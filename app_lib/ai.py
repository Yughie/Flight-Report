from promptAI import llm
from .config import VALID_AIRLINES, DATA_MIN_DATE, DATA_MAX_DATE

def extract_airline_filter(
    user_message: str,
    chat_history: list,
    current_airline: str | None = None,
    current_date_from: str | None = None,
    current_date_to: str | None = None,
    insight_history: list | None = None,
) -> dict:
    """
    Use LLM to classify user intent and extract filter parameters.
    Returns dict with 'action' (filter/clear/insight/none), 'airline', dates, etc.
    """
    airlines_list = "\n".join(f"- {a}" for a in VALID_AIRLINES)

    # Build a human-readable summary of what the viewer is currently seeing
    _filters_desc_parts: list[str] = []
    if current_airline:
        _filters_desc_parts.append(f"Airline: {current_airline}")
    else:
        _filters_desc_parts.append("Airline: All airlines (no filter)")
    if current_date_from or current_date_to:
        _filters_desc_parts.append(
            f"Date range: {current_date_from or DATA_MIN_DATE} to {current_date_to or DATA_MAX_DATE}"
        )
    else:
        _filters_desc_parts.append(f"Date range: {DATA_MIN_DATE} to {DATA_MAX_DATE} (full year)")
    current_filters_text = "\n".join(_filters_desc_parts)

    system_prompt = f"""You are an AI assistant that helps control a Power BI flight report.
Your job is to classify the user's intent and extract parameters.

Important: The dataset only contains flights in the 2015 timeline. All dates must be within 2015.

The viewer is CURRENTLY looking at the report with these active filters:
{current_filters_text}

Available airlines:
{airlines_list}

Instructions — respond with EXACTLY one of the following action types:

1. FILTER — the user wants to change the airline or date filter:
    ACTION: filter
    AIRLINE: [exact airline name from the list]  ← include only if changing airline
    DATE_FROM: YYYY-MM-DD  ← include only if changing start date
    DATE_TO: YYYY-MM-DD    ← include only if changing end date
    If the user asks to filter by date but does not provide specific dates, set DATE_FROM to 2015-01-01 and DATE_TO to 2015-12-31.

2. Single-date shorthand:
   - Phrases like "go to [date]", "until [date]", "by [date]" → DATE_TO only
   - Phrases like "start from [date]", "begin [date]", "after [date]" → DATE_FROM only
   - If unclear, default to DATE_TO for single dates

3. CLEAR — the user wants to clear/reset/remove all filters:
    ACTION: clear

4. INSIGHT — the user asks a question about the data, wants analysis, trends,
   comparisons, summaries, insights, or any question that should be answered
   using the report data. Examples: "give me an insight", "what's the cancellation rate?",
   "how is Delta performing?", "summarize the data", "any anomalies?".
    ACTION: insight

5. NONE — the message is a greeting, off-topic, or truly unrelated:
    ACTION: none
    MESSAGE: [your helpful response]

6. Fuzzy airline matching — if the user mentions an airline name but it's not exact,
   match it to the closest one.
   For example: "Delta" → "Delta Air Lines Inc.", "Southwest" → "Southwest Airlines Co."

IMPORTANT: If there is ANY doubt whether the user is asking about the data vs.
requesting a filter change, choose ACTION: insight. Only use ACTION: filter when
the user EXPLICITLY asks to change/set/show a specific filter.

FOLLOW-UP DETECTION: If the user's message is a follow-up to a previous
insight/analysis (e.g., "tell me more", "why?", "what about that?",
"elaborate", "and the delays?", "can you explain?", "go deeper",
"what else?", short clarifying questions, or any message that only makes
sense in the context of a prior data answer), classify it as ACTION: insight.
Follow-ups are VERY common — when in doubt, choose insight.

Always output only the specified keys (ACTION, AIRLINE, DATE_FROM, DATE_TO, MESSAGE) on separate lines.
"""

    # Build conversation context
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add recent chat history for context (last 4 messages)
    # Skip the very last entry if it's the same user message we're about to add
    history_slice = chat_history[-5:]
    if history_slice and history_slice[-1].get("role") == "user" and history_slice[-1].get("content") == user_message:
        history_slice = history_slice[:-1]
    for msg in history_slice:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # If there is an active insight conversation, add a hint so the classifier
    # knows recent context was data-analysis and follow-ups should be "insight"
    if insight_history:
        recent_insight = insight_history[-2:]  # last Q&A pair
        insight_summary_parts = []
        for m in recent_insight:
            role_label = "User" if m["role"] == "user" else "AI"
            snippet = m["content"][:200]
            insight_summary_parts.append(f"{role_label}: {snippet}")
        insight_ctx = "\n".join(insight_summary_parts)
        messages.append({
            "role": "system",
            "content": (
                "[Context] The user recently had this data-insight conversation:\n"
                f"{insight_ctx}\n"
                "If the new message is a follow-up to this, classify as ACTION: insight."
            ),
        })
    
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = llm.invoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse the response
        lines = content.strip().split('\n')
        result = {"action": "none", "airline": None, "date_from": None, "date_to": None, "message": content}

        def normalize_date(s: str):
            s = s.strip()
            import re
            if not s:
                return None
            # If already ISO-like
            try:
                import datetime as _dt
                min_d = _dt.date.fromisoformat(DATA_MIN_DATE)
                max_d = _dt.date.fromisoformat(DATA_MAX_DATE)
            except Exception:
                min_d = None
                max_d = None

            if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
                try:
                    d = _dt.date.fromisoformat(s)
                except Exception:
                    return None
                if min_d and d < min_d:
                    d = min_d
                if max_d and d > max_d:
                    d = max_d
                return d.isoformat()

            # Try to parse common formats using dateutil if available
            try:
                from dateutil import parser
                dt = parser.parse(s)
                d = dt.date()
                if min_d and d < min_d:
                    d = min_d
                if max_d and d > max_d:
                    d = max_d
                return d.isoformat()
            except Exception:
                # Fallback: attempt to extract YYYY-MM-DD
                m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
                if m:
                    try:
                        d = _dt.date.fromisoformat(m.group(1))
                        if min_d and d < min_d:
                            d = min_d
                        if max_d and d > max_d:
                            d = max_d
                        return d.isoformat()
                    except Exception:
                        return None
            return None

        for line in lines:
            if line.startswith("ACTION:"):
                result["action"] = line.replace("ACTION:", "").strip().lower()
            elif line.startswith("AIRLINE:"):
                airline = line.replace("AIRLINE:", "").strip()
                # Validate airline exists
                if airline in VALID_AIRLINES:
                    result["airline"] = airline
                else:
                    # Fuzzy match
                    for valid in VALID_AIRLINES:
                        if airline and (airline.lower() in valid.lower() or valid.lower() in airline.lower()):
                            result["airline"] = valid
                            break
            elif line.startswith("DATE_FROM:"):
                df = line.replace("DATE_FROM:", "").strip()
                result["date_from"] = normalize_date(df)
            elif line.startswith("DATE_TO:"):
                dt = line.replace("DATE_TO:", "").strip()
                result["date_to"] = normalize_date(dt)
            elif line.startswith("MESSAGE:"):
                result["message"] = line.replace("MESSAGE:", "").strip()

        return result
    except Exception as e:
        return {"action": "none", "airline": None, "message": f"Error processing request: {str(e)}"}


def analyze_with_data(user_message: str, insight_history: list, data_summary: str) -> str:
    """Insight Agent — answers data questions using the actual report data.

    This agent has its OWN conversation history (``insight_history``),
    separate from the filter/router chat, so it stays focused on
    data analysis without being polluted by filter commands.

    ``data_summary`` is the plain-text block produced by
    ``powerbi_data.format_data_for_ai()``.
    """
    system_prompt = f"""You are a specialised **Aviation Data Analyst AI** with operational expertise.

You are embedded inside a Power BI flight-report dashboard.  The viewer has
applied certain filters (airline and/or date range) and the data below
represents EXACTLY what they are currently seeing.

{data_summary}

─── YOUR GUIDELINES ───

1. **Be data-driven** — always reference specific numbers, percentages, and
   date ranges from the data above.  Never invent figures.
2. **Proactive insights** — when the user asks a broad question like
   "give me an insight", provide a structured analysis covering ONLY the
   sections for which actual data exists:
   • **Executive Summary** (2-3 sentences): headline performance with health badges
   • **Key Metrics**: total flights, cancellation rate vs target, avg delay per flight
   • **KPI Health Check**: reference the health badges (🟢/🟡/🔴) from the data
   • **Top Delay Contributors**: rank delay types by share, highlight the dominant cause
   • **Cancellation Analysis**: breakdown by reason with the leading driver
   • **Trend Direction**: whether metrics are improving ↑ or deteriorating ↓
   • **Monthly Highlights**: best and worst performing months
   • **Actuals vs Goals**: explicitly compare (e.g., "OTP 78.5% vs target 80% — gap of 1.5pp")
   **CRITICAL: ONLY include a section if the data above contains actual
   numbers for it.  If the data context has NO flight volume rows, NO
   monthly breakdown, NO goal comparison, or NO trend data for a topic,
   OMIT that section entirely.  Do NOT write "No data available" or
   "Unable to determine" — simply skip it.**
3. **SOP Recommendations** — when metrics are below target, reference the SOP
   recommendations from the data context. Present them as actionable next steps:
   • If cancellation rate exceeds target → cite relevant SOP-C/W/N/S codes
   • If OTP is below target → cite relevant SOP-D/LA/AL/WD/AS codes
   • If flight volume is below goal → cite SOP-FV codes
   • If one delay type dominates (>40%) → highlight it as a priority focus area
   Frame SOPs as "Recommended Actions" with urgency level (🔴 Immediate / 🟡 Short-term / 🟢 Monitor).
4. **Comparisons** — when data includes both actuals and targets/goals,
   explicitly compare them (e.g., "OTP of 78.5% vs target of 80%").
   Compute the gap and state whether it is material.
5. **Structure** — use bullet points, numbered lists, or short sections for
   readability.  Lead with the most important finding.
6. **Operational language** — use aviation industry terminology where
   appropriate (OTP, block time, turn time, CDM, MEL, etc.) but explain
   acronyms on first use.
7. **Scope awareness** — if the question is about something NOT in the data
   above, say so clearly and suggest the user change filters.
8. **Conversation continuity** — you may reference earlier answers in this
   insight conversation to avoid repeating yourself.  When the user asks a
   follow-up (e.g., "tell me more", "what about …?", "why?", "elaborate",
   "and the delays?"), use the conversation history to understand what
   they are referring to and provide a deeper or related analysis.
9. **Filter hints** — if a deeper drill-down would help, suggest the user
   change filters (e.g., "Try filtering to a specific month for more detail").
10. **Risk Escalation** — for any 🔴 CRITICAL KPIs, explicitly flag them as
    requiring management attention and provide the specific SOP actions.
11. **Never report missing sections** — if a data category (e.g., flight
    volume goal, monthly breakdown) has no rows in the data context above,
    simply do not mention it.  Do NOT output sentences like "No data for …"
    or "Unable to specify …".  Only discuss what IS present.
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Use insight-specific history (no filter noise)
    for msg in insight_history[-8:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        response = llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        return f"Sorry, I couldn't analyse the data right now: {exc}"


__all__ = ["extract_airline_filter", "analyze_with_data"]
