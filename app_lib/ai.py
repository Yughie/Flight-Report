"""AI classification & insight analysis layer.

Uses LangChain prompt templates, structured output, and LCEL chains
for robust intent classification and streaming insight generation.
"""

from __future__ import annotations

import time
import re
import datetime as _dt
from typing import Optional
from difflib import get_close_matches

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    BaseMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from promptAI import llm, llm_fast
from .config import VALID_AIRLINES, DATA_MIN_DATE, DATA_MAX_DATE


# ── Pre-computed lookup tables (built once, reused every call) ────
_AIRLINE_LOWER_MAP: dict[str, str] = {a.lower(): a for a in VALID_AIRLINES}
_AIRLINE_SHORTCUTS: dict[str, str] = {}
for _a in VALID_AIRLINES:
    # "Delta Air Lines Inc." → shortcuts: "delta", "delta air lines"
    _words = _a.lower().replace(".", "").split()
    _AIRLINE_SHORTCUTS[_words[0]] = _a                        # first word
    _AIRLINE_SHORTCUTS[" ".join(_words[:-1])] = _a            # without "Inc."/"Co."
    _AIRLINE_SHORTCUTS[" ".join(_words)] = _a                 # full lowercase
    if len(_words) >= 2:
        _AIRLINE_SHORTCUTS[" ".join(_words[:2])] = _a         # first two words

_MIN_DATE = _dt.date.fromisoformat(DATA_MIN_DATE)
_MAX_DATE = _dt.date.fromisoformat(DATA_MAX_DATE)


# ── Shared helpers ────────────────────────────────────────────────


def _fuzzy_match_airline(name: str) -> str | None:
    """Fast fuzzy airline matching using pre-built lookup + difflib."""
    if not name:
        return None
    low = name.lower().strip().rstrip(".")
    # Exact match
    if low in _AIRLINE_LOWER_MAP:
        return _AIRLINE_LOWER_MAP[low]
    # Shortcut match
    if low in _AIRLINE_SHORTCUTS:
        return _AIRLINE_SHORTCUTS[low]
    # Substring containment (both directions)
    for key, val in _AIRLINE_SHORTCUTS.items():
        if low in key or key in low:
            return val
    # difflib close match on full names
    matches = get_close_matches(low, list(_AIRLINE_LOWER_MAP.keys()), n=1, cutoff=0.5)
    if matches:
        return _AIRLINE_LOWER_MAP[matches[0]]
    return None


def _normalize_date(s: str) -> str | None:
    """Parse a date string and clamp to dataset bounds. Returns ISO or None."""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    # Already ISO
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        try:
            d = _dt.date.fromisoformat(s)
            return max(_MIN_DATE, min(_MAX_DATE, d)).isoformat()
        except ValueError:
            return None
    # Try dateutil
    try:
        from dateutil import parser as _dp
        d = _dp.parse(s).date()
        return max(_MIN_DATE, min(_MAX_DATE, d)).isoformat()
    except Exception:
        pass
    # Fallback regex
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        try:
            d = _dt.date.fromisoformat(m.group(1))
            return max(_MIN_DATE, min(_MAX_DATE, d)).isoformat()
        except ValueError:
            pass
    return None


def _to_langchain_messages(history: list[dict]) -> list[BaseMessage]:
    """Convert dict-based chat history to LangChain message objects."""
    messages: list[BaseMessage] = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
    return messages


# ══════════════════════════════════════════════════════════════════
#  INTENT CLASSIFICATION — Structured Output via LangChain
# ══════════════════════════════════════════════════════════════════


class IntentClassification(BaseModel):
    """Structured output from the intent classifier.

    LangChain's ``with_structured_output`` converts this into a tool-call
    schema so the LLM returns validated JSON instead of free-form text.
    """

    action: str = Field(
        description="One of: filter, clear, insight, none"
    )
    airline: Optional[str] = Field(
        default=None,
        description=(
            "Full airline name if user mentions or implies one "
            "(e.g. 'Delta Air Lines Inc.')"
        ),
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Start date in YYYY-MM-DD format if mentioned/implied",
    )
    date_to: Optional[str] = Field(
        default=None,
        description="End date in YYYY-MM-DD format if mentioned/implied",
    )
    message: Optional[str] = Field(
        default=None,
        description="Short response text — only for action=none (greetings, off-topic)",
    )


_CLASSIFY_SYSTEM_TEMPLATE = """\
You classify user messages for a Power BI flight report (2015 data only).

Current filters → Airline: {current_airline} | Dates: {current_dates}
Available airlines: {airlines_list}

Classification rules:
• filter — user ONLY wants to change the view (airline, dates), no analysis needed.
• clear — user wants to reset/remove all filters.
• insight — user asks about data, performance, trends, analysis, comparisons.
  ALSO include airline/date_from/date_to if the question implies a specific airline or
  time range — the system will auto-filter AND generate analysis in one step.
  Follow-ups ("tell me more", "why?", "elaborate") → insight with no filter change.
  When in doubt between filter and insight → choose insight.
• none — greetings, off-topic, non-flight-related chat. Provide a friendly message.

Date rules:
• Fuzzy-match airline names (e.g. "Delta" → "Delta Air Lines Inc.").
• "until/by [date]" → date_to only; "from/after [date]" → date_from only.
• All dates must be within 2015 (YYYY-MM-DD format).

Examples:
• "show me delta" → action=filter, airline=Delta Air Lines Inc.
• "how is delta doing?" → action=insight, airline=Delta Air Lines Inc.
• "what's the cancellation rate?" → action=insight
• "delays in march" → action=insight, date_from=2015-03-01, date_to=2015-03-31
• "flights in march" → action=filter, date_from=2015-03-01, date_to=2015-03-31
• "tell me more" → action=insight
• "reset everything" → action=clear
• "hi" → action=none, message=Hello! Ask about flight data or set filters.
• "compare airlines" → action=insight"""

_classify_prompt = ChatPromptTemplate.from_messages([
    ("system", _CLASSIFY_SYSTEM_TEMPLATE),
    MessagesPlaceholder("chat_history", optional=True),
    MessagesPlaceholder("insight_hint", optional=True),
    ("human", "{user_message}"),
])

# LCEL chain: prompt → LLM with structured output → IntentClassification
_classify_chain = _classify_prompt | llm_fast.with_structured_output(
    IntentClassification
)


def extract_airline_filter(
    user_message: str,
    chat_history: list,
    current_airline: str | None = None,
    current_date_from: str | None = None,
    current_date_to: str | None = None,
    insight_history: list | None = None,
) -> dict:
    """Classify user intent and extract filter parameters.

    Uses LangChain structured output (tool calling) for reliable parsing.
    Returns dict with 'action', 'airline', 'date_from', 'date_to',
    'message', and 'elapsed_ms' — fully compatible with existing callers.
    """
    t0 = time.perf_counter()

    # Convert recent chat history to LangChain messages (last 4 entries)
    history_slice = chat_history[-4:]
    if (
        history_slice
        and history_slice[-1].get("role") == "user"
        and history_slice[-1].get("content") == user_message
    ):
        history_slice = history_slice[:-1]
    chat_msgs = _to_langchain_messages(history_slice)

    # Build insight-context hint so follow-ups are classified correctly
    insight_hint_msgs: list[BaseMessage] = []
    if insight_history and len(insight_history) >= 2:
        last_q = insight_history[-2]["content"][:120]
        last_a = insight_history[-1]["content"][:120]
        insight_hint_msgs = [
            SystemMessage(
                content=(
                    f"[Recent insight Q&A] Q: {last_q}… A: {last_a}… "
                    "— follow-ups → action: insight"
                )
            )
        ]

    try:
        result_obj: IntentClassification = _classify_chain.invoke({
            "current_airline": current_airline or "All",
            "current_dates": (
                f"{current_date_from or DATA_MIN_DATE} to "
                f"{current_date_to or DATA_MAX_DATE}"
            ),
            "airlines_list": ", ".join(VALID_AIRLINES),
            "chat_history": chat_msgs,
            "insight_hint": insight_hint_msgs,
            "user_message": user_message,
        })

        elapsed = round((time.perf_counter() - t0) * 1000)
        return {
            "action": result_obj.action.lower().strip(),
            "airline": (
                _fuzzy_match_airline(result_obj.airline)
                if result_obj.airline
                else None
            ),
            "date_from": (
                _normalize_date(result_obj.date_from)
                if result_obj.date_from
                else None
            ),
            "date_to": (
                _normalize_date(result_obj.date_to)
                if result_obj.date_to
                else None
            ),
            "message": result_obj.message or "",
            "elapsed_ms": elapsed,
        }
    except Exception as e:
        elapsed = round((time.perf_counter() - t0) * 1000)
        return {
            "action": "none",
            "airline": None,
            "date_from": None,
            "date_to": None,
            "message": f"Error processing request: {str(e)}",
            "elapsed_ms": elapsed,
        }


# ══════════════════════════════════════════════════════════════════
#  INSIGHT ANALYSIS — LCEL Chain with streaming support
# ══════════════════════════════════════════════════════════════════

_INSIGHT_SYSTEM_TEMPLATE = """\
You are a specialised **Aviation Data Analyst AI** with operational expertise.

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
   • **Delay Causes Analysis** (ALWAYS present as ONE consolidated section — do NOT scatter delay info across other sections):
     - Start with the 2 outcome metrics: Departure Delay and Arrival Delay (total minutes and avg per flight)
     - Then list ALL 5 root-cause delay types (Late Aircraft, Airline, Weather, Air System, Security) ranked highest to lowest
     - For each root cause show: % share among causes, % of total departure delay, total minutes, avg per flight
     - Identify the #1 and #2 dominant causes and their combined share
     - Total delay types presented = 7 (2 outcomes + 5 root causes)
   • **Cancellation Analysis**: breakdown by reason with the leading driver
   • **Trend Direction**: whether metrics are improving ↑ or deteriorating ↓
   • **Monthly Highlights**: best and worst performing months
   • **Actuals vs Goals**: explicitly compare (e.g., "OTP 78.5% vs target 80% — gap of 1.5pp")
   **CRITICAL: ONLY include a section if the data above contains actual
   numbers for it.  If the data context has NO flight volume rows, NO
   monthly breakdown, NO goal comparison, or NO trend data for a topic,
   OMIT that section entirely.  Do NOT write "No data available" or
   "Unable to determine" — simply skip it.**
3. **SOP Recommendations** — when metrics are below target, reference the
   SKYbrary SOP guidance and related articles from the data context.
   SKYbrary (skybrary.aero) is the authoritative aviation safety knowledge
   base maintained by EUROCONTROL and industry partners.
   Present recommendations as actionable next steps grounded in SKYbrary references:
   • If cancellation rate exceeds target → cite relevant SKYbrary articles on
     weather disruption, crew management, or NAS procedures
   • If OTP is below target → cite SKYbrary articles on delay management,
     turnaround procedures, or ATC coordination
   • If flight volume is below goal → cite SKYbrary articles on schedule
     planning and capacity management
   • If one delay type dominates (>40%) → highlight it as a priority focus area
   When SKYbrary article URLs are provided in the data context, include them
   as references so the user can read the full SOP guidance.
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
9. **Delay reporting rule** — ALWAYS group ALL 7 delay metrics into a single **Delay Causes Analysis**
    section. Never split delay discussion across other sections. Structure it as:

    **Outcome Metrics** (what was experienced):
      - Departure Delay: X min total | Y min/flight
      - Arrival Delay: X min total | Y min/flight

    **Root Causes** (why delays happened) — ranked highest to lowest:
      1. Late Aircraft Delay — XX.X% of causes | XX.X% of dep delay | X,XXX,XXX min | X.X min/flight
      2. Airline Delay       — XX.X% of causes | XX.X% of dep delay | X,XXX,XXX min | X.X min/flight
      3. Air System Delay    — XX.X% of causes | XX.X% of dep delay | X,XXX,XXX min | X.X min/flight
      4. Weather Delay       — XX.X% of causes | XX.X% of dep delay | X,XXX,XXX min | X.X min/flight
      5. Security Delay      — XX.X% of causes | XX.X% of dep delay | X,XXX,XXX min | X.X min/flight

    Always close with: "#1 + #2 causes together = XX.X% of root cause delay minutes."
    This keeps the insight clean, complete (all 7 shown), and avoids delay data being scattered.
10. **Filter hints** — if a deeper drill-down would help, suggest the user
   change filters (e.g., "Try filtering to a specific month for more detail").
11. **Risk Escalation** — for any 🔴 CRITICAL KPIs, explicitly flag them as
    requiring management attention and reference the relevant SKYbrary SOP
    articles from the data context for authoritative procedural guidance.
12. **Never report missing sections** — if a data category (e.g., flight
    volume goal, monthly breakdown) has no rows in the data context above,
    simply do not mention it.  Do NOT output sentences like "No data for …"
    or "Unable to specify …".  Only discuss what IS present.
"""

_insight_prompt = ChatPromptTemplate.from_messages([
    ("system", _INSIGHT_SYSTEM_TEMPLATE),
    MessagesPlaceholder("history"),
    ("human", "{user_message}"),
])

# LCEL chain: prompt → LLM → extract text content
_insight_chain = _insight_prompt | llm | StrOutputParser()


def analyze_with_data(
    user_message: str, insight_history: list, data_summary: str,
) -> tuple[str, int]:
    """Non-streaming insight call. Returns (text, elapsed_ms)."""
    t0 = time.perf_counter()
    history_msgs = _to_langchain_messages(insight_history[-8:])
    try:
        text = _insight_chain.invoke({
            "data_summary": data_summary,
            "history": history_msgs,
            "user_message": user_message,
        })
        elapsed = round((time.perf_counter() - t0) * 1000)
        return text, elapsed
    except Exception as exc:
        elapsed = round((time.perf_counter() - t0) * 1000)
        return f"Sorry, I couldn't analyse the data right now: {exc}", elapsed


def analyze_with_data_stream(
    user_message: str, insight_history: list, data_summary: str,
):
    """Streaming insight generator — yields text chunks.

    Uses LCEL chain streaming: prompt → LLM → StrOutputParser.
    Each chunk is a plain string, compatible with ``st.write_stream()``.
    """
    history_msgs = _to_langchain_messages(insight_history[-8:])
    try:
        for chunk in _insight_chain.stream({
            "data_summary": data_summary,
            "history": history_msgs,
            "user_message": user_message,
        }):
            if chunk:
                yield chunk
    except Exception as exc:
        yield f"\n\nSorry, I couldn't analyse the data right now: {exc}"


__all__ = ["extract_airline_filter", "analyze_with_data", "analyze_with_data_stream"]
