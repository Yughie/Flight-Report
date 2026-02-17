from promptAI import llm
from .config import VALID_AIRLINES, DATA_MIN_DATE, DATA_MAX_DATE

def extract_airline_filter(user_message: str, chat_history: list) -> dict:
    """
    Use LLM to extract airline name from user message.
    Returns dict with 'action' (filter/clear/none) and 'airline' if applicable.
    """
    airlines_list = "\n".join(f"- {a}" for a in VALID_AIRLINES)
    
    system_prompt = f"""You are an AI assistant that helps control a Power BI flight report.
Your job is to understand user requests about filtering by airline and/or flight date range and extract parameters.

Important: The dataset only contains flights in the 2015 timeline. All dates must be within 2015.

Available airlines:
{airlines_list}

Instructions:
1. If the user wants to filter by an airline, respond with EXACTLY:
    ACTION: filter
    AIRLINE: [exact airline name from the list]

2. If the user wants to filter by a date range (between two dates), include both of these lines in YYYY-MM-DD format:
    DATE_FROM: YYYY-MM-DD
    DATE_TO: YYYY-MM-DD
    If the user asks to filter by date but does not provide specific dates, set DATE_FROM to 2015-01-01 and DATE_TO to 2015-12-31.

3. If the user provides only one date, determine context and return only the relevant key:
   - Phrases like "go to [date]", "until [date]", "by [date]", "end on [date]" → return DATE_TO: [date]
   - Phrases like "start from [date]", "begin [date]", "after [date]" → return DATE_FROM: [date]
   - If unclear, default to DATE_TO for single dates
   The application will preserve the other bound if already set.

4. If the user wants to clear/reset/remove all filters, respond with EXACTLY:
    ACTION: clear

5. If the user asks a general question or the message is not about filtering, respond with:
    ACTION: none
    MESSAGE: [your helpful response]

6. If the user mentions an airline name but it's not exact, match it to the closest one from the list.
    For example: "Delta" -> "Delta Air Lines Inc.", "Southwest" -> "Southwest Airlines Co."

Be strict about matching to valid airline names only. Always output only the specified keys (ACTION, AIRLINE, DATE_FROM, DATE_TO, MESSAGE) on separate lines.
"""

    # Build conversation context
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add recent chat history for context (last 4 messages)
    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
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

__all__ = ["extract_airline_filter"]
