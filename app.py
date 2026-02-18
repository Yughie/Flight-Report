import streamlit as st
import requests
import time

# Must be first Streamlit command
st.set_page_config(page_title="AI Flight Report", layout="wide")

from app_lib.config import *
from app_lib.ai import extract_airline_filter, analyze_with_data, analyze_with_data_stream
from app_lib.powerbi import get_aad_token, get_embed_params, render_powerbi_report
from app_lib.powerbi_data import (
    get_dataset_id,
    fetch_airline_list,
    fetch_all_report_data,
    format_data_for_ai,
    execute_dax,
)

st.title("🛫 AI-Controlled Power BI Report")

# ── Session-state initialisation ──────────────────────────────────
_defaults = {
    "report_loaded": False,
    "embed_token": None,
    "embed_url": None,
    "chat_history": [],
    "current_airline_filter": None,
    "current_date_from": DATA_MIN_DATE,   # default to 2015-01-01
    "current_date_to": DATA_MAX_DATE,     # default to 2015-12-31
    # Data-layer state
    "dataset_id": None,
    "report_data": None,         # dict returned by fetch_all_report_data
    "data_filters_key": None,    # (airline, date_from, date_to) when data was last fetched
    # Separate history for the Insight Agent (keeps insight Q&A isolated)
    "insight_history": [],
}
for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helper: fetch / refresh report data ───────────────────────────
def _refresh_report_data() -> int:
    """Fetch report data for the current filter state and cache it.
    Returns elapsed time in milliseconds.
    """
    t0 = time.perf_counter()
    airline = st.session_state.current_airline_filter
    d_from = st.session_state.current_date_from
    d_to = st.session_state.current_date_to
    new_key = (airline, d_from, d_to)

    # Skip if data is already up-to-date for these filters
    if st.session_state.data_filters_key == new_key and st.session_state.report_data:
        return round((time.perf_counter() - t0) * 1000)

    try:
        aad_token = get_aad_token()
        if not st.session_state.dataset_id:
            st.session_state.dataset_id = get_dataset_id(aad_token)
        data = fetch_all_report_data(
            aad_token,
            st.session_state.dataset_id,
            airline, d_from, d_to,
        )
        st.session_state.report_data = data
        st.session_state.data_filters_key = new_key
        
        # Show errors in sidebar if any
        if data.get("errors"):
            with st.sidebar:
                st.error("⚠️ Data Fetch Errors:")
                for err in data["errors"]:
                    st.code(err, language="text")
                    # Check for Premium capacity issues
                    if "premium" in err.lower() or "capacity" in err.lower() or "timed out" in err.lower():
                        st.warning("""
**⚠️ Premium Capacity Required**

The Power BI `executeQueries` API (used to fetch data via DAX) requires:
- Power BI **Premium** capacity, OR
- **Premium Per User (PPU)** license, OR  
- Power BI **Embedded** capacity

Your workspace appears to lack this. Options:
1. Upgrade to Premium/PPU
2. Use the embedded report visuals only (no AI insights)
3. Connect to Databricks as an alternative data source
                        """)
    except requests.exceptions.Timeout:
        error_msg = (
            "Request timed out. Your workspace likely doesn't have Premium/PPU capacity. "
            "The executeQueries API requires Premium capacity to work."
        )
        st.session_state.report_data = {"errors": [error_msg]}
        st.session_state.data_filters_key = new_key
        with st.sidebar:
            st.error(f"❌ {error_msg}")
    except Exception as exc:
        st.session_state.report_data = {"errors": [str(exc)]}
        st.session_state.data_filters_key = new_key
        # Show the error prominently
        with st.sidebar:
            st.error(f"❌ Failed to fetch data: {exc}")

    return round((time.perf_counter() - t0) * 1000)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("💬 AI Assistant")
    st.markdown(
        "Just type naturally — I'll figure out if you want to **filter** "
        "the report or **ask a question** about the data!"
    )
    st.caption("Examples: *\"show me Delta\"*, *\"what's the delay trend?\"*, *\"flights in March\"*")
    st.divider()
    
    # Current filter status
    active_filters = []
    if st.session_state.current_airline_filter:
        active_filters.append(f"Airline: **{st.session_state.current_airline_filter}**")
    if st.session_state.current_date_from or st.session_state.current_date_to:
        df = st.session_state.current_date_from or "(start)"
        dt = st.session_state.current_date_to or "(end)"
        active_filters.append(f"Dates: **{df}** to **{dt}**")
    
    if active_filters:
        st.success("🎯 **Active Filters:**")
        for filter_text in active_filters:
            st.write(filter_text)
        if st.button("🗑️ Clear All Filters", use_container_width=True):
            st.session_state.current_airline_filter = None
            st.session_state.current_date_from = DATA_MIN_DATE
            st.session_state.current_date_to = DATA_MAX_DATE
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "All filters cleared."
            })
            # Invalidate cached data so it's re-fetched on next insight question
            st.session_state.report_data = None
            st.session_state.data_filters_key = None
            st.session_state.insight_history = []
            st.rerun()
    else:
        st.info("No filters active")
    
    
    # Data status indicator & manual refresh button
    
    if st.session_state.report_data:
        errs = st.session_state.report_data.get("errors", [])
        if errs:
            st.warning(f"📊 Data partially loaded ({len(errs)} query issue(s))")
            with st.expander("View Errors", expanded=True):
                for i, err in enumerate(errs, 1):
                    st.code(f"{i}. {err}", language="text")
        else:
            # Check if data is actually empty
            dr = st.session_state.report_data.get("date_range", {})
            if dr and dr.get("TotalFlights"):
                st.success("📊 Report data loaded — ask me anything!")
            else:
                st.warning("📊 Data fetched but all values are empty/N/A")
                with st.expander("🔍 Debug: View Raw Data", expanded=False):
                    st.json(st.session_state.report_data)

    st.divider()
    
    # Chat history display
    chat_container = st.container(height=300)
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**AI:** {msg['content']}")
    
    # Chat input
    user_input = st.chat_input("Ask anything — e.g. 'show me Delta' or 'how are cancellations?'", key="chat_input")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Show user message immediately in the chat area
        with chat_container:
            st.markdown(f"**You:** {user_input}")
            _status_placeholder = st.empty()
            _status_placeholder.info("🧠 Understanding your request…")

        # 1. Determine intent (filter / clear / insight / other)
        result = extract_airline_filter(
            user_input,
            st.session_state.chat_history,
            current_airline=st.session_state.current_airline_filter,
            current_date_from=st.session_state.current_date_from,
            current_date_to=st.session_state.current_date_to,
            insight_history=st.session_state.insight_history,
        )
        classify_ms = result.get("elapsed_ms", 0)

        # Clear status for non-insight actions (insight will update it)
        if result.get("action") != "insight":
            _status_placeholder.empty()
        
        # Helper: format elapsed time badge
        def _time_badge(label: str, ms: int) -> str:
            if ms < 1000:
                return f"⏱ {label}: {ms}ms"
            return f"⏱ {label}: {ms / 1000:.1f}s"

        # 2. Handle the result
        if result.get("action") == "filter":
            parts = []
            # Airline
            if result.get("airline"):
                st.session_state.current_airline_filter = result.get("airline")
                parts.append(f"**{st.session_state.current_airline_filter}**")

            # Date range
            if result.get("date_from") or result.get("date_to"):
                existing_from = st.session_state.current_date_from
                existing_to = st.session_state.current_date_to

                rf = result.get("date_from")
                rt = result.get("date_to")

                if rf and rt:
                    df = rf
                    dt = rt
                    parts.append(f"dates {df} to {dt}")
                elif rf and not rt:
                    df = rf
                    dt = existing_to or DATA_MAX_DATE
                    parts.append(f"start date to **{df}** (end: {dt})")
                elif rt and not rf:
                    dt = rt
                    df = existing_from or DATA_MIN_DATE
                    parts.append(f"end date to **{rt}** (start: {df})")
                else:
                    df = existing_from or DATA_MIN_DATE
                    dt = existing_to or DATA_MAX_DATE
                    parts.append(f"dates {df} to {dt}")

                st.session_state.current_date_from = df
                st.session_state.current_date_to = dt

            if parts:
                response = "✅ Filtering by " + " and ".join(parts)
            else:
                response = result.get("message") or "No filters detected."

            timing_info = _time_badge("Processed", classify_ms)
            st.session_state.chat_history.append({"role": "assistant", "content": f"{response}\n\n`{timing_info}`"})

            # Invalidate cached data so next insight question fetches fresh data
            st.session_state.report_data = None
            st.session_state.data_filters_key = None
            # Reset insight conversation since filters changed
            st.session_state.insight_history = []

        elif result.get("action") == "clear":
            st.session_state.current_airline_filter = None
            st.session_state.current_date_from = DATA_MIN_DATE
            st.session_state.current_date_to = DATA_MAX_DATE
            timing_info = _time_badge("Processed", classify_ms)
            response = f"✅ Filters cleared. Showing all airlines (2015-01-01 to 2015-12-31).\n\n`{timing_info}`"
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Invalidate cached data
            st.session_state.report_data = None
            st.session_state.data_filters_key = None
            st.session_state.insight_history = []

        elif result.get("action") == "insight":
            # ── Stage 1: Apply implied filters from the question ──
            implied_parts = []
            filter_changed = False

            if result.get("airline") and result["airline"] != st.session_state.current_airline_filter:
                st.session_state.current_airline_filter = result["airline"]
                implied_parts.append(f"**{result['airline']}**")
                filter_changed = True

            if result.get("date_from") and result["date_from"] != st.session_state.current_date_from:
                st.session_state.current_date_from = result["date_from"]
                implied_parts.append(f"from **{result['date_from']}**")
                filter_changed = True

            if result.get("date_to") and result["date_to"] != st.session_state.current_date_to:
                st.session_state.current_date_to = result["date_to"]
                implied_parts.append(f"to **{result['date_to']}**")
                filter_changed = True

            if filter_changed:
                # Invalidate cache since filters changed
                st.session_state.report_data = None
                st.session_state.data_filters_key = None
                st.session_state.insight_history = []
                filter_note = "🎯 Auto-filtered to " + ", ".join(implied_parts)
                st.session_state.chat_history.append({"role": "assistant", "content": filter_note})
                # Show filter note immediately in chat
                with chat_container:
                    st.markdown(f"**AI:** {filter_note}")

            # ── Stage 2: Fetch data with progress indicator ──
            _status_placeholder.info("📊 Fetching report data…")
            t_data = time.perf_counter()
            _refresh_report_data()
            data_ms = round((time.perf_counter() - t_data) * 1000)

            # ── Stage 3: Stream the AI insight response into chat ──
            if st.session_state.report_data and not st.session_state.report_data.get("errors"):
                data_summary = format_data_for_ai(
                    st.session_state.report_data,
                    st.session_state.current_airline_filter,
                    st.session_state.current_date_from,
                    st.session_state.current_date_to,
                )

                _status_placeholder.info("🧠 Analyzing your data…")
                t_insight = time.perf_counter()
                with chat_container:
                    _status_placeholder.empty()
                    full_response = st.write_stream(
                        analyze_with_data_stream(
                            user_input,
                            st.session_state.insight_history,
                            data_summary,
                        )
                    )
                insight_ms = round((time.perf_counter() - t_insight) * 1000)

                # Build timing footer
                timing_parts = [
                    _time_badge("Classify", classify_ms),
                    _time_badge("Data", data_ms),
                    _time_badge("Analysis", insight_ms),
                ]
                total_ms = classify_ms + data_ms + insight_ms
                timing_parts.append(_time_badge("Total", total_ms))
                timing_footer = " | ".join(timing_parts)

                final_response = f"{full_response}\n\n`{timing_footer}`"

                # Track in insight-specific history (without timing)
                st.session_state.insight_history.append({"role": "user", "content": user_input})
                st.session_state.insight_history.append({"role": "assistant", "content": full_response})
                st.session_state.chat_history.append({"role": "assistant", "content": final_response})

            elif st.session_state.report_data and st.session_state.report_data.get("errors"):
                _status_placeholder.empty()
                errs = "; ".join(st.session_state.report_data["errors"])
                response = f"I couldn't fetch the report data: {errs}"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                _status_placeholder.empty()
                response = "I couldn't retrieve report data. Please try again."
                st.session_state.chat_history.append({"role": "assistant", "content": response})

        else:
            # Truly unrelated / greeting — use the LLM's own message
            response = result.get("message", "I'm here to help with your flight report! Ask me for insights or to filter the data.")
            timing_info = _time_badge("Processed", classify_ms)
            st.session_state.chat_history.append({"role": "assistant", "content": f"{response}\n\n`{timing_info}`"})
        
        st.rerun()
    
    # Clear chat button
    if st.button("🔄 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.insight_history = []
        st.rerun()

# ── Main content area ─────────────────────────────────────────────

# Auto-load the report once when the app starts
if not st.session_state.report_loaded and not st.session_state.get('auto_load_attempted'):
    st.session_state.auto_load_attempted = True
    with st.spinner("Loading Power BI Report..."):
        embed_url, token, dataset_id = get_embed_params()
        if embed_url and token:
            st.session_state.embed_url = embed_url
            st.session_state.embed_token = token
            st.session_state.dataset_id = dataset_id
            st.session_state.report_loaded = True
        else:
            st.session_state.report_loaded = False
            st.info("Unable to auto-load the report. Check workspace permissions or credentials.")

# Show the embedded report if loaded
if st.session_state.report_loaded and st.session_state.embed_token and st.session_state.embed_url:
    render_powerbi_report(
        st.session_state.embed_token,
        st.session_state.embed_url,
        st.session_state.current_airline_filter,
        st.session_state.current_date_from,
        st.session_state.current_date_to
    )
elif st.session_state.report_loaded:
    st.error("Failed to load report. Please try again.")
else:
    st.info("👆 Click **Load Report** to view the Power BI dashboard, then use the chat to filter by airline.")