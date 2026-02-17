import streamlit as st

# Must be first Streamlit command
st.set_page_config(page_title="AI Flight Report", layout="wide")

from app_lib.config import *
from app_lib.ai import extract_airline_filter
from app_lib.powerbi import get_embed_params, render_powerbi_report

st.title("🛫 AI-Controlled Power BI Report")

# Initialize session state
if 'report_loaded' not in st.session_state:
    st.session_state.report_loaded = False
if 'embed_token' not in st.session_state:
    st.session_state.embed_token = None
if 'embed_url' not in st.session_state:
    st.session_state.embed_url = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_airline_filter' not in st.session_state:
    st.session_state.current_airline_filter = None
if 'current_date_from' not in st.session_state:
    st.session_state.current_date_from = None
if 'current_date_to' not in st.session_state:
    st.session_state.current_date_to = None

# Sidebar for chat
with st.sidebar:
    st.header("💬 AI Assistant")
    st.markdown("Ask me to filter the report by airline!")
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
            st.session_state.current_date_from = None
            st.session_state.current_date_to = None
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "All filters cleared."
            })
            st.rerun()
    else:
        st.info("No filters active")
    
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
    user_input = st.chat_input("Type your request...", key="chat_input")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Process with AI
        with st.spinner("Processing..."):
            result = extract_airline_filter(user_input, st.session_state.chat_history)
        
        # Handle the result
        if result.get("action") == "filter":
            parts = []
            # Airline
            if result.get("airline"):
                st.session_state.current_airline_filter = result.get("airline")
                parts.append(f"**{st.session_state.current_airline_filter}**")

            # Date range (only set when provided). Preserve existing older/lower date if not mentioned.
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
                    # Only lower/older date provided: keep upper if set, otherwise default to dataset max
                    df = rf
                    dt = existing_to or DATA_MAX_DATE
                    parts.append(f"start date to **{df}** (end: {dt})")
                elif rt and not rf:
                    # Only upper/newer date provided: keep lower if set, otherwise default to dataset min
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
                # No concrete filter detected; show AI message
                response = result.get("message") or "No filters detected."

            st.session_state.chat_history.append({"role": "assistant", "content": response})
        elif result.get("action") == "clear":
            st.session_state.current_airline_filter = None
            st.session_state.current_date_from = None
            st.session_state.current_date_to = None
            response = "✅ Filters cleared. Showing all airlines."
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": result.get("message")})
        
        st.rerun()
    
    # Clear chat button
    if st.button("🔄 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
# Auto-load the report once when the app starts (removes manual Load/Close buttons)
if not st.session_state.report_loaded and not st.session_state.get('auto_load_attempted'):
    st.session_state.auto_load_attempted = True
    with st.spinner("Loading Power BI Report..."):
        embed_url, token = get_embed_params()
        if embed_url and token:
            st.session_state.embed_url = embed_url
            st.session_state.embed_token = token
            st.session_state.report_loaded = True
        else:
            # If loading failed, leave report_loaded False and show info in the main area
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