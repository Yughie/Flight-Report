import streamlit as st
import streamlit.components.v1 as components
import os
import time
import json
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from datetime import datetime
import html as _html

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import the LLM from promptAI
from promptAI import llm

# Import our vision pipeline modules
from screenshot_capture import capture_screenshot_sync, get_latest_screenshot, save_uploaded_screenshot, capture_screenshot_internal, save_base64_screenshot
from groq_vision import extract_dashboard_data_with_groq, format_extracted_data_for_llm

# Page config
st.set_page_config(
    page_title="Flight Report Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-size: 0.95rem;
    }
    .chat-message {
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        font-size: 1.05rem;
        color: #111; /* Ensure high-contrast text for readability */
    }
    .chat-message h1, .chat-message h2, .chat-message h3, .chat-message h4, .chat-message h5 {
        font-size: 1.1rem !important;
        margin: 0 !important;
    }
    .chat-message * {
        font-size: 1.05rem !important;
        white-space: pre-wrap !important;
        color: inherit !important;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .assistant-message {
        background-color: #eceff1; /* slightly darker for contrast */
        border-left: 4px solid #43a047;
        color: #111 !important; /* force dark text in assistant bubble */
    }
    .user-message, .assistant-message {
        color: #111 !important; /* ensure all chat bubbles use dark text */
    }
    .chat-message a, .chat-message p, .chat-message span, .chat-message small, .chat-message b {
        color: inherit !important;
    }
    .data-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    .data-success {
        background-color: #e8f5e9;
        border-left: 3px solid #4caf50;
    }
    .data-warning {
        background-color: #fff3e0;
        border-left: 3px solid #ff9800;
    }
    .quick-btn {
        margin: 2px 0;
    }
    /* Increase general block-container text to improve readability for captions/labels */
    .main .block-container p, .main .block-container small, .main .block-container span, .main .block-container div, .main .block-container li {
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'last_screenshot' not in st.session_state:
    st.session_state.last_screenshot = None
if 'last_extraction_time' not in st.session_state:
    st.session_state.last_extraction_time = None
if 'extraction_error' not in st.session_state:
    st.session_state.extraction_error = None

# ==================== Vision Pipeline Functions ====================

def capture_and_extract_data() -> tuple[dict, str]:
    """
    Complete pipeline: Screenshot → Groq Vision → Extracted Data
    Captures directly from Streamlit page (no Power BI URL needed)
    
    Returns:
        Tuple of (extracted_data_result, error_message)
    """
    try:
        # Step 1: Capture screenshot from Streamlit page
        st.session_state.extraction_error = None
        # Pass explicit Streamlit URL (helps when auto-detection fails)
        screenshot_path = capture_screenshot_sync(
            streamlit_capture=True,
            streamlit_url=os.getenv('STREAMLIT_URL') or 'http://localhost:8501',
            container_selector='iframe'
        )
        st.session_state.last_screenshot = screenshot_path
        
        # Step 2: Extract data using Groq Vision
        extraction_result = extract_dashboard_data_with_groq(screenshot_path)
        
        if extraction_result.get("success"):
            st.session_state.extracted_data = extraction_result
            st.session_state.last_extraction_time = datetime.now()
            return extraction_result, None
        else:
            error_msg = extraction_result.get("error", "Unknown extraction error")
            st.session_state.extraction_error = error_msg
            return None, error_msg
    except Exception as e:
        error_msg = str(e)
        st.session_state.extraction_error = error_msg
        return None, error_msg


def check_groq_status():
    """Check if Groq API is configured."""
    return bool(os.getenv('GROQ_API_KEY'))


def extract_from_uploaded_screenshot(uploaded_file) -> tuple[dict, str]:
    """
    Extract data from a manually uploaded screenshot.

    Returns:
        Tuple of (extracted_data_result, error_message)
    """
    try:
        st.session_state.extraction_error = None

        # Save the uploaded file to disk
        screenshot_path = save_uploaded_screenshot(uploaded_file)
        st.session_state.last_screenshot = screenshot_path

        # Extract data using Groq Vision
        extraction_result = extract_dashboard_data_with_groq(screenshot_path)

        if extraction_result.get("success"):
            st.session_state.extracted_data = extraction_result
            st.session_state.last_extraction_time = datetime.now()
            return extraction_result, None
        else:
            error_msg = extraction_result.get("error", "Unknown extraction error")
            st.session_state.extraction_error = error_msg
            return None, error_msg

    except Exception as e:
        error_msg = str(e)
        st.session_state.extraction_error = error_msg
        return None, error_msg


def check_llm_status():
    """Check if Azure OpenAI credentials are configured."""
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    return all([deployment, endpoint, api_key])


# Base system prompt used for LLM calls (dashboard context will be inserted)
BASE_SYSTEM_PROMPT = """You are an AI assistant specialized in analyzing flight data and airline operations. 
You help users understand flight patterns, delays, cancellations, and airline performance metrics.

IMPORTANT INSTRUCTIONS:
1. When dashboard data is provided below, use these ACTUAL numbers in your responses
2. Be specific with metrics and reference the real data values
3. If data shows "unclear" for any metric, acknowledge this limitation
4. Provide actionable insights based on the visible data
5. Be concise but data-driven

{dashboard_context}
"""


def get_dashboard_context() -> str:
    """Get formatted dashboard context for LLM."""
    if st.session_state.extracted_data:
        return format_extracted_data_for_llm(st.session_state.extracted_data)
    return "No dashboard data available. Please capture and extract data first using the '🔄 Capture & Analyze Dashboard' button."


def get_system_prompt_with_data():
    """Build system prompt with extracted dashboard data."""
    dashboard_context = get_dashboard_context()
    return BASE_SYSTEM_PROMPT.format(dashboard_context=dashboard_context)


def generate_auto_insights() -> tuple[str, str]:
    """
    Automatically generate insights based on extracted dashboard data.

    Returns:
        Tuple of (insights_text, error_message)
    """
    if not st.session_state.extracted_data:
        return None, "No dashboard data available"

    insight_prompt = """Based on the dashboard data provided, generate a comprehensive analysis with:

1. Executive Summary (2-3 sentences about overall performance)

2. Key Metrics Analysis:
   - Total flights and what this indicates
   - On-time performance assessment (is it good/bad for the industry?)
   - Cancellation rate analysis

3. Notable Findings from the charts:
   - Main delay causes and their impact
   - Cancellation patterns if visible

4. Actionable Recommendations (2-3 specific suggestions based on the data)

Keep the response concise but data-driven. Reference specific numbers from the extracted data."""

    return call_llm(insight_prompt)


def call_llm(user_message: str) -> tuple[str, str]:
    """Call the LLM with context and return (response, error)."""
    try: 
        system_prompt = get_system_prompt_with_data()
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        # Add conversation history (last 6 messages for context)
        for msg in st.session_state.messages[-6:]:
            messages.append({"role": msg['role'], "content": msg['text']})
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        response = llm.invoke(messages)
        return response.content, None
    except Exception as e:
        return None, f"Error: {str(e)}"


def append_message(role: str, text: str):
    """Add a message to the conversation history."""
    st.session_state.messages.append({
        'role': role, 
        'text': text,
        'timestamp': datetime.now().strftime("%H:%M")
    })


def export_chat() -> str:
    """Export chat history as text."""
    lines = ["Flight Report AI Assistant - Chat Export", "=" * 50, ""]
    for msg in st.session_state.messages:
        ts = msg.get('timestamp', '')
        role = "You" if msg['role'] == 'user' else "Assistant"
        lines.append(f"[{ts}] {role}:")
        lines.append(msg['text'])
        lines.append("")
    return "\n".join(lines)


# ==================== Build Embed URL ====================
secure_url = os.getenv('POWERBI_EMBED_URL') or os.getenv('REPORT_LINK')
if secure_url:
    p = urlparse(secure_url)
    qs = dict(parse_qsl(p.query))
    qs.update({
        'filterPaneEnabled': 'false',
        'navContentPaneEnabled': 'false'
    })
    new_query = urlencode(qs, doseq=True)
    embed_url = urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))
else:
    embed_url = None

# Check statuses
llm_ready = check_llm_status()
groq_ready = check_groq_status()

# ==================== Layout ====================

# Dashboard Vision: show at very top until a screenshot has been uploaded/analyzed
show_dashboard_vision = not (st.session_state.get('extracted_data') or st.session_state.get('uploaded_auto_analyzed'))
if show_dashboard_vision:
    # Wrap the entire top vision area so it can be hidden client-side after upload
    st.markdown('<div id="top-dashboard-vision">', unsafe_allow_html=True)
    st.markdown("##### 📸 Dashboard Vision")

    # Tab for auto vs manual capture
    capture_tab, upload_tab = st.tabs(["🤖 Auto Capture", "📤 Upload Screenshot"])

    with capture_tab:
        st.caption("📌 Click below to instantly capture the Power BI report visible below (uses internal screenshot - very fast!)")

        if st.button("📸 Capture & Analyze Dashboard", type="primary", use_container_width=True):
            if not groq_ready:
                st.error("Groq API key not configured!")
            else:
                st.session_state.trigger_capture = True
                st.rerun()

        # Handle capture after button click (on rerun)
        if st.session_state.get('trigger_capture', False):
            progress_text = st.empty()
            progress_bar = st.progress(0)

            progress_text.text("📸 Preparing internal capture...")
            progress_bar.progress(10)

            extraction_result = None
            try:
                progress_text.text("📸 Capturing Power BI dashboard...")
                progress_bar.progress(30)

                expected_path = capture_screenshot_internal(target_selector='#powerbi-report-capture')

                # Poll briefly for the file to appear
                timeout = 6.0
                interval = 0.5
                elapsed = 0.0
                screenshot_path = None
                while elapsed < timeout:
                    if os.path.exists(expected_path):
                        screenshot_path = expected_path
                        break
                    time.sleep(interval)
                    elapsed += interval

                if screenshot_path:
                    st.session_state.last_screenshot = screenshot_path
                    try:
                        progress_text.text("🖼️ Screenshot captured — displaying preview...")
                        progress_bar.progress(50)
                        st.image(screenshot_path, caption="Captured screenshot", use_container_width=True)
                    except Exception:
                        pass

                    progress_text.text("🔍 Analyzing with Groq Vision...")
                    progress_bar.progress(70)
                    extraction_result = extract_dashboard_data_with_groq(screenshot_path)
                    progress_bar.progress(100)
                    st.session_state.trigger_capture = False
                else:
                    st.info("⏳ Waiting for screenshot capture to complete... If this persists try 'Upload Screenshot'.")
                    extraction_result = None
            except Exception as e:
                progress_text.empty()
                progress_bar.empty()
                st.error(f"❌ Capture failed: {e}")
                st.info("💡 Tip: If internal capture fails, try using the 'Upload Screenshot' tab instead.")
                st.session_state.trigger_capture = False
                extraction_result = None

            if extraction_result:
                progress_text.empty()
                progress_bar.empty()

                if extraction_result.get('success'):
                    st.session_state.extracted_data = extraction_result
                    st.session_state.last_extraction_time = datetime.now()
                    st.success("✅ Dashboard analyzed successfully!")
                    st.session_state.trigger_capture = False
                    st.rerun()
                else:
                    st.error(f"❌ {extraction_result.get('error', 'Unknown extraction error')}")
                    st.session_state.trigger_capture = False

    with upload_tab:
        st.caption("📌 Upload a dashboard screenshot (smaller compact view)")

        up_col, opt_col = st.columns([3, 1])
        with up_col:
            uploader_placeholder = st.empty()
            uploaded_file = uploader_placeholder.file_uploader(
                "",
                type=['png', 'jpg', 'jpeg'],
                help="Use Win+Shift+S to take a quick screenshot and upload",
                key='uploaded_file_uploader_top'
            )
        with opt_col:
            auto_insights = st.checkbox("Auto insights", value=True, help="Generate insights after analysis", key='auto_insights_top')

        if uploaded_file is not None:
            try:
                st.image(uploaded_file, caption="Uploaded", width=260)
            except Exception:
                pass

            last_name = st.session_state.get('last_uploaded_name')
            if last_name != getattr(uploaded_file, 'name', None):
                st.session_state.last_uploaded_name = getattr(uploaded_file, 'name', None)
                st.session_state.uploaded_auto_analyzed = False

            if not st.session_state.get('uploaded_auto_analyzed', False):
                st.session_state.uploaded_auto_analyzed = True
                if not groq_ready:
                    st.error("Groq API key not configured!")
                else:
                    with st.spinner("🔍 Analyzing uploaded screenshot with Groq Vision..."):
                        uploaded_file.seek(0)
                        result, error = extract_from_uploaded_screenshot(uploaded_file)

                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.success("✅ Screenshot analyzed successfully!")
                        if auto_insights and llm_ready:
                            with st.spinner("💡 Generating AI insights..."):
                                insights, insight_error = generate_auto_insights()
                            if insights and not insight_error:
                                append_message('assistant', f"📊 **Auto-Generated Dashboard Insights:**\n\n{insights}")
                        # After a successful upload analysis, mark as analyzed.
                        # Avoid forcing a rerun here to prevent the dashboard iframe from refreshing.
                        st.session_state.uploaded_auto_analyzed = True
                        try:
                            uploader_placeholder.empty()
                        except Exception:
                            pass
                        # Inject parent-level CSS to hide the top Dashboard Vision immediately
                        try:
                            st.markdown('<style>#top-dashboard-vision{display:none !important;}</style>', unsafe_allow_html=True)
                        except Exception:
                            pass

# close the top-dashboard-vision wrapper (if present)
if show_dashboard_vision:
    st.markdown('</div>', unsafe_allow_html=True)

# continue to render main columns
left, right = st.columns([3, 1])

with left:
    st.markdown("### ✈️ Flight Report Dashboard")
    if embed_url:
        iframe_html = f'<iframe src="{embed_url}" style="border:none; width:100%; height:82vh; border-radius:8px;" allowfullscreen></iframe>'
        capture_wrapper = f'<div id="powerbi-report-capture">{iframe_html}</div>'
        components.html(capture_wrapper, height=720)
    else:
        st.error("No Power BI embed URL configured. Set REPORT_LINK or POWERBI_EMBED_URL in .env")

with right:
    st.markdown("### 🤖 AI Assistant")
    
    # Status indicators
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if llm_ready:
            st.success("AI ✅")
        else:
            st.error("AI ⚠️")
    
    with status_col2:
        if groq_ready:
            st.success("Vision ✅")
        else:
            st.warning("Vision ⚠️")
    
    if not llm_ready:
        st.caption("Configure Azure OpenAI in `.env`")
    if not groq_ready:
        st.caption("Configure GROQ_API_KEY in `.env`")

    # ==================== Quick Actions ====================
    st.markdown("##### ⚡ Quick Insights")
    qcol1, qcol2 = st.columns(2)
    
    quick_prompts = {
        "📊 Summary": "Based on the dashboard data, give me a brief executive summary of the current flight performance metrics.",
        "⏰ Delays": "Analyze the delay patterns visible in the dashboard. What are the main causes and how severe are they?",
        "🏆 Airlines": "Which airlines are performing best and worst according to the dashboard data?",
        "💡 Insights": "Provide 3 actionable insights based on the extracted dashboard data."
    }
    quick_keys = list(quick_prompts.keys())
    btns = []
    for i, k in enumerate(quick_keys):
        col = qcol1 if i % 2 == 0 else qcol2
        btn = col.button(k, key=f"quick_{i}", use_container_width=True)
        btns.append(btn)
    
    st.markdown("---")
    
    # Handle quick prompts
    selected_quick = None
    for i, (key, prompt_text) in enumerate(quick_prompts.items()):
        if btns[i]:
            selected_quick = (key, prompt_text)
            break

    # ==================== Conversation History (moved to top of chat) ====
    st.markdown("##### 💬 Conversation")

    msgs = st.session_state.get('messages', [])
    if not msgs:
        st.caption("👆 Capture the dashboard first, then ask questions!")
    else:
        chat_html = [
            "<style>",
            ".bubble{padding:10px;border-radius:10px;margin:6px 0;max-width:100%;word-wrap:break-word}",
            ".bubble b{font-weight:700}",
            ".bubble small{color:#555;font-weight:400;margin-left:6px}",
            ".user{background:#e3f2fd;border-left:4px solid #1976d2}",
            ".assistant{background:#eceff1;border-left:4px solid #43a047}",
            "</style>",
            '<div id="chat-box" style="height:360px;overflow-y:auto;padding:8px;">'
        ]

        for m in msgs:
            role = m.get('role', '')
            ts = m.get('timestamp', '')
            text = _html.escape(m.get('text', ''))
            text = text.replace('\n', '<br>')
            if role == 'user':
                chat_html.append(f'<div class="bubble user"><b>You</b> <small>({ts})</small><div style="margin-top:6px">{text}</div></div>')
            else:
                chat_html.append(f'<div class="bubble assistant"><b>Assistant</b> <small>({ts})</small><div style="margin-top:6px">{text}</div></div>')

        chat_html.append('</div>')
        chat_html.append('<script>const cb=document.getElementById("chat-box");if(cb){cb.scrollTop=cb.scrollHeight;} </script>')

        components.html('\n'.join(chat_html), height=420, scrolling=True)
    
     # ==================== Custom Prompt ====================
    prompt = st.text_area("Ask a question...", height=80, key='prompt_input', 
                          placeholder="e.g., What does the data show about cancellation rates?")
    
    bcol1, bcol2, bcol3 = st.columns([2, 1, 1])
    send = bcol1.button("🚀 Send", type="primary", use_container_width=True)
    clear = bcol2.button("🗑️", help="Clear chat", use_container_width=True)
    export = bcol3.button("📥", help="Export chat", use_container_width=True)
    
    # Dashboard Vision moved to the top of the page to save space.
    # Controls for capture/upload are available in the top banner; once an upload
    # has been analyzed the top banner hides itself to maximize dashboard space.
    
    
    
   
    # Process user input
    if llm_ready:
        if selected_quick:
            label, full_prompt = selected_quick
            append_message('user', label)
            with st.spinner('🤔 Analyzing...'):
                response, err = call_llm(full_prompt)
            if err:
                append_message('assistant', err)
            else:
                append_message('assistant', response)
            st.rerun()
        
        if send and prompt.strip():
            append_message('user', prompt)
            with st.spinner('🤔 Analyzing...'):
                response, err = call_llm(prompt)
            if err:
                append_message('assistant', err)
            else:
                append_message('assistant', response)
            st.rerun()
    
    if clear:
        st.session_state.messages = []
        st.session_state.extracted_data = None
        st.session_state.last_screenshot = None
        st.session_state.last_extraction_time = None
        st.session_state.extraction_error = None
        st.rerun()
    
    if export and st.session_state.messages:
        chat_text = export_chat()
        st.download_button(
            label="Download Chat",
            data=chat_text,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )
    
   


