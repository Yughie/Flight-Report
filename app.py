import streamlit as st
import requests
import json
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv
from promptAI import llm

# Must be first Streamlit command
st.set_page_config(page_title="AI Flight Report", layout="wide")

# Load environment variables from .env file
load_dotenv()

# --- 1. CONFIGURATION ---
TENANT_ID = os.getenv("POWERBI_TENANT_ID")
CLIENT_ID = os.getenv("POWERBI_CLIENT_ID")
CLIENT_SECRET = os.getenv("POWERBI_CLIENT_SECRET")
WORKSPACE_ID = os.getenv("POWERBI_WORKSPACE_ID")
REPORT_ID = os.getenv("POWERBI_REPORT_ID")

# Available airlines for validation (from airlines.csv)
VALID_AIRLINES = [
    "United Air Lines Inc.",
    "American Airlines Inc.",
    "US Airways Inc.",
    "Frontier Airlines Inc.",
    "JetBlue Airways",
    "Skywest Airlines Inc.",
    "Alaska Airlines Inc.",
    "Spirit Air Lines",
    "Southwest Airlines Co.",
    "Delta Air Lines Inc.",
    "Atlantic Southeast Airlines",
    "Hawaiian Airlines Inc.",
    "American Eagle Airlines Inc.",
    "Virgin America"
]

# Dataset timeline bounds (clamp defaults and parsed dates to this range)
DATA_MIN_DATE = "2015-01-01"
DATA_MAX_DATE = "2015-12-31"

# --- 2. AI FILTER EXTRACTION ---
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

# --- 3. FUNCTION TO GET ACCESS TOKEN ---
def get_embed_params():
    # 1. Get Azure AD Access Token
    auth_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    
    auth_data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET, 
        "scope": "https://analysis.windows.net/powerbi/api/.default"
    }
    
    auth_res = requests.post(auth_url, data=auth_data)
    
    if auth_res.status_code != 200:
        st.error(f"Azure Auth Failed ({auth_res.status_code})")
        st.json(auth_res.json())
        st.stop()
        
    aad_token = auth_res.json().get("access_token")

    # 2. Get Report Details
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {aad_token}'}
    report_url = f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}/reports/{REPORT_ID}"
    
    report_res = requests.get(report_url, headers=headers)
    
    if report_res.status_code != 200:
        st.error(f"Power BI API Failed ({report_res.status_code})")
        st.info("Check if your Service Principal (App) is a Member of the Power BI Workspace.")
        st.write("Full Error from Microsoft:", report_res.text)
        return None, None

    embed_data = report_res.json()

    # 3. Generate an embed token
    generate_url = f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}/reports/{REPORT_ID}/GenerateToken"
    generate_body = {"accessLevel": "View"}
    generate_res = requests.post(generate_url, headers=headers, json=generate_body)

    if generate_res.status_code != 200:
        st.error(f"Generate Embed Token Failed ({generate_res.status_code})")
        try:
            st.json(generate_res.json())
        except Exception:
            st.write(generate_res.text)
        return None, None

    embed_token = generate_res.json().get("token")
    return embed_data.get("embedUrl"), embed_token


# --- 4. POWER BI COMPONENT WITH FILTER SUPPORT ---
def render_powerbi_report(embed_token: str, embed_url: str, airline_filter: str = None, date_from: str = None, date_to: str = None):
    """
    Render Power BI report with optional airline filter applied on load.
    """
    # Prepare filter JavaScript: build combined filters (airline basic + date advanced) when provided
    # Date values passed from Python are ISO-like YYYY-MM-DD or None
    df_val = date_from or ""
    dt_val = date_to or ""
    af_val = airline_filter or ""
    filter_js = f"""
        report.on('loaded', async function() {{
            try {{
                // Track which columns were handled by slicer visuals
                let airlineSlicerFound = false;
                let dateSlicerFound = false;

                // --- STEP 1: Apply filters via slicer visuals (preferred) ---
                try {{
                    const pages = await report.getPages();
                    for (const p of pages) {{
                        const visuals = await p.getVisuals();
                        for (const v of visuals) {{
                            if (v.type === 'slicer') {{
                                try {{
                                    const state = await v.getSlicerState();
                                    const targets = state.targets || [];

                                    for (const target of targets) {{
                                        // --- Airline slicer ---
                                        if (target.column === 'AIRLINE_NAME') {{
                                            airlineSlicerFound = true;
                                            try {{
                                                if ("{af_val}" && "{af_val}" !== "") {{
                                                    const airlineFilter = {{
                                                        $schema: "http://powerbi.com/product/schema#basic",
                                                        target: {{ table: target.table, column: target.column }},
                                                        operator: "In",
                                                        values: ["{af_val}"],
                                                        filterType: models.FilterType.BasicFilter
                                                    }};
                                                    await v.setSlicerState({{ filters: [airlineFilter] }});
                                                    console.log('Airline slicer synced:', '{af_val}');
                                                }} else {{
                                                    await v.setSlicerState({{ filters: [] }});
                                                    console.log('Airline slicer cleared');
                                                }}
                                            }} catch (err) {{
                                                console.warn('Airline slicer sync failed:', err);
                                                airlineSlicerFound = false;
                                            }}
                                        }}

                                        // --- Date range slicer ---
                                        if (target.column === 'FLIGHT_DATE') {{
                                            dateSlicerFound = true;
                                            try {{
                                                if ("{df_val}" || "{dt_val}") {{
                                                    // Try multiple date value formats for compatibility
                                                    let filterApplied = false;
                                                    const formats = [
                                                        // Format 1: ISO datetime strings (most compatible)
                                                        () => {{
                                                            const filter = {{
                                                                $schema: "http://powerbi.com/product/schema#advanced",
                                                                target: {{ table: target.table, column: target.column }},
                                                                logicalOperator: "And",
                                                                conditions: [],
                                                                filterType: models.FilterType.AdvancedFilter
                                                            }};
                                                            if ("{df_val}") filter.conditions.push({{ operator: "GreaterThanOrEqual", value: "{df_val}T00:00:00" }});
                                                            if ("{dt_val}") filter.conditions.push({{ operator: "LessThanOrEqual", value: "{dt_val}T23:59:59" }});
                                                            return filter;
                                                        }},
                                                        // Format 2: Date objects
                                                        () => {{
                                                            const filter = {{
                                                                $schema: "http://powerbi.com/product/schema#advanced",
                                                                target: {{ table: target.table, column: target.column }},
                                                                logicalOperator: "And",
                                                                conditions: [],
                                                                filterType: models.FilterType.AdvancedFilter
                                                            }};
                                                            if ("{df_val}") filter.conditions.push({{ operator: "GreaterThanOrEqual", value: new Date("{df_val}T00:00:00Z") }});
                                                            if ("{dt_val}") filter.conditions.push({{ operator: "LessThanOrEqual", value: new Date("{dt_val}T23:59:59Z") }});
                                                            return filter;
                                                        }},
                                                        // Format 3: Millisecond timestamps
                                                        () => {{
                                                            const filter = {{
                                                                $schema: "http://powerbi.com/product/schema#advanced",
                                                                target: {{ table: target.table, column: target.column }},
                                                                logicalOperator: "And",
                                                                conditions: [],
                                                                filterType: models.FilterType.AdvancedFilter
                                                            }};
                                                            if ("{df_val}") filter.conditions.push({{ operator: "GreaterThanOrEqual", value: new Date("{df_val}T00:00:00Z").getTime() }});
                                                            if ("{dt_val}") filter.conditions.push({{ operator: "LessThanOrEqual", value: new Date("{dt_val}T23:59:59Z").getTime() }});
                                                            return filter;
                                                        }}
                                                    ];

                                                    for (let i = 0; i < formats.length && !filterApplied; i++) {{
                                                        try {{
                                                            const dateFilter = formats[i]();
                                                            console.log('Attempting date filter format', i + 1, ':', JSON.stringify(dateFilter, null, 2));
                                                            await v.setSlicerState({{ filters: [dateFilter] }});
                                                            console.log('✓ Date slicer synced with format', i + 1, ':', '{df_val}', 'to', '{dt_val}');
                                                            filterApplied = true;
                                                        }} catch (err) {{
                                                            console.warn('Date filter format', i + 1, 'failed:', err.message);
                                                        }}
                                                    }}

                                                    if (!filterApplied) {{
                                                        console.error('All date filter formats failed');
                                                        dateSlicerFound = false;
                                                    }}
                                                }} else {{
                                                    await v.setSlicerState({{ filters: [] }});
                                                    console.log('Date slicer cleared');
                                                }}
                                            }} catch (err) {{
                                                console.error('Date slicer sync failed:', err);
                                                dateSlicerFound = false;
                                            }}
                                        }}
                                    }}
                                }} catch (err) {{
                                    console.warn('Could not read slicer state:', v.name || v.title, err);
                                }}
                            }}
                        }}
                    }}
                }} catch (err) {{
                    console.warn('Slicer discovery error:', err);
                }}

                // --- STEP 2: Fallback to report-level filters for columns without slicers ---
                const reportFilters = [];

                if (!airlineSlicerFound && "{af_val}" && "{af_val}" !== "") {{
                    reportFilters.push({{
                        $schema: "http://powerbi.com/product/schema#basic",
                        target: {{ table: "gold_aviation_report", column: "AIRLINE_NAME" }},
                        operator: "In",
                        values: ["{af_val}"],
                        filterType: models.FilterType.BasicFilter
                    }});
                    console.log('Airline filter applied at report level (no slicer found)');
                }}

                if (!dateSlicerFound && ("{df_val}" || "{dt_val}")) {{
                    const dateFilter = {{
                        $schema: "http://powerbi.com/product/schema#advanced",
                        target: {{ table: "gold_aviation_report", column: "FLIGHT_DATE" }},
                        logicalOperator: "And",
                        conditions: [],
                        filterType: models.FilterType.AdvancedFilter
                    }};
                    if ("{df_val}") {{
                        dateFilter.conditions.push({{ operator: "GreaterThanOrEqual", value: "{df_val}T00:00:00" }});
                    }}
                    if ("{dt_val}") {{
                        dateFilter.conditions.push({{ operator: "LessThanOrEqual", value: "{dt_val}T23:59:59" }});
                    }}
                    if (dateFilter.conditions.length > 0) {{
                        reportFilters.push(dateFilter);
                        console.log('Date filter applied at report level (no slicer found):', JSON.stringify(dateFilter));
                    }}
                }}

                if (reportFilters.length > 0) {{
                    await report.setFilters(reportFilters);
                }} else if (!"{af_val}" && !"{df_val}" && !"{dt_val}") {{
                    await report.removeFilters();
                    console.log('All filters cleared');
                }}
            }} catch (err) {{
                console.error('Filter application error:', err);
                console.log('Error details:', err.message);
            }}
        }});
    """
    
    pbi_html = f"""
    <div id="reportContainer" style="height: 600px; width: 100%;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/powerbi-client/2.19.1/powerbi.min.js"></script>
    <script>
        var models = window['powerbi-client'].models;
        var config = {{
            type: 'report',
            tokenType: models.TokenType.Embed,
            accessToken: '{embed_token}',
            embedUrl: '{embed_url}',
            id: '{REPORT_ID}',
            settings: {{
                filterPaneEnabled: false,
                navContentPaneEnabled: true
            }}
        }};

        var reportContainer = document.getElementById('reportContainer');
        var report = powerbi.embed(reportContainer, config);
        
        {filter_js}
    </script>
    """
    components.html(pbi_html, height=650)

# --- 5. STREAMLIT UI ---
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