import streamlit as st
import requests
import json
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. CONFIGURATION (from secrets.toml) ---
TENANT_ID = os.getenv("POWERBI_TENANT_ID")
CLIENT_ID = os.getenv("POWERBI_CLIENT_ID")
CLIENT_SECRET = os.getenv("POWERBI_CLIENT_SECRET")
WORKSPACE_ID = os.getenv("POWERBI_WORKSPACE_ID")
REPORT_ID = os.getenv("POWERBI_REPORT_ID")

# --- 2. FUNCTION TO GET ACCESS TOKEN ---
def get_embed_params():
    # 1. Get Azure AD Access Token
    auth_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    
    # Ensure these keys are exactly lowercase
    auth_data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET, 
        "scope": "https://analysis.windows.net/powerbi/api/.default"
    }
    
    # CRITICAL: Use 'data=' instead of 'json='
    # Azure expects x-www-form-urlencoded format
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
    
    # --- THIS IS WHERE YOUR ERROR IS HAPPENING ---
    if report_res.status_code != 200:
        st.error(f"Power BI API Failed ({report_res.status_code})")
        st.info("Check if your Service Principal (App) is a Member of the Power BI Workspace.")
        st.write("Full Error from Microsoft:", report_res.text) # This reveals the truth!
        return None, None

    embed_data = report_res.json()

    # 3. Generate an embed token from the Power BI REST API (do NOT pass the AAD token directly
    #    to the browser). The embed token is what the Power BI JS SDK expects on the client.
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

# --- 3. STREAMLIT UI ---
st.title("AI-Controlled Power BI Report")

# Initialize session state
if 'report_loaded' not in st.session_state:
    st.session_state.report_loaded = False
if 'embed_token' not in st.session_state:
    st.session_state.embed_token = None
if 'embed_url' not in st.session_state:
    st.session_state.embed_url = None

# Button layout
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Load Report"):
        with st.spinner("Loading Power BI Report..."):
            embed_url, token = get_embed_params()
            if embed_url and token:
                st.session_state.embed_url = embed_url
                st.session_state.embed_token = token
                st.session_state.report_loaded = True
                st.rerun()

with col2:
    if st.button("Clear Report") and st.session_state.report_loaded:
        st.session_state.report_loaded = False
        st.session_state.embed_token = None
        st.session_state.embed_url = None
        st.rerun()

# Show the embedded report if loaded
if st.session_state.report_loaded and st.session_state.embed_token and st.session_state.embed_url:
    # This is where the JavaScript "AI control" lives
    # We use st.components.v1.html to bridge Python and the Power BI JS SDK
    pbi_html = f"""
    <div id="reportContainer" style="height: 600px; width: 100%;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/powerbi-client/2.19.1/powerbi.min.js"></script>
    <script>
        var models = window['powerbi-client'].models;
        var config = {{
            type: 'report',
            tokenType: models.TokenType.Embed,
            accessToken: '{st.session_state.embed_token}',
            embedUrl: '{st.session_state.embed_url}',
            id: '{REPORT_ID}',
            settings: {{
                filterPaneEnabled: false,
                navContentPaneEnabled: true
            }}
        }};

        var reportContainer = document.getElementById('reportContainer');
        var report = powerbi.embed(reportContainer, config);

        // This function is what your "AI" will call via message passing later
        window.applyAIFilter = async function(value) {{
            const pages = await report.getPages();
            const activePage = pages.find(p => p.isActive);
            const visuals = await activePage.getVisuals();
            const slicer = visuals.find(v => v.type === 'slicer'); // Finds the first slicer

            const filter = {{
                $schema: "http://powerbi.com/product/schema#basic",
                target: {{ table: "Store", column: "Count" }}, // EDIT THESE TO MATCH YOUR DATA
                operator: "In",
                values: [value],
                filterType: models.FilterType.BasicFilter
            }};

            await slicer.setSlicerState({{ filters: [filter] }});
        }};
    </script>
    """
    components.html(pbi_html, height=650)
elif st.session_state.report_loaded:
    st.error("Failed to load report. Please try again.")