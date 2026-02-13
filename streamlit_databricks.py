import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
import datetime
import json

# Manual date selector removed per request


AIRLINE_MAPPING = {
    "United": "United Air Lines Inc.",
    "American": "American Airlines Inc.",
    "US Airways": "US Airways Inc.",
    "Frontier": "Frontier Airlines Inc.",
    "JetBlue": "JetBlue Airways",
    "Skywest": "Skywest Airlines Inc.",
    "Alaska": "Alaska Airlines Inc.",
    "Spirit": "Spirit Air Lines",
    "Southwest": "Southwest Airlines Co.",
    "Delta": "Delta Air Lines Inc.",
    "Atlantic Southeast": "Atlantic Southeast Airlines",
    "Hawaiian": "Hawaiian Airlines Inc.",
    "American Eagle": "American Eagle Airlines Inc.",
    "Virgin": "Virgin America"
}

# 1. Load the vault
load_dotenv()

# 2. Assign variables from .env
# This keeps your keys invisible to anyone reading the code
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
version = os.getenv("AZURE_OPENAI_API_VERSION")

# 3. Initialize your AI Engine
llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    azure_deployment=deployment,
    api_version=version
)

# 4. Initialize Databricks Connection
# We build the connection string using environment variables
db_host = os.getenv("DATABRICKS_HOST")
db_path = os.getenv("DATABRICKS_HTTP_PATH")
db_token = os.getenv("DATABRICKS_TOKEN")
db_catalog = os.getenv("DATABRICKS_CATALOG")
db_schema = os.getenv("DATABRICKS_SCHEMA")

# Update the URI to include catalog and schema
db_uri = (
    f"databricks://token:{db_token}@{db_host}?"
    f"http_path={db_path}&catalog={db_catalog}&schema={db_schema}"
)
db = SQLDatabase.from_uri(db_uri)

# 5. Create the Agent
agent_executor = create_sql_agent(llm, db=db, verbose=True)

st.title("Aviation Intelligence Hub")
st.write("Securely connected to Azure and Databricks.")



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the Aviation Command Center! Ask me anything about flight performance or delays."}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def generate_advanced_url(user_query, date_tuple):
    base_url = os.getenv("PBI_REPORT_URL")
    filters = []

    # 1. Identify Airline
    selected_airline = None
    for common_name, formal_name in AIRLINE_MAPPING.items():
        if common_name.lower() in user_query.lower():
            selected_airline = formal_name
            break

    if selected_airline:
        # Encode spaces as %20 for the value
        encoded_val = selected_airline.replace(" ", "%20")
        filters.append(f"gold_aviation_report/AIRLINE_NAME%20eq%20%27{encoded_val}%27")

    # 2. Add Date Range (Standard OData V4 format)
    if len(date_tuple) == 2:
        start, end = date_tuple
        # Dates in URL should be YYYY-MM-DD
        date_filter = (
            f"gold_aviation_report/FLIGHT_DATE%20ge%20{start}%20and%20"
            f"gold_aviation_report/FLIGHT_DATE%20le%20{end}"
        )
        filters.append(date_filter)

    if filters:
        # Combine with 'and'
        full_filter_str = "%20and%20".join(filters)
        return f"{base_url}&$filter={full_filter_str}"
    
    return base_url

def extract_dates_from_prompt(user_input):
    """Uses the LLM to find start and end dates in a sentence."""
    extraction_prompt = f"""
    You are a data assistant. Extract the start and end dates from this query: "{user_input}"
    The dataset is from the year 2015. 
    If the user mentions a month, use the first and last day of that month in 2015.
    If no dates are mentioned, return "None".
    
    Return ONLY a JSON object: {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or "None".
    """
    
    response = llm.invoke(extraction_prompt).content
    
    if "None" in response or "{" not in response:
        return None
    try:
        # Clean the string in case the LLM added markdown code blocks
        clean_json = response.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except:
        return None

# Initialize session state for URL and Chat
if "current_url" not in st.session_state:
    st.session_state.current_url = os.getenv("PBI_REPORT_URL")
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    if prompt := st.chat_input("Ex: Show me United flights in the first week of January"):
        # 1. SMART DATE EXTRACTION
        detected_dates = extract_dates_from_prompt(prompt)
        
        # 2. UPDATE THE URL
        # If dates were found, we use them. If not, we keep the existing or default view.
        if detected_dates:
            start_val = detected_dates['start']
            end_val = detected_dates['end']
            
            # Use your generate_advanced_url logic but pass the detected dates
            st.session_state.current_url = generate_advanced_url(prompt, (start_val, end_val))
        else:
            # If no dates in prompt, we just slice by Airline (if mentioned)
            st.session_state.current_url = generate_advanced_url(prompt, [])

        # 3. CHAT AS USUAL
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = agent_executor.invoke({"input": prompt})
            st.markdown(response["output"])
            st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        
        st.rerun()

with col2:
    st.subheader("📊 Live Dashboard")
    # This ensures the dashboard has a default view before chatting
    if not st.session_state.messages:
        st.session_state.current_url = generate_advanced_url("", [])
    
    st.components.v1.iframe(st.session_state.current_url, height=800, scrolling=True)