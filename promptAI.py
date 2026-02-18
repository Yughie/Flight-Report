from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Full LLM for insight analysis (higher token budget)
llm = AzureChatOpenAI(
    model=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview'),
)

# Fast LLM for intent classification (structured output via tool calling)
llm_fast = AzureChatOpenAI(
    model=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview'),
    max_tokens=300,
    temperature=0,
)

