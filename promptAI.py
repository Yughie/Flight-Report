from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

llm = AzureChatOpenAI(
    model=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview'),
)

