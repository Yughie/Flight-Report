import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Power BI / Azure config
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

__all__ = [
    "TENANT_ID",
    "CLIENT_ID",
    "CLIENT_SECRET",
    "WORKSPACE_ID",
    "REPORT_ID",
    "VALID_AIRLINES",
    "DATA_MIN_DATE",
    "DATA_MAX_DATE",
]
