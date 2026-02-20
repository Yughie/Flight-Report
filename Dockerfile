# Use a stable Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install only the essential system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . .

# Streamlit port
EXPOSE 8501

# Command to run the app with security flags for Azure
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]