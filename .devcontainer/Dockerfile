FROM python:3.11-slim

# Install system packages for smoother dev experience
RUN apt-get update && apt-get install -y \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install dependencies
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source
COPY ../app ./app

# Expose the dev server port
EXPOSE 8000
