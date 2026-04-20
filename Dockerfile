FROM condaforge/miniforge3:latest

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCascade and python-magic
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglu1-mesa \
    libxi6 \
    libxrender1 \
    libxrandr2 \
    libxcursor1 \
    libxinerama1 \
    libmagic1 \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Install OCP and CadQuery via conda (not available via pip on Linux)
# OCP 7.7.2 requires Python <=3.12, so we pin python=3.11
RUN mamba install -y -c conda-forge python=3.11 cadquery=2.4.0 && mamba clean -afy

# Copy requirements and install remaining Python dependencies via pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app

# Create necessary directories
RUN mkdir -p /app/uploads /app/temp /app/outputs

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
