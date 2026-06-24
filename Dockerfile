# Use official slim Python runtime as parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

# Set working directory inside the container
WORKDIR /app

# Install system dependencies for OpenCV, FFMPEG, and graphics libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to working directory
COPY requirements.txt .

# Install python package requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application and source files
COPY app/ ./app/
COPY src/ ./src/
COPY configs/ ./configs/
# Copy metadata for pre-logged mlflow runs if they are saved in git
COPY mlruns/ ./mlruns/

# Create folders that are used by the scripts
RUN mkdir -p data models outputs artifacts

# Expose port for Streamlit
EXPOSE 8501

# Set healthcheck for the container
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit application automatically
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
