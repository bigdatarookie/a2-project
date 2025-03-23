FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter
RUN pip install --no-cache-dir jupyter

# Create directory for notebooks
RUN mkdir -p /app/notebooks

# Create volume mount point for a1-data
RUN mkdir -p /app/data

# Expose Jupyter port
EXPOSE 8888

# Set environment variables
ENV PYTHONPATH=/app
ENV JUPYTER_ENABLE_LAB=yes

# Start Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app/notebooks"] 