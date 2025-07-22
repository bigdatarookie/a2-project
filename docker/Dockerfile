FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    swig \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter pytest pytest-cov pytest-mock

# Set CUDA in PATH
ENV PATH="/usr/local/cuda-12.9/bin:${PATH}"

# Create directories for MLflow and model storage
RUN mkdir -p /app/data/models/logs /app/data/models/mlflow_experiments

# Set environment variables
ENV PYTHONPATH=/app
ENV JUPYTER_ENABLE_LAB=yes
ENV MLFLOW_TRACKING_URI=file:///app/data/models/mlflow_experiments
ENV MLFLOW_ARTIFACT_ROOT=/app/data/models/mlflow_experiments

# Copy the application code
COPY packages/ /app/packages/
COPY setup.py /app/
COPY run_tests.py /app/

# Install the local package
RUN pip3 install .

# Expose Jupyter and MLflow ports
EXPOSE 8888 5000

# Create a startup script
RUN echo '#!/bin/bash\n\
mlflow server --host 0.0.0.0 --port 5000 \
--backend-store-uri file:///app/data/models/mlflow_experiments \
--default-artifact-root /app/data/models/mlflow_experiments &\n\
mlflow_pid=$!\n\
python3 /app/run_tests.py && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app\n\
kill $mlflow_pid' > /app/start.sh && chmod +x /app/start.sh

ENTRYPOINT ["/app/start.sh"]
