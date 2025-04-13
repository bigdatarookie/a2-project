# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    swig \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --user -r requirements.txt \
    && pip install --user jupyter \
    && pip install --user pytest pytest-cov pytest-mock

# Runtime stage
FROM python:3.11-slim
WORKDIR /app

# Copy Python packages and requirements
COPY --from=builder /root/.local /root/.local
COPY requirements.txt .
ENV PATH=/root/.local/bin:$PATH

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

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

# Install the tdata package in development mode
WORKDIR /app
RUN pip install -e .

# Expose Jupyter and MLflow ports
EXPOSE 8888 5000

# Create a startup script
RUN echo '#!/bin/bash\n\
mlflow server --host 0.0.0.0 --port 5000 \
--backend-store-uri file:///app/data/models/mlflow_experiments \
--default-artifact-root /app/data/models/mlflow_experiments &\n\
mlflow_pid=$!\n\
python /app/run_tests.py && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app\n\
kill $mlflow_pid' > /app/start.sh && chmod +x /app/start.sh

# Set the entrypoint to run the startup script
ENTRYPOINT ["/app/start.sh"]
