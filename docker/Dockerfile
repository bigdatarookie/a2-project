# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
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

# Create directories
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV JUPYTER_ENABLE_LAB=yes

# Copy the application code
COPY packages/ /app/packages/
COPY setup.py /app/
COPY run_tests.py /app/

# Install the tdata package in development mode
WORKDIR /app
RUN pip install -e .

# Expose Jupyter port
EXPOSE 8888

# Set the entrypoint to run tests first; the script will exit if tests fail
ENTRYPOINT ["/bin/bash", "-c", "python /app/run_tests.py && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app"]
