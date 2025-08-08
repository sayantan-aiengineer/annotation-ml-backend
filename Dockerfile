# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim AS python-base

ARG TEST_ENV
ARG MODEL_FOLDER_URL

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-8080} \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8

# Update the base OS (removed BuildKit --mount syntax)
RUN set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install --no-install-recommends -y \
        git \
        wget \
        curl \
        unzip; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# Install base requirements
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# Install custom requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown for Google Drive downloads
RUN pip install --no-cache-dir gdown

# Install test requirements if needed
COPY requirements-test.txt .
RUN if [ "$TEST_ENV" = "true" ]; then \
        pip install --no-cache-dir -r requirements-test.txt; \
    fi

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p /data/models

# Download entire folder from Google Drive
RUN if [ -n "$MODEL_FOLDER_URL" ]; then \
        echo "Downloading model folder from Google Drive..." && \
        gdown --folder "$MODEL_FOLDER_URL" -O /data/models && \
        echo "Model folder downloaded successfully" && \
        ls -la /data/models/; \
    else \
        echo "No MODEL_FOLDER_URL provided, skipping model download"; \
    fi

EXPOSE 8080

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
