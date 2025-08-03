# ---- Multi-stage build for smaller final image ----
FROM python:3.11-slim as builder

# ---- Set environment variables ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- Install build dependencies in builder stage ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ---- Create virtual environment & install dependencies ----
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---- Final stage with minimal runtime image ----
FROM python:3.11-slim

# ---- Set environment variables ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# ---- Install only runtime dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libpoppler-cpp0v5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# ---- Copy virtual environment from builder ----
COPY --from=builder /opt/venv /opt/venv

# ---- Set working directory ----
WORKDIR /app

# ---- Copy only necessary application files ----
COPY main.py .
COPY *.py ./
# Add specific directories if needed, avoid copying unnecessary files
# COPY src/ ./src/
# COPY config/ ./config/

# ---- Create non-root user for security ----
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# ---- Run the app ----
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]