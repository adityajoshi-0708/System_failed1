# ---- Base Image ----
FROM python:3.11-slim

# ---- Set environment variables ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# ---- Set working directory ----
WORKDIR /app

# ---- Install system dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---- Create virtual environment & install dependencies ----
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---- Copy project files ----
COPY . .

# ---- Run the app ----
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
