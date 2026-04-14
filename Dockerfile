# ─────────────────────────────────────────────────────────────────────────────
# Student Academic Performance Predictor — Docker Image
# ─────────────────────────────────────────────────────────────────────────────

# 1. Base image — slim Python 3.11 keeps the image small
FROM python:3.11-slim

# 2. Metadata
LABEL maintainer="Sundara Raghav"
LABEL description="Student Academic Prediction using Ensemble Learning (Flask + Gunicorn)"

# 3. Set working directory inside the container
WORKDIR /app

# 4. Install OS-level dependencies required by psycopg2-binary
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy and install Python dependencies first (layer cache advantage)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the entire project into the image
COPY . .

# 7. Expose the Flask/Gunicorn port
EXPOSE 5000

# 8. Environment defaults (override at runtime with --env or --env-file)
ENV FLASK_ENV=production \
    FLASK_DEBUG=false \
    PYTHONUNBUFFERED=1

# 9. Start with Gunicorn (production WSGI server) — 2 workers, threaded
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]
