# Build from the REPO ROOT:  docker build -f app/Dockerfile -t btc-forecast .
# ---------- stage 1: build the virtualenv (compilers/caches stay here) ----------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY app/requirements.txt .
RUN pip install -r requirements.txt

# ---------- stage 2: slim runtime, non-root ----------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data \
    PORT=5000
RUN groupadd --system --gid 1000 app \
 && useradd  --system --uid 1000 --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# code last => dependency layers stay cached when only code changes
COPY app/app.py ./app.py
COPY src/ ./src/
RUN mkdir -p /app/models /app/data && chown -R app:app /app
USER 1000:1000
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','5000'), timeout=2)" || exit 1
# 1 worker + 4 threads keeps RAM inside the 512Mi pod limit
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 60 --access-logfile - app:app"]
