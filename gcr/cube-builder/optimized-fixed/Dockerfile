FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY optimized/ optimized/
COPY forge_middleware/ forge_middleware/
COPY forge-middleware.toml .
COPY warmup_build.py .
COPY app.py .

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "app:app"]

ENV NUMBA_CACHE_DIR=/app/.nv_jit_cache
RUN python warmup_build.py
