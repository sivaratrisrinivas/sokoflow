# CPU-only SokoFlow web UI. Weights are copied from the repo.
# Build and run:
#   docker build -t sokoflow .
#   docker run --rm -p 5000:5000 sokoflow
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=5000 \
    SOKOFLOW_MODEL_PATH=/app/sokoban_diffusion.pth

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        flask==3.0.3 \
        flask-cors==5.0.1 \
        gunicorn==23.0.0 \
        numpy==2.1.3 \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY app.py sokoban_engine.py sokoban_diffusion.py sokoban_data_gen.py ./
COPY sokoban_diffusion.pth ./
COPY templates ./templates

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import json,os,urllib.request as u; d=json.load(u.urlopen('http://127.0.0.1:%s/health'%os.environ.get('PORT','5000'))); assert d.get('model_loaded') is True"

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120 app:app"]
