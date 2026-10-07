# API image: CPU torch, local BGE weights, one Chroma collection + papers.jsonl (no Postgres).
# Build after `python -m sdrag.export --out build/data`.
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-api.txt .
# torch>=2.7: older CPU builds raise on torch.accelerator when no GPU exists, which breaks transformers 5.
# kubernetes is only used by chromadb's distributed server mode; headers/tests are build-time only.
RUN pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-api.txt \
 && pip uninstall -y kubernetes \
 && SP=/opt/venv/lib/python3.12/site-packages && rm -rf $SP/chromadb/test $SP/torch/include
# BGE saved as fp16 (the index was embedded with the fp16 model): safetensors + tokenizer/config only
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-base-en-v1.5', \
    local_dir='/tmp/bge', allow_patterns=['*.json', '*.txt', 'model.safetensors'])" \
 && python -c "from sentence_transformers import SentenceTransformer as S; \
    S('/tmp/bge', device='cpu').half().save('/opt/models/bge-base-en-v1.5')" \
 && rm -rf /tmp/bge && chmod -R a+rX /opt/models

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models
WORKDIR /app
COPY src/sdrag src/sdrag
COPY --chown=app:app build/data data
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app/src PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data EMBED_MODEL=/opt/models/bge-base-en-v1.5 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ANONYMIZED_TELEMETRY=False \
    WARMUP=1 RATE_LIMIT_PER_MIN=20 PORT=8080
USER app
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn sdrag.api:app --host 0.0.0.0 --port ${PORT}"]
