# Domain-Specific LLM Data Synthesis Platform

Cited question answering over ~500 open-access arXiv papers on self-driving cars.

Stages: ETL -> RAG API -> RAGAS evaluation + chunking ablation -> CI + Cloud Run.

## Setup

```bash
py -3.12 -m venv .venv && source .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt && pip install -e .
cp .env.example .env
docker compose up -d
```
