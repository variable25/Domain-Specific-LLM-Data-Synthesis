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

## ETL (Stage 1)

```bash
python -m sdrag.pipeline --max-papers 520          # fetch arXiv -> parse -> chunk -> embed -> Postgres
python -m sdrag.pipeline --skip-fetch --chunk-size 256 --overlap 32 --subset 100 --skip-db   # ablation index
```

- Source: arXiv API (cs.RO / cs.CV / cs.LG / cs.AI, "autonomous driving" / "self-driving" / "autonomous vehicle")
- Parse: PyMuPDF on in-memory PDF bytes (no PDFs stored, only cleaned text); clean: de-hyphenation, page numbers, arXiv stamps, reference list removed
- Chunk: LlamaIndex `SentenceSplitter` counted with the BGE tokenizer (default 384 tokens / 64 overlap)
- Embed: local `BAAI/bge-base-en-v1.5` (fp16, CUDA) -> ChromaDB collection `papers_c{size}_o{overlap}`
- Metadata: Postgres `papers` table (docker compose, host port 5433)

## RAG API (Stage 2)

```bash
uvicorn sdrag.api:app --port 8000
curl -X POST localhost:8000/query -H "Content-Type: application/json" -d '{"question": "Which sensors are fused for 3D object detection?"}'
```

- Retrieval: LlamaIndex over the existing Chroma collection (`papers_c384_o64`), queries embedded with the same local BGE model (+ BGE query prefix), top-k 5
- LLM: `LLM_PROVIDER=openai` (`gpt-5-mini`, minimal reasoning) or `groq` (free, `openai/gpt-oss-120b`)
- Answers cite `[arxiv_id]`; response has `answer`, `citations` (only ids that were actually retrieved) and `sources` (id, title, chunk, score, text)
- Spend guard: every OpenAI call is priced from its token usage into `data/openai_spend.json`; calls stop (HTTP 402) at `OPENAI_BUDGET_USD` (default $2, ~$0.0007/query)
- Tests run offline (in-memory Chroma, fake LLM); `RUN_LIVE=1 pytest -k live` hits the real index + LLM
