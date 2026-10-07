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

## Evaluation (Stage 3)

```bash
python -m sdrag.testset --n 50 --subset 100                    # synthetic test set -> eval/testset.jsonl
python -m sdrag.pipeline --skip-fetch --skip-db --subset 100 --chunk-size 256 --overlap 32   # ablation indexes (also 384/64, 512/96)
python -m sdrag.evaluate --collection papers_c384_o64 --ragas 50          # full index
python -m sdrag.evaluate --collection papers_c256_o32_s100 --ragas 20     # each ablation index
python -m sdrag.evaluate --report                                         # markdown tables below
```

- Test set: 50 questions, one per paper, from 50 of the 100 ablation-subset papers. gpt-5-mini writes a question + 1-3 sentence reference answer from a ~1.2-2.5k char passage of the cleaned text (not a chunk, so no chunking config is favoured); vague questions ("this paper", "the passage", "Table 3") are filtered out. Each row keeps its source `arxiv_ids` and passage.
- Retrieval: hit-rate@5 = a chunk from the source paper is in the top 5; MRR on its rank. Local BGE only, so all 50 questions are scored for every config at no cost.
- RAGAS 0.4 (faithfulness, answer relevancy, context precision/recall with reference): answers come from the Stage 2 engine (gpt-5-mini); the judge is gpt-5-mini behind the same spend guard (custom RAGAS LLM wrapper calling `BudgetedOpenAI`), answer-relevancy embeddings are the local BGE. To stay within the budget, RAGAS covers all 50 questions on the full index and the first 20 on each ablation index. Answers are cached in `data/eval/` so a rerun doesn't pay for them again.
- Ablation collections are suffixed `_s100` so they never overwrite the full index.

### Results

Top-k 5, answers and judge gpt-5-mini. Full index (517 papers):

| Collection | Chunks | Hit@k | MRR | RAGAS n | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `papers_c384_o64` | 15900 | 0.76 | 0.70 | 50 | 0.800 | 0.572 | 0.669 | 0.703 |

Ablation (RAGAS on the same first 20 questions; hit-rate/MRR on all 50):

| Collection | Chunks | Hit@k | MRR | RAGAS n | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `papers_c256_o32_s100` | 4366 | 0.96 | 0.90 | 20 | 0.812 | 0.763 | 0.766 | 0.975 |
| `papers_c384_o64` | 15900 | 0.76 | 0.70 | 20 | 0.820 | 0.616 | 0.596 | 0.700 |
| `papers_c384_o64_s100` | 3044 | 0.90 | 0.81 | 20 | 0.801 | 0.649 | 0.653 | 0.750 |
| `papers_c512_o96_s100` | 2333 | 0.90 | 0.86 | 20 | 0.744 | 0.674 | 0.714 | 0.775 |

- **256/32 wins the ablation on retrieval** (hit-rate 0.96 vs 0.90, MRR 0.90 vs 0.81-0.86) and on context recall/precision. Small chunks give the BGE embedding a more focused passage to match a specific factual question against.
- 512/96 trades faithfulness for context precision: longer chunks hand the LLM more side material to drift into.
- The full index scores lower than the same config on the 100-paper subset (hit-rate 0.76 vs 0.90) most likely because 5x more papers means many more near-topic distractors (e.g. dozens of CARLA or nuScenes setups). This is the realistic number.
- Answer relevancy is pulled down by honest "the excerpts don't say" answers, which RAGAS scores as 0 (non-committal).
- Caveat: the RAGAS ablation uses 20 questions, so differences under ~0.05 are noise; the hit-rate/MRR over 50 is the firmer signal. Total OpenAI cost for Stage 3 (test set, pilot, 4 RAGAS runs, re-scoring) was ~$0.92.
