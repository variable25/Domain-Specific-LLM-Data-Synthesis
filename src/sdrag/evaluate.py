"""Evaluate a Chroma collection on the synthetic test set.

Retrieval hit-rate@k / MRR run locally for every question (free). RAGAS (faithfulness, answer relevancy,
context precision/recall) runs on the first --ragas questions; answers and judge calls both go through
the OpenAI spend guard.  Run:
    python -m sdrag.evaluate --collection papers_c384_o64 --ragas 50
    python -m sdrag.evaluate --report
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from llama_index.core.llms import LLM

from sdrag import testset
from sdrag.config import DATA_DIR, OPENAI_MODEL, TOP_K

RESULTS_DIR = Path("eval/results")
ANSWER_CACHE = DATA_DIR / "eval"  # answers + retrieved contexts (large, gitignored)
JUDGE_MAX_TOKENS = 4096  # faithfulness emits one verdict per statement; 1024 truncates long answers
METRICS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]
LABELS = ["Faithfulness", "Answer relevancy", "Context precision", "Context recall"]


def retrieval_metrics(retriever, rows: list[dict]) -> list[dict]:
    """Per question: rank of the first retrieved chunk from a source paper (None = miss)."""
    out = []
    for r in rows:
        ids = [n.node.metadata["arxiv_id"] for n in retriever.retrieve(r["question"])]
        rank = next((i + 1 for i, a in enumerate(ids) if a in r["arxiv_ids"]), None)
        out.append({"id": r["id"], "rank": rank})
    return out


def summarize_retrieval(per_q: list[dict]) -> dict:
    return {"hit_rate": mean(q["rank"] is not None for q in per_q),
            "mrr": mean(1 / q["rank"] if q["rank"] else 0.0 for q in per_q)}


def answers(engine, rows: list[dict], cache: Path) -> list[dict]:
    """rag.answer for each question, cached on disk so a failed RAGAS run never re-pays for answers."""
    from sdrag.rag import answer

    done = {}
    if cache.exists():
        done = {a["id"]: a for a in map(json.loads, cache.open(encoding="utf-8"))}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("a", encoding="utf-8") as f:
        for r in rows:
            if r["id"] not in done:
                out = answer(engine, r["question"])
                done[r["id"]] = {"id": r["id"], "answer": out["answer"], "contexts": [s["text"] for s in out["sources"]]}
                f.write(json.dumps(done[r["id"]], ensure_ascii=False) + "\n")
                f.flush()
    return [done[r["id"]] for r in rows]


def _ragas_classes():
    from ragas.embeddings.base import BaseRagasEmbeddings
    from ragas.llms.base import BaseRagasLLM
    from langchain_core.outputs import Generation, LLMResult

    @dataclass
    class GuardedLLM(BaseRagasLLM):
        """RAGAS judge backed by a LlamaIndex LLM (BudgetedOpenAI), called with no extra params:
        gpt-5 models reject the temperature/stop/n that RAGAS' own LlamaIndex wrapper sends."""
        llm: LLM = field(default=None)

        def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            return LLMResult(generations=[[Generation(text=self.llm.complete(prompt.to_string()).text)]])

        async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            resp = await self.llm.acomplete(prompt.to_string())
            return LLMResult(generations=[[Generation(text=resp.text)]])

        def is_finished(self, response) -> bool:
            return True

    class BGESymmetric(BaseRagasEmbeddings):
        """Local BGE without the query prefix: answer relevancy compares question to question."""

        def embed_query(self, text):
            return self.embed_documents([text])[0]

        def embed_documents(self, texts):
            from sdrag.embed import model
            return model().encode(texts, normalize_embeddings=True).tolist()

        async def aembed_query(self, text):
            return self.embed_query(text)

        async def aembed_documents(self, texts):
            return self.embed_documents(texts)

    return GuardedLLM, BGESymmetric


def ragas_scores(rows: list[dict], answered: list[dict], judge: LLM) -> list[dict]:
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    GuardedLLM, BGESymmetric = _ragas_classes()
    ds = EvaluationDataset.from_list([
        {"user_input": r["question"], "reference": r["reference"], "response": a["answer"],
         "retrieved_contexts": a["contexts"]} for r, a in zip(rows, answered)])
    metrics = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference(), LLMContextRecall()]
    res = evaluate(ds, metrics=metrics, llm=GuardedLLM(llm=judge), embeddings=BGESymmetric(),
                   run_config=RunConfig(max_workers=8, timeout=300, max_retries=3))
    df = res.to_pandas()
    return [{"id": r["id"], **{m: _num(df.iloc[i][m]) for m in METRICS}} for i, r in enumerate(rows)]


def _num(x) -> float | None:
    return None if x is None or x != x else round(float(x), 4)  # NaN -> None


def _mean(per_q: list[dict], key: str) -> float | None:
    vals = [q[key] for q in per_q if q.get(key) is not None]
    return round(mean(vals), 4) if vals else None


def run(collection: str, n_ragas: int, top_k: int = TOP_K) -> dict:
    import chromadb

    from sdrag import budget, rag
    from sdrag.config import CHROMA_DIR

    rows = testset.load()
    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(collection)
    index = rag.load_index(col)
    per_q = retrieval_metrics(index.as_retriever(similarity_top_k=top_k), rows)
    result = {"collection": collection, "chunks": col.count(), "top_k": top_k, "n": len(rows),
              **summarize_retrieval(per_q), "ragas_n": n_ragas, "judge": OPENAI_MODEL}
    out_path = RESULTS_DIR / f"{collection}.json"
    before = budget.spent()
    if n_ragas:
        llm = rag.make_llm("openai")
        judge = rag.make_llm("openai")
        judge.max_tokens = JUDGE_MAX_TOKENS
        sub = rows[:n_ragas]
        answered = answers(rag.build_engine(index, llm, top_k), sub, ANSWER_CACHE / f"{collection}.jsonl")
        # resume: keep a previous run's scores; re-judge (all metrics) only questions missing one
        prev = {}
        if out_path.exists():
            prev = {q["id"]: q for q in json.loads(out_path.read_text(encoding="utf-8"))["per_question"]}
        todo = [i for i, r in enumerate(sub) if any(prev.get(r["id"], {}).get(m) is None for m in METRICS)]
        scores = {r["id"]: {m: prev.get(r["id"], {}).get(m) for m in METRICS} for r in sub}
        if todo:
            new = ragas_scores([sub[i] for i in todo], [answered[i] for i in todo], judge)
            scores.update({s["id"]: {m: s[m] if s[m] is not None else scores[s["id"]][m] for m in METRICS} for s in new})
        per_q = [{**q, **scores.get(q["id"], {})} for q in per_q]
        result["ragas"] = {m: _mean(per_q, m) for m in METRICS}
    result["cost_usd"] = round(budget.spent() - before, 4)
    result["per_question"] = per_q
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def _fmt(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def report(results: list[dict], ablation_n: int = 20) -> str:
    """Markdown: main-index table, then every collection on the shared first `ablation_n` questions."""
    lines = ["| Collection | Chunks | Hit@k | MRR | RAGAS n | " + " | ".join(LABELS) + " |",
             "|---|---:|---:|---:|---:|" + "---:|" * len(LABELS)]
    for r in results:
        if r["ragas_n"] > ablation_n:
            lines.append(f"| `{r['collection']}` | {r['chunks']} | {r['hit_rate']:.2f} | {r['mrr']:.2f} | "
                         f"{r['ragas_n']} | " + " | ".join(_fmt(r["ragas"][m]) for m in METRICS) + " |")
    lines += ["", f"Ablation (RAGAS on the same first {ablation_n} questions; hit-rate/MRR on all {results[0]['n']}):", "",
              lines[0], lines[1]]
    for r in results:
        sub = r["per_question"][:ablation_n]
        lines.append(f"| `{r['collection']}` | {r['chunks']} | {r['hit_rate']:.2f} | {r['mrr']:.2f} | "
                     f"{min(r['ragas_n'], ablation_n)} | " + " | ".join(_fmt(_mean(sub, m)) for m in METRICS) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection")
    ap.add_argument("--ragas", type=int, default=0, help="run RAGAS on the first N questions (0 = retrieval only)")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--report", action="store_true", help="print the markdown table from eval/results")
    ap.add_argument("--ablation-n", type=int, default=20)
    args = ap.parse_args()
    if args.report:
        results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob("*.json"))]
        print(report(results, args.ablation_n))
        return
    r = run(args.collection, args.ragas, args.top_k)
    print(json.dumps({k: v for k, v in r.items() if k != "per_question"}, indent=1))


if __name__ == "__main__":
    main()
