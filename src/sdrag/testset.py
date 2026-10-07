"""Synthetic QA test set: one grounded question per sampled paper passage.

Passages come from the cleaned paper text (not from chunks), so the test set is independent of the
chunking config being evaluated.  Run: python -m sdrag.testset --n 50 --subset 100
"""
import argparse
import json
import random
import re
from pathlib import Path

from llama_index.core.llms import LLM

from sdrag.config import TEXT_DIR
from sdrag.fetch import load_meta

TESTSET_PATH = Path("eval/testset.jsonl")
MIN_PASSAGE, MAX_PASSAGE = 1200, 2500

GEN_PROMPT = (
    "You write evaluation questions for a QA system over self-driving car research papers.\n"
    "Paper title: {title}\n\nPassage:\n---\n{passage}\n---\n\n"
    "Write ONE question a researcher might ask that is answerable from this passage alone, and a "
    "concise reference answer (1-3 sentences) using only facts stated in the passage.\n"
    "Rules: the question must be self-contained and specific (name the method, dataset or concept); "
    "the reader never sees the passage, so never write 'the passage', 'this paper', 'the authors', "
    "'the proposed method', and never refer to figures, tables, equations or sections. If the passage is unsuitable (mostly math, references, "
    'garbled text, acknowledgements), return {{"skip": true}}.\n'
    'Return only JSON: {{"question": "...", "answer": "..."}}'
)

# phrases that make a question unanswerable without knowing which paper it came from
_VAGUE = re.compile(r"\b(this|the present|the proposed) (paper|study|work|method|approach)\b|\bthe authors\b|"
                    r"\b(the|this) (passage|excerpt|text)\b|\b(figure|fig\.|table|eq\.|equation|section)\s*\d", re.IGNORECASE)
_JSON = re.compile(r"\{.*\}", re.DOTALL)


def pick_passage(text: str, rng: random.Random) -> str:
    """Consecutive paragraphs from a random start, at least MIN_PASSAGE chars when the text allows."""
    paras = [p for p in text.split("\n\n") if p.strip()]
    start = rng.randrange(max(1, len(paras) - 3))
    out = []
    for p in paras[start:]:
        out.append(p)
        if sum(map(len, out)) >= MIN_PASSAGE:
            break
    return "\n\n".join(out)[:MAX_PASSAGE]


def parse_qa(raw: str) -> dict | None:
    """LLM output -> {'question', 'answer'}, or None for a skip / malformed / vague question."""
    m = _JSON.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except ValueError:
        return None
    q, a = str(obj.get("question", "")).strip(), str(obj.get("answer", "")).strip()
    if obj.get("skip") or len(q) < 15 or not a or _VAGUE.search(q):
        return None
    return {"question": q, "answer": a}


def generate(llm: LLM, papers: list[dict], n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for p in rng.sample(papers, len(papers)):  # shuffled; stop once n questions are kept
        if len(rows) == n:
            break
        passage = pick_passage((TEXT_DIR / f"{p['arxiv_id']}.txt").read_text(encoding="utf-8"), rng)
        qa = parse_qa(llm.complete(GEN_PROMPT.format(title=p["title"], passage=passage)).text)
        if qa:
            rows.append({"id": f"q{len(rows) + 1:02d}", "question": qa["question"], "reference": qa["answer"],
                         "arxiv_ids": [p["arxiv_id"]], "title": p["title"], "passage": passage})
    return rows


def load(path: Path = TESTSET_PATH) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def main():
    from sdrag import budget
    from sdrag.rag import make_llm

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--subset", type=int, default=100, help="sample from the first N papers (the ablation subset)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    before = budget.spent()
    rows = generate(make_llm("openai"), load_meta()[: args.subset], args.n, args.seed)
    TESTSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    TESTSET_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} questions -> {TESTSET_PATH} (cost ${budget.spent() - before:.4f})")


if __name__ == "__main__":
    main()
