"""Fetch self-driving-car papers from arXiv. PDFs are parsed in memory; only cleaned text is stored."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import arxiv
import requests
from tqdm import tqdm

from sdrag.config import ARXIV_QUERY, DATA_DIR, META_PATH, PDF_DIR, TEXT_DIR
from sdrag.parse import clean, pdf_to_text

SEARCH_CACHE = DATA_DIR / "search.jsonl"
MIN_CHARS = 2_000


def search(max_papers: int) -> list[dict]:
    if SEARCH_CACHE.exists():
        papers = [json.loads(line) for line in SEARCH_CACHE.open(encoding="utf-8")]
        if len(papers) >= max_papers:
            return papers[:max_papers]
    client = arxiv.Client(page_size=200, delay_seconds=3, num_retries=5)
    query = arxiv.Search(query=ARXIV_QUERY, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance)
    papers = []
    for r in client.results(query):
        papers.append({
            "arxiv_id": r.get_short_id(),
            "title": " ".join(r.title.split()),
            "authors": [a.name for a in r.authors],
            "abstract": " ".join(r.summary.split()),
            "published": r.published.date().isoformat(),
            "categories": r.categories,
            "pdf_url": r.pdf_url.replace("arxiv.org", "export.arxiv.org"),
        })
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(SEARCH_CACHE, papers)
    return papers


def _get_pdf(url: str) -> bytes | None:
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "sdrag-research/0.1"})
            if resp.ok and resp.content[:4] == b"%PDF":
                return resp.content
        except requests.RequestException:
            pass
        time.sleep(3 * (attempt + 1))
    return None


def _ingest(paper: dict) -> int:
    """Download -> parse -> clean in memory; write only the text. Returns text length (0 = failed)."""
    out = TEXT_DIR / f"{paper['arxiv_id']}.txt"
    if not out.exists():
        local_pdf = PDF_DIR / f"{paper['arxiv_id']}.pdf"  # leftovers from an earlier run
        data = local_pdf.read_bytes() if local_pdf.exists() else _get_pdf(paper["pdf_url"])
        if data is None:
            return 0
        try:
            out.write_text(clean(pdf_to_text(data)), encoding="utf-8")
        except Exception as e:  # corrupt PDF
            print(f"skip {paper['arxiv_id']}: {e}")
            return 0
        finally:
            local_pdf.unlink(missing_ok=True)
    return out.stat().st_size


def fetch(max_papers: int, workers: int = 4) -> list[dict]:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    papers = search(max_papers)
    with ThreadPoolExecutor(workers) as pool:
        sizes = list(tqdm(pool.map(_ingest, papers), total=len(papers), desc="ingest"))
    kept = []
    for p, n in zip(papers, sizes):
        if n > MIN_CHARS:
            p["n_chars"] = n
            kept.append(p)
    _write_jsonl(META_PATH, kept)
    return kept


def load_meta() -> list[dict]:
    with META_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _write_jsonl(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
