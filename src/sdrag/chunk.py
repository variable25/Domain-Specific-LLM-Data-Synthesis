"""Token-aware chunking using the embedding model's own tokenizer."""
from functools import lru_cache

from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoTokenizer

from sdrag.config import EMBED_MODEL, TEXT_DIR


@lru_cache
def _tokenizer():
    return AutoTokenizer.from_pretrained(EMBED_MODEL).tokenize


def split(text: str, chunk_size: int, overlap: int) -> list[str]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap, tokenizer=_tokenizer())
    return splitter.split_text(text)


def chunk_papers(papers: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    chunks = []
    for p in papers:
        text = (TEXT_DIR / f"{p['arxiv_id']}.txt").read_text(encoding="utf-8")
        for i, piece in enumerate(split(text, chunk_size, overlap)):
            chunks.append({
                "id": f"{p['arxiv_id']}#{i}",
                "text": piece,
                "metadata": {"arxiv_id": p["arxiv_id"], "title": p["title"], "chunk": i,
                             "published": p["published"]},
            })
    return chunks
