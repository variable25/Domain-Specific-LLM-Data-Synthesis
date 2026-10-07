"""Embed chunks with local BGE on GPU and store them in ChromaDB."""
from functools import lru_cache

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from sdrag.config import CHROMA_DIR, EMBED_MODEL


@lru_cache
def model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(EMBED_MODEL, device=device)
    if device == "cuda":
        m.half()
    return m


def collection_name(chunk_size: int, overlap: int, subset: int = 0) -> str:
    # subset indexes get their own suffix so an ablation never overwrites the full index
    return f"papers_c{chunk_size}_o{overlap}" + (f"_s{subset}" if subset else "")


def index_chunks(chunks: list[dict], name: str, batch: int = 512) -> int:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    for i in tqdm(range(0, len(chunks), batch), desc=f"embed {name}"):
        part = chunks[i : i + batch]
        vecs = model().encode([c["text"] for c in part], batch_size=64, normalize_embeddings=True)
        col.add(ids=[c["id"] for c in part], embeddings=vecs.tolist(),
                documents=[c["text"] for c in part], metadatas=[c["metadata"] for c in part])
    return col.count()
