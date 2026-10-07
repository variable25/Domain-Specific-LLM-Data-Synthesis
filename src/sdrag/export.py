"""Copy only the serving collection + the papers.jsonl metadata snapshot into a slim dir for the Docker image.

    python -m sdrag.export --out build/data
"""
import argparse
import shutil
from pathlib import Path

import chromadb

from sdrag.config import CHROMA_DIR, COLLECTION, META_PATH


def export(out: Path, name: str = COLLECTION, batch: int = 2000) -> int:
    if out.exists():
        shutil.rmtree(out)
    src = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(name)
    dst = chromadb.PersistentClient(path=str(out / "chroma")).create_collection(name, metadata=src.metadata)
    for offset in range(0, src.count(), batch):
        part = src.get(limit=batch, offset=offset, include=["embeddings", "documents", "metadatas"])
        dst.add(ids=part["ids"], embeddings=part["embeddings"],
                documents=part["documents"], metadatas=part["metadatas"])
    if dst.count() != src.count():
        raise RuntimeError(f"exported {dst.count()} of {src.count()} chunks")
    shutil.copy(META_PATH, out / "papers.jsonl")
    return dst.count()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("build/data"))
    ap.add_argument("--collection", default=COLLECTION)
    args = ap.parse_args()
    print(f"exported {export(args.out, args.collection)} chunks of {args.collection} to {args.out}")
