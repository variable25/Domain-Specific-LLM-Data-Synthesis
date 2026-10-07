"""ETL CLI: fetch + parse/clean (in memory) -> chunk -> embed (Chroma) -> metadata (Postgres)."""
import argparse
from collections import Counter

from sdrag import chunk, db, embed, fetch
from sdrag.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_PAPERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-papers", type=int, default=MAX_PAPERS)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--subset", type=int, default=0, help="index only the first N papers")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-db", action="store_true")
    args = ap.parse_args()

    papers = fetch.load_meta() if args.skip_fetch else fetch.fetch(args.max_papers)
    if args.subset:
        papers = papers[: args.subset]
    chunks = chunk.chunk_papers(papers, args.chunk_size, args.overlap)
    counts = Counter(c["metadata"]["arxiv_id"] for c in chunks)
    for p in papers:
        p["n_chunks"] = counts[p["arxiv_id"]]
    name = embed.collection_name(args.chunk_size, args.overlap)
    n = embed.index_chunks(chunks, name)
    print(f"{len(papers)} papers -> {n} chunks in Chroma collection '{name}'")
    if not args.skip_db:
        print(f"{db.upsert_papers(papers)} papers upserted to Postgres")


if __name__ == "__main__":
    main()
