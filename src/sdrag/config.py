import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PDF_DIR = DATA_DIR / "pdfs"
TEXT_DIR = DATA_DIR / "text"
META_PATH = DATA_DIR / "papers.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://sdrag:sdrag@localhost:5433/sdrag")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# arXiv search: self-driving / autonomous driving papers
ARXIV_QUERY = (
    '(abs:"autonomous driving" OR abs:"self-driving" OR abs:"autonomous vehicle") '
    "AND (cat:cs.RO OR cat:cs.CV OR cat:cs.LG OR cat:cs.AI)"
)
MAX_PAPERS = int(os.getenv("MAX_PAPERS", "500"))

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
