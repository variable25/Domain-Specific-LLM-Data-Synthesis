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

CHUNK_SIZE = 384
CHUNK_OVERLAP = 64

# RAG / API
COLLECTION = os.getenv("COLLECTION", f"papers_c{CHUNK_SIZE}_o{CHUNK_OVERLAP}")
TOP_K = int(os.getenv("TOP_K", "5"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai (paid, small model) | groq (free)
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "1024"))  # caps per-query spend
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
# local spend guard; USD per 1M tokens for OPENAI_MODEL (gpt-5-mini list price)
OPENAI_BUDGET_USD = float(os.getenv("OPENAI_BUDGET_USD", "2.0"))
OPENAI_PRICE_IN = float(os.getenv("OPENAI_PRICE_IN", "0.25"))
OPENAI_PRICE_OUT = float(os.getenv("OPENAI_PRICE_OUT", "2.0"))
SPEND_PATH = DATA_DIR / "openai_spend.json"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
