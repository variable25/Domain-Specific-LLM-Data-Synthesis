"""FastAPI service: POST /query -> cited answer.  Run: uvicorn sdrag.api:app"""
import json
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from llama_index.core.llms import LLM
from pydantic import BaseModel, Field

from sdrag import rag
from sdrag.budget import BudgetExceeded
from sdrag.config import META_PATH, RATE_LIMIT_PER_MIN, TOP_K, WARMUP


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=TOP_K, ge=1, le=20)


class Source(BaseModel):
    arxiv_id: str
    title: str | None = None
    chunk: int | None = None
    score: float | None = None
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    sources: list[Source]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if WARMUP:  # pay the model/index load before the port opens, not on the first request
        from sdrag.embed import model
        rag.default_index()
        model().encode("warmup")
    yield


app = FastAPI(title="sdrag", description="Cited QA over arXiv self-driving papers", lifespan=lifespan)


@app.exception_handler(BudgetExceeded)
def budget_exceeded(request: Request, exc: BudgetExceeded):
    return JSONResponse(status_code=402, content={"detail": str(exc)})


def get_index():
    return rag.default_index()


@lru_cache
def get_llm() -> LLM:
    return rag.make_llm()


class RateLimiter:
    """Sliding-window cap on calls per minute for this process (Cloud Run runs at most one instance)."""

    def __init__(self, per_min: int):
        self.per_min, self.calls, self.lock = per_min, deque(), threading.Lock()

    def __call__(self) -> None:
        if not self.per_min:
            return
        now = time.monotonic()
        with self.lock:
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()
            if len(self.calls) >= self.per_min:
                raise HTTPException(429, "rate limit reached, try again in a minute")
            self.calls.append(now)


rate_limit = RateLimiter(RATE_LIMIT_PER_MIN)


@lru_cache
def papers() -> dict[str, dict]:
    """Paper metadata snapshot (data/papers.jsonl), so the container needs no Postgres."""
    with META_PATH.open(encoding="utf-8") as f:
        return {p["arxiv_id"]: p for p in map(json.loads, f)}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/papers/{arxiv_id}")
def paper(arxiv_id: str):
    if arxiv_id not in papers():
        raise HTTPException(404, f"unknown arxiv_id {arxiv_id!r}")
    return papers()[arxiv_id]


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit)])
def query(req: QueryRequest, index=Depends(get_index), llm: LLM = Depends(get_llm)):
    return rag.answer(rag.build_engine(index, llm, req.top_k), req.question)
