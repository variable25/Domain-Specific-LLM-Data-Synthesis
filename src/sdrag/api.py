"""FastAPI service: POST /query -> cited answer.  Run: uvicorn sdrag.api:app"""
from functools import lru_cache

from fastapi import Depends, FastAPI
from llama_index.core.llms import LLM
from pydantic import BaseModel, Field

from sdrag import rag
from sdrag.config import TOP_K


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


app = FastAPI(title="sdrag", description="Cited QA over arXiv self-driving papers")


def get_index():
    return rag.default_index()


@lru_cache
def get_llm() -> LLM:
    return rag.make_llm()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, index=Depends(get_index), llm: LLM = Depends(get_llm)):
    return rag.answer(rag.build_engine(index, llm, req.top_k), req.question)
