"""Cited QA: LlamaIndex query engine over the Chroma index; OpenAI or Groq as the LLM."""
import re
from functools import lru_cache

import chromadb
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore

from sdrag.config import (
    CHROMA_DIR,
    COLLECTION,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    MAX_ANSWER_TOKENS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TOP_K,
)

# BGE v1.5 expects this prefix on queries (not on passages) for retrieval
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

QA_PROMPT = PromptTemplate(
    "You answer questions about self-driving car research using only the excerpts below.\n"
    "Each excerpt starts with its arxiv_id. After every claim, cite the supporting paper(s) "
    "as [arxiv_id], e.g. [2402.16036v1]. If the excerpts do not contain the answer, "
    "say you don't know.\n\n"
    "---------------------\n{context_str}\n---------------------\n\n"
    "Question: {query_str}\nAnswer: "
)

CITATION_RE = re.compile(r"\[(\d{4}\.\d{4,5}(?:v\d+)?)\]")


class BGEEmbedding(BaseEmbedding):
    """Reuses the ETL's local BGE model so queries live in the same vector space as the index."""

    def _get_query_embedding(self, query: str) -> list[float]:
        from sdrag.embed import model
        return model().encode(BGE_QUERY_PREFIX + query, normalize_embeddings=True).tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        from sdrag.embed import model
        return model().encode(text, normalize_embeddings=True).tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


class ArxivIdOnly(BaseNodePostprocessor):
    """Show the LLM only `arxiv_id: ...` above each chunk, so it has one id to cite."""

    def _postprocess_nodes(self, nodes, query_bundle=None):
        for n in nodes:
            n.node.excluded_llm_metadata_keys = [k for k in n.node.metadata if k != "arxiv_id"]
        return nodes


def load_index(collection: chromadb.Collection | None = None,
               embed_model: BaseEmbedding | None = None) -> VectorStoreIndex:
    if collection is None:
        collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
    store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(store, embed_model=embed_model or BGEEmbedding())


def make_llm(provider: str = LLM_PROVIDER) -> LLM:
    """Always built explicitly, so LlamaIndex never falls back to its default OpenAI settings."""
    if provider == "openai":
        from llama_index.llms.openai.utils import O1_MODELS

        from sdrag.budget import BudgetedOpenAI
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
        # reasoning models (gpt-5*) count thinking tokens against the cap, so keep effort minimal
        extra = {"reasoning_effort": "minimal"} if OPENAI_MODEL in O1_MODELS else {"temperature": 0.0}
        return BudgetedOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, max_tokens=MAX_ANSWER_TOKENS, **extra)
    if provider == "groq":
        from llama_index.llms.groq import Groq
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set (see .env.example)")
        return Groq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.0, max_tokens=MAX_ANSWER_TOKENS)
    raise ValueError(f"unknown LLM_PROVIDER {provider!r}")


def build_engine(index: VectorStoreIndex, llm: LLM, top_k: int = TOP_K) -> RetrieverQueryEngine:
    return RetrieverQueryEngine.from_args(
        index.as_retriever(similarity_top_k=top_k),
        llm=llm,
        text_qa_template=QA_PROMPT,
        node_postprocessors=[ArxivIdOnly()],
    )


def answer(engine: RetrieverQueryEngine, question: str) -> dict:
    resp = engine.query(question)
    text = str(resp).strip()
    sources = [{
        "arxiv_id": n.node.metadata.get("arxiv_id"),
        "title": n.node.metadata.get("title"),
        "chunk": n.node.metadata.get("chunk"),
        "score": n.score,
        "text": n.node.get_content(),
    } for n in resp.source_nodes]
    retrieved = {s["arxiv_id"] for s in sources}
    cited = list(dict.fromkeys(c for c in CITATION_RE.findall(text) if c in retrieved))
    return {"answer": text, "citations": cited, "sources": sources}


@lru_cache
def default_index() -> VectorStoreIndex:
    return load_index()
