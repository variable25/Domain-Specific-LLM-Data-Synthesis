"""Offline tests: in-memory Chroma, mock embeddings, fake LLM (no GPU, no network, no API cost)."""
import os
import uuid

import chromadb
import pytest
from fastapi.testclient import TestClient
from llama_index.core import MockEmbedding
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata

from sdrag import api, rag

DIM = 8
PAPERS = [
    ("2401.00001v1", "LiDAR-Camera Fusion for 3D Detection", "Fusing lidar and camera improves 3D detection."),
    ("2402.00002v2", "End-to-End Planning", "End-to-end planners map sensor input to trajectories."),
]


class FakeLLM(CustomLLM):
    reply: str = "Fusion improves detection [2401.00001v1]. Unrelated [9999.99999v1]."
    prompts: list = []

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    def complete(self, prompt, formatted=False, **kwargs):
        self.prompts.append(prompt)
        return CompletionResponse(text=self.reply)

    def stream_complete(self, prompt, formatted=False, **kwargs):
        yield self.complete(prompt)


@pytest.fixture
def index():
    # same layout the ETL writes: raw chromadb ids/documents/metadatas
    col = chromadb.EphemeralClient().create_collection(f"t{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
    col.add(ids=[f"{a}#0" for a, _, _ in PAPERS],
            embeddings=[[1.0] * DIM for _ in PAPERS],
            documents=[t for _, _, t in PAPERS],
            metadatas=[{"arxiv_id": a, "title": ti, "chunk": 0, "published": "2024-01-01"} for a, ti, _ in PAPERS])
    return rag.load_index(col, embed_model=MockEmbedding(embed_dim=DIM))


def test_answer_returns_sources_and_only_retrieved_citations(index):
    llm = FakeLLM()
    out = rag.answer(rag.build_engine(index, llm, top_k=2), "Does lidar-camera fusion help?")
    assert out["answer"].startswith("Fusion improves detection")
    assert {s["arxiv_id"] for s in out["sources"]} == {a for a, _, _ in PAPERS}
    assert out["citations"] == ["2401.00001v1"]  # hallucinated id dropped
    prompt = llm.prompts[-1]
    assert "arxiv_id: 2401.00001v1" in prompt and "[arxiv_id]" in prompt
    assert "title:" not in prompt  # only the id is shown to the LLM


def test_query_endpoint(index):
    api.app.dependency_overrides = {api.get_index: lambda: index, api.get_llm: lambda: FakeLLM()}
    try:
        client = TestClient(api.app)
        r = client.post("/query", json={"question": "Does lidar-camera fusion help?", "top_k": 1})
        assert r.status_code == 200
        body = r.json()
        assert len(body["sources"]) == 1 and body["sources"][0]["title"]
        assert client.post("/query", json={"question": ""}).status_code == 422
        assert client.get("/health").json() == {"status": "ok"}
    finally:
        api.app.dependency_overrides = {}


@pytest.mark.skipif(not os.getenv("RUN_LIVE"), reason="set RUN_LIVE=1 to hit real Chroma + LLM_PROVIDER")
def test_live_groq():
    out = rag.answer(rag.build_engine(rag.default_index(), rag.make_llm()),
                     "Which sensors are fused for 3D object detection in autonomous driving?")
    assert out["sources"] and out["citations"]


def test_budget_guard_blocks_after_limit(tmp_path, monkeypatch):
    from sdrag import budget
    monkeypatch.setattr(budget, "SPEND_PATH", tmp_path / "spend.json")
    monkeypatch.setattr(budget, "OPENAI_BUDGET_USD", 0.001)
    budget.check()  # nothing spent yet
    assert budget.record(2000, 200) == pytest.approx((2000 * 0.25 + 200 * 2.0) / 1e6)
    budget.record(4000, 0)  # crosses $0.001
    with pytest.raises(budget.BudgetExceeded):
        budget.check()


def test_query_endpoint_returns_402_when_budget_spent(index):
    from sdrag.budget import BudgetExceeded

    class BrokeLLM(FakeLLM):
        def complete(self, prompt, formatted=False, **kwargs):
            raise BudgetExceeded("budget used up")

    api.app.dependency_overrides = {api.get_index: lambda: index, api.get_llm: lambda: BrokeLLM()}
    try:
        r = TestClient(api.app).post("/query", json={"question": "Does lidar-camera fusion help?"})
        assert r.status_code == 402 and "budget" in r.json()["detail"]
    finally:
        api.app.dependency_overrides = {}
