"""Offline tests for the serving extras: GCS-backed spend guard, rate limit, metadata snapshot."""
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from google.api_core.exceptions import PreconditionFailed

from sdrag import api, budget


class FakeBucket:
    """In-memory stand-in for a GCS bucket with generation preconditions."""

    def __init__(self):
        self.data, self.generation, self.before_write = None, 0, None

    def get_blob(self, name):
        if self.data is None:
            return None
        return SimpleNamespace(generation=self.generation, download_as_bytes=lambda if_generation_match: self.data)

    def blob(self, name):
        return SimpleNamespace(upload_from_string=self._upload)

    def _upload(self, data, content_type, if_generation_match):
        if self.before_write:  # simulate another instance writing between our read and write
            self.before_write, hook = None, self.before_write
            hook()
        if if_generation_match != self.generation:
            raise PreconditionFailed("generation mismatch")
        self.data, self.generation = data.encode(), self.generation + 1


@pytest.fixture
def gcs(monkeypatch):
    bucket = FakeBucket()
    monkeypatch.setattr(budget, "SPEND_PATH", "gs://spend-bucket/openai_spend.json")
    monkeypatch.setattr(budget, "_bucket", lambda name: bucket)
    return bucket


def test_gcs_spend_survives_and_retries_on_conflict(gcs, monkeypatch):
    assert budget.spent() == 0.0
    budget.record(4000, 0)  # $0.001, creates the object (generation precondition 0)

    def other_writer():
        gcs.data, gcs.generation = json.dumps({"usd": 0.5}).encode(), gcs.generation + 1
    gcs.before_write = other_writer
    total = budget.record(4000, 0)
    assert total == pytest.approx(0.501)  # re-read after the conflict, nothing lost
    assert budget.spent() == pytest.approx(0.501)

    monkeypatch.setattr(budget, "OPENAI_BUDGET_USD", 0.5)
    with pytest.raises(budget.BudgetExceeded):
        budget.check()


def test_rate_limiter_blocks_after_n_calls_per_minute():
    limit = api.RateLimiter(2)
    limit()
    limit()
    with pytest.raises(HTTPException) as e:
        limit()
    assert e.value.status_code == 429
    api.RateLimiter(0)()  # 0 = off


def test_papers_endpoint_reads_snapshot(tmp_path, monkeypatch):
    meta = tmp_path / "papers.jsonl"
    meta.write_text(json.dumps({"arxiv_id": "2401.00001v1", "title": "LiDAR-Camera Fusion"}) + "\n")
    monkeypatch.setattr(api, "META_PATH", meta)
    api.papers.cache_clear()
    try:
        client = TestClient(api.app)
        assert client.get("/papers/2401.00001v1").json()["title"] == "LiDAR-Camera Fusion"
        assert client.get("/papers/9999.99999v1").status_code == 404
    finally:
        api.papers.cache_clear()
