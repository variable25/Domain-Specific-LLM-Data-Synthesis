"""OpenAI spend guard: records token usage per call and refuses calls once the budget is spent.

The running total lives in SPEND_PATH: a local JSON file, or gs://bucket/name on Cloud Run, where the
container disk is wiped on every restart. GCS writes are conditional on the object generation, so two
concurrent writers can't lose each other's cost.
"""
import json
import threading
from functools import lru_cache
from pathlib import Path

from llama_index.llms.openai import OpenAI

from sdrag.config import OPENAI_BUDGET_USD, OPENAI_PRICE_IN, OPENAI_PRICE_OUT, SPEND_PATH

_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


class _Conflict(Exception):
    """Spend object changed between read and write."""


@lru_cache
def _bucket(name: str):
    from google.cloud import storage
    return storage.Client().bucket(name)


def _gcs_blob_name() -> tuple[str, str]:
    bucket, _, name = str(SPEND_PATH).removeprefix("gs://").partition("/")
    return bucket, name


def _parse(raw: bytes | str) -> float:
    try:
        return float(json.loads(raw)["usd"])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _read() -> tuple[float, int | None]:
    """Spent USD plus, for GCS, the object generation to write against (0 = object doesn't exist yet)."""
    if str(SPEND_PATH).startswith("gs://"):
        bucket, name = _gcs_blob_name()
        blob = _bucket(bucket).get_blob(name)
        if blob is None:
            return 0.0, 0
        return _parse(blob.download_as_bytes(if_generation_match=blob.generation)), blob.generation
    try:
        return _parse(Path(SPEND_PATH).read_text()), None
    except FileNotFoundError:
        return 0.0, None


def _write(usd: float, generation: int | None) -> None:
    data = json.dumps({"usd": round(usd, 6)})
    if str(SPEND_PATH).startswith("gs://"):
        from google.api_core.exceptions import PreconditionFailed
        bucket, name = _gcs_blob_name()
        try:
            _bucket(bucket).blob(name).upload_from_string(
                data, content_type="application/json", if_generation_match=generation)
        except PreconditionFailed as e:
            raise _Conflict from e
        return
    path = Path(SPEND_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)


def spent() -> float:
    return _read()[0]


def check() -> None:
    if spent() >= OPENAI_BUDGET_USD:
        raise BudgetExceeded(f"OpenAI budget of ${OPENAI_BUDGET_USD:.2f} used up (see {SPEND_PATH})")


def record(prompt_tokens: int, completion_tokens: int, retries: int = 5) -> float:
    cost = (prompt_tokens * OPENAI_PRICE_IN + completion_tokens * OPENAI_PRICE_OUT) / 1e6
    with _lock:
        for _ in range(retries):
            usd, generation = _read()
            try:
                _write(usd + cost, generation)
                return usd + cost
            except _Conflict:
                continue
    raise RuntimeError(f"could not record ${cost:.6f} to {SPEND_PATH} after {retries} conflicts")


def _usage(raw) -> tuple[int, int]:
    u = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
    if u is None:
        return 0, 0
    get = u.get if isinstance(u, dict) else lambda k: getattr(u, k, 0)
    return get("prompt_tokens") or 0, get("completion_tokens") or 0


class BudgetedOpenAI(OpenAI):
    """OpenAI LLM that checks the budget before each call and records its cost after."""

    def chat(self, messages, **kwargs):
        check()
        resp = super().chat(messages, **kwargs)
        record(*_usage(resp.raw))
        return resp

    def complete(self, prompt, formatted=False, **kwargs):
        check()
        resp = super().complete(prompt, formatted=formatted, **kwargs)
        record(*_usage(resp.raw))
        return resp

    async def achat(self, messages, **kwargs):
        check()
        resp = await super().achat(messages, **kwargs)
        record(*_usage(resp.raw))
        return resp

    async def acomplete(self, prompt, formatted=False, **kwargs):
        check()
        resp = await super().acomplete(prompt, formatted=formatted, **kwargs)
        record(*_usage(resp.raw))
        return resp
