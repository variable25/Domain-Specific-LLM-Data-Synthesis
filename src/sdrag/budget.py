"""Local OpenAI spend guard: records token usage per call and refuses calls once the budget is spent."""
import json
import threading

from llama_index.llms.openai import OpenAI

from sdrag.config import OPENAI_BUDGET_USD, OPENAI_PRICE_IN, OPENAI_PRICE_OUT, SPEND_PATH

_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def spent() -> float:
    try:
        return json.loads(SPEND_PATH.read_text())["usd"]
    except (FileNotFoundError, KeyError, ValueError):
        return 0.0


def check() -> None:
    if spent() >= OPENAI_BUDGET_USD:
        raise BudgetExceeded(f"OpenAI budget of ${OPENAI_BUDGET_USD:.2f} used up (see {SPEND_PATH})")


def record(prompt_tokens: int, completion_tokens: int) -> float:
    cost = (prompt_tokens * OPENAI_PRICE_IN + completion_tokens * OPENAI_PRICE_OUT) / 1e6
    with _lock:
        total = spent() + cost
        SPEND_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPEND_PATH.write_text(json.dumps({"usd": round(total, 6)}))
    return total


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
