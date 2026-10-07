"""Offline tests for the test-set generator, eval metrics, report and async spend guard."""
import asyncio
import random
from types import SimpleNamespace

import pytest

from sdrag import budget, evaluate, testset
from sdrag.embed import collection_name


def test_parse_qa_accepts_fenced_json_and_rejects_vague_or_skipped():
    ok = testset.parse_qa('```json\n{"question": "Which sensors does BEVFusion fuse?", "answer": "Camera and lidar."}\n```')
    assert ok == {"question": "Which sensors does BEVFusion fuse?", "answer": "Camera and lidar."}
    assert testset.parse_qa('{"skip": true}') is None
    assert testset.parse_qa("not json") is None
    for q in ["What does this paper propose for planning?", "According to the passage, what is used?",
              "What is shown in Table 3 for nuScenes?", "Which dataset do the authors use?"]:
        assert testset.parse_qa(f'{{"question": "{q}", "answer": "x"}}') is None, q


def test_pick_passage_is_seeded_and_bounded():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(40))
    a, b = (testset.pick_passage(text, random.Random(1)) for _ in range(2))
    assert a == b
    assert testset.MIN_PASSAGE <= len(a) <= testset.MAX_PASSAGE


def test_generate_keeps_n_valid_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(testset, "TEXT_DIR", tmp_path)
    papers = [{"arxiv_id": f"2401.0000{i}v1", "title": f"Paper {i}"} for i in range(5)]
    for p in papers:
        (tmp_path / f"{p['arxiv_id']}.txt").write_text("Lidar point clouds are voxelized. " * 50)
    replies = iter(['{"skip": true}'] + ['{"question": "How are lidar point clouds processed?", "answer": "Voxelized."}'] * 4)
    llm = SimpleNamespace(complete=lambda prompt: SimpleNamespace(text=next(replies)))
    rows = testset.generate(llm, papers, n=3)
    assert [r["id"] for r in rows] == ["q01", "q02", "q03"]
    assert len({r["arxiv_ids"][0] for r in rows}) == 3 and all(r["passage"] for r in rows)


def test_retrieval_hit_rate_and_mrr():
    ranked = {"q1": ["a", "b"], "q2": ["c", "a"], "q3": ["c", "d"]}
    node = lambda a: SimpleNamespace(node=SimpleNamespace(metadata={"arxiv_id": a}))
    retriever = SimpleNamespace(retrieve=lambda q: [node(a) for a in ranked[q]])
    rows = [{"id": q, "question": q, "arxiv_ids": ["a"]} for q in ranked]
    per_q = evaluate.retrieval_metrics(retriever, rows)
    assert [q["rank"] for q in per_q] == [1, 2, None]
    assert evaluate.summarize_retrieval(per_q) == {"hit_rate": pytest.approx(2 / 3), "mrr": pytest.approx(0.5)}


def test_report_uses_shared_question_subset():
    def result(name, ragas_n, scores):
        per_q = [{"id": f"q{i}", "rank": 1, **dict.fromkeys(evaluate.METRICS, s)} for i, s in enumerate(scores)]
        return {"collection": name, "chunks": 10, "n": len(scores), "hit_rate": 1.0, "mrr": 1.0, "ragas_n": ragas_n,
                "ragas": {m: sum(scores) / len(scores) for m in evaluate.METRICS}, "per_question": per_q}
    md = evaluate.report([result("full", 4, [1.0, 1.0, 0.0, 0.0]), result("small", 2, [0.5, 0.5])], ablation_n=2)
    main, ablation = md.split("Ablation")
    assert "`full`" in main and "| 0.500 |" in main and "`small`" not in main
    assert "`full` | 10 | 1.00 | 1.00 | 2 | 1.000" in ablation  # full index re-scored on first 2 questions


def test_ablation_collection_never_overwrites_full_index():
    assert collection_name(384, 64) == "papers_c384_o64"
    assert collection_name(384, 64, subset=100) == "papers_c384_o64_s100"


def test_async_judge_calls_respect_budget(tmp_path, monkeypatch):
    from sdrag.budget import BudgetedOpenAI
    monkeypatch.setattr(budget, "SPEND_PATH", tmp_path / "spend.json")
    monkeypatch.setattr(budget, "OPENAI_BUDGET_USD", 0.001)
    budget.record(10_000, 0)  # $0.0025 > budget
    llm = BudgetedOpenAI(model="gpt-5-mini", api_key="sk-test")
    with pytest.raises(budget.BudgetExceeded):  # raised before any network call
        asyncio.run(llm.acomplete("hi"))
