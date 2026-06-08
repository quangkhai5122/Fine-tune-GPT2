"""Diagnose the legal query-only retrieval ceiling for v16 outputs.

The script uses only existing validation artifacts. It is an analysis helper,
not a training or inference notebook component.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import evaluate, extract_answer, load_records


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def pred_from_keys(gold: list[dict], keys: list[str | None]) -> list[dict]:
    rows = []
    for i, key in enumerate(keys):
        rows.append(
            {
                "id": gold[i].get("id", i),
                "query_vi": gold[i].get("query_vi"),
                "type": gold[i].get("type"),
                "model_output": "" if key is None else f"Đáp án là: {key}",
            }
        )
    return rows


def score_key(gold: list[dict], idx: int, key: str | None) -> tuple[int, dict]:
    pred = pred_from_keys([gold[idx]], [key])
    row = evaluate(pred, [gold[idx]])["rows"][0]
    return int(row["score"]), row


def summarize(name: str, pred_items: list[dict], gold: list[dict]) -> dict:
    summary = evaluate(pred_items, gold)["summary"]
    return {
        "name": name,
        "raw": summary["raw_score"],
        "score_10": summary["score_10"],
        "buckets": summary["buckets"],
    }


def summarize_scores(name: str, scores: list[int]) -> dict:
    buckets = dict(Counter(scores))
    raw = sum(scores)
    return {
        "name": name,
        "raw": raw,
        "score_10": raw / len(scores) if scores else 0.0,
        "buckets": {0: buckets.get(0, 0), 1: buckets.get(1, 0), 5: buckets.get(5, 0), 10: buckets.get(10, 0)},
    }


def simulate_threshold(
    decisions: list[dict],
    base_scores: list[int],
    retrieval_scores: list[int],
    min_sim: float,
    min_majority_frac: float,
    min_margin: float,
) -> tuple[int, int, Counter]:
    scores = []
    used = 0
    for i, decision in enumerate(decisions):
        retrieval = decision.get("retrieval") or {}
        cond = (
            decision.get("retrieval_key") is not None
            and retrieval.get("reason") == "passed"
            and retrieval.get("top_sim", 0.0) >= min_sim
            and retrieval.get("majority_frac", 0.0) >= min_majority_frac
            and retrieval.get("margin", 0.0) >= min_margin
        )
        if cond:
            scores.append(retrieval_scores[i])
            used += 1
        else:
            scores.append(base_scores[i])
    return sum(scores), used, Counter(scores)


def main() -> None:
    gold = load_records(ROOT / "dataset" / "valid.json")
    query_dir = ROOT / "results" / "v16_query_retrieval_ranker"
    multi_dir = ROOT / "results" / "v16_multiseed"

    model_out = load_json(query_dir / "model_valid_output.json")
    final_out = load_json(query_dir / "valid_output.json")
    decisions = load_json(query_dir / "valid_query_retrieval_decisions.json")
    base_seed42e7 = load_json(multi_dir / "ensemble_candidates" / "valid_model_seed_42_epoch_07.json")

    model_keys = [extract_answer(item.get("model_output")) for item in model_out]
    base_keys = [extract_answer(item.get("model_output")) for item in base_seed42e7]
    retrieval_keys = [decision.get("retrieval_key") for decision in decisions]

    model_scores = [score_key(gold, i, key)[0] for i, key in enumerate(model_keys)]
    base_scores = [score_key(gold, i, key)[0] for i, key in enumerate(base_keys)]
    retrieval_scored = [score_key(gold, i, key) for i, key in enumerate(retrieval_keys)]
    retrieval_scores = [item[0] for item in retrieval_scored]
    retrieval_rows = [item[1] for item in retrieval_scored]

    scenarios = [
        summarize("v16_query_model", model_out, gold),
        summarize("v16_query_final", final_out, gold),
        summarize(
            "query_retrieval_all_passed_else_model",
            pred_from_keys(
                gold,
                [
                    decision.get("retrieval_key")
                    if (decision.get("retrieval") or {}).get("reason") == "passed"
                    and decision.get("retrieval_key") is not None
                    else model_keys[i]
                    for i, decision in enumerate(decisions)
                ],
            ),
            gold,
        ),
        summarize(
            "query_retrieval_if_agree_else_model",
            pred_from_keys(
                gold,
                [
                    decision.get("retrieval_key")
                    if decision.get("retrieval_key") == model_keys[i]
                    else model_keys[i]
                    for i, decision in enumerate(decisions)
                ],
            ),
            gold,
        ),
        summarize("base_seed42e7", base_seed42e7, gold),
        summarize(
            "base_seed42e7_retr_all_passed_else_base",
            pred_from_keys(
                gold,
                [
                    decision.get("retrieval_key")
                    if (decision.get("retrieval") or {}).get("reason") == "passed"
                    and decision.get("retrieval_key") is not None
                    else base_keys[i]
                    for i, decision in enumerate(decisions)
                ],
            ),
            gold,
        ),
        summarize(
            "base_seed42e7_retr_if_agree_else_base",
            pred_from_keys(
                gold,
                [
                    decision.get("retrieval_key")
                    if decision.get("retrieval_key") == base_keys[i]
                    else base_keys[i]
                    for i, decision in enumerate(decisions)
                ],
            ),
            gold,
        ),
    ]

    relation_query = Counter()
    relation_base = Counter()
    by_type_query = defaultdict(Counter)
    by_source_query = defaultdict(Counter)
    oracle_query_keys = []
    oracle_base_keys = []

    for i, decision in enumerate(decisions):
        model_score = model_scores[i]
        base_score = base_scores[i]
        retrieval_score = retrieval_scores[i]

        if retrieval_score > model_score:
            relation_query["retrieval_better"] += 1
            relation = "retrieval_better"
        elif retrieval_score < model_score:
            relation_query["retrieval_worse"] += 1
            relation = "retrieval_worse"
        else:
            relation_query["same"] += 1
            relation = "same"
        by_type_query[gold[i].get("type")][relation] += 1
        by_source_query[decision.get("source")][relation] += 1
        oracle_query_keys.append(decision.get("retrieval_key") if retrieval_score > model_score else model_keys[i])

        if retrieval_score > base_score:
            relation_base["retrieval_better"] += 1
        elif retrieval_score < base_score:
            relation_base["retrieval_worse"] += 1
        else:
            relation_base["same"] += 1
        oracle_base_keys.append(decision.get("retrieval_key") if retrieval_score > base_score else base_keys[i])

    scenarios.extend(
        [
            summarize("oracle_query_model_vs_retrieval", pred_from_keys(gold, oracle_query_keys), gold),
            summarize("oracle_base_seed42e7_vs_retrieval", pred_from_keys(gold, oracle_base_keys), gold),
        ]
    )

    top_query = []
    top_base = []
    for sim in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        for majority_frac in [0.34, 0.45, 0.50, 0.60, 0.75, 0.90, 1.00]:
            for margin in [0.00, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]:
                top_query.append(
                    {
                        "raw": simulate_threshold(decisions, model_scores, retrieval_scores, sim, majority_frac, margin)[0],
                        "used": simulate_threshold(decisions, model_scores, retrieval_scores, sim, majority_frac, margin)[1],
                        "min_sim": sim,
                        "min_majority_frac": majority_frac,
                        "min_margin": margin,
                    }
                )
                top_base.append(
                    {
                        "raw": simulate_threshold(decisions, base_scores, retrieval_scores, sim, majority_frac, margin)[0],
                        "used": simulate_threshold(decisions, base_scores, retrieval_scores, sim, majority_frac, margin)[1],
                        "min_sim": sim,
                        "min_majority_frac": majority_frac,
                        "min_margin": margin,
                    }
                )

    sim_bins = defaultdict(Counter)
    highsim_examples = {"retrieval_better": [], "retrieval_worse": []}
    for i, decision in enumerate(decisions):
        retrieval = decision.get("retrieval") or {}
        if retrieval.get("reason") != "passed" or decision.get("retrieval_key") is None:
            continue
        model_score = model_scores[i]
        retrieval_score = retrieval_scores[i]
        row = retrieval_rows[i]
        label = "same"
        if retrieval_score > model_score:
            label = "retrieval_better"
        elif retrieval_score < model_score:
            label = "retrieval_worse"
        sim_bucket = f"{math.floor(retrieval.get('top_sim', 0.0) * 10) / 10:.1f}"
        sim_bins[sim_bucket][label] += 1
        if retrieval.get("top_sim", 0.0) >= 0.80 and label in highsim_examples and len(highsim_examples[label]) < 10:
            highsim_examples[label].append(
                {
                    "id": i,
                    "type": gold[i].get("type"),
                    "model": model_keys[i],
                    "retrieval": decision.get("retrieval_key"),
                    "gold": row["gold_answer"],
                    "top_sim": round(retrieval.get("top_sim", 0.0), 3),
                    "majority_frac": retrieval.get("majority_frac"),
                    "margin": retrieval.get("margin"),
                    "source": decision.get("source"),
                }
            )

    report = {
        "scenarios": scenarios,
        "relation_query_model_vs_retrieval": dict(relation_query),
        "relation_base_seed42e7_vs_retrieval": dict(relation_base),
        "by_source_query_model_vs_retrieval": {k: dict(v) for k, v in by_source_query.items()},
        "by_type_query_model_vs_retrieval": {k: dict(v) for k, v in by_type_query.items()},
        "top_thresholds_query_model_base": sorted(top_query, key=lambda x: x["raw"], reverse=True)[:20],
        "top_thresholds_base_seed42e7": sorted(top_base, key=lambda x: x["raw"], reverse=True)[:20],
        "sim_bins_query_model_vs_retrieval": {k: dict(v) for k, v in sorted(sim_bins.items())},
        "highsim_examples": highsim_examples,
    }

    out = ROOT / "audit" / "v16_retrieval_ceiling_analysis.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
