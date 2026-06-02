"""Analyze V3 curriculum outputs.

Produces:
  - audit/curriculum_analysis.json
  - audit/curriculum_analysis.md
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import extract_gold, extract_pred, load_records


RESULT_DIR = ROOT / "results" / "curriculum"
OUT_JSON = ROOT / "audit" / "curriculum_analysis.json"
OUT_MD = ROOT / "audit" / "curriculum_analysis.md"


def _approx_eq(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _has_approx(vals: list[float], x: float | None) -> bool:
    return any(_approx_eq(v, x) for v in vals)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {
            "n": 0,
            "stage1_raw": 0,
            "stage1_score10": 0.0,
            "stage1_exact10": 0,
            "final_raw": 0,
            "final_score10": 0.0,
            "final_exact10": 0,
            "final_better": 0,
            "stage1_better": 0,
            "tie": 0,
        }
    s1 = sum(r["score_stage1"] for r in rows)
    s2 = sum(r["score_final"] for r in rows)
    return {
        "n": n,
        "stage1_raw": s1,
        "stage1_score10": s1 / n,
        "stage1_exact10": sum(r["score_stage1"] == 10 for r in rows),
        "final_raw": s2,
        "final_score10": s2 / n,
        "final_exact10": sum(r["score_final"] == 10 for r in rows),
        "final_better": sum(r["score_final"] > r["score_stage1"] for r in rows),
        "stage1_better": sum(r["score_stage1"] > r["score_final"] for r in rows),
        "tie": sum(r["score_stage1"] == r["score_final"] for r in rows),
    }


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _answer_literal_in_query(gold_str: str | None, query: str) -> bool:
    if not gold_str:
        return False
    value = gold_str.strip().replace("$", "")
    if not value or len(value) > 20:
        return False
    variants = {value, value.replace(".", ","), value.replace(",", ".")}
    return any(v and v in query for v in variants)


def main() -> None:
    train = load_records(ROOT / "dataset" / "train.json")
    valid = load_records(ROOT / "dataset" / "valid.json")
    stage1_preds = json.loads((RESULT_DIR / "valid_output_stage1.json").read_text(encoding="utf-8"))
    final_preds = json.loads((RESULT_DIR / "valid_output.json").read_text(encoding="utf-8"))
    stage1_report = json.loads((RESULT_DIR / "valid_report_stage1.json").read_text(encoding="utf-8"))
    final_report = json.loads((RESULT_DIR / "valid_report.json").read_text(encoding="utf-8"))

    train_nums_by_q: dict[str, list[float]] = defaultdict(list)
    train_resp_by_q: dict[str, set[str]] = defaultdict(set)
    for rec in train:
        q = (rec.get("query_vi") or "").strip()
        _gold_str, gold_num = extract_gold(rec)
        if gold_num is not None:
            train_nums_by_q[q].append(gold_num)
        train_resp_by_q[q].add((rec.get("response_vi") or "").strip())

    rows = []
    for i, (gold, s1_pred, s2_pred, s1_row, s2_row) in enumerate(
        zip(valid, stage1_preds, final_preds, stage1_report["rows"], final_report["rows"])
    ):
        q = (gold.get("query_vi") or "").strip()
        gold_str, gold_num = extract_gold(gold)
        p1_str, p1_num = extract_pred(s1_pred)
        p2_str, p2_num = extract_pred(s2_pred)
        seen = q in train_nums_by_q or q in train_resp_by_q
        same_answer = _has_approx(train_nums_by_q.get(q, []), gold_num)
        exact_qr = (gold.get("response_vi") or "").strip() in train_resp_by_q.get(q, set())
        rows.append(
            {
                "id": i,
                "type": gold.get("type"),
                "seen_query": seen,
                "same_answer_overlap": same_answer,
                "exact_query_response_overlap": exact_qr,
                "answer_literal_in_query": _answer_literal_in_query(gold_str, q),
                "gold_answer": gold_str,
                "gold_num": gold_num,
                "stage1_answer": p1_str,
                "stage1_num": p1_num,
                "final_answer": p2_str,
                "final_num": p2_num,
                "score_stage1": s1_row["score"],
                "score_final": s2_row["score"],
            }
        )

    slices = {
        "overall": rows,
        "seen_query": [r for r in rows if r["seen_query"]],
        "unseen_query": [r for r in rows if not r["seen_query"]],
        "same_answer_overlap": [r for r in rows if r["same_answer_overlap"]],
        "seen_but_answer_missing_or_conflict": [
            r for r in rows if r["seen_query"] and not r["same_answer_overlap"]
        ],
        "answer_literal_in_query": [r for r in rows if r["answer_literal_in_query"]],
        "no_answer_literal_in_query": [r for r in rows if not r["answer_literal_in_query"]],
    }
    by_type = {t: [r for r in rows if r["type"] == t] for t in sorted({r["type"] for r in rows})}

    transitions = Counter((r["score_stage1"], r["score_final"]) for r in rows)
    common_stage1 = Counter(str(r["stage1_answer"]) for r in rows if r["stage1_answer"] is not None)
    common_final = Counter(str(r["final_answer"]) for r in rows if r["final_answer"] is not None)

    payload = {
        "stage1_summary": stage1_report["summary"],
        "final_summary": final_report["summary"],
        "slices": {name: _summarize(part) for name, part in slices.items()},
        "by_type": {name: _summarize(part) for name, part in by_type.items()},
        "transitions": [
            {"stage1": k[0], "final": k[1], "count": v}
            for k, v in transitions.most_common()
        ],
        "common_stage1_answers": common_stage1.most_common(20),
        "common_final_answers": common_final.most_common(20),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Curriculum V3 analysis",
        "",
        "## Summary",
        "",
        "| run | raw | score/10 | exact10 | extractable | numeric_pairs | buckets |",
        "|---|---:|---:|---:|---:|---:|---|",
        (
            f"| stage1 answer-only | {stage1_report['summary']['raw_score']} | "
            f"{stage1_report['summary']['score_10']:.3f} | "
            f"{stage1_report['summary']['buckets'].get('10', stage1_report['summary']['buckets'].get(10))} | "
            f"{stage1_report['summary']['extractable']} | "
            f"{stage1_report['summary']['numeric_pairs']} | "
            f"{stage1_report['summary']['buckets']} |"
        ),
        (
            f"| final short-reason | {final_report['summary']['raw_score']} | "
            f"{final_report['summary']['score_10']:.3f} | "
            f"{final_report['summary']['buckets'].get('10', final_report['summary']['buckets'].get(10))} | "
            f"{final_report['summary']['extractable']} | "
            f"{final_report['summary']['numeric_pairs']} | "
            f"{final_report['summary']['buckets']} |"
        ),
        "",
        "## Slices",
        "",
        "| slice | n | stage1 raw | stage1 /10 | stage1 exact10 | final raw | final /10 | final exact10 | final better | stage1 better | tie |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["slices"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(row["n"]),
                    _fmt(row["stage1_raw"]),
                    _fmt(row["stage1_score10"]),
                    _fmt(row["stage1_exact10"]),
                    _fmt(row["final_raw"]),
                    _fmt(row["final_score10"]),
                    _fmt(row["final_exact10"]),
                    _fmt(row["final_better"]),
                    _fmt(row["stage1_better"]),
                    _fmt(row["tie"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## By type",
            "",
            "| type | n | stage1 raw | stage1 /10 | stage1 exact10 | final raw | final /10 | final exact10 | final better | stage1 better | tie |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in payload["by_type"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(row["n"]),
                    _fmt(row["stage1_raw"]),
                    _fmt(row["stage1_score10"]),
                    _fmt(row["stage1_exact10"]),
                    _fmt(row["final_raw"]),
                    _fmt(row["final_score10"]),
                    _fmt(row["final_exact10"]),
                    _fmt(row["final_better"]),
                    _fmt(row["stage1_better"]),
                    _fmt(row["tie"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Main transitions",
            "",
            "| stage1 score | final score | count |",
            "|---:|---:|---:|",
        ]
    )
    for item in payload["transitions"][:20]:
        lines.append(f"| {item['stage1']} | {item['final']} | {item['count']} |")

    lines.extend(
        [
            "",
            "## Common predicted answers",
            "",
            f"- stage1: {payload['common_stage1_answers']}",
            f"- final: {payload['common_final_answers']}",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
