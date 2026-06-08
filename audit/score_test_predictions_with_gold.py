from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import (  # noqa: E402
    align_predictions_with_gold,
    extract_answer,
    extract_gold,
    load_records,
    parse_number,
    rel_error,
    score_one,
)


DEFAULT_GOLD_PATH = ROOT / "dataset" / "test_gold.json"
DEFAULT_OUT_ROOT = ROOT / "test_goldanswer"


PREDICTION_TEXT_FIELDS = (
    "model_output",
    "response_vi",
    "prediction",
    "pred",
    "answer",
    "output",
)


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_name_from_path(pred_path: Path) -> str:
    stem = pred_path.stem
    parent = pred_path.parent.name
    if stem in {"test_predictions", "valid_output", "valid_predictions", "predictions"} and parent:
        return parent
    if parent and parent not in {"results", "test_goldanswer", ""}:
        return f"{parent}_{stem}"
    return stem


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", name).strip("_") or "run"


def normalize_prediction_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    if out.get("model_output") is not None:
        return out

    for field in PREDICTION_TEXT_FIELDS:
        value = out.get(field)
        if value is not None:
            out["model_output"] = str(value)
            out["_prediction_text_source"] = field
            return out

    out["model_output"] = ""
    out["_prediction_text_source"] = None
    return out


def load_predictions(path: Path) -> list[dict[str, Any]]:
    items = load_records(path)
    if not isinstance(items, list):
        raise TypeError(f"Prediction file must contain a list/jsonl of objects: {path}")
    normalized = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"Prediction row {idx} is not a JSON object: {type(item)}")
        normalized.append(normalize_prediction_record(item))
    return normalized


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return ordered[int(idx)]
    frac = idx - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


def compact_float(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def extract_pred_from_normalized(record: dict[str, Any]) -> tuple[str | None, float | None]:
    answer = extract_answer(record.get("model_output"), include_english=True)
    return answer, parse_number(answer)


def build_rows(
    pred_items: list[dict[str, Any]],
    gold_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs = align_predictions_with_gold(pred_items, gold_items)
    rows: list[dict[str, Any]] = []

    for pair_index, (pred_record, gold_record) in enumerate(pairs):
        pred_answer, pred_num = extract_pred_from_normalized(pred_record)
        gold_answer, gold_num = extract_gold(gold_record)

        extractable = pred_answer is not None
        error_value = rel_error(pred_num, gold_num)
        score = score_one(error_value, extractable)

        pred_query = pred_record.get("query_vi")
        gold_query = gold_record.get("query_vi")
        query_match = None
        if pred_query is not None and gold_query is not None:
            query_match = str(pred_query).strip() == str(gold_query).strip()

        rows.append(
            {
                "row_index": pair_index,
                "id": gold_record.get("id", pred_record.get("id", pair_index)),
                "type": gold_record.get("type") or pred_record.get("type") or "<missing>",
                "query_vi": gold_query if gold_query is not None else pred_query,
                "model_output": pred_record.get("model_output", ""),
                "prediction_text_source": pred_record.get("_prediction_text_source", "model_output"),
                "gold_response_vi": gold_record.get("response_vi"),
                "pred_answer": pred_answer,
                "pred_num": pred_num,
                "gold_answer": gold_answer,
                "gold_num": gold_num,
                "rel_error": error_value,
                "extractable": extractable,
                "pred_numeric": finite_number(pred_num),
                "gold_numeric": finite_number(gold_num),
                "score": score,
                "query_match": query_match,
            }
        )
    return rows


def summarize_by_type(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, Counter] = defaultdict(Counter)
    rel_errors: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        typ = str(row["type"])
        score = int(row["score"])
        buckets[typ]["n"] += 1
        buckets[typ]["raw_score"] += score
        buckets[typ][f"bucket_{score}"] += 1
        buckets[typ]["extractable"] += int(bool(row["extractable"]))
        buckets[typ]["numeric_pairs"] += int(bool(row["pred_numeric"] and row["gold_numeric"]))
        if isinstance(row["rel_error"], (int, float)) and math.isfinite(float(row["rel_error"])):
            rel_errors[typ].append(float(row["rel_error"]))

    out = {}
    for typ, counter in sorted(buckets.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        n = int(counter["n"])
        errors = rel_errors.get(typ, [])
        out[typ] = {
            "n": n,
            "raw_score": int(counter["raw_score"]),
            "max_raw_score": n * 10,
            "score_10": counter["raw_score"] / n if n else 0.0,
            "score_pct": counter["raw_score"] / (n * 10) if n else 0.0,
            "extractable": int(counter["extractable"]),
            "numeric_pairs": int(counter["numeric_pairs"]),
            "bucket_10": int(counter["bucket_10"]),
            "bucket_5": int(counter["bucket_5"]),
            "bucket_1": int(counter["bucket_1"]),
            "bucket_0": int(counter["bucket_0"]),
            "rel_error_median": compact_float(median(errors)) if errors else None,
            "rel_error_mean": compact_float(mean(errors)) if errors else None,
        }
    return out


def summarize_rows(
    rows: list[dict[str, Any]],
    pred_items: list[dict[str, Any]],
    gold_items: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(rows)
    raw_score = sum(int(row["score"]) for row in rows)
    score_buckets = Counter(int(row["score"]) for row in rows)
    rel_errors = [
        float(row["rel_error"])
        for row in rows
        if isinstance(row["rel_error"], (int, float)) and math.isfinite(float(row["rel_error"]))
    ]
    pred_outputs = [str(item.get("model_output", "")) for item in pred_items]
    pred_answers = [row["pred_answer"] if row["pred_answer"] is not None else "<none>" for row in rows]
    ids = [str(item.get("id")) for item in pred_items if item.get("id") is not None]
    gold_ids = [str(item.get("id")) for item in gold_items if item.get("id") is not None]
    pred_id_set = set(ids)

    query_match_values = [row["query_match"] for row in rows if row["query_match"] is not None]
    lengths = [len(text) for text in pred_outputs]

    return {
        "evaluation_kind": "Pseudo-gold test evaluation; not an official hidden-gold score",
        "scoring_rule": {
            "relative_error": "|prediction - gold| / max(1, |gold|)",
            "thresholds": {
                "<=0.01": 10,
                "<=0.10": 5,
                "<=0.50": 1,
                "otherwise_or_unextractable": 0,
            },
        },
        "n": n,
        "prediction_count": len(pred_items),
        "gold_count": len(gold_items),
        "unique_prediction_ids": len(set(ids)) if ids else None,
        "duplicate_prediction_id_count": (len(ids) - len(set(ids))) if ids else None,
        "missing_gold_ids_in_predictions": [gid for gid in gold_ids if gid not in pred_id_set][:50]
        if ids and gold_ids
        else [],
        "raw_score": raw_score,
        "max_raw_score": n * 10,
        "score_10": raw_score / n if n else 0.0,
        "score_pct": raw_score / (n * 10) if n else 0.0,
        "score_bucket_counts": {
            "10": int(score_buckets[10]),
            "5": int(score_buckets[5]),
            "1": int(score_buckets[1]),
            "0": int(score_buckets[0]),
        },
        "extractable_prediction_count": sum(int(bool(row["extractable"])) for row in rows),
        "unextractable_prediction_count": sum(int(not bool(row["extractable"])) for row in rows),
        "numeric_pair_count": sum(int(bool(row["pred_numeric"] and row["gold_numeric"])) for row in rows),
        "gold_numeric_count": sum(int(bool(row["gold_numeric"])) for row in rows),
        "pred_numeric_count": sum(int(bool(row["pred_numeric"])) for row in rows),
        "query_match_count": sum(int(value) for value in query_match_values),
        "query_mismatch_count": sum(int(not value) for value in query_match_values),
        "rel_error": {
            "count": len(rel_errors),
            "mean": compact_float(mean(rel_errors)) if rel_errors else None,
            "median": compact_float(median(rel_errors)) if rel_errors else None,
            "p90": compact_float(percentile(rel_errors, 0.90)),
            "p95": compact_float(percentile(rel_errors, 0.95)),
            "max": compact_float(max(rel_errors)) if rel_errors else None,
        },
        "output_sanity": {
            "empty_output_count": sum(int(text.strip() == "") for text in pred_outputs),
            "digit_output_count": sum(int(bool(re.search(r"\d", text))) for text in pred_outputs),
            "anchor_output_count": sum(
                int(bool(re.search(r"đáp\s*án|dap\s*an|answer|####", text, re.IGNORECASE)))
                for text in pred_outputs
            ),
            "length_min": min(lengths) if lengths else None,
            "length_median": median(lengths) if lengths else None,
            "length_mean": compact_float(mean(lengths)) if lengths else None,
            "length_max": max(lengths) if lengths else None,
            "top_pred_answers": Counter(pred_answers).most_common(20),
        },
    }


def select_examples(rows: list[dict[str, Any]], limit: int) -> dict[str, list[dict[str, Any]]]:
    def slim(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "type": row["type"],
            "score": row["score"],
            "rel_error": row["rel_error"],
            "gold_answer": row["gold_answer"],
            "pred_answer": row["pred_answer"],
            "model_output": row["model_output"],
            "query_vi": row["query_vi"],
        }

    zero_rows = [row for row in rows if int(row["score"]) == 0]
    partial_rows = [row for row in rows if int(row["score"]) in {1, 5}]
    exact_rows = [row for row in rows if int(row["score"]) == 10]
    unextractable_rows = [row for row in rows if not row["extractable"]]

    zero_rows = sorted(
        zero_rows,
        key=lambda row: (
            row["rel_error"] is None,
            float(row["rel_error"]) if isinstance(row["rel_error"], (int, float)) else -1.0,
        ),
        reverse=True,
    )

    return {
        "score_0_examples": [slim(row) for row in zero_rows[:limit]],
        "partial_score_examples": [slim(row) for row in partial_rows[:limit]],
        "score_10_examples": [slim(row) for row in exact_rows[:limit]],
        "unextractable_examples": [slim(row) for row in unextractable_rows[:limit]],
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "id",
        "type",
        "score",
        "rel_error",
        "extractable",
        "pred_answer",
        "pred_num",
        "gold_answer",
        "gold_num",
        "query_match",
        "model_output",
        "gold_response_vi",
        "query_vi",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(
    report: dict[str, Any],
    pred_path: Path,
    gold_path: Path,
    md_path: Path,
    example_limit: int,
) -> None:
    summary = report["summary"]
    by_type = report["by_type"]
    examples = report["examples"]

    lines = [
        "# Test Predictions Pseudo-Gold Evaluation",
        "",
        f"- prediction_file: `{display_path(pred_path)}`",
        f"- gold_file: `{display_path(gold_path)}`",
        "- note: pseudo-gold is useful for local comparison, but it is not the official competition gold.",
        "",
        "## Summary",
        "",
        "| n | raw_score | max_raw_score | score_10 | extractable | numeric_pairs | buckets |",
        "|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {summary['n']} | {summary['raw_score']} | {summary['max_raw_score']} | "
            f"{summary['score_10']:.4f} | {summary['extractable_prediction_count']} | "
            f"{summary['numeric_pair_count']} | {summary['score_bucket_counts']} |"
        ),
        "",
        "## By Type",
        "",
        "| type | n | raw | score_10 | b10 | b5 | b1 | b0 | extractable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for typ, row in by_type.items():
        lines.append(
            f"| {typ} | {row['n']} | {row['raw_score']} | {row['score_10']:.4f} | "
            f"{row['bucket_10']} | {row['bucket_5']} | {row['bucket_1']} | "
            f"{row['bucket_0']} | {row['extractable']} |"
        )

    lines.extend(
        [
            "",
            "## Output Sanity",
            "",
            f"- empty_output_count: {summary['output_sanity']['empty_output_count']}",
            f"- digit_output_count: {summary['output_sanity']['digit_output_count']}",
            f"- anchor_output_count: {summary['output_sanity']['anchor_output_count']}",
            (
                "- length: "
                f"min={summary['output_sanity']['length_min']}, "
                f"median={summary['output_sanity']['length_median']}, "
                f"mean={summary['output_sanity']['length_mean']}, "
                f"max={summary['output_sanity']['length_max']}"
            ),
            f"- top_pred_answers: {summary['output_sanity']['top_pred_answers'][:10]}",
            "",
            "## Score 0 Examples",
        ]
    )

    for row in examples["score_0_examples"][:example_limit]:
        lines.append(
            f"- id={row['id']} type={row['type']} pred={row['pred_answer']} "
            f"gold={row['gold_answer']} rel_error={row['rel_error']} | {str(row['query_vi'])[:220]}"
        )

    lines.extend(["", "## Unextractable Examples"])
    for row in examples["unextractable_examples"][:example_limit]:
        lines.append(
            f"- id={row['id']} type={row['type']} output={str(row['model_output'])[:160]} | "
            f"{str(row['query_vi'])[:220]}"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_prediction_file(
    pred_path: Path,
    gold_path: Path,
    out_dir: Path | None,
    example_limit: int,
) -> dict[str, Any]:
    pred_items = load_predictions(pred_path)
    gold_items = load_records(gold_path)
    if not isinstance(gold_items, list):
        raise TypeError(f"Gold file must contain a list/jsonl of objects: {gold_path}")

    rows = build_rows(pred_items, gold_items)
    summary = summarize_rows(rows, pred_items, gold_items)
    by_type = summarize_by_type(rows)
    examples = select_examples(rows, example_limit)

    report = {
        "summary": {
            **summary,
            "prediction_file": display_path(pred_path),
            "gold_file": display_path(gold_path),
        },
        "by_type": by_type,
        "examples": examples,
        "rows": rows,
    }

    if out_dir is None:
        out_dir = DEFAULT_OUT_ROOT / safe_filename(run_name_from_path(pred_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    audit_path = out_dir / "audit.json"
    csv_path = out_dir / "audit.csv"
    md_path = out_dir / "report.md"

    summary_payload = {
        "summary": report["summary"],
        "by_type": report["by_type"],
        "examples": report["examples"],
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_markdown(report, pred_path, gold_path, md_path, example_limit)

    print(f"[wrote] {display_path(summary_path)}")
    print(f"[wrote] {display_path(audit_path)}")
    print(f"[wrote] {display_path(csv_path)}")
    print(f"[wrote] {display_path(md_path)}")
    print(
        "[score] "
        f"raw={summary['raw_score']}/{summary['max_raw_score']} "
        f"score_10={summary['score_10']:.4f} "
        f"buckets={summary['score_bucket_counts']}"
    )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Kaggle test_predictions.json file against dataset/test_gold.json "
            "or another pseudo-gold file with response_vi."
        )
    )
    parser.add_argument(
        "--pred",
        required=True,
        type=Path,
        help="Path to test_predictions.json or jsonl. Rows should contain id and model_output/response_vi.",
    )
    parser.add_argument(
        "--gold",
        default=DEFAULT_GOLD_PATH,
        type=Path,
        help="Pseudo-gold file. Default: dataset/test_gold.json.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Output directory. Default: test_goldanswer/<run_name>/.",
    )
    parser.add_argument(
        "--example-limit",
        default=30,
        type=int,
        help="Number of examples to keep in summary/report sections.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_prediction_file(
        pred_path=args.pred,
        gold_path=args.gold,
        out_dir=args.out_dir,
        example_limit=max(0, int(args.example_limit)),
    )


if __name__ == "__main__":
    main()
