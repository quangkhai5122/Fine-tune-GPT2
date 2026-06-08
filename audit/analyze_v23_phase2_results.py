from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import extract_answer, load_records, parse_number, rel_error, score_one  # noqa: E402


GOLD_PATH = ROOT / "dataset" / "test_gold.json"
OUT_JSON = ROOT / "audit" / "v23_phase2_result_analysis.json"
OUT_MD = ROOT / "audit" / "v23_phase2_result_analysis.md"

RUNS = {
    "v17_phase2": ROOT / "results" / "v17_phase2" / "test_predictions.json",
    "v21_phase2": ROOT / "results" / "v21_phase2" / "test_predictions.json",
    "v23_phase2": ROOT / "results" / "v23_phase2" / "test_predictions.json",
    "v22_v17": ROOT / "results" / "v22_v17" / "test_predictions.json",
    "v22_v21": ROOT / "results" / "v22_v21" / "test_predictions.json",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def answer_num_from_output(item: dict) -> tuple[str | None, float | None]:
    answer = extract_answer(item.get("model_output"), include_english=True)
    return answer, parse_number(answer)


def answer_num_from_gold(item: dict) -> tuple[str | None, float | None]:
    answer = extract_answer(item.get("response_vi"), include_english=True)
    return answer, parse_number(answer)


def evaluate_run(pred_path: Path, gold_items: list[dict]) -> list[dict]:
    pred_items = load_records(pred_path)
    pred_by_id = {str(item.get("id")): item for item in pred_items}
    rows = []
    for idx, gold in enumerate(gold_items):
        sid = str(gold.get("id", idx))
        pred = pred_by_id[sid]
        pred_answer, pred_num = answer_num_from_output(pred)
        gold_answer, gold_num = answer_num_from_gold(gold)
        error = rel_error(pred_num, gold_num)
        score = score_one(error, pred_answer is not None)
        rows.append(
            {
                "id": gold.get("id", idx),
                "type": gold.get("type") or pred.get("type") or "<missing>",
                "query_vi": gold.get("query_vi") or pred.get("query_vi"),
                "model_output": pred.get("model_output"),
                "pred_answer": pred_answer,
                "pred_num": pred_num,
                "gold_answer": gold_answer,
                "gold_num": gold_num,
                "rel_error": error,
                "score": score,
                "extractable": pred_answer is not None,
            }
        )
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    raw = sum(int(row["score"]) for row in rows)
    buckets = Counter(int(row["score"]) for row in rows)
    errors = [
        float(row["rel_error"])
        for row in rows
        if isinstance(row.get("rel_error"), (int, float)) and math.isfinite(float(row["rel_error"]))
    ]
    by_type = defaultdict(lambda: Counter(n=0, raw=0, extractable=0, numeric=0, b10=0, b5=0, b1=0, b0=0))
    for row in rows:
        counter = by_type[row["type"]]
        score = int(row["score"])
        counter["n"] += 1
        counter["raw"] += score
        counter["extractable"] += int(bool(row["extractable"]))
        counter["numeric"] += int(row["pred_num"] is not None and row["gold_num"] is not None)
        counter[f"b{score}"] += 1
    by_type_out = {}
    for typ, counter in sorted(by_type.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        n = int(counter["n"])
        by_type_out[typ] = {
            "n": n,
            "raw": int(counter["raw"]),
            "score10": counter["raw"] / n if n else 0,
            "exact10": int(counter["b10"]),
            "bucket5": int(counter["b5"]),
            "bucket1": int(counter["b1"]),
            "bucket0": int(counter["b0"]),
            "extractable": int(counter["extractable"]),
            "numeric": int(counter["numeric"]),
        }
    return {
        "n": len(rows),
        "raw": raw,
        "max_raw": 10 * len(rows),
        "score10": raw / len(rows) if rows else 0,
        "buckets": {str(k): int(v) for k, v in sorted(buckets.items(), reverse=True)},
        "extractable": sum(int(row["extractable"]) for row in rows),
        "numeric_pairs": sum(int(row["pred_num"] is not None and row["gold_num"] is not None) for row in rows),
        "rel_error_median": median(errors) if errors else None,
        "rel_error_mean": mean(errors) if errors else None,
        "by_type": by_type_out,
        "top_pred_answers": Counter(row["pred_answer"] or "<none>" for row in rows).most_common(20),
    }


def compare_pair(base_name: str, base_rows: list[dict], target_name: str, target_rows: list[dict]) -> dict:
    base_by_id = {row["id"]: row for row in base_rows}
    target_by_id = {row["id"]: row for row in target_rows}
    deltas = []
    same_num = 0
    both_num = 0
    by_type = defaultdict(lambda: Counter(n=0, delta=0, better=0, worse=0, same=0))
    for rid, target in target_by_id.items():
        base = base_by_id[rid]
        delta = int(target["score"]) - int(base["score"])
        if target["pred_num"] is not None and base["pred_num"] is not None:
            both_num += 1
            same_num += int(abs(float(target["pred_num"]) - float(base["pred_num"])) <= 1e-9)
        counter = by_type[target["type"]]
        counter["n"] += 1
        counter["delta"] += delta
        counter["better"] += int(delta > 0)
        counter["worse"] += int(delta < 0)
        counter["same"] += int(delta == 0)
        if delta:
            deltas.append(
                {
                    "id": rid,
                    "type": target["type"],
                    "delta": delta,
                    base_name + "_score": base["score"],
                    target_name + "_score": target["score"],
                    base_name + "_pred": base["pred_answer"],
                    target_name + "_pred": target["pred_answer"],
                    "gold_answer": target["gold_answer"],
                    "query_vi": target["query_vi"],
                }
            )
    return {
        "base": base_name,
        "target": target_name,
        "raw_delta": sum(item["delta"] for item in deltas),
        "changed_score_rows": len(deltas),
        "same_numeric": same_num,
        "both_numeric": both_num,
        "by_type": {k: dict(v) for k, v in sorted(by_type.items(), key=lambda kv: (-kv[1]["n"], kv[0]))},
        "best_examples": sorted([d for d in deltas if d["delta"] > 0], key=lambda x: (-x["delta"], x["id"]))[:25],
        "worst_examples": sorted([d for d in deltas if d["delta"] < 0], key=lambda x: (x["delta"], x["id"]))[:25],
    }


def profile_selection_summary() -> dict:
    path = ROOT / "results" / "v23_phase2" / "checkpoint_selection_report.json"
    if not path.exists():
        return {}
    data = load_json(path)
    rows = []
    for item in data.get("profiles", []):
        summary = item.get("summary", {})
        profile_summary = item.get("profile_summary", {})
        rows.append(
            {
                "name": item.get("profile", {}).get("name"),
                "valid_raw": summary.get("raw_score"),
                "valid_score10": summary.get("score_10"),
                "valid_buckets": summary.get("buckets"),
                "valid_extractable": summary.get("extractable"),
                "changed_from_model": profile_summary.get("changed_from_model"),
                "source_counts": profile_summary.get("source_counts"),
                "retrieval_reason_counts": profile_summary.get("retrieval_reason_counts"),
                "sanity": item.get("sanity"),
            }
        )
    return {
        "strategy": data.get("strategy"),
        "selection_metric": data.get("selection_metric"),
        "selected_name": data.get("selected", {}).get("profile", {}).get("name"),
        "profiles": rows,
    }


def manifest_summary() -> dict:
    path = ROOT / "results" / "v23_phase2" / "v23_legal_no_solver_ranker_manifest.json"
    if not path.exists():
        return {}
    data = load_json(path)
    trained = data.get("trained_runs") or []
    wall = trained[0].get("wall_seconds") if trained else None
    return {
        "notebook_version": data.get("notebook_version"),
        "run_mode": data.get("run_mode"),
        "rule_compliance": data.get("rule_compliance"),
        "config": {
            k: data.get("config", {}).get(k)
            for k in [
                "stage_a_epochs",
                "stage_a_lr",
                "num_beams",
                "max_new_tokens",
                "ensemble_last_k_epochs",
                "legal_query_retrieval_enabled",
            ]
        },
        "train_clean_n": data.get("data", {}).get("train_clean_n"),
        "valid_clean_n": data.get("data", {}).get("valid_clean_n"),
        "wall_seconds_training": wall,
        "wall_minutes_training": wall / 60 if isinstance(wall, (int, float)) else None,
        "selection_summary": data.get("selection_summary"),
    }


def decision_source_summary(rows: list[dict]) -> dict:
    path = ROOT / "results" / "v23_phase2" / "test_v23_ranker_decisions.json"
    if not path.exists():
        return {}
    decisions = load_json(path)
    by_id = {row["id"]: row for row in rows}
    out = defaultdict(lambda: Counter(n=0, raw=0, b10=0, b5=0, b1=0, b0=0))
    for decision in decisions:
        row = by_id[decision["id"]]
        counter = out[decision.get("source", "<missing>")]
        score = int(row["score"])
        counter["n"] += 1
        counter["raw"] += score
        counter[f"b{score}"] += 1
    return {k: dict(v) for k, v in out.items()}


def oracle_summary(all_rows: dict[str, list[dict]], names: list[str]) -> dict:
    row_maps = {name: {row["id"]: row for row in all_rows[name]} for name in names if name in all_rows}
    if not row_maps:
        return {}
    ids = sorted(next(iter(row_maps.values())).keys())
    raw = 0
    buckets = Counter()
    chosen_counts = Counter()
    by_type = defaultdict(lambda: Counter(n=0, raw=0))
    examples = []
    for rid in ids:
        candidates = [(name, row_maps[name][rid]) for name in row_maps]
        name, best = max(candidates, key=lambda item: (item[1]["score"], item[0]))
        raw += int(best["score"])
        buckets[int(best["score"])] += 1
        chosen_counts[name] += 1
        by_type[best["type"]]["n"] += 1
        by_type[best["type"]]["raw"] += int(best["score"])
        if int(best["score"]) == 10 and len(examples) < 20:
            examples.append(
                {
                    "id": rid,
                    "type": best["type"],
                    "chosen_run": name,
                    "pred": best["pred_answer"],
                    "gold": best["gold_answer"],
                }
            )
    return {
        "runs": list(row_maps),
        "raw": raw,
        "score10": raw / len(ids) if ids else 0,
        "buckets": {str(k): int(v) for k, v in sorted(buckets.items(), reverse=True)},
        "chosen_counts": dict(chosen_counts.most_common()),
        "by_type": {k: dict(v) for k, v in by_type.items()},
        "examples": examples,
    }


def main() -> None:
    gold_items = load_records(GOLD_PATH)
    all_rows = {name: evaluate_run(path, gold_items) for name, path in RUNS.items() if path.exists()}
    summaries = {name: summarize_rows(rows) for name, rows in all_rows.items()}
    comparisons = {}
    if "v23_phase2" in all_rows:
        for base in ["v17_phase2", "v21_phase2"]:
            if base in all_rows:
                comparisons[f"v23_vs_{base}"] = compare_pair(base, all_rows[base], "v23_phase2", all_rows["v23_phase2"])

    analysis = {
        "note": "Pseudo-gold analysis using dataset/test_gold.json. This is not official leaderboard gold.",
        "summaries": summaries,
        "v23_profile_selection": profile_selection_summary(),
        "v23_manifest": manifest_summary(),
        "v23_decision_sources": decision_source_summary(all_rows.get("v23_phase2", [])),
        "comparisons": comparisons,
        "oracle_no_solver_v17_v21_v23": oracle_summary(all_rows, ["v17_phase2", "v21_phase2", "v23_phase2"]),
    }
    OUT_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# V23 Phase2 Result Analysis",
        "",
        "Pseudo-gold analysis using `dataset/test_gold.json`; this is not official leaderboard gold.",
        "",
        "## Score Summary",
        "| run | raw | score10 | b10 | b5 | b1 | b0 | extractable | numeric_pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        buckets = summary["buckets"]
        lines.append(
            f"| {name} | {summary['raw']} | {summary['score10']:.4f} | "
            f"{buckets.get('10', 0)} | {buckets.get('5', 0)} | {buckets.get('1', 0)} | {buckets.get('0', 0)} | "
            f"{summary['extractable']} | {summary['numeric_pairs']} |"
        )

    lines.extend(["", "## V23 Valid Profile Selection", "| profile | valid_raw | exact10 | changed | source_counts | retrieval_reasons |", "|---|---:|---:|---:|---|---|"])
    for row in analysis["v23_profile_selection"].get("profiles", []):
        buckets = row.get("valid_buckets") or {}
        lines.append(
            f"| {row['name']} | {row['valid_raw']} | {buckets.get('10', 0)} | "
            f"{row.get('changed_from_model')} | {row.get('source_counts')} | {row.get('retrieval_reason_counts')} |"
        )

    lines.extend(["", "## V23 By Type", "| type | raw | score10 | b10 | b5 | b1 | b0 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for typ, row in summaries.get("v23_phase2", {}).get("by_type", {}).items():
        lines.append(
            f"| {typ} | {row['raw']} | {row['score10']:.4f} | {row['exact10']} | {row['bucket5']} | {row['bucket1']} | {row['bucket0']} |"
        )

    lines.extend(["", "## Deltas"])
    for name, comp in comparisons.items():
        lines.append(f"### {name}")
        lines.append(
            f"- raw_delta={comp['raw_delta']}, changed_score_rows={comp['changed_score_rows']}, "
            f"same_numeric={comp['same_numeric']}/{comp['both_numeric']}"
        )
        lines.append("| type | delta | better | worse | same |")
        lines.append("|---|---:|---:|---:|---:|")
        for typ, row in comp["by_type"].items():
            lines.append(f"| {typ} | {row['delta']} | {row['better']} | {row['worse']} | {row['same']} |")

    oracle = analysis["oracle_no_solver_v17_v21_v23"]
    lines.extend(["", "## Oracle No-Solver Ensemble"])
    lines.append(f"- runs: {oracle.get('runs')}")
    lines.append(f"- raw: {oracle.get('raw')} score10={oracle.get('score10')}")
    lines.append(f"- buckets: {oracle.get('buckets')}")
    lines.append(f"- chosen_counts: {oracle.get('chosen_counts')}")

    manifest = analysis["v23_manifest"]
    lines.extend(["", "## Runtime / Compliance"])
    lines.append(f"- config: {manifest.get('config')}")
    lines.append(f"- wall_minutes_training: {manifest.get('wall_minutes_training')}")
    lines.append(f"- rule_compliance: {manifest.get('rule_compliance')}")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[wrote]", OUT_JSON.relative_to(ROOT))
    print("[wrote]", OUT_MD.relative_to(ROOT))


if __name__ == "__main__":
    main()
