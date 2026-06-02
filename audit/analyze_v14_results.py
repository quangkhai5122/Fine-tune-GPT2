"""Compare V14 retrieval runs against earlier GPT-2 math experiments."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import extract_gold  # noqa: E402
from recheck_valid_train_overlap import digest, iter_json_array, normalize  # noqa: E402


TRAIN_PATH = ROOT / "dataset" / "train.json"
VALID_PATH = ROOT / "dataset" / "valid.json"
OUT_JSON = ROOT / "audit" / "v14_result_analysis.json"
OUT_MD = ROOT / "audit" / "v14_result_analysis.md"

RUNS = {
    "4beams": ROOT / "results" / "4beams",
    "v5": ROOT / "results" / "v5",
    "v6": ROOT / "results" / "v6",
    "v6_fast": ROOT / "results" / "v6_fast",
    "v7": ROOT / "results" / "v7",
    "v8_2_lora_target": ROOT / "results" / "v8" / "2_lora_target",
    "v8_3_lora_target": ROOT / "results" / "v8" / "3_lora_target",
    "v8_select_beam2_lr1e-3": ROOT / "results" / "v8_select_checkpoint" / "beam2_lr1e-3",
    "v8_select_beam2_lr5e-4": ROOT / "results" / "v8_select_checkpoint" / "beam2_lr5e-4",
    "v8_select_beam4_lr5e-4": ROOT / "results" / "v8_select_checkpoint" / "beam4_lr5e-4",
    "v9": ROOT / "results" / "v9",
    "v10": ROOT / "results" / "v10",
    "v11": ROOT / "results" / "v11",
    "v12": ROOT / "results" / "v12",
    "v13": ROOT / "results" / "v13",
    "v14_gate_sweep": ROOT / "results" / "v14_gate_sweep",
    "v14_fobar_sv": ROOT / "results" / "v14_fobar_sv",
}

FOCUS_RUNS = ["v13", "v14_gate_sweep", "v14_fobar_sv"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summary_from_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    report = load_json(path)
    summary = report["summary"]
    buckets = summary.get("buckets", {})
    return {
        "raw_score": summary["raw_score"],
        "score_10": summary["score_10"],
        "exact10": buckets.get("10", buckets.get(10, 0)),
        "score5": buckets.get("5", buckets.get(5, 0)),
        "score1": buckets.get("1", buckets.get(1, 0)),
        "zero": buckets.get("0", buckets.get(0, 0)),
        "extractable": summary["extractable"],
        "numeric_pairs": summary.get("numeric_pairs"),
    }


def build_same_original_classes(valid_records: list[dict]) -> dict[int, str]:
    valid_nums = {i: extract_gold(record)[1] for i, record in enumerate(valid_records)}
    lookup = defaultdict(list)
    for i, record in enumerate(valid_records):
        key = normalize(record.get("original_question_vi"), "strip")
        if key:
            lookup[digest(key)].append(i)

    flags = {i: Counter() for i in range(len(valid_records))}
    for record in iter_json_array(TRAIN_PATH):
        key = normalize(record.get("original_question_vi"), "strip")
        if not key:
            continue
        valid_ids = lookup.get(digest(key))
        if not valid_ids:
            continue
        train_num = extract_gold(record)[1]
        for valid_id in valid_ids:
            valid_num = valid_nums[valid_id]
            if train_num is None or valid_num is None:
                flags[valid_id]["missing_numeric"] += 1
            elif abs(train_num - valid_num) <= 1e-9:
                flags[valid_id]["same_numeric"] += 1
            else:
                flags[valid_id]["conflicting_numeric"] += 1

    classes = {}
    for i, counter in flags.items():
        if counter["same_numeric"]:
            classes[i] = "same_original_has_same"
        elif counter["conflicting_numeric"]:
            classes[i] = "same_original_only_conflict"
        elif counter["missing_numeric"]:
            classes[i] = "same_original_only_missing"
        else:
            classes[i] = "same_original_no_match"
    return classes


def slice_stats(rows: list[dict], ids: list[int]) -> dict:
    if not ids:
        return {"n": 0, "raw_score": 0, "score_10": None, "exact10": 0, "zero": 0}
    selected = [rows[i] for i in ids]
    raw = sum(row["score"] for row in selected)
    return {
        "n": len(selected),
        "raw_score": raw,
        "score_10": raw / len(selected),
        "exact10": sum(1 for row in selected if row["score"] == 10),
        "zero": sum(1 for row in selected if row["score"] == 0),
    }


def by_type_delta(final_report: dict, model_report: dict | None) -> dict:
    out = {}
    for rec_type, final in final_report["by_type"].items():
        item = {
            "n": final["n"],
            "final_raw": final["raw_score"],
            "final_score_10": final["score_10"],
            "final_exact10": final["bucket_10"],
            "final_zero": final["bucket_0"],
        }
        if model_report and rec_type in model_report["by_type"]:
            model = model_report["by_type"][rec_type]
            item.update(
                {
                    "model_raw": model["raw_score"],
                    "model_score_10": model["score_10"],
                    "model_exact10": model["bucket_10"],
                    "model_zero": model["bucket_0"],
                    "delta_raw": final["raw_score"] - model["raw_score"],
                    "delta_exact10": final["bucket_10"] - model["bucket_10"],
                }
            )
        out[rec_type] = item
    return out


def transition_stats(final_report: dict, model_report: dict | None) -> dict | None:
    if not model_report:
        return None
    transitions = Counter()
    by_type = defaultdict(Counter)
    for final_row, model_row in zip(final_report["rows"], model_report["rows"]):
        key = f"{model_row['score']}->{final_row['score']}"
        transitions[key] += 1
        by_type[final_row["type"]][key] += 1
    return {
        "overall": dict(sorted(transitions.items())),
        "by_type": {k: dict(sorted(v.items())) for k, v in sorted(by_type.items())},
    }


def checkpoint_summary(run_dir: Path) -> dict | None:
    path = run_dir / "checkpoint_selection_report.json"
    if not path.exists():
        return None
    report = load_json(path)
    entries = []
    for entry in report.get("all_checkpoints", []):
        s = entry.get("summary", {})
        entries.append(
            {
                "label": entry.get("label"),
                "raw_score": s.get("raw_score"),
                "score_10": s.get("score_10"),
                "exact10": s.get("buckets", {}).get("10"),
                "extractable": s.get("extractable"),
                "gate_sweep_path": entry.get("gate_sweep_path"),
            }
        )
    return {
        "status": report.get("status"),
        "selected_label": (report.get("selected") or {}).get("label"),
        "selected_raw": ((report.get("selected") or {}).get("summary") or {}).get("raw_score"),
        "selected_exact10": (((report.get("selected") or {}).get("summary") or {}).get("buckets") or {}).get("10"),
        "final_retrain_epoch_count": (report.get("final_retrain") or {}).get("selected_epoch_count"),
        "selected_retrieval_gate_config_by_type": report.get("selected_retrieval_gate_config_by_type"),
        "checkpoints": entries,
    }


def focus_run_analysis(run_name: str, run_dir: Path, same_original_classes: dict[int, str]) -> dict:
    final_report = load_json(run_dir / "valid_report.json")
    model_report = load_json(run_dir / "model_valid_report.json") if (run_dir / "model_valid_report.json").exists() else None
    final_summary = summary_from_report(run_dir / "valid_report.json")
    model_summary = summary_from_report(run_dir / "model_valid_report.json") if model_report else None
    hybrid = load_json(run_dir / "hybrid_decision_report.json") if (run_dir / "hybrid_decision_report.json").exists() else None
    selected_gate = load_json(run_dir / "selected_retrieval_gate_config.json") if (run_dir / "selected_retrieval_gate_config.json").exists() else None
    gate_sweep = load_json(run_dir / "retrieval_gate_sweep_report.json") if (run_dir / "retrieval_gate_sweep_report.json").exists() else None

    class_to_ids = defaultdict(list)
    for i, cls in same_original_classes.items():
        class_to_ids[cls].append(i)
    overlap_slices = {cls: slice_stats(final_report["rows"], ids) for cls, ids in sorted(class_to_ids.items())}

    return {
        "name": run_name,
        "dir": str(run_dir.relative_to(ROOT)),
        "final_summary": final_summary,
        "model_summary": model_summary,
        "by_type_delta": by_type_delta(final_report, model_report),
        "transitions": transition_stats(final_report, model_report),
        "hybrid_decision_summary": hybrid,
        "selected_gate_config": selected_gate,
        "gate_sweep_report": gate_sweep,
        "checkpoint_selection": checkpoint_summary(run_dir),
        "same_original_overlap_slices": overlap_slices,
    }


def make_markdown(analysis: dict) -> str:
    lines = []
    lines.append("# V14 result analysis")
    lines.append("")
    lines.append("## Leaderboard so far")
    lines.append("| run | raw | score/10 | exact10 | zero | extractable |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, item in sorted(
        analysis["all_runs"].items(),
        key=lambda kv: (kv[1]["valid"]["raw_score"] if kv[1]["valid"] else -1),
        reverse=True,
    ):
        s = item["valid"]
        if not s:
            continue
        lines.append(
            f"| {name} | {s['raw_score']} | {s['score_10']:.3f} | {s['exact10']} | {s['zero']} | {s['extractable']} |"
        )

    lines.append("")
    lines.append("## V13/V14 model vs hybrid")
    lines.append("| run | model raw | final raw | delta | model exact10 | final exact10 | selected ckpt | final retrain epochs |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|")
    for name in FOCUS_RUNS:
        item = analysis["focus_runs"][name]
        final_s = item["final_summary"]
        model_s = item["model_summary"] or {"raw_score": 0, "exact10": 0}
        ckpt = item["checkpoint_selection"] or {}
        lines.append(
            f"| {name} | {model_s['raw_score']} | {final_s['raw_score']} | {final_s['raw_score'] - model_s['raw_score']} | "
            f"{model_s['exact10']} | {final_s['exact10']} | {ckpt.get('selected_label')} | {ckpt.get('final_retrain_epoch_count')} |"
        )

    lines.append("")
    lines.append("## Type-level scores")
    for name in FOCUS_RUNS:
        item = analysis["focus_runs"][name]
        lines.append(f"### {name}")
        lines.append("| type | n | model/10 | final/10 | delta raw | final exact10 | final zero |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for rec_type, row in item["by_type_delta"].items():
            lines.append(
                f"| {rec_type} | {row['n']} | {row.get('model_score_10', 0):.3f} | {row['final_score_10']:.3f} | "
                f"{row.get('delta_raw', 0)} | {row['final_exact10']} | {row['final_zero']} |"
            )
        lines.append("")

    lines.append("## Retrieval routing")
    lines.append("| run | retrieval used | used pct | output changed | fallback reasons |")
    lines.append("|---|---:|---:|---:|---|")
    for name in FOCUS_RUNS:
        hybrid = analysis["focus_runs"][name]["hybrid_decision_summary"] or {}
        lines.append(
            f"| {name} | {hybrid.get('retrieval_used')} | {hybrid.get('retrieval_used_pct', 0):.3f} | "
            f"{hybrid.get('output_changed_count', 'n/a')} | {hybrid.get('fallback_reason_counts')} |"
        )

    lines.append("")
    lines.append("## Same-original overlap slices")
    lines.append("| run | slice | n | score/10 | exact10 | zero |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for name in FOCUS_RUNS:
        for cls, stats in analysis["focus_runs"][name]["same_original_overlap_slices"].items():
            score = stats["score_10"]
            score_s = "n/a" if score is None else f"{score:.3f}"
            lines.append(f"| {name} | {cls} | {stats['n']} | {score_s} | {stats['exact10']} | {stats['zero']} |")

    lines.append("")
    lines.append("## Selected V14 gate configs")
    for name in ["v14_gate_sweep", "v14_fobar_sv"]:
        lines.append(f"### {name}")
        lines.append("```json")
        lines.append(json.dumps(analysis["focus_runs"][name]["selected_gate_config"], ensure_ascii=False, indent=2))
        lines.append("```")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("- Best current valid score remains v13: 6896 raw / 6.896.")
    lines.append("- v14_gate_sweep is second among current runs: 6806 raw / 6.806; it improves a few allowed retrieval types but loses on model-only fallback types.")
    lines.append("- v14_fobar_sv is not a candidate for direct continuation because strict FOBAR/SV retrieval still hurts conflict-heavy FOBAR and does not offset weaker model-only training.")
    lines.append("- For the next full-train, select-on-valid runs, keep v13 as the exploitation baseline and v14_gate_sweep as the safer gate-tuned challenger.")
    return "\n".join(lines) + "\n"


def main() -> None:
    valid_records = load_json(VALID_PATH)
    same_original_classes = build_same_original_classes(valid_records)

    all_runs = {}
    for name, run_dir in RUNS.items():
        valid_summary = summary_from_report(run_dir / "valid_report.json")
        if valid_summary is None:
            continue
        all_runs[name] = {
            "dir": str(run_dir.relative_to(ROOT)),
            "valid": valid_summary,
            "model_valid": summary_from_report(run_dir / "model_valid_report.json"),
        }

    focus = {name: focus_run_analysis(name, RUNS[name], same_original_classes) for name in FOCUS_RUNS}

    analysis = {
        "all_runs": all_runs,
        "focus_runs": focus,
    }
    OUT_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(make_markdown(analysis), encoding="utf-8")
    print("wrote", OUT_JSON)
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
