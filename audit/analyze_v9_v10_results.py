from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from textwrap import shorten


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
VALID_PATH = ROOT / "dataset" / "valid.json"
OUT_JSON = ROOT / "audit" / "v9_v10_results_analysis.json"
OUT_MD = ROOT / "audit" / "v9_v10_results_analysis.md"

ANSWER_PREFIX = "\u0110\u00e1p \u00e1n"
EQ_PREFIX = "Ph\u00e9p t\u00ednh"
ANCHOR = "\u0110\u00e1p \u00e1n l\u00e0"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_root(run: str) -> Path:
    return RESULTS / run


def compact_summary(summary: dict) -> dict:
    buckets = summary.get("buckets", {})
    return {
        "n": summary.get("n"),
        "raw_score": summary.get("raw_score"),
        "score_10": summary.get("score_10"),
        "extractable": summary.get("extractable"),
        "numeric_pairs": summary.get("numeric_pairs"),
        "buckets": {str(k): buckets.get(k, buckets.get(str(k))) for k in [10, 5, 1, 0]},
        "rel_error_mean": summary.get("rel_error_mean"),
    }


def output_shape(path: Path) -> dict:
    rows = load_json(path)
    lens = [len(x.get("model_output", "")) for x in rows]
    cats = Counter()
    equals_counts = []
    for row in rows:
        text = row.get("model_output", "").strip()
        if text.startswith(EQ_PREFIX):
            cats["equation_plus_answer"] += 1
        elif text.startswith(ANSWER_PREFIX):
            cats["answer_only"] += 1
        elif ANCHOR in text:
            cats["other_with_answer_anchor"] += 1
        else:
            cats["no_answer_anchor"] += 1
        before = text.split(ANCHOR)[0]
        equals_counts.append(before.count("=") if before else 0)
    return {
        "n": len(rows),
        "length_mean": statistics.mean(lens) if lens else 0.0,
        "length_median": statistics.median(lens) if lens else 0.0,
        "length_max": max(lens) if lens else 0,
        "format_counts": dict(cats),
        "equation_equals_mean": statistics.mean(equals_counts) if equals_counts else 0.0,
        "equation_equals_median": statistics.median(equals_counts) if equals_counts else 0.0,
    }


def by_type_table(report: dict) -> dict:
    out = {}
    for type_name, item in sorted(report.get("by_type", {}).items()):
        out[type_name] = {
            "n": item.get("n"),
            "raw_score": item.get("raw_score"),
            "score_10": item.get("score_10"),
            "extractable": item.get("extractable"),
            "exact10": item.get("bucket_10"),
        }
    return out


def load_run(run: str) -> dict:
    root = run_root(run)
    manifest_name = "v9_compact_equation_manifest.json" if run == "v9" else "v10_curriculum_compact_manifest.json"
    data = {
        "manifest": load_json(root / manifest_name),
        "source_valid_report": load_json(root / "source_valid_report.json"),
        "valid_report": load_json(root / "valid_report.json"),
        "checkpoint_selection": load_json(root / "checkpoint_selection_report.json"),
        "coverage": load_json(root / "compact_equation_coverage.json"),
        "valid_output": load_json(root / "valid_output.json"),
        "source_valid_output": load_json(root / "source_valid_output.json"),
        "valid_overlap_audit": load_json(root / "valid_overlap_audit.json"),
    }
    return data


def case_payload(label: str, idx: int, valid_rows: list[dict], runs: dict[str, dict]) -> dict:
    payload = {
        "label": label,
        "idx": idx,
        "type": valid_rows[idx].get("type"),
        "query_vi": valid_rows[idx].get("query_vi"),
        "gold_answer": runs["v9"]["valid_report"]["rows"][idx]["gold_answer"],
        "v9": {},
        "v10": {},
    }
    for run in ["v9", "v10"]:
        row = runs[run]["valid_report"]["rows"][idx]
        out = runs[run]["valid_output"][idx]
        payload[run] = {
            "score": row.get("score"),
            "pred_answer": row.get("pred_answer"),
            "rel_error": row.get("rel_error"),
            "model_output": out.get("model_output"),
        }
    return payload


def choose_cases(valid_rows: list[dict], runs: dict[str, dict]) -> list[dict]:
    rows9 = runs["v9"]["valid_report"]["rows"]
    rows10 = runs["v10"]["valid_report"]["rows"]
    transitions = [(rows9[i]["score"], rows10[i]["score"], i) for i in range(len(valid_rows))]
    chosen = []
    chosen += [("v10_fix", i) for s9, s10, i in transitions if s10 == 10 and s9 < 10][:5]
    chosen += [("v10_regress", i) for s9, s10, i in transitions if s9 == 10 and s10 < 10][:5]
    chosen += [("both_wrong", i) for s9, s10, i in transitions if s9 == 0 and s10 == 0][:5]
    chosen += [("near_miss", i) for s9, s10, i in transitions if s9 == 5 or s10 == 5][:5]
    out = []
    seen = set()
    for label, idx in chosen:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(case_payload(label, idx, valid_rows, runs))
    return out


def main() -> None:
    valid_rows = load_json(VALID_PATH)
    runs = {run: load_run(run) for run in ["v9", "v10"]}

    analysis = {
        "runs": {},
        "valid_score_transitions_v9_to_v10": {},
        "cases": choose_cases(valid_rows, runs),
    }
    for run, data in runs.items():
        ckpt = data["checkpoint_selection"]
        analysis["runs"][run] = {
            "config": {
                k: data["manifest"]["config"].get(k)
                for k in [
                    "stage_a_target_mode",
                    "stage_a_epochs",
                    "stage_a_lr",
                    "run_stage_b",
                    "stage_b_epochs",
                    "stage_b_lr",
                    "run_stage_c",
                    "stage_c_epochs",
                    "stage_c_lr",
                    "mixed_compact_ratio",
                    "sanitize_to_answer_only",
                    "max_new_tokens",
                    "lora_target_modules",
                ]
            },
            "source_valid_summary": compact_summary(data["source_valid_report"]["summary"]),
            "reference_valid_summary": compact_summary(data["valid_report"]["summary"]),
            "valid_by_type": by_type_table(data["valid_report"]),
            "source_valid_by_type": by_type_table(data["source_valid_report"]),
            "selected_checkpoint": {
                "label": ckpt.get("selected", {}).get("label"),
                "eval_n": ckpt.get("eval_n"),
                "selection_split": ckpt.get("selection_split"),
                "summary": compact_summary(ckpt.get("selected", {}).get("summary", {})),
                "all_checkpoints": [
                    {
                        "label": item.get("label"),
                        "raw_score": item.get("summary", {}).get("raw_score"),
                        "score_10": item.get("summary", {}).get("score_10"),
                        "exact10": item.get("summary", {}).get("buckets", {}).get("10"),
                        "extractable": item.get("summary", {}).get("extractable"),
                    }
                    for item in ckpt.get("all_checkpoints", [])
                ],
            },
            "output_shape": {
                "source_valid": output_shape(run_root(run) / "source_valid_output.json"),
                "valid": output_shape(run_root(run) / "valid_output.json"),
            },
            "source_split": data["coverage"].get("source_split"),
            "coverage_stages": data["coverage"].get("stages"),
            "valid_overlap_audit": data["valid_overlap_audit"],
        }

    rows9 = runs["v9"]["valid_report"]["rows"]
    rows10 = runs["v10"]["valid_report"]["rows"]
    transition_counts = Counter((rows9[i]["score"], rows10[i]["score"]) for i in range(len(valid_rows)))
    analysis["valid_score_transitions_v9_to_v10"] = {
        f"{k[0]}->{k[1]}": v
        for k, v in sorted(transition_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    }

    OUT_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# V9/V10 Results Analysis",
        "",
        "## Summary",
    ]
    for run in ["v9", "v10"]:
        item = analysis["runs"][run]
        lines.append(
            f"- {run}: source-valid score={item['source_valid_summary']['score_10']:.3f}, "
            f"reference valid score={item['reference_valid_summary']['score_10']:.3f}, "
            f"selected={item['selected_checkpoint']['label']}"
        )
    lines += ["", "## Output Shape"]
    for run in ["v9", "v10"]:
        for split in ["source_valid", "valid"]:
            shape = analysis["runs"][run]["output_shape"][split]
            lines.append(
                f"- {run}/{split}: {shape['format_counts']}, "
                f"median_len={shape['length_median']}, mean_equals={shape['equation_equals_mean']:.2f}"
            )
    lines += ["", "## Score Transitions V9 -> V10"]
    for k, v in analysis["valid_score_transitions_v9_to_v10"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Representative Cases"]
    for case in analysis["cases"]:
        lines += [
            f"### {case['label']} idx={case['idx']} type={case['type']}",
            f"- Q: {shorten(case['query_vi'].replace(chr(10), ' '), width=220, placeholder='...')}",
            f"- Gold: {case['gold_answer']}",
        ]
        for run in ["v9", "v10"]:
            info = case[run]
            lines.append(
                f"- {run}: score={info['score']} pred={info['pred_answer']} err={info['rel_error']} | "
                f"{shorten(info['model_output'].replace(chr(10), ' | '), width=260, placeholder='...')}"
            )
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
