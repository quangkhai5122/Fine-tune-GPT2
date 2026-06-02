"""Analyze V14.5 valid-select GPT-2 math runs and compare with prior runs."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import extract_gold, extract_pred  # noqa: E402
from recheck_valid_train_overlap import digest, iter_json_array, normalize  # noqa: E402


TRAIN_PATH = ROOT / "dataset" / "train.json"
VALID_PATH = ROOT / "dataset" / "valid.json"
OUT_JSON = ROOT / "audit" / "v145_result_analysis.json"
OUT_MD = ROOT / "audit" / "v145_result_analysis.md"

RUNS = {
    "v8_select_beam2_lr1e-3": ROOT / "results" / "v8_select_checkpoint" / "beam2_lr1e-3",
    "v12": ROOT / "results" / "v12",
    "v13": ROOT / "results" / "v13",
    "v14_gate_sweep": ROOT / "results" / "v14_gate_sweep",
    "v14_fobar_sv": ROOT / "results" / "v14_fobar_sv",
    "v14_5_v13": ROOT / "results" / "v14_5_v13",
    "v14_5_gate_sweep": ROOT / "results" / "v14_5_gate_sweep",
}

FOCUS_RUNS = ["v13", "v14_gate_sweep", "v14_5_v13", "v14_5_gate_sweep"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def report_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    report = load_json(path)
    s = report["summary"]
    b = s["buckets"]
    return {
        "raw_score": s["raw_score"],
        "score_10": s["score_10"],
        "exact10": b.get("10", b.get(10, 0)),
        "score5": b.get("5", b.get(5, 0)),
        "score1": b.get("1", b.get(1, 0)),
        "zero": b.get("0", b.get(0, 0)),
        "extractable": s["extractable"],
        "numeric_pairs": s.get("numeric_pairs"),
    }


def selected_decision_report(run_dir: Path) -> dict | None:
    direct = run_dir / "hybrid_decision_report.json"
    if direct.exists():
        return load_json(direct)
    selection_path = run_dir / "checkpoint_selection_report.json"
    if not selection_path.exists():
        return None
    selection = load_json(selection_path)
    selected = selection.get("selected") or {}
    decision_path = selected.get("hybrid_decision_path")
    if decision_path:
        local = run_dir / Path(decision_path).parent.name / Path(decision_path).name
        if local.exists():
            return load_json(local)
    return None


def checkpoint_summary(run_dir: Path) -> dict | None:
    path = run_dir / "checkpoint_selection_report.json"
    if not path.exists():
        return None
    report = load_json(path)
    selected = report.get("selected") or {}
    rows = []
    for entry in report.get("all_checkpoints", []):
        s = entry.get("summary") or {}
        b = s.get("buckets") or {}
        rows.append(
            {
                "label": entry.get("label"),
                "raw_score": s.get("raw_score"),
                "score_10": s.get("score_10"),
                "exact10": b.get("10"),
                "score5": b.get("5"),
                "score1": b.get("1"),
                "zero": b.get("0"),
                "extractable": s.get("extractable"),
                "gate_sweep_path": entry.get("gate_sweep_path"),
            }
        )
    selected_s = selected.get("summary") or {}
    selected_b = selected_s.get("buckets") or {}
    return {
        "selection_split": report.get("selection_split"),
        "selection_metric": report.get("selection_metric"),
        "selected_label": selected.get("label"),
        "selected_raw": selected_s.get("raw_score"),
        "selected_exact10": selected_b.get("10"),
        "selected_zero": selected_b.get("0"),
        "selected_extractable": selected_s.get("extractable"),
        "checkpoints": rows,
    }


def build_same_original_classes(valid_records: list[dict]) -> dict[int, str]:
    valid_nums = {i: extract_gold(record)[1] for i, record in enumerate(valid_records)}
    lookup = defaultdict(list)
    for i, record in enumerate(valid_records):
        value = normalize(record.get("original_question_vi"), "strip")
        if value:
            lookup[digest(value)].append(i)

    flags = {i: Counter() for i in range(len(valid_records))}
    for record in iter_json_array(TRAIN_PATH):
        value = normalize(record.get("original_question_vi"), "strip")
        if not value:
            continue
        valid_ids = lookup.get(digest(value))
        if not valid_ids:
            continue
        train_num = extract_gold(record)[1]
        for i in valid_ids:
            valid_num = valid_nums[i]
            if train_num is None or valid_num is None:
                flags[i]["missing_numeric"] += 1
            elif abs(train_num - valid_num) <= 1e-9:
                flags[i]["same_numeric"] += 1
            else:
                flags[i]["conflicting_numeric"] += 1

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
    helpful = transitions["0->10"] + transitions["1->10"] + transitions["5->10"]
    harmful = transitions["10->0"] + transitions["10->1"] + transitions["10->5"] + transitions["5->0"] + transitions["1->0"]
    return {
        "overall": dict(sorted(transitions.items())),
        "helpful_to_10": helpful,
        "harmful_major": harmful,
        "by_type": {k: dict(sorted(v.items())) for k, v in sorted(by_type.items())},
    }


def output_stats(output_path: Path, report_path: Path) -> dict | None:
    if not output_path.exists() or not report_path.exists():
        return None
    outputs = load_json(output_path)
    report = load_json(report_path)
    texts = [item.get("model_output", "") for item in outputs]
    pred_nums = []
    for item in outputs:
        _answer, num = extract_pred(item)
        pred_nums.append(num)
    exact_preds = Counter()
    zero_preds = Counter()
    for item, row in zip(outputs, report["rows"]):
        pred = row.get("pred_answer")
        if row["score"] == 10:
            exact_preds[str(pred)] += 1
        if row["score"] == 0:
            zero_preds[str(pred)] += 1
    return {
        "n": len(outputs),
        "mean_chars": mean(len(t) for t in texts) if texts else 0,
        "median_chars": median(len(t) for t in texts) if texts else 0,
        "single_short_rate": sum(1 for t in texts if "\n" not in t.strip() and len(t.strip()) <= 40) / len(texts) if texts else 0,
        "unique_numeric_predictions": len({x for x in pred_nums if x is not None}),
        "top_exact10_pred_answers": exact_preds.most_common(12),
        "top_zero_pred_answers": zero_preds.most_common(12),
    }


def run_analysis(run_name: str, run_dir: Path, same_original_classes: dict[int, str]) -> dict:
    final_report = load_json(run_dir / "valid_report.json")
    model_report = load_json(run_dir / "model_valid_report.json") if (run_dir / "model_valid_report.json").exists() else None

    class_ids = defaultdict(list)
    for i, cls in same_original_classes.items():
        class_ids[cls].append(i)

    return {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "valid_summary": report_summary(run_dir / "valid_report.json"),
        "model_valid_summary": report_summary(run_dir / "model_valid_report.json"),
        "checkpoint_selection": checkpoint_summary(run_dir),
        "hybrid_decision": selected_decision_report(run_dir),
        "selected_gate_config": load_json(run_dir / "selected_retrieval_gate_config.json") if (run_dir / "selected_retrieval_gate_config.json").exists() else None,
        "by_type_delta": by_type_delta(final_report, model_report),
        "transitions": transition_stats(final_report, model_report),
        "same_original_slices": {cls: slice_stats(final_report["rows"], ids) for cls, ids in sorted(class_ids.items())},
        "output_stats": output_stats(run_dir / "valid_output.json", run_dir / "valid_report.json"),
        "model_output_stats": output_stats(run_dir / "model_valid_output.json", run_dir / "model_valid_report.json") if model_report else None,
    }


def md_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    out = []
    for ri, row in enumerate(rows):
        out.append("| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |")
        if ri == 0:
            out.append("| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |")
    return out


def make_markdown(analysis: dict) -> str:
    lines: list[str] = ["# V14.5 result analysis", ""]

    leaderboard = []
    for name, item in sorted(
        analysis["runs"].items(),
        key=lambda kv: kv[1]["valid_summary"]["raw_score"],
        reverse=True,
    ):
        s = item["valid_summary"]
        leaderboard.append(
            [
                name,
                str(s["raw_score"]),
                f"{s['score_10']:.3f}",
                str(s["exact10"]),
                str(s["score5"]),
                str(s["score1"]),
                str(s["zero"]),
                str(s["extractable"]),
            ]
        )
    lines.append("## Leaderboard")
    lines.extend(md_table([["run", "raw", "score/10", "exact10", "5", "1", "0", "extractable"]] + leaderboard))
    lines.append("")

    lines.append("## Model-only vs final hybrid")
    rows = [["run", "model raw", "final raw", "delta", "model exact10", "final exact10", "selected", "split"]]
    for name in FOCUS_RUNS:
        item = analysis["runs"][name]
        f = item["valid_summary"]
        m = item["model_valid_summary"] or {"raw_score": 0, "exact10": 0}
        ck = item["checkpoint_selection"] or {}
        rows.append(
            [
                name,
                str(m["raw_score"]),
                str(f["raw_score"]),
                str(f["raw_score"] - m["raw_score"]),
                str(m["exact10"]),
                str(f["exact10"]),
                str(ck.get("selected_label")),
                str(ck.get("selection_split")),
            ]
        )
    lines.extend(md_table(rows))
    lines.append("")

    lines.append("## V14.5 checkpoint curves")
    for name in ["v14_5_v13", "v14_5_gate_sweep"]:
        lines.append(f"### {name}")
        rows = [["ckpt", "raw", "exact10", "5", "1", "0", "extractable"]]
        for ckpt in analysis["runs"][name]["checkpoint_selection"]["checkpoints"]:
            rows.append(
                [
                    str(ckpt["label"]),
                    str(ckpt["raw_score"]),
                    str(ckpt["exact10"]),
                    str(ckpt["score5"]),
                    str(ckpt["score1"]),
                    str(ckpt["zero"]),
                    str(ckpt["extractable"]),
                ]
            )
        lines.extend(md_table(rows))
        lines.append("")

    lines.append("## Type-level comparison")
    for name in FOCUS_RUNS:
        lines.append(f"### {name}")
        rows = [["type", "n", "model/10", "final/10", "delta raw", "final exact10", "final zero"]]
        for rec_type, row in analysis["runs"][name]["by_type_delta"].items():
            rows.append(
                [
                    rec_type,
                    str(row["n"]),
                    f"{row.get('model_score_10', 0):.3f}",
                    f"{row['final_score_10']:.3f}",
                    str(row.get("delta_raw", 0)),
                    str(row["final_exact10"]),
                    str(row["final_zero"]),
                ]
            )
        lines.extend(md_table(rows))
        lines.append("")

    lines.append("## Retrieval routing")
    rows = [["run", "used", "used pct", "changed", "fallback reasons"]]
    for name in FOCUS_RUNS:
        h = analysis["runs"][name]["hybrid_decision"] or {}
        rows.append(
            [
                name,
                str(h.get("retrieval_used")),
                f"{h.get('retrieval_used_pct', 0):.3f}" if h else "n/a",
                str(h.get("output_changed_count", "n/a")),
                str(h.get("fallback_reason_counts")),
            ]
        )
    lines.extend(md_table(rows))
    lines.append("")

    lines.append("## Model-to-final transitions")
    rows = [["run", "helpful to 10", "harmful major", "transitions"]]
    for name in FOCUS_RUNS:
        t = analysis["runs"][name]["transitions"] or {}
        rows.append([name, str(t.get("helpful_to_10")), str(t.get("harmful_major")), str(t.get("overall"))])
    lines.extend(md_table(rows))
    lines.append("")

    lines.append("## Same-original slices")
    rows = [["run", "slice", "n", "score/10", "exact10", "zero"]]
    for name in FOCUS_RUNS:
        for cls, stats in analysis["runs"][name]["same_original_slices"].items():
            score = stats["score_10"]
            rows.append([name, cls, str(stats["n"]), "n/a" if score is None else f"{score:.3f}", str(stats["exact10"]), str(stats["zero"])])
    lines.extend(md_table(rows))
    lines.append("")

    lines.append("## Output stats")
    rows = [["run", "mean chars", "median chars", "short single-line", "unique numeric preds"]]
    for name in FOCUS_RUNS:
        s = analysis["runs"][name]["output_stats"]
        rows.append(
            [
                name,
                f"{s['mean_chars']:.2f}",
                f"{s['median_chars']:.0f}",
                f"{s['single_short_rate']:.3f}",
                str(s["unique_numeric_predictions"]),
            ]
        )
    lines.extend(md_table(rows))
    lines.append("")

    lines.append("## Oracle ensemble upper bounds")
    rows = [["candidate runs", "oracle raw", "score/10", "exact10", "zero_all", "improved_vs_first"]]
    for item in analysis["oracle_ensembles"]:
        rows.append(
            [
                " + ".join(item["runs"]),
                str(item["raw_score"]),
                f"{item['score_10']:.3f}",
                str(item["exact10"]),
                str(item["zero_all"]),
                str(item["improved_vs_first"]),
            ]
        )
    lines.extend(md_table(rows))
    lines.append("")

    lines.append("## Selected gate config")
    for name in ["v14_5_gate_sweep"]:
        lines.append(f"### {name}")
        lines.append("```json")
        lines.append(json.dumps(analysis["runs"][name]["selected_gate_config"], ensure_ascii=False, indent=2))
        lines.append("```")
    lines.append("")

    lines.append("## Short interpretation")
    lines.append("- V13 remains best on valid.json; both V14.5 valid-select runs are below V13 despite selecting on valid.")
    lines.append("- Full-train/select-valid improved early checkpoint scores but did not improve the epoch-08 final enough; fallback types remain the largest bottleneck.")
    lines.append("- Retrieval is still responsible for most of the score above model-only, but current source-majority retrieval is near saturation for Rephrased/AnsAug.")
    lines.append("- The next high-upside path is not broader raw retrieval; it is type-specific answer-only specialization for FOBAR/SV plus safer retrieval/confidence arbitration.")
    return "\n".join(lines) + "\n"


def main() -> None:
    valid_records = load_json(VALID_PATH)
    same_original_classes = build_same_original_classes(valid_records)
    run_analysis_map = {
        name: run_analysis(name, path, same_original_classes)
        for name, path in RUNS.items()
        if (path / "valid_report.json").exists()
    }

    def oracle_for(run_names: list[str]) -> dict:
        reports = {
            name: load_json(RUNS[name] / "valid_report.json")["rows"]
            for name in run_names
        }
        raw = exact10 = zero_all = improved = 0
        for i in range(len(valid_records)):
            scores = [reports[name][i]["score"] for name in run_names]
            best = max(scores)
            raw += best
            exact10 += int(best == 10)
            zero_all += int(best == 0)
            improved += int(best > scores[0])
        return {
            "runs": run_names,
            "raw_score": raw,
            "score_10": raw / len(valid_records),
            "exact10": exact10,
            "zero_all": zero_all,
            "improved_vs_first": improved,
        }

    analysis = {
        "runs": run_analysis_map,
        "oracle_ensembles": [
            oracle_for(["v13", "v14_gate_sweep"]),
            oracle_for(["v13", "v14_gate_sweep", "v14_5_v13", "v14_5_gate_sweep"]),
            oracle_for(["v13", "v14_gate_sweep", "v14_5_v13", "v14_5_gate_sweep", "v12", "v14_fobar_sv"]),
        ],
    }
    OUT_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(make_markdown(analysis), encoding="utf-8")
    print("wrote", OUT_JSON)
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
