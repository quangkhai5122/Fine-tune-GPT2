"""Deep result analysis for V13 type-gated hybrid full-retrain run."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import extract_gold  # noqa: E402
from recheck_valid_train_overlap import digest, iter_json_array, normalize  # noqa: E402


RUN_DIR = ROOT / "results" / "v13"
VALID_PATH = ROOT / "dataset" / "valid.json"
TRAIN_PATH = ROOT / "dataset" / "train.json"
OUT_JSON = ROOT / "audit" / "v13_result_analysis.json"
OUT_MD = ROOT / "audit" / "v13_result_analysis.md"

CHANNELS = {
    "direct_query_vi": ("query_vi", "query_vi", "casefold_alnum"),
    "direct_query_vi_exact": ("query_vi", "query_vi", "strip"),
    "same_original_vi": ("original_question_vi", "original_question_vi", "strip"),
    "train_query_vs_valid_original": ("query_vi", "original_question_vi", "strip"),
    "train_original_vs_valid_query": ("original_question_vi", "query_vi", "strip"),
    "same_original_en": ("original_question_en", "original_question_en", "strip"),
}

BASELINE_REPORTS = {
    "v8_beam2_lr1e-3": ROOT / "results" / "v8_select_checkpoint" / "beam2_lr1e-3" / "valid_report.json",
    "v9": ROOT / "results" / "v9" / "valid_report.json",
    "v10": ROOT / "results" / "v10" / "valid_report.json",
    "v11": ROOT / "results" / "v11" / "valid_report.json",
    "v12": ROOT / "results" / "v12" / "valid_report.json",
    "v13_model_only": RUN_DIR / "model_valid_report.json",
    "v13_final_hybrid": RUN_DIR / "valid_report.json",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def num_key(value: float | None) -> str | None:
    if value is None:
        return None
    if abs(value - round(value)) <= 1e-9:
        return str(int(round(value)))
    return f"{value:.12g}"


def update_answer_flags(counter: Counter, train_num: float | None, valid_num: float | None) -> None:
    if train_num is None or valid_num is None:
        counter["missing_numeric"] += 1
    elif abs(train_num - valid_num) <= 1e-9:
        counter["same_numeric"] += 1
    else:
        counter["conflicting_numeric"] += 1


def flag_class(counter: Counter) -> str:
    if counter["same_numeric"] > 0:
        return "has_same"
    if counter["conflicting_numeric"] > 0:
        return "only_conflict"
    if counter["missing_numeric"] > 0:
        return "only_missing"
    return "no_match"


def score_stats(rows: list[dict], ids: set[int]) -> dict:
    if not ids:
        return {"n": 0, "raw": 0, "score_10": None, "exact10": 0, "buckets": {}}
    selected = [rows[i] for i in sorted(ids)]
    buckets = Counter(row["score"] for row in selected)
    raw = sum(row["score"] for row in selected)
    return {
        "n": len(selected),
        "raw": raw,
        "score_10": raw / len(selected),
        "exact10": buckets[10],
        "buckets": {str(k): buckets[k] for k in sorted(buckets, reverse=True)},
    }


def report_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    report = load_json(path)
    s = report["summary"]
    return {
        "raw_score": s["raw_score"],
        "score_10": s["score_10"],
        "exact10": s["buckets"]["10"],
        "extractable": s["extractable"],
        "numeric_pairs": s["numeric_pairs"],
        "buckets": s["buckets"],
    }


def output_stats(path: Path) -> dict:
    data = load_json(path)
    texts = [item.get("model_output", "") for item in data]
    lengths = [len(text) for text in texts]
    return {
        "n": len(texts),
        "mean_chars": mean(lengths),
        "median_chars": median(lengths),
        "short_single_line_rate": sum(
            1 for text in texts if "\n" not in text.strip() and len(text.strip()) <= 40
        )
        / len(texts),
    }


def main() -> None:
    valid = load_json(VALID_PATH)
    final_report = load_json(RUN_DIR / "valid_report.json")
    model_report = load_json(RUN_DIR / "model_valid_report.json")
    final_outputs = load_json(RUN_DIR / "valid_output.json")
    model_outputs = load_json(RUN_DIR / "model_valid_output.json")
    hybrid = load_json(RUN_DIR / "hybrid_decision_report.json")

    final_rows = final_report["rows"]
    model_rows = model_report["rows"]
    valid_nums = {i: extract_gold(record)[1] for i, record in enumerate(valid)}

    valid_lookup = {name: defaultdict(list) for name in CHANNELS}
    for index, record in enumerate(valid):
        for name, (_train_field, valid_field, norm) in CHANNELS.items():
            value = normalize(record.get(valid_field), norm)
            if value:
                valid_lookup[name][digest(value)].append(index)

    channel_flags = {name: {i: Counter() for i in range(len(valid))} for name in CHANNELS}
    channel_train_hits = {name: Counter() for name in CHANNELS}
    train_answer_counts = Counter()

    train_n = 0
    for record in iter_json_array(TRAIN_PATH):
        train_n += 1
        train_num = extract_gold(record)[1]
        key = num_key(train_num)
        if key is not None:
            train_answer_counts[key] += 1

        for name, (train_field, _valid_field, norm) in CHANNELS.items():
            value = normalize(record.get(train_field), norm)
            if not value:
                continue
            valid_ids = valid_lookup[name].get(digest(value))
            if not valid_ids:
                continue
            channel_train_hits[name]["hits"] += 1
            for valid_id in valid_ids:
                update_answer_flags(channel_flags[name][valid_id], train_num, valid_nums.get(valid_id))

    channel_classes = {
        name: {i: flag_class(counter) for i, counter in flags.items()}
        for name, flags in channel_flags.items()
    }
    exact10_ids = {row["id"] for row in final_rows if row["score"] == 10}
    zero_ids = {row["id"] for row in final_rows if row["score"] == 0}

    channel_summary = {}
    for name, classes in channel_classes.items():
        grouped = defaultdict(set)
        for valid_id, cls in classes.items():
            grouped[cls].add(valid_id)
        channel_summary[name] = {
            "train_hits": channel_train_hits[name]["hits"],
            "class_counts_all": {cls: len(ids) for cls, ids in sorted(grouped.items())},
            "class_counts_exact10": {cls: len(ids & exact10_ids) for cls, ids in sorted(grouped.items())},
            "class_counts_zero": {cls: len(ids & zero_ids) for cls, ids in sorted(grouped.items())},
            "score_by_class": {cls: score_stats(final_rows, ids) for cls, ids in sorted(grouped.items())},
        }

    combined_groups = {
        "direct_query_has_same": {
            i for i, cls in channel_classes["direct_query_vi"].items() if cls == "has_same"
        },
        "same_original_has_same": {
            i for i, cls in channel_classes["same_original_vi"].items() if cls == "has_same"
        },
        "same_original_only_conflict": {
            i for i, cls in channel_classes["same_original_vi"].items() if cls == "only_conflict"
        },
        "same_original_no_match": {
            i for i, cls in channel_classes["same_original_vi"].items() if cls == "no_match"
        },
        "valid_original_seen_as_train_query_has_same": {
            i for i, cls in channel_classes["train_query_vs_valid_original"].items() if cls == "has_same"
        },
        "valid_query_seen_as_train_original_has_same": {
            i for i, cls in channel_classes["train_original_vs_valid_query"].items() if cls == "has_same"
        },
    }
    combined_summary = {name: score_stats(final_rows, ids) for name, ids in combined_groups.items()}

    # Model vs hybrid deltas.
    by_type_delta = defaultdict(lambda: Counter())
    transition_counts = Counter()
    changed_output_ids = []
    for index, (final_row, model_row) in enumerate(zip(final_rows, model_rows)):
        rec_type = final_row["type"]
        delta = final_row["score"] - model_row["score"]
        by_type_delta[rec_type]["n"] += 1
        by_type_delta[rec_type]["delta_raw"] += delta
        by_type_delta[rec_type]["final_raw"] += final_row["score"]
        by_type_delta[rec_type]["model_raw"] += model_row["score"]
        by_type_delta[rec_type]["improved"] += int(delta > 0)
        by_type_delta[rec_type]["worse"] += int(delta < 0)
        by_type_delta[rec_type]["same"] += int(delta == 0)
        by_type_delta[rec_type]["final10"] += int(final_row["score"] == 10)
        by_type_delta[rec_type]["model10"] += int(model_row["score"] == 10)
        transition_counts[(model_row["score"], final_row["score"])] += 1
        if final_outputs[index].get("model_output") != model_outputs[index].get("model_output"):
            changed_output_ids.append(index)

    by_type_delta_out = {}
    for rec_type, counter in sorted(by_type_delta.items()):
        n = counter["n"]
        by_type_delta_out[rec_type] = dict(counter) | {
            "model_score_10": counter["model_raw"] / n,
            "final_score_10": counter["final_raw"] / n,
        }

    # Magnitude slices help distinguish memorized small-answer priors from larger numbers.
    magnitude_groups = {
        "abs<=10": set(),
        "10<abs<=100": set(),
        "100<abs<=1000": set(),
        "abs>1000": set(),
        "non_numeric_gold": set(),
    }
    for index, value in valid_nums.items():
        if value is None:
            magnitude_groups["non_numeric_gold"].add(index)
        elif abs(value) <= 10:
            magnitude_groups["abs<=10"].add(index)
        elif abs(value) <= 100:
            magnitude_groups["10<abs<=100"].add(index)
        elif abs(value) <= 1000:
            magnitude_groups["100<abs<=1000"].add(index)
        else:
            magnitude_groups["abs>1000"].add(index)
    magnitude_summary = {name: score_stats(final_rows, ids) for name, ids in magnitude_groups.items()}

    pred_counter_all = Counter(num_key(row.get("pred_num")) for row in final_rows if row.get("pred_num") is not None)
    pred_counter_zero = Counter(num_key(final_rows[i].get("pred_num")) for i in zero_ids if final_rows[i].get("pred_num") is not None)
    pred_counter_exact10 = Counter(num_key(final_rows[i].get("pred_num")) for i in exact10_ids if final_rows[i].get("pred_num") is not None)

    baseline_comparison = {
        name: report_summary(path) for name, path in BASELINE_REPORTS.items() if report_summary(path) is not None
    }

    result = {
        "run_dir": str(RUN_DIR.relative_to(ROOT)),
        "baseline_comparison": baseline_comparison,
        "final_summary": final_report["summary"],
        "model_summary": model_report["summary"],
        "overlap_valid_summary": report_summary(RUN_DIR / "overlap_valid_report.json"),
        "selected_checkpoint": load_json(RUN_DIR / "selected_checkpoint_info.json"),
        "final_retrain_info": load_json(RUN_DIR / "final_retrain_info.json"),
        "query_disjoint_split_report": load_json(RUN_DIR / "query_disjoint_split_report.json"),
        "valid_overlap_audit": load_json(RUN_DIR / "valid_overlap_audit.json"),
        "hybrid_decision_report": hybrid,
        "output_stats": {
            "final": output_stats(RUN_DIR / "valid_output.json"),
            "model": output_stats(RUN_DIR / "model_valid_output.json"),
            "overlap": output_stats(RUN_DIR / "overlap_valid_output.json"),
        },
        "channel_summary": channel_summary,
        "combined_overlap_summary": combined_summary,
        "exact10_overlap_breakdown": {
            "n_exact10": len(exact10_ids),
            "same_original_has_same": len(combined_groups["same_original_has_same"] & exact10_ids),
            "same_original_only_conflict": len(combined_groups["same_original_only_conflict"] & exact10_ids),
            "same_original_no_match": len(combined_groups["same_original_no_match"] & exact10_ids),
            "valid_original_seen_as_train_query_has_same": len(
                combined_groups["valid_original_seen_as_train_query_has_same"] & exact10_ids
            ),
            "valid_query_seen_as_train_original_has_same": len(
                combined_groups["valid_query_seen_as_train_original_has_same"] & exact10_ids
            ),
            "direct_query_has_same": len(combined_groups["direct_query_has_same"] & exact10_ids),
        },
        "zero_overlap_breakdown": {
            "n_zero": len(zero_ids),
            "same_original_has_same": len(combined_groups["same_original_has_same"] & zero_ids),
            "same_original_only_conflict": len(combined_groups["same_original_only_conflict"] & zero_ids),
            "same_original_no_match": len(combined_groups["same_original_no_match"] & zero_ids),
        },
        "model_vs_hybrid": {
            "delta_raw": final_report["summary"]["raw_score"] - model_report["summary"]["raw_score"],
            "delta_exact10": final_report["summary"]["buckets"]["10"] - model_report["summary"]["buckets"]["10"],
            "changed_output_count": len(changed_output_ids),
            "by_type": by_type_delta_out,
            "transition_counts": {f"{a}->{b}": count for (a, b), count in sorted(transition_counts.items())},
        },
        "magnitude_summary": magnitude_summary,
        "answer_distribution": {
            "top_pred_all": pred_counter_all.most_common(25),
            "top_pred_exact10": pred_counter_exact10.most_common(25),
            "top_pred_zero": pred_counter_zero.most_common(25),
        },
        "train_records_scanned_for_overlap": train_n,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# V13 result analysis", ""]
    lines.append("## Headline")
    lines.append(f"- final hybrid raw: {final_report['summary']['raw_score']} / {final_report['summary']['max_raw_score']} ({final_report['summary']['score_10']:.3f}/10)")
    lines.append(f"- model-only raw: {model_report['summary']['raw_score']} / {model_report['summary']['max_raw_score']} ({model_report['summary']['score_10']:.3f}/10)")
    lines.append(f"- hybrid delta: +{result['model_vs_hybrid']['delta_raw']} raw, +{result['model_vs_hybrid']['delta_exact10']} exact10")
    lines.append(f"- selected checkpoint on overlap-valid: {result['selected_checkpoint']['label']}, epoch={result['selected_checkpoint']['meta']['epoch']}, lr={result['selected_checkpoint']['meta']['stage_a_lr']}")
    lines.append(f"- final retrain: {result['final_retrain_info']['enabled']} on {result['final_retrain_info']['train_records']} clean train rows")
    lines.append("")
    lines.append("## Baseline Comparison")
    lines.append("| run | raw | score/10 | exact10 | extractable |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, summary in baseline_comparison.items():
        lines.append(f"| {name} | {summary['raw_score']} | {summary['score_10']:.3f} | {summary['exact10']} | {summary['extractable']} |")
    lines.append("")
    lines.append("## Type Delta: Model Only -> Final Hybrid")
    lines.append("| type | n | model/10 | final/10 | delta raw | improved | worse | model exact10 | final exact10 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rec_type, info in by_type_delta_out.items():
        lines.append(
            f"| {rec_type} | {info['n']} | {info['model_score_10']:.3f} | {info['final_score_10']:.3f} | "
            f"{info['delta_raw']} | {info['improved']} | {info['worse']} | {info['model10']} | {info['final10']} |"
        )
    lines.append("")
    lines.append("## Overlap Slices")
    lines.append("| group | n | score/10 | exact10 | buckets |")
    lines.append("|---|---:|---:|---:|---|")
    for name, info in combined_summary.items():
        score = "NA" if info["score_10"] is None else f"{info['score_10']:.3f}"
        lines.append(f"| {name} | {info['n']} | {score} | {info['exact10']} | {info['buckets']} |")
    lines.append("")
    lines.append("## Exact10 and Zero Breakdown")
    for key, value in result["exact10_overlap_breakdown"].items():
        lines.append(f"- exact10 `{key}`: {value}")
    for key, value in result["zero_overlap_breakdown"].items():
        lines.append(f"- zero `{key}`: {value}")
    lines.append("")
    lines.append("## Magnitude Slices")
    lines.append("| gold magnitude | n | score/10 | exact10 | buckets |")
    lines.append("|---|---:|---:|---:|---|")
    for name, info in magnitude_summary.items():
        score = "NA" if info["score_10"] is None else f"{info['score_10']:.3f}"
        lines.append(f"| {name} | {info['n']} | {score} | {info['exact10']} | {info['buckets']} |")
    lines.append("")
    lines.append("## Hybrid Routing")
    lines.append(f"- retrieval used: {hybrid['retrieval_used']} / {hybrid['n']} ({hybrid['retrieval_used_pct']:.3f})")
    lines.append(f"- allowed types: {hybrid['allowed_types']}")
    lines.append(f"- fallback reasons: {hybrid['fallback_reason_counts']}")
    lines.append(f"- output-changed count: {len(changed_output_ids)}")
    lines.append(f"- model->final transitions: {result['model_vs_hybrid']['transition_counts']}")
    lines.append("")
    lines.append("## Answer Distribution")
    lines.append(f"- top exact10 predictions: {pred_counter_exact10.most_common(15)}")
    lines.append(f"- top zero predictions: {pred_counter_zero.most_common(15)}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps(result["final_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
