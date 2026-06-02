"""Aggregate V8 select-checkpoint runs and audit train/valid source overlap.

This scans train.json once, builds per-valid overlap classes, then evaluates
every folder under results/v8_select_checkpoint.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import extract_gold  # noqa: E402
from recheck_valid_train_overlap import digest, iter_json_array, normalize  # noqa: E402


RUN_BASE = ROOT / "results" / "v8_select_checkpoint"
VALID_PATH = ROOT / "dataset" / "valid.json"
TRAIN_PATH = ROOT / "dataset" / "train.json"
OUT_JSON = ROOT / "audit" / "v8_select_checkpoint_overlap_all.json"
OUT_MD = ROOT / "audit" / "v8_select_checkpoint_overlap_all.md"

CHANNELS = {
    "direct_query_vi": ("query_vi", "query_vi", "casefold_alnum"),
    "direct_query_vi_exact": ("query_vi", "query_vi", "strip"),
    "same_original_vi": ("original_question_vi", "original_question_vi", "strip"),
    "train_query_vs_valid_original": ("query_vi", "original_question_vi", "strip"),
    "train_original_vs_valid_query": ("original_question_vi", "query_vi", "strip"),
    "same_original_en": ("original_question_en", "original_question_en", "strip"),
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def build_overlap_classes(valid: list[dict]) -> dict:
    valid_nums = {i: extract_gold(record)[1] for i, record in enumerate(valid)}
    valid_lookup = {name: defaultdict(list) for name in CHANNELS}
    for index, record in enumerate(valid):
        for name, (_train_field, valid_field, norm) in CHANNELS.items():
            value = normalize(record.get(valid_field), norm)
            if value:
                valid_lookup[name][digest(value)].append(index)

    channel_flags = {
        name: {i: Counter() for i in range(len(valid))}
        for name in CHANNELS
    }
    channel_train_hits = {name: Counter() for name in CHANNELS}

    train_n = 0
    for record in iter_json_array(TRAIN_PATH):
        train_n += 1
        train_num = extract_gold(record)[1]
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

    classes = {
        name: {i: flag_class(counter) for i, counter in flags_by_id.items()}
        for name, flags_by_id in channel_flags.items()
    }
    channel_summary = {}
    all_ids = set(range(len(valid)))
    for name, by_id in classes.items():
        channel_summary[name] = {
            "train_hits": channel_train_hits[name]["hits"],
            "class_counts_all": dict(Counter(by_id.values())),
        }
    return {
        "train_records_scanned": train_n,
        "classes": classes,
        "channel_summary": channel_summary,
        "valid_ids": all_ids,
    }


def analyze_run(run_dir: Path, overlap: dict) -> dict:
    report = load_json(run_dir / "valid_report.json")
    manifest = load_json(run_dir / "v8_manifest.json")
    selected = load_json(run_dir / "selected_checkpoint_info.json")
    rows = report["rows"]
    exact10_ids = {row["id"] for row in rows if row["score"] == 10}

    classes = overlap["classes"]
    combined_groups = {
        "direct_query_has_same": {
            i for i, cls in classes["direct_query_vi"].items() if cls == "has_same"
        },
        "same_original_has_same": {
            i for i, cls in classes["same_original_vi"].items() if cls == "has_same"
        },
        "same_original_only_conflict": {
            i for i, cls in classes["same_original_vi"].items() if cls == "only_conflict"
        },
        "same_original_no_match": {
            i for i, cls in classes["same_original_vi"].items() if cls == "no_match"
        },
        "same_original_only_missing": {
            i for i, cls in classes["same_original_vi"].items() if cls == "only_missing"
        },
        "valid_original_seen_as_train_query_has_same": {
            i for i, cls in classes["train_query_vs_valid_original"].items() if cls == "has_same"
        },
        "valid_query_seen_as_train_original_has_same": {
            i for i, cls in classes["train_original_vs_valid_query"].items() if cls == "has_same"
        },
    }
    combined_summary = {}
    for name, ids in combined_groups.items():
        stats = score_stats(rows, ids)
        stats["exact10_fraction"] = stats["exact10"] / len(exact10_ids) if exact10_ids else 0.0
        combined_summary[name] = stats

    channel_exact10 = {}
    for name, by_id in classes.items():
        counter = Counter(by_id[i] for i in exact10_ids)
        channel_exact10[name] = dict(counter)

    checkpoint_report = load_json(run_dir / "checkpoint_selection_report.json")
    curve = [
        {
            "label": entry["label"],
            "raw_score": entry["summary"]["raw_score"],
            "exact10": entry["summary"]["buckets"]["10"],
            "extractable": entry["summary"]["extractable"],
        }
        for entry in checkpoint_report.get("all_checkpoints", [])
        if entry["label"].startswith("epoch_")
    ]

    cfg = manifest.get("config", {})
    return {
        "run": run_dir.name,
        "summary": report["summary"],
        "config": {
            "stage_a_lr": cfg.get("stage_a_lr"),
            "stage_a_epochs": cfg.get("stage_a_epochs"),
            "num_beams": cfg.get("num_beams"),
            "checkpoint_eval_num_beams": cfg.get("checkpoint_eval_num_beams"),
            "lora_target_modules": cfg.get("lora_target_modules"),
            "lora_r": cfg.get("lora_r"),
            "lora_alpha": cfg.get("lora_alpha"),
        },
        "selected_checkpoint": {
            "label": selected.get("label"),
            "epoch": selected.get("meta", {}).get("epoch"),
            "raw_score": selected.get("summary", {}).get("raw_score"),
            "exact10": selected.get("summary", {}).get("buckets", {}).get("10"),
        },
        "checkpoint_curve": curve,
        "combined_summary": combined_summary,
        "channel_exact10": channel_exact10,
    }


def main() -> None:
    valid = load_json(VALID_PATH)
    overlap = build_overlap_classes(valid)
    run_dirs = sorted(p for p in RUN_BASE.iterdir() if (p / "valid_report.json").exists())
    runs = [analyze_run(run_dir, overlap) for run_dir in run_dirs]

    result = {
        "run_base": str(RUN_BASE.relative_to(ROOT)),
        "valid_records": len(valid),
        "train_records_scanned": overlap["train_records_scanned"],
        "channel_summary_all_valid": overlap["channel_summary"],
        "runs": runs,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# V8 select-checkpoint overlap analysis", ""]
    lines.append(f"- run base: `{result['run_base']}`")
    lines.append(f"- train records scanned: {result['train_records_scanned']}")
    lines.append(f"- valid records: {result['valid_records']}")
    lines.append("")
    lines.append("## Run summary")
    lines.append("| run | lr | beams | selected | raw | exact10 | extractable | same-original exact10 | no-original exact10 |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for run in runs:
        s = run["summary"]
        cfg = run["config"]
        same = run["combined_summary"]["same_original_has_same"]
        no = run["combined_summary"]["same_original_no_match"]
        lines.append(
            f"| {run['run']} | {cfg['stage_a_lr']} | {cfg['num_beams']} | "
            f"{run['selected_checkpoint']['label']} | {s['raw_score']} | {s['buckets']['10']} | "
            f"{s['extractable']} | {same['exact10']} | {no['exact10']} |"
        )
    lines.append("")
    lines.append("## Global overlap classes")
    for name, info in result["channel_summary_all_valid"].items():
        lines.append(f"- `{name}`: train_hits={info['train_hits']}, classes={info['class_counts_all']}")
    lines.append("")
    for run in runs:
        lines.append(f"## {run['run']}")
        s = run["summary"]
        lines.append(f"- raw: {s['raw_score']} / {s['max_raw_score']}; exact10={s['buckets']['10']}; extractable={s['extractable']}")
        lines.append(f"- config: `{run['config']}`")
        lines.append(f"- selected: `{run['selected_checkpoint']}`")
        lines.append("- checkpoint curve: " + ", ".join(f"{x['label']}={x['raw_score']}" for x in run["checkpoint_curve"]))
        lines.append("")
        lines.append("| group | n | raw | score/10 | exact10 | exact10 share | buckets |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for group, stats in run["combined_summary"].items():
            score = "NA" if stats["score_10"] is None else f"{stats['score_10']:.3f}"
            lines.append(
                f"| {group} | {stats['n']} | {stats['raw']} | {score} | {stats['exact10']} | "
                f"{100*stats['exact10_fraction']:.1f}% | {stats['buckets']} |"
            )
        lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
