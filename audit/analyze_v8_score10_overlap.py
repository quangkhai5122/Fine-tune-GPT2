"""Analyze whether V8 exact-10 predictions come from train overlaps.

This focuses on the current best result folder:
``results/v8_select_checkpoint/beam2_lr1e-3``.

It separates:
- direct prompt overlap: ``train.query_vi`` vs ``valid.query_vi``
- source overlap: ``train.original_question_vi`` vs ``valid.original_question_vi``
- cross-field overlap: ``train.query_vi`` vs ``valid.original_question_vi`` and
  ``train.original_question_vi`` vs ``valid.query_vi``

For each overlap channel it records whether a train match has the same numeric
answer as the valid gold answer, only conflicting numeric answers, or missing
numeric answers.
"""

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


RUN_DIR = ROOT / "results" / "v8_select_checkpoint" / "beam2_lr1e-3"
VALID_PATH = ROOT / "dataset" / "valid.json"
TRAIN_PATH = ROOT / "dataset" / "train.json"
OUT_JSON = ROOT / "audit" / "v8_score10_overlap_analysis.json"
OUT_MD = ROOT / "audit" / "v8_score10_overlap_analysis.md"

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


def main() -> None:
    valid = load_json(VALID_PATH)
    report = load_json(RUN_DIR / "valid_report.json")
    outputs = load_json(RUN_DIR / "valid_output.json")
    rows = report["rows"]

    valid_nums = {i: extract_gold(record)[1] for i, record in enumerate(valid)}
    valid_answer_keys = {i: num_key(valid_nums[i]) for i in range(len(valid))}
    pred_answer_keys = {
        row["id"]: num_key(row.get("pred_num")) for row in rows if row.get("pred_num") is not None
    }

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
    channel_examples = {name: defaultdict(list) for name in CHANNELS}
    global_train_answer_counts = Counter()

    train_n = 0
    for record in iter_json_array(TRAIN_PATH):
        train_n += 1
        train_num = extract_gold(record)[1]
        key = num_key(train_num)
        if key is not None:
            global_train_answer_counts[key] += 1

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
                if len(channel_examples[name][valid_id]) < 3:
                    channel_examples[name][valid_id].append(
                        {
                            "train_type": record.get("type"),
                            "train_answer": train_num,
                            "valid_answer": valid_nums.get(valid_id),
                            "train_query_vi": (record.get("query_vi") or "")[:260],
                            "train_original_question_vi": (record.get("original_question_vi") or "")[:260],
                        }
                    )

    exact10_ids = {row["id"] for row in rows if row["score"] == 10}
    non10_ids = set(range(len(valid))) - exact10_ids

    channel_summary = {}
    for name, flags_by_id in channel_flags.items():
        classes = {i: flag_class(counter) for i, counter in flags_by_id.items()}
        class_ids = defaultdict(set)
        for valid_id, cls in classes.items():
            class_ids[cls].add(valid_id)

        channel_summary[name] = {
            "train_hits": channel_train_hits[name]["hits"],
            "class_counts_all": {cls: len(ids) for cls, ids in sorted(class_ids.items())},
            "class_counts_exact10": {
                cls: len(ids & exact10_ids) for cls, ids in sorted(class_ids.items())
            },
            "score_by_class": {
                cls: score_stats(rows, ids) for cls, ids in sorted(class_ids.items())
            },
        }

    # Combined labels that are most useful for leakage interpretation.
    same_original_classes = {
        i: flag_class(channel_flags["same_original_vi"][i]) for i in range(len(valid))
    }
    direct_query_classes = {
        i: flag_class(channel_flags["direct_query_vi"][i]) for i in range(len(valid))
    }
    valid_original_seen_in_train_query = {
        i: flag_class(channel_flags["train_query_vs_valid_original"][i]) for i in range(len(valid))
    }
    valid_query_seen_in_train_original = {
        i: flag_class(channel_flags["train_original_vs_valid_query"][i]) for i in range(len(valid))
    }

    combined_groups = {
        "direct_query_has_same": {
            i for i, cls in direct_query_classes.items() if cls == "has_same"
        },
        "same_original_has_same": {
            i for i, cls in same_original_classes.items() if cls == "has_same"
        },
        "same_original_only_conflict": {
            i for i, cls in same_original_classes.items() if cls == "only_conflict"
        },
        "no_same_original_match": {
            i for i, cls in same_original_classes.items() if cls == "no_match"
        },
        "valid_original_seen_as_train_query_has_same": {
            i for i, cls in valid_original_seen_in_train_query.items() if cls == "has_same"
        },
        "valid_query_seen_as_train_original_has_same": {
            i for i, cls in valid_query_seen_in_train_original.items() if cls == "has_same"
        },
    }

    combined_summary = {
        name: score_stats(rows, ids) | {"exact10_fraction": (score_stats(rows, ids)["exact10"] / len(exact10_ids) if exact10_ids else 0.0)}
        for name, ids in combined_groups.items()
    }

    exact10_breakdown = {
        "n_exact10": len(exact10_ids),
        "direct_query_has_same": len(combined_groups["direct_query_has_same"] & exact10_ids),
        "same_original_has_same": len(combined_groups["same_original_has_same"] & exact10_ids),
        "same_original_only_conflict": len(combined_groups["same_original_only_conflict"] & exact10_ids),
        "no_same_original_match": len(combined_groups["no_same_original_match"] & exact10_ids),
        "valid_original_seen_as_train_query_has_same": len(
            combined_groups["valid_original_seen_as_train_query_has_same"] & exact10_ids
        ),
        "valid_query_seen_as_train_original_has_same": len(
            combined_groups["valid_query_seen_as_train_original_has_same"] & exact10_ids
        ),
    }

    pred_counter_all = Counter(pred_answer_keys.values())
    pred_counter_10 = Counter(pred_answer_keys[i] for i in exact10_ids if i in pred_answer_keys)
    gold_counter_10 = Counter(valid_answer_keys[i] for i in exact10_ids if valid_answer_keys[i] is not None)
    exact10_pred_seen_global_train = sum(
        1 for i in exact10_ids if global_train_answer_counts[pred_answer_keys.get(i)] > 0
    )

    output_lengths = [len(item.get("model_output", "")) for item in outputs]

    examples = []
    for valid_id in sorted(exact10_ids):
        if len(examples) >= 12:
            break
        examples.append(
            {
                "id": valid_id,
                "type": valid[valid_id].get("type"),
                "query_vi": valid[valid_id].get("query_vi"),
                "original_question_vi": valid[valid_id].get("original_question_vi"),
                "pred_answer": rows[valid_id].get("pred_answer"),
                "gold_answer": rows[valid_id].get("gold_answer"),
                "direct_query_class": direct_query_classes[valid_id],
                "same_original_class": same_original_classes[valid_id],
                "train_query_vs_valid_original_class": valid_original_seen_in_train_query[valid_id],
                "train_original_vs_valid_query_class": valid_query_seen_in_train_original[valid_id],
                "same_original_examples": channel_examples["same_original_vi"][valid_id],
            }
        )

    result = {
        "run_dir": str(RUN_DIR.relative_to(ROOT)),
        "summary": report["summary"],
        "selected_checkpoint": load_json(RUN_DIR / "selected_checkpoint_info.json"),
        "train_records_scanned": train_n,
        "output_length": {
            "mean_chars": mean(output_lengths),
            "median_chars": median(output_lengths),
            "short_single_line_rate": sum(
                1 for item in outputs if "\n" not in item.get("model_output", "").strip() and len(item.get("model_output", "").strip()) <= 40
            )
            / len(outputs),
        },
        "channel_summary": channel_summary,
        "combined_summary": combined_summary,
        "exact10_breakdown": exact10_breakdown,
        "answer_distribution": {
            "top_pred_all": pred_counter_all.most_common(25),
            "top_pred_exact10": pred_counter_10.most_common(25),
            "top_gold_exact10": gold_counter_10.most_common(25),
            "exact10_pred_seen_global_train": exact10_pred_seen_global_train,
        },
        "examples_exact10": examples,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# V8 score-10 overlap analysis", ""]
    lines.append(f"- run: `{result['run_dir']}`")
    lines.append(f"- raw score: {report['summary']['raw_score']} / {report['summary']['max_raw_score']}")
    lines.append(f"- exact-10: {report['summary']['buckets']['10']} / {report['summary']['n']}")
    lines.append(f"- selected checkpoint: {result['selected_checkpoint']['label']}, lr={result['selected_checkpoint']['meta']['stage_a_lr']}, epoch={result['selected_checkpoint']['meta']['epoch']}")
    lines.append(f"- output median chars: {result['output_length']['median_chars']}")
    lines.append("")
    lines.append("## Exact-10 Breakdown")
    for key, value in exact10_breakdown.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Combined Score Slices")
    lines.append("| group | n | score/10 | exact10 | exact10 share | buckets |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for name, info in combined_summary.items():
        score = "NA" if info["score_10"] is None else f"{info['score_10']:.3f}"
        lines.append(
            f"| {name} | {info['n']} | {score} | {info['exact10']} | "
            f"{100 * info['exact10_fraction']:.1f}% | {info['buckets']} |"
        )
    lines.append("")
    lines.append("## Channel Class Counts")
    for name, info in channel_summary.items():
        lines.append(f"### {name}")
        lines.append(f"- train hits: {info['train_hits']}")
        lines.append(f"- all valid classes: {info['class_counts_all']}")
        lines.append(f"- exact10 classes: {info['class_counts_exact10']}")
    lines.append("")
    lines.append("## Answer Distribution")
    lines.append(f"- exact10 predictions whose numeric answer appears somewhere in train: {exact10_pred_seen_global_train}/{len(exact10_ids)}")
    lines.append(f"- top predictions among exact10: {pred_counter_10.most_common(20)}")
    lines.append(f"- top gold answers among exact10: {gold_counter_10.most_common(20)}")
    lines.append("")
    lines.append("## Example exact-10 rows")
    for example in examples:
        lines.append(
            f"### id={example['id']} type={example['type']} pred={example['pred_answer']} gold={example['gold_answer']}"
        )
        lines.append(
            f"- classes: direct_query={example['direct_query_class']}, "
            f"same_original={example['same_original_class']}, "
            f"train_query_vs_valid_original={example['train_query_vs_valid_original_class']}, "
            f"train_original_vs_valid_query={example['train_original_vs_valid_query_class']}"
        )
        lines.append(f"- query: {(example['query_vi'] or '')[:240]}")
        lines.append(f"- original: {(example['original_question_vi'] or '')[:240]}")
        for train_example in example["same_original_examples"][:2]:
            lines.append(
                f"  - train same-original type={train_example['train_type']} "
                f"answer={train_example['train_answer']}: "
                f"{train_example['train_query_vi'][:220]}"
            )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps(exact10_breakdown, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
