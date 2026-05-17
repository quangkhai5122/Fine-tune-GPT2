"""Targeted follow-up audit for the Vietnamese GPT-2 math task.

This script complements the earlier broad audits by focusing on questions that
matter for training decisions:

1. How much do the current "simple" and "robust" numeric parsers disagree?
2. How much exact query overlap exists between train and valid?
3. Are duplicated train queries label-consistent or genuinely conflicting?
4. Which math types are most exposed to truncation at common sequence lengths?

Outputs:
  - audit/additional_audit_summary.json
  - audit/additional_audit_report.txt
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import extract_answer as extract_answer_shared
from math_eval import parse_number as parse_number_robust


DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "GPT2_vietnamese"
OUT_JSON = ROOT / "audit" / "additional_audit_summary.json"
OUT_TXT = ROOT / "audit" / "additional_audit_report.txt"

SAFE_EOS_ID = 50256
PROMPT_TEMPLATE = "Câu hỏi: {q}\nLời giải: "


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        return json.load(f) if head == "[" else [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Shared extraction logic: take the final explicit anchor when present.
# ---------------------------------------------------------------------------
RE_ANCHORS = [
    ("dap_an_la", re.compile(r"đ[áa]p\s*[áa]n\s*l[àa]\s*[:：]?\s*", re.IGNORECASE)),
    ("cau_tra_loi_la", re.compile(r"c[âa]u\s*tr[ảa]\s*l[ờo]i\s*l[àa]\s*[:：]?\s*", re.IGNORECASE)),
    ("dap_an", re.compile(r"đ[áa]p\s*[áa]n\s*[:：]\s*", re.IGNORECASE)),
    ("the_answer_is", re.compile(r"the\s*answer\s*is\s*[:：]?\s*", re.IGNORECASE)),
    ("hash4", re.compile(r"####\s*")),
]
RE_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def clean_tail(s: str) -> str:
    s = s.strip().split("\n", 1)[0].strip()
    s = re.sub(r"[.,;:。、,]+$", "", s)
    s = re.sub(
        r"\s*(đô\s*la|usd|đồng|vnd|cm2?|m2?|km2?|\$|%)\b.*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()


def extract_answer(text: str | None) -> tuple[str | None, str | None]:
    answer, anchor = extract_answer_shared(text, return_anchor=True)
    return answer, anchor


# ---------------------------------------------------------------------------
# Numeric parsers
# ---------------------------------------------------------------------------
RE_NUM_VI_DEC = re.compile(r"-?\d+,\d+")
RE_NUM_EN_DEC = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
RE_NUM_LOOSE = re.compile(r"-?\d+(?:[.,]\d+)?")
SAFE_NS = {"sqrt": math.sqrt, "pi": math.pi}


def parse_number_simple(s: str | None) -> float | None:
    """Mirror the simplified parser used in the current V2/audit path."""
    if s is None:
        return None
    t = s.strip()
    if not t:
        return None
    if RE_NUM_VI_DEC.fullmatch(t):
        try:
            return float(t.replace(",", "."))
        except ValueError:
            return None
    if RE_NUM_EN_DEC.fullmatch(t):
        try:
            value = float(t)
            return value if math.isfinite(value) else None
        except ValueError:
            return None
    cleaned = t.replace(",", "")
    match = RE_NUM_EN_DEC.search(cleaned)
    if match:
        try:
            value = float(match.group())
            return value if math.isfinite(value) else None
        except ValueError:
            return None
    match = RE_NUM_LOOSE.search(t)
    if match:
        try:
            value = float(match.group().replace(",", "."))
            return value if math.isfinite(value) else None
        except ValueError:
            return None
    return None


def answer_form(answer: str | None, value: float | None) -> str:
    if answer is None:
        return "missing"
    if value is None:
        return "non_numeric"

    text = answer.strip()
    negative = "negative_" if value < 0 else ""
    if re.fullmatch(r"-?\d+", text):
        return negative + "integer"
    if re.fullmatch(r"-?\d+,\d+", text):
        return negative + "decimal_comma"
    if re.fullmatch(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", text):
        return negative + "decimal_dot"
    if re.fullmatch(r"-?\d+(?:[eE][+-]?\d+)", text):
        return negative + "scientific"
    return negative + "numeric_other"


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def parser_audit(records: list[dict]) -> dict:
    buckets = Counter()
    forms = Counter()
    anchors = Counter()
    mismatch_examples = []
    simple_numeric = 0
    robust_numeric = 0
    mismatch_count = 0
    by_type = defaultdict(lambda: Counter())

    for record in records:
        answer, anchor = extract_answer(record.get("response_vi"))
        simple = parse_number_simple(answer)
        robust = parse_number_robust(answer)

        simple_numeric += int(simple is not None)
        robust_numeric += int(robust is not None)
        mismatch_count += int(simple != robust)
        buckets[(simple is not None, robust is not None)] += 1
        forms[answer_form(answer, robust)] += 1
        anchors[anchor or "none"] += 1
        by_type[record.get("type") or "UNKNOWN"][
            (simple is not None, robust is not None, simple != robust)
        ] += 1

        if simple != robust and len(mismatch_examples) < 20:
            mismatch_examples.append(
                {
                    "type": record.get("type"),
                    "answer": answer,
                    "simple": simple,
                    "robust": robust,
                }
            )

    type_summary = {}
    for type_name, counts in sorted(by_type.items()):
        n = sum(counts.values())
        type_summary[type_name] = {
            "n": n,
            "simple_numeric": sum(v for (simple, robust, mismatch), v in counts.items() if simple),
            "robust_numeric": sum(v for (simple, robust, mismatch), v in counts.items() if robust),
            "mismatch_count": sum(v for (simple, robust, mismatch), v in counts.items() if mismatch),
        }

    return {
        "n": len(records),
        "simple_numeric_count": simple_numeric,
        "robust_numeric_count": robust_numeric,
        "mismatch_count": mismatch_count,
        "numeric_bucket_counts": {
            "simple_yes__robust_yes": buckets[(True, True)],
            "simple_yes__robust_no": buckets[(True, False)],
            "simple_no__robust_yes": buckets[(False, True)],
            "simple_no__robust_no": buckets[(False, False)],
        },
        "answer_form_counts_robust": dict(forms),
        "last_anchor_counts": dict(anchors),
        "by_type": type_summary,
        "mismatch_examples": mismatch_examples,
    }


def overlap_and_duplicate_audit(train: list[dict], valid: list[dict]) -> dict:
    train_by_query = defaultdict(list)
    valid_by_query = defaultdict(list)

    for record in train:
        answer, _ = extract_answer(record.get("response_vi"))
        train_by_query[(record.get("query_vi") or "").strip()].append(
            {
                "record": record,
                "answer": answer,
                "robust_num": parse_number_robust(answer),
            }
        )
    for record in valid:
        answer, _ = extract_answer(record.get("response_vi"))
        valid_by_query[(record.get("query_vi") or "").strip()].append(
            {
                "record": record,
                "answer": answer,
                "robust_num": parse_number_robust(answer),
            }
        )

    overlap_queries = set(train_by_query) & set(valid_by_query)
    exact_pair_train = {
        ((record.get("query_vi") or "").strip(), (record.get("response_vi") or "").strip())
        for record in train
    }
    exact_pair_overlap = [
        record
        for record in valid
        if (
            (record.get("query_vi") or "").strip(),
            (record.get("response_vi") or "").strip(),
        )
        in exact_pair_train
    ]

    overlap_type_counts = Counter()
    same_numeric = conflicting_numeric = missing_numeric = 0
    for query in overlap_queries:
        valid_records = valid_by_query[query]
        overlap_type_counts.update(item["record"].get("type") or "UNKNOWN" for item in valid_records)

        train_nums = {item["robust_num"] for item in train_by_query[query] if item["robust_num"] is not None}
        valid_nums = {item["robust_num"] for item in valid_records if item["robust_num"] is not None}
        if not train_nums or not valid_nums:
            missing_numeric += 1
        elif train_nums == valid_nums:
            same_numeric += 1
        else:
            conflicting_numeric += 1

    duplicate_groups = [items for items in train_by_query.values() if len(items) > 1]
    exact_copy_groups = 0
    distinct_response_groups = 0
    exact_extra_copies = 0
    distinct_extra_copies = 0
    consistent_groups = 0
    conflicting_groups = 0
    groups_with_missing_numeric = 0
    extra_consistent_copies = 0
    extra_conflicting_copies = 0
    conflict_examples = []

    for items in duplicate_groups:
        responses = {(item["record"].get("response_vi") or "").strip() for item in items}
        numeric_answers = {item["robust_num"] for item in items if item["robust_num"] is not None}
        has_missing = any(item["robust_num"] is None for item in items)
        extra = len(items) - 1

        if len(responses) == 1:
            exact_copy_groups += 1
            exact_extra_copies += extra
        else:
            distinct_response_groups += 1
            distinct_extra_copies += extra

        if len(numeric_answers) <= 1:
            consistent_groups += 1
            extra_consistent_copies += extra
        else:
            conflicting_groups += 1
            extra_conflicting_copies += extra
            if len(conflict_examples) < 10:
                first_query = (items[0]["record"].get("query_vi") or "").strip()
                conflict_examples.append(
                    {
                        "query": first_query,
                        "answers": [
                            {
                                "type": item["record"].get("type"),
                                "answer": item["answer"],
                                "robust_num": item["robust_num"],
                            }
                            for item in items[:5]
                        ],
                    }
                )

        if has_missing:
            groups_with_missing_numeric += 1

    valid_type_counts = Counter(record.get("type") or "UNKNOWN" for record in valid)
    overlap_pct_by_type = {
        type_name: {
            "overlap_count": overlap_type_counts[type_name],
            "valid_count": valid_type_counts[type_name],
            "overlap_pct": round(100 * overlap_type_counts[type_name] / valid_type_counts[type_name], 2),
        }
        for type_name in sorted(valid_type_counts)
    }

    return {
        "train_unique_queries": len(train_by_query),
        "valid_unique_queries": len(valid_by_query),
        "train_duplicate_query_groups": len(duplicate_groups),
        "train_exact_copy_groups": exact_copy_groups,
        "train_distinct_response_groups": distinct_response_groups,
        "train_exact_extra_copies": exact_extra_copies,
        "train_distinct_extra_copies": distinct_extra_copies,
        "train_consistent_numeric_groups": consistent_groups,
        "train_conflicting_numeric_groups": conflicting_groups,
        "train_groups_with_missing_numeric": groups_with_missing_numeric,
        "train_extra_consistent_copies": extra_consistent_copies,
        "train_extra_conflicting_copies": extra_conflicting_copies,
        "train_conflict_examples": conflict_examples,
        "exact_query_overlap_groups": len(overlap_queries),
        "valid_records_with_query_seen_in_train": sum(len(valid_by_query[q]) for q in overlap_queries),
        "exact_query_response_overlap_records": len(exact_pair_overlap),
        "overlap_groups_same_numeric_answer": same_numeric,
        "overlap_groups_conflicting_numeric_answer": conflicting_numeric,
        "overlap_groups_missing_numeric_answer": missing_numeric,
        "valid_overlap_by_type": overlap_pct_by_type,
    }


def token_length_audit(records: list[dict], tokenizer: AutoTokenizer) -> dict:
    totals = []
    by_type = defaultdict(list)

    batch_size = 512
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        prompts = [PROMPT_TEMPLATE.format(q=(record.get("query_vi") or "").strip()) for record in chunk]
        responses = [(record.get("response_vi") or "").strip() for record in chunk]
        prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(responses, add_special_tokens=False)["input_ids"]

        for record, p_ids, r_ids in zip(chunk, prompt_ids, response_ids):
            total_len = len(p_ids) + len(r_ids) + 1
            totals.append(total_len)
            by_type[record.get("type") or "UNKNOWN"].append(total_len)

    def summarize(values: list[int]) -> dict:
        values_sorted = sorted(values)
        n = len(values_sorted)
        return {
            "n": n,
            "mean": round(mean(values_sorted), 2),
            "min": values_sorted[0],
            "p50": values_sorted[int(0.50 * (n - 1))],
            "p95": values_sorted[int(0.95 * (n - 1))],
            "p99": values_sorted[int(0.99 * (n - 1))],
            "max": values_sorted[-1],
            "over_512": sum(v > 512 for v in values_sorted),
            "over_768": sum(v > 768 for v in values_sorted),
            "over_1024": sum(v > 1024 for v in values_sorted),
        }

    return {
        "overall": summarize(totals),
        "by_type": {type_name: summarize(values) for type_name, values in sorted(by_type.items())},
    }


def build_text_report(summary: dict) -> str:
    lines: list[str] = ["# Additional data audit\n"]

    for split in ("train", "valid"):
        info = summary["parser"][split]
        lines.append(f"\n## Parser audit — {split}")
        lines.append(
            f"  simple numeric: {info['simple_numeric_count']} / {info['n']} "
            f"({100 * info['simple_numeric_count'] / info['n']:.2f}%)"
        )
        lines.append(
            f"  robust numeric: {info['robust_numeric_count']} / {info['n']} "
            f"({100 * info['robust_numeric_count'] / info['n']:.2f}%)"
        )
        lines.append(f"  parser mismatches: {info['mismatch_count']}")
        lines.append(f"  numeric buckets: {info['numeric_bucket_counts']}")
        lines.append(f"  robust answer forms: {info['answer_form_counts_robust']}")
        lines.append("  per-type parser summary:")
        for type_name, type_info in info["by_type"].items():
            lines.append(
                f"    {type_name:<18s} n={type_info['n']:>5d} "
                f"simple={type_info['simple_numeric']:>5d} "
                f"robust={type_info['robust_numeric']:>5d} "
                f"mismatch={type_info['mismatch_count']:>4d}"
            )
        lines.append("  mismatch examples:")
        for example in info["mismatch_examples"][:10]:
            lines.append(
                f"    [{example['type']}] answer={example['answer']!r} "
                f"simple={example['simple']} robust={example['robust']}"
            )

    overlap = summary["overlap_and_duplicates"]
    lines.append("\n## Overlap and duplicate audit")
    lines.append(f"  train unique queries: {overlap['train_unique_queries']}")
    lines.append(f"  train duplicate query groups: {overlap['train_duplicate_query_groups']}")
    lines.append(
        f"    exact-copy groups={overlap['train_exact_copy_groups']} "
        f"(extra copies={overlap['train_exact_extra_copies']})"
    )
    lines.append(
        f"    distinct-response groups={overlap['train_distinct_response_groups']} "
        f"(extra copies={overlap['train_distinct_extra_copies']})"
    )
    lines.append(
        f"    numeric-consistent groups={overlap['train_consistent_numeric_groups']} "
        f"(extra copies={overlap['train_extra_consistent_copies']})"
    )
    lines.append(
        f"    numeric-conflicting groups={overlap['train_conflicting_numeric_groups']} "
        f"(extra copies={overlap['train_extra_conflicting_copies']})"
    )
    lines.append(f"  exact query overlap groups (train vs valid): {overlap['exact_query_overlap_groups']}")
    lines.append(
        f"  valid records whose query is seen in train: "
        f"{overlap['valid_records_with_query_seen_in_train']}"
    )
    lines.append(
        f"  exact query+response overlap records: "
        f"{overlap['exact_query_response_overlap_records']}"
    )
    lines.append(
        f"  overlap groups same/conflicting/missing numeric answer: "
        f"{overlap['overlap_groups_same_numeric_answer']} / "
        f"{overlap['overlap_groups_conflicting_numeric_answer']} / "
        f"{overlap['overlap_groups_missing_numeric_answer']}"
    )
    lines.append("  valid overlap by type:")
    for type_name, info in overlap["valid_overlap_by_type"].items():
        lines.append(
            f"    {type_name:<18s} {info['overlap_count']:>3d}/{info['valid_count']:<3d} "
            f"({info['overlap_pct']:>5.1f}%)"
        )

    for split in ("train", "valid"):
        token_info = summary["token_lengths"][split]
        overall = token_info["overall"]
        lines.append(f"\n## Token length audit — {split}")
        lines.append(
            f"  overall: n={overall['n']} mean={overall['mean']} "
            f"p50={overall['p50']} p95={overall['p95']} p99={overall['p99']} "
            f"max={overall['max']}"
        )
        lines.append(
            f"  over 512 / 768 / 1024: "
            f"{overall['over_512']} / {overall['over_768']} / {overall['over_1024']}"
        )
        lines.append("  by type:")
        for type_name, info in token_info["by_type"].items():
            lines.append(
                f"    {type_name:<18s} "
                f">512={info['over_512']:>4d}/{info['n']:<5d} "
                f">768={info['over_768']:>4d}/{info['n']:<5d} "
                f"max={info['max']}"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    train = load_records(DATA_DIR / "train.json")
    valid = load_records(DATA_DIR / "valid.json")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    tokenizer.pad_token_id = SAFE_EOS_ID
    tokenizer.eos_token_id = SAFE_EOS_ID

    summary = {
        "parser": {
            "train": parser_audit(train),
            "valid": parser_audit(valid),
        },
        "overlap_and_duplicates": overlap_and_duplicate_audit(train, valid),
        "token_lengths": {
            "train": token_length_audit(train, tokenizer),
            "valid": token_length_audit(valid, tokenizer),
        },
    }

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TXT.write_text(build_text_report(summary), encoding="utf-8")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
