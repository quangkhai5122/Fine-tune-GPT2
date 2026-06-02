"""Re-audit train/valid question overlap without loading train.json into RAM.

The earlier audit used exact ``query_vi.strip()`` comparison. This script checks
multiple fields and normalization levels because leakage can appear as
``valid.query_vi`` matching ``train.original_question_vi`` or the English fields,
not only as ``valid.query_vi == train.query_vi``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import extract_gold  # noqa: E402


FIELDS = ["query_vi", "original_question_vi", "query_en", "original_question_en"]
NORMS = ["strip", "space", "casefold_space", "casefold_alnum"]


def normalize(text: str | None, mode: str) -> str:
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFKC", value)
    if mode == "strip":
        return value.strip()
    value = re.sub(r"\s+", " ", value).strip()
    if mode == "space":
        return value
    value = value.casefold()
    if mode == "casefold_space":
        return value
    if mode == "casefold_alnum":
        value = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in value)
        return re.sub(r"\s+", " ", value).strip()
    raise ValueError(f"unknown normalization: {mode}")


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def iter_json_array(path: Path):
    """Yield objects from this repo's pretty-printed top-level JSON array."""
    with path.open("r", encoding="utf-8") as handle:
        in_object = False
        buffer: list[str] = []
        for line in handle:
            stripped = line.strip()
            if not in_object:
                if stripped == "[" or stripped == "":
                    continue
                if stripped == "{":
                    in_object = True
                    buffer = [line]
                    continue
                if stripped == "]":
                    break
                raise ValueError(f"unexpected line outside object in {path}: {stripped[:80]}")

            buffer.append(line)
            if stripped in {"}", "},"}:
                text = "".join(buffer)
                if stripped == "},":
                    text = text.rstrip()
                    text = text[:-1]
                yield json.loads(text)
                in_object = False
                buffer = []
                continue

        if in_object:
            raise ValueError(f"unterminated object in {path}")


def load_valid(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def gold_num(record: dict) -> float | None:
    try:
        return extract_gold(record)[1]
    except Exception:
        return None


def main() -> None:
    data_dir = ROOT / "dataset"
    train_path = data_dir / "train.json"
    valid_path = data_dir / "valid.json"
    out_json = ROOT / "audit" / "valid_train_overlap_reaudit.json"
    out_md = ROOT / "audit" / "valid_train_overlap_reaudit.md"

    valid = load_valid(valid_path)
    valid_answers = {i: gold_num(record) for i, record in enumerate(valid)}

    valid_lookup = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(valid):
        for field in FIELDS:
            for norm in NORMS:
                value = normalize(record.get(field), norm)
                if value:
                    valid_lookup[(field, norm)][digest(value)].append(index)

    matched_ids = defaultdict(set)
    matched_train_records = Counter()
    answer_cmp = defaultdict(Counter)
    valid_answer_flags = defaultdict(lambda: defaultdict(Counter))
    examples = defaultdict(list)
    train_counts = Counter()
    train_seen_unique = defaultdict(set)
    train_query_dup_counts = defaultdict(Counter)

    started = time.time()
    train_n = 0
    for record in iter_json_array(train_path):
        train_n += 1
        train_answer = None
        train_answer_loaded = False

        for train_field in FIELDS:
            for norm in NORMS:
                value = normalize(record.get(train_field), norm)
                if not value:
                    continue
                key = digest(value)
                train_counts[(train_field, norm)] += 1

                if train_field in {"query_vi", "original_question_vi"} and norm in {"strip", "casefold_alnum"}:
                    train_seen_unique[(train_field, norm)].add(key)
                    if train_field == "query_vi":
                        train_query_dup_counts[norm][key] += 1

                for valid_field in FIELDS:
                    valid_ids = valid_lookup.get((valid_field, norm), {}).get(key)
                    if not valid_ids:
                        continue
                    if not train_answer_loaded:
                        train_answer = gold_num(record)
                        train_answer_loaded = True
                    combo = (train_field, valid_field, norm)
                    matched_train_records[combo] += 1
                    for valid_id in valid_ids:
                        matched_ids[combo].add(valid_id)
                        valid_answer = valid_answers.get(valid_id)
                        if train_answer is None or valid_answer is None:
                            answer_cmp[combo]["missing_numeric"] += 1
                            valid_answer_flags[combo][valid_id]["missing_numeric"] += 1
                        elif abs(train_answer - valid_answer) <= 1e-9:
                            answer_cmp[combo]["same_numeric"] += 1
                            valid_answer_flags[combo][valid_id]["same_numeric"] += 1
                        else:
                            answer_cmp[combo]["conflicting_numeric"] += 1
                            valid_answer_flags[combo][valid_id]["conflicting_numeric"] += 1

                    if len(examples[combo]) < 5:
                        valid_id = valid_ids[0]
                        examples[combo].append(
                            {
                                "valid_id": valid_id,
                                "valid_type": valid[valid_id].get("type"),
                                "train_type": record.get("type"),
                                "train_field_value": (record.get(train_field) or "")[:500],
                                "valid_field_value": (valid[valid_id].get(valid_field) or "")[:500],
                                "train_answer_num": train_answer,
                                "valid_answer_num": valid_answers.get(valid_id),
                            }
                        )

        if train_n % 20000 == 0:
            elapsed = round(time.time() - started, 1)
            print(f"streamed train records: {train_n} elapsed_s={elapsed}", flush=True)

    rows = []
    for combo, ids in matched_ids.items():
        train_field, valid_field, norm = combo
        per_valid = valid_answer_flags[combo]
        with_same = sum(1 for counter in per_valid.values() if counter["same_numeric"] > 0)
        with_conflict = sum(1 for counter in per_valid.values() if counter["conflicting_numeric"] > 0)
        only_conflict = sum(
            1
            for counter in per_valid.values()
            if counter["conflicting_numeric"] > 0 and counter["same_numeric"] == 0
        )
        with_missing = sum(1 for counter in per_valid.values() if counter["missing_numeric"] > 0)
        rows.append(
            {
                "norm": norm,
                "train_field": train_field,
                "valid_field": valid_field,
                "valid_seen": len(ids),
                "valid_total": len(valid),
                "train_hits": matched_train_records[combo],
                "same_numeric": answer_cmp[combo]["same_numeric"],
                "conflicting_numeric": answer_cmp[combo]["conflicting_numeric"],
                "missing_numeric": answer_cmp[combo]["missing_numeric"],
                "valid_with_same_numeric": with_same,
                "valid_with_conflicting_numeric": with_conflict,
                "valid_with_only_conflicting_numeric": only_conflict,
                "valid_with_missing_numeric": with_missing,
                "examples": examples[combo],
            }
        )
    rows.sort(key=lambda item: (-item["valid_seen"], item["norm"], item["train_field"], item["valid_field"]))

    duplicate_summary = {}
    for norm, counts in train_query_dup_counts.items():
        duplicate_summary[norm] = {
            "groups": sum(1 for value in counts.values() if value > 1),
            "extra_records": sum(value - 1 for value in counts.values() if value > 1),
        }

    selected_train_unique = {}
    for key, values in sorted(train_seen_unique.items()):
        selected_train_unique[f"{key[0]}::{key[1]}"] = {
            "total": train_counts[key],
            "unique": len(values),
            "duplicate_records": train_counts[key] - len(values),
        }

    report = {
        "train_records": train_n,
        "valid_records": len(valid),
        "elapsed_seconds": round(time.time() - started, 2),
        "fields": FIELDS,
        "normalizations": NORMS,
        "selected_train_unique": selected_train_unique,
        "train_query_duplicate_summary": duplicate_summary,
        "overlap_rows": rows,
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Valid/train overlap re-audit", ""]
    lines.append(f"- train records: {train_n}")
    lines.append(f"- valid records: {len(valid)}")
    lines.append(f"- elapsed seconds: {report['elapsed_seconds']}")
    lines.append("")
    lines.append("## Selected Train Duplicate Stats")
    for name, info in selected_train_unique.items():
        lines.append(
            f"- `{name}`: total={info['total']}, unique={info['unique']}, "
            f"duplicate_records={info['duplicate_records']}"
        )
    for norm, info in duplicate_summary.items():
        lines.append(
            f"- train `query_vi` duplicate groups under `{norm}`: "
            f"{info['groups']}, extra_records={info['extra_records']}"
        )
    lines.append("")
    lines.append("## Overlap Rows")
    lines.append("| norm | train field | valid field | valid seen | train hits | hit same | hit conflict | valid any same | valid only conflict |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['norm']} | {row['train_field']} | {row['valid_field']} | "
            f"{row['valid_seen']}/{row['valid_total']} | {row['train_hits']} | "
            f"{row['same_numeric']} | {row['conflicting_numeric']} | "
            f"{row['valid_with_same_numeric']} | {row['valid_with_only_conflicting_numeric']} |"
        )
    lines.append("")
    lines.append("## Top Examples")
    for row in rows[:12]:
        lines.append(
            f"### {row['norm']} train.{row['train_field']} vs valid.{row['valid_field']} "
            f"({row['valid_seen']}/{row['valid_total']})"
        )
        for example in row["examples"][:3]:
            valid_text = example["valid_field_value"].replace("\n", " ")
            train_text = example["train_field_value"].replace("\n", " ")
            lines.append(
                f"- valid_id={example['valid_id']}, valid_type={example['valid_type']}, "
                f"train_type={example['train_type']}, "
                f"train_answer={example['train_answer_num']}, valid_answer={example['valid_answer_num']}"
            )
            lines.append(f"  - valid: {valid_text[:220]}")
            lines.append(f"  - train: {train_text[:220]}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({k: report[k] for k in ["train_records", "valid_records", "elapsed_seconds"]}, ensure_ascii=False))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    for row in rows[:20]:
        print(
            f"{row['norm']:15s} train.{row['train_field']:22s} vs valid.{row['valid_field']:22s}: "
            f"valid_seen={row['valid_seen']:4d}/{row['valid_total']} "
            f"train_hits={row['train_hits']:5d} same={row['same_numeric']} "
            f"conflict={row['conflicting_numeric']} "
            f"valid_any_same={row['valid_with_same_numeric']} "
            f"valid_only_conflict={row['valid_with_only_conflicting_numeric']}"
        )


if __name__ == "__main__":
    main()
