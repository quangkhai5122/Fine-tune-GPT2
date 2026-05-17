"""Shared answer extraction and evaluation utilities for the math task.

This module is intentionally stricter than the early notebook evaluator:
- use the *last* explicit answer anchor when multiple anchors appear
- parse scalar numeric expressions such as fractions, roots, powers, and pi
- reject tuples, intervals, and symbolic expressions that are not scalar answers

Keeping these rules in one place prevents the training cleaner, audits, and
offline reports from silently drifting apart.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


VI_ANCHORS = [
    ("dap_an_la", re.compile(r"đ[áa]p\s*[áa]n\s*l[àa]\s*[:：]?\s*", re.IGNORECASE)),
    ("cau_tra_loi_la", re.compile(r"c[âa]u\s*tr[ảa]\s*l[ờo]i\s*l[àa]\s*[:：]?\s*", re.IGNORECASE)),
    ("dap_an", re.compile(r"đ[áa]p\s*[áa]n\s*[:：]\s*", re.IGNORECASE)),
]
EN_ANCHORS = [
    ("the_answer_is", re.compile(r"the\s*answer\s*is\s*[:：]?\s*", re.IGNORECASE)),
    ("hash4", re.compile(r"####\s*")),
]
BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

SAFE_NS = {"sqrt": math.sqrt, "pi": math.pi}


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        return json.load(f) if head == "[" else [json.loads(line) for line in f if line.strip()]


def _clean_tail(text: str) -> str:
    text = text.strip().split("\n", 1)[0].strip()
    text = re.sub(r"[.,;:。、,]+$", "", text)
    text = re.sub(
        r"\s*(đô\s*la|usd|đồng|vnd|cm2?|m2?|km2?|\$|%)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def extract_answer(
    text: str | None,
    *,
    include_english: bool = True,
    return_anchor: bool = False,
) -> str | tuple[str | None, str | None] | None:
    """Extract the final explicit answer string from a response.

    We intentionally use the **last** anchor match. Several dataset responses
    contain intermediate `####` or answer-like snippets before the final answer.
    """
    if not text:
        return (None, None) if return_anchor else None

    anchor_pool = VI_ANCHORS + (EN_ANCHORS if include_english else [])
    best_pos = -1
    best_tail = None
    best_name = None

    for name, pattern in anchor_pool:
        for match in pattern.finditer(text):
            if match.end() > best_pos:
                best_pos = match.end()
                best_tail = text[match.end() :]
                best_name = name

    if best_tail is not None:
        answer = _clean_tail(best_tail)
        return (answer, best_name) if return_anchor else answer

    boxes = BOXED_RE.findall(text)
    if boxes:
        answer = _clean_tail(boxes[-1])
        return (answer, "boxed") if return_anchor else answer

    return (None, None) if return_anchor else None


def parse_number(text: str | None) -> float | None:
    """Parse a finite scalar numeric answer.

    Supports:
    - integers / decimals using `.` or Vietnamese decimal comma
    - scientific notation
    - `x = 5`
    - LaTeX fractions, roots, powers, and pi expressions

    Rejects:
    - tuples / coordinate pairs / intervals
    - symbolic expressions that still contain variables
    - malformed or non-finite values
    """
    if text is None:
        return None

    value = text.strip()
    if not value:
        return None

    if re.fullmatch(r"-?\d+,\d+", value):
        try:
            return float(value.replace(",", "."))
        except ValueError:
            return None

    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value):
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
        except ValueError:
            return None

    assignment = re.match(r"^[A-Za-z_]\w*\s*=\s*(.+)$", value)
    if assignment:
        value = assignment.group(1).strip()

    if value.startswith("(") and value.endswith(")") and re.search(r"\d\s*,\s*\d", value):
        return None
    if value.startswith("[") and value.endswith("]"):
        return None

    for _ in range(3):
        new_value = re.sub(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", r"(\1)", value)
        if new_value == value:
            break
        value = new_value

    value = re.sub(r"\\text\{[^}]*\}", "", value)
    value = re.sub(r"\\mathrm\{[^}]*\}", "", value)
    value = value.replace("$", "")

    for token in ("\\,", "\\!", "\\;", "\\ ", "\\left", "\\right"):
        value = value.replace(token, "")
    for token in ("\\cdot", "\\times"):
        value = value.replace(token, "*")

    value = re.sub(r"\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", value)
    value = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", value)
    value = re.sub(r"\\sqrt\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", value)
    value = value.replace("\\pi", "pi")

    value = re.sub(r"(\d)\s*(sqrt|pi|\()", r"\1*\2", value)
    value = re.sub(r"(\))\s*(sqrt|pi|\d)", r"\1*\2", value)
    value = re.sub(r"(pi)\s*(sqrt|pi|\d|\()", r"\1*\2", value)

    has_period = "." in value
    comma_count = value.count(",")
    if comma_count == 1 and not has_period and re.search(r"\d,\d", value):
        value = re.sub(r"(?<=\d),(?=\d)", ".", value)
    elif comma_count >= 1:
        value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)

    value = re.sub(r"\s+", "", value)
    if not value or "," in value:
        return None

    leftover = re.sub(r"sqrt|pi|\d|\.|\+|\-|\*|/|\(|\)|\^|e|E", "", value)
    if leftover:
        return None

    try:
        parsed = eval(value.replace("^", "**"), {"__builtins__": {}}, SAFE_NS)
    except Exception:
        return None

    if isinstance(parsed, bool):
        return None
    if isinstance(parsed, (int, float)):
        parsed = float(parsed)
        return parsed if math.isfinite(parsed) else None
    return None


def extract_gold(record: dict) -> tuple[str | None, float | None]:
    answer = extract_answer(record.get("response_vi"), include_english=True)
    return answer, parse_number(answer)


def extract_pred(record: dict) -> tuple[str | None, float | None]:
    answer = extract_answer(record.get("model_output"), include_english=True)
    return answer, parse_number(answer)


def rel_error(pred: float | None, gold: float | None) -> float | None:
    if pred is None or gold is None:
        return None
    return abs(pred - gold) / max(1.0, abs(gold))


def score_one(error_value: float | None, extractable: bool) -> int:
    if not extractable or error_value is None:
        return 0
    if error_value <= 0.01:
        return 10
    if error_value <= 0.10:
        return 5
    if error_value <= 0.50:
        return 1
    return 0


def align_predictions_with_gold(pred_items: list[dict], gold_items: list[dict]) -> list[tuple[dict, dict]]:
    pred_has_id = all("id" in item for item in pred_items)
    gold_has_id = all("id" in item for item in gold_items)

    if pred_has_id and gold_has_id:
        pred_by_id = {str(item["id"]): item for item in pred_items}
        pairs = []
        missing = []
        for gold in gold_items:
            gold_id = str(gold["id"])
            if gold_id not in pred_by_id:
                missing.append(gold_id)
            else:
                pairs.append((pred_by_id[gold_id], gold))
        if missing:
            raise ValueError(f"Missing {len(missing)} prediction ids, e.g. {missing[:5]}")
        return pairs

    if len(pred_items) != len(gold_items):
        raise ValueError(
            f"Prediction count ({len(pred_items)}) differs from gold count ({len(gold_items)})."
        )
    return list(zip(pred_items, gold_items))


def evaluate(pred_items: list[dict], gold_items: list[dict]) -> dict[str, Any]:
    pairs = align_predictions_with_gold(pred_items, gold_items)

    rows = []
    total = 0
    buckets = {10: 0, 5: 0, 1: 0, 0: 0}
    extractable = 0
    numeric_pairs = 0
    rel_errors = []

    for pred_record, gold_record in pairs:
        gold_answer, gold_num = extract_gold(gold_record)
        pred_answer, pred_num = extract_pred(pred_record)

        is_extractable = pred_answer is not None
        extractable += int(is_extractable)

        error_value = rel_error(pred_num, gold_num)
        if gold_num is not None and pred_num is not None and error_value is not None:
            numeric_pairs += 1
            rel_errors.append(error_value)

        score = score_one(error_value, is_extractable)
        total += score
        buckets[score] = buckets.get(score, 0) + 1

        rows.append(
            {
                "id": gold_record.get("id", pred_record.get("id")),
                "type": gold_record.get("type") or pred_record.get("type"),
                "gold_answer": gold_answer,
                "gold_num": gold_num,
                "pred_answer": pred_answer,
                "pred_num": pred_num,
                "rel_error": error_value,
                "extractable": is_extractable,
                "score": score,
            }
        )

    n = len(rows)
    return {
        "summary": {
            "n": n,
            "raw_score": total,
            "max_raw_score": n * 10,
            "score_10": total / n if n else 0.0,
            "score_pct": total / (n * 10) if n else 0.0,
            "extractable": extractable,
            "numeric_pairs": numeric_pairs,
            "buckets": buckets,
            "rel_error_mean": sum(rel_errors) / len(rel_errors) if rel_errors else None,
        },
        "rows": rows,
    }


def save_evaluation_report(
    pred_path: str | Path,
    gold_records: list[dict],
    report_path: str | Path,
) -> dict[str, Any]:
    pred_path = Path(pred_path)
    report_path = Path(report_path)

    with pred_path.open("r", encoding="utf-8") as f:
        pred_items = json.load(f)

    result = evaluate(pred_items, gold_records)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
