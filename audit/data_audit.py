"""Data audit for Fine-tune GPT-2 Vietnamese math task.

Generates statistics over `query_vi` and `response_vi` for both
`train.json` and `valid.json` and dumps:
  - audit/audit_summary.json
  - audit/length_hist.txt
  - audit/type_distribution.txt
  - audit/answer_extraction_report.txt
  - audit/sample_examples.txt
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import extract_answer as standard_extract_answer
from math_eval import parse_number as standard_parse_number

DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "audit"
OUT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------
def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        return json.load(f) if head == "[" else [json.loads(ln) for ln in f if ln.strip()]


# -----------------------------------------------------------------
# Answer extraction (mirror of the notebook logic)
# -----------------------------------------------------------------
VI_ANCHORS = [
    r"Câu trả lời là\s*[:：]?",
    r"Đáp án là\s*[:：]?",
    r"Đáp án\s*[:：]",
]

EN_ANCHORS = [
    r"The answer is\s*[:：]?",
    r"####",
]

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_answer(text: str | None, anchors: list[str]) -> str | None:
    if not text:
        return None
    for anc in anchors:
        m = re.search(anc, text)
        if m:
            tail = text[m.end():].strip().split("\n")[0]
            return tail.strip().rstrip(".。、,")
    boxes = BOXED_RE.findall(text)
    if boxes:
        return boxes[-1].strip()
    return None


def parse_number(s: str | None) -> float | None:
    if s is None:
        return None
    t = s.strip()
    if not t:
        return None
    if re.fullmatch(r"-?\d+,\d+", t):
        try:
            return float(t.replace(",", "."))
        except ValueError:
            return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", t):
        try:
            v = float(t)
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    # try strip ',' (English thousand sep) and pull leading number
    m = re.search(r"-?\d+(?:[\.,]\d+)?", t.replace(",", ""))
    if m:
        try:
            v = float(m.group())
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


# -----------------------------------------------------------------
# Stats over a list of records
# -----------------------------------------------------------------
def char_stats(values: list[int]) -> dict:
    if not values:
        return {}
    values_sorted = sorted(values)
    n = len(values_sorted)

    def pct(p: float) -> int:
        idx = min(n - 1, max(0, int(p * (n - 1))))
        return values_sorted[idx]

    return {
        "n": n,
        "min": values_sorted[0],
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": values_sorted[-1],
        "mean": round(mean(values_sorted), 2),
        "median": median(values_sorted),
    }


def whitespace_token_count(text: str) -> int:
    return len(text.split())


def has_pattern(text: str, pat: re.Pattern) -> bool:
    return bool(pat.search(text))


LATEX_RE = re.compile(r"\\(?:frac|sqrt|boxed|begin|end|times|cdot|sum|int|prod|pi|sin|cos|tan|alpha|beta|theta|leq|geq|neq|in|forall|exists)")
DOLLAR_RE = re.compile(r"\$[^$]+\$")
PERCENT_RE = re.compile(r"\d+\s*%")
DOLLAR_SIGN_RE = re.compile(r"\$\d")  # "$5" style
NUMBER_RE = re.compile(r"\d")
ANGLE_RE = re.compile(r"\\angle")
NEGATIVE_RE = re.compile(r"-\d")

ANCHOR_REs = [
    ("dap_an_la", re.compile(r"Đáp án là\s*[:：]?")),
    ("cau_tra_loi_la", re.compile(r"Câu trả lời là\s*[:：]?")),
    ("dap_an", re.compile(r"Đáp án\s*[:：]")),
    ("the_answer_is", re.compile(r"The answer is\s*[:：]?")),
    ("hash4", re.compile(r"####")),
    ("boxed", BOXED_RE),
]


def audit(records: list[dict], split_name: str) -> dict:
    n = len(records)

    types = Counter()

    q_chars, q_tokens = [], []
    r_chars, r_tokens = [], []
    full_chars = []  # query + response approximate

    anchor_hits = Counter()
    extractable = 0
    numeric_extractable = 0
    extracted_ans_examples = []
    non_extractable_examples = []

    has_latex = 0
    has_dollar_math = 0
    has_percent = 0
    has_money = 0
    has_negative = 0
    response_starts_with_loi_giai = 0
    response_has_chunk_tra_loi = 0
    response_has_dap_an_la = 0
    duplicate_query_counter = Counter()

    for rec in records:
        q = (rec.get("query_vi") or "").strip()
        r = (rec.get("response_vi") or "").strip()
        t = rec.get("type") or "UNKNOWN"

        types[t] += 1
        duplicate_query_counter[q] += 1

        q_chars.append(len(q))
        q_tokens.append(whitespace_token_count(q))
        r_chars.append(len(r))
        r_tokens.append(whitespace_token_count(r))
        full_chars.append(len(q) + len(r))

        # response anchors
        for name, pat in ANCHOR_REs:
            if pat.search(r):
                anchor_hits[name] += 1

        if has_pattern(r, LATEX_RE):
            has_latex += 1
        if DOLLAR_RE.search(q) or DOLLAR_RE.search(r):
            has_dollar_math += 1
        if PERCENT_RE.search(q) or PERCENT_RE.search(r):
            has_percent += 1
        if DOLLAR_SIGN_RE.search(q):
            has_money += 1
        if NEGATIVE_RE.search(r):
            has_negative += 1
        if r.startswith("Lời giải"):
            response_starts_with_loi_giai += 1
        if "Câu trả lời" in r:
            response_has_chunk_tra_loi += 1
        if "Đáp án là" in r:
            response_has_dap_an_la += 1

        ans = standard_extract_answer(r)
        if ans is not None:
            extractable += 1
            num = standard_parse_number(ans)
            if num is not None:
                numeric_extractable += 1
            if len(extracted_ans_examples) < 30:
                extracted_ans_examples.append({
                    "type": t,
                    "ans_str": ans,
                    "ans_num": num,
                    "tail": r[-200:],
                })
        else:
            if len(non_extractable_examples) < 20:
                non_extractable_examples.append({
                    "type": t,
                    "tail": r[-300:],
                })

    duplicate_queries = sum(1 for _, c in duplicate_query_counter.items() if c > 1)

    return {
        "split": split_name,
        "n_records": n,
        "type_distribution": types.most_common(),
        "query_char_stats": char_stats(q_chars),
        "query_token_stats": char_stats(q_tokens),
        "response_char_stats": char_stats(r_chars),
        "response_token_stats": char_stats(r_tokens),
        "qr_total_char_stats": char_stats(full_chars),
        "anchor_hits": anchor_hits.most_common(),
        "extractable_count": extractable,
        "extractable_pct": round(100 * extractable / max(1, n), 2),
        "numeric_extractable_count": numeric_extractable,
        "numeric_extractable_pct": round(100 * numeric_extractable / max(1, n), 2),
        "feature_flags": {
            "has_latex": has_latex,
            "has_dollar_math": has_dollar_math,
            "has_percent": has_percent,
            "has_money_dollar": has_money,
            "has_negative_in_response": has_negative,
            "response_starts_with_loi_giai": response_starts_with_loi_giai,
            "response_contains_cau_tra_loi": response_has_chunk_tra_loi,
            "response_contains_dap_an_la": response_has_dap_an_la,
        },
        "duplicate_queries": duplicate_queries,
        "extract_examples": extracted_ans_examples[:10],
        "non_extractable_examples": non_extractable_examples[:5],
    }


def main() -> None:
    train_path = DATA_DIR / "train.json"
    valid_path = DATA_DIR / "valid.json"

    print(f"Loading {train_path} ...")
    train = load_records(train_path)
    print(f"Loading {valid_path} ...")
    valid = load_records(valid_path)

    summary = {
        "train": audit(train, "train"),
        "valid": audit(valid, "valid"),
    }

    out_json = OUT_DIR / "audit_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_json}")

    # Pretty text reports ------------------------------------------
    type_lines = ["# Type distribution\n"]
    for split in ("train", "valid"):
        type_lines.append(f"\n## {split} (n={summary[split]['n_records']})")
        for k, v in summary[split]["type_distribution"]:
            pct = 100 * v / max(1, summary[split]["n_records"])
            type_lines.append(f"  {k:<22s} {v:>7d}  ({pct:5.2f}%)")
    (OUT_DIR / "type_distribution.txt").write_text("\n".join(type_lines), encoding="utf-8")

    len_lines = ["# Length statistics\n"]
    for split in ("train", "valid"):
        s = summary[split]
        len_lines.append(f"\n## {split}")
        for k in ("query_char_stats", "query_token_stats", "response_char_stats", "response_token_stats", "qr_total_char_stats"):
            len_lines.append(f"  {k}: {s[k]}")
    (OUT_DIR / "length_stats.txt").write_text("\n".join(len_lines), encoding="utf-8")

    extr_lines = ["# Answer extraction audit\n"]
    for split in ("train", "valid"):
        s = summary[split]
        extr_lines.append(f"\n## {split}")
        extr_lines.append(f"  extractable: {s['extractable_count']} / {s['n_records']}  ({s['extractable_pct']}%)")
        extr_lines.append(f"  numeric extractable: {s['numeric_extractable_count']} / {s['n_records']}  ({s['numeric_extractable_pct']}%)")
        extr_lines.append("  anchor hits:")
        for k, v in s["anchor_hits"]:
            extr_lines.append(f"    {k:<20s} {v}")
        extr_lines.append("  feature_flags:")
        for k, v in s["feature_flags"].items():
            extr_lines.append(f"    {k:<32s} {v}")
        extr_lines.append("  extract examples (first 10):")
        for ex in s["extract_examples"]:
            extr_lines.append(f"    [{ex['type']}] ans_str={ex['ans_str']!r}  ans_num={ex['ans_num']}")
        extr_lines.append("  non-extractable examples:")
        for ex in s["non_extractable_examples"]:
            extr_lines.append(f"    [{ex['type']}] tail={ex['tail']!r}")
    (OUT_DIR / "answer_extraction_report.txt").write_text("\n".join(extr_lines), encoding="utf-8")

    # Concrete sample examples per type ----------------------------
    samples_lines = ["# Sample examples per type (from train)\n"]
    by_type: dict[str, list[dict]] = {}
    for rec in train:
        t = rec.get("type") or "UNKNOWN"
        by_type.setdefault(t, [])
        if len(by_type[t]) < 2:
            by_type[t].append(rec)
    for t, recs in by_type.items():
        samples_lines.append(f"\n## TYPE = {t}")
        for i, rec in enumerate(recs):
            samples_lines.append(f"\n--- example {i+1} ---")
            samples_lines.append(f"QUERY: {rec.get('query_vi','')[:600]}")
            samples_lines.append(f"RESPONSE: {rec.get('response_vi','')[:1200]}")
    (OUT_DIR / "sample_examples.txt").write_text("\n".join(samples_lines), encoding="utf-8")

    print("Wrote text reports to", OUT_DIR)


if __name__ == "__main__":
    main()
