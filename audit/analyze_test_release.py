from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
OUT_JSON = ROOT / "audit" / "test_release_audit.json"
OUT_MD = ROOT / "audit" / "test_release_audit.md"


WORD_RE = re.compile(r"\w+", re.UNICODE)
NUM_RE = re.compile(
    r"(?<!\w)-?\d+(?:[.,]\d+)?(?:/\d+(?:[.,]\d+)?)?(?!\w)",
    re.UNICODE,
)


def load_records(name: str) -> list[dict]:
    path = DATASET_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_alnum(text: str | None) -> str:
    text = normalize_text(text)
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)


def tokens(text: str | None) -> list[str]:
    return WORD_RE.findall(normalize_text(text))


def token_set(text: str | None) -> set[str]:
    return set(tokens(text))


def mask_numbers(text: str | None) -> str:
    return NUM_RE.sub("<NUM>", normalize_text(text))


def number_signature(text: str | None) -> tuple[str, ...]:
    return tuple(NUM_RE.findall(normalize_text(text)))


def stats(values: list[int | float]) -> dict:
    if not values:
        return {"n": 0}
    values_sorted = sorted(values)
    n = len(values_sorted)

    def q(p: float) -> float:
        if n == 1:
            return float(values_sorted[0])
        pos = (n - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return float(values_sorted[lo])
        return float(values_sorted[lo] + (values_sorted[hi] - values_sorted[lo]) * (pos - lo))

    return {
        "n": n,
        "min": values_sorted[0],
        "p25": q(0.25),
        "median": median(values_sorted),
        "mean": mean(values_sorted),
        "p75": q(0.75),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": values_sorted[-1],
    }


def exact_overlap(source: list[dict], target: list[dict], field: str = "query_vi") -> dict:
    source_norm = defaultdict(list)
    for i, rec in enumerate(source):
        source_norm[normalize_text(rec.get(field))].append(i)

    source_alnum = defaultdict(list)
    for i, rec in enumerate(source):
        source_alnum[normalize_alnum(rec.get(field))].append(i)

    exact_ids = []
    alnum_ids = []
    for j, rec in enumerate(target):
        key = normalize_text(rec.get(field))
        if key in source_norm:
            exact_ids.append(j)
        key2 = normalize_alnum(rec.get(field))
        if key2 in source_alnum:
            alnum_ids.append(j)

    return {
        "field": field,
        "exact_count": len(exact_ids),
        "alnum_count": len(alnum_ids),
        "exact_examples": exact_ids[:10],
        "alnum_examples": alnum_ids[:10],
    }


def build_inverted_index(
    records: list[dict],
    *,
    max_df: int = 2500,
) -> tuple[list[set[str]], dict[str, list[int]], set[str]]:
    sets: list[set[str]] = []
    df: Counter[str] = Counter()
    for rec in records:
        ts = token_set(rec.get("query_vi"))
        sets.append(ts)
        df.update(ts)

    skipped = {tok for tok, count in df.items() if count > max_df or len(tok) <= 1}
    inv: dict[str, list[int]] = defaultdict(list)
    for i, ts in enumerate(sets):
        for tok in ts:
            if tok in skipped:
                continue
            inv[tok].append(i)
    return sets, inv, skipped


def nearest_jaccard(source: list[dict], target: list[dict], top_k_examples: int = 12) -> dict:
    source_sets, inv, skipped = build_inverted_index(source)
    scores = []
    rows = []

    for j, rec in enumerate(target):
        ts = token_set(rec.get("query_vi"))
        query_terms = [tok for tok in ts if tok not in skipped]
        counts: Counter[int] = Counter()
        for tok in query_terms:
            for i in inv.get(tok, []):
                counts[i] += 1

        if len(counts) > 12000:
            candidate_items = counts.most_common(12000)
        else:
            candidate_items = counts.items()

        best_i = None
        best_score = 0.0
        for i, filtered_inter in candidate_items:
            inter = len(ts & source_sets[i])
            union = len(ts) + len(source_sets[i]) - inter
            score = inter / union if union else 0.0
            if score > best_score:
                best_score = score
                best_i = i
        scores.append(best_score)
        rows.append(
            {
                "test_id": rec.get("id", j),
                "test_index": j,
                "score": best_score,
                "source_index": best_i,
                "test_query": rec.get("query_vi"),
                "source_query": source[best_i].get("query_vi") if best_i is not None else None,
                "source_type": source[best_i].get("type") if best_i is not None else None,
                "test_numbers": list(number_signature(rec.get("query_vi"))),
                "source_numbers": list(number_signature(source[best_i].get("query_vi"))) if best_i is not None else [],
            }
        )

    buckets = {
        ">=0.95": sum(s >= 0.95 for s in scores),
        ">=0.90": sum(s >= 0.90 for s in scores),
        ">=0.80": sum(s >= 0.80 for s in scores),
        ">=0.70": sum(s >= 0.70 for s in scores),
        ">=0.60": sum(s >= 0.60 for s in scores),
        ">=0.50": sum(s >= 0.50 for s in scores),
        "<0.50": sum(s < 0.50 for s in scores),
    }

    examples = sorted(rows, key=lambda r: r["score"], reverse=True)[:top_k_examples]
    low_examples = sorted(rows, key=lambda r: r["score"])[:top_k_examples]

    return {
        "score_stats": stats(scores),
        "buckets": buckets,
        "skipped_high_df_token_count": len(skipped),
        "top_examples": examples,
        "low_examples": low_examples,
    }


def summarize_records(name: str, records: list[dict]) -> dict:
    query_lengths_chars = [len(rec.get("query_vi") or "") for rec in records]
    query_lengths_words = [len(tokens(rec.get("query_vi"))) for rec in records]
    number_counts = [len(number_signature(rec.get("query_vi"))) for rec in records]
    template_counts = Counter(mask_numbers(rec.get("query_vi")) for rec in records)
    type_counts = Counter(rec.get("type", "<missing>") for rec in records)
    keys = Counter(tuple(sorted(rec.keys())) for rec in records)
    id_values = [rec.get("id") for rec in records if "id" in rec]

    duplicate_queries = Counter(normalize_text(rec.get("query_vi")) for rec in records)
    duplicate_query_groups = {k: v for k, v in duplicate_queries.items() if k and v > 1}

    return {
        "name": name,
        "n": len(records),
        "keys": {", ".join(k): v for k, v in keys.items()},
        "type_counts": dict(type_counts.most_common()),
        "has_response_vi": sum("response_vi" in rec and rec.get("response_vi") not in (None, "") for rec in records),
        "id_present": sum("id" in rec for rec in records),
        "id_unique": len(set(id_values)) if id_values else 0,
        "id_min": min(id_values) if id_values else None,
        "id_max": max(id_values) if id_values else None,
        "id_is_0_to_n_minus_1": sorted(id_values) == list(range(len(records))) if id_values else False,
        "query_chars": stats(query_lengths_chars),
        "query_words": stats(query_lengths_words),
        "query_number_counts": stats(number_counts),
        "duplicate_query_groups": len(duplicate_query_groups),
        "duplicate_query_extra_records": sum(v - 1 for v in duplicate_query_groups.values()),
        "top_templates": [
            {"count": c, "template": t}
            for t, c in template_counts.most_common(25)
        ],
    }


def response_answer_map(records: list[dict]) -> dict[str, Counter[str]]:
    # Lightweight answer extraction for conflict checks, enough for anchor-style responses.
    anchor_re = re.compile(r"(?:đáp\s*án\s*là|câu\s*trả\s*lời\s*là|####)\s*:?\s*([^\n.;,]+)", re.I)
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in records:
        q = normalize_text(rec.get("query_vi"))
        resp = rec.get("response_vi") or ""
        matches = anchor_re.findall(resp)
        ans = matches[-1].strip() if matches else None
        if q and ans:
            out[q][ans] += 1
    return out


def template_family(query: str) -> str:
    q = normalize_text(query)
    if "cấp số cộng" in q:
        return "arithmetic_sequence"
    if "cấp số nhân" in q:
        return "geometric_sequence"
    if "trung bình cộng" in q:
        return "average"
    if "diện tích" in q and "hình chữ nhật" in q:
        return "rectangle_area"
    if "chu vi" in q and "hình chữ nhật" in q:
        return "rectangle_perimeter"
    if "phần trăm" in q or "%" in q:
        return "percentage"
    if "phương trình" in q or "giá trị của x" in q:
        return "equation"
    if "vận tốc" in q or "quãng đường" in q or "thời gian" in q:
        return "motion"
    if "xác suất" in q:
        return "probability"
    if "tổ hợp" in q or "hoán vị" in q:
        return "combinatorics"
    return "other"


def main() -> None:
    train = load_records("train")
    valid = load_records("valid")
    test = load_records("test")

    train_valid = train + valid

    template_counts = Counter(mask_numbers(rec.get("query_vi")) for rec in test)
    template_examples = defaultdict(list)
    for rec in test:
        t = mask_numbers(rec.get("query_vi"))
        if len(template_examples[t]) < 3:
            template_examples[t].append(
                {
                    "id": rec.get("id"),
                    "query_vi": rec.get("query_vi"),
                    "numbers": list(number_signature(rec.get("query_vi"))),
                }
            )

    family_counts = Counter(template_family(rec.get("query_vi") or "") for rec in test)
    family_examples = defaultdict(list)
    for rec in test:
        fam = template_family(rec.get("query_vi") or "")
        if len(family_examples[fam]) < 5:
            family_examples[fam].append({"id": rec.get("id"), "query_vi": rec.get("query_vi")})

    train_resp_by_query = response_answer_map(train)
    valid_resp_by_query = response_answer_map(valid)
    exact_train_answers = []
    exact_valid_answers = []
    for rec in test:
        q = normalize_text(rec.get("query_vi"))
        if q in train_resp_by_query:
            exact_train_answers.append({"id": rec.get("id"), "answers": train_resp_by_query[q].most_common(5)})
        if q in valid_resp_by_query:
            exact_valid_answers.append({"id": rec.get("id"), "answers": valid_resp_by_query[q].most_common(5)})

    audit = {
        "summaries": {
            "train": summarize_records("train", train),
            "valid": summarize_records("valid", valid),
            "test": summarize_records("test", test),
        },
        "legal_query_overlap": {
            "train_query_vs_test_query": exact_overlap(train, test, "query_vi"),
            "valid_query_vs_test_query": exact_overlap(valid, test, "query_vi"),
            "train_plus_valid_query_vs_test_query": exact_overlap(train_valid, test, "query_vi"),
        },
        "exact_answer_lookup_if_query_duplicate": {
            "train_count": len(exact_train_answers),
            "valid_count": len(exact_valid_answers),
            "train_examples": exact_train_answers[:10],
            "valid_examples": exact_valid_answers[:10],
        },
        "nearest_query_jaccard": {
            "train": nearest_jaccard(train, test),
            "valid": nearest_jaccard(valid, test),
        },
        "test_templates": {
            "unique_masked_templates": len(template_counts),
            "top_templates_with_examples": [
                {
                    "count": c,
                    "template": t,
                    "examples": template_examples[t],
                }
                for t, c in template_counts.most_common(30)
            ],
            "family_counts": dict(family_counts.most_common()),
            "family_examples": dict(family_examples),
        },
    }

    OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    test_summary = audit["summaries"]["test"]
    lines = [
        "# Test Release Audit",
        "",
        "## Schema",
        f"- test n={test_summary['n']}",
        f"- keys: {test_summary['keys']}",
        f"- has_response_vi={test_summary['has_response_vi']}",
        f"- id range: {test_summary['id_min']}..{test_summary['id_max']}, unique={test_summary['id_unique']}, contiguous={test_summary['id_is_0_to_n_minus_1']}",
        "",
        "## Type Distribution",
    ]
    for k, v in test_summary["type_counts"].items():
        lines.append(f"- {k}: {v}")

    lines.extend(
        [
            "",
            "## Length / Number Stats",
            f"- query_words: {test_summary['query_words']}",
            f"- query_chars: {test_summary['query_chars']}",
            f"- number_count_in_query: {test_summary['query_number_counts']}",
            "",
            "## Legal Query Overlap",
        ]
    )
    for name, info in audit["legal_query_overlap"].items():
        lines.append(
            f"- {name}: exact={info['exact_count']}, alnum={info['alnum_count']}"
        )

    lines.extend(
        [
            "",
            "## Nearest Query Jaccard",
            f"- train: buckets={audit['nearest_query_jaccard']['train']['buckets']}, stats={audit['nearest_query_jaccard']['train']['score_stats']}",
            f"- valid: buckets={audit['nearest_query_jaccard']['valid']['buckets']}, stats={audit['nearest_query_jaccard']['valid']['score_stats']}",
            "",
            "## Test Template Families",
        ]
    )
    for k, v in audit["test_templates"]["family_counts"].items():
        lines.append(f"- {k}: {v}")

    lines.extend(["", "## Top Masked Templates"])
    for item in audit["test_templates"]["top_templates_with_examples"][:20]:
        lines.append(f"- count={item['count']} | {item['template']}")
        for ex in item["examples"][:2]:
            lines.append(f"  - id={ex['id']} nums={ex['numbers']} query={ex['query_vi']}")

    lines.extend(["", "## Highest Train Nearest Examples"])
    for row in audit["nearest_query_jaccard"]["train"]["top_examples"][:8]:
        lines.append(
            f"- id={row['test_id']} jaccard={row['score']:.3f} source_type={row['source_type']}"
        )
        lines.append(f"  - test: {row['test_query']}")
        lines.append(f"  - nearest train: {row['source_query']}")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote] {OUT_JSON.relative_to(ROOT)}")
    print(f"[wrote] {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
