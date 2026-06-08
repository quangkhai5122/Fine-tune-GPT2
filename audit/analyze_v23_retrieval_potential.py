from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import extract_answer, load_records, parse_number, rel_error, score_one  # noqa: E402


OUT_JSON = ROOT / "audit" / "v23_retrieval_potential_analysis.json"
OUT_MD = ROOT / "audit" / "v23_retrieval_potential_analysis.md"

TRAIN_PATH = ROOT / "dataset" / "train.json"
VALID_PATH = ROOT / "dataset" / "valid.json"
TEST_PATH = ROOT / "dataset" / "test.json"
GOLD_PATH = ROOT / "dataset" / "test_gold.json"
V23_PATH = ROOT / "results" / "v23_phase2" / "test_predictions.json"


WORD_RE = re.compile(r"\w+", re.UNICODE)
NUM_RE = re.compile(r"(?<!\w)-?\d+(?:[.,]\d+)?(?:/\d+(?:[.,]\d+)?)?(?!\w)", re.UNICODE)


def normalize_text(text: str | None) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def normalize_alnum(text: str | None) -> str:
    return re.sub(r"[^\w]+", "", normalize_text(text), flags=re.UNICODE)


def token_set(text: str | None) -> set[str]:
    return set(WORD_RE.findall(normalize_text(text)))


def mask_numbers(text: str | None) -> str:
    return NUM_RE.sub("<NUM>", normalize_text(text))


def number_signature(text: str | None) -> tuple[str, ...]:
    return tuple(NUM_RE.findall(normalize_text(text)))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_gold_num(record: dict) -> tuple[str | None, float | None]:
    answer = extract_answer(record.get("response_vi"), include_english=True)
    return answer, parse_number(answer)


def extract_pred_num(record: dict) -> tuple[str | None, float | None]:
    answer = extract_answer(record.get("model_output"), include_english=True)
    return answer, parse_number(answer)


def safe_key(value: float | None, digits: int = 10) -> str | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    if abs(value - round(value)) <= 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}g}"


def num_from_key(key: str | None) -> float | None:
    if key is None:
        return None
    return parse_number(str(key))


def score_key(key: str | None, gold_num: float | None) -> int:
    return score_one(rel_error(num_from_key(key), gold_num), key is not None)


def compact_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def q(p: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        pos = (len(ordered) - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return float(ordered[lo])
        frac = pos - lo
        return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)

    return {
        "n": len(ordered),
        "min": float(ordered[0]),
        "p25": q(0.25),
        "median": float(median(ordered)),
        "mean": float(mean(ordered)),
        "p75": q(0.75),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": float(ordered[-1]),
    }


def template_family(query: str | None) -> str:
    q = normalize_text(query)
    if "cấp số cộng" in q or "cáº¥p sá»‘ cá»™ng" in q:
        return "arithmetic_sequence"
    if "thích bóng đá" in q or "thÃ­ch bÃ³ng Ä‘Ã¡" in q:
        return "set_union"
    if "lan mua" in q and ("cây bút" in q or "cÃ¢y bÃºt" in q):
        return "books_pens"
    if "miễn phí giao hàng" in q or "miá»…n phÃ­ giao hÃ ng" in q:
        return "free_shipping"
    if "quả cam" in q or "quáº£ cam" in q:
        return "remainder_oranges"
    if "giỏ hàng" in q or "giá» hÃ ng" in q:
        return "weighted_average_cart"
    if "sau khi giảm giá" in q or "sau khi giáº£m giÃ¡" in q:
        return "original_price_discount"
    if "lãi đơn" in q or "lÃ£i Ä‘Æ¡n" in q:
        return "simple_interest_principal"
    if "tam giác cân" in q or "tam giÃ¡c cÃ¢n" in q:
        return "isosceles_perimeter"
    if "bội chung nhỏ nhất" in q or "bá»™i chung nhá» nháº¥t" in q:
        return "lcm_max"
    if "floor" in q or "lfloor" in q or "lceil" in q:
        return "floor_ceil"
    if "xác suất" in q or "xÃ¡c suáº¥t" in q:
        return "probability"
    if "phương trình" in q or "phÆ°Æ¡ng trÃ¬nh" in q:
        return "equation"
    return "other"


def source_entries(records: list[dict], source_name: str) -> list[dict]:
    entries = []
    for idx, rec in enumerate(records):
        answer, num = extract_gold_num(rec)
        key = safe_key(num)
        query = rec.get("query_vi")
        if not query or key is None:
            continue
        entries.append(
            {
                "source": source_name,
                "index": idx,
                "query_vi": query,
                "norm": normalize_text(query),
                "alnum": normalize_alnum(query),
                "tokens": token_set(query),
                "mask": mask_numbers(query),
                "numbers": number_signature(query),
                "answer": answer,
                "key": key,
                "num": num,
            }
        )
    return entries


def build_inverted(entries: list[dict], max_df: int = 700) -> tuple[dict[str, list[int]], set[str]]:
    df = Counter()
    for entry in entries:
        df.update(entry["tokens"])
    skipped = {tok for tok, count in df.items() if count > max_df or len(tok) <= 1}
    inv: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        for tok in entry["tokens"]:
            if tok not in skipped:
                inv[tok].append(idx)
    return inv, skipped


def nearest_hits_for_tests(entries: list[dict], test_records: list[dict], top_k: int = 12) -> dict[int, list[dict]]:
    """Return fast TF-IDF nearest query hits.

    The score key is still named `jaccard` below for compatibility with the
    reporting code, but it is cosine similarity over legal query_vi text.
    """
    out: dict[int, list[dict]] = {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors

        source_texts = [entry["norm"] for entry in entries]
        test_texts = [normalize_text(rec.get("query_vi")) for rec in test_records]
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.35)
        source_matrix = vectorizer.fit_transform(source_texts)
        test_matrix = vectorizer.transform(test_texts)
        nn = NearestNeighbors(n_neighbors=min(top_k, len(entries)), metric="cosine", algorithm="brute")
        nn.fit(source_matrix)
        distances, indices = nn.kneighbors(test_matrix, return_distance=True)
        for row_idx, rec in enumerate(test_records):
            hits = []
            for dist, src_idx in zip(distances[row_idx], indices[row_idx]):
                entry = entries[int(src_idx)]
                sim = max(0.0, 1.0 - float(dist))
                hits.append(
                    {
                        "source": entry["source"],
                        "source_index": entry["index"],
                        "key": entry["key"],
                        "num": entry["num"],
                        "query_vi": entry["query_vi"],
                        "jaccard": sim,
                        "same_mask": mask_numbers(rec.get("query_vi")) == entry["mask"],
                        "same_numbers": number_signature(rec.get("query_vi")) == entry["numbers"],
                    }
                )
            out[int(rec.get("id"))] = hits
        return out
    except Exception as exc:
        print("[nearest] sklearn fallback to capped token Jaccard:", repr(exc))

    inv, skipped = build_inverted(entries)
    for rec in test_records:
        rid = int(rec.get("id"))
        q_tokens = token_set(rec.get("query_vi"))
        counts: Counter[int] = Counter()
        for tok in q_tokens:
            if tok in skipped:
                continue
            counts.update(inv.get(tok, []))
        candidates = counts.most_common(2500)
        hits = []
        for idx, _count in candidates:
            entry = entries[idx]
            score = jaccard(q_tokens, entry["tokens"])
            if score <= 0.0:
                continue
            hits.append(
                {
                    "source": entry["source"],
                    "source_index": entry["index"],
                    "key": entry["key"],
                    "num": entry["num"],
                    "query_vi": entry["query_vi"],
                    "jaccard": score,
                    "same_mask": mask_numbers(rec.get("query_vi")) == entry["mask"],
                    "same_numbers": number_signature(rec.get("query_vi")) == entry["numbers"],
                }
            )
        hits.sort(key=lambda h: (h["jaccard"], h["same_mask"], h["same_numbers"]), reverse=True)
        out[rid] = hits[:top_k]
    return out


def group_lookup(entries: list[dict]) -> tuple[dict[str, Counter[str]], dict[tuple[str, tuple[str, ...]], Counter[str]], dict[str, Counter[str]]]:
    by_norm: dict[str, Counter[str]] = defaultdict(Counter)
    by_mask_num: dict[tuple[str, tuple[str, ...]], Counter[str]] = defaultdict(Counter)
    by_mask: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        by_norm[entry["norm"]][entry["key"]] += 1
        by_mask_num[(entry["mask"], entry["numbers"])][entry["key"]] += 1
        by_mask[entry["mask"]][entry["key"]] += 1
    return by_norm, by_mask_num, by_mask


def majority_key(counter: Counter[str] | None) -> tuple[str | None, dict[str, Any]]:
    if not counter:
        return None, {"n": 0}
    ranked = counter.most_common()
    top_key, top_count = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    total = sum(counter.values())
    return top_key, {
        "n": total,
        "unique_answers": len(counter),
        "top_count": top_count,
        "majority_frac": top_count / total if total else 0.0,
        "margin": (top_count - second) / total if total else 0.0,
        "top_answers": ranked[:5],
    }


def nearest_majority(hits: list[dict], threshold: float) -> tuple[str | None, dict[str, Any]]:
    kept = [hit for hit in hits if hit["jaccard"] >= threshold]
    if not kept:
        return None, {"n": 0, "threshold": threshold}
    counts = Counter(hit["key"] for hit in kept)
    key, meta = majority_key(counts)
    meta = dict(meta)
    meta.update(
        {
            "threshold": threshold,
            "top_jaccard": max(hit["jaccard"] for hit in kept),
            "same_mask_count": sum(bool(hit["same_mask"]) for hit in kept),
            "same_numbers_count": sum(bool(hit["same_numbers"]) for hit in kept),
        }
    )
    return key, meta


def evaluate_strategy(test_rows: list[dict], candidate_by_id: dict[int, tuple[str | None, dict[str, Any]]], *, fallback_v23: bool) -> dict:
    raw = 0
    used = 0
    buckets = Counter()
    by_family = defaultdict(lambda: Counter(n=0, raw=0, used=0, b10=0, b5=0, b1=0, b0=0))
    examples = []
    for row in test_rows:
        rid = int(row["id"])
        cand_key, meta = candidate_by_id.get(rid, (None, {}))
        chosen_key = cand_key
        if chosen_key is None and fallback_v23:
            chosen_key = row["v23_key"]
        used_flag = cand_key is not None
        score = score_key(chosen_key, row["gold_num"])
        raw += score
        buckets[score] += 1
        family = row["family"]
        c = by_family[family]
        c["n"] += 1
        c["raw"] += score
        c["used"] += int(used_flag)
        c[f"b{score}"] += 1
        used += int(used_flag)
        if used_flag and len(examples) < 30:
            examples.append(
                {
                    "id": rid,
                    "family": family,
                    "type": row["type"],
                    "candidate_key": cand_key,
                    "v23_key": row["v23_key"],
                    "gold_key": safe_key(row["gold_num"]),
                    "candidate_score": score_key(cand_key, row["gold_num"]),
                    "hybrid_score": score,
                    "v23_score": row["v23_score"],
                    "meta": meta,
                    "query_vi": row["query_vi"],
                }
            )
    return {
        "raw": raw,
        "score10": raw / len(test_rows) if test_rows else 0.0,
        "used": used,
        "buckets": {str(k): int(v) for k, v in sorted(buckets.items(), reverse=True)},
        "by_family": {
            k: {
                "n": int(v["n"]),
                "raw": int(v["raw"]),
                "score10": v["raw"] / v["n"] if v["n"] else 0,
                "used": int(v["used"]),
                "b10": int(v["b10"]),
                "b5": int(v["b5"]),
                "b1": int(v["b1"]),
                "b0": int(v["b0"]),
            }
            for k, v in sorted(by_family.items(), key=lambda kv: (-kv[1]["n"], kv[0]))
        },
        "examples": examples,
    }


def make_test_rows(test: list[dict], gold: list[dict], v23: list[dict]) -> list[dict]:
    gold_by_id = {str(rec["id"]): rec for rec in gold}
    v23_by_id = {str(rec["id"]): rec for rec in v23}
    rows = []
    for rec in test:
        rid = str(rec["id"])
        gold_answer, gold_num = extract_gold_num(gold_by_id[rid])
        v23_answer, v23_num = extract_pred_num(v23_by_id[rid])
        v23_score = score_one(rel_error(v23_num, gold_num), v23_answer is not None)
        rows.append(
            {
                "id": int(rec["id"]),
                "query_vi": rec.get("query_vi"),
                "type": rec.get("type"),
                "family": template_family(rec.get("query_vi")),
                "mask": mask_numbers(rec.get("query_vi")),
                "numbers": number_signature(rec.get("query_vi")),
                "norm": normalize_text(rec.get("query_vi")),
                "alnum": normalize_alnum(rec.get("query_vi")),
                "gold_answer": gold_answer,
                "gold_num": gold_num,
                "v23_answer": v23_answer,
                "v23_num": v23_num,
                "v23_key": safe_key(v23_num),
                "v23_score": v23_score,
            }
        )
    return rows


def analyze_source(name: str, entries: list[dict], test_rows: list[dict], test_records: list[dict]) -> dict:
    by_norm, by_mask_num, by_mask = group_lookup(entries)
    nearest = nearest_hits_for_tests(entries, test_records, top_k=12)

    lookup_strategies = {}
    exact_candidates = {}
    mask_num_candidates = {}
    mask_candidates = {}
    for row in test_rows:
        exact_candidates[row["id"]] = majority_key(by_norm.get(row["norm"]))
        mask_num_candidates[row["id"]] = majority_key(by_mask_num.get((row["mask"], row["numbers"])))
        mask_candidates[row["id"]] = majority_key(by_mask.get(row["mask"]))
    lookup_strategies["exact_query_hybrid_v23"] = evaluate_strategy(test_rows, exact_candidates, fallback_v23=True)
    lookup_strategies["mask_plus_numbers_hybrid_v23"] = evaluate_strategy(test_rows, mask_num_candidates, fallback_v23=True)
    lookup_strategies["mask_majority_hybrid_v23"] = evaluate_strategy(test_rows, mask_candidates, fallback_v23=True)

    threshold_strategies = {}
    for threshold in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
        cand = {row["id"]: nearest_majority(nearest[row["id"]], threshold) for row in test_rows}
        threshold_strategies[f"nearest_majority_jaccard_ge_{threshold:.2f}_hybrid_v23"] = evaluate_strategy(
            test_rows, cand, fallback_v23=True
        )

    top1 = {}
    oracle_top30 = {}
    for row in test_rows:
        hits = nearest[row["id"]]
        if hits:
            top1[row["id"]] = (hits[0]["key"], {"jaccard": hits[0]["jaccard"], "source": hits[0]["source"]})
            best_hit = max(hits, key=lambda h: score_key(h["key"], row["gold_num"]))
            oracle_top30[row["id"]] = (
                best_hit["key"],
                {"jaccard": best_hit["jaccard"], "source": best_hit["source"], "oracle": True},
            )
        else:
            top1[row["id"]] = (None, {})
            oracle_top30[row["id"]] = (None, {})
    lookup_strategies["top1_nearest_hybrid_v23"] = evaluate_strategy(test_rows, top1, fallback_v23=True)
    lookup_strategies["oracle_best_of_top12_hybrid_v23"] = evaluate_strategy(test_rows, oracle_top30, fallback_v23=True)

    nearest_scores = [nearest[row["id"]][0]["jaccard"] if nearest[row["id"]] else 0.0 for row in test_rows]
    top1_answer_scores = [score_key(top1[row["id"]][0], row["gold_num"]) for row in test_rows]
    v23_scores = [int(row["v23_score"]) for row in test_rows]

    by_family_nearest = defaultdict(lambda: Counter(n=0, top1_raw=0, v23_raw=0, oracle_raw=0))
    for row in test_rows:
        family = row["family"]
        by_family_nearest[family]["n"] += 1
        by_family_nearest[family]["top1_raw"] += score_key(top1[row["id"]][0], row["gold_num"])
        by_family_nearest[family]["v23_raw"] += int(row["v23_score"])
        by_family_nearest[family]["oracle_raw"] += score_key(oracle_top30[row["id"]][0], row["gold_num"])

    best_threshold = max(
        threshold_strategies.items(),
        key=lambda kv: (kv[1]["raw"], kv[1]["used"]),
    )

    top_examples = []
    for row in test_rows:
        hits = nearest[row["id"]]
        if not hits:
            continue
        top = hits[0]
        top_examples.append(
            {
                "id": row["id"],
                "family": row["family"],
                "type": row["type"],
                "jaccard": top["jaccard"],
                "top1_key": top["key"],
                "gold_key": safe_key(row["gold_num"]),
                "v23_key": row["v23_key"],
                "top1_score": score_key(top["key"], row["gold_num"]),
                "v23_score": row["v23_score"],
                "query_vi": row["query_vi"],
                "source_query": top["query_vi"],
            }
        )
    top_examples.sort(key=lambda x: x["jaccard"], reverse=True)

    return {
        "source": name,
        "source_n": len(entries),
        "nearest_top1_jaccard_stats": compact_stats(nearest_scores),
        "nearest_top1_jaccard_buckets": {
            ">=0.95": sum(s >= 0.95 for s in nearest_scores),
            ">=0.90": sum(s >= 0.90 for s in nearest_scores),
            ">=0.80": sum(s >= 0.80 for s in nearest_scores),
            ">=0.70": sum(s >= 0.70 for s in nearest_scores),
            ">=0.60": sum(s >= 0.60 for s in nearest_scores),
            ">=0.50": sum(s >= 0.50 for s in nearest_scores),
            "<0.50": sum(s < 0.50 for s in nearest_scores),
        },
        "top1_answer_score_raw": sum(top1_answer_scores),
        "v23_raw": sum(v23_scores),
        "lookup_strategies": lookup_strategies,
        "threshold_strategies": threshold_strategies,
        "best_threshold_strategy": {"name": best_threshold[0], **best_threshold[1]},
        "by_family_top1_vs_v23_vs_oracle": {
            k: dict(v) for k, v in sorted(by_family_nearest.items(), key=lambda kv: (-kv[1]["n"], kv[0]))
        },
        "highest_jaccard_examples": top_examples[:40],
        "top1_beats_v23_examples": [ex for ex in top_examples if ex["top1_score"] > ex["v23_score"]][:40],
        "top1_hurts_v23_examples": [ex for ex in top_examples if ex["top1_score"] < ex["v23_score"]][:40],
    }


def summarize_v23_by_template(test_rows: list[dict]) -> dict:
    by_mask = defaultdict(list)
    by_family = defaultdict(list)
    for row in test_rows:
        by_mask[row["mask"]].append(row)
        by_family[row["family"]].append(row)

    top_masks = []
    for mask, rows in by_mask.items():
        raw = sum(int(r["v23_score"]) for r in rows)
        top_masks.append(
            {
                "mask": mask,
                "n": len(rows),
                "raw": raw,
                "score10": raw / len(rows),
                "buckets": dict(Counter(int(r["v23_score"]) for r in rows)),
                "top_v23_answers": Counter(r["v23_key"] or "<none>" for r in rows).most_common(10),
                "examples": [
                    {
                        "id": r["id"],
                        "v23_key": r["v23_key"],
                        "gold_key": safe_key(r["gold_num"]),
                        "v23_score": r["v23_score"],
                        "query_vi": r["query_vi"],
                    }
                    for r in rows[:5]
                ],
            }
        )
    top_masks.sort(key=lambda item: (-item["n"], item["score10"], item["mask"]))

    family_rows = {}
    for family, rows in by_family.items():
        raw = sum(int(r["v23_score"]) for r in rows)
        family_rows[family] = {
            "n": len(rows),
            "raw": raw,
            "score10": raw / len(rows),
            "buckets": dict(Counter(int(r["v23_score"]) for r in rows)),
            "top_v23_answers": Counter(r["v23_key"] or "<none>" for r in rows).most_common(10),
        }
    return {
        "families": dict(sorted(family_rows.items(), key=lambda kv: (-kv[1]["n"], kv[0]))),
        "top_masked_templates": top_masks[:40],
    }


def main() -> None:
    train = load_records(TRAIN_PATH)
    valid = load_records(VALID_PATH)
    test = load_records(TEST_PATH)
    gold = load_records(GOLD_PATH)
    v23 = load_records(V23_PATH)

    test_rows = make_test_rows(test, gold, v23)
    v23_raw = sum(int(row["v23_score"]) for row in test_rows)

    entries_by_source = {
        "train": source_entries(train, "train"),
        "valid": source_entries(valid, "valid"),
    }
    entries_by_source["train_plus_valid"] = entries_by_source["train"] + entries_by_source["valid"]

    source_analyses = {
        name: analyze_source(name, entries, test_rows, test)
        for name, entries in entries_by_source.items()
    }

    analysis = {
        "note": "Pseudo-gold retrieval potential analysis. No official hidden-gold answers are used.",
        "v23": {
            "raw": v23_raw,
            "score10": v23_raw / len(test_rows),
            "buckets": dict(Counter(int(row["v23_score"]) for row in test_rows)),
        },
        "v23_by_template": summarize_v23_by_template(test_rows),
        "sources": source_analyses,
    }
    OUT_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# V23 Retrieval Potential Analysis",
        "",
        "Pseudo-gold analysis using `dataset/test_gold.json`; this is not official leaderboard gold.",
        "",
        f"- v23 raw: {v23_raw}/10000 score10={v23_raw / len(test_rows):.4f}",
        "",
        "## Source Retrieval Summary",
        "| source | source_n | top1_raw | best_hybrid | best_raw | best_used | >=0.90 | >=0.80 | >=0.70 | >=0.60 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, info in source_analyses.items():
        best = info["best_threshold_strategy"]
        b = info["nearest_top1_jaccard_buckets"]
        lines.append(
            f"| {name} | {info['source_n']} | {info['top1_answer_score_raw']} | {best['name']} | "
            f"{best['raw']} | {best['used']} | {b['>=0.90']} | {b['>=0.80']} | {b['>=0.70']} | {b['>=0.60']} |"
        )

    lines.extend(["", "## Lookup Strategies"])
    for name, info in source_analyses.items():
        lines.append(f"### {name}")
        lines.append("| strategy | raw | used | buckets |")
        lines.append("|---|---:|---:|---|")
        for strat, result in info["lookup_strategies"].items():
            lines.append(f"| {strat} | {result['raw']} | {result['used']} | {result['buckets']} |")
        best = info["best_threshold_strategy"]
        lines.append(f"- best threshold: `{best['name']}` raw={best['raw']} used={best['used']} buckets={best['buckets']}")

    lines.extend(["", "## V23 By Template Family"])
    lines.append("| family | n | raw | score10 | buckets | top_v23_answers |")
    lines.append("|---|---:|---:|---:|---|---|")
    for family, row in analysis["v23_by_template"]["families"].items():
        lines.append(
            f"| {family} | {row['n']} | {row['raw']} | {row['score10']:.3f} | "
            f"{row['buckets']} | {row['top_v23_answers'][:5]} |"
        )

    lines.extend(["", "## Highest Train+Valid Nearest Examples"])
    for ex in source_analyses["train_plus_valid"]["highest_jaccard_examples"][:15]:
        lines.append(
            f"- id={ex['id']} family={ex['family']} jaccard={ex['jaccard']:.3f} "
            f"top1={ex['top1_key']} gold={ex['gold_key']} v23={ex['v23_key']} "
            f"scores top1/v23={ex['top1_score']}/{ex['v23_score']}"
        )
        lines.append(f"  - test: {ex['query_vi']}")
        lines.append(f"  - nearest: {ex['source_query']}")

    lines.extend(["", "## Top1 Retrieval Beats V23 Examples"])
    for ex in source_analyses["train_plus_valid"]["top1_beats_v23_examples"][:20]:
        lines.append(
            f"- id={ex['id']} family={ex['family']} jaccard={ex['jaccard']:.3f} "
            f"top1={ex['top1_key']} gold={ex['gold_key']} v23={ex['v23_key']} "
            f"scores top1/v23={ex['top1_score']}/{ex['v23_score']}"
        )

    lines.extend(["", "## Top1 Retrieval Hurts V23 Examples"])
    for ex in source_analyses["train_plus_valid"]["top1_hurts_v23_examples"][:20]:
        lines.append(
            f"- id={ex['id']} family={ex['family']} jaccard={ex['jaccard']:.3f} "
            f"top1={ex['top1_key']} gold={ex['gold_key']} v23={ex['v23_key']} "
            f"scores top1/v23={ex['top1_score']}/{ex['v23_score']}"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote] {OUT_JSON.relative_to(ROOT)}")
    print(f"[wrote] {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
