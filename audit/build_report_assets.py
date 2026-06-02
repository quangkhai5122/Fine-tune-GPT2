"""Build report-ready audit tables and figures for the GPT-2 math runs.

The script is intentionally offline and reproducible: it reads only local
artifacts under dataset/, audit/, and results/, then writes tables, figure data,
figures, and a compact summary under report/.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import extract_gold  # noqa: E402


DATA_DIR = ROOT / "dataset"
AUDIT_DIR = ROOT / "audit"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
DATA_OUT_DIR = REPORT_DIR / "data"

NAN = "NaN"

TYPE_ORDER = [
    "GSM_Rephrased",
    "GSM_AnsAug",
    "GSM_FOBAR",
    "GSM_SV",
    "MATH_AnsAug",
    "MATH_Rephrased",
    "MATH_FOBAR",
    "MATH_SV",
]

TYPE_TABLE_GROUPS = [
    ("GSM_Rephrased", lambda t: t == "GSM_Rephrased", "paraphrased GSM-style"),
    ("GSM_AnsAug", lambda t: t == "GSM_AnsAug", "answer-augmented variant"),
    ("GSM_FOBAR", lambda t: t == "GSM_FOBAR", "FOBAR perturbation from GSM sources"),
    ("GSM_SV", lambda t: t == "GSM_SV", "symbol/variable GSM-style variant"),
    ("MATH_Rephrased", lambda t: t == "MATH_Rephrased", "paraphrased MATH-style"),
    ("MATH_AnsAug", lambda t: t == "MATH_AnsAug", "answer-augmented MATH variant"),
    ("MATH_FOBAR", lambda t: t == "MATH_FOBAR", "FOBAR perturbation from MATH sources"),
    ("MATH_SV", lambda t: t == "MATH_SV", "symbol/variable MATH-style variant"),
]

SOURCE_GROUP_KEY_FIELDS = ["original_question_en", "original_question_vi", "query_vi"]

REPORT_RUNS = [
    ("v2", RESULTS_DIR / "4beams", "full-solution SFT"),
    ("v5", RESULTS_DIR / "v5", "GRPO-lite numeric"),
    ("v6", RESULTS_DIR / "v6", "answer-only LoRA"),
    ("v6_fast", RESULTS_DIR / "v6_fast", "answer-only LoRA r=16"),
    ("v7", RESULTS_DIR / "v7", "GRPO-lite reasoning"),
    ("v8", RESULTS_DIR / "v8_select_checkpoint" / "beam2_lr1e-3", "answer-only checkpoint select"),
    ("v9", RESULTS_DIR / "v9", "compact equations"),
    ("v10", RESULTS_DIR / "v10", "curriculum compact"),
    ("v11", RESULTS_DIR / "v11", "answer-only overlap-valid"),
    ("v12", RESULTS_DIR / "v12", "broad source retrieval"),
    ("v13", RESULTS_DIR / "v13", "type-gated retrieval"),
    ("v14 gate", RESULTS_DIR / "v14_gate_sweep", "retrieval gate sweep"),
    ("v14 FOBAR/SV", RESULTS_DIR / "v14_fobar_sv", "extended FOBAR/SV retrieval"),
    ("v14.5 v13", RESULTS_DIR / "v14_5_v13", "full-train valid-select v13 gate"),
    ("v14.5 gate", RESULTS_DIR / "v14_5_gate_sweep", "full-train valid-select gate sweep"),
    ("v15 ensemble", RESULTS_DIR / "v15_ensemble", "ensemble/ranker candidate selection"),
    ("v15 type-specific", RESULTS_DIR / "v15_type_specific", "SV/FOBAR expert safety route"),
]


def ensure_dirs() -> None:
    for path in (REPORT_DIR, TABLE_DIR, FIGURE_DIR, DATA_OUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        head = handle.read(1)
        handle.seek(0)
        return json.load(handle) if head == "[" else [json.loads(line) for line in handle if line.strip()]


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def norm_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"\s+", " ", text).strip()


def source_group_key(record: dict[str, Any]) -> str:
    """Mirror the source grouping used by the v11-v13 run artifacts."""
    for field in SOURCE_GROUP_KEY_FIELDS:
        value = norm_text(record.get(field))
        if value:
            return f"{field}:{value}"
    return f"source_id:{record.get('_source_id', record.get('id', 'unknown'))}"


def query_key(record: dict[str, Any]) -> str:
    value = norm_text(record.get("query_vi"))
    return value or f"query_id:{record.get('_source_id', record.get('id', 'unknown'))}"


def token_len(value: Any) -> int:
    text = clean_text(value)
    return len(text.split()) if text else 0


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return NAN
    if isinstance(value, float):
        if math.isnan(value):
            return NAN
        if abs(value - round(value)) < 1e-10:
            return str(int(round(value)))
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def pct(num: float | int | None, den: float | int | None = None, digits: int = 1) -> str:
    if num is None:
        return NAN
    value = 100.0 * num / den if den else float(num)
    return f"{value:.{digits}f}%"


def score_cell(summary: dict[str, Any] | None) -> str:
    if not summary:
        return NAN
    score = summary.get("score_10")
    raw = summary.get("raw_score")
    if score is None:
        return NAN
    return f"{score:.3f} ({raw}/10000)"


def compact_score(summary: dict[str, Any] | None) -> float | None:
    return summary.get("score_10") if summary else None


def delta_cell(left: dict[str, Any] | None, right: dict[str, Any] | None) -> str:
    if not left or not right:
        return NAN
    left_score = compact_score(left)
    right_score = compact_score(right)
    if left_score is None or right_score is None:
        return NAN
    return f"{left_score - right_score:+.3f}"


def pct_from_fraction(value: Any) -> str:
    return f"{100 * value:.1f}%" if isinstance(value, (int, float)) else NAN


def final_retrain_runtime(path: Path) -> str:
    info = load_json(path / "final_retrain_info.json", {})
    minutes = info.get("wall_minutes")
    return f"{minutes:.1f} min final retrain" if minutes is not None else NAN


def selected_checkpoint_label(path: Path) -> str:
    info = load_json(path / "selected_checkpoint_info.json", {})
    return info.get("label", NAN)


def hybrid_report(path: Path) -> dict[str, Any]:
    report = load_json(path / "hybrid_decision_report.json", {})
    if report:
        return report
    selected = load_json(path / "selected_checkpoint_info.json", {})
    if selected.get("decision_summary"):
        return selected["decision_summary"]
    checkpoint_report = load_json(path / "checkpoint_selection_report.json", {})
    if checkpoint_report.get("selected", {}).get("decision_summary"):
        return checkpoint_report["selected"]["decision_summary"]
    return {}


def retrieval_usage_cell(path: Path) -> str:
    return pct_from_fraction(hybrid_report(path).get("retrieval_used_pct"))


def output_changed_cell(path: Path) -> str:
    report = hybrid_report(path)
    changed = report.get("output_changed_count")
    n = report.get("n")
    if changed is None:
        return NAN
    return f"{changed}/{n}" if n else str(changed)


def summary_from_report(path: Path) -> dict[str, Any] | None:
    data = load_json(path)
    if not data:
        return None
    summary = data.get("summary") if isinstance(data, dict) else None
    if not summary:
        return None
    buckets = summary.get("buckets", {})
    return {
        "n": summary.get("n"),
        "raw_score": summary.get("raw_score"),
        "max_raw_score": summary.get("max_raw_score"),
        "score_10": summary.get("score_10"),
        "extractable": summary.get("extractable"),
        "numeric_pairs": summary.get("numeric_pairs"),
        "exact10": buckets.get("10", 0),
        "bucket_5": buckets.get("5", 0),
        "bucket_1": buckets.get("1", 0),
        "bucket_0": buckets.get("0", 0),
        "rel_error_mean": summary.get("rel_error_mean"),
    }


def summary_from_dict(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    buckets = summary.get("buckets", {})
    return {
        "n": summary.get("n"),
        "raw_score": summary.get("raw_score"),
        "max_raw_score": summary.get("max_raw_score"),
        "score_10": summary.get("score_10"),
        "extractable": summary.get("extractable"),
        "numeric_pairs": summary.get("numeric_pairs"),
        "exact10": buckets.get("10", 0),
        "bucket_5": buckets.get("5", 0),
        "bucket_1": buckets.get("1", 0),
        "bucket_0": buckets.get("0", 0),
        "rel_error_mean": summary.get("rel_error_mean"),
    }


def output_stats(path: Path) -> dict[str, Any] | None:
    rows = load_json(path)
    if not rows:
        return None
    texts = [clean_text(row.get("model_output", "")) for row in rows]
    lengths = [len(text) for text in texts]
    token_lengths = [token_len(text) for text in texts]
    return {
        "n": len(texts),
        "mean_chars": mean(lengths),
        "median_chars": median(lengths),
        "mean_tokens": mean(token_lengths),
        "median_tokens": median(token_lengths),
        "short_single_line_rate": sum(1 for text in texts if "\n" not in text and len(text) <= 40) / len(texts),
    }


def extract_manifest_config(path: Path) -> dict[str, Any]:
    data = load_json(path, {})
    return data.get("config", {}) if isinstance(data, dict) else {}


def write_csv(name: str, rows: list[dict[str, Any]]) -> Path:
    path = TABLE_DIR / f"{name}.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})
    return path


def escape_md(value: Any) -> str:
    text = fmt(value).replace("\n", "<br>")
    return text.replace("|", "\\|")


def rows_to_md(rows: list[dict[str, Any]], caption: str) -> str:
    if not rows:
        return caption + "\n"
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_md(row.get(column)) for column in columns) + " |")
    lines.append("")
    lines.append(f"Caption: {caption}")
    lines.append("")
    return "\n".join(lines)


def write_table(name: str, rows: list[dict[str, Any]], caption: str) -> str:
    write_csv(name, rows)
    md = rows_to_md(rows, caption)
    (TABLE_DIR / f"{name}.md").write_text(md, encoding="utf-8")
    return md


def write_data_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_source_index(train: list[dict[str, Any]]) -> dict[str, Any]:
    source_queries: dict[str, list[str]] = defaultdict(list)
    source_answer_nums: dict[str, list[float | None]] = defaultdict(list)
    query_set: set[str] = set()
    query_set_exact: set[str] = set()

    for record in train:
        source = source_group_key(record)
        query_norm = query_key(record)
        query_exact = clean_text(record.get("query_vi"))
        if source:
            source_queries[source].append(query_norm)
            source_answer_nums[source].append(extract_gold(record)[1])
        if query_norm:
            query_set.add(query_norm)
        if query_exact:
            query_set_exact.add(query_exact)

    return {
        "source_queries": source_queries,
        "source_answer_nums": source_answer_nums,
        "query_set": query_set,
        "query_set_exact": query_set_exact,
    }


def same_source_class(train_nums: list[float | None], valid_num: float | None) -> str:
    if not train_nums:
        return "source_unseen"
    numeric_train = [value for value in train_nums if value is not None]
    if valid_num is None or not numeric_train:
        return "source_seen_missing_numeric"
    if any(abs(value - valid_num) <= 1e-9 for value in numeric_train):
        return "source_seen_same_answer"
    return "source_seen_conflicting_answer"


def build_valid_overlap_rows(valid: list[dict[str, Any]], source_index: dict[str, Any]) -> list[dict[str, Any]]:
    source_queries = source_index["source_queries"]
    source_answer_nums = source_index["source_answer_nums"]
    query_set = source_index["query_set"]
    query_set_exact = source_index["query_set_exact"]

    rows = []
    for idx, record in enumerate(valid):
        source = source_group_key(record)
        query_norm = query_key(record)
        query_exact = clean_text(record.get("query_vi"))
        gold_answer, valid_num = extract_gold(record)
        same_source_queries = source_queries.get(source, [])
        if same_source_queries:
            max_sim = max(SequenceMatcher(None, query_norm, train_query).ratio() for train_query in same_source_queries)
        else:
            max_sim = 0.0
        rows.append(
            {
                "id": idx,
                "type": record.get("type") or "UNKNOWN",
                "exact_query_overlap": int(query_exact in query_set_exact),
                "normalized_query_overlap": int(query_norm in query_set),
                "max_same_source_query_similarity": max_sim,
                "source_group_overlap": int(source in source_queries),
                "source_overlap_class": same_source_class(source_answer_nums.get(source, []), valid_num),
                "gold_answer": gold_answer,
                "gold_num": valid_num,
            }
        )
    return rows


def build_table_02(train: list[dict[str, Any]], valid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    combined = train + valid
    for label, predicate, note in TYPE_TABLE_GROUPS:
        train_rows = [row for row in train if predicate(row.get("type") or "")]
        valid_rows = [row for row in valid if predicate(row.get("type") or "")]
        all_rows = [row for row in combined if predicate(row.get("type") or "")]
        source_groups = {source_group_key(row) for row in all_rows}
        rows.append(
            {
                "Type": label,
                "#Train": len(train_rows),
                "#Valid": len(valid_rows),
                "#Unique source groups": len(source_groups),
                "Avg query length": round(mean([token_len(row.get("query_vi")) for row in all_rows]), 2) if all_rows else None,
                "Avg answer length": round(mean([token_len(row.get("response_vi")) for row in all_rows]), 2) if all_rows else None,
                "Notes": note,
            }
        )
    return rows


def build_table_04() -> list[dict[str, Any]]:
    v9_cfg = extract_manifest_config(RESULTS_DIR / "v9" / "v9_compact_equation_manifest.json")
    v10_cfg = extract_manifest_config(RESULTS_DIR / "v10" / "v10_curriculum_compact_manifest.json")
    v11_cfg = extract_manifest_config(RESULTS_DIR / "v11" / "v11_answer_only_overlap_valid_manifest.json")
    v12_cfg = extract_manifest_config(RESULTS_DIR / "v12" / "v12_hybrid_retrieval_overlap_valid_manifest.json")
    v13_cfg = extract_manifest_config(RESULTS_DIR / "v13" / "v13_type_gated_hybrid_full_retrain_manifest.json")
    v14_gate_cfg = extract_manifest_config(RESULTS_DIR / "v14_gate_sweep" / "v14_retrieval_gate_sweep_manifest.json")
    v14_fobar_cfg = extract_manifest_config(RESULTS_DIR / "v14_fobar_sv" / "v14_fobar_sv_extended_retrieval_manifest.json")
    v145_v13_cfg = extract_manifest_config(RESULTS_DIR / "v14_5_v13" / "v14_5_v13_full_train_valid_select_manifest.json")
    v145_gate_cfg = extract_manifest_config(RESULTS_DIR / "v14_5_gate_sweep" / "v14_5_gate_sweep_full_train_valid_select_manifest.json")
    v15_ensemble_cfg = extract_manifest_config(RESULTS_DIR / "v15_ensemble" / "v15_ensemble_ranker_manifest.json")
    v15_expert_cfg = extract_manifest_config(RESULTS_DIR / "v15_type_specific" / "v15_type_expert_sv_fobar_manifest.json")
    v8_cfg = extract_manifest_config(RESULTS_DIR / "v8_select_checkpoint" / "beam2_lr1e-3" / "v8_manifest.json")
    return [
        {
            "Version group": "v2",
            "Target style": "full solution",
            "LoRA r": "full/standard SFT",
            "LR": NAN,
            "Epochs": NAN,
            "Max length": 768,
            "Decode": "beam4",
            "Validation for selection": "valid",
        },
        {
            "Version group": "v3",
            "Target style": "answer-only -> compact reasoning",
            "LoRA r": "SFT",
            "LR": NAN,
            "Epochs": NAN,
            "Max length": "256/384",
            "Decode": "beam2",
            "Validation for selection": "valid",
        },
        {
            "Version group": "v4",
            "Target style": "KD compact reasoning + LoRA",
            "LoRA r": 16,
            "LR": NAN,
            "Epochs": "2 stages",
            "Max length": NAN,
            "Decode": "beam2",
            "Validation for selection": "valid",
        },
        {
            "Version group": "v5/v7",
            "Target style": "LoRA + GRPO-lite",
            "LoRA r": 16,
            "LR": NAN,
            "Epochs": "staged",
            "Max length": NAN,
            "Decode": "beam2",
            "Validation for selection": "valid",
        },
        {
            "Version group": "v6/v6_fast",
            "Target style": "answer-only LoRA",
            "LoRA r": "16/32",
            "LR": NAN,
            "Epochs": "3 / 2.25",
            "Max length": 256,
            "Decode": "beam2",
            "Validation for selection": "valid",
        },
        {
            "Version group": "v8",
            "Target style": "answer-only + checkpoint selection",
            "LoRA r": v8_cfg.get("lora_r", 32),
            "LR": v8_cfg.get("stage_a_lr", NAN),
            "Epochs": v8_cfg.get("stage_a_epochs", NAN),
            "Max length": v8_cfg.get("max_length_stage_a", 256),
            "Decode": f"beam{v8_cfg.get('num_beams', 2)}",
            "Validation for selection": "valid",
        },
        {
            "Version group": "v9",
            "Target style": "compact-equation + source-disjoint selection",
            "LoRA r": v9_cfg.get("lora_r", NAN),
            "LR": v9_cfg.get("stage_a_lr", NAN),
            "Epochs": v9_cfg.get("stage_a_epochs", NAN),
            "Max length": v9_cfg.get("max_length_stage_a", NAN),
            "Decode": f"beam{v9_cfg.get('num_beams', 2)}",
            "Validation for selection": "source-disjoint",
        },
        {
            "Version group": "v10",
            "Target style": "curriculum compact-equation",
            "LoRA r": v10_cfg.get("lora_r", NAN),
            "LR": "staged",
            "Epochs": "2+4+2",
            "Max length": v10_cfg.get("max_length_stage_a", NAN),
            "Decode": f"beam{v10_cfg.get('num_beams', 2)}",
            "Validation for selection": "source-disjoint",
        },
        {
            "Version group": "v11",
            "Target style": "answer-only + overlap-valid",
            "LoRA r": v11_cfg.get("lora_r", NAN),
            "LR": v11_cfg.get("stage_a_lr", NAN),
            "Epochs": v11_cfg.get("stage_a_epochs", NAN),
            "Max length": v11_cfg.get("max_length_stage_a", NAN),
            "Decode": f"beam{v11_cfg.get('num_beams', 2)}",
            "Validation for selection": "source-overlap query-disjoint",
        },
        {
            "Version group": "v12",
            "Target style": "answer-only + broad source retrieval",
            "LoRA r": v12_cfg.get("lora_r", NAN),
            "LR": v12_cfg.get("stage_a_lr", NAN),
            "Epochs": v12_cfg.get("stage_a_epochs", NAN),
            "Max length": v12_cfg.get("max_length_stage_a", NAN),
            "Decode": f"beam{v12_cfg.get('num_beams', 2)}",
            "Validation for selection": "source-overlap query-disjoint",
        },
        {
            "Version group": "v13",
            "Target style": "type-gated retrieval + full retrain",
            "LoRA r": v13_cfg.get("lora_r", NAN),
            "LR": v13_cfg.get("stage_a_lr", NAN),
            "Epochs": v13_cfg.get("stage_a_epochs", NAN),
            "Max length": v13_cfg.get("max_length_stage_a", NAN),
            "Decode": f"beam{v13_cfg.get('num_beams', 2)}",
            "Validation for selection": "source-overlap query-disjoint",
        },
        {
            "Version group": "v14 gate sweep",
            "Target style": "retrieval gate sweep + full retrain",
            "LoRA r": v14_gate_cfg.get("lora_r", NAN),
            "LR": v14_gate_cfg.get("stage_a_lr", NAN),
            "Epochs": v14_gate_cfg.get("stage_a_epochs", NAN),
            "Max length": v14_gate_cfg.get("max_length_stage_a", NAN),
            "Decode": f"beam{v14_gate_cfg.get('num_beams', 2)}",
            "Validation for selection": "source-overlap query-disjoint",
        },
        {
            "Version group": "v14 FOBAR/SV",
            "Target style": "extended retrieval for FOBAR/SV",
            "LoRA r": v14_fobar_cfg.get("lora_r", NAN),
            "LR": v14_fobar_cfg.get("stage_a_lr", NAN),
            "Epochs": v14_fobar_cfg.get("stage_a_epochs", NAN),
            "Max length": v14_fobar_cfg.get("max_length_stage_a", NAN),
            "Decode": f"beam{v14_fobar_cfg.get('num_beams', 2)}",
            "Validation for selection": "source-overlap query-disjoint",
        },
        {
            "Version group": "v14.5 v13",
            "Target style": "full train + valid-select V13 gate",
            "LoRA r": v145_v13_cfg.get("lora_r", NAN),
            "LR": v145_v13_cfg.get("stage_a_lr", NAN),
            "Epochs": v145_v13_cfg.get("stage_a_epochs", NAN),
            "Max length": v145_v13_cfg.get("max_length_stage_a", NAN),
            "Decode": f"beam{v145_v13_cfg.get('num_beams', 2)}",
            "Validation for selection": "valid.json",
        },
        {
            "Version group": "v14.5 gate sweep",
            "Target style": "full train + valid-select gate sweep",
            "LoRA r": v145_gate_cfg.get("lora_r", NAN),
            "LR": v145_gate_cfg.get("stage_a_lr", NAN),
            "Epochs": v145_gate_cfg.get("stage_a_epochs", NAN),
            "Max length": v145_gate_cfg.get("max_length_stage_a", NAN),
            "Decode": f"beam{v145_gate_cfg.get('num_beams', 2)}",
            "Validation for selection": "valid.json",
        },
        {
            "Version group": "v15 ensemble ranker",
            "Target style": "answer-only candidates + ExtraTrees ranker",
            "LoRA r": v15_ensemble_cfg.get("lora_r", NAN),
            "LR": v15_ensemble_cfg.get("stage_a_lr", NAN),
            "Epochs": v15_ensemble_cfg.get("stage_a_epochs", NAN),
            "Max length": v15_ensemble_cfg.get("max_length_stage_a", NAN),
            "Decode": f"beam{v15_ensemble_cfg.get('num_beams', 2)}",
            "Validation for selection": "valid.json ranker labels",
        },
        {
            "Version group": "v15 type-specific expert",
            "Target style": "SV/FOBAR expert with safe routing",
            "LoRA r": v15_expert_cfg.get("lora_r", NAN),
            "LR": v15_expert_cfg.get("stage_a_lr", NAN),
            "Epochs": v15_expert_cfg.get("stage_a_epochs", NAN),
            "Max length": v15_expert_cfg.get("max_length_stage_a", NAN),
            "Decode": f"beam{v15_expert_cfg.get('num_beams', 2)}",
            "Validation for selection": "overlap-valid + valid.json route check",
        },
        {
            "Version group": "v16 legal",
            "Target style": "query_vi-only answer/retrieval candidates",
            "LoRA r": 32,
            "LR": NAN,
            "Epochs": NAN,
            "Max length": 256,
            "Decode": "beam2",
            "Validation for selection": "internal train split; no result yet",
        },
    ]


def build_table_05() -> list[dict[str, Any]]:
    fair = load_json(AUDIT_DIR / "fair_comparison.json", {})
    v2 = None
    for row in fair.get("rows", []):
        if row.get("run_id") == "v2_beam4":
            v2 = row
            break
    score = NAN
    extractability = NAN
    if v2:
        score = f"{v2.get('standard_score_10'):.3f} ({v2.get('standard_raw_score')}/10000; legacy {v2.get('legacy_raw_score')})"
        extractability = f"{v2.get('extractable')}/1000"
    return [
        {
            "Model": "v2",
            "Target": "full response",
            "Max length": 768,
            "Decode": "beam4",
            "Valid score": score,
            "Extractability": extractability,
            "Runtime": NAN,
        }
    ]


def build_table_06() -> list[dict[str, Any]]:
    rows = [
        {
            "Version": "v3",
            "Target": "answer -> compact NL",
            "Selection split": "valid",
            "Valid score": NAN,
            "Source-disjoint score": NAN,
            "Avg output length": NAN,
            "Reasoning consistency": NAN,
        }
    ]
    for version, target in [("v9", "compact equations"), ("v10", "curriculum compact")]:
        valid = summary_from_report(RESULTS_DIR / version / "valid_report.json")
        source = summary_from_report(RESULTS_DIR / version / "source_valid_report.json")
        stats = output_stats(RESULTS_DIR / version / "valid_output.json")
        rows.append(
            {
                "Version": version,
                "Target": target,
                "Selection split": "source-disjoint",
                "Valid score": score_cell(valid),
                "Source-disjoint score": score_cell(source),
                "Avg output length": round(stats["mean_chars"], 1) if stats else NAN,
                "Reasoning consistency": f"extractable {valid['extractable']}/{valid['n']}" if valid else NAN,
            }
        )
    return rows


def build_table_08() -> list[dict[str, Any]]:
    specs = [
        ("v6", "no type", "r=32", "3", "final epoch", RESULTS_DIR / "v6"),
        ("v6_fast", "no type", "r=16", "2.25", "final epoch", RESULTS_DIR / "v6_fast"),
        ("v8", "no type", "r=32", "8", "valid checkpoint", RESULTS_DIR / "v8_select_checkpoint" / "beam2_lr1e-3"),
        ("v11", "type-aware", "r=32", "8", "overlap-valid checkpoint", RESULTS_DIR / "v11"),
    ]
    rows = []
    for version, prompt, lora, epochs, selection, path in specs:
        valid = summary_from_report(path / "valid_report.json")
        overlap = summary_from_report(path / "overlap_valid_report.json")
        rows.append(
            {
                "Version": version,
                "Prompt": prompt,
                "LoRA config": lora,
                "Epochs": epochs,
                "Selection": selection,
                "Valid score": score_cell(valid),
                "Overlap-valid score": score_cell(overlap),
                "Source-disjoint score": NAN,
            }
        )
    return rows


def build_table_09() -> list[dict[str, Any]]:
    rows = []
    for version, components, penalize, failure in [
        ("v5", "numeric + anchor", "no", "score did not improve beyond SFT baseline"),
        ("v7", "numeric + anchor + reasoning", "yes", "lower extractability and low answer reward"),
    ]:
        score = summary_from_report(RESULTS_DIR / version / "valid_report.json")
        reward = load_json(RESULTS_DIR / version / "rl_reward_summary.json", {})
        runtime = f"{reward.get('wall_min'):.1f} min RL" if reward.get("wall_min") is not None else NAN
        rows.append(
            {
                "Version": version,
                "Reward components": components,
                "Penalize answer-only?": penalize,
                "Score": score_cell(score),
                "Runtime": runtime,
                "Failure mode": failure,
            }
        )
    return rows


def build_table_10() -> list[dict[str, Any]]:
    specs = [
        ("v11", "no", "none", "none", "yes", RESULTS_DIR / "v11", "0%"),
        ("v12", "yes", "broad", "majority frac", "yes", RESULTS_DIR / "v12", None),
        ("v13", "yes", "Rephrased/AnsAug", "type gate", "yes", RESULTS_DIR / "v13", None),
        ("v14 sweep", "yes", "selected types", "tuned thresholds", "yes", RESULTS_DIR / "v14_gate_sweep", None),
        ("v14 FOBAR/SV", "yes", "extended", "strict nearest", "yes", RESULTS_DIR / "v14_fobar_sv", None),
        ("v14.5 v13", "yes", "Rephrased/AnsAug", "V13 gate + valid select", "yes", RESULTS_DIR / "v14_5_v13", None),
        ("v14.5 gate", "yes", "selected types", "gate sweep + valid select", "yes", RESULTS_DIR / "v14_5_gate_sweep", None),
        ("v15 ensemble", "mixed", "32 model/hybrid candidates", "ExtraTrees ranker", "yes", RESULTS_DIR / "v15_ensemble", "ranker"),
        ("v15 type-specific", "yes", "Rephrased/AnsAug; expert not routed", "safe route check", "yes", RESULTS_DIR / "v15_type_specific", None),
    ]
    rows = []
    for version, retrieval, allowed, gate, fallback, path, forced_usage in specs:
        valid = summary_from_report(path / "valid_report.json") if path else None
        overlap = summary_from_report(path / "overlap_valid_report.json") if path else None
        hybrid = load_json(path / "hybrid_decision_report.json", {}) if path else {}
        usage = forced_usage
        if usage is None:
            usage = f"{100 * hybrid.get('retrieval_used_pct'):.1f}%" if hybrid.get("retrieval_used_pct") is not None else NAN
        rows.append(
            {
                "Version": version,
                "Retrieval": retrieval,
                "Allowed types": allowed,
                "Gate": gate,
                "Model fallback": fallback,
                "Official-valid score": score_cell(valid),
                "Overlap-valid score": score_cell(overlap),
                "Source-disjoint score": NAN,
                "Retrieval usage": usage,
            }
        )
    return rows


def build_table_11() -> list[dict[str, Any]]:
    v10_manifest = load_json(RESULTS_DIR / "v10" / "v10_curriculum_compact_manifest.json", {})
    v10_runtime = None
    if v10_manifest.get("stage_train_times"):
        v10_runtime = sum(v10_manifest["stage_train_times"].values()) / 60
    return [
        {
            "Candidate": "Best answer-only",
            "Official valid score": score_cell(summary_from_report(RESULTS_DIR / "v11" / "valid_report.json")),
            "Source-disjoint score": NAN,
            "Overlap-valid score": score_cell(summary_from_report(RESULTS_DIR / "v11" / "overlap_valid_report.json")),
            "Runtime": NAN,
            "Reasoning quality": "low/medium",
            "Final choice": "no",
        },
        {
            "Candidate": "Best compact reasoning",
            "Official valid score": score_cell(summary_from_report(RESULTS_DIR / "v10" / "valid_report.json")),
            "Source-disjoint score": score_cell(summary_from_report(RESULTS_DIR / "v10" / "source_valid_report.json")),
            "Overlap-valid score": NAN,
            "Runtime": f"{v10_runtime:.1f} min train" if v10_runtime is not None else NAN,
            "Reasoning quality": "medium",
            "Final choice": "no",
        },
        {
            "Candidate": "Best hybrid retrieval (v13)",
            "Official valid score": score_cell(summary_from_report(RESULTS_DIR / "v13" / "valid_report.json")),
            "Source-disjoint score": NAN,
            "Overlap-valid score": score_cell(summary_from_report(RESULTS_DIR / "v13" / "overlap_valid_report.json")),
            "Runtime": final_retrain_runtime(RESULTS_DIR / "v13"),
            "Reasoning quality": "retrieval-dependent",
            "Final choice": "strong baseline; no longer best score",
        },
        {
            "Candidate": "Best full-train valid-select (v14.5)",
            "Official valid score": score_cell(summary_from_report(RESULTS_DIR / "v14_5_v13" / "valid_report.json")),
            "Source-disjoint score": NAN,
            "Overlap-valid score": NAN,
            "Runtime": final_retrain_runtime(RESULTS_DIR / "v14_5_v13"),
            "Reasoning quality": "retrieval-dependent",
            "Final choice": "no; below v13/v15 despite valid selection",
        },
        {
            "Candidate": "Best diagnostic ensemble (v15)",
            "Official valid score": score_cell(summary_from_report(RESULTS_DIR / "v15_ensemble" / "valid_report.json")),
            "Source-disjoint score": NAN,
            "Overlap-valid score": NAN,
            "Runtime": final_retrain_runtime(RESULTS_DIR / "v15_ensemble"),
            "Reasoning quality": "answer-selection/retrieval-dependent",
            "Final choice": "best diagnostic score; not submission-safe as-is",
        },
        {
            "Candidate": "v14 gate sweep",
            "Official valid score": score_cell(summary_from_report(RESULTS_DIR / "v14_gate_sweep" / "valid_report.json")),
            "Source-disjoint score": NAN,
            "Overlap-valid score": score_cell(summary_from_report(RESULTS_DIR / "v14_gate_sweep" / "overlap_valid_report.json")),
            "Runtime": final_retrain_runtime(RESULTS_DIR / "v14_gate_sweep"),
            "Reasoning quality": "retrieval-dependent",
            "Final choice": "no; below v13 on official and overlap-valid",
        },
        {
            "Candidate": "v14 FOBAR/SV retrieval",
            "Official valid score": score_cell(summary_from_report(RESULTS_DIR / "v14_fobar_sv" / "valid_report.json")),
            "Source-disjoint score": NAN,
            "Overlap-valid score": score_cell(summary_from_report(RESULTS_DIR / "v14_fobar_sv" / "overlap_valid_report.json")),
            "Runtime": final_retrain_runtime(RESULTS_DIR / "v14_fobar_sv"),
            "Reasoning quality": "retrieval-dependent; weak FOBAR/SV",
            "Final choice": "no; extended retrieval hurts target variants",
        },
    ]


def pick_error_examples(valid_overlap_rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = load_json(RESULTS_DIR / "v13" / "valid_report.json", {})
    rows = report.get("rows", [])
    overlap_by_id = {row["id"]: row for row in valid_overlap_rows}
    pred_counts = Counter(str(row.get("pred_answer")) for row in rows if row.get("pred_answer") is not None)
    common_preds = {value for value, _count in pred_counts.most_common(5)}

    examples: dict[str, Any] = {}
    for row in rows:
        if not row.get("extractable") and "Parse failure" not in examples:
            examples["Parse failure"] = row["id"]
        if row.get("score") in {1, 5} and row.get("pred_num") is not None and row.get("gold_num") is not None and "Arithmetic error" not in examples:
            examples["Arithmetic error"] = row["id"]
        if row.get("score") == 0 and row.get("pred_num") is not None and row.get("gold_num") is not None and row.get("type") in {"GSM_Rephrased", "GSM_AnsAug", "MATH_Rephrased", "MATH_AnsAug"} and "Relation error" not in examples:
            examples["Relation error"] = row["id"]
        if row.get("score") == 0 and str(row.get("pred_answer")) in common_preds and "Spurious shortcut" not in examples:
            examples["Spurious shortcut"] = row["id"]
        overlap_class = overlap_by_id.get(row["id"], {}).get("source_overlap_class")
        if row.get("score") == 0 and overlap_class == "source_seen_conflicting_answer" and "Source-memory failure" not in examples:
            examples["Source-memory failure"] = row["id"]
        if row.get("score") == 0 and (row.get("type", "").endswith("_FOBAR") or row.get("type", "").endswith("_SV")) and "Type confusion" not in examples:
            examples["Type confusion"] = row["id"]
    return examples


def build_table_12(valid_overlap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = pick_error_examples(valid_overlap_rows)
    return [
        {
            "Error type": "Parse failure",
            "Description": "No final number or wrong final-answer anchor",
            "Example id": examples.get("Parse failure", NAN),
            "Diagnostic": "output format",
        },
        {
            "Error type": "Arithmetic error",
            "Description": "Correct relation pattern but wrong numeric computation",
            "Example id": examples.get("Arithmetic error", NAN),
            "Diagnostic": "equation check",
        },
        {
            "Error type": "Relation error",
            "Description": "Misread relation in the word problem",
            "Example id": examples.get("Relation error", NAN),
            "Diagnostic": "relational plan",
        },
        {
            "Error type": "Spurious shortcut",
            "Description": "Predicted a frequent answer pattern",
            "Example id": examples.get("Spurious shortcut", NAN),
            "Diagnostic": "counterfactual",
        },
        {
            "Error type": "Correct answer, wrong reasoning",
            "Description": "Answer is correct but rationale is unsupported",
            "Example id": NAN,
            "Diagnostic": "manual rationalization audit",
        },
        {
            "Error type": "Source-memory failure",
            "Description": "Same source group has conflicting train answer",
            "Example id": examples.get("Source-memory failure", NAN),
            "Diagnostic": "retrieval/source audit",
        },
        {
            "Error type": "Type confusion",
            "Description": "GSM/MATH or FOBAR/SV variant confused",
            "Example id": examples.get("Type confusion", NAN),
            "Diagnostic": "type-wise analysis",
        },
    ]


def report_by_type(path: Path) -> dict[str, dict[str, Any]]:
    report = load_json(path, {})
    return report.get("by_type", {}) if isinstance(report, dict) else {}


def by_type_score(path: Path, type_name: str) -> float | None:
    item = report_by_type(path).get(type_name)
    return item.get("score_10") if item else None


def by_type_raw(path: Path, type_name: str) -> int | None:
    item = report_by_type(path).get(type_name)
    return item.get("raw_score") if item else None


def retrieval_used_by_type(path: Path, type_name: str) -> dict[str, Any]:
    return hybrid_report(path).get("retrieval_used_by_type", {}).get(type_name, {})


def build_table_13_v14_run_comparison() -> list[dict[str, Any]]:
    specs = [
        ("v13 type-gated", RESULTS_DIR / "v13", "Rephrased/AnsAug type gate"),
        ("v14 gate sweep", RESULTS_DIR / "v14_gate_sweep", "selected thresholds"),
        ("v14 FOBAR/SV", RESULTS_DIR / "v14_fobar_sv", "extended strict nearest"),
    ]
    rows = []
    for name, path, gate in specs:
        valid = summary_from_report(path / "valid_report.json")
        model = summary_from_report(path / "model_valid_report.json")
        overlap = summary_from_report(path / "overlap_valid_report.json")
        hybrid = hybrid_report(path)
        rows.append(
            {
                "Run": name,
                "Gate policy": gate,
                "Selected checkpoint": selected_checkpoint_label(path),
                "Model-only valid": score_cell(model),
                "Hybrid valid": score_cell(valid),
                "Hybrid gain": delta_cell(valid, model),
                "Overlap-valid": score_cell(overlap),
                "Retrieval usage": retrieval_usage_cell(path),
                "Changed outputs": output_changed_cell(path),
                "Final retrain runtime": final_retrain_runtime(path),
                "Fallback reasons": "; ".join(f"{k}={v}" for k, v in sorted(hybrid.get("fallback_reason_counts", {}).items())) or NAN,
            }
        )
    return rows


def build_table_14_v14_type_gains() -> list[dict[str, Any]]:
    rows = []
    for type_name in TYPE_ORDER:
        row: dict[str, Any] = {"Type": type_name}
        for prefix, path in [
            ("v14 sweep", RESULTS_DIR / "v14_gate_sweep"),
            ("v14 FOBAR/SV", RESULTS_DIR / "v14_fobar_sv"),
        ]:
            hybrid_score = by_type_score(path / "valid_report.json", type_name)
            model_score = by_type_score(path / "model_valid_report.json", type_name)
            usage = retrieval_used_by_type(path, type_name)
            usage_pct = 100 * usage.get("retrieval_used", 0) / usage["n"] if usage.get("n") else None
            row[f"{prefix} model"] = round(model_score, 3) if model_score is not None else NAN
            row[f"{prefix} hybrid"] = round(hybrid_score, 3) if hybrid_score is not None else NAN
            row[f"{prefix} gain"] = round(hybrid_score - model_score, 3) if hybrid_score is not None and model_score is not None else NAN
            row[f"{prefix} retrieval"] = f"{usage_pct:.1f}%" if usage_pct is not None else NAN
        rows.append(row)
    return rows


def build_table_15_v14_gate_config() -> list[dict[str, Any]]:
    rows = []
    for run_name, path in [
        ("v14 sweep", RESULTS_DIR / "v14_gate_sweep"),
        ("v14 FOBAR/SV", RESULTS_DIR / "v14_fobar_sv"),
    ]:
        config = load_json(path / "selected_retrieval_gate_config.json", {})
        for type_name in TYPE_ORDER:
            gate = config.get(type_name, {})
            usage = retrieval_used_by_type(path, type_name)
            usage_pct = 100 * usage.get("retrieval_used", 0) / usage["n"] if usage.get("n") else None
            rows.append(
                {
                    "Run": run_name,
                    "Type": type_name,
                    "Enabled": gate.get("enabled", bool(gate)),
                    "Strategy": gate.get("strategy", NAN),
                    "Min majority": gate.get("min_majority_frac", NAN),
                    "Min margin": gate.get("min_margin", NAN),
                    "Min nearest Jaccard": gate.get("min_nearest_jaccard", NAN),
                    "Typed pool?": gate.get("require_typed_pool", False),
                    "Agreement gate?": gate.get("use_model_agreement_gate", False),
                    "Official retrieval usage": f"{usage_pct:.1f}%" if usage_pct is not None else NAN,
                    "Official hybrid score": round(by_type_score(path / "valid_report.json", type_name), 3)
                    if by_type_score(path / "valid_report.json", type_name) is not None
                    else NAN,
                }
            )
    return rows


def v2_standard_summary() -> dict[str, Any] | None:
    fair = load_json(AUDIT_DIR / "fair_comparison.json", {})
    for row in fair.get("rows", []):
        if row.get("run_id") == "v2_beam4":
            return {
                "n": 1000,
                "raw_score": row.get("standard_raw_score"),
                "max_raw_score": 10000,
                "score_10": row.get("standard_score_10"),
                "extractable": row.get("extractable"),
                "numeric_pairs": row.get("extractable"),
                "exact10": NAN,
                "bucket_5": NAN,
                "bucket_1": NAN,
                "bucket_0": NAN,
                "rel_error_mean": None,
            }
    return None


def run_valid_summary(run_name: str, path: Path) -> dict[str, Any] | None:
    if run_name == "v2":
        return v2_standard_summary()
    return summary_from_report(path / "valid_report.json")


def build_table_16_leaderboard() -> list[dict[str, Any]]:
    rows = [
        {
            "Version": "v3",
            "Family": "answer-only -> compact NL",
            "Official-valid score": NAN,
            "Exact10": NAN,
            "Zero": NAN,
            "Extractable": NAN,
            "Interpretation": "artifact not recoverable",
        },
        {
            "Version": "v4",
            "Family": "KD compact reasoning + LoRA",
            "Official-valid score": NAN,
            "Exact10": NAN,
            "Zero": NAN,
            "Extractable": NAN,
            "Interpretation": "artifact not recoverable",
        },
    ]
    for name, path, family in REPORT_RUNS:
        summary = run_valid_summary(name, path)
        if not summary:
            continue
        rows.append(
            {
                "Version": name,
                "Family": family,
                "Official-valid score": score_cell(summary),
                "Exact10": summary.get("exact10", NAN),
                "Zero": summary.get("bucket_0", NAN),
                "Extractable": summary.get("extractable", NAN),
                "Interpretation": "best diagnostic score" if name == "v15 ensemble" else "valid artifact",
            }
        )
    return rows


def build_table_17_model_hybrid_gain() -> list[dict[str, Any]]:
    specs = [
        ("v12", RESULTS_DIR / "v12", "broad source retrieval"),
        ("v13", RESULTS_DIR / "v13", "type-gated retrieval"),
        ("v14 gate sweep", RESULTS_DIR / "v14_gate_sweep", "tuned retrieval gates"),
        ("v14 FOBAR/SV", RESULTS_DIR / "v14_fobar_sv", "extended strict nearest retrieval"),
        ("v14.5 v13", RESULTS_DIR / "v14_5_v13", "valid-select V13 gate"),
        ("v14.5 gate", RESULTS_DIR / "v14_5_gate_sweep", "valid-select gate sweep"),
        ("v15 ensemble", RESULTS_DIR / "v15_ensemble", "ExtraTrees ranker over 32 candidates"),
        ("v15 type-specific", RESULTS_DIR / "v15_type_specific", "expert route disabled by validation"),
    ]
    rows = []
    for name, path, mechanism in specs:
        model = summary_from_report(path / "model_valid_report.json")
        final = summary_from_report(path / "valid_report.json")
        selected = summary_from_report(path / "selected_valid_report.json")
        output = output_stats(path / "valid_output.json")
        rows.append(
            {
                "Run": name,
                "Mechanism": mechanism,
                "Model-only score": score_cell(model),
                "Selected single-candidate score": score_cell(selected) if selected else NAN,
                "Final score": score_cell(final),
                "Final - model": delta_cell(final, model),
                "Final - selected": delta_cell(final, selected) if selected else NAN,
                "Median output chars": round(output["median_chars"], 1) if output else NAN,
                "Short single-line rate": pct_from_fraction(output["short_single_line_rate"]) if output else NAN,
            }
        )
    return rows


def build_table_18_v15_diagnostics() -> list[dict[str, Any]]:
    ranker = load_json(RESULTS_DIR / "v15_ensemble" / "ensemble_ranker_report.json", {})
    expert = load_json(RESULTS_DIR / "v15_type_specific" / "expert_selection_report.json", {})
    choice_counts = Counter(ranker.get("choice_counts", {}))
    top_choices = "; ".join(f"{name}={count}" for name, count in choice_counts.most_common(5)) or NAN
    best_by_type = expert.get("best_by_type", {})
    routed = expert.get("route_types", [])
    best_expert_raw: dict[str, tuple[int, str]] = {}
    for entry in expert.get("entries", []):
        label = entry.get("label", NAN)
        for type_name, item in entry.get("by_type", {}).items():
            raw = item.get("raw_score")
            if raw is None:
                continue
            if type_name not in best_expert_raw or raw > best_expert_raw[type_name][0]:
                best_expert_raw[type_name] = (raw, label)
    expert_notes = "; ".join(
        f"{t}: best_expert={best_expert_raw.get(t, (NAN, NAN))[0]}@{best_expert_raw.get(t, (NAN, NAN))[1]} baseline={item.get('baseline_raw', NAN)} route={item.get('use_expert')}"
        for t, item in sorted(best_by_type.items())
    ) or NAN
    v15_summary = summary_from_report(RESULTS_DIR / "v15_ensemble" / "valid_report.json")
    selected_summary = summary_from_report(RESULTS_DIR / "v15_ensemble" / "selected_valid_report.json")
    type_summary = summary_from_report(RESULTS_DIR / "v15_type_specific" / "valid_report.json")
    return [
        {
            "Diagnostic": "v15 ensemble ranker",
            "Value": ranker.get("ranker_kind", NAN),
            "Score": score_cell(v15_summary),
            "Details": f"{ranker.get('num_candidates', NAN)} candidates; selected baseline {score_cell(selected_summary)}",
        },
        {
            "Diagnostic": "Top ranker choices",
            "Value": top_choices,
            "Score": score_cell(v15_summary),
            "Details": "ranker often selects model-only, but uses hybrid candidates enough to improve final score",
        },
        {
            "Diagnostic": "v15 type-specific expert",
            "Value": f"route_types={routed}",
            "Score": score_cell(type_summary),
            "Details": "no target type routed to expert because expert did not beat baseline",
        },
        {
            "Diagnostic": "Expert-vs-baseline target types",
            "Value": expert_notes,
            "Score": score_cell(type_summary),
            "Details": "expert improves over epochs but remains below general baseline on target types",
        },
    ]


def build_table_19_rule_compliance() -> list[dict[str, Any]]:
    rows = [
        ("v13/v14 retrieval", "no", "yes", "yes", "no", "diagnostic only if strict query_vi/response_vi rule applies"),
        ("v14.5 valid-select", "no", "yes", "yes", "valid checkpoint selection", "diagnostic; optimized on public valid"),
        ("v15 ensemble", "no", "yes", "yes", "yes", "best diagnostic score; do not submit as-is under strict rules"),
        ("v15 type-specific", "no", "yes", "yes", "route check by type", "diagnostic; expert route disabled in final output"),
        ("v16 legal answer-only", "yes", "no", "no", "internal split only", "submission-safe direction; no local result yet"),
        ("v16 legal query retrieval", "yes", "no", "no", "internal split only", "submission-safe direction; query_vi-only nearest train query; no local result yet"),
    ]
    return [
        {
            "Pipeline": name,
            "Uses only query_vi/response_vi?": query_only,
            "Uses provided type?": uses_type,
            "Uses original/source group fields?": uses_source,
            "Uses public valid labels for final decision?": valid_labels,
            "Report status": status,
        }
        for name, query_only, uses_type, uses_source, valid_labels, status in rows
    ]


def build_table_20_source_overlap_decomposition() -> list[dict[str, Any]]:
    analysis = load_json(AUDIT_DIR / "v13_result_analysis.json", {})
    overlap = analysis.get("combined_overlap_summary", {})
    total_raw = analysis.get("final_summary", {}).get("raw_score", 6896)
    label_map = {
        "direct_query_has_same": "direct query seen with same answer",
        "same_original_has_same": "same source/original, same answer",
        "same_original_only_conflict": "same source/original, conflicting answer",
        "same_original_no_match": "source seen but no same answer match",
        "valid_original_seen_as_train_query_has_same": "valid original appears as train query",
        "valid_query_seen_as_train_original_has_same": "valid query appears as train original",
    }
    rows = []
    for key, label in label_map.items():
        item = overlap.get(key, {})
        rows.append(
            {
                "Slice": label,
                "n": item.get("n", NAN),
                "Score/10": round(item.get("score_10"), 3) if item.get("score_10") is not None else NAN,
                "Raw contribution": item.get("raw", NAN),
                "Share of v13 raw score": pct(item.get("raw"), total_raw) if item.get("raw") is not None else NAN,
                "Exact10": item.get("exact10", NAN),
                "Zero": item.get("buckets", {}).get("0", NAN),
            }
        )
    return rows


def build_table_21_oracle_ensembles() -> list[dict[str, Any]]:
    analysis = load_json(AUDIT_DIR / "v145_result_analysis.json", {})
    rows = []
    for item in analysis.get("oracle_ensembles", []):
        rows.append(
            {
                "Candidate set": " + ".join(item.get("runs", [])),
                "Oracle score": f"{item.get('score_10'):.3f} ({item.get('raw_score')}/10000)",
                "Exact10": item.get("exact10"),
                "Zero across all": item.get("zero_all"),
                "Improved vs first": item.get("improved_vs_first"),
                "Interpretation": "candidate complementarity; not a deployable reasoning score",
            }
        )
    return rows


def make_figure_01(query_split: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    colors = {
        "train": "#4C5F73",
        "valid": "#B86B43",
        "edge": "#7D8790",
        "source": "#E7E2D8",
        "heldout": "#D8E4E1",
    }

    def draw_cluster(ax, cx: float, cy: float, radius: float, valid_indices: set[int], label: str) -> None:
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        points = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
        circle = plt.Circle((cx, cy), radius * 1.25, facecolor=colors["source"], edgecolor="#B5AA98", lw=1.2, alpha=0.65)
        ax.add_patch(circle)
        for i, (x1, y1) in enumerate(points):
            for x2, y2 in points[i + 1 :]:
                ax.plot([x1, x2], [y1, y2], color=colors["edge"], lw=0.55, alpha=0.45)
        for i, (x, y) in enumerate(points):
            is_valid = i in valid_indices
            ax.scatter(x, y, s=54, color=colors["valid"] if is_valid else colors["train"], edgecolor="white", lw=0.8, zorder=3)
        ax.text(cx, cy - radius * 1.65, label, ha="center", va="top", fontsize=8.5, color="#333333")

    ax = axes[0]
    draw_cluster(ax, 0.0, 0.55, 0.32, {1, 4}, "source A")
    draw_cluster(ax, 1.0, 0.55, 0.32, {2}, "source B")
    draw_cluster(ax, 0.5, -0.25, 0.32, {0, 5}, "source C")
    ax.set_title("Source-overlap query-disjoint", fontsize=11, weight="bold")
    ax.text(
        0.5,
        -0.78,
        f"shared source groups={query_split.get('source_group_overlap', NAN)}; exact query overlap={query_split.get('query_pair_overlap', NAN)}",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    ax.scatter([], [], s=54, color=colors["train"], label="train query")
    ax.scatter([], [], s=54, color=colors["valid"], label="held-out query")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    for cx, cy, label, heldout in [
        (0.0, 0.55, "source D", False),
        (1.0, 0.55, "source E", False),
        (0.5, -0.25, "source F", True),
    ]:
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        points = [(cx + 0.32 * np.cos(a), cy + 0.32 * np.sin(a)) for a in angles]
        circle = plt.Circle(
            (cx, cy),
            0.4,
            facecolor=colors["heldout"] if heldout else colors["source"],
            edgecolor="#98ABA5" if heldout else "#B5AA98",
            lw=1.2,
            alpha=0.7,
        )
        ax.add_patch(circle)
        for i, (x1, y1) in enumerate(points):
            for x2, y2 in points[i + 1 :]:
                ax.plot([x1, x2], [y1, y2], color=colors["edge"], lw=0.55, alpha=0.45)
        for x, y in points:
            ax.scatter(x, y, s=54, color=colors["valid"] if heldout else colors["train"], edgecolor="white", lw=0.8, zorder=3)
        ax.text(cx, cy - 0.53, label + (" (valid)" if heldout else " (train)"), ha="center", va="top", fontsize=8.5, color="#333333")
    ax.set_title("Source-disjoint validation", fontsize=11, weight="bold")
    ax.text(
        0.5,
        -0.78,
        "entire original-problem groups are removed from train",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    for ax in axes:
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(-0.95, 1.1)
        ax.axis("off")
    fig.suptitle("Figure 1. Source/query overlap structure", y=1.02, fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_01_source_query_overlap_structure.png", bbox_inches="tight")
    plt.close(fig)


def make_figure_02(valid_overlap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_overlap_rows:
        grouped[row["type"]].append(row)
    for type_name in TYPE_ORDER:
        items = grouped.get(type_name, [])
        if not items:
            continue
        n = len(items)
        rows.append(
            {
                "Type": type_name,
                "Exact query overlap %": 100 * sum(item["exact_query_overlap"] for item in items) / n,
                "Normalized query similarity %": 100 * mean([item["max_same_source_query_similarity"] for item in items]),
                "Source-group overlap %": 100 * sum(item["source_group_overlap"] for item in items) / n,
            }
        )
    write_data_csv(DATA_OUT_DIR / "figure_02_overlap_heatmap_values.csv", rows)

    matrix = np.array([[row[col] for col in rows[0].keys() if col != "Type"] for row in rows])
    fig, ax = plt.subplots(figsize=(8.3, 5.3), dpi=180)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Exact query\noverlap", "Max same-source\nquery similarity", "Source-group\noverlap"], fontsize=8.5)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["Type"] for row in rows], fontsize=8.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="#222222", fontsize=8)
    ax.set_title("Figure 2. Overlap audit between train and validation sets", fontsize=12, weight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.ax.set_ylabel("%", rotation=0, labelpad=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_02_overlap_audit_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def make_figure_03(valid_overlap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report = load_json(RESULTS_DIR / "v13" / "valid_report.json", {})
    rows = report.get("rows", [])
    by_type = defaultdict(Counter)
    for row in rows:
        by_type[row["type"]][row["score"]] += 1
    chart_rows = []
    for type_name in TYPE_ORDER:
        counts = by_type.get(type_name, Counter())
        n = sum(counts.values())
        if not n:
            continue
        chart_rows.append(
            {
                "Type": type_name,
                "Exact (10)": counts[10],
                "Near (5)": counts[5],
                "Partial (1)": counts[1],
                "Fail (0)": counts[0],
                "Fail %": 100 * counts[0] / n,
            }
        )
    write_data_csv(DATA_OUT_DIR / "figure_03_error_distribution_values.csv", chart_rows)

    overlap_by_id = {row["id"]: row for row in valid_overlap_rows}
    source_class_counts = defaultdict(Counter)
    for row in rows:
        cls = overlap_by_id.get(row["id"], {}).get("source_overlap_class", "unknown")
        source_class_counts[cls][row["score"]] += 1
    class_order = ["source_seen_same_answer", "source_seen_conflicting_answer", "source_seen_missing_numeric", "source_unseen"]
    class_labels = ["seen source\nsame answer", "seen source\nconflict", "seen source\nmissing", "source-unseen"]
    fail_rates = []
    class_ns = []
    for cls in class_order:
        counts = source_class_counts.get(cls, Counter())
        n = sum(counts.values())
        class_ns.append(n)
        fail_rates.append(100 * counts[0] / n if n else 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=180, gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    y = np.arange(len(chart_rows))
    left = np.zeros(len(chart_rows))
    colors = {"Exact (10)": "#426A8C", "Near (5)": "#7BAA8D", "Partial (1)": "#D5B46A", "Fail (0)": "#A96354"}
    for key in ["Exact (10)", "Near (5)", "Partial (1)", "Fail (0)"]:
        vals = np.array([row[key] for row in chart_rows], dtype=float)
        totals = np.array([sum(row[k] for k in ["Exact (10)", "Near (5)", "Partial (1)", "Fail (0)"]) for row in chart_rows], dtype=float)
        pct_vals = 100 * vals / totals
        ax.barh(y, pct_vals, left=left, color=colors[key], edgecolor="white", linewidth=0.6, label=key)
        left += pct_vals
    ax.set_yticks(y)
    ax.set_yticklabels([row["Type"] for row in chart_rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of validation examples (%)")
    ax.set_title("Score buckets by problem type", fontsize=10.5, weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    x = np.arange(len(class_order))
    ax.bar(x, fail_rates, color="#A96354", alpha=0.85, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\n(n={n})" for label, n in zip(class_labels, class_ns)], fontsize=8)
    ax.set_ylim(0, max(100, max(fail_rates) + 10))
    ax.set_ylabel("Fail score=0 (%)")
    ax.set_title("Failure by source-overlap class", fontsize=10.5, weight="bold")
    for xi, value in zip(x, fail_rates):
        ax.text(xi, value + 2, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Figure 3. Error distribution across problem types", fontsize=12, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_03_error_distribution.png", bbox_inches="tight")
    plt.close(fig)
    return chart_rows


def signed_log10(value: float) -> float:
    return math.copysign(math.log10(1 + abs(value)), value)


def make_figure_04() -> list[dict[str, Any]]:
    report = load_json(RESULTS_DIR / "v13" / "valid_report.json", {})
    rows = [
        row
        for row in report.get("rows", [])
        if row.get("gold_num") is not None and row.get("pred_num") is not None
    ]
    data_rows = [
        {
            "id": row["id"],
            "type": row["type"],
            "gold_num": row["gold_num"],
            "pred_num": row["pred_num"],
            "score": row["score"],
            "gold_signed_log10": signed_log10(float(row["gold_num"])),
            "pred_signed_log10": signed_log10(float(row["pred_num"])),
        }
        for row in rows
    ]
    write_data_csv(DATA_OUT_DIR / "figure_04_pred_vs_gold_values.csv", data_rows)

    score_colors = {10: "#426A8C", 5: "#7BAA8D", 1: "#D5B46A", 0: "#A96354"}
    fig, ax = plt.subplots(figsize=(6.8, 6.1), dpi=180)
    for score in [0, 1, 5, 10]:
        subset = [row for row in data_rows if row["score"] == score]
        if not subset:
            continue
        ax.scatter(
            [row["gold_signed_log10"] for row in subset],
            [row["pred_signed_log10"] for row in subset],
            s=18 if score == 10 else 14,
            alpha=0.65 if score == 10 else 0.48,
            color=score_colors[score],
            edgecolor="none",
            label=f"score {score} (n={len(subset)})",
        )
    values = [row["gold_signed_log10"] for row in data_rows] + [row["pred_signed_log10"] for row in data_rows]
    lo, hi = min(values), max(values)
    pad = 0.25
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#333333", lw=1.0, ls="--", label="diagonal")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Gold numeric answer, signed log10(1+abs(x))")
    ax.set_ylabel("Predicted numeric answer, signed log10(1+abs(x))")
    ax.set_title("Figure 4. Predicted versus gold numeric answers", fontsize=12, weight="bold")
    ax.grid(True, color="#D8D8D8", lw=0.6, alpha=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_04_pred_vs_gold_numeric_answers.png", bbox_inches="tight")
    plt.close(fig)
    return data_rows


def checkpoint_curve_rows() -> list[dict[str, Any]]:
    rows = []
    for run_name, path in [
        ("v13 type-gated", RESULTS_DIR / "v13"),
        ("v14 gate sweep", RESULTS_DIR / "v14_gate_sweep"),
        ("v14 FOBAR/SV", RESULTS_DIR / "v14_fobar_sv"),
        ("v14.5 v13", RESULTS_DIR / "v14_5_v13"),
        ("v14.5 gate", RESULTS_DIR / "v14_5_gate_sweep"),
        ("v15 ensemble selected", RESULTS_DIR / "v15_ensemble"),
        ("v15 type-specific", RESULTS_DIR / "v15_type_specific"),
    ]:
        report = load_json(path / "checkpoint_selection_report.json", {})
        selected = report.get("selected", {}).get("label")
        for item in report.get("all_checkpoints", []):
            summary = item.get("summary", {})
            rows.append(
                {
                    "Run": run_name,
                    "Order": item.get("order"),
                    "Label": item.get("label"),
                    "Score": summary.get("score_10"),
                    "Raw score": summary.get("raw_score"),
                    "Exact10": summary.get("buckets", {}).get("10"),
                    "Extractable": summary.get("extractable"),
                    "Retrieval usage %": 100 * item.get("decision_summary", {}).get("retrieval_used_pct", 0)
                    if item.get("decision_summary")
                    else None,
                    "Selected": item.get("label") == selected,
                }
            )
    return rows


def make_figure_05_checkpoint_curves() -> list[dict[str, Any]]:
    rows = checkpoint_curve_rows()
    write_data_csv(DATA_OUT_DIR / "figure_05_checkpoint_curve_values.csv", rows)
    fig, ax = plt.subplots(figsize=(8.4, 5.1), dpi=180)
    colors = {
        "v13 type-gated": "#426A8C",
        "v14 gate sweep": "#7BAA8D",
        "v14 FOBAR/SV": "#A96354",
        "v14.5 v13": "#6B7F59",
        "v14.5 gate": "#8B7AAE",
        "v15 ensemble selected": "#B87A4B",
        "v15 type-specific": "#6E6E6E",
    }
    for run_name in colors:
        run_rows = [row for row in rows if row["Run"] == run_name and row["Score"] is not None]
        ax.plot(
            [row["Order"] for row in run_rows],
            [row["Score"] for row in run_rows],
            marker="o",
            ms=4,
            lw=1.8,
            color=colors[run_name],
            label=run_name,
        )
        selected_rows = [row for row in run_rows if row["Selected"]]
        if selected_rows:
            row = selected_rows[0]
            ax.scatter(row["Order"], row["Score"], s=88, color=colors[run_name], edgecolor="black", linewidth=0.8, zorder=4)
            ax.text(row["Order"], row["Score"] + 0.12, row["Label"], ha="center", va="bottom", fontsize=7.5)
    ax.set_xlabel("Checkpoint order")
    ax.set_ylabel("Overlap-valid score")
    ax.set_title("Figure 5. Checkpoint/valid selection curves", fontsize=12, weight="bold")
    ax.grid(True, color="#D8D8D8", lw=0.6, alpha=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_05_checkpoint_selection_curves.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def make_figure_06_v14_hybrid_gain() -> list[dict[str, Any]]:
    rows = []
    for type_name in TYPE_ORDER:
        for run_name, path in [
            ("v14 sweep", RESULTS_DIR / "v14_gate_sweep"),
            ("v14 FOBAR/SV", RESULTS_DIR / "v14_fobar_sv"),
        ]:
            model_score = by_type_score(path / "model_valid_report.json", type_name)
            hybrid_score = by_type_score(path / "valid_report.json", type_name)
            usage = retrieval_used_by_type(path, type_name)
            usage_pct = 100 * usage.get("retrieval_used", 0) / usage["n"] if usage.get("n") else None
            rows.append(
                {
                    "Run": run_name,
                    "Type": type_name,
                    "Hybrid gain": hybrid_score - model_score if hybrid_score is not None and model_score is not None else None,
                    "Retrieval usage %": usage_pct,
                    "Hybrid score": hybrid_score,
                    "Model score": model_score,
                }
            )
    write_data_csv(DATA_OUT_DIR / "figure_06_v14_hybrid_gain_values.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=180, sharey=True)
    colors = {"v14 sweep": "#7BAA8D", "v14 FOBAR/SV": "#A96354"}
    for ax, run_name in zip(axes, colors):
        run_rows = [row for row in rows if row["Run"] == run_name]
        y = np.arange(len(run_rows))
        gains = [row["Hybrid gain"] or 0 for row in run_rows]
        usage = [row["Retrieval usage %"] or 0 for row in run_rows]
        ax.barh(y, gains, color=colors[run_name], alpha=0.86, label="hybrid gain")
        ax.axvline(0, color="#333333", lw=0.8)
        ax2 = ax.twiny()
        ax2.plot(usage, y, color="#36454F", marker="o", ms=3.5, lw=1.2, label="retrieval usage")
        ax2.set_xlim(0, 105)
        ax2.set_xlabel("Retrieval usage (%)", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([row["Type"] for row in run_rows], fontsize=8.2)
        ax.invert_yaxis()
        ax.set_xlabel("Hybrid - model score")
        ax.set_title(run_name, fontsize=10.5, weight="bold")
        ax.grid(True, axis="x", color="#D8D8D8", lw=0.6, alpha=0.8)
    fig.suptitle("Figure 6. V14 hybrid gain versus retrieval usage by type", fontsize=12, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_06_v14_hybrid_gain_by_type.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def make_figure_07_type_score_heatmap() -> list[dict[str, Any]]:
    runs = [
        ("v13", RESULTS_DIR / "v13" / "valid_report.json"),
        ("v14 sweep", RESULTS_DIR / "v14_gate_sweep" / "valid_report.json"),
        ("v14 FOBAR/SV", RESULTS_DIR / "v14_fobar_sv" / "valid_report.json"),
        ("v14.5 v13", RESULTS_DIR / "v14_5_v13" / "valid_report.json"),
        ("v14.5 gate", RESULTS_DIR / "v14_5_gate_sweep" / "valid_report.json"),
        ("v15 ensemble", RESULTS_DIR / "v15_ensemble" / "valid_report.json"),
        ("v15 type-specific", RESULTS_DIR / "v15_type_specific" / "valid_report.json"),
    ]
    rows = []
    for type_name in TYPE_ORDER:
        row = {"Type": type_name}
        for run_name, path in runs:
            row[run_name] = by_type_score(path, type_name)
        rows.append(row)
    write_data_csv(DATA_OUT_DIR / "figure_07_type_score_heatmap_values.csv", rows)

    matrix = np.array([[row[run_name] for run_name, _ in runs] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(11.2, 5.2), dpi=180)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=10, aspect="auto")
    ax.set_xticks(np.arange(len(runs)))
    ax.set_xticklabels([run_name for run_name, _ in runs], fontsize=8.2, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["Type"] for row in rows], fontsize=8.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#222222")
    ax.set_title("Figure 7. Type-wise official-valid score comparison", fontsize=12, weight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.ax.set_ylabel("score", rotation=0, labelpad=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_07_type_wise_score_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def make_figure_08_score_trajectory() -> list[dict[str, Any]]:
    rows = []
    for order, (name, path, family) in enumerate(REPORT_RUNS, start=1):
        summary = run_valid_summary(name, path)
        if not summary or summary.get("score_10") is None:
            continue
        rows.append(
            {
                "Order": order,
                "Version": name,
                "Family": family,
                "Score": summary.get("score_10"),
                "Raw score": summary.get("raw_score"),
                "Exact10": summary.get("exact10"),
                "Extractable": summary.get("extractable"),
            }
        )
    write_data_csv(DATA_OUT_DIR / "figure_08_score_trajectory_values.csv", rows)

    fig, ax = plt.subplots(figsize=(11.8, 5.2), dpi=180)
    x = np.arange(len(rows))
    scores = [row["Score"] for row in rows]
    colors = ["#9A9A9A" if row["Version"] != "v15 ensemble" else "#B87A4B" for row in rows]
    ax.bar(x, scores, color=colors, edgecolor="#333333", linewidth=0.4)
    ax.plot(x, scores, color="#3F5F7F", lw=1.4, marker="o", ms=3.5)
    ax.set_xticks(x)
    ax.set_xticklabels([row["Version"] for row in rows], rotation=35, ha="right", fontsize=8.2)
    ax.set_ylim(0, 7.6)
    ax.set_ylabel("Official-valid score")
    ax.set_title("Figure 8. Validation score trajectory across experiment versions", fontsize=12, weight="bold")
    ax.grid(True, axis="y", color="#D8D8D8", lw=0.6, alpha=0.8)
    for idx, score in enumerate(scores):
        if rows[idx]["Version"] in {"v13", "v15 ensemble"}:
            ax.text(idx, score + 0.12, f"{score:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_08_score_trajectory_v2_v15.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def make_figure_09_model_to_final_gain() -> list[dict[str, Any]]:
    rows = []
    for name, path, mechanism in [
        ("v12", RESULTS_DIR / "v12", "broad retrieval"),
        ("v13", RESULTS_DIR / "v13", "type-gated retrieval"),
        ("v14 gate", RESULTS_DIR / "v14_gate_sweep", "gate sweep"),
        ("v14 FOBAR/SV", RESULTS_DIR / "v14_fobar_sv", "extended retrieval"),
        ("v14.5 v13", RESULTS_DIR / "v14_5_v13", "valid-select V13"),
        ("v14.5 gate", RESULTS_DIR / "v14_5_gate_sweep", "valid-select gate"),
        ("v15 ensemble", RESULTS_DIR / "v15_ensemble", "ranker"),
        ("v15 type-specific", RESULTS_DIR / "v15_type_specific", "safe expert route"),
    ]:
        model = summary_from_report(path / "model_valid_report.json")
        final = summary_from_report(path / "valid_report.json")
        if not model or not final:
            continue
        rows.append(
            {
                "Run": name,
                "Mechanism": mechanism,
                "Model score": model.get("score_10"),
                "Final score": final.get("score_10"),
                "Gain": final.get("score_10") - model.get("score_10"),
            }
        )
    write_data_csv(DATA_OUT_DIR / "figure_09_model_to_final_gain_values.csv", rows)

    fig, ax = plt.subplots(figsize=(10.4, 5.0), dpi=180)
    x = np.arange(len(rows))
    gains = [row["Gain"] for row in rows]
    ax.bar(x, gains, color="#6F8FAF", edgecolor="#333333", linewidth=0.4)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([row["Run"] for row in rows], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Final score - model-only score")
    ax.set_title("Figure 9. Retrieval/ranker contribution beyond model-only outputs", fontsize=12, weight="bold")
    ax.grid(True, axis="y", color="#D8D8D8", lw=0.6, alpha=0.8)
    for idx, gain in enumerate(gains):
        ax.text(idx, gain + 0.04, f"{gain:+.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_09_model_to_final_gain.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def make_figure_10_source_overlap_decomposition() -> list[dict[str, Any]]:
    rows = build_table_20_source_overlap_decomposition()
    write_data_csv(DATA_OUT_DIR / "figure_10_source_overlap_decomposition_values.csv", rows)
    plot_rows = [row for row in rows if isinstance(row["Raw contribution"], int)]
    labels = [row["Slice"] for row in plot_rows[:4]]
    values = [row["Raw contribution"] for row in plot_rows[:4]]
    colors = ["#567C61", "#8CA67C", "#B87A4B", "#9A9A9A"]

    fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=180)
    ax.barh(np.arange(len(values)), values, color=colors, edgecolor="#333333", linewidth=0.4)
    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlabel("Raw score contribution")
    ax.set_title("Figure 10. V13 source-overlap score decomposition", fontsize=12, weight="bold")
    ax.grid(True, axis="x", color="#D8D8D8", lw=0.6, alpha=0.8)
    for idx, value in enumerate(values):
        ax.text(value + 50, idx, str(value), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_10_source_overlap_decomposition.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def make_figure_11_same_source_score_gap() -> list[dict[str, Any]]:
    analysis = load_json(AUDIT_DIR / "v13_result_analysis.json", {})
    overlap = analysis.get("combined_overlap_summary", {})
    specs = [
        ("same answer", "same_original_has_same"),
        ("conflicting answer", "same_original_only_conflict"),
        ("no same-answer match", "same_original_no_match"),
    ]
    rows = []
    for label, key in specs:
        item = overlap.get(key, {})
        rows.append(
            {
                "Slice": label,
                "n": item.get("n"),
                "Score": item.get("score_10"),
                "Exact10": item.get("exact10"),
                "Zero": item.get("buckets", {}).get("0"),
            }
        )
    write_data_csv(DATA_OUT_DIR / "figure_11_same_source_score_gap_values.csv", rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=180)
    x = np.arange(len(rows))
    scores = [row["Score"] for row in rows]
    ax.bar(x, scores, color=["#567C61", "#B87A4B", "#9A9A9A"], edgecolor="#333333", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['Slice']}\n(n={row['n']})" for row in rows], fontsize=8.5)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Score/10")
    ax.set_title("Figure 11. Same-source agreement versus conflict", fontsize=12, weight="bold")
    ax.grid(True, axis="y", color="#D8D8D8", lw=0.6, alpha=0.8)
    for idx, score in enumerate(scores):
        ax.text(idx, score + 0.18, f"{score:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_11_same_source_score_gap.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def make_figure_12_v15_ranker_choices() -> list[dict[str, Any]]:
    ranker = load_json(RESULTS_DIR / "v15_ensemble" / "ensemble_ranker_report.json", {})
    counts = Counter(ranker.get("choice_counts", {})).most_common(12)
    rows = [{"Candidate": name, "Count": count} for name, count in counts]
    write_data_csv(DATA_OUT_DIR / "figure_12_v15_ranker_choice_values.csv", rows)

    fig, ax = plt.subplots(figsize=(9.6, 5.1), dpi=180)
    y = np.arange(len(rows))
    ax.barh(y, [row["Count"] for row in rows], color="#6F8FAF", edgecolor="#333333", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([row["Candidate"] for row in rows], fontsize=7.7)
    ax.invert_yaxis()
    ax.set_xlabel("Validation examples selected")
    ax.set_title("Figure 12. V15 ensemble ranker candidate choices", fontsize=12, weight="bold")
    ax.grid(True, axis="x", color="#D8D8D8", lw=0.6, alpha=0.8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_12_v15_ranker_choices.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def make_figure_13_rule_compliance_matrix() -> list[dict[str, Any]]:
    rows = build_table_19_rule_compliance()
    feature_cols = [
        "Uses only query_vi/response_vi?",
        "Uses provided type?",
        "Uses original/source group fields?",
        "Uses public valid labels for final decision?",
    ]
    numeric = []
    for row in rows:
        numeric.append([
            0 if row["Uses only query_vi/response_vi?"] == "yes" else 1,
            1 if row["Uses provided type?"] == "yes" else 0,
            1 if row["Uses original/source group fields?"] not in {"no"} else 0,
            1 if row["Uses public valid labels for final decision?"] in {"yes", "valid checkpoint selection", "route check by type"} else 0,
        ])
    write_data_csv(DATA_OUT_DIR / "figure_13_rule_compliance_matrix_values.csv", rows)

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180)
    matrix = np.array(numeric, dtype=float)
    im = ax.imshow(matrix, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["Pipeline"] for row in rows], fontsize=8.3)
    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels(["query-only", "type", "source/original", "valid labels"], rotation=25, ha="right", fontsize=8.2)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            label = "risk" if matrix[i, j] else "ok"
            text_color = "#FFFFFF" if matrix[i, j] else "#222222"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.2, color=text_color)
    ax.set_title("Figure 13. Diagnostic versus submission-safe pipeline features", fontsize=12, weight="bold", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_13_rule_compliance_matrix.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def write_report_additions() -> None:
    v13 = summary_from_report(RESULTS_DIR / "v13" / "valid_report.json")
    v13_model = summary_from_report(RESULTS_DIR / "v13" / "model_valid_report.json")
    v15 = summary_from_report(RESULTS_DIR / "v15_ensemble" / "valid_report.json")
    v15_model = summary_from_report(RESULTS_DIR / "v15_ensemble" / "model_valid_report.json")
    v15_selected = summary_from_report(RESULTS_DIR / "v15_ensemble" / "selected_valid_report.json")
    v10 = summary_from_report(RESULTS_DIR / "v10" / "valid_report.json")
    v10_source = summary_from_report(RESULTS_DIR / "v10" / "source_valid_report.json")
    overlap = load_json(AUDIT_DIR / "v13_result_analysis.json", {}).get("combined_overlap_summary", {})
    same = overlap.get("same_original_has_same", {})
    conflict = overlap.get("same_original_only_conflict", {})
    no_match = overlap.get("same_original_no_match", {})
    v15_stats = output_stats(RESULTS_DIR / "v15_ensemble" / "valid_output.json") or {}

    text = f"""# Bổ sung nội dung báo cáo v8-v15

## 1. Nhận xét tổng quan cho bản thảo hiện tại

Bản thảo `report/Fine_tune_GPT_2.pdf` đã có phần nền về dữ liệu, leakage, LoRA và retrieval đến v14. Phần cần bổ sung lớn nhất là biến chuỗi thí nghiệm thành một luận điểm nghiên cứu: trong thiết lập GPT-2 nhỏ, điểm validation tăng chủ yếu nhờ target alignment, source-local memory, retrieval gating và answer selection, chưa đủ để khẳng định robust mathematical reasoning.

Các bảng/hình nên được đọc theo ba lớp:

- Lớp metric: official-valid score, exact10, extractability.
- Lớp robustness: source-disjoint, FOBAR/SV, conflict-source, source-unseen.
- Lớp cơ chế: model-only, retrieval override, ranker/candidate selection, output shape.

## 2. Nội dung nên thêm vào Introduction

Nên nêu đóng góp là một protocol đánh giá reasoning cho mô hình ngôn ngữ nhỏ, không chỉ là tối ưu notebook Kaggle. Câu mở rộng phù hợp:

> Công trình này không chỉ tìm cấu hình điểm cao cho GPT-2 tiếng Việt, mà còn kiểm toán xem điểm cao đến từ năng lực reasoning chuyển giao, từ metric alignment, hay từ source-level memory trong dữ liệu augmentation.

Cần nhấn mạnh final-answer accuracy không đồng nghĩa với reasoning faithfulness. Trong task này, một mô hình answer-only có thể tối ưu đúng metric tốt hơn mô hình sinh lời giải dài, vì scorer chỉ đọc đáp án số cuối cùng.

## 3. Nội dung nên thêm vào Data Audit

Đưa overlap audit thành bằng chứng trung tâm:

- Exact query overlap trên official validation: 0/1000.
- Source-group overlap: 965/1000.
- Nhiều type gần như hoàn toàn source-seen: GSM_Rephrased và GSM_SV đạt 100% source overlap.

Điểm cần viết rõ: query-disjoint không tương đương source-disjoint. Nếu train chứa biến thể khác của cùng bài gốc, mô hình hoặc retrieval có thể học mapping cục bộ từ source/template sang đáp án mà không cần giải bài mới.

## 4. Nội dung nên thêm vào Experiments

Nên viết lại phần experiment theo bốn family:

1. Full-solution SFT: giữ rationale nhưng target dài, tín hiệu final answer bị loãng, v2 chỉ đạt 0.585.
2. Compact reasoning: cải thiện format/extractability, nhưng source-disjoint gần không tăng; v10 valid {score_cell(v10)} trong khi source-disjoint {score_cell(v10_source)}.
3. Answer-only LoRA: tăng mạnh vì loss align trực tiếp với final-answer metric.
4. Retrieval/ranker hybrid: điểm cao nhất nhưng phụ thuộc source overlap và candidate selection.

Kết quả mới cần thêm: v14.5 không vượt v13, còn v15 ensemble đạt {score_cell(v15)} nhưng là diagnostic branch vì dùng ranker/candidate selection trên public valid và dùng các feature ngoài `query_vi`/`response_vi`.

## 5. Section V. MATH REASONING VỚI CÁC MÔ HÌNH NHỎ

### 5.1 Câu hỏi nghiên cứu

Với các mô hình dưới 500M tham số, fine-tuning trên lời giải toán có tạo ra năng lực reasoning robust không, hay chủ yếu tạo ra rationalization/pattern matching?

### 5.2 Định nghĩa operational

Trong báo cáo này, robust mathematical reasoning nên được định nghĩa bằng các điều kiện quan sát được:

- Điểm không sụp mạnh trên source-disjoint hoặc source-unseen.
- Không chỉ mạnh trên Rephrased/AnsAug mà vẫn ổn trên FOBAR/SV.
- Không phụ thuộc lớn vào retrieval từ cùng source group.
- Nếu sinh rationale, rationale phải hỗ trợ đúng đáp án thay vì chỉ hợp thức hóa đáp án đã đoán.

Ngược lại, rationalization/pattern matching được nhận diện khi điểm tăng nhưng output vẫn answer-only, score cao tập trung ở source-seen same-answer, và conflict-source/no-match giảm mạnh.

### 5.3 Bằng chứng thực nghiệm

**Answer-signal dilution.** Với SFT autoregressive, loss trung bình trên toàn target. Nếu chỉ vài token cuối chứa đáp án được scorer dùng, target càng dài thì tín hiệu trực tiếp cho đáp án càng loãng. Điều này giải thích vì sao full-solution v2 đạt thấp, trong khi answer-only tăng mạnh.

**Compact reasoning chưa tạo transfer rõ.** v10 rút ngắn output và tăng extractability, nhưng source-disjoint chỉ quanh {score_cell(v10_source)}. So với v9/v10, tăng official-valid không đi kèm tăng source-disjoint tương ứng.

**Source-overlap chi phối điểm.** Với v13, same-source same-answer đạt {same.get('score_10', float('nan')):.3f}, conflict-source chỉ {conflict.get('score_10', float('nan')):.3f}, và no-match chỉ {no_match.get('score_10', float('nan')):.3f}. Khoảng cách này là bằng chứng mạnh rằng source memory là biến giải thích quan trọng.

**Retrieval/ranker đóng góp lớn hơn standalone reasoning.** v13 model-only {score_cell(v13_model)} tăng lên {score_cell(v13)} sau hybrid. v15 ensemble model-only {score_cell(v15_model)} tăng lên selected single-candidate {score_cell(v15_selected)} rồi final ranker {score_cell(v15)}. Đây là gain từ chọn đáp án/candidate, không phải bằng chứng trực tiếp rằng GPT-2 tự sinh reasoning bền vững.

**Output behavior.** v15 ensemble có median output length khoảng {v15_stats.get('median_chars', float('nan')):.1f} ký tự và short single-line rate {pct_from_fraction(v15_stats.get('short_single_line_rate'))}. Điều này cho thấy output chủ yếu là câu trả lời ngắn, không phải chain-of-thought dài.

### 5.4 Kết luận khoa học

Bằng chứng hiện tại ủng hộ kết luận thận trọng: GPT-2 nhỏ học được answer formatting, answer extraction, source-local mapping và calibration/candidate selection khá tốt. Tuy nhiên, các chỉ dấu robust reasoning vẫn yếu: source-disjoint thấp, conflict-source thấp, FOBAR/SV khó, và điểm cao nhất phụ thuộc vào retrieval/ranker. Vì vậy báo cáo không nên claim mô hình đã học robust mathematical reasoning theo nghĩa mạnh.

## 6. Phần v14.5/v15 nên viết thêm

v14.5 kiểm tra giả thuyết rằng train full + valid selection có thể vượt v13. Kết quả không ủng hộ giả thuyết này: v14.5 v13 đạt 6.823 và v14.5 gate đạt 6.807, đều thấp hơn v13 6.896.

v15 ensemble là điểm cao nhất hiện tại với 7.016, nhưng diễn giải đúng là best diagnostic score. Ranker chọn giữa 32 candidates và thường chọn cả model-only lẫn hybrid candidates. Nhánh này hữu ích để chứng minh lỗi giữa các checkpoint/candidate có tính bổ sung, nhưng không nên trình bày là bằng chứng reasoning nội tại hoặc cấu hình submission-safe.

## 7. Limitations cần thêm

- Không có source-disjoint result cho nhiều nhánh retrieval v11-v15 trong local artifacts.
- v15 ensemble dùng public valid labels cho ranker và dùng feature ngoài strict `query_vi`/`response_vi`.
- Các chỉ số reasoning quality chưa có human annotation; hiện chỉ có proxy như source-disjoint, conflict-source, output length và type-wise robustness.
- Hidden test generalization không nên được claim nếu chưa biết test có source-overlap giống valid hay không.

## 8. Đề xuất câu kết luận

> Trong giới hạn mô hình GPT-2 tiếng Việt nhỏ và runtime Kaggle, fine-tuning có thể tạo ra một solver thực dụng cho final-answer metric, nhưng bằng chứng thực nghiệm chỉ ra rằng phần lớn gain đến từ alignment với đáp án cuối, source-overlap retrieval và ranker/candidate selection. Do đó, báo cáo xem đây là một nghiên cứu về đánh giá và kiểm toán reasoning ở small LM, hơn là bằng chứng rằng mô hình đã học robust mathematical reasoning.
"""
    (REPORT_DIR / "report_additions_v8_v15.md").write_text(text, encoding="utf-8")


def collect_missing_values(table_map: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for table_name, rows in table_map.items():
        cells = []
        for i, row in enumerate(rows, start=1):
            for key, value in row.items():
                if value == NAN or value is None:
                    cells.append(f"row {i} ({row.get('Version') or row.get('Candidate') or row.get('Model') or row.get('Type') or row.get('Version group')}), {key}")
        if cells:
            missing[table_name] = cells
    return missing


def write_summary(
    train: list[dict[str, Any]],
    valid: list[dict[str, Any]],
    valid_overlap_rows: list[dict[str, Any]],
    table_map: dict[str, list[dict[str, Any]]],
    figure2_rows: list[dict[str, Any]],
) -> None:
    exact_query_seen = sum(row["exact_query_overlap"] for row in valid_overlap_rows)
    norm_query_seen = sum(row["normalized_query_overlap"] for row in valid_overlap_rows)
    source_seen = sum(row["source_group_overlap"] for row in valid_overlap_rows)
    class_counts = Counter(row["source_overlap_class"] for row in valid_overlap_rows)
    missing = collect_missing_values(table_map)

    summary = {
        "generated_from": {
            "dataset": ["dataset/train.json", "dataset/valid.json"],
            "audit": ["audit/*.json"],
            "results": ["results/*/*.json"],
        },
        "dataset": {
            "train_records": len(train),
            "valid_records": len(valid),
            "valid_exact_query_overlap": exact_query_seen,
            "valid_normalized_query_overlap": norm_query_seen,
            "valid_source_group_overlap": source_seen,
            "valid_source_group_overlap_pct": source_seen / len(valid),
            "valid_source_overlap_class_counts": dict(class_counts),
        },
        "best_available_scores": {
            "v8_answer_only_valid": summary_from_report(RESULTS_DIR / "v8_select_checkpoint" / "beam2_lr1e-3" / "valid_report.json"),
            "v10_compact_source_disjoint": summary_from_report(RESULTS_DIR / "v10" / "source_valid_report.json"),
            "v11_answer_only_overlap_valid": summary_from_report(RESULTS_DIR / "v11" / "overlap_valid_report.json"),
            "v13_hybrid_official_valid": summary_from_report(RESULTS_DIR / "v13" / "valid_report.json"),
            "v13_hybrid_overlap_valid": summary_from_report(RESULTS_DIR / "v13" / "overlap_valid_report.json"),
            "v14_gate_sweep_official_valid": summary_from_report(RESULTS_DIR / "v14_gate_sweep" / "valid_report.json"),
            "v14_gate_sweep_overlap_valid": summary_from_report(RESULTS_DIR / "v14_gate_sweep" / "overlap_valid_report.json"),
            "v14_fobar_sv_official_valid": summary_from_report(RESULTS_DIR / "v14_fobar_sv" / "valid_report.json"),
            "v14_fobar_sv_overlap_valid": summary_from_report(RESULTS_DIR / "v14_fobar_sv" / "overlap_valid_report.json"),
            "v14_5_v13_official_valid": summary_from_report(RESULTS_DIR / "v14_5_v13" / "valid_report.json"),
            "v14_5_gate_sweep_official_valid": summary_from_report(RESULTS_DIR / "v14_5_gate_sweep" / "valid_report.json"),
            "v15_ensemble_official_valid": summary_from_report(RESULTS_DIR / "v15_ensemble" / "valid_report.json"),
            "v15_type_specific_official_valid": summary_from_report(RESULTS_DIR / "v15_type_specific" / "valid_report.json"),
        },
        "figure_02_values": figure2_rows,
        "nan_or_missing_cells": missing,
        "notes": [
            "NaN means the requested metric is not available in the local dataset/audit/results artifacts.",
            "Reasoning consistency is not human-rated; tables use extractability only where explicitly marked.",
            "Source-disjoint scores are available for v9/v10 source-valid reports, but not for v11-v15 retrieval/ranker branches in this checkout.",
            "v15_ensemble is the best diagnostic validation score, but it is not submission-safe under a strict query_vi/response_vi-only rule.",
        ],
    }
    (REPORT_DIR / "report_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = [
        "# Report audit assets",
        "",
        "Generated from local `dataset/`, `audit/`, and `results/` artifacts.",
        "",
        "## Key audit findings",
        "",
        f"- Train records: {len(train):,}; official validation records: {len(valid):,}.",
        f"- Exact query overlap in official validation: {exact_query_seen}/{len(valid)}.",
        f"- Normalized query overlap in official validation: {norm_query_seen}/{len(valid)}.",
        f"- Source-group overlap by the v11-v13 source key: {source_seen}/{len(valid)} ({source_seen / len(valid):.1%}).",
        "- High source-group overlap means official validation can reward source memorization even when exact query overlap is zero.",
        "- v15 ensemble is the best diagnostic official-valid run at 7.016/10, but it depends on candidate ranker/valid labels and is not submission-safe under strict rules.",
        "- v13 remains the strongest simple retrieval baseline; v14.5 valid-select does not beat it, and v14 FOBAR/SV extended retrieval remains a negative ablation.",
        "- The core reasoning conclusion is cautious: target alignment, source-memory retrieval, and answer selection explain more gain than robust source-disjoint reasoning.",
        "",
        "## Outputs",
        "",
        "- Tables: `report/tables/*.csv`, `report/tables/*.md`, and combined `report/tables.md`.",
        "- Figures: `report/figures/figure_01_source_query_overlap_structure.png` through `figure_13_rule_compliance_matrix.png`.",
        "- Figure source data: `report/data/*.csv`.",
        "- Machine-readable summary: `report/report_audit_summary.json`.",
        "- Report text additions: `report/report_additions_v8_v15.md`.",
        "",
        "## NaN policy",
        "",
        "`NaN` marks values that cannot be recovered from the local artifacts, especially v3/v4 metrics, v11-v15 source-disjoint scores, runtimes without manifests, and human-rated reasoning quality.",
        "",
    ]
    (REPORT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    train = load_records(DATA_DIR / "train.json")
    valid = load_records(DATA_DIR / "valid.json")
    source_index = build_source_index(train)
    valid_overlap_rows = build_valid_overlap_rows(valid, source_index)
    write_data_csv(DATA_OUT_DIR / "valid_overlap_audit_by_example.csv", valid_overlap_rows)

    query_split = load_json(RESULTS_DIR / "v13" / "query_disjoint_split_report.json", {})
    make_figure_01(query_split)
    figure2_rows = make_figure_02(valid_overlap_rows)
    make_figure_03(valid_overlap_rows)
    make_figure_04()
    make_figure_05_checkpoint_curves()
    make_figure_06_v14_hybrid_gain()
    make_figure_07_type_score_heatmap()
    make_figure_08_score_trajectory()
    make_figure_09_model_to_final_gain()
    make_figure_10_source_overlap_decomposition()
    make_figure_11_same_source_score_gap()
    make_figure_12_v15_ranker_choices()
    make_figure_13_rule_compliance_matrix()

    table_map = {
        "table_02_type_distribution": build_table_02(train, valid),
        "table_04_experiment_settings": build_table_04(),
        "table_05_full_solution_sft_baseline": build_table_05(),
        "table_06_compact_reasoning_target_ablation": build_table_06(),
        "table_08_answer_only_lora_ablation": build_table_08(),
        "table_09_grpo_lite_reward_tuning": build_table_09(),
        "table_10_source_aware_retrieval_ablation": build_table_10(),
        "table_11_final_model_selection": build_table_11(),
        "table_12_qualitative_error_taxonomy": build_table_12(valid_overlap_rows),
        "table_13_v14_run_comparison": build_table_13_v14_run_comparison(),
        "table_14_v14_type_gains": build_table_14_v14_type_gains(),
        "table_15_v14_gate_config": build_table_15_v14_gate_config(),
        "table_16_v8_v15_leaderboard": build_table_16_leaderboard(),
        "table_17_model_to_final_gain": build_table_17_model_hybrid_gain(),
        "table_18_v15_diagnostics": build_table_18_v15_diagnostics(),
        "table_19_rule_compliance_matrix": build_table_19_rule_compliance(),
        "table_20_source_overlap_decomposition": build_table_20_source_overlap_decomposition(),
        "table_21_oracle_ensemble_upper_bounds": build_table_21_oracle_ensembles(),
    }
    captions = {
        "table_02_type_distribution": "**Table 2. Distribution of problem types and source groups.** MATH is split into Rephrased, AnsAug, FOBAR, and SV rather than aggregated. The source-group count here is type-local, while the overlap audit reports 12,780 global source groups and 965/1000 validation rows with seen source groups.",
        "table_04_experiment_settings": "**Table 4. Experiment settings by version group.** Configuration rows combine local manifests with notebook-level descriptions; unavailable fields are left as NaN.",
        "table_05_full_solution_sft_baseline": "**Table 5. Full-solution SFT baseline.** Long targets preserve reasoning text but are expensive and may reduce final-answer reliability under the runtime constraint.",
        "table_06_compact_reasoning_target_ablation": "**Table 6. Compact reasoning target ablation.** Compact-equation supervision tests whether shorter arithmetic-focused rationales improve robustness compared with answer-only and full natural-language targets.",
        "table_08_answer_only_lora_ablation": "**Table 8. Answer-only LoRA ablation.** Answer-only training aligns the loss with the final-answer metric and tests whether score improvements require explicit reasoning supervision.",
        "table_09_grpo_lite_reward_tuning": "**Table 9. GRPO-lite reward-tuning results.** Reward tuning tests whether direct optimization of the scoring rule improves generation beyond supervised LoRA under a limited compute budget.",
        "table_10_source_aware_retrieval_ablation": "**Table 10. Source-aware retrieval ablation.** Retrieval improves performance only when reliable source-level variants exist; therefore, the table reports retrieval usage and source-disjoint availability to distinguish memory from robust reasoning.",
        "table_11_final_model_selection": "**Table 11. Final model selection.** The submitted configuration is selected by balancing official score, runtime, robustness, and risk of overfitting to source overlap.",
        "table_12_qualitative_error_taxonomy": "**Table 12. Qualitative error taxonomy.** The taxonomy separates answer-format failures, computational failures, relational reasoning failures, and rationalization/source-memory risks. Example ids are heuristic seeds for manual inspection.",
        "table_13_v14_run_comparison": "**Table 13. V14 run-level comparison.** This table separates model-only accuracy from retrieval-assisted accuracy, so gains from source-memory retrieval are not confused with standalone reasoning ability.",
        "table_14_v14_type_gains": "**Table 14. V14 type-wise hybrid gains.** The table reports per-type model-only score, retrieval-assisted score, gain, and retrieval usage to identify where the retrieval policy helps or hurts.",
        "table_15_v14_gate_config": "**Table 15. V14 selected retrieval gates.** Gate parameters show which problem types use majority-source retrieval versus strict nearest-query retrieval and how often each gate fires on official validation.",
        "table_16_v8_v15_leaderboard": "**Table 16. Main validation leaderboard through v15.** The table updates the report with v14.5 and v15 results and separates diagnostic score from missing/unrecoverable runs.",
        "table_17_model_to_final_gain": "**Table 17. Model-only versus final hybrid/ranker output.** The table quantifies how much of the score comes from retrieval, checkpoint/candidate selection, or a ranker beyond standalone GPT-2 generation.",
        "table_18_v15_diagnostics": "**Table 18. V15 diagnostic details.** V15 ensemble is the best local validation score, while the type-specific expert does not route any target type because it remains below the general baseline.",
        "table_19_rule_compliance_matrix": "**Table 19. Diagnostic versus submission-safe pipeline features.** The table distinguishes research/audit branches from strict query_vi/response_vi-only directions.",
        "table_20_source_overlap_decomposition": "**Table 20. Source-overlap score decomposition.** V13 score is dominated by same-source same-answer examples, while conflict and no-match slices remain low.",
        "table_21_oracle_ensemble_upper_bounds": "**Table 21. Oracle ensemble upper bounds.** Candidate complementarity shows possible answer-selection upside, but these are not deployable reasoning scores.",
    }
    combined = ["# Report Tables", ""]
    for name, rows in table_map.items():
        combined.append(write_table(name, rows, captions[name]))
    (REPORT_DIR / "tables.md").write_text("\n".join(combined), encoding="utf-8")

    write_summary(train, valid, valid_overlap_rows, table_map, figure2_rows)
    write_report_additions()
    print(f"Wrote report assets to {REPORT_DIR}")


if __name__ == "__main__":
    main()
