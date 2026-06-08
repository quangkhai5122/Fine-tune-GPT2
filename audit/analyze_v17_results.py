"""Analyze finished v17 legal answer-only runs and compare with prior runs."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import evaluate, extract_answer, load_records


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def report_summary(name: str, path: str | Path, gold: list[dict]) -> dict:
    path = ROOT / path
    data = load_json(path)
    if "summary" in data:
        summary = data["summary"]
        by_type = data.get("by_type", {})
    else:
        report = evaluate(data, gold)
        summary = report["summary"]
        by_type = {}
    buckets = summary.get("buckets", {})
    return {
        "name": name,
        "raw": summary.get("raw_score"),
        "score10": summary.get("score_10"),
        "exact10": buckets.get("10", buckets.get(10)),
        "bucket5": buckets.get("5", buckets.get(5)),
        "bucket1": buckets.get("1", buckets.get(1)),
        "bucket0": buckets.get("0", buckets.get(0)),
        "extractable": summary.get("extractable"),
        "by_type": by_type,
    }


def notebook_runtime(path: str | Path) -> dict:
    nb = load_json(ROOT / path)
    durations = []
    for cell in nb.get("cells", []):
        metadata = cell.get("metadata") or {}
        execution = metadata.get("execution") or {}
        papermill = metadata.get("papermill") or {}
        duration = metadata.get("duration", papermill.get("duration", execution.get("duration")))
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
    return {
        "notebook": str(path),
        "total_seconds": sum(durations),
        "total_minutes": sum(durations) / 60,
        "cell_durations_seconds": durations,
    }


def output_behavior(path: str | Path) -> dict:
    outputs = load_json(ROOT / path)
    texts = [item.get("model_output", "") for item in outputs]
    lengths = [len(text) for text in texts]
    return {
        "n": len(texts),
        "mean_len": statistics.mean(lengths) if lengths else 0.0,
        "median_len": statistics.median(lengths) if lengths else 0.0,
        "unique_outputs": len(set(texts)),
        "multiline": sum("\n" in text for text in texts),
        "long_gt_40": sum(len(text) > 40 for text in texts),
        "first3": texts[:3],
    }


def candidate_eval(name: str, path: str | Path, gold: list[dict]) -> dict:
    outputs = load_json(ROOT / path)
    report = evaluate(outputs, gold)
    return {
        "name": name,
        "path": str(path),
        "outputs": outputs,
        "rows": report["rows"],
        "scores": [int(row["score"]) for row in report["rows"]],
        "answers": [row["pred_answer"] for row in report["rows"]],
        "summary": report["summary"],
    }


def pair_delta(base: dict, other: dict, gold: list[dict]) -> dict:
    by_type = defaultdict(lambda: {"delta": 0, "improved_rows": 0, "worse_rows": 0})
    improved = worse = same = delta = 0
    for i, rec in enumerate(gold):
        diff = base["scores"][i] - other["scores"][i]
        delta += diff
        t = rec.get("type")
        by_type[t]["delta"] += diff
        if diff > 0:
            improved += 1
            by_type[t]["improved_rows"] += 1
        elif diff < 0:
            worse += 1
            by_type[t]["worse_rows"] += 1
        else:
            same += 1
    return {
        "base": base["name"],
        "other": other["name"],
        "delta_raw": delta,
        "improved_rows": improved,
        "worse_rows": worse,
        "same_rows": same,
        "by_type": dict(sorted(by_type.items())),
    }


def oracle_score(candidates: list[dict], gold: list[dict]) -> dict:
    raw = 0
    all_same_answer_rows = 0
    for i in range(len(gold)):
        raw += max(cand["scores"][i] for cand in candidates)
        all_same_answer_rows += int(len({cand["answers"][i] for cand in candidates}) == 1)
    return {
        "candidate_names": [cand["name"] for cand in candidates],
        "n_candidates": len(candidates),
        "oracle_raw": raw,
        "oracle_score10": raw / len(gold),
        "all_same_answer_rows": all_same_answer_rows,
    }


def scan_manifest(result_dir: str) -> list[dict]:
    out = []
    for path in sorted((ROOT / result_dir).glob("*manifest*.json")):
        try:
            data = load_json(path)
        except Exception as exc:
            out.append({"path": str(path), "error": repr(exc)})
            continue
        rule = data.get("rule_compliance", {})
        cfg = data.get("config", {})
        out.append(
            {
                "path": str(path.relative_to(ROOT)),
                "uses_original_question_fields": rule.get("uses_original_question_fields"),
                "uses_type_for_prompt_or_routing": rule.get("uses_type_for_prompt_or_routing"),
                "retrieval_basis": rule.get("retrieval_basis"),
                "legal_query_retrieval_enabled": cfg.get("legal_query_retrieval_enabled"),
                "uses_valid_labels_for_checkpoint_selection": rule.get("uses_valid_labels_for_checkpoint_selection"),
            }
        )
    return out


def main() -> None:
    gold = load_records(ROOT / "dataset" / "valid.json")

    run_reports = []
    for name, path in [
        ("v17_epoch7", "results/v17_epoch7/valid_report.json"),
        ("v17_678_selected", "results/v17_678/valid_report.json"),
        ("v17_678_epoch06", "results/v17_678/v17_legal_answer_only_select_678_checkpoint_eval/model_seed_42_epoch_06_valid_report.json"),
        ("v17_678_epoch07", "results/v17_678/v17_legal_answer_only_select_678_checkpoint_eval/model_seed_42_epoch_07_valid_report.json"),
        ("v17_678_epoch08", "results/v17_678/v17_legal_answer_only_select_678_checkpoint_eval/model_seed_42_epoch_08_valid_report.json"),
        ("v16_multiseed", "results/v16_multiseed/valid_report.json"),
        ("v16_query_retrieval", "results/v16_query_retrieval_ranker/valid_report.json"),
        ("v8_select_lr1e3", "results/v8_select_checkpoint/beam2_lr1e-3/valid_report.json"),
        ("v15_ensemble_nonlegal", "results/v15_ensemble/valid_report.json"),
        ("v13_nonlegal", "results/v13/valid_report.json"),
    ]:
        if (ROOT / path).exists():
            run_reports.append(report_summary(name, path, gold))

    candidates = {}
    for name, path in [
        ("v17_epoch7", "results/v17_epoch7/valid_output.json"),
        ("v17_678_e06", "results/v17_678/ensemble_candidates/valid_model_seed_42_epoch_06.json"),
        ("v17_678_e07", "results/v17_678/ensemble_candidates/valid_model_seed_42_epoch_07.json"),
        ("v17_678_e08", "results/v17_678/ensemble_candidates/valid_model_seed_42_epoch_08.json"),
        ("v16_e06", "results/v16_multiseed/ensemble_candidates/valid_model_seed_42_epoch_06.json"),
        ("v16_e07", "results/v16_multiseed/ensemble_candidates/valid_model_seed_42_epoch_07.json"),
        ("v16_e08", "results/v16_multiseed/ensemble_candidates/valid_model_seed_42_epoch_08.json"),
        ("v16_s123e07", "results/v16_multiseed/ensemble_candidates/valid_model_seed_123_epoch_07.json"),
        ("v16_multi_final", "results/v16_multiseed/valid_output.json"),
        ("v8_lr1e3", "results/v8_select_checkpoint/beam2_lr1e-3/valid_output.json"),
        ("v16_query_final", "results/v16_query_retrieval_ranker/valid_output.json"),
    ]:
        if (ROOT / path).exists():
            candidates[name] = candidate_eval(name, path, gold)

    v17_only = [candidates[name] for name in ["v17_epoch7", "v17_678_e06", "v17_678_e07", "v17_678_e08"] if name in candidates]
    answer_only_legal = [
        candidates[name]
        for name in [
            "v17_epoch7",
            "v17_678_e06",
            "v17_678_e07",
            "v17_678_e08",
            "v16_e06",
            "v16_e07",
            "v16_e08",
            "v16_s123e07",
            "v16_multi_final",
            "v8_lr1e3",
        ]
        if name in candidates
    ]
    strict_plus_query_retrieval = answer_only_legal + ([candidates["v16_query_final"]] if "v16_query_final" in candidates else [])

    by_type_compare = {}
    report_by_name = {row["name"]: row for row in run_reports}
    for t in sorted((report_by_name.get("v17_epoch7") or {}).get("by_type", {})):
        row = {}
        for name in ["v17_epoch7", "v17_678_selected", "v16_multiseed", "v8_select_lr1e3"]:
            by_type = (report_by_name.get(name) or {}).get("by_type", {})
            row[name] = by_type.get(t, {}).get("raw_score")
        if row.get("v17_epoch7") is not None and row.get("v8_select_lr1e3") is not None:
            row["delta_v17_epoch7_vs_v8"] = row["v17_epoch7"] - row["v8_select_lr1e3"]
        if row.get("v17_epoch7") is not None and row.get("v16_multiseed") is not None:
            row["delta_v17_epoch7_vs_v16_multiseed"] = row["v17_epoch7"] - row["v16_multiseed"]
        by_type_compare[t] = row

    pairwise = []
    if "v17_epoch7" in candidates:
        for other_name in ["v17_678_e08", "v17_678_e06", "v16_e07", "v16_multi_final", "v8_lr1e3"]:
            if other_name in candidates:
                pairwise.append(pair_delta(candidates["v17_epoch7"], candidates[other_name], gold))

    manifests = {}
    for d in sorted((ROOT / "results").iterdir()):
        if d.is_dir() and d.name.startswith("v"):
            found = scan_manifest(str(d.relative_to(ROOT)))
            if found:
                manifests[d.name] = found

    report = {
        "run_reports": run_reports,
        "notebook_runtime": [
            notebook_runtime("finetune_gpt2_for_math_v17_legal_answer_only_epoch7.ipynb"),
            notebook_runtime("finetune_gpt2_for_math_v17_legal_answer_only_select_678.ipynb"),
            notebook_runtime("finetune_gpt2_for_math_v16_legal_answer_only_multiseed.ipynb"),
            notebook_runtime("finetune_gpt2_for_math_v16_legal_query_retrieval_ranker.ipynb"),
        ],
        "by_type_compare": by_type_compare,
        "candidate_summaries": {
            name: {
                "raw": cand["summary"]["raw_score"],
                "score10": cand["summary"]["score_10"],
                "buckets": cand["summary"]["buckets"],
                "extractable": cand["summary"]["extractable"],
            }
            for name, cand in candidates.items()
        },
        "oracle": {
            "v17_only": oracle_score(v17_only, gold),
            "answer_only_legal_v8_v16_v17": oracle_score(answer_only_legal, gold),
            "strict_legal_plus_query_retrieval": oracle_score(strict_plus_query_retrieval, gold),
        },
        "pairwise_vs_v17_epoch7": pairwise,
        "output_behavior": {
            "v17_epoch7": output_behavior("results/v17_epoch7/valid_output.json"),
            "v17_678": output_behavior("results/v17_678/valid_output.json"),
        },
        "manifests": manifests,
    }

    out_path = ROOT / "audit" / "v17_results_analysis.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
