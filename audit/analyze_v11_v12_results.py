from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from math_eval import evaluate, load_records

OUT_JSON = ROOT / "audit" / "v11_v12_results_analysis.json"
OUT_MD = ROOT / "audit" / "v11_v12_results_analysis.md"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_summary(summary: dict) -> dict:
    return {
        "raw_score": summary.get("raw_score"),
        "score_10": summary.get("score_10"),
        "extractable": summary.get("extractable"),
        "buckets": summary.get("buckets"),
        "rel_error_mean": summary.get("rel_error_mean"),
    }


def by_type(report: dict) -> dict:
    return {
        k: {
            "n": v.get("n"),
            "raw_score": v.get("raw_score"),
            "score_10": v.get("score_10"),
            "exact10": v.get("bucket_10"),
            "extractable": v.get("extractable"),
        }
        for k, v in sorted(report.get("by_type", {}).items())
    }


def load_run(run: str) -> dict:
    root = ROOT / "results" / run
    manifest = root / (
        "v11_answer_only_overlap_valid_manifest.json"
        if run == "v11"
        else "v12_hybrid_retrieval_overlap_valid_manifest.json"
    )
    out = {
        "manifest": load_json(manifest),
        "split": load_json(root / "query_disjoint_split_report.json"),
        "checkpoint": load_json(root / "checkpoint_selection_report.json"),
        "overlap_valid_report": load_json(root / "overlap_valid_report.json"),
        "overlap_valid_output": load_json(root / "overlap_valid_output.json"),
        "model_overlap_valid_output": load_json(root / "model_overlap_valid_output.json"),
        "valid_report": load_json(root / "valid_report.json"),
        "valid_output": load_json(root / "valid_output.json"),
        "model_valid_output": load_json(root / "model_valid_output.json"),
    }
    model_report = root / "model_valid_report.json"
    if model_report.exists():
        out["model_valid_report"] = load_json(model_report)
    return out


def gold_from_report(report: dict) -> list[dict]:
    return [
        {
            "id": row.get("id"),
            "type": row.get("type"),
            "response_vi": f"#### {row.get('gold_answer')}",
        }
        for row in report.get("rows", [])
    ]


def gate_simulations(run_data: dict, split: str) -> dict:
    """Use V12 model output as fallback and V12 hybrid output where allowed."""
    if split == "valid":
        gold = load_records(ROOT / "dataset" / "valid.json")
        hybrid_out = run_data["valid_output"]
        model_out = run_data["model_valid_output"]
        hybrid_rows = run_data["valid_report"]["rows"]
    elif split == "overlap_valid":
        gold = gold_from_report(run_data["overlap_valid_report"])
        hybrid_out = run_data["overlap_valid_output"]
        model_out = run_data["model_overlap_valid_output"]
        hybrid_rows = run_data["overlap_valid_report"]["rows"]
    else:
        raise ValueError(f"Unknown split: {split}")

    gates = {
        "current_all_retrievable": None,
        "retrieval_rephrased_ansaug_only": {"GSM_Rephrased", "MATH_Rephrased", "GSM_AnsAug", "MATH_AnsAug"},
        "retrieval_no_foobar": {"GSM_Rephrased", "MATH_Rephrased", "GSM_AnsAug", "MATH_AnsAug", "GSM_SV", "MATH_SV"},
        "retrieval_rephrased_only": {"GSM_Rephrased", "MATH_Rephrased"},
        "retrieval_rephrased_ansaug_plus_math_sv": {
            "GSM_Rephrased",
            "MATH_Rephrased",
            "GSM_AnsAug",
            "MATH_AnsAug",
            "MATH_SV",
        },
    }
    out = {}
    for name, allowed in gates.items():
        if allowed is None:
            rows = hybrid_out
        else:
            rows = [hybrid_out[i] if hybrid_rows[i].get("type") in allowed else model_out[i] for i in range(len(hybrid_out))]
        rep = evaluate(rows, gold)
        out[name] = compact_summary(rep["summary"])
    return out


def model_to_hybrid_delta(run_data: dict, split: str) -> dict:
    if split == "valid":
        model_rep = run_data["model_valid_report"]
        hybrid_rep = run_data["valid_report"]
        model_out = run_data["model_valid_output"]
        hybrid_out = run_data["valid_output"]
    elif split == "overlap_valid":
        model_rep = evaluate(run_data["model_overlap_valid_output"], gold_from_report(run_data["overlap_valid_report"]))
        hybrid_rep = run_data["overlap_valid_report"]
        model_out = run_data["model_overlap_valid_output"]
        hybrid_out = run_data["overlap_valid_output"]
    else:
        raise ValueError(f"Unknown split: {split}")
    model_rows = model_rep["rows"]
    hybrid_rows = hybrid_rep["rows"]
    changed = [i for i in range(len(model_out)) if model_out[i].get("model_output") != hybrid_out[i].get("model_output")]
    by = defaultdict(lambda: {"n": 0, "changed": 0, "delta": 0, "model_raw": 0, "hybrid_raw": 0})
    changed_set = set(changed)
    for i, row in enumerate(hybrid_rows):
        t = row.get("type") or "unknown"
        by[t]["n"] += 1
        by[t]["model_raw"] += model_rows[i]["score"]
        by[t]["hybrid_raw"] += hybrid_rows[i]["score"]
        if i in changed_set:
            by[t]["changed"] += 1
            by[t]["delta"] += hybrid_rows[i]["score"] - model_rows[i]["score"]
    transitions = Counter((model_rows[i]["score"], hybrid_rows[i]["score"]) for i in range(len(model_rows)))
    return {
        "changed_outputs": len(changed),
        "delta_raw_on_changed": sum(hybrid_rows[i]["score"] - model_rows[i]["score"] for i in changed),
        "by_type": dict(sorted((k, dict(v)) for k, v in by.items())),
        "score_transitions": {f"{k[0]}->{k[1]}": v for k, v in sorted(transitions.items(), key=lambda kv: (-kv[1], kv[0]))},
    }


def main() -> None:
    runs = {run: load_run(run) for run in ["v11", "v12"]}
    analysis = {
        "runs": {},
        "v12_gate_simulations": {
            "valid": gate_simulations(runs["v12"], "valid"),
            "overlap_valid": gate_simulations(runs["v12"], "overlap_valid"),
        },
        "v12_delta": {
            "valid": model_to_hybrid_delta(runs["v12"], "valid"),
            "overlap_valid": model_to_hybrid_delta(runs["v12"], "overlap_valid"),
        },
    }
    for run, data in runs.items():
        ckpt = data["checkpoint"]
        analysis["runs"][run] = {
            "config": {
                k: data["manifest"]["config"].get(k)
                for k in [
                    "stage_a_epochs",
                    "stage_a_lr",
                    "prompt_template",
                    "use_hybrid_retrieval",
                    "retrieval_strategy",
                    "retrieval_min_majority_frac",
                    "lora_target_modules",
                ]
            },
            "split": data["split"],
            "selected_checkpoint": {
                "label": ckpt.get("selected", {}).get("label"),
                "eval_n": ckpt.get("eval_n"),
                "summary": compact_summary(ckpt.get("selected", {}).get("summary", {})),
                "all": [
                    {
                        "label": item.get("label"),
                        "raw_score": item.get("summary", {}).get("raw_score"),
                        "score_10": item.get("summary", {}).get("score_10"),
                        "exact10": item.get("summary", {}).get("buckets", {}).get("10"),
                    }
                    for item in ckpt.get("all_checkpoints", [])
                ],
            },
            "overlap_valid": compact_summary(data["overlap_valid_report"]["summary"]),
            "valid": compact_summary(data["valid_report"]["summary"]),
            "valid_by_type": by_type(data["valid_report"]),
            "overlap_valid_by_type": by_type(data["overlap_valid_report"]),
        }
        if "model_valid_report" in data:
            analysis["runs"][run]["model_valid"] = compact_summary(data["model_valid_report"]["summary"])
            analysis["runs"][run]["model_valid_by_type"] = by_type(data["model_valid_report"])

    OUT_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# V11/V12 Results Analysis", "", "## Summary"]
    for run in ["v11", "v12"]:
        item = analysis["runs"][run]
        lines.append(
            f"- {run}: overlap={item['overlap_valid']['score_10']:.3f}, "
            f"valid={item['valid']['score_10']:.3f}, selected={item['selected_checkpoint']['label']}"
        )
        if "model_valid" in item:
            lines.append(f"  - model-only valid={item['model_valid']['score_10']:.3f}")
    for split, simulations in analysis["v12_gate_simulations"].items():
        lines += ["", f"## V12 Gate Simulations: {split}"]
        for name, summary in simulations.items():
            lines.append(f"- {name}: score={summary['score_10']:.3f}, raw={summary['raw_score']}, exact10={summary['buckets'].get(10, summary['buckets'].get('10'))}")
    lines += ["", "## V12 Hybrid Delta By Type: valid"]
    for t, d in analysis["v12_delta"]["valid"]["by_type"].items():
        lines.append(f"- {t}: model_raw={d['model_raw']}, hybrid_raw={d['hybrid_raw']}, delta={d['hybrid_raw'] - d['model_raw']}, changed={d['changed']}/{d['n']}")
    lines += ["", "## V12 Hybrid Delta By Type: overlap_valid"]
    for t, d in analysis["v12_delta"]["overlap_valid"]["by_type"].items():
        lines.append(f"- {t}: model_raw={d['model_raw']}, hybrid_raw={d['hybrid_raw']}, delta={d['hybrid_raw'] - d['model_raw']}, changed={d['changed']}/{d['n']}")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
