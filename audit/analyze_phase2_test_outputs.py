from __future__ import annotations

import ast
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import extract_pred, rel_error, score_one

DATASET_TEST = ROOT / "dataset" / "test.json"
V22_NOTEBOOK = ROOT / "finetune_gpt2_for_math_v22_solver_v17_lr3e3_epoch7_fallback.ipynb"
OUT_JSON = ROOT / "audit" / "phase2_test_output_analysis.json"
OUT_MD = ROOT / "audit" / "phase2_test_output_analysis.md"

RUNS = {
    "v17_phase2": ROOT / "results" / "v17_phase2",
    "v21_phase2": ROOT / "results" / "v21_phase2",
    "v22_v17": ROOT / "results" / "v22_v17",
    "v22_v21": ROOT / "results" / "v22_v21",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(run_dir: Path) -> list[dict]:
    path = run_dir / "test_predictions.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_json(path)


def extract_v22_solver_namespace() -> dict:
    nb = load_json(V22_NOTEBOOK)
    src = "".join(nb["cells"][8]["source"])
    start = src.find("# ============================================================\n# V22. Deterministic legal template solver overlay")
    end = src.find("\ndef run_v22_solver_v17_test", start)
    if start < 0 or end < 0:
        raise RuntimeError("Cannot locate V22 solver block")
    code = src[start:end]
    ast.parse(code)
    ns: dict = {
        "LEGAL_INPUT_FIELDS": ["query_vi"],
        "LEGAL_TARGET_FIELDS": ["response_vi"],
        "DISALLOWED_MODEL_FEATURE_FIELDS": ["type", "original_question_en", "original_question_vi"],
    }
    exec(code, ns)
    return ns


def pred_map(items: list[dict]) -> dict[str, dict]:
    return {str(item.get("id")): item for item in items}


def numeric_of_output(item: dict) -> tuple[str | None, float | None]:
    answer, num = extract_pred(item)
    return answer, num


def summarize_run(name: str, items: list[dict], test_records: list[dict]) -> dict:
    ids = [str(item.get("id")) for item in items]
    output_texts = [str(item.get("model_output", "")) for item in items]
    answers = [numeric_of_output(item) for item in items]
    extractable = [ans is not None for ans, _ in answers]
    numeric = [num is not None and math.isfinite(float(num)) for _, num in answers]
    lens = [len(t) for t in output_texts]
    digit_count = sum(bool(re.search(r"\d", text)) for text in output_texts)
    anchor_count = sum(bool(re.search(r"đáp\s*án|dap\s*an|answer|####", text, re.I)) for text in output_texts)
    type_counts = Counter((item.get("type") or "<missing>") for item in items)
    ans_counts = Counter((ans or "<none>") for ans, _ in answers)
    return {
        "name": name,
        "n": len(items),
        "expected_n": len(test_records),
        "unique_ids": len(set(ids)),
        "missing_ids": [str(rec.get("id")) for rec in test_records if str(rec.get("id")) not in set(ids)][:20],
        "duplicate_id_count": len(ids) - len(set(ids)),
        "extractable": sum(extractable),
        "numeric": sum(numeric),
        "digit_count": digit_count,
        "anchor_count": anchor_count,
        "length": {
            "min": min(lens) if lens else None,
            "median": median(lens) if lens else None,
            "mean": mean(lens) if lens else None,
            "max": max(lens) if lens else None,
        },
        "type_counts": dict(type_counts.most_common()),
        "top_answers": ans_counts.most_common(20),
        "samples": [
            {
                "id": item.get("id"),
                "type": item.get("type"),
                "model_output": str(item.get("model_output", ""))[:160],
                "answer": answers[i][0],
                "num": answers[i][1],
            }
            for i, item in enumerate(items[:8])
        ],
    }


def build_solver_labels(test_records: list[dict]) -> dict[str, dict]:
    ns = extract_v22_solver_namespace()
    labels = {}
    for idx, rec in enumerate(test_records):
        answer, rule = ns["v22_solve_query"](rec.get("query_vi", ""))
        if answer is None:
            continue
        item = {
            "id": rec.get("id", idx),
            "query_vi": rec.get("query_vi", ""),
            "type": rec.get("type"),
            "model_output": ns["v22_model_output_from_answer"](answer),
        }
        ans, num = numeric_of_output(item)
        labels[str(rec.get("id", idx))] = {
            "id": rec.get("id", idx),
            "type": rec.get("type"),
            "rule": rule,
            "answer": ans,
            "num": num,
            "query_vi": rec.get("query_vi", ""),
        }
    return labels


def compare_to_solver(run_name: str, items: list[dict], solver_labels: dict[str, dict]) -> dict:
    by_id = pred_map(items)
    rows = []
    raw = 0
    buckets = Counter()
    exact = 0
    numeric_pairs = 0
    for sid, label in solver_labels.items():
        pred = by_id.get(sid)
        if not pred:
            continue
        pred_ans, pred_num = numeric_of_output(pred)
        err = rel_error(pred_num, label["num"])
        extractable = pred_ans is not None
        score = score_one(err, extractable)
        raw += score
        buckets[score] += 1
        exact += int(score == 10)
        numeric_pairs += int(pred_num is not None and label["num"] is not None)
        rows.append(
            {
                "id": label["id"],
                "type": label["type"],
                "rule": label["rule"],
                "solver_answer": label["answer"],
                "solver_num": label["num"],
                "pred_answer": pred_ans,
                "pred_num": pred_num,
                "rel_error": err,
                "score_vs_solver": score,
                "query_vi": label["query_vi"],
                "model_output": pred.get("model_output"),
            }
        )

    by_type = defaultdict(lambda: Counter(n=0, raw=0, exact10=0, zero=0))
    by_rule = defaultdict(lambda: Counter(n=0, raw=0, exact10=0, zero=0))
    for row in rows:
        for bucket in (by_type[row["type"]], by_rule[row["rule"]]):
            bucket["n"] += 1
            bucket["raw"] += row["score_vs_solver"]
            bucket["exact10"] += int(row["score_vs_solver"] == 10)
            bucket["zero"] += int(row["score_vs_solver"] == 0)

    def compact(counter_map):
        out = {}
        for key, c in counter_map.items():
            n = c["n"]
            out[str(key)] = {
                "n": n,
                "raw": c["raw"],
                "score10": c["raw"] / n if n else None,
                "exact10": c["exact10"],
                "zero": c["zero"],
            }
        return dict(sorted(out.items(), key=lambda kv: (-kv[1]["n"], kv[0])))

    return {
        "run": run_name,
        "n_solver_labeled": len(rows),
        "raw_vs_solver": raw,
        "score10_vs_solver": raw / len(rows) if rows else None,
        "exact10_vs_solver": exact,
        "numeric_pairs": numeric_pairs,
        "buckets": dict(sorted(buckets.items(), reverse=True)),
        "by_type": compact(by_type),
        "by_rule": compact(by_rule),
        "worst_examples": [r for r in rows if r["score_vs_solver"] == 0][:40],
        "partial_examples": [r for r in rows if r["score_vs_solver"] in (1, 5)][:30],
    }


def compare_runs(predictions: dict[str, list[dict]], test_records: list[dict]) -> dict:
    maps = {name: pred_map(items) for name, items in predictions.items()}
    run_names = list(predictions)
    pairwise = {}
    for i, a in enumerate(run_names):
        for b in run_names[i + 1 :]:
            same_text = 0
            same_num = 0
            both_numeric = 0
            rows_changed = []
            for rec in test_records:
                sid = str(rec.get("id"))
                ia, ib = maps[a].get(sid), maps[b].get(sid)
                if not ia or not ib:
                    continue
                ans_a, num_a = numeric_of_output(ia)
                ans_b, num_b = numeric_of_output(ib)
                same_text += int(str(ia.get("model_output")) == str(ib.get("model_output")))
                if num_a is not None and num_b is not None:
                    both_numeric += 1
                    same_num += int(abs(num_a - num_b) <= 1e-9)
                if num_a != num_b and len(rows_changed) < 30:
                    rows_changed.append(
                        {
                            "id": rec.get("id"),
                            "type": rec.get("type"),
                            a: {"answer": ans_a, "num": num_a},
                            b: {"answer": ans_b, "num": num_b},
                            "query_vi": rec.get("query_vi"),
                        }
                    )
            pairwise[f"{a}__vs__{b}"] = {
                "same_text": same_text,
                "same_num": same_num,
                "both_numeric": both_numeric,
                "n": len(test_records),
                "changed_examples": rows_changed,
            }
    return pairwise


def read_summary_files(run_dir: Path) -> dict:
    out = {}
    for name in [
        "ensemble_or_gate_report.json",
        "selected_checkpoint_info.json",
        "checkpoint_selection_report.json",
        "selected_valid_report.json",
    ]:
        path = run_dir / name
        if not path.exists():
            continue
        try:
            data = load_json(path)
        except Exception as exc:
            out[name] = {"error": repr(exc)}
            continue
        if "summary" in data:
            out[name] = {"summary": data["summary"]}
        else:
            out[name] = {
                k: data.get(k)
                for k in [
                    "strategy",
                    "fallback_strategy",
                    "source_counts",
                    "solver_rule_counts",
                    "selected_candidate",
                    "selected_epoch",
                    "selected_profile",
                    "uses_valid_labels_for_checkpoint_selection",
                    "uses_valid_labels_for_verifier_profile_selection",
                    "output_sanity",
                ]
                if k in data
            }
            if name == "checkpoint_selection_report.json":
                rows = data.get("candidates") or []
                out[name]["num_candidates"] = len(rows)
                out[name]["candidate_summaries"] = [
                    {
                        "name": row.get("name"),
                        "epoch": row.get("epoch"),
                        "profile": row.get("profile", {}).get("name") if isinstance(row.get("profile"), dict) else row.get("profile"),
                        "summary": row.get("summary"),
                    }
                    for row in rows[:8]
                ]
    return out


def main() -> None:
    test_records = load_json(DATASET_TEST)
    predictions = {name: load_predictions(path) for name, path in RUNS.items()}
    solver_labels = build_solver_labels(test_records)

    analysis = {
        "test_n": len(test_records),
        "solver_label_n": len(solver_labels),
        "run_summaries": {
            name: summarize_run(name, items, test_records)
            for name, items in predictions.items()
        },
        "summary_files": {
            name: read_summary_files(path)
            for name, path in RUNS.items()
        },
        "vs_solver": {
            name: compare_to_solver(name, items, solver_labels)
            for name, items in predictions.items()
        },
        "pairwise": compare_runs(predictions, test_records),
    }

    OUT_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Phase 2 Test Output Analysis",
        "",
        f"- test_n: {analysis['test_n']}",
        f"- v22_solver_labeled_rows: {analysis['solver_label_n']}",
        "",
        "## Output Sanity",
        "| run | n | unique_ids | extractable | numeric | digit | anchor | len_median | top answers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, s in analysis["run_summaries"].items():
        top = ", ".join(f"{a}:{c}" for a, c in s["top_answers"][:5])
        lines.append(
            f"| {name} | {s['n']} | {s['unique_ids']} | {s['extractable']} | {s['numeric']} | "
            f"{s['digit_count']} | {s['anchor_count']} | {s['length']['median']} | {top} |"
        )

    lines.extend(
        [
            "",
            "## Agreement With V22 Solver Labels",
            "This is not official scoring. It only compares runs on rows where the deterministic v22 solver produced a high-confidence query-derived answer.",
            "",
            "| run | n | raw_vs_solver | score10_vs_solver | exact10 | buckets |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name, s in analysis["vs_solver"].items():
        lines.append(
            f"| {name} | {s['n_solver_labeled']} | {s['raw_vs_solver']} | "
            f"{s['score10_vs_solver']:.3f} | {s['exact10_vs_solver']} | {s['buckets']} |"
        )

    lines.extend(["", "## Source / Selection Metadata"])
    for name, files in analysis["summary_files"].items():
        lines.append(f"### {name}")
        for file_name, data in files.items():
            compact = json.dumps(data, ensure_ascii=False)
            lines.append(f"- {file_name}: {compact[:1200]}")

    lines.extend(["", "## Pairwise Numeric Agreement"])
    for pair, info in analysis["pairwise"].items():
        lines.append(
            f"- {pair}: same_num={info['same_num']}/{info['both_numeric']}, same_text={info['same_text']}/{info['n']}"
        )

    lines.extend(["", "## Model Failures Against Solver Labels"])
    for run_name in ["v17_phase2", "v21_phase2"]:
        comp = analysis["vs_solver"][run_name]
        lines.append(f"### {run_name}")
        for row in comp["worst_examples"][:15]:
            lines.append(
                f"- id={row['id']} type={row['type']} rule={row['rule']} "
                f"solver={row['solver_answer']} pred={row['pred_answer']} | {row['query_vi'][:180]}"
            )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote] {OUT_JSON.relative_to(ROOT)}")
    print(f"[wrote] {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
