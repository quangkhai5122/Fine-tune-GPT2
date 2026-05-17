"""Build a fair comparison table from a manifest of experiment runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import load_records, save_evaluation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args()


def _resolve(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _fmt(value: Any) -> str:
    return "NA" if value is None else str(value)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    gold_path = _resolve(manifest["gold_path"])
    gold_records = load_records(gold_path)

    rows = []
    for run in manifest["runs"]:
        pred_path = _resolve(run.get("pred_path"))
        report_path = _resolve(run.get("standard_report_path"))
        standard_summary = None
        availability = "missing_pred"

        if pred_path and pred_path.exists():
            availability = "available"
            if report_path is None:
                report_path = pred_path.with_name(pred_path.stem + "_standard_report.json")
            result = save_evaluation_report(pred_path, gold_records, report_path)
            standard_summary = result["summary"]

        rows.append(
            {
                "run_id": run["run_id"],
                "family": run.get("family"),
                "checkpoint": run.get("checkpoint"),
                "decode_mode": run.get("decode_mode"),
                "pred_path": str(pred_path) if pred_path else None,
                "availability": availability,
                "legacy_raw_score": run.get("legacy_raw_score"),
                "legacy_note": run.get("legacy_note"),
                "standard_raw_score": None if standard_summary is None else standard_summary["raw_score"],
                "standard_score_10": None if standard_summary is None else standard_summary["score_10"],
                "extractable": None if standard_summary is None else standard_summary["extractable"],
                "numeric_pairs": None if standard_summary is None else standard_summary["numeric_pairs"],
                "bucket_10": None if standard_summary is None else standard_summary["buckets"].get(10),
                "bucket_5": None if standard_summary is None else standard_summary["buckets"].get(5),
                "bucket_1": None if standard_summary is None else standard_summary["buckets"].get(1),
                "bucket_0": None if standard_summary is None else standard_summary["buckets"].get(0),
                "notes": run.get("notes"),
            }
        )

    payload = {
        "gold_path": str(gold_path),
        "evaluator": {
            "anchor_policy": "last_explicit_anchor_then_last_boxed",
            "numeric_parser": "robust_scalar_math_parser",
        },
        "rows": rows,
    }
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Fair run comparison",
        "",
        "Evaluator: `last explicit answer anchor` + robust scalar parser.",
        "",
        "| run | family | checkpoint | decode | raw standard | score/10 | exact10 | extractable | numeric pairs | legacy raw | status |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row["run_id"],
                    _fmt(row["family"]),
                    _fmt(row["checkpoint"]),
                    _fmt(row["decode_mode"]),
                    _fmt(row["standard_raw_score"]),
                    _fmt(None if row["standard_score_10"] is None else f"{row['standard_score_10']:.3f}"),
                    _fmt(row["bucket_10"]),
                    _fmt(row["extractable"]),
                    _fmt(row["numeric_pairs"]),
                    _fmt(row["legacy_raw_score"]),
                    row["availability"],
                ]
            )
            + " |"
        )

    md_lines.extend(["", "## Notes"])
    for row in rows:
        note_parts = [part for part in [row.get("legacy_note"), row.get("notes")] if part]
        if note_parts:
            md_lines.append(f"- **{row['run_id']}**: {' '.join(note_parts)}")

    args.out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
