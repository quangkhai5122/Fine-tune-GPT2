"""Re-score one prediction file with the shared standard evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import load_records, save_evaluation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path, help="Gold JSON/JSONL file.")
    parser.add_argument("--pred", required=True, type=Path, help="Prediction JSON file.")
    parser.add_argument("--out", required=True, type=Path, help="Where to write the standard report JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_records = load_records(args.gold)
    result = save_evaluation_report(args.pred, gold_records, args.out)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
