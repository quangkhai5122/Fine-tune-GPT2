"""Tiny smoke tests for the shared evaluator/parser."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_eval import extract_answer, parse_number


def main() -> None:
    assert parse_number(r"\frac{9}{20}") == 0.45
    assert parse_number(r"36\pi") == 36 * math.pi
    assert parse_number(r"10^2") == 100.0
    assert parse_number(r"-\frac{3}{2}") == -1.5
    assert parse_number("(0,4)") is None
    assert parse_number("2x-8") is None

    text = "Trung gian #### 12\nKết luận cuối cùng. Đáp án là: 37"
    assert extract_answer(text) == "37"

    print("math_eval smoke tests passed")


if __name__ == "__main__":
    main()
