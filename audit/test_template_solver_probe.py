from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path

from analyze_test_release import mask_numbers, normalize_text, number_signature


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "dataset" / "test.json"
OUT_JSON = ROOT / "audit" / "test_template_solver_probe.json"
OUT_MD = ROOT / "audit" / "test_template_solver_probe.md"


def as_number(token: str) -> Fraction:
    token = token.replace(",", ".")
    if "/" in token:
        return Fraction(token)
    return Fraction(token)


def fmt(value: Fraction | float | int) -> str:
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, int):
        return str(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.10g}"


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def solve_combination(k: int, c: int) -> int | None:
    for n in range(k, 10000):
        if math.comb(n, k) == c:
            return n
        if math.comb(n, k) > c and n > k:
            return None
    return None


def solve_query(query: str) -> tuple[str | None, str | None]:
    q = normalize_text(query)
    nums_raw = list(number_signature(query))
    nums = [as_number(x) for x in nums_raw]

    if q.startswith("trong một lớp, có") and "thích bóng đá" in q and "thích bóng rổ" in q:
        return fmt(nums[0] + nums[1] - nums[2]), "set_union_two_sports"

    if q.startswith("có") and "quả cam" in q and "mỗi hộp đựng được" in q and "còn thừa" in q:
        return fmt(int(nums[0]) % int(nums[1])), "orange_remainder"

    if "\\lfloor" in q and "\\rceil" in q:
        return fmt(math.floor(float(nums[0])) + math.ceil(float(nums[1]))), "floor_plus_ceil"

    if q.startswith("lan mua") and "quyển sách" in q and "cây bút" in q and "tổng tiền" in q:
        return fmt((nums[3] - nums[0] * nums[1]) / nums[2]), "books_pens_linear"

    if "miễn phí giao hàng" in q and "mỗi sản phẩm giá" in q:
        return fmt(math.ceil(float(nums[0] / nums[1]))), "free_shipping_ceil"

    if q.startswith("ba điểm kiểm tra") and "điểm trung bình bốn bài" in q:
        return fmt(4 * nums[3] - nums[0] - nums[1] - nums[2]), "fourth_score_average"

    if q.startswith("một kho có") and "ngày thứ nhất dùng" in q and "ngày thứ hai dùng" in q:
        return fmt(nums[0] * (1 - nums[1]) * (1 - nums[2])), "rice_remaining_after_two_days"

    if q.startswith("một cửa hàng ban đầu có") and "sản phẩm bị hỏng" in q:
        return fmt(nums[0] - nums[1] - nums[2]), "good_products_remaining"

    if q.startswith("một giỏ hàng có") and "giá trung bình" in q:
        return fmt((nums[0] * nums[1] + nums[2] * nums[3]) / (nums[0] + nums[2])), "weighted_average_cart"

    if "được" in q and "feet" in q and "yard" in q and "chỉ hỏi yard" in q:
        return fmt(nums[0] / nums[3]), "feet_to_yard"

    if q.startswith("một số lượng ban đầu") and "tăng gấp" in q and "đã qua bao nhiêu năm" in q:
        start, factor, target = float(nums[0]), float(nums[1]), float(nums[2])
        if start > 0 and factor > 0 and factor != 1:
            n = round(math.log(target / start, factor))
            if abs(start * (factor**n) - target) <= 1e-6:
                return fmt(n), "geometric_years"

    if q.startswith("một hộp có") and "bóng bị hỏng" in q and "xác suất" in q:
        return fmt((nums[0] - nums[1]) / nums[0]), "probability_not_defective"

    if q.startswith("một khoản tiền tăng") and "số tiền ban đầu" in q:
        r, years, final = float(nums[0]) / 100.0, int(nums[1]), float(nums[2])
        return fmt(final / ((1 + r) ** years)), "compound_growth_initial"

    if q.startswith("một khoản tiền gửi theo lãi đơn") and "tiền gốc ban đầu" in q:
        rate, years, interest = nums[0] / 100, nums[1], nums[2]
        if rate and years:
            return fmt(interest / (rate * years)), "simple_interest_principal"

    if q.startswith("một trò chơi có") and "còn cách ô cuối" in q:
        return fmt(nums[0] - (nums[1] + nums[2] - nums[3] + nums[4])), "board_game_remaining"

    if q.startswith("số táo và số cam có tỉ lệ") and "chênh lệch" in q:
        return fmt(abs(nums[0] - nums[1]) * nums[2] / (nums[0] + nums[1])), "ratio_difference"

    if q.startswith("một nhóm có n người") and "số cách chọn" in q:
        n = solve_combination(int(nums[0]), int(nums[1]))
        if n is not None:
            return fmt(n), "combination_inverse"

    if q.startswith("sau khi giảm giá") and "giá ban đầu" in q:
        return fmt(nums[1] / (1 - nums[0] / 100)), "discount_original_price"

    if q.startswith("một cấp số cộng") and "tổng" in q:
        a, d, n = nums[0], nums[1], nums[2]
        return fmt(n * (2 * a + (n - 1) * d) / 2), "arithmetic_sequence_sum"

    if q.startswith("xác định giá trị lớn nhất trong các bội chung nhỏ nhất"):
        base = int(nums[0])
        return fmt(max(lcm(base, int(x)) for x in nums[1:])), "max_lcm_with_base"

    if q.startswith("một tam giác cân") and "chu vi" in q:
        return fmt(2 * nums[0] + nums[1]), "isosceles_perimeter"

    m = re.search(r"biết\s+(\d+)x\s*\+\s*([+-]?\d+(?:\.\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?)", q)
    if m and "hỏi x bằng bao nhiêu" in q:
        a, b, c = Fraction(m.group(1)), Fraction(m.group(2)), Fraction(m.group(3))
        return fmt((c - b) / a), "linear_equation_ax_plus_b"

    if q.startswith("một con xúc xắc công bằng") and "giá trị kỳ vọng" in q:
        return fmt((nums[0] + nums[1]) / 2), "fair_die_expectation"

    return None, None


def main() -> None:
    records = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    rows = []
    for rec in records:
        ans, rule = solve_query(rec.get("query_vi", ""))
        rows.append(
            {
                "id": rec.get("id"),
                "type": rec.get("type"),
                "rule": rule,
                "answer": ans,
                "query_vi": rec.get("query_vi"),
                "masked_template": mask_numbers(rec.get("query_vi")),
            }
        )

    solved = [r for r in rows if r["answer"] is not None]
    by_rule: dict[str, int] = {}
    by_type: dict[str, dict[str, int]] = {}
    for r in rows:
        t = r["type"]
        by_type.setdefault(t, {"n": 0, "solved": 0})
        by_type[t]["n"] += 1
        if r["answer"] is not None:
            by_type[t]["solved"] += 1
            by_rule[r["rule"]] = by_rule.get(r["rule"], 0) + 1

    payload = {
        "n": len(rows),
        "solved": len(solved),
        "coverage": len(solved) / len(rows),
        "by_rule": dict(sorted(by_rule.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_type": by_type,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Test Template Solver Probe",
        "",
        f"- n={payload['n']}",
        f"- solved={payload['solved']} ({payload['coverage']:.1%})",
        "",
        "## By Rule",
    ]
    for k, v in payload["by_rule"].items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## By Type"])
    for k, v in sorted(by_type.items()):
        lines.append(f"- {k}: {v['solved']}/{v['n']}")
    lines.extend(["", "## Sample Solved"])
    for r in solved[:30]:
        lines.append(f"- id={r['id']} type={r['type']} rule={r['rule']} answer={r['answer']} | {r['query_vi']}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote] {OUT_JSON.relative_to(ROOT)}")
    print(f"[wrote] {OUT_MD.relative_to(ROOT)}")
    print("solved", payload["solved"], "of", payload["n"])


if __name__ == "__main__":
    main()
