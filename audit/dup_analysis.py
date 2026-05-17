"""Analyze whether duplicate query_vi in train.json are exact copies or
augmentations with different response_vi."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "dataset" / "train.json"


def load_records(p: Path):
    with p.open("r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        out = []
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out


def main():
    recs = load_records(PATH)
    print(f"total records: {len(recs)}")

    resps = defaultdict(list)   # query -> list of (response, type)
    for r in recs:
        q = (r.get("query_vi") or "").strip()
        a = (r.get("response_vi") or "").strip()
        t = r.get("type")
        resps[q].append((a, t))

    n_unique_q = len(resps)
    n_q_with_dups = sum(1 for v in resps.values() if len(v) > 1)
    n_q_only_exact_copies = 0
    n_q_with_distinct_resp = 0
    extra_copies_exact = 0
    extra_copies_distinct = 0
    types_per_q = []

    for q, items in resps.items():
        if len(items) <= 1:
            continue
        distinct_resps = {a for a, _ in items}
        distinct_types = {t for _, t in items}
        types_per_q.append(len(distinct_types))
        extra = len(items) - 1
        if len(distinct_resps) == 1:
            n_q_only_exact_copies += 1
            extra_copies_exact += extra
        else:
            n_q_with_distinct_resp += 1
            extra_copies_distinct += extra

    print(f"unique queries: {n_unique_q}")
    print(f"queries appearing >= 2 times: {n_q_with_dups}")
    print(f"  of which all copies are EXACT (same response): "
          f"{n_q_only_exact_copies}")
    print(f"  of which have >=2 DISTINCT responses          : "
          f"{n_q_with_distinct_resp}")
    print(f"extra copies (would be dropped by current FILTER):")
    print(f"  exact-duplicate copies   : {extra_copies_exact}")
    print(f"  augmentation-style copies: {extra_copies_distinct}")
    print(f"  total                    : {extra_copies_exact + extra_copies_distinct}")

    # show a couple of augmentation-style examples
    print("\n=== examples of queries with DISTINCT responses ===")
    shown = 0
    for q, items in resps.items():
        if shown >= 3:
            break
        resp_set = {a for a, _ in items}
        if len(resp_set) > 1:
            print("-" * 80)
            print("QUERY:", q[:200])
            for i, (a, t) in enumerate(items[:3]):
                print(f"  ({i}) type={t}")
                print(f"      response[:160]= {a[:160]}")
            shown += 1


if __name__ == "__main__":
    main()
