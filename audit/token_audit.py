"""Tokenizer-aware audit for the math fine-tune task.

Uses the local NlpHUST/gpt2-vietnamese tokenizer (../GPT2_vietnamese)
to compute actual token-length distributions on a sample.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "GPT2_vietnamese"
OUT = ROOT / "audit" / "token_audit.txt"

PROMPT_TEMPLATE = "Câu hỏi: {q}\nLời giải: "
SAFE_EOS_ID = 50256


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        return json.load(f) if head == "[" else [json.loads(ln) for ln in f if ln.strip()]


def percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    n = len(v)
    def p(q):
        return v[min(n - 1, int(q * (n - 1)))]
    return {
        "n": n, "min": v[0], "p50": p(0.50), "p75": p(0.75),
        "p90": p(0.90), "p95": p(0.95), "p99": p(0.99), "max": v[-1],
        "mean": round(mean(v), 1), "median": median(v),
    }


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    tok.pad_token_id = SAFE_EOS_ID
    tok.eos_token_id = SAFE_EOS_ID

    train = load_records(DATA_DIR / "train.json")
    valid = load_records(DATA_DIR / "valid.json")

    rng = random.Random(0)
    sample = rng.sample(train, k=10000)  # 10k for speed

    p_lens, r_lens, total_lens = [], [], []
    truncated_at_512 = 0
    truncated_at_768 = 0
    truncated_at_1024 = 0
    over_token_ids = 0  # token-id > 50256
    type_lens = {}

    for rec in sample:
        q = (rec.get("query_vi") or "").strip()
        r = (rec.get("response_vi") or "").strip()
        t = rec.get("type") or "UNK"

        prompt = PROMPT_TEMPLATE.format(q=q)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(r, add_special_tokens=False)["input_ids"] + [SAFE_EOS_ID]

        if any(tid > SAFE_EOS_ID for tid in p_ids + r_ids):
            over_token_ids += 1

        total = len(p_ids) + len(r_ids)
        p_lens.append(len(p_ids))
        r_lens.append(len(r_ids))
        total_lens.append(total)
        truncated_at_512 += int(total > 512)
        truncated_at_768 += int(total > 768)
        truncated_at_1024 += int(total > 1024)

        type_lens.setdefault(t, []).append(total)

    valid_total = []
    for rec in valid:
        q = (rec.get("query_vi") or "").strip()
        r = (rec.get("response_vi") or "").strip()
        prompt = PROMPT_TEMPLATE.format(q=q)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(r, add_special_tokens=False)["input_ids"] + [SAFE_EOS_ID]
        valid_total.append(len(p_ids) + len(r_ids))

    lines = []
    lines.append(f"# Tokenizer audit (sample={len(sample)}/{len(train)} of train; full valid)")
    lines.append(f"prompt template: {PROMPT_TEMPLATE!r}")
    lines.append(f"vocab size: {tok.vocab_size} | pad/eos = {SAFE_EOS_ID}")
    lines.append(f"records with token-id > {SAFE_EOS_ID}: {over_token_ids}")
    lines.append("")
    lines.append("## Train sample - prompt token lens")
    lines.append(json.dumps(percentiles(p_lens), indent=2))
    lines.append("## Train sample - response (+EOS) token lens")
    lines.append(json.dumps(percentiles(r_lens), indent=2))
    lines.append("## Train sample - total token lens (prompt + response + EOS)")
    lines.append(json.dumps(percentiles(total_lens), indent=2))
    lines.append("")
    n = len(sample)
    lines.append(f"truncated > 512  : {truncated_at_512:5d}  ({100*truncated_at_512/n:.2f}%)")
    lines.append(f"truncated > 768  : {truncated_at_768:5d}  ({100*truncated_at_768/n:.2f}%)")
    lines.append(f"truncated > 1024 : {truncated_at_1024:5d}  ({100*truncated_at_1024/n:.2f}%)")
    lines.append("")
    lines.append("## Per-type total length (train sample)")
    for t, vs in sorted(type_lens.items()):
        lines.append(f"  {t:<22s} {percentiles(vs)}")

    lines.append("")
    lines.append("## Valid - total token lens")
    lines.append(json.dumps(percentiles(valid_total), indent=2))
    nv = len(valid_total)
    lines.append(f"valid truncated > 512  : {sum(1 for x in valid_total if x>512):5d}  ({100*sum(1 for x in valid_total if x>512)/nv:.2f}%)")
    lines.append(f"valid truncated > 768  : {sum(1 for x in valid_total if x>768):5d}  ({100*sum(1 for x in valid_total if x>768)/nv:.2f}%)")
    lines.append(f"valid truncated > 1024 : {sum(1 for x in valid_total if x>1024):5d}  ({100*sum(1 for x in valid_total if x>1024)/nv:.2f}%)")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
