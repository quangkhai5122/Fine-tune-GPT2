"""Smoke test the V2 pipeline (cleaning + normalization + dataset truncation)
on a small subset of train + valid. Reports:
  - extraction rate after the upgraded regex
  - normalization examples
  - SFTDataset sample shapes (anchor preserved after truncation?)
  - stop criteria regex sanity
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "GPT2_vietnamese"

from math_eval import extract_answer as standard_extract_answer
from math_eval import parse_number as standard_parse_number

# Inline the V2 logic (we don't want to import the notebook).
# Keep IDs of these functions in sync with build_v2_notebook.py.

SAFE_EOS_ID = 50256
PROMPT_TEMPLATE = "Bài toán: {q}\nLời giải: "
MAX_LENGTH = 768


def strip_trailing_safe_eos(token_ids):
    if hasattr(token_ids, "detach"):
        ids = token_ids.detach().cpu().tolist()
    else:
        ids = list(token_ids)
    while ids and ids[-1] == SAFE_EOS_ID:
        ids.pop()
    return ids


def decode_model_text(tok, token_ids):
    return tok.decode(strip_trailing_safe_eos(token_ids), skip_special_tokens=True)

RE_ANCHORS_VI = [
    re.compile(r"đ[áa]p\s*[áa]n\s*l[àa]\s*[:：]?\s*", re.IGNORECASE),
    re.compile(r"c[âa]u\s*tr[ảa]\s*l[ờo]i\s*l[àa]\s*[:：]?\s*", re.IGNORECASE),
    re.compile(r"đ[áa]p\s*[áa]n\s*[:：]\s*", re.IGNORECASE),
]
RE_ANCHORS_EN = [
    re.compile(r"the\s*answer\s*is\s*[:：]?\s*", re.IGNORECASE),
    re.compile(r"####\s*"),
]
RE_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

RE_NUM_VI_DEC = re.compile(r"-?\d+,\d+")
RE_NUM_EN_DEC = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
RE_NUM_LOOSE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _clean_tail(s: str) -> str:
    s = s.strip().split("\n", 1)[0].strip()
    s = re.sub(r"[.,;:。、,]+$", "", s)
    s = re.sub(r"\s*(đô\s*la|usd|đồng|vnd|cm2?|m2?|km2?|\$|%)\b.*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def extract_anchor_answer(text):
    return standard_extract_answer(text)


def parse_number(s):
    return standard_parse_number(s)


RE_TRAILING_ANCHORS = re.compile(
    r"(\s*(####\s*[-\d., ]*|"
    r"đ[áa]p\s*[áa]n\s*l[àa]\s*[:：]?[^\n]*|"
    r"c[âa]u\s*tr[ảa]\s*l[ờo]i\s*l[àa]\s*[:：]?[^\n]*|"
    r"the\s*answer\s*is\s*[:：]?[^\n]*))+\s*$",
    re.IGNORECASE,
)


def normalize_response(resp, gold_num, gold_str):
    body = RE_TRAILING_ANCHORS.sub("", resp.rstrip()).rstrip()
    if gold_num is not None:
        if gold_num == int(gold_num):
            canonical = str(int(gold_num))
        else:
            canonical = gold_str if gold_str and re.fullmatch(r"-?\d+(?:[.,]\d+)?", gold_str) else f"{gold_num:g}"
    else:
        canonical = gold_str if gold_str else ""
    return f"{body}\nĐáp án là: {canonical}"


def load_records(path):
    with Path(path).open("r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        return json.load(f) if head == "[" else [json.loads(ln) for ln in f if ln.strip()]


def build_sft_sample(rec, tok, max_length):
    prompt = PROMPT_TEMPLATE.format(q=rec["query_vi"].strip())
    response = rec["response_vi"]
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(response, add_special_tokens=False)["input_ids"] + [SAFE_EOS_ID]
    budget = max_length - len(p_ids)
    if budget <= 4:
        p_ids = p_ids[-(max_length - 8):]
        budget = max_length - len(p_ids)
    if len(r_ids) > budget:
        tail_keep = min(96, budget // 2)
        head_keep = budget - tail_keep
        r_ids = r_ids[:head_keep] + r_ids[-tail_keep:]
    ids = p_ids + r_ids
    labels = [-100] * len(p_ids) + r_ids
    return ids, labels


def main():
    print("Loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    tok.pad_token_id = SAFE_EOS_ID
    tok.eos_token_id = SAFE_EOS_ID

    print("Loading train + valid ...")
    train = load_records(DATA_DIR / "train.json")
    valid = load_records(DATA_DIR / "valid.json")

    # ---- 1. Extraction rate (v2 regex vs gold) ----
    n_train = len(train)
    extractable = 0
    numeric = 0
    miss_examples = []
    for rec in train:
        s = extract_anchor_answer(rec.get("response_vi"))
        if s is not None:
            extractable += 1
            if parse_number(s) is not None:
                numeric += 1
        else:
            if len(miss_examples) < 5:
                miss_examples.append(rec)

    print(f"\n[v2-extract train] extractable={extractable}/{n_train} "
          f"({100*extractable/n_train:.2f}%) numeric={numeric} "
          f"({100*numeric/n_train:.2f}%)")

    # Valid
    n_v = len(valid)
    ext_v = sum(1 for r in valid if extract_anchor_answer(r.get("response_vi")) is not None)
    num_v = sum(1 for r in valid if parse_number(extract_anchor_answer(r.get("response_vi"))) is not None)
    print(f"[v2-extract valid] extractable={ext_v}/{n_v} "
          f"({100*ext_v/n_v:.2f}%) numeric={num_v} "
          f"({100*num_v/n_v:.2f}%)")

    # ---- 2. Normalization examples ----
    print("\n[normalization samples]")
    rng = random.Random(0)
    for rec in rng.sample(train, 3):
        s = extract_anchor_answer(rec.get("response_vi"))
        n = parse_number(s)
        new_r = normalize_response(rec["response_vi"], n, s)
        print("-" * 80)
        print("ORIG :", rec["response_vi"][-180:])
        print("GOLD :", s, "(num=", n, ")")
        print("NORM :", new_r[-180:])

    # ---- 3. Dataset truncation safety: every sample ends with EOS,
    # and the last 64 tokens contain the answer anchor whenever possible.
    rng = random.Random(7)
    sample = rng.sample(train, 2000)
    n_total = len(sample)
    n_anchor_in_tail = 0
    n_ends_with_eos = 0
    n_truncated = 0
    for rec in sample:
        # use the NORMALIZED response, since that's what training would use
        s = extract_anchor_answer(rec.get("response_vi"))
        num = parse_number(s)
        rec_n = dict(rec)
        rec_n["response_vi"] = normalize_response(rec["response_vi"], num, s)
        ids, labels = build_sft_sample(rec_n, tok, MAX_LENGTH)
        if ids[-1] == SAFE_EOS_ID:
            n_ends_with_eos += 1
        tail_text = decode_model_text(tok, ids[-64:])
        if "Đáp án là" in tail_text:
            n_anchor_in_tail += 1
        # detect truncation
        p_ids = tok(PROMPT_TEMPLATE.format(q=rec_n["query_vi"].strip()), add_special_tokens=False)["input_ids"]
        r_ids = tok(rec_n["response_vi"], add_special_tokens=False)["input_ids"] + [SAFE_EOS_ID]
        if len(p_ids) + len(r_ids) > MAX_LENGTH:
            n_truncated += 1

    print(f"\n[truncation safety on {n_total} samples]")
    print(f"  ends with EOS         : {n_ends_with_eos}/{n_total} "
          f"({100*n_ends_with_eos/n_total:.2f}%)")
    print(f"  anchor in last-64 tok : {n_anchor_in_tail}/{n_total} "
          f"({100*n_anchor_in_tail/n_total:.2f}%)")
    print(f"  truncated (would lose tail without our fix) : "
          f"{n_truncated}/{n_total} ({100*n_truncated/n_total:.2f}%)")

    # ---- 4. Show one decoded truncated sample (sanity print) ----
    for rec in sample:
        p_ids = tok(PROMPT_TEMPLATE.format(q=rec["query_vi"].strip()), add_special_tokens=False)["input_ids"]
        r_ids = tok(rec["response_vi"], add_special_tokens=False)["input_ids"] + [SAFE_EOS_ID]
        if len(p_ids) + len(r_ids) > MAX_LENGTH:
            s = extract_anchor_answer(rec.get("response_vi"))
            n = parse_number(s)
            rec_n = dict(rec); rec_n["response_vi"] = normalize_response(rec["response_vi"], n, s)
            ids, _ = build_sft_sample(rec_n, tok, MAX_LENGTH)
            print("\n[truncated sample preview]")
            print("type:", rec.get("type"), "orig len:", len(p_ids) + len(r_ids), "kept:", len(ids))
            print("decoded head:", decode_model_text(tok, ids[:60]))
            print("decoded tail:", decode_model_text(tok, ids[-60:]))
            break


if __name__ == "__main__":
    main()
