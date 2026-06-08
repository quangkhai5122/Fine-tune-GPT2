"""Create a strict-legal candidate verifier + template retrieval notebook."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V17_SCRIPT = ROOT / "audit" / "create_v17_answer_only_notebooks.py"
NOTEBOOK_PATH = ROOT / "finetune_gpt2_for_math_v18_legal_candidate_verifier_template_retrieval.ipynb"


def load_v17_module():
    spec = importlib.util.spec_from_file_location("create_v17_answer_only_notebooks", V17_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {V17_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v17 = load_v17_module()
v16 = v17.v16


V18_RUN = r'''
# ============================================================
# 8. Legal candidate verifier + template retrieval
# ============================================================
try:
    import numpy as np
except Exception as exc:
    np = None
    print("[template] numpy unavailable; linear templates disabled:", repr(exc))

NUMERIC_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
FRAC_RE = re.compile(r"\\(?:d|t)?frac\s*\{([-+]?\d+(?:[.,]\d+)?)\}\s*\{([-+]?\d+(?:[.,]\d+)?)\}")

def parse_query_number(token: str):
    token = str(token).replace(",", ".")
    try:
        value = float(token)
        return value if math.isfinite(value) else None
    except Exception:
        return None

def extract_query_numbers(text: str | None) -> list[float]:
    text = unicodedata.normalize("NFKC", str(text or ""))
    numbers = []
    consumed = set()
    for m in FRAC_RE.finditer(text):
        a = parse_query_number(m.group(1))
        b = parse_query_number(m.group(2))
        if a is not None and b not in (None, 0):
            numbers.append(a / b)
            consumed.update(range(m.start(), m.end()))
    masked = "".join(" " if i in consumed else ch for i, ch in enumerate(text))
    for m in NUMERIC_RE.finditer(masked):
        value = parse_query_number(m.group(0))
        if value is not None:
            numbers.append(value)
    return numbers

def number_signature(numbers: list[float]) -> tuple:
    return tuple(round(float(x), 8) for x in numbers)

def same_number_signature(a: list[float], b: list[float]) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= 1e-8 for x, y in zip(a, b))

def skeletonize_query(text: str | None) -> str:
    text = normalize_text_key(text)
    text = FRAC_RE.sub(" <num> ", text)
    text = NUMERIC_RE.sub(" <num> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def candidate_key_from_value(value) -> str | None:
    return _safe_num_key(value)

def canonical_candidate_key(key) -> str | None:
    if key is None:
        return None
    num = _answer_num_from_key(key)
    if num is None:
        return None
    return _safe_num_key(num)

def answer_prior_norm(key: str | None) -> float:
    if key is None or not ANSWER_PRIORS:
        return 0.0
    top = max(ANSWER_PRIORS.values()) if ANSWER_PRIORS else 1
    return math.log1p(ANSWER_PRIORS.get(key, 0)) / max(1e-9, math.log1p(top))

class LegalTemplateRetriever:
    def __init__(self, records: list[dict]):
        self.records = []
        self.texts = []
        self.token_sets = []
        self.skeleton_index = defaultdict(list)
        self.exact_query_index = defaultdict(list)
        for rec in records:
            key = _safe_num_key(rec.get("_gold_num"))
            if key is None:
                continue
            query = normalize_text_key(rec.get("query_vi"))
            nums = extract_query_numbers(rec.get("query_vi"))
            item = {
                "query_vi": rec.get("query_vi", ""),
                "query_norm": query,
                "tokens": query_tokens(query),
                "numbers": nums,
                "number_signature": number_signature(nums),
                "skeleton": skeletonize_query(rec.get("query_vi")),
                "answer_key": key,
                "answer_num": float(rec.get("_gold_num")),
                "canonical_answer": rec.get("_canonical_answer"),
            }
            self.records.append(item)
            self.texts.append(query)
            self.token_sets.append(item["tokens"])
            self.skeleton_index[item["skeleton"]].append(item)
            self.exact_query_index[item["query_norm"]].append(item)

        self.backend = "jaccard"
        self.vectorizer = None
        self.matrix = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(analyzer=RETRIEVAL_ANALYZER, ngram_range=RETRIEVAL_NGRAM_RANGE, lowercase=False, min_df=1)
            self.matrix = self.vectorizer.fit_transform(self.texts)
            self.backend = "tfidf_char_ngram"
        except Exception as exc:
            print("[retrieval] sklearn TF-IDF unavailable; using Jaccard fallback:", repr(exc))
        print("[retrieval] backend=", self.backend, "records=", len(self.records), "skeletons=", len(self.skeleton_index))

    def search(self, rec: dict, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
        if not self.records:
            return []
        q = normalize_text_key(rec.get("query_vi"))
        qtok = query_tokens(q)
        qnums = extract_query_numbers(rec.get("query_vi"))
        qskel = skeletonize_query(rec.get("query_vi"))
        if self.backend == "tfidf_char_ngram" and self.vectorizer is not None and self.matrix is not None:
            from sklearn.metrics.pairwise import linear_kernel
            qv = self.vectorizer.transform([q])
            sims = linear_kernel(qv, self.matrix).ravel()
            top_idx = sims.argsort()[-top_k:][::-1]
            rows = [(int(i), float(sims[i])) for i in top_idx if float(sims[i]) > 0.0]
        else:
            scored = [(i, jaccard(qtok, toks)) for i, toks in enumerate(self.token_sets)]
            scored.sort(key=lambda x: x[1], reverse=True)
            rows = [(i, float(s)) for i, s in scored[:top_k] if s > 0.0]
        hits = []
        for i, sim in rows:
            item = self.records[i]
            hits.append({
                **item,
                "similarity": float(sim),
                "jaccard": jaccard(qtok, item["tokens"]),
                "same_numbers": same_number_signature(qnums, item["numbers"]),
                "same_skeleton": qskel == item["skeleton"],
            })
        return hits

    def exact_query_candidate(self, rec: dict) -> dict | None:
        q = normalize_text_key(rec.get("query_vi"))
        rows = self.exact_query_index.get(q) or []
        return majority_candidate_from_rows(rows, "template_exact_query", 1.0)

    def exact_number_template_candidate(self, rec: dict) -> dict | None:
        qnums = extract_query_numbers(rec.get("query_vi"))
        rows = [
            row for row in self.skeleton_index.get(skeletonize_query(rec.get("query_vi")), [])
            if same_number_signature(qnums, row["numbers"])
        ]
        return majority_candidate_from_rows(rows, "template_exact_numbers", 0.96)

    def linear_template_candidate(self, rec: dict) -> dict | None:
        if np is None:
            return None
        qnums = extract_query_numbers(rec.get("query_vi"))
        if not qnums or len(qnums) > 6:
            return None
        rows = [
            row for row in self.skeleton_index.get(skeletonize_query(rec.get("query_vi")), [])
            if len(row["numbers"]) == len(qnums)
        ]
        min_support = max(TEMPLATE_LINEAR_MIN_SUPPORT, len(qnums) + 2)
        if len(rows) < min_support:
            return None
        try:
            x = np.array([[1.0] + [float(v) for v in row["numbers"]] for row in rows], dtype=float)
            y = np.array([float(row["answer_num"]) for row in rows], dtype=float)
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            train_pred = x @ coef
            rel = np.abs(train_pred - y) / np.maximum(1.0, np.abs(y))
            mean_rel = float(np.mean(rel))
            max_rel = float(np.max(rel))
            if mean_rel > TEMPLATE_LINEAR_MAX_MEAN_REL_ERR or max_rel > TEMPLATE_LINEAR_MAX_REL_ERR:
                return None
            pred = float(np.array([1.0] + [float(v) for v in qnums], dtype=float) @ coef)
            if not math.isfinite(pred) or abs(pred) > TEMPLATE_LINEAR_MAX_ABS_PRED:
                return None
            key = candidate_key_from_value(pred)
            if key is None:
                return None
            return {
                "key": key,
                "source": "template_linear",
                "confidence": 0.92,
                "meta": {
                    "support": len(rows),
                    "mean_rel_error": mean_rel,
                    "max_rel_error": max_rel,
                },
            }
        except Exception as exc:
            return {
                "key": None,
                "source": "template_linear_error",
                "confidence": 0.0,
                "meta": {"error": repr(exc)},
            }

def majority_candidate_from_rows(rows: list[dict], source: str, base_confidence: float) -> dict | None:
    if not rows:
        return None
    counts = Counter(row["answer_key"] for row in rows if row.get("answer_key") is not None)
    if not counts:
        return None
    key, count = counts.most_common(1)[0]
    majority_frac = count / max(1, len(rows))
    if majority_frac < TEMPLATE_MAJORITY_MIN_FRAC:
        return None
    return {
        "key": key,
        "source": source,
        "confidence": min(1.0, base_confidence * majority_frac),
        "meta": {"support": len(rows), "majority_count": count, "majority_frac": majority_frac},
    }

def retrieval_group_candidates(rec: dict, retriever: LegalTemplateRetriever) -> list[dict]:
    hits = retriever.search(rec, top_k=RETRIEVAL_TOP_K)
    if not hits:
        return []
    groups = defaultdict(list)
    for hit in hits:
        if hit.get("answer_key") is not None:
            groups[hit["answer_key"]].append(hit)
    ranked = []
    for key, vals in groups.items():
        count = len(vals)
        same_numbers = sum(1 for v in vals if v.get("same_numbers"))
        same_skeleton = sum(1 for v in vals if v.get("same_skeleton"))
        top_sim = max(v["similarity"] for v in vals)
        top_jaccard = max(v["jaccard"] for v in vals)
        ranked.append({
            "key": key,
            "count": count,
            "same_numbers": same_numbers,
            "same_skeleton": same_skeleton,
            "top_sim": top_sim,
            "top_jaccard": top_jaccard,
        })
    if not ranked:
        return []
    ranked.sort(key=lambda x: (x["count"], x["same_numbers"], x["same_skeleton"], x["top_sim"], x["top_jaccard"]), reverse=True)
    top_count = ranked[0]["count"]
    second_count = ranked[1]["count"] if len(ranked) > 1 else 0
    out = []
    for rank, row in enumerate(ranked[:RETRIEVAL_CANDIDATE_GROUPS]):
        majority_frac = row["count"] / max(1, len(hits))
        margin = (top_count - second_count) / max(1, len(hits)) if rank == 0 else 0.0
        num_bonus = 0.12 if row["same_numbers"] else 0.0
        skel_bonus = 0.08 if row["same_skeleton"] else 0.0
        confidence = (
            0.50 * row["top_sim"]
            + 0.18 * row["top_jaccard"]
            + 0.18 * majority_frac
            + 0.10 * margin
            + num_bonus
            + skel_bonus
        )
        source = "retrieval_group"
        if (
            row["top_sim"] >= RETRIEVAL_DIRECT_MIN_SIM
            and row["top_jaccard"] >= RETRIEVAL_DIRECT_MIN_JACCARD
            and majority_frac >= RETRIEVAL_DIRECT_MIN_MAJORITY_FRAC
            and margin >= RETRIEVAL_DIRECT_MIN_MARGIN
            and (row["same_numbers"] > 0 or row["same_skeleton"] > 0)
        ):
            source = "retrieval_direct_safe"
            confidence = max(confidence, 0.94)
        out.append({
            "key": row["key"],
            "source": source,
            "confidence": min(1.0, confidence),
            "meta": {
                **row,
                "rank": rank,
                "hits": len(hits),
                "majority_frac": majority_frac,
                "margin": margin,
            },
        })
    return out

def arithmetic_candidates(rec: dict) -> list[dict]:
    nums = extract_query_numbers(rec.get("query_vi"))
    if not nums or len(nums) > ARITH_MAX_NUMBERS:
        return []
    values = []
    values.extend(nums)
    values.append(sum(nums))
    if len(nums) >= 2:
        values.extend([nums[0] - nums[-1], nums[-1] - nums[0]])
    if 1 < len(nums) <= 4:
        prod = 1.0
        for v in nums:
            prod *= v
        values.append(prod)
    out = []
    seen = set()
    for value in values:
        key = candidate_key_from_value(value)
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append({
            "key": key,
            "source": "query_arithmetic",
            "confidence": 0.35,
            "meta": {"numbers": nums},
        })
    return out[:ARITH_MAX_CANDIDATES]

def add_candidate(candidates: dict, key, source: str, confidence: float, meta: dict | None = None):
    key = canonical_candidate_key(key)
    if key is None:
        return
    meta = meta or {}
    existing = candidates.get(key)
    item = {
        "key": key,
        "source": source,
        "sources": [source],
        "confidence": float(confidence),
        "meta": meta,
        "prior_norm": answer_prior_norm(key),
    }
    if existing is None:
        candidates[key] = item
    else:
        existing["confidence"] = max(existing["confidence"], float(confidence))
        existing["prior_norm"] = max(existing.get("prior_norm", 0.0), answer_prior_norm(key))
        existing.setdefault("sources", []).append(source)
        existing.setdefault("meta_by_source", {})[source] = meta

def build_candidate_set(rec: dict, idx: int, model_item: dict, retriever: LegalTemplateRetriever) -> list[dict]:
    candidates = {}
    model_key = answer_key_from_output(model_item)
    add_candidate(candidates, model_key, "model", 1.0, {"model_output": model_item.get("model_output", "")})
    for cand in retrieval_group_candidates(rec, retriever):
        add_candidate(candidates, cand["key"], cand["source"], cand["confidence"], cand.get("meta"))
    for cand in [
        retriever.exact_query_candidate(rec),
        retriever.exact_number_template_candidate(rec),
        retriever.linear_template_candidate(rec),
    ]:
        if cand and cand.get("key") is not None:
            add_candidate(candidates, cand["key"], cand["source"], cand["confidence"], cand.get("meta"))
    if INCLUDE_ARITHMETIC_CANDIDATES:
        for cand in arithmetic_candidates(rec):
            add_candidate(candidates, cand["key"], cand["source"], cand["confidence"], cand.get("meta"))
    rows = list(candidates.values())
    rows.sort(key=lambda c: (c["key"] == model_key, c.get("confidence", 0.0), c.get("prior_norm", 0.0)), reverse=True)
    model_rows = [c for c in rows if c["key"] == model_key]
    other_rows = [c for c in rows if c["key"] != model_key]
    return (model_rows + other_rows[: max(0, MAX_VERIFIER_CANDIDATES_PER_ROW - len(model_rows))])[:MAX_VERIFIER_CANDIDATES_PER_ROW]

@torch.inference_mode()
def score_candidate_logprobs(adapter_dir: Path, records: list[dict], candidate_sets: list[list[dict]]):
    if not VERIFIER_LOGPROB_ENABLED:
        return
    flat = []
    for row_idx, cands in enumerate(candidate_sets):
        for cand_idx, cand in enumerate(cands):
            if cand.get("key") is not None:
                flat.append((row_idx, cand_idx, cand["key"]))
    if not flat:
        return
    score_tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    score_tok.pad_token_id = SAFE_EOS_ID
    score_tok.eos_token_id = SAFE_EOS_ID
    score_model = load_model_for_generation(adapter_dir)
    device = next(score_model.parameters()).device
    n_pos = int(getattr(score_model.config, "n_positions", getattr(score_model.config, "max_position_embeddings", 1024)))
    vocab_n = score_model.get_input_embeddings().num_embeddings
    for start in tqdm(range(0, len(flat), VERIFIER_BATCH_SIZE), desc="verifier_logprob"):
        batch = flat[start:start + VERIFIER_BATCH_SIZE]
        encoded = []
        labels = []
        for row_idx, _cand_idx, key in batch:
            rec = records[row_idx]
            prompt_ids = score_tok(build_prompt(rec), add_special_tokens=False)["input_ids"]
            target_ids = score_tok(build_answer_only_target(key), add_special_tokens=False)["input_ids"] + [SAFE_EOS_ID]
            target_ids = [min(t, vocab_n - 1) for t in target_ids]
            prompt_budget = max(1, n_pos - len(target_ids))
            prompt_ids = [min(t, vocab_n - 1) for t in prompt_ids[-prompt_budget:]]
            ids = prompt_ids + target_ids
            lab = [-100] * len(prompt_ids) + target_ids
            encoded.append(ids)
            labels.append(lab)
        max_len = max(len(x) for x in encoded)
        input_ids = []
        attn = []
        label_ids = []
        for ids, lab in zip(encoded, labels):
            pad_n = max_len - len(ids)
            input_ids.append(ids + [SAFE_EOS_ID] * pad_n)
            attn.append([1] * len(ids) + [0] * pad_n)
            label_ids.append(lab + [-100] * pad_n)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        attn = torch.tensor(attn, dtype=torch.long, device=device)
        label_ids = torch.tensor(label_ids, dtype=torch.long, device=device)
        logits = score_model(input_ids=input_ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = label_ids[:, 1:]
        mask = shift_labels.ne(-100)
        safe_labels = shift_labels.clamp(min=0)
        token_logprobs = F.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logprobs = token_logprobs * mask
        sums = token_logprobs.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        for bidx, (row_idx, cand_idx, _key) in enumerate(batch):
            candidate_sets[row_idx][cand_idx]["logprob_sum"] = float(sums[bidx].detach().cpu())
            candidate_sets[row_idx][cand_idx]["logprob_avg"] = float((sums[bidx] / counts[bidx]).detach().cpu())
            candidate_sets[row_idx][cand_idx]["logprob_tokens"] = int(counts[bidx].detach().cpu())
    del score_model
    torch.cuda.empty_cache()

def select_candidate_for_row(cands: list[dict], model_key: str | None) -> tuple[dict, dict]:
    model_cand = next((c for c in cands if c["key"] == model_key), None)
    if model_cand is None and cands:
        model_cand = cands[0]
    model_lp = model_cand.get("logprob_avg") if model_cand else None
    chosen = model_cand
    reason = "model_default"
    best_rank = -1e9
    for cand in cands:
        if cand is model_cand:
            continue
        lp = cand.get("logprob_avg")
        lp_delta = 0.0 if (lp is None or model_lp is None) else lp - model_lp
        sources = set(cand.get("sources") or [cand.get("source")])
        safe_source = bool(sources & {"template_exact_query", "template_exact_numbers", "template_linear", "retrieval_direct_safe"})
        safe_override = safe_source and cand.get("confidence", 0.0) >= VERIFIER_SAFE_CONFIDENCE and lp_delta >= -VERIFIER_LOGPROB_TOLERANCE
        logprob_override = cand.get("confidence", 0.0) >= VERIFIER_MIN_CONFIDENCE and lp_delta >= VERIFIER_LOGPROB_OVERRIDE_MARGIN
        missing_model = model_key is None and cand.get("key") is not None
        rank_score = (
            2.0 * cand.get("confidence", 0.0)
            + 0.30 * cand.get("prior_norm", 0.0)
            + 0.75 * lp_delta
            + (0.35 if safe_source else 0.0)
        )
        cand["rank_score"] = rank_score
        cand["logprob_delta_vs_model"] = lp_delta
        if (missing_model or safe_override or logprob_override) and rank_score > best_rank:
            chosen = cand
            best_rank = rank_score
            if missing_model:
                reason = "model_missing"
            elif safe_override:
                reason = "safe_template_or_retrieval"
            else:
                reason = "logprob_override"
    if chosen is None:
        chosen = {"key": model_key, "source": "empty_fallback", "sources": ["empty_fallback"], "confidence": 0.0}
    return chosen, {
        "reason": reason,
        "model_key": model_key,
        "model_logprob_avg": model_lp,
        "chosen_key": chosen.get("key"),
        "chosen_sources": chosen.get("sources") or [chosen.get("source")],
        "chosen_confidence": chosen.get("confidence"),
        "chosen_logprob_avg": chosen.get("logprob_avg"),
        "chosen_logprob_delta_vs_model": chosen.get("logprob_delta_vs_model"),
        "num_candidates": len(cands),
    }

def choose_candidate_verifier(records: list[dict], model_outputs: list[dict], adapter_dir: Path, retriever: LegalTemplateRetriever):
    candidate_sets = []
    for idx, rec in enumerate(records):
        candidate_sets.append(build_candidate_set(rec, idx, model_outputs[idx], retriever))
    score_candidate_logprobs(adapter_dir, records, candidate_sets)
    outputs = []
    decisions = []
    source_counts = Counter()
    reason_counts = Counter()
    changed = 0
    for idx, rec in enumerate(records):
        model_key = answer_key_from_output(model_outputs[idx])
        chosen, decision = select_candidate_for_row(candidate_sets[idx], model_key)
        if chosen.get("key") != model_key:
            changed += 1
        reason_counts[decision["reason"]] += 1
        for src in decision.get("chosen_sources") or ["unknown"]:
            source_counts[src] += 1
        outputs.append(make_answer_item(rec, idx, chosen.get("key"), model_outputs[idx]))
        decisions.append({
            "id": rec.get("id", idx),
            **decision,
            "candidates": [
                {
                    "key": c.get("key"),
                    "sources": c.get("sources"),
                    "confidence": c.get("confidence"),
                    "prior_norm": c.get("prior_norm"),
                    "logprob_avg": c.get("logprob_avg"),
                    "rank_score": c.get("rank_score"),
                    "meta": c.get("meta"),
                    "meta_by_source": c.get("meta_by_source"),
                }
                for c in candidate_sets[idx]
            ],
        })
    summary = {
        "strategy": "legal_candidate_verifier_template_retrieval",
        "num_rows": len(records),
        "changed_from_model": changed,
        "source_counts": dict(source_counts.most_common()),
        "reason_counts": dict(reason_counts.most_common()),
        "retrieval_backend": retriever.backend,
        "verifier_logprob_enabled": VERIFIER_LOGPROB_ENABLED,
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "notes": "Candidate retrieval uses train query_vi only; candidate answers come from train response_vi or query-derived numeric/template heuristics.",
    }
    return outputs, decisions, summary

def run_verifier_split(records: list[dict], split_name: str, output_path: Path, report_path: Path):
    entries = candidate_entries_from_runs()
    model_candidates = generate_model_candidates(records, split_name, entries)
    primary = next((c for c in model_candidates if c["kind"] == "model_final" and int(c["seed"]) == int(TRAIN_SEEDS[0])), model_candidates[-1])
    if split_name == "valid":
        MODEL_VALID_OUTPUT_PATH.write_text(json.dumps(primary["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
        if records and records[0].get("response_vi"):
            save_eval_report(MODEL_VALID_OUTPUT_PATH, records, MODEL_VALID_REPORT_PATH)
    elif split_name == "test":
        MODEL_TEST_OUTPUT_PATH.write_text(json.dumps(primary["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    retriever = LegalTemplateRetriever(train_clean)
    outputs, decisions, summary = choose_candidate_verifier(records, primary["outputs"], Path(primary["adapter_dir"]), retriever)
    report, payload = save_outputs_and_optional_report(records, outputs, output_path, report_path, ENSEMBLE_RANKER_REPORT_PATH, summary)
    (WORKING_DIR / ("%s_candidate_verifier_decisions.json" % split_name)).write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    if report is not None:
        print("[candidate-verifier:%s]" % split_name, report["summary"])
    return model_candidates, outputs, payload

if RUN_MODE == "phase1":
    _candidates, _outputs, _payload = run_verifier_split(valid_clean, "valid", VALID_OUTPUT_PATH, VALID_REPORT_PATH)
elif RUN_MODE == "phase2":
    if not TEST_FILE.exists():
        raise FileNotFoundError("RUN_MODE='phase2' requires test.json")
    test_records = load_records(TEST_FILE)
    _candidates, _outputs, _payload = run_verifier_split(test_records, "test", TEST_OUTPUT_PATH, VALID_REPORT_PATH)
    print("[phase2] wrote", TEST_OUTPUT_PATH)
else:
    raise ValueError("Unknown RUN_MODE=" + str(RUN_MODE))
'''


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def build_config_cell() -> str:
    source = v16.build_config_cell(
        version="v18_legal_candidate_verifier_template_retrieval",
        artifact_prefix="v18_legal_candidate_verifier_template_retrieval",
        train_seeds=[42],
        stage_a_epochs=7.0,
        stage_a_lr=1e-3,
        ensemble_last_k_epochs=0,
        use_calib=False,
        calib_fraction=0.10,
        calib_max_records=None,
        legal_query_retrieval_enabled=True,
    )
    source = source.replace(
        "SAVE_EPOCH_CHECKPOINTS = True",
        "SAVE_EPOCH_CHECKPOINTS = False\nCHECKPOINT_EPOCH_LABELS_TO_SAVE = None",
    )
    source = source.replace("ENSEMBLE_INCLUDE_EPOCH_CHECKPOINTS = True", "ENSEMBLE_INCLUDE_EPOCH_CHECKPOINTS = False")
    source = source.replace("ENSEMBLE_INCLUDE_FINAL = True", "ENSEMBLE_INCLUDE_FINAL = True")
    extra = r'''

# Legal candidate verifier/template retrieval.
TEMPLATE_RETRIEVAL_ENABLED = True
VERIFIER_LOGPROB_ENABLED = True
VERIFIER_BATCH_SIZE = 16
MAX_VERIFIER_CANDIDATES_PER_ROW = 6

RETRIEVAL_TOP_K = 9
RETRIEVAL_CANDIDATE_GROUPS = 3
RETRIEVAL_DIRECT_MIN_SIM = 0.92
RETRIEVAL_DIRECT_MIN_JACCARD = 0.70
RETRIEVAL_DIRECT_MIN_MAJORITY_FRAC = 0.55
RETRIEVAL_DIRECT_MIN_MARGIN = 0.20

TEMPLATE_MAJORITY_MIN_FRAC = 0.60
TEMPLATE_LINEAR_MIN_SUPPORT = 4
TEMPLATE_LINEAR_MAX_MEAN_REL_ERR = 0.02
TEMPLATE_LINEAR_MAX_REL_ERR = 0.10
TEMPLATE_LINEAR_MAX_ABS_PRED = 1e9

INCLUDE_ARITHMETIC_CANDIDATES = True
ARITH_MAX_NUMBERS = 6
ARITH_MAX_CANDIDATES = 4

VERIFIER_SAFE_CONFIDENCE = 0.90
VERIFIER_MIN_CONFIDENCE = 0.82
VERIFIER_LOGPROB_TOLERANCE = 0.35
VERIFIER_LOGPROB_OVERRIDE_MARGIN = 0.55
'''
    return source + extra


def build_manifest_cell() -> str:
    source = v17.patched_manifest_cell("train_query_vi_template_retrieval_to_train_response_vi_candidates", False)
    source = source.replace(
        '"uses_valid_labels_for_checkpoint_selection": False,\n',
        '"uses_valid_labels_for_checkpoint_selection": False,\n'
        '        "uses_valid_labels_for_verifier_training": False,\n',
    )
    source = source.replace(
        '"legal_query_retrieval_enabled": LEGAL_QUERY_RETRIEVAL_ENABLED,\n',
        '"legal_query_retrieval_enabled": LEGAL_QUERY_RETRIEVAL_ENABLED,\n'
        '        "template_retrieval_enabled": TEMPLATE_RETRIEVAL_ENABLED,\n'
        '        "verifier_logprob_enabled": VERIFIER_LOGPROB_ENABLED,\n'
        '        "max_verifier_candidates_per_row": MAX_VERIFIER_CANDIDATES_PER_ROW,\n'
        '        "retrieval_top_k": RETRIEVAL_TOP_K,\n'
        '        "retrieval_direct_min_sim": RETRIEVAL_DIRECT_MIN_SIM,\n'
        '        "verifier_safe_confidence": VERIFIER_SAFE_CONFIDENCE,\n',
    )
    source = source.replace(
        '"ensemble_or_gate_report": str(ENSEMBLE_RANKER_REPORT_PATH),\n',
        '"ensemble_or_gate_report": str(ENSEMBLE_RANKER_REPORT_PATH),\n'
        '        "valid_candidate_verifier_decisions": str(WORKING_DIR / "valid_candidate_verifier_decisions.json"),\n'
        '        "test_candidate_verifier_decisions": str(WORKING_DIR / "test_candidate_verifier_decisions.json"),\n',
    )
    return source


def build_notebook() -> dict:
    base_nb, base_cells = v16.read_base_cells()
    nb = copy.deepcopy(base_nb)
    import_cell = v17.scrub_code_cell(base_cells[1])
    eval_cell = v17.scrub_code_cell(base_cells[3])
    eval_cell["source"] = v16.scrub_eval_cell("".join(eval_cell["source"])).splitlines(keepends=True)
    dataset_cell = v17.scrub_code_cell(base_cells[5])
    title = (
        "# V18 - Legal Candidate Verifier + Template Retrieval\n\n"
        "Single-seed answer-only SFT followed by strict-legal candidate verification. "
        "Candidate retrieval uses train `query_vi`; candidate answers come from train `response_vi` "
        "or query-derived numeric/template heuristics. No `type` routing and no `original_*` fields."
    )
    nb["cells"] = [
        markdown_cell(title),
        import_cell,
        code_cell(build_config_cell()),
        eval_cell,
        code_cell(v16.DATA_CELL),
        dataset_cell,
        code_cell(v17.patched_training_cell()),
        code_cell(v16.PROMPT_HELPER_CELL),
        code_cell(v16.GENERATION_COMMON + "\n" + V18_RUN),
        code_cell(build_manifest_cell()),
    ]
    return nb


def validate_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        try:
            ast.parse(src)
        except SyntaxError as exc:
            raise SyntaxError(f"{path.name} cell {idx}: {exc}") from exc
    print("[validate]", path.name, "cells=", len(nb.get("cells", [])))


def main() -> None:
    nb = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    validate_notebook(NOTEBOOK_PATH)
    print("[wrote]", NOTEBOOK_PATH.name, "bytes=", NOTEBOOK_PATH.stat().st_size)


if __name__ == "__main__":
    main()
