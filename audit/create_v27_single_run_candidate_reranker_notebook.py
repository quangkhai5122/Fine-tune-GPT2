"""Create v27 single-run legal candidate reranker notebook.

V27 keeps the strongest v23 answer-only training setup, but avoids uploading
outputs from previous notebooks. It creates model/retrieval candidates inside
the same Kaggle Run All session and reranks them with the fine-tuned GPT-2
itself using answer likelihood plus conservative, query-only sanity features.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V23_SCRIPT = ROOT / "audit" / "create_v23_legal_no_solver_ranker_notebook.py"
NOTEBOOK_PATH = ROOT / "finetune_gpt2_for_math_v27_single_run_candidate_reranker.ipynb"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V27_CONFIG_BLOCK = r'''DEFAULT_RETRIEVAL_GATE = {"enabled": True}

# V27 runs everything in one notebook. It trains one v23-style answer-only
# adapter, then generates model/retrieval candidates and reranks them with the
# same fine-tuned GPT-2. Defaults avoid valid-overlap profile selection.
V27_DEFAULT_PROFILE_NAME = os.environ.get("V27_PROFILE", "likelihood_conservative")
V27_SELECT_ON_VALID = os.environ.get("V27_SELECT_ON_VALID", "0") == "1"
V27_EVAL_VALID_PROFILES_IN_PHASE2 = os.environ.get("V27_EVAL_VALID_PROFILES_IN_PHASE2", "0") == "1"
V27_ENABLE_GREEDY_DECODE = os.environ.get("V27_ENABLE_GREEDY_DECODE", "1") == "1"
V27_ENABLE_SHORT_DECODE = os.environ.get("V27_ENABLE_SHORT_DECODE", "0") == "1"
V27_LIKELIHOOD_BATCH_SIZE = int(os.environ.get("V27_LIKELIHOOD_BATCH_SIZE", "8"))
V27_MAX_RETRIEVAL_CANDIDATES = int(os.environ.get("V27_MAX_RETRIEVAL_CANDIDATES", "2"))
V27_MAX_CANDIDATES_PER_ROW = int(os.environ.get("V27_MAX_CANDIDATES_PER_ROW", "5"))
V27_RETRIEVAL_TOP_K = int(os.environ.get("V27_RETRIEVAL_TOP_K", "9"))
V27_RETRIEVAL_MIN_SIM = float(os.environ.get("V27_RETRIEVAL_MIN_SIM", "0.70"))
V27_RETRIEVAL_MIN_JACCARD = float(os.environ.get("V27_RETRIEVAL_MIN_JACCARD", "0.45"))
V27_STRICT_RETRIEVAL_MIN_SIM = float(os.environ.get("V27_STRICT_RETRIEVAL_MIN_SIM", "0.92"))
V27_STRICT_RETRIEVAL_MIN_JACCARD = float(os.environ.get("V27_STRICT_RETRIEVAL_MIN_JACCARD", "0.72"))

'''


V27_TAIL = r'''
# ============================================================
# 8. V27 single-run candidate reranker, no external outputs
# ============================================================
def _bucket(summary: dict, score: int) -> int:
    buckets = summary.get("buckets", {})
    return int(buckets.get(score, buckets.get(str(score), 0)))

def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))

def _score_summary_key(summary: dict, *, tie_break: float = 0.0):
    return (
        int(summary.get("raw_score", -1)),
        _bucket(summary, 10),
        int(summary.get("extractable", 0)),
        -int(summary.get("buckets", {}).get("0", summary.get("buckets", {}).get(0, 0))),
        float(tie_break),
    )

def output_sanity(outputs: list[dict]) -> dict:
    texts = [str(o.get("model_output", "")) for o in outputs]
    return {
        "n": len(texts),
        "nonempty": sum(bool(t.strip()) for t in texts),
        "digit": sum(bool(re.search(r"\d", t)) for t in texts),
        "anchor": sum(bool(re.search(r"đáp\s*án|dap\s*an|answer|####", t, re.IGNORECASE)) for t in texts),
        "extractable": sum(extract_pred(o)[0] is not None for o in outputs),
        "samples": [t.replace("\n", " ")[:100] for t in texts[:5]],
    }

def v27_decode_profiles() -> list[dict]:
    profiles = [
        {"name": "beam2", "num_beams": 2, "max_new_tokens": 32, "priority": 30, "runtime_cost": 1.00},
    ]
    if V27_ENABLE_GREEDY_DECODE:
        profiles.append({"name": "greedy", "num_beams": 1, "max_new_tokens": 24, "priority": 20, "runtime_cost": 0.55})
    if V27_ENABLE_SHORT_DECODE:
        profiles.append({"name": "beam2_short", "num_beams": 2, "max_new_tokens": 18, "priority": 10, "runtime_cost": 0.85})
    return profiles

V27_DECODE_PROFILES = v27_decode_profiles()

V27_RANKER_PROFILES = [
    {
        "name": "likelihood_conservative",
        "ll_weight": 1.00,
        "model_vote_weight": 0.45,
        "retrieval_bonus": 0.12,
        "retrieval_model_agree_bonus": 0.20,
        "prior_weight": 0.04,
        "sanity_weight": 0.85,
        "allow_retrieval_only": False,
        "runtime_cost": 0.10,
    },
    {
        "name": "likelihood_sanity_strong",
        "ll_weight": 1.00,
        "model_vote_weight": 0.35,
        "retrieval_bonus": 0.10,
        "retrieval_model_agree_bonus": 0.20,
        "prior_weight": 0.03,
        "sanity_weight": 1.25,
        "allow_retrieval_only": False,
        "runtime_cost": 0.12,
    },
    {
        "name": "likelihood_retrieval_strict",
        "ll_weight": 1.00,
        "model_vote_weight": 0.35,
        "retrieval_bonus": 0.35,
        "retrieval_model_agree_bonus": 0.35,
        "prior_weight": 0.03,
        "sanity_weight": 0.90,
        "allow_retrieval_only": True,
        "retrieval_only_requires_strict": True,
        "runtime_cost": 0.15,
    },
]

def v27_profile_by_name(name: str | None) -> dict:
    name = name or V27_RANKER_PROFILES[0]["name"]
    for profile in V27_RANKER_PROFILES:
        if profile["name"] == name:
            return profile
    print("[v27] unknown profile", name, "->", V27_RANKER_PROFILES[0]["name"])
    return V27_RANKER_PROFILES[0]

class V27QueryRetriever:
    """Legal train-query retriever.

    It never computes answers from test numbers. It only proposes answer strings
    observed in train response_vi attached to similar train query_vi rows.
    """

    def __init__(self, records: list[dict]):
        self.records = [r for r in records if r.get("_canonical_answer") is not None and r.get("_gold_num") is not None]
        self.texts = [normalize_text_key(r.get("query_vi")) for r in self.records]
        self.token_sets = [query_tokens(t) for t in self.texts]
        self.backend = "jaccard"
        self.vectorizer = None
        self.matrix = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(analyzer=RETRIEVAL_ANALYZER, ngram_range=RETRIEVAL_NGRAM_RANGE, lowercase=False, min_df=1)
            self.matrix = self.vectorizer.fit_transform(self.texts)
            self.backend = "tfidf_char_ngram"
        except Exception as exc:
            print("[v27-retrieval] sklearn unavailable; using Jaccard fallback:", repr(exc))
        print("[v27-retrieval] backend=", self.backend, "records=", len(self.records))

    def search(self, query: str, top_k: int) -> list[dict]:
        if not self.records:
            return []
        q = normalize_text_key(query)
        if self.backend == "tfidf_char_ngram" and self.vectorizer is not None and self.matrix is not None:
            from sklearn.metrics.pairwise import linear_kernel
            qv = self.vectorizer.transform([q])
            sims = linear_kernel(qv, self.matrix).ravel()
            top_idx = sims.argsort()[-top_k:][::-1]
            rows = [(int(i), float(sims[i])) for i in top_idx if float(sims[i]) > 0.0]
        else:
            qtok = query_tokens(q)
            scored = [(i, jaccard(qtok, toks)) for i, toks in enumerate(self.token_sets)]
            scored.sort(key=lambda x: x[1], reverse=True)
            rows = [(i, float(s)) for i, s in scored[:top_k] if s > 0.0]
        qtok = query_tokens(q)
        hits = []
        for i, sim in rows:
            rec = self.records[i]
            key = _safe_num_key(rec.get("_gold_num"))
            if key is None:
                continue
            hits.append({
                "answer_key": key,
                "canonical_answer": rec.get("_canonical_answer"),
                "similarity": float(sim),
                "jaccard": jaccard(qtok, rec.get("_query_tokens") or query_tokens(rec.get("query_vi"))),
                "train_query_vi": rec.get("query_vi", ""),
            })
        return hits

def v27_retrieval_groups(rec: dict, retriever: V27QueryRetriever) -> list[dict]:
    hits = retriever.search(rec.get("query_vi", ""), V27_RETRIEVAL_TOP_K)
    if not hits:
        return []
    grouped = defaultdict(list)
    for hit in hits:
        grouped[hit["answer_key"]].append(hit)
    rows = []
    for key, vals in grouped.items():
        rows.append({
            "key": key,
            "retrieval_count": len(vals),
            "retrieval_top_sim": max(float(v.get("similarity", 0.0)) for v in vals),
            "retrieval_top_jaccard": max(float(v.get("jaccard", 0.0)) for v in vals),
            "retrieval_majority_frac": len(vals) / max(1, len(hits)),
        })
    rows.sort(key=lambda x: (x["retrieval_count"], x["retrieval_top_sim"], x["retrieval_top_jaccard"], ANSWER_PRIORS.get(x["key"], 0)), reverse=True)
    top_count = rows[0]["retrieval_count"] if rows else 0
    second_count = rows[1]["retrieval_count"] if len(rows) > 1 else 0
    for idx, row in enumerate(rows):
        row["retrieval_margin"] = (top_count - second_count) / max(1, len(hits)) if idx == 0 else 0.0
        row["retrieval_rank"] = idx + 1
    return rows

def v27_generate_decode_outputs(adapter_dir: Path, records: list[dict], split_name: str) -> dict[str, list[dict]]:
    out = {}
    for profile in V27_DECODE_PROFILES:
        path = ENSEMBLE_CANDIDATE_DIR / ("%s_v27_%s.json" % (split_name, _safe_name(profile["name"])))
        ENSEMBLE_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
        if path.exists():
            outputs = json.loads(path.read_text(encoding="utf-8"))
            print("[v27-decode] reuse", path)
        else:
            outputs = generate_model_outputs(
                adapter_dir,
                records,
                path,
                max_new_tokens=int(profile.get("max_new_tokens", MAX_NEW_TOKENS)),
                num_beams=int(profile.get("num_beams", NUM_BEAMS)),
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        out[profile["name"]] = outputs
        print("[v27-decode]", split_name, profile["name"], output_sanity(outputs))
    return out

def v27_prior_norm(key: str | None) -> float:
    if key is None or not ANSWER_PRIORS:
        return 0.0
    top = max(ANSWER_PRIORS.values()) if ANSWER_PRIORS else 1
    return math.log1p(ANSWER_PRIORS.get(key, 0)) / max(1e-9, math.log1p(top))

def v27_sanity_penalty(query: str, key: str | None) -> float:
    num = _answer_num_from_key(key)
    if num is None:
        return 3.0
    q = normalize_text_key(query)
    abs_num = abs(float(num))
    penalty = 0.0
    if abs_num > 1_000_000:
        penalty += 3.0
    if abs_num > 100_000:
        penalty += 1.0
    count_markers = [
        "bao nhiêu sản phẩm", "bao nhieu san pham",
        "bao nhiêu cây", "bao nhieu cay",
        "bao nhiêu quyển", "bao nhieu quyen",
        "bao nhiêu người", "bao nhieu nguoi",
        "bao nhiêu năm", "bao nhieu nam",
        "bao nhiêu điểm", "bao nhieu diem",
        "bao nhiêu ô", "bao nhieu o",
        "bao nhiêu quả", "bao nhieu qua",
        "hỏi n bằng bao nhiêu", "hoi n bang bao nhieu",
    ]
    if any(marker in q for marker in count_markers):
        if abs_num > 10_000:
            penalty += 3.0
        elif abs_num > 1_000:
            penalty += 1.5
    if ("xác suất" in q or "xac suat" in q or "tỉ số" in q or "ti so" in q) and abs_num > 1.5:
        penalty += 2.5
    if ("phần trăm" in q or "phan tram" in q) and abs_num > 10_000:
        penalty += 1.0
    return penalty

def v27_build_candidate_rows(records: list[dict], decode_outputs: dict[str, list[dict]], retriever: V27QueryRetriever) -> list[list[dict]]:
    rows = []
    n_profiles = max(1, len(V27_DECODE_PROFILES))
    for idx, rec in enumerate(records):
        groups = {}
        for profile in V27_DECODE_PROFILES:
            name = profile["name"]
            item = decode_outputs[name][idx]
            key = answer_key_from_output(item)
            if key is None:
                continue
            cand = groups.setdefault(key, {
                "key": key,
                "model_votes": 0,
                "model_sources": [],
                "model_priority": 0.0,
                "retrieval_count": 0,
                "retrieval_top_sim": 0.0,
                "retrieval_top_jaccard": 0.0,
                "retrieval_majority_frac": 0.0,
                "retrieval_margin": 0.0,
                "source": "model",
            })
            cand["model_votes"] += 1
            cand["model_sources"].append(name)
            cand["model_priority"] = max(cand["model_priority"], float(profile.get("priority", 0.0)))
            cand["model_frac"] = cand["model_votes"] / n_profiles
        for ret in v27_retrieval_groups(rec, retriever)[:V27_MAX_RETRIEVAL_CANDIDATES]:
            if ret["retrieval_top_sim"] < V27_RETRIEVAL_MIN_SIM or ret["retrieval_top_jaccard"] < V27_RETRIEVAL_MIN_JACCARD:
                continue
            key = ret["key"]
            cand = groups.setdefault(key, {
                "key": key,
                "model_votes": 0,
                "model_sources": [],
                "model_priority": 0.0,
                "model_frac": 0.0,
                "source": "retrieval",
            })
            cand.update(ret)
            cand["source"] = "model_retrieval" if cand.get("model_votes", 0) else "retrieval"
        scored = []
        for cand in groups.values():
            cand["prior_norm"] = v27_prior_norm(cand.get("key"))
            cand["sanity_penalty"] = v27_sanity_penalty(rec.get("query_vi", ""), cand.get("key"))
            seed_score = (
                10.0 * float(cand.get("model_votes", 0))
                + 2.0 * float(cand.get("retrieval_top_sim", 0.0))
                + float(cand.get("prior_norm", 0.0))
                - 0.5 * float(cand.get("sanity_penalty", 0.0))
            )
            cand["_seed_score"] = seed_score
            scored.append(cand)
        scored.sort(key=lambda x: x.get("_seed_score", 0.0), reverse=True)
        rows.append(scored[:V27_MAX_CANDIDATES_PER_ROW])
    return rows

def v27_score_candidates_with_model(adapter_dir: Path, records: list[dict], candidate_rows: list[list[dict]], split_name: str) -> list[list[dict]]:
    cache_path = WORKING_DIR / ("v27_%s_candidate_likelihood.json" % split_name)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        score_map = {(int(x["row_idx"]), str(x["key"])): x for x in payload.get("scores", [])}
        for idx, cands in enumerate(candidate_rows):
            for cand in cands:
                cand.update(score_map.get((idx, str(cand.get("key"))), {}))
        print("[v27-ll] reuse", cache_path)
        return candidate_rows

    flat = []
    seen = set()
    for idx, cands in enumerate(candidate_rows):
        for cand in cands:
            key = cand.get("key")
            if key is None:
                continue
            sig = (idx, str(key))
            if sig in seen:
                continue
            seen.add(sig)
            flat.append({"row_idx": idx, "key": str(key)})
    print("[v27-ll] scoring candidates:", len(flat), "split=", split_name)
    if not flat:
        return candidate_rows

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tok.pad_token_id = SAFE_EOS_ID
    tok.eos_token_id = SAFE_EOS_ID
    tok.padding_side = "right"
    model = load_model_for_generation(adapter_dir)
    device = next(model.parameters()).device
    n_pos = int(getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", N_POSITIONS)))
    max_len = min(int(MAX_LENGTH_STAGE_A), n_pos)
    pad_id = SAFE_EOS_ID
    scores = []

    @torch.inference_mode()
    def score_batch(batch: list[dict]) -> list[dict]:
        encoded = []
        max_b_len = 0
        for item in batch:
            rec = records[int(item["row_idx"])]
            prompt_ids = tok(build_prompt(rec), add_special_tokens=False)["input_ids"]
            answer_ids = tok("Đáp án là: " + str(item["key"]), add_special_tokens=False)["input_ids"]
            if not answer_ids:
                answer_ids = [SAFE_EOS_ID]
            budget = max(1, max_len - len(answer_ids))
            prompt_ids = prompt_ids[-budget:]
            ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            encoded.append((ids, labels, len(answer_ids)))
            max_b_len = max(max_b_len, len(ids))
        input_ids = torch.full((len(batch), max_b_len), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros((len(batch), max_b_len), dtype=torch.long, device=device)
        labels_t = torch.full((len(batch), max_b_len), -100, dtype=torch.long, device=device)
        for bi, (ids, labels, _ans_len) in enumerate(encoded):
            L = len(ids)
            input_ids[bi, :L] = torch.tensor(ids, dtype=torch.long, device=device)
            attention[bi, :L] = 1
            labels_t[bi, :L] = torch.tensor(labels, dtype=torch.long, device=device)
        logits = model(input_ids=input_ids, attention_mask=attention).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels_t[:, 1:]
        mask = shift_labels.ne(-100)
        safe_labels = shift_labels.clamp_min(0)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        sum_lp = (token_lp * mask).sum(dim=1)
        tok_count = mask.sum(dim=1).clamp_min(1)
        mean_lp = sum_lp / tok_count
        out = []
        for bi, item in enumerate(batch):
            out.append({
                "row_idx": int(item["row_idx"]),
                "key": str(item["key"]),
                "mean_logprob": float(mean_lp[bi].detach().cpu().item()),
                "sum_logprob": float(sum_lp[bi].detach().cpu().item()),
                "answer_token_count": int(tok_count[bi].detach().cpu().item()),
            })
        return out

    bs = max(1, int(V27_LIKELIHOOD_BATCH_SIZE))
    for start in tqdm(range(0, len(flat), bs), desc="v27-likelihood:" + split_name):
        scores.extend(score_batch(flat[start:start + bs]))
    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass

    payload = {
        "split": split_name,
        "num_scores": len(scores),
        "batch_size": bs,
        "scores": scores,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    score_map = {(int(x["row_idx"]), str(x["key"])): x for x in scores}
    for idx, cands in enumerate(candidate_rows):
        for cand in cands:
            cand.update(score_map.get((idx, str(cand.get("key"))), {}))
    return candidate_rows

def v27_retrieval_passes(cand: dict, *, strict: bool = False) -> bool:
    min_sim = V27_STRICT_RETRIEVAL_MIN_SIM if strict else V27_RETRIEVAL_MIN_SIM
    min_jacc = V27_STRICT_RETRIEVAL_MIN_JACCARD if strict else V27_RETRIEVAL_MIN_JACCARD
    return (
        float(cand.get("retrieval_top_sim", 0.0)) >= min_sim
        and float(cand.get("retrieval_top_jaccard", 0.0)) >= min_jacc
        and float(cand.get("retrieval_count", 0.0)) > 0.0
    )

def v27_select_candidate(cands: list[dict], profile: dict) -> tuple[str | None, dict]:
    best = None
    rows = []
    for cand in cands:
        key = cand.get("key")
        if key is None:
            continue
        has_model = int(cand.get("model_votes", 0)) > 0
        ret_ok = v27_retrieval_passes(cand, strict=False)
        strict_ret_ok = v27_retrieval_passes(cand, strict=True)
        if not has_model:
            if not bool(profile.get("allow_retrieval_only", False)):
                continue
            if bool(profile.get("retrieval_only_requires_strict", True)) and not strict_ret_ok:
                continue
        ll = float(cand.get("mean_logprob", -100.0))
        score = float(profile.get("ll_weight", 1.0)) * ll
        score += float(profile.get("model_vote_weight", 0.0)) * float(cand.get("model_frac", 0.0))
        score += float(profile.get("prior_weight", 0.0)) * float(cand.get("prior_norm", 0.0))
        score -= float(profile.get("sanity_weight", 0.0)) * float(cand.get("sanity_penalty", 0.0))
        if ret_ok:
            score += float(profile.get("retrieval_bonus", 0.0))
        if ret_ok and has_model:
            score += float(profile.get("retrieval_model_agree_bonus", 0.0))
        row = {
            **cand,
            "score": score,
            "has_model": has_model,
            "retrieval_passes": ret_ok,
            "strict_retrieval_passes": strict_ret_ok,
        }
        rows.append(row)
        tie = (score, int(has_model), int(strict_ret_ok), int(ret_ok), cand.get("model_votes", 0), cand.get("prior_norm", 0.0), str(key))
        if best is None or tie > best[0]:
            best = (tie, key, row)
    rows.sort(key=lambda x: x.get("score", -999.0), reverse=True)
    if best is None:
        return None, {"reason": "no_candidate", "candidates": rows[:5]}
    return best[1], {"reason": "ranked", "chosen": best[2], "candidates": rows[:5]}

def v27_apply_profile(records: list[dict], decode_outputs: dict[str, list[dict]], candidate_rows: list[list[dict]], profile: dict) -> tuple[list[dict], list[dict], dict]:
    outputs = []
    decisions = []
    source_counts = Counter()
    changed_from_beam2 = 0
    base_name = V27_DECODE_PROFILES[0]["name"]
    for idx, rec in enumerate(records):
        base_item = decode_outputs[base_name][idx]
        base_key = answer_key_from_output(base_item)
        key, debug = v27_select_candidate(candidate_rows[idx], profile)
        if key is None:
            key = base_key
            source = "fallback_beam2"
        else:
            chosen = debug.get("chosen", {})
            if int(chosen.get("model_votes", 0)) and int(chosen.get("retrieval_count", 0)):
                source = "model_retrieval"
            elif int(chosen.get("model_votes", 0)):
                source = "model"
            elif int(chosen.get("retrieval_count", 0)):
                source = "retrieval"
            else:
                source = "unknown"
        changed_from_beam2 += int(key != base_key)
        source_counts[source] += 1
        outputs.append(make_answer_item(rec, idx, key, base_item))
        decisions.append({
            "id": rec.get("id", idx),
            "base_key": base_key,
            "chosen_key": key,
            "source": source,
            "profile": profile["name"],
            "debug": debug,
        })
    summary = {
        "strategy": "v27_single_run_candidate_reranker_profile",
        "profile": profile,
        "num_rows": len(records),
        "changed_from_beam2": changed_from_beam2,
        "source_counts": dict(source_counts.most_common()),
        "decode_profiles": V27_DECODE_PROFILES,
        "retrieval_backend": "train_query_vi_response_vi",
        "default_profile_name": V27_DEFAULT_PROFILE_NAME,
        "select_on_valid": V27_SELECT_ON_VALID,
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "uses_original_question_fields": False,
        "uses_type_for_prompt_or_routing": False,
        "uses_arithmetic_or_template_solver": False,
        "uses_external_predictions": False,
        "uses_valid_labels_for_ranker_training": False,
    }
    return outputs, decisions, summary

def v27_prepare_split(adapter_dir: Path, records: list[dict], split_name: str, retriever: V27QueryRetriever) -> dict:
    decode_outputs = v27_generate_decode_outputs(adapter_dir, records, split_name)
    candidate_rows = v27_build_candidate_rows(records, decode_outputs, retriever)
    candidate_rows = v27_score_candidates_with_model(adapter_dir, records, candidate_rows, split_name)
    return {"decode_outputs": decode_outputs, "candidate_rows": candidate_rows}

def v27_evaluate_profiles_on_valid(adapter_dir: Path, retriever: V27QueryRetriever) -> dict:
    pack = v27_prepare_split(adapter_dir, valid_clean, "valid", retriever)
    rows = []
    best = None
    CHECKPOINT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for profile_name, outputs in pack["decode_outputs"].items():
        report = evaluate_predictions(outputs, valid_clean)
        row = {
            "kind": "fixed_decode",
            "name": "fixed::" + profile_name,
            "profile": {"name": profile_name},
            "summary": report["summary"],
            "by_type": report["by_type"],
            "sanity": output_sanity(outputs),
        }
        rows.append(row)
        result = {"row": row, "outputs": outputs, "report": report, "decisions": [], "selector": {"kind": "fixed_decode", "profile_name": profile_name}}
        key = _score_summary_key(report["summary"], tie_break=0.0)
        if best is None or key > best[0]:
            best = (key, result)
        (CHECKPOINT_EVAL_DIR / ("valid_v27_fixed_%s_report.json" % _safe_name(profile_name))).write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[v27-valid-fixed]", profile_name, report["summary"])

    default_result = None
    for profile in V27_RANKER_PROFILES:
        outputs, decisions, summary = v27_apply_profile(valid_clean, pack["decode_outputs"], pack["candidate_rows"], profile)
        report = evaluate_predictions(outputs, valid_clean)
        row = {
            "kind": "ranker_profile",
            "name": "ranker::" + profile["name"],
            "profile": profile,
            "summary": report["summary"],
            "by_type": report["by_type"],
            "profile_summary": summary,
            "decisions_sample": decisions[:30],
        }
        rows.append(row)
        result = {"row": row, "outputs": outputs, "report": report, "decisions": decisions, "selector": {"kind": "ranker_profile", "profile": profile}}
        if profile["name"] == V27_DEFAULT_PROFILE_NAME:
            default_result = result
        key = _score_summary_key(report["summary"], tie_break=-float(profile.get("runtime_cost", 0.0)))
        if best is None or key > best[0]:
            best = (key, result)
        (CHECKPOINT_EVAL_DIR / ("valid_v27_ranker_%s_report.json" % _safe_name(profile["name"]))).write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[v27-valid-ranker]", profile["name"], report["summary"], "changed=", row["profile_summary"]["changed_from_beam2"])

    selected = best[1] if V27_SELECT_ON_VALID else (default_result or best[1])
    payload = {
        "strategy": "v27_single_run_candidate_reranker_valid_audit",
        "selection_mode": "valid_score" if V27_SELECT_ON_VALID else "fixed_default_profile",
        "default_profile_name": V27_DEFAULT_PROFILE_NAME,
        "candidates": rows,
        "selected": selected["row"],
        "decode_profiles": V27_DECODE_PROFILES,
        "ranker_profiles": V27_RANKER_PROFILES,
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "uses_original_question_fields": False,
        "uses_type_for_prompt_or_routing": False,
        "uses_arithmetic_or_template_solver": False,
        "uses_external_predictions": False,
        "uses_valid_labels_for_profile_selection": bool(V27_SELECT_ON_VALID),
        "uses_valid_labels_for_ranker_training": False,
    }
    CHECKPOINT_SELECTION_REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_CHECKPOINT_INFO_PATH.write_text(json.dumps(selected["row"], ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_VALID_OUTPUT_PATH.write_text(json.dumps(selected["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_VALID_REPORT_PATH.write_text(json.dumps(selected["report"], ensure_ascii=False, indent=2), encoding="utf-8")
    if selected["decisions"]:
        (WORKING_DIR / "valid_v27_ranker_decisions.json").write_text(json.dumps(selected["decisions"], ensure_ascii=False, indent=2), encoding="utf-8")
    print("[v27-select-valid]", selected["row"]["name"], selected["report"]["summary"])
    return {**selected, "payload": payload}

def train_and_get_adapter() -> Path:
    entries = candidate_entries_from_runs()
    final_entry = next((e for e in entries if e["kind"] == "model_final"), None)
    if final_entry is None:
        raise RuntimeError("V27 expects one final adapter from the v23-style 7-epoch answer-only run.")
    return Path(final_entry["adapter_dir"])

def run_v27_valid():
    adapter_dir = train_and_get_adapter()
    retriever = V27QueryRetriever(train_clean)
    selected = v27_evaluate_profiles_on_valid(adapter_dir, retriever)
    summary = {
        "strategy": "v27_single_run_candidate_reranker",
        "selected": selected["row"],
        "selected_adapter_dir": str(adapter_dir),
        "selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),
        "selection_mode": "valid_score" if V27_SELECT_ON_VALID else "fixed_default_profile",
        "uses_valid_labels_for_profile_selection": bool(V27_SELECT_ON_VALID),
        "uses_valid_labels_for_ranker_training": False,
    }
    report, payload = save_outputs_and_optional_report(valid_clean, selected["outputs"], VALID_OUTPUT_PATH, VALID_REPORT_PATH, ENSEMBLE_RANKER_REPORT_PATH, summary)
    if report is not None:
        print("[v27:valid]", report["summary"])
    return selected, payload

def run_v27_test():
    adapter_dir = train_and_get_adapter()
    retriever = V27QueryRetriever(train_clean)
    selected_valid = None
    if V27_SELECT_ON_VALID or V27_EVAL_VALID_PROFILES_IN_PHASE2:
        selected_valid = v27_evaluate_profiles_on_valid(adapter_dir, retriever)
        if V27_SELECT_ON_VALID:
            profile = selected_valid["selector"]["profile"] if selected_valid["selector"]["kind"] == "ranker_profile" else v27_profile_by_name(V27_DEFAULT_PROFILE_NAME)
        else:
            profile = v27_profile_by_name(V27_DEFAULT_PROFILE_NAME)
    else:
        profile = v27_profile_by_name(V27_DEFAULT_PROFILE_NAME)
        payload = {
            "strategy": "v27_single_run_candidate_reranker_fixed_profile",
            "selection_mode": "fixed_default_profile_no_valid_selection",
            "selected": {"kind": "ranker_profile", "name": "ranker::" + profile["name"], "profile": profile},
            "decode_profiles": V27_DECODE_PROFILES,
            "ranker_profiles": V27_RANKER_PROFILES,
            "uses_valid_labels_for_profile_selection": False,
        }
        CHECKPOINT_SELECTION_REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        SELECTED_CHECKPOINT_INFO_PATH.write_text(json.dumps(payload["selected"], ensure_ascii=False, indent=2), encoding="utf-8")

    test_records = load_records(TEST_FILE)
    pack = v27_prepare_split(adapter_dir, test_records, "test", retriever)
    outputs, decisions, summary = v27_apply_profile(test_records, pack["decode_outputs"], pack["candidate_rows"], profile)
    base_name = V27_DECODE_PROFILES[0]["name"]
    MODEL_TEST_OUTPUT_PATH.write_text(json.dumps(pack["decode_outputs"][base_name], ensure_ascii=False, indent=2), encoding="utf-8")
    (WORKING_DIR / "test_v27_ranker_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(summary)
    summary.update({
        "selected_profile": profile,
        "selected_from_valid": selected_valid["row"] if selected_valid else None,
        "selected_adapter_dir": str(adapter_dir),
        "selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),
        "selection_mode": "valid_score" if V27_SELECT_ON_VALID else "fixed_default_profile",
        "uses_valid_labels_for_profile_selection": bool(V27_SELECT_ON_VALID),
        "uses_valid_labels_for_ranker_training": False,
        "uses_external_predictions": False,
    })
    _report, payload = save_outputs_and_optional_report(test_records, outputs, TEST_OUTPUT_PATH, VALID_REPORT_PATH, ENSEMBLE_RANKER_REPORT_PATH, summary)
    print("[phase2:v27] profile=", profile["name"], "wrote", TEST_OUTPUT_PATH)
    return outputs, payload

if RUN_MODE == "phase1":
    _selected, _payload = run_v27_valid()
elif RUN_MODE == "phase2":
    if not TEST_FILE.exists():
        raise FileNotFoundError("RUN_MODE='phase2' requires test.json")
    _outputs, _payload = run_v27_test()
else:
    raise ValueError("Unknown RUN_MODE=" + str(RUN_MODE))
'''


def source_of(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def clean_notebook(nb: dict) -> dict:
    nb.setdefault("metadata", {}).pop("widgets", None)
    nb.setdefault("metadata", {}).pop("papermill", None)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return nb


def replace_v23_profile_block(source: str) -> str:
    start = source.find("DEFAULT_RETRIEVAL_GATE = {\"enabled\": False}")
    if start < 0:
        raise RuntimeError("Cannot locate v23 retrieval/profile block")
    end = source.find("# RL disabled.", start)
    if end < 0:
        raise RuntimeError("Cannot locate end of v23 profile block")
    return source[:start] + V27_CONFIG_BLOCK + source[end:]


def patch_config_cell(source: str) -> str:
    source = source.replace("# 1. Config - v23_legal_no_solver_ranker", "# 1. Config - v27_single_run_candidate_reranker")
    source = source.replace('NOTEBOOK_VERSION = "v23_legal_no_solver_ranker"', 'NOTEBOOK_VERSION = "v27_single_run_candidate_reranker"')
    source = source.replace('RUN_MODE = "phase2"  # phase1 writes valid_output/report; phase2 writes test_predictions.json.', 'RUN_MODE = os.environ.get("RUN_MODE", os.environ.get("RUN_MMODE", "phase2"))  # phase1 writes valid_output/report; phase2 writes test_predictions.json.')
    source = source.replace('RUN_MODE = os.environ.get("RUN_MODE", os.environ.get("RUN_MMODE", "phase1"))  # phase1 writes valid_output/report; phase2 writes test_predictions.json.', 'RUN_MODE = os.environ.get("RUN_MODE", os.environ.get("RUN_MMODE", "phase2"))  # phase1 writes valid_output/report; phase2 writes test_predictions.json.')
    source = source.replace('ARTIFACT_PREFIX = "v23_legal_no_solver_ranker"', 'ARTIFACT_PREFIX = "v27_single_run_candidate_reranker"')
    source = source.replace("STAGE_A_LR = 0.003", 'STAGE_A_LR = float(os.environ.get("STAGE_A_LR", os.environ.get("LR", "0.003")))')
    source = source.replace("# Query-only retrieval config. Disabled in answer-only notebook.", "# Query-only retrieval config. Used only as a legal candidate source.")
    source = replace_v23_profile_block(source)
    return source


def patch_manifest_cell(source: str) -> str:
    source = source.replace('"uses_valid_labels_for_profile_selection": True,', '"uses_valid_labels_for_profile_selection": bool(globals().get("V27_SELECT_ON_VALID", False)),')
    source = source.replace('"retrieval_basis": "query_vi_train_response_retrieval_no_solver_valid_profile_select",', '"retrieval_basis": "train_query_vi_response_vi_candidate_source_only",')
    source = source.replace(
        '"selected_retrieval_gate": globals().get("SELECTED_RETRIEVAL_GATE"),\n',
        '"selected_retrieval_gate": globals().get("SELECTED_RETRIEVAL_GATE"),\n'
        '        "v27_default_profile_name": globals().get("V27_DEFAULT_PROFILE_NAME", None),\n'
        '        "v27_select_on_valid": globals().get("V27_SELECT_ON_VALID", None),\n'
        '        "v27_eval_valid_profiles_in_phase2": globals().get("V27_EVAL_VALID_PROFILES_IN_PHASE2", None),\n'
        '        "v27_decode_profiles": globals().get("V27_DECODE_PROFILES", None),\n'
        '        "v27_ranker_profiles": globals().get("V27_RANKER_PROFILES", None),\n',
    )
    source = source.replace("v23_legal_no_solver_ranker", "v27_single_run_candidate_reranker")
    return source


def build_notebook() -> dict:
    v23 = load_module(V23_SCRIPT, "create_v23_no_solver_for_v27")
    nb = v23.build_notebook()
    for idx, cell in enumerate(nb.get("cells", [])):
        src = source_of(cell)
        if not src:
            continue
        src = src.replace("# V23 - Legal No-Solver Ranker", "# V27 - Single-Run Candidate Reranker")
        src = src.replace(
            "Single-seed answer-only run plus a legal no-solver decoder/retrieval ranker.",
            "Single-seed v23-style answer-only run plus in-notebook candidate generation and GPT-2 likelihood reranking.",
        )
        src = src.replace(
            "Retrieval, when enabled by valid selection, only copies answers",
            "Retrieval is used only as an in-notebook candidate source and copies answers",
        )
        src = src.replace("v23_legal_no_solver_ranker", "v27_single_run_candidate_reranker")
        src = src.replace("V23", "V27")
        src = src.replace("v23", "v27")
        src = src.replace("v16 legal pipeline", "v27 legal pipeline")
        if idx == 2:
            src = patch_config_cell(src)
        if idx == 8:
            marker = "# ============================================================\n# 8. V27 legal no-solver decoder + train-query retrieval ranker"
            if marker not in src:
                raise RuntimeError("Cannot locate v23/v27 tail marker in cell 8")
            src = src.split(marker, 1)[0] + V27_TAIL
        if idx == 9:
            src = patch_manifest_cell(src)
        set_source(cell, src)
    return clean_notebook(nb)


def validate_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = source_of(cell)
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
