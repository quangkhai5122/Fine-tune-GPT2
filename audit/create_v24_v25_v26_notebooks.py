"""Create v24/v25/v26 legal no-solver experiment notebooks.

The three generated notebooks target different risk/reward profiles:
- v24: row-level retrieval-feature ranker, no solver, train-query retrieval
- v25: answer-only SFT with query-only distractor oversampling
- v26: answer-only diversity from epoch 7/8 and decode profiles
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V17_SCRIPT = ROOT / "audit" / "create_v17_answer_only_notebooks.py"
V23_SCRIPT = ROOT / "audit" / "create_v23_legal_no_solver_ranker_notebook.py"

V24_NOTEBOOK = ROOT / "finetune_gpt2_for_math_v24_legal_retrieval_feature_ranker.ipynb"
V25_NOTEBOOK = ROOT / "finetune_gpt2_for_math_v25_distractor_aug_answer_only.ipynb"
V26_NOTEBOOK = ROOT / "finetune_gpt2_for_math_v26_diverse_answer_only_ranker.ipynb"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v17 = load_module(V17_SCRIPT, "create_v17_answer_only_notebooks_for_v24_v26")
v23 = load_module(V23_SCRIPT, "create_v23_no_solver_for_v24")


V24_TAIL = r'''

# ============================================================
# 8b. V24 row-level retrieval-feature ranker, no solver
# ============================================================
V24_DECODE_PROFILES = [
    {"name": "beam2", "num_beams": 2, "max_new_tokens": 32, "priority": 30, "runtime_cost": 1.00},
    {"name": "greedy", "num_beams": 1, "max_new_tokens": 24, "priority": 20, "runtime_cost": 0.55},
    {"name": "beam2_short", "num_beams": 2, "max_new_tokens": 18, "priority": 10, "runtime_cost": 0.85},
]

V24_RANKER_PROFILES = [
    {
        "name": "model_vote_prior",
        "allow_retrieval_only": False,
        "retrieval_bonus": 0.00,
        "model_vote_weight": 2.00,
        "prior_weight": 0.12,
        "min_sim": 1.00,
        "min_jaccard": 1.00,
        "min_majority_frac": 1.00,
        "min_margin": 1.00,
        "runtime_cost": 0.00,
    },
    {
        "name": "retrieval_agree_boost",
        "allow_retrieval_only": False,
        "retrieval_bonus": 0.75,
        "model_vote_weight": 1.80,
        "prior_weight": 0.10,
        "min_sim": 0.82,
        "min_jaccard": 0.62,
        "min_majority_frac": 0.35,
        "min_margin": 0.08,
        "runtime_cost": 0.05,
    },
    {
        "name": "retrieval_agree_strict",
        "allow_retrieval_only": False,
        "retrieval_bonus": 1.20,
        "model_vote_weight": 1.60,
        "prior_weight": 0.08,
        "min_sim": 0.90,
        "min_jaccard": 0.70,
        "min_majority_frac": 0.45,
        "min_margin": 0.15,
        "runtime_cost": 0.10,
    },
    {
        "name": "ultra_strict_retrieval_only",
        "allow_retrieval_only": True,
        "retrieval_bonus": 1.50,
        "model_vote_weight": 1.55,
        "prior_weight": 0.05,
        "min_sim": 0.96,
        "min_jaccard": 0.82,
        "min_majority_frac": 0.70,
        "min_margin": 0.35,
        "runtime_cost": 0.15,
    },
]

V24_RETRIEVAL_TOP_K = 12
V24_RETRIEVAL_INCLUDE_VALID_FOR_TEST = os.environ.get("V24_INCLUDE_VALID_RETRIEVAL", "0") == "1"

def _v24_prior_norm(key: str | None) -> float:
    if key is None or not ANSWER_PRIORS:
        return 0.0
    top = max(ANSWER_PRIORS.values()) if ANSWER_PRIORS else 1
    return math.log1p(ANSWER_PRIORS.get(key, 0)) / max(1e-9, math.log1p(top))

def _v24_fixed_summary(profile: dict, outputs: list[dict], records: list[dict]) -> dict:
    report = evaluate_predictions(outputs, records)
    row = {
        "kind": "fixed_decode",
        "name": "fixed::" + profile["name"],
        "profile": profile,
        "summary": report["summary"],
        "by_type": report["by_type"],
    }
    path = CHECKPOINT_EVAL_DIR / ("valid_v24_fixed_%s_report.json" % _safe_name(profile["name"]))
    CHECKPOINT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"row": row, "outputs": outputs, "report": report, "selector": {"kind": "fixed_decode", "profile": profile}}

def v24_generate_decode_outputs(adapter_dir: Path, records: list[dict], split_name: str) -> dict[str, list[dict]]:
    out = {}
    for profile in V24_DECODE_PROFILES:
        out[profile["name"]] = generate_outputs_for_profile(adapter_dir, records, split_name, profile)
    return out

def v24_retriever_for_split(split_name: str) -> StrictQueryRetriever:
    # Valid profile selection must not retrieve from valid labels. For phase2,
    # optional valid-label retrieval can be enabled by env if the competition
    # setup permits valid response_vi as an external memory source.
    records = list(train_clean)
    if split_name == "test" and V24_RETRIEVAL_INCLUDE_VALID_FOR_TEST:
        records = records + list(valid_clean)
    return StrictQueryRetriever(records)

def v24_build_row_candidates(
    rec: dict,
    idx: int,
    decode_outputs: dict[str, list[dict]],
    retriever: StrictQueryRetriever,
) -> tuple[dict[str, dict], dict | None]:
    groups = {}
    fallback_item = None
    for profile in V24_DECODE_PROFILES:
        name = profile["name"]
        item = decode_outputs[name][idx]
        if fallback_item is None:
            fallback_item = item
        key = answer_key_from_output(item)
        if key is None:
            continue
        row = groups.setdefault(key, {
            "key": key,
            "model_votes": 0,
            "model_sources": [],
            "model_priority": 0.0,
            "retrieval_count": 0,
            "retrieval_top_sim": 0.0,
            "retrieval_top_jaccard": 0.0,
            "retrieval_majority_frac": 0.0,
            "retrieval_margin": 0.0,
        })
        row["model_votes"] += 1
        row["model_sources"].append(name)
        row["model_priority"] = max(row["model_priority"], float(profile.get("priority", 0.0)))

    hits = retriever.search(rec.get("query_vi", ""), V24_RETRIEVAL_TOP_K)
    if hits:
        by_key = defaultdict(list)
        for hit in hits:
            by_key[hit["answer_key"]].append(hit)
        ranked = []
        for key, vals in by_key.items():
            ranked.append({
                "key": key,
                "count": len(vals),
                "top_sim": max(float(v.get("similarity", 0.0)) for v in vals),
                "top_jaccard": max(float(v.get("jaccard", 0.0)) for v in vals),
            })
        ranked.sort(key=lambda x: (x["count"], x["top_sim"], x["top_jaccard"], ANSWER_PRIORS.get(x["key"], 0)), reverse=True)
        top_count = ranked[0]["count"] if ranked else 0
        second_count = ranked[1]["count"] if len(ranked) > 1 else 0
        for row in ranked[:3]:
            key = row["key"]
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
            })
            cand["retrieval_count"] = int(row["count"])
            cand["retrieval_top_sim"] = float(row["top_sim"])
            cand["retrieval_top_jaccard"] = float(row["top_jaccard"])
            cand["retrieval_majority_frac"] = float(row["count"]) / max(1, len(hits))
            cand["retrieval_margin"] = float(top_count - second_count) / max(1, len(hits)) if row is ranked[0] else 0.0
    return groups, fallback_item

def v24_retrieval_passes(cand: dict, profile: dict) -> bool:
    return (
        float(cand.get("retrieval_top_sim", 0.0)) >= float(profile.get("min_sim", 1.0))
        and float(cand.get("retrieval_top_jaccard", 0.0)) >= float(profile.get("min_jaccard", 1.0))
        and float(cand.get("retrieval_majority_frac", 0.0)) >= float(profile.get("min_majority_frac", 1.0))
        and float(cand.get("retrieval_margin", 0.0)) >= float(profile.get("min_margin", 1.0))
    )

def v24_select_key_for_row(cands: dict[str, dict], profile: dict) -> tuple[str | None, dict]:
    n_models = max(1, len(V24_DECODE_PROFILES))
    best = None
    rows = []
    for key, cand in cands.items():
        has_model = int(cand.get("model_votes", 0)) > 0
        ret_ok = v24_retrieval_passes(cand, profile)
        if not has_model and not (bool(profile.get("allow_retrieval_only", False)) and ret_ok):
            continue
        model_frac = float(cand.get("model_votes", 0)) / n_models
        score = (
            float(profile.get("model_vote_weight", 1.0)) * model_frac
            + float(profile.get("prior_weight", 0.0)) * _v24_prior_norm(key)
            + 0.001 * float(cand.get("model_priority", 0.0))
        )
        if ret_ok:
            score += float(profile.get("retrieval_bonus", 0.0))
            score += 0.20 * float(cand.get("retrieval_majority_frac", 0.0))
            score += 0.10 * float(cand.get("retrieval_margin", 0.0))
        debug = {
            **cand,
            "score": score,
            "has_model": has_model,
            "retrieval_passes": ret_ok,
            "model_frac": model_frac,
        }
        rows.append(debug)
        tie = (score, int(has_model), int(ret_ok), cand.get("model_votes", 0), ANSWER_PRIORS.get(key, 0), key)
        if best is None or tie > best[0]:
            best = (tie, key, debug)
    if best is None:
        return None, {"reason": "no_candidate", "candidates": rows[:5]}
    rows.sort(key=lambda x: x["score"], reverse=True)
    return best[1], {"reason": "ranked", "chosen": best[2], "candidates": rows[:5]}

def v24_apply_ranker(
    records: list[dict],
    decode_outputs: dict[str, list[dict]],
    retriever: StrictQueryRetriever,
    ranker_profile: dict,
) -> tuple[list[dict], list[dict], dict]:
    outputs = []
    decisions = []
    chosen_counts = Counter()
    changed_from_beam2 = 0
    for idx, rec in enumerate(records):
        cands, fallback = v24_build_row_candidates(rec, idx, decode_outputs, retriever)
        key, debug = v24_select_key_for_row(cands, ranker_profile)
        beam2_key = answer_key_from_output(decode_outputs[V24_DECODE_PROFILES[0]["name"]][idx])
        changed_from_beam2 += int(key != beam2_key)
        chosen_counts[key or "<none>"] += 1
        outputs.append(make_answer_item(rec, idx, key, fallback))
        decisions.append({
            "id": rec.get("id", idx),
            "chosen_key": key,
            "beam2_key": beam2_key,
            "profile": ranker_profile["name"],
            "debug": debug,
        })
    summary = {
        "strategy": "v24_legal_retrieval_feature_ranker_profile",
        "profile": ranker_profile,
        "num_rows": len(records),
        "changed_from_beam2": changed_from_beam2,
        "chosen_top": dict(chosen_counts.most_common(20)),
        "decode_profiles": V24_DECODE_PROFILES,
        "retrieval_backend": retriever.backend,
        "retrieval_include_valid_for_test": V24_RETRIEVAL_INCLUDE_VALID_FOR_TEST,
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "uses_original_question_fields": False,
        "uses_type_for_prompt_or_routing": False,
        "uses_arithmetic_or_template_solver": False,
        "uses_valid_labels_for_profile_selection": True,
        "uses_valid_labels_for_ranker_training": False,
    }
    return outputs, decisions, summary

def select_v24_on_valid(adapter_dir: Path) -> dict:
    retriever = v24_retriever_for_split("valid")
    decode_outputs = v24_generate_decode_outputs(adapter_dir, valid_clean, "valid")
    rows = []
    best = None
    for profile in V24_DECODE_PROFILES:
        result = _v24_fixed_summary(profile, decode_outputs[profile["name"]], valid_clean)
        rows.append(result["row"])
        key = _score_summary_key(result["report"]["summary"], tie_break=-float(profile.get("runtime_cost", 1.0)))
        if best is None or key > best[0]:
            best = (key, result)
        print("[v24-fixed]", profile["name"], result["report"]["summary"])
    for ranker_profile in V24_RANKER_PROFILES:
        outputs, decisions, summary = v24_apply_ranker(valid_clean, decode_outputs, retriever, ranker_profile)
        report = evaluate_predictions(outputs, valid_clean)
        row = {
            "kind": "ranker_profile",
            "name": "ranker::" + ranker_profile["name"],
            "profile": ranker_profile,
            "summary": report["summary"],
            "by_type": report["by_type"],
            "profile_summary": summary,
            "decisions_sample": decisions[:30],
        }
        path = CHECKPOINT_EVAL_DIR / ("valid_v24_ranker_%s_report.json" % _safe_name(ranker_profile["name"]))
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {"row": row, "outputs": outputs, "report": report, "decisions": decisions, "selector": {"kind": "ranker_profile", "profile": ranker_profile}}
        rows.append(row)
        key = _score_summary_key(report["summary"], tie_break=-float(ranker_profile.get("runtime_cost", 0.0)))
        if best is None or key > best[0]:
            best = (key, result)
        print("[v24-ranker]", ranker_profile["name"], report["summary"], "changed=", summary["changed_from_beam2"])
    selected = best[1]
    payload = {
        "strategy": "v24_legal_retrieval_feature_ranker_valid_selection",
        "selection_metric": "max(raw_score, exact10, extractable, -bucket0, -runtime_cost)",
        "candidates": rows,
        "selected": selected["row"],
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "uses_original_question_fields": False,
        "uses_type_for_prompt_or_routing": False,
        "uses_arithmetic_or_template_solver": False,
        "uses_valid_labels_for_profile_selection": True,
        "uses_valid_labels_for_ranker_training": False,
    }
    CHECKPOINT_SELECTION_REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_CHECKPOINT_INFO_PATH.write_text(json.dumps(selected["row"], ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_VALID_OUTPUT_PATH.write_text(json.dumps(selected["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_VALID_REPORT_PATH.write_text(json.dumps(selected["report"], ensure_ascii=False, indent=2), encoding="utf-8")
    if selected["selector"]["kind"] == "ranker_profile":
        (WORKING_DIR / "valid_v24_ranker_decisions.json").write_text(json.dumps(selected["decisions"], ensure_ascii=False, indent=2), encoding="utf-8")
    print("[v24-select]", selected["row"]["name"], selected["report"]["summary"])
    return {**selected, "payload": payload}

def train_and_get_adapter() -> Path:
    entries = candidate_entries_from_runs()
    final_entry = next((e for e in entries if e["kind"] == "model_final"), None)
    if final_entry is None:
        raise RuntimeError("V24 expects one final adapter from the 7-epoch answer-only run.")
    return Path(final_entry["adapter_dir"])

def run_v24_valid():
    adapter_dir = train_and_get_adapter()
    selected = select_v24_on_valid(adapter_dir)
    summary = {
        "strategy": "v24_legal_retrieval_feature_ranker",
        "selected": selected["row"],
        "selected_adapter_dir": str(adapter_dir),
        "selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),
        "uses_valid_labels_for_profile_selection": True,
        "uses_valid_labels_for_ranker_training": False,
    }
    report, payload = save_outputs_and_optional_report(valid_clean, selected["outputs"], VALID_OUTPUT_PATH, VALID_REPORT_PATH, ENSEMBLE_RANKER_REPORT_PATH, summary)
    if report is not None:
        print("[v24:valid]", report["summary"])
    return selected, payload

def run_v24_test():
    adapter_dir = train_and_get_adapter()
    selected_valid = select_v24_on_valid(adapter_dir)
    selector = selected_valid["selector"]
    test_records = load_records(TEST_FILE)
    test_decode_outputs = v24_generate_decode_outputs(adapter_dir, test_records, "test")
    if selector["kind"] == "fixed_decode":
        profile_name = selector["profile"]["name"]
        outputs = test_decode_outputs[profile_name]
        decisions = []
        summary = {"strategy": "v24_legal_retrieval_feature_ranker", "selected": selected_valid["row"], "source": "fixed_decode"}
    else:
        retriever = v24_retriever_for_split("test")
        outputs, decisions, summary = v24_apply_ranker(test_records, test_decode_outputs, retriever, selector["profile"])
        (WORKING_DIR / "test_v24_ranker_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    MODEL_TEST_OUTPUT_PATH.write_text(json.dumps(test_decode_outputs[V24_DECODE_PROFILES[0]["name"]], ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(summary)
    summary.update({
        "selected_from_valid": selected_valid["row"],
        "selected_adapter_dir": str(adapter_dir),
        "selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),
        "uses_valid_labels_for_profile_selection": True,
        "uses_valid_labels_for_ranker_training": False,
        "retrieval_include_valid_for_test": V24_RETRIEVAL_INCLUDE_VALID_FOR_TEST,
    })
    _report, payload = save_outputs_and_optional_report(test_records, outputs, TEST_OUTPUT_PATH, VALID_REPORT_PATH, ENSEMBLE_RANKER_REPORT_PATH, summary)
    print("[phase2:v24] wrote", TEST_OUTPUT_PATH)
    return outputs, payload

if RUN_MODE == "phase1":
    _selected, _payload = run_v24_valid()
elif RUN_MODE == "phase2":
    if not TEST_FILE.exists():
        raise FileNotFoundError("RUN_MODE='phase2' requires test.json")
    _outputs, _payload = run_v24_test()
else:
    raise ValueError("Unknown RUN_MODE=" + str(RUN_MODE))
'''


V26_TAIL = r'''
# ============================================================
# 8. V26 diverse answer-only ranker: epoch 7/8 + decode profiles
# ============================================================
V26_DECODE_PROFILES = [
    {"name": "beam2", "num_beams": 2, "max_new_tokens": 32, "priority": 30, "runtime_cost": 1.00},
    {"name": "greedy", "num_beams": 1, "max_new_tokens": 24, "priority": 20, "runtime_cost": 0.55},
]

def _bucket(summary: dict, score: int) -> int:
    buckets = summary.get("buckets", {})
    return int(buckets.get(score, buckets.get(str(score), 0)))

def _safe_candidate_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))

def _selection_key(row: dict, *, tie_break: float = 0.0):
    summary = row["summary"]
    return (
        int(summary.get("raw_score", -1)),
        _bucket(summary, 10),
        int(summary.get("extractable", 0)),
        -int(summary.get("buckets", {}).get("0", summary.get("buckets", {}).get(0, 0))),
        float(tie_break),
    )

def _decode_candidate_path(split_name: str, candidate_name: str, decode_name: str) -> Path:
    ENSEMBLE_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    return ENSEMBLE_CANDIDATE_DIR / ("%s_%s__%s.json" % (split_name, _safe_candidate_name(candidate_name), _safe_candidate_name(decode_name)))

def generate_v26_candidates(records: list[dict], split_name: str) -> list[dict]:
    entries = candidate_entries_from_runs()
    out = []
    for entry in entries:
        for profile in V26_DECODE_PROFILES:
            name = entry["name"] + "::" + profile["name"]
            path = _decode_candidate_path(split_name, entry["name"], profile["name"])
            if path.exists():
                outputs = json.loads(path.read_text(encoding="utf-8"))
                print("[v26-candidate] reuse", path)
            else:
                outputs = generate_model_outputs(
                    entry["adapter_dir"],
                    records,
                    path,
                    max_new_tokens=int(profile.get("max_new_tokens", MAX_NEW_TOKENS)),
                    num_beams=int(profile.get("num_beams", NUM_BEAMS)),
                )
            out.append({
                **entry,
                "name": name,
                "decode_profile": profile,
                "outputs": outputs,
                "path": str(path),
                "priority": float(entry.get("priority", 0.0)) * 10.0 + float(profile.get("priority", 0.0)),
            })
    print("[v26-candidates]", len(out), [c["name"] for c in out])
    return out

def v26_consensus_outputs(candidates: list[dict], records: list[dict]) -> tuple[list[dict], dict]:
    outputs, summary = choose_answer_only_consensus(candidates, records)
    summary["strategy"] = "v26_diverse_answer_only_consensus"
    summary["decode_profiles"] = V26_DECODE_PROFILES
    return outputs, summary

def v26_evaluate_selector(records: list[dict], candidates: list[dict]) -> dict:
    CHECKPOINT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    best = None
    for cand in candidates:
        report = evaluate_predictions(cand["outputs"], records)
        row = {
            "kind": "individual",
            "name": cand["name"],
            "seed": cand.get("seed"),
            "epoch": cand.get("epoch"),
            "adapter_dir": str(cand["adapter_dir"]),
            "decode_profile": cand.get("decode_profile"),
            "summary": report["summary"],
            "by_type": report["by_type"],
        }
        rows.append(row)
        (CHECKPOINT_EVAL_DIR / ("valid_v26_%s_report.json" % _safe_candidate_name(cand["name"]))).write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {"selector": {"kind": "individual", "name": cand["name"]}, "row": row, "outputs": cand["outputs"], "report": report}
        key = _selection_key(row, tie_break=float(cand.get("priority", 0.0)))
        if best is None or key > best[0]:
            best = (key, result)
        print("[v26-individual]", cand["name"], report["summary"])

    consensus_outputs, consensus_summary = v26_consensus_outputs(candidates, records)
    report = evaluate_predictions(consensus_outputs, records)
    row = {
        "kind": "consensus",
        "name": "consensus_all",
        "summary": report["summary"],
        "by_type": report["by_type"],
        "consensus_summary": consensus_summary,
    }
    rows.append(row)
    (CHECKPOINT_EVAL_DIR / "valid_v26_consensus_all_report.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"selector": {"kind": "consensus", "name": "consensus_all"}, "row": row, "outputs": consensus_outputs, "report": report}
    key = _selection_key(row, tie_break=-0.25)
    if best is None or key > best[0]:
        best = (key, result)
    print("[v26-consensus]", report["summary"])

    selected = best[1]
    payload = {
        "strategy": "v26_diverse_answer_only_valid_selection",
        "selection_metric": "max(raw_score, exact10, extractable, -bucket0)",
        "candidates": rows,
        "selected": selected["row"],
        "decode_profiles": V26_DECODE_PROFILES,
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "uses_original_question_fields": False,
        "uses_type_for_prompt_or_routing": False,
        "uses_arithmetic_or_template_solver": False,
        "uses_valid_labels_for_checkpoint_selection": True,
        "uses_valid_labels_for_ranker_training": False,
    }
    CHECKPOINT_SELECTION_REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_CHECKPOINT_INFO_PATH.write_text(json.dumps(selected["row"], ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_VALID_OUTPUT_PATH.write_text(json.dumps(selected["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_VALID_REPORT_PATH.write_text(json.dumps(selected["report"], ensure_ascii=False, indent=2), encoding="utf-8")
    print("[v26-select]", selected["row"]["name"], selected["report"]["summary"])
    return {**selected, "payload": payload}

def v26_outputs_for_selector(candidates: list[dict], records: list[dict], selector: dict) -> tuple[list[dict], dict]:
    if selector["kind"] == "individual":
        cand = next(c for c in candidates if c["name"] == selector["name"])
        return cand["outputs"], {"strategy": "v26_diverse_answer_only_ranker", "selector": selector}
    outputs, summary = v26_consensus_outputs(candidates, records)
    return outputs, summary

def run_v26_valid():
    candidates = generate_v26_candidates(valid_clean, "valid")
    selected = v26_evaluate_selector(valid_clean, candidates)
    summary = {
        "strategy": "v26_diverse_answer_only_ranker",
        "selected": selected["row"],
        "selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),
        "uses_valid_labels_for_checkpoint_selection": True,
        "uses_valid_labels_for_ranker_training": False,
    }
    report, payload = save_outputs_and_optional_report(valid_clean, selected["outputs"], VALID_OUTPUT_PATH, VALID_REPORT_PATH, ENSEMBLE_RANKER_REPORT_PATH, summary)
    MODEL_VALID_OUTPUT_PATH.write_text(json.dumps(candidates[0]["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    if report is not None:
        print("[v26:valid]", report["summary"])
    return selected, payload

def run_v26_test():
    valid_candidates = generate_v26_candidates(valid_clean, "valid")
    selected = v26_evaluate_selector(valid_clean, valid_candidates)
    selector = selected["selector"]
    test_records = load_records(TEST_FILE)
    test_candidates = generate_v26_candidates(test_records, "test")
    outputs, summary = v26_outputs_for_selector(test_candidates, test_records, selector)
    MODEL_TEST_OUTPUT_PATH.write_text(json.dumps(test_candidates[0]["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(summary)
    summary.update({
        "selected_from_valid": selected["row"],
        "selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),
        "uses_valid_labels_for_checkpoint_selection": True,
        "uses_valid_labels_for_ranker_training": False,
    })
    _report, payload = save_outputs_and_optional_report(test_records, outputs, TEST_OUTPUT_PATH, VALID_REPORT_PATH, ENSEMBLE_RANKER_REPORT_PATH, summary)
    print("[phase2:v26] wrote", TEST_OUTPUT_PATH)
    return outputs, payload

if RUN_MODE == "phase1":
    _selected, _payload = run_v26_valid()
elif RUN_MODE == "phase2":
    if not TEST_FILE.exists():
        raise FileNotFoundError("RUN_MODE='phase2' requires test.json")
    _outputs, _payload = run_v26_test()
else:
    raise ValueError("Unknown RUN_MODE=" + str(RUN_MODE))
'''


V25_AUGMENT_INSERT = r'''

# V25 query-only distractor augmentation. This uses only query_vi structure
# and response_vi targets; it does not inspect type/original fields and does
# not compute answers from query numbers.
def _v25_query_number_count(text: str | None) -> int:
    return len(re.findall(r"[-+]?\d+(?:[.,]\d+)?", str(text or "")))

def _v25_query_distractor_score(rec: dict) -> int:
    q = normalize_text_key(rec.get("query_vi"))
    toks = query_tokens(q)
    n_nums = _v25_query_number_count(q)
    score = 0
    if n_nums >= 4:
        score += 1
    if n_nums >= 6:
        score += 1
    if len(toks) >= 55:
        score += 1
    if "%" in q or "$" in q:
        score += 1
    if {"x", "n"} & toks:
        score += 1
    # ASCII fallback markers catch both plain and some mojibake-normalized rows.
    marker_text = " " + q + " "
    for marker in [" x ", " x%", " n ", " bien ", " gia tri ", " khong lien quan ", " neu ", " nhung "]:
        if marker in marker_text:
            score += 1
            break
    return score

def apply_v25_distractor_augmentation(stage_records: list[dict]) -> list[dict]:
    if not DISTRACTOR_AUGMENTATION_ENABLED:
        return stage_records
    extras = []
    max_extra = int(DISTRACTOR_AUGMENTATION_MAX_EXTRA)
    max_copies = int(DISTRACTOR_AUGMENTATION_MAX_COPIES_PER_ROW)
    for rec in stage_records:
        score = _v25_query_distractor_score(rec)
        copies = min(max_copies, max(0, score // 2))
        for copy_idx in range(copies):
            if len(extras) >= max_extra:
                break
            aug = dict(rec)
            aug["_stage"] = str(aug.get("_stage", STAGE_A_NAME)) + "_query_distractor_aug"
            aug["_v25_aug_score"] = score
            aug["_v25_aug_copy"] = copy_idx
            extras.append(aug)
        if len(extras) >= max_extra:
            break
    out = list(stage_records) + extras
    print("[v25-augmentation] base=", len(stage_records), "extra=", len(extras), "total=", len(out))
    return out
'''


def clean_notebook(nb: dict) -> dict:
    nb.setdefault("metadata", {}).pop("widgets", None)
    nb.setdefault("metadata", {}).pop("papermill", None)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return nb


def source_of(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def patch_common_config(source: str, *, version: str, artifact_prefix: str, lr: str = "0.003") -> str:
    replacements = {
        "v17_legal_answer_only_epoch7": version,
        "v17_legal_answer_only_select_678": version,
        "v23_legal_no_solver_ranker": version,
        "v17_legal_answer_only_epoch7_lr3e-3": version,
        "v17_legal_answer_only_select_78_lr3e3": version,
        "v20_5_v17_lr3e3_strict_verifier_overlay_select_78": version,
        "v24_legal_retrieval_feature_ranker": version,
        "v25_distractor_aug_answer_only": version,
        "v26_diverse_answer_only_ranker": version,
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    source = source.replace("v17_legal_answer_only_epoch7", artifact_prefix)
    source = source.replace("v17_legal_answer_only_select_678", artifact_prefix)
    source = source.replace("v23_legal_no_solver_ranker", artifact_prefix)
    source = source.replace("STAGE_A_LR = 0.001", f"STAGE_A_LR = {lr}")
    source = source.replace(
        'RUN_MODE = "phase1"  # "phase1" writes valid_output/report; "phase2" writes test_predictions.json.',
        'RUN_MODE = os.environ.get("RUN_MODE", os.environ.get("RUN_MMODE", "phase1"))  # phase1 writes valid_output/report; phase2 writes test_predictions.json.',
    )
    return source


def patch_manifest(source: str, extra_config_lines: str = "", strategy: str | None = None) -> str:
    if strategy:
        source = source.replace("none_answer_only_fixed_epoch7", strategy)
        source = source.replace("none_answer_only_valid_select_678", strategy)
        source = source.replace("query_vi_train_response_retrieval_no_solver_valid_profile_select", strategy)
    if extra_config_lines and '"selected_retrieval_gate": globals().get("SELECTED_RETRIEVAL_GATE"' in source:
        source = source.replace(
            '"selected_retrieval_gate": globals().get("SELECTED_RETRIEVAL_GATE", None),\n',
            '"selected_retrieval_gate": globals().get("SELECTED_RETRIEVAL_GATE", None),\n' + extra_config_lines,
        )
    elif extra_config_lines and '"selected_retrieval_gate": globals().get("SELECTED_RETRIEVAL_GATE")' in source:
        source = source.replace(
            '"selected_retrieval_gate": globals().get("SELECTED_RETRIEVAL_GATE"),\n',
            '"selected_retrieval_gate": globals().get("SELECTED_RETRIEVAL_GATE"),\n' + extra_config_lines,
        )
    return source


def remove_legacy_v24_profiles(source: str) -> str:
    marker = "# V24 keeps runtime below v21"
    start = source.find(marker)
    if start < 0:
        return source
    end = source.find("]\n", start)
    if end < 0:
        return source
    return (
        source[:start]
        + "# V24 row-level decode/ranker profiles are defined in cell 8.\n"
        + source[end + 2 :]
    )


def build_v24() -> dict:
    nb = v23.build_notebook()
    for idx, cell in enumerate(nb.get("cells", [])):
        src = source_of(cell)
        if not src:
            continue
        src = src.replace("# V23 - Legal No-Solver Ranker", "# V24 - Legal Retrieval Feature Ranker")
        src = src.replace(
            "Single-seed answer-only run plus a legal no-solver decoder/retrieval ranker.",
            "Single-seed answer-only run plus a row-level legal retrieval-feature ranker.",
        )
        src = src.replace("v23_legal_no_solver_ranker", "v24_legal_retrieval_feature_ranker")
        src = src.replace("V23", "V24")
        src = src.replace("v23", "v24")
        if idx == 2:
            src = remove_legacy_v24_profiles(src)
        if idx == 8:
            marker = "def apply_no_solver_profile("
            if marker not in src:
                raise RuntimeError("Cannot locate v23 no-solver profile block for v24")
            src = src.split(marker, 1)[0] + V24_TAIL
        if idx == 9:
            src = patch_manifest(
                src,
                extra_config_lines=(
                    '        "v24_decode_profiles": globals().get("V24_DECODE_PROFILES", None),\n'
                    '        "v24_ranker_profiles": globals().get("V24_RANKER_PROFILES", None),\n'
                    '        "v24_retrieval_include_valid_for_test": globals().get("V24_RETRIEVAL_INCLUDE_VALID_FOR_TEST", None),\n'
                ),
                strategy="train_query_vi_retrieval_feature_ranker_no_solver",
            )
            src = src.replace('        "v24_profiles": globals().get("V24_PROFILES", None),\n', "")
        set_source(cell, src)
    return clean_notebook(nb)


def build_v25() -> dict:
    nb = v17.build_notebook("epoch7")
    for idx, cell in enumerate(nb.get("cells", [])):
        src = source_of(cell)
        if not src:
            continue
        src = src.replace("# V17 - Legal Answer-Only Epoch 7", "# V25 - Distractor-Augmented Answer-Only")
        src = src.replace(
            "Single-seed runtime-safe answer-only run.",
            "Single-seed answer-only run with query-only distractor oversampling.",
        )
        src = patch_common_config(src, version="v25_distractor_aug_answer_only", artifact_prefix="v25_distractor_aug_answer_only")
        src = src.replace("legal_answer_only_model_consensus", "v25_distractor_aug_answer_only")
        if idx == 2:
            src = src.replace(
                "MAX_TRAIN_SAMPLES = None\n",
                (
                    "MAX_TRAIN_SAMPLES = None\n\n"
                    "# V25 augmentation keeps runtime bounded by adding at most about 17% extra rows.\n"
                    "DISTRACTOR_AUGMENTATION_ENABLED = True\n"
                    "DISTRACTOR_AUGMENTATION_MAX_EXTRA = 16000\n"
                    "DISTRACTOR_AUGMENTATION_MAX_COPIES_PER_ROW = 1\n"
                ),
            )
        if "train_stage_a = build_answer_only_records(training_clean, STAGE_A_NAME)" in src:
            src = src.replace(
                "train_stage_a = build_answer_only_records(training_clean, STAGE_A_NAME)\n",
                V25_AUGMENT_INSERT + "\ntrain_stage_a = build_answer_only_records(training_clean, STAGE_A_NAME)\ntrain_stage_a = apply_v25_distractor_augmentation(train_stage_a)\n",
            )
        if idx == 9:
            src = patch_manifest(
                src,
                extra_config_lines=(
                    '        "distractor_augmentation_enabled": globals().get("DISTRACTOR_AUGMENTATION_ENABLED", None),\n'
                    '        "distractor_augmentation_max_extra": globals().get("DISTRACTOR_AUGMENTATION_MAX_EXTRA", None),\n'
                ),
                strategy="query_only_distractor_aug_answer_only",
            )
        set_source(cell, src)
    return clean_notebook(nb)


def build_v26() -> dict:
    nb = v17.build_notebook("select_678")
    for idx, cell in enumerate(nb.get("cells", [])):
        src = source_of(cell)
        if not src:
            continue
        src = src.replace("# V17 - Legal Answer-Only Select 6/7/8", "# V26 - Diverse Answer-Only Ranker")
        src = src.replace(
            "Selects the best checkpoint among epochs 6/7/8 on valid.json.",
            "Selects among epoch 7/8 and decode-profile answer-only candidates on valid.json.",
        )
        src = patch_common_config(src, version="v26_diverse_answer_only_ranker", artifact_prefix="v26_diverse_answer_only_ranker")
        src = src.replace("CHECKPOINT_EPOCH_LABELS_TO_SAVE = ['epoch_06', 'epoch_07', 'epoch_08']", "CHECKPOINT_EPOCH_LABELS_TO_SAVE = ['epoch_07', 'epoch_08']")
        src = src.replace("legal_answer_only_valid_select_678", "v26_diverse_answer_only_ranker")
        if idx == 8:
            marker = "# ============================================================\n# 8. Legal answer-only checkpoint selection on valid.json"
            if marker not in src:
                raise RuntimeError("Cannot locate select_678 tail for v26")
            src = src.split(marker, 1)[0] + V26_TAIL
        if idx == 9:
            src = patch_manifest(
                src,
                extra_config_lines='        "v26_decode_profiles": globals().get("V26_DECODE_PROFILES", None),\n',
                strategy="answer_only_epoch78_decode_diverse_valid_select",
            )
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
    notebooks = {
        V24_NOTEBOOK: build_v24(),
        V25_NOTEBOOK: build_v25(),
        V26_NOTEBOOK: build_v26(),
    }
    for path, nb in notebooks.items():
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        validate_notebook(path)
        print("[wrote]", path.name, "bytes=", path.stat().st_size)


if __name__ == "__main__":
    main()
