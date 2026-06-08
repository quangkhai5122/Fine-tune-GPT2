"""Create v21 safe select-7/8 + strict verifier sweep notebook."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V20_5_SCRIPT = ROOT / "audit" / "create_v17_5_v20_5_select78_notebooks.py"
NOTEBOOK_PATH = ROOT / "finetune_gpt2_for_math_v21_safe_select78_strict_verifier_sweep.ipynb"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V21_SWEEP_TAIL = r'''
# ============================================================
# 8. Safe checkpoint/profile selection on valid.json
# ============================================================
def _bucket(summary: dict, score: int) -> int:
    buckets = summary.get("buckets", {})
    return int(buckets.get(score, buckets.get(str(score), 0)))

def _safe_candidate_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))

def _profile_name(profile: dict) -> str:
    return _safe_candidate_name(profile.get("name", "profile"))

def _verifier_output_path(split_name: str, candidate_name: str, profile: dict) -> Path:
    ENSEMBLE_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    return ENSEMBLE_CANDIDATE_DIR / ("%s_%s_%s_strict_verifier.json" % (split_name, _safe_candidate_name(candidate_name), _profile_name(profile)))

def _verifier_decision_path(split_name: str, candidate_name: str, profile: dict) -> Path:
    return WORKING_DIR / ("%s_candidate_verifier_decisions_%s_%s.json" % (split_name, _safe_candidate_name(candidate_name), _profile_name(profile)))

def output_sanity_for_candidate(cand: dict) -> dict:
    outputs = cand.get("outputs") or []
    texts = [str(o.get("model_output", "")) for o in outputs]
    n = len(texts)
    nonempty = sum(bool(t.strip()) for t in texts)
    digit = sum(any(ch.isdigit() for ch in t) for t in texts)
    extractable = sum(extract_pred(o)[0] is not None for o in outputs)
    samples = [t.replace("\n", " ")[:100] for t in texts[:5]]
    sanity = {
        "n": n,
        "nonempty": nonempty,
        "digit": digit,
        "extractable": extractable,
        "samples": samples,
        "min_digit_frac": MIN_CANDIDATE_DIGIT_FRAC,
    }
    sanity["usable"] = bool(n > 0 and digit >= max(1, int(MIN_CANDIDATE_DIGIT_FRAC * n)))
    print("[output-sanity]", cand.get("name"), sanity)
    return sanity

def ensure_candidate_usable(cand: dict) -> dict:
    sanity = output_sanity_for_candidate(cand)
    cand["output_sanity"] = sanity
    if not sanity["usable"]:
        print("[output-sanity] skip unusable candidate", cand.get("name"))
    return sanity

def _retrieval_metas(cand: dict) -> list[dict]:
    metas = []
    meta = cand.get("meta")
    if isinstance(meta, dict) and "top_sim" in meta:
        metas.append(meta)
    for meta in (cand.get("meta_by_source") or {}).values():
        if isinstance(meta, dict) and "top_sim" in meta:
            metas.append(meta)
    return metas

def _profile_value(profile: dict, key: str, default):
    return profile.get(key, default)

def retrieval_is_safe_for_profile(cand: dict, profile: dict) -> bool:
    for meta in _retrieval_metas(cand):
        if (
            float(meta.get("top_sim", 0.0)) >= float(_profile_value(profile, "retrieval_direct_min_sim", RETRIEVAL_DIRECT_MIN_SIM))
            and float(meta.get("top_jaccard", 0.0)) >= float(_profile_value(profile, "retrieval_direct_min_jaccard", RETRIEVAL_DIRECT_MIN_JACCARD))
            and float(meta.get("majority_frac", 0.0)) >= float(_profile_value(profile, "retrieval_direct_min_majority_frac", RETRIEVAL_DIRECT_MIN_MAJORITY_FRAC))
            and float(meta.get("margin", 0.0)) >= float(_profile_value(profile, "retrieval_direct_min_margin", RETRIEVAL_DIRECT_MIN_MARGIN))
            and (int(meta.get("same_numbers", 0)) > 0 or int(meta.get("same_skeleton", 0)) > 0)
        ):
            return True
    return False

def effective_confidence_for_profile(cand: dict, profile: dict, retrieval_safe: bool) -> float:
    confidence = float(cand.get("confidence", 0.0))
    if retrieval_safe:
        confidence = max(confidence, float(_profile_value(profile, "retrieval_direct_confidence", 0.94)))
    return confidence

def select_candidate_for_row_profile(cands: list[dict], model_key: str | None, profile: dict) -> tuple[dict, dict]:
    # Shallow-copy rows because rank/debug fields are profile-specific.
    cands = [dict(c) for c in cands]
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
        retrieval_safe = retrieval_is_safe_for_profile(cand, profile)
        safe_source = bool(sources & {"template_exact_query", "template_exact_numbers", "template_linear"}) or retrieval_safe
        eff_conf = effective_confidence_for_profile(cand, profile, retrieval_safe)
        safe_override = (
            safe_source
            and eff_conf >= float(_profile_value(profile, "verifier_safe_confidence", VERIFIER_SAFE_CONFIDENCE))
            and lp_delta >= -float(_profile_value(profile, "verifier_logprob_tolerance", VERIFIER_LOGPROB_TOLERANCE))
        )
        logprob_override = (
            eff_conf >= float(_profile_value(profile, "verifier_min_confidence", VERIFIER_MIN_CONFIDENCE))
            and lp_delta >= float(_profile_value(profile, "verifier_logprob_override_margin", VERIFIER_LOGPROB_OVERRIDE_MARGIN))
        )
        missing_model = bool(_profile_value(profile, "allow_model_missing", True)) and model_key is None and cand.get("key") is not None
        rank_score = (
            2.0 * eff_conf
            + 0.30 * cand.get("prior_norm", 0.0)
            + 0.75 * lp_delta
            + (0.35 if safe_source else 0.0)
        )
        cand["effective_confidence"] = eff_conf
        cand["profile_retrieval_safe"] = retrieval_safe
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
        "chosen_confidence": chosen.get("effective_confidence", chosen.get("confidence")),
        "chosen_logprob_avg": chosen.get("logprob_avg"),
        "chosen_logprob_delta_vs_model": chosen.get("logprob_delta_vs_model"),
        "num_candidates": len(cands),
        "profile": profile.get("name"),
    }

def prepare_verifier_candidate_sets(records: list[dict], model_outputs: list[dict], adapter_dir: Path, retriever: LegalTemplateRetriever) -> list[list[dict]]:
    candidate_sets = []
    for idx, rec in enumerate(records):
        candidate_sets.append(build_candidate_set(rec, idx, model_outputs[idx], retriever))
    score_candidate_logprobs(adapter_dir, records, candidate_sets)
    return candidate_sets

def choose_candidate_verifier_profile(
    records: list[dict],
    model_outputs: list[dict],
    adapter_dir: Path,
    retriever: LegalTemplateRetriever,
    profile: dict,
    candidate_sets: list[list[dict]] | None = None,
):
    if candidate_sets is None:
        candidate_sets = prepare_verifier_candidate_sets(records, model_outputs, adapter_dir, retriever)
    outputs = []
    decisions = []
    source_counts = Counter()
    reason_counts = Counter()
    changed = 0
    for idx, rec in enumerate(records):
        model_key = answer_key_from_output(model_outputs[idx])
        chosen, decision = select_candidate_for_row_profile(candidate_sets[idx], model_key, profile)
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
        "strategy": "v21_safe_select78_strict_verifier_sweep_profile",
        "profile": profile,
        "num_rows": len(records),
        "changed_from_model": changed,
        "source_counts": dict(source_counts.most_common()),
        "reason_counts": dict(reason_counts.most_common()),
        "retrieval_backend": retriever.backend,
        "verifier_logprob_enabled": VERIFIER_LOGPROB_ENABLED,
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "uses_valid_labels_for_profile_selection": False,
        "notes": "Base SFT saves epoch 7 and 8. Candidate retrieval uses train query_vi only; answers come from train response_vi or query-derived numeric/template heuristics.",
    }
    return outputs, decisions, summary

def verifier_profile_selection_key(row: dict):
    summary = row["summary"]
    epoch = float(row.get("epoch") or 0.0)
    profile_priority = float((row.get("profile") or {}).get("priority", 0.0))
    return (
        int(row.get("usable", True)),
        int(summary.get("raw_score", 0)),
        _bucket(summary, 10),
        int(summary.get("extractable", 0)),
        -abs(epoch - 7.0),
        epoch,
        profile_priority,
    )

def run_verifier_sweep_for_candidate(records: list[dict], split_name: str, cand: dict, retriever: LegalTemplateRetriever) -> dict:
    sanity = ensure_candidate_usable(cand)
    if not sanity["usable"]:
        row = {
            "name": cand["name"],
            "kind": cand["kind"],
            "seed": cand.get("seed"),
            "epoch": cand.get("epoch"),
            "adapter_dir": str(cand["adapter_dir"]),
            "model_output_path": cand.get("path"),
            "output_sanity": sanity,
            "usable": False,
            "profile": None,
            "summary": {"n": len(records), "raw_score": -1, "extractable": 0, "buckets": {"10": 0}},
            "sweep": [],
        }
        return {"candidate": cand, "row": row, "outputs": cand.get("outputs") or [], "decisions": [], "summary": row["summary"]}

    candidate_sets = prepare_verifier_candidate_sets(records, cand["outputs"], Path(cand["adapter_dir"]), retriever)
    sweep_rows = []
    best = None
    for profile in VERIFIER_SWEEP_PROFILES:
        outputs, decisions, verifier_summary = choose_candidate_verifier_profile(
            records, cand["outputs"], Path(cand["adapter_dir"]), retriever, profile, candidate_sets
        )
        if records and records[0].get("response_vi"):
            report = evaluate_predictions(outputs, records)
            summary = report["summary"]
            by_type = report["by_type"]
        else:
            report = None
            summary = verifier_summary
            by_type = None
        row = {
            "name": cand["name"],
            "kind": cand["kind"],
            "seed": cand.get("seed"),
            "epoch": cand.get("epoch"),
            "adapter_dir": str(cand["adapter_dir"]),
            "model_output_path": cand.get("path"),
            "profile": profile,
            "output_sanity": sanity,
            "usable": True,
            "summary": summary,
            "by_type": by_type,
            "verifier_summary": verifier_summary,
        }
        sweep_rows.append(row)
        key = verifier_profile_selection_key(row)
        if best is None or key > best[0]:
            best = (key, row, outputs, decisions, verifier_summary, report)
        print(
            "[sweep]",
            cand["name"],
            profile.get("name"),
            "raw=", summary.get("raw_score"),
            "exact10=", _bucket(summary, 10),
            "changed=", verifier_summary.get("changed_from_model"),
        )
    if best is None:
        raise RuntimeError("No verifier sweep result for " + cand["name"])
    _key, best_row, best_outputs, best_decisions, best_summary, best_report = best
    out_path = _verifier_output_path(split_name, cand["name"], best_row["profile"])
    decision_path = _verifier_decision_path(split_name, cand["name"], best_row["profile"])
    out_path.write_text(json.dumps(best_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    decision_path.write_text(json.dumps(best_decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    best_row = dict(best_row)
    best_row.update({
        "verifier_output_path": str(out_path),
        "decision_path": str(decision_path),
        "sweep": sweep_rows,
    })
    eval_path = CHECKPOINT_EVAL_DIR / (_safe_candidate_name(cand["name"]) + "_v21_sweep_valid_report.json")
    eval_path.write_text(json.dumps(best_row, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "candidate": cand,
        "row": best_row,
        "outputs": best_outputs,
        "decisions": best_decisions,
        "summary": best_summary,
        "report": best_report,
        "candidate_sets": candidate_sets,
    }

def select_best_valid_verifier_sweep(model_candidates: list[dict], records: list[dict], retriever: LegalTemplateRetriever) -> tuple[dict, dict]:
    CHECKPOINT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    best = None
    for cand in model_candidates:
        result = run_verifier_sweep_for_candidate(records, "valid", cand, retriever)
        row = result["row"]
        rows.append(row)
        key = verifier_profile_selection_key(row)
        if row.get("usable") and (best is None or key > best[0]):
            best = (key, result)
        print(
            "[select-v21]",
            cand["name"],
            "usable=", row.get("usable"),
            "best_profile=", (row.get("profile") or {}).get("name"),
            "raw=", row["summary"].get("raw_score"),
            "exact10=", _bucket(row["summary"], 10),
        )
    if best is None:
        raise RuntimeError("No usable checkpoint candidates after output sanity checks")
    selected = best[1]
    payload = {
        "strategy": "v21_safe_select78_strict_verifier_sweep",
        "selection_metric": "max(usable, raw_score, exact10, extractable, -abs(epoch-7), epoch, profile_priority)",
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "uses_valid_labels_for_checkpoint_selection": True,
        "uses_valid_labels_for_verifier_profile_selection": True,
        "uses_valid_labels_for_verifier_training": False,
        "verifier_sweep_profiles": VERIFIER_SWEEP_PROFILES,
        "num_model_candidates": len(model_candidates),
        "candidates": rows,
        "selected": selected["row"],
    }
    CHECKPOINT_SELECTION_REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_CHECKPOINT_INFO_PATH.write_text(json.dumps(selected["row"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[select-v21] selected",
        selected["candidate"]["name"],
        "profile=", selected["row"]["profile"].get("name"),
        "raw=", selected["row"]["summary"].get("raw_score"),
    )
    return selected, payload

def primary_candidate_for_reference(model_candidates: list[dict]) -> dict:
    usable = [cand for cand in model_candidates if output_sanity_for_candidate(cand).get("usable")]
    pool = usable or model_candidates
    return max(pool, key=lambda c: (float(c.get("epoch") or 0.0), float(c.get("priority", 0.0))))

def run_v21_valid(records: list[dict], split_name: str, output_path: Path, report_path: Path):
    entries = candidate_entries_from_runs()
    model_candidates = generate_model_candidates(records, split_name, entries)
    primary = primary_candidate_for_reference(model_candidates)
    MODEL_VALID_OUTPUT_PATH.write_text(json.dumps(primary["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    if records and records[0].get("response_vi"):
        save_eval_report(MODEL_VALID_OUTPUT_PATH, records, MODEL_VALID_REPORT_PATH)
    retriever = LegalTemplateRetriever(train_clean)
    selected, selection_payload = select_best_valid_verifier_sweep(model_candidates, records, retriever)
    MODEL_VALID_OUTPUT_PATH.write_text(json.dumps(selected["candidate"]["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    if records and records[0].get("response_vi"):
        save_eval_report(MODEL_VALID_OUTPUT_PATH, records, MODEL_VALID_REPORT_PATH)
    summary = {
        "strategy": "v21_safe_select78_strict_verifier_sweep",
        "selected_candidate": selected["candidate"]["name"],
        "selected_epoch": selected["candidate"].get("epoch"),
        "selected_adapter_dir": str(selected["candidate"]["adapter_dir"]),
        "selected_profile": selected["row"]["profile"],
        "selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),
        "candidate_summaries": [
            {
                "name": row["name"],
                "epoch": row["epoch"],
                "usable": row.get("usable"),
                "best_profile": (row.get("profile") or {}).get("name"),
                "raw_score": row["summary"].get("raw_score"),
                "exact10": _bucket(row["summary"], 10),
                "extractable": row["summary"].get("extractable"),
            }
            for row in selection_payload["candidates"]
        ],
        "selected_verifier_summary": selected["summary"],
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "uses_valid_labels_for_checkpoint_selection": True,
        "uses_valid_labels_for_verifier_profile_selection": True,
        "uses_valid_labels_for_verifier_training": False,
    }
    report, payload = save_outputs_and_optional_report(records, selected["outputs"], output_path, report_path, ENSEMBLE_RANKER_REPORT_PATH, summary)
    SELECTED_VALID_OUTPUT_PATH.write_text(json.dumps(selected["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_VALID_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (WORKING_DIR / ("%s_candidate_verifier_decisions.json" % split_name)).write_text(json.dumps(selected["decisions"], ensure_ascii=False, indent=2), encoding="utf-8")
    if report is not None:
        print("[v21:%s]" % split_name, report["summary"])
    return model_candidates, selected, payload

def select_entry_and_profile_for_test(entries: list[dict], retriever: LegalTemplateRetriever) -> tuple[dict, dict]:
    if VALID_FILE.exists() and valid_clean and valid_clean[0].get("response_vi"):
        valid_candidates = generate_model_candidates(valid_clean, "valid", entries)
        selected, _payload = select_best_valid_verifier_sweep(valid_candidates, valid_clean, retriever)
        SELECTED_VALID_OUTPUT_PATH.write_text(json.dumps(selected["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
        SELECTED_VALID_REPORT_PATH.write_text(json.dumps(selected["report"], ensure_ascii=False, indent=2), encoding="utf-8")
        (WORKING_DIR / "valid_candidate_verifier_decisions.json").write_text(json.dumps(selected["decisions"], ensure_ascii=False, indent=2), encoding="utf-8")
        return selected["candidate"], selected["row"]["profile"]

    # Fallback is only for environments without valid labels. Prefer epoch 7 for
    # stability, but still skip obviously collapsed outputs if valid generation exists.
    epoch7 = next((entry for entry in entries if entry["name"].endswith("::epoch_07")), None)
    fallback = epoch7 or max(entries, key=lambda e: float(e.get("epoch") or 0.0))
    profile = next(p for p in VERIFIER_SWEEP_PROFILES if p.get("name") == DEFAULT_VERIFIER_PROFILE_NAME)
    info = {
        "strategy": "v21_safe_select78_strict_verifier_sweep",
        "selection_metric": "fallback_epoch_07_when_valid_unavailable",
        "selected": {
            "name": fallback["name"],
            "kind": fallback["kind"],
            "seed": fallback.get("seed"),
            "epoch": fallback.get("epoch"),
            "adapter_dir": str(fallback["adapter_dir"]),
            "profile": profile,
        },
    }
    CHECKPOINT_SELECTION_REPORT_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_CHECKPOINT_INFO_PATH.write_text(json.dumps(info["selected"], ensure_ascii=False, indent=2), encoding="utf-8")
    return fallback, profile

def run_v21_test():
    entries = candidate_entries_from_runs()
    retriever = LegalTemplateRetriever(train_clean)
    selected_entry, selected_profile = select_entry_and_profile_for_test(entries, retriever)
    test_records = load_records(TEST_FILE)
    model_outputs = generate_model_outputs(Path(selected_entry["adapter_dir"]), test_records, MODEL_TEST_OUTPUT_PATH, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS)
    test_candidate = {**selected_entry, "outputs": model_outputs, "path": str(MODEL_TEST_OUTPUT_PATH)}
    sanity = ensure_candidate_usable(test_candidate)
    if not sanity["usable"]:
        raise RuntimeError("Selected checkpoint generated unusable test outputs; rerun with another checkpoint/seed.")
    candidate_sets = prepare_verifier_candidate_sets(test_records, model_outputs, Path(selected_entry["adapter_dir"]), retriever)
    outputs, decisions, summary = choose_candidate_verifier_profile(
        test_records, model_outputs, Path(selected_entry["adapter_dir"]), retriever, selected_profile, candidate_sets
    )
    summary = dict(summary)
    summary.update({
        "strategy": "v21_safe_select78_strict_verifier_sweep",
        "selected_candidate": selected_entry["name"],
        "selected_epoch": selected_entry.get("epoch"),
        "selected_adapter_dir": str(selected_entry["adapter_dir"]),
        "selected_profile": selected_profile,
        "model_test_output": str(MODEL_TEST_OUTPUT_PATH),
        "selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "uses_valid_labels_for_checkpoint_selection": bool(VALID_FILE.exists() and valid_clean and valid_clean[0].get("response_vi")),
        "uses_valid_labels_for_verifier_profile_selection": bool(VALID_FILE.exists() and valid_clean and valid_clean[0].get("response_vi")),
        "uses_valid_labels_for_verifier_training": False,
    })
    _report, payload = save_outputs_and_optional_report(test_records, outputs, TEST_OUTPUT_PATH, VALID_REPORT_PATH, ENSEMBLE_RANKER_REPORT_PATH, summary)
    (WORKING_DIR / "test_candidate_verifier_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[phase2] selected", selected_entry["name"], "profile=", selected_profile.get("name"), "wrote", TEST_OUTPUT_PATH)
    return test_candidate, outputs, payload

if RUN_MODE == "phase1":
    _candidates, _selected, _payload = run_v21_valid(valid_clean, "valid", VALID_OUTPUT_PATH, VALID_REPORT_PATH)
elif RUN_MODE == "phase2":
    if not TEST_FILE.exists():
        raise FileNotFoundError("RUN_MODE='phase2' requires test.json")
    _selected_candidate, _outputs, _payload = run_v21_test()
else:
    raise ValueError("Unknown RUN_MODE=" + str(RUN_MODE))
'''


def patch_config_and_text(source: str) -> str:
    replacements = {
        "v20_5_v17_lr3e3_strict_verifier_overlay_select_78": "v21_safe_select78_strict_verifier_sweep",
        "# V20.5 - V17 LR 3e-3 Select 7/8 + Strict Verifier Overlay": (
            "# V21 - Safe Select 7/8 + Strict Verifier Sweep"
        ),
        "Single-seed answer-only SFT with the v17 lr=3e-3 setup, saving epoch 7 and 8, then selecting the best strict-legal verifier overlay on valid.json.": (
            "Single-seed answer-only SFT with lr=3e-3, saving epoch 7 and 8, "
            "skipping collapsed outputs, and sweeping strict verifier profiles on valid.json."
        ),
        "v17_lr3e3_select78_plus_train_query_vi_template_retrieval_candidates": (
            "v21_safe_select78_plus_train_query_vi_template_retrieval_candidates"
        ),
        '"strategy": "v20_5_lr3e3_strict_verifier_overlay_valid_select_78"': (
            '"strategy": "v21_safe_select78_strict_verifier_sweep"'
        ),
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    source = source.replace(
        "MAX_NEW_TOKENS = 32\n",
        "MAX_NEW_TOKENS = 32\nMIN_NEW_TOKENS = 1\n",
    )
    source = source.replace(
        "VERIFIER_LOGPROB_OVERRIDE_MARGIN = 0.55\n",
        '''VERIFIER_LOGPROB_OVERRIDE_MARGIN = 0.55

# Output sanity and valid-time verifier profile sweep.
MIN_CANDIDATE_DIGIT_FRAC = 0.50
DEFAULT_VERIFIER_PROFILE_NAME = "strict"
VERIFIER_SWEEP_PROFILES = [
    {
        "name": "strict",
        "priority": 0,
        "retrieval_direct_min_sim": 0.92,
        "retrieval_direct_min_jaccard": 0.70,
        "retrieval_direct_min_majority_frac": 0.55,
        "retrieval_direct_min_margin": 0.20,
        "retrieval_direct_confidence": 0.94,
        "verifier_safe_confidence": 0.90,
        "verifier_min_confidence": 0.82,
        "verifier_logprob_tolerance": 0.35,
        "verifier_logprob_override_margin": 0.55,
        "allow_model_missing": True,
    },
    {
        "name": "balanced",
        "priority": 1,
        "retrieval_direct_min_sim": 0.88,
        "retrieval_direct_min_jaccard": 0.64,
        "retrieval_direct_min_majority_frac": 0.50,
        "retrieval_direct_min_margin": 0.12,
        "retrieval_direct_confidence": 0.94,
        "verifier_safe_confidence": 0.88,
        "verifier_min_confidence": 0.80,
        "verifier_logprob_tolerance": 0.45,
        "verifier_logprob_override_margin": 0.48,
        "allow_model_missing": True,
    },
    {
        "name": "safe_missing_only",
        "priority": -1,
        "retrieval_direct_min_sim": 0.95,
        "retrieval_direct_min_jaccard": 0.75,
        "retrieval_direct_min_majority_frac": 0.60,
        "retrieval_direct_min_margin": 0.25,
        "retrieval_direct_confidence": 0.95,
        "verifier_safe_confidence": 0.96,
        "verifier_min_confidence": 0.90,
        "verifier_logprob_tolerance": 0.20,
        "verifier_logprob_override_margin": 0.80,
        "allow_model_missing": True,
    },
]
''',
    )
    source = source.replace(
        '''            max_new_tokens=eff_new,\n            pad_token_id=SAFE_EOS_ID,\n''',
        '''            max_new_tokens=eff_new,\n            min_new_tokens=min(int(globals().get("MIN_NEW_TOKENS", 0)), eff_new),\n            pad_token_id=SAFE_EOS_ID,\n''',
    )
    source = source.replace(
        '"uses_valid_labels_for_verifier_training": False,\n',
        '"uses_valid_labels_for_verifier_training": False,\n'
        '        "uses_valid_labels_for_verifier_profile_selection": True,\n',
    )
    source = source.replace(
        '"verifier_safe_confidence": VERIFIER_SAFE_CONFIDENCE,\n',
        '"verifier_safe_confidence": VERIFIER_SAFE_CONFIDENCE,\n'
        '        "min_candidate_digit_frac": MIN_CANDIDATE_DIGIT_FRAC,\n'
        '        "default_verifier_profile_name": DEFAULT_VERIFIER_PROFILE_NAME,\n'
        '        "verifier_sweep_profiles": VERIFIER_SWEEP_PROFILES,\n',
    )
    return source


def patch_run_cell(source: str) -> str:
    marker = "def _bucket(summary: dict, score: int) -> int:"
    if marker not in source:
        raise RuntimeError("Cannot find v20.5 select tail marker")
    prefix = source.split(marker, 1)[0]
    return prefix + V21_SWEEP_TAIL


def clean_notebook(nb: dict) -> dict:
    nb.setdefault("metadata", {}).pop("widgets", None)
    nb.setdefault("metadata", {}).pop("papermill", None)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return nb


def build_notebook() -> dict:
    v20_5 = load_module(V20_5_SCRIPT, "create_v17_5_v20_5_select78_notebooks")
    nb = v20_5.build_v20_5_notebook()
    for idx, cell in enumerate(nb.get("cells", [])):
        if "source" not in cell:
            continue
        source = "".join(cell.get("source", []))
        source = patch_config_and_text(source)
        if idx == 8:
            source = patch_run_cell(source)
        cell["source"] = source.splitlines(keepends=True)
    return clean_notebook(nb)


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
