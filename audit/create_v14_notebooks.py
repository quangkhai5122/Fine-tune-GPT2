from __future__ import annotations

import copy
import json
from pathlib import Path


BASE_NOTEBOOK = Path("finetune_gpt2_for_math_v13_type_gated_full_retrain.ipynb")


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Missing expected block:\n{old[:500]}")
    return text.replace(old, new, 1)


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


ENHANCED_RETRIEVAL_BLOCK = r'''def build_retrieval_index(records: list[dict]) -> dict:
    index = defaultdict(list)
    for rec in records:
        if rec.get("_canonical_answer") is None or rec.get("_gold_num") is None:
            continue
        index[rec["_source_key"]].append({
            "type": rec.get("type") or "unknown",
            "answer_num": float(rec["_gold_num"]),
            "canonical_answer": rec["_canonical_answer"],
            "query_tokens": rec.get("_query_tokens") or query_tokens(rec.get("query_vi")),
            "query_vi": rec.get("query_vi", ""),
        })
    return dict(index)

RETRIEVAL_INDEX_OVERLAP_TRAIN = build_retrieval_index(overlap_train_clean)
RETRIEVAL_INDEX_FULL_TRAIN = build_retrieval_index(train_clean)
print("[retrieval] overlap-train source groups:", len(RETRIEVAL_INDEX_OVERLAP_TRAIN))
print("[retrieval] full-train source groups:", len(RETRIEVAL_INDEX_FULL_TRAIN))

def _num_equal(a, b, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

def _model_num(model_item: dict | None):
    if not model_item:
        return None
    try:
        _, pred_num = extract_pred(model_item)
        return pred_num
    except Exception:
        return None

def retrieval_config_for_type(rec_type: str, gate_config_by_type: dict | None = None) -> dict:
    cfg = {
        "enabled": True,
        "strategy": RETRIEVAL_STRATEGY,
        "min_majority_frac": RETRIEVAL_MIN_MAJORITY_FRAC,
        "min_margin": RETRIEVAL_MIN_MAJORITY_MARGIN,
        "min_nearest_jaccard": RETRIEVAL_MIN_NEAREST_JACCARD,
        "require_typed_pool": False,
        "use_model_agreement_gate": RETRIEVAL_USE_MODEL_AGREEMENT_GATE,
        "model_agreement_bypass_confidence": RETRIEVAL_MODEL_AGREEMENT_BYPASS_CONFIDENCE,
    }
    for source in (RETRIEVAL_TYPE_GATE_CONFIG, ACTIVE_RETRIEVAL_TYPE_GATE_CONFIG, gate_config_by_type):
        if source and rec_type in source:
            cfg.update(source[rec_type])
    return cfg

def _rank_answer_groups(pool: list[dict], rec_tokens: set[str]) -> list[dict]:
    by_num = defaultdict(list)
    for c in pool:
        by_num[c["answer_num"]].append(c)
    ranked = []
    for num, vals in by_num.items():
        nearest_j = max(jaccard(rec_tokens, v["query_tokens"]) for v in vals)
        ranked.append({
            "answer_num": num,
            "canonical_answer": vals[0]["canonical_answer"],
            "count": len(vals),
            "nearest_jaccard_same_answer": nearest_j,
        })
    ranked.sort(key=lambda x: (x["count"], x["nearest_jaccard_same_answer"], -abs(x["answer_num"])), reverse=True)
    return ranked

def _selected_from_majority(pool: list[dict], rec_tokens: set[str]) -> tuple[dict, list[dict]]:
    ranked = _rank_answer_groups(pool, rec_tokens)
    selected = dict(ranked[0])
    selected["selection_strategy"] = "source_type_majority"
    return selected, ranked

def _selected_from_nearest(pool: list[dict], rec_tokens: set[str]) -> tuple[dict, list[dict]]:
    best = max(pool, key=lambda c: jaccard(rec_tokens, c["query_tokens"]))
    ranked = _rank_answer_groups(pool, rec_tokens)
    selected = None
    for group in ranked:
        if _num_equal(group["answer_num"], best["answer_num"]):
            selected = dict(group)
            break
    if selected is None:
        selected = {
            "answer_num": best["answer_num"],
            "canonical_answer": best["canonical_answer"],
            "count": 1,
            "nearest_jaccard_same_answer": jaccard(rec_tokens, best["query_tokens"]),
        }
    selected["canonical_answer"] = best["canonical_answer"]
    selected["nearest_query_jaccard"] = jaccard(rec_tokens, best["query_tokens"])
    selected["selection_strategy"] = "source_type_nearest_query"
    return selected, ranked

def retrieve_answer_for_record(rec: dict, retrieval_index: dict, model_item: dict | None = None, gate_config_by_type: dict | None = None) -> dict:
    rec_type = rec.get("type") or "unknown"
    cfg = retrieval_config_for_type(rec_type, gate_config_by_type)
    candidates = retrieval_index.get(source_group_key(rec), [])
    if not cfg.get("enabled", True):
        return {"used": False, "reason": "type_gate_disabled", "type": rec_type, "num_candidates": len(candidates), "config": cfg}
    if RETRIEVAL_ALLOWED_TYPES is not None and rec_type not in set(RETRIEVAL_ALLOWED_TYPES):
        return {
            "used": False,
            "reason": "type_not_allowed",
            "type": rec_type,
            "allowed_types": sorted(RETRIEVAL_ALLOWED_TYPES),
            "num_candidates": len(candidates),
            "config": cfg,
        }
    if len(candidates) < RETRIEVAL_MIN_SOURCE_CANDIDATES:
        return {"used": False, "reason": "source_unseen", "num_candidates": len(candidates), "config": cfg}
    typed = [c for c in candidates if c["type"] == rec_type]
    if cfg.get("require_typed_pool") and not typed:
        return {"used": False, "reason": "typed_pool_missing", "num_candidates": len(candidates), "config": cfg}
    pool = typed or candidates
    if not pool:
        return {"used": False, "reason": "empty_pool", "num_candidates": len(candidates), "config": cfg}

    rec_tokens = query_tokens(rec.get("query_vi"))
    strategy = cfg.get("strategy", RETRIEVAL_STRATEGY)
    if strategy == "source_type_nearest_query":
        selected, ranked = _selected_from_nearest(pool, rec_tokens)
    else:
        selected, ranked = _selected_from_majority(pool, rec_tokens)

    top_count = int(selected["count"])
    other_counts = [int(g["count"]) for g in ranked if not _num_equal(g["answer_num"], selected["answer_num"])]
    second_count = max(other_counts) if other_counts else 0
    majority_frac = top_count / len(pool)
    margin = (top_count - second_count) / len(pool)
    nearest_same = float(selected.get("nearest_jaccard_same_answer") or 0.0)
    model_num = _model_num(model_item)
    model_agrees = _num_equal(model_num, selected["answer_num"])
    confidence_passed = (
        majority_frac >= float(cfg.get("min_majority_frac", 0.0))
        and margin >= float(cfg.get("min_margin", 0.0))
        and nearest_same >= float(cfg.get("min_nearest_jaccard", 0.0))
    )
    agreement_bypass = (
        bool(cfg.get("use_model_agreement_gate", False))
        and bool(cfg.get("model_agreement_bypass_confidence", False))
        and model_agrees
    )
    if not (confidence_passed or agreement_bypass):
        return {
            "used": False,
            "reason": "low_confidence_model_disagree" if model_num is not None and not model_agrees else "low_confidence",
            "strategy": strategy,
            "type": rec_type,
            "num_candidates": len(candidates),
            "pool_candidates": len(pool),
            "typed_pool": bool(typed),
            "majority_count": top_count,
            "second_count": second_count,
            "majority_frac": majority_frac,
            "top1_top2_margin": margin,
            "nearest_jaccard_same_answer": nearest_same,
            "model_num": model_num,
            "model_agrees": model_agrees,
            "confidence_passed": confidence_passed,
            "config": cfg,
        }

    return {
        "used": True,
        "strategy": strategy,
        "canonical_answer": selected["canonical_answer"],
        "answer_num": selected["answer_num"],
        "num_candidates": len(candidates),
        "pool_candidates": len(pool),
        "typed_pool": bool(typed),
        "majority_count": top_count,
        "second_count": second_count,
        "majority_frac": majority_frac,
        "top1_top2_margin": margin,
        "nearest_jaccard_same_answer": nearest_same,
        "nearest_query_jaccard": selected.get("nearest_query_jaccard"),
        "model_num": model_num,
        "model_agrees": model_agrees,
        "confidence_passed": confidence_passed,
        "agreement_bypass": agreement_bypass,
        "config": cfg,
    }

def build_hybrid_outputs(records: list[dict], model_outputs: list[dict], retrieval_index: dict, gate_config_by_type: dict | None = None, *, debug_sample: int = RETRIEVAL_DEBUG_SAMPLE):
    outputs = []
    decisions = []
    used = 0
    changed = 0
    by_type = defaultdict(lambda: {"n": 0, "retrieval_used": 0, "model_fallback": 0})
    reason_counts = Counter()
    agreement_counts = Counter()
    for idx, (rec, model_item) in enumerate(zip(records, model_outputs)):
        decision = retrieve_answer_for_record(rec, retrieval_index, model_item=model_item, gate_config_by_type=gate_config_by_type)
        rec_type = rec.get("type") or "unknown"
        by_type[rec_type]["n"] += 1
        before = model_item.get("model_output", "")
        item = {
            "id": model_item.get("id", rec.get("id", idx)),
            "query_vi": rec.get("query_vi", ""),
            "type": rec.get("type"),
            "model_output": before,
        }
        if decision.get("used"):
            item["model_output"] = build_answer_only_target(decision["canonical_answer"])
            used += 1
            changed += int(item["model_output"] != before)
            by_type[rec_type]["retrieval_used"] += 1
            agreement_counts["model_agrees" if decision.get("model_agrees") else "model_disagrees_or_unparseable"] += 1
        else:
            by_type[rec_type]["model_fallback"] += 1
            reason_counts[decision.get("reason", "not_used")] += 1
        if idx < debug_sample:
            decisions.append({
                "idx": idx,
                "type": rec.get("type"),
                "query_vi": rec.get("query_vi", "")[:300],
                "model_output_before": before,
                "model_output_after": item["model_output"],
                "decision": decision,
            })
        outputs.append(item)
    summary = {
        "enabled": True,
        "strategy": RETRIEVAL_STRATEGY,
        "allowed_types": None if RETRIEVAL_ALLOWED_TYPES is None else sorted(RETRIEVAL_ALLOWED_TYPES),
        "active_type_gate_config": gate_config_by_type or ACTIVE_RETRIEVAL_TYPE_GATE_CONFIG,
        "n": len(outputs),
        "retrieval_used": used,
        "retrieval_used_pct": used / len(outputs) if outputs else 0.0,
        "output_changed_count": changed,
        "retrieval_used_by_type": dict(sorted((k, dict(v)) for k, v in by_type.items())),
        "fallback_reason_counts": dict(reason_counts.most_common()),
        "agreement_counts": dict(agreement_counts.most_common()),
        "debug_first": decisions,
    }
    return outputs, summary

def apply_hybrid_retrieval(records: list[dict], model_outputs: list[dict], output_path: Path, retrieval_index: dict, decision_report_path: Path | None = None, gate_config_by_type: dict | None = None):
    if not USE_HYBRID_RETRIEVAL:
        output_path.write_text(json.dumps(model_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
        return model_outputs, {"enabled": False, "n": len(model_outputs), "retrieval_used": 0}
    outputs, summary = build_hybrid_outputs(records, model_outputs, retrieval_index, gate_config_by_type)
    output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    if decision_report_path is not None:
        decision_report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hybrid] wrote {len(outputs)} rows -> {output_path}; retrieval_used={summary['retrieval_used']}; changed={summary['output_changed_count']}")
    return outputs, summary

def _grid_values(grid: dict, name: str, fallback):
    vals = grid.get(name, fallback)
    return list(vals) if isinstance(vals, (list, tuple)) else [vals]

def iter_retrieval_gate_sweep_configs(rec_type: str):
    grid = dict(RETRIEVAL_GATE_SWEEP_GRID.get("_default", {}))
    grid.update(RETRIEVAL_GATE_SWEEP_GRID.get(rec_type, {}))
    strategies = _grid_values(grid, "strategy", ["source_type_majority"])
    majorities = _grid_values(grid, "min_majority_frac", [RETRIEVAL_MIN_MAJORITY_FRAC])
    margins = _grid_values(grid, "min_margin", [RETRIEVAL_MIN_MAJORITY_MARGIN])
    jaccards = _grid_values(grid, "min_nearest_jaccard", [RETRIEVAL_MIN_NEAREST_JACCARD])
    for strategy in strategies:
        for maj in majorities:
            for margin in margins:
                for jac in jaccards:
                    yield {
                        "enabled": True,
                        "strategy": strategy,
                        "min_majority_frac": float(maj),
                        "min_margin": float(margin),
                        "min_nearest_jaccard": float(jac),
                        "require_typed_pool": bool(grid.get("require_typed_pool", False)),
                        "use_model_agreement_gate": RETRIEVAL_USE_MODEL_AGREEMENT_GATE,
                        "model_agreement_bypass_confidence": RETRIEVAL_MODEL_AGREEMENT_BYPASS_CONFIDENCE,
                    }

def run_retrieval_gate_sweep_for_checkpoint(records: list[dict], model_outputs: list[dict], retrieval_index: dict, *, checkpoint_label: str, order: int) -> dict:
    model_report = evaluate_predictions(model_outputs, records)
    selected_by_type = {}
    per_type = {}
    for rec_type in RETRIEVAL_GATE_SWEEP_TYPES:
        model_type_summary = model_report["by_type"].get(rec_type)
        if not model_type_summary:
            continue
        best = None
        candidates = []
        for cfg_idx, cfg in enumerate(iter_retrieval_gate_sweep_configs(rec_type)):
            outputs, summary = build_hybrid_outputs(records, model_outputs, retrieval_index, {rec_type: cfg}, debug_sample=0)
            report = evaluate_predictions(outputs, records)
            type_summary = report["by_type"].get(rec_type, {"raw_score": 0, "bucket_10": 0, "bucket_0": 0, "n": 0})
            used_by_type = summary["retrieval_used_by_type"].get(rec_type, {})
            entry = {
                "config_index": cfg_idx,
                "config": cfg,
                "summary": type_summary,
                "raw_delta_vs_model": type_summary.get("raw_score", 0) - model_type_summary.get("raw_score", 0),
                "exact10_delta_vs_model": type_summary.get("bucket_10", 0) - model_type_summary.get("bucket_10", 0),
                "retrieval_used": used_by_type.get("retrieval_used", 0),
                "fallback": used_by_type.get("model_fallback", 0),
            }
            key = (
                entry["summary"].get("raw_score", -1),
                entry["summary"].get("bucket_10", -1),
                -entry["summary"].get("bucket_0", 10**9),
                -entry["retrieval_used"],
            )
            entry["selection_key"] = list(key)
            candidates.append(entry)
            if best is None or key > tuple(best["selection_key"]):
                best = entry
        if best is None:
            continue
        if best["summary"].get("raw_score", 0) <= model_type_summary.get("raw_score", 0):
            selected_cfg = {"enabled": False, "disabled_reason": "best_sweep_not_better_than_model_only"}
            selected = {
                "selected": {
                    "config": selected_cfg,
                    "summary": model_type_summary,
                    "raw_delta_vs_model": 0,
                    "exact10_delta_vs_model": 0,
                    "selection_note": "disabled because no retrieval gate was better than model-only for this type",
                },
                "model_only": model_type_summary,
                "top_candidates": sorted(candidates, key=lambda x: tuple(x["selection_key"]), reverse=True)[:10],
            }
        else:
            selected_cfg = best["config"]
            selected = {
                "selected": best,
                "model_only": model_type_summary,
                "top_candidates": sorted(candidates, key=lambda x: tuple(x["selection_key"]), reverse=True)[:10],
            }
        selected_by_type[rec_type] = selected_cfg
        per_type[rec_type] = selected

    combined_outputs, combined_decisions = build_hybrid_outputs(records, model_outputs, retrieval_index, selected_by_type)
    combined_report = evaluate_predictions(combined_outputs, records)
    return {
        "checkpoint_label": checkpoint_label,
        "order": order,
        "model_only_summary": model_report["summary"],
        "selected_gate_config_by_type": selected_by_type,
        "per_type": per_type,
        "combined_summary": combined_report["summary"],
        "combined_by_type": combined_report["by_type"],
        "combined_decision_summary": combined_decisions,
    }

def generate_outputs(adapter_dir: Path, records: list[dict], output_path: Path, *, max_new_tokens: int = MAX_NEW_TOKENS, num_beams: int = NUM_BEAMS, retrieval_index: dict | None = None, model_output_path: Path | None = None, decision_report_path: Path | None = None, gate_config_by_type: dict | None = None):
    model_path = model_output_path or output_path
    model_outputs = generate_model_outputs(adapter_dir, records, model_path, max_new_tokens=max_new_tokens, num_beams=num_beams)
    if USE_HYBRID_RETRIEVAL:
        if retrieval_index is None:
            raise ValueError("USE_HYBRID_RETRIEVAL=True but retrieval_index is None")
        outputs, summary = apply_hybrid_retrieval(records, model_outputs, output_path, retrieval_index, decision_report_path, gate_config_by_type)
        return outputs
    if model_path != output_path:
        shutil.copyfile(model_path, output_path)
    return model_outputs
'''


ENHANCED_SELECTION_BLOCK = r'''def select_best_checkpoint_on_overlap_valid(records_for_selection: list[dict]):
    global CHECKPOINT_SELECTION, ACTIVE_RETRIEVAL_TYPE_GATE_CONFIG
    ckpts = list_stage_checkpoints()
    if not ckpts:
        print("[select] no adapter checkpoints found; keeping current FINAL_OUTPUT_DIR")
        CHECKPOINT_SELECTION = {
            "enabled": True,
            "selection_split": "source_overlap_query_disjoint",
            "status": "no_checkpoints_found",
            "final_output_dir": str(FINAL_OUTPUT_DIR),
        }
        SELECTED_CHECKPOINT_INFO_PATH.write_text(json.dumps(CHECKPOINT_SELECTION, ensure_ascii=False, indent=2), encoding="utf-8")
        return CHECKPOINT_SELECTION

    CHECKPOINT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[select] evaluating {len(ckpts)} checkpoints on {len(records_for_selection)} overlap-valid rows")

    entries = []
    sweep_entries = []
    best_entry = None
    best_key = None
    for order, ckpt in enumerate(ckpts):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", ckpt.name)
        out_path = CHECKPOINT_EVAL_DIR / f"overlap_valid_output_{order:02d}_{safe_name}.json"
        model_out_path = CHECKPOINT_EVAL_DIR / f"model_overlap_valid_output_{order:02d}_{safe_name}.json"
        report_path = CHECKPOINT_EVAL_DIR / f"overlap_valid_report_{order:02d}_{safe_name}.json"
        decision_path = CHECKPOINT_EVAL_DIR / f"hybrid_decisions_{order:02d}_{safe_name}.json"
        gate_sweep_path = CHECKPOINT_EVAL_DIR / f"retrieval_gate_sweep_{order:02d}_{safe_name}.json"

        model_outputs = generate_model_outputs(
            ckpt,
            records_for_selection,
            model_out_path,
            max_new_tokens=CHECKPOINT_EVAL_MAX_NEW_TOKENS,
            num_beams=CHECKPOINT_EVAL_NUM_BEAMS,
        )

        gate_config_by_type = None
        gate_sweep = None
        if USE_HYBRID_RETRIEVAL and RETRIEVAL_GATE_SWEEP_ENABLED:
            gate_sweep = run_retrieval_gate_sweep_for_checkpoint(
                records_for_selection,
                model_outputs,
                RETRIEVAL_INDEX_OVERLAP_TRAIN,
                checkpoint_label=ckpt.name,
                order=order,
            )
            gate_config_by_type = gate_sweep["selected_gate_config_by_type"]
            gate_sweep_path.write_text(json.dumps(gate_sweep, ensure_ascii=False, indent=2), encoding="utf-8")
            sweep_entries.append({
                "order": order,
                "label": ckpt.name,
                "path": str(gate_sweep_path),
                "model_only_summary": gate_sweep["model_only_summary"],
                "combined_summary": gate_sweep["combined_summary"],
                "selected_gate_config_by_type": gate_config_by_type,
            })

        if USE_HYBRID_RETRIEVAL:
            outputs, decision_summary = apply_hybrid_retrieval(
                records_for_selection,
                model_outputs,
                out_path,
                RETRIEVAL_INDEX_OVERLAP_TRAIN,
                decision_report_path=decision_path,
                gate_config_by_type=gate_config_by_type,
            )
        else:
            shutil.copyfile(model_out_path, out_path)
            decision_summary = {"enabled": False}

        rep = save_eval_report(out_path, records_for_selection, report_path)
        summary = rep["summary"]
        meta = {}
        meta_path = ckpt / "checkpoint_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                meta = {"meta_error": repr(exc)}
        entry = {
            "order": order,
            "label": ckpt.name,
            "adapter_dir": str(ckpt),
            "output_path": str(out_path),
            "model_output_path": str(model_out_path),
            "report_path": str(report_path),
            "hybrid_decision_path": str(decision_path),
            "gate_sweep_path": str(gate_sweep_path) if gate_sweep is not None else None,
            "retrieval_gate_config_by_type": gate_config_by_type,
            "decision_summary": decision_summary,
            "summary": summary,
            "meta": meta,
        }
        key = _score_tuple(summary, order)
        entry["selection_key"] = list(key)
        entries.append(entry)
        print(f"[select] {ckpt.name}: raw={summary['raw_score']} exact10={summary['buckets'].get('10')} extractable={summary['extractable']} key={key}")
        if best_key is None or key > best_key:
            best_key = key
            best_entry = entry
        if not KEEP_CHECKPOINT_EVAL_OUTPUTS:
            for p in [out_path, model_out_path, decision_path, gate_sweep_path]:
                try:
                    p.unlink()
                except Exception:
                    pass

    if best_entry is None:
        raise RuntimeError("Checkpoint selection failed: no best checkpoint")

    ACTIVE_RETRIEVAL_TYPE_GATE_CONFIG = best_entry.get("retrieval_gate_config_by_type") or RETRIEVAL_TYPE_GATE_CONFIG
    SELECTED_RETRIEVAL_GATE_CONFIG_PATH.write_text(json.dumps(ACTIVE_RETRIEVAL_TYPE_GATE_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
    RETRIEVAL_GATE_SWEEP_REPORT_PATH.write_text(json.dumps({
        "enabled": RETRIEVAL_GATE_SWEEP_ENABLED,
        "sweep_types": RETRIEVAL_GATE_SWEEP_TYPES,
        "selected_checkpoint_label": best_entry["label"],
        "selected_gate_config_by_type": ACTIVE_RETRIEVAL_TYPE_GATE_CONFIG,
        "entries": sweep_entries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_dir = Path(best_entry["adapter_dir"])
    if FINAL_OUTPUT_DIR.exists():
        shutil.rmtree(FINAL_OUTPUT_DIR)
    shutil.copytree(selected_dir, FINAL_OUTPUT_DIR)
    final_hash = sha256_dir(FINAL_OUTPUT_DIR)
    (FINAL_OUTPUT_DIR / "model_hash.txt").write_text(final_hash + "\n", encoding="utf-8")

    CHECKPOINT_SELECTION = {
        "enabled": True,
        "selection_split": "source_overlap_query_disjoint",
        "selection_metric": "max(hybrid_raw_score_after_type_gate_sweep, exact10, extractable, tie_break)" if RETRIEVAL_GATE_SWEEP_ENABLED else "max(gated_hybrid_raw_score, exact10, extractable, tie_break)",
        "tie_break": CHECKPOINT_TIE_BREAK,
        "hybrid_retrieval": USE_HYBRID_RETRIEVAL,
        "retrieval_allowed_types": RETRIEVAL_ALLOWED_TYPES,
        "retrieval_gate_sweep_enabled": RETRIEVAL_GATE_SWEEP_ENABLED,
        "selected_retrieval_gate_config_by_type": ACTIVE_RETRIEVAL_TYPE_GATE_CONFIG,
        "status": "selected",
        "eval_n": len(records_for_selection),
        "num_checkpoints": len(entries),
        "selected": best_entry,
        "selected_adapter_dir": str(selected_dir),
        "final_output_dir": str(FINAL_OUTPUT_DIR),
        "final_sha256": final_hash,
        "all_checkpoints": entries,
    }
    CHECKPOINT_SELECTION_REPORT_PATH.write_text(json.dumps(CHECKPOINT_SELECTION, ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_CHECKPOINT_INFO_PATH.write_text(json.dumps(CHECKPOINT_SELECTION["selected"], ensure_ascii=False, indent=2), encoding="utf-8")
    print("[select] selected:", best_entry["label"], best_entry["summary"])
    print("[select] active retrieval gate:", ACTIVE_RETRIEVAL_TYPE_GATE_CONFIG)
    print("[select] copied to:", FINAL_OUTPUT_DIR, "sha256=", final_hash)
    return CHECKPOINT_SELECTION
'''


def find_function_block(src: str, start: str, end: str) -> str:
    a = src.index(start)
    b = src.index(end, a)
    return src[a:b]


def patch_common(nb: dict, version: str, version_slug: str, title: str, description: str, *, output_stem: str) -> dict:
    nb = copy.deepcopy(nb)
    nb["cells"][0]["source"] = lines(f"# {title}\n\n{description}\n")

    config = "".join(nb["cells"][2]["source"])
    config = config.replace("v13_type_gated_hybrid_full_retrain", version)
    config = config.replace("gpt2_math_lora_v13_answer_only", f"gpt2_math_lora_{output_stem}_answer_only")
    config = config.replace("gpt2_math_lora_v13_sft", f"gpt2_math_lora_{output_stem}_sft")
    config = config.replace("gpt2_math_lora_v13_final", f"gpt2_math_lora_{output_stem}_final")
    config = config.replace("gpt2_math_lora_v13_checkpoints", f"gpt2_math_lora_{output_stem}_checkpoints")
    config = config.replace("v13_checkpoint_eval_overlap_valid", f"{output_stem}_checkpoint_eval_overlap_valid")
    config = config.replace("gpt2_math_lora_v13_full_train_final", f"gpt2_math_lora_{output_stem}_full_train_final")
    config = replace_once(
        config,
        'HYBRID_DECISION_REPORT_PATH     = WORKING_DIR / "hybrid_decision_report.json"\n',
        'HYBRID_DECISION_REPORT_PATH     = WORKING_DIR / "hybrid_decision_report.json"\n'
        'RETRIEVAL_GATE_SWEEP_REPORT_PATH = WORKING_DIR / "retrieval_gate_sweep_report.json"\n'
        'SELECTED_RETRIEVAL_GATE_CONFIG_PATH = WORKING_DIR / "selected_retrieval_gate_config.json"\n',
    )
    config = replace_once(
        config,
        'RETRIEVAL_DEBUG_SAMPLE = 20\n',
        'RETRIEVAL_DEBUG_SAMPLE = 20\n'
        '\n'
        '# V14 confidence gate. When the model already agrees with the retrieval\n'
        '# candidate, we allow retrieval; when it disagrees, the candidate must\n'
        '# pass stricter type-specific confidence thresholds.\n'
        'RETRIEVAL_MIN_MAJORITY_MARGIN = 0.0\n'
        'RETRIEVAL_MIN_NEAREST_JACCARD = 0.0\n'
        'RETRIEVAL_USE_MODEL_AGREEMENT_GATE = True\n'
        'RETRIEVAL_MODEL_AGREEMENT_BYPASS_CONFIDENCE = True\n'
        'RETRIEVAL_TYPE_GATE_CONFIG = {}\n'
        'ACTIVE_RETRIEVAL_TYPE_GATE_CONFIG = {}\n'
        'RETRIEVAL_GATE_SWEEP_ENABLED = False\n'
        'RETRIEVAL_GATE_SWEEP_TYPES = []\n'
        'RETRIEVAL_GATE_SWEEP_GRID = {}\n',
    )
    nb["cells"][2]["source"] = lines(config)

    gen = "".join(nb["cells"][8]["source"])
    old_retrieval = find_function_block(gen, "def build_retrieval_index", "def _adapter_sort_key")
    gen = gen.replace(old_retrieval, ENHANCED_RETRIEVAL_BLOCK + "\n")
    old_selection = find_function_block(gen, "def select_best_checkpoint_on_overlap_valid", "def selected_epoch_count")
    gen = gen.replace(old_selection, ENHANCED_SELECTION_BLOCK + "\n")
    nb["cells"][8]["source"] = lines(gen)

    manifest = "".join(nb["cells"][9]["source"])
    manifest = manifest.replace('"notebook_version": "v13_type_gated_hybrid_full_retrain"', f'"notebook_version": "{version}"')
    manifest = replace_once(
        manifest,
        '        "retrieval_min_majority_frac": RETRIEVAL_MIN_MAJORITY_FRAC,\n'
        '        "retrieval_allowed_types": RETRIEVAL_ALLOWED_TYPES,\n',
        '        "retrieval_min_majority_frac": RETRIEVAL_MIN_MAJORITY_FRAC,\n'
        '        "retrieval_min_majority_margin": RETRIEVAL_MIN_MAJORITY_MARGIN,\n'
        '        "retrieval_min_nearest_jaccard": RETRIEVAL_MIN_NEAREST_JACCARD,\n'
        '        "retrieval_allowed_types": RETRIEVAL_ALLOWED_TYPES,\n'
        '        "retrieval_type_gate_config": RETRIEVAL_TYPE_GATE_CONFIG,\n'
        '        "active_retrieval_type_gate_config": ACTIVE_RETRIEVAL_TYPE_GATE_CONFIG,\n'
        '        "retrieval_gate_sweep_enabled": RETRIEVAL_GATE_SWEEP_ENABLED,\n'
        '        "retrieval_gate_sweep_types": RETRIEVAL_GATE_SWEEP_TYPES,\n',
    )
    manifest = replace_once(
        manifest,
        '        "hybrid_decision_report": str(HYBRID_DECISION_REPORT_PATH),\n',
        '        "hybrid_decision_report": str(HYBRID_DECISION_REPORT_PATH),\n'
        '        "retrieval_gate_sweep_report": str(RETRIEVAL_GATE_SWEEP_REPORT_PATH),\n'
        '        "selected_retrieval_gate_config": str(SELECTED_RETRIEVAL_GATE_CONFIG_PATH),\n',
    )
    manifest = manifest.replace('manifest_path = WORKING_DIR / "v13_type_gated_hybrid_full_retrain_manifest.json"', f'manifest_path = WORKING_DIR / "{version_slug}_manifest.json"')
    nb["cells"][9]["source"] = lines(manifest)
    return nb


def make_gate_sweep(base: dict) -> dict:
    nb = patch_common(
        base,
        "v14_retrieval_gate_sweep",
        "v14_retrieval_gate_sweep",
        "V14 - Retrieval Gate Sweep",
        "Sweep type-specific retrieval confidence gates on the source-overlap query-disjoint split. The sweep keeps the V13 type gate, but tunes majority fraction, top1-top2 margin, nearest same-answer Jaccard, and model-agreement fallback thresholds per type.",
        output_stem="v14_gate_sweep",
    )
    config = "".join(nb["cells"][2]["source"])
    config = replace_once(
        config,
        'RETRIEVAL_GATE_SWEEP_ENABLED = False\n'
        'RETRIEVAL_GATE_SWEEP_TYPES = []\n'
        'RETRIEVAL_GATE_SWEEP_GRID = {}\n',
        'RETRIEVAL_GATE_SWEEP_ENABLED = True\n'
        'RETRIEVAL_GATE_SWEEP_TYPES = ["GSM_Rephrased", "MATH_Rephrased", "GSM_AnsAug", "MATH_AnsAug"]\n'
        'RETRIEVAL_GATE_SWEEP_GRID = {\n'
        '    "_default": {\n'
        '        "strategy": ["source_type_majority"],\n'
        '        "min_majority_frac": [0.34, 0.50, 0.67, 0.75],\n'
        '        "min_margin": [0.0, 0.15, 0.33, 0.50],\n'
        '        "min_nearest_jaccard": [0.0, 0.45, 0.65, 0.80],\n'
        '    },\n'
        '    "GSM_AnsAug": {"min_majority_frac": [0.34, 0.45, 0.50, 0.67], "min_margin": [0.0, 0.10, 0.20, 0.33], "min_nearest_jaccard": [0.0, 0.35, 0.50, 0.65]},\n'
        '    "MATH_AnsAug": {"min_majority_frac": [0.34, 0.45, 0.50, 0.67], "min_margin": [0.0, 0.10, 0.20, 0.33], "min_nearest_jaccard": [0.0, 0.35, 0.50, 0.65]},\n'
        '}\n',
    )
    nb["cells"][2]["source"] = lines(config)
    return nb


def make_fobar_sv(base: dict) -> dict:
    nb = patch_common(
        base,
        "v14_fobar_sv_extended_retrieval",
        "v14_fobar_sv_extended_retrieval",
        "V14 - FOBAR/SV Extended Retrieval",
        "Extend retrieval beyond Rephrased/AnsAug to FOBAR and SV, but only with strict typed-pool and nearest-query gates. This is intentionally conservative because same-source FOBAR/SV examples often contain conflicting answers.",
        output_stem="v14_fobar_sv",
    )
    config = "".join(nb["cells"][2]["source"])
    config = config.replace(
        'RETRIEVAL_ALLOWED_TYPES = ["GSM_Rephrased", "MATH_Rephrased", "GSM_AnsAug", "MATH_AnsAug"]  # V13 gate; None means no type gate',
        'RETRIEVAL_ALLOWED_TYPES = ["GSM_Rephrased", "MATH_Rephrased", "GSM_AnsAug", "MATH_AnsAug", "GSM_FOBAR", "MATH_FOBAR", "GSM_SV", "MATH_SV"]  # V14 extends retrieval to FOBAR/SV with strict gates',
    )
    config = replace_once(
        config,
        'RETRIEVAL_TYPE_GATE_CONFIG = {}\n',
        'RETRIEVAL_TYPE_GATE_CONFIG = {\n'
        '    # Rephrased/AnsAug keep V13-style source-type majority retrieval.\n'
        '    "GSM_Rephrased": {"strategy": "source_type_majority", "min_majority_frac": 0.34, "min_margin": 0.0, "min_nearest_jaccard": 0.0},\n'
        '    "MATH_Rephrased": {"strategy": "source_type_majority", "min_majority_frac": 0.34, "min_margin": 0.0, "min_nearest_jaccard": 0.0},\n'
        '    "GSM_AnsAug": {"strategy": "source_type_majority", "min_majority_frac": 0.34, "min_margin": 0.0, "min_nearest_jaccard": 0.0},\n'
        '    "MATH_AnsAug": {"strategy": "source_type_majority", "min_majority_frac": 0.34, "min_margin": 0.0, "min_nearest_jaccard": 0.0},\n'
        '    # FOBAR/SV are conflict-heavy. Use only same-source+same-type nearest examples,\n'
        '    # and require high lexical similarity when the model does not already agree.\n'
        '    "GSM_FOBAR": {"strategy": "source_type_nearest_query", "require_typed_pool": True, "min_majority_frac": 0.0, "min_margin": 0.0, "min_nearest_jaccard": 0.82},\n'
        '    "MATH_FOBAR": {"strategy": "source_type_nearest_query", "require_typed_pool": True, "min_majority_frac": 0.0, "min_margin": 0.0, "min_nearest_jaccard": 0.82},\n'
        '    "GSM_SV": {"strategy": "source_type_nearest_query", "require_typed_pool": True, "min_majority_frac": 0.0, "min_margin": 0.0, "min_nearest_jaccard": 0.84},\n'
        '    "MATH_SV": {"strategy": "source_type_nearest_query", "require_typed_pool": True, "min_majority_frac": 0.0, "min_margin": 0.0, "min_nearest_jaccard": 0.84},\n'
        '}\n',
    )
    nb["cells"][2]["source"] = lines(config)
    return nb


def main() -> None:
    base = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    outputs = {
        "finetune_gpt2_for_math_v14_retrieval_gate_sweep.ipynb": make_gate_sweep(base),
        "finetune_gpt2_for_math_v14_fobar_sv_extended_retrieval.ipynb": make_fobar_sv(base),
    }
    for path, nb in outputs.items():
        Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
