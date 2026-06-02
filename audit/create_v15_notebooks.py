from __future__ import annotations

import copy
import json
from pathlib import Path


ENSEMBLE_BASE = Path("finetune_gpt2_for_math_v14_5_gate_sweep_full_train_valid_select.ipynb")
EXPERT_BASE = Path("finetune_gpt2_for_math_v13_type_gated_full_retrain.ipynb")


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Missing expected block:\n{old[:500]}")
    return text.replace(old, new, 1)


ENSEMBLE_CELL = r'''
# ============================================================
# V15. Ensemble/ranker over answer-only candidates
# ============================================================
if ENSEMBLE_RANKER_ENABLED:
    ENSEMBLE_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    def _resolve_artifact_path(path_value) -> Path:
        p = Path(path_value)
        if p.exists():
            return p
        local = WORKING_DIR / p.parent.name / p.name
        if local.exists():
            return local
        return p

    def _safe_num_key(value, digits: int = 10):
        if value is None:
            return None
        try:
            value = float(value)
        except Exception:
            return None
        if abs(value - round(value)) <= 1e-9:
            return str(int(round(value)))
        return f"{value:.{digits}g}"

    def _log1p_count(counter: Counter, key) -> float:
        return math.log1p(counter.get(key, 0))

    def build_answer_priors(records: list[dict]) -> dict:
        global_counts = Counter()
        type_counts = defaultdict(Counter)
        for rec in records:
            key = _safe_num_key(rec.get("_gold_num"))
            if key is None:
                continue
            rec_type = rec.get("type") or "unknown"
            global_counts[key] += 1
            type_counts[rec_type][key] += 1
        return {"global": global_counts, "by_type": type_counts}

    ANSWER_PRIORS = build_answer_priors(train_clean)

    V15_ALLOWED_RETRIEVAL_TYPES = ["GSM_Rephrased", "MATH_Rephrased", "GSM_AnsAug", "MATH_AnsAug"]
    V15_V13_GATE_CONFIG = {
        t: {
            "enabled": True,
            "strategy": "source_type_majority",
            "min_majority_frac": 0.34,
            "min_margin": 0.0,
            "min_nearest_jaccard": 0.0,
            "require_typed_pool": False,
            "use_model_agreement_gate": False,
            "model_agreement_bypass_confidence": False,
        }
        for t in V15_ALLOWED_RETRIEVAL_TYPES
    }
    V15_STRICT_GATE_CONFIG = {
        t: {
            "enabled": True,
            "strategy": "source_type_majority",
            "min_majority_frac": 0.50,
            "min_margin": 0.33,
            "min_nearest_jaccard": 0.45 if "GSM" in t else 0.0,
            "require_typed_pool": False,
            "use_model_agreement_gate": True,
            "model_agreement_bypass_confidence": True,
        }
        for t in V15_ALLOWED_RETRIEVAL_TYPES
    }
    V15_EXTRA_GATE_CONFIGS = {
        "hybrid_v13_gate": V15_V13_GATE_CONFIG,
        "hybrid_strict_gate": V15_STRICT_GATE_CONFIG,
    }

    def _unique_epoch_entries(selection: dict, top_k: int | None = None) -> list[dict]:
        entries = []
        seen = set()
        for entry in selection.get("all_checkpoints", []):
            label = str(entry.get("label") or "")
            if not label.startswith("epoch_"):
                continue
            if label in seen:
                continue
            seen.add(label)
            entries.append(entry)
        entries.sort(key=lambda e: tuple(e.get("selection_key") or [0]), reverse=True)
        if top_k:
            entries = entries[:top_k]
        entries.sort(key=lambda e: e.get("order", 10**9))
        return entries

    def _write_candidate_outputs(name: str, records: list[dict], outputs: list[dict]) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ENSEMBLE_CANDIDATE_DIR / f"{safe}.json"
        path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def collect_checkpoint_candidates(
        records: list[dict],
        selection: dict,
        retrieval_index: dict,
        *,
        split_name: str,
        use_cached: bool,
        top_k: int | None = ENSEMBLE_TOP_K_CHECKPOINTS,
    ) -> list[dict]:
        candidates = []
        entries = _unique_epoch_entries(selection, top_k)
        if not entries:
            raise RuntimeError("No epoch checkpoint entries found for ensemble candidates")

        for rank, entry in enumerate(entries):
            label = str(entry["label"])
            adapter_dir = Path(entry["adapter_dir"])

            model_outputs = None
            if use_cached:
                model_path = _resolve_artifact_path(entry["model_output_path"])
                if model_path.exists():
                    model_outputs = json.loads(model_path.read_text(encoding="utf-8"))
            if model_outputs is None:
                model_path = ENSEMBLE_CANDIDATE_DIR / f"{split_name}_model_{label}.json"
                model_outputs = generate_model_outputs(
                    adapter_dir,
                    records,
                    model_path,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=NUM_BEAMS,
                )

            if ENSEMBLE_INCLUDE_MODEL_CANDIDATES:
                candidates.append({
                    "name": f"model::{label}",
                    "kind": "model",
                    "checkpoint_label": label,
                    "checkpoint_rank": rank,
                    "checkpoint_raw": entry.get("summary", {}).get("raw_score", 0),
                    "outputs": model_outputs,
                    "path": str(_write_candidate_outputs(f"{split_name}_model_{label}", records, model_outputs)),
                })

            if ENSEMBLE_INCLUDE_RETRIEVAL_CANDIDATES:
                hybrid_outputs = None
                if use_cached:
                    hybrid_path = _resolve_artifact_path(entry["output_path"])
                    if hybrid_path.exists():
                        hybrid_outputs = json.loads(hybrid_path.read_text(encoding="utf-8"))
                if hybrid_outputs is None:
                    hybrid_outputs, _summary = build_hybrid_outputs(records, model_outputs, retrieval_index, ACTIVE_RETRIEVAL_TYPE_GATE_CONFIG, debug_sample=0)
                candidates.append({
                    "name": f"hybrid_active::{label}",
                    "kind": "hybrid_active",
                    "checkpoint_label": label,
                    "checkpoint_rank": rank,
                    "checkpoint_raw": entry.get("summary", {}).get("raw_score", 0),
                    "outputs": hybrid_outputs,
                    "path": str(_write_candidate_outputs(f"{split_name}_hybrid_active_{label}", records, hybrid_outputs)),
                })

                for gate_name, gate_cfg in V15_EXTRA_GATE_CONFIGS.items():
                    extra_outputs, _summary = build_hybrid_outputs(records, model_outputs, retrieval_index, gate_cfg, debug_sample=0)
                    candidates.append({
                        "name": f"{gate_name}::{label}",
                        "kind": gate_name,
                        "checkpoint_label": label,
                        "checkpoint_rank": rank,
                        "checkpoint_raw": entry.get("summary", {}).get("raw_score", 0),
                        "outputs": extra_outputs,
                        "path": str(_write_candidate_outputs(f"{split_name}_{gate_name}_{label}", records, extra_outputs)),
                    })
        print(f"[ensemble] collected {len(candidates)} candidates for {split_name}")
        return candidates

    def candidate_prediction_numbers(candidates: list[dict], row_idx: int) -> list[str | None]:
        keys = []
        for cand in candidates:
            _answer, pred_num = extract_pred(cand["outputs"][row_idx])
            keys.append(_safe_num_key(pred_num))
        return keys

    def make_feature_builder(candidates: list[dict], records: list[dict]):
        candidate_names = sorted({c["name"] for c in candidates})
        kinds = sorted({c["kind"] for c in candidates})
        types = sorted({r.get("type") or "unknown" for r in records})
        ckpts = sorted({c["checkpoint_label"] for c in candidates})
        name_to_i = {v: i for i, v in enumerate(candidate_names)}
        kind_to_i = {v: i for i, v in enumerate(kinds)}
        type_to_i = {v: i for i, v in enumerate(types)}
        ckpt_to_i = {v: i for i, v in enumerate(ckpts)}

        def build(row_idx: int, cand: dict, agreement_counter: Counter) -> list[float]:
            rec = records[row_idx]
            rec_type = rec.get("type") or "unknown"
            output = cand["outputs"][row_idx]
            _answer, pred_num = extract_pred(output)
            pred_key = _safe_num_key(pred_num)
            agree = agreement_counter.get(pred_key, 0) if pred_key is not None else 0
            n_cands = max(1, len(candidates))
            feats = [
                1.0,
                cand.get("checkpoint_rank", 0) / max(1, ENSEMBLE_TOP_K_CHECKPOINTS),
                cand.get("checkpoint_raw", 0) / 10000.0,
                agree / n_cands,
                float(agree),
                _log1p_count(ANSWER_PRIORS["global"], pred_key),
                _log1p_count(ANSWER_PRIORS["by_type"].get(rec_type, Counter()), pred_key),
                1.0 if pred_num is not None and abs(pred_num - round(pred_num)) <= 1e-9 else 0.0,
                math.log1p(abs(float(pred_num))) if pred_num is not None else 0.0,
                1.0 if pred_num == 0 else 0.0,
            ]
            feats.extend(1.0 if name_to_i[cand["name"]] == i else 0.0 for i in range(len(candidate_names)))
            feats.extend(1.0 if kind_to_i[cand["kind"]] == i else 0.0 for i in range(len(kinds)))
            feats.extend(1.0 if type_to_i[rec_type] == i else 0.0 for i in range(len(types)))
            feats.extend(1.0 if ckpt_to_i[cand["checkpoint_label"]] == i else 0.0 for i in range(len(ckpts)))
            return feats

        return build

    def evaluate_candidate_outputs(candidates: list[dict], records: list[dict]) -> dict:
        reports = {}
        for cand in candidates:
            reports[cand["name"]] = evaluate_predictions(cand["outputs"], records)
        return reports

    def train_ranker(candidates: list[dict], records: list[dict], reports: dict):
        build_features = make_feature_builder(candidates, records)
        X, y = [], []
        for row_idx in range(len(records)):
            keys = candidate_prediction_numbers(candidates, row_idx)
            agreement = Counter(k for k in keys if k is not None)
            for cand in candidates:
                X.append(build_features(row_idx, cand, agreement))
                y.append(reports[cand["name"]]["rows"][row_idx]["score"] / 10.0)
        try:
            from sklearn.ensemble import ExtraTreesRegressor

            model = ExtraTreesRegressor(
                n_estimators=ENSEMBLE_RANKER_TREES,
                max_depth=ENSEMBLE_RANKER_MAX_DEPTH,
                min_samples_leaf=ENSEMBLE_RANKER_MIN_SAMPLES_LEAF,
                random_state=SEED,
                n_jobs=-1,
            )
            model.fit(X, y)
            return {"kind": "sklearn_extra_trees", "model": model, "feature_builder": build_features}
        except Exception as exc:
            print("[ensemble] sklearn ranker unavailable; fallback table ranker:", repr(exc))
            global_mean = mean(y) if y else 0.0
            by_candidate = {}
            by_type_candidate = {}
            for cand in candidates:
                vals = [row["score"] / 10.0 for row in reports[cand["name"]]["rows"]]
                by_candidate[cand["name"]] = mean(vals) if vals else global_mean
                for rec_type in sorted({r.get("type") or "unknown" for r in records}):
                    vals_t = [
                        row["score"] / 10.0
                        for row, rec in zip(reports[cand["name"]]["rows"], records)
                        if (rec.get("type") or "unknown") == rec_type
                    ]
                    by_type_candidate[(rec_type, cand["name"])] = mean(vals_t) if vals_t else by_candidate[cand["name"]]
            return {
                "kind": "fallback_table",
                "global_mean": global_mean,
                "by_candidate": by_candidate,
                "by_type_candidate": by_type_candidate,
            }

    def ranker_score(ranker: dict, row_idx: int, cand: dict, records: list[dict], candidates: list[dict], agreement: Counter) -> float:
        if ranker["kind"] == "sklearn_extra_trees":
            x = ranker["feature_builder"](row_idx, cand, agreement)
            return float(ranker["model"].predict([x])[0])
        rec_type = records[row_idx].get("type") or "unknown"
        _answer, pred_num = extract_pred(cand["outputs"][row_idx])
        pred_key = _safe_num_key(pred_num)
        agree_bonus = 0.04 * agreement.get(pred_key, 0) / max(1, len(candidates)) if pred_key is not None else 0.0
        return (
            0.60 * ranker["by_type_candidate"].get((rec_type, cand["name"]), ranker["global_mean"])
            + 0.35 * ranker["by_candidate"].get(cand["name"], ranker["global_mean"])
            + agree_bonus
        )

    def choose_outputs_with_ranker(candidates: list[dict], records: list[dict], ranker: dict) -> tuple[list[dict], dict]:
        selected_outputs = []
        choice_counts = Counter()
        choice_by_type = defaultdict(Counter)
        for row_idx, rec in enumerate(records):
            keys = candidate_prediction_numbers(candidates, row_idx)
            agreement = Counter(k for k in keys if k is not None)
            scored = []
            for cand in candidates:
                score = ranker_score(ranker, row_idx, cand, records, candidates, agreement)
                scored.append((score, cand.get("checkpoint_raw", 0), -cand.get("checkpoint_rank", 999), cand))
            scored.sort(reverse=True, key=lambda x: (x[0], x[1], x[2], x[3]["name"]))
            chosen = scored[0][3]
            item = dict(chosen["outputs"][row_idx])
            selected_outputs.append(item)
            choice_counts[chosen["name"]] += 1
            choice_by_type[rec.get("type") or "unknown"][chosen["name"]] += 1
        summary = {
            "ranker_kind": ranker["kind"],
            "num_candidates": len(candidates),
            "choice_counts": dict(choice_counts.most_common()),
            "choice_by_type": {k: dict(v.most_common()) for k, v in sorted(choice_by_type.items())},
        }
        return selected_outputs, summary

    def save_ensemble_outputs(records: list[dict], outputs: list[dict], output_path: Path, report_path: Path, summary_path: Path, summary: dict):
        output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
        report = save_eval_report(output_path, records, report_path) if records and records[0].get("response_vi") else None
        payload = dict(summary)
        if report is not None:
            payload["summary"] = report["summary"]
            payload["by_type"] = report["by_type"]
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return report, payload

    def run_ensemble_for_valid():
        if VALID_OUTPUT_PATH.exists():
            shutil.copyfile(VALID_OUTPUT_PATH, SELECTED_VALID_OUTPUT_PATH)
        if VALID_REPORT_PATH.exists():
            shutil.copyfile(VALID_REPORT_PATH, SELECTED_VALID_REPORT_PATH)
        if MODEL_VALID_OUTPUT_PATH.exists():
            shutil.copyfile(MODEL_VALID_OUTPUT_PATH, SELECTED_MODEL_VALID_OUTPUT_PATH)

        candidates = collect_checkpoint_candidates(
            valid_records,
            CHECKPOINT_SELECTION,
            RETRIEVAL_INDEX_FULL_TRAIN,
            split_name="valid",
            use_cached=True,
            top_k=ENSEMBLE_TOP_K_CHECKPOINTS,
        )
        reports = evaluate_candidate_outputs(candidates, valid_records)
        candidate_summary = {
            name: {
                "summary": rep["summary"],
                "by_type": rep["by_type"],
            }
            for name, rep in reports.items()
        }
        ranker = train_ranker(candidates, valid_records, reports)
        outputs, summary = choose_outputs_with_ranker(candidates, valid_records, ranker)
        summary["candidate_reports"] = candidate_summary
        report, payload = save_ensemble_outputs(
            valid_records,
            outputs,
            VALID_OUTPUT_PATH,
            VALID_REPORT_PATH,
            ENSEMBLE_RANKER_REPORT_PATH,
            summary,
        )
        if report is not None:
            print("[ensemble:valid]", report["summary"])
        return ranker, candidates, payload

    def run_ensemble_for_test(ranker):
        top_k = ENSEMBLE_PHASE2_TOP_K_CHECKPOINTS or ENSEMBLE_TOP_K_CHECKPOINTS
        candidates = collect_checkpoint_candidates(
            test_records,
            CHECKPOINT_SELECTION,
            RETRIEVAL_INDEX_FULL_TRAIN,
            split_name="test",
            use_cached=False,
            top_k=top_k,
        )
        outputs, summary = choose_outputs_with_ranker(candidates, test_records, ranker)
        TEST_OUTPUT_PATH.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
        (WORKING_DIR / "ensemble_test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[ensemble:phase2] wrote", TEST_OUTPUT_PATH)

    if RUN_MODE == "phase1":
        _ranker, _candidates, _payload = run_ensemble_for_valid()
    elif RUN_MODE == "phase2":
        _ranker, _candidates, _payload = run_ensemble_for_valid()
        test_records = load_records(TEST_FILE)
        run_ensemble_for_test(_ranker)
'''


EXPERT_CELL = r'''
# ============================================================
# V15. Type-specific SV/FOBAR expert adapter
# ============================================================
if EXPERT_ENABLED:
    EXPERT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERT_CHECKPOINT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    def _expert_records_with_original_ids(records: list[dict], types: set[str]) -> tuple[list[dict], list[int]]:
        out, indices = [], []
        for idx, rec in enumerate(records):
            if (rec.get("type") or "unknown") in types:
                item = dict(rec)
                item["id"] = rec.get("id", idx)
                out.append(item)
                indices.append(idx)
        return out, indices

    def _raw_for_type(report: dict, rec_type: str) -> int:
        return int(report.get("by_type", {}).get(rec_type, {}).get("raw_score", 0))

    class ExpertSaveAdapterEveryEpochCallback(TrainerCallback):
        def __init__(self, root_dir: Path):
            self.root_dir = Path(root_dir)
            self.saved = []

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            if model is None:
                return
            label = _checkpoint_label(float(state.epoch or 0.0))
            ckpt_dir = save_adapter_checkpoint(
                model,
                self.root_dir,
                epoch_value=float(state.epoch or 0.0),
                final=False,
                stage_name=EXPERT_STAGE_NAME,
            )
            self.saved.append({"label": label, "path": str(ckpt_dir), "epoch": float(state.epoch or 0.0)})

    def train_expert_lora_stage(train_records_for_stage: list[dict]):
        print("\n" + "=" * 90)
        print(f"[expert:train] train={len(train_records_for_stage)} epochs={EXPERT_EPOCHS} lr={EXPERT_LR}")
        try:
            if "model" in globals():
                del globals()["model"]
            torch.cuda.empty_cache()
        except Exception:
            pass
        expert_model = build_lora_model()
        train_ds = SFTDataset(train_records_for_stage, tokenizer, MAX_LENGTH_STAGE_A, desc=EXPERT_STAGE_NAME)
        collator = PadCollator(SAFE_EOS_ID)
        callback = ExpertSaveAdapterEveryEpochCallback(EXPERT_CHECKPOINT_ROOT_DIR)
        trainer = Trainer(
            model=expert_model,
            args=build_training_args(EXPERT_OUTPUT_DIR, EXPERT_EPOCHS, EXPERT_LR),
            train_dataset=train_ds,
            data_collator=collator,
            callbacks=[callback],
        )
        t0 = time.time()
        trainer.train()
        dt = time.time() - t0
        trainer.save_model(EXPERT_OUTPUT_DIR)
        tokenizer.save_pretrained(EXPERT_OUTPUT_DIR)
        model_hash = sha256_dir(EXPERT_OUTPUT_DIR)
        (EXPERT_OUTPUT_DIR / "model_hash.txt").write_text(model_hash + "\n", encoding="utf-8")
        del trainer
        del expert_model
        torch.cuda.empty_cache()
        info = {
            "train_records": len(train_records_for_stage),
            "epochs": EXPERT_EPOCHS,
            "lr": EXPERT_LR,
            "output_dir": str(EXPERT_OUTPUT_DIR),
            "checkpoint_root_dir": str(EXPERT_CHECKPOINT_ROOT_DIR),
            "wall_minutes": dt / 60,
            "sha256": model_hash,
            "saved_checkpoints": callback.saved,
        }
        EXPERT_TRAIN_INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[expert:train] saved", EXPERT_OUTPUT_DIR, "wall_min", dt / 60)
        return info

    def list_expert_checkpoints() -> list[Path]:
        candidates = []
        if EXPERT_CHECKPOINT_ROOT_DIR.exists():
            candidates.extend([p for p in EXPERT_CHECKPOINT_ROOT_DIR.iterdir() if p.is_dir() and has_peft_adapter(p)])
        if has_peft_adapter(EXPERT_OUTPUT_DIR):
            candidates.append(EXPERT_OUTPUT_DIR)
        dedup, seen = [], set()
        for p in candidates:
            key = str(p.resolve())
            if key not in seen:
                dedup.append(p)
                seen.add(key)
        return sorted(dedup, key=_adapter_sort_key)

    def select_expert_by_valid_type(baseline_output_path: Path, baseline_report_path: Path) -> dict:
        expert_valid_records, expert_valid_indices = _expert_records_with_original_ids(valid_records, set(EXPERT_TYPES))
        if not expert_valid_records:
            raise RuntimeError("No valid records found for EXPERT_TYPES")
        baseline_outputs = json.loads(baseline_output_path.read_text(encoding="utf-8"))
        baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        baseline_type_raw = {t: _raw_for_type(baseline_report, t) for t in EXPERT_TYPES}

        entries = []
        best_by_type = {
            t: {
                "use_expert": False,
                "baseline_raw": baseline_type_raw.get(t, 0),
                "best_raw": baseline_type_raw.get(t, 0),
                "selected_label": "baseline_general",
                "output_path": None,
            }
            for t in EXPERT_TYPES
        }

        for order, ckpt in enumerate(list_expert_checkpoints()):
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", ckpt.name)
            out_path = EXPERT_EVAL_DIR / f"expert_valid_output_{order:02d}_{safe_name}.json"
            rep_path = EXPERT_EVAL_DIR / f"expert_valid_report_{order:02d}_{safe_name}.json"
            outputs = generate_model_outputs(
                ckpt,
                expert_valid_records,
                out_path,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
            )
            report = save_eval_report(out_path, expert_valid_records, rep_path)
            entry = {
                "order": order,
                "label": ckpt.name,
                "adapter_dir": str(ckpt),
                "output_path": str(out_path),
                "report_path": str(rep_path),
                "summary": report["summary"],
                "by_type": report["by_type"],
            }
            entries.append(entry)
            print(f"[expert:select] {ckpt.name}: raw={report['summary']['raw_score']} exact10={report['summary']['buckets'].get('10')}")
            for rec_type in EXPERT_TYPES:
                candidate_raw = _raw_for_type(report, rec_type)
                if candidate_raw > best_by_type[rec_type]["best_raw"]:
                    best_by_type[rec_type].update({
                        "use_expert": True,
                        "best_raw": candidate_raw,
                        "selected_label": ckpt.name,
                        "output_path": str(out_path),
                        "report_path": str(rep_path),
                    })

        combined = [dict(item) for item in baseline_outputs]
        route_types = [t for t, info in best_by_type.items() if info["use_expert"]]
        for rec_type in route_types:
            selected_path = Path(best_by_type[rec_type]["output_path"])
            selected_outputs = json.loads(selected_path.read_text(encoding="utf-8"))
            by_id = {item.get("id"): item for item in selected_outputs}
            for idx, rec in enumerate(valid_records):
                if (rec.get("type") or "unknown") != rec_type:
                    continue
                rec_id = rec.get("id", idx)
                if rec_id in by_id:
                    combined[idx] = dict(by_id[rec_id])

        shutil.copyfile(baseline_output_path, GENERAL_VALID_OUTPUT_PATH)
        shutil.copyfile(baseline_report_path, GENERAL_VALID_REPORT_PATH)
        VALID_OUTPUT_PATH.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        combined_report = save_eval_report(VALID_OUTPUT_PATH, valid_records, VALID_REPORT_PATH)

        selection = {
            "enabled": True,
            "expert_types": EXPERT_TYPES,
            "baseline_output_path": str(GENERAL_VALID_OUTPUT_PATH),
            "baseline_report_path": str(GENERAL_VALID_REPORT_PATH),
            "route_types": route_types,
            "best_by_type": best_by_type,
            "entries": entries,
            "combined_summary": combined_report["summary"],
            "combined_by_type": combined_report["by_type"],
        }
        EXPERT_SELECTION_REPORT_PATH.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[expert:valid:combined]", combined_report["summary"], "route_types=", route_types)
        return selection

    def ensure_general_valid_outputs():
        if VALID_OUTPUT_PATH.exists() and VALID_REPORT_PATH.exists():
            return
        print("[expert] building general valid outputs for expert selection")
        _ = generate_outputs(
            FINAL_OUTPUT_DIR,
            valid_records,
            VALID_OUTPUT_PATH,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            retrieval_index=RETRIEVAL_INDEX_FULL_TRAIN,
            model_output_path=MODEL_VALID_OUTPUT_PATH,
            decision_report_path=HYBRID_DECISION_REPORT_PATH,
        )
        valid_rep = save_eval_report(VALID_OUTPUT_PATH, valid_records, VALID_REPORT_PATH)
        if USE_HYBRID_RETRIEVAL and MODEL_VALID_OUTPUT_PATH.exists():
            model_valid_rep = save_eval_report(MODEL_VALID_OUTPUT_PATH, valid_records, MODEL_VALID_REPORT_PATH)
            print("[expert:general_valid:model_only]", model_valid_rep["summary"])
        print("[expert:general_valid]", valid_rep["summary"])

    def apply_expert_to_test(selection: dict):
        if not TEST_FILE.exists() or not TEST_OUTPUT_PATH.exists():
            return
        test_records_local = load_records(TEST_FILE)
        baseline_test = json.loads(TEST_OUTPUT_PATH.read_text(encoding="utf-8"))
        combined = [dict(item) for item in baseline_test]
        route_types = selection.get("route_types", [])
        for rec_type in route_types:
            selected_label = selection["best_by_type"][rec_type]["selected_label"]
            ckpt = next((p for p in list_expert_checkpoints() if p.name == selected_label), None)
            if ckpt is None:
                print("[expert:test] missing checkpoint for", rec_type, selected_label)
                continue
            subset, _indices = _expert_records_with_original_ids(test_records_local, {rec_type})
            out_path = EXPERT_EVAL_DIR / f"expert_test_output_{rec_type}_{selected_label}.json"
            expert_outputs = generate_model_outputs(
                ckpt,
                subset,
                out_path,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
            )
            by_id = {item.get("id"): item for item in expert_outputs}
            for idx, rec in enumerate(test_records_local):
                if (rec.get("type") or "unknown") != rec_type:
                    continue
                rec_id = rec.get("id", idx)
                if rec_id in by_id:
                    combined[idx] = dict(by_id[rec_id])
        shutil.copyfile(TEST_OUTPUT_PATH, GENERAL_TEST_OUTPUT_PATH)
        TEST_OUTPUT_PATH.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[expert:phase2] wrote combined", TEST_OUTPUT_PATH)

    expert_train_records = [rec for rec in train_clean if (rec.get("type") or "unknown") in set(EXPERT_TYPES)]
    expert_train_stage = build_answer_only_records(expert_train_records, EXPERT_STAGE_NAME)
    _expert_train_info = train_expert_lora_stage(expert_train_stage)

    if RUN_MODE == "phase1":
        ensure_general_valid_outputs()
        _expert_selection = select_expert_by_valid_type(VALID_OUTPUT_PATH, VALID_REPORT_PATH)
    elif RUN_MODE == "phase2":
        ensure_general_valid_outputs()
        _expert_selection = select_expert_by_valid_type(VALID_OUTPUT_PATH, VALID_REPORT_PATH)
        apply_expert_to_test(_expert_selection)
'''


def patch_common_config(src: str, *, version: str, output_stem: str) -> str:
    replacements = {
        "v14_5_gate_sweep_full_train_valid_select": version,
        "v13_type_gated_hybrid_full_retrain": version,
        "gpt2_math_lora_v145_gate_sweep_valid_select_answer_only": f"gpt2_math_lora_{output_stem}_answer_only",
        "gpt2_math_lora_v145_gate_sweep_valid_select_sft": f"gpt2_math_lora_{output_stem}_sft",
        "gpt2_math_lora_v145_gate_sweep_valid_select_final": f"gpt2_math_lora_{output_stem}_final",
        "gpt2_math_lora_v145_gate_sweep_valid_select_checkpoints": f"gpt2_math_lora_{output_stem}_checkpoints",
        "v145_gate_sweep_valid_select_checkpoint_eval_valid_json": f"{output_stem}_checkpoint_eval_valid_json",
        "gpt2_math_lora_v145_gate_sweep_valid_select_unused_final_retrain": f"gpt2_math_lora_{output_stem}_unused_final_retrain",
        "gpt2_math_lora_v13_answer_only": f"gpt2_math_lora_{output_stem}_answer_only",
        "gpt2_math_lora_v13_sft": f"gpt2_math_lora_{output_stem}_sft",
        "gpt2_math_lora_v13_final": f"gpt2_math_lora_{output_stem}_final",
        "gpt2_math_lora_v13_checkpoints": f"gpt2_math_lora_{output_stem}_checkpoints",
        "v13_checkpoint_eval_overlap_valid": f"{output_stem}_checkpoint_eval_overlap_valid",
        "gpt2_math_lora_v13_full_train_final": f"gpt2_math_lora_{output_stem}_full_train_final",
    }
    for old, new in replacements.items():
        src = src.replace(old, new)
    return src


def make_ensemble_notebook() -> dict:
    nb = json.loads(ENSEMBLE_BASE.read_text(encoding="utf-8"))
    nb = copy.deepcopy(nb)
    nb["cells"][0]["source"] = lines(
        "# V15 - Ensemble/Ranker Answer-Only\n\n"
        "Train the answer-only LoRA pipeline, collect model and retrieval candidates from the top epoch checkpoints, then train a lightweight ranker on valid.json to choose among candidate numeric answers."
    )
    config = patch_common_config("".join(nb["cells"][2]["source"]), version="v15_ensemble_ranker", output_stem="v15_ensemble_ranker")
    config = replace_once(
        config,
        'SELECTED_RETRIEVAL_GATE_CONFIG_PATH = WORKING_DIR / "selected_retrieval_gate_config.json"\n',
        'SELECTED_RETRIEVAL_GATE_CONFIG_PATH = WORKING_DIR / "selected_retrieval_gate_config.json"\n'
        'ENSEMBLE_CANDIDATE_DIR = WORKING_DIR / "ensemble_candidates"\n'
        'ENSEMBLE_RANKER_REPORT_PATH = WORKING_DIR / "ensemble_ranker_report.json"\n'
        'SELECTED_VALID_OUTPUT_PATH = WORKING_DIR / "selected_valid_output.json"\n'
        'SELECTED_VALID_REPORT_PATH = WORKING_DIR / "selected_valid_report.json"\n'
        'SELECTED_MODEL_VALID_OUTPUT_PATH = WORKING_DIR / "selected_model_valid_output.json"\n',
    )
    config = replace_once(
        config,
        'RETRIEVAL_GATE_SWEEP_GRID = {\n',
        'ENSEMBLE_RANKER_ENABLED = True\n'
        'ENSEMBLE_TOP_K_CHECKPOINTS = 8\n'
        'ENSEMBLE_PHASE2_TOP_K_CHECKPOINTS = 8\n'
        'ENSEMBLE_INCLUDE_MODEL_CANDIDATES = True\n'
        'ENSEMBLE_INCLUDE_RETRIEVAL_CANDIDATES = True\n'
        'ENSEMBLE_RANKER_TREES = 300\n'
        'ENSEMBLE_RANKER_MAX_DEPTH = 8\n'
        'ENSEMBLE_RANKER_MIN_SAMPLES_LEAF = 3\n'
        '\n'
        'RETRIEVAL_GATE_SWEEP_GRID = {\n',
    )
    nb["cells"][2]["source"] = lines(config)
    manifest = "".join(nb["cells"][9]["source"])
    manifest = manifest.replace('"notebook_version": "v14_5_gate_sweep_full_train_valid_select"', '"notebook_version": "v15_ensemble_ranker"')
    manifest = manifest.replace('manifest_path = WORKING_DIR / "v14_5_gate_sweep_full_train_valid_select_manifest.json"', 'manifest_path = WORKING_DIR / "v15_ensemble_ranker_manifest.json"')
    manifest = replace_once(
        manifest,
        '        "selected_retrieval_gate_config": str(SELECTED_RETRIEVAL_GATE_CONFIG_PATH),\n',
        '        "selected_retrieval_gate_config": str(SELECTED_RETRIEVAL_GATE_CONFIG_PATH),\n'
        '        "ensemble_ranker_report": str(ENSEMBLE_RANKER_REPORT_PATH),\n'
        '        "ensemble_candidate_dir": str(ENSEMBLE_CANDIDATE_DIR),\n',
    )
    nb["cells"][9]["source"] = lines(manifest)
    nb["cells"].insert(9, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(ENSEMBLE_CELL)})
    return nb


def make_expert_notebook() -> dict:
    nb = json.loads(EXPERT_BASE.read_text(encoding="utf-8"))
    nb = copy.deepcopy(nb)
    nb["cells"][0]["source"] = lines(
        "# V15 - Type-Specific SV/FOBAR Expert\n\n"
        "Keep the strong V13 answer-only hybrid baseline, train a separate answer-only LoRA expert for SV/FOBAR types, and route only the expert types that beat the baseline on valid.json."
    )
    config = patch_common_config("".join(nb["cells"][2]["source"]), version="v15_type_expert_sv_fobar", output_stem="v15_type_expert_sv_fobar")
    config = replace_once(
        config,
        'FINAL_RETRAIN_INFO_PATH = WORKING_DIR / "final_retrain_info.json"\n',
        'FINAL_RETRAIN_INFO_PATH = WORKING_DIR / "final_retrain_info.json"\n'
        'EXPERT_ENABLED = True\n'
        'EXPERT_TYPES = ["GSM_FOBAR", "MATH_FOBAR", "GSM_SV", "MATH_SV"]\n'
        'EXPERT_STAGE_NAME = "sv_fobar_answer_only_expert"\n'
        'EXPERT_EPOCHS = 8.0\n'
        'EXPERT_LR = 8e-4\n'
        'EXPERT_OUTPUT_DIR = WORKING_DIR / "gpt2_math_lora_v15_sv_fobar_expert_final"\n'
        'EXPERT_CHECKPOINT_ROOT_DIR = WORKING_DIR / "gpt2_math_lora_v15_sv_fobar_expert_checkpoints"\n'
        'EXPERT_EVAL_DIR = WORKING_DIR / "v15_sv_fobar_expert_eval"\n'
        'EXPERT_TRAIN_INFO_PATH = WORKING_DIR / "expert_train_info.json"\n'
        'EXPERT_SELECTION_REPORT_PATH = WORKING_DIR / "expert_selection_report.json"\n'
        'GENERAL_VALID_OUTPUT_PATH = WORKING_DIR / "general_valid_output.json"\n'
        'GENERAL_VALID_REPORT_PATH = WORKING_DIR / "general_valid_report.json"\n'
        'GENERAL_TEST_OUTPUT_PATH = WORKING_DIR / "general_test_predictions.json"\n',
    )
    nb["cells"][2]["source"] = lines(config)
    manifest = "".join(nb["cells"][9]["source"])
    manifest = manifest.replace('"notebook_version": "v13_type_gated_hybrid_full_retrain"', '"notebook_version": "v15_type_expert_sv_fobar"')
    manifest = manifest.replace('manifest_path = WORKING_DIR / "v13_type_gated_hybrid_full_retrain_manifest.json"', 'manifest_path = WORKING_DIR / "v15_type_expert_sv_fobar_manifest.json"')
    manifest = replace_once(
        manifest,
        '        "final_retrain_info": str(FINAL_RETRAIN_INFO_PATH),\n',
        '        "final_retrain_info": str(FINAL_RETRAIN_INFO_PATH),\n'
        '        "expert_train_info": str(EXPERT_TRAIN_INFO_PATH),\n'
        '        "expert_selection_report": str(EXPERT_SELECTION_REPORT_PATH),\n'
        '        "general_valid_output": str(GENERAL_VALID_OUTPUT_PATH),\n'
        '        "general_valid_report": str(GENERAL_VALID_REPORT_PATH),\n',
    )
    nb["cells"][9]["source"] = lines(manifest)
    nb["cells"].insert(9, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(EXPERT_CELL)})
    return nb


def main() -> None:
    outputs = {
        "finetune_gpt2_for_math_v15_ensemble_ranker.ipynb": make_ensemble_notebook(),
        "finetune_gpt2_for_math_v15_type_expert_sv_fobar.ipynb": make_expert_notebook(),
    }
    for path, nb in outputs.items():
        Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
