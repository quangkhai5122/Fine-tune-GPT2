from __future__ import annotations

import copy
import json
from pathlib import Path


V13_BASE = Path("finetune_gpt2_for_math_v13_type_gated_full_retrain.ipynb")
V14_GATE_BASE = Path("finetune_gpt2_for_math_v14_retrieval_gate_sweep.ipynb")


VALID_SELECT_PHASE_BLOCK = r'''if RUN_MODE == "phase1":
    if not VALID_FILE.exists():
        raise FileNotFoundError(f"RUN_MODE='phase1' but missing {VALID_FILE}")
    if SELECT_CHECKPOINTS_ON_OVERLAP_VALID:
        CHECKPOINT_SELECTION = select_best_checkpoint_on_valid(valid_records)

    if (
        CHECKPOINT_SELECTION.get("status") == "selected"
        and CHECKPOINT_SELECTION.get("eval_n") == len(valid_records)
        and CHECKPOINT_EVAL_MAX_NEW_TOKENS == MAX_NEW_TOKENS
        and CHECKPOINT_EVAL_NUM_BEAMS == NUM_BEAMS
    ):
        selected = CHECKPOINT_SELECTION["selected"]
        shutil.copyfile(selected["output_path"], VALID_OUTPUT_PATH)
        if Path(selected["model_output_path"]).exists():
            shutil.copyfile(selected["model_output_path"], MODEL_VALID_OUTPUT_PATH)
        shutil.copyfile(selected["report_path"], VALID_REPORT_PATH)
        valid_rep = json.loads(VALID_REPORT_PATH.read_text(encoding="utf-8"))
        print("[valid_json:selected] reused checkpoint-eval output/report")
    else:
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
        print("[valid_json:model_only]", model_valid_rep["summary"])
    print("[valid_json:selected]", valid_rep["summary"])
    print("Reference valid.json Score /10:", valid_rep["summary"]["score_10"])

elif RUN_MODE == "phase2":
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"RUN_MODE='phase2' but missing {TEST_FILE}")
    if not VALID_FILE.exists():
        raise FileNotFoundError(f"RUN_MODE='phase2' requires {VALID_FILE} for checkpoint selection")
    if SELECT_CHECKPOINTS_ON_OVERLAP_VALID:
        CHECKPOINT_SELECTION = select_best_checkpoint_on_valid(valid_records)
    test_records = load_records(TEST_FILE)
    _ = generate_outputs(
        FINAL_OUTPUT_DIR,
        test_records,
        TEST_OUTPUT_PATH,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        retrieval_index=RETRIEVAL_INDEX_FULL_TRAIN,
        model_output_path=MODEL_TEST_OUTPUT_PATH,
        decision_report_path=HYBRID_DECISION_REPORT_PATH,
    )
    print("[phase2] wrote", TEST_OUTPUT_PATH)
else:
    raise ValueError(f"Unknown RUN_MODE={RUN_MODE}")
'''


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Missing expected text:\n{old[:500]}")
    return text.replace(old, new, 1)


def replace_phase_block(src: str) -> str:
    start = src.index('if RUN_MODE == "phase1":')
    return src[:start] + VALID_SELECT_PHASE_BLOCK + "\n"


def patch_config(src: str, *, version: str, output_stem: str, title_comment: str) -> str:
    src = src.replace("v13_type_gated_hybrid_full_retrain", version)
    src = src.replace("v14_retrieval_gate_sweep", version)
    replacements = {
        "gpt2_math_lora_v13_answer_only": f"gpt2_math_lora_{output_stem}_answer_only",
        "gpt2_math_lora_v13_sft": f"gpt2_math_lora_{output_stem}_sft",
        "gpt2_math_lora_v13_final": f"gpt2_math_lora_{output_stem}_final",
        "gpt2_math_lora_v13_checkpoints": f"gpt2_math_lora_{output_stem}_checkpoints",
        "v13_checkpoint_eval_overlap_valid": f"{output_stem}_checkpoint_eval_valid_json",
        "gpt2_math_lora_v13_full_train_final": f"gpt2_math_lora_{output_stem}_unused_final_retrain",
        "gpt2_math_lora_v14_gate_sweep_answer_only": f"gpt2_math_lora_{output_stem}_answer_only",
        "gpt2_math_lora_v14_gate_sweep_sft": f"gpt2_math_lora_{output_stem}_sft",
        "gpt2_math_lora_v14_gate_sweep_final": f"gpt2_math_lora_{output_stem}_final",
        "gpt2_math_lora_v14_gate_sweep_checkpoints": f"gpt2_math_lora_{output_stem}_checkpoints",
        "v14_gate_sweep_checkpoint_eval_overlap_valid": f"{output_stem}_checkpoint_eval_valid_json",
        "gpt2_math_lora_v14_gate_sweep_full_train_final": f"gpt2_math_lora_{output_stem}_unused_final_retrain",
    }
    for old, new in replacements.items():
        src = src.replace(old, new)
    src = src.replace(
        "# v13_type_gated_hybrid_full_retrain: answer-only optimization for MetaMathQA-style same-source hidden tests.",
        title_comment,
    )
    src = src.replace(
        "# v14_retrieval_gate_sweep: answer-only optimization for MetaMathQA-style same-source hidden tests.",
        title_comment,
    )
    src = src.replace(
        "# The primary checkpoint metric is source-overlap query-disjoint validation\n# built from train.json. valid.json is evaluated only after checkpoint selection.",
        "# The primary checkpoint metric is valid.json. Training uses all cleaned train rows;\n# overlap-valid is kept only for diagnostics and is not used for epoch selection.",
    )
    src = replace_once(
        src,
        'SELECT_CHECKPOINTS_ON_OVERLAP_VALID = True\n',
        'SELECT_CHECKPOINTS_ON_OVERLAP_VALID = True  # kept True so epoch adapters are saved; selection split is valid.json\n'
        'CHECKPOINT_SELECTION_SPLIT_NAME = "valid_json"\n',
    )
    src = src.replace("FINAL_RETRAIN_FULL_TRAIN = True", "FINAL_RETRAIN_FULL_TRAIN = False")
    src = src.replace(
        "# V13: choose epoch on overlap-valid, then train a fresh final adapter on all\n# cleaned train rows for the selected epoch count. This keeps valid.json out of\n# selection while letting the final model use all available training labels.",
        "# V14.5: train checkpoints directly on all cleaned train rows, then choose\n# the checkpoint on valid.json. No post-selection full-train retrain is needed.",
    )
    return src


def patch_generation_cell(src: str) -> str:
    src = src.replace("select_best_checkpoint_on_overlap_valid", "select_best_checkpoint_on_valid")
    src = src.replace("source_overlap_query_disjoint", "valid_json")
    src = src.replace("overlap-valid rows", "valid.json rows")
    src = src.replace("overlap_valid_output_", "valid_output_")
    src = src.replace("model_overlap_valid_output_", "model_valid_output_")
    src = src.replace("overlap_valid_report_", "valid_report_")
    src = src.replace("RETRIEVAL_INDEX_OVERLAP_TRAIN", "RETRIEVAL_INDEX_FULL_TRAIN")
    src = src.replace(
        'RETRIEVAL_INDEX_FULL_TRAIN = build_retrieval_index(overlap_train_clean)\n'
        'RETRIEVAL_INDEX_FULL_TRAIN = build_retrieval_index(train_clean)\n'
        'print("[retrieval] overlap-train source groups:", len(RETRIEVAL_INDEX_FULL_TRAIN))',
        'RETRIEVAL_INDEX_OVERLAP_TRAIN = build_retrieval_index(overlap_train_clean)\n'
        'RETRIEVAL_INDEX_FULL_TRAIN = build_retrieval_index(train_clean)\n'
        'print("[retrieval] overlap-train source groups:", len(RETRIEVAL_INDEX_OVERLAP_TRAIN))',
    )
    src = src.replace(
        '"selection_metric": "max(gated_hybrid_raw_score, exact10, extractable, tie_break)" if RETRIEVAL_ALLOWED_TYPES is not None else "max(raw_score, exact10, extractable, tie_break)"',
        '"selection_metric": "max(valid_json_hybrid_raw_score, exact10, extractable, tie_break)" if RETRIEVAL_ALLOWED_TYPES is not None else "max(valid_json_raw_score, exact10, extractable, tie_break)"',
    )
    src = src.replace(
        '"selection_metric": "max(hybrid_raw_score_after_type_gate_sweep, exact10, extractable, tie_break)" if RETRIEVAL_GATE_SWEEP_ENABLED else "max(gated_hybrid_raw_score, exact10, extractable, tie_break)"',
        '"selection_metric": "max(valid_json_hybrid_raw_score_after_type_gate_sweep, exact10, extractable, tie_break)" if RETRIEVAL_GATE_SWEEP_ENABLED else "max(valid_json_gated_hybrid_raw_score, exact10, extractable, tie_break)"',
    )
    src = replace_phase_block(src)
    return src


def patch_manifest(src: str, *, version: str, manifest_name: str) -> str:
    src = src.replace('"notebook_version": "v13_type_gated_hybrid_full_retrain"', f'"notebook_version": "{version}"')
    src = src.replace('"notebook_version": "v14_retrieval_gate_sweep"', f'"notebook_version": "{version}"')
    src = replace_once(
        src,
        '        "select_checkpoints_on_overlap_valid": SELECT_CHECKPOINTS_ON_OVERLAP_VALID,\n',
        '        "select_checkpoints_on_overlap_valid": SELECT_CHECKPOINTS_ON_OVERLAP_VALID,\n'
        '        "checkpoint_selection_split_name": CHECKPOINT_SELECTION_SPLIT_NAME,\n',
    )
    src = src.replace('manifest_path = WORKING_DIR / "v13_type_gated_hybrid_full_retrain_manifest.json"', f'manifest_path = WORKING_DIR / "{manifest_name}"')
    src = src.replace('manifest_path = WORKING_DIR / "v14_retrieval_gate_sweep_manifest.json"', f'manifest_path = WORKING_DIR / "{manifest_name}"')
    return src


def patch_notebook(base_path: Path, *, version: str, output_stem: str, title: str, description: str, out_path: Path) -> None:
    nb = json.loads(base_path.read_text(encoding="utf-8"))
    nb = copy.deepcopy(nb)
    nb["cells"][0]["source"] = lines(f"# {title}\n\n{description}\n")

    nb["cells"][2]["source"] = lines(
        patch_config(
            "".join(nb["cells"][2]["source"]),
            version=version,
            output_stem=output_stem,
            title_comment=f"# {version}: full-train answer-only LoRA with valid.json checkpoint selection.",
        )
    )

    data_cell = "".join(nb["cells"][4]["source"])
    data_cell = replace_once(
        data_cell,
        "train_stage_a = build_answer_only_records(overlap_train_clean, STAGE_A_NAME)\n",
        "train_stage_a = build_answer_only_records(train_clean, STAGE_A_NAME)\n",
    )
    data_cell = replace_once(
        data_cell,
        "overlap_train_clean, overlap_valid_clean, query_disjoint_split_report = split_source_overlap_query_disjoint(train_clean)\n"
        "overlap_valid_eval_records = build_balanced_subset(overlap_valid_clean, OVERLAP_VALID_MAX_EVAL_RECORDS, SEED + 17)\n"
        "QUERY_DISJOINT_SPLIT_PATH.write_text(json.dumps(query_disjoint_split_report | {\"overlap_valid_eval_n\": len(overlap_valid_eval_records)}, ensure_ascii=False, indent=2), encoding=\"utf-8\")\n",
        "# Valid-select runs intentionally skip the overlap-valid holdout.\n"
        "overlap_train_clean = train_clean\n"
        "overlap_valid_clean = []\n"
        "overlap_valid_eval_records = []\n"
        "query_disjoint_split_report = {\n"
        "    \"split_name\": \"skipped_valid_select_full_train\",\n"
        "    \"reason\": \"train on all cleaned train rows and select checkpoint on valid.json\",\n"
        "    \"total_records\": len(train_clean),\n"
        "    \"train_records\": len(train_clean),\n"
        "    \"overlap_valid_records\": 0,\n"
        "    \"overlap_valid_eval_n\": 0,\n"
        "}\n"
        "QUERY_DISJOINT_SPLIT_PATH.write_text(json.dumps(query_disjoint_split_report, ensure_ascii=False, indent=2), encoding=\"utf-8\")\n",
    )
    nb["cells"][4]["source"] = lines(data_cell)

    nb["cells"][8]["source"] = lines(patch_generation_cell("".join(nb["cells"][8]["source"])))
    nb["cells"][9]["source"] = lines(
        patch_manifest(
            "".join(nb["cells"][9]["source"]),
            version=version,
            manifest_name=f"{version}_manifest.json",
        )
    )

    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out_path)


def main() -> None:
    patch_notebook(
        V13_BASE,
        version="v14_5_v13_full_train_valid_select",
        output_stem="v145_v13_valid_select",
        title="V14.5 - V13 Gate Full Train Valid Select",
        description="Train answer-only LoRA on all cleaned train rows and select the epoch directly on valid.json using the V13 type-gated source-majority retrieval policy.",
        out_path=Path("finetune_gpt2_for_math_v14_5_v13_full_train_valid_select.ipynb"),
    )
    patch_notebook(
        V14_GATE_BASE,
        version="v14_5_gate_sweep_full_train_valid_select",
        output_stem="v145_gate_sweep_valid_select",
        title="V14.5 - Gate Sweep Full Train Valid Select",
        description="Train answer-only LoRA on all cleaned train rows and select the epoch directly on valid.json, with per-type retrieval gate sweep enabled for the V14 gate-sweep policy.",
        out_path=Path("finetune_gpt2_for_math_v14_5_gate_sweep_full_train_valid_select.ipynb"),
    )


if __name__ == "__main__":
    main()
