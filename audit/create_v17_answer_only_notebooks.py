"""Create legal answer-only v17 notebooks from the v16 notebook scaffold."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V16_SCRIPT = ROOT / "audit" / "create_v16_legal_notebooks.py"

EPOCH7_NOTEBOOK = ROOT / "finetune_gpt2_for_math_v17_legal_answer_only_epoch7.ipynb"
SELECT_678_NOTEBOOK = ROOT / "finetune_gpt2_for_math_v17_legal_answer_only_select_678.ipynb"


def load_v16_module():
    spec = importlib.util.spec_from_file_location("create_v16_legal_notebooks", V16_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {V16_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v16 = load_v16_module()


SELECT_678_RUN = r'''
# ============================================================
# 8. Legal answer-only checkpoint selection on valid.json
# ============================================================
def _bucket(summary: dict, score: int) -> int:
    buckets = summary.get("buckets", {})
    return int(buckets.get(score, buckets.get(str(score), 0)))

def checkpoint_selection_key(row: dict):
    summary = row["summary"]
    epoch = float(row.get("epoch") or 0.0)
    # Primary metric is official raw_score. Tie-breaks keep exact hits high and
    # mildly prefer epoch 7, which was the strongest finished v16 checkpoint.
    return (
        int(summary.get("raw_score", 0)),
        _bucket(summary, 10),
        int(summary.get("extractable", 0)),
        -abs(epoch - 7.0),
        epoch,
    )

def select_best_valid_checkpoint(model_candidates: list[dict], records: list[dict]) -> tuple[dict, dict]:
    CHECKPOINT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    best = None
    for cand in model_candidates:
        report = evaluate_predictions(cand["outputs"], records)
        row = {
            "name": cand["name"],
            "kind": cand["kind"],
            "seed": cand.get("seed"),
            "epoch": cand.get("epoch"),
            "adapter_dir": str(cand["adapter_dir"]),
            "output_path": cand.get("path"),
            "summary": report["summary"],
            "by_type": report["by_type"],
        }
        rows.append(row)
        eval_path = CHECKPOINT_EVAL_DIR / (re.sub(r"[^A-Za-z0-9_.-]+", "_", cand["name"]) + "_valid_report.json")
        eval_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        key = checkpoint_selection_key(row)
        if best is None or key > best[0]:
            best = (key, cand, row)
        print("[select]", cand["name"], "raw=", report["summary"]["raw_score"], "exact10=", _bucket(report["summary"], 10))
    if best is None:
        raise RuntimeError("No model candidates to select")
    selected_cand = best[1]
    selected_row = best[2]
    payload = {
        "strategy": "legal_answer_only_valid_select_678",
        "selection_metric": "max(raw_score, exact10, extractable, -abs(epoch-7), epoch)",
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "num_model_candidates": len(model_candidates),
        "candidates": rows,
        "selected": selected_row,
    }
    CHECKPOINT_SELECTION_REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_CHECKPOINT_INFO_PATH.write_text(json.dumps(selected_row, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[select] selected", selected_cand["name"], "raw=", selected_row["summary"]["raw_score"])
    return selected_cand, payload

def primary_candidate_for_reference(model_candidates: list[dict]) -> dict:
    # With this notebook, final is intentionally excluded to avoid an extra
    # candidate generation pass. Epoch 8 is the closest primary reference.
    return max(model_candidates, key=lambda c: (float(c.get("epoch") or 0.0), float(c.get("priority", 0.0))))

def run_select_678_valid(records: list[dict], split_name: str, output_path: Path, report_path: Path):
    entries = candidate_entries_from_runs()
    candidates = generate_model_candidates(records, split_name, entries)
    primary = primary_candidate_for_reference(candidates)
    if split_name == "valid":
        MODEL_VALID_OUTPUT_PATH.write_text(json.dumps(primary["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
        if records and records[0].get("response_vi"):
            save_eval_report(MODEL_VALID_OUTPUT_PATH, records, MODEL_VALID_REPORT_PATH)
    selected, selection_payload = select_best_valid_checkpoint(candidates, records)
    report, payload = save_outputs_and_optional_report(
        records,
        selected["outputs"],
        output_path,
        report_path,
        ENSEMBLE_RANKER_REPORT_PATH,
        {
            "strategy": "legal_answer_only_valid_select_678",
            "selected_candidate": selected["name"],
            "selected_epoch": selected.get("epoch"),
            "selected_adapter_dir": str(selected["adapter_dir"]),
            "selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),
            "candidate_summaries": [
                {
                    "name": row["name"],
                    "epoch": row["epoch"],
                    "raw_score": row["summary"]["raw_score"],
                    "exact10": _bucket(row["summary"], 10),
                    "extractable": row["summary"]["extractable"],
                }
                for row in selection_payload["candidates"]
            ],
            "legal_input_fields": LEGAL_INPUT_FIELDS,
            "legal_target_fields": LEGAL_TARGET_FIELDS,
            "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        },
    )
    SELECTED_VALID_OUTPUT_PATH.write_text(json.dumps(selected["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    if report is not None:
        SELECTED_VALID_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[answer-only-select:%s]" % split_name, report["summary"])
    return candidates, selected, payload

def select_adapter_for_test() -> dict:
    entries = candidate_entries_from_runs()
    if VALID_FILE.exists() and valid_clean and valid_clean[0].get("response_vi"):
        candidates = generate_model_candidates(valid_clean, "valid", entries)
        selected, _payload = select_best_valid_checkpoint(candidates, valid_clean)
        return selected
    epoch7 = next((entry for entry in entries if entry["name"].endswith("::epoch_07")), None)
    fallback = epoch7 or max(entries, key=lambda e: float(e.get("epoch") or 0.0))
    info = {
        "strategy": "legal_answer_only_valid_select_678",
        "selection_metric": "fallback_epoch_07_when_valid_unavailable",
        "selected": {
            "name": fallback["name"],
            "kind": fallback["kind"],
            "seed": fallback.get("seed"),
            "epoch": fallback.get("epoch"),
            "adapter_dir": str(fallback["adapter_dir"]),
        },
    }
    CHECKPOINT_SELECTION_REPORT_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_CHECKPOINT_INFO_PATH.write_text(json.dumps(info["selected"], ensure_ascii=False, indent=2), encoding="utf-8")
    return fallback

if RUN_MODE == "phase1":
    _candidates, _selected, _payload = run_select_678_valid(valid_clean, "valid", VALID_OUTPUT_PATH, VALID_REPORT_PATH)
elif RUN_MODE == "phase2":
    if not TEST_FILE.exists():
        raise FileNotFoundError("RUN_MODE='phase2' requires test.json")
    selected = select_adapter_for_test()
    test_records = load_records(TEST_FILE)
    test_outputs = generate_model_outputs(Path(selected["adapter_dir"]), test_records, TEST_OUTPUT_PATH, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS)
    MODEL_TEST_OUTPUT_PATH.write_text(json.dumps(test_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "strategy": "legal_answer_only_valid_select_678",
        "selected_candidate": selected["name"],
        "selected_epoch": selected.get("epoch"),
        "selected_adapter_dir": str(selected["adapter_dir"]),
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
    }
    ENSEMBLE_RANKER_REPORT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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


def scrub_code_cell(cell: dict) -> dict:
    cell = copy.deepcopy(cell)
    cell["execution_count"] = None
    cell["outputs"] = []
    return cell


def patched_config(
    *,
    version: str,
    artifact_prefix: str,
    stage_a_epochs: float,
    ensemble_last_k_epochs: int,
    save_epoch_checkpoints: bool,
    checkpoint_epoch_labels_to_save: set[str] | None,
    include_epoch_checkpoints: bool,
    include_final: bool,
) -> str:
    source = v16.build_config_cell(
        version=version,
        artifact_prefix=artifact_prefix,
        train_seeds=[42],
        stage_a_epochs=stage_a_epochs,
        stage_a_lr=1e-3,
        ensemble_last_k_epochs=ensemble_last_k_epochs,
        use_calib=False,
        calib_fraction=0.10,
        calib_max_records=None,
        legal_query_retrieval_enabled=False,
    )
    labels_repr = "None" if checkpoint_epoch_labels_to_save is None else repr(sorted(checkpoint_epoch_labels_to_save))
    source = source.replace(
        "SAVE_EPOCH_CHECKPOINTS = True",
        "SAVE_EPOCH_CHECKPOINTS = %s\nCHECKPOINT_EPOCH_LABELS_TO_SAVE = %s"
        % (repr(save_epoch_checkpoints), labels_repr),
    )
    source = source.replace(
        "ENSEMBLE_INCLUDE_EPOCH_CHECKPOINTS = True",
        "ENSEMBLE_INCLUDE_EPOCH_CHECKPOINTS = %s" % repr(include_epoch_checkpoints),
    )
    source = source.replace(
        "ENSEMBLE_INCLUDE_FINAL = True",
        "ENSEMBLE_INCLUDE_FINAL = %s" % repr(include_final),
    )
    return source


def patched_training_cell() -> str:
    source = v16.TRAINING_CELL
    needle = '''        label = _checkpoint_label(float(state.epoch))
        if label in self._saved_labels:
            return control
'''
    replacement = '''        label = _checkpoint_label(float(state.epoch))
        allowed_labels = globals().get("CHECKPOINT_EPOCH_LABELS_TO_SAVE")
        if allowed_labels is not None and label not in set(allowed_labels):
            return control
        if label in self._saved_labels:
            return control
'''
    if needle not in source:
        raise RuntimeError("Cannot patch checkpoint label filter in TRAINING_CELL")
    return source.replace(needle, replacement)


def patched_manifest_cell(retrieval_basis: str, uses_valid_checkpoint_selection: bool) -> str:
    source = v16.build_manifest_cell(retrieval_basis)
    source = source.replace(
        '"uses_valid_labels_for_ranker_training": False,\n',
        '"uses_valid_labels_for_ranker_training": False,\n'
        '        "uses_valid_labels_for_checkpoint_selection": %s,\n'
        % repr(uses_valid_checkpoint_selection),
    )
    source = source.replace(
        '"ensemble_last_k_epochs": ENSEMBLE_LAST_K_EPOCHS,\n',
        '"ensemble_last_k_epochs": ENSEMBLE_LAST_K_EPOCHS,\n'
        '        "save_epoch_checkpoints": SAVE_EPOCH_CHECKPOINTS,\n'
        '        "checkpoint_epoch_labels_to_save": globals().get("CHECKPOINT_EPOCH_LABELS_TO_SAVE"),\n'
        '        "ensemble_include_epoch_checkpoints": ENSEMBLE_INCLUDE_EPOCH_CHECKPOINTS,\n'
        '        "ensemble_include_final": ENSEMBLE_INCLUDE_FINAL,\n',
    )
    source = source.replace(
        '"ensemble_or_gate_report": str(ENSEMBLE_RANKER_REPORT_PATH),\n',
        '"ensemble_or_gate_report": str(ENSEMBLE_RANKER_REPORT_PATH),\n'
        '        "checkpoint_selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),\n'
        '        "selected_checkpoint_info": str(SELECTED_CHECKPOINT_INFO_PATH),\n'
        '        "selected_valid_output": str(SELECTED_VALID_OUTPUT_PATH),\n'
        '        "selected_valid_report": str(SELECTED_VALID_REPORT_PATH),\n',
    )
    return source


def build_notebook(kind: str) -> dict:
    base_nb, base_cells = v16.read_base_cells()
    nb = copy.deepcopy(base_nb)
    import_cell = scrub_code_cell(base_cells[1])
    eval_cell = scrub_code_cell(base_cells[3])
    eval_cell["source"] = v16.scrub_eval_cell("".join(eval_cell["source"])).splitlines(keepends=True)
    dataset_cell = scrub_code_cell(base_cells[5])

    if kind == "epoch7":
        title = (
            "# V17 - Legal Answer-Only Epoch 7\n\n"
            "Single-seed runtime-safe answer-only run. Uses only `query_vi` as model input "
            "and `response_vi` as train/validation target. No retrieval, no `type` routing, "
            "no `original_*` fields."
        )
        config = patched_config(
            version="v17_legal_answer_only_epoch7",
            artifact_prefix="v17_legal_answer_only_epoch7",
            stage_a_epochs=7.0,
            ensemble_last_k_epochs=0,
            save_epoch_checkpoints=False,
            checkpoint_epoch_labels_to_save=None,
            include_epoch_checkpoints=False,
            include_final=True,
        )
        run_cell = v16.GENERATION_COMMON + "\n" + v16.ANSWER_ONLY_RUN
        manifest = patched_manifest_cell("none_answer_only_fixed_epoch7", False)
    elif kind == "select_678":
        title = (
            "# V17 - Legal Answer-Only Select 6/7/8\n\n"
            "Single-seed answer-only run with valid.json checkpoint selection across epochs 6, 7, and 8. "
            "Uses only `query_vi` as model input and `response_vi` as train/validation target. "
            "No retrieval, no `type` routing, no `original_*` fields."
        )
        config = patched_config(
            version="v17_legal_answer_only_select_678",
            artifact_prefix="v17_legal_answer_only_select_678",
            stage_a_epochs=8.0,
            ensemble_last_k_epochs=3,
            save_epoch_checkpoints=True,
            checkpoint_epoch_labels_to_save={"epoch_06", "epoch_07", "epoch_08"},
            include_epoch_checkpoints=True,
            include_final=False,
        )
        run_cell = v16.GENERATION_COMMON + "\n" + SELECT_678_RUN
        manifest = patched_manifest_cell("none_answer_only_valid_select_678", True)
    else:
        raise ValueError(kind)

    nb["cells"] = [
        markdown_cell(title),
        import_cell,
        code_cell(config),
        eval_cell,
        code_cell(v16.DATA_CELL),
        dataset_cell,
        code_cell(patched_training_cell()),
        code_cell(v16.PROMPT_HELPER_CELL),
        code_cell(run_cell),
        code_cell(manifest),
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
    notebooks = {
        EPOCH7_NOTEBOOK: build_notebook("epoch7"),
        SELECT_678_NOTEBOOK: build_notebook("select_678"),
    }
    for path, nb in notebooks.items():
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        validate_notebook(path)
        print("[wrote]", path.name, "bytes=", path.stat().st_size)


if __name__ == "__main__":
    main()
