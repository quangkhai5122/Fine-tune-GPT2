"""Create v17.5 and v20.5 select-7/8 notebooks at lr=3e-3."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V17_SCRIPT = ROOT / "audit" / "create_v17_answer_only_notebooks.py"
V20_SCRIPT = ROOT / "audit" / "create_v20_v17_lr3e3_strict_verifier_overlay_notebook.py"

V17_5_NOTEBOOK = ROOT / "finetune_gpt2_for_math_v17_5_legal_answer_only_select_78_lr3e3.ipynb"
V20_5_NOTEBOOK = ROOT / "finetune_gpt2_for_math_v20_5_v17_lr3e3_strict_verifier_overlay_select_78.ipynb"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_v17_5_source(source: str) -> str:
    replacements = {
        "v17_legal_answer_only_select_678": "v17_5_legal_answer_only_select_78_lr3e3",
        "# V17 - Legal Answer-Only Select 6/7/8": "# V17.5 - Legal Answer-Only Select 7/8 LR 3e-3",
        "across epochs 6, 7, and 8": "across epochs 7 and 8",
        "STAGE_A_LR = 0.001": "STAGE_A_LR = 0.003",
        "ENSEMBLE_LAST_K_EPOCHS = 3": "ENSEMBLE_LAST_K_EPOCHS = 2",
        "CHECKPOINT_EPOCH_LABELS_TO_SAVE = ['epoch_06', 'epoch_07', 'epoch_08']": (
            "CHECKPOINT_EPOCH_LABELS_TO_SAVE = ['epoch_07', 'epoch_08']"
        ),
        "legal_answer_only_valid_select_678": "legal_answer_only_lr3e3_valid_select_78",
        "none_answer_only_valid_select_678": "none_answer_only_lr3e3_valid_select_78",
        "num_model_candidates\": 3": "num_model_candidates\": 2",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    source = patch_numeric_fallback_and_diagnostics(source)
    return source


def patch_numeric_fallback_and_diagnostics(source: str) -> str:
    source = source.replace(
        '''BOXED_RE = re.compile(r"\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}")\n''',
        '''BOXED_RE = re.compile(r"\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}")\nFALLBACK_NUMBER_RE = re.compile(r"[-+]?\\d+(?:[.,]\\d+)?(?:\\s*/\\s*[-+]?\\d+(?:[.,]\\d+)?)?")\n''',
    )
    source = source.replace(
        '''    boxes = BOXED_RE.findall(text)\n    if boxes:\n        return _clean_tail(boxes[-1])\n    return None\n''',
        '''    boxes = BOXED_RE.findall(text)\n    if boxes:\n        return _clean_tail(boxes[-1])\n    # Some high-LR checkpoints may generate only a bare number without the\n    # "Đáp án là:" anchor. Treat the last generated numeric span as the answer\n    # so local validation does not collapse to 0 solely because of formatting.\n    number_matches = FALLBACK_NUMBER_RE.findall(str(text))\n    if number_matches:\n        return _clean_tail(number_matches[-1])\n    return None\n''',
    )
    source = source.replace(
        '''def generate_model_candidates(records: list[dict], split_name: str, entries: list[dict]) -> list[dict]:\n    candidates = []\n    for entry in entries:\n        out_path = _candidate_path(split_name, entry["name"])\n        if out_path.exists():\n            outputs = json.loads(out_path.read_text(encoding="utf-8"))\n            print("[candidates] reuse", out_path)\n        else:\n            outputs = generate_model_outputs(entry["adapter_dir"], records, out_path, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n        candidates.append({**entry, "outputs": outputs, "path": str(out_path)})\n    return candidates\n''',
        '''def _candidate_output_sanity(name: str, outputs: list[dict]) -> None:\n    digit_count = sum(bool(re.search(r"\\d", str(o.get("model_output", "")))) for o in outputs)\n    anchor_count = sum(bool(re.search(r"đáp\\s*án|dap\\s*an|answer", str(o.get("model_output", "")), re.IGNORECASE)) for o in outputs)\n    extractable_count = sum(extract_pred(o)[0] is not None for o in outputs)\n    samples = [str(o.get("model_output", "")).replace("\\n", " ")[:80] for o in outputs[:3]]\n    print("[candidate-sanity]", name, "rows=", len(outputs), "digit=", digit_count, "anchor=", anchor_count, "extractable=", extractable_count, "samples=", samples)\n\n\ndef generate_model_candidates(records: list[dict], split_name: str, entries: list[dict]) -> list[dict]:\n    candidates = []\n    for entry in entries:\n        out_path = _candidate_path(split_name, entry["name"])\n        if out_path.exists():\n            outputs = json.loads(out_path.read_text(encoding="utf-8"))\n            print("[candidates] reuse", out_path)\n        else:\n            outputs = generate_model_outputs(entry["adapter_dir"], records, out_path, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n        _candidate_output_sanity(entry["name"], outputs)\n        candidates.append({**entry, "outputs": outputs, "path": str(out_path)})\n    return candidates\n''',
    )
    return source


def clean_notebook(nb: dict) -> dict:
    nb.setdefault("metadata", {}).pop("widgets", None)
    nb.setdefault("metadata", {}).pop("papermill", None)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return nb


def build_v17_5_notebook() -> dict:
    v17 = load_module(V17_SCRIPT, "create_v17_answer_only_notebooks")
    nb = v17.build_notebook("select_678")
    for cell in nb.get("cells", []):
        if "source" not in cell:
            continue
        source = "".join(cell.get("source", []))
        cell["source"] = patch_v17_5_source(source).splitlines(keepends=True)
    return clean_notebook(nb)


V20_5_SELECT_VERIFIER_TAIL = r'''
def _bucket(summary: dict, score: int) -> int:
    buckets = summary.get("buckets", {})
    return int(buckets.get(score, buckets.get(str(score), 0)))

def verifier_checkpoint_selection_key(row: dict):
    summary = row["summary"]
    epoch = float(row.get("epoch") or 0.0)
    return (
        int(summary.get("raw_score", 0)),
        _bucket(summary, 10),
        int(summary.get("extractable", 0)),
        -abs(epoch - 7.0),
        epoch,
    )

def _safe_candidate_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))

def _verifier_output_path(split_name: str, candidate_name: str) -> Path:
    ENSEMBLE_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    return ENSEMBLE_CANDIDATE_DIR / ("%s_%s_strict_verifier.json" % (split_name, _safe_candidate_name(candidate_name)))

def _verifier_decision_path(split_name: str, candidate_name: str) -> Path:
    return WORKING_DIR / ("%s_candidate_verifier_decisions_%s.json" % (split_name, _safe_candidate_name(candidate_name)))

def run_verifier_for_candidate(records: list[dict], split_name: str, cand: dict, retriever: LegalTemplateRetriever) -> dict:
    outputs, decisions, summary = choose_candidate_verifier(records, cand["outputs"], Path(cand["adapter_dir"]), retriever)
    summary = dict(summary)
    summary.update({
        "strategy": "v20_5_lr3e3_strict_verifier_overlay_candidate",
        "model_candidate": cand["name"],
        "model_epoch": cand.get("epoch"),
        "model_adapter_dir": str(cand["adapter_dir"]),
        "model_output_path": cand.get("path"),
    })
    out_path = _verifier_output_path(split_name, cand["name"])
    decision_path = _verifier_decision_path(split_name, cand["name"])
    out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    decision_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    row = {
        "name": cand["name"],
        "kind": cand["kind"],
        "seed": cand.get("seed"),
        "epoch": cand.get("epoch"),
        "adapter_dir": str(cand["adapter_dir"]),
        "model_output_path": cand.get("path"),
        "verifier_output_path": str(out_path),
        "decision_path": str(decision_path),
        "verifier_summary": summary,
    }
    report = None
    if records and records[0].get("response_vi"):
        report = evaluate_predictions(outputs, records)
        row["summary"] = report["summary"]
        row["by_type"] = report["by_type"]
        eval_path = CHECKPOINT_EVAL_DIR / (_safe_candidate_name(cand["name"]) + "_strict_verifier_valid_report.json")
        eval_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        row["summary"] = summary
    return {
        "candidate": cand,
        "outputs": outputs,
        "decisions": decisions,
        "summary": summary,
        "report": report,
        "row": row,
        "output_path": str(out_path),
        "decision_path": str(decision_path),
    }

def select_best_valid_verifier_checkpoint(model_candidates: list[dict], records: list[dict], retriever: LegalTemplateRetriever) -> tuple[dict, dict]:
    CHECKPOINT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    best = None
    for cand in model_candidates:
        result = run_verifier_for_candidate(records, "valid", cand, retriever)
        row = result["row"]
        rows.append(row)
        key = verifier_checkpoint_selection_key(row)
        if best is None or key > best[0]:
            best = (key, result)
        print(
            "[select-verifier]",
            cand["name"],
            "raw=", row["summary"].get("raw_score"),
            "exact10=", _bucket(row["summary"], 10),
        )
    if best is None:
        raise RuntimeError("No verifier checkpoint candidates to select")
    selected = best[1]
    payload = {
        "strategy": "v20_5_lr3e3_strict_verifier_overlay_valid_select_78",
        "selection_metric": "max(raw_score, exact10, extractable, -abs(epoch-7), epoch)",
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "uses_valid_labels_for_checkpoint_selection": True,
        "uses_valid_labels_for_verifier_training": False,
        "num_model_candidates": len(model_candidates),
        "candidates": rows,
        "selected": selected["row"],
    }
    CHECKPOINT_SELECTION_REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SELECTED_CHECKPOINT_INFO_PATH.write_text(json.dumps(selected["row"], ensure_ascii=False, indent=2), encoding="utf-8")
    print("[select-verifier] selected", selected["candidate"]["name"], "raw=", selected["row"]["summary"].get("raw_score"))
    return selected, payload

def primary_candidate_for_reference(model_candidates: list[dict]) -> dict:
    return max(model_candidates, key=lambda c: (float(c.get("epoch") or 0.0), float(c.get("priority", 0.0))))

def run_select_verifier_valid(records: list[dict], split_name: str, output_path: Path, report_path: Path):
    entries = candidate_entries_from_runs()
    model_candidates = generate_model_candidates(records, split_name, entries)
    primary = primary_candidate_for_reference(model_candidates)
    if split_name == "valid":
        MODEL_VALID_OUTPUT_PATH.write_text(json.dumps(primary["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
        if records and records[0].get("response_vi"):
            save_eval_report(MODEL_VALID_OUTPUT_PATH, records, MODEL_VALID_REPORT_PATH)
    retriever = LegalTemplateRetriever(train_clean)
    selected, selection_payload = select_best_valid_verifier_checkpoint(model_candidates, records, retriever)
    summary = {
        "strategy": "v20_5_lr3e3_strict_verifier_overlay_valid_select_78",
        "selected_candidate": selected["candidate"]["name"],
        "selected_epoch": selected["candidate"].get("epoch"),
        "selected_adapter_dir": str(selected["candidate"]["adapter_dir"]),
        "selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),
        "candidate_summaries": [
            {
                "name": row["name"],
                "epoch": row["epoch"],
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
        "uses_valid_labels_for_verifier_training": False,
    }
    report, payload = save_outputs_and_optional_report(records, selected["outputs"], output_path, report_path, ENSEMBLE_RANKER_REPORT_PATH, summary)
    SELECTED_VALID_OUTPUT_PATH.write_text(json.dumps(selected["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    (WORKING_DIR / ("%s_candidate_verifier_decisions.json" % split_name)).write_text(json.dumps(selected["decisions"], ensure_ascii=False, indent=2), encoding="utf-8")
    if report is not None:
        SELECTED_VALID_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[select-verifier:%s]" % split_name, report["summary"])
    return model_candidates, selected, payload

def select_entry_for_test(entries: list[dict], retriever: LegalTemplateRetriever) -> dict:
    if VALID_FILE.exists() and valid_clean and valid_clean[0].get("response_vi"):
        valid_candidates = generate_model_candidates(valid_clean, "valid", entries)
        selected, _payload = select_best_valid_verifier_checkpoint(valid_candidates, valid_clean, retriever)
        SELECTED_VALID_OUTPUT_PATH.write_text(json.dumps(selected["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
        SELECTED_VALID_REPORT_PATH.write_text(json.dumps(selected["report"], ensure_ascii=False, indent=2), encoding="utf-8")
        (WORKING_DIR / "valid_candidate_verifier_decisions.json").write_text(json.dumps(selected["decisions"], ensure_ascii=False, indent=2), encoding="utf-8")
        return selected["candidate"]
    epoch7 = next((entry for entry in entries if entry["name"].endswith("::epoch_07")), None)
    fallback = epoch7 or max(entries, key=lambda e: float(e.get("epoch") or 0.0))
    info = {
        "strategy": "v20_5_lr3e3_strict_verifier_overlay_valid_select_78",
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

def run_select_verifier_test():
    entries = candidate_entries_from_runs()
    retriever = LegalTemplateRetriever(train_clean)
    selected_entry = select_entry_for_test(entries, retriever)
    test_records = load_records(TEST_FILE)
    model_outputs = generate_model_outputs(Path(selected_entry["adapter_dir"]), test_records, MODEL_TEST_OUTPUT_PATH, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS)
    test_candidate = {**selected_entry, "outputs": model_outputs, "path": str(MODEL_TEST_OUTPUT_PATH)}
    outputs, decisions, summary = choose_candidate_verifier(test_records, model_outputs, Path(selected_entry["adapter_dir"]), retriever)
    summary = dict(summary)
    summary.update({
        "strategy": "v20_5_lr3e3_strict_verifier_overlay_valid_select_78",
        "selected_candidate": selected_entry["name"],
        "selected_epoch": selected_entry.get("epoch"),
        "selected_adapter_dir": str(selected_entry["adapter_dir"]),
        "model_test_output": str(MODEL_TEST_OUTPUT_PATH),
        "selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "uses_valid_labels_for_checkpoint_selection": bool(VALID_FILE.exists() and valid_clean and valid_clean[0].get("response_vi")),
        "uses_valid_labels_for_verifier_training": False,
    })
    _report, payload = save_outputs_and_optional_report(test_records, outputs, TEST_OUTPUT_PATH, VALID_REPORT_PATH, ENSEMBLE_RANKER_REPORT_PATH, summary)
    (WORKING_DIR / "test_candidate_verifier_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[phase2] selected", selected_entry["name"], "wrote", TEST_OUTPUT_PATH)
    return test_candidate, outputs, payload

if RUN_MODE == "phase1":
    _candidates, _selected, _payload = run_select_verifier_valid(valid_clean, "valid", VALID_OUTPUT_PATH, VALID_REPORT_PATH)
elif RUN_MODE == "phase2":
    if not TEST_FILE.exists():
        raise FileNotFoundError("RUN_MODE='phase2' requires test.json")
    _selected_candidate, _outputs, _payload = run_select_verifier_test()
else:
    raise ValueError("Unknown RUN_MODE=" + str(RUN_MODE))
'''


def patch_v20_5_source(source: str) -> str:
    replacements = {
        "v20_v17_lr3e3_strict_verifier_overlay": "v20_5_v17_lr3e3_strict_verifier_overlay_select_78",
        "# V20 - V17 LR 3e-3 + Strict Verifier Overlay": (
            "# V20.5 - V17 LR 3e-3 Select 7/8 + Strict Verifier Overlay"
        ),
        "Single-seed answer-only SFT with the v17 epoch-7 lr=3e-3 setup, followed by strict-legal candidate verification.": (
            "Single-seed answer-only SFT with the v17 lr=3e-3 setup, saving epoch 7 and 8, "
            "then selecting the best strict-legal verifier overlay on valid.json."
        ),
        "STAGE_A_EPOCHS = 7.0": "STAGE_A_EPOCHS = 8.0",
        "SAVE_EPOCH_CHECKPOINTS = False\nCHECKPOINT_EPOCH_LABELS_TO_SAVE = None": (
            "SAVE_EPOCH_CHECKPOINTS = True\nCHECKPOINT_EPOCH_LABELS_TO_SAVE = ['epoch_07', 'epoch_08']"
        ),
        "ENSEMBLE_INCLUDE_EPOCH_CHECKPOINTS = False": "ENSEMBLE_INCLUDE_EPOCH_CHECKPOINTS = True",
        "ENSEMBLE_LAST_K_EPOCHS = 0": "ENSEMBLE_LAST_K_EPOCHS = 2",
        "ENSEMBLE_INCLUDE_FINAL = True": "ENSEMBLE_INCLUDE_FINAL = False",
        "v17_lr3e3_epoch7_plus_train_query_vi_template_retrieval_candidates": (
            "v17_lr3e3_select78_plus_train_query_vi_template_retrieval_candidates"
        ),
        '"uses_valid_labels_for_checkpoint_selection": False,': '"uses_valid_labels_for_checkpoint_selection": True,',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    source = patch_numeric_fallback_and_diagnostics(source)
    return source


def patch_v20_5_run_cell(source: str) -> str:
    marker = "def run_verifier_split(records: list[dict], split_name: str, output_path: Path, report_path: Path):"
    if marker not in source:
        raise RuntimeError("Cannot find v20 verifier run tail marker")
    prefix = source.split(marker, 1)[0]
    return prefix + V20_5_SELECT_VERIFIER_TAIL


def build_v20_5_notebook() -> dict:
    v20 = load_module(V20_SCRIPT, "create_v20_v17_lr3e3_strict_verifier_overlay_notebook")
    nb = v20.build_notebook()
    for idx, cell in enumerate(nb.get("cells", [])):
        if "source" not in cell:
            continue
        source = "".join(cell.get("source", []))
        source = patch_v20_5_source(source)
        if idx == 8:
            source = patch_v20_5_run_cell(source)
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
    notebooks = {
        V17_5_NOTEBOOK: build_v17_5_notebook(),
        V20_5_NOTEBOOK: build_v20_5_notebook(),
    }
    for path, nb in notebooks.items():
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        validate_notebook(path)
        print("[wrote]", path.name, "bytes=", path.stat().st_size)


if __name__ == "__main__":
    main()
