"""Create v17.5 safe epoch-7 lr=3e-3 answer-only notebook.

The failed select_78 run trained lr=3e-3 for 8 epochs and both epoch 7/8
checkpoints generated empty strings. This variant preserves the successful v17
epoch-7 schedule and adds inference sanity guards.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V17_SCRIPT = ROOT / "audit" / "create_v17_answer_only_notebooks.py"
NOTEBOOK_PATH = ROOT / "finetune_gpt2_for_math_v17_5_legal_answer_only_epoch7_lr3e3_safe.ipynb"
LEGACY_SELECT78_ALIAS = ROOT / "finetune_gpt2_for_math_v17_5_legal_answer_only_select_78_lr3e3.ipynb"


def load_v17_module():
    spec = importlib.util.spec_from_file_location("create_v17_answer_only_notebooks", V17_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {V17_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_numeric_fallback(source: str) -> str:
    source = source.replace(
        '''BOXED_RE = re.compile(r"\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}")\n''',
        '''BOXED_RE = re.compile(r"\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}")\nFALLBACK_NUMBER_RE = re.compile(r"[-+]?\\d+(?:[.,]\\d+)?(?:\\s*/\\s*[-+]?\\d+(?:[.,]\\d+)?)?")\n''',
    )
    source = source.replace(
        '''    boxes = BOXED_RE.findall(text)\n    if boxes:\n        return _clean_tail(boxes[-1])\n    return None\n''',
        '''    boxes = BOXED_RE.findall(text)\n    if boxes:\n        return _clean_tail(boxes[-1])\n    # Prediction fallback only: if a model emits a bare number, keep it\n    # extractable instead of turning a numeric answer into an unparseable output.\n    number_matches = FALLBACK_NUMBER_RE.findall(str(text))\n    if number_matches:\n        return _clean_tail(number_matches[-1])\n    return None\n''',
    )
    return source


def patch_generation_safety(source: str) -> str:
    source = source.replace(
        "MAX_NEW_TOKENS = 32\n",
        "MAX_NEW_TOKENS = 32\nMIN_NEW_TOKENS = 1\n",
    )
    source = source.replace(
        '''@torch.inference_mode()\ndef generate_model_outputs(adapter_dir: Path, records: list[dict], output_path: Path, *, max_new_tokens: int = MAX_NEW_TOKENS, num_beams: int = NUM_BEAMS):\n''',
        '''def output_generation_sanity(outputs: list[dict], output_path: Path) -> dict:\n    n = len(outputs)\n    texts = [str(row.get("model_output", "")) for row in outputs]\n    nonempty = sum(bool(t.strip()) for t in texts)\n    digit = sum(any(ch.isdigit() for ch in t) for t in texts)\n    anchors = ["\\\\u0111\\\\u00e1p", "dap", "answer"]\n    anchor = sum(any(marker in t.lower() for marker in anchors) for t in texts)\n    samples = [t.replace("\\\\n", " ")[:100] for t in texts[:5]]\n    info = {"n": n, "nonempty": nonempty, "digit": digit, "anchor": anchor, "samples": samples}\n    print("[output-sanity]", output_path, info)\n    if n >= 100 and nonempty == 0:\n        raise RuntimeError("All generated outputs are empty. This is EOS-collapse or a bad adapter load; do not submit this run.")\n    if n >= 100 and digit == 0:\n        raise RuntimeError("No generated output contains a digit. This run is not a valid answer-only submission.")\n    return info\n\n\n@torch.inference_mode()\ndef generate_model_outputs(adapter_dir: Path, records: list[dict], output_path: Path, *, max_new_tokens: int = MAX_NEW_TOKENS, num_beams: int = NUM_BEAMS):\n''',
    )
    source = source.replace(
        '''            max_new_tokens=eff_new,\n            pad_token_id=SAFE_EOS_ID,\n''',
        '''            max_new_tokens=eff_new,\n            min_new_tokens=min(int(globals().get("MIN_NEW_TOKENS", 0)), eff_new),\n            pad_token_id=SAFE_EOS_ID,\n''',
    )
    source = source.replace(
        '''    output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")\n    print("[infer] wrote", len(outputs), "rows ->", output_path)\n''',
        '''    output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")\n    output_generation_sanity(outputs, output_path)\n    print("[infer] wrote", len(outputs), "rows ->", output_path)\n''',
    )
    source = source.replace(
        '''        else:\n            outputs = generate_model_outputs(entry["adapter_dir"], records, out_path, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n        candidates.append({**entry, "outputs": outputs, "path": str(out_path)})\n''',
        '''        else:\n            outputs = generate_model_outputs(entry["adapter_dir"], records, out_path, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n        output_generation_sanity(outputs, out_path)\n        candidates.append({**entry, "outputs": outputs, "path": str(out_path)})\n''',
    )
    return source


def patch_source(source: str) -> str:
    replacements = {
        "# V17 - Legal Answer-Only Epoch 7": "# V17.5 - Legal Answer-Only Epoch 7 LR 3e-3 Safe",
        "Single-seed runtime-safe answer-only run.": (
            "Single-seed answer-only run using the successful 7-epoch lr=3e-3 schedule, "
            "with EOS-collapse sanity checks."
        ),
        "v17_legal_answer_only_epoch7": "v17_5_legal_answer_only_epoch7_lr3e3_safe",
        "STAGE_A_LR = 0.001": "STAGE_A_LR = 0.003",
        "none_answer_only_fixed_epoch7": "none_answer_only_lr3e3_safe_epoch7",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    source = patch_numeric_fallback(source)
    source = patch_generation_safety(source)
    return source


def clean_notebook(nb: dict) -> dict:
    nb.setdefault("metadata", {}).pop("widgets", None)
    nb.setdefault("metadata", {}).pop("papermill", None)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return nb


def build_notebook() -> dict:
    v17 = load_v17_module()
    nb = v17.build_notebook("epoch7")
    for cell in nb.get("cells", []):
        if "source" not in cell:
            continue
        source = "".join(cell.get("source", []))
        cell["source"] = patch_source(source).splitlines(keepends=True)
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
    for path in [NOTEBOOK_PATH, LEGACY_SELECT78_ALIAS]:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        validate_notebook(path)
        print("[wrote]", path.name, "bytes=", path.stat().st_size)


if __name__ == "__main__":
    main()
