"""Create v20: v17 lr=3e-3 epoch-7 answer-only + v19 strict verifier overlay."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V19_SCRIPT = ROOT / "audit" / "create_v19_v17_lr2e3_strict_verifier_overlay_notebook.py"
NOTEBOOK_PATH = ROOT / "finetune_gpt2_for_math_v20_v17_lr3e3_strict_verifier_overlay.ipynb"

BASE_VERSION = "v19_v17_lr2e3_strict_verifier_overlay"
NEW_VERSION = "v20_v17_lr3e3_strict_verifier_overlay"


def load_v19_module():
    spec = importlib.util.spec_from_file_location("create_v19_v17_lr2e3_strict_verifier_overlay_notebook", V19_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {V19_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_source(source: str) -> str:
    source = source.replace(BASE_VERSION, NEW_VERSION)
    source = source.replace(
        "# V19 - V17 LR 2e-3 + Strict Verifier Overlay",
        "# V20 - V17 LR 3e-3 + Strict Verifier Overlay",
    )
    source = source.replace(
        "Single-seed answer-only SFT with the v17 epoch-7 lr=2e-3 setup, followed by strict-legal candidate verification.",
        "Single-seed answer-only SFT with the v17 epoch-7 lr=3e-3 setup, followed by strict-legal candidate verification.",
    )
    source = source.replace("STAGE_A_LR = 0.002", "STAGE_A_LR = 0.003")
    source = source.replace(
        "v17_lr2e3_epoch7_plus_train_query_vi_template_retrieval_candidates",
        "v17_lr3e3_epoch7_plus_train_query_vi_template_retrieval_candidates",
    )
    source = source.replace(
        '"strategy": "v17_lr2e3_strict_verifier_overlay"',
        '"strategy": "v17_lr3e3_strict_verifier_overlay"',
    )
    source = source.replace(
        "Base SFT is v17-style answer-only epoch 7 at lr=2e-3.",
        "Base SFT is v17-style answer-only epoch 7 at lr=3e-3.",
    )
    return source


def build_notebook() -> dict:
    v19 = load_v19_module()
    nb = v19.build_notebook()
    for cell in nb.get("cells", []):
        if "source" not in cell:
            continue
        source = "".join(cell.get("source", []))
        cell["source"] = patch_source(source).splitlines(keepends=True)
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
