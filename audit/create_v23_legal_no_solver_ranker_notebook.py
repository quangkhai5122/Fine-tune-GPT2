"""Create v23 legal no-solver ensemble/ranker notebook.

The generated notebook is intentionally conservative:
- model input uses query_vi only
- train target / validation gold use response_vi only
- no original_question_* fields
- no type-based routing
- no arithmetic/template solver
- optional retrieval only copies answers from train response_vi keyed by train query_vi
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V17_SCRIPT = ROOT / "audit" / "create_v17_answer_only_notebooks.py"
NOTEBOOK_PATH = ROOT / "finetune_gpt2_for_math_v23_legal_no_solver_ranker.ipynb"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V23_RUN_CELL = r'''
# ============================================================
# 8. V23 legal no-solver decoder + train-query retrieval ranker
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
    n = len(texts)
    return {
        "n": n,
        "nonempty": sum(bool(t.strip()) for t in texts),
        "digit": sum(bool(re.search(r"\d", t)) for t in texts),
        "anchor": sum(bool(re.search(r"Ä‘Ã¡p\s*Ã¡n|dap\s*an|answer|####", t, re.IGNORECASE)) for t in texts),
        "extractable": sum(extract_pred(o)[0] is not None for o in outputs),
        "samples": [t.replace("\n", " ")[:100] for t in texts[:5]],
    }

def generate_outputs_for_profile(adapter_dir: Path, records: list[dict], split_name: str, profile: dict) -> list[dict]:
    out_path = ENSEMBLE_CANDIDATE_DIR / ("%s_%s.json" % (split_name, _safe_name(profile["name"])))
    ENSEMBLE_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        outputs = json.loads(out_path.read_text(encoding="utf-8"))
        print("[decode] reuse", out_path)
        return outputs
    return generate_model_outputs(
        adapter_dir,
        records,
        out_path,
        max_new_tokens=int(profile.get("max_new_tokens", MAX_NEW_TOKENS)),
        num_beams=int(profile.get("num_beams", NUM_BEAMS)),
    )

class StrictQueryRetriever:
    """Legal train-query retriever.

    This is not a solver: it never computes an answer from test numbers. It only
    returns response_vi answers attached to highly similar train query_vi rows.
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
            print("[retrieval] sklearn unavailable; using Jaccard fallback:", repr(exc))
        print("[retrieval] backend=", self.backend, "records=", len(self.records))

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
                "jaccard": jaccard(qtok, rec.get("_query_tokens") or set()),
                "train_query_vi": rec.get("query_vi", ""),
            })
        return hits

def strict_retrieval_decision(rec: dict, retriever: StrictQueryRetriever, gate: dict) -> dict:
    hits = retriever.search(rec.get("query_vi", ""), int(gate.get("top_k", RETRIEVAL_TOP_K)))
    if not hits:
        return {"used": False, "reason": "no_hits", "hits": 0}
    groups = defaultdict(list)
    for h in hits:
        groups[h["answer_key"]].append(h)
    ranked = []
    for key, vals in groups.items():
        ranked.append({
            "answer_key": key,
            "count": len(vals),
            "top_sim": max(v["similarity"] for v in vals),
            "top_jaccard": max(v["jaccard"] for v in vals),
            "canonical_answer": vals[0]["canonical_answer"],
        })
    ranked.sort(
        key=lambda x: (x["count"], x["top_sim"], x["top_jaccard"], ANSWER_PRIORS.get(x["answer_key"], 0)),
        reverse=True,
    )
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    majority_frac = top["count"] / max(1, len(hits))
    margin = (top["count"] - (second["count"] if second else 0)) / max(1, len(hits))
    passed = (
        top["top_sim"] >= float(gate.get("min_sim", 1.0))
        and top["top_jaccard"] >= float(gate.get("min_jaccard", 1.0))
        and majority_frac >= float(gate.get("min_majority_frac", 1.0))
        and margin >= float(gate.get("min_margin", 1.0))
    )
    return {
        "used": bool(passed),
        "reason": "passed" if passed else "low_confidence",
        "answer_key": top["answer_key"],
        "hits": len(hits),
        "majority_frac": majority_frac,
        "margin": margin,
        "top_sim": top["top_sim"],
        "top_jaccard": top["top_jaccard"],
        "gate": dict(gate),
        "ranked": ranked[:3],
    }

def apply_no_solver_profile(
    records: list[dict],
    model_outputs: list[dict],
    retriever: StrictQueryRetriever,
    profile: dict,
) -> tuple[list[dict], list[dict], dict]:
    retrieval_gate = profile.get("retrieval_gate") or {"enabled": False}
    retrieval_enabled = bool(retrieval_gate.get("enabled", False))
    outputs = []
    decisions = []
    source_counts = Counter()
    reason_counts = Counter()
    changed = 0
    for idx, rec in enumerate(records):
        model_item = model_outputs[idx]
        model_key = answer_key_from_output(model_item)
        chosen_key = model_key
        source = "model"
        ret_dec = {"used": False, "reason": "disabled"}
        if retrieval_enabled:
            ret_dec = strict_retrieval_decision(rec, retriever, retrieval_gate)
            if ret_dec.get("used"):
                ret_key = ret_dec.get("answer_key")
                if ret_key == model_key:
                    chosen_key = ret_key
                    source = "retrieval_model_agree"
                elif model_key is None and bool(retrieval_gate.get("allow_when_model_missing", True)):
                    chosen_key = ret_key
                    source = "retrieval_model_missing"
                elif bool(retrieval_gate.get("allow_override", False)):
                    chosen_key = ret_key
                    source = "retrieval_override"
                else:
                    source = "model_fallback_disagree"
        changed += int(chosen_key != model_key)
        source_counts[source] += 1
        reason_counts[ret_dec.get("reason", "unknown")] += 1
        outputs.append(make_answer_item(rec, idx, chosen_key, model_item))
        decisions.append({
            "id": rec.get("id", idx),
            "model_key": model_key,
            "chosen_key": chosen_key,
            "source": source,
            "retrieval": {k: v for k, v in ret_dec.items() if k != "ranked"},
        })
    summary = {
        "strategy": "v23_legal_no_solver_decoder_retrieval_ranker_profile",
        "profile": profile,
        "num_rows": len(records),
        "changed_from_model": changed,
        "source_counts": dict(source_counts.most_common()),
        "retrieval_reason_counts": dict(reason_counts.most_common()),
        "retrieval_backend": retriever.backend,
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "uses_original_question_fields": False,
        "uses_type_for_prompt_or_routing": False,
        "uses_arithmetic_or_template_solver": False,
        "uses_valid_labels_for_ranker_training": False,
        "notes": "No solver. Retrieval copies answers only from train response_vi of similar train query_vi rows, and type is copied only for output/reporting.",
    }
    return outputs, decisions, summary

def evaluate_profile_on_valid(
    records: list[dict],
    adapter_dir: Path,
    retriever: StrictQueryRetriever,
    profile: dict,
) -> dict:
    model_outputs = generate_outputs_for_profile(adapter_dir, records, "valid", profile)
    sanity = output_sanity(model_outputs)
    outputs, decisions, summary = apply_no_solver_profile(records, model_outputs, retriever, profile)
    report = evaluate_predictions(outputs, records)
    row = {
        "profile": profile,
        "summary": report["summary"],
        "by_type": report["by_type"],
        "sanity": sanity,
        "profile_summary": summary,
        "decisions_sample": decisions[:30],
    }
    out_path = CHECKPOINT_EVAL_DIR / ("valid_%s_report.json" % _safe_name(profile["name"]))
    CHECKPOINT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[v23-profile]",
        profile["name"],
        "raw=", report["summary"]["raw_score"],
        "exact10=", _bucket(report["summary"], 10),
        "changed=", summary["changed_from_model"],
        "sanity=", sanity,
    )
    return {
        "profile": profile,
        "model_outputs": model_outputs,
        "outputs": outputs,
        "decisions": decisions,
        "summary": summary,
        "report": report,
        "row": row,
    }

def select_v23_profile_on_valid(adapter_dir: Path, retriever: StrictQueryRetriever) -> dict:
    rows = []
    best = None
    for profile in V23_PROFILES:
        result = evaluate_profile_on_valid(valid_clean, adapter_dir, retriever, profile)
        rows.append(result["row"])
        tie_break = -float(profile.get("runtime_cost", 1.0))
        key = _score_summary_key(result["report"]["summary"], tie_break=tie_break)
        if best is None or key > best[0]:
            best = (key, result)
    if best is None:
        raise RuntimeError("No v23 profiles were evaluated.")
    selected = best[1]
    payload = {
        "strategy": "v23_legal_no_solver_valid_profile_selection",
        "selection_metric": "max(raw_score, exact10, extractable, -bucket0, -runtime_cost)",
        "num_profiles": len(V23_PROFILES),
        "profiles": rows,
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
    (WORKING_DIR / "valid_v23_ranker_decisions.json").write_text(json.dumps(selected["decisions"], ensure_ascii=False, indent=2), encoding="utf-8")
    print("[v23-select] selected", selected["profile"]["name"], selected["report"]["summary"])
    return selected

def train_and_get_adapter() -> Path:
    entries = candidate_entries_from_runs()
    final_entry = next((e for e in entries if e["kind"] == "model_final"), None)
    if final_entry is None:
        raise RuntimeError("V23 expects one final adapter from the 7-epoch answer-only run.")
    return Path(final_entry["adapter_dir"])

def run_v23_valid():
    adapter_dir = train_and_get_adapter()
    retriever = StrictQueryRetriever(train_clean)
    selected = select_v23_profile_on_valid(adapter_dir, retriever)
    summary = dict(selected["summary"])
    summary.update({
        "strategy": "v23_legal_no_solver_ranker",
        "selected_profile": selected["profile"],
        "selected_adapter_dir": str(adapter_dir),
        "selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),
        "uses_valid_labels_for_profile_selection": True,
        "uses_valid_labels_for_ranker_training": False,
    })
    report, payload = save_outputs_and_optional_report(
        valid_clean,
        selected["outputs"],
        VALID_OUTPUT_PATH,
        VALID_REPORT_PATH,
        ENSEMBLE_RANKER_REPORT_PATH,
        summary,
    )
    MODEL_VALID_OUTPUT_PATH.write_text(json.dumps(selected["model_outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    if report is not None:
        print("[v23:valid]", report["summary"])
    return selected, payload

def run_v23_test():
    adapter_dir = train_and_get_adapter()
    retriever = StrictQueryRetriever(train_clean)
    selected_valid = select_v23_profile_on_valid(adapter_dir, retriever)
    profile = selected_valid["profile"]
    test_records = load_records(TEST_FILE)
    model_outputs = generate_outputs_for_profile(adapter_dir, test_records, "test", profile)
    outputs, decisions, summary = apply_no_solver_profile(test_records, model_outputs, retriever, profile)
    MODEL_TEST_OUTPUT_PATH.write_text(json.dumps(model_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(summary)
    summary.update({
        "strategy": "v23_legal_no_solver_ranker",
        "selected_profile": profile,
        "selected_adapter_dir": str(adapter_dir),
        "selection_report": str(CHECKPOINT_SELECTION_REPORT_PATH),
        "uses_valid_labels_for_profile_selection": True,
        "uses_valid_labels_for_ranker_training": False,
    })
    _report, payload = save_outputs_and_optional_report(
        test_records,
        outputs,
        TEST_OUTPUT_PATH,
        VALID_REPORT_PATH,
        ENSEMBLE_RANKER_REPORT_PATH,
        summary,
    )
    (WORKING_DIR / "test_v23_ranker_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[phase2] selected_profile=", profile["name"], "wrote", TEST_OUTPUT_PATH)
    return outputs, payload

if RUN_MODE == "phase1":
    _selected, _payload = run_v23_valid()
elif RUN_MODE == "phase2":
    if not TEST_FILE.exists():
        raise FileNotFoundError("RUN_MODE='phase2' requires test.json")
    _outputs, _payload = run_v23_test()
else:
    raise ValueError("Unknown RUN_MODE=" + str(RUN_MODE))
'''


def patch_config(source: str) -> str:
    replacements = {
        "v17_legal_answer_only_epoch7": "v23_legal_no_solver_ranker",
        "# V17 - Legal Answer-Only Epoch 7": "# V23 - Legal No-Solver Ranker",
        "Single-seed runtime-safe answer-only run. Uses only `query_vi` as model input and `response_vi` as train/validation target. No retrieval, no `type` routing, no `original_*` fields.": (
            "Single-seed answer-only run plus a legal no-solver decoder/retrieval ranker. "
            "Retrieval, when enabled by valid selection, only copies answers from train `response_vi` for highly similar train `query_vi` rows."
        ),
        "STAGE_A_LR = 0.001": "STAGE_A_LR = 0.003",
        "none_answer_only_fixed_epoch7": "query_vi_train_response_retrieval_no_solver_valid_profile_select",
        "legal_answer_only_model_consensus": "v23_legal_no_solver_ranker",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    source = source.replace(
        'RUN_MODE = "phase1"  # "phase1" writes valid_output/report; "phase2" writes test_predictions.json.',
        'RUN_MODE = os.environ.get("RUN_MODE", os.environ.get("RUN_MMODE", "phase1"))  # phase1 writes valid_output/report; phase2 writes test_predictions.json.',
    )
    source = source.replace(
        "LEGAL_QUERY_RETRIEVAL_ENABLED = False",
        "LEGAL_QUERY_RETRIEVAL_ENABLED = True",
    )
    source = source.replace(
        "RETRIEVAL_TOP_K = 7",
        "RETRIEVAL_TOP_K = 9",
    )
    source = source.replace(
        "RETRIEVAL_NGRAM_RANGE = (3, 5)",
        "RETRIEVAL_NGRAM_RANGE = (3, 5)",
    )
    source = source.replace(
        "NUM_BEAMS = 2\n",
        "NUM_BEAMS = 2\nMIN_NEW_TOKENS = 1\n",
    )
    source = source.replace(
        "DEFAULT_RETRIEVAL_GATE = {\"min_sim\": 0.60, \"min_majority_frac\": 0.45, \"min_margin\": 0.10, \"model_low_conf_frac\": 0.50}\n",
        '''DEFAULT_RETRIEVAL_GATE = {"enabled": False}

# V23 keeps runtime below v21 by training one 7-epoch adapter and sweeping only
# lightweight decode/retrieval profiles. No logprob verifier, no query-derived
# arithmetic, no template solver.
V23_PROFILES = [
    {
        "name": "beam2_model_only",
        "num_beams": 2,
        "max_new_tokens": 32,
        "runtime_cost": 1.0,
        "retrieval_gate": {"enabled": False},
    },
    {
        "name": "greedy_model_only",
        "num_beams": 1,
        "max_new_tokens": 24,
        "runtime_cost": 0.5,
        "retrieval_gate": {"enabled": False},
    },
    {
        "name": "beam2_retrieval_missing_only",
        "num_beams": 2,
        "max_new_tokens": 32,
        "runtime_cost": 1.1,
        "retrieval_gate": {
            "enabled": True,
            "top_k": 9,
            "min_sim": 0.92,
            "min_jaccard": 0.72,
            "min_majority_frac": 0.55,
            "min_margin": 0.22,
            "allow_when_model_missing": True,
            "allow_override": False,
        },
    },
    {
        "name": "beam2_retrieval_ultra_strict",
        "num_beams": 2,
        "max_new_tokens": 32,
        "runtime_cost": 1.2,
        "retrieval_gate": {
            "enabled": True,
            "top_k": 9,
            "min_sim": 0.96,
            "min_jaccard": 0.82,
            "min_majority_frac": 0.70,
            "min_margin": 0.35,
            "allow_when_model_missing": True,
            "allow_override": True,
        },
    },
]
''',
    )
    return source


def patch_generation_cell(source: str) -> str:
    source = source.replace(
        '''            max_new_tokens=eff_new,\n            pad_token_id=SAFE_EOS_ID,\n''',
        '''            max_new_tokens=eff_new,\n            min_new_tokens=min(int(globals().get("MIN_NEW_TOKENS", 0)), eff_new),\n            pad_token_id=SAFE_EOS_ID,\n''',
    )
    return source


def patch_manifest(source: str) -> str:
    source = source.replace(
        '"uses_valid_labels_for_checkpoint_selection": False,\n',
        '"uses_valid_labels_for_checkpoint_selection": False,\n'
        '        "uses_valid_labels_for_profile_selection": True,\n'
        '        "uses_arithmetic_or_template_solver": False,\n',
    )
    source = source.replace(
        '"selected_retrieval_gate": globals().get("SELECTED_RETRIEVAL_GATE", None),\n',
        '"selected_retrieval_gate": globals().get("SELECTED_RETRIEVAL_GATE", None),\n'
        '        "v23_profiles": globals().get("V23_PROFILES", None),\n',
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


def build_notebook() -> dict:
    v17 = load_module(V17_SCRIPT, "create_v17_answer_only_notebooks")
    nb = v17.build_notebook("epoch7")
    for idx, cell in enumerate(nb.get("cells", [])):
        if "source" not in cell:
            continue
        source = "".join(cell.get("source", []))
        source = patch_config(source)
        if idx == 8:
            marker = "def run_answer_only_split(records: list[dict], split_name: str, output_path: Path, report_path: Path):"
            if marker not in source:
                raise RuntimeError("Cannot locate answer-only run marker in cell 8")
            source = source.split(marker, 1)[0] + V23_RUN_CELL
            source = patch_generation_cell(source)
        if idx == 9:
            source = patch_manifest(source)
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
