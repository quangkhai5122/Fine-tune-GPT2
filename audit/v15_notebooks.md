# V15 Kaggle notebooks

## 1. `finetune_gpt2_for_math_v15_ensemble_ranker.ipynb`

Goal: improve answer-only score by choosing among multiple numeric candidates instead of trusting a single checkpoint/hybrid output.

Base: `v14_5_gate_sweep_full_train_valid_select`.

Main behavior:
- trains answer-only LoRA on all cleaned train rows;
- selects checkpoints on `valid.json`;
- collects candidates from the top epoch checkpoints:
  - model-only outputs;
  - active gate-sweep hybrid outputs;
  - V13-style hybrid outputs;
  - stricter hybrid outputs;
- trains a lightweight ranker on `valid.json` labels;
- overwrites `valid_output.json` with ranker-selected answers in phase1;
- in phase2, trains the ranker on valid candidates and applies it to generated test candidates.

Important outputs:
- `valid_output.json`
- `valid_report.json`
- `selected_valid_output.json`
- `selected_valid_report.json`
- `ensemble_ranker_report.json`
- `ensemble_candidates/`
- `test_predictions.json` in phase2

Key knobs:
- `ENSEMBLE_TOP_K_CHECKPOINTS = 8`
- `ENSEMBLE_PHASE2_TOP_K_CHECKPOINTS = 8`
- `ENSEMBLE_INCLUDE_MODEL_CANDIDATES = True`
- `ENSEMBLE_INCLUDE_RETRIEVAL_CANDIDATES = True`
- `ENSEMBLE_RANKER_TREES = 300`

## 2. `finetune_gpt2_for_math_v15_type_expert_sv_fobar.ipynb`

Goal: keep the strong V13 retrieval baseline, then improve the weak fallback groups using a separate answer-only expert for `SV/FOBAR`.

Base: `v13_type_gated_full_retrain`.

Main behavior:
- trains/evaluates the V13 type-gated baseline;
- trains a fresh LoRA expert only on:
  - `GSM_FOBAR`
  - `MATH_FOBAR`
  - `GSM_SV`
  - `MATH_SV`
- evaluates expert checkpoints on valid rows of those types;
- routes a type to expert only if its valid raw score beats the V13 baseline for that type;
- overwrites `valid_output.json` with the safe combined output in phase1;
- in phase2, applies the selected expert routes to test predictions.

Important outputs:
- `valid_output.json`
- `valid_report.json`
- `general_valid_output.json`
- `general_valid_report.json`
- `expert_train_info.json`
- `expert_selection_report.json`
- `v15_sv_fobar_expert_eval/`
- `test_predictions.json` in phase2

Key knobs:
- `EXPERT_TYPES = ["GSM_FOBAR", "MATH_FOBAR", "GSM_SV", "MATH_SV"]`
- `EXPERT_EPOCHS = 8.0`
- `EXPERT_LR = 8e-4`

