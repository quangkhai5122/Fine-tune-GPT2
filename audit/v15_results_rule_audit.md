# V15 results and rule audit

Date: 2026-06-03

## Rule baseline from Task.md

`Task.md` states that the usable data fields for the model/data pipeline are:

- `query_vi` -> input question
- `response_vi` -> training target / validation gold answer

The prediction format includes `type`, but the data-use rule says to use only the Vietnamese fields. Under a strict reading, `type`, `original_question_en`, and `original_question_vi` should not be used for prompt construction, retrieval keys, routing, ranker features, or checkpoint/model selection features.

## Score summary

| version | raw_score | score_10 | exact10 | 5 | 1 | 0 | extractable |
|---|---:|---:|---:|---:|---:|---:|---:|
| v13 | 6896 | 6.896 | 678 | 5 | 91 | 226 | 999 |
| v14_5_v13 | 6823 | 6.823 | 667 | 12 | 93 | 228 | 999 |
| v14_5_gate_sweep | 6807 | 6.807 | 664 | 13 | 102 | 221 | 996 |
| v15_ensemble | 7016 | 7.016 | 686 | 12 | 96 | 206 | 1000 |
| v15_type_specific | 6927 | 6.927 | 679 | 9 | 92 | 220 | 998 |

`v15_ensemble` is the best validation score so far, +120 raw points over v13. `v15_type_specific` is only +31 over v13.

## V15 ensemble

`results/v15_ensemble/valid_report.json`:

| type | n | raw | score_10 | exact10 | zero |
|---|---:|---:|---:|---:|---:|
| GSM_AnsAug | 209 | 1506 | 7.206 | 149 | 48 |
| GSM_FOBAR | 122 | 400 | 3.279 | 36 | 46 |
| GSM_Rephrased | 197 | 1923 | 9.761 | 192 | 2 |
| GSM_SV | 97 | 456 | 4.701 | 43 | 32 |
| MATH_AnsAug | 173 | 1306 | 7.549 | 127 | 34 |
| MATH_FOBAR | 45 | 233 | 5.178 | 22 | 14 |
| MATH_Rephrased | 116 | 987 | 8.509 | 98 | 15 |
| MATH_SV | 41 | 205 | 5.000 | 19 | 15 |

Compared with v13, the main gains are GSM_Rephrased +49, MATH_SV +33, MATH_FOBAR +28, MATH_AnsAug +9, GSM_FOBAR +9, GSM_AnsAug +13. The main regression is GSM_SV -21.

The ranker report shows:

- `ranker_kind`: `sklearn_extra_trees`
- `num_candidates`: 32
- top selections:
  - `model::epoch_08`: 417
  - `hybrid_v13_gate::epoch_08`: 163
  - `hybrid_strict_gate::epoch_08`: 38
  - `hybrid_v13_gate::epoch_04`: 32
  - `hybrid_v13_gate::epoch_06`: 30

This means the score improvement is not pure retrieval. The ranker often selects the direct model candidate, but uses retrieval/hybrid candidates enough to gain over the selected single candidate baseline: 7016 vs 6857.

## V15 type-specific expert

`results/v15_type_specific/expert_selection_report.json` shows:

- `route_types`: `[]`
- For `GSM_FOBAR`, `MATH_FOBAR`, `GSM_SV`, and `MATH_SV`, `use_expert=false`.

So the final output does not actually use the SV/FOBAR expert. It falls back to the general baseline for all types.

Expert-only progress on the 305 target examples:

| checkpoint | raw | exact10 | zero |
|---|---:|---:|---:|
| epoch_01 | 562 | 46 | 169 |
| epoch_02 | 719 | 60 | 154 |
| epoch_03 | 685 | 57 | 153 |
| epoch_04 | 824 | 70 | 139 |
| epoch_05 | 1044 | 95 | 136 |
| epoch_06 | 1064 | 96 | 133 |
| epoch_07 | 1134 | 103 | 126 |
| epoch_08/final | 1137 | 104 | 128 |

The expert improves with training, but it remains below the general baseline per target type:

- GSM_FOBAR: expert 389 vs baseline 427
- GSM_SV: expert 391 vs baseline 436
- MATH_FOBAR: expert 197 vs baseline 208
- MATH_SV: expert 160 vs baseline 196

## Output behavior

Both V15 outputs are answer-only:

- no multiline reasoning
- median output length about 13 characters
- examples: `Đáp án là: 37`, `Đáp án là: 19`, `Đáp án là: 8`

The remaining gains come from answer selection and retrieval/ranking, not from generating longer chain-of-thought.

## Rule audit of current V15 notebooks

Both V15 notebooks use fields outside `query_vi`/`response_vi`.

In `finetune_gpt2_for_math_v15_ensemble_ranker.ipynb`:

- Prompt uses `type`: `PROMPT_TEMPLATE = "Dạng: {type}\nBài toán: {q}\nLời giải: "`.
- Source grouping uses original fields: `SOURCE_GROUP_KEY_FIELDS = ["original_question_en", "original_question_vi", "query_vi"]`.
- Retrieval uses `source_group_key(rec)`, therefore can retrieve by original-question identity when those fields exist.
- Candidate/ranker logic uses `rec.get("type")` and type-candidate statistics.
- The ranker is trained on `valid.json` candidates/gold labels, which is useful for local development but high-risk for leaderboard generalization if it overfits the public validation set.

In `finetune_gpt2_for_math_v15_type_expert_sv_fobar.ipynb`:

- Prompt uses `type`.
- Source grouping uses original fields.
- Expert training and routing explicitly filter by `type`.

Under a strict interpretation of `Task.md`, the current retrieval and type-specific versions are not compliant for official submission. The most serious issue is original-field retrieval; the second issue is use of `type` as model input/routing/ranker feature.

## Legal directions

### V16 legal answer-only

Use only:

- prompt input: `query_vi`
- target/gold: canonical answer extracted from train `response_vi`

Remove:

- `type` from prompt
- original fields from all split/retrieval/routing logic
- valid-trained ranker as a final decision learner

Recommended prompt:

```text
Bài toán: {query_vi}
Đáp án là:
```

Train answer-only LoRA with multiple seeds/checkpoints. Use an internal split from train for checkpoint/seed selection, and use public valid only for final reporting.

### V16 legal query-only retrieval

Build the retrieval index from train only:

- key text: train `query_vi`
- answer: extracted from train `response_vi`

At inference:

- retrieve nearest train `query_vi` by BM25/TF-IDF char n-gram/Jaccard
- use retrieval answer only when similarity is very high and nearest neighbors agree
- fallback to model when retrieval confidence is low or model disagrees

This keeps retrieval legal because it uses only train `query_vi` and `response_vi`.

### Legal ranker/calibrator

Candidates can include:

- model outputs from multiple checkpoints/seeds
- query-only retrieval answer
- simple majority answer from model candidates

Allowed features should be derived only from `query_vi`, train `response_vi`, and candidate strings:

- model agreement count
- candidate frequency across checkpoints
- query-only nearest similarity scores
- numeric answer prior from train `response_vi`
- answer parse quality
- generated answer length

Avoid:

- `type`
- `original_question_*`
- valid labels as the training target for the final ranker

### Legal type-like routing

Do not use the provided `type` label unless the organizer explicitly confirms it is allowed. Instead, derive buckets from `query_vi` text only:

- variable-solving patterns: contains `x`, `X`, `biến`, `giá trị của biến`
- equation/symbolic patterns: contains `$`, `\\frac`, `\\sqrt`, `phương trình`
- word arithmetic patterns: mostly numbers and Vietnamese word-problem verbs
- long/noisy ASY/table patterns

Then train small answer-only experts or decoding policies for those query-derived buckets.

## Recommendation

Keep `v15_ensemble` as the best research/diagnostic branch, but do not submit it as-is if the rule is enforced strictly. The next two official-safe experiments should be:

1. `v16_legal_answer_only_multiseed`: answer-only SFT, no `type`, no original fields, multi-seed/checkpoint selection by internal train split.
2. `v16_legal_query_retrieval_ranker`: query-only retrieval plus legal candidate ranker/calibrator trained on internal train split, not on public valid labels.

