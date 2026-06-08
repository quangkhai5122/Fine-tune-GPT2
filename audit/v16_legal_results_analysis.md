# V16 legal results analysis

Date: 2026-06-03

## Executive summary

Both v16 notebooks follow the strict data-use rule: model input is `query_vi`; training/validation target is `response_vi`; `type` is copied only for output/report format; `original_question_*` is not used.

However, both current v16 runs are not ideal for official submission under the `<=3h` runtime limit:

- `v16_multiseed`: best legal score so far, but runtime is about 5h31m.
- `v16_query_retrieval_ranker`: runtime is about 3h26m and score is worse than answer-only.

The best practical next direction is a v17 single-seed legal answer-only notebook that uses the same full-train setup as v16 but selects or hard-codes the best checkpoint around epoch 7.

## Scores

| run | raw | score_10 | exact10 | 5 | 1 | 0 | extractable |
|---|---:|---:|---:|---:|---:|---:|---:|
| v16_multiseed final | 5487 | 5.487 | 525 | 19 | 142 | 314 | 999 |
| v16_multiseed primary model | 5400 | 5.400 | 515 | 21 | 145 | 319 | 999 |
| v16_query_retrieval final | 4957 | 4.957 | 469 | 23 | 152 | 356 | 999 |
| v16_query_retrieval primary model | 4942 | 4.942 | 467 | 23 | 157 | 353 | 999 |
| v8_select_checkpoint beam2 lr1e-3 | 5265 | 5.265 | 502 | 21 | 140 | 337 | 997 |
| v8/3_lora_target | 2731 | 2.731 | 227 | 35 | 286 | 452 | 993 |

Strict-legal ranking from available results:

1. `v16_multiseed`: 5487, but too slow.
2. `v8_select_checkpoint/beam2_lr1e-3`: 5265, single-seed legal answer-only.
3. `v16_query_retrieval_ranker`: 4957, legal retrieval but below answer-only.

Non-strict / original-field retrieval results such as v13-v15 are higher, but they are not safe under the stated `query_vi`/`response_vi` rule.

## Runtime

From notebook cell metadata and output logs:

| run | total notebook time | training cell | inference/selection cell | notes |
|---|---:|---:|---:|---|
| v16_multiseed | ~331 min | ~321.6 min | ~9.0 min | 2 seeds, each ~160 min |
| v16_query_retrieval | ~206 min | ~146.3 min | ~59.3 min | 1 seed trained on 90% fit split, then 4 calib + 4 valid generations |
| v8_select_checkpoint beam2 lr1e-3 | estimated ~152+ min from checkpoint timestamps | single seed | checkpoint eval | likely much closer to <=3h |

`v16_multiseed` cannot be used as-is for official runtime. `v16_query_retrieval` is also over 3 hours because the query retrieval notebook spends a lot of time generating calibration and validation candidates.

## V16 multiseed details

By type:

| type | n | raw | score_10 | exact10 | zero |
|---|---:|---:|---:|---:|---:|
| GSM_AnsAug | 209 | 999 | 4.780 | 94 | 72 |
| GSM_FOBAR | 122 | 422 | 3.459 | 39 | 51 |
| GSM_Rephrased | 197 | 1622 | 8.234 | 160 | 23 |
| GSM_SV | 97 | 430 | 4.433 | 41 | 40 |
| MATH_AnsAug | 173 | 833 | 4.815 | 78 | 66 |
| MATH_FOBAR | 45 | 206 | 4.578 | 19 | 18 |
| MATH_Rephrased | 116 | 819 | 7.060 | 80 | 25 |
| MATH_SV | 41 | 156 | 3.805 | 14 | 19 |

Candidate scores:

| candidate | raw | score_10 |
|---|---:|---:|
| seed_42 epoch_07 | 5486 | 5.486 |
| seed_42 epoch_08/final | 5400 | 5.400 |
| seed_42 epoch_06 | 5297 | 5.297 |
| seed_123 epoch_07 | 5257 | 5.257 |
| seed_123 epoch_08/final | 5226 | 5.226 |
| seed_123 epoch_06 | 5134 | 5.134 |

Key finding: the final multiseed consensus is only +1 raw point over `seed_42 epoch_07`, while requiring almost double training time. The runtime-safe version should keep seed 42 and use/select epoch 7.

## V16 query-only retrieval details

Final score: 4957, only +15 over its primary model baseline 4942.

Retrieval gate selected on internal train calibration:

```json
{"min_sim": 0.45, "min_majority_frac": 0.34, "min_margin": 0.0, "model_low_conf_frac": 0.5}
```

Decision counts on valid:

- `retrieval_model_agree`: 316
- `model_fallback_disagree`: 473
- `model_consensus`: 177
- `retrieval_low_model_conf`: 34

Interpretation:

- Query-only retrieval passes often, but most passes either agree with the model or are rejected because the model disagrees.
- It only changes a small number of rows through `retrieval_low_model_conf`.
- Net gain over its own model baseline is +15 raw, not enough to offset the weaker 90%-train model.
- Calibration on internal train split did not transfer well to public valid; it selected a loose threshold (`min_sim=0.45`) that is not robust.

## Comparison with non-retrieval/legal history

| run | strict legal? | retrieval? | score_10 | note |
|---|---|---|---:|---|
| v5 | yes | no | 0.955 | early baseline |
| v6 | yes | no | 1.810 | answer-only/compact improvements |
| v8/3_lora_target | yes | no | 2.731 | answer-only checkpoint line |
| v8_select_checkpoint beam2 lr1e-3 | yes | no | 5.265 | strongest previous runtime-safe legal baseline |
| v16_multiseed | yes | no | 5.487 | best legal score, too slow |
| v16_query_retrieval | yes | query-only | 4.957 | legal but weaker |
| v11 | no under strict rule | no retrieval | 4.329 | uses `type` prompt and `original_*` source split |
| v12+ | no under strict rule | original-field retrieval | 6.194+ | strong but not submit-safe |

## Recommended next experiments

### V17 A: legal answer-only checkpoint select

Goal: preserve v16_multiseed score while fitting runtime.

Config:

- one seed: `42`
- full cleaned train: 92,874 rows
- prompt: `Bài toán: {query_vi}\nLời giải:`
- target: `Đáp án là: <canonical_num>`
- LoRA: same as v16
- train epochs: 7 or 8 with saved checkpoints
- evaluate only epoch 6/7/8 on `valid.json`
- choose best checkpoint for final output

Expected from existing candidate outputs:

- epoch 7: 5486
- epoch 8/final: 5400

Runtime estimate:

- train one seed: ~160 min
- evaluate 3 checkpoints: about 3-4 min
- total likely under 3 hours.

This should be the next primary official-safe line.

### V17 B: legal answer-only fixed epoch 7

If the official run must avoid any validation checkpoint selection, hard-code `STAGE_A_EPOCHS=7.0` and write final from epoch 7.

Expected valid score from the finished v16 candidate:

- ~5486 raw

Runtime:

- about 140 min training plus one valid/test generation.

This is the most runtime-safe answer-only candidate.

### V17 C: conservative query retrieval as optional fallback

Do not keep the current loose retrieval gate. If query-only retrieval is kept, make it conservative:

- train on full train, not 90% split
- no calibration generation during official run
- fixed gate from analysis, e.g. `min_sim >= 0.75`, `majority_frac >= 0.50`, `margin >= 0.20`
- use retrieval only when model agrees or model output is unparseable
- remove `retrieval_low_model_conf` except for very high similarity

This likely gives smaller gains but should avoid corrupting many rows and save runtime.

### V17 D: query-derived buckets, not `type`

For further legal gains, use query text patterns instead of the provided `type` label:

- variable-solving: contains `x`, `X`, `biến`, `giá trị của biến`
- equation/symbolic: contains `$`, `\frac`, `\sqrt`, `phương trình`
- long/noisy math: contains `[asy]`, very long query, many symbols
- word arithmetic: mostly numbers and Vietnamese word-problem verbs

Use these buckets only for sampling weights or post-processing heuristics. Do not route by `type`.

## Recommendation

Create two next notebooks:

1. `v17_legal_answer_only_epoch7`: seed 42, 7 epochs, no retrieval, no multiseed. This is the best runtime-safe version.
2. `v17_legal_answer_only_select_678`: seed 42, 8 epochs, save checkpoints, evaluate epoch 6/7/8 on valid, select best. This should reproduce the 5486-level result with acceptable runtime.

