# V17 results analysis

Date: 2026-06-03

## Summary

`v17_legal_answer_only_epoch7` is the best finished strict-legal, under-3h notebook in the current set.

| run | strict legal | retrieval | runtime | raw | score_10 | exact10 | 5 | 1 | 0 | extractable |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v17_epoch7 | yes | no | 141.48 min | 5450 | 5.450 | 514 | 31 | 155 | 300 | 999 |
| v17_678 selected | yes | no | 163.54 min | 5258 | 5.258 | 497 | 24 | 168 | 311 | 999 |
| v16_multiseed | yes | no | 331.28 min | 5487 | 5.487 | 525 | 19 | 142 | 314 | 999 |
| v16_query_retrieval | yes by fields | query_vi retrieval | 206.32 min | 4957 | 4.957 | 469 | 23 | 152 | 356 | 999 |
| v8_select lr1e-3 | yes | no | near runtime limit | 5265 | 5.265 | 502 | 21 | 140 | 337 | 997 |
| v15_ensemble | no | original/type retrieval/ranker | unknown | 7016 | 7.016 | 686 | 12 | 96 | 206 | 1000 |

## V17 findings

`v17_epoch7` beats `v8_select lr1e-3` by +185 raw and is only -37 raw behind `v16_multiseed`, while using less than half the runtime.

`v17_678` did not reproduce the strong v16 seed-42 epoch-7 checkpoint. Its independent run had:

| checkpoint | raw | score_10 | exact10 |
|---|---:|---:|---:|
| epoch_06 | 5240 | 5.240 | 493 |
| epoch_07 | 5185 | 5.185 | 490 |
| epoch_08 | 5258 | 5.258 | 497 |

The selected checkpoint was epoch 8. Valid selection worked correctly for that run, but the run itself was weaker than the standalone epoch-7 run.

This suggests material training variance despite fixed seed. For leaderboard, fixed epoch 7 without checkpoint selection is currently more attractive than select 6/7/8.

## By-type comparison

| type | v17_epoch7 | v17_678 | v16_multiseed | v8_lr1e-3 | v17_epoch7 vs v8 | v17_epoch7 vs v16 |
|---|---:|---:|---:|---:|---:|---:|
| GSM_AnsAug | 1037 | 877 | 999 | 876 | +161 | +38 |
| GSM_FOBAR | 388 | 415 | 422 | 392 | -4 | -34 |
| GSM_Rephrased | 1576 | 1588 | 1622 | 1665 | -89 | -46 |
| GSM_SV | 476 | 462 | 430 | 466 | +10 | +46 |
| MATH_AnsAug | 786 | 765 | 833 | 740 | +46 | -47 |
| MATH_FOBAR | 200 | 221 | 206 | 190 | +10 | -6 |
| MATH_Rephrased | 777 | 783 | 819 | 759 | +18 | -42 |
| MATH_SV | 210 | 147 | 156 | 177 | +33 | +54 |

The strongest v17 gains over v8 are `GSM_AnsAug`, `MATH_AnsAug`, and `MATH_SV`. The biggest loss is `GSM_Rephrased`, where v8 remains unusually strong.

## Candidate diversity ceiling

The legal answer-only candidates are diverse enough that better selection could matter:

| candidate pool | candidates | oracle raw | oracle score_10 | rows where all candidates agree |
|---|---:|---:|---:|---:|
| v17 only | 4 | 6669 | 6.669 | 427 |
| legal answer-only v8/v16/v17 | 10 | 7510 | 7.510 | 279 |
| legal answer-only + v16 query retrieval | 11 | 7617 | 7.617 | 253 |

This is not directly submit-safe because oracle uses gold labels, but it shows the next real opportunity: a legal candidate selector/verifier, not more checkpoint picking alone.

## Retrieval legality from v12 onward

Under the strict rule "use only `query_vi` and `response_vi`", the retrieval notebooks from v12 to v15 are not compliant:

- v12 uses `PROMPT_TEMPLATE = "Dạng: {type}..."` and `SOURCE_GROUP_KEY_FIELDS = ["original_question_en", "original_question_vi", "query_vi"]`.
- v13 uses the same type prompt and original-question source grouping.
- v14 and v14.5 use the same source grouping and type prompt, including retrieval by `source_group_key`.
- v15 ensemble/type-specific use the same original-question grouping, type prompt, type/ranker features, and valid-trained ranker logic.

The only retrieval branch that is strict-field legal is `v16_query_retrieval_ranker`: prompt uses `query_vi`, retrieval keys use train `query_vi`, and answers come from train `response_vi`. However, that run is below answer-only and exceeds the 3h runtime (`206.32 min`), so it is not a good submission candidate as-is.

## Recommendation

For immediate leaderboard priority, submit or continue from `v17_legal_answer_only_epoch7`, not `v17_678`.

Next experiments with the best chance of improving leaderboard score:

1. Run several independent `v17_epoch7`-style notebooks with small changes, one per Kaggle account:
   - fixed epoch 7, no checkpoint saving
   - seeds such as 42, 7, 2024, 3407
   - LR around `8e-4`, `1e-3`, `1.2e-3`
   - choose by valid score only after each full run

2. Build a strict-legal candidate verifier:
   - candidates: fixed epoch-7 output, lightweight query-only retrieval answer, arithmetic/template candidate
   - features: model answer logprob, candidate answer prior, query-derived numeric/symbol features, retrieval similarity, model-output parse quality
   - no `type`, no `original_*`, no valid labels for training the selector

3. Rework legal retrieval as template/formula retrieval, not answer copying:
   - retrieve train rows by `query_vi`
   - normalize numbers into slots
   - infer simple formula from same-skeleton train examples
   - copy answer only when numeric signature is nearly identical

Direct query-only answer-copy retrieval has already underperformed; the useful path is retrieval as a source of candidates/templates plus a verifier.
