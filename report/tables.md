# Report Tables

| Type | #Train | #Valid | #Unique source groups | Avg query length | Avg answer length | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| GSM_Rephrased | 20028 | 197 | 6454 | 49.68 | 90.06 | paraphrased GSM-style |
| GSM_AnsAug | 18745 | 209 | 6453 | 53.19 | 89.34 | answer-augmented variant |
| GSM_FOBAR | 10023 | 122 | 4438 | 79.22 | 151.21 | FOBAR perturbation from GSM sources |
| GSM_SV | 9869 | 97 | 4792 | 66.64 | 193.52 | symbol/variable GSM-style variant |
| MATH_Rephrased | 12477 | 116 | 4040 | 27.24 | 89.81 | paraphrased MATH-style |
| MATH_AnsAug | 16999 | 173 | 4377 | 27.62 | 84.79 | answer-augmented MATH variant |
| MATH_FOBAR | 3668 | 45 | 1696 | 65.02 | 225.93 | FOBAR perturbation from MATH sources |
| MATH_SV | 3591 | 41 | 1496 | 53.03 | 187.22 | symbol/variable MATH-style variant |

Caption: **Table 2. Distribution of problem types and source groups.** MATH is split into Rephrased, AnsAug, FOBAR, and SV rather than aggregated. The source-group count here is type-local, while the overlap audit reports 12,780 global source groups and 965/1000 validation rows with seen source groups.

| Version group | Target style | LoRA r | LR | Epochs | Max length | Decode | Validation for selection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2 | full solution | full/standard SFT | NaN | NaN | 768 | beam4 | valid |
| v3 | answer-only -> compact reasoning | SFT | NaN | NaN | 256/384 | beam2 | valid |
| v4 | KD compact reasoning + LoRA | 16 | NaN | 2 stages | NaN | beam2 | valid |
| v5/v7 | LoRA + GRPO-lite | 16 | NaN | staged | NaN | beam2 | valid |
| v6/v6_fast | answer-only LoRA | 16/32 | NaN | 3 / 2.25 | 256 | beam2 | valid |
| v8 | answer-only + checkpoint selection | 32 | 0.001 | 8 | 256 | beam2 | valid |
| v9 | compact-equation + source-disjoint selection | 32 | 0.001 | 8 | 256 | beam2 | source-disjoint |
| v10 | curriculum compact-equation | 32 | staged | 2+4+2 | 256 | beam2 | source-disjoint |
| v11 | answer-only + overlap-valid | 32 | 0.001 | 8 | 256 | beam2 | source-overlap query-disjoint |
| v12 | answer-only + broad source retrieval | 32 | 0.001 | 8 | 256 | beam2 | source-overlap query-disjoint |
| v13 | type-gated retrieval + full retrain | 32 | 0.001 | 8 | 256 | beam2 | source-overlap query-disjoint |
| v14 gate sweep | retrieval gate sweep + full retrain | 32 | 0.001 | 8 | 256 | beam2 | source-overlap query-disjoint |
| v14 FOBAR/SV | extended retrieval for FOBAR/SV | 32 | 0.001 | 8 | 256 | beam2 | source-overlap query-disjoint |
| v14.5 v13 | full train + valid-select V13 gate | 32 | 0.001 | 8 | 256 | beam2 | valid.json |
| v14.5 gate sweep | full train + valid-select gate sweep | 32 | 0.001 | 8 | 256 | beam2 | valid.json |
| v15 ensemble ranker | answer-only candidates + ExtraTrees ranker | 32 | 0.001 | 8 | 256 | beam2 | valid.json ranker labels |
| v15 type-specific expert | SV/FOBAR expert with safe routing | 32 | 0.001 | 8 | 256 | beam2 | overlap-valid + valid.json route check |
| v16 legal | query_vi-only answer/retrieval candidates | 32 | NaN | NaN | 256 | beam2 | internal train split; no result yet |

Caption: **Table 4. Experiment settings by version group.** Configuration rows combine local manifests with notebook-level descriptions; unavailable fields are left as NaN.

| Model | Target | Max length | Decode | Valid score | Extractability | Runtime |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | full response | 768 | beam4 | 0.585 (585/10000; legacy 649) | 854/1000 | NaN |

Caption: **Table 5. Full-solution SFT baseline.** Long targets preserve reasoning text but are expensive and may reduce final-answer reliability under the runtime constraint.

| Version | Target | Selection split | Valid score | Source-disjoint score | Avg output length | Reasoning consistency |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | answer -> compact NL | valid | NaN | NaN | NaN | NaN |
| v9 | compact equations | source-disjoint | 1.842 (1842/10000) | 1.392 (1392/10000) | 83.6 | extractable 959/1000 |
| v10 | curriculum compact | source-disjoint | 2.131 (2131/10000) | 1.396 (1396/10000) | 37.5 | extractable 990/1000 |

Caption: **Table 6. Compact reasoning target ablation.** Compact-equation supervision tests whether shorter arithmetic-focused rationales improve robustness compared with answer-only and full natural-language targets.

| Version | Prompt | LoRA config | Epochs | Selection | Valid score | Overlap-valid score | Source-disjoint score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v6 | no type | r=32 | 3 | final epoch | 1.810 (1810/10000) | NaN | NaN |
| v6_fast | no type | r=16 | 2.25 | final epoch | 1.789 (1789/10000) | NaN | NaN |
| v8 | no type | r=32 | 8 | valid checkpoint | 5.265 (5265/10000) | NaN | NaN |
| v11 | type-aware | r=32 | 8 | overlap-valid checkpoint | 4.329 (4329/10000) | 4.696 (4696/10000) | NaN |

Caption: **Table 8. Answer-only LoRA ablation.** Answer-only training aligns the loss with the final-answer metric and tests whether score improvements require explicit reasoning supervision.

| Version | Reward components | Penalize answer-only? | Score | Runtime | Failure mode |
| --- | --- | --- | --- | --- | --- |
| v5 | numeric + anchor | no | 0.955 (955/10000) | 8.0 min RL | score did not improve beyond SFT baseline |
| v7 | numeric + anchor + reasoning | yes | 0.639 (639/10000) | 20.9 min RL | lower extractability and low answer reward |

Caption: **Table 9. GRPO-lite reward-tuning results.** Reward tuning tests whether direct optimization of the scoring rule improves generation beyond supervised LoRA under a limited compute budget.

| Version | Retrieval | Allowed types | Gate | Model fallback | Official-valid score | Overlap-valid score | Source-disjoint score | Retrieval usage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v11 | no | none | none | yes | 4.329 (4329/10000) | 4.696 (4696/10000) | NaN | 0% |
| v12 | yes | broad | majority frac | yes | 6.194 (6194/10000) | 5.374 (5374/10000) | NaN | 90.9% |
| v13 | yes | Rephrased/AnsAug | type gate | yes | 6.896 (6896/10000) | 6.337 (6337/10000) | NaN | 62.5% |
| v14 sweep | yes | selected types | tuned thresholds | yes | 6.806 (6806/10000) | 6.145 (6145/10000) | NaN | 61.6% |
| v14 FOBAR/SV | yes | extended | strict nearest | yes | 6.379 (6379/10000) | 5.846 (5846/10000) | NaN | 82.2% |
| v14.5 v13 | yes | Rephrased/AnsAug | V13 gate + valid select | yes | 6.823 (6823/10000) | NaN | NaN | NaN |
| v14.5 gate | yes | selected types | gate sweep + valid select | yes | 6.807 (6807/10000) | NaN | NaN | NaN |
| v15 ensemble | mixed | 32 model/hybrid candidates | ExtraTrees ranker | yes | 7.016 (7016/10000) | NaN | NaN | ranker |
| v15 type-specific | yes | Rephrased/AnsAug; expert not routed | safe route check | yes | 6.927 (6927/10000) | 6.241 (6241/10000) | NaN | 62.5% |

Caption: **Table 10. Source-aware retrieval ablation.** Retrieval improves performance only when reliable source-level variants exist; therefore, the table reports retrieval usage and source-disjoint availability to distinguish memory from robust reasoning.

| Candidate | Official valid score | Source-disjoint score | Overlap-valid score | Runtime | Reasoning quality | Final choice |
| --- | --- | --- | --- | --- | --- | --- |
| Best answer-only | 4.329 (4329/10000) | NaN | 4.696 (4696/10000) | NaN | low/medium | no |
| Best compact reasoning | 2.131 (2131/10000) | 1.396 (1396/10000) | NaN | 200.4 min train | medium | no |
| Best hybrid retrieval (v13) | 6.896 (6896/10000) | NaN | 6.337 (6337/10000) | 177.7 min final retrain | retrieval-dependent | strong baseline; no longer best score |
| Best full-train valid-select (v14.5) | 6.823 (6823/10000) | NaN | NaN | NaN | retrieval-dependent | no; below v13/v15 despite valid selection |
| Best diagnostic ensemble (v15) | 7.016 (7016/10000) | NaN | NaN | NaN | answer-selection/retrieval-dependent | best diagnostic score; not submission-safe as-is |
| v14 gate sweep | 6.806 (6806/10000) | NaN | 6.145 (6145/10000) | 169.0 min final retrain | retrieval-dependent | no; below v13 on official and overlap-valid |
| v14 FOBAR/SV retrieval | 6.379 (6379/10000) | NaN | 5.846 (5846/10000) | 131.0 min final retrain | retrieval-dependent; weak FOBAR/SV | no; extended retrieval hurts target variants |

Caption: **Table 11. Final model selection.** The submitted configuration is selected by balancing official score, runtime, robustness, and risk of overfitting to source overlap.

| Error type | Description | Example id | Diagnostic |
| --- | --- | --- | --- |
| Parse failure | No final number or wrong final-answer anchor | 355 | output format |
| Arithmetic error | Correct relation pattern but wrong numeric computation | 6 | equation check |
| Relation error | Misread relation in the word problem | 18 | relational plan |
| Spurious shortcut | Predicted a frequent answer pattern | 40 | counterfactual |
| Correct answer, wrong reasoning | Answer is correct but rationale is unsupported | NaN | manual rationalization audit |
| Source-memory failure | Same source group has conflicting train answer | 30 | retrieval/source audit |
| Type confusion | GSM/MATH or FOBAR/SV variant confused | 30 | type-wise analysis |

Caption: **Table 12. Qualitative error taxonomy.** The taxonomy separates answer-format failures, computational failures, relational reasoning failures, and rationalization/source-memory risks. Example ids are heuristic seeds for manual inspection.

| Run | Gate policy | Selected checkpoint | Model-only valid | Hybrid valid | Hybrid gain | Overlap-valid | Retrieval usage | Changed outputs | Final retrain runtime | Fallback reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v13 type-gated | Rephrased/AnsAug type gate | epoch_08 | 5.389 (5389/10000) | 6.896 (6896/10000) | +1.507 | 6.337 (6337/10000) | 62.5% | NaN | 177.7 min final retrain | low_majority=28; source_unseen=42; type_not_allowed=305 |
| v14 gate sweep | selected thresholds | epoch_08 | 5.195 (5195/10000) | 6.806 (6806/10000) | +1.611 | 6.145 (6145/10000) | 61.6% | 247/1000 | 169.0 min final retrain | low_confidence=1; low_confidence_model_disagree=36; source_unseen=42; type_not_allowed=305 |
| v14 FOBAR/SV | extended strict nearest | epoch_06 | 4.626 (4626/10000) | 6.379 (6379/10000) | +1.753 | 5.846 (5846/10000) | 82.2% | 418/1000 | 131.0 min final retrain | low_confidence=1; low_confidence_model_disagree=49; source_unseen=49; typed_pool_missing=79 |

Caption: **Table 13. V14 run-level comparison.** This table separates model-only accuracy from retrieval-assisted accuracy, so gains from source-memory retrieval are not confused with standalone reasoning ability.

| Type | v14 sweep model | v14 sweep hybrid | v14 sweep gain | v14 sweep retrieval | v14 FOBAR/SV model | v14 FOBAR/SV hybrid | v14 FOBAR/SV gain | v14 FOBAR/SV retrieval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GSM_Rephrased | 8.462 | 9.513 | 1.051 | 98.5% | 7.863 | 9.503 | 1.64 | 98.5% |
| GSM_AnsAug | 4.215 | 7.177 | 2.962 | 88.5% | 3.852 | 7.105 | 3.254 | 89.0% |
| GSM_FOBAR | 2.992 | 2.992 | 0 | 0.0% | 2.27 | 1.336 | -0.934 | 67.2% |
| GSM_SV | 4.361 | 4.361 | 0 | 0.0% | 2.351 | 2.887 | 0.536 | 70.1% |
| MATH_AnsAug | 4.382 | 7.295 | 2.913 | 76.3% | 4.012 | 7.549 | 3.538 | 82.7% |
| MATH_Rephrased | 6.129 | 8.552 | 2.422 | 90.5% | 6.345 | 8.586 | 2.241 | 90.5% |
| MATH_FOBAR | 4.422 | 4.422 | 0 | 0.0% | 3.956 | 2.689 | -1.267 | 48.9% |
| MATH_SV | 4.659 | 4.659 | 0 | 0.0% | 3.878 | 3.805 | -0.073 | 53.7% |

Caption: **Table 14. V14 type-wise hybrid gains.** The table reports per-type model-only score, retrieval-assisted score, gain, and retrieval usage to identify where the retrieval policy helps or hurts.

| Run | Type | Enabled | Strategy | Min majority | Min margin | Min nearest Jaccard | Typed pool? | Agreement gate? | Official retrieval usage | Official hybrid score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v14 sweep | GSM_Rephrased | True | source_type_majority | 0.34 | 0 | 0.45 | False | True | 98.5% | 9.513 |
| v14 sweep | GSM_AnsAug | True | source_type_majority | 0.34 | 0 | 0 | False | True | 88.5% | 7.177 |
| v14 sweep | GSM_FOBAR | False | NaN | NaN | NaN | NaN | False | False | 0.0% | 2.992 |
| v14 sweep | GSM_SV | False | NaN | NaN | NaN | NaN | False | False | 0.0% | 4.361 |
| v14 sweep | MATH_AnsAug | True | source_type_majority | 0.34 | 0.2 | 0 | False | True | 76.3% | 7.295 |
| v14 sweep | MATH_Rephrased | True | source_type_majority | 0.34 | 0 | 0 | False | True | 90.5% | 8.552 |
| v14 sweep | MATH_FOBAR | False | NaN | NaN | NaN | NaN | False | False | 0.0% | 4.422 |
| v14 sweep | MATH_SV | False | NaN | NaN | NaN | NaN | False | False | 0.0% | 4.659 |
| v14 FOBAR/SV | GSM_Rephrased | True | source_type_majority | 0.34 | 0 | 0 | False | False | 98.5% | 9.503 |
| v14 FOBAR/SV | GSM_AnsAug | True | source_type_majority | 0.34 | 0 | 0 | False | False | 89.0% | 7.105 |
| v14 FOBAR/SV | GSM_FOBAR | True | source_type_nearest_query | 0 | 0 | 0.82 | True | False | 67.2% | 1.336 |
| v14 FOBAR/SV | GSM_SV | True | source_type_nearest_query | 0 | 0 | 0.84 | True | False | 70.1% | 2.887 |
| v14 FOBAR/SV | MATH_AnsAug | True | source_type_majority | 0.34 | 0 | 0 | False | False | 82.7% | 7.549 |
| v14 FOBAR/SV | MATH_Rephrased | True | source_type_majority | 0.34 | 0 | 0 | False | False | 90.5% | 8.586 |
| v14 FOBAR/SV | MATH_FOBAR | True | source_type_nearest_query | 0 | 0 | 0.82 | True | False | 48.9% | 2.689 |
| v14 FOBAR/SV | MATH_SV | True | source_type_nearest_query | 0 | 0 | 0.84 | True | False | 53.7% | 3.805 |

Caption: **Table 15. V14 selected retrieval gates.** Gate parameters show which problem types use majority-source retrieval versus strict nearest-query retrieval and how often each gate fires on official validation.

| Version | Family | Official-valid score | Exact10 | Zero | Extractable | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | answer-only -> compact NL | NaN | NaN | NaN | NaN | artifact not recoverable |
| v4 | KD compact reasoning + LoRA | NaN | NaN | NaN | NaN | artifact not recoverable |
| v2 | full-solution SFT | 0.585 (585/10000) | NaN | NaN | 854 | valid artifact |
| v5 | GRPO-lite numeric | 0.955 (955/10000) | 68 | 717 | 998 | valid artifact |
| v6 | answer-only LoRA | 1.810 (1810/10000) | 138 | 548 | 998 | valid artifact |
| v6_fast | answer-only LoRA r=16 | 1.789 (1789/10000) | 136 | 555 | 992 | valid artifact |
| v7 | GRPO-lite reasoning | 0.639 (639/10000) | 36 | 777 | 854 | valid artifact |
| v8 | answer-only checkpoint select | 5.265 (5265/10000) | 502 | 337 | 997 | valid artifact |
| v9 | compact equations | 1.842 (1842/10000) | 150 | 632 | 959 | valid artifact |
| v10 | curriculum compact | 2.131 (2131/10000) | 175 | 548 | 990 | valid artifact |
| v11 | answer-only overlap-valid | 4.329 (4329/10000) | 400 | 387 | 998 | valid artifact |
| v12 | broad source retrieval | 6.194 (6194/10000) | 606 | 300 | 998 | valid artifact |
| v13 | type-gated retrieval | 6.896 (6896/10000) | 678 | 226 | 999 | valid artifact |
| v14 gate | retrieval gate sweep | 6.806 (6806/10000) | 665 | 227 | 998 | valid artifact |
| v14 FOBAR/SV | extended FOBAR/SV retrieval | 6.379 (6379/10000) | 624 | 273 | 997 | valid artifact |
| v14.5 v13 | full-train valid-select v13 gate | 6.823 (6823/10000) | 667 | 228 | 999 | valid artifact |
| v14.5 gate | full-train valid-select gate sweep | 6.807 (6807/10000) | 664 | 221 | 996 | valid artifact |
| v15 ensemble | ensemble/ranker candidate selection | 7.016 (7016/10000) | 686 | 206 | 1000 | best diagnostic score |
| v15 type-specific | SV/FOBAR expert safety route | 6.927 (6927/10000) | 679 | 220 | 998 | valid artifact |

Caption: **Table 16. Main validation leaderboard through v15.** The table updates the report with v14.5 and v15 results and separates diagnostic score from missing/unrecoverable runs.

| Run | Mechanism | Model-only score | Selected single-candidate score | Final score | Final - model | Final - selected | Median output chars | Short single-line rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v12 | broad source retrieval | 4.489 (4489/10000) | NaN | 6.194 (6194/10000) | +1.705 | NaN | 13 | 100.0% |
| v13 | type-gated retrieval | 5.389 (5389/10000) | NaN | 6.896 (6896/10000) | +1.507 | NaN | 13 | 100.0% |
| v14 gate sweep | tuned retrieval gates | 5.195 (5195/10000) | NaN | 6.806 (6806/10000) | +1.611 | NaN | 13 | 100.0% |
| v14 FOBAR/SV | extended strict nearest retrieval | 4.626 (4626/10000) | NaN | 6.379 (6379/10000) | +1.753 | NaN | 13 | 100.0% |
| v14.5 v13 | valid-select V13 gate | 5.259 (5259/10000) | NaN | 6.823 (6823/10000) | +1.564 | NaN | 13 | 99.9% |
| v14.5 gate | valid-select gate sweep | 5.275 (5275/10000) | NaN | 6.807 (6807/10000) | +1.532 | NaN | 13 | 100.0% |
| v15 ensemble | ExtraTrees ranker over 32 candidates | 5.306 (5306/10000) | 6.857 (6857/10000) | 7.016 (7016/10000) | +1.710 | +0.159 | 13 | 100.0% |
| v15 type-specific | expert route disabled by validation | 5.183 (5183/10000) | NaN | 6.927 (6927/10000) | +1.744 | NaN | 13 | 100.0% |

Caption: **Table 17. Model-only versus final hybrid/ranker output.** The table quantifies how much of the score comes from retrieval, checkpoint/candidate selection, or a ranker beyond standalone GPT-2 generation.

| Diagnostic | Value | Score | Details |
| --- | --- | --- | --- |
| v15 ensemble ranker | sklearn_extra_trees | 7.016 (7016/10000) | 32 candidates; selected baseline 6.857 (6857/10000) |
| Top ranker choices | model::epoch_08=417; hybrid_v13_gate::epoch_08=163; hybrid_strict_gate::epoch_08=38; hybrid_v13_gate::epoch_04=32; hybrid_v13_gate::epoch_06=30 | 7.016 (7016/10000) | ranker often selects model-only, but uses hybrid candidates enough to improve final score |
| v15 type-specific expert | route_types=[] | 6.927 (6927/10000) | no target type routed to expert because expert did not beat baseline |
| Expert-vs-baseline target types | GSM_FOBAR: best_expert=398@epoch_07 baseline=427 route=False; GSM_SV: best_expert=391@epoch_08 baseline=436 route=False; MATH_FOBAR: best_expert=197@epoch_07 baseline=208 route=False; MATH_SV: best_expert=186@epoch_05 baseline=196 route=False | 6.927 (6927/10000) | expert improves over epochs but remains below general baseline on target types |

Caption: **Table 18. V15 diagnostic details.** V15 ensemble is the best local validation score, while the type-specific expert does not route any target type because it remains below the general baseline.

| Pipeline | Uses only query_vi/response_vi? | Uses provided type? | Uses original/source group fields? | Uses public valid labels for final decision? | Report status |
| --- | --- | --- | --- | --- | --- |
| v13/v14 retrieval | no | yes | yes | no | diagnostic only if strict query_vi/response_vi rule applies |
| v14.5 valid-select | no | yes | yes | valid checkpoint selection | diagnostic; optimized on public valid |
| v15 ensemble | no | yes | yes | yes | best diagnostic score; do not submit as-is under strict rules |
| v15 type-specific | no | yes | yes | route check by type | diagnostic; expert route disabled in final output |
| v16 legal answer-only | yes | no | no | internal split only | submission-safe direction; no local result yet |
| v16 legal query retrieval | yes | no | no | internal split only | submission-safe direction; query_vi-only nearest train query; no local result yet |

Caption: **Table 19. Diagnostic versus submission-safe pipeline features.** The table distinguishes research/audit branches from strict query_vi/response_vi-only directions.

| Slice | n | Score/10 | Raw contribution | Share of v13 raw score | Exact10 | Zero |
| --- | --- | --- | --- | --- | --- | --- |
| direct query seen with same answer | 10 | 10 | 100 | 1.5% | 10 | NaN |
| same source/original, same answer | 764 | 8.589 | 6562 | 95.2% | 652 | 74 |
| same source/original, conflicting answer | 174 | 1.471 | 256 | 3.7% | 19 | 105 |
| source seen but no same answer match | 38 | 2.026 | 77 | 1.1% | 7 | 24 |
| valid original appears as train query | 266 | 9.669 | 2572 | 37.3% | 257 | 7 |
| valid query appears as train original | 329 | 8.334 | 2742 | 39.8% | 273 | 44 |

Caption: **Table 20. Source-overlap score decomposition.** V13 score is dominated by same-source same-answer examples, while conflict and no-match slices remain low.

| Candidate set | Oracle score | Exact10 | Zero across all | Improved vs first | Interpretation |
| --- | --- | --- | --- | --- | --- |
| v13 + v14_gate_sweep | 7.289 (7289/10000) | 714 | 181 | 65 | candidate complementarity; not a deployable reasoning score |
| v13 + v14_gate_sweep + v14_5_v13 + v14_5_gate_sweep | 7.704 (7704/10000) | 751 | 135 | 129 | candidate complementarity; not a deployable reasoning score |
| v13 + v14_gate_sweep + v14_5_v13 + v14_5_gate_sweep + v12 + v14_fobar_sv | 7.825 (7825/10000) | 761 | 120 | 147 | candidate complementarity; not a deployable reasoning score |

Caption: **Table 21. Oracle ensemble upper bounds.** Candidate complementarity shows possible answer-selection upside, but these are not deployable reasoning scores.
