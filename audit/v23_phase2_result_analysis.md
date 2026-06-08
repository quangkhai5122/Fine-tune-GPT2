# V23 Phase2 Result Analysis

Pseudo-gold analysis using `dataset/test_gold.json`; this is not official leaderboard gold.

## Score Summary
| run | raw | score10 | b10 | b5 | b1 | b0 | extractable | numeric_pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v17_phase2 | 1281 | 1.2810 | 84 | 40 | 241 | 635 | 999 | 993 |
| v21_phase2 | 1235 | 1.2350 | 89 | 27 | 210 | 674 | 1000 | 999 |
| v23_phase2 | 1481 | 1.4810 | 101 | 45 | 246 | 608 | 1000 | 998 |
| v22_v17 | 9781 | 9.7810 | 977 | 1 | 6 | 16 | 1000 | 999 |
| v22_v21 | 9740 | 9.7400 | 973 | 0 | 10 | 17 | 1000 | 999 |

## V23 Valid Profile Selection
| profile | valid_raw | exact10 | changed | source_counts | retrieval_reasons |
|---|---:|---:|---:|---|---|
| beam2_model_only | 6325 | 615 | 0 | {'model': 1000} | {'disabled': 1000} |
| greedy_model_only | 6195 | 600 | 0 | {'model': 1000} | {'disabled': 1000} |
| beam2_retrieval_missing_only | 6325 | 615 | 0 | {'model': 815, 'retrieval_model_agree': 141, 'model_fallback_disagree': 44} | {'low_confidence': 815, 'passed': 185} |
| beam2_retrieval_ultra_strict | 6305 | 613 | 3 | {'model': 956, 'retrieval_model_agree': 41, 'retrieval_override': 3} | {'low_confidence': 956, 'passed': 44} |

## V23 By Type
| type | raw | score10 | b10 | b5 | b1 | b0 |
|---|---:|---:|---:|---:|---:|---:|
| GSM_Rephrased | 302 | 0.9292 | 19 | 10 | 62 | 234 |
| GSM_SV | 181 | 0.9330 | 7 | 10 | 61 | 116 |
| MATH_Rephrased | 283 | 1.9930 | 20 | 9 | 38 | 75 |
| Synthetic_Generalization | 101 | 0.9018 | 4 | 6 | 31 | 71 |
| MATH_SV | 108 | 1.3500 | 6 | 4 | 28 | 42 |
| MATH_AnsAug | 317 | 4.4648 | 29 | 3 | 12 | 27 |
| GSM_FOBAR | 32 | 0.7273 | 2 | 1 | 7 | 34 |
| GSM_AnsAug | 141 | 5.2222 | 13 | 1 | 6 | 7 |
| MATH_FOBAR | 16 | 3.2000 | 1 | 1 | 1 | 2 |

## Deltas
### v23_vs_v17_phase2
- raw_delta=200, changed_score_rows=420, same_numeric=76/993
| type | delta | better | worse | same |
|---|---:|---:|---:|---:|
| GSM_Rephrased | -56 | 45 | 83 | 197 |
| GSM_SV | 69 | 54 | 28 | 112 |
| MATH_Rephrased | 55 | 42 | 26 | 74 |
| Synthetic_Generalization | 20 | 21 | 14 | 77 |
| MATH_SV | 61 | 30 | 11 | 39 |
| MATH_AnsAug | 47 | 19 | 9 | 43 |
| GSM_FOBAR | 8 | 9 | 10 | 25 |
| GSM_AnsAug | -8 | 7 | 9 | 11 |
| MATH_FOBAR | 4 | 2 | 1 | 2 |
### v23_vs_v21_phase2
- raw_delta=246, changed_score_rows=368, same_numeric=106/999
| type | delta | better | worse | same |
|---|---:|---:|---:|---:|
| GSM_Rephrased | 68 | 54 | 39 | 232 |
| GSM_SV | 103 | 59 | 20 | 115 |
| MATH_Rephrased | 42 | 27 | 29 | 86 |
| Synthetic_Generalization | -18 | 17 | 22 | 73 |
| MATH_SV | 35 | 27 | 13 | 40 |
| MATH_AnsAug | 17 | 16 | 13 | 42 |
| GSM_FOBAR | 1 | 8 | 10 | 26 |
| GSM_AnsAug | 2 | 7 | 5 | 15 |
| MATH_FOBAR | -4 | 1 | 1 | 3 |

## Oracle No-Solver Ensemble
- runs: ['v17_phase2', 'v21_phase2', 'v23_phase2']
- raw: 2433 score10=2.433
- buckets: {'10': 165, '5': 81, '1': 378, '0': 376}
- chosen_counts: {'v23_phase2': 722, 'v21_phase2': 139, 'v17_phase2': 139}

## Runtime / Compliance
- config: {'stage_a_epochs': 7.0, 'stage_a_lr': 0.003, 'num_beams': 2, 'max_new_tokens': 32, 'ensemble_last_k_epochs': 0, 'legal_query_retrieval_enabled': True}
- wall_minutes_training: 140.41177717049916
- rule_compliance: {'model_input_fields': ['query_vi'], 'training_target_fields': ['response_vi'], 'disallowed_model_feature_fields': ['type', 'original_question_en', 'original_question_vi'], 'uses_original_question_fields': False, 'uses_type_for_prompt_or_routing': False, 'type_is_copied_only_for_prediction_format_and_reports': True, 'uses_valid_labels_for_ranker_training': False, 'uses_valid_labels_for_checkpoint_selection': False, 'uses_valid_labels_for_profile_selection': True, 'uses_arithmetic_or_template_solver': False, 'retrieval_basis': 'query_vi_train_response_retrieval_no_solver_valid_profile_select'}
