# V14 result analysis

## Leaderboard so far
| run | raw | score/10 | exact10 | zero | extractable |
|---|---:|---:|---:|---:|---:|
| v13 | 6896 | 6.896 | 678 | 226 | 999 |
| v14_gate_sweep | 6806 | 6.806 | 665 | 227 | 998 |
| v14_fobar_sv | 6379 | 6.379 | 624 | 273 | 997 |
| v12 | 6194 | 6.194 | 606 | 300 | 998 |
| v8_select_beam2_lr1e-3 | 5265 | 5.265 | 502 | 337 | 997 |
| v11 | 4329 | 4.329 | 400 | 387 | 998 |
| v8_select_beam2_lr5e-4 | 3608 | 3.608 | 323 | 423 | 999 |
| v8_select_beam4_lr5e-4 | 3528 | 3.528 | 318 | 446 | 998 |
| v8_3_lora_target | 2731 | 2.731 | 227 | 452 | 993 |
| v10 | 2131 | 2.131 | 175 | 548 | 990 |
| v9 | 1842 | 1.842 | 150 | 632 | 959 |
| v6 | 1810 | 1.810 | 138 | 548 | 998 |
| v6_fast | 1789 | 1.789 | 136 | 555 | 992 |
| v8_2_lora_target | 1022 | 1.022 | 69 | 679 | 973 |
| v5 | 955 | 0.955 | 68 | 717 | 998 |
| 4beams | 649 | 0.649 | 43 | 790 | 854 |
| v7 | 639 | 0.639 | 36 | 777 | 854 |

## V13/V14 model vs hybrid
| run | model raw | final raw | delta | model exact10 | final exact10 | selected ckpt | final retrain epochs |
|---|---:|---:|---:|---:|---:|---|---:|
| v13 | 5389 | 6896 | 1507 | 513 | 678 | epoch_08 | 8.0 |
| v14_gate_sweep | 5195 | 6806 | 1611 | 489 | 665 | epoch_08 | 8.0 |
| v14_fobar_sv | 4626 | 6379 | 1753 | 432 | 624 | epoch_06 | 6.0 |

## Type-level scores
### v13
| type | n | model/10 | final/10 | delta raw | final exact10 | final zero |
|---|---:|---:|---:|---:|---:|---:|
| GSM_AnsAug | 209 | 4.407 | 7.144 | 572 | 148 | 48 |
| GSM_FOBAR | 122 | 3.205 | 3.205 | 0 | 36 | 55 |
| GSM_Rephrased | 197 | 8.487 | 9.513 | 202 | 187 | 6 |
| GSM_SV | 97 | 4.918 | 4.918 | 0 | 45 | 33 |
| MATH_AnsAug | 173 | 4.538 | 7.497 | 512 | 129 | 37 |
| MATH_FOBAR | 45 | 4.556 | 4.556 | 0 | 19 | 15 |
| MATH_Rephrased | 116 | 6.603 | 8.509 | 221 | 98 | 15 |
| MATH_SV | 41 | 4.195 | 4.195 | 0 | 16 | 17 |

### v14_gate_sweep
| type | n | model/10 | final/10 | delta raw | final exact10 | final zero |
|---|---:|---:|---:|---:|---:|---:|
| GSM_AnsAug | 209 | 4.215 | 7.177 | 619 | 148 | 49 |
| GSM_FOBAR | 122 | 2.992 | 2.992 | 0 | 33 | 54 |
| GSM_Rephrased | 197 | 8.462 | 9.513 | 207 | 187 | 6 |
| GSM_SV | 97 | 4.361 | 4.361 | 0 | 40 | 38 |
| MATH_AnsAug | 173 | 4.382 | 7.295 | 504 | 123 | 34 |
| MATH_FOBAR | 45 | 4.422 | 4.422 | 0 | 18 | 16 |
| MATH_Rephrased | 116 | 6.129 | 8.552 | 281 | 98 | 14 |
| MATH_SV | 41 | 4.659 | 4.659 | 0 | 18 | 16 |

### v14_fobar_sv
| type | n | model/10 | final/10 | delta raw | final exact10 | final zero |
|---|---:|---:|---:|---:|---:|---:|
| GSM_AnsAug | 209 | 3.852 | 7.105 | 680 | 147 | 47 |
| GSM_FOBAR | 122 | 2.270 | 1.336 | -114 | 13 | 76 |
| GSM_Rephrased | 197 | 7.863 | 9.503 | 323 | 187 | 8 |
| GSM_SV | 97 | 2.351 | 2.887 | 52 | 26 | 51 |
| MATH_AnsAug | 173 | 4.012 | 7.549 | 612 | 128 | 35 |
| MATH_FOBAR | 45 | 3.956 | 2.689 | -57 | 10 | 22 |
| MATH_Rephrased | 116 | 6.345 | 8.586 | 260 | 99 | 15 |
| MATH_SV | 41 | 3.878 | 3.805 | -3 | 14 | 19 |

## Retrieval routing
| run | retrieval used | used pct | output changed | fallback reasons |
|---|---:|---:|---:|---|
| v13 | 625 | 0.625 | n/a | {'type_not_allowed': 305, 'source_unseen': 42, 'low_majority': 28} |
| v14_gate_sweep | 616 | 0.616 | 247 | {'type_not_allowed': 305, 'source_unseen': 42, 'low_confidence_model_disagree': 36, 'low_confidence': 1} |
| v14_fobar_sv | 822 | 0.822 | 418 | {'typed_pool_missing': 79, 'low_confidence_model_disagree': 49, 'source_unseen': 49, 'low_confidence': 1} |

## Same-original overlap slices
| run | slice | n | score/10 | exact10 | zero |
|---|---|---:|---:|---:|---:|
| v13 | same_original_has_same | 764 | 8.589 | 652 | 74 |
| v13 | same_original_no_match | 38 | 2.026 | 7 | 24 |
| v13 | same_original_only_conflict | 174 | 1.471 | 19 | 105 |
| v13 | same_original_only_missing | 24 | 0.042 | 0 | 23 |
| v14_gate_sweep | same_original_has_same | 764 | 8.554 | 647 | 76 |
| v14_gate_sweep | same_original_no_match | 38 | 2.263 | 6 | 18 |
| v14_gate_sweep | same_original_only_conflict | 174 | 1.063 | 12 | 109 |
| v14_gate_sweep | same_original_only_missing | 24 | 0.000 | 0 | 24 |
| v14_fobar_sv | same_original_has_same | 764 | 8.081 | 612 | 110 |
| v14_fobar_sv | same_original_no_match | 38 | 2.474 | 7 | 23 |
| v14_fobar_sv | same_original_only_conflict | 174 | 0.632 | 5 | 117 |
| v14_fobar_sv | same_original_only_missing | 24 | 0.042 | 0 | 23 |

## Selected V14 gate configs
### v14_gate_sweep
```json
{
  "GSM_Rephrased": {
    "enabled": true,
    "strategy": "source_type_majority",
    "min_majority_frac": 0.34,
    "min_margin": 0.0,
    "min_nearest_jaccard": 0.45,
    "require_typed_pool": false,
    "use_model_agreement_gate": true,
    "model_agreement_bypass_confidence": true
  },
  "MATH_Rephrased": {
    "enabled": true,
    "strategy": "source_type_majority",
    "min_majority_frac": 0.34,
    "min_margin": 0.0,
    "min_nearest_jaccard": 0.0,
    "require_typed_pool": false,
    "use_model_agreement_gate": true,
    "model_agreement_bypass_confidence": true
  },
  "GSM_AnsAug": {
    "enabled": true,
    "strategy": "source_type_majority",
    "min_majority_frac": 0.34,
    "min_margin": 0.0,
    "min_nearest_jaccard": 0.0,
    "require_typed_pool": false,
    "use_model_agreement_gate": true,
    "model_agreement_bypass_confidence": true
  },
  "MATH_AnsAug": {
    "enabled": true,
    "strategy": "source_type_majority",
    "min_majority_frac": 0.34,
    "min_margin": 0.2,
    "min_nearest_jaccard": 0.0,
    "require_typed_pool": false,
    "use_model_agreement_gate": true,
    "model_agreement_bypass_confidence": true
  }
}
```
### v14_fobar_sv
```json
{
  "GSM_Rephrased": {
    "strategy": "source_type_majority",
    "min_majority_frac": 0.34,
    "min_margin": 0.0,
    "min_nearest_jaccard": 0.0
  },
  "MATH_Rephrased": {
    "strategy": "source_type_majority",
    "min_majority_frac": 0.34,
    "min_margin": 0.0,
    "min_nearest_jaccard": 0.0
  },
  "GSM_AnsAug": {
    "strategy": "source_type_majority",
    "min_majority_frac": 0.34,
    "min_margin": 0.0,
    "min_nearest_jaccard": 0.0
  },
  "MATH_AnsAug": {
    "strategy": "source_type_majority",
    "min_majority_frac": 0.34,
    "min_margin": 0.0,
    "min_nearest_jaccard": 0.0
  },
  "GSM_FOBAR": {
    "strategy": "source_type_nearest_query",
    "require_typed_pool": true,
    "min_majority_frac": 0.0,
    "min_margin": 0.0,
    "min_nearest_jaccard": 0.82
  },
  "MATH_FOBAR": {
    "strategy": "source_type_nearest_query",
    "require_typed_pool": true,
    "min_majority_frac": 0.0,
    "min_margin": 0.0,
    "min_nearest_jaccard": 0.82
  },
  "GSM_SV": {
    "strategy": "source_type_nearest_query",
    "require_typed_pool": true,
    "min_majority_frac": 0.0,
    "min_margin": 0.0,
    "min_nearest_jaccard": 0.84
  },
  "MATH_SV": {
    "strategy": "source_type_nearest_query",
    "require_typed_pool": true,
    "min_majority_frac": 0.0,
    "min_margin": 0.0,
    "min_nearest_jaccard": 0.84
  }
}
```

## Recommendation
- Best current valid score remains v13: 6896 raw / 6.896.
- v14_gate_sweep is second among current runs: 6806 raw / 6.806; it improves a few allowed retrieval types but loses on model-only fallback types.
- v14_fobar_sv is not a candidate for direct continuation because strict FOBAR/SV retrieval still hurts conflict-heavy FOBAR and does not offset weaker model-only training.
- For the next full-train, select-on-valid runs, keep v13 as the exploitation baseline and v14_gate_sweep as the safer gate-tuned challenger.

## Selected next notebooks
- `finetune_gpt2_for_math_v14_5_v13_full_train_valid_select.ipynb`: full-train version of the current best V13 type-gated source-majority policy.
- `finetune_gpt2_for_math_v14_5_gate_sweep_full_train_valid_select.ipynb`: full-train version of V14 gate sweep, selecting both checkpoint and active gate config on `valid.json`.
- Both train `train_stage_a` from `train_clean`, set `FINAL_RETRAIN_FULL_TRAIN=False`, and call `select_best_checkpoint_on_valid(valid_records)` instead of selecting on overlap-valid.
