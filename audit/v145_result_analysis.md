# V14.5 result analysis

## Leaderboard
| run                    | raw  | score/10 | exact10 | 5  | 1   | 0   | extractable |
| ---------------------- | ---- | -------- | ------- | -- | --- | --- | ----------- |
| v13                    | 6896 | 6.896    | 678     | 5  | 91  | 226 | 999         |
| v14_5_v13              | 6823 | 6.823    | 667     | 12 | 93  | 228 | 999         |
| v14_5_gate_sweep       | 6807 | 6.807    | 664     | 13 | 102 | 221 | 996         |
| v14_gate_sweep         | 6806 | 6.806    | 665     | 12 | 96  | 227 | 998         |
| v14_fobar_sv           | 6379 | 6.379    | 624     | 9  | 94  | 273 | 997         |
| v12                    | 6194 | 6.194    | 606     | 10 | 84  | 300 | 998         |
| v8_select_beam2_lr1e-3 | 5265 | 5.265    | 502     | 21 | 140 | 337 | 997         |

## Model-only vs final hybrid
| run              | model raw | final raw | delta | model exact10 | final exact10 | selected | split                         |
| ---------------- | --------- | --------- | ----- | ------------- | ------------- | -------- | ----------------------------- |
| v13              | 5389      | 6896      | 1507  | 513           | 678           | epoch_08 | source_overlap_query_disjoint |
| v14_gate_sweep   | 5195      | 6806      | 1611  | 489           | 665           | epoch_08 | source_overlap_query_disjoint |
| v14_5_v13        | 5259      | 6823      | 1564  | 494           | 667           | epoch_08 | valid_json                    |
| v14_5_gate_sweep | 5275      | 6807      | 1532  | 497           | 664           | epoch_08 | valid_json                    |

## V14.5 checkpoint curves
### v14_5_v13
| ckpt                                             | raw  | exact10 | 5  | 1   | 0   | extractable |
| ------------------------------------------------ | ---- | ------- | -- | --- | --- | ----------- |
| epoch_01                                         | 5892 | 579     | 3  | 87  | 331 | 995         |
| epoch_02                                         | 6242 | 607     | 11 | 117 | 265 | 999         |
| epoch_03                                         | 6276 | 609     | 13 | 121 | 257 | 994         |
| epoch_04                                         | 6425 | 628     | 6  | 115 | 251 | 999         |
| epoch_05                                         | 6505 | 635     | 9  | 110 | 246 | 994         |
| epoch_06                                         | 6622 | 645     | 10 | 122 | 223 | 999         |
| epoch_07                                         | 6766 | 661     | 10 | 106 | 223 | 999         |
| epoch_08                                         | 6823 | 667     | 12 | 93  | 228 | 999         |
| gpt2_math_lora_v145_v13_valid_select_answer_only | 6823 | 667     | 12 | 93  | 228 | 999         |
| gpt2_math_lora_v145_v13_valid_select_final       | 6823 | 667     | 12 | 93  | 228 | 999         |
| gpt2_math_lora_v145_v13_valid_select_sft         | 6823 | 667     | 12 | 93  | 228 | 999         |

### v14_5_gate_sweep
| ckpt                                                    | raw  | exact10 | 5  | 1   | 0   | extractable |
| ------------------------------------------------------- | ---- | ------- | -- | --- | --- | ----------- |
| epoch_01                                                | 6120 | 595     | 8  | 130 | 267 | 1000        |
| epoch_02                                                | 6265 | 610     | 10 | 115 | 265 | 998         |
| epoch_03                                                | 6280 | 613     | 7  | 115 | 265 | 999         |
| epoch_04                                                | 6295 | 613     | 9  | 120 | 258 | 995         |
| epoch_05                                                | 6462 | 628     | 13 | 117 | 242 | 999         |
| epoch_06                                                | 6579 | 642     | 10 | 109 | 239 | 996         |
| epoch_07                                                | 6749 | 658     | 12 | 109 | 221 | 997         |
| epoch_08                                                | 6807 | 664     | 13 | 102 | 221 | 996         |
| gpt2_math_lora_v145_gate_sweep_valid_select_answer_only | 6807 | 664     | 13 | 102 | 221 | 996         |
| gpt2_math_lora_v145_gate_sweep_valid_select_final       | 6807 | 664     | 13 | 102 | 221 | 996         |
| gpt2_math_lora_v145_gate_sweep_valid_select_sft         | 6807 | 664     | 13 | 102 | 221 | 996         |

## Type-level comparison
### v13
| type           | n   | model/10 | final/10 | delta raw | final exact10 | final zero |
| -------------- | --- | -------- | -------- | --------- | ------------- | ---------- |
| GSM_AnsAug     | 209 | 4.407    | 7.144    | 572       | 148           | 48         |
| GSM_FOBAR      | 122 | 3.205    | 3.205    | 0         | 36            | 55         |
| GSM_Rephrased  | 197 | 8.487    | 9.513    | 202       | 187           | 6          |
| GSM_SV         | 97  | 4.918    | 4.918    | 0         | 45            | 33         |
| MATH_AnsAug    | 173 | 4.538    | 7.497    | 512       | 129           | 37         |
| MATH_FOBAR     | 45  | 4.556    | 4.556    | 0         | 19            | 15         |
| MATH_Rephrased | 116 | 6.603    | 8.509    | 221       | 98            | 15         |
| MATH_SV        | 41  | 4.195    | 4.195    | 0         | 16            | 17         |

### v14_gate_sweep
| type           | n   | model/10 | final/10 | delta raw | final exact10 | final zero |
| -------------- | --- | -------- | -------- | --------- | ------------- | ---------- |
| GSM_AnsAug     | 209 | 4.215    | 7.177    | 619       | 148           | 49         |
| GSM_FOBAR      | 122 | 2.992    | 2.992    | 0         | 33            | 54         |
| GSM_Rephrased  | 197 | 8.462    | 9.513    | 207       | 187           | 6          |
| GSM_SV         | 97  | 4.361    | 4.361    | 0         | 40            | 38         |
| MATH_AnsAug    | 173 | 4.382    | 7.295    | 504       | 123           | 34         |
| MATH_FOBAR     | 45  | 4.422    | 4.422    | 0         | 18            | 16         |
| MATH_Rephrased | 116 | 6.129    | 8.552    | 281       | 98            | 14         |
| MATH_SV        | 41  | 4.659    | 4.659    | 0         | 18            | 16         |

### v14_5_v13
| type           | n   | model/10 | final/10 | delta raw | final exact10 | final zero |
| -------------- | --- | -------- | -------- | --------- | ------------- | ---------- |
| GSM_AnsAug     | 209 | 4.397    | 7.057    | 556       | 145           | 51         |
| GSM_FOBAR      | 122 | 3.254    | 3.254    | 0         | 36            | 57         |
| GSM_Rephrased  | 197 | 8.284    | 9.503    | 240       | 187           | 8          |
| GSM_SV         | 97  | 4.175    | 4.175    | 0         | 38            | 38         |
| MATH_AnsAug    | 173 | 4.243    | 7.578    | 577       | 129           | 31         |
| MATH_FOBAR     | 45  | 4.400    | 4.400    | 0         | 18            | 13         |
| MATH_Rephrased | 116 | 6.914    | 8.560    | 191       | 98            | 13         |
| MATH_SV        | 41  | 4.195    | 4.195    | 0         | 16            | 17         |

### v14_5_gate_sweep
| type           | n   | model/10 | final/10 | delta raw | final exact10 | final zero |
| -------------- | --- | -------- | -------- | --------- | ------------- | ---------- |
| GSM_AnsAug     | 209 | 4.498    | 7.254    | 576       | 150           | 47         |
| GSM_FOBAR      | 122 | 3.582    | 3.582    | 0         | 39            | 48         |
| GSM_Rephrased  | 197 | 8.350    | 9.543    | 235       | 187           | 4          |
| GSM_SV         | 97  | 3.742    | 3.742    | 0         | 34            | 40         |
| MATH_AnsAug    | 173 | 4.699    | 7.434    | 473       | 126           | 33         |
| MATH_FOBAR     | 45  | 4.444    | 4.444    | 0         | 18            | 15         |
| MATH_Rephrased | 116 | 6.422    | 8.560    | 248       | 98            | 13         |
| MATH_SV        | 41  | 3.220    | 3.220    | 0         | 12            | 21         |

## Retrieval routing
| run              | used | used pct | changed | fallback reasons                                                                                         |
| ---------------- | ---- | -------- | ------- | -------------------------------------------------------------------------------------------------------- |
| v13              | 625  | 0.625    | n/a     | {'type_not_allowed': 305, 'source_unseen': 42, 'low_majority': 28}                                       |
| v14_gate_sweep   | 616  | 0.616    | 247     | {'type_not_allowed': 305, 'source_unseen': 42, 'low_confidence_model_disagree': 36, 'low_confidence': 1} |
| v14_5_v13        | 625  | 0.625    | n/a     | {'type_not_allowed': 305, 'source_unseen': 42, 'low_majority': 28}                                       |
| v14_5_gate_sweep | 622  | 0.622    | 237     | {'type_not_allowed': 305, 'source_unseen': 42, 'low_confidence_model_disagree': 31}                      |

## Model-to-final transitions
| run              | helpful to 10 | harmful major | transitions                                                                                                                                                                   |
| ---------------- | ------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| v13              | 175           | 23            | {'0->0': 203, '0->1': 3, '0->10': 110, '1->0': 12, '1->1': 87, '1->10': 49, '1->5': 1, '10->0': 10, '10->10': 503, '5->0': 1, '5->1': 1, '5->10': 16, '5->5': 4}              |
| v14_gate_sweep   | 187           | 26            | {'0->0': 202, '0->1': 2, '0->10': 114, '1->0': 14, '1->1': 92, '1->10': 58, '1->5': 1, '10->0': 10, '10->1': 1, '10->10': 478, '5->0': 1, '5->1': 1, '5->10': 15, '5->5': 11} |
| v14_5_v13        | 188           | 32            | {'0->0': 197, '0->1': 6, '0->10': 108, '1->0': 16, '1->1': 86, '1->10': 62, '10->0': 14, '10->1': 1, '10->10': 479, '5->0': 1, '5->10': 18, '5->5': 12}                       |
| v14_5_gate_sweep | 176           | 23            | {'0->0': 198, '0->1': 7, '0->10': 105, '1->0': 11, '1->1': 95, '1->10': 58, '1->5': 1, '10->0': 9, '10->10': 488, '5->0': 3, '5->10': 13, '5->5': 12}                         |

## Same-original slices
| run              | slice                       | n   | score/10 | exact10 | zero |
| ---------------- | --------------------------- | --- | -------- | ------- | ---- |
| v13              | same_original_has_same      | 764 | 8.589    | 652     | 74   |
| v13              | same_original_no_match      | 38  | 2.026    | 7       | 24   |
| v13              | same_original_only_conflict | 174 | 1.471    | 19      | 105  |
| v13              | same_original_only_missing  | 24  | 0.042    | 0       | 23   |
| v14_gate_sweep   | same_original_has_same      | 764 | 8.554    | 647     | 76   |
| v14_gate_sweep   | same_original_no_match      | 38  | 2.263    | 6       | 18   |
| v14_gate_sweep   | same_original_only_conflict | 174 | 1.063    | 12      | 109  |
| v14_gate_sweep   | same_original_only_missing  | 24  | 0.000    | 0       | 24   |
| v14_5_v13        | same_original_has_same      | 764 | 8.471    | 641     | 81   |
| v14_5_v13        | same_original_no_match      | 38  | 2.289    | 6       | 17   |
| v14_5_v13        | same_original_only_conflict | 174 | 1.511    | 20      | 107  |
| v14_5_v13        | same_original_only_missing  | 24  | 0.042    | 0       | 23   |
| v14_5_gate_sweep | same_original_has_same      | 764 | 8.480    | 641     | 78   |
| v14_5_gate_sweep | same_original_no_match      | 38  | 1.658    | 4       | 19   |
| v14_5_gate_sweep | same_original_only_conflict | 174 | 1.517    | 19      | 101  |
| v14_5_gate_sweep | same_original_only_missing  | 24  | 0.042    | 0       | 23   |

## Output stats
| run              | mean chars | median chars | short single-line | unique numeric preds |
| ---------------- | ---------- | ------------ | ----------------- | -------------------- |
| v13              | 13.08      | 13           | 1.000             | 245                  |
| v14_gate_sweep   | 13.07      | 13           | 1.000             | 249                  |
| v14_5_v13        | 13.11      | 13           | 0.999             | 247                  |
| v14_5_gate_sweep | 13.03      | 13           | 1.000             | 240                  |

## Oracle ensemble upper bounds
| candidate runs                                                           | oracle raw | score/10 | exact10 | zero_all | improved_vs_first |
| ------------------------------------------------------------------------ | ---------- | -------- | ------- | -------- | ----------------- |
| v13 + v14_gate_sweep                                                     | 7289       | 7.289    | 714     | 181      | 65                |
| v13 + v14_gate_sweep + v14_5_v13 + v14_5_gate_sweep                      | 7704       | 7.704    | 751     | 135      | 129               |
| v13 + v14_gate_sweep + v14_5_v13 + v14_5_gate_sweep + v12 + v14_fobar_sv | 7825       | 7.825    | 761     | 120      | 147               |

## Selected gate config
### v14_5_gate_sweep
```json
{
  "GSM_Rephrased": {
    "enabled": true,
    "strategy": "source_type_majority",
    "min_majority_frac": 0.34,
    "min_margin": 0.33,
    "min_nearest_jaccard": 0.0,
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
    "min_margin": 0.0,
    "min_nearest_jaccard": 0.0,
    "require_typed_pool": false,
    "use_model_agreement_gate": true,
    "model_agreement_bypass_confidence": true
  }
}
```

## Short interpretation
- V13 remains best on valid.json; both V14.5 valid-select runs are below V13 despite selecting on valid.
- Full-train/select-valid improved early checkpoint scores but did not improve the epoch-08 final enough; fallback types remain the largest bottleneck.
- Retrieval is still responsible for most of the score above model-only, but current source-majority retrieval is near saturation for Rephrased/AnsAug.
- The next high-upside path is not broader raw retrieval; it is type-specific answer-only specialization for FOBAR/SV plus safer retrieval/confidence arbitration.
