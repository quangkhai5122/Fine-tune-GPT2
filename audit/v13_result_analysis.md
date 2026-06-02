# V13 result analysis

## Headline
- final hybrid raw: 6896 / 10000 (6.896/10)
- model-only raw: 5389 / 10000 (5.389/10)
- hybrid delta: +1507 raw, +165 exact10
- selected checkpoint on overlap-valid: epoch_08, epoch=8.0, lr=0.001
- final retrain: True on 92874 clean train rows

## Baseline Comparison
| run | raw | score/10 | exact10 | extractable |
|---|---:|---:|---:|---:|
| v8_beam2_lr1e-3 | 5265 | 5.265 | 502 | 997 |
| v9 | 1842 | 1.842 | 150 | 959 |
| v10 | 2131 | 2.131 | 175 | 990 |
| v11 | 4329 | 4.329 | 400 | 998 |
| v12 | 6194 | 6.194 | 606 | 998 |
| v13_model_only | 5389 | 5.389 | 513 | 998 |
| v13_final_hybrid | 6896 | 6.896 | 678 | 999 |

## Type Delta: Model Only -> Final Hybrid
| type | n | model/10 | final/10 | delta raw | improved | worse | model exact10 | final exact10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM_AnsAug | 209 | 4.407 | 7.144 | 572 | 70 | 15 | 87 | 148 |
| GSM_FOBAR | 122 | 3.205 | 3.205 | 0 | 0 | 0 | 36 | 36 |
| GSM_Rephrased | 197 | 8.487 | 9.513 | 202 | 28 | 6 | 162 | 187 |
| GSM_SV | 97 | 4.918 | 4.918 | 0 | 0 | 0 | 45 | 45 |
| MATH_AnsAug | 173 | 4.538 | 7.497 | 512 | 56 | 3 | 74 | 129 |
| MATH_FOBAR | 45 | 4.556 | 4.556 | 0 | 0 | 0 | 19 | 19 |
| MATH_Rephrased | 116 | 6.603 | 8.509 | 221 | 25 | 0 | 74 | 98 |
| MATH_SV | 41 | 4.195 | 4.195 | 0 | 0 | 0 | 16 | 16 |

## Overlap Slices
| group | n | score/10 | exact10 | buckets |
|---|---:|---:|---:|---|
| direct_query_has_same | 10 | 10.000 | 10 | {'10': 10} |
| same_original_has_same | 764 | 8.589 | 652 | {'10': 652, '5': 1, '1': 37, '0': 74} |
| same_original_only_conflict | 174 | 1.471 | 19 | {'10': 19, '5': 4, '1': 46, '0': 105} |
| same_original_no_match | 38 | 2.026 | 7 | {'10': 7, '1': 7, '0': 24} |
| valid_original_seen_as_train_query_has_same | 266 | 9.669 | 257 | {'10': 257, '1': 2, '0': 7} |
| valid_query_seen_as_train_original_has_same | 329 | 8.334 | 273 | {'10': 273, '1': 12, '0': 44} |

## Exact10 and Zero Breakdown
- exact10 `n_exact10`: 678
- exact10 `same_original_has_same`: 652
- exact10 `same_original_only_conflict`: 19
- exact10 `same_original_no_match`: 7
- exact10 `valid_original_seen_as_train_query_has_same`: 257
- exact10 `valid_query_seen_as_train_original_has_same`: 273
- exact10 `direct_query_has_same`: 10
- zero `n_zero`: 226
- zero `same_original_has_same`: 74
- zero `same_original_only_conflict`: 105
- zero `same_original_no_match`: 24

## Magnitude Slices
| gold magnitude | n | score/10 | exact10 | buckets |
|---|---:|---:|---:|---|
| abs<=10 | 419 | 6.473 | 266 | {'10': 266, '1': 52, '0': 101} |
| 10<abs<=100 | 354 | 7.249 | 252 | {'10': 252, '5': 4, '1': 26, '0': 72} |
| 100<abs<=1000 | 162 | 8.000 | 128 | {'10': 128, '5': 1, '1': 11, '0': 22} |
| abs>1000 | 42 | 7.667 | 32 | {'10': 32, '1': 2, '0': 8} |
| non_numeric_gold | 23 | 0.000 | 0 | {'0': 23} |

## Hybrid Routing
- retrieval used: 625 / 1000 (0.625)
- allowed types: ['GSM_AnsAug', 'GSM_Rephrased', 'MATH_AnsAug', 'MATH_Rephrased']
- fallback reasons: {'type_not_allowed': 305, 'source_unseen': 42, 'low_majority': 28}
- output-changed count: 238
- model->final transitions: {'0->0': 203, '0->1': 3, '0->10': 110, '1->0': 12, '1->1': 87, '1->5': 1, '1->10': 49, '5->0': 1, '5->1': 1, '5->5': 4, '5->10': 16, '10->0': 10, '10->10': 503}

## Answer Distribution
- top exact10 predictions: [('3', 35), ('2', 30), ('4', 30), ('5', 22), ('6', 22), ('8', 20), ('10', 19), ('20', 17), ('1', 15), ('50', 13), ('12', 13), ('60', 11), ('15', 11), ('9', 10), ('16', 9)]
- top zero predictions: [('2', 18), ('4', 16), ('5', 15), ('3', 14), ('12', 10), ('8', 10), ('1', 9), ('20', 8), ('15', 8), ('50', 8), ('30', 7), ('6', 6), ('10', 6), ('7', 6), ('0', 6)]
