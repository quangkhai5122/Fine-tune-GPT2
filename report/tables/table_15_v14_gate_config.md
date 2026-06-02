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
