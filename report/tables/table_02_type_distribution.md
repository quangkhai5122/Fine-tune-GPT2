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
