| Version | Prompt | LoRA config | Epochs | Selection | Valid score | Overlap-valid score | Source-disjoint score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v6 | no type | r=32 | 3 | final epoch | 1.810 (1810/10000) | NaN | NaN |
| v6_fast | no type | r=16 | 2.25 | final epoch | 1.789 (1789/10000) | NaN | NaN |
| v8 | no type | r=32 | 8 | valid checkpoint | 5.265 (5265/10000) | NaN | NaN |
| v11 | type-aware | r=32 | 8 | overlap-valid checkpoint | 4.329 (4329/10000) | 4.696 (4696/10000) | NaN |

Caption: **Table 8. Answer-only LoRA ablation.** Answer-only training aligns the loss with the final-answer metric and tests whether score improvements require explicit reasoning supervision.
