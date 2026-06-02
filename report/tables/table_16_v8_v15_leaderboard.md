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
