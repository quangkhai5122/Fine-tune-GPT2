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
