# Fair run comparison

Evaluator: `last explicit answer anchor` + robust scalar parser.

| run | family | checkpoint | decode | raw standard | score/10 | exact10 | extractable | numeric pairs | legacy raw | status |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline_greedy_legacy | baseline | baseline_1epoch | greedy | NA | NA | NA | NA | NA | 344 | missing_pred |
| v2_beam4 | v2 | v2_1epoch | beam4 | 585 | 0.585 | 38 | 854 | 799 | 649 | available |
| v2_greedy_same_ckpt | v2 | v2_1epoch | greedy | 529 | 0.529 | 34 | 953 | 926 | NA | available |
| v2_beam2_same_ckpt | v2 | v2_1epoch | beam2 | 646 | 0.646 | 41 | 955 | 929 | NA | available |
| v2_beam4_same_ckpt | v2 | v2_1epoch | beam4 | 591 | 0.591 | 35 | 946 | 916 | NA | available |

## Notes
- **baseline_greedy_legacy**: Legacy notebook summary only; raw prediction file is not present locally, so this row cannot be re-scored by the standard evaluator yet. Recover or regenerate baseline_valid_output.json before using this row in fair comparisons.
- **v2_beam4**: Legacy V2 report used the weaker parser. Existing local raw predictions from the 4-beam Kaggle run.
- **v2_greedy_same_ckpt**: Pending same-checkpoint decoding ablation.
- **v2_beam2_same_ckpt**: Pending same-checkpoint decoding ablation.
- **v2_beam4_same_ckpt**: Pending rerun from the same checkpoint for a clean decoding ablation; current historical beam4 run remains above as `v2_beam4`.
