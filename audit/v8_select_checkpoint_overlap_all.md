# V8 select-checkpoint overlap analysis

- run base: `results\v8_select_checkpoint`
- train records scanned: 95400
- valid records: 1000

## Run summary
| run | lr | beams | selected | raw | exact10 | extractable | same-original exact10 | no-original exact10 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| beam2_lr1e-3 | 0.001 | 2 | epoch_08 | 5265 | 502 | 997 | 479 | 4 |
| beam2_lr5e-4 | 0.0005 | 2 | epoch_06 | 3608 | 323 | 999 | 303 | 4 |
| beam4_lr5e-4 | 0.0005 | 4 | epoch_08 | 3528 | 318 | 998 | 297 | 5 |

## Global overlap classes
- `direct_query_vi`: train_hits=28, classes={'no_match': 988, 'has_same': 10, 'only_missing': 1, 'only_conflict': 1}
- `direct_query_vi_exact`: train_hits=0, classes={'no_match': 1000}
- `same_original_vi`: train_hits=6924, classes={'has_same': 764, 'only_conflict': 174, 'no_match': 38, 'only_missing': 24}
- `train_query_vs_valid_original`: train_hits=1792, classes={'has_same': 266, 'no_match': 479, 'only_conflict': 237, 'only_missing': 18}
- `train_original_vs_valid_query`: train_hits=1968, classes={'no_match': 641, 'has_same': 329, 'only_missing': 12, 'only_conflict': 18}
- `same_original_en`: train_hits=6981, classes={'has_same': 767, 'only_conflict': 174, 'no_match': 35, 'only_missing': 24}

## beam2_lr1e-3
- raw: 5265 / 10000; exact10=502; extractable=997
- config: `{'stage_a_lr': 0.001, 'stage_a_epochs': 8.0, 'num_beams': 2, 'checkpoint_eval_num_beams': 2, 'lora_target_modules': ['c_attn', 'c_proj', 'c_fc'], 'lora_r': 32, 'lora_alpha': 64}`
- selected: `{'label': 'epoch_08', 'epoch': 8.0, 'raw_score': 5265, 'exact10': 502}`
- checkpoint curve: epoch_01=1094, epoch_02=2088, epoch_03=3496, epoch_04=4298, epoch_05=4830, epoch_06=5046, epoch_07=5252, epoch_08=5265

| group | n | raw | score/10 | exact10 | exact10 share | buckets |
|---|---:|---:|---:|---:|---:|---|
| direct_query_has_same | 10 | 90 | 9.000 | 9 | 1.8% | {'10': 9, '0': 1} |
| same_original_has_same | 764 | 4958 | 6.490 | 479 | 95.4% | {'10': 479, '5': 16, '1': 88, '0': 181} |
| same_original_only_conflict | 174 | 253 | 1.454 | 19 | 3.8% | {'10': 19, '5': 4, '1': 43, '0': 108} |
| same_original_no_match | 38 | 54 | 1.421 | 4 | 0.8% | {'10': 4, '5': 1, '1': 9, '0': 24} |
| same_original_only_missing | 24 | 0 | 0.000 | 0 | 0.0% | {'0': 24} |
| valid_original_seen_as_train_query_has_same | 266 | 2200 | 8.271 | 217 | 43.2% | {'10': 217, '5': 3, '1': 15, '0': 31} |
| valid_query_seen_as_train_original_has_same | 329 | 1585 | 4.818 | 148 | 29.5% | {'10': 148, '5': 10, '1': 55, '0': 116} |

## beam2_lr5e-4
- raw: 3608 / 10000; exact10=323; extractable=999
- config: `{'stage_a_lr': 0.0005, 'stage_a_epochs': 8.0, 'num_beams': 2, 'checkpoint_eval_num_beams': 2, 'lora_target_modules': ['c_attn', 'c_proj', 'c_fc'], 'lora_r': 32, 'lora_alpha': 64}`
- selected: `{'label': 'epoch_06', 'epoch': 6.0, 'raw_score': 3608, 'exact10': 323}`
- checkpoint curve: epoch_01=1172, epoch_02=1652, epoch_03=2120, epoch_04=2989, epoch_05=3292, epoch_06=3608, epoch_07=3582, epoch_08=3570

| group | n | raw | score/10 | exact10 | exact10 share | buckets |
|---|---:|---:|---:|---:|---:|---|
| direct_query_has_same | 10 | 91 | 9.100 | 9 | 2.8% | {'10': 9, '1': 1} |
| same_original_has_same | 764 | 3313 | 4.336 | 303 | 93.8% | {'10': 303, '5': 25, '1': 158, '0': 278} |
| same_original_only_conflict | 174 | 231 | 1.328 | 16 | 5.0% | {'10': 16, '5': 3, '1': 56, '0': 99} |
| same_original_no_match | 38 | 63 | 1.658 | 4 | 1.2% | {'10': 4, '5': 3, '1': 8, '0': 23} |
| same_original_only_missing | 24 | 1 | 0.042 | 0 | 0.0% | {'1': 1, '0': 23} |
| valid_original_seen_as_train_query_has_same | 266 | 1658 | 6.233 | 156 | 48.3% | {'10': 156, '5': 10, '1': 48, '0': 52} |
| valid_query_seen_as_train_original_has_same | 329 | 982 | 2.985 | 88 | 27.2% | {'10': 88, '5': 7, '1': 67, '0': 167} |

## beam4_lr5e-4
- raw: 3528 / 10000; exact10=318; extractable=998
- config: `{'stage_a_lr': 0.0005, 'stage_a_epochs': 8.0, 'num_beams': 4, 'checkpoint_eval_num_beams': 4, 'lora_target_modules': ['c_attn', 'c_proj', 'c_fc'], 'lora_r': 32, 'lora_alpha': 64}`
- selected: `{'label': 'epoch_08', 'epoch': 8.0, 'raw_score': 3528, 'exact10': 318}`
- checkpoint curve: epoch_01=1179, epoch_02=1572, epoch_03=2322, epoch_04=3038, epoch_05=3259, epoch_06=3458, epoch_07=3486, epoch_08=3528

| group | n | raw | score/10 | exact10 | exact10 share | buckets |
|---|---:|---:|---:|---:|---:|---|
| direct_query_has_same | 10 | 72 | 7.200 | 7 | 2.2% | {'10': 7, '1': 2, '0': 1} |
| same_original_has_same | 764 | 3244 | 4.246 | 297 | 93.4% | {'10': 297, '5': 24, '1': 154, '0': 289} |
| same_original_only_conflict | 174 | 224 | 1.287 | 16 | 5.0% | {'10': 16, '5': 4, '1': 44, '0': 110} |
| same_original_no_match | 38 | 60 | 1.579 | 5 | 1.6% | {'10': 5, '1': 10, '0': 23} |
| same_original_only_missing | 24 | 0 | 0.000 | 0 | 0.0% | {'0': 24} |
| valid_original_seen_as_train_query_has_same | 266 | 1641 | 6.169 | 155 | 48.7% | {'10': 155, '5': 9, '1': 46, '0': 56} |
| valid_query_seen_as_train_original_has_same | 329 | 913 | 2.775 | 80 | 25.2% | {'10': 80, '5': 10, '1': 63, '0': 176} |

