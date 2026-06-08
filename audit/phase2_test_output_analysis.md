# Phase 2 Test Output Analysis

- test_n: 1000
- v22_solver_labeled_rows: 974

## Output Sanity
| run | n | unique_ids | extractable | numeric | digit | anchor | len_median | top answers |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v17_phase2 | 1000 | 1000 | 999 | 994 | 1000 | 999 | 13.0 | 5:84, 4:45, 50:34, 10:32, 22:29 |
| v21_phase2 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 13.0 | 20:45, 10:37, 60:34, 6:34, 5:32 |
| v22_v17 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 13.0 | 5:36, 10:27, 100:24, 20:23, 8:23 |
| v22_v21 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 13.0 | 5:34, 10:27, 100:24, 20:23, 2:23 |

## Agreement With V22 Solver Labels
This is not official scoring. It only compares runs on rows where the deterministic v22 solver produced a high-confidence query-derived answer.

| run | n | raw_vs_solver | score10_vs_solver | exact10 | buckets |
|---|---:|---:|---:|---:|---|
| v17_phase2 | 974 | 1156 | 1.187 | 73 | {10: 73, 5: 38, 1: 236, 0: 627} |
| v21_phase2 | 974 | 1127 | 1.157 | 79 | {10: 79, 5: 26, 1: 207, 0: 662} |
| v22_v17 | 974 | 9730 | 9.990 | 973 | {10: 973, 0: 1} |
| v22_v21 | 974 | 9730 | 9.990 | 973 | {10: 973, 0: 1} |

## Source / Selection Metadata
### v17_phase2
- ensemble_or_gate_report.json: {"strategy": "legal_answer_only_model_consensus"}
### v21_phase2
- ensemble_or_gate_report.json: {"strategy": "v21_safe_select78_strict_verifier_sweep", "source_counts": {"model": 994, "retrieval_group": 150, "query_arithmetic": 45}, "selected_candidate": "model::seed_42::epoch_08", "selected_epoch": 8.0, "selected_profile": {"name": "balanced", "priority": 1, "retrieval_direct_min_sim": 0.88, "retrieval_direct_min_jaccard": 0.64, "retrieval_direct_min_majority_frac": 0.5, "retrieval_direct_min_margin": 0.12, "retrieval_direct_confidence": 0.94, "verifier_safe_confidence": 0.88, "verifier_min_confidence": 0.8, "verifier_logprob_tolerance": 0.45, "verifier_logprob_override_margin": 0.48, "allow_model_missing": true}, "uses_valid_labels_for_checkpoint_selection": true, "uses_valid_labels_for_verifier_profile_selection": true}
- selected_checkpoint_info.json: {"summary": {"n": 1000, "raw_score": 6347, "max_raw_score": 10000, "score_10": 6.347, "score_pct": 0.6347, "extractable": 1000, "numeric_pairs": 980, "buckets": {"10": 616, "5": 13, "1": 122, "0": 249}, "rel_error_mean": 36.68439814830564}}
- checkpoint_selection_report.json: {"strategy": "v21_safe_select78_strict_verifier_sweep", "uses_valid_labels_for_checkpoint_selection": true, "uses_valid_labels_for_verifier_profile_selection": true, "num_candidates": 2, "candidate_summaries": [{"name": "model::seed_42::epoch_07", "epoch": 7.0, "profile": "balanced", "summary": {"n": 1000, "raw_score": 6317, "max_raw_score": 10000, "score_10": 6.317, "score_pct": 0.6317, "extractable": 1000, "numeric_pairs": 980, "buckets": {"10": 614, "5": 12, "1": 117, "0": 257}, "rel_error_mean": 37.94834948893446}}, {"name": "model::seed_42::epoch_08", "epoch": 8.0, "profile": "balanced", "summary": {"n": 1000, "raw_score": 6347, "max_raw_score": 10000, "score_10": 6.347, "score_pct": 0.6347, "extractable": 1000, "numeric_pairs": 980, "buckets": {"10": 616, "5": 13, "1": 122, "0": 249}, "rel_error_mean": 36.68439814830564}}]}
- selected_valid_report.json: {"summary": {"n": 1000, "raw_score": 6347, "max_raw_score": 10000, "score_10": 6.347, "score_pct": 0.6347, "extractable": 1000, "numeric_pairs": 980, "buckets": {"10": 616, "5": 13, "1": 122, "0": 249}, "rel_error_mean": 36.68439814830564}}
### v22_v17
- ensemble_or_gate_report.json: {"strategy": "v22_solver_overlay", "fallback_strategy": "v17_lr3e3_epoch7_answer_only_consensus", "source_counts": {"solver": 974, "fallback": 26}, "solver_rule_counts": {"set_union_two_sports": 39, "orange_remainder": 39, "floor_plus_ceil": 39, "linear_equation_ax_plus_b": 39, "books_pens_linear": 38, "free_shipping_ceil": 38, "fourth_score_average": 38, "rice_remaining_after_two_days": 38, "good_products_remaining": 38, "weighted_average_cart": 38, "feet_to_yard": 38, "geometric_years": 38, "probability_not_defective": 38, "compound_growth_initial": 38, "simple_interest_principal": 38, "board_game_remaining": 38, "ratio_difference": 38, "combination_inverse": 37, "discount_original_price": 37, "arithmetic_sequence_sum": 36, "max_lcm_with_base": 36, "isosceles_perimeter": 35, "fair_die_expectation": 5, "three_coins_at_least_one_head": 3, "money_left_after_spending": 2, "icecream_multiset_combinations": 2, "decimal_subtraction_comma": 2, "defined_operator_a2_ab_b2": 1, "basketball_lineup_permutation": 1, "parabola_y_intercepts_count": 1, "positive_integer_reciprocal_pairs": 1, "clock_hand_inverse_minutes": 1, "quadratic_discriminant": 1, "land_sale_profit": 1, "non_black_cows": 1, 
### v22_v21
- ensemble_or_gate_report.json: {"strategy": "v22_solver_overlay", "fallback_strategy": "v21_select78_strict_verifier_sweep", "source_counts": {"solver": 974, "fallback": 26}, "solver_rule_counts": {"set_union_two_sports": 39, "orange_remainder": 39, "floor_plus_ceil": 39, "linear_equation_ax_plus_b": 39, "books_pens_linear": 38, "free_shipping_ceil": 38, "fourth_score_average": 38, "rice_remaining_after_two_days": 38, "good_products_remaining": 38, "weighted_average_cart": 38, "feet_to_yard": 38, "geometric_years": 38, "probability_not_defective": 38, "compound_growth_initial": 38, "simple_interest_principal": 38, "board_game_remaining": 38, "ratio_difference": 38, "combination_inverse": 37, "discount_original_price": 37, "arithmetic_sequence_sum": 36, "max_lcm_with_base": 36, "isosceles_perimeter": 35, "fair_die_expectation": 5, "three_coins_at_least_one_head": 3, "money_left_after_spending": 2, "icecream_multiset_combinations": 2, "decimal_subtraction_comma": 2, "defined_operator_a2_ab_b2": 1, "basketball_lineup_permutation": 1, "parabola_y_intercepts_count": 1, "positive_integer_reciprocal_pairs": 1, "clock_hand_inverse_minutes": 1, "quadratic_discriminant": 1, "land_sale_profit": 1, "non_black_cows": 1, "fun
- selected_checkpoint_info.json: {"summary": {"n": 1000, "raw_score": 5861, "max_raw_score": 10000, "score_10": 5.861, "score_pct": 0.5861, "extractable": 1000, "numeric_pairs": 980, "buckets": {"10": 562, "5": 21, "1": 136, "0": 281}, "rel_error_mean": 5.546036864158181}}
- checkpoint_selection_report.json: {"strategy": "v21_safe_select78_strict_verifier_sweep", "uses_valid_labels_for_checkpoint_selection": true, "uses_valid_labels_for_verifier_profile_selection": true, "num_candidates": 2, "candidate_summaries": [{"name": "model::seed_42::epoch_07", "epoch": 7.0, "profile": "balanced", "summary": {"n": 1000, "raw_score": 5623, "max_raw_score": 10000, "score_10": 5.623, "score_pct": 0.5623, "extractable": 1000, "numeric_pairs": 980, "buckets": {"10": 539, "5": 18, "1": 143, "0": 300}, "rel_error_mean": 5.4205408329034706}}, {"name": "model::seed_42::epoch_08", "epoch": 8.0, "profile": "balanced", "summary": {"n": 1000, "raw_score": 5861, "max_raw_score": 10000, "score_10": 5.861, "score_pct": 0.5861, "extractable": 1000, "numeric_pairs": 980, "buckets": {"10": 562, "5": 21, "1": 136, "0": 281}, "rel_error_mean": 5.546036864158181}}]}
- selected_valid_report.json: {"summary": {"n": 1000, "raw_score": 5861, "max_raw_score": 10000, "score_10": 5.861, "score_pct": 0.5861, "extractable": 1000, "numeric_pairs": 980, "buckets": {"10": 562, "5": 21, "1": 136, "0": 281}, "rel_error_mean": 5.546036864158181}}

## Pairwise Numeric Agreement
- v17_phase2__vs__v21_phase2: same_num=77/994, same_text=77/1000
- v17_phase2__vs__v22_v17: same_num=80/994, same_text=78/1000
- v17_phase2__vs__v22_v21: same_num=74/994, same_text=72/1000
- v21_phase2__vs__v22_v17: same_num=86/1000, same_text=82/1000
- v21_phase2__vs__v22_v21: same_num=87/1000, same_text=83/1000
- v22_v17__vs__v22_v21: same_num=983/1000, same_text=983/1000

## Model Failures Against Solver Labels
### v17_phase2
- id=0 type=Synthetic_Generalization rule=arithmetic_sequence_sum solver=407 pred=22 | Một cấp số cộng có số hạng đầu là 8, công sai là 1. Tổng 22 số hạng đầu tiên là bao nhiêu?
- id=1 type=GSM_Rephrased rule=set_union_two_sports solver=72 pred=7 | Trong một lớp, có 62 học sinh thích bóng đá, 27 học sinh thích bóng rổ, và 17 học sinh thích cả hai môn. Hỏi có bao nhiêu học sinh thích ít nhất một trong hai môn? Có 6 học sinh ng
- id=2 type=GSM_SV rule=books_pens_linear solver=20 pred=4 | Lan mua 7 quyển sách, mỗi quyển giá 25 nghìn đồng, và một số cây bút, mỗi cây giá 8 nghìn đồng. Tổng tiền Lan trả là 335 nghìn đồng. Hỏi Lan mua bao nhiêu cây bút? Bạn của Lan mua 
- id=3 type=Synthetic_Generalization rule=arithmetic_sequence_sum solver=477 pred=6 | Một cấp số cộng có số hạng đầu là 1, công sai là 3. Tổng 18 số hạng đầu tiên là bao nhiêu?
- id=4 type=GSM_Rephrased rule=set_union_two_sports solver=125 pred=7 | Trong một lớp, có 73 học sinh thích bóng đá, 69 học sinh thích bóng rổ, và 17 học sinh thích cả hai môn. Hỏi có bao nhiêu học sinh thích ít nhất một trong hai môn? Có 8 học sinh ng
- id=5 type=Synthetic_Generalization rule=max_lcm_with_base solver=170 pred=580 | Xác định giá trị lớn nhất trong các bội chung nhỏ nhất của 10 với các số 6, 9, 12, 14, 16, 17. Hãy trả lời bằng một số nguyên.
- id=9 type=MATH_Rephrased rule=basketball_lineup_permutation solver=95040 pred=95 | Có bao nhiêu cách khác nhau để chọn đội hình xuất phát bao gồm trung vệ, tiền đạo, tiền đạo sút, người bảo vệ điểm và người bảo vệ bắn từ một đội bóng rổ gồm 12 thành viên, trong đ
- id=11 type=MATH_AnsAug rule=parabola_y_intercepts_count solver=0 pred=7 | Đồ thị của parabol $x = 2y^2 - 3y + 7$ có bao nhiêu giao điểm $y$?
- id=12 type=Synthetic_Generalization rule=arithmetic_sequence_sum solver=232 pred=27 | Một cấp số cộng có số hạng đầu là 15, công sai là 4. Tổng 8 số hạng đầu tiên là bao nhiêu?
- id=14 type=MATH_SV rule=combination_inverse solver=19 pred=168 | Một nhóm có n người. Số cách chọn 2 người từ nhóm là 171. Hỏi n bằng bao nhiêu? Buổi họp bắt đầu lúc 2 giờ, nhưng thời gian này không liên quan.
- id=16 type=GSM_Rephrased rule=rice_remaining_after_two_days solver=192 pred=325 | Một kho có 360 kg gạo. Ngày thứ nhất dùng 1/5 số gạo. Ngày thứ hai dùng 1/3 số gạo còn lại. Hỏi sau hai ngày còn lại bao nhiêu kg gạo? Ngày thứ ba dự kiến nhập thêm 28 kg, nhưng ch
- id=17 type=MATH_SV rule=combination_inverse solver=7 pred=56 | Một nhóm có n người. Số cách chọn 2 người từ nhóm là 21. Hỏi n bằng bao nhiêu? Buổi họp bắt đầu lúc 3 giờ, nhưng thời gian này không liên quan.
- id=18 type=GSM_SV rule=discount_original_price solver=160 pred=44000 | Sau khi giảm giá 25%, một món hàng còn giá 120 nghìn đồng. Hỏi giá ban đầu của món hàng là bao nhiêu nghìn đồng? Cửa hàng còn treo thêm biển giảm 45% cho một sản phẩm khác, nhưng b
- id=21 type=MATH_FOBAR rule=clock_hand_inverse_minutes solver=1800 pred=10 | Kim giây của đồng hồ trong hình dưới đây dài 6 cm. Đầu của kim giây này di chuyển được bao xa tính bằng cm trong khoảng thời gian X phút? Hãy thể hiện câu trả lời của bạn dưới dạng
- id=23 type=GSM_Rephrased rule=orange_remainder solver=10 pred=78 | Có 234 quả cam, mỗi hộp đựng được 32 quả. Sau khi đóng đầy nhiều hộp nhất có thể, còn thừa bao nhiêu quả cam? Ngày hôm sau cửa hàng nhập thêm 8 quả cam, nhưng số cam này không tính
### v21_phase2
- id=0 type=Synthetic_Generalization rule=arithmetic_sequence_sum solver=407 pred=181 | Một cấp số cộng có số hạng đầu là 8, công sai là 1. Tổng 22 số hạng đầu tiên là bao nhiêu?
- id=1 type=GSM_Rephrased rule=set_union_two_sports solver=72 pred=17 | Trong một lớp, có 62 học sinh thích bóng đá, 27 học sinh thích bóng rổ, và 17 học sinh thích cả hai môn. Hỏi có bao nhiêu học sinh thích ít nhất một trong hai môn? Có 6 học sinh ng
- id=2 type=GSM_SV rule=books_pens_linear solver=20 pred=12000 | Lan mua 7 quyển sách, mỗi quyển giá 25 nghìn đồng, và một số cây bút, mỗi cây giá 8 nghìn đồng. Tổng tiền Lan trả là 335 nghìn đồng. Hỏi Lan mua bao nhiêu cây bút? Bạn của Lan mua 
- id=3 type=Synthetic_Generalization rule=arithmetic_sequence_sum solver=477 pred=37 | Một cấp số cộng có số hạng đầu là 1, công sai là 3. Tổng 18 số hạng đầu tiên là bao nhiêu?
- id=4 type=GSM_Rephrased rule=set_union_two_sports solver=125 pred=35 | Trong một lớp, có 73 học sinh thích bóng đá, 69 học sinh thích bóng rổ, và 17 học sinh thích cả hai môn. Hỏi có bao nhiêu học sinh thích ít nhất một trong hai môn? Có 8 học sinh ng
- id=5 type=Synthetic_Generalization rule=max_lcm_with_base solver=170 pred=15 | Xác định giá trị lớn nhất trong các bội chung nhỏ nhất của 10 với các số 6, 9, 12, 14, 16, 17. Hãy trả lời bằng một số nguyên.
- id=9 type=MATH_Rephrased rule=basketball_lineup_permutation solver=95040 pred=95.04 | Có bao nhiêu cách khác nhau để chọn đội hình xuất phát bao gồm trung vệ, tiền đạo, tiền đạo sút, người bảo vệ điểm và người bảo vệ bắn từ một đội bóng rổ gồm 12 thành viên, trong đ
- id=10 type=GSM_Rephrased rule=free_shipping_ceil solver=5 pred=46000 | Một cửa hàng miễn phí giao hàng nếu tiền hàng đạt ít nhất 120 nghìn đồng. Mỗi sản phẩm giá 25 nghìn đồng. Hỏi cần mua ít nhất bao nhiêu sản phẩm để được miễn phí giao hàng? Nếu khô
- id=11 type=MATH_AnsAug rule=parabola_y_intercepts_count solver=0 pred=11 | Đồ thị của parabol $x = 2y^2 - 3y + 7$ có bao nhiêu giao điểm $y$?
- id=14 type=MATH_SV rule=combination_inverse solver=19 pred=5 | Một nhóm có n người. Số cách chọn 2 người từ nhóm là 171. Hỏi n bằng bao nhiêu? Buổi họp bắt đầu lúc 2 giờ, nhưng thời gian này không liên quan.
- id=15 type=GSM_SV rule=fourth_score_average solver=58 pred=28 | Ba điểm kiểm tra của Minh là 14, 43 và 33. Minh cần điểm trung bình bốn bài bằng 37. Hỏi bài kiểm tra thứ tư Minh cần đạt bao nhiêu điểm? Giáo viên nói bài thứ năm sẽ được cộng thê
- id=16 type=GSM_Rephrased rule=rice_remaining_after_two_days solver=192 pred=60 | Một kho có 360 kg gạo. Ngày thứ nhất dùng 1/5 số gạo. Ngày thứ hai dùng 1/3 số gạo còn lại. Hỏi sau hai ngày còn lại bao nhiêu kg gạo? Ngày thứ ba dự kiến nhập thêm 28 kg, nhưng ch
- id=18 type=GSM_SV rule=discount_original_price solver=160 pred=595000 | Sau khi giảm giá 25%, một món hàng còn giá 120 nghìn đồng. Hỏi giá ban đầu của món hàng là bao nhiêu nghìn đồng? Cửa hàng còn treo thêm biển giảm 45% cho một sản phẩm khác, nhưng b
- id=19 type=GSM_Rephrased rule=free_shipping_ceil solver=8 pred=1000 | Một cửa hàng miễn phí giao hàng nếu tiền hàng đạt ít nhất 200 nghìn đồng. Mỗi sản phẩm giá 25 nghìn đồng. Hỏi cần mua ít nhất bao nhiêu sản phẩm để được miễn phí giao hàng? Nếu khô
- id=21 type=MATH_FOBAR rule=clock_hand_inverse_minutes solver=1800 pred=30 | Kim giây của đồng hồ trong hình dưới đây dài 6 cm. Đầu của kim giây này di chuyển được bao xa tính bằng cm trong khoảng thời gian X phút? Hãy thể hiện câu trả lời của bạn dưới dạng
