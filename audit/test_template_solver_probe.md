# Test Template Solver Probe

- n=1000
- solved=836 (83.6%)

## By Rule
- floor_plus_ceil: 39
- linear_equation_ax_plus_b: 39
- orange_remainder: 39
- set_union_two_sports: 39
- board_game_remaining: 38
- books_pens_linear: 38
- compound_growth_initial: 38
- feet_to_yard: 38
- fourth_score_average: 38
- free_shipping_ceil: 38
- geometric_years: 38
- good_products_remaining: 38
- probability_not_defective: 38
- ratio_difference: 38
- rice_remaining_after_two_days: 38
- simple_interest_principal: 38
- weighted_average_cart: 38
- combination_inverse: 37
- discount_original_price: 37
- arithmetic_sequence_sum: 36
- max_lcm_with_base: 36
- isosceles_perimeter: 35
- fair_die_expectation: 5

## By Type
- GSM_AnsAug: 0/27
- GSM_FOBAR: 39/44
- GSM_Rephrased: 306/325
- GSM_SV: 189/194
- MATH_AnsAug: 0/71
- MATH_FOBAR: 0/5
- MATH_Rephrased: 115/142
- MATH_SV: 75/80
- Synthetic_Generalization: 112/112

## Sample Solved
- id=0 type=Synthetic_Generalization rule=arithmetic_sequence_sum answer=407 | Một cấp số cộng có số hạng đầu là 8, công sai là 1. Tổng 22 số hạng đầu tiên là bao nhiêu?
- id=1 type=GSM_Rephrased rule=set_union_two_sports answer=72 | Trong một lớp, có 62 học sinh thích bóng đá, 27 học sinh thích bóng rổ, và 17 học sinh thích cả hai môn. Hỏi có bao nhiêu học sinh thích ít nhất một trong hai môn? Có 6 học sinh nghỉ học hôm đó, nhưng không tính vào các số liệu trên.
- id=2 type=GSM_SV rule=books_pens_linear answer=20 | Lan mua 7 quyển sách, mỗi quyển giá 25 nghìn đồng, và một số cây bút, mỗi cây giá 8 nghìn đồng. Tổng tiền Lan trả là 335 nghìn đồng. Hỏi Lan mua bao nhiêu cây bút? Bạn của Lan mua thêm 5 cục tẩy, nhưng không tính vào tổng tiền của Lan.
- id=3 type=Synthetic_Generalization rule=arithmetic_sequence_sum answer=477 | Một cấp số cộng có số hạng đầu là 1, công sai là 3. Tổng 18 số hạng đầu tiên là bao nhiêu?
- id=4 type=GSM_Rephrased rule=set_union_two_sports answer=125 | Trong một lớp, có 73 học sinh thích bóng đá, 69 học sinh thích bóng rổ, và 17 học sinh thích cả hai môn. Hỏi có bao nhiêu học sinh thích ít nhất một trong hai môn? Có 8 học sinh nghỉ học hôm đó, nhưng không tính vào các số liệu trên.
- id=5 type=Synthetic_Generalization rule=max_lcm_with_base answer=170 | Xác định giá trị lớn nhất trong các bội chung nhỏ nhất của 10 với các số 6, 9, 12, 14, 16, 17. Hãy trả lời bằng một số nguyên.
- id=7 type=Synthetic_Generalization rule=isosceles_perimeter answer=61 | Một tam giác cân có hai cạnh bằng nhau dài 24 cm và cạnh đáy dài 13 cm. Chu vi tam giác là bao nhiêu cm?
- id=8 type=Synthetic_Generalization rule=isosceles_perimeter answer=41 | Một tam giác cân có hai cạnh bằng nhau dài 18 cm và cạnh đáy dài 5 cm. Chu vi tam giác là bao nhiêu cm?
- id=10 type=GSM_Rephrased rule=free_shipping_ceil answer=5 | Một cửa hàng miễn phí giao hàng nếu tiền hàng đạt ít nhất 120 nghìn đồng. Mỗi sản phẩm giá 25 nghìn đồng. Hỏi cần mua ít nhất bao nhiêu sản phẩm để được miễn phí giao hàng? Nếu không đủ điều kiện thì phí giao hàng là 15 nghìn đồng, nhưng câu hỏi chỉ hỏi số sản phẩm tối thiểu.
- id=12 type=Synthetic_Generalization rule=arithmetic_sequence_sum answer=232 | Một cấp số cộng có số hạng đầu là 15, công sai là 4. Tổng 8 số hạng đầu tiên là bao nhiêu?
- id=14 type=MATH_SV rule=combination_inverse answer=19 | Một nhóm có n người. Số cách chọn 2 người từ nhóm là 171. Hỏi n bằng bao nhiêu? Buổi họp bắt đầu lúc 2 giờ, nhưng thời gian này không liên quan.
- id=15 type=GSM_SV rule=fourth_score_average answer=58 | Ba điểm kiểm tra của Minh là 14, 43 và 33. Minh cần điểm trung bình bốn bài bằng 37. Hỏi bài kiểm tra thứ tư Minh cần đạt bao nhiêu điểm? Giáo viên nói bài thứ năm sẽ được cộng thêm 8 điểm thưởng, nhưng bài đó không tính vào trung bình này.
- id=16 type=GSM_Rephrased rule=rice_remaining_after_two_days answer=192 | Một kho có 360 kg gạo. Ngày thứ nhất dùng 1/5 số gạo. Ngày thứ hai dùng 1/3 số gạo còn lại. Hỏi sau hai ngày còn lại bao nhiêu kg gạo? Ngày thứ ba dự kiến nhập thêm 28 kg, nhưng chưa nhập trong bài toán này.
- id=17 type=MATH_SV rule=combination_inverse answer=7 | Một nhóm có n người. Số cách chọn 2 người từ nhóm là 21. Hỏi n bằng bao nhiêu? Buổi họp bắt đầu lúc 3 giờ, nhưng thời gian này không liên quan.
- id=18 type=GSM_SV rule=discount_original_price answer=160 | Sau khi giảm giá 25%, một món hàng còn giá 120 nghìn đồng. Hỏi giá ban đầu của món hàng là bao nhiêu nghìn đồng? Cửa hàng còn treo thêm biển giảm 45% cho một sản phẩm khác, nhưng biển này không áp dụng cho món hàng đang hỏi.
- id=19 type=GSM_Rephrased rule=free_shipping_ceil answer=8 | Một cửa hàng miễn phí giao hàng nếu tiền hàng đạt ít nhất 200 nghìn đồng. Mỗi sản phẩm giá 25 nghìn đồng. Hỏi cần mua ít nhất bao nhiêu sản phẩm để được miễn phí giao hàng? Nếu không đủ điều kiện thì phí giao hàng là 10 nghìn đồng, nhưng câu hỏi chỉ hỏi số sản phẩm tối thiểu.
- id=22 type=Synthetic_Generalization rule=isosceles_perimeter answer=23 | Một tam giác cân có hai cạnh bằng nhau dài 10 cm và cạnh đáy dài 3 cm. Chu vi tam giác là bao nhiêu cm?
- id=23 type=GSM_Rephrased rule=orange_remainder answer=10 | Có 234 quả cam, mỗi hộp đựng được 32 quả. Sau khi đóng đầy nhiều hộp nhất có thể, còn thừa bao nhiêu quả cam? Ngày hôm sau cửa hàng nhập thêm 8 quả cam, nhưng số cam này không tính cho câu hỏi trên.
- id=24 type=MATH_SV rule=combination_inverse answer=13 | Một nhóm có n người. Số cách chọn 2 người từ nhóm là 78. Hỏi n bằng bao nhiêu? Buổi họp bắt đầu lúc 8 giờ, nhưng thời gian này không liên quan.
- id=25 type=Synthetic_Generalization rule=max_lcm_with_base answer=234 | Xác định giá trị lớn nhất trong các bội chung nhỏ nhất của 18 với các số 2, 6, 10, 12, 13, 14. Hãy trả lời bằng một số nguyên.
- id=26 type=GSM_Rephrased rule=good_products_remaining answer=120 | Một cửa hàng ban đầu có 199 sản phẩm. Buổi sáng bán được 55 sản phẩm và phát hiện 24 sản phẩm bị hỏng. Hỏi trước khi nhập thêm hàng, cửa hàng còn bao nhiêu sản phẩm tốt? Buổi chiều cửa hàng nhập thêm 30 sản phẩm mới, nhưng không tính vào câu hỏi.
- id=27 type=GSM_Rephrased rule=orange_remainder answer=6 | Có 604 quả cam, mỗi hộp đựng được 13 quả. Sau khi đóng đầy nhiều hộp nhất có thể, còn thừa bao nhiêu quả cam? Ngày hôm sau cửa hàng nhập thêm 19 quả cam, nhưng số cam này không tính cho câu hỏi trên.
- id=28 type=GSM_SV rule=fourth_score_average answer=26 | Ba điểm kiểm tra của Minh là 45, 19 và 18. Minh cần điểm trung bình bốn bài bằng 27. Hỏi bài kiểm tra thứ tư Minh cần đạt bao nhiêu điểm? Giáo viên nói bài thứ năm sẽ được cộng thêm 7 điểm thưởng, nhưng bài đó không tính vào trung bình này.
- id=29 type=GSM_Rephrased rule=weighted_average_cart answer=65/2 | Một giỏ hàng có 4 món loại A, mỗi món giá 25 nghìn đồng, và 4 món loại B, mỗi món giá 40 nghìn đồng. Hỏi giá trung bình của một món trong giỏ là bao nhiêu nghìn đồng? Người bán giảm thêm 10 nghìn đồng cho đơn hàng khác, không liên quan đến giỏ này.
- id=31 type=MATH_Rephrased rule=floor_plus_ceil answer=17 | Tính $\lfloor -9.1 \rfloor + \lceil 26.3 \rceil$. Số 88 chỉ là mã bài và không tham gia vào phép tính.
- id=32 type=GSM_Rephrased rule=good_products_remaining answer=28 | Một cửa hàng ban đầu có 61 sản phẩm. Buổi sáng bán được 23 sản phẩm và phát hiện 10 sản phẩm bị hỏng. Hỏi trước khi nhập thêm hàng, cửa hàng còn bao nhiêu sản phẩm tốt? Buổi chiều cửa hàng nhập thêm 70 sản phẩm mới, nhưng không tính vào câu hỏi.
- id=33 type=Synthetic_Generalization rule=arithmetic_sequence_sum answer=81 | Một cấp số cộng có số hạng đầu là 5, công sai là 1. Tổng 9 số hạng đầu tiên là bao nhiêu?
- id=34 type=Synthetic_Generalization rule=max_lcm_with_base answer=240 | Xác định giá trị lớn nhất trong các bội chung nhỏ nhất của 15 với các số 3, 9, 14, 15, 16, 18. Hãy trả lời bằng một số nguyên.
- id=35 type=MATH_Rephrased rule=feet_to_yard answer=40 | Một người đi được 120 feet với tốc độ 5 feet mỗi phút. Hỏi quãng đường đó bằng bao nhiêu yard, biết 1 yard = 3 feet? Thời gian đi là 24 phút và số 9 là ghi chú, nhưng câu hỏi chỉ hỏi yard.
- id=36 type=GSM_SV rule=fourth_score_average answer=59 | Ba điểm kiểm tra của Minh là 41, 16 và 44. Minh cần điểm trung bình bốn bài bằng 40. Hỏi bài kiểm tra thứ tư Minh cần đạt bao nhiêu điểm? Giáo viên nói bài thứ năm sẽ được cộng thêm 4 điểm thưởng, nhưng bài đó không tính vào trung bình này.
