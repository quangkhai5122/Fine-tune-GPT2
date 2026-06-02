# V9/V10 Results Analysis

## Summary
- v9: source-valid score=1.392, reference valid score=1.842, selected=stage_a_compact_equation_lora_epoch_05
- v10: source-valid score=1.396, reference valid score=2.131, selected=stage_c_mixed_replay_lora_epoch_02

## Output Shape
- v9/source_valid: {'equation_plus_answer': 857, 'no_answer_anchor': 19, 'answer_only': 123, 'other_with_answer_anchor': 1}, median_len=76.0, mean_equals=2.95
- v9/valid: {'equation_plus_answer': 910, 'answer_only': 83, 'no_answer_anchor': 5, 'other_with_answer_anchor': 2}, median_len=72.0, mean_equals=2.97
- v10/source_valid: {'answer_only': 714, 'equation_plus_answer': 284, 'no_answer_anchor': 2}, median_len=13.0, mean_equals=0.92
- v10/valid: {'answer_only': 641, 'equation_plus_answer': 358, 'no_answer_anchor': 1}, median_len=14.0, mean_equals=1.17

## Score Transitions V9 -> V10
- 0->0: 404
- 0->1: 134
- 1->0: 83
- 0->10: 82
- 1->1: 75
- 10->10: 67
- 10->0: 49
- 10->1: 32
- 1->10: 23
- 0->5: 12
- 5->0: 12
- 5->1: 10
- 1->5: 6
- 5->5: 6
- 5->10: 3
- 10->5: 2

## Representative Cases
### v10_fix idx=37 type=MATH_AnsAug
- Q: Bán kính của một hình cầu là đơn vị $p$ và bán kính của một bán cầu là đơn vị $2p$. Tỉ số giữa thể tích của hình cầu và thể tích của bán cầu là bao nhiêu?
- Gold: \frac{1}{4}
- v9: score=0 pred=113.994 err=113.744 | Phép tính: ch : ( 4 ) / ( 3 ) pi r ^ 3; ( 4pi ) / ( 9 ) | Đáp án là: 113.994
- v10: score=10 pred=0.25 err=0.0 | Đáp án là: 0.25

### v10_fix idx=42 type=MATH_Rephrased
- Q: Nếu sáu số nguyên riêng biệt được chọn ngẫu nhiên từ tập hợp {1, 2, 3, ..., 10} thì xác suất để số nguyên nhỏ thứ hai trong số các số được chọn là 3 là bao nhiêu?
- Gold: \frac{1}{3}
- v9: score=1 pred=0.125 err=0.20833333333333331 | Phép tính: 2 ^ 6 = 128 k; ( 1 + 2 + 3 + \dots + 1 ) / ( 2 ^ 6 ) = ( 4 + 6 + 9 ) / ( 8 ) = ( 25 ) / ( 64 ) = \boxed; ( 5 ) / ( 4 ) | Đáp án là: 0.125
- v10: score=10 pred=0.333333 err=3.3333333332441484e-07 | Đáp án là: 0.333333

### v10_fix idx=53 type=MATH_AnsAug
- Q: Tìm bội số chung nhỏ nhất của 24 và 90.
- Gold: 360
- v9: score=0 pred=120 err=0.6666666666666666 | Đáp án là: 120
- v10: score=10 pred=360 err=0.0 | Đáp án là: 360

### v10_fix idx=88 type=MATH_Rephrased
- Q: Giá trị của $\log_{10}{28471}$ nằm giữa hai số nguyên liên tiếp $a$ và $b$. Tính tổng của $a$ và $b$.
- Gold: 9
- v9: score=1 pred=7 err=0.2222222222222222 | Phép tính: 10 ^ a; 10 ^ b; 10 ^ 4; a = 4 v; b = 5; a + b = 4 + 5 = 7 | Đáp án là: 7
- v10: score=10 pred=9 err=0.0 | Đáp án là: 9

### v10_fix idx=96 type=GSM_SV
- Q: Một người chuyển phát nhanh đã nhận được x gói hàng ngày hôm qua và số lượng gấp đôi ngày hôm nay. Tất cả những thứ này sẽ được giao vào ngày mai. 240 gói hàng sẽ được giao vào ngày mai. Giá trị của biến x chưa biết...
- Gold: 80
- v9: score=1 pred=120 err=0.5 | Phép tính: x * 2 = 240 H; x : 2x = 240; 2 : 2x / 2 = 240 / 2 x = 120 | Đáp án là: 120
- v10: score=10 pred=80 err=0.0 | Phép tính: x + 2x = 240 H; x : 3x = 240; 3 : 3x / 3 = 240 / 3 x = 80 | Đáp án là: 80

### v10_regress idx=11 type=GSM_Rephrased
- Q: Nếu con chó của Sandra sinh ra 7 chú chó con và bác sĩ thú y của cô ấy đã cung cấp cho cô ấy 105 phần sữa công thức để cho các chú chó con ăn trong 5 ngày, thì Sandra nên cho mỗi chú chó con ăn bao nhiêu lần một ngày?
- Gold: 3
- v9: score=10 pred=3 err=0.0 | Phép tính: 105 / 5 = 21 l; 21 / 7 = 3 l | Đáp án là: 3
- v10: score=1 pred=2 err=0.3333333333333333 | Đáp án là: 2

### v10_regress idx=36 type=MATH_Rephrased
- Q: Chandra có bốn chiếc bát, mỗi chiếc có một màu khác nhau (đỏ, xanh dương, vàng và xanh lá cây), và cô ấy cũng có một chiếc ly phù hợp với màu của mỗi chiếc bát. Nếu cô ấy chọn một cái bát và một cái ly từ tủ của mình...
- Gold: 16
- v9: score=10 pred=16 err=0.0 | Phép tính: 4 \time 4 = 16 c | Đáp án là: 16
- v10: score=0 pred=6 err=0.625 | Đáp án là: 6

### v10_regress idx=44 type=MATH_AnsAug
- Q: Tìm giá trị của $a$ sao cho các đường thẳng có phương trình đã cho vuông góc với nhau. \begin{align*} y &= 2x+13 \\ 6y+ax &= 6. \end{align*}
- Gold: 3
- v9: score=10 pred=3 err=0.0 | Phép tính: 2x + 13; 6y + x = - ( a ) / ( 6 ) + ( 6 ) / ( 2 ) = - ( 2 ) / ( 3 ) + a; 6 ( 2 ) + a = - 6; 10 + a = 6; a = 3 | Đáp án là: 3
- v10: score=0 pred=5 err=0.6666666666666666 | Đáp án là: 5

### v10_regress idx=49 type=MATH_Rephrased
- Q: Số nào khi chia cho 2 sẽ bằng kết quả khi trừ đi 2?
- Gold: 4
- v9: score=10 pred=4 err=0.0 | Phép tính: ng ( x ) / ( 2 ) = x - 2; x = 2 ( x - 2 ); x = 2x - 4; x = 4 | Đáp án là: 4
- v10: score=0 pred=1 err=0.75 | Đáp án là: 1

### v10_regress idx=56 type=GSM_AnsAug
- Q: Micah có thể gõ 20 từ mỗi phút và Isaiah có thể gõ 40 từ mỗi phút. Ê-sai có thể gõ nhiều hơn Mi-ca bao nhiêu từ trong một giờ?
- Gold: 1200
- v9: score=10 pred=1200 err=0.0 | Phép tính: t * 60; t = 1200 t | Đáp án là: 1200
- v10: score=1 pred=1800 err=0.5 | Phép tính: 20 x 60 = 1200 t; 20 + 1200 = 1800 t | Đáp án là: 1800

### both_wrong idx=7 type=GSM_FOBAR
- Q: Grace bắt đầu công việc kinh doanh cảnh quan của riêng mình. Cô tính phí 6 đô la một giờ cho việc cắt cỏ, 11 đô la cho việc nhổ cỏ và x đô la cho việc phủ lớp phủ. Vào tháng 9, cô cắt cỏ trong 63 giờ, nhổ cỏ trong 9...
- Gold: 9
- v9: score=0 pred=1 err=0.8888888888888888 | Phép tính: 6 * 63 = 294; 9 * 11 = 99; 10 * x = 10x; 294 + 99 + 10x; t : 294 + 99 , 10 + 10x = 567; c : 363 , 10 USD + 10x - 363 , 10; c : 10x - 384 , 10 = 567; 10x - 256 , 10 = 675; c : - 256 , 20 USD = 675; x = 1 | Đáp án là: 1
- v10: score=0 pred=4 err=0.5555555555555556 | Đáp án là: 4

### both_wrong idx=9 type=GSM_AnsAug
- Q: Bob được hỗ trợ tiền thuê nhà vì anh ấy có thu nhập thấp. Nếu anh ta được tăng lương 0,50 USD/giờ và làm việc 40 giờ một tuần, anh ta sẽ thực sự kiếm được bao nhiêu tiền một tuần nếu trợ cấp nhà ở của anh ta giảm đi...
- Gold: 5
- v9: score=0 pred=40 err=7.0 | Phép tính: 0 , 50 USD / gi; 40 * 0 , 5 USD = 20 USD m; 60 USD - 20 USD = 40 USD m | Đáp án là: 40
- v10: score=0 pred=2 err=0.6 | Phép tính: 40 * 0 , 60 USD = 60 USD m; 60 * 0 , 50 USD = 30 USD m; 30 USD - 60 USD = 2 USD m | Đáp án là: 2

### both_wrong idx=10 type=GSM_FOBAR
- Q: John phải thay vòng bi cho những chiếc máy mà anh ấy làm việc cùng. Anh ta có 10 chiếc máy và mỗi chiếc có 30 vòng bi. Thông thường nó có giá x $ cho mỗi ổ bi nhưng hiện tại đang có đợt giảm giá với giá chỉ 0,75 USD....
- Gold: 1
- v9: score=0 pred=200 err=199.0 | Phép tính: 10 * 30 = 300 v; 0 , 75 USD - 0 , 20 USD = 0 , 20; 0 , 20 * 300 = 60; 0 , 25 USD * x = 0 , 25x; 0 , 50 USD * x; 0 , 5 USD * x - 0 , 25; t : 0 , 50; la * x = 120; c : x = 200 | Đáp án là: 200
- v10: score=0 pred=100 err=99.0 | Đáp án là: 100

### both_wrong idx=12 type=GSM_SV
- Q: Bob đặt mua x gói sữa chua Hy Lạp từ Costco để sử dụng trong tháng. Tuy nhiên, khi nhận hàng, anh nhận ra 40% số gói đã hết hạn sử dụng. Anh quyết định trả lại những gói đó. Nếu mỗi gói có giá 12 USD thì Bob được hoàn...
- Gold: 80
- v9: score=0 pred=320 err=3.0 | Phép tính: chua : 40 % c; x - 40 % * x = 384 H; x : 0 , 4x = 384; 0 , 4 : x = 384 / 0 , 4 x = 320 | Đáp án là: 320
- v10: score=0 pred=2 err=0.975 | Đáp án là: 2

### both_wrong idx=14 type=GSM_FOBAR
- Q: Lizzie có số bút chì màu bằng một nửa Bobbie. Bobbie có số bút chì màu nhiều gấp ba lần Billie. Nếu Billie có x bút chì màu thì Lizzie có bao nhiêu bút màu? Nếu chúng ta biết câu trả lời cho câu hỏi trên là 27 thì giá...
- Gold: 18
- v9: score=0 pred=3 err=0.8333333333333334 | Phép tính: x / 2 b; 3 * ( x / 2 ) = 3x / 2; 3x + 3x = 9x; t : 9x = 27; c : x = 3 | Đáp án là: 3
- v10: score=0 pred=3 err=0.8333333333333334 | Phép tính: 3 * x; 3 * ( 3 * x ) = 9x; t : 9x = 27; c : x = 3 | Đáp án là: 3

### near_miss idx=4 type=GSM_Rephrased
- Q: Nếu có một bãi đỗ xe 1000 ô tô được chia thành 3 khu, trong đó khu 1 có 320 chỗ và khu 2 có nhiều hơn khu 3 200 chỗ thì khu 2 của bãi đậu xe có bao nhiêu chỗ?
- Gold: 440
- v9: score=5 pred=420 err=0.045454545454545456 | Phép tính: 320 + 200 = 620; 1000 - 620 = 420 | Đáp án là: 420
- v10: score=5 pred=400 err=0.09090909090909091 | Phép tính: 320 + 200 = 600; 1000 - 600 = 400 | Đáp án là: 400

### near_miss idx=21 type=GSM_Rephrased
- Q: Nếu Michael để dành 5 chiếc bánh quy của mình để tặng Sarah và Sarah để dành một phần ba trong số 9 chiếc bánh nướng nhỏ của mình để tặng Michael thì Sarah có tổng cộng bao nhiêu món tráng miệng?
- Gold: 11
- v9: score=5 pred=12 err=0.09090909090909091 | Phép tính: 9 / 3 = 3; 9 + 3 = 12 | Đáp án là: 12
- v10: score=1 pred=15 err=0.36363636363636365 | Đáp án là: 15

### near_miss idx=124 type=MATH_SV
- Q: Trung bình cộng của bốn số là X. Hai số là 10 và 18, hai số còn lại bằng nhau. Tích của hai số bằng nhau là 256. Giá trị của biến X chưa biết là bao nhiêu?
- Gold: 15
- v9: score=0 pred=96 err=5.4 | Phép tính: sau : ( 4 * x ) / 4 = X H; x : 4x = X + 256; nh : 4x - 4x = 256 - 4x 256 = 256 - 256 x = 96 | Đáp án là: 96
- v10: score=5 pred=16 err=0.06666666666666667 | Đáp án là: 16

### near_miss idx=129 type=MATH_SV
- Q: Ba số hạng đầu tiên của dãy số học lần lượt là 1, X và 19. Giá trị của số hạng thứ 21 là 181. Giá trị của biến X chưa biết là bao nhiêu?
- Gold: 10
- v9: score=0 pred=21 err=1.1 | Đáp án là: 21
- v10: score=5 pred=9 err=0.1 | Đáp án là: 9

### near_miss idx=131 type=GSM_AnsAug
- Q: Mary đi mua hàng tạp hóa vào thứ bảy. Cô ấy chỉ mua sắm tại một cửa hàng cụ thể nơi cô ấy được cấp khoản tín dụng 100 đô la, số tiền này phải được thanh toán đầy đủ trước chuyến mua sắm tiếp theo. Tuần đó, cô ấy đã...
- Gold: 62
- v9: score=5 pred=60 err=0.03225806451612903 | Phép tính: la + 23; la = 40; 100 USD - 40 USD = 60 | Đáp án là: 60
- v10: score=0 pred=195 err=2.1451612903225805 | Phép tính: 100 - 15 = 65; 65 + 23 = 90; 100 + 65 + 90 = 195 | Đáp án là: 195
