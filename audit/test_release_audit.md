# Test Release Audit

## Schema
- test n=1000
- keys: {'id, query_vi, type': 1000}
- has_response_vi=0
- id range: 0..999, unique=1000, contiguous=True

## Type Distribution
- GSM_Rephrased: 325
- GSM_SV: 194
- MATH_Rephrased: 142
- Synthetic_Generalization: 112
- MATH_SV: 80
- MATH_AnsAug: 71
- GSM_FOBAR: 44
- GSM_AnsAug: 27
- MATH_FOBAR: 5

## Length / Number Stats
- query_words: {'n': 1000, 'min': 4, 'p25': 31.0, 'median': 45.0, 'mean': 43.687, 'p75': 53.0, 'p90': 61.0, 'p95': 66.0, 'p99': 84.00999999999999, 'max': 286}
- query_chars: {'n': 1000, 'min': 15, 'p25': 143.0, 'median': 198.0, 'mean': 188.427, 'p75': 238.0, 'p90': 264.0, 'p95': 276.0, 'p99': 333.2299999999998, 'max': 1227}
- number_count_in_query: {'n': 1000, 'min': 0, 'p25': 3.0, 'median': 4.0, 'mean': 3.927, 'p75': 5.0, 'p90': 6.0, 'p95': 7.0, 'p99': 7.009999999999991, 'max': 24}

## Legal Query Overlap
- train_query_vs_test_query: exact=0, alnum=1
- valid_query_vs_test_query: exact=0, alnum=0
- train_plus_valid_query_vs_test_query: exact=0, alnum=1

## Nearest Query Jaccard
- train: buckets={'>=0.95': 1, '>=0.90': 5, '>=0.80': 19, '>=0.70': 65, '>=0.60': 145, '>=0.50': 193, '<0.50': 807}, stats={'n': 1000, 'min': 0.0, 'p25': 0.2830188679245283, 'median': 0.3157894736842105, 'mean': 0.3714172287229051, 'p75': 0.3548387096774194, 'p90': 0.6776774193548387, 'p95': 0.7222222222222222, 'p99': 0.8269871794871794, 'max': 1.0}
- valid: buckets={'>=0.95': 0, '>=0.90': 0, '>=0.80': 0, '>=0.70': 2, '>=0.60': 6, '>=0.50': 47, '<0.50': 953}, stats={'n': 1000, 'min': 0.13333333333333333, 'p25': 0.23214285714285715, 'median': 0.25, 'mean': 0.2792739765441808, 'p75': 0.29545454545454547, 'p90': 0.42930402930402944, 'p95': 0.48484848484848486, 'p99': 0.5715476190476189, 'max': 0.7105263157894737}

## Test Template Families
- other: 670
- percentage: 121
- motion: 80
- equation: 50
- probability: 43
- arithmetic_sequence: 36

## Top Masked Templates
- count=39 | trong một lớp, có <NUM> học sinh thích bóng đá, <NUM> học sinh thích bóng rổ, và <NUM> học sinh thích cả hai môn. hỏi có bao nhiêu học sinh thích ít nhất một trong hai môn? có <NUM> học sinh nghỉ học hôm đó, nhưng không tính vào các số liệu trên.
  - id=1 nums=['62', '27', '17', '6'] query=Trong một lớp, có 62 học sinh thích bóng đá, 27 học sinh thích bóng rổ, và 17 học sinh thích cả hai môn. Hỏi có bao nhiêu học sinh thích ít nhất một trong hai môn? Có 6 học sinh nghỉ học hôm đó, nhưng không tính vào các số liệu trên.
  - id=4 nums=['73', '69', '17', '8'] query=Trong một lớp, có 73 học sinh thích bóng đá, 69 học sinh thích bóng rổ, và 17 học sinh thích cả hai môn. Hỏi có bao nhiêu học sinh thích ít nhất một trong hai môn? Có 8 học sinh nghỉ học hôm đó, nhưng không tính vào các số liệu trên.
- count=39 | có <NUM> quả cam, mỗi hộp đựng được <NUM> quả. sau khi đóng đầy nhiều hộp nhất có thể, còn thừa bao nhiêu quả cam? ngày hôm sau cửa hàng nhập thêm <NUM> quả cam, nhưng số cam này không tính cho câu hỏi trên.
  - id=23 nums=['234', '32', '8'] query=Có 234 quả cam, mỗi hộp đựng được 32 quả. Sau khi đóng đầy nhiều hộp nhất có thể, còn thừa bao nhiêu quả cam? Ngày hôm sau cửa hàng nhập thêm 8 quả cam, nhưng số cam này không tính cho câu hỏi trên.
  - id=27 nums=['604', '13', '19'] query=Có 604 quả cam, mỗi hộp đựng được 13 quả. Sau khi đóng đầy nhiều hộp nhất có thể, còn thừa bao nhiêu quả cam? Ngày hôm sau cửa hàng nhập thêm 19 quả cam, nhưng số cam này không tính cho câu hỏi trên.
- count=39 | tính $\lfloor <NUM> \rfloor + \lceil <NUM> \rceil$. số <NUM> chỉ là mã bài và không tham gia vào phép tính.
  - id=31 nums=['-9.1', '26.3', '88'] query=Tính $\lfloor -9.1 \rfloor + \lceil 26.3 \rceil$. Số 88 chỉ là mã bài và không tham gia vào phép tính.
  - id=163 nums=['-6.7', '18.9', '26'] query=Tính $\lfloor -6.7 \rfloor + \lceil 18.9 \rceil$. Số 26 chỉ là mã bài và không tham gia vào phép tính.
- count=38 | lan mua <NUM> quyển sách, mỗi quyển giá <NUM> nghìn đồng, và một số cây bút, mỗi cây giá <NUM> nghìn đồng. tổng tiền lan trả là <NUM> nghìn đồng. hỏi lan mua bao nhiêu cây bút? bạn của lan mua thêm <NUM> cục tẩy, nhưng không tính vào tổng tiền của lan.
  - id=2 nums=['7', '25', '8', '335', '5'] query=Lan mua 7 quyển sách, mỗi quyển giá 25 nghìn đồng, và một số cây bút, mỗi cây giá 8 nghìn đồng. Tổng tiền Lan trả là 335 nghìn đồng. Hỏi Lan mua bao nhiêu cây bút? Bạn của Lan mua thêm 5 cục tẩy, nhưng không tính vào tổng tiền của Lan.
  - id=72 nums=['9', '30', '12', '378', '5'] query=Lan mua 9 quyển sách, mỗi quyển giá 30 nghìn đồng, và một số cây bút, mỗi cây giá 12 nghìn đồng. Tổng tiền Lan trả là 378 nghìn đồng. Hỏi Lan mua bao nhiêu cây bút? Bạn của Lan mua thêm 5 cục tẩy, nhưng không tính vào tổng tiền của Lan.
- count=38 | một cửa hàng miễn phí giao hàng nếu tiền hàng đạt ít nhất <NUM> nghìn đồng. mỗi sản phẩm giá <NUM> nghìn đồng. hỏi cần mua ít nhất bao nhiêu sản phẩm để được miễn phí giao hàng? nếu không đủ điều kiện thì phí giao hàng là <NUM> nghìn đồng, nhưng câu hỏi chỉ hỏi số sản phẩm tối thiểu.
  - id=10 nums=['120', '25', '15'] query=Một cửa hàng miễn phí giao hàng nếu tiền hàng đạt ít nhất 120 nghìn đồng. Mỗi sản phẩm giá 25 nghìn đồng. Hỏi cần mua ít nhất bao nhiêu sản phẩm để được miễn phí giao hàng? Nếu không đủ điều kiện thì phí giao hàng là 15 nghìn đồng, nhưng câu hỏi chỉ hỏi số sản phẩm tối thiểu.
  - id=19 nums=['200', '25', '10'] query=Một cửa hàng miễn phí giao hàng nếu tiền hàng đạt ít nhất 200 nghìn đồng. Mỗi sản phẩm giá 25 nghìn đồng. Hỏi cần mua ít nhất bao nhiêu sản phẩm để được miễn phí giao hàng? Nếu không đủ điều kiện thì phí giao hàng là 10 nghìn đồng, nhưng câu hỏi chỉ hỏi số sản phẩm tối thiểu.
- count=38 | ba điểm kiểm tra của minh là <NUM>, <NUM> và <NUM>. minh cần điểm trung bình bốn bài bằng <NUM>. hỏi bài kiểm tra thứ tư minh cần đạt bao nhiêu điểm? giáo viên nói bài thứ năm sẽ được cộng thêm <NUM> điểm thưởng, nhưng bài đó không tính vào trung bình này.
  - id=15 nums=['14', '43', '33', '37', '8'] query=Ba điểm kiểm tra của Minh là 14, 43 và 33. Minh cần điểm trung bình bốn bài bằng 37. Hỏi bài kiểm tra thứ tư Minh cần đạt bao nhiêu điểm? Giáo viên nói bài thứ năm sẽ được cộng thêm 8 điểm thưởng, nhưng bài đó không tính vào trung bình này.
  - id=28 nums=['45', '19', '18', '27', '7'] query=Ba điểm kiểm tra của Minh là 45, 19 và 18. Minh cần điểm trung bình bốn bài bằng 27. Hỏi bài kiểm tra thứ tư Minh cần đạt bao nhiêu điểm? Giáo viên nói bài thứ năm sẽ được cộng thêm 7 điểm thưởng, nhưng bài đó không tính vào trung bình này.
- count=38 | một kho có <NUM> kg gạo. ngày thứ nhất dùng <NUM> số gạo. ngày thứ hai dùng <NUM> số gạo còn lại. hỏi sau hai ngày còn lại bao nhiêu kg gạo? ngày thứ ba dự kiến nhập thêm <NUM> kg, nhưng chưa nhập trong bài toán này.
  - id=16 nums=['360', '1/5', '1/3', '28'] query=Một kho có 360 kg gạo. Ngày thứ nhất dùng 1/5 số gạo. Ngày thứ hai dùng 1/3 số gạo còn lại. Hỏi sau hai ngày còn lại bao nhiêu kg gạo? Ngày thứ ba dự kiến nhập thêm 28 kg, nhưng chưa nhập trong bài toán này.
  - id=41 nums=['240', '1/4', '1/2', '28'] query=Một kho có 240 kg gạo. Ngày thứ nhất dùng 1/4 số gạo. Ngày thứ hai dùng 1/2 số gạo còn lại. Hỏi sau hai ngày còn lại bao nhiêu kg gạo? Ngày thứ ba dự kiến nhập thêm 28 kg, nhưng chưa nhập trong bài toán này.
- count=38 | một cửa hàng ban đầu có <NUM> sản phẩm. buổi sáng bán được <NUM> sản phẩm và phát hiện <NUM> sản phẩm bị hỏng. hỏi trước khi nhập thêm hàng, cửa hàng còn bao nhiêu sản phẩm tốt? buổi chiều cửa hàng nhập thêm <NUM> sản phẩm mới, nhưng không tính vào câu hỏi.
  - id=26 nums=['199', '55', '24', '30'] query=Một cửa hàng ban đầu có 199 sản phẩm. Buổi sáng bán được 55 sản phẩm và phát hiện 24 sản phẩm bị hỏng. Hỏi trước khi nhập thêm hàng, cửa hàng còn bao nhiêu sản phẩm tốt? Buổi chiều cửa hàng nhập thêm 30 sản phẩm mới, nhưng không tính vào câu hỏi.
  - id=32 nums=['61', '23', '10', '70'] query=Một cửa hàng ban đầu có 61 sản phẩm. Buổi sáng bán được 23 sản phẩm và phát hiện 10 sản phẩm bị hỏng. Hỏi trước khi nhập thêm hàng, cửa hàng còn bao nhiêu sản phẩm tốt? Buổi chiều cửa hàng nhập thêm 70 sản phẩm mới, nhưng không tính vào câu hỏi.
- count=38 | một giỏ hàng có <NUM> món loại a, mỗi món giá <NUM> nghìn đồng, và <NUM> món loại b, mỗi món giá <NUM> nghìn đồng. hỏi giá trung bình của một món trong giỏ là bao nhiêu nghìn đồng? người bán giảm thêm <NUM> nghìn đồng cho đơn hàng khác, không liên quan đến giỏ này.
  - id=29 nums=['4', '25', '4', '40', '10'] query=Một giỏ hàng có 4 món loại A, mỗi món giá 25 nghìn đồng, và 4 món loại B, mỗi món giá 40 nghìn đồng. Hỏi giá trung bình của một món trong giỏ là bao nhiêu nghìn đồng? Người bán giảm thêm 10 nghìn đồng cho đơn hàng khác, không liên quan đến giỏ này.
  - id=48 nums=['6', '15', '5', '30', '10'] query=Một giỏ hàng có 6 món loại A, mỗi món giá 15 nghìn đồng, và 5 món loại B, mỗi món giá 30 nghìn đồng. Hỏi giá trung bình của một món trong giỏ là bao nhiêu nghìn đồng? Người bán giảm thêm 10 nghìn đồng cho đơn hàng khác, không liên quan đến giỏ này.
- count=38 | một người đi được <NUM> feet với tốc độ <NUM> feet mỗi phút. hỏi quãng đường đó bằng bao nhiêu yard, biết <NUM> yard = <NUM> feet? thời gian đi là <NUM> phút và số <NUM> là ghi chú, nhưng câu hỏi chỉ hỏi yard.
  - id=35 nums=['120', '5', '1', '3', '24', '9'] query=Một người đi được 120 feet với tốc độ 5 feet mỗi phút. Hỏi quãng đường đó bằng bao nhiêu yard, biết 1 yard = 3 feet? Thời gian đi là 24 phút và số 9 là ghi chú, nhưng câu hỏi chỉ hỏi yard.
  - id=50 nums=['300', '3', '1', '3', '100', '4'] query=Một người đi được 300 feet với tốc độ 3 feet mỗi phút. Hỏi quãng đường đó bằng bao nhiêu yard, biết 1 yard = 3 feet? Thời gian đi là 100 phút và số 4 là ghi chú, nhưng câu hỏi chỉ hỏi yard.
- count=38 | một số lượng ban đầu là <NUM> và tăng gấp <NUM> lần sau mỗi năm. sau một số năm, số lượng đạt <NUM>. hỏi đã qua bao nhiêu năm? số <NUM> là mã thí nghiệm, không dùng để tính.
  - id=38 nums=['5', '4', '1280', '42'] query=Một số lượng ban đầu là 5 và tăng gấp 4 lần sau mỗi năm. Sau một số năm, số lượng đạt 1280. Hỏi đã qua bao nhiêu năm? Số 42 là mã thí nghiệm, không dùng để tính.
  - id=56 nums=['5', '4', '1280', '81'] query=Một số lượng ban đầu là 5 và tăng gấp 4 lần sau mỗi năm. Sau một số năm, số lượng đạt 1280. Hỏi đã qua bao nhiêu năm? Số 81 là mã thí nghiệm, không dùng để tính.
- count=38 | một hộp có <NUM> bóng đèn, trong đó có <NUM> bóng bị hỏng. chọn ngẫu nhiên một bóng đèn. xác suất chọn được bóng không bị hỏng là bao nhiêu? sau đó người ta kiểm tra thêm <NUM> bóng khác, nhưng việc kiểm tra này không liên quan đến xác suất vừa hỏi.
  - id=42 nums=['46', '9', '2'] query=Một hộp có 46 bóng đèn, trong đó có 9 bóng bị hỏng. Chọn ngẫu nhiên một bóng đèn. Xác suất chọn được bóng không bị hỏng là bao nhiêu? Sau đó người ta kiểm tra thêm 2 bóng khác, nhưng việc kiểm tra này không liên quan đến xác suất vừa hỏi.
  - id=103 nums=['32', '5', '2'] query=Một hộp có 32 bóng đèn, trong đó có 5 bóng bị hỏng. Chọn ngẫu nhiên một bóng đèn. Xác suất chọn được bóng không bị hỏng là bao nhiêu? Sau đó người ta kiểm tra thêm 2 bóng khác, nhưng việc kiểm tra này không liên quan đến xác suất vừa hỏi.
- count=38 | một khoản tiền tăng <NUM>% mỗi năm. sau <NUM> năm, số tiền là <NUM> đô la. hỏi số tiền ban đầu là bao nhiêu đô la? con số <NUM> chỉ là mã giao dịch và không dùng trong phép tính.
  - id=45 nums=['10', '2', '121', '5'] query=Một khoản tiền tăng 10% mỗi năm. Sau 2 năm, số tiền là 121 đô la. Hỏi số tiền ban đầu là bao nhiêu đô la? Con số 5 chỉ là mã giao dịch và không dùng trong phép tính.
  - id=91 nums=['10', '2', '1210', '8'] query=Một khoản tiền tăng 10% mỗi năm. Sau 2 năm, số tiền là 1210 đô la. Hỏi số tiền ban đầu là bao nhiêu đô la? Con số 8 chỉ là mã giao dịch và không dùng trong phép tính.
- count=38 | một khoản tiền gửi theo lãi đơn <NUM>% mỗi năm trong <NUM> năm thu được <NUM> đô la tiền lãi. hỏi số tiền gốc ban đầu là bao nhiêu đô la? ngân hàng có thêm phí dịch vụ <NUM> đô la cho tài khoản khác, không liên quan đến khoản tiền này.
  - id=60 nums=['4', '1', '48', '7'] query=Một khoản tiền gửi theo lãi đơn 4% mỗi năm trong 1 năm thu được 48 đô la tiền lãi. Hỏi số tiền gốc ban đầu là bao nhiêu đô la? Ngân hàng có thêm phí dịch vụ 7 đô la cho tài khoản khác, không liên quan đến khoản tiền này.
  - id=104 nums=['4', '5', '100', '2'] query=Một khoản tiền gửi theo lãi đơn 4% mỗi năm trong 5 năm thu được 100 đô la tiền lãi. Hỏi số tiền gốc ban đầu là bao nhiêu đô la? Ngân hàng có thêm phí dịch vụ 2 đô la cho tài khoản khác, không liên quan đến khoản tiền này.
- count=38 | một trò chơi có <NUM> ô từ ô bắt đầu đến ô cuối. người chơi đi <NUM> ô, sau đó đi thêm <NUM> ô nhưng bị lùi <NUM> ô, rồi đi tiếp <NUM> ô. hỏi trước khi nhận thẻ thưởng, người chơi còn cách ô cuối bao nhiêu ô? thẻ thưởng có thể cho đi thêm <NUM> ô, nhưng chưa được dùng trong câu hỏi này.
  - id=61 nums=['48', '15', '6', '8', '5', '10'] query=Một trò chơi có 48 ô từ ô bắt đầu đến ô cuối. Người chơi đi 15 ô, sau đó đi thêm 6 ô nhưng bị lùi 8 ô, rồi đi tiếp 5 ô. Hỏi trước khi nhận thẻ thưởng, người chơi còn cách ô cuối bao nhiêu ô? Thẻ thưởng có thể cho đi thêm 10 ô, nhưng chưa được dùng trong câu hỏi này.
  - id=85 nums=['48', '5', '9', '2', '10', '6'] query=Một trò chơi có 48 ô từ ô bắt đầu đến ô cuối. Người chơi đi 5 ô, sau đó đi thêm 9 ô nhưng bị lùi 2 ô, rồi đi tiếp 10 ô. Hỏi trước khi nhận thẻ thưởng, người chơi còn cách ô cuối bao nhiêu ô? Thẻ thưởng có thể cho đi thêm 6 ô, nhưng chưa được dùng trong câu hỏi này.
- count=38 | số táo và số cam có tỉ lệ <NUM>:<NUM>. tổng số quả là <NUM>. hỏi số táo và số cam chênh lệch nhau bao nhiêu quả? trong giỏ còn có <NUM> quả lê, nhưng lê không tính vào tỉ lệ trên.
  - id=161 nums=['9', '3', '180', '8'] query=Số táo và số cam có tỉ lệ 9:3. Tổng số quả là 180. Hỏi số táo và số cam chênh lệch nhau bao nhiêu quả? Trong giỏ còn có 8 quả lê, nhưng lê không tính vào tỉ lệ trên.
  - id=179 nums=['5', '6', '165', '4'] query=Số táo và số cam có tỉ lệ 5:6. Tổng số quả là 165. Hỏi số táo và số cam chênh lệch nhau bao nhiêu quả? Trong giỏ còn có 4 quả lê, nhưng lê không tính vào tỉ lệ trên.
- count=37 | một nhóm có n người. số cách chọn <NUM> người từ nhóm là <NUM>. hỏi n bằng bao nhiêu? buổi họp bắt đầu lúc <NUM> giờ, nhưng thời gian này không liên quan.
  - id=14 nums=['2', '171', '2'] query=Một nhóm có n người. Số cách chọn 2 người từ nhóm là 171. Hỏi n bằng bao nhiêu? Buổi họp bắt đầu lúc 2 giờ, nhưng thời gian này không liên quan.
  - id=17 nums=['2', '21', '3'] query=Một nhóm có n người. Số cách chọn 2 người từ nhóm là 21. Hỏi n bằng bao nhiêu? Buổi họp bắt đầu lúc 3 giờ, nhưng thời gian này không liên quan.
- count=37 | sau khi giảm giá <NUM>%, một món hàng còn giá <NUM> nghìn đồng. hỏi giá ban đầu của món hàng là bao nhiêu nghìn đồng? cửa hàng còn treo thêm biển giảm <NUM>% cho một sản phẩm khác, nhưng biển này không áp dụng cho món hàng đang hỏi.
  - id=18 nums=['25', '120', '45'] query=Sau khi giảm giá 25%, một món hàng còn giá 120 nghìn đồng. Hỏi giá ban đầu của món hàng là bao nhiêu nghìn đồng? Cửa hàng còn treo thêm biển giảm 45% cho một sản phẩm khác, nhưng biển này không áp dụng cho món hàng đang hỏi.
  - id=40 nums=['40', '240', '5'] query=Sau khi giảm giá 40%, một món hàng còn giá 240 nghìn đồng. Hỏi giá ban đầu của món hàng là bao nhiêu nghìn đồng? Cửa hàng còn treo thêm biển giảm 5% cho một sản phẩm khác, nhưng biển này không áp dụng cho món hàng đang hỏi.
- count=36 | một cấp số cộng có số hạng đầu là <NUM>, công sai là <NUM>. tổng <NUM> số hạng đầu tiên là bao nhiêu?
  - id=0 nums=['8', '1', '22'] query=Một cấp số cộng có số hạng đầu là 8, công sai là 1. Tổng 22 số hạng đầu tiên là bao nhiêu?
  - id=3 nums=['1', '3', '18'] query=Một cấp số cộng có số hạng đầu là 1, công sai là 3. Tổng 18 số hạng đầu tiên là bao nhiêu?
- count=36 | xác định giá trị lớn nhất trong các bội chung nhỏ nhất của <NUM> với các số <NUM>, <NUM>, <NUM>, <NUM>, <NUM>, <NUM>. hãy trả lời bằng một số nguyên.
  - id=5 nums=['10', '6', '9', '12', '14', '16', '17'] query=Xác định giá trị lớn nhất trong các bội chung nhỏ nhất của 10 với các số 6, 9, 12, 14, 16, 17. Hãy trả lời bằng một số nguyên.
  - id=25 nums=['18', '2', '6', '10', '12', '13', '14'] query=Xác định giá trị lớn nhất trong các bội chung nhỏ nhất của 18 với các số 2, 6, 10, 12, 13, 14. Hãy trả lời bằng một số nguyên.

## Highest Train Nearest Examples
- id=453 jaccard=1.000 source_type=MATH_AnsAug
  - test: Phương trình bậc hai $x^2 - 3x + 9 = x + 41$ có hai nghiệm. Sự khác biệt tích cực giữa các giải pháp này là gì?
  - nearest train: Phương trình bậc hai $x^2-3x+9=x+41$ có hai nghiệm. Sự khác biệt tích cực giữa các giải pháp này là gì?
- id=200 jaccard=0.930 source_type=GSM_SV
  - test: Một con bọ cạp mù sống sót nhờ bắt được động vật nhiều chân. Nó cần ăn nhiều động vật nhiều chân để tồn tại: tổng cộng 800 đoạn cơ thể mỗi ngày. Nếu nó đã ăn x cuốn chiếu có 60 đốt và 2 cuốn chiếu dài gấp đôi, thì nó cần ăn 10 cuốn chiếu 50 đốt để đạt tổng số hàng ngày. Giá trị của biến x chưa biết là bao nhiêu? Giá trị của biến x chưa biết là bao nhiêu?
  - nearest train: Một con bọ cạp mù sống sót nhờ bắt được động vật nhiều chân. Nó cần ăn nhiều động vật nhiều chân để tồn tại: tổng cộng 800 đoạn cơ thể mỗi ngày. Nếu nó đã ăn một con rết có x đốt và 2 con rết dài gấp đôi, thì nó cần ăn 10 con rết 50 đốt để đạt tổng số hàng ngày. Giá trị của biến x chưa biết là bao nhiêu? Giá trị của biến x chưa biết là bao nhiêu?
- id=472 jaccard=0.913 source_type=MATH_AnsAug
  - test: Nếu hai đa giác đều có cùng chu vi và đa giác thứ nhất có 38 cạnh với chiều dài cạnh gấp đôi đa giác thứ hai thì đa giác thứ hai có bao nhiêu cạnh?
  - nearest train: Hai đa giác đều có cùng chu vi. Nếu hình thứ nhất có 38 cạnh và chiều dài cạnh gấp đôi cạnh thứ hai thì hình thứ hai có bao nhiêu cạnh?
- id=576 jaccard=0.909 source_type=GSM_Rephrased
  - test: Nếu Terry kiếm được 24 đô la mỗi ngày và Jordan kiếm được 30 đô la mỗi ngày, thì sự khác biệt giữa thu nhập hàng tuần của họ là bao nhiêu nếu cả hai đều làm việc trong 7 ngày?
  - nearest train: Nếu thu nhập hàng ngày của Terry là 24 đô la và thu nhập hàng ngày của Jordan là 30 đô la, thì sự khác biệt giữa thu nhập hàng tuần của họ là bao nhiêu nếu cả hai đều làm việc trong 7 ngày?
- id=273 jaccard=0.900 source_type=GSM_Rephrased
  - test: Số giờ trung bình mà Harry ngủ mỗi đêm là bao nhiêu, biết rằng vào Thứ Hai, anh ấy ngủ 8 giờ, vào Thứ Ba trong 7 giờ, vào Thứ Tư trong 8 giờ, vào Thứ Năm trong 10 giờ và vào Thứ Sáu trong 7 giờ?
  - nearest train: Số giờ trung bình mà Harry ngủ mỗi đêm là bao nhiêu, biết rằng vào Thứ Hai, Thứ Ba, Thứ Tư, Thứ Năm và Thứ Sáu, anh ấy ngủ lần lượt là 8, 7, 8, 10 và 7 giờ?
- id=734 jaccard=0.885 source_type=MATH_Rephrased
  - test: Tổng của tất cả các giá trị của $k$ sao cho phương trình $2x^2-kx+8=0$ có hai nghiệm số nguyên phân biệt?
  - nearest train: Xác định tổng tất cả các giá trị của k sao cho phương trình $2x^2 - kx + 8 = 0$ có hai nghiệm nguyên phân biệt.
- id=648 jaccard=0.878 source_type=MATH_Rephrased
  - test: Hàm $y=\frac{x^3+8x^2+21x+18}{x+2}$ có thể được đơn giản hóa thành hàm $y=Ax^2+Bx+C$, được xác định ở mọi nơi ngoại trừ tại $x =Đ$. Tổng các giá trị của $A$, $B$, $C$ và $D$ là bao nhiêu?
  - nearest train: Tổng các giá trị của A, B, C và D trong hàm $y = \frac{x^3 + 8x^2 + 21x + 18}{x + 2}$, có thể đơn giản hóa là $ y = Ax^2 + Bx + C$ và được xác định ở mọi nơi ngoại trừ tại $x = D$?
- id=405 jaccard=0.846 source_type=MATH_Rephrased
  - test: Có bao nhiêu chữ số trong biểu diễn cơ số 7 của $956$?
  - nearest train: Trong cách biểu diễn cơ số 7, số 956 có bao nhiêu chữ số?
