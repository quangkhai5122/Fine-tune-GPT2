# V8 score-10 overlap analysis

- run: `results\v8_select_checkpoint\beam2_lr1e-3`
- raw score: 5265 / 10000
- exact-10: 502 / 1000
- selected checkpoint: epoch_08, lr=0.001, epoch=8.0
- output median chars: 13.0

## Exact-10 Breakdown
- `n_exact10`: 502
- `direct_query_has_same`: 9
- `same_original_has_same`: 479
- `same_original_only_conflict`: 19
- `no_same_original_match`: 4
- `valid_original_seen_as_train_query_has_same`: 217
- `valid_query_seen_as_train_original_has_same`: 148

## Combined Score Slices
| group | n | score/10 | exact10 | exact10 share | buckets |
|---|---:|---:|---:|---:|---|
| direct_query_has_same | 10 | 9.000 | 9 | 1.8% | {'10': 9, '0': 1} |
| same_original_has_same | 764 | 6.490 | 479 | 95.4% | {'10': 479, '5': 16, '1': 88, '0': 181} |
| same_original_only_conflict | 174 | 1.454 | 19 | 3.8% | {'10': 19, '5': 4, '1': 43, '0': 108} |
| no_same_original_match | 38 | 1.421 | 4 | 0.8% | {'10': 4, '5': 1, '1': 9, '0': 24} |
| valid_original_seen_as_train_query_has_same | 266 | 8.271 | 217 | 43.2% | {'10': 217, '5': 3, '1': 15, '0': 31} |
| valid_query_seen_as_train_original_has_same | 329 | 4.818 | 148 | 29.5% | {'10': 148, '5': 10, '1': 55, '0': 116} |

## Channel Class Counts
### direct_query_vi
- train hits: 28
- all valid classes: {'has_same': 10, 'no_match': 988, 'only_conflict': 1, 'only_missing': 1}
- exact10 classes: {'has_same': 9, 'no_match': 493, 'only_conflict': 0, 'only_missing': 0}
### direct_query_vi_exact
- train hits: 0
- all valid classes: {'no_match': 1000}
- exact10 classes: {'no_match': 502}
### same_original_vi
- train hits: 6924
- all valid classes: {'has_same': 764, 'no_match': 38, 'only_conflict': 174, 'only_missing': 24}
- exact10 classes: {'has_same': 479, 'no_match': 4, 'only_conflict': 19, 'only_missing': 0}
### train_query_vs_valid_original
- train hits: 1792
- all valid classes: {'has_same': 266, 'no_match': 479, 'only_conflict': 237, 'only_missing': 18}
- exact10 classes: {'has_same': 217, 'no_match': 198, 'only_conflict': 85, 'only_missing': 2}
### train_original_vs_valid_query
- train hits: 1968
- all valid classes: {'has_same': 329, 'no_match': 641, 'only_conflict': 18, 'only_missing': 12}
- exact10 classes: {'has_same': 148, 'no_match': 353, 'only_conflict': 1, 'only_missing': 0}
### same_original_en
- train hits: 6981
- all valid classes: {'has_same': 767, 'no_match': 35, 'only_conflict': 174, 'only_missing': 24}
- exact10 classes: {'has_same': 481, 'no_match': 2, 'only_conflict': 19, 'only_missing': 0}

## Answer Distribution
- exact10 predictions whose numeric answer appears somewhere in train: 493/502
- top predictions among exact10: [('3', 31), ('2', 27), ('4', 25), ('8', 18), ('5', 18), ('10', 15), ('50', 13), ('6', 13), ('1', 12), ('7', 10), ('20', 10), ('60', 9), ('16', 8), ('80', 8), ('25', 7), ('0', 7), ('9', 7), ('15', 7), ('18', 6), ('120', 6)]
- top gold answers among exact10: [('3', 31), ('2', 27), ('4', 25), ('8', 18), ('5', 18), ('10', 15), ('50', 13), ('6', 13), ('1', 12), ('7', 10), ('20', 10), ('60', 9), ('16', 8), ('80', 8), ('25', 7), ('0', 7), ('9', 7), ('15', 7), ('18', 6), ('120', 6)]

## Example exact-10 rows
### id=0 type=GSM_Rephrased pred=37 gold=37
- classes: direct_query=no_match, same_original=has_same, train_query_vs_valid_original=has_same, train_original_vs_valid_query=no_match
- query: Nếu Susan đang chơi một trò chơi cờ bàn có 48 ô từ ô bắt đầu đến ô cuối chiến thắng và ở lượt đầu tiên, cô ấy tiến về phía trước tám ô, ở lượt thứ hai, cô ấy di chuyển hai ô nhưng bị đẩy lùi lại năm ô và ở lượt thứ ba. đến lượt cô ấy tiến v
- original: Susan đang chơi một trò chơi board game có 48 ô tính từ ô đầu tiên đến ô cuối cùng của trò chơi. Ở lượt đầu tiên, cô ấy tiến về phía trước tám ô. Ở lượt thứ hai, cô ấy di chuyển hai ô, nhưng đáp xuống một ô khiến cô ấy lùi lại năm ô. Ở lượt
  - train same-original type=GSM_Rephrased answer=37.0: Nếu Susan đang chơi một trò chơi cờ bàn có 48 ô, và ở lượt đầu tiên cô ấy tiến về phía trước tám ô, ở lượt thứ hai, cô ấy di chuyển hai ô nhưng bị đẩy lùi lại năm ô, và ở lượt thứ ba, cô ấy tiến về phía trước thêm sáu ô,
  - train same-original type=GSM_Rephrased answer=37.0: Nếu Susan đang chơi một trò chơi cờ bàn có 48 ô, trong đó cô ấy tiến về phía trước tám ô ở lượt đầu tiên, hai ô ở lượt thứ hai (nhưng bị đẩy lùi lại năm ô) và sáu ô nữa ở lượt thứ ba, thì có bao nhiêu ô trống? cô ấy vẫn 
### id=1 type=MATH_Rephrased pred=19 gold=19
- classes: direct_query=no_match, same_original=has_same, train_query_vs_valid_original=no_match, train_original_vs_valid_query=no_match
- query: Nếu $\angle PQR = \angle PRQ$, và độ dài của QR và PR lần lượt là 5 và 7 thì chu vi của tam giác PQR là bao nhiêu?
- original: Trong sơ đồ, $\angle PQR=\angle PRQ$. Nếu $QR=5$ và $PR=7$, chu vi của $\tam giác PQR$ là bao nhiêu? [asy] draw((0,0)--(2.5,7.43)--(5,0)--cycle); nhãn("5",(2.5,0),S); nhãn("$Q$",(0,0),SW); nhãn("$R$",(5,0),SE); nhãn("$P$",(2.5,7.43),N); nhã
  - train same-original type=MATH_Rephrased answer=19.0: Cho $\angle PQR = \angle PRQ$, và độ dài của QR và PR lần lượt là 5 và 7, vậy chu vi của tam giác PQR là bao nhiêu?
  - train same-original type=MATH_Rephrased answer=19.0: Cho $\angle PQR = \angle PRQ$, và $QR = 5$ và $PR = 7$, chu vi của tam giác PQR là bao nhiêu?
### id=2 type=MATH_SV pred=8 gold=8
- classes: direct_query=no_match, same_original=has_same, train_query_vs_valid_original=only_conflict, train_original_vs_valid_query=no_match
- query: Một con súc sắc tám mặt có các mặt được đánh số từ 1 đến X. Giá trị kỳ vọng của con xúc xắc là 4,5. Giá trị của biến X chưa biết là bao nhiêu?
- original: Một con súc sắc tám mặt có các mặt được đánh số từ 1 đến 8. Giá trị kỳ vọng của việc tung xúc xắc là bao nhiêu?
  - train same-original type=MATH_AnsAug answer=4.5: Một con súc sắc tám mặt có các mặt được đánh số từ 1 đến 8. Giá trị kỳ vọng của việc tung xúc xắc là bao nhiêu?
  - train same-original type=MATH_Rephrased answer=4.5: Giá trị trung bình của việc tung một con súc sắc tám mặt là bao nhiêu?
### id=3 type=GSM_Rephrased pred=39 gold=39
- classes: direct_query=no_match, same_original=has_same, train_query_vs_valid_original=has_same, train_original_vs_valid_query=no_match
- query: Sau khi bắt đầu với 50 quả bóng bay, Claire đưa cho một bé gái 1 quả bóng bay, mất 12 quả bóng bay, cho thêm 9 quả bóng nữa và lấy 11 quả bóng bay từ đồng nghiệp của mình. Hiện tại Claire có bao nhiêu quả bóng bay?
- original: Claire chịu trách nhiệm phát bóng bay miễn phí cho tất cả trẻ em tại hội chợ. Cô ấy bắt đầu với 50 quả bóng bay. Khi chuyền 1 quả bóng bay cho một bé gái thì có 12 quả bóng bay đi mất. Trong ba mươi phút tiếp theo, cô ấy cho thêm 9 quả nữa 
  - train same-original type=GSM_Rephrased answer=39.0: Bắt đầu với 50 quả bóng bay, Claire đưa cho một bé gái 1 quả bóng bay nhưng 12 quả bóng bay đi mất. Trong vòng ba mươi phút tiếp theo, cô đưa thêm 9 quả bóng bay và lấy đi 11 quả từ đồng nghiệp. Claire hiện có bao nhiêu 
  - train same-original type=GSM_Rephrased answer=39.0: Sau khi bắt đầu với 50 quả bóng bay, Claire chuyền 1 quả bóng bay cho một cô bé, làm mất 12 quả bóng bay, tặng thêm 9 quả bóng nữa và lấy đi 11 quả bóng bay từ đồng nghiệp của mình. Hiện tại Claire có bao nhiêu quả bóng 
### id=4 type=GSM_Rephrased pred=440 gold=440
- classes: direct_query=no_match, same_original=has_same, train_query_vs_valid_original=has_same, train_original_vs_valid_query=no_match
- query: Nếu có một bãi đỗ xe 1000 ô tô được chia thành 3 khu, trong đó khu 1 có 320 chỗ và khu 2 có nhiều hơn khu 3 200 chỗ thì khu 2 của bãi đậu xe có bao nhiêu chỗ?
- original: Bãi đậu xe 1000 ô tô được chia làm 3 khu. Có 320 chỗ ở khu vực 1 và 200 chỗ ở khu vực 2 nhiều hơn khu vực 3. Có bao nhiêu chỗ trống ở khu vực 2 của bãi đậu xe?
  - train same-original type=GSM_Rephrased answer=440.0: Nếu một bãi đỗ xe 1000 ô tô được chia thành 3 khu, trong đó khu 1 có 320 chỗ và khu 2 có nhiều hơn khu 3 200 chỗ thì khu 2 của bãi đậu xe có bao nhiêu chỗ?
  - train same-original type=GSM_AnsAug answer=440.0: Bãi đậu xe 1000 ô tô được chia làm 3 khu. Có 320 chỗ ở khu vực 1 và 200 chỗ ở khu vực 2 nhiều hơn khu vực 3. Có bao nhiêu chỗ trống ở khu vực 2 của bãi đậu xe?
### id=5 type=GSM_AnsAug pred=90 gold=90
- classes: direct_query=no_match, same_original=has_same, train_query_vs_valid_original=no_match, train_original_vs_valid_query=has_same
- query: Hans đặt phòng ở khách sạn. Khách sạn có 10 tầng, mỗi tầng có 10 phòng giống nhau. Do xảy ra tai nạn nên tầng cuối cùng không còn chỗ cho khách. Xem xét không có khách nào khác, Hans có thể được nhận vào bao nhiêu phòng khác nhau?
- original: Hans đặt phòng ở khách sạn. Khách sạn có 10 tầng, mỗi tầng có 10 phòng giống nhau. Do xảy ra tai nạn nên tầng cuối cùng không còn chỗ cho khách. Xem xét không có khách nào khác, Hans có thể được nhận vào bao nhiêu phòng khác nhau?
  - train same-original type=GSM_Rephrased answer=90.0: Nếu khách sạn có 10 tầng với 10 phòng giống nhau ở mỗi tầng, nhưng tầng cuối cùng không còn trống do tai nạn, thì Hans có thể được nhận phòng ở bao nhiêu phòng khác nhau, giả sử không có khách nào khác?
  - train same-original type=GSM_Rephrased answer=90.0: Nếu có 10 tầng trong một khách sạn, mỗi tầng có 10 phòng giống nhau và tầng cuối cùng không có khách do tai nạn, thì Hans có thể được nhận vào bao nhiêu phòng khác nhau, vì không có khách nào khác?
### id=6 type=GSM_SV pred=12 gold=12
- classes: direct_query=no_match, same_original=has_same, train_query_vs_valid_original=only_conflict, train_original_vs_valid_query=no_match
- query: Bà Dunbar đang cắm hoa cho đám cưới của cháu gái bà. Cô ấy cần làm 5 bó hoa và 7 món đồ trang trí bàn ăn. Cô sử dụng x bông hồng trắng để trang trí mỗi bàn và 5 bông hồng trắng trong mỗi bó hoa. Cô ấy cần tổng cộng 109 bông hồng trắng để ho
- original: Bà Dunbar đang cắm hoa cho đám cưới của cháu gái bà. Cô ấy cần làm 5 bó hoa và 7 món đồ trang trí bàn ăn. Cô sử dụng 12 bông hồng trắng để trang trí bàn tiệc và 5 bông hồng trắng trong mỗi bó hoa. Cô ấy cần tổng cộng bao nhiêu bông hồng trắ
  - train same-original type=GSM_Rephrased answer=109.0: Đối với đám cưới của cháu gái, bà Dunbar đang thực hiện việc cắm hoa, bao gồm 5 bó hoa và 7 đồ trang trí trên bàn. Mỗi cách trang trí bàn ăn cần 12 bông hồng trắng, trong khi mỗi bó hoa cần 5 bông hồng trắng. Cô ấy cần t
  - train same-original type=GSM_Rephrased answer=109.0: Đối với đám cưới của cháu gái mình, bà Dunbar đang thực hiện các kiểu cắm hoa bao gồm 5 bó hoa và 7 đồ trang trí trên bàn. Mỗi cách trang trí bàn cần có 12 bông hồng trắng và mỗi bó hoa cần có 5 bông hồng trắng. Bà Dunba
### id=8 type=MATH_Rephrased pred=60 gold=60
- classes: direct_query=no_match, same_original=has_same, train_query_vs_valid_original=has_same, train_original_vs_valid_query=no_match
- query: Xác định giá trị cao nhất trong số các bội số chung nhỏ nhất của 12 và 2, 12 và 4, 12 và 6, 12 và 8, 12 và 10, 12 và 12. Hãy thể hiện câu trả lời của bạn dưới dạng số nguyên.
- original: Giá trị lớn nhất trong số $\operatorname{lcm[12,2],$ $\operatorname{lcm[12,4],$ $\operatorname{lcm[12,6],$ $\operatorname{lcm là bao nhiêu [12,8],$ $\operatorname{lcm[12,10],$ và $\operatorname{lcm[12,12]?$ Thể hiện câu trả lời của bạn dưới
  - train same-original type=MATH_AnsAug answer=60.0: Giá trị lớn nhất trong số $\operatorname{lcm[12,2],$ $\operatorname{lcm[12,4],$ $\operatorname{lcm[12,6],$ $\operatorname{lcm là bao nhiêu [12,8],$ $\operatorname{lcm[12,10],$ và $\operatorname{lcm[12,12]?$ Thể hiện câu 
  - train same-original type=MATH_AnsAug answer=60.0: Giá trị lớn nhất trong số $\operatorname{lcm[12,2],$ $\operatorname{lcm[12,4],$ $\operatorname{lcm[12,6],$ $\operatorname{lcm là bao nhiêu [12,8],$ $\operatorname{lcm[12,10],$ và $\operatorname{lcm[12,12]?$ Thể hiện câu 
### id=9 type=GSM_AnsAug pred=5 gold=5
- classes: direct_query=no_match, same_original=has_same, train_query_vs_valid_original=no_match, train_original_vs_valid_query=has_same
- query: Bob được hỗ trợ tiền thuê nhà vì anh ấy có thu nhập thấp. Nếu anh ta được tăng lương 0,50 USD/giờ và làm việc 40 giờ một tuần, anh ta sẽ thực sự kiếm được bao nhiêu tiền một tuần nếu trợ cấp nhà ở của anh ta giảm đi 60 USD/tháng?
- original: Bob được hỗ trợ tiền thuê nhà vì anh ấy có thu nhập thấp. Nếu anh ta được tăng lương 0,50 USD/giờ và làm việc 40 giờ một tuần, anh ta sẽ thực sự kiếm được bao nhiêu tiền một tuần nếu trợ cấp nhà ở của anh ta giảm đi 60 USD/tháng?
  - train same-original type=GSM_Rephrased answer=5.0: Nếu Bob nhận được hỗ trợ tiền thuê nhà do tình trạng thu nhập thấp của anh ấy và được tăng lương 0,50 đô la mỗi giờ khi làm việc 40 giờ một tuần, anh ấy sẽ kiếm thêm bao nhiêu thu nhập mỗi tuần nếu trợ cấp nhà ở của anh 
  - train same-original type=GSM_Rephrased answer=5.0: Nếu Bob, người nhận được hỗ trợ tiền thuê nhà do tình trạng thu nhập thấp, được tăng lương 0,50 đô la mỗi giờ và làm việc 40 giờ mỗi tuần, thì anh ta sẽ kiếm thêm bao nhiêu thu nhập mỗi tuần nếu trợ cấp nhà ở của anh ta 
### id=10 type=GSM_FOBAR pred=1 gold=1
- classes: direct_query=no_match, same_original=only_conflict, train_query_vs_valid_original=only_conflict, train_original_vs_valid_query=no_match
- query: John phải thay vòng bi cho những chiếc máy mà anh ấy làm việc cùng. Anh ta có 10 chiếc máy và mỗi chiếc có 30 vòng bi. Thông thường nó có giá x $ cho mỗi ổ bi nhưng hiện tại đang có đợt giảm giá với giá chỉ 0,75 USD. Ngoài ra, vì mua số lượ
- original: John phải thay vòng bi cho những chiếc máy mà anh ấy làm việc cùng. Anh ta có 10 chiếc máy và mỗi chiếc có 30 vòng bi. Thông thường nó có giá 1 USD cho mỗi ổ bi nhưng hiện tại đang có đợt giảm giá với giá chỉ 0,75 USD. Ngoài ra, vì mua số l
  - train same-original type=GSM_FOBAR answer=20.0: John phải thay vòng bi cho những chiếc máy mà anh ấy làm việc cùng. Anh ta có 10 chiếc máy và mỗi chiếc có 30 vòng bi. Thông thường nó có giá 1 USD cho mỗi ổ bi nhưng hiện tại đang có đợt giảm giá với giá chỉ 0,75 USD. N
  - train same-original type=GSM_Rephrased answer=120.0: Nếu John cần thay vòng bi cho 10 máy của mình, với mỗi máy cần 30 vòng bi và chi phí thông thường là 1 USD cho mỗi vòng bi nhưng hiện đang được bán với giá 0,75 USD, kèm theo chiết khấu thêm 20% khi mua số lượng lớn, thì
### id=11 type=GSM_Rephrased pred=3 gold=3
- classes: direct_query=no_match, same_original=has_same, train_query_vs_valid_original=has_same, train_original_vs_valid_query=no_match
- query: Nếu con chó của Sandra sinh ra 7 chú chó con và bác sĩ thú y của cô ấy đã cung cấp cho cô ấy 105 phần sữa công thức để cho các chú chó con ăn trong 5 ngày, thì Sandra nên cho mỗi chú chó con ăn bao nhiêu lần một ngày?
- original: Con chó của Sandra đã sinh được 7 chú chó con. Bác sĩ thú y đã đưa cho cô ấy 105 phần sữa công thức để cho chó con uống trong 5 ngày. Sandra nên cho chó con ăn bao nhiêu lần một ngày?
  - train same-original type=GSM_Rephrased answer=3.0: Nếu con chó của Sandra sinh ra 7 chú chó con và cô ấy có 105 phần sữa công thức để cho chúng ăn trong 5 ngày thì Sandra nên cho mỗi chú chó con ăn bao nhiêu lần một ngày?
  - train same-original type=GSM_FOBAR answer=7.0: Con chó của Sandra đã sinh ra x con chó con. Bác sĩ thú y đã đưa cho cô ấy 105 phần sữa công thức để cho chó con uống trong 5 ngày. Sandra nên cho chó con ăn bao nhiêu lần một ngày? Nếu chúng ta biết câu trả lời cho câu 
### id=14 type=GSM_FOBAR pred=18 gold=18
- classes: direct_query=no_match, same_original=only_conflict, train_query_vs_valid_original=only_conflict, train_original_vs_valid_query=no_match
- query: Lizzie có số bút chì màu bằng một nửa Bobbie. Bobbie có số bút chì màu nhiều gấp ba lần Billie. Nếu Billie có x bút chì màu thì Lizzie có bao nhiêu bút màu? Nếu chúng ta biết câu trả lời cho câu hỏi trên là 27 thì giá trị của biến x chưa bi
- original: Lizzie có số bút chì màu bằng một nửa Bobbie. Bobbie có số bút chì màu nhiều gấp ba lần Billie. Nếu Billie có 18 cây bút chì màu thì Lizzie có bao nhiêu cây bút chì màu?
  - train same-original type=GSM_AnsAug answer=27.0: Lizzie có số bút chì màu bằng một nửa Bobbie. Bobbie có số bút chì màu nhiều gấp ba lần Billie. Nếu Billie có 18 cây bút chì màu thì Lizzie có bao nhiêu cây bút chì màu?
  - train same-original type=GSM_AnsAug answer=27.0: Lizzie có số bút chì màu bằng một nửa Bobbie. Bobbie có số bút chì màu nhiều gấp ba lần Billie. Nếu Billie có 18 cây bút chì màu thì Lizzie có bao nhiêu cây bút chì màu?
