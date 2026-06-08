# V23 Retrieval Potential Analysis

Pseudo-gold analysis using `dataset/test_gold.json`; this is not official leaderboard gold.

- v23 raw: 1481/10000 score10=1.4810

## Source Retrieval Summary
| source | source_n | top1_raw | best_hybrid | best_raw | best_used | >=0.90 | >=0.80 | >=0.70 | >=0.60 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| train | 92880 | 1166 | nearest_majority_jaccard_ge_0.55_hybrid_v23 | 1464 | 153 | 8 | 30 | 67 | 118 |
| valid | 977 | 447 | nearest_majority_jaccard_ge_0.70_hybrid_v23 | 1499 | 9 | 2 | 4 | 9 | 17 |
| train_plus_valid | 93857 | 1253 | nearest_majority_jaccard_ge_0.55_hybrid_v23 | 1484 | 154 | 8 | 29 | 67 | 120 |

## Lookup Strategies
### train
| strategy | raw | used | buckets |
|---|---:|---:|---|
| exact_query_hybrid_v23 | 1481 | 0 | {'10': 101, '5': 45, '1': 246, '0': 608} |
| mask_plus_numbers_hybrid_v23 | 1481 | 0 | {'10': 101, '5': 45, '1': 246, '0': 608} |
| mask_majority_hybrid_v23 | 1481 | 0 | {'10': 101, '5': 45, '1': 246, '0': 608} |
| top1_nearest_hybrid_v23 | 1166 | 1000 | {'10': 90, '5': 24, '1': 146, '0': 740} |
| oracle_best_of_top12_hybrid_v23 | 2456 | 1000 | {'10': 158, '5': 98, '1': 386, '0': 358} |
- best threshold: `nearest_majority_jaccard_ge_0.55_hybrid_v23` raw=1464 used=153 buckets={'10': 105, '5': 39, '1': 219, '0': 637}
### valid
| strategy | raw | used | buckets |
|---|---:|---:|---|
| exact_query_hybrid_v23 | 1481 | 0 | {'10': 101, '5': 45, '1': 246, '0': 608} |
| mask_plus_numbers_hybrid_v23 | 1481 | 0 | {'10': 101, '5': 45, '1': 246, '0': 608} |
| mask_majority_hybrid_v23 | 1481 | 0 | {'10': 101, '5': 45, '1': 246, '0': 608} |
| top1_nearest_hybrid_v23 | 447 | 1000 | {'10': 20, '5': 19, '1': 152, '0': 809} |
| oracle_best_of_top12_hybrid_v23 | 2764 | 1000 | {'10': 122, '5': 196, '1': 564, '0': 118} |
- best threshold: `nearest_majority_jaccard_ge_0.70_hybrid_v23` raw=1499 used=9 buckets={'10': 103, '5': 45, '1': 244, '0': 608}
### train_plus_valid
| strategy | raw | used | buckets |
|---|---:|---:|---|
| exact_query_hybrid_v23 | 1481 | 0 | {'10': 101, '5': 45, '1': 246, '0': 608} |
| mask_plus_numbers_hybrid_v23 | 1481 | 0 | {'10': 101, '5': 45, '1': 246, '0': 608} |
| mask_majority_hybrid_v23 | 1481 | 0 | {'10': 101, '5': 45, '1': 246, '0': 608} |
| top1_nearest_hybrid_v23 | 1253 | 1000 | {'10': 96, '5': 28, '1': 153, '0': 723} |
| oracle_best_of_top12_hybrid_v23 | 2532 | 1000 | {'10': 162, '5': 108, '1': 372, '0': 358} |
- best threshold: `nearest_majority_jaccard_ge_0.55_hybrid_v23` raw=1484 used=154 buckets={'10': 107, '5': 39, '1': 219, '0': 635}

## V23 By Template Family
| family | n | raw | score10 | buckets | top_v23_answers |
|---|---:|---:|---:|---|---|
| other | 491 | 1120 | 2.281 | {10: 83, 0: 230, 1: 150, 5: 28} | [('40', 45), ('20', 42), ('6', 31), ('10', 22), ('8', 22)] |
| equation | 50 | 121 | 2.420 | {0: 32, 10: 11, 1: 6, 5: 1} | [('49', 9), ('20', 5), ('14', 3), ('143', 3), ('11', 2)] |
| probability | 43 | 46 | 1.070 | {0: 25, 5: 7, 1: 11} | [('0.3', 22), ('0.285714', 4), ('0.7', 3), ('0.714286', 2), ('0.0192308', 2)] |
| floor_ceil | 41 | 25 | 0.610 | {1: 10, 0: 29, 10: 1, 5: 1} | [('8', 10), ('11', 7), ('23', 4), ('10', 3), ('-2', 2)] |
| remainder_oranges | 40 | 4 | 0.100 | {1: 4, 0: 36} | [('26', 7), ('42', 5), ('6', 4), ('32', 3), ('5', 3)] |
| set_union | 39 | 4 | 0.103 | {0: 35, 1: 4} | [('2', 12), ('32', 4), ('20', 4), ('6', 3), ('28', 2)] |
| books_pens | 38 | 60 | 1.579 | {0: 14, 1: 20, 10: 4} | [('4', 11), ('14', 10), ('20', 7), ('6', 3), ('21', 2)] |
| free_shipping | 38 | 10 | 0.263 | {0: 32, 1: 5, 5: 1} | [('2', 15), ('12', 14), ('37500', 5), ('1800', 3), ('1500', 1)] |
| simple_interest_principal | 38 | 6 | 0.158 | {0: 32, 1: 6} | [('363', 21), ('198', 11), ('420', 3), ('<none>', 1), ('331', 1)] |
| weighted_average_cart | 38 | 0 | 0.000 | {0: 38} | [('8', 9), ('276', 6), ('1000', 5), ('384', 5), ('20000', 3)] |
| original_price_discount | 37 | 6 | 0.162 | {0: 35, 1: 1, 5: 1} | [('924', 4), ('3420', 4), ('9', 3), ('1315', 2), ('54400', 2)] |
| arithmetic_sequence | 36 | 0 | 0.000 | {0: 36} | [('5', 21), ('11', 4), ('6.75', 3), ('4.5', 2), ('22', 1)] |
| lcm_max | 36 | 13 | 0.361 | {0: 27, 1: 8, 5: 1} | [('60', 11), ('17', 6), ('39', 4), ('15', 3), ('9', 3)] |
| isosceles_perimeter | 35 | 66 | 1.886 | {1: 21, 5: 5, 10: 2, 0: 7} | [('24', 13), ('43', 8), ('23', 3), ('45', 3), ('28', 3)] |

## Highest Train+Valid Nearest Examples
- id=153 family=other jaccard=1.000 top1=5 gold=5 v23=5 scores top1/v23=10/10
  - test: Rút gọn: $|{-3^2+4}|$
  - nearest: Rút gọn $2(3-i)+i(2+i)$.
- id=240 family=equation jaccard=1.000 top1=4 gold=9 v23=9 scores top1/v23=0/10
  - test: Giá trị của $x$ trong phương trình $9^4+9^4+9^4=3^x$ là bao nhiêu?
  - nearest: Giá trị của $n$ trong phương trình $4^6 = 8^n$ là bao nhiêu?
- id=427 family=other jaccard=1.000 top1=64 gold=224 v23=224 scores top1/v23=0/10
  - test: Kết quả của việc đánh giá $(4 + 8)^2 + (4^2 + 8^2)$ là gì?
  - nearest: Kết quả của việc đánh giá $(2^2)^3$ là gì?
- id=453 family=equation jaccard=1.000 top1=12 gold=12 v23=12 scores top1/v23=10/10
  - test: Phương trình bậc hai $x^2 - 3x + 9 = x + 41$ có hai nghiệm. Sự khác biệt tích cực giữa các giải pháp này là gì?
  - nearest: Phương trình bậc hai $x^2-3x+9=x+41$ có hai nghiệm. Sự khác biệt tích cực giữa các giải pháp này là gì?
- id=935 family=other jaccard=1.000 top1=224 gold=22 v23=200 scores top1/v23=0/0
  - test: Tính: $5^2-3(4)+3^2$.
  - nearest: Tính $(4+8)^2+(4^2+8^2)$.
- id=59 family=other jaccard=0.929 top1=5 gold=4 v23=4 scores top1/v23=1/10
  - test: Sam có 18 con bò. 5 hơn một nửa số bò có màu đen. Có bao nhiêu con bò không có màu đen?
  - nearest: Sam có 18 con bò. x hơn một nửa số bò có màu đen. Có bao nhiêu con bò không có màu đen? Nếu chúng ta biết câu trả lời cho câu hỏi trên là 4 thì giá trị của biến x chưa biết là bao nhiêu?
- id=178 family=equation jaccard=0.915 top1=11 gold=11 v23=11 scores top1/v23=10/10
  - test: Một parabol có phương trình $y = x^2 + bx + c$ đi qua các điểm $(2,3)$ và $(4,3)$. $c$ là gì?
  - nearest: Cho một parabol có phương trình $y = x^2 + bx + c$ đi qua các điểm (2,3) và (4,3), giá trị của c là bao nhiêu?
- id=971 family=other jaccard=0.910 top1=50 gold=10 v23=0.6 scores top1/v23=0/0
  - test: Bob mua 50 feet dây. Anh ấy sử dụng một phần 5 của nó để tạo ra một tác phẩm nghệ thuật nhỏ. Anh ta lấy phần còn lại và đưa một nửa cho người bạn. Sau đó, anh cắt đoạn dài 2 foot. Anh ấy nhận được bao nhiêu phần?
  - nearest: Bob mua 50 feet dây. Anh ấy sử dụng một phần 5 của nó để tạo ra một tác phẩm nghệ thuật nhỏ. Anh ta lấy phần còn lại và đưa x% cho người bạn. Sau đó, anh cắt đoạn dài 2 foot. Anh ấy nhận được 10 phần. Giá trị của biến x chưa biết là bao nhiêu? Giá trị của biến x chưa biết là bao nhiêu?
- id=653 family=other jaccard=0.896 top1=64 gold=64 v23=64 scores top1/v23=10/10
  - test: Giá trị của biểu thức $(a^2 + b)^2 - (a^2 - b)^2$ khi $a = 4$ và $b = 1$ là bao nhiêu?
  - nearest: Tính giá trị của biểu thức $(a^2+b)^2 - (a^2-b)^2$ khi $a=4$ và $b=1$.
- id=55 family=other jaccard=0.887 top1=200 gold=6000 v23=6000 scores top1/v23=0/10
  - test: Mike quyết định phát triển một mảnh đất. Ông mua 200 mẫu đất với giá 70 USD/mẫu. Sau khi phát triển, ông đã bán một nửa diện tích với giá 200 USD/mẫu. Anh ta đã kiếm được bao nhiêu lợi nhuận?
  - nearest: Mike quyết định phát triển một mảnh đất. Ông mua 200 mẫu đất với giá 70 USD/mẫu. Sau khi phát triển, ông đã bán một nửa diện tích với giá x đô la một mẫu Anh. Anh ta đã kiếm được bao nhiêu lợi nhuận? Nếu chúng ta biết câu trả lời cho câu hỏi trên là 6000 thì giá trị của biến x chưa biết là bao nhiêu?
- id=993 family=other jaccard=0.882 top1=80 gold=11000 v23=47000 scores top1/v23=0/0
  - test: James quyết định thay chiếc xe của mình. Anh ta đã bán chiếc ô tô trị giá 20.000 đô la của mình với giá 80% giá trị của nó và sau đó có thể mặc cả để mua một chiếc ô tô giá 30.000 đô la với giá 90% giá trị của nó. Anh ta đã hết bao nhiêu tiền?
  - nearest: James quyết định thay chiếc xe của mình. Anh ta đã bán chiếc ô tô trị giá 20.000 đô la của mình với giá x% giá trị của nó và sau đó có thể mặc cả để mua một chiếc ô tô giá 30.000 đô la với giá 90% giá trị của nó. Anh ta đã hết túi 11000. Giá trị của biến x chưa biết là bao nhiêu? Giá trị của biến x chưa biết là bao nhiêu?
- id=750 family=other jaccard=0.867 top1=9 gold=9 v23=-9 scores top1/v23=10/0
  - test: Cho rằng đa thức $f(x) = x^4 + ax^3 + bx^2 + cx + d$ có hệ số thực và $f(2i) = f(2 + i) = 0$, thì giá trị của $a + b + c + d$?
  - nearest: Đa thức $f(x)=x^4+ax^3+bx^2+cx+d$ có hệ số thực và $f(2i)=f(2+i)=0$. $a+b+c+d$ là gì?
- id=985 family=other jaccard=0.865 top1=57 gold=6840 v23=3 scores top1/v23=0/0
  - test: Tích của ba số nguyên dương khác nhau bằng X$. Tổng của ba số nguyên là 57. Giá trị của biến X chưa biết là bao nhiêu?
  - nearest: Tích của ba số nguyên dương khác nhau bằng $7^3$. Tổng của ba số nguyên là bao nhiêu?
- id=150 family=other jaccard=0.864 top1=38 gold=38 v23=38 scores top1/v23=10/10
  - test: Xác định giá trị nguyên của $n$, trong đó $0 \le n \le 180$, sao cho $\cos n^\circ = \cos 758^\circ$.
  - nearest: Tìm số nguyên $n,$ $0 \le n \le 180,$ sao cho $\cos n^\circ = \cos 758^\circ.$
- id=732 family=other jaccard=0.853 top1=20 gold=220 v23=220 scores top1/v23=0/10
  - test: Jack mua 3 cuốn sách mỗi tháng với giá 20 USD mỗi cuốn. Anh ta bán lại chúng vào cuối năm với giá 500 USD. Anh ta đã mất bao nhiêu tiền?
  - nearest: Jack mua 3 cuốn sách mỗi tháng với giá x $ mỗi cuốn. Anh ta bán lại chúng vào cuối năm với giá 500 USD. Anh ta đã mất bao nhiêu tiền? Nếu chúng ta biết câu trả lời cho câu hỏi trên là 220 thì giá trị của biến x chưa biết là bao nhiêu?

## Top1 Retrieval Beats V23 Examples
- id=750 family=other jaccard=0.867 top1=9 gold=9 v23=-9 scores top1/v23=10/0
- id=920 family=other jaccard=0.822 top1=112 gold=112 v23=108 scores top1/v23=10/5
- id=894 family=other jaccard=0.822 top1=8 gold=8 v23=16 scores top1/v23=10/0
- id=92 family=other jaccard=0.815 top1=16 gold=12 v23=49 scores top1/v23=1/0
- id=157 family=other jaccard=0.777 top1=20 gold=20 v23=3 scores top1/v23=10/0
- id=648 family=other jaccard=0.772 top1=14 gold=14 v23=13 scores top1/v23=10/5
- id=928 family=other jaccard=0.771 top1=3 gold=3 v23=9 scores top1/v23=10/0
- id=13 family=other jaccard=0.762 top1=8 gold=8 v23=5 scores top1/v23=10/1
- id=745 family=other jaccard=0.753 top1=2 gold=2 v23=23 scores top1/v23=10/0
- id=338 family=other jaccard=0.745 top1=15 gold=10 v23=150 scores top1/v23=1/0
- id=77 family=other jaccard=0.745 top1=4 gold=4 v23=2 scores top1/v23=10/1
- id=652 family=other jaccard=0.741 top1=42 gold=42 v23=46 scores top1/v23=10/5
- id=84 family=other jaccard=0.729 top1=625 gold=625 v23=15 scores top1/v23=10/0
- id=278 family=probability jaccard=0.728 top1=144 gold=144 v23=174 scores top1/v23=10/1
- id=250 family=other jaccard=0.708 top1=0.125 gold=0.125 v23=0.25 scores top1/v23=10/1
- id=467 family=other jaccard=0.706 top1=0.2 gold=0.5466666667 v23=9.6491 scores top1/v23=1/0
- id=432 family=other jaccard=0.696 top1=1000 gold=1000 v23=3600 scores top1/v23=10/0
- id=64 family=other jaccard=0.690 top1=2 gold=2 v23=1 scores top1/v23=10/1
- id=381 family=other jaccard=0.682 top1=10 gold=10 v23=5 scores top1/v23=10/1
- id=779 family=other jaccard=0.677 top1=23 gold=30 v23=10 scores top1/v23=1/0

## Top1 Retrieval Hurts V23 Examples
- id=240 family=equation jaccard=1.000 top1=4 gold=9 v23=9 scores top1/v23=0/10
- id=427 family=other jaccard=1.000 top1=64 gold=224 v23=224 scores top1/v23=0/10
- id=59 family=other jaccard=0.929 top1=5 gold=4 v23=4 scores top1/v23=1/10
- id=55 family=other jaccard=0.887 top1=200 gold=6000 v23=6000 scores top1/v23=0/10
- id=732 family=other jaccard=0.853 top1=20 gold=220 v23=220 scores top1/v23=0/10
- id=559 family=other jaccard=0.841 top1=3 gold=41 v23=41 scores top1/v23=0/10
- id=777 family=other jaccard=0.840 top1=5 gold=20 v23=20 scores top1/v23=0/10
- id=399 family=other jaccard=0.822 top1=100 gold=72 v23=70 scores top1/v23=1/5
- id=717 family=other jaccard=0.819 top1=9 gold=8 v23=8 scores top1/v23=1/10
- id=124 family=other jaccard=0.816 top1=2 gold=980 v23=980 scores top1/v23=0/10
- id=139 family=other jaccard=0.811 top1=2 gold=60 v23=40 scores top1/v23=0/1
- id=160 family=other jaccard=0.800 top1=5 gold=40 v23=45 scores top1/v23=0/1
- id=426 family=other jaccard=0.784 top1=7 gold=27 v23=27 scores top1/v23=0/10
- id=811 family=other jaccard=0.780 top1=8640 gold=60 v23=40 scores top1/v23=0/1
- id=741 family=equation jaccard=0.754 top1=50 gold=3 v23=3 scores top1/v23=0/10
- id=527 family=other jaccard=0.741 top1=30 gold=105 v23=105 scores top1/v23=0/10
- id=748 family=other jaccard=0.739 top1=60 gold=3 v23=4 scores top1/v23=0/1
- id=63 family=other jaccard=0.734 top1=39 gold=14 v23=14 scores top1/v23=0/10
- id=715 family=other jaccard=0.725 top1=2 gold=16 v23=16 scores top1/v23=0/10
- id=979 family=other jaccard=0.720 top1=100 gold=3000 v23=3000 scores top1/v23=0/10
