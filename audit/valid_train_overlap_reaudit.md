# Valid/train overlap re-audit

- train records: 95400
- valid records: 1000
- elapsed seconds: 57.08

## Selected Train Duplicate Stats
- `original_question_vi::casefold_alnum`: total=95400, unique=13060, duplicate_records=82340
- `original_question_vi::strip`: total=95400, unique=13069, duplicate_records=82331
- `query_vi::casefold_alnum`: total=95400, unique=57443, duplicate_records=37957
- `query_vi::strip`: total=95400, unique=57806, duplicate_records=37594
- train `query_vi` duplicate groups under `strip`: 17224, extra_records=37594
- train `query_vi` duplicate groups under `casefold_alnum`: 17324, extra_records=37957

## Overlap Rows
| norm | train field | valid field | valid seen | train hits | hit same | hit conflict | valid any same | valid only conflict |
|---|---|---|---:|---:|---:|---:|---:|---:|
| casefold_alnum | original_question_en | original_question_en | 968/1000 | 7006 | 3278 | 4013 | 767 | 177 |
| casefold_alnum | original_question_vi | original_question_vi | 965/1000 | 6951 | 3257 | 3977 | 764 | 177 |
| casefold_space | original_question_en | original_question_en | 965/1000 | 6981 | 3278 | 3958 | 767 | 174 |
| space | original_question_en | original_question_en | 965/1000 | 6981 | 3278 | 3958 | 767 | 174 |
| strip | original_question_en | original_question_en | 965/1000 | 6981 | 3278 | 3958 | 767 | 174 |
| casefold_space | original_question_vi | original_question_vi | 962/1000 | 6924 | 3255 | 3922 | 764 | 174 |
| space | original_question_vi | original_question_vi | 962/1000 | 6924 | 3255 | 3922 | 764 | 174 |
| strip | original_question_vi | original_question_vi | 962/1000 | 6924 | 3255 | 3922 | 764 | 174 |
| casefold_alnum | query_en | original_question_en | 533/1000 | 1809 | 1038 | 801 | 273 | 242 |
| casefold_space | query_en | original_question_en | 530/1000 | 1805 | 1038 | 789 | 273 | 239 |
| space | query_en | original_question_en | 530/1000 | 1805 | 1038 | 789 | 273 | 239 |
| strip | query_en | original_question_en | 530/1000 | 1805 | 1038 | 789 | 273 | 239 |
| casefold_alnum | query_vi | original_question_vi | 525/1000 | 1797 | 1030 | 797 | 267 | 240 |
| casefold_space | query_vi | original_question_vi | 521/1000 | 1792 | 1029 | 785 | 266 | 237 |
| space | query_vi | original_question_vi | 521/1000 | 1792 | 1029 | 785 | 266 | 237 |
| strip | query_vi | original_question_vi | 521/1000 | 1792 | 1029 | 785 | 266 | 237 |
| casefold_alnum | original_question_en | query_en | 361/1000 | 2016 | 1209 | 807 | 331 | 18 |
| casefold_alnum | original_question_vi | query_vi | 360/1000 | 1993 | 1198 | 795 | 329 | 19 |
| casefold_space | original_question_en | query_en | 360/1000 | 1991 | 1209 | 782 | 331 | 17 |
| space | original_question_en | query_en | 360/1000 | 1991 | 1209 | 782 | 331 | 17 |
| strip | original_question_en | query_en | 360/1000 | 1991 | 1209 | 782 | 331 | 17 |
| casefold_space | original_question_vi | query_vi | 359/1000 | 1968 | 1198 | 770 | 329 | 18 |
| space | original_question_vi | query_vi | 359/1000 | 1968 | 1198 | 770 | 329 | 18 |
| strip | original_question_vi | query_vi | 359/1000 | 1968 | 1198 | 770 | 329 | 18 |
| casefold_alnum | original_question_vi | original_question_en | 33/1000 | 30 | 14 | 18 | 14 | 17 |
| casefold_space | original_question_vi | original_question_en | 33/1000 | 30 | 14 | 18 | 14 | 17 |
| space | original_question_vi | original_question_en | 33/1000 | 30 | 14 | 18 | 14 | 17 |
| strip | original_question_vi | original_question_en | 33/1000 | 30 | 14 | 18 | 14 | 17 |
| casefold_alnum | query_en | query_en | 22/1000 | 40 | 34 | 4 | 19 | 1 |
| casefold_alnum | query_vi | query_vi | 12/1000 | 28 | 23 | 4 | 10 | 1 |
| casefold_alnum | original_question_vi | query_en | 10/1000 | 11 | 7 | 4 | 7 | 3 |
| casefold_space | original_question_vi | query_en | 10/1000 | 11 | 7 | 4 | 7 | 3 |
| space | original_question_vi | query_en | 10/1000 | 11 | 7 | 4 | 7 | 3 |
| strip | original_question_vi | query_en | 10/1000 | 11 | 7 | 4 | 7 | 3 |
| casefold_alnum | query_vi | original_question_en | 9/1000 | 8 | 6 | 2 | 6 | 2 |
| casefold_space | query_vi | original_question_en | 9/1000 | 8 | 6 | 2 | 6 | 2 |
| space | query_vi | original_question_en | 9/1000 | 8 | 6 | 2 | 6 | 2 |
| strip | query_vi | original_question_en | 9/1000 | 8 | 6 | 2 | 6 | 2 |
| casefold_space | query_en | query_en | 7/1000 | 10 | 10 | 0 | 7 | 0 |
| space | query_en | query_en | 7/1000 | 10 | 10 | 0 | 7 | 0 |
| strip | query_en | query_en | 7/1000 | 10 | 10 | 0 | 7 | 0 |
| casefold_alnum | query_vi | query_en | 4/1000 | 4 | 4 | 0 | 4 | 0 |
| casefold_space | query_vi | query_en | 4/1000 | 4 | 4 | 0 | 4 | 0 |
| space | query_vi | query_en | 4/1000 | 4 | 4 | 0 | 4 | 0 |
| strip | query_vi | query_en | 4/1000 | 4 | 4 | 0 | 4 | 0 |
| casefold_alnum | original_question_en | original_question_vi | 2/1000 | 16 | 6 | 10 | 2 | 0 |
| casefold_alnum | query_en | original_question_vi | 2/1000 | 5 | 3 | 2 | 1 | 1 |
| casefold_alnum | query_en | query_vi | 2/1000 | 4 | 4 | 0 | 2 | 0 |
| casefold_space | original_question_en | original_question_vi | 2/1000 | 16 | 6 | 10 | 2 | 0 |
| casefold_space | query_en | original_question_vi | 2/1000 | 5 | 3 | 2 | 1 | 1 |
| casefold_space | query_en | query_vi | 2/1000 | 4 | 4 | 0 | 2 | 0 |
| space | original_question_en | original_question_vi | 2/1000 | 16 | 6 | 10 | 2 | 0 |
| space | query_en | original_question_vi | 2/1000 | 5 | 3 | 2 | 1 | 1 |
| space | query_en | query_vi | 2/1000 | 4 | 4 | 0 | 2 | 0 |
| strip | original_question_en | original_question_vi | 2/1000 | 16 | 6 | 10 | 2 | 0 |
| strip | query_en | original_question_vi | 2/1000 | 5 | 3 | 2 | 1 | 1 |
| strip | query_en | query_vi | 2/1000 | 4 | 4 | 0 | 2 | 0 |
| casefold_alnum | original_question_en | query_vi | 1/1000 | 8 | 4 | 4 | 1 | 0 |
| casefold_space | original_question_en | query_vi | 1/1000 | 8 | 4 | 4 | 1 | 0 |
| space | original_question_en | query_vi | 1/1000 | 8 | 4 | 4 | 1 | 0 |
| strip | original_question_en | query_vi | 1/1000 | 8 | 4 | 4 | 1 | 0 |

## Top Examples
### casefold_alnum train.original_question_en vs valid.original_question_en (968/1000)
- valid_id=458, valid_type=GSM_AnsAug, train_type=GSM_SV, train_answer=2.0, valid_answer=9.0
  - valid: Bob, Tom, Sally, and Jerry had dinner at their favorite pizzeria. They decide to share 2 pizzas. Bob ate half of a pizza on his own. Tom ate one-third of a pizza. Sally wasn't very hungry and only ate one-sixth of a pizz
  - train: Bob, Tom, Sally, and Jerry had dinner at their favorite pizzeria. They decide to share 2 pizzas. Bob ate half of a pizza on his own. Tom ate one-third of a pizza. Sally wasn't very hungry and only ate one-sixth of a pizz
- valid_id=241, valid_type=MATH_AnsAug, train_type=MATH_SV, train_answer=9.7, valid_answer=36.0
  - valid: A square and a triangle have equal perimeters. The lengths of the three sides of the triangle are $6.1$ cm, $8.2$ cm and $9.7$ cm. What is the area of the square in square centimeters?
  - train: A square and a triangle have equal perimeters. The lengths of the three sides of the triangle are $6.1$ cm, $8.2$ cm and $9.7$ cm. What is the area of the square in square centimeters?
- valid_id=852, valid_type=MATH_AnsAug, train_type=MATH_Rephrased, train_answer=150.0, valid_answer=150.0
  - valid: The measure of the angles of a pentagon are in the ratio of 3:3:3:4:5. What is the number of degrees in the measure of the largest angle?
  - train: The measure of the angles of a pentagon are in the ratio of 3:3:3:4:5. What is the number of degrees in the measure of the largest angle?
### casefold_alnum train.original_question_vi vs valid.original_question_vi (965/1000)
- valid_id=458, valid_type=GSM_AnsAug, train_type=GSM_SV, train_answer=2.0, valid_answer=9.0
  - valid: Bob, Tom, Sally và Jerry ăn tối tại tiệm bánh pizza yêu thích của họ. Họ quyết định chia nhau 2 chiếc pizza. Bob đã tự mình ăn nửa chiếc bánh pizza. Tom đã ăn một phần ba chiếc bánh pizza. Sally không đói lắm và chỉ ăn 1
  - train: Bob, Tom, Sally và Jerry ăn tối tại tiệm bánh pizza yêu thích của họ. Họ quyết định chia nhau 2 chiếc pizza. Bob đã tự mình ăn nửa chiếc bánh pizza. Tom đã ăn một phần ba chiếc bánh pizza. Sally không đói lắm và chỉ ăn 1
- valid_id=241, valid_type=MATH_AnsAug, train_type=MATH_SV, train_answer=9.7, valid_answer=36.0
  - valid: Một hình vuông và một hình tam giác có chu vi bằng nhau. Độ dài ba cạnh của tam giác là $6,1$ cm, $8,2$ cm và $9,7$ cm. Diện tích của hình vuông tính bằng cm vuông là bao nhiêu?
  - train: Một hình vuông và một hình tam giác có chu vi bằng nhau. Độ dài ba cạnh của tam giác là $6,1$ cm, $8,2$ cm và $9,7$ cm. Diện tích của hình vuông tính bằng cm vuông là bao nhiêu?
- valid_id=852, valid_type=MATH_AnsAug, train_type=MATH_Rephrased, train_answer=150.0, valid_answer=150.0
  - valid: Số đo các góc của ngũ giác đều tỉ lệ 3:3:3:4:5. Góc lớn nhất có số đo là bao nhiêu độ?
  - train: Số đo các góc của ngũ giác đều tỉ lệ 3:3:3:4:5. Góc lớn nhất có số đo là bao nhiêu độ?
### casefold_space train.original_question_en vs valid.original_question_en (965/1000)
- valid_id=458, valid_type=GSM_AnsAug, train_type=GSM_SV, train_answer=2.0, valid_answer=9.0
  - valid: Bob, Tom, Sally, and Jerry had dinner at their favorite pizzeria. They decide to share 2 pizzas. Bob ate half of a pizza on his own. Tom ate one-third of a pizza. Sally wasn't very hungry and only ate one-sixth of a pizz
  - train: Bob, Tom, Sally, and Jerry had dinner at their favorite pizzeria. They decide to share 2 pizzas. Bob ate half of a pizza on his own. Tom ate one-third of a pizza. Sally wasn't very hungry and only ate one-sixth of a pizz
- valid_id=241, valid_type=MATH_AnsAug, train_type=MATH_SV, train_answer=9.7, valid_answer=36.0
  - valid: A square and a triangle have equal perimeters. The lengths of the three sides of the triangle are $6.1$ cm, $8.2$ cm and $9.7$ cm. What is the area of the square in square centimeters?
  - train: A square and a triangle have equal perimeters. The lengths of the three sides of the triangle are $6.1$ cm, $8.2$ cm and $9.7$ cm. What is the area of the square in square centimeters?
- valid_id=852, valid_type=MATH_AnsAug, train_type=MATH_Rephrased, train_answer=150.0, valid_answer=150.0
  - valid: The measure of the angles of a pentagon are in the ratio of 3:3:3:4:5. What is the number of degrees in the measure of the largest angle?
  - train: The measure of the angles of a pentagon are in the ratio of 3:3:3:4:5. What is the number of degrees in the measure of the largest angle?
### space train.original_question_en vs valid.original_question_en (965/1000)
- valid_id=458, valid_type=GSM_AnsAug, train_type=GSM_SV, train_answer=2.0, valid_answer=9.0
  - valid: Bob, Tom, Sally, and Jerry had dinner at their favorite pizzeria. They decide to share 2 pizzas. Bob ate half of a pizza on his own. Tom ate one-third of a pizza. Sally wasn't very hungry and only ate one-sixth of a pizz
  - train: Bob, Tom, Sally, and Jerry had dinner at their favorite pizzeria. They decide to share 2 pizzas. Bob ate half of a pizza on his own. Tom ate one-third of a pizza. Sally wasn't very hungry and only ate one-sixth of a pizz
- valid_id=241, valid_type=MATH_AnsAug, train_type=MATH_SV, train_answer=9.7, valid_answer=36.0
  - valid: A square and a triangle have equal perimeters. The lengths of the three sides of the triangle are $6.1$ cm, $8.2$ cm and $9.7$ cm. What is the area of the square in square centimeters?
  - train: A square and a triangle have equal perimeters. The lengths of the three sides of the triangle are $6.1$ cm, $8.2$ cm and $9.7$ cm. What is the area of the square in square centimeters?
- valid_id=852, valid_type=MATH_AnsAug, train_type=MATH_Rephrased, train_answer=150.0, valid_answer=150.0
  - valid: The measure of the angles of a pentagon are in the ratio of 3:3:3:4:5. What is the number of degrees in the measure of the largest angle?
  - train: The measure of the angles of a pentagon are in the ratio of 3:3:3:4:5. What is the number of degrees in the measure of the largest angle?
### strip train.original_question_en vs valid.original_question_en (965/1000)
- valid_id=458, valid_type=GSM_AnsAug, train_type=GSM_SV, train_answer=2.0, valid_answer=9.0
  - valid: Bob, Tom, Sally, and Jerry had dinner at their favorite pizzeria. They decide to share 2 pizzas. Bob ate half of a pizza on his own. Tom ate one-third of a pizza. Sally wasn't very hungry and only ate one-sixth of a pizz
  - train: Bob, Tom, Sally, and Jerry had dinner at their favorite pizzeria. They decide to share 2 pizzas. Bob ate half of a pizza on his own. Tom ate one-third of a pizza. Sally wasn't very hungry and only ate one-sixth of a pizz
- valid_id=241, valid_type=MATH_AnsAug, train_type=MATH_SV, train_answer=9.7, valid_answer=36.0
  - valid: A square and a triangle have equal perimeters. The lengths of the three sides of the triangle are $6.1$ cm, $8.2$ cm and $9.7$ cm. What is the area of the square in square centimeters?
  - train: A square and a triangle have equal perimeters. The lengths of the three sides of the triangle are $6.1$ cm, $8.2$ cm and $9.7$ cm. What is the area of the square in square centimeters?
- valid_id=852, valid_type=MATH_AnsAug, train_type=MATH_Rephrased, train_answer=150.0, valid_answer=150.0
  - valid: The measure of the angles of a pentagon are in the ratio of 3:3:3:4:5. What is the number of degrees in the measure of the largest angle?
  - train: The measure of the angles of a pentagon are in the ratio of 3:3:3:4:5. What is the number of degrees in the measure of the largest angle?
### casefold_space train.original_question_vi vs valid.original_question_vi (962/1000)
- valid_id=458, valid_type=GSM_AnsAug, train_type=GSM_SV, train_answer=2.0, valid_answer=9.0
  - valid: Bob, Tom, Sally và Jerry ăn tối tại tiệm bánh pizza yêu thích của họ. Họ quyết định chia nhau 2 chiếc pizza. Bob đã tự mình ăn nửa chiếc bánh pizza. Tom đã ăn một phần ba chiếc bánh pizza. Sally không đói lắm và chỉ ăn 1
  - train: Bob, Tom, Sally và Jerry ăn tối tại tiệm bánh pizza yêu thích của họ. Họ quyết định chia nhau 2 chiếc pizza. Bob đã tự mình ăn nửa chiếc bánh pizza. Tom đã ăn một phần ba chiếc bánh pizza. Sally không đói lắm và chỉ ăn 1
- valid_id=241, valid_type=MATH_AnsAug, train_type=MATH_SV, train_answer=9.7, valid_answer=36.0
  - valid: Một hình vuông và một hình tam giác có chu vi bằng nhau. Độ dài ba cạnh của tam giác là $6,1$ cm, $8,2$ cm và $9,7$ cm. Diện tích của hình vuông tính bằng cm vuông là bao nhiêu?
  - train: Một hình vuông và một hình tam giác có chu vi bằng nhau. Độ dài ba cạnh của tam giác là $6,1$ cm, $8,2$ cm và $9,7$ cm. Diện tích của hình vuông tính bằng cm vuông là bao nhiêu?
- valid_id=852, valid_type=MATH_AnsAug, train_type=MATH_Rephrased, train_answer=150.0, valid_answer=150.0
  - valid: Số đo các góc của ngũ giác đều tỉ lệ 3:3:3:4:5. Góc lớn nhất có số đo là bao nhiêu độ?
  - train: Số đo các góc của ngũ giác đều tỉ lệ 3:3:3:4:5. Góc lớn nhất có số đo là bao nhiêu độ?
### space train.original_question_vi vs valid.original_question_vi (962/1000)
- valid_id=458, valid_type=GSM_AnsAug, train_type=GSM_SV, train_answer=2.0, valid_answer=9.0
  - valid: Bob, Tom, Sally và Jerry ăn tối tại tiệm bánh pizza yêu thích của họ. Họ quyết định chia nhau 2 chiếc pizza. Bob đã tự mình ăn nửa chiếc bánh pizza. Tom đã ăn một phần ba chiếc bánh pizza. Sally không đói lắm và chỉ ăn 1
  - train: Bob, Tom, Sally và Jerry ăn tối tại tiệm bánh pizza yêu thích của họ. Họ quyết định chia nhau 2 chiếc pizza. Bob đã tự mình ăn nửa chiếc bánh pizza. Tom đã ăn một phần ba chiếc bánh pizza. Sally không đói lắm và chỉ ăn 1
- valid_id=241, valid_type=MATH_AnsAug, train_type=MATH_SV, train_answer=9.7, valid_answer=36.0
  - valid: Một hình vuông và một hình tam giác có chu vi bằng nhau. Độ dài ba cạnh của tam giác là $6,1$ cm, $8,2$ cm và $9,7$ cm. Diện tích của hình vuông tính bằng cm vuông là bao nhiêu?
  - train: Một hình vuông và một hình tam giác có chu vi bằng nhau. Độ dài ba cạnh của tam giác là $6,1$ cm, $8,2$ cm và $9,7$ cm. Diện tích của hình vuông tính bằng cm vuông là bao nhiêu?
- valid_id=852, valid_type=MATH_AnsAug, train_type=MATH_Rephrased, train_answer=150.0, valid_answer=150.0
  - valid: Số đo các góc của ngũ giác đều tỉ lệ 3:3:3:4:5. Góc lớn nhất có số đo là bao nhiêu độ?
  - train: Số đo các góc của ngũ giác đều tỉ lệ 3:3:3:4:5. Góc lớn nhất có số đo là bao nhiêu độ?
### strip train.original_question_vi vs valid.original_question_vi (962/1000)
- valid_id=458, valid_type=GSM_AnsAug, train_type=GSM_SV, train_answer=2.0, valid_answer=9.0
  - valid: Bob, Tom, Sally và Jerry ăn tối tại tiệm bánh pizza yêu thích của họ. Họ quyết định chia nhau 2 chiếc pizza. Bob đã tự mình ăn nửa chiếc bánh pizza. Tom đã ăn một phần ba chiếc bánh pizza. Sally không đói lắm và chỉ ăn 1
  - train: Bob, Tom, Sally và Jerry ăn tối tại tiệm bánh pizza yêu thích của họ. Họ quyết định chia nhau 2 chiếc pizza. Bob đã tự mình ăn nửa chiếc bánh pizza. Tom đã ăn một phần ba chiếc bánh pizza. Sally không đói lắm và chỉ ăn 1
- valid_id=241, valid_type=MATH_AnsAug, train_type=MATH_SV, train_answer=9.7, valid_answer=36.0
  - valid: Một hình vuông và một hình tam giác có chu vi bằng nhau. Độ dài ba cạnh của tam giác là $6,1$ cm, $8,2$ cm và $9,7$ cm. Diện tích của hình vuông tính bằng cm vuông là bao nhiêu?
  - train: Một hình vuông và một hình tam giác có chu vi bằng nhau. Độ dài ba cạnh của tam giác là $6,1$ cm, $8,2$ cm và $9,7$ cm. Diện tích của hình vuông tính bằng cm vuông là bao nhiêu?
- valid_id=852, valid_type=MATH_AnsAug, train_type=MATH_Rephrased, train_answer=150.0, valid_answer=150.0
  - valid: Số đo các góc của ngũ giác đều tỉ lệ 3:3:3:4:5. Góc lớn nhất có số đo là bao nhiêu độ?
  - train: Số đo các góc của ngũ giác đều tỉ lệ 3:3:3:4:5. Góc lớn nhất có số đo là bao nhiêu độ?
### casefold_alnum train.query_en vs valid.original_question_en (533/1000)
- valid_id=668, valid_type=MATH_Rephrased, train_type=MATH_AnsAug, train_answer=0.020833333333333332, valid_answer=0.020833333333333332
  - valid: Megan has lost Fatima's phone number. Megan knows that the first three digits are either 296 or 299. The remaining four digits are 0, 1, 6 and 7, but she isn't sure of the order of these digits. If Megan randomly dials a
  - train: Megan has lost Fatima's phone number. Megan knows that the first three digits are either 296 or 299. The remaining four digits are 0, 1, 6 and 7, but she isn't sure of the order of these digits. If Megan randomly dials a
- valid_id=975, valid_type=MATH_Rephrased, train_type=MATH_AnsAug, train_answer=20.0, valid_answer=20.0
  - valid: What is the sum of the prime factors of 91?
  - train: What is the sum of the prime factors of 91?
- valid_id=794, valid_type=GSM_Rephrased, train_type=GSM_AnsAug, train_answer=60.0, valid_answer=60.0
  - valid: Allen ordered five boxes of pizza, which cost $7 each box. He then gave a tip which amounts to 1/7 of the total cost of his order. If he gave the delivery man $100, how much change did he receive?
  - train: Allen ordered five boxes of pizza, which cost $7 each box. He then gave a tip which amounts to 1/7 of the total cost of his order. If he gave the delivery man $100, how much change did he receive?
### casefold_space train.query_en vs valid.original_question_en (530/1000)
- valid_id=668, valid_type=MATH_Rephrased, train_type=MATH_AnsAug, train_answer=0.020833333333333332, valid_answer=0.020833333333333332
  - valid: Megan has lost Fatima's phone number. Megan knows that the first three digits are either 296 or 299. The remaining four digits are 0, 1, 6 and 7, but she isn't sure of the order of these digits. If Megan randomly dials a
  - train: Megan has lost Fatima's phone number. Megan knows that the first three digits are either 296 or 299. The remaining four digits are 0, 1, 6 and 7, but she isn't sure of the order of these digits. If Megan randomly dials a
- valid_id=975, valid_type=MATH_Rephrased, train_type=MATH_AnsAug, train_answer=20.0, valid_answer=20.0
  - valid: What is the sum of the prime factors of 91?
  - train: What is the sum of the prime factors of 91?
- valid_id=794, valid_type=GSM_Rephrased, train_type=GSM_AnsAug, train_answer=60.0, valid_answer=60.0
  - valid: Allen ordered five boxes of pizza, which cost $7 each box. He then gave a tip which amounts to 1/7 of the total cost of his order. If he gave the delivery man $100, how much change did he receive?
  - train: Allen ordered five boxes of pizza, which cost $7 each box. He then gave a tip which amounts to 1/7 of the total cost of his order. If he gave the delivery man $100, how much change did he receive?
### space train.query_en vs valid.original_question_en (530/1000)
- valid_id=668, valid_type=MATH_Rephrased, train_type=MATH_AnsAug, train_answer=0.020833333333333332, valid_answer=0.020833333333333332
  - valid: Megan has lost Fatima's phone number. Megan knows that the first three digits are either 296 or 299. The remaining four digits are 0, 1, 6 and 7, but she isn't sure of the order of these digits. If Megan randomly dials a
  - train: Megan has lost Fatima's phone number. Megan knows that the first three digits are either 296 or 299. The remaining four digits are 0, 1, 6 and 7, but she isn't sure of the order of these digits. If Megan randomly dials a
- valid_id=975, valid_type=MATH_Rephrased, train_type=MATH_AnsAug, train_answer=20.0, valid_answer=20.0
  - valid: What is the sum of the prime factors of 91?
  - train: What is the sum of the prime factors of 91?
- valid_id=794, valid_type=GSM_Rephrased, train_type=GSM_AnsAug, train_answer=60.0, valid_answer=60.0
  - valid: Allen ordered five boxes of pizza, which cost $7 each box. He then gave a tip which amounts to 1/7 of the total cost of his order. If he gave the delivery man $100, how much change did he receive?
  - train: Allen ordered five boxes of pizza, which cost $7 each box. He then gave a tip which amounts to 1/7 of the total cost of his order. If he gave the delivery man $100, how much change did he receive?
### strip train.query_en vs valid.original_question_en (530/1000)
- valid_id=668, valid_type=MATH_Rephrased, train_type=MATH_AnsAug, train_answer=0.020833333333333332, valid_answer=0.020833333333333332
  - valid: Megan has lost Fatima's phone number. Megan knows that the first three digits are either 296 or 299. The remaining four digits are 0, 1, 6 and 7, but she isn't sure of the order of these digits. If Megan randomly dials a
  - train: Megan has lost Fatima's phone number. Megan knows that the first three digits are either 296 or 299. The remaining four digits are 0, 1, 6 and 7, but she isn't sure of the order of these digits. If Megan randomly dials a
- valid_id=975, valid_type=MATH_Rephrased, train_type=MATH_AnsAug, train_answer=20.0, valid_answer=20.0
  - valid: What is the sum of the prime factors of 91?
  - train: What is the sum of the prime factors of 91?
- valid_id=794, valid_type=GSM_Rephrased, train_type=GSM_AnsAug, train_answer=60.0, valid_answer=60.0
  - valid: Allen ordered five boxes of pizza, which cost $7 each box. He then gave a tip which amounts to 1/7 of the total cost of his order. If he gave the delivery man $100, how much change did he receive?
  - train: Allen ordered five boxes of pizza, which cost $7 each box. He then gave a tip which amounts to 1/7 of the total cost of his order. If he gave the delivery man $100, how much change did he receive?
