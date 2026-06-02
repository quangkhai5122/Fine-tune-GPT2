# Bổ sung nội dung báo cáo v8-v15

## 1. Nhận xét tổng quan cho bản thảo hiện tại

Bản thảo `report/Fine_tune_GPT_2.pdf` đã có phần nền về dữ liệu, leakage, LoRA và retrieval đến v14. Phần cần bổ sung lớn nhất là biến chuỗi thí nghiệm thành một luận điểm nghiên cứu: trong thiết lập GPT-2 nhỏ, điểm validation tăng chủ yếu nhờ target alignment, source-local memory, retrieval gating và answer selection, chưa đủ để khẳng định robust mathematical reasoning.

Các bảng/hình nên được đọc theo ba lớp:

- Lớp metric: official-valid score, exact10, extractability.
- Lớp robustness: source-disjoint, FOBAR/SV, conflict-source, source-unseen.
- Lớp cơ chế: model-only, retrieval override, ranker/candidate selection, output shape.

## 2. Nội dung nên thêm vào Introduction

Nên nêu đóng góp là một protocol đánh giá reasoning cho mô hình ngôn ngữ nhỏ, không chỉ là tối ưu notebook Kaggle. Câu mở rộng phù hợp:

> Công trình này không chỉ tìm cấu hình điểm cao cho GPT-2 tiếng Việt, mà còn kiểm toán xem điểm cao đến từ năng lực reasoning chuyển giao, từ metric alignment, hay từ source-level memory trong dữ liệu augmentation.

Cần nhấn mạnh final-answer accuracy không đồng nghĩa với reasoning faithfulness. Trong task này, một mô hình answer-only có thể tối ưu đúng metric tốt hơn mô hình sinh lời giải dài, vì scorer chỉ đọc đáp án số cuối cùng.

## 3. Nội dung nên thêm vào Data Audit

Đưa overlap audit thành bằng chứng trung tâm:

- Exact query overlap trên official validation: 0/1000.
- Source-group overlap: 965/1000.
- Nhiều type gần như hoàn toàn source-seen: GSM_Rephrased và GSM_SV đạt 100% source overlap.

Điểm cần viết rõ: query-disjoint không tương đương source-disjoint. Nếu train chứa biến thể khác của cùng bài gốc, mô hình hoặc retrieval có thể học mapping cục bộ từ source/template sang đáp án mà không cần giải bài mới.

## 4. Nội dung nên thêm vào Experiments

Nên viết lại phần experiment theo bốn family:

1. Full-solution SFT: giữ rationale nhưng target dài, tín hiệu final answer bị loãng, v2 chỉ đạt 0.585.
2. Compact reasoning: cải thiện format/extractability, nhưng source-disjoint gần không tăng; v10 valid 2.131 (2131/10000) trong khi source-disjoint 1.396 (1396/10000).
3. Answer-only LoRA: tăng mạnh vì loss align trực tiếp với final-answer metric.
4. Retrieval/ranker hybrid: điểm cao nhất nhưng phụ thuộc source overlap và candidate selection.

Kết quả mới cần thêm: v14.5 không vượt v13, còn v15 ensemble đạt 7.016 (7016/10000) nhưng là diagnostic branch vì dùng ranker/candidate selection trên public valid và dùng các feature ngoài `query_vi`/`response_vi`.

## 5. Section V. MATH REASONING VỚI CÁC MÔ HÌNH NHỎ

### 5.1 Câu hỏi nghiên cứu

Với các mô hình dưới 500M tham số, fine-tuning trên lời giải toán có tạo ra năng lực reasoning robust không, hay chủ yếu tạo ra rationalization/pattern matching?

### 5.2 Định nghĩa operational

Trong báo cáo này, robust mathematical reasoning nên được định nghĩa bằng các điều kiện quan sát được:

- Điểm không sụp mạnh trên source-disjoint hoặc source-unseen.
- Không chỉ mạnh trên Rephrased/AnsAug mà vẫn ổn trên FOBAR/SV.
- Không phụ thuộc lớn vào retrieval từ cùng source group.
- Nếu sinh rationale, rationale phải hỗ trợ đúng đáp án thay vì chỉ hợp thức hóa đáp án đã đoán.

Ngược lại, rationalization/pattern matching được nhận diện khi điểm tăng nhưng output vẫn answer-only, score cao tập trung ở source-seen same-answer, và conflict-source/no-match giảm mạnh.

### 5.3 Bằng chứng thực nghiệm

**Answer-signal dilution.** Với SFT autoregressive, loss trung bình trên toàn target. Nếu chỉ vài token cuối chứa đáp án được scorer dùng, target càng dài thì tín hiệu trực tiếp cho đáp án càng loãng. Điều này giải thích vì sao full-solution v2 đạt thấp, trong khi answer-only tăng mạnh.

**Compact reasoning chưa tạo transfer rõ.** v10 rút ngắn output và tăng extractability, nhưng source-disjoint chỉ quanh 1.396 (1396/10000). So với v9/v10, tăng official-valid không đi kèm tăng source-disjoint tương ứng.

**Source-overlap chi phối điểm.** Với v13, same-source same-answer đạt 8.589, conflict-source chỉ 1.471, và no-match chỉ 2.026. Khoảng cách này là bằng chứng mạnh rằng source memory là biến giải thích quan trọng.

**Retrieval/ranker đóng góp lớn hơn standalone reasoning.** v13 model-only 5.389 (5389/10000) tăng lên 6.896 (6896/10000) sau hybrid. v15 ensemble model-only 5.306 (5306/10000) tăng lên selected single-candidate 6.857 (6857/10000) rồi final ranker 7.016 (7016/10000). Đây là gain từ chọn đáp án/candidate, không phải bằng chứng trực tiếp rằng GPT-2 tự sinh reasoning bền vững.

**Output behavior.** v15 ensemble có median output length khoảng 13.0 ký tự và short single-line rate 100.0%. Điều này cho thấy output chủ yếu là câu trả lời ngắn, không phải chain-of-thought dài.

### 5.4 Kết luận khoa học

Bằng chứng hiện tại ủng hộ kết luận thận trọng: GPT-2 nhỏ học được answer formatting, answer extraction, source-local mapping và calibration/candidate selection khá tốt. Tuy nhiên, các chỉ dấu robust reasoning vẫn yếu: source-disjoint thấp, conflict-source thấp, FOBAR/SV khó, và điểm cao nhất phụ thuộc vào retrieval/ranker. Vì vậy báo cáo không nên claim mô hình đã học robust mathematical reasoning theo nghĩa mạnh.

## 6. Phần v14.5/v15 nên viết thêm

v14.5 kiểm tra giả thuyết rằng train full + valid selection có thể vượt v13. Kết quả không ủng hộ giả thuyết này: v14.5 v13 đạt 6.823 và v14.5 gate đạt 6.807, đều thấp hơn v13 6.896.

v15 ensemble là điểm cao nhất hiện tại với 7.016, nhưng diễn giải đúng là best diagnostic score. Ranker chọn giữa 32 candidates và thường chọn cả model-only lẫn hybrid candidates. Nhánh này hữu ích để chứng minh lỗi giữa các checkpoint/candidate có tính bổ sung, nhưng không nên trình bày là bằng chứng reasoning nội tại hoặc cấu hình submission-safe.

## 7. Limitations cần thêm

- Không có source-disjoint result cho nhiều nhánh retrieval v11-v15 trong local artifacts.
- v15 ensemble dùng public valid labels cho ranker và dùng feature ngoài strict `query_vi`/`response_vi`.
- Các chỉ số reasoning quality chưa có human annotation; hiện chỉ có proxy như source-disjoint, conflict-source, output length và type-wise robustness.
- Hidden test generalization không nên được claim nếu chưa biết test có source-overlap giống valid hay không.

## 8. Đề xuất câu kết luận

> Trong giới hạn mô hình GPT-2 tiếng Việt nhỏ và runtime Kaggle, fine-tuning có thể tạo ra một solver thực dụng cho final-answer metric, nhưng bằng chứng thực nghiệm chỉ ra rằng phần lớn gain đến từ alignment với đáp án cuối, source-overlap retrieval và ranker/candidate selection. Do đó, báo cáo xem đây là một nghiên cứu về đánh giá và kiểm toán reasoning ở small LM, hơn là bằng chứng rằng mô hình đã học robust mathematical reasoning.
