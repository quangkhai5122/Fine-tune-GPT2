# HF teacher KD artifact for V4

Mục tiêu: tạo `train_kd_compact.jsonl` bằng một notebook Kaggle riêng, dùng open-weight teacher như `Qwen/Qwen2.5-Math-7B-Instruct`, sau đó upload JSONL này làm Kaggle input cho notebook `finetune_gpt2_for_math_v4.ipynb`.

V4 không phụ thuộc teacher là OpenAI hay Hugging Face. V4 chỉ cần artifact đúng schema và sẽ validate lại:

```json
{
  "source_id": 123,
  "query_vi": "...",
  "type": "GSM_Rephrased",
  "gold_answer": "1200",
  "compact_reasoning": "...",
  "final_answer": "1200",
  "teacher_model": "Qwen/Qwen2.5-Math-7B-Instruct",
  "verified": true
}
```

## Kaggle teacher notebook

Notebook teacher nên chạy riêng với GPU. Add input gồm:

- dataset chứa `train.json`;
- code files `create_train_kd_compact_hf.py` và `math_eval.py`;
- nếu Kaggle Internet OFF, add model HF đã được Kaggle cache/model input; nếu Internet ON, script có thể tải trực tiếp từ Hugging Face.

### 1. Install dependencies

```python
!pip install -q -U transformers accelerate bitsandbytes sentencepiece tqdm
```

### 2. Copy code files vào working dir

Điều chỉnh path theo tên Kaggle input của bạn.

```python
!cp /kaggle/input/gpt2-math-code/create_train_kd_compact_hf.py /kaggle/working/
!cp /kaggle/input/gpt2-math-code/math_eval.py /kaggle/working/
%cd /kaggle/working
```

### 3. Dry run

```python
!python create_train_kd_compact_hf.py \
  --train /kaggle/input/math-dataset/train.json \
  --out /kaggle/working/train_kd_compact.jsonl \
  --teacher Qwen/Qwen2.5-Math-7B-Instruct \
  --load-in-4bit \
  --batch-size 1 \
  --max-records 5 \
  --dry-run
```

### 4. Smoke test thật trên 20 records

```python
!python create_train_kd_compact_hf.py \
  --train /kaggle/input/math-dataset/train.json \
  --out /kaggle/working/train_kd_compact_smoke.jsonl \
  --teacher Qwen/Qwen2.5-Math-7B-Instruct \
  --load-in-4bit \
  --batch-size 1 \
  --max-records 20
```

Kiểm tra nhanh output:

```python
!head -n 3 /kaggle/working/train_kd_compact_smoke.jsonl
!wc -l /kaggle/working/train_kd_compact_smoke.jsonl
!wc -l /kaggle/working/train_kd_compact_smoke.errors.jsonl || true
```

### 5. Full run

Với khoảng 95k train rows, full generation có thể lâu. Nên chạy chunk để tránh mất tiến độ.

Ví dụ chunk theo index:

```python
!python create_train_kd_compact_hf.py \
  --train /kaggle/input/math-dataset/train.json \
  --out /kaggle/working/train_kd_compact_00000_20000.jsonl \
  --teacher Qwen/Qwen2.5-Math-7B-Instruct \
  --load-in-4bit \
  --batch-size 2 \
  --start-id 0 \
  --end-id 20000
```

Hoặc chia modulo shard, tiện để chạy nhiều notebook/session độc lập:

```python
!python create_train_kd_compact_hf.py \
  --train /kaggle/input/math-dataset/train.json \
  --out /kaggle/working/train_kd_compact_shard0of5.jsonl \
  --teacher Qwen/Qwen2.5-Math-7B-Instruct \
  --load-in-4bit \
  --batch-size 2 \
  --num-shards 5 \
  --shard-index 0
```

Sau khi có các shard:

```python
!cat /kaggle/working/train_kd_compact_shard*.jsonl > /kaggle/working/train_kd_compact.jsonl
!wc -l /kaggle/working/train_kd_compact.jsonl
```

Download hoặc tạo Kaggle Dataset từ `/kaggle/working/train_kd_compact.jsonl`, rồi add dataset đó vào notebook V4.

## Recommended teacher settings

- Default: `Qwen/Qwen2.5-Math-7B-Instruct`, `--load-in-4bit`, `--batch-size 1` hoặc `2`.
- Nếu Kaggle GPU yếu hoặc quá chậm: thử `Qwen/Qwen2.5-Math-1.5B-Instruct`.
- Không cần 72B cho V4 vì teacher chỉ tạo compact reasoning từ train row + gold answer; artifact còn được validate numeric answer trước khi train.

## Important notes

- Teacher artifact chỉ được tạo từ `train.json`, không dùng `valid.json` hoặc `test.json`.
- Prompt yêu cầu JSON deterministic và script dùng `do_sample=False`.
- Record nào teacher trả sai answer hoặc JSON lỗi sẽ bị ghi vào `.errors.jsonl`, không đưa vào KD.
- V4 sẽ fallback answer-only cho train rows thiếu KD hợp lệ, nên không bắt buộc phải đạt coverage 100%, nhưng coverage càng cao càng tốt cho stage2 reasoning.
