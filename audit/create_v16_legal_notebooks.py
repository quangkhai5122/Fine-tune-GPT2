from __future__ import annotations

import ast
import copy
import json
from pathlib import Path


BASE_NOTEBOOK = Path("finetune_gpt2_for_math_v15_ensemble_ranker.ipynb")

ANSWER_ONLY_NOTEBOOK = Path("finetune_gpt2_for_math_v16_legal_answer_only_multiseed.ipynb")
QUERY_RETRIEVAL_NOTEBOOK = Path("finetune_gpt2_for_math_v16_legal_query_retrieval_ranker.ipynb")


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def read_base_cells() -> tuple[dict, list[dict]]:
    nb = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]
    return nb, cells


def scrub_eval_cell(source: str) -> str:
    # Keep the official-style evaluator but remove the model-irrelevant type
    # distribution print. The evaluator may still report by_type for analysis.
    source = source.replace(
        "valid_records = load_records(VALID_FILE)\n",
        "valid_records = load_records(VALID_FILE) if VALID_FILE.exists() else []\n",
    )
    return source.replace(
        'print("train type distribution:", dict(Counter(r.get("type") for r in train_records).most_common()))\n',
        'print("model/data fields: query_vi -> input, response_vi -> target/gold")\n',
    )


CONFIG_TEMPLATE = r'''# ============================================================
# 1. Config - __VERSION__
# ============================================================
def first_existing(*paths) -> Path:
    for p in map(Path, paths):
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find any path: " + " | ".join(map(str, paths)))

DATA_DIR = first_existing(
    "/kaggle/input/dataset-math",
    "/kaggle/input/datasets/kimanh2002/dataset-math",
    "dataset",
)

MODEL_NAME = str(first_existing(
    "/kaggle/input/nlphustgpt2-vietnamese",
    "/kaggle/input/nlphustgpt2-vietnamese/gpt2-vietnamese",
    "/kaggle/input/datasets/kimanh2002/nlphustgpt2-vietnamese",
    "GPT2_vietnamese",
))

TRAIN_FILE = Path(DATA_DIR) / "train.json"
VALID_FILE = Path(DATA_DIR) / "valid.json"
TEST_FILE  = Path(DATA_DIR) / "test.json"

NOTEBOOK_VERSION = "__VERSION__"
RUN_MODE = "phase1"  # "phase1" writes valid_output/report; "phase2" writes test_predictions.json.

# Legal prompt: the model input uses query_vi only. The target/gold is extracted
# from response_vi. `type` and `original_*` are not used for prompt, routing,
# retrieval keys, model selection, or ranker features.
PROMPT_TEMPLATE = "Bài toán: {q}\nLời giải: "
LEGAL_INPUT_FIELDS = ["query_vi"]
LEGAL_TARGET_FIELDS = ["response_vi"]
DISALLOWED_MODEL_FEATURE_FIELDS = ["type", "original_question_en", "original_question_vi"]

SAFE_EOS_ID = 50256
N_POSITIONS = 1024

WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
ARTIFACT_PREFIX = "__ARTIFACT_PREFIX__"

STAGE_A_OUTPUT_DIR = WORKING_DIR / (ARTIFACT_PREFIX + "_answer_only_runs")
SFT_OUTPUT_DIR     = WORKING_DIR / (ARTIFACT_PREFIX + "_sft_primary")
FINAL_OUTPUT_DIR   = WORKING_DIR / (ARTIFACT_PREFIX + "_final_primary")

VALID_OUTPUT_PATH               = WORKING_DIR / "valid_output.json"
VALID_REPORT_PATH               = WORKING_DIR / "valid_report.json"
MODEL_VALID_OUTPUT_PATH         = WORKING_DIR / "model_valid_output.json"
MODEL_VALID_REPORT_PATH         = WORKING_DIR / "model_valid_report.json"
VALID_OVERLAP_AUDIT_PATH        = WORKING_DIR / "valid_overlap_audit.json"
QUERY_DISJOINT_SPLIT_PATH       = WORKING_DIR / "query_only_internal_split_report.json"
ENSEMBLE_CANDIDATE_DIR          = WORKING_DIR / "ensemble_candidates"
ENSEMBLE_RANKER_REPORT_PATH     = WORKING_DIR / "ensemble_or_gate_report.json"
SELECTED_VALID_OUTPUT_PATH      = WORKING_DIR / "selected_valid_output.json"
SELECTED_VALID_REPORT_PATH      = WORKING_DIR / "selected_valid_report.json"
RL_LOG_PATH                     = WORKING_DIR / "rl_training_log.jsonl"
RL_SUMMARY_PATH                 = WORKING_DIR / "rl_reward_summary.json"
TEST_OUTPUT_PATH                = WORKING_DIR / "test_predictions.json"
MODEL_TEST_OUTPUT_PATH          = WORKING_DIR / "model_test_predictions.json"

CHECKPOINT_ROOT_DIR              = WORKING_DIR / (ARTIFACT_PREFIX + "_checkpoints")
CHECKPOINT_EVAL_DIR              = WORKING_DIR / (ARTIFACT_PREFIX + "_checkpoint_eval")
CHECKPOINT_SELECTION_REPORT_PATH = WORKING_DIR / "checkpoint_selection_report.json"
SELECTED_CHECKPOINT_INFO_PATH    = WORKING_DIR / "selected_checkpoint_info.json"

# Legal train-internal calibration. Only train.json response_vi is used for this.
USE_INTERNAL_CALIBRATION_SPLIT = __USE_INTERNAL_CALIBRATION_SPLIT__
CALIB_FRACTION = __CALIB_FRACTION__
CALIB_MAX_RECORDS = __CALIB_MAX_RECORDS__
TRAIN_ON_FIT_SPLIT_ONLY = USE_INTERNAL_CALIBRATION_SPLIT
INFERENCE_RETRIEVAL_USES_FULL_TRAIN = True

# Smoke/debug knobs. Keep None for real Kaggle runs.
MAX_TRAIN_SAMPLES = None
MAX_VALID_SAMPLES = None
DROP_EXACT_DUPLICATES = True
DROP_NON_EXTRACTABLE = True

# Answer-only LoRA.
STAGE_A_NAME = "stage_a_legal_answer_only_lora"
STAGE_A_EPOCHS = __STAGE_A_EPOCHS__
STAGE_A_LR = __STAGE_A_LR__
MAX_LENGTH_STAGE_A = 256
TRAIN_SEEDS = __TRAIN_SEEDS__
SEED = TRAIN_SEEDS[0]
SAVE_EPOCH_CHECKPOINTS = True

# Backward-compatible no-op stage variables for reused cells/manifest.
USE_KD = False
REQUIRE_KD_FILE = False
RUN_STAGE_B = False
STAGE_B_EPOCHS = 0.0
STAGE_B_LR = 0.0
MAX_LENGTH_STAGE_B = 256
RUN_STAGE_C = False
STAGE_C_EPOCHS = 0.0
STAGE_C_LR = 0.0

# Trainer.
PER_DEVICE_BATCH_SIZE = 16
GRAD_ACCUM = 2
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

# LoRA.
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]

# Generation and legal candidate selection.
MAX_NEW_TOKENS = 32
NUM_BEAMS = 2
DECODE_BATCH_SIZE = 8
NO_REPEAT_NGRAM = 4
REPETITION_PENALTY = 1.15
LENGTH_PENALTY = 0.9
SANITIZE_TO_ANSWER_ONLY = True
INFER_FP16 = True

ENSEMBLE_INCLUDE_EPOCH_CHECKPOINTS = True
ENSEMBLE_LAST_K_EPOCHS = __ENSEMBLE_LAST_K_EPOCHS__
ENSEMBLE_INCLUDE_FINAL = True
ENSEMBLE_CANDIDATE_LIMIT = None

# Query-only retrieval config. Disabled in answer-only notebook.
LEGAL_QUERY_RETRIEVAL_ENABLED = __LEGAL_QUERY_RETRIEVAL_ENABLED__
RETRIEVAL_TOP_K = 7
RETRIEVAL_ANALYZER = "char_wb"
RETRIEVAL_NGRAM_RANGE = (3, 5)
RETRIEVAL_SWEEP_ENABLED = LEGAL_QUERY_RETRIEVAL_ENABLED and USE_INTERNAL_CALIBRATION_SPLIT
RETRIEVAL_GATE_GRID = [
    {"min_sim": 0.45, "min_majority_frac": 0.34, "min_margin": 0.00, "model_low_conf_frac": 0.50},
    {"min_sim": 0.50, "min_majority_frac": 0.34, "min_margin": 0.00, "model_low_conf_frac": 0.50},
    {"min_sim": 0.55, "min_majority_frac": 0.34, "min_margin": 0.10, "model_low_conf_frac": 0.50},
    {"min_sim": 0.60, "min_majority_frac": 0.45, "min_margin": 0.10, "model_low_conf_frac": 0.50},
    {"min_sim": 0.65, "min_majority_frac": 0.50, "min_margin": 0.15, "model_low_conf_frac": 0.60},
    {"min_sim": 0.70, "min_majority_frac": 0.50, "min_margin": 0.20, "model_low_conf_frac": 0.60},
]
DEFAULT_RETRIEVAL_GATE = {"min_sim": 0.60, "min_majority_frac": 0.45, "min_margin": 0.10, "model_low_conf_frac": 0.50}

# RL disabled.
RL_ENABLED = False
RL_MAX_STEPS = 0

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print("TRAIN_FILE       :", TRAIN_FILE)
print("VALID_FILE       :", VALID_FILE)
print("TEST_FILE        :", TEST_FILE, "| exists:", TEST_FILE.exists())
print("MODEL_NAME       :", MODEL_NAME)
print("NOTEBOOK_VERSION :", NOTEBOOK_VERSION)
print("RUN_MODE         :", RUN_MODE)
print("LEGAL_FIELDS     :", {"input": LEGAL_INPUT_FIELDS, "target": LEGAL_TARGET_FIELDS})
print("TRAIN_SEEDS      :", TRAIN_SEEDS)
print("QUERY_RETRIEVAL  :", LEGAL_QUERY_RETRIEVAL_ENABLED)
print("CHECKPOINT_ROOT  :", CHECKPOINT_ROOT_DIR)
'''


DATA_CELL = r'''# ============================================================
# 3. Clean data + legal query-only train/calibration split
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
tokenizer.pad_token_id = SAFE_EOS_ID
tokenizer.eos_token_id = SAFE_EOS_ID
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

def stable_fraction(value) -> float:
    h = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0x100000000

def normalize_text_key(text: str | None) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def query_key(rec: dict) -> str:
    value = normalize_text_key(rec.get("query_vi"))
    return value or "query_id:" + str(rec.get("_source_id", rec.get("id", "unknown")))

def query_tokens(text: str | None) -> set[str]:
    return set(re.findall(r"\w+", normalize_text_key(text)))

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def format_prompt(rec: dict) -> str:
    return PROMPT_TEMPLATE.format(q=(rec.get("query_vi") or "").strip())

def build_answer_only_target(canonical_answer: str) -> str:
    return "Đáp án là: " + str(canonical_answer)

def save_valid_overlap_audit(train_recs: list[dict], valid_recs: list[dict], path: Path):
    # Legal audit axis: exact query_vi overlap only. original_* fields are ignored.
    train_q = Counter(query_key(r) for r in train_recs)
    seen_query = []
    for i, rec in enumerate(valid_recs):
        if query_key(rec) in train_q:
            seen_query.append(i)
    report = {
        "audit_fields": ["query_vi"],
        "train_n": len(train_recs),
        "valid_n": len(valid_recs),
        "train_unique_query": len(train_q),
        "valid_seen_query": len(seen_query),
        "valid_seen_query_pct": len(seen_query) / len(valid_recs) if valid_recs else 0.0,
        "valid_seen_query_ids_first20": seen_query[:20],
        "note": "original_question_en/original_question_vi are intentionally not read by the v16 legal pipeline.",
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[valid-query-overlap]", json.dumps({k: report[k] for k in ["train_n", "valid_n", "valid_seen_query"]}, ensure_ascii=False))
    return report

def clean_records(records: list[dict], split: str) -> list[dict]:
    seen_exact = set()
    out = []
    dropped_dup = dropped_empty = dropped_non_numeric = 0
    for source_id, rec in enumerate(records):
        q = (rec.get("query_vi") or "").strip()
        r = (rec.get("response_vi") or "").strip()
        if not q or (split.startswith("train") and not r):
            dropped_empty += 1
            continue
        if split.startswith("train") and DROP_EXACT_DUPLICATES:
            key = (q, r)
            if key in seen_exact:
                dropped_dup += 1
                continue
            seen_exact.add(key)
        gold_str, gold_num = extract_gold(rec) if r else (None, None)
        canonical = canonicalize_answer(gold_num)
        if split.startswith("train") and DROP_NON_EXTRACTABLE and canonical is None:
            dropped_non_numeric += 1
            continue
        item = {
            **rec,
            "query_vi": q,
            "response_vi": r,
            "_source_id": source_id,
            "_query_key": None,
            "_query_tokens": None,
            "_gold_str": gold_str,
            "_gold_num": gold_num,
            "_canonical_answer": canonical,
        }
        item["_query_key"] = query_key(item)
        item["_query_tokens"] = query_tokens(q)
        out.append(item)
    print(f"[clean:{split}] kept={len(out)} dropped_dup={dropped_dup} dropped_empty={dropped_empty} dropped_non_numeric={dropped_non_numeric}")
    return out

def build_random_subset(records: list[dict], max_records: int | None, seed: int) -> list[dict]:
    if max_records is None or max_records >= len(records):
        return list(records)
    rng = random.Random(seed)
    rows = list(records)
    rng.shuffle(rows)
    return rows[:max_records]

def split_train_calibration(records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    if not USE_INTERNAL_CALIBRATION_SPLIT:
        report = {
            "enabled": False,
            "fit_records": len(records),
            "calib_records": 0,
            "split_fields": ["query_vi"],
        }
        return list(records), [], report
    fit, calib = [], []
    for rec in records:
        if stable_fraction(rec["_query_key"]) < CALIB_FRACTION:
            calib.append(rec)
        else:
            fit.append(rec)
    if CALIB_MAX_RECORDS and len(calib) > CALIB_MAX_RECORDS:
        calib = sorted(calib, key=lambda r: stable_fraction("calib:" + r["_query_key"]))[:CALIB_MAX_RECORDS]
    if not fit or not calib:
        raise RuntimeError("Internal calibration split is empty; adjust CALIB_FRACTION/CALIB_MAX_RECORDS")
    fit_queries = {r["_query_key"] for r in fit}
    calib_queries = {r["_query_key"] for r in calib}
    report = {
        "enabled": True,
        "split_name": "query_vi_hash_internal_calibration",
        "split_fields": ["query_vi"],
        "calib_fraction": CALIB_FRACTION,
        "fit_records": len(fit),
        "calib_records": len(calib),
        "query_overlap": len(fit_queries & calib_queries),
        "note": "Used only for legal threshold/model-candidate calibration from train.json.",
    }
    return fit, calib, report

def build_answer_only_records(records: list[dict], stage_name: str) -> list[dict]:
    out = []
    for rec in records:
        canonical = rec.get("_canonical_answer")
        if canonical is not None:
            out.append({**rec, "response_vi": build_answer_only_target(canonical), "_stage": stage_name})
    print(f"[build:{stage_name}] {len(out)}")
    return out

valid_overlap_audit = save_valid_overlap_audit(train_records, valid_records, VALID_OVERLAP_AUDIT_PATH)
train_clean = clean_records(train_records, "train")
valid_clean = clean_records(valid_records, "valid_reference")

train_fit_clean, train_calib_clean, query_split_report = split_train_calibration(train_clean)
QUERY_DISJOINT_SPLIT_PATH.write_text(json.dumps(query_split_report, ensure_ascii=False, indent=2), encoding="utf-8")
print("[internal-split]", json.dumps(query_split_report, ensure_ascii=False))

training_clean = train_fit_clean if TRAIN_ON_FIT_SPLIT_ONLY else train_clean
training_clean = build_random_subset(training_clean, MAX_TRAIN_SAMPLES, SEED)

train_stage_a = build_answer_only_records(training_clean, STAGE_A_NAME)
full_train_stage_a = build_answer_only_records(train_clean, STAGE_A_NAME + "_full_train_reference")
calib_stage_a = build_answer_only_records(train_calib_clean, STAGE_A_NAME + "_calib") if train_calib_clean else []

print("\nExample target:")
print(train_stage_a[0]["response_vi"])
print("Example prompt:")
print(format_prompt(train_stage_a[0]))
'''


TRAINING_CELL = r'''# ============================================================
# 5. PEFT LoRA SFT: legal answer-only, optional multi-seed
# ============================================================
def build_training_args_for_seed(output_dir: Path, epochs: float, lr: float, seed: int):
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        group_by_length=True,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    if "optim" in sig.parameters and torch.cuda.is_available():
        kwargs["optim"] = "adamw_torch_fused"
    return TrainingArguments(**kwargs)

def build_lora_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.config.pad_token_id = SAFE_EOS_ID
    model.config.eos_token_id = SAFE_EOS_ID
    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        fan_in_fan_out=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def _checkpoint_label(epoch_value: float | None, *, final: bool = False) -> str:
    if final:
        return f"final_epoch_{float(STAGE_A_EPOCHS):.2f}".replace(".", "p")
    if epoch_value is None:
        return "epoch_unknown"
    ev = float(epoch_value)
    if abs(ev - round(ev)) < 1e-3:
        return f"epoch_{int(round(ev)):02d}"
    return f"epoch_{ev:.2f}".replace(".", "p")

class SeedCheckpointCallback(TrainerCallback):
    def __init__(self, root_dir: Path, run_state: dict):
        self.root_dir = Path(root_dir)
        self.run_state = run_state
        self._saved_labels = set()

    def on_epoch_end(self, args, state, control, **kwargs):
        model_obj = kwargs.get("model")
        if model_obj is None or state.epoch is None:
            return control
        label = _checkpoint_label(float(state.epoch))
        if label in self._saved_labels:
            return control
        ckpt_dir = self.root_dir / label
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_obj.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        meta = {
            "label": label,
            "seed": self.run_state["seed"],
            "epoch": float(state.epoch),
            "adapter_dir": str(ckpt_dir),
            "stage_name": STAGE_A_NAME,
            "prompt_template": PROMPT_TEMPLATE,
            "legal_input_fields": LEGAL_INPUT_FIELDS,
            "legal_target_fields": LEGAL_TARGET_FIELDS,
            "saved_at_unix": time.time(),
        }
        (ckpt_dir / "checkpoint_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            meta["sha256"] = sha256_dir(ckpt_dir)
            (ckpt_dir / "model_hash.txt").write_text(meta["sha256"] + "\n", encoding="utf-8")
        except Exception as exc:
            meta["sha256_error"] = repr(exc)
        self.run_state["checkpoints"].append(meta)
        self._saved_labels.add(label)
        print("[checkpoint] saved", label, "->", ckpt_dir)
        return control

def train_lora_for_seed(seed: int, train_records_for_stage: list[dict]) -> dict:
    seed_everything(seed)
    run_name = f"seed_{seed}"
    output_dir = STAGE_A_OUTPUT_DIR / (run_name + "_final")
    checkpoint_root = CHECKPOINT_ROOT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    run_state = {"seed": seed, "run_name": run_name, "checkpoints": []}

    print("\n" + "=" * 90)
    print("[train]", run_name, "records=", len(train_records_for_stage), "epochs=", STAGE_A_EPOCHS, "lr=", STAGE_A_LR)
    model = build_lora_model()
    train_ds = SFTDataset(train_records_for_stage, tokenizer, MAX_LENGTH_STAGE_A, desc=run_name)
    collator = PadCollator(SAFE_EOS_ID)
    eff_batch = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM * max(1, torch.cuda.device_count())
    print("[train]", run_name, "eff_batch=", eff_batch, "steps/epoch=", math.ceil(len(train_ds) / eff_batch))

    callbacks = []
    if SAVE_EPOCH_CHECKPOINTS:
        callbacks.append(SeedCheckpointCallback(checkpoint_root, run_state))

    trainer = Trainer(
        model=model,
        args=build_training_args_for_seed(output_dir, STAGE_A_EPOCHS, STAGE_A_LR, seed),
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    model_hash = sha256_dir(output_dir)
    (output_dir / "model_hash.txt").write_text(model_hash + "\n", encoding="utf-8")
    run_state.update({
        "final_adapter_dir": str(output_dir),
        "checkpoint_root": str(checkpoint_root),
        "wall_seconds": wall,
        "final_sha256": model_hash,
    })
    (checkpoint_root / "checkpoint_index.json").write_text(json.dumps(run_state["checkpoints"], ensure_ascii=False, indent=2), encoding="utf-8")
    print("[train]", run_name, "wall_min=", round(wall / 60, 2), "saved=", output_dir)
    del trainer, model
    torch.cuda.empty_cache()
    return run_state

TRAINED_RUNS = []
for seed in TRAIN_SEEDS:
    TRAINED_RUNS.append(train_lora_for_seed(int(seed), train_stage_a))

PRIMARY_RUN = TRAINED_RUNS[0]
PRIMARY_ADAPTER_DIR = Path(PRIMARY_RUN["final_adapter_dir"])
SFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copytree(PRIMARY_ADAPTER_DIR, SFT_OUTPUT_DIR, dirs_exist_ok=True)
shutil.copytree(PRIMARY_ADAPTER_DIR, FINAL_OUTPUT_DIR, dirs_exist_ok=True)
(CHECKPOINT_ROOT_DIR / "trained_runs.json").write_text(json.dumps(TRAINED_RUNS, ensure_ascii=False, indent=2), encoding="utf-8")
print("[train] trained_runs=", len(TRAINED_RUNS), "primary=", PRIMARY_ADAPTER_DIR)
'''


PROMPT_HELPER_CELL = r'''# ============================================================
# 6. Prompt helper / RL disabled
# ============================================================
def build_prompt(rec: dict) -> str:
    return format_prompt(rec)

rl_summary = {
    "enabled": False,
    "reason": "This notebook is legal answer-only SFT; no GRPO/RL stage.",
    "final_dir": str(FINAL_OUTPUT_DIR),
}
RL_SUMMARY_PATH.write_text(json.dumps(rl_summary, ensure_ascii=False, indent=2), encoding="utf-8")

try:
    del model
except NameError:
    pass
torch.cuda.empty_cache()
'''


GENERATION_COMMON = r'''# ============================================================
# 7. Generation + legal answer-only candidate utilities
# ============================================================
def has_peft_adapter(path_like) -> bool:
    p = Path(path_like)
    return p.exists() and (p / "adapter_config.json").exists()

def load_model_for_generation(adapter_dir: Path):
    dtype = torch.float16 if (INFER_FP16 and torch.cuda.is_available()) else torch.float32
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, local_files_only=True)
    base.config.pad_token_id = SAFE_EOS_ID
    base.config.eos_token_id = SAFE_EOS_ID
    if has_peft_adapter(adapter_dir):
        print("[infer] base + adapter:", adapter_dir)
        gen_model = PeftModel.from_pretrained(base, str(adapter_dir), local_files_only=True)
        try:
            gen_model = gen_model.merge_and_unload()
            print("[infer] merged LoRA adapter")
        except Exception as exc:
            print("[infer] merge failed; using PEFT wrapper:", repr(exc))
    else:
        print("[infer] no adapter at", adapter_dir, "; using base model")
        gen_model = base
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_model.to(device)
    gen_model.eval()
    return gen_model

class StopOnAnswerLine(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int, eos_id: int = SAFE_EOS_ID, patience_tokens: int = 8):
        self.tok = tokenizer
        self.prompt_len = prompt_len
        self.eos_id = eos_id
        self.patience = patience_tokens
        self.re_answer = re.compile(r"đáp\s*án\s*l[àa]\s*[:：]\s*-?\d", re.IGNORECASE)
        self._matched_at = None

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        seq = input_ids[0]
        if seq[-1].item() == self.eos_id:
            return True
        gen_tail = seq[self.prompt_len:]
        if gen_tail.numel() < 4:
            return False
        text = decode_model_text(self.tok, gen_tail)
        m = self.re_answer.search(text)
        if not m:
            return False
        if self._matched_at is None:
            self._matched_at = gen_tail.numel()
        if "\n" in text[m.end():]:
            return True
        if gen_tail.numel() - self._matched_at >= self.patience:
            return True
        return False

@torch.inference_mode()
def generate_model_outputs(adapter_dir: Path, records: list[dict], output_path: Path, *, max_new_tokens: int = MAX_NEW_TOKENS, num_beams: int = NUM_BEAMS):
    gen_tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    gen_tok.pad_token_id = SAFE_EOS_ID
    gen_tok.eos_token_id = SAFE_EOS_ID
    gen_tok.padding_side = "left"
    gen_tok.truncation_side = "left"
    gen_model = load_model_for_generation(adapter_dir)
    device = next(gen_model.parameters()).device
    outputs = []
    n_pos = int(getattr(gen_model.config, "n_positions", getattr(gen_model.config, "max_position_embeddings", 1024)))
    vocab_n = gen_model.get_input_embeddings().num_embeddings
    for idx, rec in enumerate(tqdm(records, desc="generate:" + Path(adapter_dir).name)):
        prompt = build_prompt(rec)
        prompt_budget = max(8, n_pos - max_new_tokens)
        full_ids = gen_tok(prompt, add_special_tokens=False)["input_ids"]
        if len(full_ids) > prompt_budget:
            full_ids = full_ids[-prompt_budget:]
        full_ids = [min(t, vocab_n - 1) for t in full_ids]
        ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(ids)
        prompt_len = ids.shape[1]
        eff_new = max(8, min(max_new_tokens, n_pos - prompt_len))
        gen_kwargs = dict(
            input_ids=ids,
            attention_mask=attn,
            max_new_tokens=eff_new,
            pad_token_id=SAFE_EOS_ID,
            eos_token_id=SAFE_EOS_ID,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            stopping_criteria=StoppingCriteriaList([StopOnAnswerLine(gen_tok, prompt_len=prompt_len)]),
        )
        if num_beams and num_beams > 1:
            gen_kwargs.update(dict(num_beams=num_beams, do_sample=False, early_stopping=True, length_penalty=LENGTH_PENALTY))
        else:
            gen_kwargs.update(dict(num_beams=1, do_sample=False))
        seqs = gen_model.generate(**gen_kwargs)
        text = decode_model_text(gen_tok, seqs[0, prompt_len:])
        if SANITIZE_TO_ANSWER_ONLY:
            text = sanitize_model_output(text)
        outputs.append({
            "id": rec.get("id", idx),
            "query_vi": rec.get("query_vi", ""),
            "type": rec.get("type"),  # copied only for required output format/reporting
            "model_output": text.strip(),
        })
    output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[infer] wrote", len(outputs), "rows ->", output_path)
    del gen_model
    torch.cuda.empty_cache()
    return outputs

def _safe_num_key(value, digits: int = 10):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if abs(value - round(value)) <= 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}g}"

def _answer_num_from_key(key):
    if key is None:
        return None
    try:
        return float(str(key).replace(",", "."))
    except Exception:
        return None

def answer_key_from_output(item: dict):
    _answer, pred_num = extract_pred(item)
    return _safe_num_key(pred_num)

def build_answer_priors(records: list[dict]) -> Counter:
    counts = Counter()
    for rec in records:
        key = _safe_num_key(rec.get("_gold_num"))
        if key is not None:
            counts[key] += 1
    return counts

ANSWER_PRIORS = build_answer_priors(train_clean)

def candidate_entries_from_runs() -> list[dict]:
    entries = []
    seed_order = {int(seed): i for i, seed in enumerate(TRAIN_SEEDS)}
    for run in TRAINED_RUNS:
        seed = int(run["seed"])
        ckpts = list(run.get("checkpoints") or [])
        ckpts.sort(key=lambda x: float(x.get("epoch") or 0.0))
        if ENSEMBLE_INCLUDE_EPOCH_CHECKPOINTS and ENSEMBLE_LAST_K_EPOCHS:
            ckpts = ckpts[-int(ENSEMBLE_LAST_K_EPOCHS):]
            for ck in ckpts:
                epoch = float(ck.get("epoch") or 0.0)
                entries.append({
                    "name": "model::seed_%s::%s" % (seed, ck["label"]),
                    "kind": "model_epoch",
                    "seed": seed,
                    "epoch": epoch,
                    "adapter_dir": Path(ck["adapter_dir"]),
                    "priority": 1000 * (1 + epoch) - seed_order.get(seed, 0),
                })
        if ENSEMBLE_INCLUDE_FINAL:
            entries.append({
                "name": "model::seed_%s::final" % seed,
                "kind": "model_final",
                "seed": seed,
                "epoch": float(STAGE_A_EPOCHS),
                "adapter_dir": Path(run["final_adapter_dir"]),
                "priority": 10000 + 1000 * float(STAGE_A_EPOCHS) - seed_order.get(seed, 0),
            })
    if ENSEMBLE_CANDIDATE_LIMIT:
        entries = entries[:int(ENSEMBLE_CANDIDATE_LIMIT)]
    print("[candidates] entries=", len(entries), [e["name"] for e in entries])
    return entries

def _candidate_path(split_name: str, candidate_name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_name)
    ENSEMBLE_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    return ENSEMBLE_CANDIDATE_DIR / ("%s_%s.json" % (split_name, safe))

def generate_model_candidates(records: list[dict], split_name: str, entries: list[dict]) -> list[dict]:
    candidates = []
    for entry in entries:
        out_path = _candidate_path(split_name, entry["name"])
        if out_path.exists():
            outputs = json.loads(out_path.read_text(encoding="utf-8"))
            print("[candidates] reuse", out_path)
        else:
            outputs = generate_model_outputs(entry["adapter_dir"], records, out_path, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS)
        candidates.append({**entry, "outputs": outputs, "path": str(out_path)})
    return candidates

def make_answer_item(rec: dict, idx: int, key: str | None, fallback_item: dict | None = None) -> dict:
    if key is not None:
        text = "Đáp án là: " + str(key)
    elif fallback_item:
        text = fallback_item.get("model_output", "")
    else:
        text = ""
    return {
        "id": rec.get("id", idx),
        "query_vi": rec.get("query_vi", ""),
        "type": rec.get("type"),  # output-format copy only
        "model_output": text,
    }

def model_consensus_for_row(model_candidates: list[dict], row_idx: int):
    groups = defaultdict(list)
    fallback = None
    for cand in model_candidates:
        item = cand["outputs"][row_idx]
        if fallback is None:
            fallback = item
        key = answer_key_from_output(item)
        if key is not None:
            groups[key].append(cand)
    if not groups:
        return {
            "key": None,
            "agreement_count": 0,
            "agreement_frac": 0.0,
            "fallback": fallback,
            "chosen_candidate": None,
        }
    scored = []
    n = max(1, len(model_candidates))
    for key, vals in groups.items():
        best_priority = max(float(v.get("priority", 0.0)) for v in vals)
        prior = ANSWER_PRIORS.get(key, 0)
        num = _answer_num_from_key(key)
        scored.append((len(vals), prior, best_priority, -abs(num or 0.0), key, vals))
    scored.sort(reverse=True, key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    count, prior, best_priority, _neg_abs, key, vals = scored[0]
    chosen_candidate = max(vals, key=lambda c: float(c.get("priority", 0.0)))
    return {
        "key": key,
        "agreement_count": int(count),
        "agreement_frac": float(count) / n,
        "fallback": chosen_candidate["outputs"][row_idx],
        "chosen_candidate": chosen_candidate["name"],
        "prior": int(prior),
    }

def choose_answer_only_consensus(model_candidates: list[dict], records: list[dict]) -> tuple[list[dict], dict]:
    outputs = []
    agreement_hist = Counter()
    chosen_model_counts = Counter()
    for idx, rec in enumerate(records):
        decision = model_consensus_for_row(model_candidates, idx)
        outputs.append(make_answer_item(rec, idx, decision["key"], decision.get("fallback")))
        agreement_hist[str(decision["agreement_count"])] += 1
        if decision.get("chosen_candidate"):
            chosen_model_counts[decision["chosen_candidate"]] += 1
    summary = {
        "strategy": "legal_answer_only_model_consensus",
        "num_model_candidates": len(model_candidates),
        "agreement_hist": dict(agreement_hist.most_common()),
        "chosen_model_counts": dict(chosen_model_counts.most_common()),
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
    }
    return outputs, summary

def save_outputs_and_optional_report(records: list[dict], outputs: list[dict], output_path: Path, report_path: Path, summary_path: Path, summary: dict):
    output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = dict(summary)
    if records and records[0].get("response_vi"):
        report = save_eval_report(output_path, records, report_path)
        payload["summary"] = report["summary"]
        payload["by_type"] = report["by_type"]
    else:
        report = None
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, payload
'''


ANSWER_ONLY_RUN = r'''
def run_answer_only_split(records: list[dict], split_name: str, output_path: Path, report_path: Path):
    entries = candidate_entries_from_runs()
    candidates = generate_model_candidates(records, split_name, entries)
    primary = next((c for c in candidates if c["kind"] == "model_final" and int(c["seed"]) == int(TRAIN_SEEDS[0])), candidates[-1])
    if split_name == "valid":
        MODEL_VALID_OUTPUT_PATH.write_text(json.dumps(primary["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
        if records and records[0].get("response_vi"):
            save_eval_report(MODEL_VALID_OUTPUT_PATH, records, MODEL_VALID_REPORT_PATH)
    elif split_name == "test":
        MODEL_TEST_OUTPUT_PATH.write_text(json.dumps(primary["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    outputs, summary = choose_answer_only_consensus(candidates, records)
    report, payload = save_outputs_and_optional_report(records, outputs, output_path, report_path, ENSEMBLE_RANKER_REPORT_PATH, summary)
    if report is not None:
        print("[answer-only:%s]" % split_name, report["summary"])
    return candidates, outputs, payload

if RUN_MODE == "phase1":
    _candidates, _outputs, _payload = run_answer_only_split(valid_clean, "valid", VALID_OUTPUT_PATH, VALID_REPORT_PATH)
elif RUN_MODE == "phase2":
    if not TEST_FILE.exists():
        raise FileNotFoundError("RUN_MODE='phase2' requires test.json")
    test_records = load_records(TEST_FILE)
    _candidates, _outputs, _payload = run_answer_only_split(test_records, "test", TEST_OUTPUT_PATH, VALID_REPORT_PATH)
    print("[phase2] wrote", TEST_OUTPUT_PATH)
else:
    raise ValueError("Unknown RUN_MODE=" + str(RUN_MODE))
'''


QUERY_RETRIEVAL_RUN = r'''
# ============================================================
# 8. Legal query_vi-only retrieval + train-internal gate sweep
# ============================================================
class QueryOnlyRetriever:
    def __init__(self, records: list[dict]):
        self.records = [r for r in records if r.get("_canonical_answer") is not None and r.get("_gold_num") is not None]
        self.texts = [normalize_text_key(r.get("query_vi")) for r in self.records]
        self.token_sets = [query_tokens(t) for t in self.texts]
        self.backend = "jaccard"
        self.vectorizer = None
        self.matrix = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(analyzer=RETRIEVAL_ANALYZER, ngram_range=RETRIEVAL_NGRAM_RANGE, lowercase=False, min_df=1)
            self.matrix = self.vectorizer.fit_transform(self.texts)
            self.backend = "tfidf_char_ngram"
        except Exception as exc:
            print("[retrieval] sklearn TF-IDF unavailable; using Jaccard fallback:", repr(exc))
        print("[retrieval] backend=", self.backend, "records=", len(self.records))

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
        if not self.records:
            return []
        q = normalize_text_key(query)
        if self.backend == "tfidf_char_ngram" and self.vectorizer is not None and self.matrix is not None:
            from sklearn.metrics.pairwise import linear_kernel
            qv = self.vectorizer.transform([q])
            sims = linear_kernel(qv, self.matrix).ravel()
            top_idx = sims.argsort()[-top_k:][::-1]
            rows = [(int(i), float(sims[i])) for i in top_idx if float(sims[i]) > 0.0]
        else:
            qtok = query_tokens(q)
            scored = [(i, jaccard(qtok, toks)) for i, toks in enumerate(self.token_sets)]
            scored.sort(key=lambda x: x[1], reverse=True)
            rows = [(i, float(s)) for i, s in scored[:top_k] if s > 0.0]
        hits = []
        qtok = query_tokens(q)
        for i, sim in rows:
            rec = self.records[i]
            hits.append({
                "query_vi": rec.get("query_vi", ""),
                "answer_key": _safe_num_key(rec.get("_gold_num")),
                "canonical_answer": rec.get("_canonical_answer"),
                "answer_num": float(rec.get("_gold_num")),
                "similarity": float(sim),
                "jaccard": jaccard(qtok, rec.get("_query_tokens") or set()),
            })
        return hits

def retrieval_decision_for_record(rec: dict, retriever: QueryOnlyRetriever, gate: dict) -> dict:
    hits = retriever.search(rec.get("query_vi", ""), top_k=RETRIEVAL_TOP_K)
    if not hits:
        return {"used": False, "reason": "no_hits", "hits": 0}
    groups = defaultdict(list)
    for h in hits:
        if h["answer_key"] is not None:
            groups[h["answer_key"]].append(h)
    if not groups:
        return {"used": False, "reason": "no_numeric_hits", "hits": len(hits)}
    ranked = []
    for key, vals in groups.items():
        ranked.append({
            "answer_key": key,
            "count": len(vals),
            "top_sim": max(v["similarity"] for v in vals),
            "top_jaccard": max(v["jaccard"] for v in vals),
            "canonical_answer": vals[0]["canonical_answer"],
        })
    ranked.sort(key=lambda x: (x["count"], x["top_sim"], x["top_jaccard"], ANSWER_PRIORS.get(x["answer_key"], 0)), reverse=True)
    top = ranked[0]
    second_count = ranked[1]["count"] if len(ranked) > 1 else 0
    majority_frac = top["count"] / max(1, len(hits))
    margin = (top["count"] - second_count) / max(1, len(hits))
    passed = (
        top["top_sim"] >= float(gate.get("min_sim", 0.0))
        and majority_frac >= float(gate.get("min_majority_frac", 0.0))
        and margin >= float(gate.get("min_margin", 0.0))
    )
    return {
        "used": bool(passed),
        "reason": "passed" if passed else "low_confidence",
        "answer_key": top["answer_key"],
        "hits": len(hits),
        "majority_frac": majority_frac,
        "margin": margin,
        "top_sim": top["top_sim"],
        "top_jaccard": top["top_jaccard"],
        "gate": dict(gate),
        "ranked": ranked[:3],
    }

def choose_query_retrieval_hybrid(model_candidates: list[dict], records: list[dict], retriever: QueryOnlyRetriever, gate: dict):
    outputs = []
    decisions = []
    reason_counts = Counter()
    source_counts = Counter()
    for idx, rec in enumerate(records):
        model_dec = model_consensus_for_row(model_candidates, idx)
        ret_dec = retrieval_decision_for_record(rec, retriever, gate)
        chosen_key = model_dec["key"]
        source = "model_consensus"
        if ret_dec.get("used"):
            ret_key = ret_dec.get("answer_key")
            if ret_key == model_dec["key"]:
                chosen_key = ret_key
                source = "retrieval_model_agree"
            elif model_dec["agreement_frac"] <= float(gate.get("model_low_conf_frac", 0.5)):
                chosen_key = ret_key
                source = "retrieval_low_model_conf"
            else:
                source = "model_fallback_disagree"
        reason_counts[ret_dec.get("reason", "unknown")] += 1
        source_counts[source] += 1
        outputs.append(make_answer_item(rec, idx, chosen_key, model_dec.get("fallback")))
        decisions.append({
            "id": rec.get("id", idx),
            "model_key": model_dec.get("key"),
            "model_agreement_frac": model_dec.get("agreement_frac"),
            "retrieval_key": ret_dec.get("answer_key"),
            "retrieval_used": ret_dec.get("used", False),
            "source": source,
            "retrieval": {k: v for k, v in ret_dec.items() if k != "ranked"},
        })
    summary = {
        "strategy": "legal_query_vi_retrieval_with_model_consensus_fallback",
        "gate": dict(gate),
        "num_model_candidates": len(model_candidates),
        "retrieval_backend": retriever.backend,
        "source_counts": dict(source_counts.most_common()),
        "retrieval_reason_counts": dict(reason_counts.most_common()),
        "legal_input_fields": LEGAL_INPUT_FIELDS,
        "legal_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
    }
    return outputs, decisions, summary

def sweep_retrieval_gate_on_calib(model_candidates: list[dict], calib_records: list[dict], retriever: QueryOnlyRetriever) -> tuple[dict, dict]:
    if not RETRIEVAL_SWEEP_ENABLED or not calib_records:
        return dict(DEFAULT_RETRIEVAL_GATE), {"enabled": False, "selected_gate": dict(DEFAULT_RETRIEVAL_GATE)}
    rows = []
    best = None
    for gate in RETRIEVAL_GATE_GRID:
        outputs, decisions, summary = choose_query_retrieval_hybrid(model_candidates, calib_records, retriever, gate)
        report = evaluate_predictions(outputs, calib_records)
        row = {
            "gate": dict(gate),
            "summary": report["summary"],
            "source_counts": summary["source_counts"],
            "retrieval_reason_counts": summary["retrieval_reason_counts"],
        }
        rows.append(row)
        key = (report["summary"]["raw_score"], report["summary"]["extractable"], -sum(1 for d in decisions if d["source"].startswith("retrieval")))
        if best is None or key > best[0]:
            best = (key, row)
    selected = dict(best[1]["gate"])
    payload = {
        "enabled": True,
        "split": "train_internal_calibration",
        "calib_records": len(calib_records),
        "selected_gate": selected,
        "rows": rows,
    }
    return selected, payload

def run_query_retrieval_split(records: list[dict], split_name: str, output_path: Path, report_path: Path, selected_gate: dict, retriever: QueryOnlyRetriever):
    entries = candidate_entries_from_runs()
    model_candidates = generate_model_candidates(records, split_name, entries)
    primary = next((c for c in model_candidates if c["kind"] == "model_final" and int(c["seed"]) == int(TRAIN_SEEDS[0])), model_candidates[-1])
    if split_name == "valid":
        MODEL_VALID_OUTPUT_PATH.write_text(json.dumps(primary["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
        if records and records[0].get("response_vi"):
            save_eval_report(MODEL_VALID_OUTPUT_PATH, records, MODEL_VALID_REPORT_PATH)
    elif split_name == "test":
        MODEL_TEST_OUTPUT_PATH.write_text(json.dumps(primary["outputs"], ensure_ascii=False, indent=2), encoding="utf-8")
    outputs, decisions, summary = choose_query_retrieval_hybrid(model_candidates, records, retriever, selected_gate)
    summary["decisions_sample"] = decisions[:50]
    report, payload = save_outputs_and_optional_report(records, outputs, output_path, report_path, ENSEMBLE_RANKER_REPORT_PATH, summary)
    (WORKING_DIR / ("%s_query_retrieval_decisions.json" % split_name)).write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    if report is not None:
        print("[query-retrieval:%s]" % split_name, report["summary"])
    return model_candidates, outputs, payload

fit_retriever = QueryOnlyRetriever(train_fit_clean if train_fit_clean else train_clean)
full_retriever = QueryOnlyRetriever(train_clean if INFERENCE_RETRIEVAL_USES_FULL_TRAIN else (train_fit_clean if train_fit_clean else train_clean))

if RETRIEVAL_SWEEP_ENABLED and train_calib_clean:
    calib_entries = candidate_entries_from_runs()
    calib_model_candidates = generate_model_candidates(train_calib_clean, "calib", calib_entries)
    SELECTED_RETRIEVAL_GATE, gate_payload = sweep_retrieval_gate_on_calib(calib_model_candidates, train_calib_clean, fit_retriever)
else:
    SELECTED_RETRIEVAL_GATE, gate_payload = dict(DEFAULT_RETRIEVAL_GATE), {"enabled": False, "selected_gate": dict(DEFAULT_RETRIEVAL_GATE)}
(WORKING_DIR / "selected_query_retrieval_gate.json").write_text(json.dumps(gate_payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("[retrieval-gate]", json.dumps(gate_payload.get("selected_gate"), ensure_ascii=False))

if RUN_MODE == "phase1":
    _candidates, _outputs, _payload = run_query_retrieval_split(valid_clean, "valid", VALID_OUTPUT_PATH, VALID_REPORT_PATH, SELECTED_RETRIEVAL_GATE, full_retriever)
elif RUN_MODE == "phase2":
    if not TEST_FILE.exists():
        raise FileNotFoundError("RUN_MODE='phase2' requires test.json")
    test_records = load_records(TEST_FILE)
    _candidates, _outputs, _payload = run_query_retrieval_split(test_records, "test", TEST_OUTPUT_PATH, VALID_REPORT_PATH, SELECTED_RETRIEVAL_GATE, full_retriever)
    print("[phase2] wrote", TEST_OUTPUT_PATH)
else:
    raise ValueError("Unknown RUN_MODE=" + str(RUN_MODE))
'''


MANIFEST_TEMPLATE = r'''# ============================================================
# 9. Output manifest
# ============================================================
def _path_exists_str(p):
    try:
        return Path(p).exists()
    except Exception:
        return False

def _maybe_json_summary(path: Path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("summary", data)
    except Exception as exc:
        return {"error": repr(exc)}

manifest = {
    "notebook_version": NOTEBOOK_VERSION,
    "run_mode": RUN_MODE,
    "created_at_unix": time.time(),
    "rule_compliance": {
        "model_input_fields": LEGAL_INPUT_FIELDS,
        "training_target_fields": LEGAL_TARGET_FIELDS,
        "disallowed_model_feature_fields": DISALLOWED_MODEL_FEATURE_FIELDS,
        "uses_original_question_fields": False,
        "uses_type_for_prompt_or_routing": False,
        "type_is_copied_only_for_prediction_format_and_reports": True,
        "uses_valid_labels_for_ranker_training": False,
        "retrieval_basis": "__RETRIEVAL_BASIS__",
    },
    "data": {
        "train_file": str(TRAIN_FILE),
        "valid_file": str(VALID_FILE),
        "test_file": str(TEST_FILE),
        "valid_overlap_audit": str(VALID_OVERLAP_AUDIT_PATH),
        "query_internal_split_report": str(QUERY_DISJOINT_SPLIT_PATH),
        "train_clean_n": len(globals().get("train_clean", [])),
        "train_fit_n": len(globals().get("train_fit_clean", [])),
        "train_calib_n": len(globals().get("train_calib_clean", [])),
        "valid_clean_n": len(globals().get("valid_clean", [])),
    },
    "config": {
        "prompt_template": PROMPT_TEMPLATE,
        "safe_eos_id": SAFE_EOS_ID,
        "train_seeds": TRAIN_SEEDS,
        "stage_a_epochs": STAGE_A_EPOCHS,
        "stage_a_lr": STAGE_A_LR,
        "max_length_stage_a": MAX_LENGTH_STAGE_A,
        "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_beams": NUM_BEAMS,
        "ensemble_last_k_epochs": ENSEMBLE_LAST_K_EPOCHS,
        "legal_query_retrieval_enabled": LEGAL_QUERY_RETRIEVAL_ENABLED,
        "selected_retrieval_gate": globals().get("SELECTED_RETRIEVAL_GATE"),
        "use_internal_calibration_split": USE_INTERNAL_CALIBRATION_SPLIT,
        "calib_fraction": CALIB_FRACTION,
    },
    "dirs": {
        "stage_a_output_dir": str(STAGE_A_OUTPUT_DIR),
        "sft_output_dir": str(SFT_OUTPUT_DIR),
        "final_output_dir": str(FINAL_OUTPUT_DIR),
        "checkpoint_root_dir": str(CHECKPOINT_ROOT_DIR),
        "checkpoint_eval_dir": str(CHECKPOINT_EVAL_DIR),
    },
    "outputs": {
        "valid_output": str(VALID_OUTPUT_PATH),
        "valid_report": str(VALID_REPORT_PATH),
        "model_valid_output": str(MODEL_VALID_OUTPUT_PATH),
        "model_valid_report": str(MODEL_VALID_REPORT_PATH),
        "ensemble_or_gate_report": str(ENSEMBLE_RANKER_REPORT_PATH),
        "ensemble_candidate_dir": str(ENSEMBLE_CANDIDATE_DIR),
        "test_predictions": str(TEST_OUTPUT_PATH),
        "model_test_predictions": str(MODEL_TEST_OUTPUT_PATH),
    },
    "trained_runs": globals().get("TRAINED_RUNS", []),
    "reference_valid_summary": _maybe_json_summary(VALID_REPORT_PATH),
    "model_reference_valid_summary": _maybe_json_summary(MODEL_VALID_REPORT_PATH),
    "selection_summary": _maybe_json_summary(ENSEMBLE_RANKER_REPORT_PATH),
    "final_output_dir_exists": _path_exists_str(FINAL_OUTPUT_DIR),
}
manifest_path = WORKING_DIR / (NOTEBOOK_VERSION + "_manifest.json")
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print("[manifest] wrote", manifest_path)
'''


def build_config_cell(
    *,
    version: str,
    artifact_prefix: str,
    train_seeds: list[int],
    stage_a_epochs: float,
    stage_a_lr: float,
    ensemble_last_k_epochs: int,
    use_calib: bool,
    calib_fraction: float,
    calib_max_records: int | None,
    legal_query_retrieval_enabled: bool,
) -> str:
    return (
        CONFIG_TEMPLATE
        .replace("__VERSION__", version)
        .replace("__ARTIFACT_PREFIX__", artifact_prefix)
        .replace("__TRAIN_SEEDS__", repr(train_seeds))
        .replace("__STAGE_A_EPOCHS__", repr(stage_a_epochs))
        .replace("__STAGE_A_LR__", repr(stage_a_lr))
        .replace("__ENSEMBLE_LAST_K_EPOCHS__", repr(ensemble_last_k_epochs))
        .replace("__USE_INTERNAL_CALIBRATION_SPLIT__", repr(use_calib))
        .replace("__CALIB_FRACTION__", repr(calib_fraction))
        .replace("__CALIB_MAX_RECORDS__", repr(calib_max_records))
        .replace("__LEGAL_QUERY_RETRIEVAL_ENABLED__", repr(legal_query_retrieval_enabled))
    )


def build_manifest_cell(retrieval_basis: str) -> str:
    return MANIFEST_TEMPLATE.replace("__RETRIEVAL_BASIS__", retrieval_basis)


def build_notebook(*, kind: str) -> dict:
    base_nb, base_cells = read_base_cells()
    nb = copy.deepcopy(base_nb)
    import_cell = copy.deepcopy(base_cells[1])
    eval_cell = copy.deepcopy(base_cells[3])
    eval_cell["source"] = scrub_eval_cell("".join(eval_cell["source"])).splitlines(keepends=True)
    dataset_cell = copy.deepcopy(base_cells[5])

    if kind == "answer_only":
        title = "# V16 - Legal Answer-Only Multi-Seed\n\nUses only `query_vi` as model input and `response_vi` as train/validation target. No `type` or `original_*` fields are used for prompt, routing, retrieval, or model selection."
        config = build_config_cell(
            version="v16_legal_answer_only_multiseed",
            artifact_prefix="v16_legal_answer_only_multiseed",
            train_seeds=[42, 123],
            stage_a_epochs=8.0,
            stage_a_lr=1e-3,
            ensemble_last_k_epochs=3,
            use_calib=False,
            calib_fraction=0.10,
            calib_max_records=None,
            legal_query_retrieval_enabled=False,
        )
        run_cell = GENERATION_COMMON + "\n" + ANSWER_ONLY_RUN
        manifest = build_manifest_cell("none_answer_only_model_consensus")
    elif kind == "query_retrieval":
        title = "# V16 - Legal Query-Only Retrieval Ranker\n\nUses only train `query_vi` for retrieval keys and train `response_vi` for answers/calibration. No `original_*` fields and no `type` routing."
        config = build_config_cell(
            version="v16_legal_query_retrieval_ranker",
            artifact_prefix="v16_legal_query_retrieval_ranker",
            train_seeds=[42],
            stage_a_epochs=8.0,
            stage_a_lr=1e-3,
            ensemble_last_k_epochs=3,
            use_calib=True,
            calib_fraction=0.10,
            calib_max_records=1000,
            legal_query_retrieval_enabled=True,
        )
        run_cell = GENERATION_COMMON + "\n" + QUERY_RETRIEVAL_RUN
        manifest = build_manifest_cell("train_query_vi_similarity_to_train_response_vi_answer")
    else:
        raise ValueError(kind)

    nb["cells"] = [
        markdown_cell(title),
        import_cell,
        code_cell(config),
        eval_cell,
        code_cell(DATA_CELL),
        dataset_cell,
        code_cell(TRAINING_CELL),
        code_cell(PROMPT_HELPER_CELL),
        code_cell(run_cell),
        code_cell(manifest),
    ]
    return nb


def validate_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        try:
            ast.parse(src)
        except SyntaxError as exc:
            raise SyntaxError(f"{path.name} cell {idx}: {exc}") from exc
    print("[validate]", path, "cells=", len(nb.get("cells", [])))


def main() -> None:
    notebooks = {
        ANSWER_ONLY_NOTEBOOK: build_notebook(kind="answer_only"),
        QUERY_RETRIEVAL_NOTEBOOK: build_notebook(kind="query_retrieval"),
    }
    for path, nb in notebooks.items():
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        validate_notebook(path)
        print("[wrote]", path, "bytes=", path.stat().st_size)


if __name__ == "__main__":
    main()
