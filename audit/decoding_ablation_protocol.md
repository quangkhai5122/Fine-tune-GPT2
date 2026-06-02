# Decoding ablation protocol

Goal: compare decoding strategies while changing **only** the decoder.

## Fixed across all runs

- Same trained checkpoint
- Same `valid.json`
- Same prompt template
- Same `MAX_NEW_TOKENS`
- Same robust evaluator from `math_eval.py`
- Same output schema

## Required runs

| run id | checkpoint | decode |
|---|---|---|
| `v2_greedy_same_ckpt` | same V2 checkpoint | greedy |
| `v2_beam2_same_ckpt` | same V2 checkpoint | beam=2 |
| `v2_beam4_same_ckpt` | same V2 checkpoint | beam=4 |

In `finetune_gpt2_for_math_v2.ipynb`, set:

```python
RUN_DECODING_ABLATIONS = True
```

The notebook will emit all three runs from the same trained checkpoint.

## Recommended output layout

```text
results/
  ablations/
    greedy/
      valid_output.json
    beam2/
      valid_output.json
    beam4/
      valid_output.json
```

Then run:

```powershell
python audit\compare_runs.py `
  --manifest audit\fair_runs_manifest.json `
  --out-json audit\fair_comparison.json `
  --out-md audit\fair_comparison.md
```

## Important note

Do **not** compare:

- baseline trained with one pipeline
- V2 trained with another pipeline
- and different decoders

as if that were a clean decoding ablation.  
That comparison is useful descriptively, but not causal.
