# LLM/VLM GPU Memory Estimator

Command-line tool and library for estimating per-GPU VRAM requirements for LLM/VLM fine-tuning.

## Quick Start

### Estimation Only (No GPU Required)

```bash
python scripts/estimate_memory_budget.py \
  --model meta-llama/Llama-2-7b-hf \
  --dtype bf16 --seq-len 4096 \
  --per-device-batch 2 --grad-accum 128 \
  --zero 2 --dp 8 --tp 1 --pp 1 \
  --flashattn true --grad-checkpoint true
```

### Empirical Probe (GPU Required)

```bash
python scripts/estimate_memory_budget.py \
  --model meta-llama/Llama-2-7b-hf \
  --dtype bf16 --seq-len 4096 \
  --per-device-batch 2 --grad-accum 128 \
  --probe true --warmup-steps 3 --probe-steps 6
```

## Installation

### For Estimation Only (CPU)

```bash
pip install transformers accelerate torch pandas
```

CPU-only torch is sufficient because estimation only:
- Loads model configs (no actual model weights)
- Uses `init_empty_weights()` for parameter counting (meta tensors)
- Performs mathematical calculations

### For Empirical Probe (GPU)

```bash
# CUDA 12.1
pip install transformers accelerate pandas
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage Examples

### 1. Full Fine-Tuning (7B Model)

```bash
python scripts/estimate_memory_budget.py \
  --model meta-llama/Llama-2-7b-hf \
  --dtype bf16 \
  --seq-len 4096 \
  --per-device-batch 2 \
  --grad-accum 128 \
  --zero 2 \
  --dp 8 \
  --flashattn true \
  --grad-checkpoint true
```

**Expected Output:**
```
=== Memory Estimate (per GPU) ===
Model: meta-llama/Llama-2-7b-hf
dtype=bf16, seq_len=4096, micro_batch=2, GA=128
DP=8, TP=1, PP=1, ZeRO=2, LoRA=False, QLoRA=False

Base weights        :  13.48 GiB
Trainable state     :   6.74 GiB
Attention activ.    :   2.25 GiB
MLP activ.          :   3.37 GiB
Output logits       :   1.00 GiB
...
Total (peak)        :  33.21 GiB
```

### 2. LoRA Fine-Tuning (13B Model)

```bash
python scripts/estimate_memory_budget.py \
  --model meta-llama/Llama-2-13b-hf \
  --dtype bf16 \
  --seq-len 8192 \
  --per-device-batch 1 \
  --grad-accum 256 \
  --lora true \
  --lora-rank 8 \
  --lora-target-modules q_proj k_proj v_proj o_proj \
  --zero 3 \
  --dp 16 \
  --flashattn true \
  --grad-checkpoint true
```

### 3. QLoRA (70B Model)

```bash
python scripts/estimate_memory_budget.py \
  --model meta-llama/Llama-2-70b-hf \
  --dtype bf16 \
  --qlora true \
  --lora-rank 64 \
  --seq-len 4096 \
  --per-device-batch 1 \
  --grad-accum 512 \
  --zero 3 \
  --dp 32 \
  --tp 4 \
  --flashattn true \
  --grad-checkpoint true
```

### 4. With Empirical Probe

```bash
python scripts/estimate_memory_budget.py \
  --model mistralai/Mistral-7B-v0.1 \
  --dtype bf16 \
  --seq-len 4096 \
  --per-device-batch 1 \
  --grad-accum 64 \
  --flashattn true \
  --grad-checkpoint true \
  --probe true \
  --warmup-steps 3 \
  --probe-steps 6
```

**Expected Output:**
```
--- Empirical Probe ---
max_memory_allocated : 12.45 GiB
max_memory_reserved  : 13.89 GiB
Diff (estimate - reserved): +1.32 GiB
```

## Parameters Reference

### Model & Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | required | HuggingFace model ID or local path |
| `--dtype` | str | bf16 | Precision: bf16, fp16, fp32, fp8, int8, nf4 |
| `--seq-len` | int | 2048 | Maximum sequence length |
| `--per-device-batch` | int | 1 | Micro-batch size per GPU |
| `--grad-accum` | int | 1 | Gradient accumulation steps |

### Parallelism

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dp` | int | 1 | Data parallel replicas |
| `--tp` | int | 1 | Tensor parallel degree |
| `--pp` | int | 1 | Pipeline parallel stages |
| `--zero` | int | 0 | DeepSpeed ZeRO stage (0/1/2/3) |

### Optimizations

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--flashattn` | bool | false | Enable FlashAttention |
| `--grad-checkpoint` | bool | false | Enable gradient checkpointing |
| `--fused-loss` | bool | false | Enable fused loss (Liger) |
| `--liger` | bool | false | Enable Liger kernels (flag only) |

### LoRA / QLoRA

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--lora` | bool | false | Enable LoRA |
| `--qlora` | bool | false | Enable QLoRA (4-bit base weights) |
| `--lora-rank` | int | 8 | LoRA rank |
| `--lora-alpha` | int | 16 | LoRA alpha |
| `--lora-target-modules` | str[] | defaults | Target modules (space-separated) |

### Advanced Tuning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--grad-ckpt-reduction` | float | 0.35 | Activation retention ratio |
| `--attn-act-factor-no-fa` | float | 4.0 | Attention factor without FA |
| `--attn-act-factor-fa` | float | 0.8 | Attention factor with FA |
| `--mlp-act-factor` | float | 6.0 | MLP activation factor |
| `--fragmentation-gib` | float | 5.0 | Fragmentation reserve (GiB) |
| `--misc-overhead-gib` | float | 3.0 | Misc overhead (GiB) |

### Probe

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--probe` | bool | false | Run empirical probe |
| `--warmup-steps` | int | 3 | Warmup iterations |
| `--probe-steps` | int | 6 | Measurement iterations |
| `--estimate-only` | bool | false | Skip probe even if --probe is set |

## Memory Breakdown

The estimator models per-GPU VRAM as:

```
Total Memory = Base Weights + Trainable State + Activations + Logits + Overheads + Peak Buffer
```

### 1. Base Weights
- Full precision: `params × bytes(dtype)`
- QLoRA: `params × 0.56` (4-bit + scales/zeros overhead)
- Sharding: Divided by DP (ZeRO-3), TP, PP

### 2. Trainable State
- **Gradients**: `trainable_params × bytes(dtype)`
- **Optimizer** (AdamW 32-bit): `trainable_params × 8`
- **Master weights**: `trainable_params × 4` (optional)
- For LoRA: Only LoRA parameters are trainable
- Sharding: ZeRO stages, TP

### 3. Activations
- **Attention** (non-FA): `B × heads × S² × layers + B × S × layers × H × factor`
- **Attention** (FA): `B × S × layers × H × factor` (linear in S)
- **MLP**: `B × S × layers × H × mlp_factor`
- **Gradient checkpointing**: Multiply by `grad_ckpt_reduction` (0.2-0.5)
- Sharding: Divided by TP, PP

### 4. Logits
- `B × S × vocab × logits_factor`
- With fused loss: Multiply by 0.2 (5× reduction)
- Sharding: Divided by TP (vocab-parallel)

### 5. Overheads
- CUDA/cuBLAS: ~2.5 GiB
- DeepSpeed: 1-3 GiB (depends on ZeRO stage)
- Fragmentation: 2-10 GiB (user-tunable)
- Misc: 1-5 GiB

### 6. Peak Buffer
- ZeRO-3 all-gather: Temporary un-sharding during forward/backward
- Optimizer step: Temporary buffers
- Loss computation: Softmax intermediates

## Calibration Workflow

For production use, calibrate the estimator against empirical measurements:

### Step 1: Run Baseline Estimate

```bash
python scripts/estimate_memory_budget.py \
  --model YOUR_MODEL \
  --dtype bf16 \
  --seq-len 4096 \
  --per-device-batch 2 \
  --flashattn true \
  --grad-checkpoint true
```

### Step 2: Run Empirical Probe

```bash
python scripts/estimate_memory_budget.py \
  --model YOUR_MODEL \
  --dtype bf16 \
  --seq-len 4096 \
  --per-device-batch 2 \
  --flashattn true \
  --grad-checkpoint true \
  --probe true
```

### Step 3: Adjust Factors

If estimate < measured:
- Attention too low? Increase `--attn-act-factor-*`
- MLP too low? Increase `--mlp-act-factor`
- Logits peak? Disable `--fused-loss` or increase `--logits-factor`
- Fragmentation? Increase `--fragmentation-gib`

If estimate > measured:
- Lower the corresponding factors

### Step 4: Validate

Re-run probe with adjusted factors until estimate ≈ measured (within 10-15%).

## FAQ

### Q: Do I need a GPU to run estimates?

**A:** No! The estimator only needs CPU. It:
- Loads model configs (no weights)
- Uses meta tensors for param counting
- Performs mathematical calculations

GPU is only needed for the `--probe` option.

### Q: How accurate are the estimates?

**A:** Typical accuracy after calibration:
- ±10-15% for common configurations
- ±20-30% for unusual configs (very long sequences, exotic parallelism)

**Always keep peak ≤ 75-80% of physical HBM** for safety margin.

### Q: Can I use this for custom models?

**A:** Yes, if your model:
- Has a HuggingFace-compatible config
- Uses standard transformer architecture

For custom vision encoders or exotic architectures, you may need to adjust factors.

### Q: What about CPU offloading?

**A:** The `--cpu-offload-optimizer` flag is tracked but doesn't affect estimates yet. For offload scenarios:
- Use conservative estimates
- Run empirical probes to measure actual memory
- Offload performance varies widely by hardware

### Q: How do I convert training configs?

**HuggingFace Trainer:**
```python
per_device_train_batch_size → --per-device-batch
gradient_accumulation_steps → --grad-accum
bf16=True → --dtype bf16
gradient_checkpointing=True → --grad-checkpoint true
```

**DeepSpeed:**
```yaml
zero_optimization.stage: 3 → --zero 3
train_batch_size → dp × per_device_batch × grad_accum
```

**LLaMA-Factory:**
```yaml
flash_attn: fa2 → --flashattn true
enable_liger_kernel: true → --liger true --fused-loss true
quantization_bit: 4 → --qlora true
```

## Known Limitations

1. **Vision encoders**: Very rough approximation (assumes ViT-style)
2. **Custom architectures**: May need manual tuning
3. **Framework overhead**: Varies by versions
4. **Communication buffers**: Not explicitly modeled (included in overheads)
5. **Attention score precision**: Assumes FP32 for non-FA, actual behavior may vary

## Contributing

Found an issue or want to improve the estimator?
- Report bugs via GitHub issues
- Submit PRs with improvements
- Share your calibration results

## License

MIT License

---

**For web UI, see:** [Gradio app](../../app.py)
**For documentation, see:** [Main docs](../../docs/)
