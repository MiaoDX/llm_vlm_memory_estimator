# LLM/VLM Memory Estimation - Implementation Reference

> Technical reference for memory estimation formulas and calibration factors used in the implementation.

## Overview

Estimates per-GPU VRAM requirements for LLM/VLM fine-tuning accounting for:
- **Precision**: fp32, bf16, fp16, int8, int4/nf4 (QLoRA)
- **PEFT**: LoRA/QLoRA with configurable rank and target modules
- **Optimizations**: FlashAttention, gradient checkpointing, fused loss kernels
- **Parallelism**: Data Parallel (DP), Tensor Parallel (TP), Pipeline Parallel (PP)
- **DeepSpeed ZeRO**: Stages 0/1/2/3 with optimizer/gradient/parameter sharding
- **Vision**: VLM support with vision encoder memory estimation

## Safety Margins

**Recommended headroom:**
- Keep peak ≤ 75-80% of physical HBM
- Example: H200 141 GB → target ≤ ~113 GB per GPU

**Built-in cushions:**
- CUDA/cuBLAS context: ~2.5 GiB
- Miscellaneous framework overhead: ~3.0 GiB
- Allocator fragmentation: ~5.0 GiB (configurable)
- DeepSpeed overhead: 0-3 GiB (stage-dependent)

**Memory limitations:**
- Cannot reliably emulate *more* HBM (CUDA Unified Memory is slow and unrealistic)
- Can enforce *less* memory via MIG partitions or PyTorch memory fraction caps

## Memory Components

Per-GPU VRAM breakdown:

```
Total = Weights + Trainable_State + Activations + Logits + Overheads + Peak_Buffer
```

### 1. Base Weights

**Formula:** `W_bytes = P × bytes_per_param(dtype)`

**Precision bytes per parameter:**
- fp32/float32: 4.0
- bf16/bfloat16/fp16/float16: 2.0
- int8/fp8: 1.0
- int4: 0.5
- nf4 (QLoRA): 0.56 (includes scales/zeros overhead)

**Sharding:**
- ZeRO-3 (full fine-tune): `W_bytes /= DP` (sharded across data-parallel ranks)
- ZeRO-3 (QLoRA): Base weights replicated (no DP sharding), only LoRA params sharded
- Tensor Parallel: `W_bytes /= TP` (sharded across tensor-parallel ranks)
- Pipeline Parallel: `W_bytes ≈ W_bytes / PP` (approximate per-stage residency)

### 2. Trainable State

**Components:**
- Gradients: `trainable_params × bytes(dtype)`
- Optimizer (AdamW 32-bit): `trainable_params × 8` (2 states × 4 bytes)
- Master weights (optional FP32): `trainable_params × 4`

**Trainable parameter count:**
- Full fine-tuning: `trainable_params = total_params`
- LoRA: `trainable_params ≈ 2 × rank × hidden × target_modules × layers`
  - Heuristic estimation from model architecture
  - Override via `--lora-params-override` for exact count

**ZeRO sharding:**
- Stage 0: No sharding
- Stage 1: `optimizer_state /= DP`
- Stage 2: `(optimizer_state + gradients) /= DP`
- Stage 3: `(optimizer_state + gradients + trainable_params) /= DP`

**Tensor Parallel:** All trainable tensors divided by `TP`

### 3. Activations

**Attention (without FlashAttention):**
```
A_attn = B × heads × S² × (L/PP) + B × S × (L/PP) × H × attn_factor_no_fa
```
- Explicit O(S²) score/probability matrices
- Default `attn_factor_no_fa = 4.0`

**Attention (with FlashAttention):**
```
A_attn = B × S × (L/PP) × H × attn_factor_fa
```
- Linear in sequence length (no materialized scores)
- Default `attn_factor_fa = 0.8` (~80% reduction)

**MLP:**
```
A_mlp = B × S × (L/PP) × H × mlp_factor
```
- Default `mlp_factor = 6.0`

**Gradient Checkpointing:**
- Multiplies activation terms by `grad_ckpt_reduction`
- Default `grad_ckpt_reduction = 0.35` (retains 35%, recomputes 65%)

**Tensor Parallel:** Activations divided by `TP`

### 4. Logits

**Formula:**
```
A_logits = (B × S × V / TP) × logits_factor × reduction
```

**Reduction:**
- Standard: `reduction = 1.0`
- Fused loss (Liger): `reduction = 0.2` (80% reduction via in-kernel computation)

**Sharding:** Logits divided by `TP` (vocab-parallel)

### 5. KV Cache

**Formula (when enabled):**
```
A_kv = (2 × B × S × (L/PP) × H) / TP
```

**Note:** Typically disabled during training

### 6. Vision Encoder (VLM)

**Formula:**
```
A_vis = B × image_patches × vision_layers × vision_hidden
```

Where:
- `image_patches = (image_size / patch_size)²`
- Affected by gradient checkpointing when enabled
- Assumed replicated (not TP-sharded) in most VLMs

**Configuration:**
- `--vision-hidden`: Vision encoder hidden size (default: 1024)
- `--vision-layers`: Vision encoder layers (default: 24)
- `--vision-image-size`: Image size in pixels (default: 448)
- `--vision-patch`: ViT patch size (default: 14)

### 7. Overheads

**Fixed cushions (GiB):**
- CUDA/cuBLAS context: 2.5 (configurable via `--cuda-libs-gib`)
- Miscellaneous overhead: 3.0 (configurable via `--misc-overhead-gib`)
- Fragmentation: 5.0 (configurable via `--fragmentation-gib`)
- DeepSpeed: Stage-dependent (configurable via `--deepspeed-overhead-gib`):
  - Stage 0: 0.0 GiB
  - Stage 1: 1.5 GiB
  - Stage 2: 2.0 GiB
  - Stage 3: 3.0 GiB

### 8. Peak Memory

**Formula:**
```
peak = steady + transient_overhead + zero3_allgather_overhead
```

**Components:**
- Steady: Sum of all memory components
- Transient: Small buffers proportional to batch size
- ZeRO-3 all-gather: Temporary parameter materialization during forward/backward

## Calibration Factors

All tunable via CLI for empirical calibration:

| Factor | Flag | Default | Purpose |
|--------|------|---------|---------|
| Attention (no FA) | `--attn-act-factor-no-fa` | 4.0 | Attention activation scaling |
| Attention (with FA) | `--attn-act-factor-fa` | 0.8 | FlashAttention reduction |
| MLP | `--mlp-act-factor` | 6.0 | MLP activation scaling |
| Logits | `--logits-factor` | 1.0 | Output logits scaling |
| Grad checkpoint | `--grad-ckpt-reduction` | 0.35 | Activation retention fraction |
| Fragmentation | `--fragmentation-gib` | 5.0 | Allocator fragmentation cushion |

## Configuration Mapping

### HuggingFace Trainer → CLI Flags

| Trainer Config | CLI Flag | Notes |
|----------------|----------|-------|
| `per_device_train_batch_size` | `--per-device-batch` | Micro-batch size per GPU |
| `gradient_accumulation_steps` | `--grad-accum` | Gradient accumulation |
| `bf16=True` | `--dtype bf16` | BFloat16 precision |
| `fp16=True` | `--dtype fp16` | Float16 precision |
| `gradient_checkpointing=True` | `--grad-checkpoint true` | Activation checkpointing |
| `optim="adamw_*"` | `--optimizer adamw` | Optimizer type |
| PEFT `r` | `--lora-rank` | LoRA rank |
| PEFT `lora_alpha` | `--lora-alpha` | LoRA alpha |
| PEFT `target_modules` | `--lora-target-modules` | Target modules list |

### LLaMA-Factory → CLI Flags

| LLaMA-Factory | CLI Flag | Notes |
|---------------|----------|-------|
| `flash_attn: fa2` | `--flashattn true` | FlashAttention 2 |
| `enable_liger_kernel: true` | `--liger true --fused-loss true` | Liger kernels |
| `zero_stage: N` | `--zero N` | DeepSpeed ZeRO stage |
| `quantization_bit: 4` + LoRA | `--qlora true` | 4-bit QLoRA (NF4) |

### DeepSpeed Config → CLI Flags

| DeepSpeed | CLI Flag | Notes |
|-----------|----------|-------|
| `zero_optimization.stage` | `--zero` | ZeRO stage (0/1/2/3) |
| `train_micro_batch_size_per_gpu` | `--per-device-batch` | Micro-batch size |
| Data parallel size | `--dp` | Number of DP ranks |

## Implementation Notes

**Parameter Counting:**
- Uses HuggingFace `transformers` library with `accelerate.init_empty_weights()`
- Counts parameters on CPU without GPU/memory requirements
- Supports local paths and HuggingFace Hub model IDs

**LoRA Parameter Estimation:**
- Heuristic-based estimation from model architecture
- Counts: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Formula: `lora_params ≈ 2 × rank × hidden_dim × num_target_modules × num_layers`
- Use `--lora-params-override` if exact count is known

**Accuracy:**
- Estimates typically within ±10-15% after calibration
- Accuracy depends on framework versions and kernel implementations
- Use empirical validation for production workloads

**Limitations:**
- Vision encoder estimation is approximate (assumes ViT-style architecture)
- Custom architectures may require manual factor tuning
- Framework overhead varies by version and configuration
- Communication buffers approximated in fixed overheads

## CLI Reference

**Basic usage:**
```bash
python -m llm_memory_estimator \
  --model MODEL_NAME \
  --dtype bf16 \
  --seq-len 4096 \
  --per-device-batch 2 \
  --grad-accum 128 \
  --dp 8 \
  --zero 2 \
  --flashattn true \
  --grad-checkpoint true
```

**Full parameter list:** `python -m llm_memory_estimator --help`

## References

- DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- QLoRA: https://arxiv.org/abs/2305.14314
- LoRA: https://arxiv.org/abs/2106.09685
