# LLM/VLM Memory Estimation & Verification Playbook

> One-stop Markdown you can iterate on. Includes rationale, formulas, flags mapping, safety guidance, and a reference to the **estimator + probe script** for convenience.

---

## Goals
- **Estimate per-GPU VRAM** (steady & peak) for LLM/VLM fine-tuning from configs.
- Account for common knobs: **dtype**, **LoRA/QLoRA**, **FlashAttention**, **Liger/fused loss**, **grad checkpointing**, **KV cache policy**, **optimizer type/precision**, **DeepSpeed ZeRO 0/1/2/3**, **DP/TP/PP**.
- **Validate empirically** with a synthetic forward/backward probe that logs peak allocator stats.
- Provide **conservative margins** and a **calibration loop** to reduce “estimate 100 GB, reality 200 GB” surprises.

---

## Quick Start

### 1) Estimate only (no GPU run)
```bash
python scripts/estimate_memory_budget.py \
  --model meta-llama/Llama-2-7b-hf \
  --dtype bf16 --seq-len 4096 \
  --per-device-batch 2 --grad-accum 128 \
  --zero 2 --dp 8 --tp 1 --pp 1 \
  --flashattn true --liger true --fused-loss true --grad-checkpoint true
```

### 2) Estimate w/ LoRA
```bash
python scripts/estimate_memory_budget.py \
  --model mistralai/Mistral-7B-v0.1 \
  --lora true --lora-rank 8 --lora-target-modules q_proj k_proj v_proj o_proj \
  --dtype bf16 --seq-len 4096 --per-device-batch 1 --grad-accum 128 \
  --flashattn true --fused-loss true
```

### 3) Empirical probe (allocates VRAM!)
```bash
python scripts/estimate_memory_budget.py \
  --model mistralai/Mistral-7B-v0.1 \
  --dtype bf16 --seq-len 4096 --per-device-batch 1 --grad-accum 64 \
  --probe true --warmup-steps 3 --probe-steps 6
```

---

## Safety & Headroom
- Treat the estimator as **conservative** but not infallible.
- Keep **peak ≤ 75–80%** of physical HBM (e.g., H200 141 GB → target ≤ ~113 GB per GPU).
- Add **fragmentation** (+2–6 GB) and **CUDA/cuBLAS** (+~2–3 GB) cushions.
- Prefer **down-sweeping micro-batch** with gradient accumulation to fit.

**“Fake GPU memory?”**
- You cannot reliably emulate *more* HBM. CUDA Unified Memory can oversubscribe but is non-representative and very slow.
- You can emulate *less* memory: **MIG** partitions (hardware-level caps) or **PyTorch per-process memory fraction** (allocator-governed cap). Use this to enforce safety margins, not to pretend you own a bigger GPU.

---

## Estimation Model (Concepts)
We break down per-GPU VRAM into:

1) **Base weights**
   - Modeled as `params × bytes(dtype)`; if **QLoRA**/NF4, base weights are treated as ≈0.55–0.6 byte/param to account for scales/zeros overhead.
   - With **ZeRO-3** + **DP=N**, parameters are sharded across data-parallel ranks (effective **÷N** per rank) for full fine-tuning; for QLoRA we conservatively keep base weights replicated across DP ranks (no ZeRO sharding) but still shard them across TP ranks.
   - With **TP=T**, layer parameters are sharded across tensor-parallel ranks (effective **÷T** per rank).
   - With **PP=K**, we approximate per-rank residency by dividing total parameters by `K` (each pipeline stage holds roughly `1/K` of the layers/params).

2) **Trainable state**
   - **Full FT**: includes gradients, optimizer moments (AdamW ≈ `2 × trainable_params × 4 B`), and optional FP32 master weights.
   - **LoRA/PEFT**: applies the same formulas but only to **LoRA params** (base weights are treated as frozen/quantized); LoRA params are also divided by the pipeline parallel degree.
   - **ZeRO** scaling: stage-1 shards optimizer; stage-2 shards optimizer+grads; stage-3 shards optimizer+grads+params across DP.
   - **TP** shards trainable tensors across `T` when computing gradients, optimizer state, and (for full FT) master weights.

3) **Activations**
   - Modeled as scaling with **micro-batch × seq_len × hidden_size × layers**, split into **Attention** and **MLP** terms.
   - **Non-FlashAttention**: attention activations include an explicit `O(S²)` score/prob term (roughly `B × heads × S × S × layers_per_stage`) plus a linear-in-`S` term controlled by a tunable factor.
   - **FlashAttention**: uses a separate attention factor and is modeled as roughly linear in `S`, reflecting that large score/prob matrices are not materialized.
   - **Gradient checkpointing** multiplies activation terms by a reduction factor (recompute instead of store).
   - **Liger/fused loss** reduces large transient tensors (e.g., logits buffer before loss).

4) **Logits / Loss buffers**
   - Modeled as `B × L × V × bytes(dtype)` scaled by a tunable logits factor.
   - With fused loss (e.g., Liger), a strong reduction factor is applied to approximate in-kernel loss computation.

5) **KV cache** (usually off in training; on in some RLHF/inference-like loops)
   - Modeled as `≈ 2 × B × L × layers_per_stage × H × bytes(dtype)` when enabled.

6) **Vision branch** (VLM)
   - Adds ViT/CNN feature map activations based on image size, patching, layers, and hidden size via a rough ViT-style term. We assume the vision encoder is replicated (not TP-sharded) in most VLMs; if your vision stack uses TP, adjust factors accordingly.

7) **Overheads**
   - CUDA/cuBLAS handles, allocator reserve, DeepSpeed/runtime overhead, and miscellaneous framework state modeled as fixed GiB cushions.
   - **Fragmentation** reserve to absorb allocator fragmentation and additional buffers.

**Totals**
- `steady = sum(categories)`
- `peak = steady + small_transient_overhead`

---

## Core Formulas (Simplified)
Let `P = total_params`, `H = hidden_size`, `L = num_layers`, `V = vocab_size`, `B = micro_batch`, `S = seq_len`, `N = DP`, `T = TP`.

**Weights per rank (steady):**
- If QLoRA: `W_bytes ≈ P × 0.56` (NF4 base weights with scales/zeros) else `P × bytes(dtype)`.
- With ZeRO-3 and full FT: `W_bytes /= N` to reflect parameter sharding across data-parallel ranks; for QLoRA we keep base weights replicated across DP ranks (no ZeRO sharding).
- With tensor parallel: `W_bytes /= T` to reflect per-shard residency.
- With pipeline parallel: we approximate per-rank parameters as `P / PP` (each stage holds a slice of layers).

**Trainable (full FT vs LoRA):**
- Full fine-tuning: `TrainParams = P` and we model gradients, optimizer states, and optional master weights for all parameters.
- LoRA/PEFT: `TrainParams ≈ 2 × r × H × modules × L` (heuristic; see code) or an override count; only these are treated as trainable.
- Gradients: `G_bytes = TrainParams × bytes(dtype)`.
- Optimizer (AdamW 32-bit): `O_bytes ≈ TrainParams × 8`.
- Master weights (optional): `M_bytes ≈ TrainParams × 4`.
- ZeRO scaling: Stage-1 → `O_bytes /= N`; Stage-2 → `(O_bytes + G_bytes) /= N`; Stage-3 → additionally shards trainable params across `N`.
- Tensor parallel: divides trainable tensors across `T` (gradients, optimizer state, and master weights are divided by `T` in the implementation).

**Activations (heuristic, tuned):**
- Attention (non-FA): `A_attn ≈ B × heads × S² × L_rank + B × S × L_rank × H × attn_factor_no_fa`, where `L_rank ≈ L / PP` and the factor term absorbs additional tensors.
- Attention (FA): `A_attn ≈ B × S × L_rank × H × attn_factor_fa`, modeled as roughly linear in `S`.
- MLP: `A_mlp ≈ B × S × L_rank × H × mlp_factor`.
- Gradient checkpointing: both attention and MLP terms are multiplied by `gc_reduction` when enabled.
- Tensor parallel: activations are approximated as sharded across `T` (both attention and MLP activations are divided by `T`).

**Logits / loss:**
- `A_logits ≈ (B × S × V / T) × logits_factor × (0.2 if fused_loss else 1.0)` (logits are divided by `T` to reflect vocab-parallel projections).

**KV cache:** `A_kv ≈ (2 × B × S × L_rank × H) / T` when KV cache is enabled.

**Vision activations (ViT-ish):** `A_vis ≈ B × image_patches × vision_layers × vision_hidden`, scaled by checkpoint reduction when applicable.

**Overheads:** `A_overhead = CUDA + misc + DeepSpeed + fragmentation` (GiB sums).

**Peak:** `peak ≈ steady + small_transient_overhead + zero3_allgather_overhead`, where:
- The transient term is a simple function of `B` (optimizer/loss buffers).
- The ZeRO-3 term approximates the incremental peak from materializing full parameters per rank during all-gather/reduce-scatter (scales with `P`, `dtype`, `N`, and `T`).

---

## Config Mapping Cheat Sheet

### Hugging Face / Trainer / TRL
- `per_device_train_batch_size` → `--per-device-batch`
- `gradient_accumulation_steps` → `--grad-accum`
- `bf16`/`fp16` → `--dtype`
- `gradient_checkpointing` → `--grad-checkpoint true`
- `optim` (`adamw_*`, `adafactor`) → `--optimizer`, `--optimizer-bits`
- LoRA (via PEFT): `lora_r`, `lora_alpha`, `target_modules` → `--lora true`, `--lora-rank`, `--lora-target-modules`

### LLaMA-Factory
- `flash_attn: fa2` → `--flashattn true`
- `enable_liger_kernel: true` → `--liger true` (and often `--fused-loss true`)
- `zero_stage: 0/1/2/3`, `deepspeed` YAML → `--zero`, `--dp` (and TP/PP if using)
- `quantization_bit: 4` + LoRA → `--qlora true` (weights in NF4)

### DeepSpeed
- `zero_optimization.stage` → `--zero`
- `zero_optimization.offload_optimizer.device: cpu` → `--dp-offload-optimizer true`
- `train_batch_size` = `dp × per_device_batch × grad_accum`
- TP/PP often configured via launcher — map to `--tp` / `--pp`; TP is used to shard parameters/activations/logits in the estimator, PP is used to approximate per-stage residency.

---

## Empirical Probe Protocol
1) **Match config flags** exactly (dtype, FA/Liger, LoRA, ckpt, sequence length, micro-batch).
2) Warm up a few steps, then measure **`torch.cuda.max_memory_allocated()`** and **`...reserved()`**.
3) Compare **Estimate (peak)** vs **Reserved**; adjust multipliers to close the gap (see *Calibration*).
4) Optional: **binary search** micro-batch (constant global batch via GA) to auto-find the largest safe size.

> The included script prints a category table + totals, then an optional probe section with measured peaks.

---

## Calibration Workflow
- Start with defaults (`--fragmentation-gib 5`, FA factors, logits factor, etc.).
- Run a probe on one representative config.
- If **estimate < measured**, increase the relevant terms:
  - Attention too low? Raise `--attn-act-factor-*`.
  - MLP too low? Raise `--mlp-act-factor`.
  - Loss peak? Increase `--logits-factor` or disable `--fused-loss` in estimate to be conservative.
  - Fragmentation spike? Increase `--fragmentation-gib`.
- Re-run until estimate ≥ measured by ~10–15%.

---

## Multi-GPU & Parallelism
- **DP + ZeRO**: applies the ZeRO stage rules to shard optimizer state, gradients, and (for ZeRO-3) parameters across data-parallel ranks; the estimator also adds an explicit ZeRO-3 all-gather peak term. Always check per-rank usage against headroom.
- **TP**: parameters and trainable state are sharded across `T`; activations and logits are also divided by `T` to approximate tensor/vocab parallelism, so TP runs are more accurately modeled (you can still tune activation/logits factors for your stack).
- **PP**: pipeline parallel degree is used to approximate how many layers/params live on a given stage (roughly `1/PP` of the model per rank), which reduces the previous over-estimation for PP>1; exact stage distribution can still shift peak locations, so use probes for final safety.
- Prefer **per-rank peaks** as the primary capacity constraint.

---

## Known Pitfalls
- FlashAttention/Liger availability depends on installed kernels; your **probe** may not reflect the “ideal” if they’re missing.
- 8-bit optimizers vary in state layout; the script uses **conservative** approximations by default.
- Activation checkpointing patterns differ across frameworks (per-block vs per-layer); adjust `--grad-ckpt-reduction` accordingly.
- Vision encoders vary widely; our VLM term is deliberately rough — extend with your specific backbone math for accuracy.

---

## Roadmap (Next Extensions)
- Precise modeling for **bitsandbytes** optim states & offload.
- **DeepSpeed** multi-GPU probe wrapper (stage-aware) for true distributed measurement.
- Detection hooks for **FA2/Liger** presence to auto-switch multipliers.
- Per-architecture activation shapes (Llama/Mistral/Qwen/GPT-NeoX) for tighter factors.
- VLM backbones (CLIP/ViT variants) with exact tensor shapes.

---

## Estimator Script
The estimator + probe implementation lives in `scripts/estimate_memory_budget.py`. It implements the concepts and formulas described above and exposes all of the tunable factors used in the examples. Some flags (e.g., `--liger`, `--dp-offload-optimizer`) are currently tracked for configuration mirroring and probe context but do not yet change the VRAM estimates directly.

Usage examples (from repo root):

```bash
python scripts/estimate_memory_budget.py --help

# Example: estimate only
python scripts/estimate_memory_budget.py \
  --model meta-llama/Llama-2-7b-hf \
  --dtype bf16 --seq-len 4096 \
  --per-device-batch 2 --grad-accum 128 \
  --zero 2 --dp 8 --tp 1 --pp 1 \
  --flashattn true --liger true --fused-loss true --grad-checkpoint true
```

---

## Repro Template (fill this for each run)
```
### Experiment ID
Model:
Commit/Env: CUDA=, PyTorch=, Transformers=, FlashAttn=, Liger=
GPU(s): Model=, Count=, HBM per GPU=
Parallelism: DP=, TP=, PP=, ZeRO=

### Config
Seq len=, micro-batch=, GA=, dtype=, optimizer= (bits=), ckpt=?, LoRA=?, QLoRA=?, FA=?, Liger=?, fused-loss=?

### Estimator Output
Base weights= GiB
Trainable state= GiB
Attention activ.= GiB
MLP activ.= GiB
Logits= GiB
KV cache= GiB
Vision activ.= GiB
Misc= GiB, CUDA= GiB, DS= GiB, Frag= GiB
Total (steady)= GiB
Total (peak)= GiB

### Probe Output
max_memory_allocated= GiB
max_memory_reserved= GiB
Diff (estimate - reserved)= GiB

### Notes
-
```
