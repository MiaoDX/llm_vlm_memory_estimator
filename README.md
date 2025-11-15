# LLM/VLM Memory Estimator

Estimate GPU memory requirements for fine-tuning Large Language Models and Vision-Language Models.

## Features

- ✅ LoRA / QLoRA (4-bit quantization)
- ✅ FlashAttention & Gradient Checkpointing
- ✅ DeepSpeed ZeRO (0/1/2/3)
- ✅ Multi-GPU (DP/TP/PP)
- ✅ Web UI & CLI
- ✅ Supports Qwen, Llama, Mistral, etc.

## Quick Start

### Web UI

```bash
# Using uv (recommended)
uv sync
uv run python -m llm_memory_estimator.app
# Or: uv run python llm_memory_estimator/app.py
# Visit http://localhost:7860

# Using pip
pip install -e .
python -m llm_memory_estimator.app
```

### Command Line

```bash
# Using uv (recommended)
uv sync
uv run llm-memory-estimator --help

# Using pip
pip install -e .

# Basic estimation (with uv)
uv run llm-memory-estimator \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bf16 \
  --seq-len 2048 \
  --per-device-batch 2 \
  --grad-accum 128 \
  --zero 2 \
  --dp 8 \
  --flashattn true \
  --grad-checkpoint true
```

## Examples

### Full Fine-Tuning (7B Model)

```bash
uv run llm-memory-estimator \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bf16 \
  --seq-len 2048 \
  --per-device-batch 2 \
  --grad-accum 128 \
  --zero 2 \
  --dp 8 \
  --flashattn true \
  --grad-checkpoint true
```

**Result:** ~14 GiB/GPU (estimated, may vary based on actual runtime behavior)

### QLoRA (7B Model)

```bash
uv run llm-memory-estimator \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bf16 \
  --qlora true \
  --lora-rank 8 \
  --seq-len 2048 \
  --per-device-batch 1 \
  --grad-accum 64 \
  --zero 3 \
  --dp 4 \
  --tp 2 \
  --flashattn true \
  --grad-checkpoint true
```

**Result:** ~8 GiB/GPU (estimated, may vary based on actual runtime behavior)

### LoRA (72B Model)

```bash
uv run llm-memory-estimator \
  --model Qwen/Qwen2.5-72B-Instruct \
  --dtype bf16 \
  --lora true \
  --lora-rank 64 \
  --seq-len 2048 \
  --per-device-batch 1 \
  --grad-accum 512 \
  --zero 3 \
  --dp 32 \
  --tp 4 \
  --flashattn true \
  --grad-checkpoint true
```

**Result:** ~18 GiB/GPU (estimated, may vary based on actual runtime behavior)

### Vision-Language Model

```bash
uv run llm-memory-estimator \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bf16 \
  --seq-len 2048 \
  --per-device-batch 1 \
  --grad-accum 64 \
  --zero 2 \
  --dp 4 \
  --flashattn true \
  --grad-checkpoint true \
  --use-vision true \
  --vision-hidden 1024 \
  --vision-layers 24 \
  --vision-image-size 448 \
  --vision-patch 14
```

### Empirical Probe (GPU Required)

```bash
uv run llm-memory-estimator \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bf16 \
  --seq-len 2048 \
  --flashattn true \
  --grad-checkpoint true \
  --probe true \
  --warmup-steps 3 \
  --probe-steps 6
```

Compares estimated vs actual GPU memory usage.

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | HuggingFace model ID | Required |
| `--dtype` | Precision (bf16/fp16/fp32/int8/nf4) | bf16 |
| `--seq-len` | Sequence length | 2048 |
| `--per-device-batch` | Batch size per GPU | 1 |
| `--grad-accum` | Gradient accumulation steps | 1 |
| `--dp` | Data parallelism degree | 1 |
| `--tp` | Tensor parallelism degree | 1 |
| `--pp` | Pipeline parallelism degree | 1 |
| `--zero` | DeepSpeed ZeRO stage (0/1/2/3) | 0 |
| `--flashattn` | Enable FlashAttention | false |
| `--grad-checkpoint` | Enable gradient checkpointing | false |
| `--lora` | Enable LoRA | false |
| `--qlora` | Enable QLoRA (4-bit) | false |
| `--lora-rank` | LoRA rank | 8 |
| `--fused-loss` | Enable fused loss (Liger) | false |
| `--probe` | Run empirical GPU probe | false |
| `--use-vision` | Enable vision encoder | false |

For complete parameter list: `uv run llm-memory-estimator --help`

## How It Works

The estimator calculates per-GPU memory as:

```
Total = Weights + Gradients + Optimizer States + Activations + Logits + Overheads
```

**Key components:**
- **Weights:** Model parameters (sharded by TP/PP/ZeRO-3)
- **Trainable State:** Gradients + optimizer states (sharded by ZeRO-1/2/3)
- **Activations:** Attention + MLP intermediate tensors (reduced by gradient checkpointing & FlashAttention)
- **Logits:** Output layer activations (reduced by fused loss)
- **Overheads:** CUDA context, DeepSpeed, fragmentation

## Requirements

**Estimation** (no GPU needed):
- Python 3.8+
- transformers >= 4.36.0
- accelerate >= 0.25.0
- torch >= 2.1.0 (CPU version OK)

**Empirical Probe** (GPU required):
- CUDA-enabled PyTorch

## Installation

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

**Note:** This project uses **CPU-only PyTorch** by default (~650 MB vs 1.7 GB CUDA version, 1.xGB vs 7.x GB for full virtual env), which is sufficient for memory estimation. The empirical probe feature (GPU required) is not yet implemented.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/MiaoDX/llm-memory-estimator.git
cd llm-memory-estimator
uv sync

# Run the CLI
uv run llm-memory-estimator --help

# Run the web UI
uv run python -m llm_memory_estimator.app
```

### Using pip

```bash
git clone https://github.com/MiaoDX/llm-memory-estimator.git
cd llm-memory-estimator
pip install -e .
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! This tool uses mathematical models calibrated against empirical measurements. Estimates typically accurate within ±10-15%.
