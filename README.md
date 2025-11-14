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
pip install -e ".[gradio]"
python app.py
# Visit http://localhost:7860
```

### Command Line

```bash
pip install -e .

# Basic estimation
python -m llm_memory_estimator \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bf16 \
  --seq-len 4096 \
  --per-device-batch 2 \
  --grad-accum 128 \
  --zero 2 \
  --dp 8 \
  --flashattn true \
  --grad-checkpoint true
```

### Python API

```python
from llm_memory_estimator.cli import MemoryEstimator, EstimatorInputs

cfg = EstimatorInputs(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="bf16",
    seq_len=4096,
    per_device_batch=2,
    grad_accum=128,
    dp=8,
    zero=2,
    flashattn=True,
    grad_checkpoint=True
)

estimator = MemoryEstimator(cfg)
result = estimator.estimate()
print(f"Peak memory: {result.peak_total_gib:.2f} GiB per GPU")
```

## Examples

### Full Fine-Tuning (7B Model)

```bash
python -m llm_memory_estimator \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bf16 \
  --seq-len 4096 \
  --per-device-batch 2 \
  --zero 2 \
  --dp 8 \
  --flashattn true \
  --grad-checkpoint true
```

**Result:** ~14 GiB/GPU (fits on A100 40GB)

### QLoRA (7B Model)

```bash
python -m llm_memory_estimator \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bf16 \
  --qlora true \
  --lora-rank 8 \
  --seq-len 4096 \
  --per-device-batch 1 \
  --zero 3 \
  --dp 4 \
  --tp 2 \
  --flashattn true \
  --grad-checkpoint true
```

**Result:** ~8 GiB/GPU (fits on RTX 4090)

### LoRA (72B Model)

```bash
python -m llm_memory_estimator \
  --model Qwen/Qwen2.5-72B-Instruct \
  --dtype bf16 \
  --lora true \
  --lora-rank 64 \
  --seq-len 4096 \
  --zero 3 \
  --dp 32 \
  --tp 4 \
  --flashattn true \
  --grad-checkpoint true
```

**Result:** ~18 GiB/GPU (fits on A100 40GB)

### Vision-Language Model

```bash
python -m llm_memory_estimator \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bf16 \
  --seq-len 4096 \
  --per-device-batch 1 \
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
python -m llm_memory_estimator \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bf16 \
  --seq-len 4096 \
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
| `--seq-len` | Sequence length | 4096 |
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

For complete parameter list: `python -m llm_memory_estimator --help`

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

```bash
git clone https://github.com/MiaoDX/llm-memory-estimator.git
cd llm-memory-estimator
pip install -e .

# With Gradio UI
pip install -e ".[gradio]"
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! This tool uses mathematical models calibrated against empirical measurements. Estimates typically accurate within ±10-15%.
