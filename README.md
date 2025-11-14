# 🧮 LLM/VLM Memory Estimator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Estimate per-GPU VRAM requirements for LLM/VLM fine-tuning with support for:
- ✅ **LoRA** and **QLoRA** (4-bit quantization)
- ✅ **FlashAttention** and **Gradient Checkpointing**
- ✅ **DeepSpeed ZeRO** (stages 0/1/2/3)
- ✅ **Multi-GPU parallelism** (DP/TP/PP)
- ✅ **Fused kernels** (Liger loss)
- ✅ **Gradio Web UI** for interactive estimation
- ✅ **Command-line tool** with empirical probe

---

## 🚀 Quick Start

### Option 1: Web UI (Gradio)

```bash
# Install
pip install -e .

# Run Gradio app
python app.py

# Or with gradio extras
pip install -e ".[gradio]"
gradio app.py
```

Visit `http://localhost:7860` to use the interactive UI.

### Option 2: Command Line

```bash
# Install
pip install -e .

# Run estimation
python -m estimate_memory_budget \
  --model meta-llama/Llama-2-7b-hf \
  --dtype bf16 --seq-len 4096 \
  --per-device-batch 2 --grad-accum 128 \
  --zero 2 --dp 8 \
  --flashattn true --grad-checkpoint true

# Or use the CLI script directly
python cli.py --model meta-llama/Llama-2-7b-hf --dtype bf16 --seq-len 4096
```

### Option 3: Python API

```python
from cli import MemoryEstimator, EstimatorInputs

cfg = EstimatorInputs(
    model="meta-llama/Llama-2-7b-hf",
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

---

## 📦 Installation

### From Source (Development)

```bash
git clone https://github.com/YOUR_USERNAME/llm-memory-estimator.git
cd llm-memory-estimator
pip install -e .
```

### With Gradio UI

```bash
pip install -e ".[gradio]"
```

### Requirements

**For Estimation Only** (No GPU needed):
```
transformers >= 4.36.0
accelerate >= 0.25.0
torch >= 2.1.0  # CPU version is sufficient
pandas >= 2.0.0
```

**For Empirical Probe** (GPU required):
```bash
# Install CUDA-enabled torch separately
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**For Web UI**:
```
gradio >= 4.40.0
```

---

## 💡 Features

### 🎛️ Supported Configurations

| Feature | Support | Notes |
|---------|---------|-------|
| **Models** | Any HF model | LLaMA, Mistral, Qwen, GPT, T5, etc. |
| **Precision** | fp32, bf16, fp16, fp8, int8, nf4 | QLoRA uses nf4 (4-bit) |
| **LoRA** | ✅ | Parameter-efficient fine-tuning |
| **QLoRA** | ✅ | 4-bit base + LoRA adapters |
| **FlashAttention** | ✅ | Reduces attention O(N²) → O(N) |
| **Grad Checkpoint** | ✅ | Trade compute for memory |
| **Fused Loss** | ✅ | Liger kernel (~5x logits reduction) |
| **ZeRO** | 0/1/2/3 | DeepSpeed optimizer sharding |
| **Data Parallel** | ✅ | Multi-GPU data replication |
| **Tensor Parallel** | ✅ | Model layer sharding |
| **Pipeline Parallel** | ✅ | Model stage distribution |

### 📊 What It Estimates

The tool provides comprehensive memory breakdown:

1. **Base Weights** - Model parameters (sharded by TP/PP, ZeRO-3)
2. **Trainable State** - Gradients + optimizer + master weights
3. **Activations** - Attention + MLP intermediate tensors
4. **Logits** - Output vocabulary projections
5. **Overheads** - CUDA context, DeepSpeed, fragmentation
6. **Peak Buffers** - ZeRO-3 all-gather, optimizer step temporaries

### 🎯 Use Cases

**Research & Planning:**
- "Can I fine-tune Llama-2 70B on 8x A100 80GB?"
- "How many GPUs needed for Mistral-7B with QLoRA?"
- "Will gradient checkpointing let me fit larger batches?"

**Production Optimization:**
- Compare LoRA vs QLoRA vs full fine-tuning costs
- Find optimal batch size for given hardware
- Calibrate estimates against empirical measurements

**Education:**
- Understand memory breakdown for different techniques
- Learn how ZeRO, TP, PP affect memory distribution
- Visualize impact of FlashAttention, gradient checkpointing

---

## 📖 Documentation

- **[CLI Reference](./CLI_README.md)** - Command-line usage and all parameters
- **[Playbook](./docs/llm_vlm_memory_estimation_verification_playbook_markdown.md)** - Complete technical guide
- **[Web UI Guide](#gradio-web-ui)** - Gradio app usage

---

## 🌐 Gradio Web UI

The package includes a full-featured Gradio web interface:

```bash
python app.py
```

**Features:**
- ✅ 5 organized tabs (Basic, Parallelism, Optimizations, LoRA, Advanced)
- ✅ 20+ configurable parameters
- ✅ Real-time memory calculation
- ✅ Breakdown table and GPU recommendations
- ✅ 4 preset example configurations
- ✅ Mobile-responsive design

**Deploy to HuggingFace Spaces:**

```bash
# The app.py is ready for HF Spaces deployment
# Just push the entire folder to a HF Space repository
```

See [HuggingFace Spaces Guide](https://huggingface.co/docs/hub/spaces) for deployment instructions.

---

## 🔬 Examples

### Example 1: Llama-2 7B Full Fine-Tuning

```bash
python cli.py \
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

**Result:** ~14 GiB/GPU → Fits on A100 40GB

### Example 2: Mistral 7B with QLoRA

```bash
python cli.py \
  --model mistralai/Mistral-7B-v0.1 \
  --dtype bf16 \
  --qlora true \
  --lora-rank 8 \
  --seq-len 4096 \
  --per-device-batch 1 \
  --grad-accum 64 \
  --zero 3 \
  --dp 4 \
  --tp 2 \
  --flashattn true \
  --grad-checkpoint true
```

**Result:** ~8 GiB/GPU → Fits on RTX 4090

### Example 3: Llama-2 70B with LoRA

```bash
python cli.py \
  --model meta-llama/Llama-2-70b-hf \
  --dtype bf16 \
  --lora true \
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

**Result:** ~18 GiB/GPU → Fits on A100 40GB

### Example 4: Empirical Validation (GPU Required)

```bash
python cli.py \
  --model meta-llama/Llama-2-7b-hf \
  --dtype bf16 \
  --seq-len 4096 \
  --per-device-batch 2 \
  --flashattn true \
  --grad-checkpoint true \
  --probe true \
  --warmup-steps 3 \
  --probe-steps 6
```

**Output:**
```
=== Memory Estimate (per GPU) ===
Total (peak)        :  14.32 GiB

--- Empirical Probe ---
max_memory_allocated : 13.45 GiB
max_memory_reserved  : 15.21 GiB
Diff (estimate - reserved): -0.89 GiB
```

---

## 🎓 How It Works

### Memory Model

The estimator breaks down per-GPU VRAM into:

```
Total = Weights + Trainable_State + Activations + Logits + Overheads + Peak_Buffer
```

**Weights:**
- Full precision: `params × bytes(dtype)`
- QLoRA: `params × 0.56` (4-bit + scales/zeros)
- Sharded by: TP, PP, ZeRO-3 (for trainable)

**Trainable State:**
- Gradients: `trainable_params × bytes(dtype)`
- Optimizer (AdamW): `trainable_params × 8` (2 states in fp32)
- Master weights: `trainable_params × 4` (optional fp32 copy)
- For LoRA: Only adapter params are trainable
- Sharded by: ZeRO stages, TP

**Activations:**
- Attention (non-FA): `B × heads × S² × layers` + linear term
- Attention (FA): `B × S × layers × H × factor` (linear in S)
- MLP: `B × S × layers × H × mlp_factor`
- Gradient checkpointing: Multiply by reduction factor (0.2-0.5)
- Sharded by: TP, PP

**Logits:**
- `B × S × vocab × logits_factor`
- With fused loss: 5× reduction
- Sharded by: TP (vocab-parallel)

### Calibration

For production accuracy, calibrate against empirical measurements:

1. Run estimate
2. Run `--probe true` on actual GPU
3. Adjust tuning factors to match
4. Use calibrated factors for future estimates

See [Playbook](./docs/) for detailed calibration guide.

---

## ⚙️ Configuration

### Environment Variables

```bash
# Force CPU-only (disable CUDA)
export CUDA_VISIBLE_DEVICES=""

# Use specific GPU for probe
export CUDA_VISIBLE_DEVICES=0
```

### Config Files

The package supports modular configuration through its component architecture:

```
estimate_memory_budget/
├── components/      # Model components (attention, MLP, etc.)
├── config/          # Configuration dataclasses
├── core/            # Core estimation logic
├── formulas/        # Memory calculation formulas
├── optimizations/   # Optimization-specific adjustments
├── parallelism/     # DP/TP/PP/ZeRO logic
├── probe/           # Empirical measurement
└── utils/           # Helper utilities
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/llm-memory-estimator.git
cd llm-memory-estimator
pip install -e ".[dev]"

# Run tests (if available)
pytest

# Format code
black .
```

---

## 📝 Notes

### Accuracy

- Estimates are **approximate** (±10-15% after calibration)
- Actual memory depends on kernels, framework versions, runtime optimizations
- **Rule of thumb:** Keep peak ≤ 75-80% of physical HBM

### Limitations

- Vision encoders: Rough approximation (ViT-style assumed)
- Custom architectures: May need manual tuning
- Communication buffers: Included in overheads (not explicitly modeled)
- Framework overhead: Varies by version

### Requirements

- **Estimation:** CPU-only, no GPU needed
- **Probe:** Requires CUDA GPU
- **Models:** HuggingFace-compatible configs

---

## 📄 License

MIT License - See [LICENSE](./LICENSE) for details

---

## 🙏 Acknowledgments

Built for the ML community. Based on memory modeling research from transformer training at scale.

**Inspired by:**
- DeepSpeed memory optimization techniques
- HuggingFace Transformers architecture
- Community feedback on GPU memory estimation

---

## 🔗 Links

- **Documentation:** [Full Playbook](./docs/)
- **CLI Guide:** [CLI README](./CLI_README.md)
- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/llm-memory-estimator/issues)
- **Discussions:** [GitHub Discussions](https://github.com/YOUR_USERNAME/llm-memory-estimator/discussions)

---

**Made with ❤️ for efficient LLM training**
