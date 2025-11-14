"""
Memory estimation constants and tunable factors.

These constants are used throughout the estimation process.
Factors marked as "tunable" can be calibrated against empirical measurements.
"""

# Memory units
GiB = 1024 ** 3
MiB = 1024 ** 2

# Bytes per parameter for different datatypes
DTYPE_BYTES = {
    "fp32": 4.0,
    "float32": 4.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "int8": 1.0,
    "fp8": 1.0,      # FP8 formats are typically 1 byte per element
    "int4": 0.5,     # 4-bit quantization
    "nf4": 0.5,      # QLoRA NF4 quantized weights
}

# Default tunable factors for activation memory estimation
# These can be overridden in EstimatorConfig for calibration

class ActivationFactors:
    """
    Tunable factors for activation memory estimation.

    These approximate how many hidden_size elements per token per layer
    are saved in memory during training.
    """

    # Attention activations without FlashAttention
    # Includes Q/K/V projections, attention scores (O(S²)), and context
    ATTN_NO_FLASH_ATTENTION = 4.0

    # Attention activations with FlashAttention
    # Much lower because score/prob matrices aren't materialized
    ATTN_WITH_FLASH_ATTENTION = 0.8

    # MLP activations (intermediate states, gate projections for SwiGLU, etc.)
    MLP = 6.0

    # Gradient checkpointing reduction factor
    # When checkpointing is enabled, this fraction of activations is retained
    # (checkpointed layers are recomputed during backward, not stored)
    GRAD_CHECKPOINT_REDUCTION = 0.35

    # Output logits memory factor
    # Controls the scaling of B × S × V logits buffer
    LOGITS = 1.0

    # Logits reduction with fused loss (e.g., Liger kernel)
    # Fused cross-entropy computes loss in-kernel without materializing full logits
    FUSED_LOSS_REDUCTION = 0.2


class OverheadDefaults:
    """Default overhead values in GiB"""

    # CUDA context, cuBLAS handles, etc.
    CUDA_LIBS = 2.5

    # Miscellaneous framework overhead (Python objects, metadata, etc.)
    MISC = 3.0

    # Allocator fragmentation cushion
    FRAGMENTATION = 5.0

    # DeepSpeed runtime overhead (varies by ZeRO stage)
    DEEPSPEED_BASE = 1.0
    DEEPSPEED_STAGE_1 = 1.5
    DEEPSPEED_STAGE_2 = 2.0
    DEEPSPEED_STAGE_3 = 3.0


def to_gib(x_bytes: float) -> float:
    """Convert bytes to GiB"""
    return float(x_bytes) / GiB


def to_mib(x_bytes: float) -> float:
    """Convert bytes to MiB"""
    return float(x_bytes) / MiB


def clamp_nonneg(x: float) -> float:
    """Clamp value to non-negative"""
    return max(0.0, float(x))
