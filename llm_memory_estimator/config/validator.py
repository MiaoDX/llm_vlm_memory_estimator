"""
Configuration validation.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import EstimatorConfig

from ..formulas.constants import DTYPE_BYTES


def validate_config(config: 'EstimatorConfig') -> None:
    """
    Validate estimator configuration.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Model
    if not config.model or not isinstance(config.model, str):
        raise ValueError("model must be a non-empty string")

    # Training shape
    if config.seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {config.seq_len}")
    if config.per_device_batch <= 0:
        raise ValueError(f"per_device_batch must be positive, got {config.per_device_batch}")
    if config.grad_accum < 1:
        raise ValueError(f"grad_accum must be >= 1, got {config.grad_accum}")

    # Dtype
    if config.dtype.lower() not in DTYPE_BYTES:
        valid_dtypes = ", ".join(DTYPE_BYTES.keys())
        raise ValueError(
            f"Invalid dtype '{config.dtype}'. Must be one of: {valid_dtypes}"
        )

    # Parallelism
    if config.dp < 1:
        raise ValueError(f"dp must be >= 1, got {config.dp}")
    if config.tp < 1:
        raise ValueError(f"tp must be >= 1, got {config.tp}")
    if config.pp < 1:
        raise ValueError(f"pp must be >= 1, got {config.pp}")
    if config.zero_stage not in {0, 1, 2, 3}:
        raise ValueError(f"zero_stage must be 0, 1, 2, or 3, got {config.zero_stage}")

    # LoRA validation
    if config.lora is not None:
        lora_rank = config.lora.get("rank", 0)
        if lora_rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {lora_rank}")
        lora_alpha = config.lora.get("alpha", 0)
        if lora_alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {lora_alpha}")

    # Vision validation
    if config.vision is not None and config.vision.get("enabled", False):
        image_size = config.vision.get("image_size", 0)
        patch_size = config.vision.get("patch_size", 0)
        if image_size <= 0:
            raise ValueError(f"vision.image_size must be positive, got {image_size}")
        if patch_size <= 0:
            raise ValueError(f"vision.patch_size must be positive, got {patch_size}")
        if image_size % patch_size != 0:
            raise ValueError(
                f"vision.image_size ({image_size}) must be divisible by "
                f"patch_size ({patch_size})"
            )

    # Overheads (should be non-negative)
    if config.cuda_libs_gib < 0:
        raise ValueError(f"cuda_libs_gib must be >= 0, got {config.cuda_libs_gib}")
    if config.misc_overhead_gib < 0:
        raise ValueError(f"misc_overhead_gib must be >= 0, got {config.misc_overhead_gib}")
    if config.fragmentation_gib < 0:
        raise ValueError(f"fragmentation_gib must be >= 0, got {config.fragmentation_gib}")

    # Tunable factors (should be positive)
    if config.attn_act_factor_no_fa <= 0:
        raise ValueError(f"attn_act_factor_no_fa must be positive, got {config.attn_act_factor_no_fa}")
    if config.attn_act_factor_fa <= 0:
        raise ValueError(f"attn_act_factor_fa must be positive, got {config.attn_act_factor_fa}")
    if config.mlp_act_factor <= 0:
        raise ValueError(f"mlp_act_factor must be positive, got {config.mlp_act_factor}")
    if config.logits_factor < 0:
        raise ValueError(f"logits_factor must be >= 0, got {config.logits_factor}")
    if not (0 < config.grad_ckpt_reduction <= 1):
        raise ValueError(
            f"grad_ckpt_reduction must be in (0, 1], got {config.grad_ckpt_reduction}"
        )

    # Probe settings
    if config.probe_steps < 1:
        raise ValueError(f"probe_steps must be >= 1, got {config.probe_steps}")
    if config.probe_warmup < 0:
        raise ValueError(f"probe_warmup must be >= 0, got {config.probe_warmup}")
