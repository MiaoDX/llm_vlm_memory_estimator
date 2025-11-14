"""
Configuration dataclass for memory estimation.

EstimatorConfig is designed to be produced by both:
1. EstimatorConfigBuilder (programmatic API)
2. load_yaml_config() / load_json_config() (file-based)

They must produce identical structures to avoid divergence.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class EstimatorConfig:
    """
    Configuration for memory estimation.

    This dataclass captures all parameters needed for memory estimation.
    It's designed to be easily serializable to/from YAML/JSON and
    constructable via the fluent builder API.

    Attributes:
        model: HuggingFace model ID or local path
        is_seq2seq: Whether model is seq2seq (vs. causal LM)

        # Training shape
        seq_len: Sequence length
        per_device_batch: Micro-batch size per GPU
        grad_accum: Gradient accumulation steps

        # Precision
        dtype: Weight/activation dtype (bf16, fp16, fp32, int8, int4, nf4)

        # Parallelism
        dp: Data parallel degree
        tp: Tensor parallel degree
        pp: Pipeline parallel stages
        zero_stage: DeepSpeed ZeRO stage (0/1/2/3)

        # LoRA configuration (structured)
        lora: Optional dict with keys: rank, alpha, quantized, target_modules, etc.

        # Optimizations - structured with parameters!
        # Format: {"optimization_name": {"param1": value1, ...}}
        # Example: {"flashattention": {}, "gradient_checkpointing": {"reduction": 0.3}}
        optimizations: Dict[str, Dict[str, Any]]

        # Overheads (GiB)
        cuda_libs_gib: CUDA context, cuBLAS handles, etc.
        misc_overhead_gib: Miscellaneous framework overhead
        fragmentation_gib: Allocator fragmentation cushion
        deepspeed_overhead_gib: DeepSpeed runtime overhead (overridden based on ZeRO stage)

        # Tunable factors for calibration
        attn_act_factor_no_fa: Attention activation factor without FlashAttention
        attn_act_factor_fa: Attention activation factor with FlashAttention
        mlp_act_factor: MLP activation factor
        logits_factor: Output logits memory factor
        grad_ckpt_reduction: Activation retention with gradient checkpointing

        # Probe settings
        probe_enabled: Whether to run empirical probe
        probe_steps: Number of probe steps
        probe_warmup: Number of warmup steps before measuring

        # Vision (VLM) configuration
        vision: Optional dict with keys: enabled, image_size, patch_size, layers, hidden, etc.

        # Advanced
        vocab_size_override: Override vocab size from model config
        use_kv_cache: Enable KV cache (typically False for training)
    """

    # Model
    model: str
    is_seq2seq: bool = False
    vocab_size_override: Optional[int] = None

    # Training shape
    seq_len: int = 2048
    per_device_batch: int = 1
    grad_accum: int = 1

    # Precision
    dtype: str = "bf16"

    # Parallelism
    dp: int = 1
    tp: int = 1
    pp: int = 1
    zero_stage: int = 0

    # LoRA configuration (structured)
    # Example: {"rank": 8, "alpha": 16, "quantized": False, "target_modules": ["q_proj", "v_proj"]}
    lora: Optional[Dict[str, Any]] = None

    # Optimizations - structured with parameters!
    # Format: {"flashattention": {}, "liger": {}, "gradient_checkpointing": {"reduction": 0.35}}
    optimizations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Overheads (GiB)
    cuda_libs_gib: float = 2.5
    misc_overhead_gib: float = 3.0
    fragmentation_gib: float = 5.0
    deepspeed_overhead_gib: float = 1.0  # Adjusted based on zero_stage

    # Tunable factors (calibration knobs)
    attn_act_factor_no_fa: float = 4.0
    attn_act_factor_fa: float = 0.8
    mlp_act_factor: float = 6.0
    logits_factor: float = 1.0
    grad_ckpt_reduction: float = 0.35

    # Probe settings
    probe_enabled: bool = False
    probe_steps: int = 6
    probe_warmup: int = 3

    # Vision (VLM) configuration
    # Example: {"enabled": True, "image_size": 448, "patch_size": 14, "layers": 24, "hidden": 1024}
    vision: Optional[Dict[str, Any]] = None

    # Advanced
    use_kv_cache: bool = False  # Typically False for training

    def __post_init__(self):
        """Validation and normalization after initialization"""
        # Normalize dtype
        self.dtype = self.dtype.lower()

        # Ensure optimizations is a dict
        if self.optimizations is None:
            self.optimizations = {}

        # Auto-adjust DeepSpeed overhead based on ZeRO stage
        from ..formulas.constants import OverheadDefaults
        if self.zero_stage == 1:
            self.deepspeed_overhead_gib = max(
                self.deepspeed_overhead_gib,
                OverheadDefaults.DEEPSPEED_STAGE_1
            )
        elif self.zero_stage == 2:
            self.deepspeed_overhead_gib = max(
                self.deepspeed_overhead_gib,
                OverheadDefaults.DEEPSPEED_STAGE_2
            )
        elif self.zero_stage >= 3:
            self.deepspeed_overhead_gib = max(
                self.deepspeed_overhead_gib,
                OverheadDefaults.DEEPSPEED_STAGE_3
            )

    @property
    def has_lora(self) -> bool:
        """Check if LoRA is enabled"""
        return self.lora is not None and self.lora.get("rank", 0) > 0

    @property
    def has_vision(self) -> bool:
        """Check if vision (VLM) is enabled"""
        return self.vision is not None and self.vision.get("enabled", False)

    @property
    def total_batch_size(self) -> int:
        """Calculate total global batch size"""
        return self.per_device_batch * self.grad_accum * self.dp

    def get_optimization_params(self, opt_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific optimization.

        Args:
            opt_name: Optimization name

        Returns:
            Parameter dict (empty if optimization not enabled)
        """
        return self.optimizations.get(opt_name, {})

    def is_optimization_enabled(self, opt_name: str) -> bool:
        """Check if an optimization is enabled"""
        return opt_name in self.optimizations
