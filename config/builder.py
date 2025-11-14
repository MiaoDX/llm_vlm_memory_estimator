"""
Fluent builder for EstimatorConfig.

The builder provides a readable API for constructing estimation configurations.
It produces the same EstimatorConfig structure as YAML/JSON loading would.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self
else:
    from typing import TypeVar
    Self = TypeVar("Self", bound="EstimatorConfigBuilder")

from ..core.config import EstimatorConfig


class EstimatorConfigBuilder:
    """
    Fluent builder for EstimatorConfig.

    Example:
        config = (EstimatorConfigBuilder()
            .for_model("meta-llama/Llama-2-7b-hf")
            .with_training(seq_len=4096, batch=2, grad_accum=64)
            .with_dtype("bf16")
            .with_parallelism(dp=8, tp=1, pp=1)
            .with_zero(stage=2)
            .enable_optimization("flashattention")
            .enable_optimization("liger")
            .build())
    """

    def __init__(self):
        """Initialize builder with default config"""
        self._model: Optional[str] = None
        self._seq_len: int = 2048
        self._per_device_batch: int = 1
        self._grad_accum: int = 1
        self._dtype: str = "bf16"
        self._dp: int = 1
        self._tp: int = 1
        self._pp: int = 1
        self._zero_stage: int = 0
        self._lora: Optional[Dict[str, Any]] = None
        self._optimizations: Dict[str, Dict[str, Any]] = {}
        self._vision: Optional[Dict[str, Any]] = None
        self._probe_enabled: bool = False
        self._probe_steps: int = 6
        self._probe_warmup: int = 3
        self._overrides: Dict[str, Any] = {}

    def for_model(self, model: str) -> Self:
        """
        Set model name or path.

        Args:
            model: HuggingFace model ID or local path

        Returns:
            Self for chaining
        """
        self._model = model
        return self

    def with_training(
        self,
        seq_len: int,
        batch: int,
        grad_accum: int = 1
    ) -> Self:
        """
        Set training shape parameters.

        Args:
            seq_len: Sequence length
            batch: Per-device micro-batch size
            grad_accum: Gradient accumulation steps

        Returns:
            Self for chaining
        """
        self._seq_len = seq_len
        self._per_device_batch = batch
        self._grad_accum = grad_accum
        return self

    def with_dtype(self, dtype: str) -> Self:
        """
        Set precision dtype.

        Args:
            dtype: One of: bf16, fp16, fp32, int8, int4, nf4

        Returns:
            Self for chaining
        """
        self._dtype = dtype
        return self

    def with_parallelism(
        self,
        dp: int = 1,
        tp: int = 1,
        pp: int = 1
    ) -> Self:
        """
        Set parallelism configuration.

        Args:
            dp: Data parallel degree
            tp: Tensor parallel degree
            pp: Pipeline parallel stages

        Returns:
            Self for chaining
        """
        self._dp = dp
        self._tp = tp
        self._pp = pp
        return self

    def with_zero(self, stage: int) -> Self:
        """
        Enable DeepSpeed ZeRO.

        Args:
            stage: ZeRO stage (0/1/2/3)

        Returns:
            Self for chaining
        """
        if stage not in {0, 1, 2, 3}:
            raise ValueError(f"ZeRO stage must be 0, 1, 2, or 3, got {stage}")
        self._zero_stage = stage
        return self

    def enable_optimization(self, name: str, **params: Any) -> Self:
        """
        Enable an optimization with optional parameters.

        Args:
            name: Optimization name (e.g., "flashattention", "liger")
            **params: Optimization-specific parameters

        Returns:
            Self for chaining

        Example:
            .enable_optimization("gradient_checkpointing", reduction=0.3)
        """
        self._optimizations[name] = params
        return self

    def with_lora(
        self,
        rank: int = 8,
        alpha: int = 16,
        quantized: bool = False,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Self:
        """
        Enable LoRA fine-tuning.

        Args:
            rank: LoRA rank
            alpha: LoRA alpha
            quantized: Use QLoRA (4-bit base weights)
            target_modules: List of module names to apply LoRA to
            **kwargs: Additional LoRA parameters

        Returns:
            Self for chaining
        """
        self._lora = {
            "rank": rank,
            "alpha": alpha,
            "quantized": quantized,
            **kwargs
        }
        if target_modules is not None:
            self._lora["target_modules"] = target_modules
        return self

    def with_vision(
        self,
        enabled: bool = True,
        image_size: int = 448,
        patch_size: int = 14,
        layers: int = 24,
        hidden: int = 1024,
        **kwargs: Any
    ) -> Self:
        """
        Enable vision (VLM) configuration.

        Args:
            enabled: Whether vision is enabled
            image_size: Image resolution
            patch_size: Patch size for ViT
            layers: Number of vision encoder layers
            hidden: Vision hidden dimension
            **kwargs: Additional vision parameters

        Returns:
            Self for chaining
        """
        self._vision = {
            "enabled": enabled,
            "image_size": image_size,
            "patch_size": patch_size,
            "layers": layers,
            "hidden": hidden,
            **kwargs
        }
        return self

    def with_probe(self, steps: int = 6, warmup: int = 3) -> Self:
        """
        Enable empirical probe.

        Args:
            steps: Number of probe steps
            warmup: Number of warmup steps

        Returns:
            Self for chaining
        """
        self._probe_enabled = True
        self._probe_steps = steps
        self._probe_warmup = warmup
        return self

    def override(self, **kwargs: Any) -> Self:
        """
        Override any config field directly.

        Use this for less common fields not covered by dedicated methods.

        Args:
            **kwargs: Field name -> value pairs

        Returns:
            Self for chaining

        Example:
            .override(cuda_libs_gib=3.0, fragmentation_gib=10.0)
        """
        self._overrides.update(kwargs)
        return self

    def build(self) -> EstimatorConfig:
        """
        Build and validate configuration.

        Returns:
            EstimatorConfig instance

        Raises:
            ValueError: If required fields missing or invalid
        """
        if self._model is None:
            raise ValueError("Model must be specified via for_model()")

        # Build config
        config = EstimatorConfig(
            model=self._model,
            seq_len=self._seq_len,
            per_device_batch=self._per_device_batch,
            grad_accum=self._grad_accum,
            dtype=self._dtype,
            dp=self._dp,
            tp=self._tp,
            pp=self._pp,
            zero_stage=self._zero_stage,
            lora=self._lora,
            optimizations=self._optimizations,
            vision=self._vision,
            probe_enabled=self._probe_enabled,
            probe_steps=self._probe_steps,
            probe_warmup=self._probe_warmup,
            **self._overrides  # Apply any direct overrides
        )

        # Validate (basic checks)
        from .validator import validate_config
        validate_config(config)

        return config
