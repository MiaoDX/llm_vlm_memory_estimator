"""
Model weights memory component calculator.
"""

from typing import TYPE_CHECKING
from .base import MemoryComponent, ComponentMemory
from ..core.types import MemoryImpactArea
from ..formulas.constants import DTYPE_BYTES

if TYPE_CHECKING:
    from ..core.config import EstimatorConfig
    from ..core.model_info import ModelInfo
    from ..core.estimator import MemoryEstimator


class WeightsComponent:
    """
    Calculator for model weights memory.

    Handles:
    - Base model weights
    - LoRA weights (if enabled)
    - Quantization effects on weight memory
    """

    def __init__(
        self,
        config: 'EstimatorConfig',
        model_info: 'ModelInfo',
        estimator: 'MemoryEstimator'
    ):
        self.config = config
        self.model_info = model_info
        self.estimator = estimator

    def get_area(self) -> MemoryImpactArea:
        return MemoryImpactArea.WEIGHTS

    def calculate(self) -> ComponentMemory:
        """
        Calculate memory for model weights.

        Considers:
        - Base parameter count
        - Weight dtype (including quantization)
        - LoRA parameters (if enabled)
        """
        # Determine which parameters are trainable
        if self.config.has_lora:
            # LoRA: base weights + LoRA adapters
            lora_params = self.model_info.estimate_lora_params(self.config)

            # Base weights dtype (may be quantized for QLoRA)
            lora_cfg = self.config.lora
            if lora_cfg.get("quantized", False):
                # QLoRA: 4-bit base weights
                base_dtype = "nf4"
            else:
                base_dtype = self.config.dtype

            base_bytes = self.model_info.param_count * DTYPE_BYTES.get(base_dtype, 2.0)

            # LoRA weights always stored in training precision
            lora_bytes = lora_params * DTYPE_BYTES.get(self.config.dtype, 2.0)

            total_bytes = base_bytes + lora_bytes

            breakdown = {
                "base_weights": base_bytes,
                "lora_weights": lora_bytes,
            }
        else:
            # Full fine-tuning: all weights in training dtype
            total_bytes = self.model_info.param_count * DTYPE_BYTES.get(self.config.dtype, 2.0)
            breakdown = {
                "model_weights": total_bytes,
            }

        # Apply optimization effects (though weights typically aren't optimized much)
        from .base import apply_effects
        effects = self.estimator.get_effects_for(MemoryImpactArea.WEIGHTS)

        base_result = ComponentMemory(
            name="weights",
            bytes=total_bytes,
            breakdown=breakdown,
            area=MemoryImpactArea.WEIGHTS
        )

        return apply_effects(
            base_result,
            effects,
            self.config,
            context={"param_count": self.model_info.param_count}
        )
