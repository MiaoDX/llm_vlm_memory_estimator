"""
Trainable parameters state (gradients, optimizer states, master weights).
"""

from typing import TYPE_CHECKING
from .base import MemoryComponent, ComponentMemory
from ..core.types import MemoryImpactArea
from ..formulas.constants import DTYPE_BYTES

if TYPE_CHECKING:
    from ..core.config import EstimatorConfig
    from ..core.model_info import ModelInfo
    from ..core.estimator import MemoryEstimator


class TrainableStateComponent:
    """
    Calculator for trainable parameter state memory.

    Handles:
    - Gradients
    - Optimizer states (AdamW, SGD, Adafactor)
    - Master weights (fp32 copy)
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
        # This component covers multiple areas
        return MemoryImpactArea.OPTIMIZER_STATE

    def calculate(self) -> ComponentMemory:
        """
        Calculate memory for trainable parameter states.

        Returns combined memory for:
        - Gradients (always fp32 for numeric stability)
        - Optimizer states (depends on optimizer)
        - Master weights (optional fp32 copy)
        """
        # Determine trainable parameter count
        if self.config.has_lora:
            trainable_params = self.model_info.estimate_lora_params(self.config)
        else:
            trainable_params = self.model_info.param_count

        # Gradients (always fp32 for stability)
        grad_bytes = trainable_params * 4.0

        # Optimizer states
        optimizer_bytes = self._calculate_optimizer_states(trainable_params)

        # Master weights (fp32 copy for mixed precision training)
        lora_cfg = self.config.lora if self.config.has_lora else {}
        master_weights_bytes = 0.0
        if not self.config.has_lora:  # Full fine-tuning
            # Only if dtype is not already fp32
            if self.config.dtype != "fp32" and self.config.dtype != "float32":
                master_weights_bytes = trainable_params * 4.0

        total_bytes = grad_bytes + optimizer_bytes + master_weights_bytes

        breakdown = {
            "gradients": grad_bytes,
            "optimizer_states": optimizer_bytes,
        }
        if master_weights_bytes > 0:
            breakdown["master_weights"] = master_weights_bytes

        # Apply optimization effects
        from .base import apply_effects

        # Combine effects from all relevant areas
        grad_effects = self.estimator.get_effects_for(MemoryImpactArea.GRADIENTS)
        opt_effects = self.estimator.get_effects_for(MemoryImpactArea.OPTIMIZER_STATE)

        base_result = ComponentMemory(
            name="trainable_state",
            bytes=total_bytes,
            breakdown=breakdown,
            area=MemoryImpactArea.OPTIMIZER_STATE
        )

        # Apply gradient effects first
        result = apply_effects(
            base_result,
            grad_effects,
            self.config,
            context={"param_count": trainable_params}
        )

        # Then apply optimizer effects
        result = apply_effects(
            result,
            opt_effects,
            self.config,
            context={"param_count": trainable_params}
        )

        return result

    def _calculate_optimizer_states(self, trainable_params: int) -> float:
        """
        Calculate optimizer state memory based on optimizer type.

        Args:
            trainable_params: Number of trainable parameters

        Returns:
            Optimizer state memory in bytes
        """
        # TODO: Extract optimizer name from config.optimizations
        # For now, assume AdamW (most common)
        # AdamW stores 2 states per parameter: momentum + variance
        # Each state is typically fp32 (4 bytes)
        return trainable_params * 2 * 4.0
