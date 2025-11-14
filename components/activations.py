"""
Activation memory components (attention and MLP).
"""

from typing import TYPE_CHECKING
from .base import MemoryComponent, ComponentMemory
from ..core.types import MemoryImpactArea
from ..formulas.constants import DTYPE_BYTES

if TYPE_CHECKING:
    from ..core.config import EstimatorConfig
    from ..core.model_info import ModelInfo
    from ..core.estimator import MemoryEstimator


class AttentionActivationsComponent:
    """
    Calculator for attention activation memory.

    Handles:
    - Attention intermediate activations
    - FlashAttention memory reduction
    - Gradient checkpointing effects
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
        return MemoryImpactArea.ACTIVATIONS_ATTENTION

    def calculate(self) -> ComponentMemory:
        """
        Calculate attention activation memory.

        Memory depends on:
        - Batch size, sequence length
        - Whether FlashAttention is enabled
        - Gradient checkpointing
        """
        B = self.config.per_device_batch
        L = self.config.seq_len
        H = self.model_info.hidden_size
        layers = self.model_info.num_layers

        dtype_bytes = DTYPE_BYTES.get(self.config.dtype, 2.0)

        # Determine activation factor based on FlashAttention
        is_flashattn = self.config.is_optimization_enabled("flashattention")
        if is_flashattn:
            factor = self.config.attn_act_factor_fa
        else:
            factor = self.config.attn_act_factor_no_fa

        # Base activation memory
        # Approximation: B * L * H * factor * layers
        base_bytes = B * L * H * factor * layers * dtype_bytes

        breakdown = {
            "attention_activations": base_bytes,
        }

        # Apply optimization effects (FlashAttention, gradient checkpointing)
        from .base import apply_effects
        effects = self.estimator.get_effects_for(MemoryImpactArea.ACTIVATIONS_ATTENTION)

        base_result = ComponentMemory(
            name="attention_activations",
            bytes=base_bytes,
            breakdown=breakdown,
            area=MemoryImpactArea.ACTIVATIONS_ATTENTION
        )

        return apply_effects(
            base_result,
            effects,
            self.config,
            context={
                "B": B,
                "L": L,
                "H": H,
                "layers": layers,
            }
        )


class MLPActivationsComponent:
    """
    Calculator for MLP activation memory.
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
        return MemoryImpactArea.ACTIVATIONS_MLP

    def calculate(self) -> ComponentMemory:
        """
        Calculate MLP activation memory.
        """
        B = self.config.per_device_batch
        L = self.config.seq_len
        H = self.model_info.hidden_size
        layers = self.model_info.num_layers

        dtype_bytes = DTYPE_BYTES.get(self.config.dtype, 2.0)

        # MLP activations
        factor = self.config.mlp_act_factor
        base_bytes = B * L * H * factor * layers * dtype_bytes

        breakdown = {
            "mlp_activations": base_bytes,
        }

        # Apply optimization effects (gradient checkpointing)
        from .base import apply_effects
        effects = self.estimator.get_effects_for(MemoryImpactArea.ACTIVATIONS_MLP)

        base_result = ComponentMemory(
            name="mlp_activations",
            bytes=base_bytes,
            breakdown=breakdown,
            area=MemoryImpactArea.ACTIVATIONS_MLP
        )

        return apply_effects(
            base_result,
            effects,
            self.config,
            context={
                "B": B,
                "L": L,
                "H": H,
                "layers": layers,
            }
        )
