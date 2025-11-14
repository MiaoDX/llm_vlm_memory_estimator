"""
Logits and KV cache memory components.
"""

from typing import TYPE_CHECKING
from .base import MemoryComponent, ComponentMemory
from ..core.types import MemoryImpactArea
from ..formulas.constants import DTYPE_BYTES

if TYPE_CHECKING:
    from ..core.config import EstimatorConfig
    from ..core.model_info import ModelInfo
    from ..core.estimator import MemoryEstimator


class LogitsComponent:
    """
    Calculator for output logits memory.

    Handles:
    - Logits tensor (B, L, vocab_size)
    - Fused loss effects
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
        return MemoryImpactArea.LOGITS

    def calculate(self) -> ComponentMemory:
        """
        Calculate logits memory.

        Logits tensor: (batch, seq_len, vocab_size)
        """
        B = self.config.per_device_batch
        L = self.config.seq_len
        V = self.model_info.vocab_size

        dtype_bytes = DTYPE_BYTES.get(self.config.dtype, 2.0)

        # Logits memory with factor (fused loss can reduce)
        factor = self.config.logits_factor
        base_bytes = B * L * V * factor * dtype_bytes

        breakdown = {
            "logits": base_bytes,
        }

        # Apply optimization effects (fused loss)
        from .base import apply_effects
        effects = self.estimator.get_effects_for(MemoryImpactArea.LOGITS)

        base_result = ComponentMemory(
            name="logits",
            bytes=base_bytes,
            breakdown=breakdown,
            area=MemoryImpactArea.LOGITS
        )

        return apply_effects(
            base_result,
            effects,
            self.config,
            context={
                "B": B,
                "L": L,
                "vocab_size": V,
            }
        )


class KVCacheComponent:
    """
    Calculator for KV cache memory (typically only for inference).
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
        return MemoryImpactArea.KV_CACHE

    def calculate(self) -> ComponentMemory:
        """
        Calculate KV cache memory (if enabled).

        Typically disabled for training, enabled for inference.
        """
        if not self.config.use_kv_cache:
            return ComponentMemory(
                name="kv_cache",
                bytes=0.0,
                breakdown={"kv_cache": 0.0},
                area=MemoryImpactArea.KV_CACHE
            )

        B = self.config.per_device_batch
        L = self.config.seq_len
        H = self.model_info.hidden_size
        layers = self.model_info.num_layers

        dtype_bytes = DTYPE_BYTES.get(self.config.dtype, 2.0)

        # KV cache: 2 (K, V) * B * L * H * layers
        base_bytes = 2 * B * L * H * layers * dtype_bytes

        breakdown = {
            "kv_cache": base_bytes,
        }

        # Apply optimization effects
        from .base import apply_effects
        effects = self.estimator.get_effects_for(MemoryImpactArea.KV_CACHE)

        base_result = ComponentMemory(
            name="kv_cache",
            bytes=base_bytes,
            breakdown=breakdown,
            area=MemoryImpactArea.KV_CACHE
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
