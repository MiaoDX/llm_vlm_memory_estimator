"""
Vision (VLM) and system overhead components.
"""

from typing import TYPE_CHECKING
from .base import MemoryComponent, ComponentMemory
from ..core.types import MemoryImpactArea
from ..formulas.constants import DTYPE_BYTES

if TYPE_CHECKING:
    from ..core.config import EstimatorConfig
    from ..core.model_info import ModelInfo
    from ..core.estimator import MemoryEstimator


class VisionComponent:
    """
    Calculator for vision encoder memory (VLM models).
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
        return MemoryImpactArea.VISION_ACTIVATIONS

    def calculate(self) -> ComponentMemory:
        """
        Calculate vision encoder memory.

        Returns zero if vision is disabled.
        """
        if not self.config.has_vision:
            return ComponentMemory(
                name="vision",
                bytes=0.0,
                breakdown={"vision": 0.0},
                area=MemoryImpactArea.VISION_ACTIVATIONS
            )

        vision_cfg = self.config.vision
        B = self.config.per_device_batch
        image_size = vision_cfg.get("image_size", 448)
        patch_size = vision_cfg.get("patch_size", 14)
        vision_hidden = vision_cfg.get("hidden", 1024)
        vision_layers = vision_cfg.get("layers", 24)

        dtype_bytes = DTYPE_BYTES.get(self.config.dtype, 2.0)

        # Number of patches
        num_patches = (image_size // patch_size) ** 2

        # Vision activations (rough estimate)
        # B * num_patches * vision_hidden * vision_layers * factor
        factor = 4.0  # Heuristic for ViT activations
        base_bytes = B * num_patches * vision_hidden * vision_layers * factor * dtype_bytes

        breakdown = {
            "vision_activations": base_bytes,
        }

        # Apply optimization effects
        from .base import apply_effects
        effects = self.estimator.get_effects_for(MemoryImpactArea.VISION_ACTIVATIONS)

        base_result = ComponentMemory(
            name="vision",
            bytes=base_bytes,
            breakdown=breakdown,
            area=MemoryImpactArea.VISION_ACTIVATIONS
        )

        return apply_effects(
            base_result,
            effects,
            self.config,
            context={
                "B": B,
                "num_patches": num_patches,
                "vision_hidden": vision_hidden,
                "vision_layers": vision_layers,
            }
        )


class OverheadsComponent:
    """
    Calculator for system overheads (CUDA, miscellaneous, fragmentation, DeepSpeed).
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
        # Multi-area component
        return MemoryImpactArea.MISC_OVERHEAD

    def calculate(self) -> ComponentMemory:
        """
        Calculate system overhead memory.

        Includes:
        - CUDA context, cuBLAS, etc.
        - Miscellaneous framework overhead
        - Allocator fragmentation
        - DeepSpeed runtime overhead
        """
        GiB = 1024 ** 3

        cuda_bytes = self.config.cuda_libs_gib * GiB
        misc_bytes = self.config.misc_overhead_gib * GiB
        fragmentation_bytes = self.config.fragmentation_gib * GiB
        deepspeed_bytes = self.config.deepspeed_overhead_gib * GiB

        total_bytes = cuda_bytes + misc_bytes + fragmentation_bytes + deepspeed_bytes

        breakdown = {
            "cuda_libs": cuda_bytes,
            "misc_overhead": misc_bytes,
            "fragmentation": fragmentation_bytes,
            "deepspeed_overhead": deepspeed_bytes,
        }

        # Apply optimization effects
        from .base import apply_effects

        # Collect effects from all overhead areas
        cuda_effects = self.estimator.get_effects_for(MemoryImpactArea.CUDA_OVERHEAD)
        misc_effects = self.estimator.get_effects_for(MemoryImpactArea.MISC_OVERHEAD)
        frag_effects = self.estimator.get_effects_for(MemoryImpactArea.FRAGMENTATION)
        deepspeed_effects = self.estimator.get_effects_for(MemoryImpactArea.DEEPSPEED_OVERHEAD)

        all_effects = cuda_effects + misc_effects + frag_effects + deepspeed_effects

        base_result = ComponentMemory(
            name="overheads",
            bytes=total_bytes,
            breakdown=breakdown,
            area=MemoryImpactArea.MISC_OVERHEAD
        )

        return apply_effects(
            base_result,
            all_effects,
            self.config,
            context={}
        )
