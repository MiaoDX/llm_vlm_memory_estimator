"""
Component factory for creating memory component calculators.

This module provides the ComponentFactory class that instantiates
the appropriate memory component calculators based on configuration.
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import EstimatorConfig
    from ..core.model_info import ModelInfo
    from ..core.estimator import MemoryEstimator
    from ..components.base import MemoryComponent


class ComponentFactory:
    """
    Factory for creating memory component calculators.

    This factory instantiates all relevant component calculators
    based on the estimator configuration.
    """

    @staticmethod
    def create_all_components(
        config: 'EstimatorConfig',
        model_info: 'ModelInfo',
        estimator: 'MemoryEstimator'
    ) -> List['MemoryComponent']:
        """
        Create all relevant memory component calculators.

        Args:
            config: Estimator configuration
            model_info: Model architecture information
            estimator: MemoryEstimator instance (for accessing optimization effects)

        Returns:
            List of memory component calculators
        """
        from ..components.weights import WeightsComponent
        from ..components.trainable_state import TrainableStateComponent
        from ..components.activations import (
            AttentionActivationsComponent,
            MLPActivationsComponent,
        )
        from ..components.logits_kv import LogitsComponent, KVCacheComponent
        from ..components.overheads import VisionComponent, OverheadsComponent

        components = []

        # Always include base components
        components.append(WeightsComponent(config, model_info, estimator))
        components.append(TrainableStateComponent(config, model_info, estimator))
        components.append(AttentionActivationsComponent(config, model_info, estimator))
        components.append(MLPActivationsComponent(config, model_info, estimator))
        components.append(LogitsComponent(config, model_info, estimator))

        # Optional components
        if config.use_kv_cache:
            components.append(KVCacheComponent(config, model_info, estimator))

        if config.has_vision:
            components.append(VisionComponent(config, model_info, estimator))

        # Always include overheads
        components.append(OverheadsComponent(config, model_info, estimator))

        return components
