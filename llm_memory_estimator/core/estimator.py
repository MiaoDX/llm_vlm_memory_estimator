"""
Main memory estimation orchestrator.

The MemoryEstimator coordinates:
- Model info loading
- Optimization activation (local state, not global)
- Component calculation via factory
- Parallelism scaling
- Result aggregation
"""

from typing import List, Optional

from .config import EstimatorConfig
from .model_info import ModelInfo
from .results import EstimationResult
from .types import MemoryImpactArea
from ..optimizations.base import Optimization, OptimizationEffect
from ..optimizations import OPTIMIZATION_REGISTRY


class MemoryEstimator:
    """
    Main orchestrator for memory estimation.

    Each estimator instance holds its own list of activated optimization instances.
    This design allows multiple estimators to coexist safely without global state conflicts.

    Attributes:
        config: Estimation configuration
        model_info: Model architecture details
        _active_optimizations: List of activated optimization instances (local state)
    """

    def __init__(self, config: EstimatorConfig):
        """
        Initialize estimator and activate optimizations.

        Args:
            config: Estimation configuration

        Raises:
            ValueError: If conflicting optimizations are enabled
        """
        self.config = config
        self.model_info = ModelInfo.from_config(
            config.model,
            config.vocab_size_override
        )

        # Each estimator has its own activated optimization instances (local state)
        self._active_optimizations: List[Optimization] = []
        self._activate_optimizations()

    def _activate_optimizations(self) -> None:
        """
        Instantiate and activate optimizations from config.

        Validates that no conflicting optimizations are enabled.

        Raises:
            ValueError: If conflicting optimizations detected
        """
        for opt_name, opt_params in self.config.optimizations.items():
            # Get class from global registry (definition only, not state)
            try:
                OptClass = OPTIMIZATION_REGISTRY.get(opt_name)
            except ValueError as e:
                # Unknown optimization - provide helpful error with suggestions
                suggestions = OPTIMIZATION_REGISTRY.suggest_similar(opt_name)
                suggestion_text = ""
                if suggestions:
                    suggestion_text = f" Did you mean: {', '.join(suggestions)}?"
                raise ValueError(
                    f"Unknown optimization '{opt_name}'.{suggestion_text}\n"
                    f"Available: {', '.join(OPTIMIZATION_REGISTRY.list_available())}"
                ) from e

            # Instantiate with parameters
            try:
                opt_instance = OptClass(**opt_params)
            except TypeError as e:
                raise ValueError(
                    f"Failed to instantiate optimization '{opt_name}' with params {opt_params}: {e}"
                ) from e

            # Check conflicts with already-active optimizations
            for active in self._active_optimizations:
                if opt_name in active.conflicts_with():
                    raise ValueError(
                        f"Optimization '{opt_name}' conflicts with '{active.name}'. "
                        f"Cannot enable both simultaneously."
                    )
                if active.name in opt_instance.conflicts_with():
                    raise ValueError(
                        f"Optimization '{active.name}' conflicts with '{opt_name}'. "
                        f"Cannot enable both simultaneously."
                    )

            self._active_optimizations.append(opt_instance)

    def get_effects_for(self, area: MemoryImpactArea) -> List[OptimizationEffect]:
        """
        Get all optimization effects for a specific memory area from active optimizations.

        This method is called by component calculators to retrieve applicable effects.

        Args:
            area: Memory impact area to query

        Returns:
            List of applicable optimization effects
        """
        effects = []
        for opt in self._active_optimizations:
            for effect in opt.get_effects(self.config):
                if effect.area == area:
                    effects.append(effect)
        return effects

    def estimate(self) -> EstimationResult:
        """
        Run memory estimation.

        Returns:
            Estimation results with detailed breakdown

        Raises:
            RuntimeError: If estimation fails
        """
        # Import here to avoid circular dependency
        from ..utils.factory import ComponentFactory
        from ..parallelism import create_parallelism_strategy
        from ..parallelism.aggregator import MemoryAggregator

        # Create component calculators via factory
        components = ComponentFactory.create_all_components(
            self.config,
            self.model_info,
            self  # Pass estimator so components can call get_effects_for()
        )

        # Calculate each component
        component_results = []
        for component in components:
            if component is not None:  # Some components may be None (e.g., vision if disabled)
                try:
                    result = component.calculate()
                    component_results.append(result)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to calculate component {component.get_area()}: {e}"
                    ) from e

        # Apply parallelism scaling
        parallelism = create_parallelism_strategy(self.config)
        scaled_results = parallelism.scale_components(component_results)

        # Aggregate into final result
        aggregator = MemoryAggregator()
        result = aggregator.aggregate(scaled_results, self.config, self.model_info)

        return result

    def probe(self) -> 'ProbeResult':
        """
        Run empirical probe with synthetic data.

        Requires PyTorch and transformers to be installed.

        Returns:
            Probe results with measured memory

        Raises:
            RuntimeError: If probe dependencies not installed or probe fails
        """
        try:
            from ..probe.runner import ProbeRunner
        except ImportError as e:
            raise RuntimeError(
                "Probe requires PyTorch and transformers. "
                "Install with: pip install torch transformers accelerate\n"
                f"Original error: {e}"
            ) from e

        runner = ProbeRunner(self.config, self.model_info, self._active_optimizations)
        return runner.run()

    def list_active_optimizations(self) -> List[str]:
        """Get list of active optimization names"""
        return [opt.name for opt in self._active_optimizations]

    def get_optimization_documentation(self, opt_name: str) -> Optional[str]:
        """
        Get documentation for an active optimization.

        Args:
            opt_name: Optimization name

        Returns:
            Documentation string, or None if optimization not active
        """
        for opt in self._active_optimizations:
            if opt.name == opt_name:
                return opt.get_documentation()
        return None
