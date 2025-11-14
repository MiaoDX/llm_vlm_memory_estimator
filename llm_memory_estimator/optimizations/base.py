"""
Base types for optimization system.

This module defines the core Optimization protocol and OptimizationEffect dataclass.
All optimizations (built-in and third-party) must implement the Optimization protocol.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Callable, Optional, List, Union, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.config import EstimatorConfig
    from ..probe.modifiers import ProbeModifier

from ..core.types import MemoryImpactArea


# Type aliases for clarity and type safety
FormulaFn = Callable[[int, int, int, int], float]  # (B, L, H, layers) -> memory_bytes
ConditionFn = Callable[['EstimatorConfig'], bool]  # config -> should_apply


class EffectType(Enum):
    """
    How an optimization modifies memory usage.

    MULTIPLY: Scale existing memory by a factor (e.g., 0.8 for FlashAttention)
    REPLACE_FORMULA: Replace the entire calculation with a new formula
    ADD: Add or subtract a fixed amount of memory
    """
    MULTIPLY = "multiply"
    REPLACE_FORMULA = "replace"
    ADD = "add"


@dataclass(frozen=True)
class OptimizationEffect:
    """
    Describes how an optimization affects memory in a specific area.

    Immutable for safety in concurrent contexts and caching.

    Attributes:
        area: Which memory component this affects
        effect_type: How the effect is applied (multiply/replace/add)
        value: Either a scalar (for multiply/add) or a formula function (for replace)
        condition: Optional callable to determine if effect applies
        description: Human-readable explanation for reporting
    """
    area: MemoryImpactArea
    effect_type: EffectType

    # Value type depends on effect_type:
    # - MULTIPLY/ADD: float (multiplier or addend)
    # - REPLACE_FORMULA: FormulaFn (takes B, L, H, layers, returns bytes)
    value: Union[float, FormulaFn]

    # Optional condition - when is this effect active?
    # If None, always applies. If provided, calls condition(config) -> bool
    condition: Optional[ConditionFn] = None

    # Human-readable explanation for breakdown reporting
    description: str = ""

    def applies_to(self, config: 'EstimatorConfig') -> bool:
        """
        Check if this effect applies given the configuration.

        Args:
            config: Estimator configuration

        Returns:
            True if effect should be applied, False otherwise
        """
        if self.condition is None:
            return True
        return self.condition(config)

    def __post_init__(self):
        """Validate effect type and value match"""
        if self.effect_type in (EffectType.MULTIPLY, EffectType.ADD):
            if not isinstance(self.value, (int, float)):
                raise TypeError(
                    f"{self.effect_type.value} effect requires numeric value, "
                    f"got {type(self.value)}"
                )
        elif self.effect_type == EffectType.REPLACE_FORMULA:
            if not callable(self.value):
                raise TypeError(
                    f"{self.effect_type.value} effect requires callable value, "
                    f"got {type(self.value)}"
                )


class Optimization(Protocol):
    """
    Protocol for optimization techniques.

    All optimizations (built-in and third-party plugins) must implement this protocol.
    The registry will validate implementations at registration time.

    Example:
        class FlashAttentionOptimization:
            @property
            def name(self) -> str:
                return "flashattention"

            def get_effects(self, config):
                return [OptimizationEffect(...)]

            def conflicts_with(self) -> List[str]:
                return []

            def get_probe_modifier(self):
                return None

            def get_documentation(self) -> str:
                return "FlashAttention reduces memory..."
    """

    @property
    def name(self) -> str:
        """
        Unique identifier for this optimization.

        Must be lowercase, alphanumeric + underscores only.
        Used in config files and CLI arguments.
        """
        ...

    def get_effects(self, config: 'EstimatorConfig') -> List[OptimizationEffect]:
        """
        Return all memory effects this optimization provides.

        Effects describe how this optimization modifies memory usage in different areas.
        Multiple effects can target different memory impact areas.

        Args:
            config: Full estimator configuration for conditional effects

        Returns:
            List of optimization effects
        """
        ...

    def conflicts_with(self) -> List[str]:
        """
        Names of optimizations this conflicts with.

        The registry will enforce that conflicting optimizations cannot be
        activated simultaneously.

        Returns:
            List of optimization names that conflict with this one
        """
        ...

    def get_probe_modifier(self) -> Optional['ProbeModifier']:
        """
        Optional: how to configure empirical probe for this optimization.

        Returns a ProbeModifier that sets up the probe environment
        (e.g., enabling FlashAttention in the model).

        Returns:
            ProbeModifier instance, or None if no special setup needed
        """
        ...

    def get_documentation(self) -> str:
        """
        User-facing documentation for this optimization.

        Should explain:
        - What the optimization does
        - Memory savings expected
        - Any requirements or limitations
        - Configuration parameters

        Returns:
            Multi-line documentation string
        """
        ...
