"""
Base types for memory component system.

This module defines the MemoryComponent protocol, ComponentMemory dataclass,
and the shared apply_effects() helper to ensure consistent effect application.
"""

from dataclasses import dataclass, field
from typing import Protocol, Dict, List, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.config import EstimatorConfig

from ..core.types import MemoryImpactArea
from ..optimizations.base import OptimizationEffect, EffectType


@dataclass
class ComponentMemory:
    """
    Result of a memory component calculation.

    Attributes:
        name: Human-readable component name (e.g., "attention_activations")
        bytes: Total memory in bytes for this component
        breakdown: Detailed breakdown showing calculation steps
        area: Which memory impact area this component corresponds to
    """
    name: str
    bytes: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    area: MemoryImpactArea = MemoryImpactArea.MISC_OVERHEAD  # Default fallback

    def to_gib(self) -> float:
        """Convert bytes to GiB"""
        from ..formulas.constants import to_gib
        return to_gib(self.bytes)


class MemoryComponent(Protocol):
    """
    Protocol for memory component calculators.

    Each calculator is responsible for estimating one aspect of memory usage
    (e.g., weights, attention activations, optimizer state).

    Calculators use the shared apply_effects() helper to apply optimization
    effects consistently.
    """

    def calculate(self) -> ComponentMemory:
        """
        Calculate memory for this component.

        Should compute base memory, then call apply_effects() to apply
        any active optimization effects.

        Returns:
            ComponentMemory with bytes and detailed breakdown
        """
        ...

    def get_area(self) -> MemoryImpactArea:
        """
        Which memory impact area this component covers.

        Used to route optimization effects to the correct calculator.

        Returns:
            MemoryImpactArea enum value
        """
        ...


def apply_effects(
    base_memory: ComponentMemory,
    effects: List[OptimizationEffect],
    config: 'EstimatorConfig',
    context: Dict[str, Any]
) -> ComponentMemory:
    """
    Shared helper to apply optimization effects to a memory component.

    This function ensures consistent application logic across all calculators,
    preventing duplication and subtle inconsistencies.

    Args:
        base_memory: Initial component memory calculation (before optimizations)
        effects: List of applicable optimization effects for this component
        config: Estimator configuration (for conditional effects)
        context: Dictionary with formula parameters for REPLACE_FORMULA effects
                 Typically includes: B (batch), L (seq_len), H (hidden_size),
                 layers, num_heads, vocab_size, etc.

    Returns:
        Modified ComponentMemory with effects applied and breakdown updated

    Example:
        base = ComponentMemory("attention", bytes=1000, breakdown={}, area=ACTIVATIONS_ATTENTION)
        effects = [OptimizationEffect(area=ACTIVATIONS_ATTENTION, effect_type=MULTIPLY, value=0.8)]
        result = apply_effects(base, effects, config, {"B": 2, "L": 2048, ...})
        # result.bytes == 800
    """
    modified_bytes = base_memory.bytes
    breakdown = base_memory.breakdown.copy()

    # Always record the base calculation
    if "base" not in breakdown:
        breakdown["base"] = base_memory.bytes

    # Apply each effect in sequence
    for effect in effects:
        # Check if effect applies to this configuration
        if not effect.applies_to(config):
            continue

        effect_desc = effect.description or effect.effect_type.value

        if effect.effect_type == EffectType.MULTIPLY:
            # Type narrowing: value must be float for MULTIPLY
            assert isinstance(effect.value, (int, float)), \
                f"MULTIPLY effect requires numeric value, got {type(effect.value)}"

            modified_bytes *= effect.value
            breakdown[effect_desc] = modified_bytes

        elif effect.effect_type == EffectType.REPLACE_FORMULA:
            # Type narrowing: value must be callable for REPLACE_FORMULA
            assert callable(effect.value), \
                f"REPLACE_FORMULA effect requires callable, got {type(effect.value)}"

            # Extract parameters from context
            try:
                # Standard formula signature: (B, L, H, layers)
                B = context.get("B", 1)
                L = context.get("L", 2048)
                H = context.get("H", 4096)
                layers = context.get("layers", 32)

                modified_bytes = effect.value(B, L, H, layers)
                breakdown[effect_desc] = modified_bytes

            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"REPLACE_FORMULA effect failed: missing required context parameters. "
                    f"Expected: B, L, H, layers. Got: {list(context.keys())}. Error: {e}"
                )

        elif effect.effect_type == EffectType.ADD:
            # Type narrowing: value must be float for ADD
            assert isinstance(effect.value, (int, float)), \
                f"ADD effect requires numeric value, got {type(effect.value)}"

            modified_bytes += effect.value
            breakdown[effect_desc] = modified_bytes

    return ComponentMemory(
        name=base_memory.name,
        bytes=max(0.0, modified_bytes),  # Clamp to non-negative
        breakdown=breakdown,
        area=base_memory.area
    )
