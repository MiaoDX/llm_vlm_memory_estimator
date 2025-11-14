"""
Optimization system for memory estimation.

This package provides:
- OPTIMIZATION_REGISTRY: Global registry of available optimizations
- Built-in optimizations (FlashAttention, gradient checkpointing, etc.)
- Third-party plugin discovery

The registry is populated automatically when this module is imported.
"""

from .registry import OPTIMIZATION_REGISTRY
from .base import Optimization, OptimizationEffect, EffectType

# Import built-in optimizations to trigger registration
from . import builtin

# Auto-discover third-party plugins
OPTIMIZATION_REGISTRY.auto_discover_plugins()

__all__ = [
    'OPTIMIZATION_REGISTRY',
    'Optimization',
    'OptimizationEffect',
    'EffectType',
]
