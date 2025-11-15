"""
GPU memory probing module.

Provides empirical measurement of actual GPU memory usage.
"""

from .runner import ProbeRunner
from .results import DeviceInfo, PhaseMetrics, ProbeResult, ComparisonReport

__all__ = [
    "ProbeRunner",
    "DeviceInfo",
    "PhaseMetrics",
    "ProbeResult",
    "ComparisonReport",
]
