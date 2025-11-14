"""
LLM Memory Estimator - GPU Memory Estimation for LLM/VLM Fine-tuning

This package provides tools to estimate GPU memory requirements for training
Large Language Models and Vision-Language Models.
"""

__version__ = "0.1.0"

# Core exports
from .core.estimator import MemoryEstimator
from .core.config import EstimatorConfig
from .core.results import EstimationResult

__all__ = [
    "MemoryEstimator",
    "EstimatorConfig",
    "EstimationResult",
    "__version__",
]
