#!/usr/bin/env python3
"""
Hugging Face Spaces entry point for LLM/VLM Memory Estimator
"""
import sys
from pathlib import Path

# Add project root to Python path to allow imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_memory_estimator.app import demo

if __name__ == "__main__":
    demo.launch()
