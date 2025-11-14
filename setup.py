#!/usr/bin/env python3
"""
Setup script for llm-memory-estimator package
"""
from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "LLM/VLM GPU Memory Estimator"

# Read requirements
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
if os.path.exists(requirements_path):
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "torch>=2.1.0",
        "pandas>=2.0.0",
    ]

setup(
    name="llm_memory_estimator",
    version="0.1.0",
    author="LLM Memory Estimator Contributors",
    author_email="",
    description="Estimate GPU memory requirements for LLM/VLM fine-tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llm-memory-estimator/llm_memory_estimator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gradio": ["gradio>=4.40.0"],
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "llm-memory-estimator=llm_memory_estimator.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.json", "docs/*"],
    },
)
