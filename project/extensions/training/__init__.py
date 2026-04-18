"""
Training extension package.

Implementation lives in ``pipeline.py`` (large deps: torch, transformers). This
``__init__`` stays lightweight so ``import extensions.training`` does not load
GPU/ML libraries — use ``import training`` (shim) or
``from extensions.training.pipeline import …`` when needed.
"""
