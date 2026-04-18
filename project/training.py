"""Compatibility shim — bind ``training`` to ``extensions.training.pipeline`` (includes private names)."""

import sys

from extensions.training import pipeline as _pipeline

sys.modules[__name__] = _pipeline
