"""clinrobust: robustness audits for clinical prediction models.

A reproducible protocol for evaluating machine-learning staging/diagnostic
models under missing modalities, progressive missingness and measurement noise.
"""
__version__ = "0.1.0"

from . import config, data, models, robustness, reporting  # noqa: F401
