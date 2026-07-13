"""Optional tuner namespace.

The laptop configuration uses fixed compact hyperparameters by default. This
module is intentionally light so CI does not require a database-backed study.
"""

from __future__ import annotations


def tuning_not_run_reason() -> str:
    return "Laptop mode uses pre-defined compact hyperparameters; exhaustive tuning is disabled."
