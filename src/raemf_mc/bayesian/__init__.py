"""Optional Variational Bayesian scenario layer.

PyMC and ArviZ are imported lazily by :class:`VariationalScenarioModel`, so
the original RAEMF-MC pipeline has no Bayesian runtime dependency when the
feature is disabled.
"""

from raemf_mc.bayesian.variational import VariationalPosteriorResult, VariationalScenarioModel

__all__ = ["VariationalPosteriorResult", "VariationalScenarioModel"]
