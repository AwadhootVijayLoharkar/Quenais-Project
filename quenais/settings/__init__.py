"""
Grouped settings for the QuEnAIS pipeline.

Config holds molecule identity, paths and the solver choice; everything
else lives in one of the dataclasses here and is reached as cfg.<group>.<field>
(cfg.dmet.bath_tolerance, cfg.gqe.ngates). Grouping rather than flattening
is deliberate -- the flat version of this is what grew to 800 lines.

Nothing in this subpackage imports NumPy, PySCF, Qiskit or CUDA-Q.
"""

from quenais.settings.asf import DEFAULT_ASF_PARAMS, AsfSettings
from quenais.settings.dmet import REFERENCE_METHODS, DmetSettings
from quenais.settings.gqe import (
    CUDAQ_SIMULATOR_TARGETS,
    DMET_POOL_SPECS,
    OPERATOR_POOL_SPECS,
    GqeSettings,
)
from quenais.settings.qiskit_solver import (
    ANSATZE,
    BACKENDS,
    MAPPINGS,
    QiskitSolverSettings,
)
from quenais.settings.tiering import TM_ELEMENTS, TierSettings

__all__ = [
    "AsfSettings",
    "DmetSettings",
    "GqeSettings",
    "QiskitSolverSettings",
    "TierSettings",
    "DEFAULT_ASF_PARAMS",
    "REFERENCE_METHODS",
    "OPERATOR_POOL_SPECS",
    "DMET_POOL_SPECS",
    "CUDAQ_SIMULATOR_TARGETS",
    "ANSATZE",
    "MAPPINGS",
    "BACKENDS",
    "TM_ELEMENTS",
]
