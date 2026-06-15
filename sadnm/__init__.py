"""
SADnm: next generation of the SWOT Assimilated Discharge algorithm.

Per-reach river discharge from SWOT, composed of physically-interpretable stages:

  Stage 1  uniform-flow inversion   (uniform_flow.invert_reach)
  Stage 2  temporal assimilation    (core.smooth_reach; learned GP hydrograph prior)
  Stage 3  mass conservation        (spatial.graph_gmrf; optional, cross-reach)

Pure numpy/scipy. The deployment entry point is `run_reach`; hyper-parameters are
loaded frozen via `load_config` (configs calibrated once, offline).

"""
from sadnm.core import (
    TemporalParams, SADnmResult, run_reach, smooth_reach, fit_temporal,
)
from sadnm.config import load_config, save_config, DEFAULT_TEMPORAL, DEFAULT_KERNEL
from sadnm.uniform_flow import invert_reach, InversionConfig, ReachResult
from sadnm.temporal import KERNELS, ou_kernel

__version__ = "0.1.0"

__all__ = [
    "TemporalParams", "SADnmResult", "run_reach", "smooth_reach", "fit_temporal",
    "load_config", "save_config", "DEFAULT_TEMPORAL", "DEFAULT_KERNEL",
    "invert_reach", "InversionConfig", "ReachResult",
    "KERNELS", "ou_kernel", "__version__",
]
