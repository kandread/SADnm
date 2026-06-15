"""
Frozen SADnm deployment configuration.

The Stage-2 GP hyper-parameters are calibrated ONCE, offline, on the training
gauges (scripts/calibrate_sadnm.py) and shipped here as constants. Deployment
loads these and never refits. A SADnm module run needs no gauge data.

`configs/sadnm_params.json` is the artifact (written by the
calibration script); the constants below are the checked-in fallback / default
and must match it. The calibration script overwrites the JSON.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from sadnm.core import TemporalParams
from sadnm.uniform_flow import InversionConfig

CONFIG_PATH = Path(__file__).resolve().parent / 'sadnm_params.json'

# Calibrated on the 130 training gauges. Matches configs/sadnm_params.json.
DEFAULT_TEMPORAL = TemporalParams(sigma_proc=0.398, tau=5.62, sigma_obs=0.305)
DEFAULT_KERNEL = 'ou'


def save_config(temporal: TemporalParams, kernel: str = DEFAULT_KERNEL,
                inversion: InversionConfig = InversionConfig(), path: Path = CONFIG_PATH) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        'temporal': dataclasses.asdict(temporal),
        'kernel': kernel,
        'inversion': dataclasses.asdict(inversion),
        'note': 'Calibrated once on the 130 SADnm training gauges; frozen for deployment.',
    }
    path.write_text(json.dumps(blob, indent=2))
    return path


def load_config(path: Path = CONFIG_PATH):
    """Returns (TemporalParams, kernel_name, InversionConfig). Falls back to the
    checked-in defaults if the JSON artifact is absent."""
    path = Path(path)
    if not path.exists():
        return DEFAULT_TEMPORAL, DEFAULT_KERNEL, InversionConfig()
    blob = json.loads(path.read_text())
    temporal = TemporalParams(**blob['temporal'])
    inv = InversionConfig(**blob['inversion'])
    return temporal, blob.get('kernel', DEFAULT_KERNEL), inv
