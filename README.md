# SADnm

**Next generation of the SWOT Assimilated Discharge algorithm**

Per-reach river discharge from SWOT observations. A clean numpy/scipy package that replaces
the original (Julia) SAD which used rejection sampling and a Local Ensemble Kalman Filter for estimation.

## Stages

1. **Uniform-flow inversion** (`uniform_flow.invert_reach`): Dingman cross-section & Manning equation, flow anchored to the monthly prior; one independent
   Q per overpass from that overpass's WSE node profile.
2. **Temporal assimilation** (`core.smooth_reach`): a learned Gaussian Process hydrograph prior over the log-Q anomaly denoises the per-overpass estimates and yields predictive uncertainty. Hyper-parameters are calibrated once, offline, and
   shipped frozen (`sadnm/sadnm_params.json`); deployment **never** refits.
3. **Mass conservation** (`spatial.graph_gmrf`, optional): spatial coupling across SWORD-connected reaches; needs cross-reach orchestration, so it is an optional enhancement, not part of the per-reach core.

## Validated result

On the SVS validation gauges, SADnm produces valid estimates (with per-overpass uncertainty) for 2,522 reaches with the following validation statistics
|                     | 32th percentile | Median | 68th percentile |
|---------------------|-----------------|--------|-----------------|
| NSE                 | -0.04           | 0.30   | 0.59            |
| KGE                 | 0.08            | 0.30   | 0.49            |
| Pearson correlation | 0.75            | 0.88   | 0.94            |
| Absolute nBIAS      | 0.16            | 0.27   | 0.43            |


## Install

```bash
pip install -e .            # core (numpy, scipy)
pip install -e .[io]        # + netCDF4 for I/O and Stage 3
```

## Usage

```python
import sadnm

params, kernel_name, inv_cfg = sadnm.load_config()      # frozen, shipped config
res = sadnm.run_reach(
    wse_norm, width_norm, node_mask, overpass_mask, node_id, overpass_time_s,
    monthly_q, norm_stats, mu, params,
    eval_overpass_idx=overpass_idx, cfg=inv_cfg,
)
# res.q (smoothed discharge), res.log_sigma (uncertainty), res.ok / res.cal_resid
```

## Deployment

The Confluence FLPE wrapper (entry point + Dockerfile) lives in the [Confluence SAD module repository](https://github.com/SWOT-Confluence/sad) and depends on this package; it replaces the Julia SAD module while preserving the `<reach_id>_sad.nc` output specification.
