"""
Per-reach uniform flow inversion. It applies Manning's equation (the
steady uniform-flow formula, friction slope ~ water-surface slope) on a Dingman
cross-section, as distinct from a gradually-varied-flow (GVF) solve.

Pipeline per reach (monthly prior):

  1. denoise_stage_width : robust line fit through the node WSE profile per
     overpass (outlier reject, evaluate at a consistent reference node) -> a clean
     stage series H(t); reach-mean width W(t) as a static shape estimator.
  2. estimate_r          : Dingman shape r from the width-height geometry
     (1/r = TLS slope of log W vs log depth) -- NOT fit to Q.
  3. calibrate_level     : with the exponent fixed from geometry, fit the level to
     the monthly prior on across-month aggregates (anchors level/bias to prior).
  4. invert              : Q(t) = exp(a*log(H(t)-Z0) + C) from observed stage.

The discharge dynamics come from the height physics, the level
from the prior (bias = prior).
"""

import datetime
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import UnivariateSpline

A_FLOOR = 5.0 / 3.0
_EPOCH = datetime.datetime(2000, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class InversionConfig:
    rmse_max: float = 1.0 # m, profile-quality gate (spline/line residual)
    min_nodes: int = 5
    min_good_overpass: int = 8
    min_months: int = 4
    min_geom_overpass: int = 8
    inv_r_min: float = 0.2 # 1/r bounds  (r in [0.3, 5])
    inv_r_max: float = 1.0 / 0.3
    d0_grid_lo: float = 0.2 # m, baseflow-depth search
    d0_grid_hi: float = 20.0
    d0_grid_n: int = 25
    min_eval_pairs: int = 5
    # Spline & spatial-Z0 water-surface treatment.
    # NOTE: measured to beat the reference at matched (~25%) coverage but to degrade
    # mid/full coverage vs the robust line (it only helps the cleanest reaches). Kept
    # as an option for the accuracy-selected mode; robust line is the balanced default.
    use_spline: bool = False # smoothing spline & spatial bed datum (vs robust line)
    spline_s: float = 0.01 # smoothing ~ (0.1 m)^2 per node (UnivariateSpline s factor)
    mono_min: float = 0.85 # min monotonic fraction of the splined profile


@dataclass
class ReachResult:
    ok: bool
    reason: str = ""
    q: np.ndarray | None = None # inverted Q at matched overpass indices
    overpass_idx: np.ndarray | None = None
    r_shape: float = np.nan # Dingman shape r
    d0: float = np.nan # baseflow depth (m)
    C: float = np.nan # level coefficient
    W0: float = np.nan # width at the baseflow depth d0 (m)
    Hmin: float = np.nan # lowest calibration stage, in the WSE datum (m)
    cal_resid: float = np.nan # calibration residual (selection metric)
    n_used: int = 0
    used_spline: bool = False # spline path vs robust-line fallback


def months_of(overpass_time_s: np.ndarray) -> np.ndarray:
    return np.array(
        [(_EPOCH + datetime.timedelta(seconds=float(t))).month - 1 for t in overpass_time_s],
        dtype=np.int32,
    )


def robust_stage(x, y, x_ref, cfg: InversionConfig, max_iter: int = 3):
    """Robust linear fit y~x with ITERATED 3-sigma outlier rejection.

    A single pass under-rejects when large outliers inflate sigma; iterating
    shrinks sigma as the worst outliers drop out and catches the rest.
    Returns (stage@x_ref, rmse_of_inliers, n_used).
    """
    if len(x) < cfg.min_nodes:
        return np.nan, np.inf, 0
    A = np.vstack([x, np.ones_like(x)]).T
    keep = np.ones(len(x), dtype=bool)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(max_iter):
        resid = y - A @ coef
        s = max(np.std(resid[keep]), 1e-6)
        new_keep = np.abs(resid) < 3 * s
        if new_keep.sum() < cfg.min_nodes or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= cfg.min_nodes else keep
            break
        keep = new_keep
        coef = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
    resid = y[keep] - A[keep] @ coef
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return float(coef[0] * x_ref + coef[1]), rmse, int(keep.sum())


def denoise_stage_width(
    wse_norm: np.ndarray, # (ob, nb) normalised node WSE
    width_norm: np.ndarray, # (ob, nb) normalised node width
    node_mask: np.ndarray, # (ob, nb) bool
    overpass_mask: np.ndarray, # (ob,) bool
    node_id: np.ndarray, # (nb,) int (0 = padding)
    norm_stats: dict, # wse_mean/std, width_mean/std
    cfg: InversionConfig,
):
    """Return (stage(ob,), width(ob,), good(ob,) bool). NaN where rejected."""
    n_real = int(np.sum(node_id > 0))
    nt = wse_norm.shape[0]
    stage = np.full(nt, np.nan)
    width = np.full(nt, np.nan)
    if n_real < cfg.min_nodes:
        return stage, width, np.zeros(nt, bool)

    x_all = np.arange(n_real, dtype=np.float64)
    x_ref = (n_real - 1) / 2.0
    for t in range(nt):
        if not overpass_mask[t]:
            continue
        valid = node_mask[t, :n_real]
        if valid.sum() < cfg.min_nodes:
            continue
        H = wse_norm[t, :n_real][valid] * norm_stats['wse_std'] + norm_stats['wse_mean']
        st, rmse, _ = robust_stage(x_all[valid], H, x_ref, cfg)
        if rmse <= cfg.rmse_max:
            stage[t] = st
            W = width_norm[t, :n_real][valid] * norm_stats['width_std'] + norm_stats['width_mean']
            Wpos = W[W > 0]
            width[t] = float(np.mean(Wpos)) if Wpos.size else np.nan
    return stage, width, np.isfinite(stage)


def spline_stage_width(
    wse_norm, width_norm, node_mask, overpass_mask, node_id, norm_stats,
    cfg: InversionConfig,
):
    """
    Spline & spatial-Z0 stage extraction. Per overpass, fit a smoothing spline
    through the node WSE profile (reject on RMSE / monotonicity), evaluate at all
    node positions. Reference each node to its own minimum-stage value (the bed
    datum Z0(x)); the reach stage rise is the median over nodes of the rise above
    that datum. Under uniform flow the median per-node flow law reduces exactly to
    a reach flow law on this median rise (y^a monotonic => median commutes).

    Returns (stage(ob,), width(ob,), good(ob,) bool) where `stage` is the rise above
    the per-node bed (>= 0), or None if extraction fails.
    """
    n_real = int(np.sum(node_id > 0))
    nt = wse_norm.shape[0]
    if n_real < max(4, cfg.min_nodes):
        return None
    x_all = np.arange(n_real, dtype=np.float64)
    H_sp = np.full((nt, n_real), np.nan)
    width = np.full(nt, np.nan)
    good = np.zeros(nt, bool)

    for t in range(nt):
        if not overpass_mask[t]:
            continue
        valid = node_mask[t, :n_real]
        if valid.sum() < max(4, cfg.min_nodes):
            continue
        xv = x_all[valid]
        Hv = wse_norm[t, :n_real][valid] * norm_stats['wse_std'] + norm_stats['wse_mean']
        try:
            spl = UnivariateSpline(xv, Hv, k=3, s=cfg.spline_s * len(xv))
        except Exception:
            continue
        rmse = float(np.sqrt(np.mean((Hv - spl(xv)) ** 2)))
        if rmse > cfg.rmse_max:
            continue
        Heval = spl(x_all)
        d = np.diff(Heval)
        mono = max(np.mean(d >= 0), np.mean(d <= 0)) if len(d) else 1.0
        if mono < cfg.mono_min:
            continue
        H_sp[t] = Heval
        good[t] = True
        W = width_norm[t, :n_real][valid] * norm_stats['width_std'] + norm_stats['width_mean']
        Wpos = W[W > 0]
        width[t] = float(np.mean(Wpos)) if Wpos.size else np.nan

    if good.sum() < cfg.min_good_overpass:
        return None
    # spatial bed datum Z0(x): the splined surface at the single lowest flow overpass
    # (uniform flow => bed parallel to the low-flow water surface). Referencing every
    # node to one overpass keeps the stage rise consistent (=0 at the low-flow time).
    reach_stage = np.where(good, np.nanmedian(H_sp, axis=1), np.inf)
    t_low = int(np.argmin(reach_stage))
    H_ref = H_sp[t_low]
    stage = np.full(nt, np.nan)
    for t in np.where(good)[0]:
        stage[t] = float(np.median(H_sp[t] - H_ref)) # median rise above the bed profile
    return stage, width, good


def estimate_r(dH_all: np.ndarray, logW_all: np.ndarray, d0: float, cfg: InversionConfig):
    """Geometric 1/r = TLS slope of log W vs log depth, bounded; returns (a, r, inv_r)."""
    ld = np.log(dH_all + d0)
    if np.std(ld) < 1e-6 or np.std(logW_all) < 1e-6:
        return np.nan, np.nan, np.nan
    inv_r = float(np.clip(np.std(logW_all) / np.std(ld), cfg.inv_r_min, cfg.inv_r_max))
    return A_FLOOR + inv_r, 1.0 / inv_r, inv_r


def invert_reach(
    wse_norm, width_norm, node_mask, overpass_mask, node_id,
    overpass_time_s, monthly_q, norm_stats,
    eval_overpass_idx: np.ndarray | None = None,
    cfg: InversionConfig = InversionConfig(),
) -> ReachResult:
    """Run the full per-reach inversion. monthly_q: (12,) climatology."""
    used_spline = False
    out = None
    if cfg.use_spline:
        out = spline_stage_width(
            wse_norm, width_norm, node_mask, overpass_mask, node_id, norm_stats, cfg)
        used_spline = out is not None
    if out is None:
        # fall back to the robust line where the spline gate rejects the reach
        # (spline accuracy on clean reaches, line coverage on the rest).
        out = denoise_stage_width(
            wse_norm, width_norm, node_mask, overpass_mask, node_id, norm_stats, cfg)
    stage, width, good = out

    months = months_of(overpass_time_s)
    Qp = np.where((months >= 0) & (months < 12), monthly_q[months], np.nan)
    cal = good & np.isfinite(Qp) & (Qp > 0)
    if cal.sum() < cfg.min_good_overpass or np.ptp(stage[np.where(cal)[0]]) < 1e-3:
        return ReachResult(ok=False, reason="too few/flat calibration overpasses")

    Hmin = np.nanmin(stage[cal])

    # monthly aggregation for the level fit
    cal_months = months[cal]
    cal_dH = stage[cal] - Hmin
    cal_Qp = Qp[cal]
    uniq = np.unique(cal_months)
    if len(uniq) < cfg.min_months:
        return ReachResult(ok=False, reason="too few distinct months")
    dH_m = np.array([cal_dH[cal_months == m].mean() for m in uniq])
    logQp_m = np.log(np.array([cal_Qp[cal_months == m][0] for m in uniq]))

    # width-height geometry across all valid overpasses
    geo = cal & np.isfinite(width)
    if geo.sum() < cfg.min_geom_overpass or np.nanstd(width[geo]) < 1e-6:
        return ReachResult(ok=False, reason="insufficient width-height geometry")
    dH_geo = stage[geo] - Hmin
    logW_geo = np.log(np.maximum(width[geo], 1e-3))

    # search baseflow depth d0; exponent from geometry, level to prior.
    best = None
    for log_d0 in np.linspace(np.log(cfg.d0_grid_lo), np.log(cfg.d0_grid_hi), cfg.d0_grid_n):
        d0 = float(np.exp(log_d0))
        a, r_shape, inv_r = estimate_r(dH_geo, logW_geo, d0, cfg)
        if not np.isfinite(a):
            continue
        ld_m = np.log(dH_m + d0)
        C = float(np.mean(logQp_m) - a * np.mean(ld_m))
        resid = float(np.sum((a * ld_m + C - logQp_m) ** 2))
        if best is None or resid < best[0]:
            best = (resid, d0, a, r_shape, C)
    if best is None:
        return ReachResult(ok=False, reason="calibration failed")
    resid, d0, a, r_shape, C = best

    # Width coefficient kappa in W = kappa * y^(1/r), evaluated at the winning d0. The
    # centroid intercept is the one consistent with the std-ratio (TLS) slope estimate_r
    # uses; W0 is then the width at baseflow depth.
    inv_r = 1.0 / r_shape
    log_kappa = float(np.mean(logW_geo) - inv_r * np.mean(np.log(dH_geo + d0)))
    W0 = float(np.exp(log_kappa) * d0 ** inv_r)

    # invert Q at requested overpasses (default: all good)
    if eval_overpass_idx is None:
        eval_overpass_idx = np.where(good)[0]
    ok_eval = good[eval_overpass_idx]
    idx = np.asarray(eval_overpass_idx)[ok_eval]
    if len(idx) < cfg.min_eval_pairs:
        return ReachResult(ok=False, reason="too few evaluable overpasses")
    dH = np.maximum(stage[idx] - Hmin, 0.0)
    q = np.maximum(np.exp(a * np.log(dH + d0) + C), 1e-6)

    return ReachResult(
        ok=True, q=q, overpass_idx=idx, r_shape=r_shape, d0=d0, C=C,
        W0=W0, Hmin=float(Hmin),
        cal_resid=np.sqrt(resid / max(len(dH_m), 1)), n_used=int(len(idx)),
        used_spline=used_spline,
    )


@dataclass(frozen=True)
class EffectiveCrossSection:
    """Channel geometry implied by a Stage-1 fit.

    These are *effective* quantities, not surveyed ones. `W0`, `A0`, `bed_elevation`
    and `n_eff` all inherit the monthly prior's level through `C`, so any bias in the
    prior climatology transfers straight into them. Only `r` is data-driven.
    """
    r: float
    d0: float # baseflow depth (m)
    W0: float # width at d0 (m)
    A0: float # flow area below d0 (m^2)
    bed_elevation: float # in the WSE datum (m)
    n_eff: float # effective Manning n


def effective_cross_section(res, slope: float) -> EffectiveCrossSection:
    """Effective cross-section implied by a `ReachResult` or `SADnmResult`.

    `n_eff` inverts the Stage-1 level constant, which absorbs roughness and slope
    jointly as `exp(C) = W0 * (r/(r+1))^(5/3) * S^(1/2) / (n * d0^(1/r))`. SADnm never
    uses slope, so `slope` (reach water-surface slope, m/m) must come from the caller's
    own observations; `n_eff` is NaN without a positive one.
    """
    r, d0, C, W0 = res.r_shape, res.d0, res.C, res.W0
    shape = r / (r + 1.0)
    sqrt_s = np.sqrt(slope) if slope > 0 else np.nan
    return EffectiveCrossSection(
        r=r,
        d0=d0,
        W0=W0,
        A0=shape * W0 * d0,
        bed_elevation=res.Hmin - d0,
        n_eff=float(W0 * shape ** (5.0 / 3.0) * sqrt_s / (np.exp(C) * d0 ** (1.0 / r))),
    )
