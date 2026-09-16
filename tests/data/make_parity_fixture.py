"""Regenerate `parity_fixture.npz` from a full preprocessed SWOT zarr store.

The parity test needs real SWOT observations, but the production store is ~4 GB and
cannot live in the repo. This carves a small deterministic slice of it: the first N
validation reaches by sorted `reach_idx` that have at least `min_pairs` same-day gauge
matches. Selection is by sort order, never by whether a reach inverts well.

Each reach is trimmed from its padded bucket shape (e.g. 256x128) down to its valid
overpass/node extent (e.g. 57x51). That is lossless for `invert_reach` -- the masks are
carried along -- and is what keeps the fixture around 1 MB rather than ~100 MB.

Requires `zarr`; not needed to run the test suite.

    python tests/data/make_parity_fixture.py --store /path/to/store.zarr
"""
import argparse
from pathlib import Path

import numpy as np


def build(store_path: Path, n_reaches: int, min_pairs: int) -> dict:
    import zarr

    s = zarr.open(str(store_path), mode='r')
    obs = s['obs']
    val_g = np.array(s['gauge']['val_reach_idx'])
    ov = np.array(s['gauge']['val_pairs']['overpass_idx'])
    q = np.array(s['gauge']['val_pairs']['q'])
    pm = np.array(s['gauge']['val_pairs']['mask'])

    pos = {}
    for bkey in obs.keys():
        for li, gp in enumerate(np.array(obs[bkey]['reach_idx'])):
            pos[int(gp)] = (bkey, int(li))

    selected = []
    for i in np.argsort(val_g):
        gpos = int(val_g[i])
        if gpos in pos and pm[i].astype(bool).sum() >= min_pairs:
            selected.append(int(i))
        if len(selected) >= n_reaches:
            break

    out = {}
    for n, i in enumerate(selected):
        gpos = int(val_g[i])
        bkey, li = pos[gpos]
        bg = obs[bkey]

        omask = np.array(bg['overpass_mask'][li]).astype(bool)
        nmask = np.array(bg['node_mask'][li])
        keep_nodes = nmask[omask].any(axis=0)
        sl = np.ix_(omask, keep_nodes)
        pmask = pm[i].astype(bool)

        out[f'{n}_wse'] = np.clip(np.array(bg['wse_norm'][li]), -10, 10)[sl].astype(np.float32)
        out[f'{n}_wid'] = np.clip(np.array(bg['width_norm'][li]), -10, 10)[sl].astype(np.float32)
        out[f'{n}_nmask'] = nmask[sl]
        out[f'{n}_omask'] = np.ones(int(omask.sum()), bool)
        out[f'{n}_nid'] = np.array(bg['node_id'][li])[keep_nodes]
        out[f'{n}_time'] = np.array(bg['overpass_time_s'][li])[omask]
        out[f'{n}_mq'] = np.array(s['prior']['monthly_q'][gpos]).astype(np.float32)
        out[f'{n}_ns'] = np.array([
            s['norm']['wse_mean'][gpos], s['norm']['wse_std'][gpos],
            s['norm']['width_mean'][gpos], s['norm']['width_std'][gpos]], np.float64)
        # gauge overpass indices refer to the padded frame; remap into the trimmed one
        out[f'{n}_oidx'] = (np.cumsum(omask) - 1)[ov[i][pmask]].astype(np.int32)
        out[f'{n}_gq'] = np.maximum(q[i][pmask], 1e-6).astype(np.float32)

    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store', type=Path, default=Path('data/store.zarr'))
    p.add_argument('--out', type=Path, default=Path(__file__).parent / 'parity_fixture.npz')
    p.add_argument('--n-reaches', type=int, default=40)
    p.add_argument('--min-pairs', type=int, default=5)
    args = p.parse_args()

    arrays = build(args.store, args.n_reaches, args.min_pairs)
    np.savez_compressed(args.out, **arrays)
    n = len({k.split('_')[0] for k in arrays})
    print(f"wrote {args.out} - {n} reaches, {args.out.stat().st_size / 1e6:.2f} MB")


if __name__ == '__main__':
    main()
