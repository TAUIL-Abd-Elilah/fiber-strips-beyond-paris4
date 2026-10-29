"""PREREG v3.1 erratum: two checks on fiber geometry and the frozen plans only (no strip is made, opened or scored).

    python tools/erratum_geometry_checks.py OUT.json

1. Crossing rule (gap 2, PREREG 7). ``find_crossings`` queries, per H fiber sample, only the nearest V fiber sample
   (k = 1) within the tolerance. Here every H/V pair whose closest approach is within the tolerance is found
   exactly (all V samples within the tolerance of every H sample), on the PHerc0332 fibers at 6 sv (the plan) and
   3 sv (the tight subset and ``score --crossings``), and compared with the k = 1 rule: pairs missed, and pairs whose
   recorded closest approach is larger than the true one.
2. Piece tube (gap 6). infer.py's tube input holds every fiber point inside the decision's crop. A piece holds the
   fiber from 300 sv before its first head to 160 sv after its last; the observed path (256 sv behind) and the
   prediction window lie inside it, so the only input that can differ from a full run's (apart from the kink repair
   of PREREG 4.6) is the tube: fiber points outside the piece that come back within the crop. For every planned head
   of every run-set piece, the fiber samples outside the piece (on the full controlled span) within the crop's
   bounding sphere are counted: radius = crop reach (58.05 trace voxels = 116.1 sv: depth 144, width 128, behind
   72, spacing 0.5 of the checkpoint) + the tube margin (3 tube_sigma spacing = 1.5 trace voxels = 3 sv). A sphere
   holds the crop box, so zero means the tube is the full run's.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
SCORE = HERE.parent
sys.path.insert(0, str(SCORE))
import score_strips as S  # noqa: E402

CROP = dict(depth=144, width=128, behind=72, spacing=0.5)  # checkpoint config.json "fine" (trace voxels)
TUBE_SIGMA = 1.0
PREDICTION_SV = 32.  # the checkpoint's prediction window, 16 trace voxels: infer.py's crop axis looks this far ahead
REACH_TRACE = math.sqrt((max(CROP['behind'], CROP['depth']-CROP['behind'])*CROP['spacing'])**2
                        + 2*((CROP['width']-1)/2*CROP['spacing'])**2)+CROP['spacing']
RADIUS_SV = (REACH_TRACE+3*TUBE_SIGMA*CROP['spacing'])*2.
SETS = [('primary', 'results/matched_stretches.json', 'data/fibers0139', 'PHerc0139/20260102150214@L0',
         'data/full_run_map_0139.npz'),
        ('secondary', 'results/matched_stretches_secondary_run40.json', 'data/fibers0139',
         'PHerc0139/20260102150214@L0', 'data/full_run_map_0139.npz'),
        ('0332', 'results/crossings_0332_plan.json', 'data/fibers0332', 'PHerc0332/20251211183505@L0',
         'data/full_run_map_0332.npz')]


def exact_pairs(fibers, tol):
    """Every H/V pair (tagged fibers) with closest approach <= tol, and that approach (sample to sample)."""
    H = [f for f in fibers if f.family == 'H' and f.family_source != 'geometry']
    V = [f for f in fibers if f.family == 'V' and f.family_source != 'geometry']
    trees = {f.name: cKDTree(f.points) for f in V}
    vbox = {f.name: (f.points.min(0)-tol, f.points.max(0)+tol) for f in V}
    out = {}
    for fh in H:
        lo, hi = fh.points.min(0), fh.points.max(0)
        for fv in V:
            blo, bhi = vbox[fv.name]
            if np.any(hi < blo) or np.any(lo > bhi):
                continue
            d, _ = trees[fv.name].query(fh.points, k=1, distance_upper_bound=tol)
            if np.isfinite(d).any():
                out[(fh.name, fv.name)] = float(d[np.isfinite(d)].min())
    return out


def crossing_rule(fibers, tol):
    k1 = {(c['h_fiber'], c['v_fiber']): c['closest_approach_sv'] for c in S.find_crossings(fibers, tol)}
    ex = exact_pairs(fibers, tol)
    missed = sorted(set(ex)-set(k1))
    extra = sorted(set(k1)-set(ex))
    larger = sorted(k for k in set(ex) & set(k1) if k1[k] > ex[k]+1e-9)
    return dict(tol_sv=tol, pairs_exact=len(ex), pairs_k1=len(k1), missed_by_k1=len(missed),
                missed=[dict(h=h, v=v, closest_approach_sv=round(ex[(h, v)], 3)) for h, v in missed],
                found_by_k1_not_exact=len(extra),
                k1_approach_larger=[dict(h=h, v=v, k1_sv=round(k1[(h, v)], 3), exact_sv=round(ex[(h, v)], 3))
                                    for h, v in larger])


def piece_tube(set_name, plan_path, fiber_dir, space, map_path):
    """Per run-set piece and planned head: fiber samples outside the piece (a) within the crop's bounding sphere
    (RADIUS_SV; a superset of the crop for any crop orientation) and (b) within the crop box's reach about the crop
    axis approximated by the fiber itself (the head to the fiber point PREDICTION_SV ahead, as infer.py's crop axis
    follows the fiber ahead): |along| <= 72 + 3 sv and radial <= 63.5 sqrt(2) + 3 sv (any rotation about the axis).
    Heads "scoring" are those within BLEND_SV (full-run arclength) of one of the piece's windows: the decisions
    stitched into rows a window can score."""
    P = json.loads((SCORE/plan_path).read_text())
    windows = {w['id']: w for w in P.get('stretches', P.get('windows', []))}
    pieces = [p for p in P['pieces'] if p.get('run', True)]
    names = {p['fiber'] for p in pieces}
    fibers, _ = S.load_fibers(SCORE/fiber_dir, space=space, names=names)
    by_name = {f.name: f for f in fibers}
    maps = S.load_full_run_maps(SCORE/map_path, fibers)
    half_along = CROP['behind']*CROP['spacing']*2.+3*TUBE_SIGMA*CROP['spacing']*2.
    half_radial = (CROP['width']-1)/2*CROP['spacing']*2.*math.sqrt(2)+3*TUBE_SIGMA*CROP['spacing']*2.
    rows, tot = [], dict(heads=0, scoring=0, sphere=0, sphere_scoring=0, box=0, box_scoring=0)
    for p in pieces:
        f, m = by_name[p['fiber']], maps[p['fiber']]
        k0, k1 = p['head_k']
        Sh = m.head_full(np.arange(k0, k1+1))
        hs = m.to_raw(Sh)
        idx = np.clip(np.searchsorted(f.s, hs), 0, len(f.s)-1)
        heads = f.points[idx]
        ahead = f.points[np.clip(np.searchsorted(f.s, hs+PREDICTION_SV), 0, len(f.s)-1)]
        axis = ahead-heads
        axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9)
        scoring = np.zeros(len(hs), bool)
        for wid in p['windows']:
            w = windows[wid]
            A0, A1 = m.to_full(w['s0']), m.to_full(w['s1'])
            scoring |= (Sh > A0-S.BLEND_SV) & (Sh < A1+S.BLEND_SV)
        a, b = p['parent_arc_sv']
        outside = (f.s < a-1e-9) | (f.s > b+1e-9)
        sphere = np.zeros(len(hs), bool)
        box = np.zeros(len(hs), bool)
        nearest = None
        if outside.any():
            op = f.points[outside]
            tree = cKDTree(op)
            d, _ = tree.query(heads, k=1)
            nearest = float(d.min())
            for i, h in enumerate(tree.query_ball_point(heads, RADIUS_SV)):
                if not h:
                    continue
                sphere[i] = True
                rel = op[h]-heads[i]
                along = rel@axis[i]
                radial = np.linalg.norm(rel-along[:, None]*axis[i], axis=1)
                box[i] = bool(np.any((np.abs(along) <= half_along) & (radial <= half_radial)))
        tot['heads'] += len(hs)
        tot['scoring'] += int(scoring.sum())
        tot['sphere'] += int(sphere.sum())
        tot['sphere_scoring'] += int((sphere & scoring).sum())
        tot['box'] += int(box.sum())
        tot['box_scoring'] += int((box & scoring).sum())
        rows.append(dict(piece=p['id'], fiber=p['fiber'], heads=len(hs), heads_scoring=int(scoring.sum()),
                         outside_samples=int(outside.sum()), heads_sphere=int(sphere.sum()),
                         heads_sphere_scoring=int((sphere & scoring).sum()), heads_box=int(box.sum()),
                         heads_box_scoring=int((box & scoring).sum()),
                         nearest_outside_point_to_a_head_sv=nearest))
    hit = [r for r in rows if r['heads_sphere']]
    return dict(set=set_name, pieces=len(rows), totals=tot, pieces_sphere=len(hit),
                pieces_box_scoring=sum(r['heads_box_scoring'] > 0 for r in rows), affected=hit,
                box_half_along_sv=half_along, box_half_radial_sv=half_radial,
                min_nearest_outside_sv=min((r['nearest_outside_point_to_a_head_sv'] for r in rows
                                            if r['nearest_outside_point_to_a_head_sv'] is not None), default=None))


def main(out):
    fibers0332, _ = S.load_fibers(SCORE/'data'/'fibers0332', space='PHerc0332/20251211183505@L0')
    cr = {f'{tol:g}': crossing_rule(fibers0332, tol) for tol in (S.CROSS_TOL_0332, S.CROSS_TOL)}
    plan = json.loads((SCORE/'results'/'crossings_0332_plan.json').read_text())
    cr['plan_crossings_found'] = plan['plan']['crossings_found']
    tube = [piece_tube(*s) for s in SETS]
    rep = dict(tool='tools/erratum_geometry_checks.py', scorer_sha256=S.sha256_file(SCORE/'score_strips.py'),
               scope='fiber geometry and the frozen plans only; no strip', crossing_rule_0332=cr,
               piece_tube=dict(radius_sv=RADIUS_SV, crop_reach_trace=REACH_TRACE, crop=CROP, tube_sigma=TUBE_SIGMA,
                               sets=tube))
    Path(out).write_text(json.dumps(S.jsonable(rep), indent=1))
    brief = dict(crossing_rule_0332={k: ({kk: vv for kk, vv in v.items() if kk not in ('missed',)}
                                         if isinstance(v, dict) else v) for k, v in cr.items()},
                 piece_tube=[{k: v for k, v in t.items() if k != 'affected'} | dict(
                     affected=[(r['piece'], r['heads'], r['heads_scoring'], r['heads_sphere'],
                                r['heads_sphere_scoring'], r['heads_box'], r['heads_box_scoring'],
                                round(r['nearest_outside_point_to_a_head_sv'], 1)) for r in t['affected']])
                     for t in tube], radius_sv=RADIUS_SV)
    print(json.dumps(S.jsonable(brief), indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
