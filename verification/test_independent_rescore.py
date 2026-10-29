"""Tests of verification/independent_rescore.py on synthetic data and on the Paris 4 fixture (PREREG 10.7 item 5).
They read no PHerc0139 or PHerc0332 strip and no scorer report.

    C:/Users/PC/miniconda3/envs/vesuvius/python.exe -m pytest verification/test_independent_rescore.py -q
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import independent_rescore as R  # noqa: E402

FIX = REPO/'score'/'tests'/'fixtures'/'paris4_real'
PARIS4_REF = REPO/'score'/'results'/'erratum_v31'/'paris4_piece_check.json'
AXIS = R.Axis(np.array([[0., 200., -1000.], [0., 200., 5000.]]))   # a vertical umbilicus at x = 0, y = 200 (sv)


# ---- synthetic helpers -----------------------------------------------------------------------------------------

def grid_from(origin, du, dv, rows, cols):
    """Nodes origin + i du + j dv (sv)."""
    i, j = np.mgrid[0:rows, 0:cols]
    return np.asarray(origin, float) + i[..., None]*np.asarray(du, float) + j[..., None]*np.asarray(dv, float)


def wall(rows=41, cols=81, x=500.):
    """A vertical plane x = const: rows along z, columns along y, 5 sv apart, from y = z = 0."""
    return grid_from((x, 0., 0.), (0., 0., 5.), (0., 5., 0.), rows, cols)


def write_tifxyz(folder, grid, valid=None, l0_per_unit=4.0, trace=None):
    """A tifxyz folder in file units (l0_per_unit base voxels per unit: strips 4, segments 1); invalid nodes -1."""
    import tifffile
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    g = np.asarray(grid, float)*(R.L0_PER_SV/l0_per_unit)
    for k, c in enumerate('xyz'):
        a = g[..., k].astype(np.float32)
        if valid is not None:
            a[~valid] = -1
        tifffile.imwrite(str(folder/f'{c}.tif'), a)
    (folder/'meta.json').write_text(json.dumps(dict(format='tifxyz', type='seg')))
    if trace is not None:
        (folder/'trace.json').write_text(json.dumps(trace))
    return folder


def memory_opener(surfaces):
    return lambda name, pts: (R.BilinearSurface(*surfaces[name], name), None) if name in surfaces else \
        (None, f'{name} unknown')


def straight_fiber(name, a, b, family='H'):
    return R.Fiber(name, family, *R.sample_polyline(np.array([a, b], float)))


def score_one(tmp_path, strip, strip_valid, fiber, stretches, seg_grid, seg_valid=None, axis=AXIS, piece='P0001'):
    sdir = write_tifxyz(tmp_path/piece/'strip', strip, strip_valid)
    seg_valid = np.ones(seg_grid.shape[:2], bool) if seg_valid is None else seg_valid
    return R.score_pieces('test', [dict(piece=piece, fiber=fiber.name, strip_dir=R.strip_tifxyz_dir(sdir.parent))],
                          {fiber.name: fiber}, stretches, memory_opener({'wall': (seg_grid, seg_valid)}), axis)


def offset_strip(h, rows=60, cols=25):
    """A strip parallel to the wall at x = 500 + h: rows along y every 4 sv from y = 20, columns along z
    every 5 sv centred on z = 100."""
    return grid_from((500. + h, 20., 100. - 5.*(cols//2)), (0., 4., 0.), (0., 0., 5.), rows, cols)


# ---- independence ----------------------------------------------------------------------------------------------

def test_does_not_import_the_scorer():
    src = Path(R.__file__).read_text(encoding='utf-8')
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or '').split('.')[0])
    assert imported <= {'__future__', 'argparse', 'dataclasses', 'hashlib', 'itertools', 'json', 'math', 'os', 're',
                        'sys', 'time', 'pathlib', 'numpy', 'scipy', 'tifffile'}, imported
    assert 'importlib' not in src and '__import__' not in src and 'exec(' not in src
    # in a fresh interpreter (the pytest session may hold score_strips from score/tests)
    code = ('import sys; sys.path.insert(0, sys.argv[1]); import independent_rescore; '
            'print("score_strips" in sys.modules)')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2')
    run = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(HERE)], capture_output=True, text=True, env=env,
                         timeout=300)
    assert run.returncode == 0 and run.stdout.strip() == 'False', run.stderr
    assert set(R.readings()) == {f'R{i}' for i in range(1, 13)}
    assert '1e-3 sv' in R.tolerance_text() and 'agree_within_solver_bounds' in R.tolerance_text()
    src.encode('ascii')          # the docstring and messages print on any Windows console


# ---- fibers: span and sampling (4.1) -----------------------------------------------------------------------------

def test_sampling_every_sv_plus_the_end():
    pts, s = R.sample_polyline(np.array([[0., 0., 0.], [10.5, 0., 0.]]))
    assert s.tolist() == [float(i) for i in range(11)] + [10.5]
    np.testing.assert_allclose(pts[:, 0], s)
    pts, s = R.sample_polyline(np.array([[0., 0., 0.], [4., 0., 0.], [4., 3., 0.]]))     # length exactly 7
    assert s.tolist() == [float(i) for i in range(8)]
    np.testing.assert_allclose(pts[5], [4., 1., 0.])
    assert pts[-1].tolist() == [4., 3., 0.]


def test_fiber_json_controlled_span_and_family(tmp_path):
    line = [[4.*i, 0., 8.] for i in range(30)]             # level 0: one sv between line points
    raw = dict(line_points=line, control_points=[dict(position=line[17]), dict(position=[13., 0.1, 8.]),
                                                 dict(position=line[5])],
               hv_classification=dict(manual_tag='', automatic_tag='V'))
    (tmp_path/'kb_x.json').write_text(json.dumps(raw))
    fib, why = R.read_fiber(tmp_path/'kb_x.json')
    assert why is None and fib.name == 'kb_x' and fib.family == 'V'
    assert fib.line_range == (3, 17)                       # the control at x = 13 anchors at line point 3 (x = 12)
    assert fib.length == 14. and len(fib.s) == 15
    np.testing.assert_allclose(fib.pts[0], [3., 0., 2.])
    raw['hv_classification'] = dict(manual_tag='H', automatic_tag='V')
    raw['control_points'] = raw['control_points'][:1]
    (tmp_path/'kb_y.json').write_text(json.dumps(raw))
    fib, why = R.read_fiber(tmp_path/'kb_y.json')
    assert fib is None and 'controlled span' in why
    fibers, skipped = R.load_fiber_dir(tmp_path)
    assert list(fibers) == ['kb_x'] and list(skipped) == ['kb_y']


# ---- the distance (5.1) ------------------------------------------------------------------------------------------

def test_plane_signed_distance_reach_and_sign():
    g = wall()
    surf = R.BilinearSurface(g, np.ones(g.shape[:2], bool))
    pts = np.array([[501.5, 120.3, 77.7], [498.2, 33.3, 150.1], [500., 200., 100.], [530., 50., 50.],
                    [530.01, 50., 50.], [469.5, 50., 50.]])
    pr = surf.project(pts, AXIS)
    assert pr.found.all()
    np.testing.assert_allclose(pr.d, [1.5, -1.8, 0., 30., 30.01, -30.5], atol=1e-9)    # outward (+x) positive
    np.testing.assert_allclose(pr.side, 0., atol=1e-9)
    assert pr.covered().tolist() == [True, True, True, True, False, False]           # |d| <= 30 is in (R3)
    raw = R.BilinearSurface(g, np.ones(g.shape[:2], bool)).project(pts)               # no umbilicus: one fixed sign
    np.testing.assert_allclose(np.abs(raw.d), np.abs(pr.d), atol=1e-12)
    assert len(set(np.sign(raw.d[[0, 1]]*pr.d[[0, 1]]))) == 1


def test_edges_holes_and_the_sideways_residual():
    g = wall()                                             # y in [0, 400], z in [0, 200]
    valid = np.ones(g.shape[:2], bool)
    valid[20, 26] = False                                  # node (z 100, y 130): its 4 cells (y 125..135, z 95..105)
    surf = R.BilinearSurface(g, valid)
    pts = np.array([[501., 400.9, 100.],                   # 0.9 sv beyond the y = 400 edge: covered
                    [501., 401.1, 100.],                   # 1.1 sv beyond: off the mesh
                    [501., 130., 100.],                    # over the hole's centre: 5 sv sideways
                    [501., 134.5, 100.],                   # 0.5 sv inside the hole's edge: covered
                    [501., 136., 100.],                    # beside the hole
                    [501., 401., 100.]])                   # exactly 1 sv beyond the edge: covered (R3)
    pr = surf.project(pts, AXIS)
    np.testing.assert_allclose(pr.side, [0.9, 1.1, 5., 0.5, 0., 1.], atol=1e-9)
    assert pr.side[5] == 1.
    np.testing.assert_allclose(pr.d, 1., atol=1e-9)
    assert pr.covered().tolist() == [True, False, False, True, True, True]
    assert R.uncovered_reason(pr).tolist() == [0, 2, 2, 0, 0, 0]
    assert pr.boundary.tolist() == [True, True, True, True, False, True]           # feet on the edge of the mesh


def test_tilted_plane_matches_the_analytic_distance():
    rng = np.random.default_rng(1)
    n = np.array([1., 2., 2.])/3.
    u = np.array([2., -1., 0.])/math.sqrt(5.)
    v = np.cross(n, u)
    g = grid_from((1000., 1000., 1000.), 5*u, 5*v, 30, 30)
    surf = R.BilinearSurface(g, np.ones((30, 30), bool))
    ab = rng.uniform(20., 120., size=(300, 2))
    h = rng.uniform(-25., 25., size=300)
    pr = surf.project(1000. + ab[:, :1]*u + ab[:, 1:]*v + h[:, None]*n)
    assert pr.covered().all()
    np.testing.assert_allclose(np.abs(pr.d), np.abs(h), atol=1e-9)
    np.testing.assert_allclose(pr.side, 0., atol=1e-8)
    assert len(set(np.sign(pr.d*h))) == 1


@pytest.mark.parametrize('reverse', [False, True])
def test_cylinder_outward_sign_and_facet_sagitta(reverse):
    radius = 200.
    th = np.arange(0., 1.0, 5./radius)
    th = th[::-1] if reverse else th                       # the raw normal flips; outward stays positive
    z = np.arange(0., 205., 5.)
    g = np.stack([radius*np.cos(th)[None, :].repeat(len(z), 0), radius*np.sin(th)[None, :].repeat(len(z), 0),
                  z[:, None].repeat(len(th), 1)], -1)
    axis = R.Axis(np.array([[0., 0., -100.], [0., 0., 400.]]))
    surf = R.BilinearSurface(g, np.ones(g.shape[:2], bool))
    ang = np.linspace(0.1, 0.9, 17)
    sag = radius*(1. - math.cos(2.5/radius))               # chord sagitta of a 5 sv cell, 0.0156 sv
    for rho, sign in ((201.5, 1.), (198.5, -1.)):
        pts = np.stack([rho*np.cos(ang), rho*np.sin(ang), np.full_like(ang, 100.)], -1)
        pr = surf.project(pts, axis)
        assert pr.covered().all()
        assert np.all(np.sign(pr.d) == sign)
        assert np.all(np.abs(pr.d) >= 1.5 - sag - 1e-9) and np.all(np.abs(pr.d) <= 1.5 + sag + 1e-9)


def folded_plane(fold_deg):
    """z = 0 for y <= 50, falling at ``fold_deg`` beyond: a crease along y = 50, convex from above."""
    t = math.radians(fold_deg)
    y = np.arange(0., 101., 5.)
    g = np.zeros((21, len(y), 3))
    g[..., 0] = np.arange(21)[:, None]*5.
    g[..., 1] = np.where(y <= 50., y, 50. + (y - 50.)*math.cos(t))[None, :]
    g[..., 2] = np.where(y <= 50., 0., -(y - 50.)*math.sin(t))[None, :]
    return R.BilinearSurface(g, np.ones(g.shape[:2], bool))


def test_crease_keeps_every_normal_and_uses_the_best_fitting_one():
    """A point above the 20 deg crease inside its normal cone has its foot on the crease; d and the sideways
    residual over both normals are kept, and the normal that fits p - foot best gives d (R7)."""
    t = math.radians(20.)
    surf = folded_plane(20.)
    for r, phi in ((10., 4.), (10., 16.), (1., 16.), (3., 16.)):        # phi from the flat side's normal
        phi = math.radians(phi)
        pr = surf.project(np.array([[50., 50. + r*math.sin(phi), r*math.cos(phi)]]))
        assert pr.found[0] and pr.crease[0] and not pr.boundary[0]
        np.testing.assert_allclose(pr.foot[0], [50., 50., 0.], atol=1e-9)
        lo_hi = sorted([r*math.cos(t - phi), r*math.cos(phi)])
        np.testing.assert_allclose(sorted(np.abs([pr.d_lo[0], pr.d_hi[0]])), lo_hi, atol=1e-9)
        side = sorted([r*math.sin(phi), r*math.sin(t - phi)])
        np.testing.assert_allclose([pr.side_lo[0], pr.side_hi[0]], side, atol=1e-9)
        best = min(phi, t - phi)
        assert abs(pr.d[0]) == pytest.approx(r*math.cos(best)) and pr.side[0] == pytest.approx(r*math.sin(best))
        c = R.classify(pr)
        # r = 1: sideways <= 0.28 sv for both normals, but |d| differs by 0.036 sv between them: not robust;
        # r = 3: the best normal leaves 0.21 sv sideways, the other 0.83 sv: not robust
        assert c['covered'][0] and not c['robust'][0] and c['sensitive'][0]
    # a 9.7 deg fold, 5.9 sv off, 4.8 deg from one normal and 4.9 from the other: |d| varies by 0.00087 sv only,
    # but one normal leaves 0.504 sv sideways: not robust (every normal at the foot counts)
    r, phi = 5.9, math.radians(4.8)
    pr = folded_plane(9.7).project(np.array([[50., 50. + r*math.sin(phi), r*math.cos(phi)]]))
    assert pr.side_lo[0] < 0.5 < pr.side_hi[0] and abs(pr.d_hi[0] - pr.d_lo[0]) < 1e-3 and abs(pr.d[0]) < 6.
    c = R.classify(pr)
    assert c['covered'][0] and not c['robust'][0] and c['sensitive'][0]


def test_an_inner_crease_foot_is_sensitive_whatever_its_residual():
    """On a 60 deg fold a point 30 deg inside the cone is 5 sv sideways of both normals at 10 sv: uncovered here,
    but its foot is inside the mesh, so another solver may cover it (TOLERANCE AND VERDICTS)."""
    phi, r = math.radians(30.), 10.
    pr = folded_plane(60.).project(np.array([[50., 50. + r*math.sin(phi), r*math.cos(phi)]]))
    assert pr.crease[0] and not pr.boundary[0] and pr.side_lo[0] == pytest.approx(5.)
    assert pr.side_lo[0] > max(R.SENSITIVE_SIDE_SV, R.SENSITIVE_SIDE_REL*pr.dist[0])
    c = R.classify(pr)
    assert not c['covered'][0] and c['sensitive'][0]


def test_segment_crop_gives_the_full_grid_projection(tmp_path):
    g = wall(rows=60, cols=100)
    valid = np.ones(g.shape[:2], bool)
    valid[30:33, 40:44] = False
    write_tifxyz(tmp_path/'seg', g, valid, l0_per_unit=1.0)
    rng = np.random.default_rng(2)
    pts = np.stack([500. + rng.uniform(-35., 35., 400), rng.uniform(180., 260., 400), rng.uniform(120., 180., 400)],
                   -1)
    full = R.BilinearSurface(*R.read_tifxyz(tmp_path/'seg', 1.0)).project(pts, AXIS)
    surf, note = R.segment_opener({'seg': tmp_path/'seg'}, 1.0)('seg', pts)
    assert note is None and surf.shape[0] < 60 and surf.shape[1] < 100
    crop = surf.project(pts, AXIS)
    assert np.array_equal(full.covered(), crop.covered()) and full.covered().sum() > 100
    np.testing.assert_allclose(crop.d, full.d, atol=1e-12)
    np.testing.assert_allclose(crop.side, full.side, atol=1e-12)


def test_umbilicus_interpolation_check_counts_sign_changes():
    """R8's check: between two control points the linear umbilicus and the nearest control point can lie on either
    side of a sheet; the outward sign then differs and is counted."""
    axis = R.Axis(np.array([[0., 200., 0.], [1000., 200., 1000.]]))   # at z = 100: linear x = 100, nearest x = 0
    np.testing.assert_allclose(axis.xy(np.array([100.])), [[100., 200.]])
    np.testing.assert_allclose(axis.xy_nearest(np.array([100., 900., -5., 2000.])),
                               [[0., 200.], [1000., 200.], [0., 200.], [1000., 200.]])
    g = wall(x=50.)
    pts = np.array([[51., 150., 100.], [49., 150., 90.]])
    pr = R.BilinearSurface(g, np.ones(g.shape[:2], bool)).project(pts, axis)
    np.testing.assert_allclose(pr.d, [-1., 1.], atol=1e-9)        # outward is -x: the umbilicus is at x 90..100
    assert pr.sign_alt.tolist() == [True, True]
    diag = R.solver_diagnostics(pr, R.classify(pr))
    assert diag['outward_sign_differs_with_nearest_control_point'] == 2


# ---- tier 2: solver sensitivity ----------------------------------------------------------------------------------

def test_classes_robust_sensitive_and_firm():
    g = wall()
    surf = R.BilinearSurface(g, np.ones(g.shape[:2], bool))
    pts = np.array([[501., 100., 100.],      # near and inside: robust
                    [506.5, 100., 100.],     # beyond 6 sv: sensitive
                    [501., 400.7, 100.],     # covered, 0.7 sv beyond the edge: sensitive (sideways > 0.5 sv)
                    [501., 401.4, 100.],     # uncovered, 1.4 sv beyond the edge: sensitive (another solver may cover)
                    [501., 403., 100.],      # 3 sv beyond the edge at 1 sv: firmly uncovered
                    [520., 405., 100.],      # 5 sv beyond the edge at 20 sv: 5 <= 0.35 x 20.6, sensitive
                    [530.5, 100., 100.],     # |d| 30.5: sqrt(30.5^2 - 1) - (0.1 + 0.02 x 30.5) <= 30: sensitive
                    [534., 100., 100.]])     # |d| 34: firmly uncovered
    pr = surf.project(pts, AXIS)
    c = R.classify(pr)
    assert c['robust'].tolist() == [True] + [False]*7
    assert c['sensitive'].tolist() == [False, True, True, True, False, True, True, False]
    assert c['covered'].tolist() == [True, True, True, False, False, False, False, False]
    dist = pr.dist[1]
    assert c['lo'][1] == pytest.approx(math.sqrt(dist**2 - 1.) - 0.1 - 0.02*dist)
    assert c['hi'][1] == pytest.approx(dist + 0.1 + 0.1*dist)
    assert c['dlo'][1] == c['lo'][1] and c['dhi'][1] == c['hi'][1]            # outward side: d = +|d|
    ing = R.tier2_ingredients(c, quantiles=True)
    assert ing['n_robust'] == 1 and len(ing['lo']) == 5 and ing['robust_abs'].tolist() == [1.]
    assert list(ing['lo']) == sorted(ing['lo']) and list(ing['hi']) == sorted(ing['hi'], reverse=True)


def test_interval_arithmetic():
    ing = dict(n_robust=2, robust_abs_sum=3., robust_signed_sum=-1., robust_abs=np.array([1., 2.]),
               lo=np.array([5., 7., 10.]), hi=np.array([20., 12., 8.]), dlo=np.array([-20., 5., 7.]),
               dhi=np.array([12., 8., -5.]))
    e = 2*R.ROBUST_SLACK_SV + R.ROBUST_SLACK_REL*3.
    assert R.count_interval(ing) == (2, 5)
    lo, hi = R.mean_interval(ing, 4)                      # 2 robust + the 2 smallest lows / the 2 largest highs
    assert lo == pytest.approx((3. - e + 12.)/4) and hi == pytest.approx((3. + e + 32.)/4)
    lo, hi = R.mean_interval(ing, 3, signed=True)
    assert lo == pytest.approx((-1. - e - 20.)/3) and hi == pytest.approx((-1. + e + 12.)/3)
    assert R.mean_interval(ing, 6) is None and R.mean_interval(ing, 1) is None
    lo, hi = R.mean_interval_any_count(ing)
    assert lo == pytest.approx((3. - e)/2) and hi == pytest.approx((3. + e + 32.)/4)    # the largest is at k = 2
    lo, hi = R.quantile_interval(ing, 1, 50)
    assert lo == pytest.approx(np.median([1. - R.robust_slack(1.), 2. - R.robust_slack(2.), 5.]))
    assert hi == pytest.approx(np.median([1. + R.robust_slack(1.), 2. + R.robust_slack(2.), 20.]))
    assert R.within_interval(ing, 2, 2.) == (1 + 0, 2 + 0)
    assert R.within_interval(ing, 3, 9.) == (2 + 1, 2 + 2)      # all 3 covered: the one with hi 8 is within 9
    lo, hi = R.median_interval_any_count(ing, 1)
    assert lo == pytest.approx(R.quantile_interval(ing, 1, 50)[0])
    assert hi == pytest.approx(max(R.quantile_interval(ing, k, 50)[1] for k in (1, 2, 3)))


def tilted_strip(h=1.5, tilt_deg=20., rows=60, cols=25):
    """offset_strip with its columns tilted about the row line: the side columns lie up to 60 sin(tilt) sv off."""
    t = math.radians(tilt_deg)
    g = offset_strip(h, rows, cols)
    v = (np.arange(cols) - cols//2)*5.
    g[..., 0] = 500. + h + v[None, :]*math.sin(t)
    g[..., 2] = 100. + v[None, :]*math.cos(t)
    return g


def test_piece_tier2_reads_a_solver_difference_and_still_flags_a_defect(tmp_path):
    fiber = straight_fiber('f', (500.7, 0., 100.), (500.7, 300., 100.))
    st = dict(id='S0001', fiber='f', segment='wall', s0=100., s1=160., piece='P0001')
    res = score_one(tmp_path, tilted_strip(), None, fiber, [st], wall())
    p = R.plain(res['pieces']['P0001'])
    diag = p['solver_sensitivity']['diagnostics']
    assert p['strip_nodes'] == 400 and diag['robust'] == 16*7 and diag['sensitive_covered'] == 400 - 16*7
    assert diag['covered_by_abs_d_sv']['12-30'] > 0 and diag['crease_spread_over_0p001_sv'] == 0
    d, cls = res['node_d']['P0001'], res['node_cls']['P0001']
    drop = np.argsort(-np.where(cls['sensitive'] & cls['covered'], cls['abs_d'], -1.))[:9]  # another solver drops 9
    keep = np.ones(len(cls['d']), bool)
    keep[drop] = False
    kd = cls['d'][keep & cls['covered']]
    entry = dict(strip='P0001', stretches=['S0001'], assoc_median_sv=p['assoc_median_sv'], strip_nodes=int(len(kd)),
                 strip_mean_abs_sv=float(np.abs(kd).mean()), strip_signed_mean_sv=float(kd.mean()),
                 fiber_floor_mean_abs_sv=p['fiber_floor_mean_abs_sv'])
    it = R.compare_piece('test', 'P0001', p, [entry], dict(per_strip=[entry]), [])
    assert it['verdict'] == 'agree_within_solver_bounds' and it['ok'] and not it['agree']
    assert not it['fields']['strip_nodes']['agree'] and it['fields']['strip_nodes']['bounds']['within']
    b = it['fields']['strip_mean_abs_sv']['bounds']
    assert b['lo'] <= entry['strip_mean_abs_sv'] <= b['hi'] and b['width'] < 2.
    for bad in (dict(strip_mean_abs_sv=entry['strip_mean_abs_sv'] + 3.),          # a defect beyond the bounds
                dict(strip_signed_mean_sv=-entry['strip_signed_mean_sv']),       # the sign flipped
                dict(strip_nodes=100),                                           # robust nodes lost
                dict(assoc_median_sv=p['assoc_median_sv'] + 0.01)):              # no solver in it: tier 1 only
        e2 = dict(entry, **bad)
        assert R.compare_piece('test', 'P0001', p, [e2], dict(per_strip=[e2]), [])['verdict'] == 'differs', bad


# ---- rows and nodes (5.2) ----------------------------------------------------------------------------------------

def test_strip_at_a_known_offset(tmp_path):
    fiber = straight_fiber('f', (500.7, 0., 100.), (500.7, 300., 100.))
    st = dict(id='S0001', fiber='f', segment='wall', s0=100., s1=160., piece='P0001', samples=61, mean_d=0.7,
              median_abs_d=0.7, p90_abs_d=0.7, max_abs_d=0.7)
    res = score_one(tmp_path, offset_strip(1.5), None, fiber, [st], wall())
    p, s = res['pieces']['P0001'], res['stretches']['S0001']
    assert p['rows_scored'] == 16 and p['strip_nodes'] == 400                  # rows at y = 100, 104, ..., 160
    assert p['strip_mean_abs_sv'] == pytest.approx(1.5, abs=1e-9)
    assert p['strip_signed_mean_sv'] == pytest.approx(1.5, abs=1e-9)
    assert p['strip_median_abs_sv'] == pytest.approx(1.5) and p['strip_p90_abs_sv'] == pytest.approx(1.5)
    assert p['strip_within1'] == 0. and p['strip_within2'] == 1.
    assert p['assoc_median_sv'] == pytest.approx(0.8, abs=1e-9) and not p['rejected']
    assert p['fiber_floor_rows'] == 16 and p['fiber_floor_mean_abs_sv'] == pytest.approx(0.7, abs=1e-9)
    assert p['stretches'] == ['S0001'] and s['rows_scored'] == 16 and s['strip_nodes'] == 400
    assert p['solver_sensitivity']['diagnostics']['robust'] == 400
    gt = s['ground_truth_check']
    assert gt['samples'] == 61 and gt['covered'] == 61 and gt['agree'] and gt['verdict'] == 'agree'
    assert gt['mean_d_sv'] == pytest.approx(0.7, abs=1e-9)


def test_node_statistics_on_distinct_offsets(tmp_path):
    """Per-node offsets 0.05 .. 2.5 sv (1.0 and 2.0 among them, exact in float32) over the flat wall: mean,
    median, p90, the within-1 / within-2 counts (inclusive) and the max, against numpy on the same offsets."""
    fiber = straight_fiber('f', (500.7, 0., 100.), (500.7, 300., 100.))
    st = dict(id='S0001', fiber='f', segment='wall', s0=100., s1=160., piece='P0001')
    strip = offset_strip(0.)
    h = 0.05*(1 + (np.arange(60*25).reshape(60, 25)*7) % 50)
    strip[..., 0] = 500. + h
    res = score_one(tmp_path, strip, None, fiber, [st], wall())
    p = res['pieces']['P0001']
    h32 = strip[..., 0].astype(np.float32).astype(np.float64) - 500.                 # as the tif stores them
    hh = h32[20:36].ravel()                                # the scored rows: y = 100 .. 160
    assert p['strip_nodes'] == 400 and (hh == 1.).sum() > 0 and (hh == 2.).sum() > 0
    assert p['strip_mean_abs_sv'] == pytest.approx(hh.mean(), abs=1e-9)
    assert p['strip_median_abs_sv'] == pytest.approx(np.median(hh), abs=1e-9)
    assert p['strip_p90_abs_sv'] == pytest.approx(np.percentile(hh, 90), abs=1e-9)
    assert np.percentile(hh, 90) - np.percentile(hh, 80) > 0.1
    assert p['strip_within1_nodes'] == int((hh <= 1.).sum()) > int((hh < 1.).sum())
    assert p['strip_within2_nodes'] == int((hh <= 2.).sum()) > int((hh < 2.).sum())
    assert p['strip_max_abs_sv'] == pytest.approx(hh.max(), abs=1e-9)


def test_invalid_nodes_and_the_centre_fallback(tmp_path):
    fiber = straight_fiber('f', (500.7, 0., 100.), (500.7, 300., 100.))
    st = dict(id='S0001', fiber='f', segment='wall', s0=100., s1=160., piece='P0001')
    valid = np.ones((60, 25), bool)
    valid[22, 12:16] = False                               # the centre node of row 22: centre = mean of 21 nodes
    valid[25] = False                                      # no valid node: no centre, not scored
    res = score_one(tmp_path, offset_strip(1.5), valid, fiber, [st], wall())
    p = res['pieces']['P0001']
    assert p['centre_fallback_rows'] == 1 and p['rows_with_centre'] == 59
    assert p['rows_scored'] == 15 and p['strip_nodes'] == 400 - 4 - 25
    assert p['strip_mean_abs_sv'] == pytest.approx(1.5, abs=1e-9)
    prep = R.prepare_rows(offset_strip(1.5), valid, fiber, [st])
    zc = 40. + 5.*np.mean([c for c in range(25) if not 12 <= c < 16])
    np.testing.assert_allclose(prep['centre'][22], [501.5, 108., zc])
    assert prep['dist'][22] == pytest.approx(math.hypot(0.8, 100. - zc))


def test_segment_holes_leave_nodes_uncovered(tmp_path):
    fiber = straight_fiber('f', (500., 0., 100.), (500., 300., 100.))
    st = dict(id='S0001', fiber='f', segment='wall', s0=100., s1=160., piece='P0001')
    seg_valid = np.ones((41, 81), bool)
    seg_valid[20, 26] = False                              # a 10 x 10 sv hole at y 125..135, z 95..105
    res = score_one(tmp_path, offset_strip(1.0), None, fiber, [st], wall(), seg_valid)
    p = res['pieces']['P0001']
    # nodes over the hole and more than 1 sv inside its edge: row y = 128, 132 at z = 100 (column 12): 2 nodes
    assert p['uncovered'] == dict(no_foot=0, sideways_over_1sv=2, beyond_30sv=0)
    assert p['strip_nodes'] == 400 - 2


def test_fiber_floor_follows_the_row_sample_and_its_coverage(tmp_path):
    """A fiber tilted off the wall (d = 0.01 y) over a hole at y 105..115: the floor is |d| of the sample at each
    scored row's arclength, averaged over the rows whose sample is covered (5.6, R9)."""
    fiber = straight_fiber('f', (500., 0., 100.), (503., 300., 100.))
    st = dict(id='S0001', fiber='f', segment='wall', s0=100., s1=160., piece='P0001')
    seg_valid = np.ones((41, 81), bool)
    seg_valid[20, 22] = False                              # the 4 cells around node (z 100, y 110)
    res = score_one(tmp_path, offset_strip(1.5), None, fiber, [st], wall(), seg_valid)
    p = res['pieces']['P0001']
    k = np.arange(100, 161, 4)                             # each row's nearest sample is s = y (= k)
    cov = (fiber.pts[k, 1] <= 106.) | (fiber.pts[k, 1] >= 114.)
    assert cov.sum() == len(k) - 2
    assert p['rows_scored'] == len(k) and p['fiber_floor_rows'] == len(k) - 2
    expected = np.abs(fiber.pts[k[cov], 0] - 500.).mean()
    assert p['fiber_floor_mean_abs_sv'] == pytest.approx(expected, abs=1e-9)
    assert abs(expected - np.abs(fiber.pts[k, 0] - 500.).mean()) > 0.01         # coverage matters here
    assert abs(expected - np.abs(fiber.pts[k[cov] + 3, 0] - 500.).mean()) > 0.01  # so does the sample


def test_the_30sv_row_window_and_the_30sv_reach(tmp_path):
    fiber = straight_fiber('f', (500., 0., 100.), (500., 300., 100.))
    st = dict(id='S0001', fiber='f', segment='wall', s0=0., s1=300., piece='P0001')
    strip = offset_strip(1.0)
    strip[30, :, 0] = 530.                                 # centre exactly 30 sv from the fiber: scored (R3)
    strip[31, :, 0] = 530. + 1e-4                          # just beyond: not scored
    strip[32, :, 0] = 529.999
    strip[33, :, 0] = 529.5 + 0.1*(np.arange(25) - 12)     # centre 29.5 sv; columns 18..24 beyond |d| = 30
    res = score_one(tmp_path, strip, None, fiber, [st], wall())
    p = res['pieces']['P0001']
    assert p['rows_in_stretches'] == 60 and p['rows_in_stretches_beyond_30sv'] == 1
    assert p['rows_scored'] == 59 and p['nodes_on_scored_rows'] == 59*25
    assert p['uncovered'] == dict(no_foot=0, sideways_over_1sv=0, beyond_30sv=7)
    assert p['strip_nodes'] == 59*25 - 7
    assert p['strip_max_abs_sv'] == pytest.approx(30., abs=1e-4)


def test_stretch_bounds_are_inclusive_and_rows_go_to_their_own_stretch(tmp_path):
    fiber = straight_fiber('f', (500., 0., 100.), (500., 300., 100.))
    sts = [dict(id='S0001', fiber='f', segment='wall', s0=100., s1=140., piece='P0001'),
           dict(id='S0002', fiber='f', segment='wall', s0=141., s1=200., piece='P0001'),
           dict(id='S0003', fiber='f', segment='wall', s0=240., s1=250., piece='P0002')]
    res = score_one(tmp_path, offset_strip(1.0), None, fiber, sts, wall())
    p, st = res['pieces']['P0001'], res['stretches']
    assert st['S0001']['rows_scored'] == 11                 # y = 100 .. 140: both bounds in
    assert st['S0002']['rows_scored'] == 15                 # y = 144 .. 200
    assert p['rows_in_other_pieces_stretches'] == 3         # y = 240, 244, 248 lie in P0002's stretch
    assert st['S0003']['rows_from_other_pieces']['P0001']['rows_scored'] == 3
    assert p['stretches'] == ['S0001', 'S0002', 'S0003'] and p['strip_nodes'] == 29*25
    alt = p['own_windows_only']                             # R6's other reading, reported beside it
    assert alt['rows_scored'] == 26 and alt['stretches'] == ['S0001', 'S0002'] and alt['strip_nodes'] == 26*25
    # a scorer that reads R6 the other way: not a defect, the item agrees under the other reading
    entry = dict(strip='P0001', stretches=['S0001', 'S0002'], assoc_median_sv=p['assoc_median_sv'],
                 strip_nodes=alt['strip_nodes'], strip_mean_abs_sv=alt['strip_mean_abs_sv'],
                 strip_signed_mean_sv=alt['strip_signed_mean_sv'], fiber_floor_mean_abs_sv=alt['fiber_floor_mean_abs_sv'])
    it = R.compare_piece('test', 'P0001', R.plain(p), [entry], dict(per_strip=[entry]), [])
    assert it['verdict'] == 'agree_under_other_reading' and it['reading'] == 'R6' and it['ok']
    assert it['own_windows_only_reading']['verdict'] == 'agree'
    assert not it['fields']['strip_nodes']['agree'] and not it['fields']['strip_nodes']['bounds']['within']
    assert it['fields']['strip_mean_abs_sv']['agree']
    # the scorer skips a strip whose rows lie only in another piece's stretches: also the other reading
    p2 = dict(R.plain(p), own_windows_only=dict(alt, rows_scored=0))
    it = R.compare_piece('test', 'P0001', p2, [], dict(per_strip=[], skips=[]), [])
    assert it['verdict'] == 'agree_under_other_reading'


@pytest.mark.parametrize('h, rejected', [(5.5, True), (5.0, False), (4.9, False)])
def test_association_rejects_a_strip_off_its_fiber(tmp_path, h, rejected):
    fiber = straight_fiber('f', (500., 0., 100.), (500., 300., 100.))
    st = dict(id='S0001', fiber='f', segment='wall', s0=100., s1=160., piece='P0001')
    res = score_one(tmp_path, offset_strip(h), None, fiber, [st], wall())
    p = res['pieces']['P0001']
    assert p['rejected'] is rejected and p['assoc_median_sv'] == pytest.approx(h, abs=1e-4)   # float32 strip
    if h == 5.0:
        assert p['assoc_median_sv'] == 5.                  # exactly the bound: in (R3)
    assert ('strip_nodes' not in p) if rejected else (p['strip_nodes'] == 400)
    pool = R.pooled(res['pieces'], res['node_d'])
    assert pool['scored_fibers'] == (0 if rejected else 1)


def test_association_median_is_over_every_row_with_a_centre(tmp_path):
    """Rows 0-9 lie 40 sv off the fiber, rows 10-59 at 0.5 .. 2.95 sv, the stretch holds rows 30-59 (R5): the median
    over all 60 rows (1.975) differs from the one over rows within 30 sv (1.725) or in the stretch (2.225)."""
    fiber = straight_fiber('f', (500., 0., 100.), (500., 300., 100.))
    st = dict(id='S0001', fiber='f', segment='wall', s0=140., s1=256., piece='P0001')
    strip = offset_strip(0.)
    hr = np.r_[np.full(10, 40.), 0.5 + 0.05*np.arange(50)]
    strip[..., 0] = 500. + hr[:, None]
    res = score_one(tmp_path, strip, None, fiber, [st], wall())
    p = res['pieces']['P0001']
    assert p['assoc_median_sv'] == pytest.approx(np.median(hr), abs=1e-4) == pytest.approx(1.975, abs=1e-4)
    assert np.median(hr[hr <= 30.]) == pytest.approx(1.725) and np.median(hr[30:]) == pytest.approx(2.225)
    assert p['rows_scored'] == 30 and not p['rejected']


def test_a_strip_with_no_row_in_a_window_has_no_entry_on_either_side(tmp_path):
    """The strip's rows (y 20 .. 96) end before the window (100 .. 160): no row is scored. The scorer then writes
    no per_strip entry ('no strip row inside a matched stretch'), which agrees; a scored piece without an entry
    differs."""
    fiber = straight_fiber('f', (500.7, 0., 100.), (500.7, 300., 100.))
    st = dict(id='S0001', fiber='f', segment='wall', s0=100., s1=160., piece='P0001')
    p = R.plain(score_one(tmp_path/'a', offset_strip(1.5, rows=20), None, fiber, [st], wall())['pieces']['P0001'])
    assert p['rows_scored'] == 0 and p['strip_nodes'] == 0 and not p['rejected']
    skips = [dict(strip='P0001', fiber='f', reason='no strip row inside a matched stretch')]
    it = R.compare_piece('test', 'P0001', p, [], dict(per_strip=[], skips=skips), [])
    assert it['verdict'] == 'agree' and it['scorer_skips'] == skips and it['mine_status'] == 'no scored row'
    entry = dict(strip='P0001', stretches=[], assoc_median_sv=p['assoc_median_sv'], strip_nodes=0,
                 strip_mean_abs_sv=None, strip_signed_mean_sv=None, fiber_floor_mean_abs_sv=None)
    assert R.compare_piece('test', 'P0001', p, [entry], dict(per_strip=[entry]), [])['verdict'] == 'agree'
    scored = R.plain(score_one(tmp_path/'b', offset_strip(1.5), None, fiber, [st], wall())['pieces']['P0001'])
    assert R.compare_piece('test', 'P0001', scored, [], dict(per_strip=[], skips=[]), [])['verdict'] == 'differs'


def test_pooled_numbers_keep_only_fibers_with_25_covered_nodes():
    pieces = {'P1': dict(fiber='a'), 'P2': dict(fiber='b'), 'P3': dict(fiber='b')}
    node_d = {'P1': np.full(24, 1.), 'P2': np.full(20, -3.), 'P3': np.full(5, 0.5)}
    pool = R.pooled(pieces, node_d)
    assert pool['scored_fibers'] == 1 and pool['scored_fiber_names'] == ['b'] and pool['strip']['nodes'] == 25
    assert pool['strip']['mean_abs_sv'] == pytest.approx((20*3 + 5*0.5)/25)
    assert pool['strip']['signed_mean_sv'] == pytest.approx((-60 + 2.5)/25)
    assert pool['strip']['within1_nodes'] == 5 and pool['strip']['within2_nodes'] == 5


def test_pooled_comparison_with_no_scored_fiber():
    """No primary fiber reaches 25 covered nodes: the scorer writes point None, which agrees with 0 nodes here."""
    mine = R.plain(R.pooled({'P0': dict(fiber='a'), 'P1': dict(fiber='b')},
                            {'P0': np.full(10, 1.), 'P1': np.full(3, .5)}))
    theirs = dict(accuracy=dict(all=dict(scored_fibers=0, strip=dict(point=None, ci=None, fibers=0))))
    it = R.compare_pooled(mine, theirs)
    assert it['verdict'] == 'agree', it['fields']
    it = R.compare_pooled(mine, dict(accuracy=dict(all=dict(scored_fibers=1, strip=dict(point=dict(
        nodes=26, mean=1., median=1., p90=1., within1=1., within2=1.))))))
    assert it['verdict'] == 'differs'
    scored = R.plain(R.pooled({'P0': dict(fiber='a')}, {'P0': np.full(30, 1.)}))
    assert R.compare_pooled(scored, theirs)['verdict'] == 'differs'


def test_pooled_tier2_keeps_the_scorers_fiber_count():
    """Two fibers, each 20 robust nodes at 1 sv and 10 sensitive ones at about 10 sv: each may reach 25 covered
    nodes. A scorer that reports 1 scored fiber with 50 nodes cannot be explained (one fiber has at most 30)."""
    def t2():
        return dict(n_robust=20, robust_abs_sum=20., robust_signed_sum=20., robust_abs=[1.]*20, lo=[9.]*10,
                    hi=[11.]*10, dlo=[9.]*10, dhi=[11.]*10)
    pool = dict(fibers=2, scored_fibers=2, tier2_per_fiber=dict(a=t2(), b=t2()), strip=dict(
        nodes=60, mean_abs_sv=4., median_abs_sv=1., p90_abs_sv=10., within1_nodes=40, within2_nodes=40))
    point = dict(nodes=50, mean=2.8, median=1., p90=10., within1=0.8, within2=0.8)
    it = R.compare_pooled(pool, dict(accuracy=dict(all=dict(scored_fibers=1, strip=dict(point=point)))))
    assert it['verdict'] == 'differs' and not it['fields']['nodes']['bounds']['within']
    it = R.compare_pooled(pool, dict(accuracy=dict(all=dict(scored_fibers=2, strip=dict(point=point)))))
    assert it['verdict'] == 'agree_within_solver_bounds' and it['tier2_fibers'] == ['a', 'b']


def test_pooled_tier2_with_the_25_node_rule(tmp_path):
    fiber = straight_fiber('f', (500.7, 0., 100.), (500.7, 300., 100.))
    st = dict(id='S0001', fiber='f', segment='wall', s0=100., s1=160., piece='P0001')
    res = score_one(tmp_path, tilted_strip(), None, fiber, [st], wall())
    pool = R.plain(R.pooled(res['pieces'], res['node_d'], res['node_cls']))
    t2 = pool['tier2_per_fiber']['f']
    assert t2['n_robust'] == 112 and len(t2['lo']) == 288 and len(t2['robust_abs']) == 112
    cls = res['node_cls']['P0001']

    def point_without(drop):
        keep = cls['covered'].copy()
        keep[drop] = False
        a = cls['abs_d'][keep]
        return dict(nodes=int(keep.sum()), mean=float(a.mean()), median=float(np.median(a)),
                    p90=float(np.percentile(a, 90)), within1=float(np.mean(a <= 1.)), within2=float(np.mean(a <= 2.)))

    def verdict(point):
        return R.compare_pooled(pool, dict(accuracy=dict(all=dict(scored_fibers=1, strip=dict(point=point)))))
    far = np.argsort(-np.where(cls['sensitive'], cls['abs_d'], -1.))
    point = point_without(far[:9])                       # another solver loses the 9 farthest nodes
    it = verdict(point)
    assert it['verdict'] == 'agree_within_solver_bounds', {k: v for k, v in it['fields'].items() if not v['agree']}
    assert it['tier2_fibers'] == ['f'] and it['fields']['median']['bounds']['width'] < 2.
    for bad in (dict(median=point['median'] + 2.), dict(p90=point['p90'] - 3.), dict(mean=point['mean'] * 1.5),
                dict(within2=point['within2'] / 2.)):
        assert verdict(dict(point, **bad))['verdict'] == 'differs', bad
    it = verdict(point_without(far[:128]))               # 128 of 288 lost: the subset is nearly free, the interval
    assert it['verdict'] == 'agree_within_solver_bounds' and it['fields']['median']['bounds']['width'] > 5.  # wide


# ---- crossings (7) -----------------------------------------------------------------------------------------------

def test_crossing_point_follows_the_k1_rule():
    h = straight_fiber('h', (0., 0., 0.), (100., 0., 0.))
    v = straight_fiber('v', (50., 3., -50.), (50., 3., 50.), 'V')
    v2 = straight_fiber('v2', (52., 2., -50.), (52., 2., 50.), 'V')
    vt = R.v_sample_index({'h': h, 'v': v, 'v2': v2})
    cp = R.crossing_point(h, v, vt)
    # the H sample at x = 50 has v2 nearest (2.83 < 3), so the pair (h, v) takes x = 49 at sqrt(10)
    assert cp['h_s'] == 49. and cp['v_s'] == 50.
    assert cp['closest_approach_sv'] == pytest.approx(math.sqrt(10.))
    np.testing.assert_allclose(cp['point_sv'], [49.5, 1.5, 0.])
    cp2 = R.crossing_point(h, v2, vt)
    assert cp2['closest_approach_sv'] == pytest.approx(2.) and cp2['h_s'] == 52.
    far = straight_fiber('far', (50., 20., -50.), (50., 20., 50.), 'V')
    cp3 = R.crossing_point(h, far, R.v_sample_index({'v': v, 'far': far}))
    assert cp3['rule'].startswith('pairwise') and cp3['closest_approach_sv'] == pytest.approx(20.)


def test_crossing_agreement_of_two_strips_at_a_known_separation():
    hs = grid_from((0., -60., 0.), (4., 0., 0.), (0., 5., 0.), 26, 25)       # plane z = 0
    vs = grid_from((-10., -50., 2.), (0., 4., 0.), (5., 0., 0.), 26, 25)     # plane z = 2
    ok_h, ok_v = np.ones(hs.shape[:2], bool), np.ones(vs.shape[:2], bool)
    ag = R.crossing_agreement(hs, ok_h, vs, ok_v, [50., 0., 1.])
    assert ag['n_directions'] == 2 and ag['agreement_sv'] == pytest.approx(2., abs=1e-12)
    assert all(d['usable'] and d['nodes_within_30sv'] >= 5 and d['covered'] == d['nodes_within_30sv']
               for d in ag['directions'])
    assert ag['tier2']['possible_directions'] == [2]


def test_crossing_agreement_is_the_mean_of_two_unequal_medians():
    """Perpendicular strips through the point (50, 0, 0): H in z = 0 (rows x = 0, 5, ..., 100), V in x = 50 (rows
    z = -57.5, -52.5, ...). H nodes project onto V at |x - 50|, V nodes onto H at |z|."""
    hs = grid_from((0., -60., 0.), (5., 0., 0.), (0., 5., 0.), 21, 25)
    vs = grid_from((50., -60., -57.5), (0., 0., 5.), (0., 5., 0.), 24, 25)
    ok_h, ok_v = np.ones(hs.shape[:2], bool), np.ones(vs.shape[:2], bool)
    p = np.array([50., 0., 0.])
    hn, vn = hs.reshape(-1, 3), vs.reshape(-1, 3)
    m1 = np.median(np.abs(hn[np.linalg.norm(hn - p, axis=1) <= 30., 0] - 50.))
    m2 = np.median(np.abs(vn[np.linalg.norm(vn - p, axis=1) <= 30., 2]))
    assert abs(m1 - m2) > 0.5
    ag = R.crossing_agreement(hs, ok_h, vs, ok_v, p)
    h2v, v2h = ag['directions']
    assert h2v['median_abs_d_sv'] == pytest.approx(m1, abs=1e-9)
    assert v2h['median_abs_d_sv'] == pytest.approx(m2, abs=1e-9)
    assert ag['n_directions'] == 2 and ag['agreement_sv'] == pytest.approx((m1 + m2)/2., abs=1e-9)


def test_crossing_five_node_rule_and_the_30sv_radius():
    vs = grid_from((-10., -50., 2.), (0., 4., 0.), (5., 0., 0.), 26, 25)     # plane z = 2, fine grid
    ok_v = np.ones(vs.shape[:2], bool)
    # a coarse H strip, nodes 30 sv apart: exactly 5 nodes within 30 sv of (50, 0, 0), 4 of them at exactly 30 sv
    coarse = grid_from((20., -30., 0.), (30., 0., 0.), (0., 30., 0.), 3, 3)
    ok_c = np.ones((3, 3), bool)
    ag = R.crossing_agreement(coarse, ok_c, vs, ok_v, [50., 0., 0.])
    h2v, v2h = ag['directions']
    assert h2v['nodes_within_30sv'] == 5 and h2v['covered'] == 5 and h2v['usable']
    assert v2h['usable'] and ag['n_directions'] == 2 and ag['agreement_sv'] == pytest.approx(2., abs=1e-9)
    # the V strip cut at x = 40: one of the 5 nodes falls 20 sv off its edge, 4 covered: not usable
    ag = R.crossing_agreement(coarse, ok_c, vs[:, 10:], ok_v[:, 10:], [50., 0., 0.])
    h2v = ag['directions'][0]
    assert h2v['nodes_within_30sv'] == 5 and h2v['covered'] == 4 and not h2v['usable']
    assert 'covered by the other strip' in h2v['reason'] and ag['n_directions'] == 1
    # 4 nodes within 30 sv: not usable
    point = [62.5, 12.5, 1.]
    coarse2 = grid_from((0., -50., 0.), (25., 0., 0.), (0., 25., 0.), 5, 5)  # 25 sv cells
    ag = R.crossing_agreement(coarse2, np.ones((5, 5), bool), vs, ok_v, point)
    h2v, v2h = ag['directions']
    assert h2v['nodes_within_30sv'] == 4 and not h2v['usable'] and 'fewer than 5 nodes' in h2v['reason']
    assert v2h['usable'] and ag['n_directions'] == 1 and ag['agreement_sv'] == pytest.approx(2., abs=1e-9)
    away = coarse2 + np.array([300., 0., 0.])                                # no H node near, V nodes not covered
    ag = R.crossing_agreement(away, np.ones((5, 5), bool), vs, ok_v, point)
    assert ag['n_directions'] == 0 and ag['agreement_sv'] is None
    assert ag['directions'][1]['covered'] == 0 and 'covered by the other strip' in ag['directions'][1]['reason']
    assert ag['tier2']['possible_directions'] == [0]


def crossing_record(ag, point):
    return dict(R.plain(ag), missing=ag['agreement_sv'] is None, recomputed=dict(point_sv=list(point),
                                                                                  closest_approach_sv=1.))


def test_crossing_comparison_missing_status_and_tier2():
    hs = grid_from((0., -60., 0.), (5., 0., 0.), (0., 5., 0.), 21, 25)
    vs = grid_from((50., -60., -57.5), (0., 0., 5.), (0., 5., 0.), 24, 25)
    p = [50., 0., 0.]
    ag = R.crossing_agreement(hs, np.ones(hs.shape[:2], bool), vs, np.ones(vs.shape[:2], bool), p)
    mine = crossing_record(ag, p)
    row = dict(id='X0000', closest_approach_sv=1., point_sv=p, agreement_sv=ag['agreement_sv'], directions=2)

    def verdict(rows, missing=(), m=mine):
        items, _ = R.compare_crossings(dict(per_crossing={'X0000': m}, planned=1),
                                       dict(crossings=dict(planned=1, missing=list(missing), per_crossing=rows)))
        return items[0]
    assert verdict([row])['verdict'] == 'agree'
    it = verdict([dict(row, agreement_sv=ag['agreement_sv'] + 0.05)])        # far nodes: a solver difference
    assert it['verdict'] == 'agree_within_solver_bounds' and it['fields']['agreement_sv']['bounds']['within']
    assert verdict([dict(row, agreement_sv=ag['agreement_sv'] + 40.)])['verdict'] == 'differs'
    assert verdict([dict(row, directions=0)])['verdict'] == 'differs'
    assert verdict([dict(row, point_sv=[51., 0., 0.])])['verdict'] == 'differs'           # tier 1 only
    it = verdict([], missing=['X0000'])                                       # the scorer lists it missing
    assert it['verdict'] == 'differs' and it['fields']['missing'] == dict(mine=False, scorer=True, agree=False,
                                                                           bounds=dict(possible_directions=[2],
                                                                                       within=False))
    assert verdict([dict(row, agreement_sv=None)])['verdict'] == 'differs'
    no_strip = dict(id='X0000', missing=True, missing_reason='no strip for H')
    assert verdict([], missing=['X0000'], m=no_strip)['verdict'] == 'agree'
    assert verdict([row], m=no_strip)['verdict'] == 'differs'


@pytest.mark.parametrize('off, rejected', [(5.0, False), (5.5, True)])
def test_crossing_strip_association_bound_is_inclusive(tmp_path, off, rejected):
    """An H strip exactly 5 sv off its fiber runs along it (5.2, R3); 5.5 sv off it is rejected (R12)."""
    ws = tmp_path/'score'
    _fiber_json(ws/'data'/'fibers0332'/'h1.json', (0., 0., 1000.), (100., 0., 1000.), 'H')
    _fiber_json(ws/'data'/'fibers0332'/'v1.json', (50., -50., 1001.), (50., 50., 1001.), 'V')
    strips = ws/'runs'/'v3'/'strips'/'0332'
    write_tifxyz(strips/'C0000'/'strip', grid_from((0., -60., 1000. + off), (4., 0., 0.), (0., 5., 0.), 26, 25))
    write_tifxyz(strips/'C0001'/'strip', grid_from((-10., -50., 1001.), (0., 4., 0.), (5., 0., 0.), 26, 25))
    plan = dict(crossings=[dict(id='X0000', h_fiber='h1', v_fiber='v1', closest_approach_sv=1.0,
                                point_sv=[50., 0., 1000.5], windows=['X0000H', 'X0000V'], run=True)],
                windows=[dict(id='X0000H', family='H', fiber='h1', piece='C0000'),
                         dict(id='X0000V', family='V', fiber='v1', piece='C0001')],
                pieces=[dict(id='C0000', fiber='h1'), dict(id='C0001', fiber='v1')])
    cr = R.rescore_crossings(ws, dict(crossings=[dict(id='X0000')]), plan, ws/'runs'/'v3'/'strips')
    x = cr['per_crossing']['X0000']
    assert x['association']['H']['assoc_median_sv'] == off and x['association']['H']['rejected'] is rejected
    assert x['missing'] is rejected
    if not rejected:
        assert x['agreement_sv'] == pytest.approx(off - 1., abs=1e-9) and x['n_directions'] == 2


def test_crossing_point_and_strips_end_to_end(tmp_path):
    """rescore_crossings on a synthetic workspace: (a) the H strip runs 8 sv off its fiber: rejected (5.2, R12), the
    crossing is missing; (b) the V fiber has two pieces and the other piece covers the crossing best: R12's other
    reading pairs it, and a scorer that does so agrees under that reading."""
    ws = tmp_path/'score'
    _fiber_json(ws/'data'/'fibers0332'/'h1.json', (0., 0., 1000.), (100., 0., 1000.), 'H')
    _fiber_json(ws/'data'/'fibers0332'/'v1.json', (50., -150., 1001.), (50., 50., 1001.), 'V')
    strips = ws/'runs'/'v3'/'strips'/'0332'
    write_tifxyz(strips/'C0000'/'strip', grid_from((0., -60., 1008.), (4., 0., 0.), (0., 5., 0.), 26, 25))
    write_tifxyz(strips/'C0001'/'strip', grid_from((-10., -150., 1001.), (0., 4., 0.), (5., 0., 0.), 26, 25))
    write_tifxyz(strips/'C0002'/'strip', grid_from((-10., -50., 1001.), (0., 4., 0.), (5., 0., 0.), 26, 25))
    plan = dict(crossings=[dict(id='X0000', h_fiber='h1', v_fiber='v1', closest_approach_sv=1.0,
                                point_sv=[50., 0., 1000.5], windows=['X0000H', 'X0000V'], run=True)],
                windows=[dict(id='X0000H', family='H', fiber='h1', piece='C0000'),
                         dict(id='X0000V', family='V', fiber='v1', piece='C0001')],
                pieces=[dict(id='C0000', fiber='h1'), dict(id='C0001', fiber='v1'), dict(id='C0002', fiber='v1')])
    cr = R.rescore_crossings(ws, dict(crossings=[dict(id='X0000')]), plan, ws/'runs'/'v3'/'strips')
    x = cr['per_crossing']['X0000']
    assert x['missing'] and x['missing_reason'].startswith('strip rejected') and 'tier2' not in x
    assert x['association']['H']['assoc_median_sv'] == pytest.approx(8., abs=1e-4) and x['association']['H']['rejected']
    assert x['association']['V']['assoc_median_sv'] == pytest.approx(0., abs=1e-4)
    assert x['other_reading']['h_piece'] is None and x['other_reading']['missing']     # 8 sv: no strip of h1 there
    items, _ = R.compare_crossings(cr, dict(crossings=dict(planned=1, missing=['X0000'], per_crossing=[])))
    assert items[0]['verdict'] == 'agree'
    # (b) the H strip on its fiber
    write_tifxyz(strips/'C0000'/'strip', grid_from((0., -60., 1000.), (4., 0., 0.), (0., 5., 0.), 26, 25))
    cr = R.rescore_crossings(ws, dict(crossings=[dict(id='X0000')]), plan, ws/'runs'/'v3'/'strips')
    x = cr['per_crossing']['X0000']
    assert not x['association']['H']['rejected'] and x['v_piece'] == 'C0001'
    # the window's piece C0001 ends 50 sv before the crossing: no node within 30 sv, H nodes 20+ sv off its edge
    assert x['missing'] and x['missing_reason'] == 'no usable direction (5-node rule)'
    assert x['tier2']['possible_directions'] == [0]
    o = x['other_reading']
    assert (o['h_piece'], o['v_piece']) == ('C0000', 'C0002') and not o['missing']
    assert o['agreement_sv'] == pytest.approx(1., abs=1e-9) and o['n_directions'] == 2
    rc = x['recomputed']
    row = dict(id='X0000', closest_approach_sv=rc['closest_approach_sv'], point_sv=rc['point_sv'],
               agreement_sv=o['agreement_sv'], directions=o['n_directions'])
    items, _ = R.compare_crossings(R.plain(cr), dict(crossings=dict(planned=1, missing=[], per_crossing=[row])))
    assert items[0]['verdict'] == 'agree_under_other_reading' and items[0]['reading'] == 'R12'
    assert items[0]['other_reading']['verdict'] == 'agree'
    items, _ = R.compare_crossings(R.plain(cr), dict(crossings=dict(planned=1, missing=['X0000'], per_crossing=[])))
    assert items[0]['verdict'] == 'agree'                # a scorer that reads R12 as this file does


# ---- the Paris 4 fixture (PREREG 10.7 item 5) -------------------------------------------------------------------

def paris4_piece_a():
    """Piece A of fiber 778459 (fixture 'pieces/QA') scored against the G0 full-fiber strip as a stand-in segment,
    window 400 to 440 sv, the whole G0 polyline as the fiber, no umbilicus: as tools/paris4_piece_check.py ran it."""
    fj = json.loads((FIX/'fiber_778459.json').read_text())
    fiber = R.Fiber('g0_778459', fj['family'], *R.sample_polyline(np.asarray(fj['xyz_l0'], float)/R.L0_PER_SV))
    st = dict(id='WA', fiber=fiber.name, segment='g0strip_778459', s0=fj['window_sv'][0], s1=fj['window_sv'][1],
              piece='QA')
    res = R.score_pieces('paris4', [dict(piece='QA', fiber=fiber.name, strip_dir=FIX/'pieces'/'QA'/'strip')],
                         {fiber.name: fiber}, [st],
                         R.segment_opener({'g0strip_778459': FIX/'g0_fiber_778459'}, l0_per_unit=4.0))
    return res['pieces']['QA'], R.pooled(res['pieces'], res['node_d'])


@pytest.mark.skipif(not FIX.is_dir() or not PARIS4_REF.is_file(), reason='Paris 4 fixture not present')
def test_paris4_fixture_reproduces_the_erratum_numbers():
    ref = next(r for r in json.loads(PARIS4_REF.read_text())['piece_vs_full'] if r['tag'] == 'A')['score_end_to_end']
    piece, pool = paris4_piece_a()
    assert piece['strip_nodes'] == ref['strip_nodes'] == 150
    assert abs(piece['strip_mean_abs_sv'] - ref['strip_mean_abs_sv']) <= 1e-12          # 4.98e-6 sv
    assert abs(piece['assoc_median_sv'] - ref['assoc_median_sv']) <= 1e-12              # 1.0607 sv (R2, R5)
    agg = ref['aggregate']
    assert pool['scored_fibers'] == agg['scored_fibers'] == 1
    s = pool['strip']
    assert s['nodes'] == agg['strip']['nodes'] == 150
    assert abs(s['mean_abs_sv'] - agg['strip']['mean']) <= 1e-12
    assert s['median_abs_sv'] == agg['strip']['median'] == 0. and s['p90_abs_sv'] == agg['strip']['p90'] == 0.
    assert s['within1'] == agg['strip']['within1'] == 1. and s['within2'] == agg['strip']['within2'] == 1.
    assert piece['rows_scored'] == 6 and piece['rows_with_centre'] == 24 and not piece['notes']
    assert piece['solver_sensitivity']['diagnostics']['robust'] == 150


# ---- the command line on a synthetic workspace -------------------------------------------------------------------

def _fiber_json(path, a, b, family, n=None):
    a, b = np.asarray(a, float)*4., np.asarray(b, float)*4.    # level 0
    n = n or int(np.linalg.norm(b - a)/4.) + 1
    line = [list(map(float, a + (b - a)*t)) for t in np.linspace(0., 1., n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(type='vc3d_fiber', version=4, line_points=line,
                                    control_points=[dict(position=line[0]), dict(position=line[-1])],
                                    hv_classification=dict(manual_tag='', automatic_tag=family))))


def make_workspace(root):
    ws = root/'score'
    write_tifxyz(ws/'data'/'segments'/'segA', wall(), l0_per_unit=1.0)
    (ws/'data'/'umbilicus').mkdir(parents=True)
    (ws/'data'/'umbilicus'/'PHerc0139_20260102150214_umbilicus.json').write_text(json.dumps(dict(control_points=[
        dict(x=1600, y=800, z=-4000), dict(x=1600, y=800, z=20000)])))        # level 0: x = 400, y = 200 sv
    _fiber_json(ws/'data'/'fibers0139'/'kb_f1.json', (500.7, 0., 100.), (500.7, 300., 100.), 'H')
    _fiber_json(ws/'data'/'fibers0332'/'dj_h1.json', (0., 0., 1000.), (100., 0., 1000.), 'H')
    _fiber_json(ws/'data'/'fibers0332'/'dj_v1.json', (50., -50., 1001.), (50., 50., 1001.), 'V')
    raw = json.loads((ws/'data'/'fibers0332'/'dj_v1.json').read_text())
    raw['control_points'] = raw['control_points'][:1]                                  # unusable, like 2 real ones
    (ws/'data'/'fibers0332'/'kb_bad.json').write_text(json.dumps(raw))
    gt = dict(mean_d=0.7, median_abs_d=0.7, p90_abs_d=0.7, max_abs_d=0.7)
    seg = [dict(name='segA', source='data\\segments\\segA')]
    sec = dict(segments=seg, stretches=[
        dict(id='S0000', fiber='kb_f1', family='H', segment='segA', s0=60., s1=90., samples=31, piece='P0000',
             run=True, **gt),
        dict(id='S0001', fiber='kb_f1', family='H', segment='segA', s0=100., s1=160., samples=61, piece='P0000',
             run=True, **gt),
        dict(id='S0002', fiber='kb_f1', family='H', segment='segA', s0=250., s1=290., samples=41, piece='P0001',
             run=False, **gt)],
        pieces=[dict(id='P0000', fiber='kb_f1', windows=['S0000', 'S0001'], run=True),
                dict(id='P0001', fiber='kb_f1', windows=['S0002'], run=False)])
    pri = dict(segments=seg, stretches=[dict(id='S0000', fiber='kb_f1', family='H', segment='segA', s0=100.,
                                             s1=180., samples=81, piece='P0000', run=True, **gt)],
               pieces=[dict(id='P0000', fiber='kb_f1', windows=['S0000'], run=True)])
    cro = dict(crossings=[dict(id='X0000', h_fiber='dj_h1', v_fiber='dj_v1', closest_approach_sv=1.0,
                               point_sv=[50., 0., 1000.5], h_s=50., v_s=50., windows=['X0000H', 'X0000V'], run=True,
                               z_tercile=0)],
               windows=[dict(id='X0000H', crossing='X0000', fiber='dj_h1', family='H', piece='C0000', run=True),
                        dict(id='X0000V', crossing='X0000', fiber='dj_v1', family='V', piece='C0001', run=True)],
               pieces=[dict(id='C0000', fiber='dj_h1', windows=['X0000H']),
                       dict(id='C0001', fiber='dj_v1', windows=['X0000V'])])
    (ws/'results').mkdir(parents=True)
    for rel, obj in ((R.PLAN_FILES['secondary'], sec), (R.PLAN_FILES['primary'], pri), (R.PLAN_FILES['0332'], cro)):
        (ws/rel).write_text(json.dumps(obj))
    strips = ws/'runs'/'v3'/'strips'
    write_tifxyz(strips/'secondary'/'P0000'/'strip', offset_strip(1.5))
    write_tifxyz(strips/'primary'/'P0000'/'strip', offset_strip(-1.0))      # the inner side: d = -1
    write_tifxyz(strips/'0332'/'C0000'/'strip', grid_from((0., -60., 1000.), (4., 0., 0.), (0., 5., 0.), 26, 25))
    write_tifxyz(strips/'0332'/'C0001'/'strip', grid_from((-10., -50., 1001.), (0., 4., 0.), (5., 0., 0.), 26, 25))
    sample = dict(secondary_stretches=[dict(id='S0001', fiber='kb_f1', segment='segA', s0=100., s1=160.,
                                            piece='P0000')],
                  primary_stretches=[dict(id='S0000', fiber='kb_f1', segment='segA', s0=100., s1=180.,
                                          piece='P0000')],
                  crossings=[dict(id='X0000', h_fiber='dj_h1', v_fiber='dj_v1')],
                  sources={rel: hashlib.sha256((ws/rel).read_bytes()).hexdigest() for rel in R.PLAN_FILES.values()})
    (root/'sample.json').write_text(json.dumps(sample))
    return ws, root/'sample.json'


def fake_scorer_reports(ws, rep):
    """Scorer-shaped reports holding this tool's own numbers (the structure of cmd_score / cmd_crossings)."""
    out = ws/'results'/'v3'
    out.mkdir(parents=True, exist_ok=True)
    for set_name in ('primary', 'secondary'):
        per = [dict(strip=pid, fiber=p['fiber'], family=p['family'], stretches=p['stretches'],
                    assoc_median_sv=p['assoc_median_sv'], notes=[], strip_nodes=p['strip_nodes'],
                    strip_mean_abs_sv=p['strip_mean_abs_sv'], strip_signed_mean_sv=p['strip_signed_mean_sv'],
                    fiber_floor_mean_abs_sv=p['fiber_floor_mean_abs_sv'])
               for pid, p in rep[set_name]['pieces'].items()]
        body = dict(provenance=dict(scorer_sha256='x'), skips=[], per_strip=per)
        if set_name == 'primary':
            s = rep['primary']['pooled']['strip']
            body['accuracy'] = dict(all=dict(scored_fibers=rep['primary']['pooled']['scored_fibers'], strip=dict(
                point=dict(nodes=s['nodes'], mean=s['mean_abs_sv'], median=s['median_abs_sv'], p90=s['p90_abs_sv'],
                           within1=s['within1'], within2=s['within2']))))
        (out/R.REPORT_FILES[set_name]).write_text(json.dumps(body))
    cr = rep['crossings_0332']
    per = [dict(id=cid, h_fiber=c['h_fiber'], v_fiber=c['v_fiber'], closest_approach_sv=c['plan']['closest_approach_sv'],
                point_sv=c['recomputed']['point_sv'], agreement_sv=c['agreement_sv'], directions=c['n_directions'])
           for cid, c in cr['per_crossing'].items() if not c['missing']]
    (out/R.REPORT_FILES['0332']).write_text(json.dumps(dict(provenance={}, crossings=dict(
        label='no ground truth', planned=cr['planned'], missing=cr['missing'], per_crossing=per))))
    return out


def test_command_line_recompute_and_compare(tmp_path, capsys):
    ws, sample = make_workspace(tmp_path)
    out = tmp_path/'rescore_report.json'
    assert R.main(['--workspace', str(ws), '--sample', str(sample), '--out', str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep['errors'] == [] and all(p['matches_sample'] for p in rep['inputs']['plans'].values())
    sec = rep['secondary']
    assert list(sec['pieces']) == ['P0000'] and sorted(sec['stretches']) == ['S0000', 'S0001']
    assert sec['stretches']['S0001']['in_sample'] and not sec['stretches']['S0000']['in_sample']
    p = sec['pieces']['P0000']
    assert p['stretches'] == ['S0000', 'S0001'] and p['rows_scored'] == 8 + 16       # y 60..88 and 100..160
    assert p['strip_nodes'] == 24*25 and p['strip_signed_mean_sv'] == pytest.approx(1.5, abs=1e-6)  # outward: +x
    assert p['fiber_floor_mean_abs_sv'] == pytest.approx(0.7, abs=1e-6)
    pri = rep['primary']
    assert pri['pieces']['P0000']['strip_signed_mean_sv'] == pytest.approx(-1., abs=1e-6)  # inward: negative
    assert pri['pooled']['scored_fibers'] == 1 and pri['pooled']['strip']['nodes'] == 21*25
    assert rep['ground_truth_check_summary'] == dict(stretches=3, with_plan_values=3, agree=3,
                                                     agree_within_solver_bounds=0, differs=0)
    x = rep['crossings_0332']['per_crossing']['X0000']
    assert not x['missing'] and x['n_directions'] == 2 and x['agreement_sv'] == pytest.approx(1., abs=1e-9)
    assert x['point_vs_plan_max_abs_sv'] < 1e-9 and x['h_piece'] == 'C0000' and x['v_piece'] == 'C0001'
    assert x['association']['H']['assoc_median_sv'] == pytest.approx(0., abs=1e-4) and 'other_reading' not in x
    assert rep['crossings_0332']['fibers_skipped'] == {'kb_bad': 'no controlled span (fewer than 2 control points '
                                                                 'or line points)'}

    reports = fake_scorer_reports(ws, rep)
    assert R.main(['--workspace', str(ws), '--sample', str(sample), '--out', str(out), '--compare-only']) == 0
    cmp_ = json.loads(out.read_text())['comparison']
    zero = dict(agree_within_solver_bounds=0, agree_under_other_reading=0, differs=0)
    assert cmp_['summary']['all_exact'] and cmp_['summary']['no_item_differs']
    assert cmp_['summary']['by_kind'] == dict(
        ground_truth=dict(items=3, agree=3, **zero), piece=dict(items=2, agree=2, **zero),
        pooled=dict(items=1, agree=1, **zero), crossing=dict(items=1, agree=1, **zero),
        crossing_set=dict(items=1, agree=1, **zero))

    body = json.loads((reports/R.REPORT_FILES['secondary']).read_text())
    body['per_strip'][0]['strip_mean_abs_sv'] += 0.004                 # beyond 1e-3 sv, within a robust allowance
    (reports/R.REPORT_FILES['secondary']).write_text(json.dumps(body))
    x_body = json.loads((reports/R.REPORT_FILES['0332']).read_text())
    x_body['crossings']['per_crossing'][0]['agreement_sv'] += 0.0005    # within the tolerance
    (reports/R.REPORT_FILES['0332']).write_text(json.dumps(x_body))
    assert R.main(['--workspace', str(ws), '--sample', str(sample), '--out', str(out), '--compare-only']) == 0
    cmp_ = json.loads(out.read_text())['comparison']
    assert cmp_['summary']['verdicts'] == dict(agree=7, agree_within_solver_bounds=1, agree_under_other_reading=0,
                                               differs=0)
    assert not cmp_['summary']['all_exact'] and cmp_['summary']['no_item_differs']

    body['per_strip'][0]['strip_mean_abs_sv'] += 0.1                   # beyond every bound
    body['per_strip'][0]['stretches'] = [1]                            # an index into the run-only list: S0001
    (reports/R.REPORT_FILES['secondary']).write_text(json.dumps(body))
    assert R.main(['--workspace', str(ws), '--sample', str(sample), '--out', str(out), '--compare-only']) == 1
    cmp_ = json.loads(out.read_text())['comparison']
    bad = cmp_['summary']['disagreements']
    assert bad == [dict(kind='piece', set='secondary', id='P0000', status=None, verdict='differs', reading=None,
                        fields=['stretches', 'strip_mean_abs_sv'])]
    piece = next(it for it in cmp_['items'] if it['kind'] == 'piece' and it['set'] == 'secondary')
    assert piece['fields']['stretches']['scorer'] == ['S0001'] and not piece['fields']['stretches']['bounds']['within']
    assert piece['fields']['strip_mean_abs_sv']['diff'] == pytest.approx(-0.104)
    assert not piece['fields']['strip_mean_abs_sv']['bounds']['within']
    assert cmp_['summary']['fields']['crossing.agreement_sv']['max_abs_diff'] == pytest.approx(0.0005)
    assert 'DIFFERS piece secondary P0000' in capsys.readouterr().out


def test_ground_truth_only_reads_no_strip_and_no_report(tmp_path):
    ws, sample = make_workspace(tmp_path)
    (ws/'runs').rename(tmp_path/'runs_hidden')
    out = tmp_path/'gt.json'
    assert R.main(['--workspace', str(ws), '--sample', str(sample), '--out', str(out), '--gt-only']) == 0
    rep = json.loads(out.read_text())
    assert rep['mode'] == 'gt-only' and 'crossings_0332' not in rep
    assert rep['comparison']['summary']['by_kind'] == dict(ground_truth=dict(
        items=3, agree=3, agree_within_solver_bounds=0, agree_under_other_reading=0, differs=0))
    assert 'strip_nodes' not in rep['secondary']['pieces']['P0000']


def test_ground_truth_tier2():
    """A plan value off by more than 1e-3 sv but within a robust node's allowance agrees within the solver bounds;
    a wrong sample count does not (it does not depend on the solver)."""
    g = wall()
    fiber = straight_fiber('f', (500.7, 0., 100.), (500.7, 300., 100.))
    sel = np.flatnonzero((fiber.s >= 100.) & (fiber.s <= 160.))
    pr = R.BilinearSurface(g, np.ones(g.shape[:2], bool)).project(fiber.pts[sel], AXIS)
    st = dict(samples=61, mean_d=0.7, median_abs_d=0.7, p90_abs_d=0.7, max_abs_d=0.7)
    assert R.gt_record(st, pr, len(sel))['verdict'] == 'agree'
    rec = R.gt_record(dict(st, max_abs_d=0.71), pr, len(sel))          # 0.01 sv: 0.005 + 2 % of 0.7 allows 0.019
    assert rec['verdict'] == 'agree_within_solver_bounds' and not rec['vs_plan']['max_abs_d']['agree']
    assert R.gt_record(dict(st, mean_d=-0.7), pr, len(sel))['verdict'] == 'differs'
    assert R.gt_record(dict(st, samples=62), pr, len(sel))['verdict'] == 'differs'


def test_an_input_error_gives_exit_code_2_even_when_nothing_differs(tmp_path):
    ws, sample = make_workspace(tmp_path)
    plan = ws/R.PLAN_FILES['secondary']
    plan.write_text(plan.read_text()+' ')                       # its sha256 is no longer the sample's
    out = tmp_path/'r.json'
    assert R.main(['--workspace', str(ws), '--sample', str(sample), '--out', str(out)]) == 2
    rep = json.loads(out.read_text())
    assert rep['errors'] == [f"{R.PLAN_FILES['secondary']}: sha256 differs from the one recorded in the sample"]
    fake_scorer_reports(ws, rep)                                # a comparison that agrees does not hide the error
    assert R.main(['--workspace', str(ws), '--sample', str(sample), '--out', str(out), '--compare-only']) == 2
    assert json.loads(out.read_text())['comparison']['summary']['no_item_differs']


def test_a_missing_umbilicus_gives_exit_code_2(tmp_path):
    ws, sample = make_workspace(tmp_path)
    (ws/R.UMBILICUS_0139).unlink()
    assert R.main(['--workspace', str(ws), '--sample', str(sample), '--out', str(tmp_path/'r.json'), '--gt-only']) == 2
