"""Synthetic checks of score_strips.py: known offsets recovered, baselines computed right, bootstrap reproducible.

Geometry: wrap segments are cylinders about the z axis through (CX, CY) (the umbilicus), meshed every 5 sv like
the PHerc0139 tifxyz (scale 0.05 -> 20 base voxels); a V fiber runs along z on the cylinder, an H fiber around it.
Units: sv = 4 base voxels. Segments and fibers are written in base voxels, strips in patch voxels (= sv)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score_strips as S  # noqa: E402

CX, CY = 500., 500.
BASE = 4.  # base voxels per sv


# ---- fixtures ------------------------------------------------------------------------------------------------

def cyl(R, phi, z):
    phi, z = np.broadcast_arrays(np.asarray(phi, float), np.asarray(z, float))
    return np.stack([CX+R*np.cos(phi), CY+R*np.sin(phi), z], -1)


def cylinder_grid(R, z0, z1, phi0, phi1, step=5.):
    zs = np.arange(z0, z1+1e-9, step)
    n = int(round((phi1-phi0)*R/step))
    phis = np.linspace(phi0, phi1, n+1)
    return cyl(R, phis[None, :], zs[:, None])


def write_tifxyz(d, grid_sv, file_unit_base, valid=None, extra_json=None):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(grid_sv).all(-1) if valid is None else valid
    g = np.where(valid[..., None], grid_sv*BASE/file_unit_base, -1.).astype(np.float32)
    for i, c in enumerate('xyz'):
        tifffile.imwrite(d/f'{c}.tif', g[..., i])
    pts = g[valid]
    (d/'meta.json').write_text(json.dumps(dict(format='tifxyz', type='seg', scale=[.05, .05],
                                               bbox=[pts.min(0).tolist(), pts.max(0).tolist()])))
    if extra_json:
        (d/'trace.json').write_text(json.dumps(extra_json))
    return d


def write_umbilicus(path):
    path.write_text(json.dumps(dict(control_points=[dict(x=CX*BASE, y=CY*BASE, z=-1e5, score=100),
                                                    dict(x=CX*BASE, y=CY*BASE, z=1e6, score=100)])))
    return path


def write_fiber(path, pts_sv, family='V', controls=None, version=4):
    line = (np.asarray(pts_sv)*BASE).tolist()
    idx = controls if controls is not None else [0, len(line)//2, len(line)-1]
    raw = dict(type='vc3d_fiber', version=version, line_points=line,
               control_points=[dict(position=line[i]) for i in idx],
               hv_classification=dict(automatic_tag=family, manual_tag=''), username='test')
    path.write_text(json.dumps(raw))
    return path


def v_fiber(R, phi0, z0=100., z1=500., offset=0.):
    z = np.arange(z0, z1+1e-9, 2.)
    return cyl(R+offset, phi0, z)


def h_fiber(R, z, phia, phib):
    phi = np.arange(phia, phib+1e-12, 2./R)
    return cyl(R, phi, z)


def v_strip(R, phi0, z0, z1, delta=0., half=60., col_step=5., row_step=4.):
    zs = np.arange(z0, z1+1e-9, row_step)
    v = np.arange(-half, half+1e-9, col_step)
    return cyl(R+delta, phi0+v[None, :]/(R+delta), zs[:, None])  # rows along the fiber (z)


def h_strip(R, z, phia, phib, delta=0., half=60., col_step=5., row_step=4.):
    phis = np.arange(phia, phib+1e-12, row_step/R)
    v = np.arange(-half, half+1e-9, col_step)
    return cyl(R+delta, phis[:, None], z+v[None, :])  # rows along the fiber (phi), columns across (z)


@pytest.fixture
def scene(tmp_path):
    """Two wrap segments (R = 200 and the next winding at R = 212) and the umbilicus."""
    segs = tmp_path/'segments'
    write_tifxyz(segs/'w001'/'mesh'/'w001.tifxyz', cylinder_grid(200., 0., 700., -0.9, 0.9), 1.)
    write_tifxyz(segs/'w002'/'mesh'/'w002.tifxyz', cylinder_grid(212., 0., 700., -0.9, 0.9), 1.)
    umb = write_umbilicus(tmp_path/'umbilicus.json')
    return tmp_path, segs, umb


def sources(segs):
    return S.discover_segments([segs], None, 1.)


# ---- projection ------------------------------------------------------------------------------------------------

def test_plane_offsets_recovered_exactly():
    rng = np.random.default_rng(0)
    e1, e2 = S.unit(np.array([1., .2, .1])), None
    e2 = S.unit(np.cross(e1, [0., 0.3, 1.]))
    n = S.unit(np.cross(e1, e2))
    r, c = np.meshgrid(np.arange(30)*5., np.arange(40)*5., indexing='ij')
    grid = 100.+r[..., None]*e2+c[..., None]*e1
    surf = S.GridSurface(grid)
    feet = 100.+rng.uniform(5, 140, (200, 1))*e2+rng.uniform(5, 190, (200, 1))*e1
    d = rng.uniform(-20, 20, 200)
    got, cov, foot, nrm = surf.project(feet+d[:, None]*n)
    sign = np.sign((nrm*n).sum(1))
    assert cov.all()
    np.testing.assert_allclose(got*sign, d, atol=1e-8)
    np.testing.assert_allclose(foot, feet, atol=1e-8)


def test_cylinder_offsets_recovered_outward_signed():
    rng = np.random.default_rng(1)
    axis = S.Umbilicus([[CX, CY, -10.], [CX, CY, 1e4]])
    surf = S.GridSurface(cylinder_grid(200., 0., 400., -1., 1.), axis=axis)
    phi, z, d = rng.uniform(-.9, .9, 500), rng.uniform(20, 380, 500), rng.uniform(-15, 15, 500)
    got, cov, _, _ = surf.project(cyl(200.+d, phi, z))
    assert cov.all()
    np.testing.assert_allclose(got, d, atol=0.02)  # chord sagitta of a 5 sv cell at R = 200 is 0.016 sv


def test_uncovered_beyond_edge_hole_and_reach():
    grid = cylinder_grid(200., 0., 200., -.5, .5)
    valid = np.ones(grid.shape[:2], bool)
    valid[18:24, 18:24] = False  # a hole
    surf = S.GridSurface(grid, valid)
    hole_centre = grid[21, 21]
    beyond_edge = cyl(200., 0., 210.)  # 10 sv above the top row
    far = cyl(240., 0., 100.)          # 40 sv off: beyond SEARCH_REACH
    inside = cyl(201., .2, 100.)
    _, cov, _, _ = surf.project(np.stack([hole_centre, beyond_edge, far, inside]))
    assert cov.tolist() == [False, False, False, True]


# ---- fibers and matching -----------------------------------------------------------------------------------------

def test_fiber_json_controlled_span_family_and_skips(tmp_path):
    pts = v_fiber(200., 0.)                    # 201 points, 400 sv
    write_fiber(tmp_path/'a.json', pts, 'V', controls=[50, 100, 150])
    f = S.load_fiber_json(tmp_path/'a.json')
    assert f.family == 'V' and f.family_source == 'automatic'
    assert abs(f.length-200.) < 1e-6           # points 50..150, 2 sv apart
    np.testing.assert_allclose(f.points[0], pts[50], atol=1e-6)
    write_fiber(tmp_path/'b.json', pts, 'unknown', controls=[10])
    assert S.load_fiber_json(tmp_path/'b.json') is None  # a seed: no controlled span (villa skips these too)
    write_fiber(tmp_path/'c.json', h_fiber(200., 300., -.5, .5), 'unknown', controls=[0, 100])
    assert S.load_fiber_json(tmp_path/'c.json').family == 'H'  # geometry fallback
    full = S.load_fiber_json(tmp_path/'a.json', span='full')
    assert abs(full.length-400.) < 1e-6


def test_matching_rule(scene):
    tmp, segs, umb = scene
    fib = tmp/'fibers'
    fib.mkdir()
    write_fiber(fib/'on.json', v_fiber(200., 0., offset=.5))             # on w001
    write_fiber(fib/'off4.json', v_fiber(200., .1, offset=4.))           # same sheet, too far: candidate only
    write_fiber(fib/'gap.json', v_fiber(200., .2, offset=6.))            # mid-gap: 6 sv from both windings
    mixed = np.concatenate([v_fiber(200., .3, 100., 340., .5), v_fiber(200., .3, 342., 500., 4.)])
    write_fiber(fib/'mixed.json', mixed)                                 # 60 % on, 40 % at 4 sv
    jump = np.concatenate([v_fiber(200., -.3, 100., 300., .3), v_fiber(212., -.3, 302., 500., -.3)])
    write_fiber(fib/'jump.json', jump)                                   # on w001 then on w002 (union)
    fibers, _ = S.load_fibers(fib)
    recs, names = S.match_fibers(fibers, sources(segs), S.Umbilicus.from_json(umb), log=lambda *a: None)
    r = {x['fiber']: x for x in recs}
    assert r['on']['matched'] and abs(r['on']['median_abs_d_candidate']-.5) < .03
    assert abs(r['on']['mean_d_on']-.5) < .03 and r['on']['runs'][0]['segment'] == names.index('w001')
    assert r['off4']['candidate'] and not r['off4']['matched']
    assert abs(r['off4']['median_abs_d_candidate']-4.) < .03
    assert not r['gap']['matched']
    assert r['mixed']['candidate'] and not r['mixed']['matched'] and .55 < r['mixed']['on_fraction'] < .65
    assert r['jump']['matched'] and {x['segment'] for x in r['jump']['runs']} == {0, 1}
    g1 = S.gate_g1(recs)
    assert not g1['pass'] and g1['matched_long'] == 2  # too few fibers


def test_g1_noise_floor_and_count():
    def rec(name, noise, matched=True):
        return dict(fiber=name, family='V', candidate=True, matched=matched, long_enough=True,
                    median_abs_d_candidate=noise, median_abs_d_on=min(noise, 2.), mean_d_on=0.,
                    on_fraction=.9 if matched else .1, flags={})
    good = [rec(f'f{i}', 1.0) for i in range(30)]
    assert S.gate_g1(good)['pass']
    assert not S.gate_g1(good[:29])['pass']
    noisy = good+[rec(f'n{i}', 3.5, matched=False) for i in range(40)]  # candidates far from the meshes
    g = S.gate_g1(noisy)
    assert g['noise_floor_sv'] == 3.5 and not g['pass']


# ---- strips: accuracy and baselines ------------------------------------------------------------------------------

def run_score(tmp, segs, umb, fibers_dir, strips_root, normals=None):
    fibers, _ = S.load_fibers(fibers_dir)
    axis = S.Umbilicus.from_json(umb)
    srcs = sources(segs)
    recs, _ = S.match_fibers(fibers, srcs, axis, log=lambda *a: None)
    strips = [S.load_strip(d) for d in S.discover_strips(strips_root)]
    per_fiber, skips = S.score_strips(strips, fibers, recs, srcs, axis, normals, log=lambda *a: None)
    return per_fiber, skips, fibers, strips, recs


@pytest.mark.parametrize('delta', [0., 1.25, -3.])
def test_known_strip_offsets_recovered_v_and_h(scene, delta):
    tmp, segs, umb = scene
    fib, st = tmp/'fibers', tmp/'strips'
    fib.mkdir()
    write_fiber(fib/'v1.json', v_fiber(200., 0.), 'V')
    write_fiber(fib/'h1.json', h_fiber(200., 350., -.7, .7), 'H')
    write_tifxyz(st/'fiber_1', v_strip(200., 0., 175., 425., delta), BASE, extra_json=dict(family='V', fiber_along='rows'))
    hs = h_strip(200., 350., -.6, .6, delta)
    write_tifxyz(st/'fiber_2', np.swapaxes(hs, 0, 1), BASE, extra_json=dict(family='H', fiber_along='columns'))
    per_fiber, skips, *_ = run_score(tmp, segs, umb, fib, st)
    assert not skips and set(per_fiber) == {'v1', 'h1'}
    for name in ('v1', 'h1'):
        e = per_fiber[name]['err']['strip']
        assert np.isfinite(e).sum() >= 0.95*per_fiber[name]['valid'].sum()
        np.testing.assert_allclose(e[np.isfinite(e)], delta, atol=0.03)
        floor = per_fiber[name]['fiber_floor']
        np.testing.assert_allclose(floor[np.isfinite(floor)], 0., atol=0.02)
    agg = S.aggregate(per_fiber)
    assert abs(agg['strip']['point']['mean']-abs(delta)) < 0.03
    assert agg['strip']['point']['within2'] == (1. if abs(delta) <= 2 else 0.)
    assert abs(agg['strip_signed_mean_sv']-delta) < 0.03


def villa_display_grid(rows, keep, family):
    """villa sheet_strip/infer_all.display_grid@6f2024569 (bruniss), reduced to rows/keep: the published layout."""
    z = rows[..., 2]
    r, c = np.nonzero(keep)
    rising = lambda index: np.sum((index-index.mean())*(z[r, c]-z[r, c].mean())) > 0
    along_columns = family == 'H'
    flip_rows = not along_columns and len(r) > 1 and rising(r)
    flip_columns = along_columns and len(c) > 1 and rising(c)
    index = (slice(None, None, -1 if flip_rows else 1), slice(None, None, -1 if flip_columns else 1))
    rows, keep = rows[index], keep[index]
    if along_columns:
        rows, keep = np.swapaxes(rows, 0, 1), keep.T
    return rows, keep, dict(fiber_along='columns' if along_columns else 'rows', fiber_reversed=bool(flip_rows),
                            across_reversed=bool(flip_columns))


def decisions_v(R, phi0, z0, z1, tpp=.5, flips=(), expected=None, height_offset=0.):
    """infer.py-style decisions along a V fiber on the cylinder: trace voxels; heights follow the cylinder
    (plus ``height_offset`` sv along the frame normal); some decisions with a flipped (u, v) frame."""
    f = np.arange(-35., 36., 2.)
    v = np.arange(-30., 30.01, 2.5)
    L = (z1-z0)*tpp
    arcs = np.arange(12., L+1e-9, 8.)
    pos = cyl(R, phi0, z0+arcs/tpp)*tpp
    eu = np.array([math.cos(phi0), math.sin(phi0), 0.])
    ev = np.array([-math.sin(phi0), math.cos(phi0), 0.])
    ef = np.array([0., 0., 1.])
    frames = np.tile(np.stack([eu, ev, ef], 1), (len(arcs), 1, 1))
    Rt = R*tpp
    u = np.sqrt((Rt+height_offset*tpp)**2-v**2)-Rt
    heights = np.tile(u, (len(arcs), len(f), 1))
    for k in flips:
        frames[k, :, :2] *= -1
        heights[k] = -heights[k][:, ::-1]
    out = dict(arcs=arcs, positions=pos, frames=frames, heights=heights, f=f, v=v, trace_per_patch=tpp)
    if expected is not None:
        out['expected_error_vox'] = np.full_like(heights, expected)
    return out, L


def write_infer_output(out_dir, dec, L, family='V', display=False, trim=75.):
    """An infer.py-style output (OUT/strip tifxyz + OUT/strip.npz) or, with ``display``, an infer_all-style strip
    in the published layout with the decisions beside it."""
    tpp = dec['trace_per_patch']
    oriented, oframes, _ = S.orient_decisions(dec['heights'], dec['frames'])
    rows, weight, _ = S.stitch(oriented, oframes, dec['positions'], dec['arcs'], dec['f'], dec['v'], 16.)
    df = dec['f'][1]-dec['f'][0]
    arc = dec['arcs'][0]+dec['f'][0]+np.arange(len(rows))*df
    keep = (weight > 0) & ((arc >= trim*tpp) & (arc <= L-trim*tpp))[:, None]
    rows = rows/tpp
    out_dir = Path(out_dir)
    if display:
        rows, keep, layout = villa_display_grid(rows, keep, family)
        d = write_tifxyz(out_dir, np.where(keep[..., None], rows, np.nan), BASE,
                         extra_json=dict(family=family, blend=16., **layout))
        np.savez(d/'decisions.npz', **dec)
    else:
        d = write_tifxyz(out_dir/'strip', np.where(keep[..., None], rows, np.nan), BASE, extra_json=dict(blend=16.))
        np.savez(out_dir/'strip.npz', **dec)
    return d


@pytest.mark.parametrize('display', [False, True])
def test_frame_plane_baseline_matches_tangent_plane_geometry(scene, display):
    tmp, segs, umb = scene
    fib, st = tmp/'fibers', tmp/'strips'
    fib.mkdir()
    write_fiber(fib/'v1.json', v_fiber(200., 0.), 'V')
    dec, L = decisions_v(200., 0., 100., 500., flips=(3, 4, 5, 20), expected=.6)
    write_infer_output(st/'fiber_1', dec, L, display=display)
    per_fiber, skips, *_ = run_score(tmp, segs, umb, fib, st)
    assert not skips
    e = per_fiber['v1']
    strip, plane = e['err']['strip'], e['err']['frame_plane']
    assert np.isfinite(plane).sum() == np.isfinite(strip).sum() > 1000
    np.testing.assert_allclose(strip[np.isfinite(strip)], 0., atol=0.03)
    # The frame plane is the tangent plane through the fiber: at across offset v it lies sqrt(R^2+v^2)-R outside.
    v_sv = dec['v']/dec['trace_per_patch']
    want = np.sqrt(200.**2+v_sv**2)-200.
    cols = np.isfinite(plane).all(0)
    got = np.nanmean(plane, axis=0)
    got = got if not display else got  # columns are v-ordered in both layouts (V strips keep their columns)
    np.testing.assert_allclose(np.sort(got[cols]), np.sort(want[cols]), atol=0.03)
    agg = S.aggregate(per_fiber)
    assert abs(agg['frame_plane']['stats']['point']['mean']-want.mean()) < 0.03
    assert e['expected'] is not None and np.allclose(e['expected'][e['valid']], .6)


def test_lasagna_plane_baseline_exact_and_tilted(scene):
    tmp, segs, umb = scene
    fib, st = tmp/'fibers', tmp/'strips'
    fib.mkdir()
    write_fiber(fib/'v1.json', v_fiber(200., 0.), 'V')
    write_tifxyz(st/'fiber_1', v_strip(200., 0., 175., 425.), BASE, extra_json=dict(family='V', fiber_along='rows'))
    fibers, _ = S.load_fibers(fib)
    f = fibers[0]
    results = {}
    for theta in (0., 10.):
        t = math.radians(theta)
        n = np.array([math.cos(t), math.sin(t), 0.])  # rotated about the fiber (z)
        n = n*(-1 if theta else 1)  # Lasagna normals are unoriented: the sign must not matter
        normals = {'v1': (np.tile(n, (len(f.points), 1)), np.ones(len(f.points), bool))}
        per_fiber, *_ = run_score(tmp, segs, umb, fib, st, normals)
        lp = per_fiber['v1']['err']['lasagna_plane']
        results[theta] = lp
        # brute force: nodes fiber + v*b with b = t x n, distance to the R = 200 cylinder
        b = np.cross(S.tangents(f.points)[0], n)
        offs = (np.arange(25)-12)*5.
        nodes = f.points[0]+offs[:, None]*b
        want = np.abs(np.hypot(nodes[:, 0]-CX, nodes[:, 1]-CY)-200.)
        got = np.nanmean(np.abs(lp), axis=0)
        np.testing.assert_allclose(got, want, atol=0.03)
        ang = per_fiber['v1']['lasagna_angle']
        assert abs(np.nanmedian(ang)-theta) < 1.  # a facet normal is up to half a cell angle (0.7 deg) off
    assert np.nanmean(np.abs(results[10.])) > np.nanmean(np.abs(results[0.]))+1.


def test_lasagna_decoding_and_sampling():
    n = S.unit(np.array([.48, -.36, .8]))
    nx = np.full((4, 4, 4), np.uint8(round(n[0]*127+128)))
    ny = np.full((4, 4, 4), np.uint8(round(n[1]*127+128)))
    pts = np.array([[6., 6., 6.], [9., 5.5, 10.]])  # sv; level scale 16 base = 4 sv per voxel (array: 16 sv)
    got, ok = S.lasagna_normals_at(pts, nx, ny, 16.)
    assert ok.all()
    assert np.all(np.abs(np.abs((got*n).sum(1))-1) < 1e-3)
    nx[...] = 0
    ny[...] = 0
    _, ok = S.lasagna_normals_at(pts, nx, ny, 16.)
    assert not ok.any()  # 0/0 = no data (villa: valid where nx or ny is set)


# ---- statistics ---------------------------------------------------------------------------------------------------

def test_bootstrap_reproducible_and_paired_delta():
    rng = np.random.default_rng(5)
    per = [np.abs(rng.normal(0, 1+i % 3, 50+i)) for i in range(40)]
    a = S.bootstrap_stats(per, seed=7)
    b = S.bootstrap_stats(per, seed=7)
    c = S.bootstrap_stats(per, seed=8)
    assert a == b
    assert a['point'] == c['point'] and a['ci'] != c['ci']
    for k in ('mean', 'median', 'p90', 'within1', 'within2'):
        lo, hi = a['ci'][k]
        assert lo <= a['point'][k] <= hi
    d = S.paired_delta(per, [x+1. for x in per], seed=7)
    assert abs(d['baseline_minus_strip_mean_sv']-1.) < 1e-12 and d['beaten']
    assert abs(d['ci'][0]-1.) < 1e-9 and abs(d['ci'][1]-1.) < 1e-9
    same = S.paired_delta(per, per, seed=7)
    assert not same['beaten'] and same['baseline_minus_strip_mean_sv'] == 0.
    assert S.paired_delta(per, [x+1. for x in per], seed=7) == d


def synthetic_entries(n, err, family='V', expected=None, rng=None):
    rng = rng or np.random.default_rng(0)
    out = {}
    for i in range(n):
        e = rng.normal(0, err, (40, 25))
        out[f'{family}{i}'] = dict(fiber=f'{family}{i}', family=family, strip=f's{i}',
                                   err=dict(strip=e, frame_plane=e*4+np.sign(e)*3, lasagna_plane=e*2+np.sign(e)),
                                   expected=None if expected is None else np.full((40, 25), expected),
                                   valid=np.ones((40, 25), bool), fiber_floor=np.full(40, .4),
                                   lasagna_angle=np.full(40, 5.), row_seg=np.zeros(40, int), notes=[])
    return out


def test_gate_g2_transfers_and_degrades():
    g1 = dict(**{'pass': True})
    good = S.aggregate(synthetic_entries(30, .8))
    assert S.gate_g2(good, g1)['verdict'] == 'transfers'
    bad = S.aggregate(synthetic_entries(30, 2.5))
    assert S.gate_g2(bad, g1)['verdict'] == 'degrades'
    few = S.aggregate(synthetic_entries(10, .8))
    assert S.gate_g2(few, g1)['verdict'].startswith('no accuracy claim')
    assert S.gate_g2(good, {'pass': False})['verdict'].startswith('no accuracy claim')


def test_calibration_bins():
    rng = np.random.default_rng(3)
    ents = {}
    ents.update(synthetic_entries(5, .5, 'V', expected=.6, rng=rng))
    ents.update(synthetic_entries(5, 2., 'H', expected=1.7, rng=rng))
    cal = S.calibration(ents)
    filled = [b for b in cal['bins'] if b['nodes']]
    assert [(b['lo'], b['nodes']) for b in filled] == [(.5, 5000), (1.5, 5000)]
    assert abs(filled[0]['mean_observed']-.5*math.sqrt(2/math.pi)) < .02
    assert abs(filled[1]['mean_observed']-2.*math.sqrt(2/math.pi)) < .06
    assert cal['spearman_fiber'] > .8


# ---- crossings, units, CLI -----------------------------------------------------------------------------------------

@pytest.mark.parametrize('delta', [0., 3.])
def test_crossing_agreement(tmp_path, delta):
    fib, st = tmp_path/'fibers', tmp_path/'strips'
    fib.mkdir()
    write_fiber(fib/'v1.json', v_fiber(200., 0.), 'V')
    write_fiber(fib/'h1.json', h_fiber(200., 300., -.7, .7), 'H')
    write_tifxyz(st/'fiber_1', v_strip(200., 0., 175., 425., delta), BASE, extra_json=dict(family='V', fiber_along='rows'))
    write_tifxyz(st/'fiber_2', np.swapaxes(h_strip(200., 300., -.6, .6), 0, 1), BASE,
                 extra_json=dict(family='H', fiber_along='columns'))
    fibers, _ = S.load_fibers(fib)
    strips = [S.load_strip(d) for d in S.discover_strips(st)]
    res = S.crossing_agreement(strips, fibers)
    assert res['crossings'] == 1
    assert abs(res['median']-delta) < 0.05


def test_strip_in_wrong_units_is_rejected(scene):
    tmp, segs, umb = scene
    fib, st = tmp/'fibers', tmp/'strips'
    fib.mkdir()
    write_fiber(fib/'v1.json', v_fiber(200., 0.), 'V')
    write_tifxyz(st/'fiber_1', v_strip(200., 0., 175., 425.), 1.)  # written in base voxels, read as patch voxels
    per_fiber, skips, *_ = run_score(tmp, segs, umb, fib, st)
    assert not per_fiber and 'unit or association error' in skips[0]['reason']


def test_cli_end_to_end(scene):
    tmp, segs, umb = scene
    fib, st = tmp/'fibers', tmp/'strips'
    fib.mkdir()
    write_fiber(fib/'v1.json', v_fiber(200., 0.), 'V')
    write_fiber(fib/'h1.json', h_fiber(200., 350., -.7, .7), 'H')
    dec, L = decisions_v(200., 0., 100., 500., expected=.6, height_offset=1.)
    write_infer_output(st/'fiber_1', dec, L, display=True)
    write_tifxyz(st/'fiber_2', np.swapaxes(h_strip(200., 350., -.6, .6, .5), 0, 1), BASE,
                 extra_json=dict(family='H', fiber_along='columns'))
    m, r = tmp/'matches.json', tmp/'report.json'
    S.main(['match', '--fibers', str(fib), '--segments', str(segs), '--umbilicus', str(umb), '--out', str(m)])
    S.main(['score', '--fibers', str(fib), '--segments', str(segs), '--umbilicus', str(umb), '--matches', str(m),
            '--strips', str(st), '--crossings', '--out', str(r)])
    rep = json.loads(r.read_text())
    assert rep['G1']['matched_long'] == 2 and not rep['G1']['pass']
    assert rep['G2']['verdict'].startswith('no accuracy claim')
    allf = rep['accuracy']['all']
    assert allf['scored_fibers'] == 2
    v = rep['accuracy']['V']['strip']['point']
    assert abs(v['mean']-1.) < 0.05  # heights 1 sv outside the cylinder along the frame normal
    assert rep['accuracy']['V']['frame_plane']['available_fibers'] == 1
    assert rep['accuracy']['H']['frame_plane']['available_fibers'] == 0  # no decisions for the H strip
    assert rep['calibration']['fibers'] == 1
    assert rep['crossings']['crossings'] == 1
    assert rep['provenance']['scorer_sha256'] and rep['provenance']['constants']['MATCH_MAX_D'] == 2.0
    assert allf['strip_coverage']['fraction'] > .95
    assert 'unflagged (no fold-back or jump flag)' in rep['sensitivity'] and 'annotator v1' in rep['sensitivity']


def test_fibers_from_urls_in_memory(tmp_path):
    write_fiber(tmp_path/'a.json', v_fiber(200., 0.), 'V')
    (tmp_path/'bad.json').write_text('{not json')
    urls = tmp_path/'urls.txt'
    urls.write_text('\n'.join([(tmp_path/'a.json').as_uri(), (tmp_path/'bad.json').as_uri(),
                               (tmp_path/'missing.json').as_uri()]))
    fibers, skipped = S.load_fibers(None, urls_file=urls)
    assert [f.name for f in fibers] == ['a'] and abs(fibers[0].length-400.) < 1e-6
    assert sorted(n for n, _ in skipped) == ['bad', 'missing']


def test_zarr_v2_chunk_reader_and_lasagna(tmp_path):
    import numcodecs
    codec = dict(id='blosc', cname='lz4', clevel=5, shuffle=1, blocksize=0)
    for name, value in (('nx', 128+int(round(.6*127))), ('ny', 128)):
        d = tmp_path/name
        d.mkdir()
        (d/'.zarray').write_text(json.dumps(dict(shape=[64, 64, 64], chunks=[32, 32, 32], dtype='|u1', fill_value=0,
                                                 order='C', filters=None, dimension_separator='/',
                                                 compressor=codec, zarr_format=2)))
        arr = np.full((32, 32, 32), value, np.uint8)
        arr[0, 0, 0] = 7
        for key in ('0/0/0', '0/0/1'):  # the other chunks are missing: fill value 0 (no data)
            (d/key).parent.mkdir(parents=True, exist_ok=True)
            (d/key).write_bytes(numcodecs.get_codec(codec).encode(arr.tobytes()))
    nx, ny = S.ZarrV2Chunks(tmp_path/'nx'), S.ZarrV2Chunks(tmp_path/'ny')
    got = nx.take(np.array([[0, 0, 0], [0, 0, 33], [5, 5, 40], [40, 40, 40], [0, 0, 99]]))
    assert got.tolist() == [7, 128+76, 128+76, 0, 0]
    pts = np.array([[60., 20., 20.], [20., 20., 200.]])  # sv: x 60 -> level-4 voxel 14.5 (chunk 0); z 200 -> chunk 1 (missing)
    n, ok = S.lasagna_normals_at(pts, nx, ny, 16.)
    assert ok.tolist() == [True, False]
    assert abs(abs(n[0, 0])-.6) < .01 and abs(abs(n[0, 2])-.8) < .01


def test_projection_batches_agree():
    rng = np.random.default_rng(9)
    surf = S.GridSurface(cylinder_grid(200., 0., 300., -1., 1.))
    pts = cyl(200.+rng.uniform(-5, 5, 1000), rng.uniform(-.9, .9, 1000), rng.uniform(10, 290, 1000))
    one = surf.project(pts)
    many = surf.project(pts, batch=97)
    for x, y in zip(one, many):
        np.testing.assert_array_equal(x, y)
