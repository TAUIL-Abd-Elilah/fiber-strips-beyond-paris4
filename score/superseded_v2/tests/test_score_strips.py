"""Synthetic checks of score_strips.py (v2): known offsets recovered, the stretch rule, the inference plan, the
budget and the seeded subsample, baselines computed right, bootstrap reproducible.

Geometry: wrap segments are cylinders about the z axis through (CX, CY) (the umbilicus), meshed every 5 sv like
the PHerc0139 tifxyz (scale 0.05 -> 20 base voxels); a V fiber runs along z on the cylinder, an H fiber around it.
Units: sv = 4 base voxels. Segments and fibers are written in base voxels, strips in patch voxels (= sv).

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_score_strips.py
"""
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
               hv_classification=dict(automatic_tag=family, manual_tag=''), username='test',
               vc_open_data_coordinate_space='TEST/1@L0')
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


def match(fibers, srcs, axis, min_len=S.MIN_STRETCH_SV):
    """The match subcommand's selection, in memory: stretches (with ids), G1."""
    hits, _ = S.fiber_segment_distances(fibers, srcs, axis, log=lambda *a: None)
    names = [s.name for s in srcs]
    stretches, near, dropped, match_abs = [], [], [], []
    for k, f in enumerate(fibers):
        acc, nr, dr = S.find_stretches(f, hits[k], names, min_len)
        for st in acc:
            idx, d = hits[k][st['segment_index']]
            sel = (idx >= st['i0']) & (idx <= st['i1'])
            match_abs.append(np.abs(d[sel]))
        stretches += acc
        near += nr
        dropped += dr
    for n, st in enumerate(stretches):
        st['id'], st['_n'] = f'S{n:04d}', n
    return stretches, S.gate_g1(stretches, match_abs, near, len(fibers), min_len), dropped


# ---- projection ------------------------------------------------------------------------------------------------

def test_plane_offsets_recovered_exactly():
    rng = np.random.default_rng(0)
    e1 = S.unit(np.array([1., .2, .1]))
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


def test_projection_batches_agree():
    rng = np.random.default_rng(9)
    surf = S.GridSurface(cylinder_grid(200., 0., 300., -1., 1.))
    pts = cyl(200.+rng.uniform(-5, 5, 1000), rng.uniform(-.9, .9, 1000), rng.uniform(10, 290, 1000))
    one = surf.project(pts)
    many = surf.project(pts, batch=97)
    for x, y in zip(one, many):
        np.testing.assert_array_equal(x, y)


# ---- fibers ------------------------------------------------------------------------------------------------------

def test_fiber_json_controlled_span_family_uuid_and_skips(tmp_path):
    pts = v_fiber(200., 0.)                    # 201 points, 400 sv
    write_fiber(tmp_path/'a.json', pts, 'V', controls=[50, 100, 150])
    f = S.load_fiber_json(tmp_path/'a.json')
    assert f.family == 'V' and f.family_source == 'automatic'
    assert abs(f.length-200.) < 1e-6           # points 50..150, 2 sv apart
    np.testing.assert_allclose(f.points[0], pts[50], atol=1e-6)
    assert f.line_offset == 50 and len(f.line) == 101 and abs(f.line_s[-1]-f.length) < 1e-9
    import uuid
    assert f.uuid == str(uuid.uuid5(uuid.NAMESPACE_URL, 'vc3d_fiber:TEST/1@L0/a.json')) and f.uuid_source == 'uuid5(file)'
    assert f.afv_id == int(f.uuid.replace('-', '')[:13], 16)
    write_fiber(tmp_path/'b.json', pts, 'unknown', controls=[10])
    assert S.load_fiber_json(tmp_path/'b.json') is None  # a seed: no controlled span (villa skips these too)
    write_fiber(tmp_path/'c.json', h_fiber(200., 300., -.5, .5), 'unknown', controls=[0, 100])
    assert S.load_fiber_json(tmp_path/'c.json').family_source == 'geometry'
    full = S.load_fiber_json(tmp_path/'a.json', span='full')
    assert abs(full.length-400.) < 1e-6 and full.line_offset == 0


def test_fiber_flags_foldback_and_jump(tmp_path):
    z = np.r_[np.arange(100., 300.01, 2.), np.arange(298., 280., -2.), np.arange(282., 300., 2.),
              np.arange(330., 500.01, 2.)]  # a fold-back near z 300 and a 30 sv jump
    write_fiber(tmp_path/'f.json', cyl(200., 0., z), 'V', controls=[0, len(z)-1])
    f = S.load_fiber_json(tmp_path/'f.json')
    assert f.flags['foldbacks'] >= 1 and f.flags['jumps'] == 1
    assert all(150. < x < 260. for x in f.flags['foldback_s'])


def test_fibers_from_urls_in_memory(tmp_path):
    write_fiber(tmp_path/'a.json', v_fiber(200., 0.), 'V')
    (tmp_path/'bad.json').write_text('{not json')
    urls = tmp_path/'urls.txt'
    urls.write_text('\n'.join([(tmp_path/'a.json').as_uri(), (tmp_path/'bad.json').as_uri(),
                               (tmp_path/'missing.json').as_uri()]))
    fibers, skipped = S.load_fibers(None, urls_file=urls)
    assert [f.name for f in fibers] == ['a'] and abs(fibers[0].length-400.) < 1e-6
    assert sorted(n for n, _ in skipped) == ['bad', 'missing']


# ---- the stretch rule (PREREG 4) -----------------------------------------------------------------------------------

def test_stretch_rule_offsets_contiguity_and_noise_floors(scene):
    tmp, segs, umb = scene
    fib = tmp/'fibers'
    fib.mkdir()
    write_fiber(fib/'on.json', v_fiber(200., 0., offset=.5))                 # 400 sv at +0.5: one stretch
    write_fiber(fib/'off3.json', v_fiber(200., .1, offset=3.))               # 3 sv off: near run only
    write_fiber(fib/'gap.json', v_fiber(200., .2, offset=6.5))               # mid-gap: nothing
    half = np.concatenate([v_fiber(200., .3, 100., 340., .5), v_fiber(200., .3, 342., 500., 4.)])
    write_fiber(fib/'part.json', half)                                       # 240 sv on, then 4 sv off
    longp = np.concatenate([v_fiber(200., .4, 100., 400., -.5), v_fiber(200., .4, 402., 500., 4.)])
    write_fiber(fib/'longpart.json', longp)                                  # 300 sv on (-0.5), then off
    blip = v_fiber(200., -.2, offset=.5)
    blip[100, :2] += (blip[100, :2]-[CX, CY])/200.*2.2                       # one sample at +2.7: splits
    write_fiber(fib/'blip.json', blip)
    jump = np.concatenate([v_fiber(200., -.3, 100., 300., .3), v_fiber(212., -.3, 302., 500., -.3)])
    write_fiber(fib/'jump.json', jump)                                       # 200 sv on w001, 200 on w002
    fibers, _ = S.load_fibers(fib)
    srcs = sources(segs)
    axis = S.Umbilicus.from_json(umb)
    st, g1, _ = match(fibers, srcs, axis)
    by = {}
    for r in st:
        by.setdefault(r['fiber'], []).append(r)
    assert set(by) == {'on', 'longpart'}
    assert len(by['on']) == 1 and abs(by['on'][0]['length_sv']-400.) < 1e-6 and by['on'][0]['segment'] == 'w001'
    assert abs(by['on'][0]['mean_d']-.5) < .03 and abs(by['on'][0]['median_abs_d']-.5) < .03
    lp = by['longpart'][0]
    assert abs(lp['mean_d']+.5) < .03 and 295. <= lp['length_sv'] <= 302.
    assert g1['fibers_with_stretch'] == 2 and not g1['pass']
    assert abs(g1['noise_floor_match_sv']-.5) < .03
    # NF6 pools the near runs (|d| <= 6, >= 2.5 mm): on, blip (0.5), off3 (3), part and longpart (0.5 then 4),
    # gap (5.5 from w002), nothing from jump (200 sv per segment): 1344 at 0.5, 401 at 3, 260 at 4, 401 at 5.5
    assert abs(g1['noise_floor_nf6_sv']-.5) < .03 and g1['near_runs'] == 6  # gap is 5.5 sv from w002
    nf6_far = S.gate_g1(st, [np.full(10, .5)]*len(st), [np.full(10, 3.)]*5, 7)
    assert nf6_far['noise_floor_nf6_sv'] == 3. and not nf6_far['pass']
    # with villa README's --min-run 40 the shorter runs count, the blip splits into two, the jump gives two segments
    st40, g40, _ = match(fibers, srcs, axis, min_len=40.)
    by40 = {}
    for r in st40:
        by40.setdefault(r['fiber'], []).append(r)
    assert len(by40['blip']) == 2 and len(by40['part']) == 1 and abs(by40['part'][0]['length_sv']-240.) <= 2.
    assert {r['segment'] for r in by40['jump']} == {'w001', 'w002'}
    assert 'off3' not in by40 and 'gap' not in by40


def test_g1_needs_30_fibers_and_both_noise_floors():
    def stretch(name):
        return dict(fiber=name, family='V', length_mm=3., median_abs_d=.5)
    st = [stretch(f'f{i}') for i in range(30)]
    ok = [np.full(300, .5)]*30
    assert S.gate_g1(st, ok, [np.full(300, 1.)], 30)['pass']
    assert not S.gate_g1(st[:29], ok[:29], [np.full(300, 1.)], 30)['pass']
    g = S.gate_g1(st, ok, [np.full(300, 2.5)], 30)  # fibers scattered within half a gap of the meshes
    assert g['noise_floor_nf6_sv'] == 2.5 and not g['pass']


def test_overlapping_segments_resolved_once(tmp_path):
    segs = tmp_path/'segments'
    write_tifxyz(segs/'a'/'mesh'/'a.tifxyz', cylinder_grid(200., 0., 700., -0.9, 0.9), 1.)
    write_tifxyz(segs/'b'/'mesh'/'b.tifxyz', cylinder_grid(200.2, 300., 700., -0.9, 0.9), 1.)  # duplicate, z 300-700
    umb = write_umbilicus(tmp_path/'umbilicus.json')
    write_fiber(tmp_path/'v.json', v_fiber(200., 0., 100., 650.))
    fibers = [S.load_fiber_json(tmp_path/'v.json')]
    st, _, dropped = match(fibers, sources(segs), S.Umbilicus.from_json(umb))
    assert len(st) == 1 and st[0]['segment'] == 'a' and abs(st[0]['length_sv']-550.) < 1e-6
    assert len(dropped) == 1 and dropped[0]['segment'] == 'b'


# ---- inference plan, budget and subsample (PREREG 4.6-4.8) --------------------------------------------------------

def test_head_grid_matches_full_fiber_decisions():
    heads = S.head_grid(1000., 300., 600.)
    full = np.arange(S.FIRST_HEAD_SV, 1000.+1e-9, S.DECISION_STEP_SV)  # infer_all: arange(12, L, 8) trace voxels
    assert set(heads.tolist()) <= set(full.tolist())
    assert heads[0] >= 300.-S.HEAD_MARGIN_SV and heads[0]-S.DECISION_STEP_SV < 300.-S.HEAD_MARGIN_SV
    assert heads[-1] <= 600.+S.HEAD_MARGIN_SV and heads[-1]+S.DECISION_STEP_SV > 600.+S.HEAD_MARGIN_SV
    # every row in [s0, s1] has all the decisions within the blend of it
    for row in np.arange(300., 600.01, 4.):
        need = full[np.abs(full-row) < S.BLEND_SV]
        assert set(need.tolist()) <= set(heads.tolist())
    assert S.head_grid(1000., 0., 100.)[0] == S.FIRST_HEAD_SV  # clipped at the fiber start like a full run


def test_pieces_cut_merge_and_arc(tmp_path):
    write_fiber(tmp_path/'v.json', v_fiber(200., 0., 0., 3000.), controls=[10, 1400])  # line points every 2 sv
    f = S.load_fiber_json(tmp_path/'v.json')
    assert f.line_offset == 10 and abs(f.length-2780.) < 1e-6
    ws = [dict(id='a', s0=1000., s1=1300.), dict(id='b', s0=1500., s1=1600.), dict(id='c', s0=2400., s1=2500.),
          dict(id='d', s0=20., s1=80.)]
    pieces = S.number_pieces(S.plan_pieces(f, ws), 'P', ws)
    groups = [p['windows'] for p in pieces]
    assert groups == [['d'], ['a', 'b'], ['c']]  # a and b merge (their pieces overlap); c is far; d at the start
    pd, pab, pc = pieces
    assert pd['parent_arc_sv'][0] == 0. and pd['heads_sv'][0] == S.FIRST_HEAD_SV
    h0, h1 = pab['heads_sv']
    assert pab['parent_arc_sv'][0] <= h0-S.PIECE_BEFORE_SV < pab['parent_arc_sv'][0]+2.+1e-9
    assert pab['parent_arc_sv'][1] >= h1+S.PIECE_AFTER_SV
    assert pab['line_index_range'][0] == 10+int(round(pab['parent_arc_sv'][0]/2.))
    assert pab['decisions'] == int(round((h1-h0)/S.DECISION_STEP_SV))+1
    np.testing.assert_allclose(pab['arc_patch_vox'], [h0-pab['parent_arc_sv'][0], h1-pab['parent_arc_sv'][0]],
                               atol=1e-3)
    assert ws[0]['piece'] == ws[1]['piece'] == pab['id'] and ws[0]['piece_afv_id'] == pab['afv_id']
    assert pab['afv_id'] == int(S.piece_uuid(f.uuid, *pab['line_index_range']).replace('-', '')[:13], 16)
    # the decisions infer.py makes on the piece (arange(START/2, END/2, 8) trace voxels) are the planned ones
    lo, hi = pab['arc_patch_vox']
    assert len(np.arange(lo/2., hi/2.+1e-9, 8.)) == pab['decisions']


def test_ct_footprint_exact_against_brute_force():
    chunks, shape = [8, 8, 8], [64, 64, 64]
    pts = np.array([[40., 40., 40.], [120., 90., 60.], [130., 140., 200.]])  # L0; level 2 -> /4
    got = S.ct_footprint(pts, 2, 6., chunks, shape)
    p = pts/4.
    dense = np.concatenate([a+(b-a)*np.linspace(0, 1, 2000)[:, None] for a, b in zip(p[:-1], p[1:])])[:, ::-1]

    def brute(radius):
        out = set()
        for k in np.ndindex(8, 8, 8):
            lo, hi = np.array(k)*8., np.array(k)*8.+7.
            if (np.linalg.norm(np.clip(dense, lo, hi)-dense, axis=1) <= radius).any():
                out.add(k)
        return out
    assert brute(5.5) <= got <= brute(6.)  # exact up to the 1-voxel sampling of the polyline
    assert len(got) > 20


def units(n_strata=3, per=6, ct_each=10, overlap=False):
    out = []
    for s in range(n_strata):
        for i in range(per):
            base = 0 if overlap else (s*per+i)*ct_each
            out.append(dict(id=f'U{s}{i}', fiber=f'f{s}{i % 3}', stratum=s, decisions=100,
                            _ct={(base+j, 0, 0) for j in range(ct_each)}))
    return out


def test_budget_and_subsample_caps_strata_and_seed():
    zarray = dict(chunks=[128, 128, 128], shape=[1000, 1000, 1000], dtype='|u1')
    us = units()
    b = S.budget(us, zarray)
    assert b['decisions'] == 1800 and b['ct_chunks'] == 180 and abs(b['ct_gib']-180*2/1024) < 1e-3
    assert abs(b['cpu_hours']-(1800/S.CPU_DECISIONS_PER_S+S.PROCESS_OVERHEAD_S*18)/3600) < 1e-3
    cap_gib = 60*2/1024  # room for 60 chunks = 6 units
    a1 = S.stratified_subsample(us, lambda u: u['stratum'], zarray, 100., cap_gib, seed=1)
    a2 = S.stratified_subsample(us, lambda u: u['stratum'], zarray, 100., cap_gib, seed=1)
    a3 = S.stratified_subsample(us, lambda u: u['stratum'], zarray, 100., cap_gib, seed=2)
    assert a1 == a2 and len(a1) == 6 and a1 != a3
    assert [int(i[1]) for i in a1] == [0, 1, 2, 0, 1, 2]  # round robin over the strata
    chosen = [u for u in us if u['id'] in a1]
    assert len({u['fiber'] for u in chosen}) == 6  # new fibers first
    assert S.budget(chosen, zarray)['ct_gib'] <= cap_gib+1e-9
    cached = {(j, 0, 0) for j in range(180)}  # everything cached: only the CPU cap binds
    cpu_cap = (1000/S.CPU_DECISIONS_PER_S+10*S.PROCESS_OVERHEAD_S)/3600+1e-6
    a4 = S.stratified_subsample(us, lambda u: u['stratum'], zarray, cpu_cap, 0., cached=cached, seed=1)
    assert len(a4) == 10
    a5 = S.stratified_subsample(us, lambda u: u['stratum'], zarray, 100., 100., seed=1, max_units=4)
    assert len(a5) == 4


# ---- strips: accuracy and baselines ------------------------------------------------------------------------------

def run_score(tmp, segs, umb, fibers_dir, strips_root, normals=None, pieces=None, min_len=S.MIN_STRETCH_SV):
    fibers, _ = S.load_fibers(fibers_dir)
    axis = S.Umbilicus.from_json(umb)
    srcs = sources(segs)
    st, g1, _ = match(fibers, srcs, axis, min_len)
    strips = [S.load_strip(d) for d in S.discover_strips(strips_root)]
    entries, skips = S.score_strips(strips, fibers, st, srcs, axis, normals, log=lambda *a: None, pieces=pieces)
    return entries, skips, fibers, strips, st


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
    entries, skips, *_ = run_score(tmp, segs, umb, fib, st)
    assert not skips and {e['fiber'] for e in entries.values()} == {'v1', 'h1'}
    for e in entries.values():
        err = e['err']['strip']
        assert np.isfinite(err).sum() >= 0.95*e['valid'].sum()
        np.testing.assert_allclose(err[np.isfinite(err)], delta, atol=0.03)
        floor = e['fiber_floor']
        np.testing.assert_allclose(floor[np.isfinite(floor)], 0., atol=0.02)
    agg = S.aggregate(entries)
    assert agg['scored_fibers'] == 2 and agg['scored_stretches'] == 2
    assert abs(agg['strip']['point']['mean']-abs(delta)) < 0.03
    assert agg['strip']['point']['within2'] == (1. if abs(delta) <= 2 else 0.)
    assert abs(agg['strip_signed_mean_sv']-delta) < 0.03
    assert S.aggregate(entries, 'H')['scored_fibers'] == 1 and S.aggregate(entries, 'V')['scored_fibers'] == 1


def test_only_rows_inside_stretches_are_scored_against_their_segment(scene):
    tmp, segs, umb = scene
    fib, st = tmp/'fibers', tmp/'strips'
    fib.mkdir()
    # on w001 for z 100-400 (300 sv: a stretch), then 6.5 sv outward (mid-gap) for z 402-600
    write_fiber(fib/'v1.json', np.concatenate([v_fiber(200., 0., 100., 400.), v_fiber(200., 0., 402., 600., 6.5)]))
    # a strip that jumps to the next winding (R 212) for z > 300: inside the stretch it is scored against w001
    lower = v_strip(200., 0., 120., 330.)
    upper = v_strip(212., 0., 334., 420.)
    write_tifxyz(st/'x', np.concatenate([lower, upper]), BASE)
    entries, skips, _, _, stretches = run_score(tmp, segs, umb, fib, st)
    assert len(stretches) == 1 and not skips
    e = next(iter(entries.values()))
    err = e['err']['strip']
    rows_scored = np.flatnonzero(e['row_seg'] >= 0)
    z_rows = np.r_[np.arange(120., 330.01, 4.), np.arange(334., 420.01, 4.)]
    assert z_rows[rows_scored].max() <= 400. and z_rows[rows_scored].min() >= 100.
    assert (e['row_seg'][z_rows > 400.] < 0).all()  # beyond the stretch: not scored
    jumped = (z_rows > 330.) & (e['row_seg'] >= 0)
    assert jumped.sum() >= 15
    assert np.allclose(np.nanmedian(err[jumped], axis=1), 12., atol=.05)  # one winding gap, not a hit on w002


def test_strip_named_after_piece_scores_its_stretches_only(scene):
    tmp, segs, umb = scene
    fib, st = tmp/'fibers', tmp/'strips'
    fib.mkdir()
    write_fiber(fib/'v1.json', v_fiber(200., 0., 100., 700.))
    # two stretches on one fiber: 40 sv min run, the fiber 6.5 sv off in z 380-420
    pts = v_fiber(200., 0., 100., 700.)
    off = (pts[:, 2] > 380.) & (pts[:, 2] < 420.)
    pts[off, :2] += (pts[off, :2]-[CX, CY])/200.*6.5
    write_fiber(fib/'v1.json', pts)
    write_tifxyz(st/'P0001'/'strip', v_strip(200., 0., 110., 690., 1.), BASE)
    fibers, _ = S.load_fibers(fib)
    stretches, _, _ = match(fibers, sources(segs), S.Umbilicus.from_json(umb), 40.)
    assert len(stretches) == 2
    pieces = [dict(id='P0001', windows=[stretches[1]['id']])]
    strips = [S.load_strip(d) for d in S.discover_strips(st)]
    entries, skips = S.score_strips(strips, fibers, stretches, sources(segs), S.Umbilicus.from_json(umb),
                                    log=lambda *a: None, pieces=pieces)
    e = entries['P0001']
    assert e['stretches'] == [stretches[1]['id']]
    assert set(np.unique(e['row_stretch'][e['row_stretch'] >= 0]).tolist()) == {stretches[1]['_n']}


def villa_display_grid(rows, keep, family):
    """villa sheet_strip/infer_all.display_grid@6f2024569 (bruniss), reduced to rows/keep: the published layout."""
    z = rows[..., 2]
    r, c = np.nonzero(keep)

    def rising(index):
        return np.sum((index-index.mean())*(z[r, c]-z[r, c].mean())) > 0
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
    entries, skips, *_ = run_score(tmp, segs, umb, fib, st)
    assert not skips
    e = next(iter(entries.values()))
    strip, plane = e['err']['strip'], e['err']['frame_plane']
    assert np.isfinite(plane).sum() == np.isfinite(strip).sum() > 1000
    np.testing.assert_allclose(strip[np.isfinite(strip)], 0., atol=0.03)
    # The frame plane is the tangent plane through the fiber: at across offset v it lies sqrt(R^2+v^2)-R outside.
    v_sv = dec['v']/dec['trace_per_patch']
    want = np.sqrt(200.**2+v_sv**2)-200.
    cols = np.isfinite(plane).all(0)
    got = np.nanmean(plane, axis=0)
    np.testing.assert_allclose(np.sort(got[cols]), np.sort(want[cols]), atol=0.03)
    agg = S.aggregate(entries)
    assert abs(agg['frame_plane']['stats']['point']['mean']-want.mean()) < 0.03
    assert agg['frame_plane']['delta']['fibers'] if agg['frame_plane']['delta'] else True
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
        entries, *_ = run_score(tmp, segs, umb, fib, st, normals)
        e = next(iter(entries.values()))
        lp = e['err']['lasagna_plane']
        results[theta] = lp
        b = np.cross(S.tangents(f.points)[0], n)  # brute force: nodes fiber + v*b, distance to the R = 200 cylinder
        offs = (np.arange(25)-12)*5.
        nodes = f.points[0]+offs[:, None]*b
        want = np.abs(np.hypot(nodes[:, 0]-CX, nodes[:, 1]-CY)-200.)
        got = np.nanmean(np.abs(lp), axis=0)
        np.testing.assert_allclose(got, want, atol=0.03)
        ang = e['lasagna_angle']
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
    pts = np.array([[60., 20., 20.], [20., 20., 200.]])  # sv: x 60 -> level-4 voxel 14.5 (chunk 0); z 200 -> chunk 1
    n, ok = S.lasagna_normals_at(pts, nx, ny, 16.)
    assert ok.tolist() == [True, False]
    assert abs(abs(n[0, 0])-.6) < .01 and abs(abs(n[0, 2])-.8) < .01
    assert S.fetch_zarray(tmp_path/'nx')['chunks'] == [32, 32, 32]


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


def synthetic_entries(n, err, family='V', expected=None, rng=None, strips_per_fiber=1):
    rng = rng or np.random.default_rng(0)
    out = {}
    for i in range(n):
        for j in range(strips_per_fiber):
            e = rng.normal(0, err, (40, 25))
            out[f'{family}{i}_{j}'] = dict(fiber=f'{family}{i}', family=family, strip=f's{i}_{j}',
                                           stretches=[f'S{i}_{j}'],
                                           err=dict(strip=e, frame_plane=e*4+np.sign(e)*3,
                                                    lasagna_plane=e*2+np.sign(e)),
                                           expected=None if expected is None else np.full((40, 25), expected),
                                           valid=np.ones((40, 25), bool), fiber_floor=np.full(40, .4),
                                           lasagna_angle=np.full(40, 5.), row_seg=np.zeros(40, int),
                                           row_stretch=np.full(40, i*strips_per_fiber+j), notes=[])
    return out


def test_gate_g2_transfers_and_degrades():
    g1 = {'pass': True}
    good = S.aggregate(synthetic_entries(30, .8))
    assert S.gate_g2(good, g1)['verdict'] == 'transfers'
    bad = S.aggregate(synthetic_entries(30, 2.5))
    assert S.gate_g2(bad, g1)['verdict'] == 'degrades'
    few = S.aggregate(synthetic_entries(10, .8))
    assert S.gate_g2(few, g1)['verdict'].startswith('no accuracy claim')
    assert S.gate_g2(good, {'pass': False})['verdict'].startswith('no accuracy claim')
    no_base = synthetic_entries(30, .8)
    for e in no_base.values():
        del e['err']['lasagna_plane']
    assert S.gate_g2(S.aggregate(no_base), g1)['verdict'] == 'degrades'  # a missing baseline counts as not beaten


def test_clusters_are_fibers_not_strips():
    ents = synthetic_entries(15, .8, strips_per_fiber=3)
    agg = S.aggregate(ents)
    assert agg['scored_fibers'] == 15 and agg['strip']['fibers'] == 15 and agg['scored_stretches'] == 45
    assert agg['strip']['point']['nodes'] == 15*3*40*25


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
    assert res['crossings'] == 1 and res['tight_subset']['crossings'] == 1
    assert abs(res['median']-delta) < 0.05
    cr = S.find_crossings(fibers)
    assert len(cr) == 1 and cr[0]['id'] == 'X0000' and cr[0]['closest_approach_sv'] < 1.
    plan = [dict(cr[0]), dict(cr[0], id='X0001', v_fiber='missing')]
    res2 = S.crossing_agreement(strips, fibers, plan=plan)
    assert res2['crossings'] == 1 and res2['missing'] == ['X0001'] and res2['planned'] == 2


def test_crossing_windows_centred_and_inside(tmp_path):
    write_fiber(tmp_path/'v1.json', v_fiber(200., 0., 100., 500.), 'V')
    write_fiber(tmp_path/'h1.json', h_fiber(200., 130., -.7, .7), 'H')  # crosses 30 sv from the V fiber's start
    fibers, _ = S.load_fibers(tmp_path)
    cr = S.find_crossings(fibers)
    kept, ws = S.crossing_windows(cr, {f.name: f for f in fibers})
    assert len(kept) == 1 and len(ws) == 2
    w = {x['fiber']: x for x in ws}
    assert w['v1']['s0'] == 0. and abs(w['v1']['s1']-S.MIN_STRETCH_SV) < 1e-9  # shifted to fit the fiber
    assert abs((w['h1']['s0']+w['h1']['s1'])/2-cr[0]['h_s']) < 1e-6           # centred on the crossing


def test_strip_in_wrong_units_is_rejected(scene):
    tmp, segs, umb = scene
    fib, st = tmp/'fibers', tmp/'strips'
    fib.mkdir()
    write_fiber(fib/'v1.json', v_fiber(200., 0.), 'V')
    write_tifxyz(st/'fiber_1', v_strip(200., 0., 175., 425.), 1.)  # written in base voxels, read as patch voxels
    entries, skips, *_ = run_score(tmp, segs, umb, fib, st)
    assert not entries and 'unit or association error' in skips[0]['reason']


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
    mj = json.loads(m.read_text())
    assert mj['analysis'] == 'primary' and len(mj['stretches']) == 2 and len(mj['pieces']) == 2
    assert all(st_['piece'] in {p['id'] for p in mj['pieces']} for st_ in mj['stretches'])
    S.main(['score', '--fibers', str(fib), '--segments', str(segs), '--umbilicus', str(umb), '--matches', str(m),
            '--strips', str(st), '--crossings', '--out', str(r)])
    rep = json.loads(r.read_text())
    assert rep['G1']['fibers_with_stretch'] == 2 and not rep['G1']['pass']
    assert rep['G2']['verdict'].startswith('no accuracy claim')
    allf = rep['accuracy']['all']
    assert allf['scored_fibers'] == 2 and rep['stretches_scored'] == 2
    v = rep['accuracy']['V']['strip']['point']
    assert abs(v['mean']-1.) < 0.05  # heights 1 sv outside the cylinder along the frame normal
    assert rep['accuracy']['V']['frame_plane']['available_fibers'] == 1
    assert rep['accuracy']['H']['frame_plane']['available_fibers'] == 0  # no decisions for the H strip
    assert rep['calibration']['fibers'] == 1
    assert rep['crossings']['crossings'] == 1
    assert rep['provenance']['scorer_sha256'] and rep['provenance']['constants']['MATCH_MAX_D'] == 2.0
    assert allf['strip_coverage']['fraction'] > .95
    assert 'annotator v1' in rep['sensitivity'] and any(k.startswith('unflagged') for k in rep['sensitivity'])
    m2, r2 = tmp/'matches2.json', tmp/'report2.json'
    S.main(['match', '--fibers', str(fib), '--segments', str(segs), '--umbilicus', str(umb), '--out', str(m2),
            '--min-stretch-mm', '0.38384', '--analysis', 'secondary'])
    assert abs(json.loads(m2.read_text())['min_stretch_sv']-40.) < 1e-9
    S.main(['score', '--fibers', str(fib), '--segments', str(segs), '--umbilicus', str(umb), '--matches', str(m2),
            '--strips', str(st), '--out', str(r2)])
    rep2 = json.loads(r2.read_text())
    assert rep2['analysis'] == 'secondary' and rep2['G2']['verdict'].startswith('secondary: no accuracy claim')
