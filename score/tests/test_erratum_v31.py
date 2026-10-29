"""PREREG v3.1 erratum (section 10.7): the rules in the code that v3's text did not state, the strata rule, the new
FREEZE pins, and the scorer on REAL infer.py output (a small Paris 4 piece run, tests/fixtures/paris4_real).

The fixture is Paris 4 only (CC BY-NC 4.0 derived data, LICENSE-DATA.md): villa's sheet_strip/infer.py at the pinned
worktree on a piece of G0 fiber 778459 (tools/paris4_piece_check.py, PREREG 10.7 checks 5 and 6), the G0 CPU strip of
the whole fiber (runs/g0, infer_all) and that fiber's AFV polyline.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_erratum_v31.py
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
import score_strips as S  # noqa: E402

FIX = ROOT/'tests'/'fixtures'/'paris4_real'


def line_fiber(name, pts, family='H'):
    pts = np.asarray(pts, np.float64)
    s = S.arclength(pts)
    return S.Fiber(name, family, 'manual', pts, s, {}, pts, s, 0, '00000000-0000-5000-8000-000000000000', 'test',
                   name+'.json')


# ---- 4.8 rule 2: the stratum segment ------------------------------------------------------------------------------

def test_stratum_segment_is_the_one_holding_the_most_stretch_length():
    """v3.1: the segment holding the most total stretch length in the piece (the code), not the segment of the
    piece's longest stretch (v3's text): one 60 sv stretch on X against 40 + 40 sv on Y gives Y."""
    x = np.arange(0., 1201.)
    f = line_fiber('f', np.c_[x, np.zeros_like(x), np.zeros_like(x)])
    ws = [dict(id='a', s0=300., s1=360., segment='X'), dict(id='b', s0=380., s1=420., segment='Y'),
          dict(id='c', s0=440., s1=480., segment='Y')]
    (piece,) = S.plan_pieces(f, ws)
    assert piece['segment'] == 'Y' and piece['windows'] == ['a', 'b', 'c']
    tie = [dict(id='a', s0=300., s1=360., segment='Z'), dict(id='b', s0=380., s1=440., segment='Y')]
    assert S.plan_pieces(f, tie)[0]['segment'] == 'Y'  # a tie goes to the first name in sorted order


def test_strata_reading_check_reproduces_the_frozen_plan():
    """tools/strata_reading_check.py reproduced the frozen scan and subsample with the code's strata; the literal
    reading of v3's text differs in the 13 pieces PREREG 10.7 names (P0169 a tie) and gives a 9.1 GiB cap."""
    rep = ROOT/'results'/'erratum_v31'/'strata_reading_check.json'
    plan = ROOT/'results'/'matched_stretches_secondary_run40.json'
    if not rep.exists() or not plan.exists():
        pytest.skip('plan or check not here')
    r = json.loads(rep.read_text())
    assert r['plan_sha256'] == S.sha256_file(plan)
    c = r['checks']
    assert c['code_scan_equals_frozen'] and c['code_run_equals_frozen'] and c['code_cap'] == 9.6
    assert c['cached_at_plan_time'] == c['cached_at_plan_time_plan'] == 35
    named = ['P0002', 'P0021', 'P0034', 'P0074', 'P0084', 'P0092', 'P0093', 'P0117', 'P0119', 'P0126', 'P0128',
             'P0133', 'P0169']
    assert [d['piece'] for d in r['differing']] == named
    assert [t['piece'] for t in r['ties']] == ['P0169']
    assert r['literal']['cap_gib'] == 9.1 and r['literal']['run']['pieces'] == 45 and r['literal']['run']['fibers'] == 36
    # recomputed here from the plan alone: the literal segment differs exactly in those pieces
    P = json.loads(plan.read_text())
    st = {s['id']: s for s in P['stretches']}
    got = []
    for p in P['pieces']:
        sts = [st[i] for i in p['windows']]
        best = max(s['s1']-s['s0'] for s in sts)
        top = sorted({s['segment'] for s in sts if s['s1']-s['s0'] == best})
        if top[0] != p['segment'] or len(top) > 1:
            got.append(p['id'])
    assert got == named


# ---- rules in the code now stated (PREREG 5.2, 5.5, 7) --------------------------------------------------------------

def test_paired_bootstrap_seed_is_20261029():
    rng = np.random.default_rng(1)
    a = [np.abs(rng.normal(0, 1, 30+i)) for i in range(12)]
    b = [x+rng.normal(.3, .2, len(x)) for x in a]
    d = S.paired_delta(a, b, seed=S.SEED)
    draws = np.random.default_rng(20261029).integers(0, len(a), size=(S.BOOTSTRAP, len(a)))
    cnt = np.array([len(x) for x in a], float)
    dm = (np.array([x.sum() for x in b])[draws].sum(1)-np.array([x.sum() for x in a])[draws].sum(1))/cnt[draws].sum(1)
    assert d['ci'] == [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))]
    assert S.SEED == 20261028 and S.SEED+2 == 20261030  # crossing_agreement's bootstrap: default_rng(seed + 2)


def test_paired_delta_keeps_fibers_with_25_nodes_covered_by_both():
    ents = {}
    for i in range(4):
        e = np.full((2, 20), .5)
        base = np.full((2, 20), 1.5)
        if i == 0:
            base[:, 12:] = np.nan  # 24 nodes covered by both: this fiber drops out of the paired comparison
        ents[f'f{i}'] = dict(fiber=f'f{i}', family='H', strip=f's{i}', stretches=[f'S{i}'],
                             err=dict(strip=e, frame_plane=base, lasagna_plane=base.copy()), expected=None,
                             valid=np.ones((2, 20), bool), fiber_floor=np.full(2, .1), lasagna_angle=np.full(2, 1.),
                             row_seg=np.zeros(2, int), row_stretch=np.full(2, i), notes=[])
    agg = S.aggregate(ents)
    assert agg['scored_fibers'] == 4  # 40 strip nodes each
    assert agg['frame_plane']['available_fibers'] == 3 and agg['frame_plane']['delta']['fibers'] == 3


def test_row_centre_falls_back_to_the_mean_valid_node():
    rows = np.zeros((3, 5, 3))
    rows[..., 0] = np.arange(5)[None]
    rows[..., 1] = np.arange(3)[:, None]
    valid = np.ones((3, 5), bool)
    valid[1, 2] = False           # centre node invalid: the mean of the row's valid nodes
    valid[2] = False              # no valid node: NaN (not scored)
    st = S.Strip('t', 't', np.where(valid[..., None], rows, np.nan), valid, None, None, {}, 1., [])
    c = S.row_centres(st)
    assert np.allclose(c[0], [2, 0, 0]) and np.allclose(c[1], [np.mean([0, 1, 3, 4]), 1, 0])
    assert np.isnan(c[2]).all()


def test_find_crossings_takes_only_the_nearest_v_sample():
    """PREREG 7 (v3.1): one query per H sample for the nearest V sample within the tolerance (k = 1). V2 is within
    6 sv of the H fiber, but V1 is nearer to every H sample V2 is near, so the pair (H, V2) is not found."""
    t = np.arange(-50., 51.)
    h = line_fiber('h', np.c_[t, np.zeros_like(t), np.zeros_like(t)], 'H')
    v1 = line_fiber('v1', np.c_[np.zeros_like(t), np.full_like(t, 2.), t], 'V')
    v2 = line_fiber('v2', np.c_[np.zeros_like(t), np.full_like(t, 5.), t], 'V')
    pairs = {(c['h_fiber'], c['v_fiber']) for c in S.find_crossings([h, v1, v2], 6.)}
    assert pairs == {('h', 'v1')}
    assert {(c['h_fiber'], c['v_fiber']) for c in S.find_crossings([h, v2], 6.)} == {('h', 'v2')}


def test_geometry_checks_record():
    """tools/erratum_geometry_checks.py: on PHerc0332 the k = 1 rule finds exactly the pairs an exhaustive search
    finds (48 at 6 sv, 4 at 3 sv); the piece-tube counts PREREG 10.7 states."""
    rep = ROOT/'results'/'erratum_v31'/'geometry_checks.json'
    if not rep.exists():
        pytest.skip('check not here')
    r = json.loads(rep.read_text())
    for tol, n in (('6', 48), ('3', 4)):
        c = r['crossing_rule_0332'][tol]
        assert c['pairs_exact'] == c['pairs_k1'] == n and c['missed_by_k1'] == 0 and not c['k1_approach_larger']
    tube = {t['set']: t for t in r['piece_tube']['sets']}
    assert tube['primary']['totals']['sphere'] == 0
    assert tube['secondary']['totals']['box_scoring'] == 121 and tube['secondary']['pieces_box_scoring'] == 6
    assert tube['0332']['totals']['box_scoring'] == 112 and tube['0332']['pieces_box_scoring'] == 6


def test_p0104_one_head_at_1_01_sv_all_others_within_0_235():
    plan = ROOT/'results'/'matched_stretches_secondary_run40.json'
    afv = ROOT/'data'/'matched_0139_secondary_run40.afv'
    full_afv = ROOT.parent/'data'/'human_afv'/'PHerc0139_human_controlled.afv'
    if not (plan.exists() and afv.exists() and full_afv.exists()):
        pytest.skip('data not here')
    import make_piece_afv as MP
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import AFVFibers
    from vesuvius.neural_tracing.tracers.shared.geometry import interp_at
    p = next(p for p in json.loads(plan.read_text())['pieces'] if p['id'] == 'P0104')
    with open(ROOT/'results'/'runplan_0139_secondary_run40.tsv') as fh:
        r = next(r for r in csv.DictReader(fh, delimiter='\t') if r['piece'] == 'P0104')
    a, full = AFVFibers(str(afv), 8., split='all'), AFVFibers(str(full_afv), 8., split='all')
    geo = a[a.id_to_index[p['afv_id']]]
    arcs = MP.infer_heads(float(r['arc_start_patch_vox']), float(r['arc_end_patch_vox']), float(geo.length))
    k0, k1 = p['head_k']
    parent = full[full.id_to_index[p['parent_afv_id']]]
    ref = interp_at(np.asarray(parent.points), np.asarray(parent.s), 12.+8.*np.arange(k0, k1+1))
    dev = np.linalg.norm(interp_at(np.asarray(geo.points), np.asarray(geo.s), arcs)-ref, axis=1)*2.
    assert len(dev) == 187 and abs(dev.max()-1.01) < 0.01
    assert np.sort(dev)[-2] <= 0.235 and np.sort(dev)[-2] > 0.23


# ---- the scorer on REAL infer.py output (Paris 4 fixture; PREREG 10.7 checks 5 and 6) ---------------------------------

def fixture_fiber():
    d = json.loads((FIX/'fiber_778459.json').read_text())
    line = np.asarray(d['xyz_l0'], np.float64)/S.SCORE_UNIT_BASE
    pts, s = S.resample(line, S.SAMPLE_STEP)
    f = S.Fiber('g0_778459', d['family'], 'afv', pts, s, {}, line, S.arclength(line), 0,
                '00000000-0000-5000-8000-000000000000', 'test', 'g0_778459')
    return f, d


def canonical(st):
    rows, valid = st.rows, st.valid
    if st.trace.get('fiber_reversed'):
        rows, valid = rows[::-1], valid[::-1]
    if st.trace.get('across_reversed'):
        rows, valid = rows[:, ::-1], valid[:, ::-1]
    return rows, valid


def test_real_infer_py_output_is_read_with_its_frame_plane():
    """The scorer's reading path (discover_strips, load_strip, frames_from_decisions) on a real infer.py output: the
    stitched decisions reproduce every strip node, so the frame-plane baseline and expected_error are available."""
    dirs = S.discover_strips(FIX/'pieces')
    assert [Path(d).relative_to(FIX).as_posix() for d in dirs] == ['pieces/QA/strip']
    st = S.load_strip(dirs[0])
    assert st.name == 'QA' and st.notes == [] and st.rows.shape == (64, 25, 3) and int(st.valid.sum()) == 600
    assert st.frame_rows is not None and np.isfinite(st.frame_rows[st.valid]).all()
    assert st.expected is not None and np.isfinite(st.expected[st.valid]).all()
    assert abs(st.column_step-5.) < .1 and st.trace['blend'] == 16. and st.trace['trace_per_patch'] == .5
    z = np.load(S.find_decisions(dirs[0]))
    assert len(z['arcs']) == 8 and 'expected_error_vox' in z.files
    o, of, _ = S.orient_decisions(z['heights'], z['frames'])
    rows, w, _ = S.stitch(o, of, z['positions'], z['arcs'], z['f'], z['v'], 16.)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(rows[w > 0]/.5).query(st.rows[st.valid], k=1)
    assert d.max() < 0.005  # sv; the frame-plane pairing needs <= 0.05 sv for >= 99 % of the nodes


def test_real_infer_py_output_scores_end_to_end():
    """score_strips + aggregate on the real piece strip, against the G0 full-fiber strip of the same fiber standing in
    for a segment: every window row scored, the strip within float rounding of the full run, the frame plane there."""
    fiber, d = fixture_fiber()
    seg = S.SegmentSource('g0strip_778459', FIX/'g0_fiber_778459', 4.)
    s0, s1 = d['window_sv']
    st = dict(id='WA', fiber=fiber.name, s0=s0, s1=s1, segment=seg.name, _n=0, family=d['family'], annotator='g0')
    strip = S.load_strip(FIX/'pieces'/'QA'/'strip')
    entries, skips = S.score_strips([strip], [fiber], [st], [seg], log=lambda *a: None,
                                    pieces=[dict(id='QA', windows=['WA'])])
    assert not skips and list(entries) == ['QA']
    e = entries['QA']
    assert e['stretches'] == ['WA'] and e['assoc_median_sv'] < S.ASSOC_MAX_SV
    assert int(np.isfinite(e['err']['strip']).sum()) == 150 and int(np.isfinite(e['err']['frame_plane']).sum()) == 150
    assert np.nanmax(np.abs(e['err']['strip'])) < 0.01
    agg = S.aggregate(entries)
    assert agg['scored_fibers'] == 1 and agg['frame_plane']['available_fibers'] == 1
    assert 1. < agg['frame_plane']['stats']['point']['mean'] < 4.  # 2.14 sv: the frame plane is not the sheet


def test_real_piece_rows_equal_the_full_run():
    """PREREG 10.7 check 6 on the fixture: the piece's window rows equal the full-fiber run's (G0) within float
    rounding; rows that take a decision outside the piece do not."""
    fiber, d = fixture_fiber()
    pr, pv = canonical(S.load_strip(FIX/'pieces'/'QA'/'strip'))
    gr, gv = canonical(S.load_strip(FIX/'g0_fiber_778459'))
    k0, k1 = d['head_k']
    off = 4*k0  # piece row r is full row r + 8 k0 / 2
    arc = 12.-35.25+(np.arange(len(pr))+off)*2.  # full-run arclength of the piece's rows (trace voxels)
    full_set = (arc > 12.+8.*(k0-1)+16.) & (arc < 12.+8.*(k1+1)-16.)
    dist = np.linalg.norm(pr-gr[off:off+len(pr)], axis=-1)
    both = pv & gv[off:off+len(pr)]
    assert both.sum() == 600
    assert dist[both & full_set[:, None]].max() < 0.05     # measured 0.038 sv
    assert dist[both & ~full_set[:, None]].max() > 1.      # measured 1.20 sv: edge rows, never scored


# ---- FREEZE v3.1 pins the code that makes the strips -----------------------------------------------------------------

def test_freeze_pins_the_villa_worktree_patches_and_checkpoint():
    import write_freeze as W
    assert len(W.VILLA_CHANGED) == 9 and len(W.CHECKPOINT_FILES) == 4 and len(W.PINS) == 9+6+4
    for rel in W.PINS:
        if rel.startswith('../patches/'):  # shipped with the scorer
            assert (ROOT/rel).resolve().is_file(), rel
    if not all((ROOT/rel).resolve().is_file() for rel in W.PINS):
        pytest.skip('the pinned villa worktree or the checkpoint snapshot is not here (README, How to reproduce)')
    if shutil.which('git') is None:
        pytest.skip('git not here')
    head, changed = W.worktree_state()
    assert head == W.VILLA_BASE and changed == sorted(W.VILLA_CHANGED)
    tracked = subprocess.run(['git', '-C', str(W.VILLA), 'diff', '--name-only'], capture_output=True, text=True,
                             check=True).stdout.split()
    assert sorted(tracked) == sorted(W.VILLA_CHANGED[:7])
