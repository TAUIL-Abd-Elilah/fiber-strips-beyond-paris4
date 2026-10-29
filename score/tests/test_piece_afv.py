"""Integration checks (v3) of the planned AFVs and run plans (tools/make_piece_afv.py) on the real PHerc0139/0332 data:

- the pieces are the JSONs' own line points bit for bit, inside the controlled span, with the converter's identity;
- P1: the decisions villa's own infer.py head code makes for each run-plan row (``--arc`` as written) are exactly
  the planned ones (count and positions), for every piece of every plan;
- P2: those heads lie within 1 sv of the full infer_all run's heads (12 + 8k trace voxels on the full AFV's
  kink-repaired AFVFibers geometry), except in the pieces named in PREREG v3 4.6 (HEAD_DEV_EXCEPTIONS: the cut piece's
  kink repair differs from the full fiber's inside the head span), and each piece holds every full-run head within
  the blend reach of its windows;
- the full-run map (tools/full_run_map.py) is AFVFibers' own geometry, and the full AFV is the same controlled span.

Skipped when the data, the AFVs or the checkpoint are not on this machine.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_piece_afv.py
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent/'tools'))
sys.path.insert(0, str(ROOT/'tools'))
import score_strips as S  # noqa: E402

conv = pytest.importorskip('fibers_json_to_afv')  # puts villa_6f2/vesuvius/src first on sys.path
from vesuvius.neural_tracing.tracers.shared.annotations.afv import AFVFibers, read_afv_polylines  # noqa: E402
from vesuvius.neural_tracing.tracers.shared.geometry import interp_at  # noqa: E402
from full_run_map import fiber_map  # noqa: E402

FULL = {'0139': ROOT.parent/'data'/'human_afv'/'PHerc0139_human_controlled.afv',
        '0332': ROOT.parent/'data'/'human_afv'/'PHerc0332_human_controlled.afv'}
MAPS = {'0139': 'data/full_run_map_0139.npz', '0332': 'data/full_run_map_0332.npz'}
SETS = [('results/matched_stretches.json', 'data/fibers0139', 'data/matched_0139.afv', 'results/runplan_0139_primary.tsv',
         '0139'),
        ('results/matched_stretches_secondary_run40.json', 'data/fibers0139', 'data/matched_0139_secondary_run40.afv',
         'results/runplan_0139_secondary_run40.tsv', '0139'),
        ('results/crossings_0332_plan.json', 'data/fibers0332', 'data/crossings_0332.afv',
         'results/runplan_0332_crossings.tsv', '0332')]
CHECKPOINT = (ROOT.parent/'hf_home'/'hub'/'models--scrollprize--fiber_strip_small_v1'/'snapshots' /
              '7bd163a22c839ff312879ae1916195d0c9729c0e'/'last.pt')
HEAD_TOL_SV = 1.0
# PREREG v3 4.6: pieces whose own kink repair differs from the full fiber's at one place inside the head span; the
# heads after it are this far (max, sv) from the full run's. Every other piece is within HEAD_TOL_SV.
HEAD_DEV_EXCEPTIONS = {'results/runplan_0139_primary.tsv': {},
                       'results/runplan_0139_secondary_run40.tsv': {'P0047': 3.49, 'P0104': 1.01, 'P0114': 7.81},
                       'results/runplan_0332_crossings.tsv': {'C0001': 5.73, 'C0004': 1.26}}


def need(*paths):
    for p in paths:
        if not (ROOT/p).exists():
            pytest.skip(f'{p} not here')


def read_runplan(path):
    with open(ROOT/path) as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


# ---- villa's own infer.py head code ---------------------------------------------------------------------------------

class _Captured(Exception):
    pass


@pytest.fixture(scope='module')
def infer_heads():
    """A function (afv, fiber id, ARC_START, ARC_END strings) -> (decision arclengths in trace voxels, the fiber) as
    villa's sheet_strip/infer.py main() computes them, run unchanged up to the model call: its StripPredictor is
    replaced by a stand-in carrying the checkpoint's own vol_spec (grid_scale, ct_grid_scale -> trace_per_patch),
    whose predict() hands back the arcs. Arguments as in the run plan (PREREG 2)."""
    if not CHECKPOINT.exists():
        pytest.skip('checkpoint not here')
    import torch
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip import infer as INF
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolumeSpec
    assert 'villa_6f2' in Path(INF.__file__).parts  # the pinned worktree's infer.py
    spec =CTVolumeSpec.from_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=False)['vol_spec'])
    assert spec.grid_scale == 8. and spec.ct_grid_scale/spec.grid_scale == 0.5

    class Stand:
        def __init__(self, *args, **kwargs):
            self.cfg, self.ck, self.data, self.spec = None, {}, {}, spec
            self.trace_per_patch = spec.ct_grid_scale/spec.grid_scale
            self.f, self.v = np.arange(-40., 41.), np.arange(-60., 61.)  # only f's spacing is read before predict
            self.vol = None

        def predict(self, fiber, arcs, on_items=None, log=None):
            raise _Captured(np.asarray(arcs, np.float64), fiber)

    def run(afv, fiber_id, arc_start, arc_end, tmp):
        real = INF.StripPredictor
        INF.StripPredictor = Stand
        try:
            INF.main(['--afv', str(afv), '--fiber', str(fiber_id), '--arc', arc_start, arc_end, '--out', str(tmp),
                      '--checkpoint', 'unused', '--device', 'cpu', '--index', 'none', '--no-render', '--step', '8',
                      '--blend', '16', '--trim', '75'])
        except _Captured as c:
            return c.args
        finally:
            INF.StripPredictor = real
        raise AssertionError('infer.py did not reach the model call')
    return run


@pytest.mark.parametrize('plan, fibers, afv, runplan, scroll', SETS)
def test_pieces_are_json_points_bit_for_bit(plan, fibers, afv, runplan, scroll):
    need(plan, fibers, afv, runplan)
    P = json.loads((ROOT/plan).read_text())
    pieces = {p['afv_id']: p for p in P['pieces']}
    got = {row['id']: (row, xyz) for row, xyz in read_afv_polylines(ROOT/afv)}
    assert set(got) == set(pieces)
    for fid, (row, xyz) in got.items():
        p = pieces[fid]
        raw = json.loads((ROOT/fibers/p['parent_file']).read_text())
        line = np.asarray(raw['line_points'], np.float64)
        A, B = p['line_index_range']
        assert np.array_equal(xyz, line[A:B+1])  # bit for bit
        ann = json.loads(row['annotation'])
        assert ann['piece']['id'] == p['id'] and ann['family'] == p['family'] == row['family']
        ca, cb = ann['piece']['controlled_span']
        assert ca <= A < B <= cb
        assert row['name'] == p['uuid'] and fid == int(p['uuid'].replace('-', '')[:13], 16)
    with sqlite3.connect(f'file:{(ROOT/afv).resolve()}?mode=ro', uri=True) as c:
        meta = {k: json.loads(v) for k, v in c.execute('SELECT key,value FROM metadata')}
    assert meta['complete'] is True and meta['fiber_count'] == len(pieces)
    rows = read_runplan(runplan)
    assert {int(r['afv_id']) for r in rows} == set(pieces)
    for r in rows:
        p = pieces[int(r['afv_id'])]
        assert r['run'] == str(bool(p.get('run', True)))
        assert int(r['decisions']) == int(r['decisions_planned']) == p['decisions']
        assert [int(r['head_k_first']), int(r['head_k_last'])] == p['head_k']


@pytest.mark.parametrize('plan, fibers, afv, runplan, scroll', SETS)
def test_infer_py_makes_exactly_the_planned_heads_on_the_full_run_grid(plan, fibers, afv, runplan, scroll,
                                                                       infer_heads, tmp_path):
    """P1 and P2: villa's own head code on each run-plan row gives exactly the planned number of decisions, and
    every one lies within HEAD_TOL_SV of the full run's head k (full AFV, AFVFibers geometry), except in the named
    pieces, whose deviations are the ones PREREG v3 states."""
    need(plan, afv, runplan)
    if not FULL[scroll].exists():
        pytest.skip('full AFV not here')
    P = json.loads((ROOT/plan).read_text())
    pieces = {p['id']: p for p in P['pieces']}
    full = AFVFibers(str(FULL[scroll]), 8., split='all')
    worst, total, over = 0., 0, {}
    for r in read_runplan(runplan):
        p = pieces[r['piece']]
        arcs, fiber = infer_heads(ROOT/afv, r['afv_id'], r['arc_start_patch_vox'], r['arc_end_patch_vox'], tmp_path)
        k0, k1 = p['head_k']
        assert len(arcs) == p['decisions'] == k1-k0+1, r['piece']  # exact (P1)
        assert np.allclose(np.diff(arcs), 8.)
        parent = full[full.id_to_index[p['parent_afv_id']]]
        n_full = len(np.arange(12., parent.length+1e-9, 8.))
        assert 0 <= k0 <= k1 < n_full
        ref = interp_at(np.asarray(parent.points), np.asarray(parent.s), 12.+8.*np.arange(k0, k1+1))
        got = interp_at(np.asarray(fiber.points), np.asarray(fiber.s), arcs)
        dev = np.linalg.norm(got-ref, axis=1)*2.  # sv
        assert abs(float(r['max_head_dev_sv'])-dev.max()) < 1e-3
        if dev.max() > HEAD_TOL_SV:
            over[r['piece']] = float(dev.max())
        else:
            worst = max(worst, float(dev.max()))
        total += len(arcs)
    assert total == sum(p['decisions'] for p in pieces.values())
    named = HEAD_DEV_EXCEPTIONS[runplan]
    assert set(over) == set(named), (sorted(over), sorted(named))  # P2: the exceptions are exactly the named ones
    for k, v in named.items():
        assert abs(over[k]-v) < 0.01, (k, over[k], v)
    print(f'{runplan}: {len(pieces)} pieces, {total} decisions, max head deviation {worst:.3f} sv outside {over}')


@pytest.mark.parametrize('plan, fibers, afv, runplan, scroll', SETS)
def test_pieces_hold_every_full_run_head_within_the_blend_of_their_windows(plan, fibers, afv, runplan, scroll):
    """Each window's rows get every decision a full run would give them: all full-run heads within BLEND_SV (in the
    full run's arclength) of the window are among the piece's heads."""
    need(plan, fibers, MAPS[scroll])
    P = json.loads((ROOT/plan).read_text())
    windows = {w['id']: w for w in P.get('stretches', P.get('windows', []))}
    names = {p['fiber'] for p in P['pieces']}
    space = 'PHerc0332/20251211183505@L0' if scroll == '0332' else 'PHerc0139/20260102150214@L0'
    fibers_, _ = S.load_fibers(ROOT/fibers, names=names, space=space)
    maps = S.load_full_run_maps(ROOT/MAPS[scroll], fibers_)
    for p in P['pieces']:
        m = maps[p['fiber']]
        k0, k1 = p['head_k']
        np.testing.assert_allclose(p['heads_sv'], m.to_raw(m.head_full([k0, k1])), atol=1e-9)
        for wid in p['windows']:
            w = windows[wid]
            A0, A1 = m.to_full(w['s0']), m.to_full(w['s1'])
            S_all = m.head_full(np.arange(len(np.arange(24., m.length_full+2e-9, 16.))))
            need_k = np.flatnonzero((S_all > A0-S.BLEND_SV) & (S_all < A1+S.BLEND_SV))
            assert len(need_k) and k0 <= need_k.min() and need_k.max() <= k1, (p['id'], wid)


@pytest.mark.parametrize('scroll', ['0139', '0332'])
def test_full_run_map_is_afvfibers_geometry_and_same_controlled_span(scroll):
    need(MAPS[scroll])
    if not FULL[scroll].exists():
        pytest.skip('full AFV not here')
    z = np.load(ROOT/MAPS[scroll])
    full = AFVFibers(str(FULL[scroll]), 8., split='all')
    polys = {row['id']: (row, xyz) for row, xyz in read_afv_polylines(FULL[scroll])}
    assert sorted(int(i) for i in z['ids']) == sorted(polys)
    fibers_dir = ROOT/('data/fibers0139' if scroll == '0139' else 'data/fibers0332')
    for n, fid in enumerate(z['ids'][::9]):
        fid = int(fid)
        row, xyz = polys[fid]
        ann = json.loads(row['annotation'])
        s_raw, s_rep, pts, _ = fiber_map(xyz)
        geo = full[full.id_to_index[fid]]
        assert np.array_equal(pts, np.asarray(geo.points))
        raw_bp, full_bp = z[f'm{fid}_raw'], z[f'm{fid}_full']
        np.testing.assert_allclose(np.interp(s_rep*2., full_bp, raw_bp), s_raw*2., atol=1e-6)
        # the full AFV is the JSON's controlled span, bit for bit, starting where the scorer's starts
        raw = json.loads((fibers_dir/ann['source_file']).read_text())
        A, B = ann['line_index_range']
        assert np.array_equal(xyz, np.asarray(raw['line_points'], np.float64)[A:B+1])
        space = raw.get('vc_open_data_coordinate_space') or ('PHerc0332/20251211183505@L0' if scroll == '0332'
                                                             else 'PHerc0139/20260102150214@L0')
        f = S.parse_fiber(raw, Path(ann['source_file']).stem, space=space)
        assert f.line_offset == A and abs(f.length-s_raw[-1]*2.) < 1e-6 and f.afv_id == fid


def test_scorer_identity_matches_the_converter():
    need('data/fibers0139')
    files = sorted((ROOT/'data/fibers0139').glob('*.json'))
    assert len(files) == 620
    for path in files[::7]:
        raw = json.loads(path.read_text())
        f = S.parse_fiber(raw, path.stem)
        if f is None:
            continue
        u, src = conv.fiber_uuid(raw, raw.get('vc_open_data_coordinate_space'), path)
        assert f.uuid == u and f.uuid_source == src


def test_matched_fibers_have_the_full_afv_ids_of_the_setup_run():
    need('results/matched_stretches_secondary_run40.json')
    if not FULL['0139'].exists():
        pytest.skip('setup AFV not here')
    P = json.loads((ROOT/'results/matched_stretches_secondary_run40.json').read_text())
    ids = {row['id']: json.loads(row['annotation'])['source_file'] for row, _ in read_afv_polylines(FULL['0139'])}
    for p in P['pieces']:
        assert ids[p['parent_afv_id']] == p['parent_file']


def test_v3_secondary_subsample_rule():
    """A1: the secondary CT cap is the smallest on the pre-registered grid whose subsample has >= 36 fibers."""
    need('results/matched_stretches_secondary_run40.json')
    plan = json.loads((ROOT/'results/matched_stretches_secondary_run40.json').read_text())['plan']
    caps = plan['caps']
    lo, hi, step = S.CT_CAP_SCAN_0139_SECONDARY
    assert len(caps['ct_cap_scan']) == int(round((hi-lo)/step))+1
    ok = [s for s in caps['ct_cap_scan'] if s['fibers'] >= S.MIN_RUN_FIBERS_SECONDARY]
    assert caps['ct_new_gib'] == (ok[0]['ct_cap_gib'] if ok else hi)
    assert plan['run']['ct_new_gib'] <= caps['ct_new_gib']+1e-9
    assert plan['run']['fibers'] == next(s['fibers'] for s in caps['ct_cap_scan']
                                         if s['ct_cap_gib'] == caps['ct_new_gib'])
