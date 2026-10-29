"""Integration checks of the planned AFVs (tools/make_piece_afv.py) on the real PHerc0139/0332 data: the pieces are the
JSONs' own line points bit for bit, inside the controlled span, with the converter's identity rules, and the run
plan's decisions match the plan. Skipped when the data or the AFVs are not on this machine.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_piece_afv.py
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent/'tools'))
import score_strips as S  # noqa: E402

conv = pytest.importorskip('fibers_json_to_afv')
from vesuvius.neural_tracing.tracers.shared.annotations.afv import AFVFibers, read_afv_polylines  # noqa: E402

SETS = [('results/matched_stretches.json', 'data/fibers0139', 'data/matched_0139.afv', 'results/runplan_0139_primary.tsv'),
        ('results/matched_stretches_secondary_run40.json', 'data/fibers0139', 'data/matched_0139_secondary_run40.afv',
         'results/runplan_0139_secondary_run40.tsv'),
        ('results/crossings_0332_plan.json', 'data/fibers0332', 'data/crossings_0332.afv',
         'results/runplan_0332_crossings.tsv')]


def need(*paths):
    for p in paths:
        if not (ROOT/p).exists():
            pytest.skip(f'{p} not here')


@pytest.mark.parametrize('plan, fibers, afv, runplan', SETS)
def test_pieces_are_json_points_bit_for_bit(plan, fibers, afv, runplan):
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
    fibers_afv = AFVFibers(str(ROOT/afv), 8., split='all')  # what the strip inference opens
    assert len(fibers_afv) == len(pieces)
    with open(ROOT/runplan) as fh:
        rows = list(csv.DictReader(fh, delimiter='\t'))
    assert {int(r['afv_id']) for r in rows} == set(pieces)
    for r in rows:
        p = pieces[int(r['afv_id'])]
        assert abs(int(r['decisions'])-p['decisions']) <= max(2, 0.05*p['decisions'])  # kink repair shortens a bit
        assert r['run'] == str(bool(p.get('run', True)))
        geo = fibers_afv[fibers_afv.id_to_index[p['afv_id']]]
        lo, hi = float(r['arc_start_patch_vox']), float(r['arc_end_patch_vox'])
        assert 24. <= lo + 1e-6 and hi/2. <= geo.length + 1e-6 and lo < hi


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
    full = ROOT.parent/'data'/'human_afv'/'PHerc0139_human_controlled.afv'
    need('results/matched_stretches_secondary_run40.json')
    if not full.exists():
        pytest.skip('setup AFV not here')
    P = json.loads((ROOT/'results/matched_stretches_secondary_run40.json').read_text())
    ids = {row['id']: json.loads(row['annotation'])['source_file'] for row, _ in read_afv_polylines(full)}
    for p in P['pieces']:
        assert ids[p['parent_afv_id']] == p['parent_file']
