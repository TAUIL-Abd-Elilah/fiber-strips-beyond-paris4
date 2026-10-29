"""Tests of fibers_json_to_afv.py on the downloaded sample fibers (data/samples/0139, data/samples/0332).

    python -m pytest -q tools/test_fibers_json_to_afv.py      (or: python tools/test_fibers_json_to_afv.py)
"""
import copy
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fibers_json_to_afv as conv  # noqa: E402
from vesuvius.neural_tracing.tracers.shared.annotations.afv import AFVFibers, read_afv_polylines  # noqa: E402

SAMPLES = HERE.parent/'data'/'samples'
VOL = dict(
    **{'0139': ('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/'
                '20260102150214-2.399um-0.2m-78keV-masked.zarr', [76953, 26511, 26511])},
    **{'0332': ('s3://vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr',
                [33592, 15761, 15761])})


def _span(raw, span):
    line = np.asarray(raw['line_points'], np.float64)
    if span == 'full':
        return line
    controls = np.asarray([c['position'] for c in raw['control_points']], np.float64)
    anchors = np.linalg.norm(line[None]-controls[:, None], axis=2).argmin(1)
    return line[anchors.min():anchors.max()+1]


def _by_file(afv):
    return {json.loads(row['annotation'])['source_file']: (row, xyz) for row, xyz in read_afv_polylines(afv)}


def check_scroll(scroll, tmp, span='controlled'):
    url, shape = VOL[scroll]
    afv = Path(tmp)/f'{scroll}_{span}.afv'
    report = conv.convert([SAMPLES/scroll], afv, url, span, shape)
    files = sorted((SAMPLES/scroll).glob('*.json'))
    assert report['converted'] == len(files) == 5, report
    got = _by_file(afv)
    for path in files:
        raw = json.loads(path.read_text())
        row, xyz = got[path.name]
        assert np.array_equal(xyz, _span(raw, span)), path.name  # bit for bit, native L0 xyz
        a = json.loads(row['annotation'])
        hv = raw['hv_classification']
        assert row['family'] == (hv['manual_tag'] if hv.get('manual_tag') in ('H', 'V') else hv['automatic_tag'])
        if raw.get('uuid'):
            assert a['uuid'] == row['name'] == raw['uuid'].lower() and a['uuid_source'] == 'json'
        else:
            assert a['uuid_source'] == 'uuid5(file)'
        assert row['id'] == int(a['uuid'].replace('-', '')[:13], 16)
    with sqlite3.connect(afv) as c:
        meta = {k: json.loads(v) for k, v in c.execute('SELECT key,value FROM metadata')}
    assert meta['frame']['vc_open_data_source_coordinate_level'] == 0
    assert meta['root']['coordinate_base_shape_zyx'] == shape
    fibers = AFVFibers(afv, 8., split='all')  # as the strip inference opens it
    for i in range(len(fibers)):
        f = fibers[i]
        assert f.length > 0 and f.points.shape[1] == 3
        # AFVFibers divides L0 by the grid scale (8): trace voxels; the strip is written in patch voxels (L0/4).
        row, xyz = got[json.loads(fibers.connection().execute(
            'SELECT annotation FROM fibers WHERE id=?', (fibers.catalog[i][0],)).fetchone()[0])['source_file']]
        assert np.allclose(f.points[0], xyz[0]/8., atol=1e-9) or np.allclose(f.points[-1], xyz[-1]/8., atol=1e-9)
    return afv, got


def test_0139(tmp_path):
    check_scroll('0139', tmp_path)


def test_0332(tmp_path):
    check_scroll('0332', tmp_path)


def test_full_span(tmp_path):
    for scroll in VOL:
        check_scroll(scroll, tmp_path, 'full')


def test_round_trip(tmp_path):
    """JSON -> AFV -> JSON (minimal v4) -> AFV: same ids, uuids, families and polylines, bit for bit."""
    for scroll in VOL:
        afv, got = check_scroll(scroll, tmp_path)
        back = conv.afv_to_json(afv, tmp_path/f'{scroll}_json')
        assert len(back) == 5
        afv2 = tmp_path/f'{scroll}_again.afv'
        conv.convert([tmp_path/f'{scroll}_json'], afv2, VOL[scroll][0], 'controlled', VOL[scroll][1])
        again = _by_file(afv2)
        assert set(again) == set(got)
        for name, (row, xyz) in got.items():
            row2, xyz2 = again[name]
            assert np.array_equal(xyz, xyz2)
            assert (row['id'], row['name'], row['family']) == (row2['id'], row2['name'], row2['family'])


def test_rejects_wrong_volume(tmp_path):
    url, shape = VOL['0332']
    try:
        conv.convert([SAMPLES/'0139'], tmp_path/'x.afv', url, 'controlled', shape)
    except ValueError as error:
        assert 'is not the volume' in str(error)
    else:
        raise AssertionError('0139 fibers accepted as 0332')


def test_skips_untagged_and_seed_only(tmp_path):
    src = sorted((SAMPLES/'0332').glob('*.json'))[0]
    raw = json.loads(src.read_text())
    d = tmp_path/'odd'
    d.mkdir()
    untagged = copy.deepcopy(raw)
    untagged['hv_classification'] = dict(manual_tag='', automatic_tag='')
    untagged['uuid'] = '00000000-0000-4000-8000-000000000001'
    (d/'untagged.json').write_text(json.dumps(untagged))
    seed = copy.deepcopy(raw)
    seed['control_points'] = [dict(position=seed['line_points'][0])]
    seed['uuid'] = '00000000-0000-4000-8000-000000000002'
    (d/'seed.json').write_text(json.dumps(seed))
    manual = copy.deepcopy(raw)
    manual['hv_classification'] = dict(raw['hv_classification'], manual_tag='V' if raw['hv_classification']['automatic_tag'] == 'H' else 'H')
    manual['uuid'] = '00000000-0000-4000-8000-000000000003'
    (d/'manual.json').write_text(json.dumps(manual))
    report = conv.convert([d], tmp_path/'odd.afv', *VOL['0332'][:1], 'controlled', VOL['0332'][1])
    assert report['converted'] == 1 and len(report['skipped']) == 2 and report['family_disagreements'] == 1
    (row, _), = read_afv_polylines(tmp_path/'odd.afv')
    assert row['family'] == manual['hv_classification']['manual_tag']


if __name__ == '__main__':
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            with tempfile.TemporaryDirectory() as tmp:
                fn(Path(tmp))
            print('ok', name)
