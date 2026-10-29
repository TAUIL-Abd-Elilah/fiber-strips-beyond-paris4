"""Tests of tools/package_strips.py: on copies of the real Paris 4 infer.py output (score/tests/fixtures/paris4_real,
placed into a temporary layout like the run's), on synthetic strips with planted local paths in every member type
(JSON values and keys, nested lists, npz string arrays, TIFF tags), and on the frozen plan files (score/results).

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tools/test_package_strips.py

CPU only, no network; the real run's strips are never read.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import tifffile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import package_strips as P  # noqa: E402

FIXTURE = REPO/'score'/'tests'/'fixtures'/'paris4_real'
QA = FIXTURE/'pieces'/'QA'
RESULTS = REPO/'score'/'results'
B = '\\'
QA_AFV_ID, QA_DECISIONS = 1803139159717749, 8
QA_AFV = 'runs'+B+'erratum_v31'+B+'paris4'+B+'paris4_pieces.afv'  # the fixture's own (relative) afv
RUN_AFV = 'C:'+B+'VesuviusTemp'+B+'strips'+B+'score'+B+'data'+B+'matched_0139.afv'  # as run_all.ps1 passes it
# An independent leak check for the tests (not the tool's own patterns): the spec's local-path patterns.
LEAK_TEXT = re.compile(r'(?i)(?<![a-z0-9])[a-z]:[\\/]|users[\\/]+pc|vesuviustemp|miniconda|vesuvius progress prizes'
                       r'|prizetemp|hf_home|(?<![a-z0-9:/.])/[a-z]/')
LEAK_BINARY = re.compile(rb'(?i)users[\\/]+pc|vesuviustemp|miniconda|vesuvius progress prizes|prizetemp|hf_home')

pytestmark = pytest.mark.skipif(not QA.is_dir(), reason='needs score/tests/fixtures/paris4_real')


# ---- helpers ------------------------------------------------------------------------------------------------------

def sha(data):
    return hashlib.sha256(data).hexdigest()


def plant_afv(piece_dir, path):
    """Replace the fixture's relative afv by ``path`` in summary.json and strip/trace.json, byte for byte."""
    for rel in ('summary.json', 'strip/trace.json'):
        f = piece_dir/rel
        data = f.read_bytes()
        old = json.dumps(QA_AFV).encode()
        assert data.count(old) == 1
        f.write_bytes(data.replace(old, json.dumps(path).encode()))


def make_piece(dst, afv=RUN_AFV):
    shutil.copytree(QA, dst)
    if afv is not None:
        plant_afv(dst, afv)
    return dst


def runplan_header():
    return (RESULTS/'runplan_0139_primary.tsv').read_text(encoding='utf-8').splitlines()[0].split('\t')


def write_runplan(path, rows):
    header = runplan_header()
    lines = ['\t'.join(header)]
    for r in rows:
        base = dict(afv_id=str(QA_AFV_ID), arc_start_patch_vox='300.704493', arc_end_patch_vox='413.704493',
                    decisions=str(QA_DECISIONS), decisions_planned=str(QA_DECISIONS), run='True', run_order='0',
                    parent_afv_id='778459', head_k_first='21', head_k_last='28', max_head_dev_sv='0.0')
        base.update(r)
        lines.append('\t'.join(str(base.get(h, '')) for h in header))
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def stretch_plan(pieces):
    """A plan JSON shaped like matched_stretches*.json: pieces [(id, fiber, family, windows, run)]."""
    out = dict(provenance=dict(constants=dict(CROSS_TOL=3.0)), pieces=[], stretches=[])
    for k, (pid, fiber, family, windows, run) in enumerate(pieces):
        out['pieces'].append(dict(afv_id=QA_AFV_ID, uuid=f'00000000-0000-5000-8000-{k:012d}', fiber=fiber,
                                  parent_file=fiber+'.json', parent_uuid=f'11111111-0000-5000-8000-{k:012d}',
                                  parent_afv_id=778459, family=family, annotator=fiber.split('_')[0],
                                  line_index_range=[10, 200], decisions=QA_DECISIONS, windows=windows, id=pid,
                                  run=run))
        for j, w in enumerate(windows):
            out['stretches'].append(dict(id=w, fiber=fiber, segment='20260115000000-w044_2026011522',
                                         s0=1000.0+100*j, s1=1060.0+100*j, length_sv=60.0, length_mm=0.57576,
                                         piece=pid, run=run))
    return out


def crossing_plan():
    """A plan JSON shaped like crossings_0332_plan.json: crossing X0000 between C0000 (H) and C0001 (V)."""
    pieces = [('C0000', 'dj_20260810T113807458_000001', 'H', ['X0000H']),
              ('C0001', 'lt_20261013T095553471_000027', 'V', ['X0000V'])]
    out = dict(provenance=dict(constants=dict(CROSS_TOL=3.0)), pieces=[], windows=[], crossings=[])
    for k, (pid, fiber, family, windows) in enumerate(pieces):
        out['pieces'].append(dict(afv_id=QA_AFV_ID, uuid=f'22222222-0000-5000-8000-{k:012d}', fiber=fiber,
                                  parent_file=fiber+'.json', parent_uuid=f'33333333-0000-5000-8000-{k:012d}',
                                  parent_afv_id=778459, family=family, annotator=fiber.split('_')[0],
                                  line_index_range=[0, 400], decisions=QA_DECISIONS, windows=windows, id=pid,
                                  run=True, segment=''))
        out['windows'].append(dict(id=windows[0], crossing='X0000', fiber=fiber, family=family,
                                   annotator=fiber.split('_')[0], s0=8049.7, s1=8310.3, run=True, piece=pid))
    out['crossings'].append(dict(h_fiber=pieces[0][1], v_fiber=pieces[1][1], closest_approach_sv=2.5,
                                 point_sv=[1.0, 2.0, 3.0], id='X0000', windows=['X0000H', 'X0000V'], z_tercile=0,
                                 run=True))
    return out


def make_plans(score_dir):
    """Synthetic plan files for all three sets, for the fixture's AFV id and decisions."""
    res = score_dir/'results'
    res.mkdir(parents=True)
    write_runplan(res/'runplan_0139_primary.tsv',
                  [dict(piece='P0000', family='H', fiber='kb_20260904T021448778_000085', windows='S0000',
                        segments_or_crossings='20260115000000-w044_2026011522')])
    sec = [('P0003', 'dj_20260904T151437210_000003', 'V', ['S0018', 'S0019'], True),
           ('P0005', 'dj_20260904T160405930_000005', 'H', ['S0026'], True),
           ('P0006', 'dj_20260904T160405930_000005', 'H', ['S0027'], True),
           ('P0007', 'lt_20260918T092404850_000160', 'H', ['S0058'], True),
           ('P0008', 'lt_20260918T092404850_000160', 'H', ['S0079'], False)]
    write_runplan(res/'runplan_0139_secondary_run40.tsv',
                  [dict(piece=p, family=fam, fiber=f, windows=','.join(w), run=str(run),
                        segments_or_crossings='20260115000000-w044_2026011522') for p, f, fam, w, run in sec])
    (res/'matched_stretches_secondary_run40.json').write_text(json.dumps(stretch_plan(sec)), encoding='utf-8')
    (res/'matched_stretches.json').write_text(json.dumps(stretch_plan(
        [('P0000', 'kb_20260904T021448778_000085', 'H', ['S0000'], True)])), encoding='utf-8')
    cp = crossing_plan()
    (res/'crossings_0332_plan.json').write_text(json.dumps(cp), encoding='utf-8')
    write_runplan(res/'runplan_0332_crossings.tsv',
                  [dict(piece=p['id'], family=p['family'], fiber=p['fiber'], windows=p['windows'][0],
                        segments_or_crossings='X0000') for p in cp['pieces']])
    return res


@pytest.fixture
def layout(tmp_path):
    """A run-like layout: score/results (synthetic plans) and score/runs/v3/strips with copies of the fixture:
    primary P0000; secondary P0003 (with an expected_error.tif), P0005 (with a render and a temporary folder),
    P0006 (incomplete: no summary.json), P0007 missing; 0332 C0000, C0001 missing."""
    score = tmp_path/'score'
    plans = make_plans(score)
    strips = score/'runs'/'v3'/'strips'
    make_piece(strips/'primary'/'P0000')
    p3 = make_piece(strips/'secondary'/'P0003')
    tifffile.imwrite(p3/'strip'/'expected_error.tif', np.full((64, 25), 0.75, np.float32), photometric='minisblack')
    p5 = make_piece(strips/'secondary'/'P0005')
    (p5/'render.png').write_bytes(b'\x89PNG\r\n\x1a\n not a real png')
    (p5/'.tifxyz_tmp_abc123').mkdir()
    (p5/'.tifxyz_tmp_abc123'/'x.tif').write_bytes(b'partial')
    shutil.copy(FIXTURE/'fiber_778459.json', p5/'strip'/'fiber_778459.json')  # a raw fiber JSON: never packaged
    p6 = make_piece(strips/'secondary'/'P0006')
    (p6/'summary.json').unlink()
    make_piece(strips/'0332'/'C0000')
    return dict(root=tmp_path, plans=plans, strips=strips, out=tmp_path/'release')


def run(layout, *extra):
    return P.main(['--strips', str(layout['strips']), '--plans', str(layout['plans']), '--out', str(layout['out']),
                   *extra])


def views(name, data):
    """Every text view of a released member, for the independent leak check: raw text, JSON strings, TIFF tag values
    and npz string arrays; and the raw bytes (for the long names only)."""
    texts = []
    if name.endswith('.json') or name.endswith('.md') or name.endswith('.txt'):
        text = data.decode('utf-8')
        texts.append(text)
        if name.endswith('.json'):
            def walk(node):
                if isinstance(node, dict):
                    for k, v in node.items():
                        texts.append(k)
                        walk(v)
                elif isinstance(node, list):
                    for v in node:
                        walk(v)
                elif isinstance(node, str):
                    texts.append(node)
            walk(json.loads(text))
    elif name.endswith('.tif'):
        with tifffile.TiffFile(io.BytesIO(data)) as t:
            for page in t.pages:
                texts += [str(tag.value) for tag in page.tags.values() if isinstance(tag.value, (str, bytes))]
    elif name.endswith('.npz'):
        z = np.load(io.BytesIO(data), allow_pickle=False)
        for k in z.files:
            texts.append(k)
            if z[k].dtype.kind in 'US':
                texts += [v.decode('latin-1') if isinstance(v, bytes) else v for v in z[k].reshape(-1).tolist()]
    return texts


def assert_no_leak(name, data):
    for text in views(name, data):
        assert not LEAK_TEXT.search(text), (name, LEAK_TEXT.search(text).group(0))
    raw = data
    if name.endswith('.npz'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            raw = b''.join(z.read(n) for n in z.namelist())
    for blob in (raw, raw.replace(b'\0', b'')):
        assert not LEAK_BINARY.search(blob), (name, LEAK_BINARY.search(blob).group(0))


# ---- scrub_string -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('text, expected', [
    (RUN_AFV, '<local>'),
    ('C:/VesuviusTemp/strips/data/g0/g0_fibers.afv', '<local>'),
    ('c:/users/pc/x.json', '<local>'),
    ('C:'+B+B+'Users'+B+B+'PC', '<local>'),                                     # doubled backslashes
    ('/c/Users/PC/miniconda3/envs/vesuvius/python.exe', '<local>'),              # MSYS / Git Bash
    ('/mnt/d/PrizeTemp/strips_ct_cache', '<local>'),                             # WSL
    (B+B+'?'+B+'C:'+B+'long'+B+'path.txt', '<local>'),                           # extended-length
    (B+B+'server'+B+'share'+B+'f.json', '<local>'),                              # UNC
    ('file:///C:/Users/PC/x.json', '<local>'),
    ('file:///home/me/x.json', '<local>'),
    ('C%3A%2FUsers%2FPC%2Fx.json', '<local>'),                                   # URL-encoded
    ('D:/Competition/Vesuvius progress prizes/_public/x.json', '<local>'),       # spaces in a folder
    ('D:/Competition/Vesuvius progress prizes', '<local>'),
    ('/Users/someone/file', '<local>'),
    ('cd D:/Competition/Vesuvius progress prizes && python run.py', 'cd <local> && python run.py'),
    ('python C:/a/b.py --out D:'+B+'x'+B+'y --n 3', 'python <local> --out <local> --n 3'),
    ('C:/env/python.exe -I runs/v3/infer_cpu.py', '<local> -I runs/v3/infer_cpu.py'),  # options end a path
    ('D:/My Data/run 2/x.json > D:/out log/y.txt', '<local> > <local>'),
    ('C:/a/b.json, D:/c/d.json', '<local>, <local>'),
    ('File "C:'+B+'a'+B+'b.py", line 42', 'File "<local>", line 42'),
    ('see /tmp/x and more', 'see <local> and more'),
    ('hf_home/hub/models--scrollprize/x.pt', '<local>'),                         # a fragment naming the workspace
    ('cache in VesuviusTemp'+B+'strips', 'cache in <local>'),
    ('HF_HOME=C:'+B+'VesuviusTemp'+B+'strips'+B+'hf_home', '<local>=<local>'),   # the variable's name too
])
def test_scrub_string_replaces_local_paths(text, expected):
    new, spans = P.scrub_string(text)
    assert new == expected
    assert spans
    assert not LEAK_TEXT.search(new)


@pytest.mark.parametrize('text', [
    'hf://scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e/last.pt',
    's3://vesuvius-challenge-open-data/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr',
    'https://dl.ash2txt.org/other/annotators/0139_fibers/',
    'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/x.zarr/2',
    QA_AFV, 'data/fibers0139', 'PHerc0332/20251211183505@L0', 'patch voxels', 'tifxyz', 'Note: e:x', 'a/c/d',
    'e27688aa-5aa7-4ac9-8496-4ff4eaaa6ff2', '',
])
def test_scrub_string_leaves_everything_else(text):
    assert P.scrub_string(text) == (text, [])


# ---- JSON -------------------------------------------------------------------------------------------------------

def test_scrub_json_edits_only_the_strings_with_paths():
    ck = '"hf://scrollprize/fiber_strip_small_v1@7bd/last.pt"'
    raw = ('{\r\n "afv": ' + json.dumps(RUN_AFV) + ',\r\n "n": 1.50, "e": 1e-05,\r\n'
           ' "nested": {"C:/Users/PC/a.json": [1, "/c/x/y", {"deep": ["ok", "D:'+B*2+'PrizeTemp'+B*2+'z"]}]},\r\n'
           ' "escaped": "' + B + 'u0043:' + B*2 + 'Users' + B*2 + 'PC",\r\n'
           ' "argv": ["--afv", "C:/x/y.afv", "--out", "relative/out"],\r\n'
           ' "note": "made in C:/VesuviusTemp/strips/score today",\r\n'
           ' "rel": ' + json.dumps(QA_AFV) + ',\r\n "ck": ' + ck + '\r\n}\r\n')
    new, changed, kept = P.scrub_json(raw.encode())
    expected = ('{\r\n "afv": "<local>",\r\n "n": 1.50, "e": 1e-05,\r\n'
                ' "nested": {"<local>": [1, "<local>", {"deep": ["ok", "<local>"]}]},\r\n'
                ' "escaped": "<local>",\r\n'
                ' "argv": ["--afv", "<local>", "--out", "relative/out"],\r\n'
                ' "note": "made in <local> today",\r\n'
                ' "rel": ' + json.dumps(QA_AFV) + ',\r\n "ck": ' + ck + '\r\n}\r\n')
    assert new.decode() == expected  # every other byte kept: CRLF, 1.50, 1e-05, spacing, escapes
    assert changed == ['afv', 'nested.{<local>}', 'nested.<local>[1]', 'nested.<local>[2].deep[1]', 'escaped',
                       'argv[1]', 'note']
    assert kept == ['argv[3]', 'rel']


def test_scrub_json_on_a_real_villa_trace_with_a_local_path():
    data = (FIXTURE/'g0_fiber_778459'/'trace.json').read_bytes()  # infer_all's own trace.json of the G0 run
    new, changed, kept = P.scrub_json(data)
    assert new == data.replace(b'"C:/VesuviusTemp/strips/data/g0/g0_fibers.afv"', b'"<local>"')
    assert changed == ['afv'] and kept == [] and b'\r\n' in new
    assert P.scan_member('strip/trace.json', new) == [] and P.scan_member('strip/trace.json', data)


def test_scrub_json_without_paths_is_the_same_bytes():
    data = (QA/'summary.json').read_bytes()
    new, changed, kept = P.scrub_json(data)
    assert new is data and changed == [] and kept == ['afv']


def test_scrub_json_refuses_to_merge_keys():
    with pytest.raises(P.ScrubError, match='equal keys'):
        P.scrub_json(json.dumps({'C:/a/x.json': 1, 'D:/b/y.json': 2}).encode())


def test_scrub_json_refuses_non_json():
    with pytest.raises(P.ScrubError, match='not JSON'):
        P.scrub_json(b'{"afv": "C:/x", ')


def test_scrub_text_line_by_line():
    data = ('first C:/Users/PC/x.txt\r\nsecond\r\nsee /c/data/y z\n/c/My Data/file one.txt\n').encode()
    new, changed, _ = P.scrub_text(data)
    assert new == b'first <local>\r\nsecond\r\nsee <local> z\n<local>\n'  # a line that is one path: all of it
    assert changed == ['line 1', 'line 3', 'line 4']


# ---- npz --------------------------------------------------------------------------------------------------------

def npz_bytes(compressed=True, **arrays):
    buf = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(buf, **arrays)
    return buf.getvalue()


def members(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return {n: z.read(n) for n in z.namelist()}


@pytest.mark.parametrize('compressed', [True, False])
def test_scrub_npz_string_arrays_only(compressed):
    rng = np.random.default_rng(0)
    arrays = dict(heights=rng.normal(size=(4, 36, 25)), flipped=np.zeros(4, bool),
                  fiber_path=np.array('C:/VesuviusTemp/strips/score/data/matched_0139.afv'),
                  paths=np.array([['/c/Users/PC/a', 'ok'], ['D:' + B + 'PrizeTemp' + B + 'b', 'C:/x']]),
                  bpaths=np.array([b'C:/Users/PC/miniconda3', b'fine']), short=np.array(['C:/a', 'b']),
                  fortran=np.asfortranarray(np.array([['C:/x/y', 'a'], ['b', 'c']])))
    data = npz_bytes(compressed, **arrays)
    new, changed = P.scrub_npz(data)
    assert sorted(changed) == sorted(['fiber_path', 'paths[0]', 'paths[2]', 'paths[3]', 'bpaths[0]', 'short[0]',
                                      'fortran[0]'])
    before, after = members(data), members(new)
    assert list(before) == list(after)
    assert after['heights.npy'] == before['heights.npy'] and after['flipped.npy'] == before['flipped.npy']
    z = np.load(io.BytesIO(new), allow_pickle=False)
    assert z['fiber_path'].item() == '<local>' and z['fiber_path'].dtype == arrays['fiber_path'].dtype
    assert z['paths'].tolist() == [['<local>', 'ok'], ['<local>', '<local>']]
    assert z['paths'].dtype == arrays['paths'].dtype
    assert z['bpaths'].tolist() == [b'<local>', b'fine'] and z['bpaths'].dtype == arrays['bpaths'].dtype
    assert z['short'].tolist() == ['<local>', 'b'] and z['short'].dtype == np.dtype('<U7')  # widened to fit
    assert z['fortran'].tolist() == [['<local>', 'a'], ['b', 'c']]
    with zipfile.ZipFile(io.BytesIO(new)) as zz, zipfile.ZipFile(io.BytesIO(data)) as zo:
        assert [i.compress_type for i in zz.infolist()] == [i.compress_type for i in zo.infolist()]
    assert P.scan_member('strip.npz', new) == []
    assert_no_leak('strip.npz', new)


def test_scrub_npz_without_strings_is_the_same_bytes():
    data = (QA/'strip.npz').read_bytes()
    new, changed = P.scrub_npz(data)
    assert new is data and changed == []
    assert P.scan_member('strip.npz', data) == []


def test_npz_paths_that_cannot_be_scrubbed_are_found():
    obj = np.empty(2, object)
    obj[:] = ['C:/Users/PC/x', 'y']
    data = npz_bytes(True, obj=obj, f=np.zeros(3))
    new, changed = P.scrub_npz(data)  # never unpickled, so never changed
    assert new is data and changed == []
    assert any('obj.npy' in w for w, _ in P.scan_member('strip.npz', data))
    numeric = np.frombuffer(b'xxVesuviusTemp..', np.float32)  # the name in a float array's bytes
    hits = P.scan_member('strip.npz', npz_bytes(True, a=numeric))
    assert any(m.lower() == 'vesuviustemp' for _, m in hits)
    utf32 = np.array(['ok']), np.frombuffer('VesuviusTemp'.encode('utf-32-le'), np.float32)
    assert P.scan_member('strip.npz', npz_bytes(False, a=utf32[0], b=utf32[1]))  # UTF-32 bytes, NULs dropped


def test_npz_numeric_bytes_like_a_drive_are_not_a_hit():
    blob = b'aaaC:/bbb /c/dd' + b'\0'
    data = npz_bytes(True, a=np.frombuffer(blob, np.float32))
    assert P.scan_member('strip.npz', data) == []


# ---- TIFF -------------------------------------------------------------------------------------------------------

def tiff_bytes(arr=None, **kwargs):
    buf = io.BytesIO()
    tifffile.imwrite(buf, np.zeros((16, 16), np.float32) if arr is None else arr, photometric='minisblack',
                     **kwargs)
    return buf.getvalue()


@pytest.mark.parametrize('kwargs, tag', [
    (dict(description='made from C:' + B + 'Users' + B + 'PC' + B + 'x.afv', metadata=None), '270'),
    (dict(extratags=[(315, 's', 0, '/c/Users/PC/me', True)]), '315'),                 # Artist
    (dict(extratags=[(65000, 's', 0, 'D:/PrizeTemp/strips_ct_cache', True)]), '65000'),  # private ASCII tag
    (dict(extratags=[(65001, 's', 0, 'C:/', True)]), '65001'),                        # inline (4 bytes)
    (dict(extratags=[(65002, 'B', 14, b'E:/data/x.bin\x00', True)]), '65002'),          # BYTE tag
    (dict(software='C:/Users/PC/miniconda3/python.exe'), '305'),
])
def test_tiff_tag_paths_are_found(kwargs, tag):
    hits = P.scan_member('strip/x.tif', tiff_bytes(**kwargs))
    assert any(f'tag {tag} ' in w for w, _ in hits), hits


def test_tiff_pixel_bytes_like_a_drive_are_not_a_hit():
    blob = b'aaaC:/bbb /c/dd "/d/x' + b'\0'*3
    arr = np.frombuffer(blob, np.float32).reshape(1, -1).copy()
    data = tiff_bytes(arr, compression=None)
    assert blob[:-3] in data  # the bytes are in the file as written
    assert P.scan_member('strip/x.tif', data) == []
    named = np.frombuffer(b'xxUsers/PC/xx', np.uint8).reshape(1, -1).copy()
    assert P.scan_member('strip/mask.tif', tiff_bytes(named, compression=None))  # a workspace name in pixels


def test_fixture_tiffs_are_clean():
    for name in ('x.tif', 'y.tif', 'z.tif', 'mask.tif'):
        assert P.scan_member(f'strip/{name}', (QA/'strip'/name).read_bytes()) == []


# ---- end to end on the fixture ------------------------------------------------------------------------------------

def read_zip(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        infos = z.infolist()
        assert all(i.compress_type == zipfile.ZIP_STORED for i in infos)
        return [i.filename for i in infos], {i.filename: z.read(i) for i in infos if not i.is_dir()}


def test_release_of_fixture_copies(layout, capsys):
    assert run(layout) == 0
    report = json.loads(capsys.readouterr().out)
    out = layout['out']
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ['PHerc0139_primary_strips.zip', 'PHerc0139_secondary_strips.zip', 'PHerc0332_crossing_strips.zip',
         'README.md', 'manifest.json', 'SHA256SUMS.txt'])
    # zips: stored, the run's layout, published member order
    names, data = read_zip(out/'PHerc0139_secondary_strips.zip')
    files = ['mask.tif', 'meta.json', 'trace.json', 'x.tif', 'y.tif', 'z.tif']
    assert names == (['secondary/', 'secondary/P0003/', 'secondary/P0003/strip/']
                     + ['secondary/P0003/strip/'+f for f in ['expected_error.tif']+files]
                     + ['secondary/P0003/strip.npz', 'secondary/P0003/summary.json',
                        'secondary/P0005/', 'secondary/P0005/strip/']
                     + ['secondary/P0005/strip/'+f for f in files]
                     + ['secondary/P0005/strip.npz', 'secondary/P0005/summary.json'])
    # members: TIFFs and strip.npz byte for byte; JSON only the afv value changed
    src = layout['strips']/'secondary'/'P0003'
    for rel in ['strip/'+f for f in ['expected_error.tif', 'mask.tif', 'meta.json', 'x.tif', 'y.tif', 'z.tif']] \
            + ['strip.npz']:
        assert data['secondary/P0003/'+rel] == (src/rel).read_bytes(), rel
    for rel in ('summary.json', 'strip/trace.json'):
        before = (src/rel).read_bytes()
        assert data['secondary/P0003/'+rel] == before.replace(json.dumps(RUN_AFV).encode(), b'"<local>"')
        assert b'\r\n' in data['secondary/P0003/'+rel]
    # manifest: from the plans; sha256 of every member
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    strips = {(s['set'], s['piece']): s for s in manifest['strips']}
    assert sorted(strips) == [('0332', 'C0000'), ('primary', 'P0000'), ('secondary', 'P0003'),
                              ('secondary', 'P0005')]
    p3 = strips[('secondary', 'P0003')]
    assert (p3['fiber'], p3['annotator'], p3['family'], p3['afv_id']) == \
        ('dj_20260904T151437210_000003', 'dj', 'V', QA_AFV_ID)
    assert [c['id'] for c in p3['stretches']] == ['S0018', 'S0019']
    assert p3['stretches'][0]['segment'] == '20260115000000-w044_2026011522'
    c0 = strips[('0332', 'C0000')]
    assert c0['family'] == 'H' and c0['annotator'] == 'dj'
    assert c0['crossing_windows'] == [dict(id='X0000H', crossing='X0000', family='H', s0_sv=8049.7, s1_sv=8310.3,
                                           partner_window='X0000V', partner_piece='C0001',
                                           partner_fiber='lt_20261013T095553471_000027', closest_approach_sv=2.5,
                                           tight=True, z_tercile=0)]
    assert manifest['crossings'][0]['windows'] == {'X0000H': dict(piece='C0000', packaged=True),
                                                   'X0000V': dict(piece='C0001', packaged=False)}
    for zip_name in ('PHerc0139_primary_strips.zip', 'PHerc0139_secondary_strips.zip',
                     'PHerc0332_crossing_strips.zip'):
        _, zdata = read_zip(out/zip_name)
        listed = {m['name']: m for s in manifest['strips'] if s['zip'] == zip_name for m in s['members']}
        assert sorted(listed) == sorted(zdata)
        for name, blob in zdata.items():
            assert listed[name]['sha256'] == sha(blob) and listed[name]['bytes'] == len(blob)
            assert listed[name]['scrubbed'] == name.endswith(('summary.json', 'trace.json'))
            assert_no_leak(name, blob)
    sec = manifest['sets']['secondary']
    assert (sec['planned_pieces'], sec['packaged_pieces'], sec['missing_pieces']) == (4, 2, ['P0007'])
    assert sec['incomplete_pieces'] == [dict(piece='P0006', missing=['summary.json'])]
    assert sec['left_out'] == {'P0005': ['.tifxyz_tmp_abc123/', 'render.png', 'strip/fiber_778459.json']}
    assert not any(n.endswith(('fiber_778459.json', 'render.png')) or 'tifxyz_tmp' in n for n in names)
    assert manifest['sets']['0332']['missing_pieces'] == ['C0001']
    # what held paths, reported
    assert manifest['scrub']['fields'] == {'strip/trace.json': {'afv': 4}, 'summary.json': {'afv': 4}}
    assert report['fields_that_held_paths'] == manifest['scrub']['fields']
    assert manifest['provenance']['freeze'] == dict(found=False)
    assert manifest['grid'] == dict(row_step_voxels=4.0, column_step_voxels=5.0, columns=25)
    # SHA256SUMS and the top-level files
    sums = dict(reversed(line.split('  ', 1)) for line in (out/'SHA256SUMS.txt').read_text().splitlines())
    assert sorted(sums) == sorted(p.name for p in out.iterdir() if p.name != 'SHA256SUMS.txt')
    for name, digest in sums.items():
        assert sha((out/name).read_bytes()) == digest
    text = (out/'README.md').read_text(encoding='utf-8')
    for needle in ('CC BY-NC 4.0', 'sean (bruniss', 'villa PR #2110', '#2163', 'slchicoineau-creator', 'Baz248',
                   'dj, kb and lt', 'Vesuvius Challenge open data', '`summary.json` `afv`: 4 strips',
                   'secondary: no strip for P0007', 'P0006 left out', 'AI assistance',
                   '`afv` held the path of the AFV file that infer.py read.'):  # one AFV for all sets here
        assert needle in text, needle
    for name in ('README.md', 'manifest.json', 'SHA256SUMS.txt'):
        assert_no_leak(name, (out/name).read_bytes())


def test_released_strips_score_like_the_originals(layout, tmp_path):
    keep = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # no __pycache__ in the frozen score/ tree
    sys.path.insert(0, str(REPO/'score'))
    try:
        import score_strips
    finally:
        sys.path.remove(str(REPO/'score'))
        sys.dont_write_bytecode = keep
    assert run(layout) == 0
    extracted = tmp_path/'extracted'
    with zipfile.ZipFile(layout['out']/'PHerc0139_primary_strips.zip') as z:
        z.extractall(extracted)
    a = score_strips.load_strip(QA/'strip')
    b = score_strips.load_strip(extracted/'primary'/'P0000'/'strip')
    assert b.name == 'P0000' and b.frame_rows is not None and b.expected is not None
    for field in ('rows', 'valid', 'expected', 'frame_rows'):
        assert np.array_equal(getattr(a, field), getattr(b, field), equal_nan=True), field
    assert {k: v for k, v in a.trace.items() if k != 'afv'} == {k: v for k, v in b.trace.items() if k != 'afv'}
    assert b.trace['afv'] == '<local>'


def test_readme_names_the_runner_afv_only_when_every_strip_shows_it(layout, capsys):
    tails = {'secondary': 'matched_0139_secondary_run40.afv', '0332': 'crossings_0332.afv'}
    for set_name, tail in tails.items():
        for piece in (layout['strips']/set_name).iterdir():
            for rel in ('summary.json', 'strip/trace.json'):
                f = piece/rel
                if f.is_file():
                    f.write_bytes(f.read_bytes().replace(json.dumps(RUN_AFV).encode(),
                                                         json.dumps(RUN_AFV.replace('matched_0139.afv', tail)).encode()))
    assert run(layout) == 0
    capsys.readouterr()
    text = (layout['out']/'README.md').read_text(encoding='utf-8')
    assert ('`afv` held the path of the piece AFV that infer.py read: `score/data/matched_0139.afv` (primary), '
            '`score/data/matched_0139_secondary_run40.afv` (secondary) and `score/data/crossings_0332.afv` '
            '(0332), which the\nrepository rebuilds') in text


def test_dry_run_writes_nothing(layout, capsys):
    assert run(layout, '--dry-run') == 0
    report = json.loads(capsys.readouterr().out)
    assert not layout['out'].exists()
    assert report['dry_run'] is True and report['local_paths_left'] == 0
    assert report['fields_that_held_paths'] == {'strip/trace.json': {'afv': 4}, 'summary.json': {'afv': 4}}
    assert report['sets']['secondary']['missing_pieces'] == ['P0007']


def test_relative_paths_are_kept_and_listed(layout, capsys):
    shutil.rmtree(layout['strips']/'primary'/'P0000')
    make_piece(layout['strips']/'primary'/'P0000', afv=None)  # the fixture as it is: a relative afv
    assert run(layout, '--sets', 'primary') == 0
    report = json.loads(capsys.readouterr().out)
    assert report['fields_that_held_paths'] == {}
    assert report['relative_paths_kept'] == {'strip/trace.json': {'afv': 1}, 'summary.json': {'afv': 1}}
    assert sorted(p.name for p in layout['out'].iterdir()) == ['PHerc0139_primary_strips.zip', 'README.md',
                                                               'SHA256SUMS.txt', 'manifest.json']
    _, data = read_zip(layout['out']/'PHerc0139_primary_strips.zip')
    for rel in ('summary.json', 'strip/trace.json', 'strip.npz', 'strip/x.tif'):
        assert data['primary/P0000/'+rel] == (QA/rel).read_bytes()  # nothing to replace: byte for byte


def test_planted_paths_in_every_member_type(layout, capsys):
    piece = layout['strips']/'secondary'/'P0003'
    meta = piece/'strip'/'meta.json'
    planted = json.loads(meta.read_bytes())
    planted['source'] = {'C:/Users/PC/data/x.json': ['/c/data/y', {'deep': 'D:' + B + 'PrizeTemp' + B + 'z'}],
                         'list': [['file:///C:/VesuviusTemp/a'], 'ok']}
    meta.write_bytes(json.dumps(planted, indent=2).replace('\n', '\r\n').encode())
    summary = piece/'summary.json'
    s = json.loads(summary.read_bytes())
    s['argv'] = ['--afv', RUN_AFV, '--cache-dir', 'C:/VesuviusTemp/strips/ct_cache', '--batch', '32']
    s['env'] = {'CACHE_DIR': 'C:' + B + 'VesuviusTemp' + B + 'strips' + B + 'hf_home'}
    s['note'] = 'python C:/Users/PC/miniconda3/envs/vesuvius/python.exe -I runs/v3/infer_cpu.py'
    summary.write_bytes(json.dumps(s, indent=1).encode())
    z = dict(np.load(piece/'strip.npz', allow_pickle=False))
    z['out_dir'] = np.array('C:/VesuviusTemp/strips/score/runs/v3/strips/secondary/P0003')
    z['inputs'] = np.array(['/c/Users/PC/x', 'kept', 'D:/Competition/Vesuvius progress prizes/y'])
    z['raw'] = np.array([b'C:' + B.encode() + b'hf_home', b'ok'])
    np.savez_compressed(piece/'strip.npz', **z)
    original = {k: v for k, v in members((piece/'strip.npz').read_bytes()).items()
                if k not in ('out_dir.npy', 'inputs.npy', 'raw.npy')}
    assert len(original) == 10
    assert run(layout, '--sets', 'secondary') == 0
    report = json.loads(capsys.readouterr().out)
    assert report['fields_that_held_paths'] == {
        'strip.npz': {'inputs[0]': 1, 'inputs[2]': 1, 'out_dir': 1, 'raw[0]': 1},
        'strip/meta.json': {'source.list[0][0]': 1, 'source.{<local>}': 1, 'source.<local>[0]': 1,
                            'source.<local>[1].deep': 1},
        'strip/trace.json': {'afv': 2},
        'summary.json': {'afv': 2, 'argv[1]': 1, 'argv[3]': 1, 'env.CACHE_DIR': 1, 'note': 1}}
    _, data = read_zip(layout['out']/'PHerc0139_secondary_strips.zip')
    for name, blob in data.items():
        assert_no_leak(name, blob)
    got = json.loads(data['secondary/P0003/strip/meta.json'])
    assert got['source'] == {'<local>': ['<local>', {'deep': '<local>'}], 'list': [['<local>'], 'ok']}
    assert {k: v for k, v in got.items() if k != 'source'} == {k: v for k, v in planted.items() if k != 'source'}
    got = json.loads(data['secondary/P0003/summary.json'])
    assert got['argv'] == ['--afv', '<local>', '--cache-dir', '<local>', '--batch', '32']
    assert got['env'] == {'CACHE_DIR': '<local>'}
    assert got['note'] == 'python <local> -I runs/v3/infer_cpu.py'
    npz = data['secondary/P0003/strip.npz']
    after = members(npz)
    for name, blob in original.items():
        assert after[name] == blob, name  # every other array byte for byte
    zz = np.load(io.BytesIO(npz), allow_pickle=False)
    assert zz['out_dir'].item() == '<local>'
    assert zz['inputs'].tolist() == ['<local>', 'kept', '<local>']
    assert zz['raw'].tolist() == [b'<local>', b'ok']


@pytest.mark.parametrize('where', ['tiff_description', 'tiff_private_tag', 'npz_object', 'npz_numeric_name'])
def test_paths_that_cannot_be_removed_stop_the_release(layout, capsys, where):
    piece = layout['strips']/'secondary'/'P0003'
    if where.startswith('tiff'):
        arr = tifffile.imread(piece/'strip'/'y.tif')
        kwargs = (dict(description='C:/Users/PC/y.tif', metadata=None) if where == 'tiff_description' else
                  dict(extratags=[(65000, 's', 0, '/c/VesuviusTemp/y', True)]))
        tifffile.imwrite(piece/'strip'/'y.tif', arr, photometric='minisblack', **kwargs)
    else:
        z = dict(np.load(piece/'strip.npz', allow_pickle=False))
        if where == 'npz_object':
            obj = np.empty(1, object)
            obj[0] = 'C:/Users/PC/x'
            z['obj'] = obj
        else:
            z['blob'] = np.frombuffer(b'..miniconda3....', np.float32)
        np.savez_compressed(piece/'strip.npz', **z)
    assert run(layout) == 1
    err = capsys.readouterr().err
    assert 'local paths left' in err
    assert ('y.tif' in err) if where.startswith('tiff') else ('strip.npz' in err)
    assert not layout['out'].exists()  # nothing released, no staging left


def test_a_scrub_bug_is_caught_by_the_scan(layout, capsys, monkeypatch):
    monkeypatch.setattr(P, 'scrub_member', lambda name, data: (data, [], []))
    assert run(layout) == 1
    assert 'local paths left' in capsys.readouterr().err
    assert not layout['out'].exists()


def test_verify_zip_reads_back(tmp_path):
    leaky = b'{"afv": "C:/Users/PC/x"}'
    path = tmp_path/'t.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(zipfile.ZipInfo('s/a.json'), leaky)
        info = zipfile.ZipInfo('s/b.json')
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, b'{}')
    hits, problems = P.verify_zip(path, {'s/a.json': sha(leaky), 's/b.json': '0'*64})
    assert hits and all('s/a.json' in w for w, _ in hits)
    assert any('not stored' in p for p in problems) and any('differs' in p for p in problems)


# ---- refusals -----------------------------------------------------------------------------------------------------

def test_strip_that_does_not_match_its_plan(layout, capsys):
    f = layout['plans']/'runplan_0139_primary.tsv'
    f.write_text(f.read_text(encoding='utf-8').replace(str(QA_AFV_ID), '3329813072112693'), encoding='utf-8')
    plan = layout['plans']/'matched_stretches.json'
    plan.write_text(plan.read_text(encoding='utf-8').replace(str(QA_AFV_ID), '3329813072112693'), encoding='utf-8')
    assert run(layout) == 2
    err = capsys.readouterr().err
    assert "primary/P0000: summary.json fiber 1803139159717749 is not the plan's AFV id 3329813072112693" in err
    assert not layout['out'].exists()


def test_settings_and_decisions_are_checked(layout, capsys):
    f = layout['strips']/'primary'/'P0000'/'summary.json'
    f.write_bytes(f.read_bytes().replace(b'"blend": 16.0', b'"blend": 8.0').replace(b'"decisions": 8',
                                                                                    b'"decisions": 9'))
    assert run(layout) == 2
    err = capsys.readouterr().err
    assert 'summary.json blend 8.0 is not 16' in err and 'summary.json decisions 9' in err


def test_plan_files_that_disagree(layout, capsys):
    plan = layout['plans']/'matched_stretches.json'
    plan.write_text(plan.read_text(encoding='utf-8').replace('"family": "H"', '"family": "V"'), encoding='utf-8')
    assert run(layout, '--dry-run') == 2
    assert 'the plan files disagree' in capsys.readouterr().err


def test_unexpected_piece_folder(layout, capsys):
    make_piece(layout['strips']/'secondary'/'P0008')  # planned, but run = False
    assert run(layout) == 2
    assert 'secondary/P0008' in capsys.readouterr().err


def test_require_complete(layout, capsys):
    assert run(layout, '--require-complete') == 3
    err = capsys.readouterr().err
    assert 'secondary/P0007' in err and 'secondary/P0006' in err and '0332/C0001' in err


def test_out_folder_rules(layout, capsys):
    bad = layout['strips']/'release'
    assert P.main(['--strips', str(layout['strips']), '--plans', str(layout['plans']), '--out', str(bad)]) == 2
    layout['out'].mkdir()
    (layout['out']/'notes.txt').write_text('mine')
    assert run(layout) == 2
    (layout['out']/'notes.txt').unlink()
    assert run(layout) == 0
    first = (layout['out']/'manifest.json').read_bytes()
    assert run(layout) == 2  # a release is there already
    assert run(layout, '--overwrite') == 0
    assert json.loads((layout['out']/'manifest.json').read_bytes())['strips'] == json.loads(first)['strips']
    assert not (layout['out']/'.staging').exists()
    capsys.readouterr()


def test_unfinished_run_is_refused(layout, capsys):
    run_dir = layout['strips'].parent
    (run_dir/'run_all.ps1').write_text('# runner')
    assert run(layout, '--dry-run') == 2
    assert 'not finished' in capsys.readouterr().err
    (run_dir/'DONE').write_bytes('\ufeffrc=1 stage=finished finished=2026-10-29 09:00:00\r\n'.encode())
    assert run(layout) == 0
    capsys.readouterr()
    manifest = json.loads((layout['out']/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['run_status'] == 'DONE: rc=1 stage=finished finished=2026-10-29 09:00:00'


def test_freeze_check(layout, capsys):
    freeze = layout['plans'].parent/'FREEZE.txt'
    lines = ['Freeze (test)', 'sha256:']
    for f in sorted(layout['plans'].iterdir()):
        lines.append(f'  {sha(f.read_bytes())} *results/{f.name}')
    freeze.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    assert run(layout, '--dry-run') == 0
    assert json.loads(capsys.readouterr().out)['plans_frozen'] is True
    f = layout['plans']/'runplan_0332_crossings.tsv'
    f.write_bytes(f.read_bytes()+b'\n')
    assert run(layout, '--dry-run') == 2
    assert 'runplan_0332_crossings.tsv is not as frozen' in capsys.readouterr().err


# ---- the frozen plan files ------------------------------------------------------------------------------------------

@pytest.mark.skipif(not (RESULTS/'runplan_0139_secondary_run40.tsv').is_file(), reason='needs score/results')
def test_frozen_plans_map_as_preregistered():
    plans, files, freeze = P.load_plans(RESULTS, list(P.SETS))
    if (REPO/'score'/'FREEZE.txt').is_file():
        assert freeze['found'] and freeze['plans_verified']
        assert all(f['frozen'] for f in files.values()) and len(files) == 6
    prim, sec, x332 = (plans[s]['pieces'] for s in ('primary', 'secondary', '0332'))
    # PREREG 4.3, 4.6, 4.10 and 7
    assert sorted(prim) == ['P0000', 'P0001']
    assert [c['id'] for p in prim.values() for c in p['covers']] == ['S0000', 'S0001']
    assert (prim['P0000']['fiber'], prim['P0000']['annotator'], prim['P0000']['family']) == \
        ('kb_20260904T021448778_000085', 'kb', 'H')
    assert prim['P0000']['covers'][0]['segment'] == '20260115000000-w044_2026011522'
    assert sum(p['decisions'] for p in prim.values()) == 58
    assert len(sec) == 46 and sum(p['decisions'] for p in sec.values()) == 3513
    assert sum(len(p['covers']) for p in sec.values()) == 284
    fibers = {p['fiber']: p['family'] for p in sec.values()}
    assert len(fibers) == 36 and sorted(fibers.values()).count('H') == 14
    assert len(x332) == 25 and sum(p['decisions'] for p in x332.values()) == 558
    assert sum(len(p['covers']) for p in x332.values()) == 30 and len({p['fiber'] for p in x332.values()}) == 24
    assert len(plans['0332']['crossings']) == 15 and sum(c['tight'] for c in plans['0332']['crossings']) == 3
    assert [c['id'] for c in x332['C0013']['covers']] == ['X0025H', 'X0024H', 'X0022H']
    assert x332['C0000']['covers'][0]['partner_piece'] == 'C0008'
    annotators = {p['annotator'] for s in plans.values() for p in s['pieces'].values()}
    assert annotators == {'lt', 'kb', 'dj'}
    assert all(p['family'] in 'HV' for s in plans.values() for p in s['pieces'].values())


@pytest.mark.skipif(not (RESULTS/'runplan_0139_secondary_run40.tsv').is_file(), reason='needs score/results')
def test_frozen_plans_with_no_strips_and_a_wrong_strip(tmp_path, capsys):
    strips = tmp_path/'strips'
    strips.mkdir()
    args = ['--strips', str(strips), '--plans', str(RESULTS), '--out', str(tmp_path/'release')]
    assert P.main(args+['--dry-run']) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report['sets']['secondary']['missing_pieces']) == 46 and report['sets']['secondary']['zip'] is None
    make_piece(strips/'primary'/'P0000')  # a Paris 4 strip under a PHerc0139 piece id
    assert P.main(args) == 2
    assert "fiber 1803139159717749 is not the plan's AFV id 3329813072112693" in capsys.readouterr().err
    assert not (tmp_path/'release').exists()


def test_command_line(layout):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2')
    cmd = [sys.executable, '-I', str(HERE/'package_strips.py'), '--strips', str(layout['strips']),
           '--plans', str(layout['plans']), '--out', str(layout['out']), '--dry-run']
    done = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)['local_paths_left'] == 0
    assert 'no FREEZE.txt' in done.stderr
