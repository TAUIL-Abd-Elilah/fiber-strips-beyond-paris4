"""Tests of tools/make_figures.py on SYNTHETIC reports only (never on the scored run's reports).

Two kinds of reports:
  - written by the frozen scorer itself (score_strips.py match / score / crossplan / crossings, run in memory as
    score/tests/test_score_strips.py does) on a synthetic scene with known offsets: wrap segments are cylinders about
    the umbilicus, five V fibers and one H fiber lie on one of them, their strips sit known distances outside it, and
    one more V fiber 4.5 sv off every segment gives a crossing outside the tight subset;
  - written field by field from cmd_score / cmd_crossings, for the edge cases (one fiber, no baseline, no floor, no
    calibration, no crossing with both strips, malformed files) and for a result that degrades (the strip worse than
    the baselines, observed |d| above expected_error, a crossing exactly at the 3 sv tight limit).

What is drawn is checked on the figure models (the backend-independent list of what each panel draws, with the
report value behind every point), and both backends are checked to draw every point of the model, with every legend
outside all axes.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 python -m pytest -q tools/test_make_figures.py
"""
from __future__ import annotations

import builtins
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import make_figures as MF  # noqa: E402


def _scorer_test_helpers():
    """score/tests/test_score_strips.py's scene helpers (and through it the frozen scorer), under a private name."""
    spec = importlib.util.spec_from_file_location('_score_strips_test_helpers',
                                                  REPO/'score'/'tests'/'test_score_strips.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _scorer_test_helpers()
S = T.S
HAVE_MPL = importlib.util.find_spec('matplotlib') is not None
VILLA_INFER = Path('C:/VesuviusTemp/strips/villa_6f2/vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/'
                   'infer.py')

PHI = {'v_a': -.6, 'v_b': -.3, 'v_c': 0., 'v_d': .3, 'v_e': .6}
OFFSET = {'v_a': .2, 'v_b': .5, 'v_c': .8, 'v_d': 1.1, 'v_e': 2.6}    # sv: each V strip outside the segment
EXPECTED = {'v_a': .4, 'v_b': .6, 'v_c': .9, 'v_d': 1.3, 'v_e': 3.2}  # sv: the model's expected error (5 bins)
H_DELTA = 1.4                                                          # sv: the H strip outside the segment
RUNPLAN_COLUMNS = ['piece', 'afv_id', 'family', 'fiber', 'arc_start_patch_vox', 'arc_end_patch_vox', 'decisions',
                   'decisions_planned', 'run', 'run_order', 'windows', 'segments_or_crossings', 'parent_afv_id',
                   'head_k_first', 'head_k_last', 'max_head_dev_sv']


def write_runplan(path, pieces):
    """A run plan TSV in make_piece_afv.py's columns (score/results/runplan_*.tsv) for the synthetic pieces."""
    lines = ['\t'.join(RUNPLAN_COLUMNS)]
    for n, pc in enumerate(pieces):
        lines.append('\t'.join(str(x) for x in (
            pc['id'], pc['afv_id'], pc['family'], pc['fiber'], f"{pc['arc_patch_vox'][0]:.6f}",
            f"{pc['arc_patch_vox'][1]:.6f}", pc['decisions'], pc['decisions'], 'True', n, ','.join(pc['windows']),
            pc['segment'], pc['parent_afv_id'], pc['head_k'][0], pc['head_k'][1], '0.0')))
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    """Workspaces laid out as C:/VesuviusTemp/strips/score after the run, with reports written by the scorer:
    'full' (secondary + primary with Lasagna normals, PHerc0332-style crossings) and 'nolas' (no Lasagna normals,
    no crossings report). The strips live outside the workspaces: the tool must never need runs/v3/strips."""
    root = tmp_path_factory.mktemp('world')
    data = root/'data'
    segs, fib = data/'segments', data/'fibers'
    T.write_tifxyz(segs/'w001'/'mesh'/'w001.tifxyz', T.cylinder_grid(200., 0., 700., -.9, .9), 1.)
    T.write_tifxyz(segs/'w002'/'mesh'/'w002.tifxyz', T.cylinder_grid(212., 0., 700., -.9, .9), 1.)
    umb = T.write_umbilicus(data/'umbilicus.json')
    fib.mkdir(parents=True)
    for name, phi in PHI.items():
        T.write_fiber(fib/f'{name}.json', T.v_fiber(200., phi), 'V')
    T.write_fiber(fib/'h_a.json', T.h_fiber(200., 300., -.75, .75), 'H')
    T.write_fiber(fib/'v_off.json', T.v_fiber(204.5, .45), 'V')  # 4.5 sv off every segment: no stretch
    common = ['--fibers', str(fib), '--segments', str(segs), '--umbilicus', str(umb)]
    m_sec, m_pri = root/'matched_secondary.json', root/'matched_primary.json'
    S.main(['match', *common, '--out', str(m_sec), '--min-stretch-mm', '0.38384', '--analysis', 'secondary'])
    S.main(['match', *common, '--out', str(m_pri)])
    plan = json.loads(m_sec.read_text())
    strips = root/'strips_secondary'  # infer.py layout: <piece id>/strip (tifxyz) and <piece id>/strip.npz
    for pc in plan['pieces']:
        if pc['fiber'] == 'h_a':  # no strip.npz: no frame-plane baseline for this piece
            T.write_tifxyz(strips/pc['id']/'strip', np.swapaxes(T.h_strip(200., 300., -.65, .65, H_DELTA), 0, 1),
                           T.BASE, extra_json=dict(family='H', fiber_along='columns'))
        else:
            dec, L = T.decisions_v(200., PHI[pc['fiber']], 100., 500., expected=EXPECTED[pc['fiber']],
                                   height_offset=OFFSET[pc['fiber']])
            T.write_infer_output(strips/pc['id'], dec, L)
    arrays = {}
    for f in S.load_fibers(fib)[0]:  # the Lasagna normal of a cylinder: radial
        d = f.points[:, :2]-[T.CX, T.CY]
        arrays[f.name+'__n'] = np.c_[d/np.linalg.norm(d, axis=1, keepdims=True), np.zeros(len(d))].astype(np.float32)
        arrays[f.name+'__valid'] = np.ones(len(d), bool)
    normals = root/'lasagna_normals.npz'
    np.savez_compressed(normals, **arrays)

    def workspace(name, lasagna):
        ws = root/name/'score'
        res = ws/'results'/'v3'
        res.mkdir(parents=True)
        las = ['--lasagna-normals', str(normals)] if lasagna else []
        S.main(['score', *common, '--crossings', '--matches', str(m_sec), '--run-only', '--strips', str(strips), *las,
                '--out', str(res/'secondary_report.json')])
        S.main(['score', *common, '--crossings', '--matches', str(m_pri), '--strips', str(strips), *las,
                '--out', str(res/'primary_case_studies.json')])
        write_runplan(ws/'results'/'runplan_0139_secondary_run40.tsv', plan['pieces'])
        return ws
    full, nolas = workspace('full', True), workspace('nolas', False)
    xplan, s332 = root/'crossings_plan.json', root/'strips_0332'
    S.main(['crossplan', '--fibers', str(fib), '--out', str(xplan)])
    T.write_tifxyz(s332/'C_h_a'/'strip', np.swapaxes(T.h_strip(200., 300., -.7, .7), 0, 1), T.BASE,
                   extra_json=dict(family='H', fiber_along='columns'))
    for name, phi in PHI.items():
        T.write_tifxyz(s332/f'C_{name}'/'strip', T.v_strip(200., phi, 175., 425., OFFSET[name]), T.BASE,
                       extra_json=dict(family='V', fiber_along='rows'))
    T.write_tifxyz(s332/'C_v_off'/'strip', T.v_strip(204.5, .45, 175., 425.), T.BASE,
                   extra_json=dict(family='V', fiber_along='rows'))
    S.main(['crossings', '--fibers', str(fib), '--plan', str(xplan), '--strips', str(s332),
            '--out', str(full/MF.REPORTS['crossings_0332'])])
    return {'full': full, 'nolas': nolas, 'plan': plan, 'root': root}


def run(capsys, *args):
    rc = MF.main([str(a) for a in args])
    return rc, capsys.readouterr().out


def captions(out):
    return json.loads((Path(out)/'captions.json').read_text(encoding='utf-8'))


def report(ws, key):
    return json.loads((Path(ws)/MF.REPORTS[key]).read_text(encoding='utf-8'))


# ---- reports written by the scorer ---------------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_MPL, reason='needs matplotlib for the PNGs')
def test_scorer_reports_make_all_four_figures(world, tmp_path, capsys):
    Image = pytest.importorskip('PIL.Image')
    out = tmp_path/'figures'
    rc, text = run(capsys, '--workspace', world['full'], '--out', out)
    assert rc == 0 and 'SKIP' not in text and 'WARNING' not in text
    cap = captions(out)
    assert cap['backend'].startswith('matplotlib') and all(r['status'] == 'read' for r in cap['reports'].values())
    assert cap['reports']['secondary']['sha256'] == MF.sha256_file(world['full']/MF.REPORTS['secondary'])
    for name in MF.FIGURES:
        e = cap['figures'][name]
        assert 'skipped' not in e and e['files'] == [f'{name}.png', f'{name}.svg'] and e['caption'], name
        assert 'stale_files_from_an_earlier_run' not in e
        with Image.open(out/f'{name}.png') as im:
            w, h = e['size_inches']
            assert abs(im.size[0]-w*150) < 1 and abs(im.size[1]-h*150) < 1  # Agg truncates to whole pixels
            assert all(abs(d-150) < .5 for d in im.info['dpi'])
        assert ET.parse(out/f'{name}.svg').getroot().tag.endswith('svg')


def test_per_fiber_means_reproduce_the_scorer_aggregate(world):
    rep = report(world['full'], 'secondary')
    fibers = MF.fiber_table(rep['per_strip'], 25)
    check = MF.check_against_report(fibers, rep['accuracy']['all'])
    assert check['ok'], check
    assert sorted(n for n, f in fibers.items() if f['scored']) == ['h_a', 'v_a', 'v_b', 'v_c', 'v_d', 'v_e']
    # one-fiber family: its per-fiber mean is the family's pooled mean as the scorer states it; the V family pools
    h = rep['accuracy']['H']['strip']['point']
    assert fibers['h_a']['nodes']['strip'] == h['nodes'] and abs(fibers['h_a']['mean']['strip']-h['mean']) < 1e-12
    v = rep['accuracy']['V']['strip']['point']
    vs = [fibers[n] for n in OFFSET]
    assert sum(f['nodes']['strip'] for f in vs) == v['nodes']
    assert abs(sum(f['sum']['strip'] for f in vs)/v['nodes']-v['mean']) < 1e-12
    for name, off in OFFSET.items():
        assert abs(fibers[name]['mean']['strip']-off) < .05
    assert abs(fibers['h_a']['mean']['strip']-H_DELTA) < .05
    # frame plane only where strip.npz exists (the V pieces); the Lasagna plane everywhere
    assert fibers['h_a']['nodes']['frame_plane'] == 0 and all(fibers[n]['nodes']['frame_plane'] >= 25 for n in OFFSET)
    assert all(f['nodes']['lasagna_plane'] >= 25 for f in fibers.values())


def test_example_strips_from_the_scorer_report(world, tmp_path, capsys):
    out = tmp_path/'fig ures'  # a space, as in the repository's own path
    rc, text = run(capsys, '--workspace', world['full'], '--out', out)
    assert rc == 0
    ex = captions(out)['example_strips']
    assert ex['scored_fibers'] == 6 and ex['consistency_with_report']['ok'] and ex['runplan_status'] == 'read'
    med, p90 = ex['examples']['median'], ex['examples']['p90']
    # means 0.2 (v_a) 0.5 0.8 (v_c) 1.1 1.4 (h_a) 2.6 (v_e): median = rank 3 of 6 (the lower middle), p90 = rank 6
    assert (med['fiber'], med['rank'], p90['fiber'], p90['rank']) == ('v_c', 3, 'v_e', 6)
    pieces = {pc['fiber']: pc for pc in world['plan']['pieces']}
    ws = world['full'].resolve()
    render_dir = ws.parent/'renders'/'v3_secondary'
    for e, kind in ((med, 'median'), (p90, 'p90')):
        (p,) = e['pieces']
        pc = pieces[e['fiber']]
        assert p['piece'] == pc['id'] and p['afv_id'] == pc['afv_id'] and p['run_in_plan'] == 'True'
        mkdir, copy_strip, copy_files, render, copy_png = p['commands']
        src = ws/'runs'/'v3'/'strips'/'secondary'/pc['id']
        assert MF.ps_quote(src/'strip'/'*') in copy_strip and MF.ps_quote(src/'strip.npz') in copy_files
        assert MF.ps_quote(src/'summary.json') in copy_files
        tool = MF.ps_quote(MF.RENDER_TOOL)
        assert render.startswith(f'& {MF.ps_quote(MF.PYTHON)} -I -u {tool} ')
        assert f" --villa {MF.ps_quote(ws.parent/'villa_6f2')} --out {MF.ps_quote(render_dir/pc['id'])} " in render
        assert '--render-only' not in render  # infer.py --render-only opens the CT without the normalization binding
        assert f'--checkpoint {MF.CHECKPOINT} --ct {MF.CT_0139} ' in render and render.endswith('--device cpu')
        assert copy_png.endswith(MF.ps_quote(out.resolve()/f"example_strip_{kind}_{pc['id']}.png"))
        assert all(c in text for c in p['commands'])  # printed
    assert "$env:CUDA_VISIBLE_DEVICES = '-1'" in ex['environment'] and "$env:HF_HUB_OFFLINE = '1'" in ex['environment']
    # printed, never run, and the run's strips never needed: neither the render copy nor runs/ exists
    assert not (world['full'].parent/'renders').exists() and not (world['full']/'runs').exists()


def test_relative_workspace_and_render_dir_give_absolute_commands(world, tmp_path, capsys, monkeypatch):
    """--workspace and --render-dir relative to the current folder: the printed commands still name absolute paths
    (HF_HOME, ct_cache, the villa worktree, the run's strips, the render copy), so they work wherever they are pasted."""
    monkeypatch.chdir(tmp_path)
    rel_ws = os.path.relpath(world['full'], tmp_path)
    assert not Path(rel_ws).is_absolute()
    rc, text = run(capsys, '--workspace', rel_ws, '--out', 'figs', '--render-dir', 'rd', '--backend', 'svg')
    assert rc == 0
    cap = captions(tmp_path/'figs')
    ws, rd = world['full'].resolve(), tmp_path.resolve()/'rd'
    assert cap['workspace'] == str(ws)
    ex = cap['example_strips']
    assert f"$env:HF_HOME = {MF.ps_quote(ws.parent/'hf_home')}" in ex['environment']
    for kind, e in ex['examples'].items():
        for p in e['pieces']:
            mkdir, copy_strip, copy_files, render, copy_png = p['commands']
            assert MF.ps_quote(rd/p['piece']/'strip') in mkdir
            assert MF.ps_quote(ws/'runs'/'v3'/'strips'/'secondary'/p['piece']/'strip'/'*') in copy_strip
            assert f"-I -u {MF.ps_quote(MF.RENDER_TOOL)} " in render
            assert f"--villa {MF.ps_quote(ws.parent/'villa_6f2')} " in render
            assert f"--out {MF.ps_quote(rd/p['piece'])} " in render
            assert f"--cache-dir {MF.ps_quote(ws.parent/'ct_cache')} " in render
            assert copy_png.endswith(MF.ps_quote(tmp_path.resolve()/'figs'/f"example_strip_{kind}_{p['piece']}.png"))
    assert not rd.exists()  # printed, never run


def test_lasagna_baseline_unavailable(world, tmp_path, capsys):
    out = tmp_path/'figures'
    rc, text = run(capsys, '--workspace', world['nolas'], '--out', out)
    assert rc == 0
    cap = captions(out)
    f1, f2 = cap['figures']['fig1_secondary_distance'], cap['figures']['fig2_secondary_per_fiber']
    assert [s['surface'] for s in f1['data']['node_level']] == ['strip', 'frame_plane', 'fiber_floor']
    assert any('Lasagna plane baseline is not in the report (available_fibers 0)' in n for n in f1['notes'])
    assert 'lasagna_plane' not in f1['data']['per_fiber']
    assert list(f2['data']) == ['frame_plane']
    assert any(n.startswith('Lasagna plane: no scored fiber') for n in f2['notes'])
    assert sorted(r['fiber'] for r in f2['data']['frame_plane']['fibers']) == sorted(OFFSET)  # no strip.npz for h_a
    assert f2['data']['frame_plane']['delta']['fibers'] == 5
    assert cap['figures']['fig3_0332_crossings']['skipped'].startswith('crossings_0332 report missing')
    assert 'SKIP fig3_0332_crossings' in text


def test_crossings_figure_marks_the_tight_subset(world, tmp_path, capsys):
    out = tmp_path/'figures'
    run(capsys, '--workspace', world['full'], '--out', out, '--backend', 'svg')
    e = captions(out)['figures']['fig3_0332_crossings']
    rows = {r['v_fiber']: r for r in e['data']['per_crossing']}
    assert set(rows) == set(PHI) | {'v_off'} and e['data']['missing'] == []
    assert [v for v, r in rows.items() if not r['tight']] == ['v_off']
    assert 4. < rows['v_off']['closest_approach_sv'] < 5.
    assert abs(rows['v_off']['agreement_sv']-4.5) < .1
    assert all(abs(rows[v]['agreement_sv']-OFFSET[v]) < .1 for v in OFFSET)
    assert MF.NO_GT in e['caption']
    svg = (out/'fig3_0332_crossings.svg').read_text(encoding='utf-8')
    assert MF.NO_GT in svg and all(r['id'] in svg for r in rows.values())


def test_calibration_figure_uses_the_reported_bins(world, tmp_path, capsys):
    out = tmp_path/'figures'
    run(capsys, '--workspace', world['full'], '--out', out, '--backend', 'svg')
    e = captions(out)['figures']['fig4_calibration']
    d = e['data']
    filled = [b for b in report(world['full'], 'secondary')['calibration']['bins'] if b['nodes']]
    assert [(b['lo'], b['hi'], b['nodes']) for b in d['secondary_bins']] == [(b['lo'], b['hi'], b['nodes'])
                                                                              for b in filled]
    assert [b['lo'] for b in d['secondary_bins']] == [0., .5, .75, 1.25, 3.] and d['secondary_bins'][-1]['hi'] is None
    assert d['primary_bins'] and d['secondary_summary']['fibers'] == 5  # the H piece has no expected_error
    for b, (name, exp) in zip(d['secondary_bins'], sorted(EXPECTED.items(), key=lambda kv: kv[1])):
        assert abs(b['mean_expected']-exp) < 1e-6 and abs(b['mean_observed']-OFFSET[name]) < .05
    assert 'bin edges 0, 0.5, 0.75, 1, 1.25, 1.5, 2, 3 sv and up' in e['caption']
    prim = report(world['full'], 'primary')
    assert prim['G1']['pass'] is False and 'G1 failed: no accuracy claim' in e['caption']
    assert f"{prim['stretches_scored']} of {prim['stretches_planned']} stretches scored" in e['caption']


def test_svg_backend_when_matplotlib_does_not_import(world, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(MF, 'matplotlib_or_reason', lambda: (None, 'ImportError: no matplotlib here'))
    out = tmp_path/'figures'
    rc, _ = run(capsys, '--workspace', world['full'], '--out', out)
    assert rc == 0
    cap = captions(out)
    assert cap['backend'] == 'SVG written directly, no PNG (matplotlib not used: ImportError: no matplotlib here)'
    assert not list(out.glob('*.png'))
    for name in MF.FIGURES:
        assert cap['figures'][name]['files'] == [f'{name}.svg']
        assert ET.parse(out/f'{name}.svg').getroot().tag.endswith('svg')
    n = len(cap['figures']['fig3_0332_crossings']['data']['per_crossing'])
    svg = (out/'fig3_0332_crossings.svg').read_text(encoding='utf-8')
    assert svg.count('<circle') == n+2  # one per crossing, and the two legend keys (tight, not tight)


def test_dry_run_writes_nothing(world, tmp_path, capsys):
    out = tmp_path/'figures'
    rc, text = run(capsys, '--workspace', world['full'], '--out', out, '--dry-run')
    assert rc == 0 and not out.exists()
    assert all(f'would write {name}' in text for name in MF.FIGURES)
    assert 'render_strip.py' in text and 'caption:' in text


def test_tool_opens_only_the_reports_and_the_run_plan(world, tmp_path, capsys, monkeypatch):
    """Nothing under the workspace but the three reports and the run plan is opened, listed or stat'ed."""
    seen = []
    real_open = builtins.open

    def spy_open(file, *a, **k):
        seen.append(str(file))
        return real_open(file, *a, **k)
    monkeypatch.setattr(builtins, 'open', spy_open)
    for name in ('is_file', 'exists', 'is_dir', 'iterdir', 'glob', 'rglob', 'stat', 'read_text', 'read_bytes'):
        def spy(self, *a, _real=getattr(pathlib.Path, name), **k):
            seen.append(str(self))
            return _real(self, *a, **k)
        monkeypatch.setattr(pathlib.Path, name, spy)
    rc, _ = run(capsys, '--workspace', world['full'], '--out', tmp_path/'figures')
    paths = list(seen)
    monkeypatch.undo()
    assert rc == 0
    ws = str(world['full'].resolve())
    touched = {str(Path(p).resolve()) for p in paths if str(Path(p).resolve()).startswith(ws)}
    allowed = {str((world['full']/rel).resolve()) for rel in [*MF.REPORTS.values(), MF.RUNPLAN]}
    assert touched and touched <= allowed, touched-allowed


# ---- reports written field by field (cmd_score / cmd_crossings) ----------------------------------------------------

def pooled(nodes, mean, median, p90, within1, within2):
    return {'nodes': nodes, 'mean': mean, 'median': median, 'p90': p90, 'within1': within1, 'within2': within2}


def ci5(mean, median, p90, within1, within2):
    return {'mean': mean, 'median': median, 'p90': p90, 'within1': within1, 'within2': within2}


def no_baseline():
    return {'available_fibers': 0, 'stats': None, 'strip_on_same_nodes': None, 'delta': None}


CONSTANTS = {'G2_MAX_MEAN': 1.5, 'G2_MIN_WITHIN2': .7, 'MIN_NODES': 25, 'CROSS_TOL': 3., 'CROSS_RADIUS': 30.,
             'SV_UM': 9.596}


def one_fiber_report():
    """cmd_score's report for one scored fiber: the strip's CI is None (bootstrap_stats needs 2 fibers), no strip.npz
    (no frame plane), no normals, no fiber floor (no covered row), no expected_error (calibration None)."""
    acc = {'family': 'all', 'subset': None,
           'strip_coverage': {'nodes_on_scored_rows': 1100, 'covered': 1000, 'fraction': 1000/1100},
           'scored_fibers': 1, 'scored_stretches': 3, 'strip': {'point': pooled(1000, 1.2, .9, 2.6, .55, .82),
                                                                'ci': None, 'fibers': 1},
           'strip_signed_mean_sv': .1, 'fiber_floor': None,
           'fiber_floor_note': 'the human fiber itself, one point per scored row: the ground truth noise floor',
           'frame_plane': no_baseline(), 'lasagna_plane': no_baseline(), 'lasagna_vs_segment_normal_deg': None}
    empty = dict(acc, family='V', strip_coverage={'nodes_on_scored_rows': 0, 'covered': 0, 'fraction': None},
                 scored_fibers=0, scored_stretches=0, strip={'point': None, 'ci': None, 'fibers': 0},
                 strip_signed_mean_sv=None)
    per_strip = [{'strip': 'P0007', 'fiber': 'kb_x', 'family': 'H', 'stretches': ['S0001', 'S0002', 'S0003'],
                  'assoc_median_sv': .5, 'notes': ['no decisions (strip.npz) beside the strip: frame-plane baseline '
                                                   'unavailable'], 'strip_nodes': 1000, 'strip_mean_abs_sv': 1.2,
                  'strip_signed_mean_sv': .1, 'fiber_floor_mean_abs_sv': None}]
    return {'provenance': {'scorer_version': '3.0.0', 'scorer_sha256': 'x', 'prereg_sha256': 'y',
                           'constants': dict(CONSTANTS)},
            'matches_sha256': 'z', 'G1': {'pass': True}, 'skips': [], 'stretches_planned': 3, 'stretches_scored': 3,
            'accuracy': {'all': acc, 'H': acc, 'V': empty},
            'G2': {'verdict': 'secondary: no accuracy claim (G1 failed or fewer than 30 scored fibers)'},
            'analysis': 'secondary', 'sensitivity': {'note': 'reported, not gating (PREREG 4.9)'},
            'calibration': None, 'crossings': None, 'per_strip': per_strip}


# fiber, piece, family, strip nodes and mean |d|, frame plane nodes and mean, Lasagna plane nodes and mean
EDGE_ROWS = [('f1', 'P01', 'V', 400, 1.0, 380, 3.0, 400, 2.0), ('f2', 'P02', 'H', 300, 1.5, 290, 3.5, 300, 2.2),
             ('f3', 'P03', 'V', 500, 2.0, 480, 4.0, 500, 2.5), ('f4', 'P04', 'H', 200, 2.5, 190, 4.5, 200, 2.0),
             ('f5', 'P05', 'V', 250, 9.0, 240, 3.0, 250, 1.0), ('f5', 'P06', 'V', 250, 11.0, 240, 3.2, 250, 1.2),
             ('f6', 'P07', 'H', 350, 3.0, 340, 5.0, 350, 2.8), ('f7', 'P08', 'V', 450, 3.5, 440, 6.0, 450, 3.0),
             ('f8', 'P09', 'H', 300, 4.0, 290, 6.5, 300, 3.5)]
EDGE_BINS = [(0., .5, .4, 2.9, 2.5), (.5, .75, .62, 3.3, 2.9), (.75, 1., .88, 3.8, 3.1), (1., 1.25, 1.1, 4.4, 3.6),
             (1.25, 1.5, 1.35, 4.9, 4.0), (1.5, 2., 1.7, 5.2, 4.1), (2., 3., None, None, None),
             (3., None, None, None, None)]  # lo, hi, mean expected, mean and median observed


def cal_report(bins, nodes_per_bin, fibers):
    out = [{'lo': lo, 'hi': hi, 'nodes': nodes_per_bin if e is not None else 0, 'mean_expected': e,
            'mean_observed': o, 'median_observed': md, 'within2': .3 if e is not None else None}
           for lo, hi, e, o, md in bins]
    return {'nodes': sum(b['nodes'] for b in out), 'fibers': fibers, 'bins': out, 'weighted_abs_gap_sv': 3.1,
            'observed_over_expected': 4.1, 'spearman_node': .21, 'spearman_fiber': .3, 'note': '...'}


def edge_secondary():
    """cmd_score's secondary report written field by field, shaped like a result that degrades: the strip worse than
    a baseline for some fibers (the p90 example fiber f5: strip 10.0 sv against frame plane 3.1 and Lasagna plane
    1.1 sv), observed |d| far above expected_error. In panel (a) of figure 1 three pairs of marks coincide (within
    1 sv: strip and frame plane; within 2 sv and median: strip and Lasagna plane) and every other mark lies far from
    the others."""
    rows = [{'strip': s, 'fiber': f, 'family': fam, 'stretches': [f'S{s}'], 'assoc_median_sv': .5, 'notes': [],
             'strip_nodes': n, 'frame_plane_nodes': nf, 'lasagna_plane_nodes': nl, 'strip_mean_abs_sv': m,
             'frame_plane_mean_abs_sv': mf, 'lasagna_plane_mean_abs_sv': ml, 'strip_signed_mean_sv': .1,
             'fiber_floor_mean_abs_sv': 1.2} for f, s, fam, n, m, nf, mf, nl, ml in EDGE_ROWS]
    nodes = sum(r['strip_nodes'] for r in rows)
    mean = sum(r['strip_nodes']*r['strip_mean_abs_sv'] for r in rows)/nodes
    acc = {'family': 'all', 'subset': None,
           'strip_coverage': {'nodes_on_scored_rows': nodes+100, 'covered': nodes, 'fraction': nodes/(nodes+100)},
           'scored_fibers': 8, 'scored_stretches': 9,
           'strip': {'point': pooled(nodes, mean, 2.6, 10.5, .2, .45),
                     'ci': ci5([2.6, 5.4], [2.4, 2.8], [9.5, 12.], [.12, .28], [.4, .5]), 'fibers': 8},
           'strip_signed_mean_sv': .4, 'fiber_floor': pooled(140, 1.2, 1.0, 1.9, .5, .92),
           'fiber_floor_note': 'the human fiber itself, one point per scored row: the ground truth noise floor',
           'frame_plane': {'available_fibers': 8,
                           'stats': {'point': pooled(2890, 4.4, 6.0, 8.0, .2, .25),
                                     'ci': ci5([3.6, 5.1], [5.5, 6.5], [7.5, 8.5], [.12, .28], [.2, .3]), 'fibers': 8},
                           'strip_on_same_nodes': pooled(2890, 3.8, 2.5, 10.4, .2, .45),
                           'delta': {'fibers': 8, 'nodes': 2890, 'baseline_minus_strip_mean_sv': .6, 'ci': [-1.2, 2.],
                                     'beaten': False, 'strip_minus_baseline_within2': .2, 'ci_within2': [.1, .3]}},
           'lasagna_plane': {'available_fibers': 8,
                             'stats': {'point': pooled(3000, 2.2, 2.6, 4.0, .03, .45),
                                       'ci': ci5([1.8, 2.7], [2.4, 2.8], [3.6, 4.4], [.01, .05], [.4, .5]),
                                       'fibers': 8},
                             'strip_on_same_nodes': pooled(3000, 3.9, 2.6, 10.5, .2, .45),
                             'delta': {'fibers': 8, 'nodes': 3000, 'baseline_minus_strip_mean_sv': -1.7,
                                       'ci': [-3.5, -.2], 'beaten': False, 'strip_minus_baseline_within2': 0.,
                                       'ci_within2': [-.1, .1]}},
           'lasagna_vs_segment_normal_deg': {'median': 8., 'p90': 20.}}
    return {'provenance': {'scorer_version': '3.0.0', 'constants': dict(CONSTANTS)}, 'G1': {'pass': True},
            'skips': [], 'stretches_planned': 9, 'stretches_scored': 9, 'accuracy': {'all': acc},
            'G2': {'verdict': 'secondary: degrades'}, 'analysis': 'secondary',
            'calibration': cal_report(EDGE_BINS, 400, 8), 'crossings': None, 'per_strip': rows}


def edge_primary():
    """The primary case studies' report as far as figure 4 reads it (G1 failed; two stretches; calibration)."""
    bins = [(0., .5, .45, 6.1, 5.0), (.5, .75, .6, 7.4, 6.0)]
    return {'provenance': {'scorer_version': '3.0.0', 'constants': dict(CONSTANTS)}, 'G1': {'pass': False},
            'stretches_planned': 2, 'stretches_scored': 2, 'analysis': 'primary',
            'G2': {'verdict': 'no accuracy claim (G1 failed or fewer than 30 scored fibers)'},
            'calibration': cal_report(bins, 300, 2)}


def edge_crossings():
    """cmd_crossings' report written field by field: a tight crossing with a large agreement (X0000), one exactly at
    the 3 sv tight limit (X0002, inside it) and one just beyond it (X0003)."""
    per = [('X0000', 'h1', 'v1', .8, 9.5, 2), ('X0001', 'h1', 'v2', 2.2, 1.1, 2), ('X0002', 'h2', 'v3', 3.0, 2.4, 1),
           ('X0003', 'h3', 'v4', 3.05, 3.2, 2), ('X0004', 'h4', 'v5', 5.7, 1.9, 2)]
    rows = [{'id': i, 'h_fiber': h, 'v_fiber': v, 'closest_approach_sv': c, 'point_sv': [0., 0., 0.],
             'agreement_sv': a, 'directions': d} for i, h, v, c, a, d in per]
    a = sorted(r['agreement_sv'] for r in rows)
    return {'provenance': {'constants': dict(CONSTANTS)},
            'crossings': {'label': 'no ground truth: consistency between two predictions, not accuracy', 'planned': 6,
                          'missing': ['X0005'], 'crossings': 5, 'fibers_h': 4, 'fibers_v': 5, 'mean': sum(a)/5,
                          'median': a[2], 'p90': 8.0, 'within2': .4,
                          'ci': {'mean': [1.5, 6.], 'median': [1.1, 9.5], 'p90': [3., 9.5], 'within2': [0., .8]},
                          'tight_subset': {'rule': 'closest approach <= 3.0 sv', 'crossings': 3, 'mean': 13/3,
                                           'median': 2.4, 'within2': 1/3},
                          'per_crossing': rows}}


def write_reports(ws, **reports):
    res = ws/'results'/'v3'
    res.mkdir(parents=True, exist_ok=True)
    for key, data in reports.items():
        (res/Path(MF.REPORTS[key]).name).write_text(data if isinstance(data, str) else json.dumps(data),
                                                     encoding='utf-8')
    return ws


def test_one_fiber_report_without_baselines_floor_or_calibration(tmp_path, capsys):
    no_cross = {'provenance': {}, 'crossings': {'label': 'no ground truth: consistency between two predictions, not '
                                                         'accuracy', 'planned': 15, 'missing': ['X0000', 'X0003'],
                                                'crossings': 0, 'note': 'no H/V crossings with overlapping strips'}}
    ws = write_reports(tmp_path/'score', secondary=one_fiber_report(), crossings_0332=no_cross)
    out = tmp_path/'figures'
    rc, text = run(capsys, '--workspace', ws, '--out', out, '--backend', 'svg')
    assert rc == 0
    cap = captions(out)
    f1 = cap['figures']['fig1_secondary_distance']
    assert [s['surface'] for s in f1['data']['node_level']] == ['strip']
    assert f1['data']['node_level'][0]['ci'] == {k: None for k in ('mean', 'median', 'p90', 'within1', 'within2')}
    assert len(f1['notes']) == 6 and (out/'fig1_secondary_distance.svg').exists()  # 3 absent series, 2 in panel (b)
    assert f1['data']['per_fiber'] == {'strip': [1.2]}
    assert cap['figures']['fig2_secondary_per_fiber']['skipped'].startswith('no baseline per fiber')
    assert cap['figures']['fig4_calibration']['skipped'].startswith('the secondary report has no calibration')
    assert 'no H/V crossings with overlapping strips' in cap['figures']['fig3_0332_crossings']['skipped']
    assert cap['figures']['fig3_0332_crossings']['skipped'].startswith('no crossing with an agreement')
    ex = cap['example_strips']
    assert ex['examples']['median']['fiber'] == ex['examples']['p90']['fiber'] == 'kb_x'
    assert ex['runplan_status'].startswith('missing') and 'commands' not in ex['examples']['median']['pieces'][0]
    assert 'no command (missing' in text and 'WARNING' not in text


def test_missing_reports_skip_with_reasons(world, tmp_path, capsys):
    ws = tmp_path/'score'
    ws.mkdir()
    out = tmp_path/'figures'
    rc, text = run(capsys, '--workspace', ws, '--out', out)
    assert rc == 0
    cap = captions(out)
    for name in MF.FIGURES:
        assert cap['figures'][name]['skipped'].startswith(('secondary report missing', 'crossings_0332 report missing'))
        assert f'SKIP {name}' in text
    assert cap['example_strips'] == {'skipped': 'no secondary report'}
    assert not list(out.glob('*.png')) and not list(out.glob('*.svg'))
    # only the crossings report: figure 3 alone
    only = write_reports(tmp_path/'only332')
    shutil.copy(world['full']/MF.REPORTS['crossings_0332'], only/MF.REPORTS['crossings_0332'])
    rc, _ = run(capsys, '--workspace', tmp_path/'only332', '--out', tmp_path/'fig332', '--backend', 'svg')
    made = [n for n, e in captions(tmp_path/'fig332')['figures'].items() if 'skipped' not in e]
    assert rc == 0 and made == ['fig3_0332_crossings']


def test_stale_figures_are_reported_not_deleted(tmp_path, capsys):
    out = tmp_path/'figures'
    out.mkdir()
    (out/'fig4_calibration.svg').write_text('<svg/>', encoding='utf-8')
    (out/'fig1_secondary_distance.png').write_bytes(b'an earlier matplotlib run')
    ws = write_reports(tmp_path/'score', secondary=one_fiber_report())
    run(capsys, '--workspace', ws, '--out', out, '--backend', 'svg')
    figs = captions(out)['figures']
    e = figs['fig4_calibration']
    assert e['stale_files_from_an_earlier_run'] == ['fig4_calibration.svg'] and (out/'fig4_calibration.svg').exists()
    # a figure drawn as SVG only next to a PNG of an earlier run: the PNG is reported as stale, and kept
    e = figs['fig1_secondary_distance']
    assert e['files'] == ['fig1_secondary_distance.svg']
    assert e['stale_files_from_an_earlier_run'] == ['fig1_secondary_distance.png']
    assert (out/'fig1_secondary_distance.png').read_bytes() == b'an earlier matplotlib run'


def test_a_failed_drawing_reports_its_own_files_apart_from_stale_ones(tmp_path, capsys, monkeypatch):
    """The PNG written just before the SVG failed is this run's (perhaps incomplete), not a stale one."""
    out = tmp_path/'figures'
    out.mkdir()
    (out/'fig1_secondary_distance.svg').write_text('<svg/>', encoding='utf-8')  # from an earlier run

    def png_then_fail(model, png, svg, started):
        started.append(png)
        Path(png).write_bytes(b'png of this run')
        raise RuntimeError('svg writer broke')
    monkeypatch.setattr(MF, 'matplotlib_or_reason', lambda: (types.SimpleNamespace(__version__='test'), None))
    monkeypatch.setattr(MF, 'render_mpl', png_then_fail)
    ws = write_reports(tmp_path/'score', secondary=one_fiber_report())
    rc, text = run(capsys, '--workspace', ws, '--out', out)
    assert rc == 0
    e = captions(out)['figures']['fig1_secondary_distance']
    assert e['skipped'] == 'drawing failed: RuntimeError: svg writer broke'
    assert e['written_by_this_run_before_the_failure'] == ['fig1_secondary_distance.png']
    assert e['stale_files_from_an_earlier_run'] == ['fig1_secondary_distance.svg']
    assert 'written by this run before the failure, may be incomplete: fig1_secondary_distance.png' in text


def test_malformed_reports_never_stop_the_tool(tmp_path, capsys):
    junk = {'accuracy': {'all': {'strip': {'point': 'x'}}, 'H': 5},
            'per_strip': [1, None, {'fiber': 5}, {'fiber': 'a', 'strip_nodes': 'many', 'strip_mean_abs_sv': [1]},
                          {'fiber': 'b', 'strip_nodes': True, 'strip_mean_abs_sv': 1.}],
            'calibration': {'bins': [{'nodes': 10, 'mean_expected': None, 'mean_observed': 1.}, 'bin']}}
    cross = {'crossings': {'per_crossing': [{'id': 'X0000', 'agreement_sv': 'abc', 'closest_approach_sv': 1.},
                                            {'id': 'X0001', 'agreement_sv': float('nan'), 'closest_approach_sv': 1.}],
                           'planned': 2, 'missing': ['X0002']}}
    ws = write_reports(tmp_path/'score', secondary=json.dumps(junk), primary='[1, 2]',
                       crossings_0332=json.dumps(cross).replace('NaN', 'null'))
    rc, text = run(capsys, '--workspace', ws, '--out', tmp_path/'figures', '--backend', 'svg')
    assert rc == 0
    cap = captions(tmp_path/'figures')
    assert cap['reports']['primary']['status'].startswith('not a JSON object')
    figs = cap['figures']
    assert figs['fig1_secondary_distance']['skipped'].startswith('the secondary report has no scored strip nodes')
    assert figs['fig2_secondary_per_fiber']['skipped'] == 'no scored fiber in the secondary report (per_strip)'
    assert figs['fig3_0332_crossings']['skipped'] == ('no crossing with an agreement in the report '
                                                      '(planned 2, missing 1)')
    assert figs['fig4_calibration']['skipped'] == "the secondary report's calibration has no filled bin"
    assert cap['example_strips']['skipped'] == 'no scored fiber in the secondary report (per_strip)'
    (ws/MF.REPORTS['secondary']).write_text('{"accuracy": ', encoding='utf-8')  # truncated
    rc, text = run(capsys, '--workspace', ws, '--out', tmp_path/'figures2', '--backend', 'svg')
    assert rc == 0 and captions(tmp_path/'figures2')['reports']['secondary']['status'].startswith('unreadable')


@pytest.mark.parametrize('change', ['strip nodes', 'scored fibers', 'pooled mean'])
def test_warning_when_per_strip_does_not_reproduce_the_aggregate(tmp_path, capsys, change):
    """Each of the three numbers checked, changed alone in accuracy.all, raises the WARNING."""
    sec = edge_secondary()
    acc = sec['accuracy']['all']
    if change == 'strip nodes':
        acc['strip']['point']['nodes'] += 1
    elif change == 'scored fibers':
        acc['scored_fibers'] += 1
    else:
        acc['strip']['point']['mean'] += 1e-6
    ws = write_reports(tmp_path/'score', secondary=sec)
    rc, text = run(capsys, '--workspace', ws, '--out', tmp_path/'figures', '--backend', 'svg')
    assert rc == 0 and "WARNING: per_strip does not reproduce the report's aggregate" in text
    assert not captions(tmp_path/'figures')['example_strips']['consistency_with_report']['ok']


def test_no_warning_when_per_strip_reproduces_the_aggregate(tmp_path, capsys):
    ws = write_reports(tmp_path/'score', secondary=edge_secondary())
    rc, text = run(capsys, '--workspace', ws, '--out', tmp_path/'figures', '--backend', 'svg')
    check = captions(tmp_path/'figures')['example_strips']['consistency_with_report']
    assert rc == 0 and 'WARNING' not in text and check['ok']
    assert check['scored_fibers'] == [8, 8] and check['strip_nodes'] == [3000, 3000]


# ---- what is drawn: the figure models against the report ------------------------------------------------------------

def models_for(ws, primary_overlay=True):
    reports, status = MF.read_reports(ws)
    fibers, check, picks = MF.per_fiber(reports['secondary'])
    return MF.build_models(reports, status, fibers, picks, primary_overlay), reports, picks


def panel_named(model, name):
    (p,) = [p for p in model['panels'] if p['name'] == name]
    return p


def find(model, panel, role):
    return [it for it in panel_named(model, panel)['items'] if it.get('role') == role]


def the(model, panel, role):
    (it,) = find(model, panel, role)
    return it


def fiber_means(per_strip, surface):
    """{fiber: (nodes, node-weighted mean |d|)} of one surface, straight from the report's per_strip rows."""
    acc = {}
    for r in per_strip:
        n, m = r.get(f'{surface}_nodes') or 0, r.get(f'{surface}_mean_abs_sv')
        if n and m is not None:
            a = acc.setdefault(r['fiber'], [0, 0.])
            a[0] += n
            a[1] += n*m
    return {f: (n, s/n) for f, (n, s) in acc.items()}


def edge_workspace(tmp_path, **extra):
    return write_reports(tmp_path/'edge'/'score', secondary=edge_secondary(), primary=edge_primary(),
                         crossings_0332=edge_crossings(), **extra)


@pytest.mark.parametrize('which', ['scorer', 'edge'])
def test_fig1_draws_the_reported_points_thresholds_and_fiber_means(world, tmp_path, which):
    ws = world['full'] if which == 'scorer' else edge_workspace(tmp_path)
    models, reports, _ = models_for(ws)
    m, sec = models['fig1_secondary_distance'], reports['secondary']
    acc, C = sec['accuracy']['all'], MF.constants(sec)
    sources = {'strip': (acc['strip']['point'], acc['strip']['ci'], 'nodes'),
               'frame_plane': (acc['frame_plane']['stats']['point'], acc['frame_plane']['stats']['ci'], 'nodes'),
               'lasagna_plane': (acc['lasagna_plane']['stats']['point'], acc['lasagna_plane']['stats']['ci'], 'nodes'),
               'fiber_floor': (acc['fiber_floor'], None, 'rows')}
    quantities = ('mean', 'median', 'p90', 'within1', 'within2')
    for base in ('frame_plane', 'lasagna_plane'):  # so drawing the strip-on-same-nodes numbers would be caught
        assert any(acc[base]['stats']['point'][q] != acc[base]['strip_on_same_nodes'][q] for q in quantities)
    offsets = {r['surface']: r['drawn_offset'] for r in m['data']['node_level']}
    assert list(offsets) == list(sources)
    for key, (pt, ci, unit) in sources.items():
        it = the(m, 'a', f'series:{key}')
        assert it['tags'] == ['median', 'p90', 'within1', 'within2'], key
        drawn = dict(zip(it['tags'], zip(it['x'], it['y'], it['xint'], it['yint'])))
        off = offsets[key]
        for q, frac in (('median', .5), ('p90', .9)):  # x = the reported value, y = its CDF level
            assert drawn[q] == (pt[q], frac+off.get(q, 0.), ci[q] if ci else None, None), (key, q)
        for q, thr in (('within1', 1.), ('within2', 2.)):  # y = the reported fraction, x = its threshold
            assert drawn[q] == (thr+off.get(q, 0.), pt[q], None, ci[q] if ci else None), (key, q)
        assert all(abs(o) <= 1.5*(.015 if q in ('median', 'p90') else .012*m['panels'][0]['xlim'][1])
                   for q, o in off.items())
        mean = the(m, 'means', f'mean:{key}')
        assert mean['x'] == [pt['mean']] and mean['xint'] == [ci['mean'] if ci else None]
        assert it['label'].endswith(f"({pt['nodes']:,} {unit})")  # the floor's count is of scored rows
    assert the(m, 'a', 'within2_line')['x'] == 2.0
    seg = the(m, 'a', 'g2_within2')
    assert seg['y0'] == seg['y1'] == C['G2_MIN_WITHIN2'] == .7 and seg['x0'] < 2.0 < seg['x1']
    assert the(m, 'means', 'g2_mean')['x'] == C['G2_MAX_MEAN'] == 1.5
    for name in ('a', 'means'):
        assert panel_named(m, name)['xlabel'].endswith('|d| to the matched wrap segment (sv)')
    # panel (b): each surface's steps at its per-fiber means (fibers with MIN_NODES of its nodes); the floor's mean
    strip = fiber_means(sec['per_strip'], 'strip')
    for s in MF.SURFACES:
        vals = sorted(mm for f, (n, mm) in fiber_means(sec['per_strip'], s).items()
                      if n >= 25 and strip[f][0] >= 25)
        ln = the(m, 'b', f'ecdf:{s}')
        k = len(vals)
        assert ln['x'][1:-1] == pytest.approx([v for v in vals for _ in (0, 1)], rel=1e-12)
        assert ln['y'][1:-1] == [y for i in range(k) for y in (i/k, (i+1)/k)]
        assert m['data']['per_fiber'][s] == pytest.approx(vals, rel=1e-12)
    assert the(m, 'b', 'floor_mean')['x'] == acc['fiber_floor']['mean']
    assert f"mean {acc['strip']['point']['mean']:.2f} sv" in m['caption']


def test_dodge_moves_only_marks_that_overlap():
    assert MF.dodge([(0, 1), (2, 3)], .1) == [0., 0.]
    assert MF.dodge([(0, 1), (.5, 2), (5, 6)], .1) == [-.05, .05, 0.]
    assert MF.dodge([(1.4, 2), (0, 1), (.9, 1.5)], .1) == pytest.approx([-.1, 0., .1])  # a chain, spread in list order
    assert MF.dodge([(0, 1), (0, 1), (0, 1), (0, 1)], .2) == pytest.approx([-.3, -.1, .1, .3])
    assert MF.dodge([(3, 4)], .1) == [0.] and MF.dodge([], .1) == []


def test_fig1_marks_move_only_where_they_overlap(tmp_path):
    """The edge report's coinciding pairs are pulled apart, symmetrically and visibly; every other mark sits exactly
    at its value; the caption gives the largest move."""
    m = models_for(edge_workspace(tmp_path))[0]['fig1_secondary_distance']
    moved = {(r['surface'], q): o for r in m['data']['node_level'] for q, o in r['drawn_offset'].items()}
    pairs = {'within1': ('strip', 'frame_plane'), 'within2': ('strip', 'lasagna_plane'),
             'median': ('strip', 'lasagna_plane')}
    assert set(moved) == {(s, q) for q, pair in pairs.items() for s in pair}
    for q, (a, b) in pairs.items():
        assert moved[(a, q)] == -moved[(b, q)] < 0
    xmax = panel_named(m, 'a')['xlim'][1]
    assert moved[('strip', 'within1')] == pytest.approx(-.5*.012*xmax)
    assert moved[('strip', 'median')] == pytest.approx(-.5*.015)
    assert (f"by at most {.5*.015:.3f} up or down (median, p90) and {.5*.012*xmax:.2f} sv sideways"
            in m['caption'])
    # no overlap at all: the caption says every mark is at its value
    sec = edge_secondary()
    fp, lp = sec['accuracy']['all']['frame_plane']['stats'], sec['accuracy']['all']['lasagna_plane']['stats']
    fp['point']['within1'], fp['ci']['within1'] = .6, [.55, .65]
    lp['point']['median'], lp['ci']['median'] = 1.6, [1.5, 1.7]
    lp['point']['within2'], lp['ci']['within2'] = .7, [.65, .75]
    m = models_for(write_reports(tmp_path/'apart'/'score', secondary=sec))[0]['fig1_secondary_distance']
    assert all(r['drawn_offset'] == {} for r in m['data']['node_level'])
    assert 'no two marks overlap, so every mark is at its value' in m['caption']
    # points far apart but CI bars overlapping are moved too: within 2 sv, the frame plane's bar [0.20, 0.42] reaches
    # the strip's [0.40, 0.50] (points 0.15 apart); median, the Lasagna plane's bar [1.5, 2.45] reaches the strip's
    # [2.4, 2.8] (points 1 sv apart)
    fp['point']['within2'], fp['ci']['within2'] = .3, [.2, .42]
    lp['ci']['median'] = [1.5, 2.45]
    m = models_for(write_reports(tmp_path/'bars'/'score', secondary=sec))[0]['fig1_secondary_distance']
    moved = {(r['surface'], q) for r in m['data']['node_level'] for q in r['drawn_offset']}
    assert moved == {('strip', 'within2'), ('frame_plane', 'within2'), ('strip', 'median'), ('lasagna_plane', 'median')}


@pytest.mark.parametrize('which', ['scorer', 'edge'])
def test_fig2_draws_strip_against_baseline_per_fiber_and_the_reports_difference(world, tmp_path, which):
    ws = world['full'] if which == 'scorer' else edge_workspace(tmp_path)
    models, reports, picks = models_for(ws)
    m, sec = models['fig2_secondary_per_fiber'], reports['secondary']
    acc = sec['accuracy']['all']
    strip = fiber_means(sec['per_strip'], 'strip')
    for base in ('frame_plane', 'lasagna_plane'):
        bm = fiber_means(sec['per_strip'], base)
        want = {f: (bm[f][1], strip[f][1]) for f in bm if bm[f][0] >= 25 and strip[f][0] >= 25}
        drawn = {}
        for it in find(m, base, 'fibers'):
            drawn.update({f: (x, y) for f, x, y in zip(it['tags'], it['x'], it['y'])})
        assert drawn.keys() == want.keys()
        for f, xy in want.items():  # x: the baseline's mean, y: the strip's
            assert drawn[f] == pytest.approx(xy, rel=1e-12), (base, f)
        p = panel_named(m, base)
        assert p['xlabel'] == f'mean |d|, {MF.LABEL[base]} (sv)' and p['ylabel'] == 'mean |d|, strip (sv)'
        dia = the(m, base, 'pooled')
        assert dia['x'] == [acc[base]['stats']['point']['mean']]
        assert dia['y'] == [acc[base]['strip_on_same_nodes']['mean']]
        assert dia['xint'] == [acc[base]['stats']['ci']['mean']]
        d = acc[base]['delta']
        # baseline minus strip, with the report's sign: the pooled means on the shared nodes give the same number
        assert d['baseline_minus_strip_mean_sv'] == pytest.approx(
            acc[base]['stats']['point']['mean']-acc[base]['strip_on_same_nodes']['mean'])
        assert f"baseline \u2212 strip = {d['baseline_minus_strip_mean_sv']:.2f} sv" in p['title']
        assert (': beaten\n' in p['title']) == d['beaten'] and (': not beaten\n' in p['title']) != d['beaten']
        for kind in ('median', 'p90'):
            f = picks[kind]['fiber']
            rings = find(m, base, f'example:{kind}')
            assert [(r['x'][0], r['y'][0]) for r in rings] == ([drawn[f]] if f in want else [])
    if which == 'edge':  # the p90 example fiber f5: strip 10.0 sv against frame plane 3.1 and Lasagna plane 1.1 sv
        assert picks['p90']['fiber'] == 'f5'
        assert the(m, 'frame_plane', 'example:p90')['x'] + the(m, 'frame_plane', 'example:p90')['y'] == \
            pytest.approx([3.1, 10.])
        assert the(m, 'lasagna_plane', 'example:p90')['x'][0] == pytest.approx(1.1)


def test_min_nodes_boundary_is_inclusive(tmp_path):
    """A fiber with exactly MIN_NODES strip nodes is scored and one with a node fewer is not; a baseline with exactly
    MIN_NODES nodes is drawn in figures 1 (b) and 2; MIN_NODES is the report's constant."""
    fibers = MF.fiber_table(per_strip([('a', 'P1', 25, 1.), ('b', 'P2', 24, 2.), ('c', 'P3', 26, 3.)]), 25)
    assert fibers['a']['scored'] and not fibers['b']['scored'] and fibers['c']['scored']
    sec = edge_secondary()
    for r in sec['per_strip']:
        r['frame_plane_nodes'] = {'f1': 25, 'f2': 24}.get(r['fiber'], r['frame_plane_nodes'])
    m = models_for(write_reports(tmp_path/'b'/'score', secondary=sec))[0]
    drawn = {f for it in find(m['fig2_secondary_per_fiber'], 'frame_plane', 'fibers') for f in it['tags']}
    assert 'f1' in drawn and 'f2' not in drawn and len(drawn) == 7
    assert len(m['fig1_secondary_distance']['data']['per_fiber']['frame_plane']) == 7
    # the report says MIN_NODES 30: fibers, baselines and the no-comparison subtitle follow it
    sec = edge_secondary()
    sec['provenance']['constants']['MIN_NODES'] = 30
    sec['accuracy']['all']['frame_plane']['delta'] = None
    for r in sec['per_strip']:
        r['frame_plane_nodes'] = {'f1': 30, 'f2': 29}.get(r['fiber'], r['frame_plane_nodes'])
    f2 = models_for(write_reports(tmp_path/'c'/'score', secondary=sec))[0]['fig2_secondary_per_fiber']
    drawn = {f for it in find(f2, 'frame_plane', 'fibers') for f in it['tags']}
    assert 'f1' in drawn and 'f2' not in drawn
    assert panel_named(f2, 'frame_plane')['title'].endswith('(fewer than 2 fibers with 30 nodes covered by both)')


def test_fig3_tight_subset_is_inclusive_filled_and_drawn_at_its_values(tmp_path):
    cro = edge_crossings()
    m = models_for(write_reports(tmp_path/'score', crossings_0332=cro))[0]['fig3_0332_crossings']
    tight, loose = the(m, 'crossings', 'tight'), the(m, 'crossings', 'loose')
    assert tight['tags'] == ['X0000', 'X0001', 'X0002'] and tight['filled']  # X0002 sits at exactly 3.0 sv
    assert loose['tags'] == ['X0003', 'X0004'] and not loose['filled']
    rows = {r['id']: r for r in cro['crossings']['per_crossing']}
    for it in (tight, loose):  # x: closest approach of the two human fibers, y: agreement of the two strips
        assert list(zip(it['x'], it['y'])) == [(rows[i]['closest_approach_sv'], rows[i]['agreement_sv'])
                                               for i in it['tags']]
    assert {r['id']: r['tight'] for r in m['data']['per_crossing']} == {
        'X0000': True, 'X0001': True, 'X0002': True, 'X0003': False, 'X0004': False}
    res = cro['crossings']
    assert the(m, 'crossings', 'median')['y'] == res['median']
    band = the(m, 'crossings', 'median_ci')
    assert [band['a'], band['b']] == res['ci']['median'] and band['axis'] == 'y'
    region = the(m, 'crossings', 'tight_region')
    assert (region['a'], region['b'], region['axis']) == (0, 3.0, 'x')
    assert MF.NO_GT in panel_named(m, 'crossings')['title'] and MF.NO_GT in m['caption']
    assert 'missing:\n  X0005' in m['texts'][0]['s'] and 'One usable direction only: X0002.' in m['caption']


def test_fig4_draws_mean_and_median_observed_per_bin(tmp_path, capsys):
    ws = edge_workspace(tmp_path)
    m = models_for(ws)[0]['fig4_calibration']
    sec, prim = edge_secondary(), edge_primary()
    filled = [b for b in sec['calibration']['bins'] if b['nodes']]
    mean, median = the(m, 'calibration', 'bins_mean'), the(m, 'calibration', 'bins_median')
    assert mean['x'] == median['x'] == [b['mean_expected'] for b in filled]
    assert mean['y'] == [b['mean_observed'] for b in filled] and mean['filled']
    assert median['y'] == [b['median_observed'] for b in filled] and not median['filled']
    assert mean['tags'] == ['[0.00, 0.50)', '[0.50, 0.75)', '[0.75, 1.00)', '[1.00, 1.25)', '[1.25, 1.50)',
                            '[1.50, 2.00)']
    pm = the(m, 'calibration', 'primary_mean')
    assert pm['y'] == [b['mean_observed'] for b in prim['calibration']['bins']] and not pm['filled']
    assert '[1.50, 2.00): 400; 1.70, 5.20, 4.10' in m['texts'][0]['s']  # the bin table beside the plot
    assert '2 of 2 stretches scored; G1 failed: no accuracy claim' in m['caption']
    # --no-primary-overlay: the secondary set only, said in a note
    m = models_for(ws, primary_overlay=False)[0]['fig4_calibration']
    assert not find(m, 'calibration', 'primary_mean') and not m['data']['primary_bins']
    assert m['notes'] == ['The primary case studies are left out (--no-primary-overlay).']
    rc, text = run(capsys, '--workspace', ws, '--out', tmp_path/'figs', '--backend', 'svg', '--no-primary-overlay')
    assert rc == 0 and not captions(tmp_path/'figs')['figures']['fig4_calibration']['data']['primary_bins']


# ---- both backends: every point drawn, every legend outside the axes ---------------------------------------------

def low_example_secondary():
    """The edge report with four fibers near 0 sv, so the median example fiber (f4, 0.05 sv) sits at the bottom of
    figure 2, under a 10 sv fiber: its ring label must stay inside the axes."""
    sec = edge_secondary()
    for r in sec['per_strip']:
        r['strip_mean_abs_sv'] = {'f1': .02, 'f2': .03, 'f3': .04, 'f4': .05}.get(r['fiber'], r['strip_mean_abs_sv'])
    point = sec['accuracy']['all']['strip']['point']
    point['mean'] = sum(r['strip_nodes']*r['strip_mean_abs_sv'] for r in sec['per_strip'])/point['nodes']
    return sec


def layout_cases(world, tmp_path):
    """Every figure model of the scorer-built reports (with and without Lasagna normals), the degrading edge reports
    (also with the median example fiber near 0 sv) and the one-fiber report."""
    one = write_reports(tmp_path/'one'/'score', secondary=one_fiber_report())
    low = write_reports(tmp_path/'low'/'score', secondary=low_example_secondary())
    cases = {}
    for name, ws in (('scorer', world['full']), ('scorer_nolas', world['nolas']), ('edge', edge_workspace(tmp_path)),
                     ('low_example', low), ('one_fiber', one)):
        models, _, picks = models_for(ws)
        if name == 'low_example':
            assert picks['median']['fiber'] == 'f4' and picks['median']['mean_abs_sv'] == .05
        for fig, m in models.items():
            if 'skipped' not in m:
                cases[f'{name}:{fig}'] = m
    assert len(cases) == 4+3+4+3+1
    return cases


def _overlap(a, b):
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def _inside(a, b, tol=1.):
    return a.x0 >= b.x0-tol and a.y0 >= b.y0-tol and a.x1 <= b.x1+tol and a.y1 <= b.y1+tol


def mpl_layout_problems(fig):
    """What a reader would see covered or cut: a legend over an axes, over its ticks, labels or title, over a figure
    text or another legend, or over a data vertex; a legend or text outside the figure; an annotation outside its
    axes; two axes' labels or titles running into each other."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    legends = [(i, ax.get_legend()) for i, ax in enumerate(fig.axes) if ax.get_legend() is not None]
    for _, lg in legends:
        lg.set_visible(False)
    tight = [ax.get_tightbbox(r) for ax in fig.axes]  # all an axes draws but its legend
    for _, lg in legends:
        lg.set_visible(True)
    boxes = [(i, lg.get_window_extent(r)) for i, lg in legends]
    problems = []
    for i, lb in boxes:
        if not _inside(lb, fig.bbox):
            problems.append(f'legend of axes {i} leaves the figure')
        for j, ax in enumerate(fig.axes):
            if _overlap(lb, ax.get_window_extent(r)):
                problems.append(f'legend of axes {i} over axes {j}')
            if _overlap(lb, tight[j]):
                problems.append(f'legend of axes {i} over the ticks, labels or title of axes {j}')
            for ln in ax.lines:
                hidden = [p for p in ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
                          if lb.contains(*p)]
                if hidden:
                    problems.append(f'legend of axes {i} covers {len(hidden)} data vertices of axes {j}')
        for t in fig.texts:
            if _overlap(lb, t.get_window_extent(r)):
                problems.append(f'legend of axes {i} over the figure text {t.get_text()[:30]!r}')
        for k, lb2 in boxes:
            if k > i and _overlap(lb, lb2):
                problems.append(f'legends of axes {i} and {k} overlap')
    for t in fig.texts:
        tb = t.get_window_extent(r)
        if not _inside(tb, fig.bbox):
            problems.append(f'figure text {t.get_text()[:30]!r} leaves the figure')
        problems += [f'figure text {t.get_text()[:30]!r} over axes {j}' for j, ab in enumerate(tight)
                     if _overlap(tb, ab)]
    for j, ax in enumerate(fig.axes):
        ab = ax.get_window_extent(r)
        problems += [f'annotation {t.get_text()!r} leaves axes {j}' for t in ax.texts
                     if not _inside(t.get_window_extent(r), ab)]
        problems += [f'axes {j} and {k} run into each other' for k in range(j+1, len(tight))
                     if _overlap(tight[j], tight[k])]
    return problems


@pytest.mark.skipif(not HAVE_MPL, reason='needs matplotlib')
def test_matplotlib_legends_never_cover_data_and_nothing_is_cut(world, tmp_path):
    mpl, reason = MF.matplotlib_or_reason()
    assert mpl is not None, reason
    from matplotlib import pyplot as plt
    problems = []
    for name, m in layout_cases(world, tmp_path).items():
        with MF.mpl_style():
            fig = MF.draw_mpl(m)
            try:
                assert sum(ax.get_legend() is not None for ax in fig.axes) == sum(
                    bool(p['legend_at']) and any(it.get('label') for it in p['items']) for p in m['panels']) > 0
                problems += [f'{name}: {p}' for p in mpl_layout_problems(fig)]
            finally:
                plt.close(fig)
    assert not problems, '\n'.join(problems)


@pytest.mark.skipif(not HAVE_MPL, reason='needs matplotlib')
def test_matplotlib_draws_every_point_of_the_model(world, tmp_path):
    MF.matplotlib_or_reason()
    from matplotlib import pyplot as plt
    for name, m in layout_cases(world, tmp_path).items():
        with MF.mpl_style():
            fig = MF.draw_mpl(m)
            try:
                assert len(fig.axes) == len(m['panels'])
                for ax, p in zip(fig.axes, m['panels']):
                    lines = [(ln.get_marker(), ln.get_markerfacecolor(), np.asarray(ln.get_xydata(), float))
                             for ln in ax.lines]
                    for it in p['items']:
                        if it['kind'] != 'points':
                            continue
                        want = np.c_[np.asarray(it['x'], float), np.asarray(it['y'], float)]
                        hits = [(mk, fc) for mk, fc, xy in lines if xy.shape == want.shape and np.array_equal(xy, want)
                                and mk == it['marker']]
                        assert hits, (name, p['name'], it['role'])
                        assert any((fc == 'none') != it['filled'] for _, fc in hits), (name, p['name'], it['role'])
            finally:
                plt.close(fig)


def _tag(e):
    return e.tag.rsplit('}', 1)[-1]


def _box(e):
    x, y, w, h = (float(e.get(k)) for k in ('x', 'y', 'width', 'height'))
    return types.SimpleNamespace(x0=x, y0=y, x1=x+w, y1=y+h)


def _centre(e):
    """The centre a point marker was drawn at (circle, square, diamond or triangle of the SVG backend)."""
    if _tag(e) == 'circle':
        return float(e.get('cx')), float(e.get('cy'))
    if _tag(e) == 'rect':
        b = _box(e)
        return (b.x0+b.x1)/2, (b.y0+b.y1)/2
    pts = [tuple(map(float, p.split(','))) for p in e.get('points').split()]
    if len(pts) == 4:  # diamond
        return (pts[0][0]+pts[2][0])/2, (pts[1][1]+pts[3][1])/2
    return pts[0][0], (.7*pts[0][1]+1.1*pts[1][1])/1.8  # triangle: apex at 1.1 r, base at 0.7 r from the centre


def test_svg_backend_draws_every_point_and_keeps_legends_off_the_axes(world, tmp_path):
    for name, m in layout_cases(world, tmp_path).items():
        svg = tmp_path/(name.replace(':', '_')+'.svg')
        MF.render_svg(m, svg, [])
        root = ET.parse(svg).getroot()
        W, H = (float(v) for v in root.get('viewBox').split()[2:])
        page = types.SimpleNamespace(x0=0., y0=0., x1=W, y1=H)
        axes = [_box(next(e for e in cp if _tag(e) == 'rect')) for cp in root.iter() if _tag(cp) == 'clipPath']
        legends = [_box(e) for e in root.iter() if _tag(e) == 'rect' and e.get('class') == 'legend']
        marks = [_centre(e) for e in root.iter() if e.get('class') == 'pt']
        assert len(axes) == len(m['panels']) and legends, name
        for i, p in enumerate(m['panels']):  # every point of the model, where the model puts it
            (group,) = [g for g in root.iter() if _tag(g) == 'g' and g.get('clip-path') == f'url(#clip{i})']
            got = [_centre(e) for e in group if e.get('class') == 'pt']
            l, b, w, h = p['rect']
            (x0, x1), (y0, y1) = p['xlim'], p['ylim']
            want = [(l*W+(x-x0)/(x1-x0)*w*W, (1-b-h)*H+h*H-(y-y0)/(y1-y0)*h*H)
                    for it in p['items'] if it['kind'] == 'points' for x, y in zip(it['x'], it['y'])]
            assert len(got) == len(want), (name, p['name'])
            for c in want:
                assert any(abs(c[0]-g[0]) < .02 and abs(c[1]-g[1]) < .02 for g in got), (name, p['name'], c)
        for lb in legends:
            assert _inside(lb, page), name
            for ab in axes:
                assert not _overlap(lb, ab), name
                if lb.x0 < ab.x1 and ab.x0 < lb.x1 and lb.y0 >= ab.y1:  # under this axes: below its x label too
                    assert lb.y0-ab.y1 >= 36, name
            assert not [c for c in marks if lb.x0 <= c[0] <= lb.x1 and lb.y0 <= c[1] <= lb.y1], name
        assert not [(a, b) for k, a in enumerate(legends) for b in legends[k+1:] if _overlap(a, b)], name


@pytest.mark.skipif(not HAVE_MPL, reason='needs matplotlib for the PNGs')
def test_review_case_points_are_not_hidden_by_a_legend(tmp_path, capsys):
    """The review's case, through the CLI: the points it found under the legends (fig2 p90 example (3.1, 10.0) and
    (1.1, 10.0), fig3 X0000 (0.8, 9.5), fig4 bin [1.5, 2) at (1.7, 5.2)) lie in no legend of the written figures."""
    MF.matplotlib_or_reason()
    from matplotlib import pyplot as plt
    ws = edge_workspace(tmp_path)
    rc, _ = run(capsys, '--workspace', ws, '--out', tmp_path/'figs')
    assert rc == 0 and all('skipped' not in e for e in captions(tmp_path/'figs')['figures'].values())
    models = models_for(ws)[0]
    for fig_name, panel, xy in (('fig2_secondary_per_fiber', 'frame_plane', (3.1, 10.)),
                                ('fig2_secondary_per_fiber', 'lasagna_plane', (1.1, 10.)),
                                ('fig3_0332_crossings', 'crossings', (.8, 9.5)),
                                ('fig4_calibration', 'calibration', (1.7, 5.2))):
        m = models[fig_name]
        i = [p['name'] for p in m['panels']].index(panel)
        with MF.mpl_style():
            fig = MF.draw_mpl(m)
            try:
                fig.canvas.draw()
                r = fig.canvas.get_renderer()
                px = fig.axes[i].transData.transform(xy)
                assert fig.axes[i].get_window_extent(r).contains(*px)
                assert not [ax for ax in fig.axes if ax.get_legend() is not None
                            and ax.get_legend().get_window_extent(r).contains(*px)], (fig_name, xy)
            finally:
                plt.close(fig)


# ---- the example-strip rule (fixed before the run) -----------------------------------------------------------------

def per_strip(spec):
    return [{'strip': s, 'fiber': f, 'family': 'V', 'stretches': [], 'strip_nodes': n, 'strip_mean_abs_sv': m}
            for f, s, n, m in spec]


def pick(means, nodes=100):
    return MF.pick_examples(MF.fiber_table(per_strip([(f, f'P_{f}', nodes, m) for f, m in means.items()]), 25))


def test_example_rule_median_lower_middle_and_p90_nearest_rank():
    p = pick({'a': 3., 'b': 1., 'c': 2., 'd': 5., 'e': 4.})  # odd count: the middle
    assert (p['median']['fiber'], p['median']['rank'], p['p90']['fiber'], p['p90']['rank']) == ('a', 3, 'd', 5)
    p = pick({f'f{i}': float(i) for i in range(1, 7)})  # even count: the lower of the two middle values
    assert (p['median']['fiber'], p['median']['mean_abs_sv'], p['p90']['fiber']) == ('f3', 3., 'f6')
    p = pick({f'f{i:02d}': float(i) for i in range(1, 11)})  # 10 fibers: p90 = rank ceil(9.0) = 9
    assert (p['median']['fiber'], p['p90']['fiber']) == ('f05', 'f09')
    p = pick({f'f{i:02d}': float(i) for i in range(1, 37)})  # 36, the planned secondary fibers: ranks 18 and 33
    assert (p['median']['rank'], p['p90']['rank'], p['n']) == (18, 33, 36)
    p = pick({f'f{i:02d}': float(i) for i in range(1, 31)})  # 30, the G1' minimum: ranks 15 and 27
    assert (p['median']['rank'], p['p90']['rank']) == (15, 27)
    one = pick({'only': 1.})
    assert one['median']['fiber'] == one['p90']['fiber'] == 'only'
    assert pick({}) is None


def test_example_rule_ties_go_to_the_smaller_fiber_name():
    p = pick({'x': 1., 'c': 2., 'b': 2., 'd': 3.})  # rank 2 of 4 holds 2.0, shared by b and c
    assert p['median']['fiber'] == 'b' and p['median']['fibers_with_this_value'] == 2
    p = pick({'a': 1., 'b': 2., 'r': 5., 'q': 5., 'p': 5.})  # p90 = rank 5, the entry (5.0, r): the value goes to p
    assert (p['median']['fiber'], p['p90']['fiber'], p['p90']['fibers_with_this_value']) == ('p', 'p', 3)


def test_fiber_means_are_node_weighted_and_need_min_nodes():
    fibers = MF.fiber_table(per_strip([('a', 'P2', 300, 2.), ('a', 'P1', 100, 1.), ('b', 'P3', 24, .1),
                                       ('c', 'P4', 0, None), ('c', 'P5', 25, 4.)]), 25)
    assert fibers['a']['mean']['strip'] == 1.75 and fibers['a']['nodes']['strip'] == 400
    assert [s['strip'] for s in fibers['a']['strips']] == ['P1', 'P2']
    assert not fibers['b']['scored'] and fibers['c']['scored'] and fibers['c']['mean']['strip'] == 4.
    p = MF.pick_examples(fibers)
    assert (p['n'], p['median']['fiber'], p['p90']['fiber']) == (2, 'a', 'c')


# ---- the render command ---------------------------------------------------------------------------------------------

def test_render_block_quotes_paths_and_renders_a_copy():
    ws, rd, out = Path('C:/W/score'), Path("C:/W/ren ders"), Path("C:/F G/it's")
    mkdir, copy_strip, copy_files, render, copy_png = MF.render_block('P0001', 123, 'median', ws, rd, out)
    assert mkdir == f"New-Item -ItemType Directory -Force -Path {MF.ps_quote(rd/'P0001'/'strip')} | Out-Null"
    assert MF.ps_quote(out/'example_strip_median_P0001.png') in copy_png and "it''s" in copy_png
    assert f"--out {MF.ps_quote(rd/'P0001')} " in render and str(ws/'runs'/'v3'/'strips') not in render
    assert sorted(MF.RENDER_READS) == ['strip', 'strip.npz', 'summary.json']


def test_render_inputs_match_a_real_infer_py_output():
    """The copy holds what villa's render_saved reads; the Paris 4 fixture is a real infer.py OUT of one piece."""
    out = REPO/'score'/'tests'/'fixtures'/'paris4_real'/'pieces'/'QA'
    assert sorted(p.name for p in out.iterdir()) == sorted(MF.RENDER_READS)
    assert {p.name for p in (out/'strip').iterdir()} >= {'x.tif', 'y.tif', 'z.tif', 'meta.json'}
    summary = json.loads((out/'summary.json').read_text())
    assert {'row_step_voxels', 'column_step_voxels', 'first_row_arc_patch_voxels', 'fiber'} <= set(summary)
    tool = MF.RENDER_TOOL.read_text(encoding='utf-8')
    assert tool.index('INF.bind_normalization(spec)') < tool.index('CTVolume(spec)')


@pytest.mark.skipif(not VILLA_INFER.is_file(),
                    reason='needs the pinned villa worktree C:/VesuviusTemp/strips/villa_6f2')
def test_render_command_flags_exist_in_the_pinned_villa():
    src = VILLA_INFER.read_text(encoding='utf-8')
    runtime = (VILLA_INFER.parents[1]/'tracing'/'runtime.py').read_text(encoding='utf-8')
    for flag in ('--checkpoint', '--device'):  # tools/render_strip.py parses these with infer.add_checkpoint_arguments
        assert f"'{flag}'" in src
    for name in ('def add_checkpoint_arguments', 'def bind_normalization', 'def render_saved'):
        assert name in src
    assert MF.RENDER_TOOL.is_file()
    for flag in ('--ct', '--cache-dir'):
        assert f"'{flag}'" in runtime
    body = src[src.index('def render_saved'):src.index('def main')]
    assert "out/'summary.json'" in body and "out/'strip.npz'" in body and "out/'strip'/f'{axis}.tif'" in body
    assert "save_render(out/'render.png'" in body
    main = src[src.index('def main'):]
    branch = main[main.index('if args.render_only:'):main.index('StripPredictor(')]
    assert 'render_saved(' in branch and 'bind_normalization' not in branch  # why tools/render_strip.py exists
