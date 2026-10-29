"""Tests of tools/fill_results.py on synthetic reports only; no output of the real run is read.

The reports are made by the frozen scorer's own code (score/score_strips.py) on synthetic inputs built with the
helpers of score/tests/test_score_strips.py:
  - primary: `match`, then `score --crossings` with Lasagna normals, on a cylinder scene whose two fibers carry the
    primary fibers' names; S0000's strip has infer.py decisions (strip.npz), S0001's has none, so its frame-plane
    baseline is unavailable. Also `score` with no strip at all.
  - secondary: `score --run-only --crossings` with the loaders and score_strips replaced by synthetic per-strip node
    errors in score_strips' entry layout, so aggregate, the gates, sensitivity, calibration and per_strip are the
    scorer's: 'secondary' (32 scored fibers, 12 H and 20 V: transfers), 'degraded' (no Lasagna normals, no
    expected_error, 4 H fibers under 25 nodes, two skipped strips: degrades) and 'empty' (no scored strip).
  - PHerc0332: `crossings` on the 15 H/V crossings (6 sv) of a cylinder scene, 5 of them without a V strip, 2 of the
    rest not tight.
The runner's records (log.txt with a pre-freeze attempt, status.json, DONE, V3_FINAL) are written in run_all.ps1's
and relaunch_v3.ps1's formats (UTF-8 with a BOM, CRLF), as is a synthetic FREEZE.txt with the frozen scorer's sha256.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tools/test_fill_results.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO/'score'/'tests'))
import fill_results as F  # noqa: E402
from test_score_strips import (BASE, CX, CY, cylinder_grid, decisions_v, h_fiber, h_strip, v_fiber,  # noqa: E402
                               v_strip, write_fiber, write_infer_output, write_tifxyz, write_umbilicus)
import score_strips as S  # noqa: E402  (score/ is on sys.path through test_score_strips)

FIB0, FIB1 = F.PRIMARY_FIBER['S0000'], F.PRIMARY_FIBER['S0001']
FREEZE_LOCAL = '2026-10-28 20:00:00'
MINUS = '\u2212'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- synthetic reports, made by the scorer's own code -----------------------------------------------------------

def build_primary(root):
    """The runner's primary scoring (match, then score --crossings --lasagna-normals) on a two-fiber scene."""
    segs, fib, st = root/'segments', root/'fibers', root/'strips'
    write_tifxyz(segs/'w001'/'mesh'/'w001.tifxyz', cylinder_grid(200., 0., 700., -0.9, 0.9), 1.)
    write_tifxyz(segs/'w002'/'mesh'/'w002.tifxyz', cylinder_grid(212., 0., 700., -0.9, 0.9), 1.)
    umb = write_umbilicus(root/'umbilicus.json')
    fib.mkdir(parents=True)
    write_fiber(fib/f'{FIB0}.json', v_fiber(200., 0.), 'V')
    write_fiber(fib/f'{FIB1}.json', h_fiber(200., 350., -.7, .7), 'H')
    matches = root/'matches.json'
    S.main(['match', '--fibers', str(fib), '--segments', str(segs), '--umbilicus', str(umb), '--out', str(matches)])
    plan = json.loads(matches.read_text())
    assert [(p['id'], p['windows'], p['fiber']) for p in plan['pieces']] == [('P0000', ['S0000'], FIB0),
                                                                           ('P0001', ['S0001'], FIB1)]
    dec, length = decisions_v(200., 0., 100., 500., expected=.6, height_offset=.8)
    write_infer_output(st/'P0000', dec, length)                                 # OUT/P0000/strip + strip.npz
    write_tifxyz(st/'P0001'/'strip', h_strip(200., 350., -.6, .6, -.4), BASE)  # no strip.npz
    arrays = {}
    for f in S.load_fibers(fib)[0]:  # radial Lasagna normals at every fiber sample
        n = f.points-[CX, CY, 0.]
        n[:, 2] = 0.
        arrays[f.name+'__n'] = n/np.linalg.norm(n, axis=1, keepdims=True)
        arrays[f.name+'__valid'] = np.ones(len(n), bool)
    np.savez(root/'lasagna.npz', **arrays)
    common = ['score', '--fibers', str(fib), '--segments', str(segs), '--umbilicus', str(umb), '--crossings',
              '--matches', str(matches), '--lasagna-normals', str(root/'lasagna.npz')]
    S.main(common+['--strips', str(st), '--out', str(root/'primary.json')])
    (root/'no_strips').mkdir()
    S.main(common+['--strips', str(root/'no_strips'), '--out', str(root/'primary_empty.json')])
    return root/'primary.json', root/'primary_empty.json'


def secondary_matches(path, variant):
    nf, nh = (34, 4) if variant == 'degraded' else (32, 12)
    stretches = []
    for i in range(nf):
        ann = ('dj', 'kb', 'lt')[i % 3]
        for j in range(2):
            stretches.append(dict(id=f'S{len(stretches):04d}', fiber=f'{ann}_20260901T0000{i:02d}000_{i:06d}',
                                  family='H' if i < nh else 'V', annotator=ann, segment='w001', s0=100.*j,
                                  s1=100.*j+60., foldbacks=int(i < 3 and j == 0), jumps=0, run=True))
    stretches.append(dict(id=f'S{len(stretches):04d}', fiber='dj_not_in_the_run', family='V', annotator='dj',
                          segment='w001', s0=0., s1=60., foldbacks=0, jumps=0, run=False))
    g1 = {'fibers_with_stretch': nf, 'noise_floor_match_sv': .82, 'noise_floor_nf6_sv': 1.76, 'pass': True}
    path.write_text(json.dumps(dict(analysis='secondary', segments=[dict(name='w001', source='synthetic')], G1=g1,
                                    stretches=stretches, pieces=[])))


def synthetic_score_strips(variant):
    """score_strips stand-in: one strip per fiber over its run stretches, in score_strips' entry layout; strip errors
    around 0.4 sv, frame plane 3|e| + 2, Lasagna plane 2|e| + 1 on the same nodes."""
    def fn(strips, fibers, stretches, segments, axis=None, normals=None, log=print, pieces=None):
        if variant == 'empty':
            return {}, []
        rng = np.random.default_rng(20261029)
        by_fiber = {}
        for st in stretches:
            by_fiber.setdefault(st['fiber'], []).append(st)
        entries = {}
        for k, (fiber, sts) in enumerate(sorted(by_fiber.items())):
            per = 2 if variant == 'degraded' and sts[0]['family'] == 'H' else 6  # 2 rows x 2 stretches x 5 = 20 nodes
            R, C = per*len(sts), 5
            e = rng.normal(.1, .5, (R, C))
            err = dict(strip=e, frame_plane=3*e+2*np.sign(e))
            if variant != 'degraded':
                err['lasagna_plane'] = 2*e+np.sign(e)
            expected = None if variant == 'degraded' else np.abs(e)*.8+rng.uniform(.2, .4, (R, C))
            name = f'P{k:04d}'
            entries[name] = dict(strip=name, fiber=fiber, family=sts[0]['family'], assoc_median_sv=.3,
                                 stretches=[st['id'] for st in sts], rows=R, cols=C, notes=[], err=err,
                                 expected=expected, valid=np.ones((R, C), bool), row_seg=np.zeros(R, int),
                                 row_stretch=np.repeat([st['_n'] for st in sts], per),
                                 fiber_floor=rng.normal(0., .8, R), lasagna_angle=np.full(R, 5.))
        skips = []
        if variant == 'degraded':
            skips = [dict(strip='P0090', reason=f'no fiber within {S.ASSOC_MAX_SV} sv (median 41.3 sv): '
                                                'unit or association error'),
                     dict(strip='P0091', fiber='kb_x', reason='no strip row inside a matched stretch')]
        return entries, skips
    return fn


def build_secondary(root, variant):
    """The runner's secondary scoring (score --run-only --crossings) with synthetic strips in place of real ones."""
    matches, out = root/f'matches_{variant}.json', root/f'secondary_{variant}.json'
    secondary_matches(matches, variant)
    with mock.patch.multiple(S, load_fibers=lambda *a, **k: ([], []),
                             discover_segments=lambda *a, **k: [SimpleNamespace(name='w001')],
                             discover_strips=lambda *a, **k: [], score_strips=synthetic_score_strips(variant)):
        S.main(['score', '--matches', str(matches), '--run-only', '--strips', str(root), '--crossings',
                '--out', str(out)])
    return out


def build_0332(root):
    """The runner's PHerc0332 scoring (crossings --plan) on 5 H x 3 V fibers; v2 has no strip, h4 is 4.5 sv out."""
    fib, st = root/'fibers', root/'strips'
    fib.mkdir(parents=True)
    vs = {'v0': (0., .3), 'v1': (.3, -.5), 'v2': (-.3, None)}  # phi, strip offset (None: no strip)
    hs = {'h0': (200., 150., 0.), 'h1': (200., 220., 1.), 'h2': (200., 290., -.2), 'h3': (200., 360., .4),
          'h4': (204.5, 430., 0.)}  # radius, z, strip offset
    for name, (phi, delta) in vs.items():
        write_fiber(fib/f'{name}.json', v_fiber(200., phi), 'V')
        if delta is not None:
            write_tifxyz(st/name, v_strip(200., phi, 120., 480., delta), BASE)
    for name, (radius, z, delta) in hs.items():
        write_fiber(fib/f'{name}.json', h_fiber(radius, z, -.7, .7), 'H')
        write_tifxyz(st/name, h_strip(radius, z, -.6, .6, delta), BASE)
    fibers = S.load_fibers(fib)[0]
    crossings = S.find_crossings(fibers, S.CROSS_TOL_0332)
    for c in crossings:
        c['run'] = True
    plan = root/'plan.json'
    plan.write_text(json.dumps(S.jsonable(dict(crossings=crossings))))
    S.main(['crossings', '--fibers', str(fib), '--plan', str(plan), '--strips', str(st), '--out',
            str(root/'crossings.json')])
    # the same plan scored with no strip at all (the scorer's own crossing code)
    res = S.crossing_agreement([], fibers, plan=crossings)
    (root/'crossings_none.json').write_text(json.dumps(S.jsonable(dict(provenance=S.provenance(argparse.Namespace()),
                                                                       crossings=res))))
    return root/'crossings.json', root/'crossings_none.json'


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp('reports')
    (root/'p').mkdir()
    (root/'x').mkdir()
    primary, primary_empty = build_primary(root/'p')
    x0332, x0332_none = build_0332(root/'x')
    return dict(primary=primary, primary_empty=primary_empty, secondary=build_secondary(root, 'normal'),
                degraded=build_secondary(root, 'degraded'), empty=build_secondary(root, 'empty'),
                x0332=x0332, x0332_none=x0332_none)


# ---- synthetic runner records ------------------------------------------------------------------------------------

PIECES = {'primary': [('P0000', 29), ('P0001', 29)], 'secondary': [('P0003', 99), ('P0005', 4), ('P0006', 418)],
          '0332': [('C0000', 21), ('C0001', 21)]}
CHUNKS = {'primary': (2, 140), 'secondary': (46, 4914), '0332': (25, 1009)}
NORMAL = {('primary', 'P0000'): [(0, 61.3)], ('primary', 'P0001'): [(0, 58.9)],
          ('secondary', 'P0003'): [(0, 150.2)], ('secondary', 'P0005'): [(1, 30.5), (0, 31.)],
          ('secondary', 'P0006'): [(0, 300.4)], ('0332', 'C0000'): [(0, 45.2)], ('0332', 'C0001'): [(0, 44.8)]}


class Log:
    def __init__(self, start):
        self.t, self.lines = datetime.strptime(start, '%Y-%m-%d %H:%M:%S'), []

    def stamp(self):
        return self.t.strftime('%Y-%m-%d %H:%M:%S')

    def add(self, msg, seconds=1.):
        self.t += timedelta(seconds=seconds)
        self.lines.append(f'[{self.stamp()}] {msg}')


def pre_freeze_attempt(log):
    """The v3 runner's first attempt of 28 Oct: primary prefetch, then a stop in prefetch:secondary (rc 3)."""
    log.add('START v3 runner pid 43784 (BelowNormal), CUDA_VISIBLE_DEVICES=-1, HF_HUB_OFFLINE=1, C: 20.1 GiB free', 0)
    log.add('prefetch primary: start (C: 20.0 GiB free)', 37)
    log.add('[prefetch] primary: 2 pieces, 140 chunks (0.27 GiB), 0 cached, 140 to fetch (0.27 GiB), 2 connections')
    log.add('[prefetch] primary: 140/140 chunks, 280 MiB, 448 s, 0.63 MiB/s, C: 19.7 GiB free', 448)
    log.add('[prefetch] primary: done: 140 fetched (0 missing objects -> fill), 280 MiB in 448 s; 0 of 140 chunks '
            'not cached', 0)
    log.add('prefetch primary: rc 0 (C: 19.7 GiB free)')
    log.add('prefetch secondary: start (C: 19.7 GiB free)')
    log.add('[prefetch] secondary: 46 pieces, 4914 chunks (9.60 GiB), 0 cached, 4914 to fetch (9.60 GiB), 2 '
            'connections')
    log.add('[prefetch] secondary: 512/4914 chunks, 1024 MiB, 1496 s, 0.68 MiB/s, C: 10.3 GiB free', 1496)
    log.add('[prefetch] STOP: C: has 9.98 GiB free (< 10); 512/4914 fetched', 0)
    log.add('prefetch secondary: rc 3 (C: 9.98 GiB free)')
    log.add('ERROR (rc 3): C: below 10 GiB free during prefetch secondary', 0)
    log.add('DONE rc 3', 0)


def attempt(log, pid, outcomes, fetch=0, crash_before=None, score_rc=None, stop_in=None):
    """One run_all.ps1 attempt, logged as run_all.ps1 and prefetch_v3.py write it. outcomes[(set, piece)]: a list of
    (rc, seconds) per infer.py attempt, or 'skip' (done in an earlier attempt). crash_before: (set, piece) where the
    attempt fails with rc 10; stop_in: a set whose inference is still running (no end). Returns the status.json."""
    started = log.t.strftime('%Y-%m-%d %H:%M:%S')
    log.add(f'START v3 runner pid {pid} (BelowNormal), CUDA_VISIBLE_DEVICES=-1, HF_HUB_OFFLINE=1, C: 31.2 GiB free', 0)
    log.add('villa worktree C:\\VesuviusTemp\\strips\\villa_6f2 at 6f2024569404a46b0f008eee292773945c07a684 (+ '
            '#2163 + normalization patch, as G0); checkpoint '
            'scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e', 0)
    log.add('verify FREEZE.txt: rc 0; FREEZE.txt v3.2: 100 files as frozen ', 21)
    log.add('[prefetch] 0139: cache metadata present (C:\\VesuviusTemp\\strips\\ct_cache\\s3\\...)', 3)
    log.add('preflight 0139 rc 0 {"preflight": "ok", "seconds": 40.2} ', 60)
    log.add('preflight 0332 rc 0 {"preflight": "ok", "seconds": 38.0} ', 50)
    for s, (npieces, chunks) in CHUNKS.items():
        todo = fetch if s == 'secondary' else 0
        log.add(f'prefetch {s}: start (C: 31.2 GiB free)')
        log.add(f'[prefetch] {s}: {npieces} pieces, {chunks} chunks ({chunks*2/1024:.2f} GiB), {chunks-todo} cached, '
                f'{todo} to fetch ({todo*2/1024:.2f} GiB), 2 connections', 3)
        if todo:
            log.add(f'[prefetch] {s}: {todo}/{todo} chunks, {2*todo} MiB, 9 s, 0.67 MiB/s, C: 31.0 GiB free', 9)
        log.add(f'[prefetch] {s}: done: {todo} fetched (0 missing objects -> fill), {2*todo} MiB in 9 s; 0 of '
                f'{chunks} chunks not cached', 0)
        log.add(f'prefetch {s}: rc 0 (C: 31.0 GiB free)')
    status = dict(stage='start', pid=pid, started=started, rc=None, pieces={}, score={})
    failed = 0
    for s, plist in PIECES.items():
        st = dict(planned=len(plist), decisions=sum(d for _, d in plist), ok=0, skipped=0, failed=0,
                  failed_pieces=[])
        status['pieces'][s] = st
        status['stage'] = f'infer:{s}'
        log.add(f"infer {s}: {len(plist)} pieces, {st['decisions']} decisions")
        for piece, dec in plist:
            if crash_before == (s, piece):
                log.add('ERROR (rc 10): Exception calling "WaitForExit" with "0" argument(s)', 5)
                log.add('DONE rc 10', 0)
                status.update(stage=f'failed: infer:{s}', rc=10, error='WaitForExit')
                return status
            plan = outcomes[(s, piece)]
            if plan == 'skip':
                st['skipped'] += 1
                log.add(f'infer {s} {piece}: done already, skipped', 0)
                continue
            for a, (rc, sec) in enumerate(plan, 1):
                log.add(f'infer {s} {piece} (attempt {a}): rc {rc}, {dec} decisions, {sec} s, '
                        f'{round(dec/sec, 3)} decisions/s incl. start-up', sec)
            if plan[-1][0] == 0:
                st['ok'] += 1
            else:
                st['failed'] += 1
                st['failed_pieces'].append(piece)
                failed += 1
            if stop_in == s:
                return status
        log.add(f"infer {s}: ok {st['ok']}, skipped {st['skipped']}, failed {st['failed']}", 0)
    status['stage'] = 'score'
    for k, sec in (('primary', 12.3), ('secondary', 80.1), ('0332', 9.9)):
        rc = (score_rc or {}).get(k, 0)
        log.add(f'score {k}: rc {rc} in {sec} s (report in results\\v3; not read by the runner)', sec)
        status['score'][k] = dict(rc=rc, seconds=sec)
    rc = 1 if failed or any((score_rc or {}).values()) else 0
    log.add(f'DONE rc {rc}', 0)
    status.update(stage='finished', rc=rc)
    return status


def write_ps(path, text):
    """As PowerShell 5.1 writes it with -Encoding UTF8: a BOM, CRLF line ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(('\ufeff'+text.replace('\n', '\r\n')).encode('utf-8'))


def write_run(ws, scenario='normal', score_rc=None, offset='+0000'):
    write_ps(ws/'FREEZE.txt', f'Freeze of pre-registration v3.2 (synthetic, for tests)\n'
                              f'frozen_at_local: {FREEZE_LOCAL} {offset}\nfrozen_at_utc:   (synthetic)\nsha256:\n'
                              f"  {sha256(REPO/'score'/'PREREG.md')} *PREREG.md\n"
                              f"  {sha256(REPO/'score'/'score_strips.py')} *score_strips.py\n")
    log = Log('2026-10-28 14:04:33')
    pre_freeze_attempt(log)
    log.t = datetime(2026, 10, 28, 20, 5, 0)
    if scenario == 'normal':
        status = attempt(log, 5120, NORMAL, fetch=3, score_rc=score_rc)
    elif scenario == 'resumed':  # an attempt that fails with rc 10, then a relaunch that resumes
        attempt(log, 5120, NORMAL, crash_before=('secondary', 'P0005'))
        log.t = datetime(2026, 10, 28, 21, 0, 0)
        outcomes = dict(NORMAL)
        outcomes.update({('primary', 'P0000'): 'skip', ('primary', 'P0001'): 'skip', ('secondary', 'P0003'): 'skip',
                         ('secondary', 'P0006'): [(1, 40.), (1, 41.)]})
        status = attempt(log, 6200, outcomes)
    else:  # 'unfinished': still in the secondary set's inference
        status = attempt(log, 5120, NORMAL, stop_in='secondary')
    status.update(updated=log.stamp(), prefetch=dict(set='0332', fetched=0, to_fetch=0, chunks=1009, mib=0.0))
    write_ps(ws/'runs'/'v3'/'log.txt', '\n'.join(log.lines)+'\n')
    write_ps(ws/'runs'/'v3'/'status.json', json.dumps(status, indent=4))
    if status['stage'] in ('finished',) or status['stage'].startswith('failed'):
        write_ps(ws/'runs'/'v3'/'DONE', f"rc={status['rc']} stage={status['stage']} finished={log.stamp()}\n")
        log.add('', 1)
        write_ps(ws/'runs'/'v3'/'V3_FINAL', f"rc={status['rc']} finished={log.stamp()}\n")
    return status


def workspace(tmp, built, scenario='normal', primary='primary', secondary='secondary', x0332='x0332', drop=(),
              score_rc=None, offset='+0000'):
    ws = tmp/'ws'
    (ws/'results'/'v3').mkdir(parents=True)
    for key, src in (('primary', primary), ('secondary', secondary), ('0332', x0332)):
        if key not in drop:
            shutil.copy(built[src], ws/F.REPORTS[key])
    write_run(ws, scenario, score_rc, offset)
    return ws


# The README as committed in a2f1b56 (29 Oct 2026, 02:29 UTC), before the results: the placeholders that
# fill_results.py filled. The repo README itself has been filled since, so the tests read this copy.
TEMPLATE = REPO/'tools'/'fixtures'/'README_before_results.md'


def readme_copy(tmp, deviations='none since the v3.2 freeze'):
    """The pre-result README (read only), with RESULT_DEVIATIONS replaced by hand as the author did, unless None."""
    text = TEMPLATE.read_bytes().decode('utf-8')
    if deviations is not None:
        text = text.replace('{{RESULT_DEVIATIONS}}', deviations)
    path = tmp/'README.md'
    path.write_bytes(text.encode('utf-8'))
    return path


def values(ws):
    c = F.Ctx(ws)
    return {name: F.resolve(c, name)[::2] for name in F.BY_NAME}  # (value, state)


def f2(x):
    return f'{x:.2f}'.replace('-', MINUS)


def s2(x):
    text = f'{x:.2f}'
    return MINUS+text[1:] if text.startswith('-') else '+'+text


def p0(x):
    return f'{x*100:.0f} %'


# ---- tests ---------------------------------------------------------------------------------------------------------

def test_every_readme_placeholder_has_one_source_entry():
    assert sha256(TEMPLATE) == '664c3690d3fb3afa1e148e09bf26827b544eadc86cdb8314471180fdfb735c27'  # = git show a2f1b56:README.md
    names = F.PLACEHOLDER.findall(TEMPLATE.read_bytes().decode('utf-8'))
    assert len(names) == 86 and len(set(names)) == 86
    assert set(names) == set(F.BY_NAME) and len(F.FIELDS) == 86
    assert all(source and callable(fn) for _, source, fn in F.FIELDS)


def test_reads_the_real_freeze_record(tmp_path):
    shutil.copy(REPO/'score'/'FREEZE.txt', tmp_path/'FREEZE.txt')  # the repo's copy of the workspace's FREEZE.txt
    fz = F.Ctx(tmp_path).freeze()
    assert fz['local'] == '2026-10-29 02:10:33' and fz['offset'] == timedelta(0)
    assert fz['pins']['score_strips.py'] == sha256(REPO/'score'/'score_strips.py')
    assert fz['pins']['PREREG.md'] == sha256(REPO/'score'/'PREREG.md')


def test_formats():
    assert F.num(1.234) == '1.23' and F.num(-.004) == '0.00' and F.num(-1.5) == MINUS+'1.50'
    assert F.signed(.47) == '+0.47' and F.signed(-.12) == MINUS+'0.12' and F.signed(.001) == '0.00'
    assert F.pct(.8249) == '82 %' and F.pct(1.) == '100 %'
    assert F.pct_points(.45) == '+45 %' and F.pct_points(-.031) == MINUS+'3 %' and F.pct_points(.001) == '0 %'
    assert F.count(12345) == '12,345' and F.ci([1.234, 2.346]) == '1.23, 2.35'
    assert F.ci([-.5, .25], F.signed) == MINUS+'0.50, +0.25'
    assert F.utc('2026-10-29 03:15:10', timedelta(hours=1)) == '29 Oct 2026 at 02:15:10 UTC'
    assert F.md('a | b\n c') == 'a \\| b c'
    with pytest.raises(F.NotAvailable):
        F.num(None)


def test_normal_case_fills_every_placeholder(tmp_path, built, capsys):
    ws = workspace(tmp_path, built)
    readme = readme_copy(tmp_path)
    assert F.main(['--workspace', str(ws), '--readme', str(readme)]) == 0
    assert 'wrote' in capsys.readouterr().out
    text = readme.read_bytes().decode('utf-8')
    assert '{{RESULT_' not in text.replace('{{RESULT_...}}', '')
    v = values(ws)
    assert {n for n, (_, state) in v.items() if state != 'ok'} == {'DEVIATIONS', 'P_S0001_FRAME'}
    assert v['DEVIATIONS'][1] == 'unresolved'  # the author's: replaced by hand above
    # run record: the run starts at its first start after the freeze; the 28 Oct attempt's 1,304 MiB are not counted
    assert ('Started 28 Oct 2026 at 20:05:00 UTC, finished 28 Oct 2026 at '
            f"{(F.Ctx(ws).final()['finished'])[11:]} UTC, exit code 0. Pieces\nok / failed: primary 2 / 0, "
            'secondary\n3 / 0, PHerc0332 2 /\n0. Decisions 621 of 4,129 planned. CPU time\n0.20 h; CT fetched 0.01 GiB.'
            ) in text
    # secondary, from the scorer's report
    rep = json.loads(built['secondary'].read_text())
    for fam, label, fibers in (('all', 'All', 32), ('H', 'H', 12), ('V', 'V', 20)):
        a = rep['accuracy'][fam]
        p, lo_hi = a['strip']['point'], a['strip']['ci']['mean']
        row = (f"| {label} | {fibers} | {p['nodes']:,} | {f2(p['mean'])} [{f2(lo_hi[0])}, {f2(lo_hi[1])}] | "
               f"{f2(p['median'])} | {f2(p['p90'])} | {p0(p['within1'])} | {p0(p['within2'])} | "
               f"{s2(a['strip_signed_mean_sv'])} |")
        assert row in text
    assert rep['G2']['verdict'] == 'secondary: transfers'
    assert '- **Verdict:** secondary: transfers.' in text
    assert "**G1':** 32 of 36 fibers scored with at least 25 covered nodes, so G1'\npasses." in text
    for base, label in (('frame_plane', '(a) Frame plane'), ('lasagna_plane', '(b) Lasagna-normal plane')):
        b = rep['accuracy']['all'][base]
        d = b['delta']
        assert (f"| {label} | {f2(b['stats']['point']['mean'])} | {s2(d['baseline_minus_strip_mean_sv'])} "
                f"[{s2(d['ci'][0])}, {s2(d['ci'][1])}] | +{d['strip_minus_baseline_within2']*100:.0f} % "
                f"(strip {MINUS} baseline) | yes |") in text
    cal = rep['calibration']
    assert (f"weighted gap {f2(cal['weighted_abs_gap_sv'])} sv; observed / predicted\n"
            f"  {f2(cal['observed_over_expected'])}; Spearman rho {f2(cal['spearman_node'])} (nodes), "
            f"{f2(cal['spearman_fiber'])} (fibers).") in text
    floor = rep['accuracy']['all']['fiber_floor']
    assert (f"scored rows): mean {f2(floor['mean'])} sv, median {f2(floor['median'])} sv over {floor['nodes']:,} "
            'rows.') in text
    unflagged = rep['sensitivity']['unflagged stretches (no fold-back or jump)']
    lo, hi = unflagged['strip']['ci']['mean']
    assert v['SEC_SENS_UNFLAGGED'] == (f"mean \\|d\\| {f2(unflagged['strip']['point']['mean'])} sv [{f2(lo)}, "
                                       f"{f2(hi)}], {p0(unflagged['strip']['point']['within2'])} within 2 sv (32 "
                                       'fibers)', 'ok')
    annotators = v['SEC_SENS_ANNOTATOR'][0].split('; ')
    assert [a.split(':')[0] for a in annotators] == ['dj', 'kb', 'lt']
    assert [a.endswith(f'({n} fibers)') for a, n in zip(annotators, (11, 11, 10))] == [True]*3
    assert v['SEC_REJECTED'] == ('none', 'ok')
    # primary case studies: the per-annotator aggregate is the stretch alone
    prim = json.loads(built['primary'].read_text())
    kb = prim['sensitivity']['annotator kb']
    strip0 = next(e for e in prim['per_strip'] if e['stretches'] == ['S0000'])
    assert kb['strip']['point']['nodes'] == strip0['strip_nodes']
    assert abs(kb['strip']['point']['mean']-strip0['strip_mean_abs_sv']) < 1e-9
    assert (f"| {strip0['strip_nodes']:,} | {f2(kb['strip']['point']['mean'])} | {p0(kb['strip']['point']['within2'])} "
            f"| {s2(kb['strip_signed_mean_sv'])} | {f2(kb['frame_plane']['stats']['point']['mean'])} | "
            f"{f2(kb['lasagna_plane']['stats']['point']['mean'])} | {f2(kb['fiber_floor']['mean'])} |") in text
    assert abs(kb['strip']['point']['mean']-.8) < .05 and kb['strip_signed_mean_sv'] > 0
    assert abs(float(v['P_S0001_BIAS'][0].replace(MINUS, '-'))+.4) < .03  # the H strip lies 0.4 sv inward
    assert v['P_S0001_FRAME'] == ('not available (no decisions (strip.npz) beside the strip: frame-plane baseline '
                                  'unavailable)', 'na')
    # PHerc0332: 15 planned, 10 with both strips, 8 of them tight; the 5 crossings of v2 are missing
    x = json.loads(built['x0332'].read_text())['crossings']
    assert x['planned'] == 15 and x['crossings'] == 10 and x['tight_subset']['crossings'] == 8
    assert (f"| All 15 planned | 10 | {f2(x['mean'])} [{f2(x['ci']['mean'][0])}, {f2(x['ci']['mean'][1])}] | "
            f"{f2(x['median'])} | {f2(x['p90'])} | 80 % |") in text
    assert (f"| Tight subset (3, ≤ 3 sv) | 8 | {f2(x['tight_subset']['mean'])} | "
            f"{f2(x['tight_subset']['median'])} | | |") in text
    assert 'Missing crossings: 5 (X0002, X0005, X0008, X0011, X0014).' in text
    cr = prim['crossings']
    assert cr['planned'] == 1 and cr['crossings'] == 1
    assert ('H/V crossings found among the 0139 strips: 1 (secondary: no H/V fiber pair within 3 sv; primary: 1 of 1 '
            f"H/V fiber pairs within 3 sv with both strips, agreement mean {f2(cr['mean'])} sv, median "
            f"{f2(cr['median'])} sv).") in text
    assert '| after v3.2 | | none since the v3.2 freeze (PREREG 10.6: none at freeze time) |' in text


def test_dry_run_prints_every_placeholder_and_writes_nothing(tmp_path, built, capsys):
    ws = workspace(tmp_path, built)
    readme = readme_copy(tmp_path, deviations=None)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert F.main(['--workspace', str(ws), '--readme', str(readme), '--dry-run']) == 1  # DEVIATIONS: no source
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith('{{RESULT_')]
    assert len(lines) == 86 and all(' -> ' in ln and ln.endswith(')') for ln in lines)
    assert lines[0].startswith('{{RESULT_DEVIATIONS}} -> UNRESOLVED: no source: deviations after the v3.2 freeze')
    assert '{{RESULT_RUN_RC}} -> 0 (runs/v3/V3_FINAL: rc=' in out
    assert '86 placeholders' in out and '84 filled, 1 not available, 1 unresolved' in out and 'nothing written' in out
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


def test_placeholder_without_source_leaves_the_readme_untouched(tmp_path, built, capsys):
    ws = workspace(tmp_path, built)
    readme = readme_copy(tmp_path, deviations=None)
    readme.write_bytes(readme.read_bytes()+b'\nOne more: {{RESULT_NOT_A_FIELD}}.\n')
    before = readme.read_bytes()
    assert F.main(['--workspace', str(ws), '--readme', str(readme)]) == 1
    out = capsys.readouterr().out
    assert readme.read_bytes() == before and not list(tmp_path.glob('*.fill_tmp'))
    assert 'not written; unresolved: RESULT_DEVIATIONS, RESULT_NOT_A_FIELD' in out
    assert '{{RESULT_NOT_A_FIELD}} -> UNRESOLVED: no source defined in tools/fill_results.py (none)' in out


def test_missing_report_gives_explicit_text(tmp_path, built, capsys):
    ws = workspace(tmp_path, built, drop=('secondary',), score_rc={'secondary': 1})
    readme = readme_copy(tmp_path)
    assert F.main(['--workspace', str(ws), '--readme', str(readme)]) == 0
    text = readme.read_bytes().decode('utf-8')
    reason = 'not available (results/v3/secondary_report.json absent; score secondary rc 1)'
    assert f'- **Verdict:** secondary: {reason}.' in text
    v = values(ws)
    for name in F.BY_NAME:
        if name.startswith('SEC_') and 'PIECES' not in name:
            assert v[name] == (reason, 'na'), name
    assert v['0139_CROSSINGS'][0].startswith('secondary: not available (results/v3/secondary_report.json absent; '
                                             'score secondary rc 1); primary: 1 of 1 H/V fiber pairs')
    assert v['RUN_RC'] == ("1 (some pieces or a score failed; last stage 'finished')", 'ok')
    assert v['P_S0000_MEAN'][1] == 'ok' and v['0332_MEAN'][1] == 'ok'
    # a report the scorer could not fill: no crossing has both strips (the scorer's own crossing code, no strip)
    shutil.copy(built['x0332_none'], ws/F.REPORTS['0332'])
    v = values(ws)
    assert v['0332_N'] == ('0', 'ok') and v['0332_TIGHT_N'] == ('0', 'ok')
    assert v['0332_MEAN'] == ('not available (no H/V crossings with overlapping strips)', 'na')
    assert v['0332_MISSING'][0].startswith('15 (X0000, X0001, ')
    (ws/F.REPORTS['0332']).unlink()
    assert values(ws)['0332_TIGHT_MEDIAN'] == ('not available (results/v3/crossings_0332_report.json absent; score '
                                               '0332 rc 0)', 'na')


def test_unavailable_baseline_family_without_fiber_and_rejected_strip(tmp_path, built):
    ws = workspace(tmp_path, built, secondary='degraded')
    rep = json.loads(built['degraded'].read_text())
    assert rep['G2']['verdict'] == 'secondary: degrades' and not rep['G2']['checks']['beats_lasagna_plane']
    assert rep['accuracy']['all']['scored_fibers'] == 30 and rep['accuracy']['H']['scored_fibers'] == 0
    v = values(ws)
    assert v['SEC_VERDICT'] == ('degrades', 'ok') and v['SEC_G1P'] == ('passes', 'ok')
    gap = 'no Lasagna normals for any of the 34 strips'
    for name in ('SEC_LASAGNA_MEAN', 'SEC_LASAGNA_DIFF', 'SEC_LASAGNA_DIFF_CI', 'SEC_LASAGNA_W2_DIFF'):
        assert v[name] == (f'not available ({gap})', 'na'), name
    assert v['SEC_LASAGNA_BEATEN'] == (f'no (not computed: {gap}; counts as not beaten)', 'ok')
    assert v['SEC_FRAME_BEATEN'] == ('yes', 'ok')
    assert v['SEC_H_FIBERS'] == ('0', 'ok')
    for what in ('NODES', 'MEAN', 'MEAN_CI', 'MEDIAN', 'P90', 'W1', 'W2', 'BIAS'):
        assert v[f'SEC_H_{what}'] == ('not available (no scored H fiber)', 'na')
    assert v['SEC_CAL_GAP'] == ('not available (no fiber with 25 scored nodes that carry expected_error)', 'na')
    assert v['SEC_REJECTED'] == ('1 (P0090); other skips: 1 (P0091: no strip row inside a matched stretch)', 'ok')


def test_sets_without_scored_fiber(tmp_path, built):
    ws = workspace(tmp_path, built, primary='primary_empty', secondary='empty')
    v = values(ws)
    assert v['SEC_FIBERS_SCORED'] == ('0', 'ok') and v['SEC_G1P'] == ('fails', 'ok')
    assert v['SEC_VERDICT'] == ('no accuracy claim (G1 failed or fewer than 30 scored fibers)', 'ok')
    for name in ('SEC_ALL_NODES', 'SEC_ALL_MEAN', 'SEC_ALL_MEAN_CI', 'SEC_ALL_BIAS', 'SEC_FIBER_FLOOR',
                 'SEC_SENS_UNFLAGGED', 'SEC_FRAME_MEAN', 'SEC_FRAME_DIFF', 'SEC_LASAGNA_DIFF_CI'):
        assert v[name] == ('not available (no scored fiber)', 'na'), name
    assert v['SEC_FRAME_BEATEN'] == ('no (not computed: no scored fiber; counts as not beaten)', 'ok')
    assert v['SEC_SENS_ANNOTATOR'][0] == ('dj: not available (no scored fiber); kb: not available (no scored '
                                          'fiber); lt: not available (no scored fiber)')
    assert v['P_S0000_NODES'] == ('not available (no scored strip for S0000)', 'na')
    assert v['0139_CROSSINGS'][0] == ('0 (secondary: no H/V fiber pair within 3 sv; primary: 0 of 1 H/V fiber pairs '
                                      'within 3 sv with both strips)')


def test_unfinished_run_is_not_written(tmp_path, built, capsys):
    ws = workspace(tmp_path, built, scenario='unfinished', drop=('primary', 'secondary', '0332'))
    readme = readme_copy(tmp_path)
    before = readme.read_bytes()
    assert F.main(['--workspace', str(ws), '--readme', str(readme)]) == 1
    out = capsys.readouterr().out
    assert readme.read_bytes() == before
    assert 'not written; unresolved: RESULT_RUN_END, RESULT_RUN_RC' in out
    assert '{{RESULT_RUN_RC}} -> UNRESOLVED: the run has not finished: runs/v3/V3_FINAL absent' in out
    v = values(ws)
    assert v['SEC_ALL_MEAN'] == ("not available (results/v3/secondary_report.json absent; the run's last stage was "
                                 "'infer:secondary')", 'na')
    assert v['0332_PIECES_OK'] == ("not available (the run did not reach inference of the 0332 set (last stage "
                                   "'infer:secondary'))", 'na')


def test_resumed_run_with_a_failed_piece(tmp_path, built):
    ws = workspace(tmp_path, built, scenario='resumed')
    v = values(ws)
    assert v['RUN_START'] == ('28 Oct 2026 at 20:05:00 UTC', 'ok')  # the first start after the freeze, not 21:00
    assert v['RUN_RC'] == ("1 (some pieces or a score failed; last stage 'finished')", 'ok')
    assert v['PRIMARY_PIECES_OK'] == ('2', 'ok') and v['PRIMARY_PIECES_FAILED'] == ('0', 'ok')
    assert v['SEC_PIECES_OK'] == ('2', 'ok') and v['SEC_PIECES_FAILED'] == ('1 (P0006)', 'ok')
    assert v['DECISIONS_TOTAL'] == ('203', 'ok')  # P0006 (418) failed twice
    seconds = 61.3+58.9+150.2+30.5+31.+40.+41.+45.2+44.8  # every attempt of both campaigns, failed ones too
    assert v['CPU_HOURS'] == (f'{seconds/3600:.2f}', 'ok') and v['CT_GIB'] == ('0.00', 'ok')


def test_report_not_from_the_frozen_scorer_or_older_than_the_run(tmp_path, built):
    ws = workspace(tmp_path, built, offset='+0100')
    path = ws/F.REPORTS['0332']
    rep = json.loads(path.read_text())
    rep['provenance']['scorer_sha256'] = '0'*64
    path.write_text(json.dumps(rep))
    path = ws/F.REPORTS['primary']
    rep = json.loads(path.read_text())
    rep['provenance']['time'] = '2026-10-28 20:04:59'
    path.write_text(json.dumps(rep))
    v = values(ws)
    assert v['0332_N'][0].startswith('not available (results/v3/crossings_0332_report.json was not written by the '
                                     'frozen scorer (scorer_sha256 000000000000, frozen score_strips.py d7d6d1946bc4')
    assert v['P_S0000_MEAN'] == ('not available (results/v3/primary_case_studies.json was written at 2026-10-28 '
                                 '20:04:59, before the run started (2026-10-28 20:05:00))', 'na')
    assert v['SEC_ALL_MEAN'][1] == 'ok'
    assert v['RUN_START'] == ('28 Oct 2026 at 19:05:00 UTC', 'ok')  # local time at +01:00 (FREEZE.txt)
