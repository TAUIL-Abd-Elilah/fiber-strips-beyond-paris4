#!/usr/bin/env python
"""Fill the {{RESULT_<NAME>}} placeholders of README.md from the frozen scorer's reports and the runner's records.

    python tools/fill_results.py --workspace C:/VesuviusTemp/strips/score [--readme README.md] [--dry-run]

Every placeholder maps to one field of one source, or to a fixed formatting of a few fields (FIELDS below; each
placeholder's source is written beside it, and --dry-run prints "placeholder -> value (source)" for all of them and
writes nothing). The scorer's verdict and gate results are used as the scorer wrote them; no gate is recomputed.

Sources, all under --workspace (the author's score/ folder):
  results/v3/primary_case_studies.json    score_strips.py score on the primary set (runs/v3/run_all.ps1)
  results/v3/secondary_report.json        score_strips.py score on the secondary run subsample (--run-only)
  results/v3/crossings_0332_report.json   score_strips.py crossings on the PHerc0332 plan
  runs/v3/log.txt, status.json, DONE      runs/v3/run_all.ps1 (log.txt also holds prefetch_v3.py's lines)
  runs/v3/V3_FINAL                        runs/relaunch_v3.ps1: the run's final status
  FREEZE.txt                              the v3.2 freeze: its local time and UTC offset, the frozen sha256

The run is the runner's work from its first start at or after the freeze (log.txt). Log times are the runner
machine's local time; they are given in UTC with the offset FREEZE.txt records (frozen_at_local). A report is used
only if the frozen scorer wrote it (its provenance sha256 of score_strips.py and PREREG.md equal FREEZE.txt's), after
the run started, and status.json does not record its scoring step as failed. Missing data gives "not available
(reason)". README.md is written in place only when every placeholder in it resolves. Unresolved are: a placeholder
without a source (RESULT_DEVIATIONS: deviations after the freeze are the author's statement, PREREG 10.6; replace it
by hand), RESULT_RUN_END and RESULT_RUN_RC while runs/v3/V3_FINAL is absent (the run has not finished), and any value
of a report whose layout is not the scorer's. Then nothing is written and the exit code is 1. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r'\{\{RESULT_([A-Z0-9_]+)\}\}')
STAMP = r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d'  # Get-Date -Format 'yyyy-MM-dd HH:mm:ss' and time.strftime of the scorer
MINUS = '\u2212'  # negative numbers as the PREREG writes them ("from \u22126 to 6 sv")
MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

REPORTS = {'primary': 'results/v3/primary_case_studies.json',  # run_all.ps1, score stage ($jobs)
           'secondary': 'results/v3/secondary_report.json',
           '0332': 'results/v3/crossings_0332_report.json'}
ANALYSIS = {'primary': 'primary', 'secondary': 'secondary'}  # report['analysis'] (the crossings report has none)
LOG, STATUS, DONE, FINAL, FREEZE = ('runs/v3/log.txt', 'runs/v3/status.json', 'runs/v3/DONE', 'runs/v3/V3_FINAL',
                                    'FREEZE.txt')
RC_MEANING = {0: 'all done', 1: 'some pieces or a score failed', 2: 'CUDA visible', 3: 'low disk',
              4: 'prefetch incomplete', 5: 'FREEZE mismatch', 10: 'other error'}  # run_all.ps1, header
# Fixed by the README's own text (Results; PREREG 4.3 and 7): the case-study rows and the 0332 row label.
PRIMARY_FIBER = {'S0000': 'kb_20260904T021448778_000085', 'S0001': 'lt_20260918T092404850_000160'}
PLANNED_0332 = 15
UNFLAGGED = 'unflagged stretches (no fold-back or jump)'  # score_strips.cmd_score's sensitivity key
BASE_NAME = {'frame_plane': 'frame plane', 'lasagna_plane': 'Lasagna plane'}
FAMILY_NAME = {'all': 'fiber', 'H': 'H fiber', 'V': 'V fiber'}

INFER_RUN = re.compile(r'^infer (\S+) (\S+) \(attempt (\d+)\): rc (-?\d+), (\d+) decisions, (\d+(?:[.,]\d+)?) s,')
INFER_SKIP = re.compile(r'^infer (\S+) (\S+): done already, skipped$')
PREFETCH_START = re.compile(r'^\[prefetch\] (\S+): \d+ pieces, \d+ chunks ')
PREFETCH_MIB = re.compile(r'^\[prefetch\] (\S+): (?:done: \d+ fetched .*?, |\d+/\d+ chunks, )(\d+) MiB')


class NotAvailable(Exception):
    """Missing data: the placeholder becomes 'not available (reason)'."""


class Unresolved(Exception):
    """No value can be given: no source, the run has not finished, or a report is not laid out as the scorer's."""


# ---- formatting (README style: 2 decimals for sv, 'lo, hi' inside the README's brackets, whole percent 'NN %',
#      thousands with commas, signed values with + and U+2212, times '29 Oct 2026 at 02:10:33 UTC') -----------------

def num(x, nd=2):
    if x is None:
        raise NotAvailable('no value in the report')
    text = f'{float(x):.{nd}f}'
    if float(text) == 0:
        text = text.lstrip('-')
    return text.replace('-', MINUS)


def signed(x, nd=2):
    text = num(x, nd)
    return text if text.startswith(MINUS) or float(text) == 0 else '+' + text


def pct(x):
    if x is None:
        raise NotAvailable('no value in the report')
    text = f'{float(x) * 100:.0f}'
    return (text.lstrip('-') if float(text) == 0 else text.replace('-', MINUS)) + ' %'


def pct_points(x):
    """A difference of two fractions, in percentage points, signed."""
    if x is None:
        raise NotAvailable('no value in the report')
    text = f'{float(x) * 100:.0f}'
    if float(text) == 0:
        return '0 %'
    return (MINUS + text[1:] if text.startswith('-') else '+' + text) + ' %'


def count(n):
    return f'{int(n):,}'


def ci(pair, fmt=num):
    if not pair or len(pair) != 2:
        raise NotAvailable('no CI in the report')
    return f'{fmt(pair[0])}, {fmt(pair[1])}'


def utc(stamp, offset):
    t = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S') - offset
    return f'{t.day} {MONTHS[t.month - 1]} {t.year} at {t:%H:%M:%S} UTC'


def plural(n, word):
    return f'{n} {word}' + ('' if n == 1 else 's')


def md(value):
    """One line, with '|' escaped as the README's tables need it."""
    return ' '.join(str(value).split()).replace('|', '\\|')


def need(d, *keys):
    """d[k0][k1]...; a missing key is a report layout this script does not know (Unresolved), not missing data."""
    cur = d
    for k in keys:
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            raise Unresolved(f"report field {'.'.join(map(str, keys))} missing") from None
    return cur


def min_nodes(rep):
    return int(((rep.get('provenance') or {}).get('constants') or {}).get('MIN_NODES', 25))


# ---- the sources ----------------------------------------------------------------------------------------------

class Ctx:
    """The workspace's sources, each read once; a lookup returns a value or raises NotAvailable / Unresolved."""

    def __init__(self, workspace):
        self.ws = Path(workspace)
        self._memo = {}

    def _once(self, key, load):
        if key not in self._memo:
            try:
                self._memo[key] = (None, load())
            except (NotAvailable, Unresolved) as error:
                self._memo[key] = (error, None)
        error, value = self._memo[key]
        if error is not None:
            raise type(error)(*error.args)
        return value

    def text(self, rel):
        path = self.ws/rel
        if not path.is_file():
            raise NotAvailable(f'{rel} absent')
        try:  # PowerShell 5.1 writes UTF-8 with a BOM
            return path.read_bytes().decode('utf-8', errors='replace').replace('\ufeff', '')
        except OSError as error:
            raise NotAvailable(f'{rel} unreadable ({error})') from None

    def freeze(self):
        """FREEZE.txt: the freeze's local time and UTC offset (frozen_at_local) and the sha256 pins."""
        def load():
            text = self.text(FREEZE)
            m = re.search(rf'^frozen_at_local:\s*({STAMP})\s*([+-])(\d\d):?(\d\d)', text, re.M)
            if not m:
                raise NotAvailable('FREEZE.txt has no frozen_at_local line')
            sign = 1 if m[2] == '+' else -1
            pins = {path: sha for sha, path in re.findall(r'^\s*([0-9a-f]{64}) \*(.+?)\s*$', text, re.M)}
            return dict(local=m[1], offset=sign*timedelta(hours=int(m[3]), minutes=int(m[4])), pins=pins)
        return self._once('freeze', load)

    def log(self):
        """runs/v3/log.txt as (local time, message); lines without a time stamp (continuations) are dropped."""
        def load():
            out = []
            for line in self.text(LOG).splitlines():
                m = re.match(rf'^\[({STAMP})\] (.*)$', line.strip())
                if m:
                    out.append((m[1], m[2]))
            return out
        return self._once('log', load)

    def start(self):
        """The run's start: the first 'START v3 runner' line of log.txt at or after the freeze (both local times of
        the same machine). Earlier starts are the v3 runner's attempts of 28 Oct (PREREG 10.7 item 7)."""
        def load():
            frozen = self.freeze()['local']
            for stamp, msg in self.log():
                if stamp >= frozen and msg.startswith('START v3 runner'):
                    return stamp
            raise NotAvailable(f'no runner start in {LOG} at or after the freeze ({frozen})')
        return self._once('start', load)

    def run_log(self):
        start = self.start()
        return [(stamp, msg) for stamp, msg in self.log() if stamp >= start]

    def final(self):
        """runs/v3/V3_FINAL, 'rc=<rc> finished=<local time>': written once by relaunch_v3.ps1 when no retry follows."""
        def load():
            if not (self.ws/FINAL).is_file():
                raise Unresolved(f'the run has not finished: {FINAL} absent')
            text = self.text(FINAL)
            m = re.search(rf'rc=(-?\d+) finished=({STAMP})', text)
            if not m:
                raise Unresolved(f'{FINAL} unreadable: {text.strip()[:80]!r}')
            return dict(rc=int(m[1]), finished=m[2])
        return self._once('final', load)

    def done(self):
        """runs/v3/DONE, 'rc=<rc> stage=<stage> finished=<local time>' (run_all.ps1), or None."""
        def load():
            if not (self.ws/DONE).is_file():
                return None
            m = re.search(rf'rc=(-?\d+) stage=(.*?) finished=({STAMP})', self.text(DONE))
            return dict(rc=int(m[1]), stage=m[2], finished=m[3]) if m else None
        return self._once('done', load)

    def status(self):
        """runs/v3/status.json: the run's last attempt (run_all.ps1 Save-Status)."""
        def load():
            try:
                return json.loads(self.text(STATUS))
            except ValueError as error:
                raise NotAvailable(f'{STATUS} unreadable ({error})') from None
        return self._once('status', load)

    def score_state(self, key):
        """(rc of the scoring step of a report, the run's last stage) from status.json; None where unknown."""
        try:
            st = self.status()
        except NotAvailable:
            return None, None
        rc = ((st.get('score') or {}).get(key) or {}).get('rc')
        return rc, st.get('stage')

    def report(self, key):
        return self._once(('report', key), lambda: self._load_report(key))

    def _load_report(self, key):
        rel = REPORTS[key]
        path = self.ws/rel
        rc, stage = self.score_state(key)
        if not path.is_file():
            why = f'{rel} absent'
            if rc is not None:
                why += f'; score {key} rc {rc}'
            elif stage:
                why += f"; the run's last stage was '{stage}'"
            raise NotAvailable(why)
        if rc not in (None, 0):
            raise NotAvailable(f'score {key} exited with rc {rc} ({STATUS}); its report is not used')
        try:
            rep = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise NotAvailable(f'{rel} unreadable ({type(error).__name__})') from None
        prov = rep.get('provenance') or {}
        pins = self.freeze()['pins']
        for field, frozen in (('scorer_sha256', 'score_strips.py'), ('prereg_sha256', 'PREREG.md')):
            if frozen not in pins:
                raise NotAvailable(f'{FREEZE} has no sha256 of {frozen}')
            if prov.get(field) != pins[frozen]:
                raise NotAvailable(f'{rel} was not written by the frozen scorer ({field} '
                                   f'{str(prov.get(field))[:12]}, frozen {frozen} {pins[frozen][:12]})')
        try:
            start = self.start()
        except NotAvailable:
            start = None
        if start is not None and str(prov.get('time', '')) < start:
            raise NotAvailable(f"{rel} was written at {prov.get('time')}, before the run started ({start})")
        if key in ANALYSIS and rep.get('analysis') != ANALYSIS[key]:
            raise NotAvailable(f"{rel} is a report of analysis {rep.get('analysis')!r}, not {ANALYSIS[key]!r}")
        return rep

    def pieces(self, key):
        """status.json pieces.<set>: planned, decisions, ok, skipped (finished in an earlier attempt), failed,
        failed_pieces of the run's last attempt."""
        st = self.status()
        p = (st.get('pieces') or {}).get(key)
        if p is None:
            raise NotAvailable(f"the run did not reach inference of the {key} set (last stage '{st.get('stage')}')")
        return p

    def pieces_log(self):
        """Per (set, piece), from the run's infer lines in log.txt: the outcome of its last event (ok / failed; a
        'done already, skipped' piece is ok), its planned decisions and the wall seconds of all its attempts."""
        def load():
            out = {}
            for _, msg in self.run_log():
                m = INFER_RUN.match(msg)
                if m:
                    d = out.setdefault((m[1], m[2]), dict(outcome=None, decisions=None, seconds=0.))
                    d['decisions'] = int(m[5])
                    d['seconds'] += float(m[6].replace(',', '.'))
                    d['outcome'] = 'ok' if int(m[4]) == 0 else 'failed'
                    continue
                m = INFER_SKIP.match(msg)
                if m:
                    out.setdefault((m[1], m[2]), dict(outcome=None, decisions=None, seconds=0.))['outcome'] = 'ok'
            if not out:
                raise NotAvailable(f'no infer.py run in {LOG} since the run started')
            return out
        return self._once('pieces_log', load)

    def primary_case(self, sid):
        """Stretch sid of the primary report: its per-strip entry and its annotator's sensitivity aggregate. In the
        primary set each annotator has one fiber with one stretch (kb: S0000, lt: S0001), so that aggregate holds
        the stretch's nodes alone; this is checked here."""
        def load():
            rep = self.report('primary')
            fiber = PRIMARY_FIBER[sid]
            ann = fiber.split('_')[0]
            strips = need(rep, 'per_strip')
            mine = [e for e in strips if sid in (e.get('stretches') or [])]
            if not mine:
                skips = '; '.join(f"{s.get('strip')}: {s.get('reason')}" for s in rep.get('skips') or [])
                raise NotAvailable(f'no scored strip for {sid}' + (f' (skips: {skips})' if skips else ''))
            e = mine[0]
            if e.get('fiber') != fiber:
                raise NotAvailable(f"{sid} was scored on fiber {e.get('fiber')}, not {fiber}")
            same = [x for x in strips if str(x.get('fiber', '')).split('_')[0] == ann]
            if same != [e] or e.get('stretches') != [sid]:
                raise NotAvailable(f'the annotator {ann} aggregate is not {sid} alone')
            return e, need(rep, 'sensitivity', f'annotator {ann}'), min_nodes(rep)
        return self._once(('case', sid), load)


# ---- values ---------------------------------------------------------------------------------------------------

def no_source(c):
    raise Unresolved("no source: deviations after the v3.2 freeze are the author's statement (PREREG 10.6); "
                     'replace this placeholder by hand')


def run_start(c):
    return utc(c.start(), c.freeze()['offset'])


def run_end(c):
    return utc(c.final()['finished'], c.freeze()['offset'])


def run_rc(c):
    rc = c.final()['rc']
    done = c.done()
    text = str(rc)
    if rc != 0:
        text += f" ({RC_MEANING.get(rc, 'not a run_all.ps1 exit code')}"
        if done and done['stage']:
            text += f"; last stage '{done['stage']}'"
        text += ')'
    if done and done['rc'] != rc:
        text += f"; {DONE} says rc {done['rc']}"
    return text


def pieces_ok(key):
    def fn(c):
        p = c.pieces(key)
        return count(int(p['ok']) + int(p['skipped']))
    return fn


def pieces_failed(key):
    def fn(c):
        p = c.pieces(key)
        failed = int(p['failed'])
        ids = p.get('failed_pieces') or []
        ids = [ids] if isinstance(ids, str) else list(ids)
        extra = [', '.join(map(str, ids))] if failed and ids else []
        rest = int(p['planned']) - int(p['ok']) - int(p['skipped']) - failed
        if rest > 0:
            extra.append(f"{rest} of {p['planned']} not run")
        return count(failed) + (f" ({'; '.join(extra)})" if extra else '')
    return fn


def decisions_total(c):
    pl = c.pieces_log()
    ok = {k: d for k, d in pl.items() if d['outcome'] == 'ok'}
    unknown = sorted(f'{s} {p}' for (s, p), d in ok.items() if d['decisions'] is None)
    if unknown:
        raise NotAvailable(f"the decisions of {', '.join(unknown)} are not in {LOG}")
    try:
        st = c.status().get('pieces') or {}
    except NotAvailable:
        st = {}
    for key, p in st.items():  # the log and status.json must count the same finished pieces
        n_log = sum(1 for (s, _) in ok if s == key)
        n_status = int(p.get('ok', 0)) + int(p.get('skipped', 0))
        if n_log != n_status:
            raise NotAvailable(f'{LOG} shows {n_log} finished {key} pieces, {STATUS} {n_status}')
    return count(sum(d['decisions'] for d in ok.values()))


def cpu_hours(c):
    return num(sum(d['seconds'] for d in c.pieces_log().values())/3600)


def ct_gib(c):
    total, cur, seen = 0., 0., False
    for _, msg in c.run_log():
        if PREFETCH_START.match(msg):  # a prefetch_v3.py invocation starts: its MiB count from 0
            total, cur, seen = total+cur, 0., True
            continue
        m = PREFETCH_MIB.match(msg)
        if m:
            cur = max(cur, float(m[2]))
    if not seen:
        raise NotAvailable(f'no CT prefetch in {LOG} since the run started')
    return num((total+cur)/1024)


def strip_baseline_gap(e, base, mn):
    """Why one strip has no baseline value: the scorer's own note, no normals, or too few nodes covered by both."""
    if f'{base}_nodes' not in e:
        if base == 'frame_plane':
            return next((n for n in e.get('notes') or [] if 'frame-plane' in n), 'no frame-plane nodes on this strip')
        return 'no Lasagna normals for this fiber'
    return f'fewer than {mn} nodes covered by both the strip and the {BASE_NAME[base]}'


def case(sid, what):
    def fn(c):
        e, agg, mn = c.primary_case(sid)
        if what == 'nodes':
            return count(need(e, 'strip_nodes'))
        point = need(agg, 'strip', 'point')
        if point is None:
            raise NotAvailable(f"{e.get('strip_nodes')} covered nodes, fewer than {mn}: not scored")
        if what == 'mean':
            return num(need(point, 'mean'))
        if what == 'within2':
            return pct(need(point, 'within2'))
        if what == 'bias':
            return signed(need(agg, 'strip_signed_mean_sv'))
        if what == 'floor':
            floor = need(agg, 'fiber_floor')
            if floor is None:
                raise NotAvailable('no covered fiber point on the scored rows')
            return num(need(floor, 'mean'))
        if need(agg, what).get('stats') is None:  # what: 'frame_plane' or 'lasagna_plane'
            raise NotAvailable(strip_baseline_gap(e, what, mn))
        return num(need(agg, what, 'stats', 'point', 'mean'))
    return fn


def sec_fibers(fam):
    return lambda c: count(need(c.report('secondary'), 'accuracy', fam, 'scored_fibers'))


def sec(fam, what):
    def fn(c):
        agg = need(c.report('secondary'), 'accuracy', fam)
        strip = need(agg, 'strip')
        point = need(strip, 'point')
        if point is None:
            raise NotAvailable(f'no scored {FAMILY_NAME[fam]}')
        if what == 'mean_ci':
            if strip.get('ci') is None:
                raise NotAvailable(f"{plural(strip.get('fibers'), 'scored ' + FAMILY_NAME[fam])}: "
                                   'no bootstrap CI with fewer than 2')
            return ci(need(strip, 'ci', 'mean'))
        if what == 'bias':
            return signed(need(agg, 'strip_signed_mean_sv'))
        if what == 'nodes':
            return count(need(point, 'nodes'))
        if what in ('within1', 'within2'):
            return pct(need(point, what))
        return num(need(point, what))  # mean, median, p90
    return fn


def g1_prime(c):
    return 'passes' if need(c.report('secondary'), 'G2', 'checks', 'scored_fibers_ge_30') else 'fails'


def report_baseline_gap(rep, base):
    """Why the secondary set has no baseline value or comparison (PREREG 5.5: >= 25 nodes covered by both, on at
    least 2 fibers for the paired comparison)."""
    agg = need(rep, 'accuracy', 'all')
    mn = min_nodes(rep)
    if not agg.get('scored_fibers'):
        return 'no scored fiber'
    strips = rep.get('per_strip') or []  # every strip associated with its fiber, scored or not
    lacking = [e for e in strips if f'{base}_nodes' not in e]
    if strips and len(lacking) == len(strips):
        if base == 'frame_plane':
            note = next((n for e in lacking for n in e.get('notes') or [] if 'frame-plane' in n), None)
            return f'no frame-plane nodes on any of the {len(strips)} strips' + (f' ({note})' if note else '')
        return f'no Lasagna normals for any of the {len(strips)} strips'
    n = int(need(agg, base).get('available_fibers') or 0)
    if n == 0:
        return f'no fiber with {mn} nodes covered by both the strip and the {BASE_NAME[base]}'
    return f"{plural(n, 'fiber')} with {mn} nodes covered by both; the paired comparison needs 2"


def baseline(base, what):
    def fn(c):
        rep = c.report('secondary')
        b = need(rep, 'accuracy', 'all', base)
        if what == 'mean':
            if b.get('stats') is None:
                raise NotAvailable(report_baseline_gap(rep, base))
            return num(need(b, 'stats', 'point', 'mean'))
        if what == 'beaten':  # the scorer's gate check; a comparison it could not make counts as not beaten
            beaten = need(rep, 'G2', 'checks', f'beats_{base}')
            if b.get('delta') is None:
                return f'no (not computed: {report_baseline_gap(rep, base)}; counts as not beaten)'
            return 'yes' if beaten else 'no'
        d = b.get('delta')
        if d is None:
            raise NotAvailable(report_baseline_gap(rep, base))
        if what == 'diff':
            return signed(need(d, 'baseline_minus_strip_mean_sv'))
        if what == 'diff_ci':
            return ci(need(d, 'ci'), signed)
        return pct_points(need(d, 'strip_minus_baseline_within2')) + ' (strip \u2212 baseline)'  # what == 'w2'
    return fn


def verdict(c):
    v = str(need(c.report('secondary'), 'G2', 'verdict'))
    if not v.startswith('secondary: '):  # the README writes the prefix itself
        raise Unresolved(f"the secondary verdict {v!r} lacks the scorer's 'secondary: ' prefix")
    return v[len('secondary: '):]


def fiber_floor(c):
    agg = need(c.report('secondary'), 'accuracy', 'all')
    floor = need(agg, 'fiber_floor')
    if floor is None:
        raise NotAvailable('no scored fiber' if not agg.get('scored_fibers') else 'no covered fiber point')
    return (f"mean {num(need(floor, 'mean'))} sv, median {num(need(floor, 'median'))} sv over "
            f"{count(need(floor, 'nodes'))} rows")


def calibration(what):
    def fn(c):
        rep = c.report('secondary')
        cal = need(rep, 'calibration')
        if cal is None:
            raise NotAvailable(f'no fiber with {min_nodes(rep)} scored nodes that carry expected_error')
        v = need(cal, what)
        if v is None:
            if what == 'spearman_fiber' and int(cal.get('fibers') or 0) < 3:
                raise NotAvailable(f"{plural(int(cal.get('fibers') or 0), 'fiber')}: rho needs 3")
            if what == 'observed_over_expected':
                raise NotAvailable('mean expected_error is 0')
            raise NotAvailable('undefined (constant values)')
        return num(v)
    return fn


def sens_line(agg):
    point = need(agg, 'strip', 'point')
    if point is None:
        raise NotAvailable('no scored fiber')
    cis = need(agg, 'strip').get('ci')
    return (f"mean |d| {num(need(point, 'mean'))} sv" + (f" [{ci(need(cis, 'mean'))}]" if cis else ' (no CI)')
            + f", {pct(need(point, 'within2'))} within 2 sv ({plural(int(need(agg, 'scored_fibers')), 'fiber')})")


def sens_unflagged(c):
    return sens_line(need(c.report('secondary'), 'sensitivity', UNFLAGGED))


def sens_annotator(c):
    sens = need(c.report('secondary'), 'sensitivity')
    keys = sorted(k for k in sens if k.startswith('annotator '))
    if not keys:
        raise Unresolved('report field sensitivity has no annotator entry')
    parts = []
    for k in keys:
        try:
            parts.append(f"{k[len('annotator '):]}: {sens_line(sens[k])}")
        except NotAvailable as error:
            parts.append(f"{k[len('annotator '):]}: not available ({error})")
    return '; '.join(parts)


def rejected(c):
    skips = need(c.report('secondary'), 'skips')
    bad = [s for s in skips if 'unit or association error' in str(s.get('reason', ''))]
    text = f"{len(bad)} ({', '.join(str(s.get('strip')) for s in bad)})" if bad else 'none'
    other = [s for s in skips if s not in bad]
    if other:
        text += f'; other skips: {len(other)} (' + '; '.join(
            ' '.join(str(s[k]) for k in ('strip', 'stretch') if s.get(k)) + f": {s.get('reason')}" for s in other) + ')'
    return text


def x0332(what):
    def fn(c):
        res = need(c.report('0332'), 'crossings')
        n = int(need(res, 'crossings'))
        if what == 'n':
            planned = int(need(res, 'planned'))
            return count(n) + (f' (of {planned} planned)' if planned != PLANNED_0332 else '')
        if what == 'missing':
            ids = need(res, 'missing')
            return f"{len(ids)} ({', '.join(map(str, ids))})" if ids else 'none'
        if what == 'tight_n':
            return count(need(res, 'tight_subset', 'crossings')) if n else '0'
        if n == 0:
            raise NotAvailable(str(res.get('note') or 'no crossing with both strips'))
        if what in ('tight_mean', 'tight_median'):
            tight = need(res, 'tight_subset')
            if not tight.get('crossings'):
                raise NotAvailable('no tight crossing with both strips')
            return num(need(tight, what[len('tight_'):]))
        if what == 'mean_ci':
            return ci(need(res, 'ci', 'mean'))
        if what == 'within2':
            return pct(need(res, 'within2'))
        return num(need(res, what))  # mean, median, p90
    return fn


def crossings_0139(c):
    parts, total, known = [], 0, True
    for key in ('secondary', 'primary'):
        try:
            rep = c.report(key)
            cr = need(rep, 'crossings')
            if cr is None:
                raise NotAvailable('not computed (score without --crossings)')
            tol = float(need(rep, 'provenance', 'constants', 'CROSS_TOL'))
            n, planned = int(need(cr, 'crossings')), int(need(cr, 'planned'))
            if planned == 0:
                parts.append(f'{key}: no H/V fiber pair within {tol:g} sv')
            else:
                text = f'{key}: {n} of {planned} H/V fiber pairs within {tol:g} sv with both strips'
                if n:
                    text += f", agreement mean {num(need(cr, 'mean'))} sv, median {num(need(cr, 'median'))} sv"
                parts.append(text)
            total += n
        except NotAvailable as error:
            parts.append(f'{key}: not available ({error})')
            known = False
    if not known and all(': not available (' in p for p in parts):
        raise NotAvailable('; '.join(parts))
    return f"{total} ({'; '.join(parts)})" if known else '; '.join(parts)


# ---- placeholder -> source -------------------------------------------------------------------------------------

LOGS = 'runs/v3/log.txt (run_all.ps1, prefetch_v3.py lines), from the run start: '
STAT = 'runs/v3/status.json: '
P = 'primary_case_studies.json: '
AKB, ALT = "sensitivity['annotator kb']", "sensitivity['annotator lt']"  # = S0000 alone, = S0001 alone (checked)
S = 'secondary_report.json: '
X = 'crossings_0332_report.json: crossings.'

FIELDS = [
    ('DEVIATIONS', 'none: the author states any deviation after the v3.2 freeze (PREREG 10.6)', no_source),
    ('RUN_START', 'runs/v3/log.txt: first "START v3 runner" at or after FREEZE.txt frozen_at_local; UTC with its '
                  'offset', run_start),
    ('RUN_END', 'runs/v3/V3_FINAL: finished= (relaunch_v3.ps1); UTC with the FREEZE.txt offset', run_end),
    ('RUN_RC', 'runs/v3/V3_FINAL: rc= (the last run_all.ps1 exit code; meaning from its header, stage from '
               'runs/v3/DONE)', run_rc),
    ('PRIMARY_PIECES_OK', STAT + 'pieces.primary.ok + pieces.primary.skipped (skipped = done in an earlier attempt)',
     pieces_ok('primary')),
    ('PRIMARY_PIECES_FAILED', STAT + 'pieces.primary.failed, failed_pieces (planned pieces not run noted)',
     pieces_failed('primary')),
    ('SEC_PIECES_OK', STAT + 'pieces.secondary.ok + pieces.secondary.skipped', pieces_ok('secondary')),
    ('SEC_PIECES_FAILED', STAT + 'pieces.secondary.failed, failed_pieces', pieces_failed('secondary')),
    ('0332_PIECES_OK', STAT + 'pieces.0332.ok + pieces.0332.skipped', pieces_ok('0332')),
    ('0332_PIECES_FAILED', STAT + 'pieces.0332.failed, failed_pieces', pieces_failed('0332')),
    ('DECISIONS_TOTAL', LOGS + "'infer <set> <piece> (attempt n): rc 0, N decisions': planned N of every piece whose "
                        "last outcome is ok (checked against status.json's counts)", decisions_total),
    ('CPU_HOURS', LOGS + "'infer ... (attempt n): ..., S s': wall seconds of every infer.py attempt (incl. start-up "
                  'and failed attempts; scoring excluded) / 3600', cpu_hours),
    ('CT_GIB', LOGS + "prefetch_v3.py 'MiB' of each invocation ('done:' or last progress line) / 1024: CT fetched "
               'by the run itself', ct_gib),
    ('P_S0000_NODES', P + "per_strip[strip of S0000].strip_nodes", case('S0000', 'nodes')),
    ('P_S0000_MEAN', P + AKB + '.strip.point.mean', case('S0000', 'mean')),
    ('P_S0000_W2', P + AKB + '.strip.point.within2', case('S0000', 'within2')),
    ('P_S0000_BIAS', P + AKB + '.strip_signed_mean_sv', case('S0000', 'bias')),
    ('P_S0000_FRAME', P + AKB + '.frame_plane.stats.point.mean (nodes covered by both)', case('S0000', 'frame_plane')),
    ('P_S0000_LASAGNA', P + AKB + '.lasagna_plane.stats.point.mean (nodes covered by both)',
     case('S0000', 'lasagna_plane')),
    ('P_S0000_FLOOR', P + AKB + '.fiber_floor.mean', case('S0000', 'floor')),
    ('P_S0001_NODES', P + "per_strip[strip of S0001].strip_nodes", case('S0001', 'nodes')),
    ('P_S0001_MEAN', P + ALT + '.strip.point.mean', case('S0001', 'mean')),
    ('P_S0001_W2', P + ALT + '.strip.point.within2', case('S0001', 'within2')),
    ('P_S0001_BIAS', P + ALT + '.strip_signed_mean_sv', case('S0001', 'bias')),
    ('P_S0001_FRAME', P + ALT + '.frame_plane.stats.point.mean (nodes covered by both)', case('S0001', 'frame_plane')),
    ('P_S0001_LASAGNA', P + ALT + '.lasagna_plane.stats.point.mean (nodes covered by both)',
     case('S0001', 'lasagna_plane')),
    ('P_S0001_FLOOR', P + ALT + '.fiber_floor.mean', case('S0001', 'floor')),
    ('SEC_FIBERS_SCORED', S + 'accuracy.all.scored_fibers (fibers with >= 25 covered nodes)', sec_fibers('all')),
    ('SEC_G1P', S + "G2.checks.scored_fibers_ge_30 (the scorer's G1' check): passes / fails", g1_prime),
    ('SEC_ALL_FIBERS', S + 'accuracy.all.scored_fibers', sec_fibers('all')),
    ('SEC_ALL_NODES', S + 'accuracy.all.strip.point.nodes', sec('all', 'nodes')),
    ('SEC_ALL_MEAN', S + 'accuracy.all.strip.point.mean', sec('all', 'mean')),
    ('SEC_ALL_MEAN_CI', S + 'accuracy.all.strip.ci.mean', sec('all', 'mean_ci')),
    ('SEC_ALL_MEDIAN', S + 'accuracy.all.strip.point.median', sec('all', 'median')),
    ('SEC_ALL_P90', S + 'accuracy.all.strip.point.p90', sec('all', 'p90')),
    ('SEC_ALL_W1', S + 'accuracy.all.strip.point.within1', sec('all', 'within1')),
    ('SEC_ALL_W2', S + 'accuracy.all.strip.point.within2', sec('all', 'within2')),
    ('SEC_ALL_BIAS', S + 'accuracy.all.strip_signed_mean_sv', sec('all', 'bias')),
    ('SEC_H_FIBERS', S + 'accuracy.H.scored_fibers', sec_fibers('H')),
    ('SEC_H_NODES', S + 'accuracy.H.strip.point.nodes', sec('H', 'nodes')),
    ('SEC_H_MEAN', S + 'accuracy.H.strip.point.mean', sec('H', 'mean')),
    ('SEC_H_MEAN_CI', S + 'accuracy.H.strip.ci.mean', sec('H', 'mean_ci')),
    ('SEC_H_MEDIAN', S + 'accuracy.H.strip.point.median', sec('H', 'median')),
    ('SEC_H_P90', S + 'accuracy.H.strip.point.p90', sec('H', 'p90')),
    ('SEC_H_W1', S + 'accuracy.H.strip.point.within1', sec('H', 'within1')),
    ('SEC_H_W2', S + 'accuracy.H.strip.point.within2', sec('H', 'within2')),
    ('SEC_H_BIAS', S + 'accuracy.H.strip_signed_mean_sv', sec('H', 'bias')),
    ('SEC_V_FIBERS', S + 'accuracy.V.scored_fibers', sec_fibers('V')),
    ('SEC_V_NODES', S + 'accuracy.V.strip.point.nodes', sec('V', 'nodes')),
    ('SEC_V_MEAN', S + 'accuracy.V.strip.point.mean', sec('V', 'mean')),
    ('SEC_V_MEAN_CI', S + 'accuracy.V.strip.ci.mean', sec('V', 'mean_ci')),
    ('SEC_V_MEDIAN', S + 'accuracy.V.strip.point.median', sec('V', 'median')),
    ('SEC_V_P90', S + 'accuracy.V.strip.point.p90', sec('V', 'p90')),
    ('SEC_V_W1', S + 'accuracy.V.strip.point.within1', sec('V', 'within1')),
    ('SEC_V_W2', S + 'accuracy.V.strip.point.within2', sec('V', 'within2')),
    ('SEC_V_BIAS', S + 'accuracy.V.strip_signed_mean_sv', sec('V', 'bias')),
    ('SEC_FRAME_MEAN', S + 'accuracy.all.frame_plane.stats.point.mean', baseline('frame_plane', 'mean')),
    ('SEC_FRAME_DIFF', S + 'accuracy.all.frame_plane.delta.baseline_minus_strip_mean_sv',
     baseline('frame_plane', 'diff')),
    ('SEC_FRAME_DIFF_CI', S + 'accuracy.all.frame_plane.delta.ci', baseline('frame_plane', 'diff_ci')),
    ('SEC_FRAME_W2_DIFF', S + 'accuracy.all.frame_plane.delta.strip_minus_baseline_within2',
     baseline('frame_plane', 'w2')),
    ('SEC_FRAME_BEATEN', S + 'G2.checks.beats_frame_plane (why not, from accuracy.all.frame_plane)',
     baseline('frame_plane', 'beaten')),
    ('SEC_LASAGNA_MEAN', S + 'accuracy.all.lasagna_plane.stats.point.mean', baseline('lasagna_plane', 'mean')),
    ('SEC_LASAGNA_DIFF', S + 'accuracy.all.lasagna_plane.delta.baseline_minus_strip_mean_sv',
     baseline('lasagna_plane', 'diff')),
    ('SEC_LASAGNA_DIFF_CI', S + 'accuracy.all.lasagna_plane.delta.ci', baseline('lasagna_plane', 'diff_ci')),
    ('SEC_LASAGNA_W2_DIFF', S + 'accuracy.all.lasagna_plane.delta.strip_minus_baseline_within2',
     baseline('lasagna_plane', 'w2')),
    ('SEC_LASAGNA_BEATEN', S + 'G2.checks.beats_lasagna_plane (why not, from accuracy.all.lasagna_plane)',
     baseline('lasagna_plane', 'beaten')),
    ('SEC_VERDICT', S + "G2.verdict verbatim, without the 'secondary: ' prefix the README writes", verdict),
    ('SEC_FIBER_FLOOR', S + 'accuracy.all.fiber_floor.mean, .median, .nodes (one per scored row)', fiber_floor),
    ('SEC_CAL_GAP', S + 'calibration.weighted_abs_gap_sv', calibration('weighted_abs_gap_sv')),
    ('SEC_CAL_RATIO', S + 'calibration.observed_over_expected', calibration('observed_over_expected')),
    ('SEC_CAL_RHO_NODE', S + 'calibration.spearman_node', calibration('spearman_node')),
    ('SEC_CAL_RHO_FIBER', S + 'calibration.spearman_fiber', calibration('spearman_fiber')),
    ('SEC_SENS_UNFLAGGED', S + f"sensitivity['{UNFLAGGED}']: scored_fibers, strip.point.mean, strip.ci.mean, "
                           'strip.point.within2', sens_unflagged),
    ('SEC_SENS_ANNOTATOR', S + "sensitivity['annotator *'], each: scored_fibers, strip.point.mean, strip.ci.mean, "
                           'strip.point.within2', sens_annotator),
    ('SEC_REJECTED', S + "skips whose reason is the scorer's 'unit or association error' (other skips listed)",
     rejected),
    ('0332_N', X + 'crossings (crossings with an agreement value; planned checked against 15)', x0332('n')),
    ('0332_MEAN', X + 'mean', x0332('mean')),
    ('0332_MEAN_CI', X + 'ci.mean', x0332('mean_ci')),
    ('0332_MEDIAN', X + 'median', x0332('median')),
    ('0332_P90', X + 'p90', x0332('p90')),
    ('0332_W2', X + 'within2', x0332('within2')),
    ('0332_TIGHT_N', X + 'tight_subset.crossings', x0332('tight_n')),
    ('0332_TIGHT_MEAN', X + 'tight_subset.mean', x0332('tight_mean')),
    ('0332_TIGHT_MEDIAN', X + 'tight_subset.median', x0332('tight_median')),
    ('0332_MISSING', X + 'missing', x0332('missing')),
    ('0139_CROSSINGS', 'secondary_report.json and primary_case_studies.json: crossings (score --crossings): '
                       'crossings, planned, mean, median; provenance.constants.CROSS_TOL', crossings_0139),
]
BY_NAME = {name: (source, fn) for name, source, fn in FIELDS}


def resolve(c, name):
    """(value, source, state) with state 'ok', 'na' (not available) or 'unresolved'. Never raises."""
    if name not in BY_NAME:
        return 'no source defined in tools/fill_results.py', 'none', 'unresolved'
    source, fn = BY_NAME[name]
    try:
        value = md(fn(c))
        if not value:
            raise Unresolved('empty value')
        return value, source, 'ok'
    except NotAvailable as error:
        return md(f'not available ({error})'), source, 'na'
    except Unresolved as error:
        return md(error), source, 'unresolved'
    except Exception as error:  # noqa: BLE001 (an unexpected report layout must block the write, not crash it)
        return md(f'{type(error).__name__}: {error}'), source, 'unresolved'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--workspace', required=True, help="the author's score/ folder (C:/VesuviusTemp/strips/score)")
    ap.add_argument('--readme', default=str(REPO/'README.md'), help='default: README.md of this repo')
    ap.add_argument('--dry-run', action='store_true', help='print placeholder -> value (source); write nothing')
    a = ap.parse_args(argv)
    try:  # a console that cannot show a character must not stop the fill
        sys.stdout.reconfigure(errors='backslashreplace')
    except (AttributeError, ValueError):
        pass
    readme = Path(a.readme)
    if not readme.is_file():
        print(f'{readme}: no such file')
        return 2
    text = readme.read_bytes().decode('utf-8')
    names = list(dict.fromkeys(PLACEHOLDER.findall(text)))  # README order, each once
    c = Ctx(a.workspace)
    rows = [(name, *resolve(c, name)) for name in names]
    for name, value, source, state in rows:
        shown = f'UNRESOLVED: {value}' if state == 'unresolved' else value
        print(f'{{{{RESULT_{name}}}}} -> {shown} ({source})')
    unresolved = [name for name, _, _, state in rows if state == 'unresolved']
    n_na = sum(state == 'na' for *_, state in rows)
    print(f'{len(rows)} placeholders in {readme}: {len(rows) - n_na - len(unresolved)} filled, {n_na} not '
          f'available, {len(unresolved)} unresolved')
    if a.dry_run:
        print('dry run: nothing written')
        return 1 if unresolved else 0
    if unresolved:
        print(f'{readme} not written; unresolved: ' + ', '.join(f'RESULT_{n}' for n in unresolved))
        return 1
    if not rows:
        print(f'{readme}: no placeholder left; nothing to do')
        return 0
    values = {name: value for name, value, _, _ in rows}
    tmp = readme.with_name(readme.name + '.fill_tmp')
    tmp.write_bytes(PLACEHOLDER.sub(lambda m: values[m[1]], text).encode('utf-8'))
    os.replace(tmp, readme)
    print(f'wrote {readme}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
