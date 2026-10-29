#!/usr/bin/env python
"""README figures from the frozen scorer's reports, made after the scored run (PREREG v3.2; score/score_strips.py).

    CUDA_VISIBLE_DEVICES=-1 python tools/make_figures.py --workspace C:/VesuviusTemp/strips/score \
        --out figures/ --no-primary-overlay [--dry-run]

Reads only these files of the workspace (no strip, no log):
  results/v3/secondary_report.json          ``score --run-only``: the PHerc0139 secondary set (PREREG 4.10)
  results/v3/primary_case_studies.json      ``score``: the two primary case-study stretches (G1 failed)
  results/v3/crossings_0332_report.json     ``crossings --plan``: PHerc0332 H/V crossings (no ground truth)
  results/runplan_0139_secondary_run40.tsv  the frozen run plan (piece -> AFV id), for the example-strip commands
as score/runs/v3/run_all.ps1 writes them (the commands are in score/PREREG.md section 11).

What the reports hold, and so what is drawn. ``score`` writes no per-node distances. Per set and family it writes
pooled summaries of |d| (nodes, mean, median, p90, fractions within 1 and 2 sv) with fiber-bootstrap 95 % CIs: for
the strip on all its scored nodes, for each baseline on the nodes covered by both the strip and that baseline
(``stats``, with the strip on the same nodes and the paired difference), and for the fiber floor (pooled over one
point per scored row, no CI); the calibration bins; and per strip, i.e. per infer.py piece (``per_strip``), the node
count and mean |d| of each surface. Per-fiber means come from ``per_strip`` alone: node-weighted over a fiber's
strips, which is the scorer's own fiber cluster (aggregate(): the covered strip nodes on a fiber's scored rows), and a
fiber counts as scored with at least MIN_NODES strip nodes, as in aggregate(). Both are checked against the report's
pooled numbers. Nothing is recomputed from strips. ``crossings`` writes, per crossing, the agreement and the closest
approach of the two human fibers.

Figures, as PNG (150 dpi) and SVG (matplotlib when it imports; otherwise the SVG is written directly and there is no
PNG), distances in sv (level-2 voxels, 9.596 um), with figures/captions.json (per figure: caption, notes, the plotted
numbers as a table, and the report's sha256):
  fig1_secondary_distance    |d| to the segment: the reported points of the pooled per-node distribution for the strip,
                             both baselines and the fiber floor, the G2 thresholds; per-fiber cumulative distributions
  fig2_secondary_per_fiber   per-fiber mean |d|, strip against each baseline, with the report's fiber-bootstrap CIs
  fig3_0332_crossings        PHerc0332 per-crossing agreement, the tight subset marked (no ground truth)
  fig4_calibration           the model's expected_error against the observed |d|, per bin (PREREG 5.7): the secondary
                             set, with the primary case studies' bins overlaid (hollow; G1 failed, no accuracy claim)
                             unless --no-primary-overlay
Every legend hangs below its panel, outside all axes, so no legend can cover a data point. In figure 1 (a), marks of
different surfaces that would overlap are moved apart, perpendicular to the value they show, and the caption says by
how much; every other mark sits at its value. A figure whose data is missing is skipped with the reason (printed, and
in captions.json); a missing or malformed report never stops the tool.

Example strips (rule chosen on 29 Oct 2026 at about 02:31 UTC, while the run was at inference and before any report
existed; it was not published before the results): among the scored secondary fibers ranked by mean |d|, the fiber at
the median (an even count takes the lower of the two middle values) and the fiber at the 90th percentile (nearest
rank, ceil(0.9 n)); a value shared by several fibers goes to the smaller fiber name. For each piece of those fibers
a render command is printed, never run: tools/render_strip.py, which runs villa's own render_saved (villa imported
from the pinned worktree C:/VesuviusTemp/strips/villa_6f2; it refuses to run unless CUDA_VISIBLE_DEVICES=-1) with the
normalization binding of our patch, on a copy of the piece's output, so the run's folders stay as the run left them.
villa's infer.py --render-only cannot be used: it opens the CT without that binding and stops on PHerc0139.
The workspace and render folder are resolved to absolute paths, so the printed commands work from any folder.

Written with Claude (Anthropic); checked by tools/test_make_figures.py on synthetic reports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import textwrap
import time
from pathlib import Path
from xml.sax.saxutils import escape

VERSION = '1.1'
HERE = Path(__file__).resolve()

# ---- inputs (score/runs/v3/run_all.ps1, score/PREREG.md section 11) -------------------------------------------------
REPORTS = {'secondary': 'results/v3/secondary_report.json',
           'primary': 'results/v3/primary_case_studies.json',
           'crossings_0332': 'results/v3/crossings_0332_report.json'}
RUNPLAN = 'results/runplan_0139_secondary_run40.tsv'
RUN_STRIPS = 'runs/v3/strips/secondary'     # infer.py OUT/<piece id> of the secondary set: named, never opened here
RENDER_TOOL = HERE.parent/'render_strip.py'  # render_saved with the normalization binding (--render-only lacks it)
VILLA = 'villa_6f2'                          # the pinned worktree, beside the workspace (run_all.ps1)
PYTHON = r'C:\Users\PC\miniconda3\envs\vesuvius\python.exe'   # run_all.ps1 $Py
CHECKPOINT = 'scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e'
CT_0139 = 's3://vesuvius-challenge-open-data/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr'
RENDER_READS = ('strip', 'strip.npz', 'summary.json')  # villa infer.render_saved: OUT/summary.json, strip.npz, strip/

# PREREG v3.2 values, used only when a report does not record them (its provenance.constants win)
DEFAULTS = {'G2_MAX_MEAN': 1.5, 'G2_MIN_WITHIN2': 0.70, 'MIN_NODES': 25, 'CROSS_TOL': 3.0, 'CROSS_RADIUS': 30.0,
            'SV_UM': 9.596}
WITHIN = (('within1', 1.0), ('within2', 2.0))  # score_strips.pooled(): fractions with |d| <= 1 and <= 2 sv (in code)
SURFACES = ('strip', 'frame_plane', 'lasagna_plane')
LABEL = {'strip': 'strip', 'frame_plane': 'frame plane', 'lasagna_plane': 'Lasagna plane',
         'fiber_floor': 'fiber floor'}
FIGURES = ('fig1_secondary_distance', 'fig2_secondary_per_fiber', 'fig3_0332_crossings', 'fig4_calibration')
RULE = ('among the scored secondary fibers (at least MIN_NODES covered strip nodes), ranked by mean |d| (node-weighted '
        "over the fiber's strips, from per_strip): the median, rank (n+1)//2 (an even count takes the lower of the two "
        'middle values), and the 90th percentile, nearest rank ceil(0.9 n); a value shared by several fibers goes to '
        'the smaller fiber name')
NO_GT = 'no ground truth: consistency, not accuracy'

# ---- style: the dataviz reference palette on a light surface; one hue per surface in every figure ------------------
SURFACE, TEXT, TEXT2, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1', '#b9b8b2'
COLOR = {'strip': '#2a78d6', 'frame_plane': '#eb6834', 'lasagna_plane': '#1baf7a', 'fiber_floor': '#6b6a65'}
MARKER = {'strip': 'o', 'frame_plane': 's', 'lasagna_plane': '^', 'fiber_floor': 'D'}
FAMILY_MARKER = {'V': 'o', 'H': '^'}
MARK = 6.5            # pt: the default marker size (matplotlib ms; the SVG backend draws the same size)
LEGEND_FS = 7.        # pt: legend text
LEGEND_GAP = .5       # in: from an axes' bottom edge to the top of the legend under it (tick labels and x label fit)
LEGEND_ROW = .17      # in per legend entry (matplotlib needs about .16 at 7 pt; the legend hangs from its top edge)
LEGEND_PAD = .16      # in: the legend frame's padding, top and bottom together


# ---- small helpers -------------------------------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    """(object, None), or (None, reason) for a missing, unreadable or non-object file. Never raises."""
    p = Path(path)
    if not p.is_file():
        return None, f'missing: {p}'
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        return None, f'unreadable: {p} ({type(error).__name__}: {error})'
    if not isinstance(data, dict):
        return None, f'not a JSON object: {p}'
    return data, None


def dig(obj, *keys):
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        else:
            return None
    return obj


def num(x):
    """A finite float, else None (JSON null for NaN/inf in the scorer's jsonable(), strings, bools)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def count(x):
    """A node count from JSON (non-negative integer), else 0."""
    if isinstance(x, bool):
        return 0
    if isinstance(x, int):
        return max(x, 0)
    if isinstance(x, float) and math.isfinite(x) and x == int(x):
        return max(int(x), 0)
    return 0


def interval(x):
    """[lo, hi] of two finite numbers (a CI as the scorer writes it), else None."""
    if isinstance(x, (list, tuple)) and len(x) == 2:
        lo, hi = num(x[0]), num(x[1])
        if lo is not None and hi is not None:
            return [lo, hi]
    return None


def f2(x, nd=2):
    return 'n/a' if x is None else f'{x:.{nd}f}'


def pct(x, nd=0):
    return 'n/a' if x is None else f'{100*x:.{nd}f} %'


def ci_txt(iv, nd=2, percent=False):
    if iv is None:
        return ''
    if percent:
        return f' [{100*iv[0]:.{nd}f}, {100*iv[1]:.{nd}f}] %'
    return f' [{iv[0]:.{nd}f}, {iv[1]:.{nd}f}]'


def wrap_ids(ids, per_line=4):
    """A list of ids as lines of at most ``per_line`` (a figure's side text must stay inside the figure)."""
    return ',\n  '.join(', '.join(ids[i:i+per_line]) for i in range(0, len(ids), per_line))


def constants(report):
    """PREREG constants as the report recorded them (provenance.constants), else the PREREG v3.2 defaults."""
    c = dig(report, 'provenance', 'constants')
    out = {}
    for k, v in DEFAULTS.items():
        got = num(c.get(k)) if isinstance(c, dict) else None
        out[k] = got if got is not None else v
    out['MIN_NODES'] = int(out['MIN_NODES'])
    out['source'] = 'report' if isinstance(c, dict) else 'PREREG v3.2 defaults (no provenance.constants in the report)'
    return out


def skip(reason):
    return {'skipped': reason}


def ps_quote(s):
    return "'" + str(s).replace("'", "''") + "'"


def inch_rect(l, b, w, h, W, H):
    """An axes rectangle given in inches (left, bottom, width, height) as figure fractions."""
    return [l/W, b/H, w/W, h/H]


def legend_rows(items):
    return sum(1 for it in items if it.get('label'))


def legend_height(n):
    """Inches reserved for a legend of n one-line entries."""
    return n*LEGEND_ROW+LEGEND_PAD if n else 0.


def legend_below(left, bottom, W, H):
    """Figure-fraction position of the top-left corner of the legend of the axes whose left and bottom edges (inches)
    are given: under its tick labels and x label, outside every axes."""
    return [left/W, (bottom-LEGEND_GAP)/H]


def dodge(extents, step):
    """Offsets that keep overlapping marks apart. Marks whose extents (lo, hi) overlap, directly or through other
    marks, form a group and are spread by ``step`` about their value, in list order; a mark that overlaps no other mark
    keeps offset 0, i.e. is drawn at its value."""
    order = sorted(range(len(extents)), key=lambda i: tuple(extents[i]))
    groups, reach = [], -math.inf
    for i in order:
        lo, hi = extents[i]
        if groups and lo <= reach:
            groups[-1].append(i)
            reach = max(reach, hi)
        else:
            groups.append([i])
            reach = hi
    off = [0.]*len(extents)
    for g in groups:
        for k, i in enumerate(sorted(g)):
            off[i] = (k-(len(g)-1)/2)*step
    return off


# ---- primitives of a figure model (drawn by either backend) ---------------------------------------------------------
# 'role' names what an item shows and 'tags' what each of its points is (both for captions.json readers and tests;
# the backends ignore them).

def points(x, y, color, marker='o', filled=True, size=MARK, label=None, xint=None, yint=None, z=3, role=None,
           tags=None):
    return {'kind': 'points', 'x': list(x), 'y': list(y), 'color': color, 'marker': marker, 'filled': filled,
            'size': size, 'label': label, 'xint': list(xint) if xint else None, 'yint': list(yint) if yint else None,
            'z': z, 'role': role, 'tags': list(tags) if tags is not None else None}


def line(x, y, color, width=1.6, dash=None, label=None, z=2, role=None):
    return {'kind': 'line', 'x': list(x), 'y': list(y), 'color': color, 'width': width, 'dash': dash, 'label': label,
            'z': z, 'role': role}


def vline(x, color=TEXT2, width=1., dash='dash', label=None, z=1.5, role=None):
    return {'kind': 'vline', 'x': x, 'color': color, 'width': width, 'dash': dash, 'label': label, 'z': z,
            'role': role}


def hline(y, color=TEXT2, width=1., dash='dash', label=None, z=1.5, role=None):
    return {'kind': 'hline', 'y': y, 'color': color, 'width': width, 'dash': dash, 'label': label, 'z': z,
            'role': role}


def segment(x0, y0, x1, y1, color=TEXT, width=2., dash=None, z=2.5, role=None):
    return {'kind': 'segment', 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'color': color, 'width': width, 'dash': dash,
            'z': z, 'role': role}


def span(a, b, color, alpha=.08, axis='x', label=None, z=.5, role=None):
    return {'kind': 'span', 'a': a, 'b': b, 'color': color, 'alpha': alpha, 'axis': axis, 'label': label, 'z': z,
            'role': role}


def text(x, y, s, size=7.5, color=TEXT2, ha='left', va='center', coords='data', box=False, z=5, role=None):
    return {'kind': 'text', 'x': x, 'y': y, 's': s, 'size': size, 'color': color, 'ha': ha, 'va': va,
            'coords': coords, 'box': box, 'z': z, 'role': role}


def panel(rect, xlim, ylim, items, xlabel='', ylabel='', title='', xticks=None, yticks=None, grid='both',
          legend_at=None, name=''):
    """``legend_at``: figure-fraction position of the legend's top-left corner (outside the axes), or None."""
    return {'rect': rect, 'xlim': list(xlim), 'ylim': list(ylim), 'items': items, 'xlabel': xlabel, 'ylabel': ylabel,
            'title': title, 'xticks': xticks, 'yticks': yticks, 'grid': grid, 'legend_at': legend_at, 'name': name}


PERCENT_TICKS = ([0., .25, .5, .75, 1.], ['0', '25 %', '50 %', '75 %', '100 %'])


# ---- per fiber, from per_strip; the example-strip rule ------------------------------------------------------------

def fiber_table(per_strip, min_nodes):
    """Per fiber, from the report's per_strip rows: its strips (pieces), and per surface the node count and the
    node-weighted mean |d| over the fiber's strips. Each strip's {surface}_mean_abs_sv is the mean over its covered
    nodes, which lie on scored rows only, so the weighted mean is the scorer's per-fiber cluster mean (aggregate()).
    scored: at least ``min_nodes`` strip nodes (aggregate()'s MIN_NODES rule)."""
    fibers = {}
    for row in per_strip if isinstance(per_strip, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get('fiber'), str):
            continue
        f = fibers.setdefault(row['fiber'], {'fiber': row['fiber'], 'family': row.get('family'), 'strips': [],
                                             'nodes': dict.fromkeys(SURFACES, 0), 'sum': dict.fromkeys(SURFACES, 0.)})
        st = {'strip': str(row.get('strip')),
              'stretches': row.get('stretches') if isinstance(row.get('stretches'), list) else [],
              'fiber_floor_mean_abs_sv': num(row.get('fiber_floor_mean_abs_sv'))}
        for s in SURFACES:
            n, m = count(row.get(f'{s}_nodes')), num(row.get(f'{s}_mean_abs_sv'))
            st[f'{s}_nodes'], st[f'{s}_mean_abs_sv'] = n, m
            if n and m is not None:
                f['nodes'][s] += n
                f['sum'][s] += n*m
        f['strips'].append(st)
    for f in fibers.values():
        f['strips'].sort(key=lambda st: st['strip'])
        f['mean'] = {s: (f['sum'][s]/f['nodes'][s] if f['nodes'][s] else None) for s in SURFACES}
        f['scored'] = f['nodes']['strip'] >= min_nodes and f['mean']['strip'] is not None
    return fibers


def check_against_report(fibers, acc):
    """The per_strip derivation against the report's own aggregate: scored fiber count, strip nodes, pooled mean."""
    scored = [f for f in fibers.values() if f['scored']]
    nodes = sum(f['nodes']['strip'] for f in scored)
    mean = sum(f['sum']['strip'] for f in scored)/nodes if nodes else None
    point = dig(acc, 'strip', 'point')
    rep_n = dig(acc, 'scored_fibers')
    # no scored fiber: the scorer's pooled() gives point None, which is 0 nodes
    rep_nodes = point.get('nodes') if isinstance(point, dict) else (0 if isinstance(dig(acc, 'strip'), dict) else None)
    rep_mean = num(dig(acc, 'strip', 'point', 'mean'))
    same_mean = (mean is None and rep_mean is None) or (
        mean is not None and rep_mean is not None and abs(mean-rep_mean) <= 1e-9*max(1., abs(rep_mean)))
    return {'scored_fibers': [len(scored), rep_n], 'strip_nodes': [nodes, rep_nodes],
            'pooled_mean_sv': [mean, rep_mean], 'ok': bool(rep_n == len(scored) and rep_nodes == nodes and same_mean),
            'note': 'derived from per_strip, then as the report states it (accuracy.all)'}


def pick_examples(fibers):
    """The example-strip rule (RULE; chosen during the run at about 02:31 UTC on 29 Oct, not pre-registered) on the
    scored fibers; None when there is none."""
    ranked = sorted((f['mean']['strip'], f['fiber']) for f in fibers.values() if f['scored'])
    n = len(ranked)
    if not n:
        return None

    def at(rank):
        value = ranked[rank-1][0]
        tied = [name for m, name in ranked if m == value]
        return {'fiber': min(tied), 'rank': rank, 'mean_abs_sv': value, 'fibers_with_this_value': len(tied)}
    return {'n': n, 'median': at((n+1)//2), 'p90': at(-(-9*n//10))}


def read_runplan(path):
    """{piece: row} of a frozen run plan TSV, or (None, reason)."""
    p = Path(path)
    if not p.is_file():
        return None, f'missing: {p}'
    try:
        with open(p, newline='', encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh, delimiter='\t'))
    except (OSError, csv.Error, UnicodeDecodeError) as error:
        return None, f'unreadable: {p} ({type(error).__name__}: {error})'
    table = {r['piece']: r for r in rows if r.get('piece')}
    return (table, None) if table else (None, f'no piece rows in {p}')


def render_block(piece, afv_id, kind, ws, render_dir, out_dir):
    """PowerShell lines that copy one piece's infer.py output and render the copy with tools/render_strip.py (villa's
    render_saved plus the normalization binding; ``afv_id`` names the fiber in the comment printed above them)."""
    src, dst = Path(ws)/RUN_STRIPS/piece, Path(render_dir)/piece
    png = Path(out_dir)/f'example_strip_{kind}_{piece}.png'
    return [
        f'New-Item -ItemType Directory -Force -Path {ps_quote(dst/"strip")} | Out-Null',
        f'Copy-Item -Force -Path {ps_quote(src/"strip"/"*")} -Destination {ps_quote(dst/"strip")}',
        f'Copy-Item -Force -Path {ps_quote(src/"strip.npz")},{ps_quote(src/"summary.json")} '
        f'-Destination {ps_quote(dst)}',
        f'& {ps_quote(PYTHON)} -I -u {ps_quote(RENDER_TOOL)} --villa {ps_quote(Path(ws).parent/VILLA)} '
        f'--out {ps_quote(dst)} --checkpoint {CHECKPOINT} --ct {CT_0139} '
        f'--cache-dir {ps_quote(Path(ws).parent/"ct_cache")} --device cpu',
        f'Copy-Item -Force -Path {ps_quote(dst/"render.png")} -Destination {ps_quote(png)}',
    ]


def example_strips(sec, fibers, picks, check, ws, render_dir, out_dir):
    """The two example fibers, their pieces and the render commands (printed by main, never run). ``ws``,
    ``render_dir`` and ``out_dir`` are absolute (main resolves them)."""
    if sec is None:
        return skip('no secondary report')
    if picks is None:
        return skip('no scored fiber in the secondary report (per_strip)')
    plan, plan_reason = read_runplan(Path(ws)/RUNPLAN)
    root = Path(ws).parent
    env = (f"$env:CUDA_VISIBLE_DEVICES = '-1'; $env:HF_HOME = {ps_quote(root/'hf_home')}; "
           "$env:HF_HUB_OFFLINE = '1'")
    out = {'rule': RULE, 'scored_fibers': picks['n'], 'consistency_with_report': check, 'runplan': RUNPLAN,
           'runplan_status': 'read' if plan else plan_reason, 'environment': env, 'examples': {}}
    for kind in ('median', 'p90'):
        pk = picks[kind]
        f = fibers[pk['fiber']]
        ex = {'fiber': f['fiber'], 'family': f['family'], 'mean_abs_sv': pk['mean_abs_sv'], 'rank': pk['rank'],
              'of': picks['n'], 'fibers_with_this_value': pk['fibers_with_this_value'],
              'strip_nodes': f['nodes']['strip'], 'pieces': []}
        for st in f['strips']:
            piece = st['strip']
            row = plan.get(piece) if plan else None
            item = {'piece': piece, 'strip_nodes': st['strip_nodes'], 'strip_mean_abs_sv': st['strip_mean_abs_sv'],
                    'stretches': st['stretches']}
            if row is None:
                item['no_command'] = plan_reason if plan is None else f'piece {piece} is not in {RUNPLAN}'
            elif row.get('fiber') != f['fiber']:
                item['no_command'] = f"{RUNPLAN} gives piece {piece} to fiber {row.get('fiber')}, not {f['fiber']}"
            elif not str(row.get('afv_id', '')).isdigit():
                item['no_command'] = f"piece {piece} has no integer afv_id in {RUNPLAN}"
            else:
                item['afv_id'] = int(row['afv_id'])
                item['run_in_plan'] = row.get('run')
                item['commands'] = render_block(piece, item['afv_id'], kind, ws, render_dir, out_dir)
            ex['pieces'].append(item)
        out['examples'][kind] = ex
    return out


def print_examples(ex):
    print('\n== Example strips: selection (rule chosen at about 02:31 UTC on 29 Oct, during the run, before any report) and render commands (printed, never run) ==')
    if 'skipped' in ex:
        print(f"  no example strips: {ex['skipped']}")
        return
    print(f"  rule: {ex['rule']}")
    c = ex['consistency_with_report']
    if c is not None and not c['ok']:
        print(f"  WARNING: per_strip does not reproduce the report's aggregate: {c}")
    if ex['runplan_status'] != 'read':
        print(f"  run plan: {ex['runplan_status']}")
    for kind, e in ex['examples'].items():
        tie = f", {e['fibers_with_this_value']} fibers share this value" if e['fibers_with_this_value'] > 1 else ''
        print(f"  {kind}: fiber {e['fiber']} ({e['family']}), mean |d| {e['mean_abs_sv']:.3f} sv over "
              f"{e['strip_nodes']} nodes, rank {e['rank']} of {e['of']}{tie}; pieces "
              + ', '.join(p['piece'] for p in e['pieces']))
    print('  # PowerShell, after the scored run. Each block copies the piece output (strip/, strip.npz, summary.json)')
    print("  # and renders the copy with tools/render_strip.py (villa's render_saved with the normalization binding;")
    print("  # infer.py --render-only stops on PHerc0139); the run's folders are not written. Not run by this tool.")
    print(f"  {ex['environment']}")
    for kind, e in ex['examples'].items():
        for p in e['pieces']:
            if 'commands' in p:
                print(f"  # {kind} example: fiber {e['fiber']}, piece {p['piece']} (AFV id {p['afv_id']})")
                for cmd in p['commands']:
                    print(f'  {cmd}')
            else:
                print(f"  # {kind} example: piece {p['piece']}: no command ({p['no_command']})")


# ---- figure 1: the secondary set's distance to the segment -------------------------------------------------------

def fig1_distance(sec, C, fibers):
    acc = dig(sec, 'accuracy', 'all')
    point = dig(acc, 'strip', 'point')
    if not isinstance(point, dict) or num(point.get('mean')) is None:
        return skip('the secondary report has no scored strip nodes (accuracy.all.strip.point is empty)')
    notes = ['The report holds no per-node distances: panel (a) shows the points of the pooled per-node distribution '
             'that it states; panel (b) the per-fiber means derived from per_strip.']
    if sec.get('analysis') != 'secondary':
        notes.append(f"The report's analysis is {sec.get('analysis')!r}, not 'secondary'.")
    series = [{'key': 'strip', 'point': point, 'ci': dig(acc, 'strip', 'ci'), 'on': 'all scored nodes',
               'unit': 'nodes'}]
    for base in ('frame_plane', 'lasagna_plane'):
        bp = dig(acc, base, 'stats', 'point')
        if isinstance(bp, dict) and num(bp.get('mean')) is not None:
            series.append({'key': base, 'point': bp, 'ci': dig(acc, base, 'stats', 'ci'),
                           'on': 'nodes covered by both', 'unit': 'nodes'})
        else:
            notes.append(f"The {LABEL[base]} baseline is not in the report (available_fibers "
                         f"{dig(acc, base, 'available_fibers')}): not drawn.")
    floor = dig(acc, 'fiber_floor')
    if isinstance(floor, dict) and num(floor.get('mean')) is not None:
        # pooled() calls the count 'nodes'; the floor has one value per scored row (aggregate()), so it counts rows
        series.append({'key': 'fiber_floor', 'point': floor, 'ci': None,
                       'on': 'the human fiber, one point per scored row', 'unit': 'rows'})
    else:
        notes.append('The fiber floor is not in the report: not drawn.')
    for s in series:
        s['ci'] = s['ci'] if isinstance(s['ci'], dict) else {}
        s['offset'] = {}
    K = len(series)
    scored = [f for f in fibers.values() if f['scored']]
    ecdf = {}
    for s in SURFACES:
        vals = sorted(f['mean'][s] for f in scored if f['nodes'][s] >= C['MIN_NODES'] and f['mean'][s] is not None)
        if vals:
            ecdf[s] = vals
    two = bool(ecdf)
    W = 11.4 if two else 7.6
    left, aw, ah, mh, gap = 1.25, (4.75 if two else 6.0), 2.6, 1.25, .62
    thr2 = dict(WITHIN)['within2']
    w2, g2 = C['G2_MIN_WITHIN2'], C['G2_MAX_MEAN']
    xs = [thr2, g2]
    for s in series:
        for q in ('mean', 'median', 'p90'):
            xs.append(num(s['point'].get(q)) or 0.)
            iv = interval(s['ci'].get(q))
            if iv:
                xs.append(iv[1])
    # room right of the 2 sv line for its two-line G2 label (under 1.3 in): thr2 + .04 xmax + 1.3 in <= xmax
    xmax = max(max(xs)*1.1, thr2/(1-.04-1.3*1.02/aw))
    xlim, ylim = (-.02*xmax, xmax), (0., 1.03)
    hw = MARK/72/aw*(xlim[1]-xlim[0])/2   # half a marker, in sv
    hh = MARK/72/ah*(ylim[1]-ylim[0])/2   # half a marker, in fraction units
    dx, dy = .012*xmax, .015
    # (a) the CDF points the report gives. Per quantity, marks (point and CI bar) of different surfaces that would
    # overlap are moved apart perpendicular to the value they show (dodge); every other mark sits at its value.
    marks = [{'X': [], 'Y': [], 'XI': [], 'YI': [], 'tags': []} for _ in series]
    for q, frac in (('median', .5), ('p90', .9)):
        got = [(k, num(s['point'].get(q)), interval(s['ci'].get(q))) for k, s in enumerate(series)]
        got = [g for g in got if g[1] is not None]
        ext = [((min(v, iv[0]) if iv else v)-hw, (max(v, iv[1]) if iv else v)+hw) for _, v, iv in got]
        for (k, v, iv), off in zip(got, dodge(ext, dy)):
            m = marks[k]
            m['X'].append(v)
            m['Y'].append(frac+off)
            m['XI'].append(iv)
            m['YI'].append(None)
            m['tags'].append(q)
            if off:
                series[k]['offset'][q] = off
    for key, thr in WITHIN:
        got = [(k, num(s['point'].get(key)), interval(s['ci'].get(key))) for k, s in enumerate(series)]
        got = [g for g in got if g[1] is not None]
        ext = [((min(v, iv[0]) if iv else v)-hh, (max(v, iv[1]) if iv else v)+hh) for _, v, iv in got]
        for (k, v, iv), off in zip(got, dodge(ext, dx)):
            m = marks[k]
            m['X'].append(thr+off)
            m['Y'].append(v)
            m['XI'].append(None)
            m['YI'].append(iv)
            m['tags'].append(key)
            if off:
                series[k]['offset'][key] = off
    items = []
    for s, m in zip(series, marks):
        n = count(s['point'].get('nodes'))
        items.append(points(m['X'], m['Y'], COLOR[s['key']], MARKER[s['key']],
                            label=f"{LABEL[s['key']]}: {s['on']} ({n:,} {s['unit']})", xint=m['XI'], yint=m['YI'],
                            role=f"series:{s['key']}", tags=m['tags']))
    items += [vline(thr2, role='within2_line'),
              segment(thr2-.03*xmax, w2, thr2+.03*xmax, w2, TEXT, 2.4, role='g2_within2'),
              text(thr2+.04*xmax, w2-.012, f'G2: at least {pct(w2)} of\nnodes within {thr2:g} sv', va='top', color=TEXT,
                   role='g2_within2_label')]
    # pooled means with their CIs (below panel (a), same x axis)
    mitems = []
    for k, s in enumerate(series):
        y = K-1-k
        m = num(s['point'].get('mean'))
        iv = interval(s['ci'].get('mean'))
        mitems.append(points([m], [y], COLOR[s['key']], MARKER[s['key']], xint=[iv], role=f"mean:{s['key']}"))
        mitems.append(text(max(m, iv[1] if iv else m)+.015*xmax, y, f'{m:.2f}', color=TEXT, role='mean_value'))
    mitems += [vline(g2, role='g2_mean'),
               text(g2+.012*xmax, K-.42, f'G2: pooled mean \u2264 {g2:g} sv', va='bottom', color=TEXT,
                    role='g2_mean_label')]
    # (b) per fiber
    floor_mean = num(dig(floor, 'mean')) if isinstance(floor, dict) else None
    bitems = []
    if two:
        xm = max([v for vals in ecdf.values() for v in vals]+[floor_mean or 0.])*1.08 or 1.
        for s, vals in ecdf.items():
            n = len(vals)
            X, Y = [0.], [0.]
            for i, v in enumerate(vals):
                X += [v, v]
                Y += [i/n, (i+1)/n]
            bitems.append(line(X+[xm], Y+[1.], COLOR[s], 1.8, label=f'{LABEL[s]} ({n} fibers)', role=f'ecdf:{s}'))
        if floor_mean is not None:
            bitems.append(vline(floor_mean, COLOR['fiber_floor'], 1.2, 'dot', role='floor_mean',
                                label=f'fiber floor, pooled mean ({floor_mean:.2f} sv)'))
    else:
        notes.append('Panel (b) is omitted: the report has no per_strip rows of a scored fiber.')
    for s in SURFACES[1:]:
        if s not in ecdf and two:
            notes.append(f'Panel (b): no fiber has {C["MIN_NODES"]} or more {LABEL[s]} nodes in per_strip.')
    # layout (inches): legends hang below the means panel and below panel (b), outside every axes
    b0 = .1+legend_height(max(legend_rows(items), legend_rows(bitems)))+LEGEND_GAP
    pa_bottom = b0+mh+gap
    H = pa_bottom+ah+.75
    xlab = '|d| to the matched wrap segment (sv)'
    panels = [panel(inch_rect(left, pa_bottom, aw, ah, W, H), xlim, ylim, items, xlabel=f'x = {xlab}',
                    ylabel='fraction of nodes with |d| \u2264 x',
                    title='(a) node level: the points of the pooled distribution that the report states',
                    yticks=([0., .25, .5, .75, .9, 1.], ['0', '25 %', '50 %', '75 %', '90 %', '100 %']),
                    legend_at=legend_below(left, b0, W, H), name='a'),
              panel(inch_rect(left, b0, aw, mh, W, H), xlim, (-.6, K+.05), mitems, xlabel=xlab, ylabel='pooled mean',
                    yticks=(list(range(K)), [LABEL[s['key']] for s in reversed(series)]), grid='x', name='means')]
    if two:
        panels.append(panel(inch_rect(7.35, b0, 3.75, mh+gap+ah, W, H), (-.02*xm, xm), (0, 1.03), bitems,
                            xlabel='mean |d| of a fiber (sv)', ylabel='fraction of fibers with mean |d| \u2264 x',
                            title=f'(b) per fiber ({len(scored)} scored): mean over its covered nodes',
                            yticks=PERCENT_TICKS, legend_at=legend_below(7.35, b0, W, H), name='b'))
    sp = series[0]['point']
    sci = series[0]['ci']
    parts = []
    for s in series[1:]:
        iv = interval(s['ci'].get('mean'))
        parts.append(f"{LABEL[s['key']]} {f2(num(s['point'].get('mean')))} sv{ci_txt(iv)}")
    verdict = dig(sec, 'G2', 'verdict')
    skips = dig(sec, 'skips')
    offy = max([abs(o) for s in series for q, o in s['offset'].items() if q in ('median', 'p90')] or [0.])
    offx = max([abs(o) for s in series for q, o in s['offset'].items() if q in ('within1', 'within2')] or [0.])
    moved = (f"marks of different surfaces (point or CI bar) that would overlap are moved apart, perpendicular to the "
             f"value they show, here by at most {offy:.3f} up or down (median, p90) and {offx:.2f} sv sideways "
             f"(within 1 and 2 sv); every other mark is at its value" if offx or offy else
             'no two marks overlap, so every mark is at its value')
    caption = (
        f"Secondary set (PHerc0139, PREREG 4.10): distance |d| from the predicted sheet to the matched wrap segment, "
        f"in sv (1 sv = {C['SV_UM']:g} \u00b5m); {dig(acc, 'scored_fibers')} scored fibers, "
        f"{dig(sec, 'stretches_scored')} of {dig(sec, 'stretches_planned')} planned stretches scored"
        + (f" (the report lists {len(skips)} skips)" if isinstance(skips, list) and skips else '') + '. '
        f"(a) The report holds no per-node distances, only these points of the pooled per-node distribution: the "
        f"median and p90 (horizontal bars: fiber-bootstrap 95 % CI) and the fractions within 1 and 2 sv (vertical "
        f"bars: CI); {moved}. Strip: all scored "
        f"nodes; frame plane and Lasagna plane: the nodes covered by both the strip and that baseline; fiber floor: "
        f"the human fiber itself, one point per scored row (its count is of rows; no CI in the report). Below: "
        f"pooled mean |d| with its CI. "
        f"Strip: mean {f2(num(sp.get('mean')))} sv{ci_txt(interval(sci.get('mean')))}, median "
        f"{f2(num(sp.get('median')))} sv, p90 {f2(num(sp.get('p90')))} sv, within 2 sv "
        f"{pct(num(sp.get('within2')), 1)}{ci_txt(interval(sci.get('within2')), 1, True)}"
        + (f"; {'; '.join(parts)}" if parts else '') + '. '
        f"G2 (PREREG 6) needs a pooled mean \u2264 {g2:g} sv, at least {pct(w2)} of nodes within 2 sv and both "
        f"baselines beaten; the report's verdict: \u201c{verdict}\u201d. "
        + ("(b) Per fiber: the cumulative fraction of scored fibers by their mean |d| over each surface's own "
           "covered nodes (node-weighted over the fiber's strips, from per_strip; a baseline only for fibers with "
           f"at least {C['MIN_NODES']} of its nodes); the dotted line is the fiber floor's pooled mean."
           if two else ''))
    data = {'node_level': [{'surface': s['key'], 'on': s['on'], 'count_is_of': s['unit'],
                            **{q: num(s['point'].get(q)) for q in ('nodes', 'mean', 'median', 'p90', 'within1',
                                                                    'within2')},
                            'ci': {q: interval(s['ci'].get(q)) for q in ('mean', 'median', 'p90', 'within1',
                                                                          'within2')},
                            'drawn_offset': dict(s['offset'])}
                           for s in series],
            'per_fiber': {s: vals for s, vals in ecdf.items()}}
    return {'size': (W, H), 'panels': panels, 'caption': caption, 'notes': notes, 'data': data,
            'title': 'PHerc0139 secondary set: distance from the predicted sheet to the wrap segment'}


# ---- figure 2: per fiber, strip against each baseline ------------------------------------------------------------

def fig2_per_fiber(sec, C, fibers, picks):
    acc = dig(sec, 'accuracy', 'all') or {}
    mn = C['MIN_NODES']
    scored = sorted((f for f in fibers.values() if f['scored']), key=lambda f: f['fiber'])
    if not scored:
        return skip('no scored fiber in the secondary report (per_strip)')
    notes, chosen = [], []
    for base in ('frame_plane', 'lasagna_plane'):
        rows = [f for f in scored if f['nodes'][base] >= mn and f['mean'][base] is not None]
        if rows:
            chosen.append((base, rows))
        else:
            notes.append(f'{LABEL[base]}: no scored fiber has {mn} or more {LABEL[base]} nodes in per_strip '
                         f'(report available_fibers {dig(acc, base, "available_fibers")}): no panel.')
    if not chosen:
        return skip('no baseline per fiber in the secondary report: ' + ' '.join(notes))
    built, data, cap = [], {}, []
    for base, rows in chosen:
        stats, delta = dig(acc, base, 'stats'), dig(acc, base, 'delta')
        bx, bci = num(dig(stats, 'point', 'mean')), interval(dig(stats, 'ci', 'mean'))
        sy = num(dig(acc, base, 'strip_on_same_nodes', 'mean'))
        vals = [f['mean'][base] for f in rows]+[f['mean']['strip'] for f in rows]
        vals += [v for v in (bx, sy, bci[1] if bci else None) if v is not None]
        lim = max(vals)*1.08 if max(vals) > 0 else 1.
        items = [line([0, lim], [0, lim], TEXT2, 1., label='equal mean |d|', z=1, role='diagonal')]
        families = sorted({f['family'] for f in rows}, key=lambda x: (x not in FAMILY_MARKER, str(x)))
        for fam in families:
            fr = [f for f in rows if f['family'] == fam]
            items.append(points([f['mean'][base] for f in fr], [f['mean']['strip'] for f in fr], COLOR[base],
                                FAMILY_MARKER.get(fam, 's'),
                                label=f"{fam if fam in FAMILY_MARKER else 'untagged'} fibers ({len(fr)})",
                                role='fibers', tags=[f['fiber'] for f in fr]))
        if bx is not None and sy is not None:
            items.append(points([bx], [sy], TEXT, 'D', size=7.5, xint=[bci], z=4, role='pooled',
                                label='pooled, nodes covered by both (bar: 95 % CI)'))
        for kind in ('median', 'p90'):
            name = picks[kind]['fiber'] if picks else None
            hit = [f for f in rows if f['fiber'] == name]
            if hit:
                x, y = hit[0]['mean'][base], hit[0]['mean']['strip']
                right = x > .6*lim  # label on the side with room
                items += [points([x], [y], TEXT, 'o', filled=False, size=12.5, z=3.5, role=f'example:{kind}',
                                 tags=[name]),
                          text(x+(-.035 if right else .035)*lim, y, f'{kind} example', size=7, color=TEXT,
                               ha='right' if right else 'left', va='bottom' if y < .06*lim else 'center',
                               role=f'example_label:{kind}')]
        if isinstance(delta, dict) and num(delta.get('baseline_minus_strip_mean_sv')) is not None:
            d, dci = num(delta.get('baseline_minus_strip_mean_sv')), interval(delta.get('ci'))
            beaten = 'beaten' if delta.get('beaten') else 'not beaten'
            subtitle = (f"baseline \u2212 strip = {d:.2f} sv{ci_txt(dci)}: {beaten}\n(paired fiber bootstrap, "
                        f"{delta.get('fibers')} fibers, nodes covered by both)")
            cap.append(f"{LABEL[base]}: {len(rows)} fibers plotted; the report's paired comparison has "
                       f"{delta.get('fibers')} fibers and {count(delta.get('nodes')):,} nodes, baseline \u2212 strip "
                       f"mean |d| = {d:.2f} sv{ci_txt(dci)}, {beaten}")
        else:
            subtitle = f'no paired comparison in the report\n(fewer than 2 fibers with {mn} nodes covered by both)'
            cap.append(f'{LABEL[base]}: {len(rows)} fibers plotted; no paired comparison in the report')
        built.append((base, rows, lim, items, subtitle))
        data[base] = {'fibers': [{'fiber': f['fiber'], 'family': f['family'], 'strip_mean_abs_sv': f['mean']['strip'],
                                  'strip_nodes': f['nodes']['strip'], 'baseline_mean_abs_sv': f['mean'][base],
                                  'baseline_nodes': f['nodes'][base]} for f in rows],
                      'pooled_baseline_mean_sv': bx, 'pooled_baseline_mean_ci': bci,
                      'pooled_strip_mean_on_same_nodes_sv': sy,
                      'delta': delta if isinstance(delta, dict) else None}
    # layout (inches): square panels side by side, each legend hanging below its panel's x label
    side, gap, left, top = 3.9, 1.2, 1.0, 1.4
    bottom = .1+legend_height(max(legend_rows(b[3]) for b in built))+LEGEND_GAP
    need = left+len(built)*side+(len(built)-1)*gap+.4
    W, H = max(need, 7.4), bottom+side+top  # one panel: wide enough for the figure title
    x0 = left+(W-need)/2
    panels = []
    for j, (base, rows, lim, items, subtitle) in enumerate(built):
        pl = x0+j*(side+gap)
        panels.append(panel(inch_rect(pl, bottom, side, side, W, H), (0, lim), (0, lim), items,
                            xlabel=f'mean |d|, {LABEL[base]} (sv)', ylabel='mean |d|, strip (sv)',
                            title=f'strip vs {LABEL[base]}\n{subtitle}', legend_at=legend_below(pl, bottom, W, H),
                            name=base))
    examples = (f" Rings: the example-strip fibers ({picks['median']['fiber']}, median; {picks['p90']['fiber']}, p90)."
                if picks else '')
    caption = (
        'Secondary set, per scored fiber: the mean |d| of the strip (y) against each baseline (x), in sv; below the '
        "diagonal the strip is nearer the segment than the baseline. A fiber's means are over each surface's own "
        "covered nodes on its scored rows (node-weighted over its strips, from per_strip), so its two values need not "
        f"come from identical nodes; a baseline is plotted for fibers with at least {mn} of its nodes. The black "
        "diamond and the paired difference are the report's, on the nodes covered by both: the diamond's bar is the "
        "fiber-bootstrap 95 % CI of the baseline's mean (the report has no CI for the strip on those nodes). "
        + '; '.join(cap) + '.' + examples)
    return {'size': (W, H), 'panels': panels, 'caption': caption, 'notes': notes, 'data': data,
            'title': 'PHerc0139 secondary set: per-fiber mean distance, strip against each baseline'}


# ---- figure 3: PHerc0332 crossings -------------------------------------------------------------------------------

def fig3_crossings(cr, C):
    res = dig(cr, 'crossings')
    if not isinstance(res, dict):
        return skip('the crossings report has no "crossings" object')
    rows = [r for r in res.get('per_crossing') or [] if isinstance(r, dict)
            and num(r.get('agreement_sv')) is not None and num(r.get('closest_approach_sv')) is not None]
    missing = [str(m) for m in res.get('missing') or []]
    if not rows:
        return skip(f"no crossing with an agreement in the report (planned {res.get('planned')}, missing "
                    f"{len(missing)}{'; ' + str(res['note']) if res.get('note') else ''})")
    tol = C['CROSS_TOL']

    def is_tight(r):  # crossing_agreement(): the tight subset is closest_approach_sv <= CROSS_TOL
        return num(r['closest_approach_sv']) <= tol
    tight = [r for r in rows if is_tight(r)]
    loose = [r for r in rows if not is_tight(r)]
    med, mci = num(res.get('median')), interval(dig(res, 'ci', 'median'))
    xmax = max([num(r['closest_approach_sv']) for r in rows]+[tol])*1.15
    ymax = max([num(r['agreement_sv']) for r in rows]+[2.]+([mci[1]] if mci else []))*1.15
    items = [span(0, tol, TEXT2, .07, label=f'tight subset region: closest approach \u2264 {tol:g} sv',
                  role='tight_region'),
             hline(2.0, role='2sv'), text(xmax*.99, 2.0+.01*ymax, '2 sv', ha='right', va='bottom', size=7)]
    if med is not None and mci is not None:
        items += [span(mci[0], mci[1], COLOR['strip'], .1, axis='y', z=.6, role='median_ci'),
                  hline(med, COLOR['strip'], 1., None, label=f'median of all crossings, {med:.2f} sv (band: 95 % CI)',
                        role='median')]
    if tight:
        items.append(points([num(r['closest_approach_sv']) for r in tight], [num(r['agreement_sv']) for r in tight],
                            COLOR['strip'], 'o', label=f'crossing in the tight subset ({len(tight)})', role='tight',
                            tags=[str(r.get('id')) for r in tight]))
    if loose:
        items.append(points([num(r['closest_approach_sv']) for r in loose], [num(r['agreement_sv']) for r in loose],
                            COLOR['strip'], 'o', filled=False, role='loose', tags=[str(r.get('id')) for r in loose],
                            label=f'crossing at more than {tol:g} sv ({len(loose)})'))
    for r in rows:  # crossing ids beside their points, on the side with room
        x, y = num(r['closest_approach_sv']), num(r['agreement_sv'])
        right = x > .75*xmax
        items.append(text(x+(-.02 if right else .02)*xmax, y, str(r.get('id')), size=6.3,
                          ha='right' if right else 'left', va='bottom' if y < .06*ymax else 'center', role='id'))
    W, pl, pw, ph = 9.2, .85, 5.3, 4.25
    bottom = .1+legend_height(legend_rows(items))+LEGEND_GAP
    H = bottom+ph+.83
    pnl = panel(inch_rect(pl, bottom, pw, ph, W, H), (-.03*xmax, xmax), (0, ymax), items,
                xlabel='closest approach of the H and the V human fiber (sv)',
                ylabel='agreement between the two strips (sv)',
                title=f'PHerc0332 H/V crossings \u2014 {NO_GT}', legend_at=legend_below(pl, bottom, W, H),
                name='crossings')
    one_dir = [str(r.get('id')) for r in rows if r.get('directions') == 1]
    t = dig(res, 'tight_subset') or {}
    ci = dig(res, 'ci') or {}
    summary = (f"{NO_GT}\n\n"
               f"{len(rows)} crossings with an agreement\n(planned {res.get('planned')}, missing {len(missing)})\n\n"
               f"all crossings (95 % CI, bootstrap\nover crossings):\n"
               f"  median {f2(med)} sv{ci_txt(mci)}\n"
               f"  mean {f2(num(res.get('mean')))} sv{ci_txt(interval(ci.get('mean')))}\n"
               f"  p90 {f2(num(res.get('p90')))} sv{ci_txt(interval(ci.get('p90')))}\n"
               f"  within 2 sv {pct(num(res.get('within2')))}{ci_txt(interval(ci.get('within2')), 0, True)}\n\n"
               f"tight subset ({t.get('crossings')} crossings, no CI):\n"
               f"  median {f2(num(t.get('median')))} sv, mean {f2(num(t.get('mean')))} sv\n"
               f"  within 2 sv {pct(num(t.get('within2')))}")
    if missing:
        summary += '\n\nmissing:\n  ' + wrap_ids(missing)
    texts = [text(6.45/W, (H-1.0)/H, summary, size=7.3, color=TEXT, va='top', coords='figure', role='summary')]
    caption = (
        f"PHerc0332, {NO_GT}: per planned crossing that has both strips, the agreement (sv): the median |d| from one "
        f"strip's nodes within {C['CROSS_RADIUS']:g} sv of the crossing point to the other strip's surface, averaged "
        f"over the two directions, against the closest approach of the two human fibers (sv). Filled: the tight subset "
        f"(closest approach \u2264 {tol:g} sv), reported apart (PREREG 7). Two strips can agree and both be "
        f"wrong, and at "
        f"3 to 6 sv the two fibers may lie on opposite faces of one sheet or on neighbouring sheets. All crossings: "
        f"median {f2(med)} sv{ci_txt(mci)}, mean {f2(num(res.get('mean')))} sv{ci_txt(interval(ci.get('mean')))}, "
        f"within 2 sv {pct(num(res.get('within2')))}; tight subset ({t.get('crossings')}): median "
        f"{f2(num(t.get('median')))} sv. "
        + (f"Missing (no agreement): {', '.join(missing)}. " if missing else '')
        + (f"One usable direction only: {', '.join(one_dir)}." if one_dir else ''))
    data = {'per_crossing': [{'id': r.get('id'), 'h_fiber': r.get('h_fiber'), 'v_fiber': r.get('v_fiber'),
                              'closest_approach_sv': num(r['closest_approach_sv']),
                              'agreement_sv': num(r['agreement_sv']), 'directions': r.get('directions'),
                              'tight': is_tight(r)} for r in rows],
            'missing': missing, 'tight_rule_sv': tol, 'label': res.get('label')}
    return {'size': (W, H), 'panels': [pnl], 'texts': texts, 'caption': caption, 'notes': [], 'data': data,
            'title': 'PHerc0332: H/V crossing agreement'}


# ---- figure 4: calibration of expected_error ---------------------------------------------------------------------

def cal_bins(cal):
    out = []
    for b in (cal.get('bins') or []) if isinstance(cal, dict) else []:
        if not isinstance(b, dict) or count(b.get('nodes')) <= 0:
            continue
        e, o = num(b.get('mean_expected')), num(b.get('mean_observed'))
        if e is None or o is None:
            continue
        out.append({'lo': num(b.get('lo')), 'hi': num(b.get('hi')), 'nodes': count(b.get('nodes')), 'mean_expected': e,
                    'mean_observed': o, 'median_observed': num(b.get('median_observed')),
                    'within2': num(b.get('within2'))})
    return out


def bin_name(b):
    lo = f2(b['lo']) if b['lo'] is not None else '?'
    return f"[{lo}, {f2(b['hi'])})" if b['hi'] is not None else f'\u2265 {lo}'


def bin_edges(cal):
    los = [num(b.get('lo')) for b in cal.get('bins') or [] if isinstance(b, dict)]
    los = [v for v in los if v is not None]
    return (', '.join(f'{v:g}' for v in los)+' sv and up') if los else 'as in the report'


def cal_summary(cal):
    return (f"{count(cal.get('nodes')):,} nodes, {cal.get('fibers')} fibers; weighted |gap| "
            f"{f2(num(cal.get('weighted_abs_gap_sv')))} sv; observed/expected "
            f"{f2(num(cal.get('observed_over_expected')))}; "
            f"Spearman \u03c1 node {f2(num(cal.get('spearman_node')))}, fiber {f2(num(cal.get('spearman_fiber')))}")


def bin_table(bins):
    return '\n'.join(f"  {bin_name(b)}: {b['nodes']:,}; {b['mean_expected']:.2f}, {b['mean_observed']:.2f}, "
                     f"{f2(b['median_observed'])}" for b in bins)


def primary_label(prim):
    """What the primary report says of itself: stretches scored, and G1 (PREREG 6: it fails, so no accuracy claim)."""
    claim = ('G1 failed: no accuracy claim' if dig(prim, 'G1', 'pass') is False
             else f"verdict: {dig(prim, 'G2', 'verdict')}")
    return f"{dig(prim, 'stretches_scored')} of {dig(prim, 'stretches_planned')} stretches scored; {claim}"


def fig4_calibration(sec, prim, C, overlay=True):
    cal = dig(sec, 'calibration')
    if not isinstance(cal, dict):
        return skip(f"the secondary report has no calibration (no fiber with expected_error on {C['MIN_NODES']} or "
                    'more scored nodes)')
    bins = cal_bins(cal)
    if not bins:
        return skip('the secondary report\'s calibration has no filled bin')
    notes = []
    pcal = dig(prim, 'calibration') if overlay else None
    pbins = cal_bins(pcal) if isinstance(pcal, dict) else []
    if not overlay:
        notes.append('The primary case studies are left out (--no-primary-overlay).')
    elif not pbins:
        notes.append('The primary case studies have no calibration in their report (or no report): secondary set '
                     'only.')
    vals = [b['mean_expected'] for b in bins+pbins]+[b['mean_observed'] for b in bins+pbins]
    vals += [b['median_observed'] for b in bins if b['median_observed'] is not None]
    lim = max(vals)*1.12 if max(vals) > 0 else 1.
    items = [line([0, lim], [0, lim], TEXT2, 1., label='observed = expected', z=1, role='diagonal'),
             points([b['mean_expected'] for b in bins], [b['mean_observed'] for b in bins], COLOR['strip'], 'o',
                    label='secondary set: mean observed |d|', role='bins_mean', tags=[bin_name(b) for b in bins])]
    med = [b for b in bins if b['median_observed'] is not None]
    if med:
        items.append(points([b['mean_expected'] for b in med], [b['median_observed'] for b in med], COLOR['strip'],
                            's', filled=False, size=6, label='secondary set: median observed |d|',
                            role='bins_median', tags=[bin_name(b) for b in med]))
    if pbins:
        items.append(points([b['mean_expected'] for b in pbins], [b['mean_observed'] for b in pbins],
                            COLOR['fiber_floor'], '^', filled=False, size=6.5, role='primary_mean',
                            tags=[bin_name(b) for b in pbins], label='primary case studies: mean observed |d|'))
    W, pl, side = 9.4, .85, 4.4
    bottom = .1+legend_height(legend_rows(items))+LEGEND_GAP
    H = bottom+side+.88
    pnl = panel(inch_rect(pl, bottom, side, side, W, H), (0, lim), (0, lim), items,
                xlabel='expected_error, bin mean (sv)', ylabel='observed |d| to the matched segment (sv)',
                title='bins of the predicted error (filled bins; table at the right)',
                legend_at=legend_below(pl, bottom, W, H), name='calibration')
    head = 'per bin of expected_error (sv): nodes;\nmean expected, mean and median\nobserved |d| (sv)\n'
    summary = 'secondary set:\n' + cal_summary(cal).replace('; ', '\n') + '\n\n' + head + bin_table(bins)
    plabel = primary_label(prim) if pbins else None
    if pbins:
        summary += ('\n\nprimary case studies\n(' + textwrap.fill(plabel, 40) + '):\n'
                    + cal_summary(pcal).replace('; ', '\n') + '\n' + bin_table(pbins))
    texts = [text(5.6/W, (H-1.0)/H, summary, size=7.3, color=TEXT, va='top', coords='figure', role='summary')]
    caption = (
        "Calibration of the model's expected_error (PREREG 5.7): the scored nodes binned by their predicted error (bin "
        f"edges {bin_edges(cal)}); per filled bin the mean predicted error (x) against the mean "
        "(filled) and median (hollow) observed |d| to the matched wrap segment (y), in sv; the table gives each bin's "
        "nodes and values. expected_error is the "
        "model's predicted error along its frame normal, learned against Paris 4 patches; the observed |d| is the "
        "normal distance to the PHerc0139 mesh, which carries the mesh's own disagreement with the human fibers (the "
        f"fiber floor). Secondary set: {cal_summary(cal)}. weighted |gap| is the scorer's weighted_abs_gap_sv: the "
        "node-weighted mean over bins of |bin mean observed \u2212 bin mean expected|, not a per-node mean."
        + (f" Primary case studies (hollow triangles; {plabel}): {cal_summary(pcal)}." if pbins else ''))
    data = {'secondary_bins': bins, 'primary_bins': pbins,
            'secondary_summary': {k: cal.get(k) for k in ('nodes', 'fibers', 'weighted_abs_gap_sv',
                                                           'observed_over_expected', 'spearman_node',
                                                           'spearman_fiber')}}
    return {'size': (W, H), 'panels': [pnl], 'texts': texts, 'caption': caption, 'notes': notes, 'data': data,
            'title': "Calibration of the model's expected_error"}


# ---- backend 1: matplotlib (PNG 150 dpi + SVG) -------------------------------------------------------------------

MPL_DASH = {None: 'solid', 'dash': (0, (4, 2.5)), 'dot': (0, (1, 2))}
MPL_RC = {'font.family': 'DejaVu Sans', 'font.size': 8., 'svg.hashsalt': 'make_figures', 'axes.unicode_minus': True}


def matplotlib_or_reason():
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        from matplotlib import pyplot  # noqa: F401
        return matplotlib, None
    except Exception as error:  # noqa: BLE001 - any import failure means: write the SVG directly
        return None, f'{type(error).__name__}: {error}'


def mpl_style():
    """The rc context the figures are drawn and saved in."""
    import matplotlib
    return matplotlib.rc_context(MPL_RC)


def _mpl_points(ax, it):
    c, z = it['color'], it['z']
    for ints, horizontal in ((it['xint'], True), (it['yint'], False)):
        for i, iv in enumerate(ints or []):
            if iv is None:
                continue
            xs, ys = (iv, [it['y'][i]]*2) if horizontal else ([it['x'][i]]*2, iv)
            ax.plot(xs, ys, color=c, lw=1.2, zorder=z-.1, solid_capstyle='butt')
            ax.plot(xs, ys, ls='none', marker='|' if horizontal else '_', ms=5, mew=1.2, color=c, zorder=z-.1)
    face, edge = (c, SURFACE) if it['filled'] else ('none', c)  # hollow: see-through, so overlaps stay visible
    ax.plot(it['x'], it['y'], ls='none', marker=it['marker'], ms=it['size'], mfc=face, mec=edge,
            mew=1.1 if it['filled'] else 1.4, label=it['label'] or '_nolegend_', zorder=z)


def _mpl_panel(ax, p):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(*p['xlim'])
    ax.set_ylim(*p['ylim'])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(.8)
    ax.tick_params(colors=AXIS, labelcolor=TEXT2, labelsize=7.5, length=3, width=.6)
    if p['xticks']:
        ax.set_xticks(p['xticks'][0], p['xticks'][1])
    if p['yticks']:
        ax.set_yticks(p['yticks'][0], p['yticks'][1])
    if p['grid']:
        ax.grid(True, axis=p['grid'], color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
    ax.set_xlabel(p['xlabel'], color=TEXT, fontsize=8.5)
    ax.set_ylabel(p['ylabel'], color=TEXT, fontsize=8.5)
    if p['title']:
        ax.set_title(p['title'], color=TEXT, fontsize=8.8, loc='left', pad=6, linespacing=1.4)
    for it in p['items']:
        k, lab = it['kind'], (it.get('label') or '_nolegend_')
        if k == 'points':
            _mpl_points(ax, it)
        elif k == 'line':
            ax.plot(it['x'], it['y'], color=it['color'], lw=it['width'], ls=MPL_DASH[it['dash']], label=lab,
                    zorder=it['z'], solid_joinstyle='round', solid_capstyle='round')
        elif k == 'vline':
            ax.axvline(it['x'], color=it['color'], lw=it['width'], ls=MPL_DASH[it['dash']], label=lab, zorder=it['z'])
        elif k == 'hline':
            ax.axhline(it['y'], color=it['color'], lw=it['width'], ls=MPL_DASH[it['dash']], label=lab, zorder=it['z'])
        elif k == 'segment':
            ax.plot([it['x0'], it['x1']], [it['y0'], it['y1']], color=it['color'], lw=it['width'],
                    ls=MPL_DASH[it['dash']], zorder=it['z'], solid_capstyle='butt')
        elif k == 'span':
            fn = ax.axvspan if it['axis'] == 'x' else ax.axhspan
            fn(it['a'], it['b'], color=it['color'], alpha=it['alpha'], lw=0, label=lab, zorder=it['z'])
        elif k == 'text':
            box = dict(boxstyle='round,pad=0.35', fc=SURFACE, ec=GRID, lw=.6) if it['box'] else None
            ax.text(it['x'], it['y'], it['s'], transform=ax.transAxes if it['coords'] == 'axes' else ax.transData,
                    ha=it['ha'], va=it['va'], fontsize=it['size'], color=it['color'], zorder=it['z'], bbox=box,
                    linespacing=1.3)
    if p.get('legend_at') and any(it.get('label') for it in p['items']):
        # hung from its top-left corner, given in figure fractions below the axes: outside every axes
        ax.legend(loc='upper left', bbox_to_anchor=tuple(p['legend_at']), bbox_transform=ax.figure.transFigure,
                  borderaxespad=0., fontsize=LEGEND_FS, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                  framealpha=.95, labelcolor=TEXT, borderpad=.5, handlelength=1.6)


def draw_mpl(model):
    """The matplotlib Figure of a figure model; call inside mpl_style(); the caller saves and closes it."""
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=model['size'], facecolor=SURFACE)
    try:
        for p in model['panels']:
            _mpl_panel(fig.add_axes(p['rect']), p)
        W, H = model['size']
        fig.text(.5, 1-.14/H, model['title'], ha='center', va='top', fontsize=10.5, color=TEXT, fontweight='bold')
        for t in model.get('texts', []):
            fig.text(t['x'], t['y'], t['s'], ha=t['ha'], va=t['va'], fontsize=t['size'], color=t['color'],
                     linespacing=1.35)
    except Exception:
        plt.close(fig)
        raise
    return fig


def render_mpl(model, png, svg, started):
    """PNG (150 dpi, unless png is None) and SVG; each path is appended to ``started`` before it is written."""
    from matplotlib import pyplot as plt
    with mpl_style():
        fig = draw_mpl(model)
        try:
            if png is not None:
                started.append(png)
                fig.savefig(png, dpi=150, facecolor=SURFACE)
            started.append(svg)
            fig.savefig(svg, format='svg', facecolor=SURFACE, metadata={'Date': None})
        finally:
            plt.close(fig)


# ---- backend 2: SVG written directly (no matplotlib; no PNG) ---------------------------------------------------------

PX = 96.          # SVG px per inch
PT = 96./72.      # SVG px per point
SVG_DASH = {None: '', 'dash': ' stroke-dasharray="5.3,3.3"', 'dot': ' stroke-dasharray="1.3,2.7"'}


def nice_ticks(lo, hi, n=5):
    span_ = hi-lo
    if not span_ > 0:
        return [lo]
    raw = span_/n
    mag = 10**math.floor(math.log10(raw))
    step = mag*10
    for m in (1, 2, 2.5, 5, 10):
        if m*mag >= raw:
            step = m*mag
            break
    t, out = math.ceil(lo/step-1e-9)*step, []
    while t <= hi+1e-9*step:
        out.append(0. if abs(t) < 1e-12 else round(t, 10))
        t += step
    return out


def tick_label(v):
    return f'{v:.4g}'


def _svg_text(x, y, s, size=7.5, color=TEXT2, ha='left', va='center', weight='normal', rotate=None):
    fs = size*PT
    lines = str(s).split('\n')
    lh = 1.3*fs
    block = lh*(len(lines)-1)
    first = {'top': y+.8*fs, 'center': y-block/2+.35*fs, 'bottom': y-block-.2*fs}.get(va, y)
    anchor = {'left': 'start', 'center': 'middle', 'right': 'end'}[ha]
    rot = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ''
    spans = ''.join(f'<tspan x="{x:.2f}" y="{first+i*lh:.2f}">{escape(ln)}</tspan>' for i, ln in enumerate(lines))
    return (f'<text font-size="{fs:.2f}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}"{rot}>'
            f'{spans}</text>')


def _svg_marker(cx, cy, marker, size, color, filled, cls=None):
    r = size*PT/2
    fill, stroke, sw = (color, SURFACE, 1.1*PT) if filled else ('none', color, 1.4*PT)
    style = (f'class="{cls}" ' if cls else '')+f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}"'
    if marker == 's':
        a = r*.9
        return f'<rect x="{cx-a:.2f}" y="{cy-a:.2f}" width="{2*a:.2f}" height="{2*a:.2f}" {style}/>'
    if marker in ('^', 'v', 'D'):
        if marker == 'D':
            pts = [(cx, cy-r*1.1), (cx+r*.8, cy), (cx, cy+r*1.1), (cx-r*.8, cy)]
        else:
            s = -1 if marker == '^' else 1
            pts = [(cx, cy+s*r*1.1), (cx+r, cy-s*r*.7), (cx-r, cy-s*r*.7)]
        return f'<polygon points="{" ".join(f"{a:.2f},{b:.2f}" for a, b in pts)}" {style}/>'
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" {style}/>'


def _svg_panel(p, idx, W, H):
    l, b, w, h = p['rect']
    X0, Y0, PW, PH = l*W, (1-b-h)*H, w*W, h*H
    (x0, x1), (y0, y1) = p['xlim'], p['ylim']

    def fx(x):
        return X0+(x-x0)/(x1-x0)*PW

    def fy(y):
        return Y0+PH-(y-y0)/(y1-y0)*PH
    g = [f'<clipPath id="clip{idx}"><rect x="{X0:.2f}" y="{Y0:.2f}" width="{PW:.2f}" height="{PH:.2f}"/></clipPath>']
    xt = p['xticks'] or (nice_ticks(x0, x1), None)
    yt = p['yticks'] or (nice_ticks(y0, y1), None)
    xlab = xt[1] or [tick_label(v) for v in xt[0]]
    ylab = yt[1] or [tick_label(v) for v in yt[0]]
    for v, s in zip(xt[0], xlab):
        if not x0-1e-9 <= v <= x1+1e-9:
            continue
        if p['grid'] in ('both', 'x'):
            g.append(f'<line x1="{fx(v):.2f}" y1="{Y0:.2f}" x2="{fx(v):.2f}" y2="{Y0+PH:.2f}" stroke="{GRID}" '
                     'stroke-width="0.8"/>')
        g.append(f'<line x1="{fx(v):.2f}" y1="{Y0+PH:.2f}" x2="{fx(v):.2f}" y2="{Y0+PH+4:.2f}" stroke="{AXIS}"/>')
        g.append(_svg_text(fx(v), Y0+PH+6, s, 7.5, TEXT2, 'center', 'top'))
    for v, s in zip(yt[0], ylab):
        if not y0-1e-9 <= v <= y1+1e-9:
            continue
        if p['grid'] in ('both', 'y'):
            g.append(f'<line x1="{X0:.2f}" y1="{fy(v):.2f}" x2="{X0+PW:.2f}" y2="{fy(v):.2f}" stroke="{GRID}" '
                     'stroke-width="0.8"/>')
        g.append(f'<line x1="{X0-4:.2f}" y1="{fy(v):.2f}" x2="{X0:.2f}" y2="{fy(v):.2f}" stroke="{AXIS}"/>')
        g.append(_svg_text(X0-6, fy(v), s, 7.5, TEXT2, 'right', 'center'))
    g.append(f'<polyline points="{X0:.2f},{Y0:.2f} {X0:.2f},{Y0+PH:.2f} {X0+PW:.2f},{Y0+PH:.2f}" fill="none" '
             f'stroke="{AXIS}" stroke-width="1.07"/>')
    if p['xlabel']:
        g.append(_svg_text(X0+PW/2, Y0+PH+22, p['xlabel'], 8.5, TEXT, 'center', 'top'))
    if p['ylabel']:  # left of the widest tick label
        reach = 10+max((len(s) for s in ylab), default=1)*7.5*PT*.56+8.5*PT*.6
        g.append(_svg_text(X0-reach, Y0+PH/2, p['ylabel'], 8.5, TEXT, 'center', 'center', rotate=-90))
    if p['title']:
        g.append(_svg_text(X0, Y0-8, p['title'], 8.8, TEXT, 'left', 'bottom'))
    body, labels, legend = [], [], []
    for it in sorted(p['items'], key=lambda it: it['z']):
        k = it['kind']
        if k == 'points':
            c = it['color']
            for ints, horizontal in ((it['xint'], True), (it['yint'], False)):
                for i, iv in enumerate(ints or []):
                    if iv is None:
                        continue
                    if horizontal:
                        ya, xa, xb = fy(it['y'][i]), fx(iv[0]), fx(iv[1])
                        body += [f'<line x1="{xa:.2f}" y1="{ya:.2f}" x2="{xb:.2f}" y2="{ya:.2f}" stroke="{c}" '
                                 'stroke-width="1.6"/>'] + [
                            f'<line x1="{xe:.2f}" y1="{ya-3.3:.2f}" x2="{xe:.2f}" y2="{ya+3.3:.2f}" stroke="{c}" '
                            'stroke-width="1.6"/>' for xe in (xa, xb)]
                    else:
                        xa, ya, yb = fx(it['x'][i]), fy(iv[0]), fy(iv[1])
                        body += [f'<line x1="{xa:.2f}" y1="{ya:.2f}" x2="{xa:.2f}" y2="{yb:.2f}" stroke="{c}" '
                                 'stroke-width="1.6"/>'] + [
                            f'<line x1="{xa-3.3:.2f}" y1="{ye:.2f}" x2="{xa+3.3:.2f}" y2="{ye:.2f}" stroke="{c}" '
                            'stroke-width="1.6"/>' for ye in (ya, yb)]
            for x, y in zip(it['x'], it['y']):
                body.append(_svg_marker(fx(x), fy(y), it['marker'], it['size'], c, it['filled'], 'pt'))
        elif k in ('line', 'segment', 'vline', 'hline'):
            if k == 'line':
                pts = [(fx(x), fy(y)) for x, y in zip(it['x'], it['y'])]
            elif k == 'segment':
                pts = [(fx(it['x0']), fy(it['y0'])), (fx(it['x1']), fy(it['y1']))]
            elif k == 'vline':
                pts = [(fx(it['x']), Y0), (fx(it['x']), Y0+PH)]
            else:
                pts = [(X0, fy(it['y'])), (X0+PW, fy(it['y']))]
            body.append(f'<polyline points="{" ".join(f"{a:.2f},{b:.2f}" for a, b in pts)}" fill="none" '
                        f'stroke="{it["color"]}" stroke-width="{it["width"]*PT:.2f}" stroke-linejoin="round"'
                        f'{SVG_DASH[it["dash"]]}/>')
        elif k == 'span':
            if it['axis'] == 'x':
                xa, xb = sorted((fx(it['a']), fx(it['b'])))
                body.append(f'<rect x="{xa:.2f}" y="{Y0:.2f}" width="{xb-xa:.2f}" height="{PH:.2f}" '
                            f'fill="{it["color"]}" fill-opacity="{it["alpha"]}"/>')
            else:
                ya, yb = sorted((fy(it['a']), fy(it['b'])))
                body.append(f'<rect x="{X0:.2f}" y="{ya:.2f}" width="{PW:.2f}" height="{yb-ya:.2f}" '
                            f'fill="{it["color"]}" fill-opacity="{it["alpha"]}"/>')
        elif k == 'text':
            x, y = (X0+it['x']*PW, Y0+(1-it['y'])*PH) if it['coords'] == 'axes' else (fx(it['x']), fy(it['y']))
            labels.append(_svg_text(x, y, it['s'], it['size'], it['color'], it['ha'], it['va']))
        if it.get('label'):
            legend.append(('marker' if k == 'points' else ('rect' if k == 'span' else 'line'), it))
    g.append(f'<g clip-path="url(#clip{idx})">' + ''.join(body) + '</g>')
    g += labels
    if p.get('legend_at') and legend:
        g += _svg_legend(legend, p['legend_at'][0]*W, (1-p['legend_at'][1])*H)
    return g


def _svg_legend(entries, x, y):
    """A legend hung from its top-left corner (x, y), in px."""
    fs, row = LEGEND_FS*PT, LEGEND_FS*PT*1.5
    width = 30+max(len(it['label']) for _, it in entries)*fs*.56
    height = row*len(entries)+8
    g = [f'<rect class="legend" x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="3" '
         f'fill="{SURFACE}" fill-opacity="0.95" stroke="{GRID}"/>']
    for i, (kind, it) in enumerate(entries):
        cy = y+4+row*(i+.5)
        if kind == 'marker':
            g.append(_svg_marker(x+14, cy, it['marker'], min(it['size'], 7), it['color'], it['filled']))
        elif kind == 'rect':
            g.append(f'<rect x="{x+6:.2f}" y="{cy-4:.2f}" width="16" height="8" fill="{it["color"]}" '
                     f'fill-opacity="{max(it["alpha"], .25)}"/>')
        else:
            g.append(f'<line x1="{x+5:.2f}" y1="{cy:.2f}" x2="{x+23:.2f}" y2="{cy:.2f}" stroke="{it["color"]}" '
                     f'stroke-width="{it["width"]*PT:.2f}"{SVG_DASH[it["dash"]]}/>')
        g.append(_svg_text(x+28, cy, it['label'], LEGEND_FS, TEXT, 'left', 'center'))
    return g


def render_svg(model, svg, started):
    W, H = model['size'][0]*PX, model['size'][1]*PX
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.2f} {H:.2f}" '
           'font-family="DejaVu Sans, Verdana, Arial, sans-serif">',
           f'<rect x="0" y="0" width="{W:.2f}" height="{H:.2f}" fill="{SURFACE}"/>']
    for i, p in enumerate(model['panels']):
        out += _svg_panel(p, i, W, H)
    out.append(_svg_text(W/2, .14*PX, model['title'], 10.5, TEXT, 'center', 'top', 'bold'))
    for t in model.get('texts', []):
        out.append(_svg_text(t['x']*W, (1-t['y'])*H, t['s'], t['size'], t['color'], t['ha'], t['va']))
    out.append('</svg>')
    started.append(Path(svg))
    Path(svg).write_text('\n'.join(out), encoding='utf-8')


# ---- main ----------------------------------------------------------------------------------------------------------

def _safe_stdout():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='backslashreplace')
        except (AttributeError, ValueError, OSError):
            pass


def read_reports(ws):
    """The three reports of the workspace (None where missing or unusable) and their status."""
    reports, status = {}, {}
    for key, rel in REPORTS.items():
        data, reason = read_json(Path(ws)/rel)
        reports[key] = data
        status[key] = {'path': rel, 'status': 'read' if data is not None else reason}
        if data is not None:
            status[key].update(sha256=sha256_file(Path(ws)/rel), analysis=data.get('analysis'),
                               scorer_version=dig(data, 'provenance', 'scorer_version'),
                               scorer_sha256=dig(data, 'provenance', 'scorer_sha256'),
                               prereg_sha256=dig(data, 'provenance', 'prereg_sha256'))
    return reports, status


def per_fiber(sec):
    """(fibers, consistency check, example picks) of the secondary report; empty for no report."""
    if sec is None:
        return {}, None, None
    fibers = fiber_table(dig(sec, 'per_strip'), constants(sec)['MIN_NODES'])
    return fibers, check_against_report(fibers, dig(sec, 'accuracy', 'all')), pick_examples(fibers)


def build_models(reports, status, fibers, picks, primary_overlay=True):
    sec, prim, cro = reports['secondary'], reports['primary'], reports['crossings_0332']
    c_sec, c_cro = constants(sec), constants(cro)

    def need(key):
        return skip(f"{key} report {status[key]['status']}")
    builders = {
        'fig1_secondary_distance': lambda: fig1_distance(sec, c_sec, fibers) if sec else need('secondary'),
        'fig2_secondary_per_fiber': lambda: fig2_per_fiber(sec, c_sec, fibers, picks) if sec else need('secondary'),
        'fig3_0332_crossings': lambda: fig3_crossings(cro, c_cro) if cro else need('crossings_0332'),
        'fig4_calibration': lambda: (fig4_calibration(sec, prim, c_sec, primary_overlay) if sec
                                     else need('secondary')),
    }
    models = {}
    for name, build in builders.items():
        try:
            models[name] = build()
        except Exception as error:  # noqa: BLE001 - a malformed report skips its figure, never stops the tool
            models[name] = skip(f'error while reading the report for this figure: {type(error).__name__}: {error}')
    return models


def main(argv=None):
    _safe_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workspace', required=True, type=Path,
                    help="the scorer's folder (score/), holding results/v3/ and the frozen run plans")
    ap.add_argument('--out', required=True, type=Path, help='output folder for the figures and captions.json')
    ap.add_argument('--dry-run', action='store_true', help='read, select and print; write nothing')
    ap.add_argument('--render-dir', type=Path,
                    help='where the printed example-strip commands copy and render a piece '
                         '(default: <workspace>/../renders/v3_secondary)')
    ap.add_argument('--backend', choices=('auto', 'svg'), default='auto',
                    help='auto: matplotlib when it imports (PNG + SVG), else SVG only; svg: SVG written directly')
    ap.add_argument('--no-primary-overlay', action='store_true',
                    help="figure 4: the secondary set's calibration only, without the primary case studies' bins "
                         '(2 stretches; G1 failed, no accuracy claim)')
    args = ap.parse_args(argv)
    # absolute, so the printed commands (HF_HOME, the villa worktree, ct_cache, the render copy) work from any folder
    ws, out = args.workspace.resolve(), args.out.resolve()
    render_dir = (args.render_dir or ws.parent/'renders'/'v3_secondary').resolve()
    print(f'[make_figures {VERSION}] workspace {ws}; out {out}' + ('; dry run: nothing is written' if args.dry_run
                                                                   else ''))
    reports, status = read_reports(ws)
    for key in REPORTS:
        print(f"  {key}: {status[key]['status']}")
    sec = reports['secondary']
    fibers, check, picks, examples = {}, None, None, skip('no secondary report')
    try:
        fibers, check, picks = per_fiber(sec)
        if check is not None and not check['ok']:
            print(f'  WARNING: per_strip does not reproduce the secondary aggregate: {check}')
        examples = example_strips(sec, fibers, picks, check, ws, render_dir, out)
    except Exception as error:  # noqa: BLE001
        examples = skip(f'error while reading per_strip: {type(error).__name__}: {error}')
    models = build_models(reports, status, fibers, picks, primary_overlay=not args.no_primary_overlay)

    mpl, reason = (None, 'not loaded for a dry run') if args.dry_run else (
        (None, '--backend svg') if args.backend == 'svg' else matplotlib_or_reason())
    if args.dry_run:
        backend = 'none (dry run)'
    elif mpl is not None:
        backend = f'matplotlib {mpl.__version__} (PNG 150 dpi + SVG)'
    else:
        backend = f'SVG written directly, no PNG (matplotlib not used: {reason})'
    print(f'  backend: {backend}')
    if not args.dry_run:
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(f'  cannot create {out}: {error}; nothing written')
            print_examples(examples)
            return 1
    entries = {}
    for name in FIGURES:
        m = models[name]
        png, svg = out/f'{name}.png', out/f'{name}.svg'
        started = []  # the files of this figure this run began to write, in order
        if 'skipped' not in m and not args.dry_run:
            try:
                if mpl is not None:
                    render_mpl(m, png, svg, started)
                else:
                    render_svg(m, svg, started)
            except Exception as error:  # noqa: BLE001
                m = models[name] = skip(f'drawing failed: {type(error).__name__}: {error}')
        stale = [p.name for p in (png, svg) if p not in started and p.exists()]
        if 'skipped' in m:
            partial = [p.name for p in started if p.exists()]
            entries[name] = {'skipped': m['skipped']}
            if partial:
                entries[name]['written_by_this_run_before_the_failure'] = partial
            if stale:
                entries[name]['stale_files_from_an_earlier_run'] = stale
            print(f"  SKIP {name}: {m['skipped']}"
                  + (f' (written by this run before the failure, may be incomplete: {", ".join(partial)})'
                     if partial else '')
                  + (f' (stale files left: {", ".join(stale)})' if stale else ''))
            continue
        files = [p.name for p in started]
        entries[name] = {'files': files, 'size_inches': list(m['size']), 'title': m['title'], 'caption': m['caption'],
                         'notes': m['notes'], 'data': m['data']}
        if stale and not args.dry_run:
            entries[name]['stale_files_from_an_earlier_run'] = stale
        print(f"  {'would write' if args.dry_run else 'wrote'} {name}"
              + ('' if args.dry_run else f": {', '.join(files)}")
              + (f' (stale file left from an earlier run: {", ".join(stale)})' if stale and not args.dry_run else ''))
        for note in m['notes']:
            print(f'    note: {note}')
        if args.dry_run:
            print(f"    caption: {m['caption']}")
    print_examples(examples)
    captions = {'tool': 'tools/make_figures.py', 'version': VERSION, 'tool_sha256': sha256_file(HERE),
                'generated_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), 'workspace': str(ws),
                'backend': backend, 'units': 'sv = level-2 voxels of the 2.4 um scans (9.596 um)', 'reports': status,
                'figures': entries, 'example_strips': examples}
    if not args.dry_run:
        try:
            (out/'captions.json').write_text(json.dumps(captions, indent=1, ensure_ascii=False), encoding='utf-8')
            print('  wrote captions.json')
        except (OSError, TypeError, ValueError) as error:
            print(f'  captions.json not written: {type(error).__name__}: {error}')
            return 1
    made = [n for n in FIGURES if 'skipped' not in entries[n]]
    print(f"[make_figures] {len(made)} of {len(FIGURES)} figures {'would be ' if args.dry_run else ''}made"
          + (f"; skipped: {', '.join(n for n in FIGURES if n not in made)}" if len(made) < len(FIGURES) else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
