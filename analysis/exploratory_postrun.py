#!/usr/bin/env python
"""Exploratory analyses of the finished v3 scored run: EXPLORATORY, NOT PRE-REGISTERED.

    python analysis/exploratory_postrun.py [--workspace C:/VesuviusTemp/strips/score] [--out PATH]

What this is. The scored run of score/PREREG.md v3.2 finished on 29 Oct 2026 (02:13 to 03:40 UTC, rc 0). The five
analyses below were decided on 29 Oct 2026, AFTER its three reports had been seen. They are descriptive only. They
change no pre-registered number, gate or verdict: the G2 gate of PREREG 5.3 and 6 is node-weighted (nodes pooled
over fibers) and stands as reported. Every number this script writes carries the label
'exploratory, not pre-registered'.

  E1  per fiber (secondary set): node-weighted mean |d| of the strip per fiber, the fiber floor, their difference;
  E2  weighting: H and V shares of nodes and fibers; the pooled mean with equal weight per fiber;
  E3  per strip: the frame-plane and Lasagna-plane means against the strip's;
  E4  PHerc0332: agreement against the fibers' closest approach; why X0019, X0021 and X0027 have no value
      (PREREG 7's 5-node rule checked against the strips);
  E5  P0005: why no strip row lies inside its matched stretch (PREREG 4.6, the 75 sv end trim).

Inputs (read only; nothing under --workspace is written):
  results/v3/primary_case_studies.json, secondary_report.json, crossings_0332_report.json   the three reports
  results/matched_stretches.json, matched_stretches_secondary_run40.json, crossings_0332_plan.json   the frozen plans
  results/runplan_0139_primary.tsv, runplan_0139_secondary_run40.tsv, runplan_0332_crossings.tsv   their --arc
      starts (E5: how much the kink-repaired arclength is shorter before a piece's first head)
  runs/v3/strips/0332/C*/strip (+ strip.npz, summary.json)      the 25 PHerc0332 strips (E4)
  runs/v3/strips/secondary/P0005/strip (+ strip.npz)            the one skipped secondary strip (E5)
  data/fibers0332/<fiber>.json    the 24 fibers of the 15 run crossings (E4: strip association, arclengths)
  data/fibers0139/dj_20260904T160405930_000005.json             P0005's fiber (E5)
E1 to E3 use the reports only.

Standard library + numpy only: no scipy, no tifffile, no villa, and score/score_strips.py is not imported. The few
scorer rules needed for E4 and E5 are re-implemented here from score_strips.py (sha256 d7d6d194...; read for
this purpose): the tifxyz reading (a node is valid when x, y, z are finite and none is -1), the fiber's controlled
span and its 1 sv resampling (parse_fiber, resample), row centres (row_centres), strip-to-fiber association
(associate: the median distance of the row centres to the fiber), the crossing rule (crossing_agreement:
near_strip and the two 5-node rules) and the projection onto a strip surface (GridSurface.project: Gauss-Newton
on the bilinear grid, covered when the sideways residual is <= 1 sv and |d| <= 30 sv). Nearest-neighbour queries
are brute force instead of cKDTree. A compact TIFF reader (classic little/big-endian TIFF, LZW or none, predictor
1, strips or tiles) replaces tifffile. The checks block of the output shows that these re-implementations
reproduce the reports: the 12 PHerc0332 agreements, the stretch and crossing geometry of the plans, the per_strip
totals.

Seeds (exploratory only): 29102026 for the bootstrap of the equal-weight mean (E2) and 29102027 for the
permutation p of the rank correlations (E4). The pre-registered seeds 20261028 to 20261030 are not reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import sys
import time
import warnings
from pathlib import Path

import numpy as np

LABEL = 'exploratory, not pre-registered'
WHEN = ('decided on 29 Oct 2026, after the scored run (29 Oct 2026 02:13-03:40 UTC) and its three reports had been '
        'seen; descriptive only; no pre-registered number, gate or verdict changes')

# Pre-registered constants used here (score_strips.py / PREREG.md v3.2); they are read, not changed.
SCORE_UNIT_BASE = 4.0      # base voxels per sv
SAMPLE_STEP = 1.0          # sv between fiber samples
SEARCH_REACH = 30.0        # sv
TANGENT_TOL = 1.0          # sv
ASSOC_MAX_SV = 5.0         # sv
ROW_MAX_SV = 30.0          # sv
CROSS_RADIUS = 30.0        # sv (PREREG 7)
CROSS_MIN_NODES = 5        # PREREG 7: a direction needs >= 5 near nodes and >= 5 of them covered
CROSS_TOL = 3.0            # sv: the tight subset
TRIM_SV = 75.0             # infer.py --trim (patch voxels = sv)
G2_MAX_MEAN = 1.5          # sv
SEED_BOOT = 29102026       # exploratory only
SEED_PERM = 29102027       # exploratory only
BOOT_B = 2000
PERM_B = 200000
P0005_FIBER = 'dj_20260904T160405930_000005'


# ---- small helpers -------------------------------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _holds_num(v):
    if _is_num(v):
        return True
    if isinstance(v, (list, tuple)):
        return any(_holds_num(x) for x in v if not isinstance(x, dict))
    return False


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj


def labelled(obj):
    """Every dict that holds a number (directly or in a list of numbers) gets 'label' as its first key."""
    if isinstance(obj, dict):
        inner = {k: labelled(v) for k, v in obj.items() if k != 'label'}
        if any(_holds_num(v) for v in inner.values()) or 'label' in obj:
            return {'label': LABEL, **inner}
        return inner
    if isinstance(obj, list):
        return [labelled(v) for v in obj]
    return obj


def r(x, nd=6):
    return None if x is None or not math.isfinite(float(x)) else round(float(x), nd)


def stats_over(values):
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], np.float64)
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()),
                max=float(v.max()), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)))


def rankdata(x):
    """Ranks 1..n, ties get their average rank."""
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j+1 < len(x) and sx[j+1] == sx[i]:
            j += 1
        ranks[order[i:j+1]] = (i+j)/2.+1.
        i = j+1
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau_b(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    conc = disc = tx = ty = 0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            dx, dy = np.sign(x[i]-x[j]), np.sign(y[i]-y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc+disc+tx)*(conc+disc+ty))
    return float((conc-disc)/den) if den else float('nan')


def permutation_p(x, y, B, seed, chunk=20000):
    """Two-sided Monte Carlo permutation p of Spearman's rho and Kendall's tau (y permuted against x; tau's p only
    without ties, where tau-b = (C - D) / (n (n - 1) / 2))."""
    rng = np.random.default_rng(seed)
    xv, yv = np.asarray(x, np.float64), np.asarray(y, np.float64)
    n = len(xv)
    rx, ry = rankdata(xv), rankdata(yv)
    rho0, tau0 = spearman(xv, yv), kendall_tau_b(xv, yv)
    a = rx-rx.mean()
    iu, ju = np.triu_indices(n, 1)
    sx = np.sign(xv[iu]-xv[ju])
    off = ~np.eye(n, dtype=bool)
    ties = bool(np.any(sx == 0) or (yv[:, None] == yv[None, :])[off].any())
    hit_rho = hit_tau = done = 0
    while done < B:
        m = min(chunk, B-done)
        perms = np.argsort(rng.random((m, n)), axis=1)
        pr = ry[perms]
        pc = pr-pr.mean(1, keepdims=True)
        rho = (pc @ a)/np.sqrt((pc**2).sum(1)*(a**2).sum())
        hit_rho += int((np.abs(rho) >= abs(rho0)-1e-12).sum())
        if not ties:
            yy = yv[perms]
            tau = (np.sign(yy[:, iu]-yy[:, ju])*sx).sum(1)/len(iu)
            hit_tau += int((np.abs(tau) >= abs(tau0)-1e-12).sum())
        done += m
    return (1+hit_rho)/(B+1), (None if ties else (1+hit_tau)/(B+1))


# ---- TIFF (stdlib + numpy): classic TIFF, LZW or no compression, predictor 1, strips or tiles --------------------

_TIFF_TYPES = {1: ('B', 1), 2: ('s', 1), 3: ('H', 2), 4: ('I', 4), 5: ('I', 8), 6: ('b', 1), 7: ('B', 1),
               8: ('h', 2), 9: ('i', 4), 10: ('i', 8), 11: ('f', 4), 12: ('d', 8), 16: ('Q', 8), 17: ('q', 8)}


def lzw_decode(data):
    """TIFF LZW (MSB-first codes of 9 to 12 bits, Clear 256, EOI 257, the code width growing one code early, as
    libtiff)."""
    out = bytearray()
    buf = bytes(data)+b'\x00\x00\x00'
    total = len(data)*8
    table = [bytes((i,)) for i in range(256)]+[b'', b'']
    nbits, bitpos, prev = 9, 0, None
    while bitpos+nbits <= total:
        byte = bitpos >> 3
        word = (buf[byte] << 16) | (buf[byte+1] << 8) | buf[byte+2]
        code = (word >> (24-nbits-(bitpos & 7))) & ((1 << nbits)-1)
        bitpos += nbits
        if code == 256:
            del table[258:]
            nbits, prev = 9, None
            continue
        if code == 257:
            break
        if prev is None:
            entry = table[code]
            out += entry
            prev = entry
            continue
        if code < len(table):
            entry = table[code]
            table.append(prev+entry[:1])
        elif code == len(table):
            entry = prev+prev[:1]
            table.append(entry)
        else:
            raise ValueError(f'LZW: code {code} beyond the table ({len(table)})')
        out += entry
        prev = entry
        if len(table) >= (1 << nbits)-1 and nbits < 12:
            nbits += 1
    return bytes(out)


def read_tiff(path):
    """The first image of a classic TIFF as a numpy array (one sample per pixel)."""
    buf = Path(path).read_bytes()
    if buf[:2] == b'II':
        e = '<'
    elif buf[:2] == b'MM':
        e = '>'
    else:
        raise ValueError(f'{path}: not a TIFF')
    if struct.unpack(e+'H', buf[2:4])[0] != 42:
        raise ValueError(f'{path}: BigTIFF or another TIFF variant: not supported here')
    ifd = struct.unpack(e+'I', buf[4:8])[0]
    n = struct.unpack(e+'H', buf[ifd:ifd+2])[0]
    tags = {}
    for k in range(n):
        off = ifd+2+12*k
        tag, typ, cnt = struct.unpack(e+'HHI', buf[off:off+8])
        if typ not in _TIFF_TYPES:
            continue
        fmt, size = _TIFF_TYPES[typ]
        nbytes = size*cnt
        raw = buf[off+8:off+8+nbytes] if nbytes <= 4 else \
            buf[struct.unpack(e+'I', buf[off+8:off+12])[0]:][:nbytes]
        if typ == 2:
            tags[tag] = raw.split(b'\x00')[0].decode('latin-1')
        elif typ in (5, 10):
            v = struct.unpack(e+fmt*(2*cnt), raw)
            tags[tag] = [v[i]/v[i+1] if v[i+1] else math.nan for i in range(0, 2*cnt, 2)]
        else:
            tags[tag] = list(struct.unpack(e+fmt*cnt, raw))
    W, H = int(tags[256][0]), int(tags[257][0])
    bps = int(tags.get(258, [1])[0])
    comp = int(tags.get(259, [1])[0])
    spp = int(tags.get(277, [1])[0])
    pred = int(tags.get(317, [1])[0])
    sfmt = int(tags.get(339, [1])[0])
    if spp != 1 or pred != 1 or comp not in (1, 5) or bps % 8:
        raise ValueError(f'{path}: unsupported TIFF (samples {spp}, predictor {pred}, compression {comp}, '
                         f'bits {bps})')
    kind = {1: 'u', 2: 'i', 3: 'f'}[sfmt]
    dtype = np.dtype(f'{e}{kind}{bps//8}')
    out = np.zeros((H, W), dtype)

    def decode(chunk, count):
        data = chunk if comp == 1 else lzw_decode(chunk)
        need = count*dtype.itemsize
        if len(data) < need:
            data = data+b'\x00'*(need-len(data))
        return np.frombuffer(data[:need], dtype)

    if 322 in tags:
        tw, tl = int(tags[322][0]), int(tags[323][0])
        across = -(-W//tw)
        for i, (o, c) in enumerate(zip(tags[324], tags[325])):
            ty, tx = divmod(i, across)
            tile = decode(buf[o:o+c], tw*tl).reshape(tl, tw)
            r0, c0 = ty*tl, tx*tw
            h, w = min(tl, H-r0), min(tw, W-c0)
            out[r0:r0+h, c0:c0+w] = tile[:h, :w]
    else:
        rps = int(tags.get(278, [H])[0])
        for i, (o, c) in enumerate(zip(tags[273], tags[279])):
            r0 = i*rps
            h = min(rps, H-r0)
            out[r0:r0+h] = decode(buf[o:o+c], h*W).reshape(h, W)
    return out.astype(dtype.newbyteorder('='))


# ---- strips --------------------------------------------------------------------------------------------------------

class StripData:
    """One infer.py output (OUT/<piece id>): the tifxyz rows (sv, NaN off the strip), validity, mask, trace and
    the decision layout of strip.npz (arcs, f), as the scorer's load_strip reads them."""

    def __init__(self, strip_dir):
        d = Path(strip_dir)
        self.dir = d
        self.name = d.name if d.name != 'strip' else d.parent.name
        xyz = [read_tiff(d/f'{c}.tif').astype(np.float64) for c in 'xyz']
        grid = np.stack(xyz, -1)
        valid = np.isfinite(grid).all(-1) & ~(grid == -1).any(-1)
        grid *= 4.0/SCORE_UNIT_BASE  # strip unit base 4 (patch voxels = sv)
        grid[~valid] = np.nan
        self.mask = read_tiff(d/'mask.tif') > 0 if (d/'mask.tif').exists() else None
        trace = {}
        for cand in (d/'trace.json', d.parent/'summary.json'):
            if cand.exists():
                trace = {**json.loads(cand.read_text()), **trace}
        if trace.get('fiber_along') == 'columns':
            grid, valid = np.swapaxes(grid, 0, 1), valid.T
            self.mask = None if self.mask is None else self.mask.T
        self.rows, self.valid, self.trace = grid, valid, trace
        self.R, self.C = valid.shape
        self.files = {p.name: sha256_file(p) for p in sorted(d.glob('*')) if p.is_file()}
        npz = d.parent/'strip.npz'
        self.layout = None
        if npz.exists():
            self.files['../strip.npz'] = sha256_file(npz)
            with np.load(npz) as z:
                arcs, f = z['arcs'].astype(np.float64), z['f'].astype(np.float64)
                tpp = float(z['trace_per_patch'])
            blend = float(trace.get('blend', 16.))
            df = f[1]-f[0]
            ridx = np.rint((arcs[:, None]-arcs[0]+f[None]-f[0])/df).astype(np.int64)
            window = np.where(np.abs(f) < blend, np.cos(np.pi*f/(2*blend))**2, 0.)
            weight = np.bincount(ridx.ravel(), weights=np.broadcast_to(window, ridx.shape).ravel(),
                                 minlength=self.R)
            self.layout = dict(heads_sv=arcs/tpp, row_arc_sv=(arcs[0]+f[0]+np.arange(self.R)*df)/tpp,
                               row_step_sv=df/tpp, support=weight[:self.R] > 0, rows_from_npz=int(ridx.max())+1,
                               blend_sv=blend/tpp)

    def row_centres(self):
        """score_strips.row_centres: the centre column's node, else the mean of the row's valid nodes."""
        c = self.C//2
        centre = self.rows[:, c].copy()
        bad = ~self.valid[:, c]
        if bad.any():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                centre[bad] = np.nanmean(np.where(self.valid[bad][..., None], self.rows[bad], np.nan), axis=1)
        return centre

    def row_classes(self, trim_sv=TRIM_SV):
        """Every row: valid (all its nodes valid), partial (some), or invalid; invalid rows split into 'no decision'
        (stitch weight 0), 'start trim' (decisions, piece arclength < trim) and 'end trim' (decisions, beyond the
        last valid row: piece arclength > L - trim)."""
        lay = self.layout
        nvalid = self.valid.sum(1)
        full, part, none = nvalid == self.C, (nvalid > 0) & (nvalid < self.C), nvalid == 0
        out = dict(rows=self.R, columns=self.C, valid_nodes=int(self.valid.sum()), rows_all_valid=int(full.sum()),
                   rows_partly_valid=int(part.sum()), rows_without_valid_node=int(none.sum()),
                   mask_equals_xyz_validity=None if self.mask is None else bool(np.array_equal(self.mask,
                                                                                               self.valid)))
        if lay is None:
            return out
        arc = lay['row_arc_sv']
        sup = lay['support']
        vrows = np.flatnonzero(nvalid > 0)
        last_valid = int(vrows.max()) if len(vrows) else -1
        no_dec = none & ~sup
        start_trim = none & sup & (arc < trim_sv)
        end_trim = none & sup & (arc >= trim_sv) & (np.arange(self.R) > last_valid)
        other = none & sup & ~start_trim & ~end_trim

        def span(sel):
            idx = np.flatnonzero(sel)
            return None if not len(idx) else [float(arc[idx[0]]), float(arc[idx[-1]])]
        out.update(rows_from_strip_npz=lay['rows_from_npz'], row_step_sv=lay['row_step_sv'],
                   decision_heads_piece_arc_sv=[float(lay['heads_sv'][0]), float(lay['heads_sv'][-1])],
                   decisions=int(len(lay['heads_sv'])),
                   rows_with_decisions=int(sup.sum()), rows_with_decisions_piece_arc_sv=span(sup),
                   valid_rows_piece_arc_sv=span(nvalid > 0),
                   rows_dropped_no_decision=int(no_dec.sum()),
                   rows_dropped_start_trim=int(start_trim.sum()), start_trim_rows_piece_arc_sv=span(start_trim),
                   rows_dropped_end_trim=int(end_trim.sum()), end_trim_rows_piece_arc_sv=span(end_trim),
                   rows_dropped_other=int(other.sum()),
                   valid_rows_without_decisions=int(((nvalid > 0) & ~sup).sum()))
        if end_trim.any() and last_valid >= 0:
            out['piece_repaired_length_sv_bracket'] = [float(arc[last_valid]+trim_sv),
                                                       float(arc[np.flatnonzero(end_trim)[0]]+trim_sv)]
        elif last_valid >= 0:
            out['piece_repaired_length_sv_at_least'] = float(arc[last_valid]+trim_sv)
        return out


def discover_strips(root):
    """score_strips.discover_strips: every directory holding an x.tif, sorted, except patch_sheet."""
    dirs = sorted({p.parent for p in Path(root).rglob('x.tif')})
    return [d for d in dirs if d.name != 'patch_sheet']


# ---- fibers (score_strips.parse_fiber, controlled span, 1 sv samples) -------------------------------------------------

def _xyz(p):
    if isinstance(p, dict):
        p = p.get('position', p.get('xyz'))
    return [float(p[0]), float(p[1]), float(p[2])]


def arclength(points):
    if len(points) < 2:
        return np.zeros(len(points))
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def resample(points, step):
    s = arclength(points)
    if s[-1] <= 0:
        return points[:1].copy(), np.zeros(1)
    t = np.arange(0., s[-1]+1e-9, step)
    if s[-1]-t[-1] > 1e-6:
        t = np.r_[t, s[-1]]
    keep = np.r_[True, np.diff(s) > 1e-12]
    return np.stack([np.interp(t, s[keep], points[keep, i]) for i in range(3)], 1), t


def nearest(q, pts, budget=1_500_000):
    """Index and distance of the nearest point of pts (N, 3) for each q (M, 3); brute force (ties: first index)."""
    q, pts = np.asarray(q, np.float64).reshape(-1, 3), np.asarray(pts, np.float64).reshape(-1, 3)
    idx, dist = np.zeros(len(q), np.int64), np.full(len(q), np.inf)
    if not len(q) or not len(pts):
        return idx, dist
    step = max(1, budget//len(pts))
    for a in range(0, len(q), step):
        d2 = ((q[a:a+step, None, :]-pts[None, :, :])**2).sum(-1)
        j = d2.argmin(1)
        idx[a:a+step] = j
        dist[a:a+step] = np.sqrt(d2[np.arange(len(j)), j])
    return idx, dist


class FiberData:
    def __init__(self, path):
        raw = json.loads(Path(path).read_text())
        self.name, self.file, self.sha256 = Path(path).stem, Path(path).name, sha256_file(path)
        line = np.asarray([_xyz(p) for p in raw.get('line_points', [])], np.float64)*1.0/SCORE_UNIT_BASE
        ctrl = np.asarray([_xyz(p) for p in raw.get('control_points', [])], np.float64)*1.0/SCORE_UNIT_BASE
        if len(line) < 2 or len(ctrl) < 2:
            raise ValueError(f'{path}: no controlled span')
        anchors, _ = nearest(ctrl, line)
        a, b = int(anchors.min()), int(anchors.max())
        if b <= a:
            raise ValueError(f'{path}: no controlled span')
        self.line_offset, self.n_line = a, int(len(line))
        self.points, self.s = resample(line[a:b+1], SAMPLE_STEP)
        self.length = float(self.s[-1])
        hv = raw.get('hv_classification') or {}
        manual, auto = (hv.get('manual_tag') or ''), (hv.get('automatic_tag') or '')
        self.family = manual if manual in ('H', 'V') else (auto if auto in ('H', 'V') else '?')

    def sample_at(self, s):
        i = int(np.argmin(np.abs(self.s-s)))
        return i, float(abs(self.s[i]-s))


def associate(strip, fibers, only=None):
    """score_strips.associate: the fiber whose samples the row centres run closest to (median of the nearest
    distances); candidates are the fibers owning the nearest sample of some row centre; None beyond 5 sv."""
    centre = strip.row_centres()
    ok = np.isfinite(centre).all(1)
    if not ok.any():
        return None, math.inf
    if only is not None:
        cands = [only]
    else:
        allpts = np.concatenate([f.points for f in fibers])
        owner = np.concatenate([np.full(len(f.points), k) for k, f in enumerate(fibers)])
        idx, _ = nearest(centre[ok], allpts)
        cands = [int(k) for k in np.unique(owner[idx])]
    best, best_med = None, math.inf
    for k in cands:
        _, dk = nearest(centre[ok], fibers[k].points)
        med = float(np.median(dk))
        if med < best_med:
            best, best_med = k, med
    return (None, best_med) if best_med > ASSOC_MAX_SV else (best, best_med)


# ---- surface projection (port of score_strips.GridSurface.project; brute-force k nearest nodes) ---------------------

class Surface:
    def __init__(self, grid, valid=None):
        g = np.asarray(grid, np.float64)
        ok = np.isfinite(g).all(-1)
        valid = ok if valid is None else (np.asarray(valid, bool) & ok)
        self.R, self.C = g.shape[:2]
        self.grid = np.where(valid[..., None], g, np.nan)
        if self.R > 1 and self.C > 1:
            self.cell_ok = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
        else:
            self.cell_ok = np.zeros((max(self.R-1, 0), max(self.C-1, 0)), bool)
        corner = np.zeros_like(valid)
        if self.cell_ok.size:
            for dr in (0, 1):
                for dc in (0, 1):
                    corner[dr:self.R-1+dr, dc:self.C-1+dc] |= self.cell_ok
        rr, cc = np.nonzero(corner)
        self.node_rc = np.stack([rr, cc], 1)
        self.nodes = self.grid[rr, cc]

    def _ok(self, i, j):
        inside = (i >= 0) & (j >= 0) & (i < self.R-1) & (j < self.C-1)
        out = np.zeros(i.shape, bool)
        out[inside] = self.cell_ok[i[inside], j[inside]]
        return out

    def _eval(self, i, j, a, b):
        g = self.grid
        p00, p10, p01, p11 = g[i, j], g[i+1, j], g[i, j+1], g[i+1, j+1]
        a, b = a[:, None], b[:, None]
        s = (1-a)*(1-b)*p00+a*(1-b)*p10+(1-a)*b*p01+a*b*p11
        sr = (1-b)*(p10-p00)+b*(p11-p01)
        sc = (1-a)*(p01-p00)+a*(p11-p10)
        return s, sr, sc

    def _knn(self, p, k, bound):
        m, n = len(p), len(self.nodes)
        dist, idx = np.full((m, k), np.inf), np.full((m, k), n, np.int64)
        step = max(1, 1_000_000//max(n, 1))
        for a in range(0, m, step):
            q = p[a:a+step]
            d2 = ((q[:, None, :]-self.nodes[None, :, :])**2).sum(-1)
            part = np.argpartition(d2, k-1, axis=1)[:, :k] if n > k else np.tile(np.arange(n), (len(q), 1))
            dd = np.take_along_axis(d2, part, 1)
            order = np.argsort(dd, axis=1, kind='stable')
            part, dd = np.take_along_axis(part, order, 1), np.take_along_axis(dd, order, 1)
            within = dd < bound*bound
            dist[a:a+len(q)] = np.where(within, np.sqrt(dd), np.inf)
            idx[a:a+len(q)] = np.where(within, part, n)
        return dist, idx

    def project(self, points, k=4, reach=SEARCH_REACH, iters=8):
        """Signed normal distance (M,) and covered (M,) of points (M, 3) sv (no umbilicus: the sign is the grid's)."""
        p = np.asarray(points, np.float64).reshape(-1, 3)
        m = len(p)
        d_out, cov_out = np.full(m, np.nan), np.zeros(m, bool)
        if m == 0 or not len(self.nodes) or not self.cell_ok.any():
            return d_out, cov_out
        k = min(k, len(self.nodes))
        dist, idx = self._knn(p, k, reach+2*TANGENT_TOL+8.)
        has = np.isfinite(dist)
        pi, ki = np.nonzero(has)
        if not len(pi):
            return d_out, cov_out
        q = p[pi]
        rc = self.node_rc[idx[pi, ki]]
        r0, c0 = rc[:, 0], rc[:, 1]
        i, j = r0.copy(), c0.copy()
        chosen = np.zeros(len(q), bool)
        for di, dj in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
            ii, jj = r0+di, c0+dj
            ok = ~chosen & self._ok(ii, jj)
            i[ok], j[ok], chosen[ok] = ii[ok], jj[ok], True
        a = (r0-i).astype(np.float64)
        b = (c0-j).astype(np.float64)
        q, pi, i, j, a, b = q[chosen], pi[chosen], i[chosen], j[chosen], a[chosen], b[chosen]
        for _ in range(iters):
            s, sr, sc = self._eval(i, j, a, b)
            e = q-s
            a11, a12, a22 = (sr*sr).sum(1), (sr*sc).sum(1), (sc*sc).sum(1)
            b1, b2 = (sr*e).sum(1), (sc*e).sum(1)
            det = a11*a22-a12*a12
            det = np.where(np.abs(det) < 1e-12, np.nan, det)
            da = (a22*b1-a12*b2)/det
            db = (a11*b2-a12*b1)/det
            da, db = np.nan_to_num(da), np.nan_to_num(db)
            step = np.maximum(np.maximum(np.abs(da), np.abs(db))/2., 1.)
            da, db = da/step, db/step
            rr = np.clip(i+a+da, 0., self.R-1.)
            cc = np.clip(j+b+db, 0., self.C-1.)
            ni = np.clip(np.floor(rr).astype(np.int64), 0, self.R-2)
            nj = np.clip(np.floor(cc).astype(np.int64), 0, self.C-2)
            move = self._ok(ni, nj)
            a = np.where(move, rr-ni, np.clip(a+da, 0., 1.))
            b = np.where(move, cc-nj, np.clip(b+db, 0., 1.))
            i, j = np.where(move, ni, i), np.where(move, nj, j)
        s, sr, sc = self._eval(i, j, a, b)
        nrm = np.cross(sc, sr)
        nrm = nrm/np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
        e = q-s
        d = (e*nrm).sum(1)
        tang = np.linalg.norm(e-d[:, None]*nrm, axis=1)
        cov = np.isfinite(d) & (tang <= TANGENT_TOL) & (np.abs(d) <= reach)
        score = np.where(cov, np.linalg.norm(e, axis=1), np.inf)
        order = np.lexsort((score, pi))
        first = np.r_[True, pi[order][1:] != pi[order][:-1]]
        best = order[first]
        best = best[np.isfinite(score[best])]
        d_out[pi[best]], cov_out[pi[best]] = d[best], True
        return d_out, cov_out


# ---- E1 to E3: the reports' per_strip entries ---------------------------------------------------------------------------

def per_fiber(per_strip):
    """Fibers from per_strip: node-weighted (strip_nodes) mean |d|, signed mean and fiber floor."""
    groups = {}
    for e in per_strip:
        groups.setdefault(e['fiber'], []).append(e)
    out = []
    for name in sorted(groups):
        es = groups[name]
        n = np.array([e['strip_nodes'] for e in es], np.float64)
        m = np.array([e['strip_mean_abs_sv'] for e in es], np.float64)
        sg = np.array([e['strip_signed_mean_sv'] for e in es], np.float64)
        fl = [(e['strip_nodes'], e['fiber_floor_mean_abs_sv']) for e in es if e['fiber_floor_mean_abs_sv'] is not None]
        floor = (sum(a*b for a, b in fl)/sum(a for a, _ in fl)) if fl else None
        mean = float((n*m).sum()/n.sum())
        out.append(dict(fiber=name, family=es[0]['family'], strips=[e['strip'] for e in es],
                        stretches=int(sum(len(e['stretches']) for e in es)), nodes=int(n.sum()),
                        strip_mean_abs_sv=mean, strip_signed_mean_sv=float((n*sg).sum()/n.sum()),
                        fiber_floor_mean_abs_sv=floor,
                        strip_minus_fiber_floor_sv=None if floor is None else mean-floor,
                        at_or_below_1p5_sv=bool(mean <= G2_MAX_MEAN)))
    return out


def fiber_summary(fibers):
    means = [f['strip_mean_abs_sv'] for f in fibers]
    diffs = [f['strip_minus_fiber_floor_sv'] for f in fibers if f['strip_minus_fiber_floor_sv'] is not None]
    return dict(fibers=len(fibers), nodes=int(sum(f['nodes'] for f in fibers)),
                strip_mean_abs_sv_over_fibers=stats_over(means),
                fibers_at_or_below_1p5_sv=int(sum(m <= G2_MAX_MEAN for m in means)),
                fibers_at_or_below_2_sv=int(sum(m <= 2. for m in means)),
                fiber_floor_mean_abs_sv_over_fibers=stats_over([f['fiber_floor_mean_abs_sv'] for f in fibers]),
                strip_minus_fiber_floor_sv_over_fibers=stats_over(diffs),
                fibers_strip_below_fiber_floor=int(sum(d < 0 for d in diffs)))


def bootstrap_replicates(means, seed, B):
    rng = np.random.default_rng(seed)
    v = np.asarray(means, np.float64)
    draws = rng.integers(0, len(v), size=(B, len(v)))
    return v[draws].mean(1)


def bootstrap_equal_weight(means, seed, B):
    reps = bootstrap_replicates(means, seed, B)
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def e1_e2(rep):
    ps = rep['per_strip']
    fibers = per_fiber(ps)
    acc = rep['accuracy']
    checks = {}
    for fam in ('all', 'H', 'V'):
        sel = [e for e in ps if fam == 'all' or e['family'] == fam]
        n = sum(e['strip_nodes'] for e in sel)
        pooled = sum(e['strip_nodes']*e['strip_mean_abs_sv'] for e in sel)/n
        fl_n = sum(e['strip_nodes'] for e in sel if e['fiber_floor_mean_abs_sv'] is not None)
        fl = sum(e['strip_nodes']*e['fiber_floor_mean_abs_sv'] for e in sel
                 if e['fiber_floor_mean_abs_sv'] is not None)/fl_n
        a = acc[fam]
        checks[fam] = dict(per_strip_nodes=int(n), report_nodes=a['strip']['point']['nodes'],
                           per_strip_node_weighted_mean_abs_sv=pooled, report_pooled_mean_abs_sv=a['strip']['point']['mean'],
                           abs_difference_sv=abs(pooled-a['strip']['point']['mean']),
                           per_strip_fibers=len({e['fiber'] for e in sel}), report_scored_fibers=a['scored_fibers'],
                           fiber_floor_node_weighted_pooled_sv=fl,
                           report_fiber_floor_row_pooled_sv=a['fiber_floor']['mean'],
                           report_fiber_floor_rows=a['fiber_floor']['nodes'],
                           fiber_floor_weighting_difference_sv=fl-a['fiber_floor']['mean'])
    multi = [f['fiber'] for f in fibers if len(f['strips']) > 1]
    e1 = dict(
        set='secondary (PREREG 4.10), secondary_report.json per_strip',
        method=('per fiber: mean |d| = sum(strip_nodes * strip_mean_abs_sv) / sum(strip_nodes) over the fiber\'s '
                'strips (per_strip); this equals the fiber\'s pooled mean over its covered nodes, because per_strip '
                'counts and averages exactly the covered nodes of the scored rows. Fiber floor: '
                'fiber_floor_mean_abs_sv weighted the same way by strip_nodes (node-weighted, as asked); per_strip '
                'has no count of floor rows, so for the fibers with two strips this is a node-, not row-, '
                'weighted mean of the two strips\' floors (the report pools the floor over rows: one point per '
                'scored row). Summary statistics give each fiber equal weight.'),
        fibers_with_two_strips=multi,
        all=fiber_summary(fibers),
        H=fiber_summary([f for f in fibers if f['family'] == 'H']),
        V=fiber_summary([f for f in fibers if f['family'] == 'V']),
        per_fiber=sorted(fibers, key=lambda f: (f['family'], f['strip_mean_abs_sv'])))
    # E2
    nodes = {fam: acc[fam]['strip']['point']['nodes'] for fam in ('all', 'H', 'V')}
    nf = {fam: acc[fam]['scored_fibers'] for fam in ('all', 'H', 'V')}
    ns = {fam: acc[fam]['scored_stretches'] for fam in ('all', 'H', 'V')}
    means_all = [f['strip_mean_abs_sv'] for f in fibers]
    by_nodes = sorted(fibers, key=lambda f: -f['nodes'])
    cum = np.cumsum([f['nodes'] for f in by_nodes])/nodes['all']
    worst = max(fibers, key=lambda f: f['strip_mean_abs_sv'])
    rest = [f for f in fibers if f is not worst]
    influence = dict(
        excluded_fiber=worst['fiber'], excluded_strips=worst['strips'], excluded_family=worst['family'],
        excluded_fiber_mean_abs_sv=worst['strip_mean_abs_sv'], excluded_fiber_nodes=worst['nodes'],
        equal_weight_mean_abs_sv_without_it=float(np.mean([f['strip_mean_abs_sv'] for f in rest])),
        equal_weight_median_abs_sv_without_it=float(np.median([f['strip_mean_abs_sv'] for f in rest])),
        node_weighted_mean_abs_sv_without_it=float(sum(f['nodes']*f['strip_mean_abs_sv'] for f in rest)
                                                   / sum(f['nodes'] for f in rest)),
        note=('the influence of the one fiber with the largest mean |d|, chosen after seeing the per-fiber list: '
              'descriptive only'))
    e2 = dict(
        note=('PREREG 5.3 ("pooled over nodes") and 6 (G2: "a pooled mean |d| of at most 1.5 sv; at least 70 % of '
              'nodes within 2 sv") make the pre-registered gate node-weighted: nodes are pooled over all scored '
              'fibers, the fiber is only the bootstrap cluster (5.4). The re-weightings below are descriptive '
              'only; they are not a gate and change neither the G2 verdict ("secondary: transfers") nor any '
              'pre-registered number.'),
        node_share=dict(H=nodes['H']/nodes['all'], V=nodes['V']/nodes['all'], nodes=nodes),
        fiber_share=dict(H=nf['H']/nf['all'], V=nf['V']/nf['all'], fibers=nf),
        stretch_share=dict(H=ns['H']/ns['all'], V=ns['V']/ns['all'], stretches=ns),
        nodes_per_fiber=dict(all=nodes['all']/nf['all'], H=nodes['H']/nf['H'], V=nodes['V']/nf['V']),
        pooled_mean_abs_sv_node_weighted_as_reported=dict(
            quoted_from_the_preregistered_report=True,
            all=acc['all']['strip']['point']['mean'], H=acc['H']['strip']['point']['mean'],
            V=acc['V']['strip']['point']['mean'], all_ci95_fiber_bootstrap=acc['all']['strip']['ci']['mean'],
            source=('secondary_report.json accuracy.*.strip: the pre-registered numbers, quoted unchanged for '
                    'comparison (the label of this file applies to the comparison, not to these values)')),
        equal_weight_per_fiber_mean_abs_sv=dict(
            all=float(np.mean(means_all)),
            H=float(np.mean([f['strip_mean_abs_sv'] for f in fibers if f['family'] == 'H'])),
            V=float(np.mean([f['strip_mean_abs_sv'] for f in fibers if f['family'] == 'V'])),
            all_ci95_percentile_bootstrap_over_fibers=bootstrap_equal_weight(means_all, SEED_BOOT, BOOT_B),
            share_of_replicates_at_or_below_1_5_sv=float(
                (bootstrap_replicates(means_all, SEED_BOOT, BOOT_B) <= 1.5).mean()),
            note_1_5_sv=('1.5 sv is the threshold of the node-pooled G2 gate; it is shown here for scale only and '
                         'does not apply to this re-weighted mean'),
            bootstrap=dict(B=BOOT_B, seed=SEED_BOOT, unit='fiber (each fiber\'s mean |d| drawn with replacement)'),
            method='the mean over the 36 fibers of each fiber\'s node-weighted mean |d| (E1)'),
        equal_weight_per_family_mean_abs_sv=dict(
            value=(acc['H']['strip']['point']['mean']+acc['V']['strip']['point']['mean'])/2.,
            method='(pooled H mean + pooled V mean) / 2'),
        influence_of_the_largest_per_fiber_mean=influence,
        node_concentration=dict(
            largest_fibers=[dict(fiber=f['fiber'], family=f['family'], nodes=f['nodes'],
                                 node_share=f['nodes']/nodes['all'], strip_mean_abs_sv=f['strip_mean_abs_sv'])
                            for f in by_nodes[:5]],
            cumulative_node_share_of_largest=dict(top1=float(cum[0]), top3=float(cum[2]), top5=float(cum[4]),
                                                  top10=float(cum[9])),
            largest_5_pooled_mean_abs_sv=float(sum(f['nodes']*f['strip_mean_abs_sv'] for f in by_nodes[:5])
                                               / sum(f['nodes'] for f in by_nodes[:5])),
            other_fibers_pooled_mean_abs_sv=float(sum(f['nodes']*f['strip_mean_abs_sv'] for f in by_nodes[5:])
                                                  / sum(f['nodes'] for f in by_nodes[5:])),
            other_fibers=len(by_nodes)-5,
            fibers_holding_half_the_nodes=int(np.searchsorted(cum, 0.5)+1)))
    return e1, e2, checks


def e3(rep, set_name):
    rows = []
    for e in rep['per_strip']:
        s, fp, lp = e['strip_mean_abs_sv'], e.get('frame_plane_mean_abs_sv'), e.get('lasagna_plane_mean_abs_sv')
        rows.append(dict(strip=e['strip'], fiber=e['fiber'], family=e['family'], strip_nodes=e['strip_nodes'],
                         frame_plane_nodes=e.get('frame_plane_nodes'), lasagna_plane_nodes=e.get('lasagna_plane_nodes'),
                         strip_mean_abs_sv=s, frame_plane_mean_abs_sv=fp, lasagna_plane_mean_abs_sv=lp,
                         frame_minus_strip_sv=None if fp is None else fp-s,
                         lasagna_minus_strip_sv=None if lp is None else lp-s,
                         strip_lower_than_frame_plane=None if fp is None else bool(s < fp),
                         strip_lower_than_lasagna_plane=None if lp is None else bool(s < lp)))

    def count(sel):
        f = [x for x in sel if x['strip_lower_than_frame_plane'] is not None]
        la = [x for x in sel if x['strip_lower_than_lasagna_plane'] is not None]
        both = [x for x in sel if x['strip_lower_than_frame_plane'] is not None
                and x['strip_lower_than_lasagna_plane'] is not None]
        return dict(strips=len(sel),
                    strip_lower_than_frame_plane=sum(x['strip_lower_than_frame_plane'] for x in f),
                    strips_with_frame_plane=len(f),
                    strip_lower_than_lasagna_plane=sum(x['strip_lower_than_lasagna_plane'] for x in la),
                    strips_with_lasagna_plane=len(la),
                    strip_lower_than_both=sum(x['strip_lower_than_frame_plane'] and x['strip_lower_than_lasagna_plane']
                                              for x in both),
                    strip_lower_than_neither=sum((not x['strip_lower_than_frame_plane'])
                                                 and (not x['strip_lower_than_lasagna_plane']) for x in both),
                    frame_minus_strip_sv_over_strips=stats_over([x['frame_minus_strip_sv'] for x in f]),
                    lasagna_minus_strip_sv_over_strips=stats_over([x['lasagna_minus_strip_sv'] for x in la]))
    rel = []
    for x in rows:
        for k in ('frame_plane_nodes', 'lasagna_plane_nodes'):
            if x[k] is not None and x['strip_nodes']:
                rel.append((abs(x[k]-x['strip_nodes'])/x['strip_nodes'], x['strip'], k))
    rel.sort(reverse=True)
    return dict(set=set_name,
                method=('per strip, per_strip\'s strip_mean_abs_sv against frame_plane_mean_abs_sv and '
                        'lasagna_plane_mean_abs_sv. Each mean is over that surface\'s own covered nodes on the '
                        'scored rows (the node counts can differ a little between surfaces); this is not the '
                        'pre-registered paired comparison, which uses the nodes covered by both (5.5). "Strip lower" '
                        'is a plain comparison of the two means, with no interval.'),
                all=count(rows), H=count([x for x in rows if x['family'] == 'H']),
                V=count([x for x in rows if x['family'] == 'V']),
                strips_where_a_baseline_is_lower=[x for x in rows if not x['strip_lower_than_frame_plane']
                                                  or not x['strip_lower_than_lasagna_plane']],
                largest_node_count_mismatches=[dict(strip=s, surface=k, relative_difference=v) for v, s, k in rel[:5]],
                per_strip=rows)


# ---- E4: PHerc0332 crossings -------------------------------------------------------------------------------------------

def e4(ws, crep, plan, strips_root, fibers_dir):
    per = crep['crossings']['per_crossing']
    rows = []
    for c in per:
        rows.append(dict(id=c['id'], h_fiber=c['h_fiber'], v_fiber=c['v_fiber'],
                         closest_approach_sv=c['closest_approach_sv'], agreement_sv=c['agreement_sv'],
                         agreement_minus_closest_approach_sv=c['agreement_sv']-c['closest_approach_sv'],
                         agreement_over_closest_approach=c['agreement_sv']/c['closest_approach_sv'],
                         tight=bool(c['closest_approach_sv'] <= CROSS_TOL), directions=c['directions']))
    x = [r_['closest_approach_sv'] for r_ in rows]
    y = [r_['agreement_sv'] for r_ in rows]
    p_rho, p_tau = permutation_p(x, y, PERM_B, SEED_PERM)
    diff = [r_['agreement_minus_closest_approach_sv'] for r_ in rows]
    part1 = dict(
        source='crossings_0332_report.json crossings.per_crossing (the 12 crossings with a value)',
        per_crossing=sorted(rows, key=lambda q: q['closest_approach_sv']),
        difference_agreement_minus_closest_approach_sv=stats_over(diff),
        crossings_agreement_below_closest_approach=int(sum(d < 0 for d in diff)),
        crossings_agreement_above_closest_approach=int(sum(d > 0 for d in diff)),
        crossings_within_1sv_of_closest_approach=int(sum(abs(d) <= 1. for d in diff)),
        rank_correlation=dict(n=len(rows), spearman_rho=spearman(x, y), kendall_tau_b=kendall_tau_b(x, y),
                              permutation_p_two_sided_spearman=p_rho, permutation_p_two_sided_kendall=p_tau,
                              permutations=PERM_B, seed=SEED_PERM,
                              note='12 crossings: a rank correlation this small in n is weak evidence either way'),
        pearson_r=float(np.corrcoef(x, y)[0, 1]))

    # the strips: the scorer's crossing rule re-run, and PREREG 7's text rule on the planned pieces
    run = [c for c in plan['crossings'] if c.get('run', True)]
    names = sorted({c['h_fiber'] for c in run} | {c['v_fiber'] for c in run})
    fibers = [FiberData(Path(fibers_dir)/f'{n}.json') for n in names]  # sorted by file name, as load_fibers
    by_name = {f.name: k for k, f in enumerate(fibers)}
    strips = [StripData(d) for d in discover_strips(strips_root)]
    assoc = {}
    for st in strips:
        k, med = associate(st, fibers)
        assoc[st.name] = (k, med)
    by_fiber = {}
    for st in strips:
        k = assoc[st.name][0]
        if k is not None:
            by_fiber.setdefault(fibers[k].name, []).append(st)
    windows = {w['id']: w for w in plan['windows']}
    pieces = {p['id']: p for p in plan['pieces']}
    sbyname = {st.name: st for st in strips}
    surfaces = {}

    def surface(st):
        if st.name not in surfaces:
            surfaces[st.name] = Surface(st.rows, st.valid)
        return surfaces[st.name]

    def near_count(st, point):
        with np.errstate(invalid='ignore'):
            dist = np.linalg.norm(st.rows-point, axis=-1)
        near = st.valid & (dist <= CROSS_RADIUS)
        dv = dist[st.valid]
        return near, (float(dv.min()) if len(dv) else None)

    def near_strip(name, point):
        best, best_n = None, 0
        for st in by_fiber.get(name, []):
            n = int(near_count(st, point)[0].sum())
            if n > best_n:
                best, best_n = st, n
        return best

    def direction(a, b, point):
        near, _ = near_count(a, point)
        out = dict(nodes_strip=a.name, onto_strip=b.name, near_nodes=int(near.sum()))
        if near.sum() < CROSS_MIN_NODES:
            out.update(covered=None, used=False, reason=f'fewer than {CROSS_MIN_NODES} valid nodes of {a.name} '
                                                        f'within {CROSS_RADIUS:g} sv of the crossing point')
            return out, None
        d, cov = surface(b).project(a.rows[near])
        out['covered'] = int(cov.sum())
        if cov.sum() >= CROSS_MIN_NODES:
            med = float(np.median(np.abs(d[cov])))
            out.update(used=True, median_abs_d_sv=med)
            return out, med
        out.update(used=False, reason=f'fewer than {CROSS_MIN_NODES} of them covered by {b.name}')
        return out, None

    reported = {c['id']: c for c in per}
    details, missing_code, missing_text, repro = [], [], [], []
    for c in run:
        point = np.asarray(c['point_sv'], np.float64)
        rec = dict(id=c['id'], closest_approach_sv=c['closest_approach_sv'], tight=bool(c['closest_approach_sv'] <= CROSS_TOL),
                   z_tercile=c.get('z_tercile'), point_sv=c['point_sv'])
        # code path (score_strips.crossing_agreement)
        st_h, st_v = near_strip(c['h_fiber'], point), near_strip(c['v_fiber'], point)
        code = dict(h_strip=None if st_h is None else st_h.name, v_strip=None if st_v is None else st_v.name)
        meds = []
        if st_h is None or st_v is None:
            code['status'] = 'missing: no strip of the ' + ' and '.join(
                s for s, st in (('H fiber', st_h), ('V fiber', st_v)) if st is None) + \
                f' has a valid node within {CROSS_RADIUS:g} sv of the crossing point (near_strip)'
        else:
            dirs = []
            for a, b in ((st_h, st_v), (st_v, st_h)):
                dd, med = direction(a, b, point)
                dirs.append(dd)
                if med is not None:
                    meds.append(med)
            code['directions'] = dirs
            code['status'] = 'value' if meds else 'missing: neither direction usable'
        if meds:
            code['agreement_sv'] = float(np.mean(meds))
        if c['id'] in reported:
            rv = reported[c['id']]['agreement_sv']
            code['agreement_sv_report'] = rv
            code['abs_difference_vs_report_sv'] = abs(code.get('agreement_sv', math.nan)-rv)
            repro.append(code['abs_difference_vs_report_sv'])
            code['directions_report'] = reported[c['id']]['directions']
        if not meds:
            missing_code.append(c['id'])
        rec['scorer_rule_recomputed'] = code
        # PREREG 7's text rule on the planned strips (the pieces holding the crossing's two windows)
        text, tmeds = {}, []
        sides = {}
        for side in ('H', 'V'):
            w = windows[f"{c['id']}{side}"]
            pc = pieces[w['piece']]
            st = sbyname.get(w['piece'])
            fib = fibers[by_name[w['fiber']]]
            s_cross = c['h_s'] if side == 'H' else c['v_s']
            info = dict(window=w['id'], window_s_sv=[w['s0'], w['s1']], piece=w['piece'], fiber=w['fiber'],
                        fiber_length_sv=fib.length, crossing_s_sv=s_cross,
                        crossing_to_fiber_start_sv=s_cross, crossing_to_fiber_end_sv=fib.length-s_cross,
                        piece_fiber_s_sv=pc['parent_arc_sv'],
                        piece_starts_at_true_fiber_start=bool(abs(pc['parent_arc_sv'][0]) < 1e-6),
                        piece_ends_at_true_fiber_end=bool(abs(pc['parent_arc_sv'][1]-fib.length) < 1e-6),
                        strip_present=st is not None)
            if st is not None:
                k, med = assoc[st.name]
                info['strip_associated_fiber'] = None if k is None else fibers[k].name
                info['strip_association_median_sv'] = med
                near, dmin = near_count(st, point)
                info['valid_nodes_within_30sv'] = int(near.sum())
                info['nearest_valid_node_to_point_sv'] = dmin
                cen = st.row_centres()
                okr = np.isfinite(cen).all(1)
                if okr.any():
                    ii, dd = nearest(cen[okr], fib.points)
                    rs = fib.s[ii]
                    info['valid_rows_fiber_s_sv'] = [float(rs.min()), float(rs.max())]
                    info['crossing_to_nearest_valid_row_along_fiber_sv'] = float(np.min(np.abs(rs-s_cross)))
                info['rows'] = st.row_classes()
            sides[side] = (st, info)
        for a_side, b_side in (('H', 'V'), ('V', 'H')):
            a, b = sides[a_side][0], sides[b_side][0]
            if a is None or b is None:
                text[f'{a_side}_onto_{b_side}'] = dict(used=False, reason='strip missing')
                continue
            dd, med = direction(a, b, point)
            text[f'{a_side}_onto_{b_side}'] = dd
            if med is not None:
                tmeds.append(med)
        text['status'] = 'value' if tmeds else 'missing'
        if tmeds:
            text['agreement_sv'] = float(np.mean(tmeds))
        if not tmeds:
            missing_text.append(c['id'])
        rec['prereg7_text_rule_on_planned_strips'] = text
        rec['H'], rec['V'] = sides['H'][1], sides['V'][1]
        details.append(rec)

    # the fibers' geometry against the plan (validates the fiber parse)
    geo = []
    for c in run:
        fh, fv = fibers[by_name[c['h_fiber']]], fibers[by_name[c['v_fiber']]]
        ih, eh = fh.sample_at(c['h_s'])
        iv, ev = fv.sample_at(c['v_s'])
        p = (fh.points[ih]+fv.points[iv])/2
        geo.append(dict(id=c['id'], s_match_sv=max(eh, ev),
                        point_abs_difference_sv=float(np.abs(p-np.asarray(c['point_sv'])).max()),
                        closest_approach_abs_difference_sv=abs(float(np.linalg.norm(fh.points[ih]-fv.points[iv]))
                                                               - c['closest_approach_sv'])))
    def num(v, fmt='.1f'):
        return 'n/a' if v is None else format(v, fmt)

    reasons = []
    for rec in details:
        if rec['id'] not in missing_code:
            continue
        lines = []
        for side in ('H', 'V'):
            i_ = rec[side]
            rw = i_.get('rows', {})
            which, dist_end = min((('start', i_['crossing_to_fiber_start_sv']), ('end', i_['crossing_to_fiber_end_sv'])),
                                  key=lambda t_: t_[1])
            true_end = i_['piece_starts_at_true_fiber_start'] if which == 'start' else i_['piece_ends_at_true_fiber_end']
            text_ = (f"{side} strip {i_['piece']} (window {i_['window']}, fiber {i_['fiber']}): "
                     f"{i_.get('valid_nodes_within_30sv')} valid nodes within {CROSS_RADIUS:g} sv of the crossing point "
                     f"(nearest valid node {num(i_.get('nearest_valid_node_to_point_sv'))} sv)")
            if dist_end <= TRIM_SV+CROSS_RADIUS:
                dropped = rw.get('rows_dropped_start_trim' if which == 'start' else 'rows_dropped_end_trim')
                span = rw.get('start_trim_rows_piece_arc_sv' if which == 'start' else 'end_trim_rows_piece_arc_sv')
                text_ += (f"; the crossing lies {dist_end:.1f} sv from the fiber's {which} (the fiber is "
                          f"{i_['fiber_length_sv']:.1f} sv long), and the piece "
                          f"{'reaches that true fiber ' + which if true_end else 'is cut before it'}; the {TRIM_SV:g} sv "
                          f"{which} trim removed the {dropped} rows there that had decisions (piece arclength "
                          f"{[round(v, 1) for v in span] if span else span} sv); the nearest valid row is "
                          f"{num(i_.get('crossing_to_nearest_valid_row_along_fiber_sv'))} sv along the fiber from the "
                          f"crossing")
            else:
                vs = i_.get('valid_rows_fiber_s_sv')
                text_ += (f"; the crossing is {dist_end:.0f} sv from the nearest fiber end, within the strip's valid "
                          f"rows (fiber s {num(vs[0], '.0f') if vs else 'n/a'} to {num(vs[1], '.0f') if vs else 'n/a'} sv)")
            text_ += (f"; rows only partly valid (holes in the mask): {rw.get('rows_partly_valid')}; mask.tif equals "
                      f"the x/y/z validity: {rw.get('mask_equals_xyz_validity')}.")
            lines.append(text_)
        t = rec['prereg7_text_rule_on_planned_strips']
        lines.append('PREREG 7 text rule on the planned strips: ' + '; '.join(
            f"{k.replace('_onto_', ' onto ')}: {v.get('near_nodes')} near nodes, covered {v.get('covered')}"
            + ('' if v.get('used') else f" -> skipped ({v.get('reason')})") for k, v in t.items()
            if isinstance(v, dict)) + f"; status {t['status']}.")
        lines.append(f"Scorer path: {rec['scorer_rule_recomputed']['status']}.")
        reasons.append(dict(id=rec['id'], closest_approach_sv=rec['closest_approach_sv'], tight=rec['tight'],
                            explanation=lines))
    proximity = []
    for rec in details:
        row = dict(id=rec['id'], status=rec['scorer_rule_recomputed']['status'].split(':')[0],
                   closest_approach_sv=rec['closest_approach_sv'])
        for side in ('H', 'V'):
            i_ = rec[side]
            row[f'{side}_crossing_to_nearest_fiber_end_sv'] = min(i_['crossing_to_fiber_start_sv'],
                                                                  i_['crossing_to_fiber_end_sv'])
            row[f'{side}_valid_nodes_within_30sv'] = i_.get('valid_nodes_within_30sv')
        dirs = rec['scorer_rule_recomputed'].get('directions', [])
        row['covered_per_direction'] = [d_.get('covered') for d_ in dirs]
        proximity.append(row)
    near_end = [p_ for p_ in proximity if min(p_['H_crossing_to_nearest_fiber_end_sv'],
                                              p_['V_crossing_to_nearest_fiber_end_sv']) <= TRIM_SV+CROSS_RADIUS]
    part2_prox = dict(
        rule_of_thumb=(f'on a fiber end that is a true end, the strip keeps no row within {TRIM_SV:g} sv of it; a '
                       f'crossing closer than about {TRIM_SV-CROSS_RADIUS:g} sv to it has no strip node within '
                       f'{CROSS_RADIUS:g} sv on that side (straight fiber), and one closer than '
                       f'{TRIM_SV+CROSS_RADIUS:g} sv has fewer'),
        crossings_within_105sv_of_a_fiber_end=near_end,
        all_run_crossings=sorted(proximity, key=lambda p_: min(p_['H_crossing_to_nearest_fiber_end_sv'],
                                                              p_['V_crossing_to_nearest_fiber_end_sv'])))
    part2 = dict(
        end_proximity=part2_prox,
        method=('The scorer\'s crossing rule (score_strips.crossing_agreement) re-run in numpy on the 25 strips and '
                'the 24 fibers of the 15 run crossings: every strip associated with its nearest fiber (median row '
                'centre distance <= 5 sv); per fiber, the strip with the most valid nodes within 30 sv of the '
                'crossing point (near_strip; none if no node); each direction needs >= 5 such nodes and >= 5 of '
                'them covered by the other strip (projection port of GridSurface.project). Separately, PREREG 7\'s '
                'text rule evaluated on the strips of the pieces that hold the crossing\'s two windows, both '
                'directions always tried. Row classes from strip.npz: a row with stitch weight > 0 that is invalid '
                'and lies within 75 sv of the piece start, or after the last valid row, was removed by infer.py\'s '
                'end trim (rows of piece arclength < 75 sv or > L - 75 sv, L the piece\'s kink-repaired length).'),
        strip_association=[dict(strip=st.name, fiber=None if assoc[st.name][0] is None else fibers[assoc[st.name][0]].name,
                                median_sv=assoc[st.name][1]) for st in strips],
        reproduction=dict(crossings_with_value_recomputed=len(run)-len(missing_code),
                          missing_recomputed=missing_code, missing_report=crep['crossings']['missing'],
                          same_missing=bool(missing_code == crep['crossings']['missing']),
                          max_abs_agreement_difference_vs_report_sv=max(repro) if repro else None,
                          missing_under_prereg7_text_rule=missing_text),
        fiber_geometry_vs_plan=dict(max_point_abs_difference_sv=max(g['point_abs_difference_sv'] for g in geo),
                                    max_closest_approach_abs_difference_sv=max(g['closest_approach_abs_difference_sv']
                                                                               for g in geo),
                                    max_s_mismatch_sv=max(g['s_match_sv'] for g in geo)),
        missing=reasons,
        per_crossing_detail=details)
    files = dict(fibers={f.file: f.sha256 for f in fibers},
                 strips={st.name: st.files for st in strips})
    return dict(agreement_vs_closest_approach=part1, missing_crossings=part2), files, fibers


# ---- E5: P0005 and the end trim in every plan ----------------------------------------------------------------------

def trim_reach(plan_pieces, units, fiber_length, kind):
    """Per window (stretch or crossing window) of a piece: its raw-arclength distance to the piece's ends and the
    overlap with the first and last 75 sv of the piece, with whether each end is a true fiber end."""
    out = []
    for pc in plan_pieces:
        if not pc.get('run', True):
            continue
        L = fiber_length[pc['fiber']]
        p0, p1 = pc['parent_arc_sv']
        true0, true1 = abs(p0) < 1e-6, abs(p1-L) < 1e-6
        for wid in pc['windows']:
            w = units.get(wid)
            if w is None:
                continue
            ov0 = max(0., min(w['s1'], p0+TRIM_SV)-max(w['s0'], p0))
            ov1 = max(0., min(w['s1'], p1)-max(w['s0'], p1-TRIM_SV))
            out.append(dict(kind=kind, window=wid, piece=pc['id'], fiber=pc['fiber'], window_s_sv=[w['s0'], w['s1']],
                            piece_s_sv=[p0, p1], fiber_length_sv=L, window_start_minus_piece_start_sv=w['s0']-p0,
                            piece_end_minus_window_end_sv=p1-w['s1'], start_is_true_fiber_start=bool(true0),
                            end_is_true_fiber_end=bool(true1), overlap_start_trim_sv=ov0, overlap_end_trim_sv=ov1,
                            window_length_sv=w['s1']-w['s0']))
    return out


def trim_summary(rows, scored):
    cut0 = [x['window_start_minus_piece_start_sv'] for x in rows if not x['start_is_true_fiber_start']]
    cut1 = [x['piece_end_minus_window_end_sv'] for x in rows if not x['end_is_true_fiber_end']]
    touch = [x for x in rows if x['overlap_start_trim_sv'] > 0 or x['overlap_end_trim_sv'] > 0]
    return dict(prereg_4_6_claim='the trim never reaches a scored row, except at a true fiber end',
                claim_holds_in_this_plan=not any(
                    (x['overlap_start_trim_sv'] > 0 and not x['start_is_true_fiber_start'])
                    or (x['overlap_end_trim_sv'] > 0 and not x['end_is_true_fiber_end']) for x in rows),
                windows=len(rows),
                min_distance_window_to_a_cut_piece_start_sv=min(cut0) if cut0 else None,
                min_distance_window_to_a_cut_piece_end_sv=min(cut1) if cut1 else None,
                min_margin_beyond_the_trim_zone_at_a_cut_start_sv=min(cut0)-TRIM_SV if cut0 else None,
                min_margin_beyond_the_trim_zone_at_a_cut_end_sv=min(cut1)-TRIM_SV if cut1 else None,
                windows_reaching_a_trim_zone_at_a_cut_end=sum(
                    (x['overlap_start_trim_sv'] > 0 and not x['start_is_true_fiber_start'])
                    or (x['overlap_end_trim_sv'] > 0 and not x['end_is_true_fiber_end']) for x in rows),
                windows_reaching_a_trim_zone_at_a_true_fiber_end=sum(
                    (x['overlap_start_trim_sv'] > 0 and x['start_is_true_fiber_start'])
                    or (x['overlap_end_trim_sv'] > 0 and x['end_is_true_fiber_end']) for x in rows),
                windows_entirely_inside_a_trim_zone=sum(
                    x['overlap_start_trim_sv']+x['overlap_end_trim_sv'] >= x['window_length_sv']-1e-9 for x in touch),
                touching=[dict(window=x['window'], piece=x['piece'], fiber=x['fiber'], window_s_sv=x['window_s_sv'],
                               fiber_length_sv=x['fiber_length_sv'], overlap_start_trim_sv=x['overlap_start_trim_sv'],
                               overlap_end_trim_sv=x['overlap_end_trim_sv'], window_length_sv=x['window_length_sv'],
                               scored_in_report=bool(scored(x['window'])))
                          for x in touch])


def repair_offsets(ws, plan_pieces, tsv_name):
    """Per run piece: the raw piece arclength of its first head (the plan's arc_patch_vox[0]) minus the run plan's
    --arc start (the kink-repaired arclength there), i.e. how much shorter the repaired arclength is before the first
    head. infer.py trims in the repaired arclength."""
    lines = (Path(ws)/'results'/tsv_name).read_text(encoding='utf-8').splitlines()
    head = lines[0].split('\t')
    ip, ia = head.index('piece'), head.index('arc_start_patch_vox')
    arc0 = {f[ip]: float(f[ia]) for f in (line.split('\t') for line in lines[1:] if line.strip())}
    offs = {pc['id']: pc['arc_patch_vox'][0]-arc0[pc['id']] for pc in plan_pieces
            if pc.get('run', True) and pc['id'] in arc0 and pc.get('arc_patch_vox')}
    k = max(offs, key=lambda i: abs(offs[i])) if offs else None
    return dict(run_plan=tsv_name, pieces=len(offs), max_abs_offset_sv=abs(offs[k]) if k else None, piece=k)


def e5(ws, sec_plan, prim_plan, cross_plan, sec_rep, prim_rep, cross_rep, cross_fibers):
    st_id = 'P0005'
    pc = next(p for p in sec_plan['pieces'] if p['id'] == st_id)
    stretches = {s['id']: s for s in sec_plan['stretches']}
    sts = [stretches[w] for w in pc['windows']]
    frec = next(f for f in sec_plan['fibers'] if f['fiber'] == pc['fiber'])
    fib = FiberData(Path(ws)/'data'/'fibers0139'/f"{pc['fiber']}.json")
    strip = StripData(Path(ws)/'runs'/'v3'/'strips'/'secondary'/st_id/'strip')
    # the plan's geometry reproduced by the fiber parse
    st0 = sts[0]
    centre = fib.points[(st0['i0']+st0['i1'])//2]*SCORE_UNIT_BASE
    parse_check = dict(length_sv=fib.length, plan_length_sv=frec['length_sv'],
                       length_abs_difference_sv=abs(fib.length-frec['length_sv']),
                       line_offset=fib.line_offset, plan_line_offset=frec['line_offset'],
                       stretch_centre_l0_abs_difference=float(np.abs(centre-np.asarray(st0['centre_l0'])).max()),
                       stretch_s0_s1_are_samples=bool(abs(fib.s[st0['i0']]-st0['s0']) < 1e-9
                                                      and abs(fib.s[st0['i1']]-st0['s1']) < 1e-9))
    # association and rows (score_strips.score_strips: only the planned fiber)
    k, med = associate(strip, [fib], only=0)
    cen = strip.row_centres()
    okr = np.isfinite(cen).all(1)
    row_s, row_d = np.full(strip.R, np.nan), np.full(strip.R, np.inf)
    if okr.any():
        ii, dd = nearest(cen[okr], fib.points)
        row_s[okr], row_d[okr] = fib.s[ii], dd
    per_stretch = []
    for s in sts:
        inside = (row_s >= s['s0']) & (row_s <= s['s1']) & (row_d <= ROW_MAX_SV)
        arc = strip.layout['row_arc_sv']
        on_piece = (arc >= s['s0']-pc['parent_arc_sv'][0]) & (arc <= s['s1']-pc['parent_arc_sv'][0])
        per_stretch.append(dict(
            stretch=s['id'], segment=s['segment'], fiber_s_sv=[s['s0'], s['s1']], length_sv=s['length_sv'],
            piece_s_sv=[s['s0']-pc['parent_arc_sv'][0], s['s1']-pc['parent_arc_sv'][0]],
            stretch_end_minus_fiber_start_sv=s['s1'], fiber_length_sv=fib.length,
            rows_inside_stretch_scorer_rule=int(inside.sum()),
            strip_rows_whose_piece_arclength_lies_in_the_stretch=int(on_piece.sum()),
            of_which_with_decisions=int((on_piece & strip.layout['support']).sum()),
            of_which_valid=int((on_piece & (strip.valid.sum(1) > 0)).sum()),
            of_which_removed_by_start_trim=int((on_piece & strip.layout['support'] & (arc < TRIM_SV)
                                                & (strip.valid.sum(1) == 0)).sum()),
            plan_head_k=s['head_k'], plan_heads_sv=s['heads_sv']))
    valid_rows = np.flatnonzero(strip.valid.sum(1) > 0)
    rows = strip.row_classes()
    p5 = dict(
        skip_in_report=[x for x in sec_rep['skips'] if x['strip'] == st_id],
        piece=dict(id=st_id, fiber=pc['fiber'], family=pc['family'], fiber_line_index_range=pc['line_index_range'],
                   fiber_s_sv=pc['parent_arc_sv'], length_sv=pc['length_sv'], head_k=pc['head_k'],
                   heads_sv=pc['heads_sv'], heads_full_run_sv=pc['heads_full_sv'], decisions=pc['decisions'],
                   windows=pc['windows'], starts_at_true_fiber_start=bool(abs(pc['parent_arc_sv'][0]) < 1e-6),
                   fiber_controlled_span_first_line_index=frec['line_offset'],
                   note=('the piece starts at line index 150 of the JSON, the first point of the fiber\'s controlled '
                         'span (the line_offset of the plan\'s fiber record): the start of the fiber in the full AFV '
                         'too, so a full infer_all run starts there as well (first head k = 0 at 24 sv)')),
        stretches=per_stretch,
        fiber_parse_vs_plan=parse_check,
        strip=dict(association_median_sv=med, associated=k is not None,
                   summary_arc_trace=strip.trace.get('arc_trace'),
                   first_row_arc_patch_voxels=strip.trace.get('first_row_arc_patch_voxels'),
                   trim_patch_voxels=strip.trace.get('trim_patch_voxels'), rows=rows,
                   valid_rows=[dict(row=int(i), piece_arc_sv=float(strip.layout['row_arc_sv'][i]),
                                    fiber_s_sv=float(row_s[i]), centre_to_fiber_sv=float(row_d[i]),
                                    valid_nodes=int(strip.valid[i].sum())) for i in valid_rows]),
        files=dict(fiber={fib.file: fib.sha256}, strip=strip.files))
    expl = (f"P0005 holds one stretch, {per_stretch[0]['stretch']}, at fiber arclength "
            f"{per_stretch[0]['fiber_s_sv'][0]:.1f} to {per_stretch[0]['fiber_s_sv'][1]:.1f} sv: it starts at the "
            f"first sample of the fiber's controlled span (the fiber is {fib.length:.1f} sv long). The piece starts "
            f"there too (fiber s {pc['parent_arc_sv'][0]:.1f} to {pc['parent_arc_sv'][1]:.1f} sv), so its start is a "
            f"true fiber end. Its {pc['decisions']} decisions (heads {pc['heads_sv'][0]:g} to {pc['heads_sv'][1]:g} "
            f"sv) give rows from {rows['rows_with_decisions_piece_arc_sv'][0]:.1f} to "
            f"{rows['rows_with_decisions_piece_arc_sv'][1]:.1f} sv of the piece; infer.py's {TRIM_SV:g} sv end trim "
            f"removes every row below {TRIM_SV:g} sv ({rows['rows_dropped_start_trim']} rows with decisions, piece "
            f"arclength {rows['start_trim_rows_piece_arc_sv']}), which includes all "
            f"{per_stretch[0]['of_which_with_decisions']} rows lying in the stretch. The {len(valid_rows)} rows "
            f"left ({rows['valid_nodes']} nodes, piece arclength {rows['valid_rows_piece_arc_sv']}) lie at fiber "
            f"arclength {float(np.nanmin(row_s[valid_rows])):.0f} to {float(np.nanmax(row_s[valid_rows])):.0f} sv, "
            f"beyond the stretch's end at {per_stretch[0]['fiber_s_sv'][1]:.0f} sv, so the scorer finds no row "
            f"inside it. The mask is not the cause: mask.tif equals the x/y/z validity and no row is partly valid. "
            f"PREREG 4.6 says the trim reaches a scored row only at a true fiber end, where a full run trims the "
            f"same way: this is that case (a full run would also leave this stretch without a row). The fiber is "
            f"still scored through P0006 (31 stretches), so no fiber was lost.")
    p5['explanation'] = expl
    # the trim against every plan: does a window reach the first/last 75 sv of its piece, and only at true ends?
    sec_len = {f['fiber']: f['length_sv'] for f in sec_plan['fibers']}
    sec_units = {s['id']: s for s in sec_plan['stretches'] if s.get('run', True)}
    prim_len = {f['fiber']: f['length_sv'] for f in prim_plan['fibers']}
    prim_units = {s['id']: s for s in prim_plan['stretches']}
    cross_len = {f.name: f.length for f in cross_fibers}
    cross_units = {w['id']: w for w in cross_plan['windows'] if w.get('run')}
    sec_scored = {s for e in sec_rep['per_strip'] for s in e['stretches']}
    prim_scored = {s for e in prim_rep['per_strip'] for s in e['stretches']}
    cross_valued = {c['id'] for c in cross_rep['crossings']['per_crossing']}
    offsets = dict(secondary_run=repair_offsets(ws, sec_plan['pieces'], 'runplan_0139_secondary_run40.tsv'),
                   primary=repair_offsets(ws, prim_plan['pieces'], 'runplan_0139_primary.tsv'),
                   pherc0332_run_windows=repair_offsets(ws, cross_plan['pieces'], 'runplan_0332_crossings.tsv'))
    worst = max((o for o in offsets.values() if o['piece']), key=lambda o: o['max_abs_offset_sv'])
    plans = dict(
        note=('raw JSON arclength; infer.py trims in the piece\'s kink-repaired arclength, which is shorter by the '
              f'repairs inside the piece (before the first head by up to {worst["max_abs_offset_sv"]:.1f} sv, piece '
              f'{worst["piece"]}; per plan in kink_repair_offset_at_first_head), so the overlaps and distances are '
              'approximate. "distance" is from a window to the cut piece end; "margin beyond the trim zone" '
              f'subtracts the {TRIM_SV:g} sv trim. A window reaching a trim zone loses its rows there (whole rows). '
              'scored_in_report: the stretch is in a per_strip entry (secondary, primary), or the crossing has a '
              'value (PHerc0332).'),
        kink_repair_offset_at_first_head=offsets,
        secondary_run=trim_summary(trim_reach(sec_plan['pieces'], sec_units, sec_len, 'stretch'),
                                   lambda w: w in sec_scored),
        primary=trim_summary(trim_reach(prim_plan['pieces'], prim_units, prim_len, 'stretch'),
                             lambda w: w in prim_scored),
        pherc0332_run_windows=trim_summary(trim_reach(cross_plan['pieces'], cross_units, cross_len, 'crossing window'),
                                           lambda w: cross_units[w]['crossing'] in cross_valued))
    return dict(P0005=p5, end_trim_in_the_plans=plans)


# ---- main --------------------------------------------------------------------------------------------------------------

def main(argv=None):
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workspace', default='C:/VesuviusTemp/strips/score', help='the frozen score/ tree (read only)')
    ap.add_argument('--out', default=str(here.parent/'exploratory_postrun.json'))
    args = ap.parse_args(argv)
    t0 = time.time()
    ws = Path(args.workspace)
    res = ws/'results'
    rep_paths = dict(primary=res/'v3'/'primary_case_studies.json', secondary=res/'v3'/'secondary_report.json',
                     crossings=res/'v3'/'crossings_0332_report.json')
    plan_paths = dict(primary=res/'matched_stretches.json', secondary=res/'matched_stretches_secondary_run40.json',
                      crossings=res/'crossings_0332_plan.json')
    reps = {k: json.loads(p.read_text()) for k, p in rep_paths.items()}
    plans = {k: json.loads(p.read_text()) for k, p in plan_paths.items()}
    sec, prim, crep = reps['secondary'], reps['primary'], reps['crossings']

    e1, e2, chk = e1_e2(sec)
    e3_sec = e3(sec, 'secondary (PREREG 4.10)')
    e3_prim = e3(prim, 'primary case studies (G1 failed: no accuracy claim; 2 strips)')
    e4_out, e4_files, cross_fibers = e4(ws, crep, plans['crossings'], ws/'runs'/'v3'/'strips'/'0332',
                                        ws/'data'/'fibers0332')
    e5_out = e5(ws, plans['secondary'], plans['primary'], plans['crossings'], sec, prim, crep, cross_fibers)

    manifests = {}
    for tag, mf in (('fibers0139', ws/'data'/'fibers0139_manifest.json'),
                    ('fibers0332', ws/'data'/'fibers0332_manifest.json')):
        if mf.exists():
            manifests[tag] = {x['fiber']: x.get('sha256') for x in json.loads(mf.read_text()).get('files', [])}
    used = {**e4_files['fibers'], **e5_out['P0005']['files']['fiber']}
    man = {**manifests.get('fibers0139', {}), **manifests.get('fibers0332', {})}
    out = dict(
        label=LABEL,
        when=WHEN,
        what=('Exploratory analyses E1 to E5 of the v3 scored run of score/PREREG.md v3.2 (fiber_strip_small_v1 off '
              'Paris 4). Units: sv = level-2 voxels = 9.596 um. Every number in this file is exploratory, not '
              'pre-registered; the pre-registered results are the three reports, unchanged.'),
        provenance=dict(script='analysis/exploratory_postrun.py', script_sha256=sha256_file(here),
                        python=sys.version.split()[0],
                        numpy=np.__version__, platform=platform.platform(),
                        time_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), workspace=str(ws),
                        reports={k: dict(path=str(p), sha256=sha256_file(p)) for k, p in rep_paths.items()},
                        plans={k: dict(path=str(p), sha256=sha256_file(p)) for k, p in plan_paths.items()},
                        report_scorer_sha256=sec['provenance']['scorer_sha256'],
                        report_prereg_sha256=sec['provenance']['prereg_sha256'],
                        fiber_files_read=used,
                        fiber_files_match_manifest=all(man.get(k) == v for k, v in used.items()) if man else None,
                        strips_read=dict(pherc0332=e4_files['strips'], secondary_P0005=e5_out['P0005']['files']['strip']),
                        seconds=None),
        checks=dict(
            note=('do the re-implementations reproduce the reports and plans? report_* and plan_* values are quoted '
                  'unchanged from the pre-registered reports and frozen plans for comparison'),
            per_strip_vs_report_pooled=chk,
            pherc0332_crossing_rule=e4_out['missing_crossings']['reproduction'],
            pherc0332_fiber_geometry_vs_plan=e4_out['missing_crossings']['fiber_geometry_vs_plan'],
            p0005_fiber_parse_vs_plan=e5_out['P0005']['fiber_parse_vs_plan']),
        E1_per_fiber=e1,
        E2_weighting=e2,
        E3_per_strip_baselines=dict(secondary=e3_sec, primary_case_studies=e3_prim),
        E4_pherc0332=e4_out,
        E5_P0005_and_end_trim=e5_out)
    out['provenance']['seconds'] = round(time.time()-t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(labelled(jsonable(out)), indent=1))
    print(f'wrote {args.out} ({time.time()-t0:.0f} s)')


if __name__ == '__main__':
    main()
