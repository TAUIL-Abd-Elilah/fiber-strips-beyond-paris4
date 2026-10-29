#!/usr/bin/env python
"""Pre-registered scorer for fiber_strip_small_v1 sheet strips off Paris 4 (see PREREG.md beside this file).

Question: does the Paris4-trained strip model place the sheet correctly around human-traced fibers on PHerc0139?
Ground truth: the PHerc0139 wrap segments (VC3D tifxyz) meshed on the same 2.399 um volume 20260102150214 as the
human fibers. PHerc0332 has no ground truth: only the H/V crossing agreement is reported there.

Subcommands (all CPU only; no torch, no strip model):

  match      human fiber JSONs + segments (+ umbilicus) -> fiber selection, noise floor, gate G1 (no strips needed)
  lasagna    human fiber JSONs + Lasagna nx/ny zarr (local or https) -> Lasagna normals at the fiber samples (.npz)
  score      matches + strips (+ Lasagna normals) -> per-node accuracy, baselines, bootstrap, calibration, gate G2
  crossings  strips + fibers -> H/V crossing agreement (no ground truth)

Units. Everything is scored in "sv" (score voxels) = level 2 of the 2.4 um scans = 4 base voxels (9.6 um).
Human fiber JSONs and segment tifxyz are in base voxels (level 0; ``--fiber-unit-base 1 --segment-unit-base 1``);
strips are in the strip model's "patch voxels" (9.6 um; ``--strip-unit-base 4``). Each input's scale is
recorded in the report, and the strips are checked against the fibers (their centre column must run within
``ASSOC_MAX_SV`` of the fiber, else the strip is rejected as a unit or association error).

Normal distance. A segment (or strip) is a grid of nodes; a point is projected onto the bilinear surface of the
grid (Gauss-Newton from its nearest nodes) and its signed distance along the surface normal there is taken;
outward (away from the umbilicus) is positive when an umbilicus is given. A point whose foot falls off the valid
grid (beyond an edge or a hole) is "uncovered" and not scored. Surface normals follow villa's
sheets/scripts/match_fibers_to_patches.py (cross product of the grid's column and row derivatives).

Credits: strip format, stitching and layout are villa's sheet_strip (bruniss/sean, ScrollPrize/villa #2110;
``orient_decisions``/``stitch``/``stitch_values`` below are copied from
vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/infer.py at 6f2024569, MIT licence); the fiber
to patch matching rule follows villa's match_fibers_to_patches.py; Windows inference fixes are #2163
(slchicoineau-creator, Baz248); held-out human-fiber evaluation practice follows modaribrahim/fiber-tracer-heldout-eval
and villa #2143; human-fiber noise (fold-backs, jumps, duplicates) per ChaseHendrick's #2248.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCORER_VERSION = '1.0.0'

# ---- Pre-registered constants (PREREG.md section 4-6). Do not change after the freeze. -------------------------
SCORE_UNIT_BASE = 4.0       # base voxels per score voxel (sv): level 2 of the 2.4 um scans
SAMPLE_STEP = 1.0           # sv between fiber samples
MATCH_MAX_D = 2.0           # sv: a fiber sample lies on a segment when |normal distance| <= this
CANDIDATE_MAX_D = 6.0       # sv: about half a winding gap: the sample is on that segment's sheet
MATCH_MIN_FRACTION = 0.70   # fraction of the fiber's (controlled) length on segments
MIN_RUN = 40.0              # sv: shorter on-segment runs are contacts, not runs (villa README uses --min-run 40)
MIN_LENGTH_SV = 2.5e3/9.6   # 2.5 mm: infer_all's default --min-length-mm
SEARCH_REACH = 30.0         # sv: farthest normal distance measured (larger: uncovered)
TANGENT_TOL = 1.0           # sv: a foot farther than this sideways from the point's normal line is off the grid
ASSOC_MAX_SV = 5.0          # sv: median distance from a strip's centre column to its fiber
ROW_MAX_SV = 10.0           # sv: a strip row whose centre is farther from the fiber is not scored
MIN_NODES = 25              # covered strip nodes for a fiber to count as scored (one strip row)
G1_MIN_FIBERS = 30
G1_MAX_NOISE = 2.0          # sv
G2_MAX_MEAN = 1.5           # sv
G2_MIN_WITHIN2 = 0.70
BOOTSTRAP = 2000
BOOTSTRAP_PCT = 1000        # replicates for the median / p90 CIs (concatenating nodes is slower)
SEED = 20261028
CAL_EDGES = (0., .5, .75, 1., 1.25, 1.5, 2., 3., math.inf)
CROSS_TOL = 3.0             # sv: closest approach of an H and a V fiber that counts as a crossing
CROSS_RADIUS = 30.0         # sv: strip nodes this close to the crossing point are compared
LASAGNA_SCALE = 16.0        # base voxels per voxel of the Lasagna nx/ny level used (level 4)
TANGENT_HALF = 4            # samples each side for the fiber tangent

CONSTANTS = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, (int, float, tuple))}


# ---- Small helpers ----------------------------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def jsonable(obj):
    """numpy scalars/arrays and non-finite floats to JSON-safe values (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj


def arclength(points):
    if len(points) < 2:
        return np.zeros(len(points))
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def resample(points, step):
    """Points every ``step`` along a polyline (first and last kept); and their arclengths."""
    s = arclength(points)
    if s[-1] <= 0:
        return points[:1].copy(), np.zeros(1)
    t = np.arange(0., s[-1]+1e-9, step)
    if s[-1]-t[-1] > 1e-6:
        t = np.r_[t, s[-1]]
    keep = np.r_[True, np.diff(s) > 1e-12]
    out = np.stack([np.interp(t, s[keep], points[keep, i]) for i in range(3)], 1)
    return out, t


def tangents(points, half=TANGENT_HALF):
    n = len(points)
    if n < 2:
        return np.tile([0., 0., 1.], (n, 1))
    idx = np.arange(n)
    d = points[np.minimum(idx+half, n-1)]-points[np.maximum(idx-half, 0)]
    return d/np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


def unit(v):
    return v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


# ---- Umbilicus ----------------------------------------------------------------------------------------------

class Umbilicus:
    """The scroll axis: control points (K, 3) xyz in sv, interpolated in z."""

    def __init__(self, xyz):
        xyz = np.asarray(xyz, np.float64)
        order = np.argsort(xyz[:, 2])
        self.xyz = xyz[order]

    @classmethod
    def from_json(cls, path, unit_base=1.0, score_unit_base=SCORE_UNIT_BASE):
        cp = json.loads(Path(path).read_text())['control_points']
        xyz = np.array([[c['x'], c['y'], c['z']] for c in cp], np.float64)*unit_base/score_unit_base
        return cls(xyz)

    def xy(self, z):
        z = np.asarray(z, np.float64)
        return np.stack([np.interp(z, self.xyz[:, 2], self.xyz[:, i]) for i in (0, 1)], -1)


# ---- Surfaces: tifxyz grids and point projection --------------------------------------------------------------

def read_tifxyz(source, unit_base, score_unit_base=SCORE_UNIT_BASE):
    """A tifxyz (local dir, or an https base URL) as (grid (R, C, 3) sv with NaN off it, valid (R, C), meta)."""
    import tifffile
    def load(name):
        if isinstance(source, str) and source.startswith(('http://', 'https://')):
            with urllib.request.urlopen(source.rstrip('/')+'/'+name, timeout=300) as r:
                data = r.read()
            return data
        return (Path(source)/name).read_bytes()
    try:
        meta = json.loads(load('meta.json'))
    except Exception:
        meta = {}
    xyz = [np.asarray(tifffile.imread(io.BytesIO(load(f'{c}.tif'))), np.float64) for c in 'xyz']
    grid = np.stack(xyz, -1)
    valid = np.isfinite(grid).all(-1) & ~(grid == -1).any(-1)
    grid = np.where(valid[..., None], grid*unit_base/score_unit_base, np.nan)
    return grid, valid, meta


class GridSurface:
    """A grid of surface nodes (R, C, 3) in sv; projects points onto its bilinear surface."""

    def __init__(self, grid, valid=None, name='', axis: Umbilicus | None = None):
        g = np.asarray(grid, np.float64)
        ok = np.isfinite(g).all(-1)
        valid = ok if valid is None else (np.asarray(valid, bool) & ok)
        self.name, self.axis = name, axis
        self.R, self.C = g.shape[:2]
        self.grid = np.where(valid[..., None], g, np.nan)
        self.valid = valid
        if self.R > 1 and self.C > 1:
            self.cell_ok = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
        else:
            self.cell_ok = np.zeros((max(self.R-1, 0), max(self.C-1, 0)), bool)
        corner = np.zeros_like(valid)
        if self.cell_ok.size:
            for dr in (0, 1):
                for dc in (0, 1):
                    corner[dr:self.R-1+dr, dc:self.C-1+dc] |= self.cell_ok
        r, c = np.nonzero(corner)
        self.node_rc = np.stack([r, c], 1)
        self.nodes = self.grid[r, c]
        self.tree = cKDTree(self.nodes) if len(self.nodes) else None
        self.lo = self.nodes.min(0) if len(self.nodes) else np.full(3, np.inf)
        self.hi = self.nodes.max(0) if len(self.nodes) else np.full(3, -np.inf)

    def near_bbox(self, points, reach=SEARCH_REACH):
        points = np.asarray(points, np.float64)
        return np.all((points >= self.lo-reach) & (points <= self.hi+reach), axis=-1)

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

    def project(self, points, k=4, reach=SEARCH_REACH, iters=8, batch=50000):
        """Signed normal distance (M,), covered (M,), foot (M, 3), unit normal (M, 3) of points (M, 3) sv."""
        p = np.asarray(points, np.float64).reshape(-1, 3)
        if len(p) > batch:  # bounded memory: about 1 KB of temporaries per point and candidate
            parts = [self.project(p[i:i+batch], k, reach, iters, batch) for i in range(0, len(p), batch)]
            return tuple(np.concatenate(x) for x in zip(*parts))
        m = len(p)
        d_out, cov_out = np.full(m, np.nan), np.zeros(m, bool)
        foot_out, n_out = np.full((m, 3), np.nan), np.full((m, 3), np.nan)
        if m == 0 or self.tree is None or not self.cell_ok.any():
            return d_out, cov_out, foot_out, n_out
        k = min(k, len(self.nodes))
        dist, idx = self.tree.query(p, k=k, distance_upper_bound=reach+2*TANGENT_TOL+8.)
        dist, idx = dist.reshape(m, k), idx.reshape(m, k)
        has = np.isfinite(dist)
        pi, ki = np.nonzero(has)
        if not len(pi):
            return d_out, cov_out, foot_out, n_out
        q = p[pi]
        rc = self.node_rc[idx[pi, ki]]
        r0, c0 = rc[:, 0], rc[:, 1]
        # the first valid cell having the node as a corner
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
            step = np.maximum(np.maximum(np.abs(da), np.abs(db))/2., 1.)  # at most two cells per iteration
            da, db = da/step, db/step
            r = np.clip(i+a+da, 0., self.R-1.)
            c = np.clip(j+b+db, 0., self.C-1.)
            ni = np.clip(np.floor(r).astype(np.int64), 0, self.R-2)
            nj = np.clip(np.floor(c).astype(np.int64), 0, self.C-2)
            move = self._ok(ni, nj)
            a = np.where(move, r-ni, np.clip(a+da, 0., 1.))
            b = np.where(move, c-nj, np.clip(b+db, 0., 1.))
            i, j = np.where(move, ni, i), np.where(move, nj, j)
        s, sr, sc = self._eval(i, j, a, b)
        n = unit(np.cross(sc, sr))
        e = q-s
        d = (e*n).sum(1)
        tang = np.linalg.norm(e-d[:, None]*n, axis=1)
        if self.axis is not None:
            radial = s[:, :2]-self.axis.xy(s[:, 2])
            flip = (n[:, :2]*radial).sum(1) < 0
            n[flip] *= -1
            d[flip] *= -1
        cov = np.isfinite(d) & (tang <= TANGENT_TOL) & (np.abs(d) <= reach)
        # per point: the covered candidate closest to it
        score = np.where(cov, np.linalg.norm(e, axis=1), np.inf)
        order = np.lexsort((score, pi))
        first = np.r_[True, pi[order][1:] != pi[order][:-1]]
        best = order[first]
        best = best[np.isfinite(score[best])]
        tgt = pi[best]
        d_out[tgt], cov_out[tgt], foot_out[tgt], n_out[tgt] = d[best], True, s[best], n[best]
        return d_out, cov_out, foot_out, n_out


class SegmentSource:
    """A ground-truth tifxyz segment (local dir or https base URL), loaded on demand."""

    def __init__(self, name, source, unit_base=1.0):
        self.name, self.source, self.unit_base = name, str(source), unit_base
        self.bbox = None
        try:
            meta = self._meta()
            if 'bbox' in meta:
                self.bbox = np.asarray(meta['bbox'], np.float64)*unit_base/SCORE_UNIT_BASE
        except Exception:
            pass

    def _meta(self):
        if self.source.startswith(('http://', 'https://')):
            with urllib.request.urlopen(self.source.rstrip('/')+'/meta.json', timeout=60) as r:
                return json.loads(r.read())
        return json.loads((Path(self.source)/'meta.json').read_text())

    def may_touch(self, points, reach=SEARCH_REACH):
        if self.bbox is None or not len(points):
            return True
        return bool(np.any(np.all((points >= self.bbox[0]-reach) & (points <= self.bbox[1]+reach), axis=1)))

    def load(self, axis=None):
        grid, valid, _ = read_tifxyz(self.source, self.unit_base)
        return GridSurface(grid, valid, self.name, axis)


def discover_segments(paths, urls_file=None, unit_base=1.0):
    """Segment sources from tifxyz dirs (each dir holding x.tif, or a tree of them) and/or a file of base URLs."""
    out = []
    for root in paths or []:
        root = Path(root)
        dirs = [root] if (root/'x.tif').exists() else sorted({p.parent for p in root.rglob('x.tif')})
        for d in dirs:
            name = d.name if not d.name.endswith('.tifxyz') else d.parent.parent.name
            out.append(SegmentSource(name, d, unit_base))
    if urls_file:
        for line in Path(urls_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.rstrip('/').split('/')
                name = parts[parts.index('mesh')-1] if 'mesh' in parts else parts[-1]
                out.append(SegmentSource(name, line, unit_base))
    return out


# ---- Human fibers -------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Fiber:
    name: str
    family: str           # 'H', 'V' or '?'
    family_source: str    # 'manual', 'automatic', 'geometry'
    points: np.ndarray    # dense samples (N, 3) sv, every SAMPLE_STEP along the scored span
    s: np.ndarray         # their arclengths (sv)
    flags: dict

    @property
    def length(self):
        return float(self.s[-1]) if len(self.s) else 0.


def _xyz(p):
    if isinstance(p, dict):
        p = p.get('position', p.get('xyz'))
    return [float(p[0]), float(p[1]), float(p[2])]


def load_fiber_json(path, unit_base=1.0, span='controlled', step=SAMPLE_STEP):
    """A VC3D fiber JSON file as a Fiber in sv (``parse_fiber``), or None when it has no scored span."""
    return parse_fiber(json.loads(Path(path).read_text()), Path(path).stem, unit_base, span, step)


def parse_fiber(raw, name, unit_base=1.0, span='controlled', step=SAMPLE_STEP):
    """A VC3D fiber JSON (v3/v4; line_points xyz) as a Fiber in sv, or None when it has no scored span.

    ``span='controlled'``: the line between the outer human control points (villa's annotations/vc3d.load_fibers
    keeps only that); ``'full'``: every line point. Flags: fold-backs (the line turning > 150 degrees within a few
    voxels) and jumps (a line step > 3x the median step and > 8 sv), for the #2248-style sensitivity analysis."""
    line = np.asarray([_xyz(p) for p in raw.get('line_points', [])], np.float64)*unit_base/SCORE_UNIT_BASE
    controls = np.asarray([_xyz(p) for p in raw.get('control_points', [])], np.float64)*unit_base/SCORE_UNIT_BASE
    flags = dict(version=raw.get('version'), n_line=int(len(line)), n_controls=int(len(controls)),
                 username=raw.get('username'))
    if len(line) < 2:
        return None
    if span == 'controlled':
        if len(controls) < 2:
            return None
        dist, anchors = cKDTree(line).query(controls)
        flags['control_anchor_max_sv'] = float(dist.max())
        if np.any(np.diff(anchors) <= 0) and not np.all(np.diff(anchors) < 0):
            flags['unordered_controls'] = True
        a, b = int(anchors.min()), int(anchors.max())
        if b <= a:
            return None
        line = line[a:b+1]
    steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
    med = float(np.median(steps)) if len(steps) else 0.
    flags['jumps'] = int(np.sum(steps > max(3*med, 8.)))
    pts, s = resample(line, step)
    if len(pts) < 2:
        return None
    t = tangents(pts, 2)
    reach = 3
    if len(t) > 2*reach:
        cos = (t[:-2*reach]*t[2*reach:]).sum(1)
        flags['foldbacks'] = int(np.sum(np.diff(np.r_[0, (cos < math.cos(math.radians(150))).astype(np.int8)]) == 1))
    else:
        flags['foldbacks'] = 0
    hv = raw.get('hv_classification') or {}
    manual, auto = (hv.get('manual_tag') or ''), (hv.get('automatic_tag') or '')
    if manual in ('H', 'V'):
        family, source = manual, 'manual'
    elif auto in ('H', 'V'):
        family, source = auto, 'automatic'
    else:
        dz = abs(pts[-1, 2]-pts[0, 2])
        family, source = ('V' if dz > 0.5*s[-1] else 'H'), 'geometry'
    return Fiber(name, family, source, pts, s, flags)


def _fetch_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read())


def load_fibers(fiber_dir=None, unit_base=1.0, span='controlled', urls_file=None, threads=8):
    """Fibers from a directory of JSONs and/or a file of JSON URLs (read into memory, nothing saved)."""
    items = []
    if fiber_dir:
        items += [(p.stem, (lambda p=p: json.loads(p.read_text()))) for p in sorted(Path(fiber_dir).glob('*.json'))]
    if urls_file:
        urls = [u.strip() for u in Path(urls_file).read_text().splitlines() if u.strip() and not u.startswith('#')]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(threads) as pool:
            raws = list(pool.map(lambda u: _try(_fetch_json, u), urls))
        items += [(u.rstrip('/').split('/')[-1].rsplit('.', 1)[0], (lambda r=r: r)) for u, r in zip(urls, raws)]
    fibers, skipped = [], []
    for name, get in items:
        try:
            raw = get()
            if isinstance(raw, Exception):
                raise raw
            f = parse_fiber(raw, name, unit_base, span)
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
            skipped.append((name, f'unreadable: {type(error).__name__}: {error}'))
            continue
        if f is None:
            skipped.append((name, 'no scored span (fewer than 2 control points or line points)'))
        else:
            fibers.append(f)
    return fibers, skipped


def _try(fn, *args):
    try:
        return fn(*args)
    except Exception as error:  # reported per fiber by load_fibers
        return error


# ---- Fiber selection (PREREG section 4) -----------------------------------------------------------------------

def match_fibers(fibers, segments, axis=None, log=print):
    """For every fiber sample: the segment whose surface is nearest along its normal (covered feet only).
    Returns per-fiber selection records (runs on segments, fractions, noise) and segment names."""
    best_abs = [np.full(len(f.points), np.inf) for f in fibers]
    best_d = [np.full(len(f.points), np.nan) for f in fibers]
    best_seg = [np.full(len(f.points), -1, np.int64) for f in fibers]
    allpts = np.concatenate([f.points for f in fibers]) if fibers else np.zeros((0, 3))
    owner = np.concatenate([np.full(len(f.points), k) for k, f in enumerate(fibers)]) if fibers else np.zeros(0, int)
    offset = np.r_[0, np.cumsum([len(f.points) for f in fibers])]
    names = []
    t0 = time.time()
    for si, seg in enumerate(segments):
        names.append(seg.name)
        if not seg.may_touch(allpts):
            continue
        surf = seg.load(axis)
        near = np.flatnonzero(surf.near_bbox(allpts))
        if len(near):
            d, cov, _, _ = surf.project(allpts[near])
            g = near[cov]
            dd = d[cov]
            fk = owner[g]
            local = g-offset[fk]
            for k in np.unique(fk):
                sel = fk == k
                li, dv = local[sel], dd[sel]
                better = np.abs(dv) < best_abs[k][li]
                best_abs[k][li[better]] = np.abs(dv[better])
                best_d[k][li[better]] = dv[better]
                best_seg[k][li[better]] = si
        log(f'[match] segment {si+1}/{len(segments)} {seg.name}: {len(surf.nodes)} nodes, {len(near)} samples near, '
            f'{time.time()-t0:.0f} s')
        del surf
    records = [selection_record(f, best_d[k], best_seg[k]) for k, f in enumerate(fibers)]
    return records, names


def selection_record(fiber, d, seg):
    n = len(d)
    a = np.abs(d)
    covered = seg >= 0
    cand = covered & (a <= CANDIDATE_MAX_D)
    on = covered & (a <= MATCH_MAX_D)
    runs = []
    k = 0
    while k < n:
        if not on[k]:
            k += 1
            continue
        e = k
        while e+1 < n and on[e+1] and seg[e+1] == seg[k]:
            e += 1
        if fiber.s[e]-fiber.s[k] >= MIN_RUN:
            runs.append(dict(segment=int(seg[k]), s0=float(fiber.s[k]), s1=float(fiber.s[e]),
                             mean_d=float(np.mean(d[k:e+1])), samples=int(e-k+1)))
        k = e+1
    on_samples = sum(r['samples'] for r in runs)
    rec = dict(fiber=fiber.name, family=fiber.family, family_source=fiber.family_source,
               length_sv=fiber.length, length_mm=fiber.length*9.6e-3, samples=n,
               covered_fraction=float(covered.mean()) if n else 0.,
               candidate_fraction=float(cand.mean()) if n else 0.,
               on_fraction=float(on_samples/n) if n else 0.,
               on_fraction_any_run=float(on.mean()) if n else 0.,
               median_abs_d_candidate=float(np.median(a[cand])) if cand.any() else None,
               median_abs_d_on=float(np.median(a[on])) if on.any() else None,
               mean_d_on=float(np.mean(d[on])) if on.any() else None,
               runs=runs, flags=fiber.flags)
    rec['candidate'] = rec['candidate_fraction'] >= MATCH_MIN_FRACTION
    rec['matched'] = rec['on_fraction'] >= MATCH_MIN_FRACTION
    rec['long_enough'] = fiber.length >= MIN_LENGTH_SV
    return rec


def gate_g1(records):
    cand = [r for r in records if r['candidate'] and r['median_abs_d_candidate'] is not None]
    matched = [r for r in records if r['matched'] and r['long_enough']]
    noise = float(np.median([r['median_abs_d_candidate'] for r in cand])) if cand else None
    by_family = {fam: dict(candidates=sum(r['family'] == fam for r in cand),
                           matched=sum(r['family'] == fam for r in matched)) for fam in ('H', 'V', '?')}
    on_noise = [r['median_abs_d_on'] for r in matched if r['median_abs_d_on'] is not None]
    result = dict(
        fibers=len(records), candidates=len(cand), matched_long=len(matched),
        matched_any_length=sum(r['matched'] for r in records), by_family=by_family,
        noise_floor_sv=noise,
        noise_floor_definition='median over candidate fibers of the per-fiber median |normal distance| of its samples '
                               f'within {CANDIDATE_MAX_D} sv of a segment',
        matched_fiber_median_abs_d_sv=float(np.median(on_noise)) if on_noise else None,
        matched_fiber_mean_d_sv=float(np.mean([r['mean_d_on'] for r in matched])) if matched else None,
        sensitivity_matched_at_50pct=sum(r['on_fraction'] >= .5 and r['long_enough'] for r in records),
        flagged_matched=sum(bool(r['flags'].get('foldbacks') or r['flags'].get('jumps')) for r in matched))
    result['pass'] = bool(len(matched) >= G1_MIN_FIBERS and noise is not None and noise <= G1_MAX_NOISE)
    result['rule'] = (f'>= {G1_MIN_FIBERS} matched fibers (>= {MATCH_MIN_FRACTION:.0%} of length within '
                      f'{MATCH_MAX_D} sv of a segment, runs >= {MIN_RUN} sv, length >= 2.5 mm) and noise floor '
                      f'<= {G1_MAX_NOISE} sv')
    return result


# ---- Strips --------------------------------------------------------------------------------------------------

def orient_decisions(heights, frames):
    """villa sheet_strip/infer.py@6f2024569 (bruniss), copied: each decision's v axis turned to agree with the
    previous one's. Returns the oriented heights, frames and which decisions flipped."""
    flipped = np.zeros(len(frames), bool)
    for k in range(1, len(frames)):
        flipped[k] = flipped[k-1] ^ (float(frames[k, :, 1] @ frames[k-1, :, 1]) < 0)
    oriented = np.where(flipped[:, None, None], heights[:, :, ::-1], heights)
    oriented_frames = frames.copy()
    oriented_frames[flipped, :, 1] *= -1
    oriented_frames[flipped, :, 0] *= -1
    oriented[flipped] *= -1
    return oriented, oriented_frames, flipped


def stitch(heights, frames, positions, arcs, f, v, blend, weights=None):
    """villa sheet_strip/infer.py@6f2024569 (bruniss), copied: every decision's strip points averaged into rows at
    arclength arcs[0]+f[0]+r*df, weight cos^2(pi f / 2 blend). Returns rows (R, Nv, 3), weight (R, Nv), spread."""
    df = f[1]-f[0]
    rows = np.rint((arcs[:, None]-arcs[0]+f[None]-f[0])/df).astype(np.int64)
    window = np.where(np.abs(f) < blend, np.cos(np.pi*f/(2*blend))**2, 0.)
    w = window[None, :, None]*(np.ones_like(heights) if weights is None else weights)
    local = np.stack(np.broadcast_arrays(heights, v[None, None], f[None, :, None]), -1)
    world = positions[:, None, None]+np.einsum('kfvj,kij->kfvi', local, frames)
    shape = (int(rows.max())+1, len(v))
    total, first, second = np.zeros(shape), np.zeros((*shape, 3)), np.zeros(shape)
    flat_rows = np.broadcast_to(rows[..., None], heights.shape).ravel()
    flat_cols = np.broadcast_to(np.arange(len(v)), heights.shape).ravel()
    np.add.at(total, (flat_rows, flat_cols), w.ravel())
    np.add.at(first, (flat_rows, flat_cols), w.reshape(-1, 1)*world.reshape(-1, 3))
    np.add.at(second, (flat_rows, flat_cols), w.ravel()*(world**2).sum(-1).ravel())
    mean = first/np.maximum(total, 1e-12)[..., None]
    spread = np.sqrt(np.maximum(second/np.maximum(total, 1e-12)-(mean**2).sum(-1), 0.))
    return mean, total, spread


def stitch_values(values, arcs, f, blend, weights=None):
    """villa sheet_strip/infer.py@6f2024569 (bruniss), copied: the stitch weights' average of a per-point value."""
    df = f[1]-f[0]
    rows = np.rint((arcs[:, None]-arcs[0]+f[None]-f[0])/df).astype(np.int64)
    window = np.where(np.abs(f) < blend, np.cos(np.pi*f/(2*blend))**2, 0.)
    w = window[None, :, None]*(np.ones_like(values) if weights is None else weights)
    shape = (int(rows.max())+1, values.shape[2])
    total, first = np.zeros(shape), np.zeros(shape)
    flat_rows = np.broadcast_to(rows[..., None], values.shape).ravel()
    flat_cols = np.broadcast_to(np.arange(values.shape[2]), values.shape).ravel()
    np.add.at(total, (flat_rows, flat_cols), w.ravel())
    np.add.at(first, (flat_rows, flat_cols), (w*values).ravel())
    return np.where(total > 0, first/np.maximum(total, 1e-12), np.nan)


@dataclasses.dataclass
class Strip:
    name: str
    path: str
    rows: np.ndarray             # (R, C, 3) sv, rows along the fiber, NaN off the strip
    valid: np.ndarray            # (R, C)
    expected: np.ndarray | None  # (R, C) sv, the model's expected |error|
    frame_rows: np.ndarray | None  # (R, C, 3) sv: the decisions' frame planes (u = 0), stitched like the strip
    trace: dict
    column_step: float           # sv between columns
    notes: list


def find_decisions(tifxyz_dir):
    d = Path(tifxyz_dir)
    for cand in (d/'decisions.npz', d/'strip.npz', d.parent/'strip.npz'):
        if cand.exists():
            return cand
    return None


def load_strip(tifxyz_dir, strip_unit_base=4.0):
    """A strip tifxyz (infer.py OUT/strip or infer_all OUT/strips/B/fiber_ID) as a Strip, rows along the fiber.
    The decisions (infer.py's strip.npz: arcs, positions, frames, heights, f, v, trace_per_patch; beside the
    tifxyz or in its parent) give the frame-plane baseline; without them it is unavailable."""
    import tifffile
    d = Path(tifxyz_dir)
    scale = strip_unit_base/SCORE_UNIT_BASE
    grid, valid, meta = read_tifxyz(d, strip_unit_base)
    trace = {}
    for cand in (d/'trace.json', d.parent/'summary.json'):
        if cand.exists():
            trace = {**json.loads(cand.read_text()), **trace}
    notes = []
    expected = None
    if (d/'expected_error.tif').exists():
        expected = np.asarray(tifffile.imread(d/'expected_error.tif'), np.float64)*scale
    transposed = trace.get('fiber_along') == 'columns'
    if transposed:  # infer_all display layout of an H fiber: back to rows along the fiber
        grid, valid = np.swapaxes(grid, 0, 1), valid.T
        expected = None if expected is None else expected.T
    frame_rows = None
    npz = find_decisions(d)
    if npz is not None:
        try:
            frame_rows, exp2, note = frames_from_decisions(npz, grid, valid, trace, strip_unit_base)
            notes += note
            if expected is None and exp2 is not None:
                expected = exp2
        except Exception as error:  # a broken decisions file must not stop the scoring of the strip itself
            notes.append(f'decisions unusable ({type(error).__name__}: {error}); frame-plane baseline unavailable')
    else:
        notes.append('no decisions (strip.npz) beside the strip: frame-plane baseline unavailable')
    if expected is not None:
        expected = np.where(valid, expected, np.nan)
    steps = np.linalg.norm(np.diff(grid, axis=1), axis=-1)
    column_step = float(np.nanmedian(steps)) if np.isfinite(steps).any() else 5.
    name = d.name if d.name not in ('strip',) else d.parent.name
    return Strip(name, str(d), grid, valid, expected, frame_rows, trace, column_step, notes)


def frames_from_decisions(npz_path, grid, valid, trace, strip_unit_base):
    """The frame-plane nodes (u = 0 in every decision's frame, stitched as the strip) on the strip's grid; nodes are
    paired with the tifxyz by position (the stitched heights reproduce the tifxyz nodes)."""
    z = np.load(npz_path)
    scale = strip_unit_base/SCORE_UNIT_BASE
    tpp = float(z['trace_per_patch'])
    blend = float(trace.get('blend', 16.))
    arcs, f, v = z['arcs'].astype(np.float64), z['f'].astype(np.float64), z['v'].astype(np.float64)
    heights, frames, positions = z['heights'].astype(np.float64), z['frames'].astype(np.float64), \
        z['positions'].astype(np.float64)
    oriented, oframes, flipped = orient_decisions(heights, frames)
    rows, weight, _ = stitch(oriented, oframes, positions, arcs, f, v, blend)
    plane, _, _ = stitch(np.zeros_like(oriented), oframes, positions, arcs, f, v, blend)
    rows, plane = rows/tpp*scale, plane/tpp*scale
    ok = weight > 0
    tree = cKDTree(rows[ok])
    rc = np.argwhere(ok)
    target = np.argwhere(valid)
    dist, idx = tree.query(grid[valid], k=1)
    good = dist <= 0.05  # sv; float32 tifxyz rounding is ~1e-3 at these coordinates
    notes = []
    if good.mean() < 0.99:
        notes.append(f'decisions reproduce only {good.mean():.1%} of the strip nodes: frame-plane baseline unavailable')
        return None, None, notes
    out = np.full(grid.shape, np.nan)
    src = rc[idx]
    out[target[:, 0], target[:, 1]] = plane[src[:, 0], src[:, 1]]
    out[target[~good, 0], target[~good, 1]] = np.nan
    expected = None
    if 'expected_error_vox' in z.files:
        ev = z['expected_error_vox'].astype(np.float64)
        ev = np.where(flipped[:, None, None], ev[:, :, ::-1], ev)
        stitched = stitch_values(ev, arcs, f, blend)*scale
        expected = np.full(grid.shape[:2], np.nan)
        expected[target[:, 0], target[:, 1]] = stitched[src[:, 0], src[:, 1]]
    return out, expected, notes


def discover_strips(root):
    root = Path(root)
    dirs = sorted({p.parent for p in root.rglob('x.tif')})
    return [d for d in dirs if d.name != 'patch_sheet']


# ---- Strip <-> fiber association and node sets -------------------------------------------------------------------

class FiberIndex:
    def __init__(self, fibers):
        self.fibers = fibers
        self.by_name = {f.name: k for k, f in enumerate(fibers)}
        pts = [f.points for f in fibers]
        self.owner = np.concatenate([np.full(len(p), k) for k, p in enumerate(pts)]) if pts else np.zeros(0, int)
        self.local = np.concatenate([np.arange(len(p)) for p in pts]) if pts else np.zeros(0, int)
        self.tree = cKDTree(np.concatenate(pts)) if pts else None
        self.trees = {}

    def tree_of(self, k):
        if k not in self.trees:
            self.trees[k] = cKDTree(self.fibers[k].points)
        return self.trees[k]


def row_centres(strip):
    c = strip.rows.shape[1]//2
    centre = strip.rows[:, c].copy()
    bad = ~strip.valid[:, c]
    if bad.any():
        import warnings
        with warnings.catch_warnings():  # rows with no valid node stay NaN
            warnings.simplefilter('ignore', RuntimeWarning)
            centre[bad] = np.nanmean(np.where(strip.valid[bad][..., None], strip.rows[bad], np.nan), axis=1)
    return centre


def associate(strip, index: FiberIndex):
    """The fiber a strip belongs to: the one its row centres run closest to (median); None if > ASSOC_MAX_SV."""
    centre = row_centres(strip)
    ok = np.isfinite(centre).all(1)
    if not ok.any() or index.tree is None:
        return None, math.inf
    dist, idx = index.tree.query(centre[ok], k=1)
    owners = index.owner[idx]
    best, best_med = None, math.inf
    for k in np.unique(owners):
        dk, _ = index.tree_of(int(k)).query(centre[ok], k=1)
        med = float(np.median(dk))
        if med < best_med:
            best, best_med = int(k), med
    if best_med > ASSOC_MAX_SV:
        return None, best_med
    return best, best_med


# ---- Lasagna normals -----------------------------------------------------------------------------------------

class ZarrV2Chunks:
    """Minimal read-only zarr v2 array (local dir or https URL): whole chunks, decoded with numcodecs, cached."""

    def __init__(self, source):
        self.source = str(source).rstrip('/')
        self.meta = json.loads(self._get('.zarray'))
        self.shape, self.chunks = tuple(self.meta['shape']), tuple(self.meta['chunks'])
        self.dtype = np.dtype(self.meta['dtype'])
        self.fill = self.meta.get('fill_value') or 0
        self.sep = self.meta.get('dimension_separator', '.')
        self.codec = None
        if self.meta.get('compressor'):
            import numcodecs
            self.codec = numcodecs.get_codec(self.meta['compressor'])
        self.cache = {}

    def _get(self, key):
        if self.source.startswith(('http://', 'https://')):
            try:
                with urllib.request.urlopen(self.source+'/'+key, timeout=120) as r:
                    return r.read()
            except urllib.error.HTTPError as error:
                if error.code in (403, 404):
                    return None
                raise
        p = Path(self.source)/key
        return p.read_bytes() if p.exists() else None

    def chunk(self, cz, cy, cx):
        key = (cz, cy, cx)
        if key not in self.cache:
            raw = self._get(self.sep.join(map(str, key)))
            if raw is None:
                arr = np.full(self.chunks, self.fill, self.dtype)
            else:
                data = self.codec.decode(raw) if self.codec is not None else raw
                arr = np.frombuffer(data, self.dtype).reshape(self.chunks)
            self.cache[key] = arr
        return self.cache[key]

    def take(self, zyx):
        """Values at integer voxel indices (M, 3) zyx; fill outside the array."""
        zyx = np.asarray(zyx, np.int64)
        out = np.full(len(zyx), self.fill, self.dtype)
        inside = np.all((zyx >= 0) & (zyx < np.array(self.shape)), axis=1)
        ch = zyx//np.array(self.chunks)
        keys = {tuple(int(v) for v in c) for c in ch[inside]}
        todo = [k for k in keys if k not in self.cache]
        if len(todo) > 1:  # fetch missing chunks in parallel (remote stores: latency-bound)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(8) as pool:
                list(pool.map(lambda k: self.chunk(*k), todo))
        for key in keys:
            sel = inside & np.all(ch == key, axis=1)
            loc = zyx[sel]-np.array(key)*np.array(self.chunks)
            out[sel] = self.chunk(*key)[loc[:, 0], loc[:, 1], loc[:, 2]]
        return out


def decode_lasagna(nx_u8, ny_u8):
    """Lasagna's uint8 hemisphere normals (villa lasagna: (u8 - 128)/127, nz = sqrt(1 - nx^2 - ny^2)) as unit xyz
    vectors; valid where nx or ny is set."""
    nx = (np.asarray(nx_u8, np.float64)-128.)/127.
    ny = (np.asarray(ny_u8, np.float64)-128.)/127.
    nz = np.sqrt(np.clip(1.-nx*nx-ny*ny, 0., None))
    valid = (np.asarray(nx_u8) != 0) | (np.asarray(ny_u8) != 0)
    return unit(np.stack([nx, ny, nz], -1)), valid


def lasagna_normals_at(points_sv, nx, ny, level_scale=LASAGNA_SCALE):
    """Unoriented Lasagna normals (M, 3) at points (M, 3) sv: the principal axis of the trilinearly weighted outer
    products of the 8 neighbouring voxels' normals (sign-free); valid (M,) where any neighbour has a normal.
    ``nx``/``ny``: objects with ``take(zyx)`` (ZarrV2Chunks) or numpy arrays (z, y, x) at ``level_scale`` base
    voxels per voxel."""
    take = (lambda a, zyx: a.take(zyx)) if hasattr(nx, 'take') and not isinstance(nx, np.ndarray) else \
        (lambda a, zyx: _np_take(a, zyx))
    q = np.asarray(points_sv, np.float64)*SCORE_UNIT_BASE/level_scale-0.5  # voxel-centre coordinates, xyz
    base = np.floor(q).astype(np.int64)
    frac = q-base
    tensor = np.zeros((len(q), 3, 3))
    wsum = np.zeros(len(q))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (np.where(dx, frac[:, 0], 1-frac[:, 0])*np.where(dy, frac[:, 1], 1-frac[:, 1])
                     * np.where(dz, frac[:, 2], 1-frac[:, 2]))
                idx = base+np.array([dx, dy, dz])
                zyx = idx[:, ::-1]
                n, ok = decode_lasagna(take(nx, zyx), take(ny, zyx))
                w = np.where(ok, w, 0.)
                tensor += w[:, None, None]*n[:, :, None]*n[:, None, :]
                wsum += w
    valid = wsum > 1e-9
    out = np.full((len(q), 3), np.nan)
    if valid.any():
        _, vecs = np.linalg.eigh(tensor[valid])
        out[valid] = vecs[:, :, -1]
    return out, valid


def _np_take(a, zyx):
    zyx = np.asarray(zyx, np.int64)
    out = np.zeros(len(zyx), a.dtype)
    inside = np.all((zyx >= 0) & (zyx < np.array(a.shape)), axis=1)
    out[inside] = a[zyx[inside, 0], zyx[inside, 1], zyx[inside, 2]]
    return out


def lasagna_plane_rows(fiber, row_s, ncols, column_step, normals, nvalid):
    """The Lasagna-normal plane baseline on a strip's grid: for each row at fiber arclength ``row_s``, the plane
    through the fiber point holding the fiber tangent, normal to the Lasagna normal there (made perpendicular to
    the tangent); nodes at the strip's column offsets across the fiber. NaN where no Lasagna normal is near."""
    k = np.clip(np.searchsorted(fiber.s, row_s), 0, len(fiber.s)-1)
    good = np.flatnonzero(nvalid)
    out = np.full((len(row_s), ncols, 3), np.nan)
    if not len(good):
        return out
    pos = np.searchsorted(good, k)
    lo, hi = good[np.clip(pos-1, 0, len(good)-1)], good[np.clip(pos, 0, len(good)-1)]
    nearest = np.where(np.abs(lo-k) <= np.abs(hi-k), lo, hi)
    usable = np.abs(fiber.s[nearest]-fiber.s[k]) <= 10.
    t = tangents(fiber.points)[k]
    n = normals[nearest]
    n = unit(n-(n*t).sum(1, keepdims=True)*t)
    b = unit(np.cross(t, n))
    offs = (np.arange(ncols)-ncols//2)*column_step
    rows = fiber.points[k][:, None, :]+offs[None, :, None]*b[:, None, :]
    out[usable] = rows[usable]
    return out


# ---- Scoring (PREREG section 5) ---------------------------------------------------------------------------------

SURFACES = ('strip', 'frame_plane', 'lasagna_plane')


def score_strips(strips, fibers, records, segments, axis=None, normals=None, log=print):
    """Per fiber (matched, with a strip): signed normal distances of strip / baseline nodes to the matched segment
    surface, on strip rows inside the fiber's matched runs. Returns per-fiber node arrays and a log of skips."""
    index = FiberIndex(fibers)
    rec_by_name = {r['fiber']: r for r in records}
    jobs = {}       # segment index -> list of (fiber_key, surface, node positions (M,3), node ids)
    per_fiber = {}
    skips = []
    for strip in strips:
        k, med = associate(strip, index)
        if k is None:
            skips.append(dict(strip=strip.name, reason=f'no fiber within {ASSOC_MAX_SV} sv (median {med:.1f} sv): '
                                                       'unit or association error'))
            continue
        fiber = fibers[k]
        rec = rec_by_name.get(fiber.name)
        if rec is None or not rec['matched']:
            skips.append(dict(strip=strip.name, fiber=fiber.name, reason='fiber not matched to a segment'))
            continue
        if fiber.name in per_fiber:
            skips.append(dict(strip=strip.name, fiber=fiber.name, reason='second strip of the same fiber (ignored)'))
            continue
        centre = row_centres(strip)
        okc = np.isfinite(centre).all(1)
        row_s = np.full(len(centre), np.nan)
        row_d = np.full(len(centre), np.inf)
        if okc.any():
            dd, ii = index.tree_of(k).query(centre[okc], k=1)
            row_s[okc], row_d[okc] = fiber.s[ii], dd
        row_seg = np.full(len(centre), -1)
        for run in rec['runs']:
            inside = (row_s >= run['s0']) & (row_s <= run['s1']) & (row_d <= ROW_MAX_SV)
            row_seg[inside] = run['segment']
        nodes = {'strip': strip.rows}
        if strip.frame_rows is not None:
            nodes['frame_plane'] = strip.frame_rows
        if normals is not None and fiber.name in normals:
            nv, nvalid = normals[fiber.name]
            ls = np.where(np.isfinite(row_s), row_s, 0.)
            lrows = lasagna_plane_rows(fiber, ls, strip.rows.shape[1], strip.column_step, nv, nvalid)
            lrows[~np.isfinite(row_s)] = np.nan
            nodes['lasagna_plane'] = lrows
        R, C = strip.valid.shape
        entry = dict(strip=strip.name, fiber=fiber.name, family=fiber.family, assoc_median_sv=med,
                     rows=R, cols=C, notes=strip.notes,
                     err={s: np.full((R, C), np.nan) for s in nodes},
                     expected=strip.expected, valid=strip.valid, row_seg=row_seg,
                     fiber_floor=np.full(R, np.nan), lasagna_angle=np.full(R, np.nan))
        per_fiber[fiber.name] = entry
        fiber_at_row = fiber.points[np.clip(np.searchsorted(fiber.s, np.nan_to_num(row_s)), 0, len(fiber.s)-1)]
        for seg in np.unique(row_seg[row_seg >= 0]):
            rr = np.flatnonzero(row_seg == seg)
            for surf, grid in nodes.items():
                mask = np.zeros((R, C), bool)
                mask[rr] = strip.valid[rr] & np.isfinite(grid[rr]).all(-1)
                rc = np.argwhere(mask)
                jobs.setdefault(int(seg), []).append((fiber.name, surf, grid[mask], rc))
            jobs.setdefault(int(seg), []).append((fiber.name, '_floor', fiber_at_row[rr], rr))
    t0 = time.time()
    for seg in sorted(jobs):
        surf = segments[seg].load(axis)
        for fname, kind, pts, ids in jobs[seg]:
            d, cov, _, nrm = surf.project(pts)
            e = per_fiber[fname]
            if kind == '_floor':
                e['fiber_floor'][ids] = np.where(cov, d, np.nan)
                if normals is not None and fname in normals:
                    nv, nval = normals[fname]
                    _, kk = index.tree_of(index.by_name[fname]).query(pts, k=1)
                    ang = np.degrees(np.arccos(np.clip(np.abs((nv[kk]*nrm).sum(1)), 0, 1)))
                    e['lasagna_angle'][ids] = np.where(cov & nval[kk], ang, np.nan)
            else:
                e['err'][kind][ids[:, 0], ids[:, 1]] = np.where(cov, d, np.nan)
        log(f'[score] segment {segments[seg].name}: {len(jobs[seg])} node sets, {time.time()-t0:.0f} s')
        del surf
    return per_fiber, skips


def pooled(values_list):
    v = np.concatenate([np.abs(x) for x in values_list]) if values_list else np.zeros(0)
    if not len(v):
        return None
    return dict(nodes=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                within1=float((v <= 1).mean()), within2=float((v <= 2).mean()))


def bootstrap_stats(per_fiber_abs, seed=SEED, B=BOOTSTRAP, B_pct=BOOTSTRAP_PCT):
    """Cluster bootstrap over fibers (each replicate: fibers drawn with replacement, nodes pooled). Returns the
    point estimates and 95% percentile CIs of mean, median, p90, within1, within2."""
    n = len(per_fiber_abs)
    point = pooled(per_fiber_abs)
    if point is None or n < 2:
        return dict(point=point, ci=None, fibers=n)
    rng = np.random.default_rng(seed)
    cnt = np.array([len(x) for x in per_fiber_abs], np.float64)
    sums = np.array([x.sum() for x in per_fiber_abs])
    w1 = np.array([(x <= 1).sum() for x in per_fiber_abs], np.float64)
    w2 = np.array([(x <= 2).sum() for x in per_fiber_abs], np.float64)
    draws = rng.integers(0, n, size=(B, n))
    tot = cnt[draws].sum(1)
    reps = dict(mean=sums[draws].sum(1)/tot, within1=w1[draws].sum(1)/tot, within2=w2[draws].sum(1)/tot)
    med, p90 = np.empty(B_pct), np.empty(B_pct)
    for b in range(B_pct):
        v = np.concatenate([per_fiber_abs[i] for i in draws[b]])
        med[b], p90[b] = np.percentile(v, [50, 90])
    reps.update(median=med, p90=p90)
    ci = {k: [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))] for k, r in reps.items()}
    return dict(point=point, ci=ci, fibers=n, replicates=dict(B=B, B_percentiles=B_pct, seed=seed))


def paired_delta(a_list, b_list, seed=SEED, B=BOOTSTRAP):
    """Baseline (b) minus strip (a) on the same fibers and nodes: mean |d| difference and within-2 difference
    (strip - baseline), cluster bootstrap over fibers with shared draws."""
    n = len(a_list)
    if n < 2:
        return None
    rng = np.random.default_rng(seed+1)
    cnt = np.array([len(x) for x in a_list], np.float64)
    sa, sb = np.array([x.sum() for x in a_list]), np.array([x.sum() for x in b_list])
    wa, wb = np.array([(x <= 2).sum() for x in a_list], float), np.array([(x <= 2).sum() for x in b_list], float)
    draws = rng.integers(0, n, size=(B, n))
    tot = cnt[draws].sum(1)
    dmean = (sb[draws].sum(1)-sa[draws].sum(1))/tot
    dw2 = (wa[draws].sum(1)-wb[draws].sum(1))/tot
    point_mean = float((sb.sum()-sa.sum())/cnt.sum())
    point_w2 = float((wa.sum()-wb.sum())/cnt.sum())
    ci_mean = [float(np.percentile(dmean, 2.5)), float(np.percentile(dmean, 97.5))]
    ci_w2 = [float(np.percentile(dw2, 2.5)), float(np.percentile(dw2, 97.5))]
    return dict(fibers=n, nodes=int(cnt.sum()),
                baseline_minus_strip_mean_sv=point_mean, ci=ci_mean, beaten=bool(ci_mean[0] > 0),
                strip_minus_baseline_within2=point_w2, ci_within2=ci_w2)


def aggregate(per_fiber, family=None, seed=SEED, names=None, label=None):
    """Pooled accuracy of the strips and baselines over the scored fibers (optionally one family, or a subset of
    fiber names for the sensitivity analyses)."""
    entries = [e for e in per_fiber.values() if (family is None or e['family'] == family)
               and (names is None or e['fiber'] in names)]
    out = dict(family=family or 'all', subset=label)
    on_rows = sum(int(e['valid'][e['row_seg'] >= 0].sum()) for e in entries)
    covered = sum(int(np.isfinite(e['err']['strip']).sum()) for e in entries)
    out['strip_coverage'] = dict(nodes_on_scored_rows=on_rows, covered=covered,
                                 fraction=covered/on_rows if on_rows else None)
    strip_abs, signed, keep = [], [], []
    for e in entries:
        v = e['err']['strip'][np.isfinite(e['err']['strip'])]
        if len(v) >= MIN_NODES:
            keep.append(e)
            strip_abs.append(np.abs(v))
            signed.append(v)
    out['scored_fibers'] = len(keep)
    out['strip'] = bootstrap_stats(strip_abs, seed)
    out['strip_signed_mean_sv'] = float(np.mean(np.concatenate(signed))) if signed else None
    floors = [np.abs(e['fiber_floor'][np.isfinite(e['fiber_floor'])]) for e in keep]
    floors = [x for x in floors if len(x)]
    out['fiber_floor'] = pooled(floors)
    out['fiber_floor_note'] = 'the human fiber itself, one point per scored row: the ground truth noise floor'
    for base in ('frame_plane', 'lasagna_plane'):
        a_list, b_list = [], []
        for e in keep:
            if base not in e['err']:
                continue
            both = np.isfinite(e['err']['strip']) & np.isfinite(e['err'][base])
            if both.sum() >= MIN_NODES:
                a_list.append(np.abs(e['err']['strip'][both]))
                b_list.append(np.abs(e['err'][base][both]))
        out[base] = dict(available_fibers=len(b_list), stats=bootstrap_stats(b_list, seed) if b_list else None,
                         strip_on_same_nodes=pooled(a_list) if a_list else None,
                         delta=paired_delta(a_list, b_list, seed) if len(a_list) >= 2 else None)
    angles = np.concatenate([e['lasagna_angle'][np.isfinite(e['lasagna_angle'])] for e in keep]) if keep else []
    out['lasagna_vs_segment_normal_deg'] = (dict(median=float(np.median(angles)), p90=float(np.percentile(angles, 90)))
                                            if len(angles) else None)
    return out


def calibration(per_fiber):
    pred, obs, fp, fo = [], [], [], []
    for e in per_fiber.values():
        if e['expected'] is None:
            continue
        err = e['err']['strip']
        ok = np.isfinite(err) & np.isfinite(e['expected'])
        if ok.sum() < MIN_NODES:
            continue
        pred.append(e['expected'][ok])
        obs.append(np.abs(err[ok]))
        fp.append(e['expected'][ok].mean())
        fo.append(np.abs(err[ok]).mean())
    if not pred:
        return None
    from scipy.stats import spearmanr
    p, o = np.concatenate(pred), np.concatenate(obs)
    bins = []
    for lo, hi in zip(CAL_EDGES[:-1], CAL_EDGES[1:]):
        sel = (p >= lo) & (p < hi)
        bins.append(dict(lo=lo, hi=hi, nodes=int(sel.sum()),
                         mean_expected=float(p[sel].mean()) if sel.any() else None,
                         mean_observed=float(o[sel].mean()) if sel.any() else None,
                         median_observed=float(np.median(o[sel])) if sel.any() else None,
                         within2=float((o[sel] <= 2).mean()) if sel.any() else None))
    filled = [b for b in bins if b['nodes']]
    ece = sum(b['nodes']*abs(b['mean_observed']-b['mean_expected']) for b in filled)/len(p)
    rho = spearmanr(p, o).statistic if len(p) > 2 else float('nan')
    rho_f = spearmanr(fp, fo).statistic if len(fp) > 2 else float('nan')
    return dict(nodes=int(len(p)), fibers=len(fp), bins=bins, weighted_abs_gap_sv=float(ece),
                observed_over_expected=float(o.mean()/p.mean()) if p.mean() > 0 else None,
                spearman_node=float(rho), spearman_fiber=float(rho_f),
                note='expected_error is the model\'s predicted |error| along its frame normal (patch voxels = sv); '
                     'observed is |normal distance| to the matched segment')


def gate_g2(agg, g1):
    s = agg['strip']['point']
    checks = dict(g1_pass=bool(g1.get('pass')) if g1 else False)
    checks['scored_fibers_ge_30'] = agg['scored_fibers'] >= G1_MIN_FIBERS
    checks['mean_le_1.5'] = bool(s is not None and s['mean'] <= G2_MAX_MEAN)
    checks['within2_ge_0.70'] = bool(s is not None and s['within2'] >= G2_MIN_WITHIN2)
    for base in ('frame_plane', 'lasagna_plane'):
        d = agg[base]['delta']
        checks[f'beats_{base}'] = bool(d is not None and d['beaten'])
    if not (checks['g1_pass'] and checks['scored_fibers_ge_30']):
        verdict = 'no accuracy claim (G1 failed or fewer than 30 scored fibers)'
    elif all(checks.values()):
        verdict = 'transfers'
    else:
        verdict = 'degrades'
    return dict(verdict=verdict, checks=checks,
                rule=f'transfers = pooled mean |d| <= {G2_MAX_MEAN} sv AND >= {G2_MIN_WITHIN2:.0%} of nodes within 2 '
                     'sv AND both baselines beaten (paired fiber-bootstrap 95% CI of baseline - strip mean |d| above 0)'
                     '; otherwise degrades')


# ---- Crossing agreement (PREREG section 7) ------------------------------------------------------------------------

def crossing_agreement(strips, fibers, seed=SEED, log=print):
    """Where an H strip's fiber and a V strip's fiber cross (closest approach <= CROSS_TOL sv), the normal distance
    between the two predicted sheets: each strip's nodes within CROSS_RADIUS of the crossing projected onto the
    other strip's surface; per crossing the mean of the two medians. No ground truth."""
    index = FiberIndex(fibers)
    assoc = []
    for st in strips:
        k, med = associate(st, index)
        if k is not None:
            assoc.append((st, k))
    H = [(st, k) for st, k in assoc if fibers[k].family == 'H']
    V = [(st, k) for st, k in assoc if fibers[k].family == 'V']
    if not H or not V:
        return dict(crossings=0, note='needs strips of both families')
    vpts = np.concatenate([fibers[k].points for _, k in V])
    vown = np.concatenate([np.full(len(fibers[k].points), j) for j, (_, k) in enumerate(V)])
    vtree = cKDTree(vpts)
    surfaces = {}

    def surface(st):
        if st.path not in surfaces:
            surfaces[st.path] = GridSurface(st.rows, st.valid, st.name)
        return surfaces[st.path]

    rows = []
    for st_h, kh in H:
        fh = fibers[kh]
        hits = vtree.query_ball_point(fh.points, CROSS_TOL)
        pairs = {}
        for i, lst in enumerate(hits):
            for j in lst:
                vj = int(vown[j])
                dist = float(np.linalg.norm(fh.points[i]-vpts[j]))
                if vj not in pairs or dist < pairs[vj][0]:
                    pairs[vj] = (dist, (fh.points[i]+vpts[j])/2)
        for vj, (dist, point) in pairs.items():
            st_v, kv = V[vj]
            meds = []
            for a, b in ((st_h, st_v), (st_v, st_h)):
                near = a.valid & (np.linalg.norm(a.rows-point, axis=-1) <= CROSS_RADIUS)
                if near.sum() < 5:
                    continue
                d, cov, _, _ = surface(b).project(a.rows[near])
                if cov.sum() >= 5:
                    meds.append(float(np.median(np.abs(d[cov]))))
            if meds:
                rows.append(dict(h_fiber=fh.name, v_fiber=fibers[kv].name, closest_approach_sv=dist,
                                 point_sv=point.tolist(), agreement_sv=float(np.mean(meds)), directions=len(meds)))
    if not rows:
        return dict(crossings=0, note='no H/V crossings with overlapping strips')
    a = np.array([r['agreement_sv'] for r in rows])
    rng = np.random.default_rng(seed+2)
    draws = rng.integers(0, len(a), size=(BOOTSTRAP, len(a)))
    reps = a[draws]
    ci = dict(mean=[float(np.percentile(reps.mean(1), q)) for q in (2.5, 97.5)],
              median=[float(np.percentile(np.median(reps, 1), q)) for q in (2.5, 97.5)],
              within2=[float(np.percentile((reps <= 2).mean(1), q)) for q in (2.5, 97.5)])
    return dict(label='no ground truth: consistency between two predictions, not accuracy',
                crossings=len(rows), fibers_h=len({r['h_fiber'] for r in rows}),
                fibers_v=len({r['v_fiber'] for r in rows}),
                mean=float(a.mean()), median=float(np.median(a)), p90=float(np.percentile(a, 90)),
                within2=float((a <= 2).mean()), ci=ci, per_crossing=rows)


# ---- CLI ---------------------------------------------------------------------------------------------------------

def provenance(args):
    here = Path(__file__).resolve()
    prereg = here.parent/'PREREG.md'
    return dict(scorer=str(here), scorer_version=SCORER_VERSION, scorer_sha256=sha256_file(here),
                prereg_sha256=sha256_file(prereg) if prereg.exists() else None,
                argv=sys.argv[1:], constants=CONSTANTS, time=time.strftime('%Y-%m-%d %H:%M:%S'),
                units=dict(score_unit_base=SCORE_UNIT_BASE, fiber_unit_base=getattr(args, 'fiber_unit_base', None),
                           segment_unit_base=getattr(args, 'segment_unit_base', None),
                           strip_unit_base=getattr(args, 'strip_unit_base', None)))


def load_axis(args):
    if getattr(args, 'umbilicus', None):
        return Umbilicus.from_json(args.umbilicus, args.umbilicus_unit_base)
    return None


def load_normals(path):
    if not path:
        return None
    z = np.load(path)
    out = {}
    for key in z.files:
        if key.endswith('__n'):
            name = key[:-3]
            out[name] = (z[key], z[name+'__valid'])
    return out


def cmd_match(args):
    fibers, skipped = load_fibers(args.fibers, args.fiber_unit_base, args.span, args.fiber_urls)
    segs = discover_segments(args.segments, args.segment_urls, args.segment_unit_base)
    print(f'{len(fibers)} fibers ({len(skipped)} skipped), {len(segs)} segments', flush=True)
    records, names = match_fibers(fibers, segs, load_axis(args))
    g1 = gate_g1(records)
    out = dict(provenance=provenance(args), segments=[dict(name=s.name, source=s.source) for s in segs],
               skipped=skipped, G1=g1, fibers=records)
    Path(args.out).write_text(json.dumps(jsonable(out), indent=1))
    print(json.dumps(jsonable(g1), indent=1))


def cmd_lasagna(args):
    fibers, _ = load_fibers(args.fibers, args.fiber_unit_base, args.span, args.fiber_urls)
    nx, ny = ZarrV2Chunks(args.nx), ZarrV2Chunks(args.ny)
    arrays = {}
    t0 = time.time()
    for k, f in enumerate(fibers):
        n, ok = lasagna_normals_at(f.points, nx, ny, args.level_scale)
        arrays[f.name+'__n'] = n.astype(np.float32)
        arrays[f.name+'__valid'] = ok
        if (k+1) % 25 == 0:
            print(f'[lasagna] {k+1}/{len(fibers)} fibers, {len(nx.cache)} chunks, {time.time()-t0:.0f} s', flush=True)
        if len(nx.cache) > 4000:  # bound memory: ~32 KiB per chunk and array
            nx.cache.clear()
            ny.cache.clear()
    np.savez_compressed(args.out, **arrays)
    print(f'[lasagna] wrote {args.out}: {len(fibers)} fibers', flush=True)


def cmd_score(args):
    fibers, skipped = load_fibers(args.fibers, args.fiber_unit_base, args.span, args.fiber_urls)
    matches = json.loads(Path(args.matches).read_text())
    segs = discover_segments(args.segments, args.segment_urls, args.segment_unit_base)
    names = [s.name for s in segs]
    if names != [s['name'] for s in matches['segments']]:
        raise SystemExit('the segments differ from those of the matches file (same list, same order required)')
    strips = [load_strip(d, args.strip_unit_base) for d in discover_strips(args.strips)]
    print(f'{len(strips)} strips, {len(fibers)} fibers, {len(segs)} segments', flush=True)
    normals = load_normals(args.lasagna_normals)
    axis = load_axis(args)
    per_fiber, skips = score_strips(strips, fibers, matches['fibers'], segs, axis, normals)
    g1 = matches['G1']
    report = dict(provenance=provenance(args), G1=g1, skips=skips)
    report['accuracy'] = {fam or 'all': aggregate(per_fiber, fam) for fam in (None, 'H', 'V')}
    report['G2'] = gate_g2(report['accuracy']['all'], g1)
    flags = {r['fiber']: r.get('flags', {}) for r in matches['fibers']}
    unflagged = {n for n, f in flags.items() if not f.get('foldbacks') and not f.get('jumps')}
    sens = {'unflagged (no fold-back or jump flag)': aggregate(per_fiber, names=unflagged, label='unflagged')}
    for prefix in sorted({n.split('_')[0] for n in per_fiber}):
        sens[f'annotator {prefix}'] = aggregate(per_fiber, names={n for n in per_fiber if n.split('_')[0] == prefix},
                                                label=f'annotator {prefix}')
    report['sensitivity'] = dict(note='reported, not gating (PREREG 4.9)', **sens)
    report['calibration'] = calibration(per_fiber)
    report['crossings'] = crossing_agreement(strips, fibers) if args.crossings else None
    report['per_fiber'] = [dict(fiber=e['fiber'], strip=e['strip'], family=e['family'],
                                assoc_median_sv=e['assoc_median_sv'], notes=e['notes'],
                                **{f'{s}_nodes': int(np.isfinite(e['err'][s]).sum()) for s in e['err']},
                                **{f'{s}_mean_abs_sv': (float(np.nanmean(np.abs(e['err'][s])))
                                                        if np.isfinite(e['err'][s]).any() else None)
                                   for s in e['err']},
                                strip_signed_mean_sv=(float(np.nanmean(e['err']['strip']))
                                                      if np.isfinite(e['err']['strip']).any() else None),
                                fiber_floor_mean_abs_sv=(float(np.nanmean(np.abs(e['fiber_floor'])))
                                                         if np.isfinite(e['fiber_floor']).any() else None))
                           for e in per_fiber.values()]
    Path(args.out).write_text(json.dumps(jsonable(report), indent=1))
    print(json.dumps(jsonable(dict(G2=report['G2'], all=report['accuracy']['all']['strip']['point'])), indent=1))


def cmd_crossings(args):
    fibers, _ = load_fibers(args.fibers, args.fiber_unit_base, args.span, args.fiber_urls)
    strips = [load_strip(d, args.strip_unit_base) for d in discover_strips(args.strips)]
    res = crossing_agreement(strips, fibers)
    Path(args.out).write_text(json.dumps(jsonable(dict(provenance=provenance(args), crossings=res)), indent=1))
    print(json.dumps(jsonable({k: v for k, v in res.items() if k != 'per_crossing'}), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, segments=True, strips=False):
        p.add_argument('--fibers', help='directory of VC3D fiber JSONs')
        p.add_argument('--fiber-urls', help='file of fiber JSON URLs (read into memory, nothing saved)')
        p.add_argument('--fiber-unit-base', type=float, default=1., help='base voxels per fiber JSON unit')
        p.add_argument('--span', choices=('controlled', 'full'), default='controlled')
        if segments:
            p.add_argument('--segments', nargs='*', default=[], help='tifxyz dirs (or trees of them)')
            p.add_argument('--segment-urls', help='file of tifxyz base URLs (read into memory, nothing saved)')
            p.add_argument('--segment-unit-base', type=float, default=1.)
            p.add_argument('--umbilicus', help='umbilicus.json (control points x, y, z)')
            p.add_argument('--umbilicus-unit-base', type=float, default=1.)
        if strips:
            p.add_argument('--strips', required=True, help='root of strip tifxyz dirs')
            p.add_argument('--strip-unit-base', type=float, default=4., help='base voxels per strip unit')
        p.add_argument('--out', required=True)

    p = sub.add_parser('match')
    common(p)
    p.set_defaults(func=cmd_match)
    p = sub.add_parser('lasagna')
    common(p, segments=False)
    p.add_argument('--nx', required=True, help='Lasagna nx zarr array (level dir; local or https)')
    p.add_argument('--ny', required=True)
    p.add_argument('--level-scale', type=float, default=LASAGNA_SCALE)
    p.set_defaults(func=cmd_lasagna)
    p = sub.add_parser('score')
    common(p, strips=True)
    p.add_argument('--matches', required=True)
    p.add_argument('--lasagna-normals', help='.npz from the lasagna subcommand')
    p.add_argument('--crossings', action='store_true', help='also the H/V crossing agreement (0139 calibration)')
    p.set_defaults(func=cmd_score)
    p = sub.add_parser('crossings')
    common(p, segments=False, strips=True)
    p.set_defaults(func=cmd_crossings)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
