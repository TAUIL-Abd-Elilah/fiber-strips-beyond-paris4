#!/usr/bin/env python
"""Pre-registered scorer for fiber_strip_small_v1 sheet strips off Paris 4 (see PREREG.md beside this file).

Question: does the Paris4-trained strip model place the sheet correctly around human-traced fibers on PHerc0139?
Ground truth: the PHerc0139 wrap segments (VC3D tifxyz) meshed on the same 2.399 um volume 20260102150214 as the
human fibers. Only the fiber STRETCHES that lie on a segment are run and scored (PREREG section 4). PHerc0332 has
no ground truth: only the H/V crossing agreement on a fixed set of crossings is reported there.

Subcommands (all CPU only; no torch, no strip model):

  match      human fiber JSONs + segments (+ umbilicus) -> matched stretches, noise floors, gate G1, the inference
             plan (pieces, decision heads), CPU and CT budget, seeded subsample when over budget
  crossplan  human fiber JSONs of a scroll without ground truth (PHerc0332) -> H/V crossings, windows, budget and
             the seeded crossing subsample
  lasagna    Lasagna nx/ny zarr (local or https) -> Lasagna normals at the samples of the matched stretches (.npz)
  score      matches + strips (+ Lasagna normals) -> per-node accuracy, baselines, bootstrap, calibration, gate G2
  crossings  strips + fibers (+ a crossplan) -> H/V crossing agreement (no ground truth)

Units. Everything is scored in "sv" (score voxels) = level 2 of the 2.4 um scans = 4 base voxels (9.596 um on
PHerc0139/0332). Human fiber JSONs and segment tifxyz are in base voxels (level 0; ``--fiber-unit-base 1
--segment-unit-base 1``); strips are in the strip model's "patch voxels" (CT level 2 = sv; ``--strip-unit-base 4``).
The strip model's trace voxels (AFVFibers grid_scale 8) are 2 sv: one decision every 8 trace voxels = 16 sv.

Normal distance. A segment (or strip) is a grid of nodes; a point is projected onto the bilinear surface of the
grid (Gauss-Newton from its nearest nodes) and its signed distance along the surface normal there is taken;
outward (away from the umbilicus) is positive when an umbilicus is given. A point whose foot falls off the valid
grid (beyond an edge or a hole) is "uncovered" and not scored. Surface normals follow villa's
sheets/scripts/match_fibers_to_patches.py (cross product of the grid's column and row derivatives).

Credits: strip format, stitching and layout are villa's sheet_strip (bruniss/sean, ScrollPrize/villa #2110;
``orient_decisions``/``stitch``/``stitch_values`` below are copied from
vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/infer.py at 6f2024569, MIT licence); the fiber
to segment matching rule follows villa's sheets/scripts/match_fibers_to_patches.py (sean); Windows inference fixes
are #2163 (slchicoineau-creator, Baz248); held-out human-fiber evaluation practice follows
modaribrahim/fiber-tracer-heldout-eval and villa #2143; human-fiber noise (fold-backs, jumps, duplicates) per
ChaseHendrick's #2248.
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
import urllib.error
import urllib.request
import uuid as uuidlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCORER_VERSION = '2.0.0'

# ---- Pre-registered constants (PREREG.md sections 4-7). Do not change after the freeze. ----------------------------
SCORE_UNIT_BASE = 4.0       # base voxels per score voxel (sv): level 2 of the 2.4 um scans
SV_UM = 9.596               # um per sv on the 2.399 um scans (PHerc0139 20260102150214, PHerc0332 20251211183505)
SAMPLE_STEP = 1.0           # sv between fiber samples
MATCH_MAX_D = 2.0           # sv (= 8 base voxels): a fiber sample lies on a segment when |normal distance| <= this
CANDIDATE_MAX_D = 6.0       # sv: about half a winding gap; the window of the unbiased noise floor NF6
MIN_STRETCH_SV = 2500./SV_UM  # 2.5 mm (260.5 sv): infer_all's default --min-length-mm, the shortest strip fiber
SEARCH_REACH = 30.0         # sv: farthest normal distance measured when scoring (larger: uncovered)
TANGENT_TOL = 1.0           # sv: a foot farther than this sideways from the point's normal line is off the grid
ASSOC_MAX_SV = 5.0          # sv: median distance from a strip's centre column to its fiber
ROW_MAX_SV = 30.0           # sv: a strip row whose centre is farther from the fiber is not scored (rows on the
                            # next winding, ~12 sv off, are scored: they count as errors of about a gap)
MIN_NODES = 25              # covered strip nodes for a fiber to count as scored (one strip row)
G1_MIN_FIBERS = 30
G1_MAX_NOISE = 2.0          # sv
G2_MAX_MEAN = 1.5           # sv
G2_MIN_WITHIN2 = 0.70
BOOTSTRAP = 2000
BOOTSTRAP_PCT = 1000        # replicates for the median / p90 CIs (concatenating nodes is slower)
SEED = 20261028
CAL_EDGES = (0., .5, .75, 1., 1.25, 1.5, 2., 3., math.inf)
CROSS_TOL = 3.0             # sv: closest approach of an H and a V fiber that counts as a crossing (the tight subset)
CROSS_TOL_0332 = 6.0        # sv: the PHerc0332 plan's crossings (half a winding gap; PREREG 7)
CROSS_RADIUS = 30.0         # sv: strip nodes this close to the crossing point are compared
LASAGNA_SCALE = 16.0        # base voxels per voxel of the Lasagna nx/ny level used (level 4)
TANGENT_HALF = 4            # samples each side for the fiber tangent
# Inference plan: villa sheet_strip defaults (infer.py/infer_all.py at 6f2024569) in sv; 1 trace voxel = 2 sv.
DECISION_STEP_SV = 16.0     # --step 8 trace voxels
FIRST_HEAD_SV = 24.0        # MIN_HEAD_ARC 12 trace voxels: the first decision of a full-fiber run
BLEND_SV = 32.0             # --blend 16 trace voxels: a row takes the decisions whose heads are within this
HEAD_MARGIN_SV = 40.0       # heads planned this far beyond each stretch end (BLEND_SV plus a row step and slack)
PIECE_BEFORE_SV = 300.0     # fiber kept before the first head: n_history 128 trace voxels (256 sv) + 40 + slack
PIECE_AFTER_SV = 160.0      # fiber kept after the last head: crop reach 58 trace voxels (116 sv) + 40 + slack
CT_LEVEL = 2
CT_RADIUS = 170.0           # level-2 voxels around the decision heads' fiber path (the setup run's footprint radius)
CPU_DECISIONS_PER_S = 1.35  # measured by the setup run (CPU, villa 6f2024569 + #2163)
PROCESS_OVERHEAD_S = 30.0   # assumed per infer.py process (model, frame model, CT store); not measured
CT_MIB_PER_S = 0.58         # measured S3 CT download rate on this machine (setup run)
CPU_CAP_H_0139 = 8.0
CT_CAP_GIB_0139 = 8.0       # new CT downloads; C: has ~20 GB free and must keep >= 8 GB
CPU_CAP_H_0332 = 2.0
CT_CAP_GIB_0332 = 2.0
MAX_CROSSINGS_0332 = 60

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


def consecutive_runs(idx):
    """(first, last) of each run of consecutive integers in a sorted integer array."""
    idx = np.asarray(idx, np.int64)
    if not len(idx):
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[0, breaks+1]
    ends = np.r_[breaks, len(idx)-1]
    return [(int(idx[a]), int(idx[b])) for a, b in zip(starts, ends)]


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
                return r.read()
        return (Path(source)/name).read_bytes()
    try:
        meta = json.loads(load('meta.json'))
    except Exception:
        meta = {}
    xyz = [np.asarray(tifffile.imread(io.BytesIO(load(f'{c}.tif'))), np.float64) for c in 'xyz']
    grid = np.stack(xyz, -1)
    del xyz
    valid = np.isfinite(grid).all(-1) & ~(grid == -1).any(-1)
    grid *= unit_base/score_unit_base
    grid[~valid] = np.nan
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
    """Segment sources from tifxyz dirs (each dir holding x.tif, or a tree of them) and/or a file of base URLs;
    named by their segment directory (``<segment>/x.tif`` or ``<segment>/mesh/<name>.tifxyz/x.tif``), sorted."""
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
    names = [s.name for s in out]
    if len(set(names)) != len(names):
        raise ValueError(f'duplicate segment names: {sorted({n for n in names if names.count(n) > 1})}')
    return sorted(out, key=lambda s: s.name)


# ---- Human fibers -------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Fiber:
    name: str             # file stem
    family: str           # 'H', 'V' or '?'
    family_source: str    # 'manual', 'automatic', 'geometry'
    points: np.ndarray    # dense samples (N, 3) sv, every SAMPLE_STEP along the controlled span
    s: np.ndarray         # their arclengths (sv)
    flags: dict
    line: np.ndarray = None       # the controlled line points (L, 3) sv, as in the JSON
    line_s: np.ndarray = None     # their arclengths (sv); line_s[-1] == s[-1]
    line_offset: int = 0          # index of line[0] in the JSON's line_points
    uuid: str = ''
    uuid_source: str = ''
    file: str = ''

    @property
    def length(self):
        return float(self.s[-1]) if len(self.s) else 0.

    @property
    def afv_id(self):
        return int(self.uuid.replace('-', '')[:13], 16) if self.uuid else None

    @property
    def annotator(self):
        return self.name.split('_')[0]


def _xyz(p):
    if isinstance(p, dict):
        p = p.get('position', p.get('xyz'))
    return [float(p[0]), float(p[1]), float(p[2])]


def fiber_uuid(raw, space, file_name):
    """fibers_json_to_afv.py's identity: the JSON's uuid, else uuid5(URL, 'vc3d_fiber:' + space + '/' + file)."""
    if raw.get('uuid'):
        return str(raw['uuid']).lower(), 'json'
    return str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, f'vc3d_fiber:{space}/{file_name}')), 'uuid5(file)'


def load_fiber_json(path, unit_base=1.0, span='controlled', step=SAMPLE_STEP, space=None):
    """A VC3D fiber JSON file as a Fiber in sv (``parse_fiber``), or None when it has no scored span."""
    return parse_fiber(json.loads(Path(path).read_text()), Path(path).stem, unit_base, span, step, space)


def parse_fiber(raw, name, unit_base=1.0, span='controlled', step=SAMPLE_STEP, space=None):
    """A VC3D fiber JSON (v3/v4; line_points xyz) as a Fiber in sv, or None when it has no scored span.

    ``span='controlled'``: the line between the outer human control points (villa's annotations/vc3d.load_fibers
    and fibers_json_to_afv.py keep only that); ``'full'``: every line point. Flags (for the #2248-style sensitivity
    analysis): fold-backs (the resampled line turning > 150 degrees within 6 sv) and jumps (a line step > 3x the
    median step and > 8 sv), with their arclengths."""
    line = np.asarray([_xyz(p) for p in raw.get('line_points', [])], np.float64)*unit_base/SCORE_UNIT_BASE
    controls = np.asarray([_xyz(p) for p in raw.get('control_points', [])], np.float64)*unit_base/SCORE_UNIT_BASE
    flags = dict(version=raw.get('version'), n_line=int(len(line)), n_controls=int(len(controls)),
                 username=raw.get('username'))
    if len(line) < 2:
        return None
    a = 0
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
    line_s = arclength(line)
    steps = np.diff(line_s)
    med = float(np.median(steps)) if len(steps) else 0.
    jump_at = line_s[1:][steps > max(3*med, 8.)]
    flags['jumps'] = int(len(jump_at))
    flags['jump_s'] = [float(x) for x in jump_at]
    pts, s = resample(line, step)
    if len(pts) < 2:
        return None
    t = tangents(pts, 2)
    reach = 3
    fold_at = []
    if len(t) > 2*reach:
        cos = (t[:-2*reach]*t[2*reach:]).sum(1)
        starts = np.flatnonzero(np.diff(np.r_[0, (cos < math.cos(math.radians(150))).astype(np.int8)]) == 1)
        fold_at = s[starts+reach].tolist()
    flags['foldbacks'] = int(len(fold_at))
    flags['foldback_s'] = [float(x) for x in fold_at]
    hv = raw.get('hv_classification') or {}
    manual, auto = (hv.get('manual_tag') or ''), (hv.get('automatic_tag') or '')
    if manual in ('H', 'V'):
        family, source = manual, 'manual'
    elif auto in ('H', 'V'):
        family, source = auto, 'automatic'
    else:
        dz = abs(pts[-1, 2]-pts[0, 2])
        family, source = ('V' if dz > 0.5*s[-1] else 'H'), 'geometry'
    file_name = name+'.json'
    sp = raw.get('vc_open_data_coordinate_space') or space or ''
    fuuid, usrc = fiber_uuid(raw, sp, file_name)
    return Fiber(name, family, source, pts, s, flags, line, line_s, a, fuuid, usrc, file_name)


def _fetch_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read())


def load_fibers(fiber_dir=None, unit_base=1.0, span='controlled', urls_file=None, threads=8, space=None, names=None):
    """Fibers from a directory of JSONs and/or a file of JSON URLs (read into memory, nothing saved); ``names``
    restricts them to those file stems."""
    items = []
    if fiber_dir:
        items += [(p.stem, (lambda p=p: json.loads(p.read_text()))) for p in sorted(Path(fiber_dir).glob('*.json'))
                  if names is None or p.stem in names]
    if urls_file:
        urls = [u.strip() for u in Path(urls_file).read_text().splitlines() if u.strip() and not u.startswith('#')]
        urls = [u for u in urls if names is None or u.rstrip('/').split('/')[-1].rsplit('.', 1)[0] in names]
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
            f = parse_fiber(raw, name, unit_base, span, space=space)
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


# ---- Fiber-to-segment distances (PREREG section 4) -----------------------------------------------------------

def _project_worker(job):
    """One segment: (index, nodes, samples near its bbox, sample indices with |d| <= CANDIDATE_MAX_D, their d)."""
    si, name, source, unit_base, samples_path, axis_xyz, reach = job
    t0 = time.time()
    pts = np.load(samples_path, mmap_mode='r')
    seg = SegmentSource(name, source, unit_base)
    if not seg.may_touch(pts, reach):
        return si, 0, 0, np.zeros(0, np.int64), np.zeros(0, np.float32), time.time()-t0
    surf = seg.load(Umbilicus(axis_xyz) if axis_xyz is not None else None)
    near = np.flatnonzero(surf.near_bbox(pts, reach))
    d, cov, _, _ = surf.project(np.asarray(pts[near]), reach=reach)
    keep = cov & (np.abs(d) <= CANDIDATE_MAX_D)
    return si, len(surf.nodes), len(near), near[keep].astype(np.int64), d[keep].astype(np.float32), time.time()-t0


def fiber_segment_distances(fibers, segments, axis=None, workers=1, tmpdir=None, log=print):
    """For every fiber sample and segment: the signed normal distance where the foot is on the segment's valid mesh
    and |d| <= CANDIDATE_MAX_D. Returns hits[k] = {segment index: (sample indices (sorted), d)} and a log."""
    import tempfile
    allpts = np.concatenate([f.points for f in fibers]) if fibers else np.zeros((0, 3))
    owner = np.concatenate([np.full(len(f.points), k) for k, f in enumerate(fibers)]) if fibers else np.zeros(0, int)
    offset = np.r_[0, np.cumsum([len(f.points) for f in fibers])]
    hits = [dict() for _ in fibers]
    seglog = []
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmp:
        path = str(Path(tmp)/'samples.npy')
        np.save(path, allpts)
        jobs = [(si, s.name, s.source, s.unit_base, path, None if axis is None else axis.xyz, CANDIDATE_MAX_D)
                for si, s in enumerate(segments)]
        t0 = time.time()
        if workers > 1:
            import multiprocessing
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(workers) as pool:
                results = []
                for r in pool.imap_unordered(_project_worker, jobs):
                    results.append(r)
                    log(f'[match] {len(results)}/{len(jobs)} {segments[r[0]].name}: {r[1]} nodes, {r[2]} samples '
                        f'near, {len(r[3])} within {CANDIDATE_MAX_D} sv, {r[5]:.0f} s (total {time.time()-t0:.0f} s)')
        else:
            results = []
            for job in jobs:
                r = _project_worker(job)
                results.append(r)
                log(f'[match] {len(results)}/{len(jobs)} {segments[r[0]].name}: {r[1]} nodes, {r[2]} near, '
                    f'{len(r[3])} within {CANDIDATE_MAX_D} sv, {r[5]:.0f} s')
    for si, nodes, near, idx, d, sec in sorted(results, key=lambda r: r[0]):
        seglog.append(dict(segment=segments[si].name, nodes=int(nodes), samples_near_bbox=int(near),
                           samples_within_6sv=int(len(idx)), seconds=round(float(sec), 1)))
        if not len(idx):
            continue
        fk = owner[idx]
        for k in np.unique(fk):
            sel = fk == k
            li = idx[sel]-offset[k]
            order = np.argsort(li)
            hits[int(k)][si] = (li[order], d[sel][order].astype(np.float64))
    return hits, seglog


# ---- Stretch selection (PREREG section 4) ---------------------------------------------------------------------

def _in_interval_count(positions, s0, s1):
    return int(sum(s0 <= p <= s1 for p in positions))


def stretch_record(fiber, si, seg_name, i0, i1, d_of):
    """A stretch of ``fiber`` on segment ``si`` over samples i0..i1 (inclusive); ``d_of`` maps sample -> d."""
    d = np.array([d_of[i] for i in range(i0, i1+1)])
    a = np.abs(d)
    s0, s1 = float(fiber.s[i0]), float(fiber.s[i1])
    z = fiber.points[i0:i1+1, 2]
    return dict(fiber=fiber.name, file=fiber.file, uuid=fiber.uuid, family=fiber.family,
                family_source=fiber.family_source, annotator=fiber.annotator, segment=seg_name, segment_index=si,
                i0=int(i0), i1=int(i1), s0=s0, s1=s1, length_sv=s1-s0, length_mm=(s1-s0)*SV_UM*1e-3,
                samples=int(len(d)), mean_d=float(d.mean()), median_abs_d=float(np.median(a)),
                p90_abs_d=float(np.percentile(a, 90)), max_abs_d=float(a.max()),
                z_sv=[float(z.min()), float(z.max())], centre_l0=(fiber.points[(i0+i1)//2]*SCORE_UNIT_BASE).tolist(),
                foldbacks=_in_interval_count(fiber.flags.get('foldback_s', []), s0, s1),
                jumps=_in_interval_count(fiber.flags.get('jump_s', []), s0, s1))


def find_stretches(fiber, fiber_hits, seg_names, min_len=MIN_STRETCH_SV):
    """Matched stretches of one fiber (PREREG 4.2-4.4): per segment, maximal runs of consecutive samples with
    |d| <= MATCH_MAX_D, at least MIN_STRETCH_SV long; overlaps between segments resolved greedily (longest first,
    a later stretch keeps its longest non-overlapping part if that is still long enough). Also the near runs
    (|d| <= CANDIDATE_MAX_D, >= MIN_STRETCH_SV) for the unbiased noise floor NF6, and the dropped overlaps.
    ``min_len``: the shortest stretch (sv), MIN_STRETCH_SV for the primary set."""
    cands, near = [], []
    for si, (idx, d) in sorted(fiber_hits.items()):
        d_of = dict(zip(idx.tolist(), d.tolist()))
        on = np.abs(d) <= MATCH_MAX_D
        for i0, i1 in consecutive_runs(idx[on]):
            if fiber.s[i1]-fiber.s[i0] >= min_len:
                cands.append((si, i0, i1, d_of))
        for i0, i1 in consecutive_runs(idx):
            if fiber.s[i1]-fiber.s[i0] >= MIN_STRETCH_SV:
                near.append(np.abs(np.array([d_of[i] for i in range(i0, i1+1)])))
    cands.sort(key=lambda c: (-(fiber.s[c[2]]-fiber.s[c[1]]), seg_names[c[0]], c[1]))
    taken = np.zeros(len(fiber.s), bool)
    accepted, dropped = [], []
    for si, i0, i1, d_of in cands:
        free = np.flatnonzero(~taken[i0:i1+1])+i0
        parts = consecutive_runs(free)
        best = max(parts, key=lambda p: fiber.s[p[1]]-fiber.s[p[0]]) if parts else None
        if best is None or fiber.s[best[1]]-fiber.s[best[0]] < min_len:
            dropped.append(dict(fiber=fiber.name, segment=seg_names[si], s0=float(fiber.s[i0]),
                                s1=float(fiber.s[i1]), reason='overlaps a longer stretch on another segment'))
            continue
        b0, b1 = best
        taken[b0:b1+1] = True
        accepted.append(stretch_record(fiber, si, seg_names[si], b0, b1, d_of))
        if (b0, b1) != (i0, i1):
            accepted[-1]['trimmed_for_overlap'] = [float(fiber.s[i0]), float(fiber.s[i1])]
    accepted.sort(key=lambda r: r['s0'])
    return accepted, near, dropped


def gate_g1(stretches, match_abs, near_abs, fibers_total, min_len=MIN_STRETCH_SV):
    """G1 (PREREG 6): >= G1_MIN_FIBERS fibers with a matched stretch, NF_match <= G1_MAX_NOISE (true by
    construction: every stretch sample has |d| <= MATCH_MAX_D) and NF6 <= G1_MAX_NOISE. ``match_abs``: per stretch,
    |d| of its samples; ``near_abs``: per near run, |d| of its samples."""
    fibers = sorted({r['fiber'] for r in stretches})
    nf_match = float(np.median(np.concatenate(match_abs))) if match_abs else None
    per_fiber = {}
    for r, a in zip(stretches, match_abs):
        per_fiber.setdefault(r['fiber'], []).append(a)
    nf_match_fiber = float(np.median([np.median(np.concatenate(v)) for v in per_fiber.values()])) if per_fiber         else None
    nf6 = float(np.median(np.concatenate(near_abs))) if near_abs else None
    by_family = {fam: dict(fibers=len({r['fiber'] for r in stretches if r['family'] == fam}),
                           stretches=sum(r['family'] == fam for r in stretches),
                           length_mm=float(sum(r['length_mm'] for r in stretches if r['family'] == fam)))
                 for fam in ('H', 'V')}
    out = dict(fibers_loaded=fibers_total, fibers_with_stretch=len(fibers), stretches=len(stretches),
               total_length_mm=float(sum(r['length_mm'] for r in stretches)), by_family=by_family,
               noise_floor_match_sv=nf_match, noise_floor_match_per_fiber_median_sv=nf_match_fiber,
               noise_floor_match_definition=f'median |d| over all samples of the matched stretches (bounded by '
                                            f'{MATCH_MAX_D} sv by the selection)',
               noise_floor_nf6_sv=nf6,
               noise_floor_nf6_definition=f'median |d| over all samples of the near runs: runs of consecutive '
                                          f'samples on one segment with |d| <= {CANDIDATE_MAX_D} sv, >= 2.5 mm',
               near_runs=len(near_abs))
    out['pass'] = bool(len(fibers) >= G1_MIN_FIBERS and nf_match is not None and nf_match <= G1_MAX_NOISE
                       and nf6 is not None and nf6 <= G1_MAX_NOISE)
    out['rule'] = (f'>= {G1_MIN_FIBERS} fibers with a matched stretch (|d| <= {MATCH_MAX_D} sv on one segment over '
                   f'>= {min_len:.1f} sv ({min_len*SV_UM*1e-3:.3f} mm) of consecutive 1 sv samples) AND NF_match <= '
                   f'{G1_MAX_NOISE} sv AND NF6 <= {G1_MAX_NOISE} sv')
    return out


# ---- Inference plan, CPU and CT budget (PREREG section 4.6-4.8) -------------------------------------------------

def head_grid(length, s0, s1):
    """Decision heads (sv, fiber arclength) a full-fiber run would place within HEAD_MARGIN_SV of [s0, s1]:
    FIRST_HEAD_SV + k * DECISION_STEP_SV, inside the fiber."""
    lo, hi = max(FIRST_HEAD_SV, s0-HEAD_MARGIN_SV), min(length, s1+HEAD_MARGIN_SV)
    k0 = math.ceil((lo-FIRST_HEAD_SV)/DECISION_STEP_SV-1e-9)
    k1 = math.floor((hi-FIRST_HEAD_SV)/DECISION_STEP_SV+1e-9)
    return FIRST_HEAD_SV+DECISION_STEP_SV*np.arange(k0, k1+1)


def piece_uuid(parent_uuid, a, b):
    return str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, f'vc3d_fiber_piece:{parent_uuid}:{a}-{b}'))


def plan_pieces(fiber, windows):
    """AFV pieces of ``fiber`` holding the windows (dicts with id, s0, s1): each window's fiber from PIECE_BEFORE_SV
    before its first head to PIECE_AFTER_SV after its last head (heads: ``head_grid``), cut at line points
    (bit-identical JSON points); overlapping pieces merged. One infer.py run per piece: its decision heads are the
    full-fiber grid points from its windows' first head to their last head (gaps between windows included), and its
    --arc range is that, in patch voxels (= sv) of the piece's own arclength before AFVFibers' kink repair."""
    L = fiber.length
    spans = []
    for w in windows:
        heads = head_grid(L, w['s0'], w['s1'])
        w['heads_sv'] = [float(heads[0]), float(heads[-1])] if len(heads) else None
        w['decisions_own'] = int(len(heads))
        if not len(heads):
            continue
        p0, p1 = max(0., heads[0]-PIECE_BEFORE_SV), min(L, heads[-1]+PIECE_AFTER_SV)
        a = int(np.clip(np.searchsorted(fiber.line_s, p0+1e-9, 'right')-1, 0, len(fiber.line_s)-1))
        b = int(np.clip(np.searchsorted(fiber.line_s, p1-1e-9, 'left'), 0, len(fiber.line_s)-1))
        spans.append([a, b, [w]])
    spans.sort(key=lambda x: x[0])
    merged = []
    for a, b, ws in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][2] += ws
        else:
            merged.append([a, b, ws])
    pieces = []
    for a, b, ws in merged:
        A, B = a+fiber.line_offset, b+fiber.line_offset
        pu = piece_uuid(fiber.uuid, A, B)
        ps0 = float(fiber.line_s[a])
        h0, h1 = min(w['heads_sv'][0] for w in ws), max(w['heads_sv'][1] for w in ws)
        n_heads = int(round((h1-h0)/DECISION_STEP_SV))+1
        seg_len = {}
        for w in ws:
            seg_len[w.get('segment', '')] = seg_len.get(w.get('segment', ''), 0.)+w['s1']-w['s0']
        piece = dict(afv_id=int(pu.replace('-', '')[:13], 16), uuid=pu, fiber=fiber.name, parent_file=fiber.file,
                     parent_uuid=fiber.uuid, parent_afv_id=fiber.afv_id, family=fiber.family,
                     annotator=fiber.annotator, line_index_range=[A, B],
                     parent_arc_sv=[ps0, float(fiber.line_s[b])], length_sv=float(fiber.line_s[b]-ps0),
                     heads_sv=[float(h0), float(h1)], decisions=n_heads,
                     arc_patch_vox=[round(h0-ps0, 3), round(h1-ps0, 3)],
                     segment=max(sorted(seg_len), key=lambda k: seg_len[k]),
                     windows=[w['id'] for w in sorted(ws, key=lambda w: w['s0'])])
        pieces.append(piece)
    return pieces


def number_pieces(pieces, prefix, windows):
    """Piece ids ``prefix`` + 4 digits in (fiber, start) order; each window gets its piece id and AFV id."""
    pieces.sort(key=lambda p: (p['fiber'], p['parent_arc_sv'][0]))
    wid = {w['id']: w for w in windows}
    for n, p in enumerate(pieces):
        p['id'] = f'{prefix}{n:04d}'
        for i in p['windows']:
            wid[i]['piece'] = p['id']
            wid[i]['piece_afv_id'] = p['afv_id']
    return pieces


def fetch_zarray(source):
    """A zarr v2 array's .zarray (``source``: the level dir, local or https)."""
    source = str(source).rstrip('/')
    if source.startswith(('http://', 'https://')):
        with urllib.request.urlopen(source+'/.zarray', timeout=60) as r:
            return json.loads(r.read())
    return json.loads((Path(source)/'.zarray').read_text())


def ct_footprint(points_l0, level, radius, chunks, shape):
    """Chunk keys (z, y, x) of level ``level`` whose box lies within ``radius`` level voxels of the polyline
    (sampled every <= 1 voxel, so a chunk at most 0.5 voxel beyond ``radius`` may be missed). As
    tools/prefetch_ct.footprint of the setup run (C:/VesuviusTemp/strips), which samples every <= 8 voxels."""
    c = np.asarray(chunks, float)
    m = int(np.ceil(radius/c.min()))
    p = np.asarray(points_l0, np.float64)/(2**level)
    if len(p) == 1:
        p = np.repeat(p, 2, 0)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    dense = [p[:1]]
    for a, b, n in zip(p[:-1], p[1:], np.ceil(seg/1.).astype(int)):
        t = np.linspace(0, 1, max(n, 1)+1)[1:, None]
        dense.append(a+(b-a)*t)
    zyx = np.concatenate(dense)[:, ::-1]
    zyx = np.unique(np.round(zyx, 1), axis=0)
    base = np.floor(zyx/c).astype(np.int64)
    keys = set()
    for off in np.array(np.meshgrid(*[range(-m, m+1)]*3, indexing='ij')).reshape(3, -1).T:
        k = base+off
        lo, hi = k*c, (k+1)*c-1
        d = np.linalg.norm(np.clip(zyx, lo, hi)-zyx, axis=1)
        ok = (d <= radius) & np.all(k >= 0, 1) & np.all(k*c < np.asarray(shape), 1)
        keys.update(map(tuple, np.unique(k[ok], axis=0).tolist()))
    return keys


def head_path_l0(fiber, heads_sv):
    """Fiber points (L0) between the first and last decision head."""
    lo, hi = heads_sv
    sel = (fiber.s >= lo-1e-9) & (fiber.s <= hi+1e-9)
    pts = fiber.points[sel] if sel.any() else fiber.points[[int(np.argmin(np.abs(fiber.s-lo)))]]
    return pts*SCORE_UNIT_BASE


def piece_ct(fibers_by_name, pieces, zarray, cache_dir=None, volume_path=None):
    """Per piece: its CT chunk keys (level CT_LEVEL within CT_RADIUS of its head path, ``_ct``); returns the keys
    already in villa's remote-CT cache."""
    chunks, shape = zarray['chunks'], zarray['shape']
    for p in pieces:
        p['_ct'] = ct_footprint(head_path_l0(fibers_by_name[p['fiber']], p['heads_sv']), CT_LEVEL, CT_RADIUS,
                                chunks, shape)
        p['ct_chunks'] = len(p['_ct'])
    cached = set()
    if cache_dir and volume_path:
        sep = zarray.get('dimension_separator', '.')
        root = Path(cache_dir)/'s3'/'vesuvius-challenge-open-data'/volume_path/str(CT_LEVEL)
        allk = set().union(*[p['_ct'] for p in pieces]) if pieces else set()
        cached = {k for k in allk if (root/sep.join(map(str, k))).exists()}
    return cached


def budget(pieces, zarray, cached=frozenset()):
    """CPU hours (decisions at CPU_DECISIONS_PER_S plus PROCESS_OVERHEAD_S per infer.py run = per piece) and CT
    (union of chunk keys; uncompressed chunk bytes) of a set of pieces."""
    chunk_bytes = int(np.prod(zarray['chunks']))*np.dtype(zarray.get('dtype', '|u1')).itemsize
    keys = set().union(*[p['_ct'] for p in pieces]) if pieces else set()
    new = keys-set(cached)
    dec = int(sum(p['decisions'] for p in pieces))
    cpu_s = dec/CPU_DECISIONS_PER_S+PROCESS_OVERHEAD_S*len(pieces)
    return dict(pieces=len(pieces), windows=int(sum(len(p.get('windows', [p['id']])) for p in pieces)),
                fibers=len({p['fiber'] for p in pieces}), decisions=dec,
                cpu_hours=round(cpu_s/3600, 3), cpu_hours_decisions_only=round(dec/CPU_DECISIONS_PER_S/3600, 3),
                ct_chunks=len(keys), ct_gib=round(len(keys)*chunk_bytes/2**30, 3),
                ct_chunks_cached=len(keys & set(cached)), ct_new_gib=round(len(new)*chunk_bytes/2**30, 3),
                ct_download_hours=round(len(new)*chunk_bytes/2**20/CT_MIB_PER_S/3600, 2),
                rates=dict(decisions_per_s=CPU_DECISIONS_PER_S, overhead_s_per_run=PROCESS_OVERHEAD_S,
                           ct_mib_per_s=CT_MIB_PER_S, ct_level=CT_LEVEL, ct_radius=CT_RADIUS,
                           chunk_bytes=chunk_bytes))


def stratified_subsample(units, strata_key, zarray, cpu_cap_h, ct_cap_gib, cached=frozenset(), seed=SEED,
                         max_units=None):
    """The seeded subsample (PREREG 4.8): units (pieces; crossings for PHerc0332) grouped by ``strata_key``;
    strata in sorted key order; within a stratum a seeded permutation of the units sorted by id (numpy
    default_rng(seed), drawn stratum by stratum in that order). Round robin over the strata: each turn offers a
    stratum's next unit, preferring units of fibers not yet chosen (a stratum's units of already chosen fibers wait
    until its other units are offered); a unit is taken when the CPU hours (decisions plus PROCESS_OVERHEAD_S per
    infer.py run; a unit with ``n_runs`` counts that many runs) and the new CT GiB of the selection stay within the
    caps, else it is passed over for good. Stops when every stratum is exhausted (or ``max_units`` are taken).
    Returns the chosen ids in selection order."""
    chunk_bytes = int(np.prod(zarray['chunks']))*np.dtype(zarray.get('dtype', '|u1')).itemsize
    rng = np.random.default_rng(seed)
    groups = {}
    for w in units:
        groups.setdefault(strata_key(w), []).append(w)
    queues = {}
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda w: w['id'])
        queues[key] = [g[i] for i in rng.permutation(len(g))]
    chosen, fibers, keys = [], set(), set()
    dec, nwin, nproc = 0, 0, 0
    cached = set(cached)
    while any(queues.values()):
        for key in sorted(queues):
            q = queues[key]
            if not q:
                continue
            pick = next((w for w in q if w['fiber'] not in fibers), q[0])
            q.remove(pick)
            new_keys = keys | pick['_ct']
            new_dec = dec+pick['decisions']
            cpu_h = (new_dec/CPU_DECISIONS_PER_S+PROCESS_OVERHEAD_S*(nproc+pick.get('n_runs', 1)))/3600
            ct_gib = len(new_keys-cached)*chunk_bytes/2**30
            if cpu_h <= cpu_cap_h and ct_gib <= ct_cap_gib:
                chosen.append(pick['id'])
                fibers.add(pick['fiber'])
                keys, dec, nwin, nproc = new_keys, new_dec, nwin+1, nproc+pick.get('n_runs', 1)
                if max_units is not None and nwin >= max_units:
                    return chosen
    return chosen


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


def associate(strip, index: FiberIndex, only=None):
    """The fiber a strip belongs to: the one its row centres run closest to (median); None if > ASSOC_MAX_SV.
    ``only``: a fiber index to test alone (the strip's planned fiber)."""
    centre = row_centres(strip)
    ok = np.isfinite(centre).all(1)
    if not ok.any() or index.tree is None:
        return None, math.inf
    if only is not None:
        cands = [only]
    else:
        _, idx = index.tree.query(centre[ok], k=1)
        cands = np.unique(index.owner[idx])
    best, best_med = None, math.inf
    for k in cands:
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
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(self.source+'/'+key, timeout=120) as r:
                        return r.read()
                except urllib.error.HTTPError as error:
                    if error.code in (403, 404):
                        return None
                    if attempt == 3:
                        raise
                except OSError:
                    if attempt == 3:
                        raise
                time.sleep(2+3*attempt)
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


def score_strips(strips, fibers, stretches, segments, axis=None, normals=None, log=print, pieces=None):
    """Per strip: signed normal distances of strip / baseline nodes to the matched segment surface, on strip rows
    inside matched stretches of the strip's fiber (each row against its stretch's segment). A strip named after a
    piece id (infer.py OUT/<piece id>/strip, the run plan's layout) scores that piece's stretches; one named after a
    stretch id scores that stretch; any other strip scores the stretches of its fiber that no earlier strip (sorted
    by path) has claimed. The strip must run along its fiber (median centre distance <= ASSOC_MAX_SV). Returns
    per-strip entries and a log of skips."""
    index = FiberIndex(fibers)
    by_fiber, by_id = {}, {}
    for st in stretches:
        by_fiber.setdefault(st['fiber'], []).append(st)
        by_id[st['id']] = st
    by_piece = {}
    for pc in pieces or []:
        by_piece[pc['id']] = [by_id[i] for i in pc['windows'] if i in by_id]
    seg_index = {s.name: i for i, s in enumerate(segments)}
    jobs, entries, skips, claimed = {}, {}, [], {}
    for strip in sorted(strips, key=lambda s: s.path):
        planned = by_piece.get(strip.name) or ([by_id[strip.name]] if strip.name in by_id else None)
        only = index.by_name.get(planned[0]['fiber']) if planned else None
        if planned and only is None:
            skips.append(dict(strip=strip.name, reason=f"planned fiber {planned[0]['fiber']} not loaded"))
            continue
        k, med = associate(strip, index, only)
        if k is None:
            skips.append(dict(strip=strip.name, reason=f'no fiber within {ASSOC_MAX_SV} sv (median {med:.1f} sv): '
                                                       'unit or association error'))
            continue
        fiber = fibers[k]
        cands = planned if planned else by_fiber.get(fiber.name, [])
        cands = [st for st in cands if st['id'] not in claimed]
        if not cands:
            skips.append(dict(strip=strip.name, fiber=fiber.name,
                              reason='no unclaimed matched stretch on this fiber'))
            continue
        centre = row_centres(strip)
        okc = np.isfinite(centre).all(1)
        row_s = np.full(len(centre), np.nan)
        row_d = np.full(len(centre), np.inf)
        if okc.any():
            dd, ii = index.tree_of(k).query(centre[okc], k=1)
            row_s[okc], row_d[okc] = fiber.s[ii], dd
        R, C = strip.valid.shape
        row_seg = np.full(R, -1)
        row_st = np.full(R, -1)
        used = []
        for st in cands:
            inside = (row_s >= st['s0']) & (row_s <= st['s1']) & (row_d <= ROW_MAX_SV)
            if inside.any():
                if st['segment'] not in seg_index:
                    skips.append(dict(strip=strip.name, stretch=st['id'], reason=f"segment {st['segment']} not loaded"))
                    continue
                row_seg[inside] = seg_index[st['segment']]
                row_st[inside] = st['_n']
                used.append(st['id'])
        if not used:
            skips.append(dict(strip=strip.name, fiber=fiber.name, reason='no strip row inside a matched stretch'))
            continue
        for sid in used:
            claimed[sid] = strip.name
        nodes = {'strip': strip.rows}
        if strip.frame_rows is not None:
            nodes['frame_plane'] = strip.frame_rows
        if normals is not None and fiber.name in normals:
            nv, nvalid = normals[fiber.name]
            ls = np.where(np.isfinite(row_s), row_s, 0.)
            lrows = lasagna_plane_rows(fiber, ls, C, strip.column_step, nv, nvalid)
            lrows[~np.isfinite(row_s)] = np.nan
            nodes['lasagna_plane'] = lrows
        entry = dict(strip=strip.name, fiber=fiber.name, family=fiber.family, assoc_median_sv=med, stretches=used,
                     rows=R, cols=C, notes=strip.notes, err={s: np.full((R, C), np.nan) for s in nodes},
                     expected=strip.expected, valid=strip.valid, row_seg=row_seg, row_stretch=row_st,
                     fiber_floor=np.full(R, np.nan), lasagna_angle=np.full(R, np.nan))
        entries[strip.name] = entry
        fiber_at_row = fiber.points[np.clip(np.searchsorted(fiber.s, np.nan_to_num(row_s)), 0, len(fiber.s)-1)]
        for seg in np.unique(row_seg[row_seg >= 0]):
            rr = np.flatnonzero(row_seg == seg)
            for surf, grid in nodes.items():
                mask = np.zeros((R, C), bool)
                mask[rr] = strip.valid[rr] & np.isfinite(grid[rr]).all(-1)
                rc = np.argwhere(mask)
                jobs.setdefault(int(seg), []).append((strip.name, surf, grid[mask], rc))
            jobs.setdefault(int(seg), []).append((strip.name, '_floor', fiber_at_row[rr], rr))
    t0 = time.time()
    for seg in sorted(jobs):
        surf = segments[seg].load(axis)
        for sname, kind, pts, ids in jobs[seg]:
            d, cov, _, nrm = surf.project(pts)
            e = entries[sname]
            if kind == '_floor':
                e['fiber_floor'][ids] = np.where(cov, d, np.nan)
                if normals is not None and e['fiber'] in normals:
                    nv, nval = normals[e['fiber']]
                    _, kk = index.tree_of(index.by_name[e['fiber']]).query(pts, k=1)
                    ang = np.degrees(np.arccos(np.clip(np.abs((nv[kk]*nrm).sum(1)), 0, 1)))
                    e['lasagna_angle'][ids] = np.where(cov & nval[kk], ang, np.nan)
            else:
                e['err'][kind][ids[:, 0], ids[:, 1]] = np.where(cov, d, np.nan)
        log(f'[score] segment {segments[seg].name}: {len(jobs[seg])} node sets, {time.time()-t0:.0f} s')
        del surf
    return entries, skips


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


def fiber_clusters(entries, family=None, names=None, stretch_ids=None):
    """Bootstrap clusters: per fiber, its strips' node arrays (``kind`` -> list of (R, C) arrays and row masks);
    optionally one family, a subset of fiber names, or a subset of stretch ids (rows of other stretches dropped)."""
    groups = {}
    for e in entries.values():
        if family is not None and e['family'] != family:
            continue
        if names is not None and e['fiber'] not in names:
            continue
        rows = e['row_seg'] >= 0
        if stretch_ids is not None:
            keep = np.isin(e['row_stretch'], list(stretch_ids))
            rows = rows & keep
        groups.setdefault(e['fiber'], []).append((e, rows))
    return groups


def _cat(parts):
    return np.concatenate(parts) if parts else np.zeros(0)


def aggregate(entries, family=None, seed=SEED, names=None, label=None, stretch_ids=None):
    """Pooled accuracy of the strips and baselines over the scored fibers (optionally one family, or a subset of
    fiber names / stretch ids for the sensitivity analyses). Clusters are fibers (all their strips' nodes)."""
    groups = fiber_clusters(entries, family, names, stretch_ids)
    out = dict(family=family or 'all', subset=label)
    on_rows = sum(int(e['valid'][rows].sum()) for g in groups.values() for e, rows in g)
    covered = sum(int(np.isfinite(e['err']['strip'][rows]).sum()) for g in groups.values() for e, rows in g)
    out['strip_coverage'] = dict(nodes_on_scored_rows=on_rows, covered=covered,
                                 fraction=covered/on_rows if on_rows else None)
    strip_abs, signed, keep = [], [], []
    for fname in sorted(groups):
        v = _cat([e['err']['strip'][rows][np.isfinite(e['err']['strip'][rows])] for e, rows in groups[fname]])
        if len(v) >= MIN_NODES:
            keep.append(fname)
            strip_abs.append(np.abs(v))
            signed.append(v)
    out['scored_fibers'] = len(keep)
    out['scored_stretches'] = len({s for f in keep for e, _ in groups[f] for s in e['stretches']
                                   if stretch_ids is None or s in stretch_ids})
    out['strip'] = bootstrap_stats(strip_abs, seed)
    out['strip_signed_mean_sv'] = float(np.mean(np.concatenate(signed))) if signed else None
    floors = [np.abs(_cat([e['fiber_floor'][rows][np.isfinite(e['fiber_floor'][rows])] for e, rows in groups[f]]))
              for f in keep]
    floors = [x for x in floors if len(x)]
    out['fiber_floor'] = pooled(floors)
    out['fiber_floor_note'] = 'the human fiber itself, one point per scored row: the ground truth noise floor'
    for base in ('frame_plane', 'lasagna_plane'):
        a_list, b_list = [], []
        for f in keep:
            pa, pb = [], []
            for e, rows in groups[f]:
                if base not in e['err']:
                    continue
                both = np.isfinite(e['err']['strip']) & np.isfinite(e['err'][base]) & rows[:, None]
                pa.append(np.abs(e['err']['strip'][both]))
                pb.append(np.abs(e['err'][base][both]))
            a, b = _cat(pa), _cat(pb)
            if len(a) >= MIN_NODES:
                a_list.append(a)
                b_list.append(b)
        out[base] = dict(available_fibers=len(b_list), stats=bootstrap_stats(b_list, seed) if b_list else None,
                         strip_on_same_nodes=pooled(a_list) if a_list else None,
                         delta=paired_delta(a_list, b_list, seed) if len(a_list) >= 2 else None)
    angles = _cat([e['lasagna_angle'][rows][np.isfinite(e['lasagna_angle'][rows])]
                   for f in keep for e, rows in groups[f]])
    out['lasagna_vs_segment_normal_deg'] = (dict(median=float(np.median(angles)), p90=float(np.percentile(angles, 90)))
                                            if len(angles) else None)
    return out


def calibration(entries):
    """Calibration of the model's expected_error against the observed |d| (PREREG 5.6), nodes and fibers."""
    groups = fiber_clusters(entries)
    pred, obs, fp, fo = [], [], [], []
    for fname in sorted(groups):
        pp, oo = [], []
        for e, rows in groups[fname]:
            if e['expected'] is None:
                continue
            err = e['err']['strip']
            ok = np.isfinite(err) & np.isfinite(e['expected']) & rows[:, None]
            pp.append(e['expected'][ok])
            oo.append(np.abs(err[ok]))
        p, o = _cat(pp), _cat(oo)
        if len(p) < MIN_NODES:
            continue
        pred.append(p)
        obs.append(o)
        fp.append(p.mean())
        fo.append(o.mean())
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


# ---- Crossings (PREREG section 7) ---------------------------------------------------------------------------------

def find_crossings(fibers, tol=CROSS_TOL):
    """H/V fiber pairs whose closest approach is <= ``tol`` sv: one crossing per pair, at the closest approach
    (the midpoint of the two closest samples), with both fibers' arclengths there."""
    H = [k for k, f in enumerate(fibers) if f.family == 'H' and f.family_source != 'geometry']
    V = [k for k, f in enumerate(fibers) if f.family == 'V' and f.family_source != 'geometry']
    if not H or not V:
        return []
    vpts = np.concatenate([fibers[k].points for k in V])
    vown = np.concatenate([np.full(len(fibers[k].points), j) for j, k in enumerate(V)])
    vloc = np.concatenate([np.arange(len(fibers[k].points)) for k in V])
    tree = cKDTree(vpts)
    out = []
    for kh in H:
        fh = fibers[kh]
        dist, idx = tree.query(fh.points, k=1, distance_upper_bound=tol)
        hit = np.isfinite(dist)
        best = {}
        for i in np.flatnonzero(hit):
            j = int(vown[idx[i]])
            if j not in best or dist[i] < best[j][0]:
                best[j] = (float(dist[i]), int(i), int(vloc[idx[i]]))
        for j, (dmin, i, l) in sorted(best.items()):
            fv = fibers[V[j]]
            out.append(dict(h_fiber=fh.name, v_fiber=fv.name, closest_approach_sv=dmin,
                            point_sv=((fh.points[i]+fv.points[l])/2).tolist(), h_s=float(fh.s[i]), v_s=float(fv.s[l])))
    out.sort(key=lambda c: (c['h_fiber'], c['v_fiber']))
    for n, c in enumerate(out):
        c['id'] = f'X{n:04d}'
    return out


def crossing_windows(crossings, fibers_by_name):
    """Per crossing, a MIN_STRETCH_SV window on each fiber centred on the crossing (shifted to fit the fiber; a
    fiber shorter than the window gives none, and drops the crossing)."""
    half = MIN_STRETCH_SV/2
    windows, kept = [], []
    for c in crossings:
        ws = []
        for side in ('h', 'v'):
            f = fibers_by_name[c[f'{side}_fiber']]
            if f.length < MIN_STRETCH_SV:
                break
            s0 = min(max(0., c[f'{side}_s']-half), f.length-MIN_STRETCH_SV)
            ws.append(dict(id=f"{c['id']}{side.upper()}", crossing=c['id'], fiber=f.name, family=f.family,
                           annotator=f.annotator, s0=float(s0), s1=float(s0+MIN_STRETCH_SV)))
        if len(ws) == 2:
            c['windows'] = [w['id'] for w in ws]
            windows += ws
            kept.append(c)
    return kept, windows


def crossing_agreement(strips, fibers, seed=SEED, plan=None, log=print):
    """Where an H strip's fiber and a V strip's fiber cross (closest approach <= CROSS_TOL sv), the normal distance
    between the two predicted sheets: each strip's nodes within CROSS_RADIUS of the crossing projected onto the
    other strip's surface; per crossing the mean of the two medians. No ground truth. ``plan``: only these
    crossings (a crossplan's 'crossings' list); planned crossings without both strips are listed as missing."""
    index = FiberIndex(fibers)
    assoc = []
    for st in strips:
        k, med = associate(st, index)
        if k is not None:
            assoc.append((st, k))
    by_fiber = {}
    for st, k in assoc:
        by_fiber.setdefault(fibers[k].name, []).append(st)
    if plan is None:
        cr = find_crossings(fibers)
    else:
        cr = [dict(c) for c in plan]
    surfaces = {}

    def surface(st):
        if st.path not in surfaces:
            surfaces[st.path] = GridSurface(st.rows, st.valid, st.name)
        return surfaces[st.path]

    def near_strip(name, point):
        best, best_n = None, 0
        for st in by_fiber.get(name, []):
            n = int((st.valid & (np.linalg.norm(st.rows-point, axis=-1) <= CROSS_RADIUS)).sum())
            if n > best_n:
                best, best_n = st, n
        return best

    rows, missing = [], []
    for c in cr:
        point = np.asarray(c['point_sv'], np.float64)
        st_h, st_v = near_strip(c['h_fiber'], point), near_strip(c['v_fiber'], point)
        if st_h is None or st_v is None:
            missing.append(c['id'])
            continue
        meds = []
        for a, b in ((st_h, st_v), (st_v, st_h)):
            near = a.valid & (np.linalg.norm(a.rows-point, axis=-1) <= CROSS_RADIUS)
            if near.sum() < 5:
                continue
            d, cov, _, _ = surface(b).project(a.rows[near])
            if cov.sum() >= 5:
                meds.append(float(np.median(np.abs(d[cov]))))
        if meds:
            rows.append(dict(id=c['id'], h_fiber=c['h_fiber'], v_fiber=c['v_fiber'],
                             closest_approach_sv=c['closest_approach_sv'], point_sv=point.tolist(),
                             agreement_sv=float(np.mean(meds)), directions=len(meds)))
        else:
            missing.append(c['id'])
    base = dict(label='no ground truth: consistency between two predictions, not accuracy',
                planned=len(cr), missing=missing)
    if not rows:
        return dict(base, crossings=0, note='no H/V crossings with overlapping strips')
    a = np.array([r['agreement_sv'] for r in rows])
    rng = np.random.default_rng(seed+2)
    draws = rng.integers(0, len(a), size=(BOOTSTRAP, len(a)))
    reps = a[draws]
    ci = dict(mean=[float(np.percentile(reps.mean(1), q)) for q in (2.5, 97.5)],
              median=[float(np.percentile(np.median(reps, 1), q)) for q in (2.5, 97.5)],
              p90=[float(np.percentile(np.percentile(reps, 90, axis=1), q)) for q in (2.5, 97.5)],
              within2=[float(np.percentile((reps <= 2).mean(1), q)) for q in (2.5, 97.5)])
    tight = np.array([r['agreement_sv'] for r in rows if r['closest_approach_sv'] <= CROSS_TOL])
    return dict(base, crossings=len(rows), fibers_h=len({r['h_fiber'] for r in rows}),
                fibers_v=len({r['v_fiber'] for r in rows}),
                mean=float(a.mean()), median=float(np.median(a)), p90=float(np.percentile(a, 90)),
                within2=float((a <= 2).mean()), ci=ci,
                tight_subset=dict(rule=f'closest approach <= {CROSS_TOL} sv', crossings=int(len(tight)),
                                  mean=float(tight.mean()) if len(tight) else None,
                                  median=float(np.median(tight)) if len(tight) else None,
                                  within2=float((tight <= 2).mean()) if len(tight) else None),
                per_crossing=rows)


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


def _public(w):
    return {k: v for k, v in w.items() if not k.startswith('_')}


def _ct_inputs(args):
    if not args.ct_zarray:
        return None, None
    zarray = fetch_zarray(args.ct_zarray)
    path = None
    if args.ct_zarray.startswith(('http://', 'https://')):
        parts = args.ct_zarray.rstrip('/').split('/')
        path = '/'.join(parts[3:-1])  # <sample>/volumes/<name>.zarr (the level dir stripped)
    return zarray, path


def cmd_match(args):
    min_len = args.min_stretch_mm*1e3/SV_UM
    fibers, skipped = load_fibers(args.fibers, args.fiber_unit_base, args.span, args.fiber_urls, space=args.space)
    segs = discover_segments(args.segments, args.segment_urls, args.segment_unit_base)
    untagged = [f.name for f in fibers if f.family_source == 'geometry']
    tagged = [f for f in fibers if f.family_source != 'geometry']
    print(f'{len(fibers)} fibers ({len(skipped)} skipped, {len(untagged)} without an H/V tag), {len(segs)} segments',
          flush=True)
    axis = load_axis(args)
    hits, seglog = fiber_segment_distances(tagged, segs, axis, args.workers, args.tmpdir)
    names = [s.name for s in segs]
    stretches, near_abs, dropped = [], [], []
    for k, f in enumerate(tagged):
        acc, near, drop = find_stretches(f, hits[k], names, min_len)
        stretches += acc
        near_abs += near
        dropped += drop
    stretches.sort(key=lambda r: (r['fiber'], r['s0']))
    for n, st in enumerate(stretches):
        st['id'] = f'S{n:04d}'
    by_name = {f.name: f for f in tagged}
    kof = {f.name: k for k, f in enumerate(tagged)}
    match_abs = []
    for st in stretches:
        idx, d = hits[kof[st['fiber']]][st['segment_index']]
        sel = (idx >= st['i0']) & (idx <= st['i1'])
        match_abs.append(np.abs(d[sel]))
    g1 = gate_g1(stretches, match_abs, near_abs, len(fibers), min_len)
    # inference plan: pieces (one infer.py run each)
    pieces = []
    for fname in sorted({st['fiber'] for st in stretches}):
        pieces += plan_pieces(by_name[fname], [st for st in stretches if st['fiber'] == fname])
    number_pieces(pieces, 'P', stretches)
    zarray, vpath = _ct_inputs(args)
    plan = dict(pieces=len(pieces))
    if zarray is not None:
        cached = piece_ct(by_name, pieces, zarray, args.ct_cache, vpath)
        full = budget(pieces, zarray, cached)
        plan.update(volume_zarray=args.ct_zarray, zarray=dict(shape=zarray['shape'], chunks=zarray['chunks'],
                                                              dtype=zarray.get('dtype')), all_matched=full)
        over = full['cpu_hours'] > args.cpu_cap_h or full['ct_new_gib'] > args.ct_cap_gib
        if over:
            run_ids = stratified_subsample(pieces, lambda u: (u['segment'], u['family']), zarray, args.cpu_cap_h,
                                           args.ct_cap_gib, cached)
        else:
            run_ids = [pc['id'] for pc in pieces]
        order = {i: n for n, i in enumerate(run_ids)}
        for pc in pieces:
            pc['run'] = pc['id'] in order
            pc['run_order'] = order.get(pc['id'])
        run_p = [pc for pc in pieces if pc['run']]
        run_w = {i for pc in run_p for i in pc['windows']}
        for st in stretches:
            st['run'] = st['id'] in run_w
        run_st = [st for st in stretches if st['run']]
        plan.update(subsampled=bool(over), caps=dict(cpu_hours=args.cpu_cap_h, ct_new_gib=args.ct_cap_gib),
                    run=budget(run_p, zarray, cached),
                    run_stretches=dict(stretches=len(run_st), fibers=len({st['fiber'] for st in run_st}),
                                       length_mm=float(sum(st['length_mm'] for st in run_st)),
                                       by_family={fam: dict(stretches=sum(st['family'] == fam for st in run_st),
                                                            fibers=len({st['fiber'] for st in run_st
                                                                        if st['family'] == fam}))
                                                  for fam in ('H', 'V')},
                                       segments=len({st['segment'] for st in run_st})))
    out = dict(provenance=provenance(args), analysis=args.analysis, min_stretch_mm=args.min_stretch_mm,
               min_stretch_sv=min_len, segments=[dict(name=s.name, source=s.source) for s in segs],
               segment_log=seglog, fibers_skipped=skipped, fibers_untagged=untagged, G1=g1, plan=plan,
               stretches=[_public(st) for st in stretches], pieces=[_public(pc) for pc in pieces],
               dropped_overlaps=dropped,
               fibers=[dict(fiber=f.name, file=f.file, uuid=f.uuid, uuid_source=f.uuid_source, afv_id=f.afv_id,
                            family=f.family, family_source=f.family_source, length_sv=f.length,
                            length_mm=f.length*SV_UM*1e-3, line_offset=f.line_offset,
                            stretches=sum(st['fiber'] == f.name for st in stretches),
                            foldbacks=f.flags.get('foldbacks'), jumps=f.flags.get('jumps')) for f in fibers])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(jsonable(out), indent=1))
    print(json.dumps(jsonable(dict(G1=g1, plan=plan)), indent=1))


def cmd_crossplan(args):
    fibers, skipped = load_fibers(args.fibers, args.fiber_unit_base, args.span, args.fiber_urls, space=args.space)
    by_name = {f.name: f for f in fibers}
    crossings = find_crossings(fibers, args.cross_tol)
    kept, windows = crossing_windows(crossings, by_name)
    wid = {w['id']: w for w in windows}
    zarray, vpath = _ct_inputs(args)
    plan = dict(fibers=len(fibers), crossings_found=len(crossings), crossings_with_windows=len(kept),
                windows=len(windows))
    pieces = []
    if zarray is not None and kept:
        # the crossing is the unit of the subsample, costed by its two windows alone (own heads, own CT)
        units = []
        for c in kept:
            ws = [wid[i] for i in c['windows']]
            own = [plan_pieces(by_name[w['fiber']], [dict(w)])[0] for w in ws]
            ct = [ct_footprint(head_path_l0(by_name[o['fiber']], o['heads_sv']), CT_LEVEL, CT_RADIUS,
                               zarray['chunks'], zarray['shape']) for o in own]
            units.append(dict(id=c['id'], fiber=c['id'], n_runs=2, decisions=sum(o['decisions'] for o in own),
                              _ct=ct[0] | ct[1]))
        zs = np.array([c['point_sv'][2] for c in kept])
        edges = np.quantile(zs, [1/3, 2/3])
        for c, u in zip(kept, units):
            c['z_tercile'] = u['z_tercile'] = int(np.searchsorted(edges, c['point_sv'][2], 'right'))
        sep = zarray.get('dimension_separator', '.')
        cached = set()
        if args.ct_cache and vpath:
            root = Path(args.ct_cache)/'s3'/'vesuvius-challenge-open-data'/vpath/str(CT_LEVEL)
            allk = set().union(*[u['_ct'] for u in units])
            cached = {k for k in allk if (root/sep.join(map(str, k))).exists()}
        # strata: (0 for the tight crossings, closest approach <= CROSS_TOL, else 1; z tercile): the tight ones are
        # offered in the first round
        for c, u in zip(kept, units):
            u['tight'] = int(c['closest_approach_sv'] > CROSS_TOL)
        chosen = stratified_subsample(units, lambda u: (u['tight'], u['z_tercile']), zarray, args.cpu_cap_h,
                                      args.ct_cap_gib, cached, max_units=args.max_crossings)
        order = {i: n for n, i in enumerate(chosen)}
        for c in kept:
            c['run'] = c['id'] in order
            c['run_order'] = order.get(c['id'])
        for w in windows:
            w['run'] = w['crossing'] in order
        # the pieces of the chosen crossings' windows (merged per fiber), with the exact budget
        run_w = [w for w in windows if w['run']]
        for fname in sorted({w['fiber'] for w in run_w}):
            pieces += plan_pieces(by_name[fname], [w for w in run_w if w['fiber'] == fname])
        number_pieces(pieces, 'C', windows)
        for pc in pieces:
            pc['run'] = True
        piece_ct(by_name, pieces, zarray)
        plan.update(volume_zarray=args.ct_zarray, z_tercile_edges_sv=edges.tolist(),
                    caps=dict(cpu_hours=args.cpu_cap_h, ct_new_gib=args.ct_cap_gib, max_crossings=args.max_crossings),
                    run_crossings=len(chosen),
                    run_by_tercile={t: sum(bool(c['run']) and c['z_tercile'] == t for c in kept) for t in range(3)},
                    run=budget(pieces, zarray, cached),
                    all_crossings_estimate=dict(units=len(units), decisions=int(sum(u['decisions'] for u in units)),
                                                ct_gib=round(len(set().union(*[u['_ct'] for u in units]))
                                                             * int(np.prod(zarray['chunks']))/2**30, 3)))
    out = dict(provenance=provenance(args), analysis='consistency (no ground truth)', plan=plan,
               fibers_skipped=skipped, crossings=kept, windows=[_public(w) for w in windows],
               pieces=[_public(pc) for pc in pieces])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(jsonable(out), indent=1))
    print(json.dumps(jsonable(plan), indent=1))


def cmd_lasagna(args):
    matches = json.loads(Path(args.matches).read_text())
    stretches = matches['stretches']
    names = {st['fiber'] for st in stretches}
    fibers, _ = load_fibers(args.fibers, args.fiber_unit_base, args.span, args.fiber_urls, names=names)
    nx, ny = ZarrV2Chunks(args.nx), ZarrV2Chunks(args.ny)
    arrays = {}
    t0 = time.time()
    for k, f in enumerate(fibers):
        sel = np.zeros(len(f.s), bool)
        for st in stretches:
            if st['fiber'] == f.name:
                sel |= (f.s >= st['s0']-ROW_MAX_SV) & (f.s <= st['s1']+ROW_MAX_SV)
        n = np.full((len(f.s), 3), np.nan, np.float32)
        ok = np.zeros(len(f.s), bool)
        if sel.any():
            nn, oo = lasagna_normals_at(f.points[sel], nx, ny, args.level_scale)
            n[sel], ok[sel] = nn, oo
        arrays[f.name+'__n'] = n
        arrays[f.name+'__valid'] = ok
        print(f'[lasagna] {k+1}/{len(fibers)} {f.name}: {int(sel.sum())} samples, {ok.sum()} valid, '
              f'{len(nx.cache)} chunks, {time.time()-t0:.0f} s', flush=True)
        if len(nx.cache) > 4000:  # bound memory: ~32 KiB per chunk and array
            nx.cache.clear()
            ny.cache.clear()
    np.savez_compressed(args.out, **arrays)
    print(f'[lasagna] wrote {args.out}: {len(fibers)} fibers', flush=True)


def cmd_score(args):
    matches = json.loads(Path(args.matches).read_text())
    stretches = matches['stretches']
    if args.run_only:
        stretches = [st for st in stretches if st.get('run', True)]
    for n, st in enumerate(stretches):
        st['_n'] = n
    pieces = matches.get('pieces', [])
    names = {st['fiber'] for st in stretches}
    fibers, skipped = load_fibers(args.fibers, args.fiber_unit_base, args.span, args.fiber_urls, names=names)
    segs = discover_segments(args.segments, args.segment_urls, args.segment_unit_base)
    if [s.name for s in segs] != [s['name'] for s in matches['segments']]:
        raise SystemExit('the segments differ from those of the matches file (same list, same order required)')
    strips = [load_strip(d, args.strip_unit_base) for d in discover_strips(args.strips)]
    print(f'{len(strips)} strips, {len(fibers)} fibers, {len(stretches)} stretches, {len(segs)} segments', flush=True)
    normals = load_normals(args.lasagna_normals)
    axis = load_axis(args)
    entries, skips = score_strips(strips, fibers, stretches, segs, axis, normals, pieces=pieces)
    g1 = matches['G1']
    report = dict(provenance=provenance(args), matches_sha256=sha256_file(args.matches), G1=g1, skips=skips,
                  stretches_planned=len(stretches),
                  stretches_scored=len({s for e in entries.values() for s in e['stretches']}))
    report['accuracy'] = {fam or 'all': aggregate(entries, fam) for fam in (None, 'H', 'V')}
    report['G2'] = gate_g2(report['accuracy']['all'], g1)
    report['analysis'] = matches.get('analysis', 'primary')
    if report['analysis'] != 'primary':  # PREREG 6.3: the secondary set's verdict is labelled, never the primary one
        report['G2']['verdict'] = f"{report['analysis']}: {report['G2']['verdict']}"
    unflagged = {st['_n'] for st in stretches if not st.get('foldbacks') and not st.get('jumps')}
    sens = {'unflagged stretches (no fold-back or jump)': aggregate(entries, stretch_ids=unflagged,
                                                                    label='unflagged')}
    for prefix in sorted({st['annotator'] for st in stretches}):
        sens[f'annotator {prefix}'] = aggregate(entries, names={st['fiber'] for st in stretches
                                                                if st['annotator'] == prefix},
                                                label=f'annotator {prefix}')
    report['sensitivity'] = dict(note='reported, not gating (PREREG 4.9)', **sens)
    report['calibration'] = calibration(entries)
    report['crossings'] = crossing_agreement(strips, fibers) if args.crossings else None
    report['per_strip'] = [dict(strip=e['strip'], fiber=e['fiber'], family=e['family'], stretches=e['stretches'],
                                assoc_median_sv=e['assoc_median_sv'], notes=e['notes'],
                                **{f'{s}_nodes': int(np.isfinite(e['err'][s]).sum()) for s in e['err']},
                                **{f'{s}_mean_abs_sv': (float(np.nanmean(np.abs(e['err'][s])))
                                                        if np.isfinite(e['err'][s]).any() else None)
                                   for s in e['err']},
                                strip_signed_mean_sv=(float(np.nanmean(e['err']['strip']))
                                                      if np.isfinite(e['err']['strip']).any() else None),
                                fiber_floor_mean_abs_sv=(float(np.nanmean(np.abs(e['fiber_floor'])))
                                                         if np.isfinite(e['fiber_floor']).any() else None))
                           for e in entries.values()]
    Path(args.out).write_text(json.dumps(jsonable(report), indent=1))
    print(json.dumps(jsonable(dict(G2=report['G2'], all=report['accuracy']['all']['strip']['point'])), indent=1))


def cmd_crossings(args):
    plan = json.loads(Path(args.plan).read_text()) if args.plan else None
    crossings = None
    names = None
    if plan is not None:
        crossings = [c for c in plan['crossings'] if c.get('run', True) or args.all_planned]
        names = {c['h_fiber'] for c in crossings} | {c['v_fiber'] for c in crossings}
    fibers, _ = load_fibers(args.fibers, args.fiber_unit_base, args.span, args.fiber_urls, names=names)
    strips = [load_strip(d, args.strip_unit_base) for d in discover_strips(args.strips)]
    res = crossing_agreement(strips, fibers, plan=crossings)
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
        p.add_argument('--space', help="coordinate space for JSONs without one, e.g. 'PHerc0332/20251211183505@L0'")
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

    def ct(p, cpu, gib):
        p.add_argument('--ct-zarray', help='the CT level-2 array (dir or https URL) for the CT footprint')
        p.add_argument('--ct-cache', help="villa's remote-CT cache dir (cached chunks are not counted as downloads)")
        p.add_argument('--cpu-cap-h', type=float, default=cpu)
        p.add_argument('--ct-cap-gib', type=float, default=gib)

    p = sub.add_parser('match')
    common(p)
    ct(p, CPU_CAP_H_0139, CT_CAP_GIB_0139)
    p.add_argument('--min-stretch-mm', type=float, default=MIN_STRETCH_SV*SV_UM*1e-3,
                   help='shortest matched stretch (default 2.5 mm: the primary rule)')
    p.add_argument('--analysis', choices=('primary', 'secondary'), default='primary',
                   help="'secondary': the pre-registered secondary set (PREREG 4.10); its verdict is labelled")
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--tmpdir', help='where the shared sample array is written while matching')
    p.set_defaults(func=cmd_match)
    p = sub.add_parser('crossplan')
    common(p, segments=False)
    ct(p, CPU_CAP_H_0332, CT_CAP_GIB_0332)
    p.add_argument('--max-crossings', type=int, default=MAX_CROSSINGS_0332)
    p.add_argument('--cross-tol', type=float, default=CROSS_TOL_0332,
                   help='closest approach (sv) of an H and a V fiber that counts as a crossing')
    p.set_defaults(func=cmd_crossplan)
    p = sub.add_parser('lasagna')
    common(p, segments=False)
    p.add_argument('--matches', required=True)
    p.add_argument('--nx', required=True, help='Lasagna nx zarr array (level dir; local or https)')
    p.add_argument('--ny', required=True)
    p.add_argument('--level-scale', type=float, default=LASAGNA_SCALE)
    p.set_defaults(func=cmd_lasagna)
    p = sub.add_parser('score')
    common(p, strips=True)
    p.add_argument('--matches', required=True)
    p.add_argument('--lasagna-normals', help='.npz from the lasagna subcommand')
    p.add_argument('--crossings', action='store_true', help='also the H/V crossing agreement (0139 calibration)')
    p.add_argument('--run-only', action='store_true', help="only the stretches marked 'run' (the subsample)")
    p.set_defaults(func=cmd_score)
    p = sub.add_parser('crossings')
    common(p, segments=False, strips=True)
    p.add_argument('--plan', help='crossplan JSON: score its crossings marked run only')
    p.add_argument('--all-planned', action='store_true', help='with --plan: every planned crossing, not only run')
    p.set_defaults(func=cmd_crossings)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
