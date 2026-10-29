"""Write the arclength map of a full ``infer_all`` run for every fiber of a full AFV (PREREG v3, 4.6).

    python tools/full_run_map.py FULL.afv OUT.npz

A full run (villa sheet_strip/infer_all.py at 6f2024569) opens each fiber with AFVFibers (grid_scale 8: trace
voxels; every segment densified to <= 1 trace voxel by ``dense_line``, then ``repair_kinks``) and puts its decision
heads at ``numpy.arange(MIN_HEAD_ARC = 12, length + 1e-9, 8)`` trace voxels of that repaired geometry, i.e. at
24 + 16k sv. The scorer plans on the raw JSON polyline (sv = 2 trace voxels). ``repair_kinks`` keeps the vertex
count of the densified line and only moves vertices inside a repaired run, so densified vertex i before repair and
vertex i after it are the same annotation point: this gives the exact map between the raw arclength s and the
full-run arclength S at every vertex (linear between vertices). The map is stored as its breakpoints: the vertices
where s - S changes (inside or at the ends of repaired runs) plus both ends; between breakpoints s - S is
constant, so linear interpolation over them reproduces the map at every vertex.

Per fiber (AFV id): ``m<id>_raw`` and ``m<id>_full`` (sv, increasing), and in the arrays ``ids``, ``files``,
``length_raw_sv``, ``length_full_sv``, ``kink_repairs``, ``heads`` (the number of full-run decisions). The geometry
is recomputed here with villa's own functions and checked against AFVFibers' (vertex for vertex, exactly).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

STRIPS = Path(__file__).resolve().parents[2]
VILLA_SRC = STRIPS/'villa_6f2'/'vesuvius'/'src'
if str(VILLA_SRC) not in sys.path:
    sys.path.insert(0, str(VILLA_SRC))

from vesuvius.neural_tracing.tracers.shared.annotations.afv import AFVFibers, read_afv_polylines  # noqa: E402
from vesuvius.neural_tracing.tracers.shared.annotations.repair import repair_kinks  # noqa: E402
from vesuvius.neural_tracing.tracers.shared.geometry import arclength, dense_line  # noqa: E402

TOOL = 'full_run_map.py v1'
GRID_SCALE = 8.     # AFVFibers grid_scale of the strip model (trace voxels)
SV_PER_TRACE = 2.   # 8 base voxels per trace voxel, 4 per sv
MIN_HEAD_ARC = 12.  # trace voxels (infer.py / infer_all.py)
STEP = 8.           # trace voxels (--step, infer_all default)


def fiber_map(raw_l0):
    """(raw s, full S, repaired points, kink runs) of one fiber's raw L0 polyline, both arclengths in trace voxels."""
    dense = dense_line(np.asarray(raw_l0, np.float64)/GRID_SCALE, 1.)
    s_raw = arclength(dense)
    repaired, runs = repair_kinks(dense, s_raw)
    return s_raw, arclength(repaired), repaired, runs


def breakpoints(s_raw, s_full, tol=1e-9):
    """Indices of the vertices where s - S changes (both ends always kept)."""
    d = s_raw-s_full
    keep = np.zeros(len(d), bool)
    keep[[0, -1]] = True
    ch = np.abs(np.diff(d)) > tol
    keep[1:] |= ch
    keep[:-1] |= ch
    return np.flatnonzero(keep)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    afv_path, out = Path(argv[0]), Path(argv[1])
    t0 = time.time()
    afv = AFVFibers(str(afv_path), GRID_SCALE, split='all')
    polylines = {row['id']: (row, xyz) for row, xyz in read_afv_polylines(afv_path)}
    arrays, ids, files, lraw, lfull, kinks, heads, nbp = {}, [], [], [], [], [], [], []
    worst = 0.
    for idx in range(len(afv)):
        fid = int(afv.catalog[idx][0])
        row, xyz = polylines[fid]
        s_raw, s_full, repaired, runs = fiber_map(xyz)
        geo = afv[idx]
        ref = np.asarray(geo.points)
        if ref.shape != repaired.shape or not np.array_equal(ref, repaired):
            raise SystemExit(f'fiber {fid}: geometry differs from AFVFibers')
        worst = max(worst, float(abs(geo.length-s_full[-1])))
        k = breakpoints(s_raw, s_full)
        arrays[f'm{fid}_raw'] = s_raw[k]*SV_PER_TRACE
        arrays[f'm{fid}_full'] = s_full[k]*SV_PER_TRACE
        ids.append(fid)
        files.append(json.loads(row['annotation'])['source_file'])
        lraw.append(s_raw[-1]*SV_PER_TRACE)
        lfull.append(s_full[-1]*SV_PER_TRACE)
        kinks.append(len(runs))
        heads.append(len(np.arange(MIN_HEAD_ARC, s_full[-1]+1e-9, STEP)))
        nbp.append(len(k))
    h = hashlib.sha256()
    with open(afv_path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    meta = dict(tool=TOOL, afv=str(afv_path), afv_sha256=h.hexdigest(), fibers=len(ids),
                kink_repaired_fibers=int(sum(k > 0 for k in kinks)), full_run_decisions=int(sum(heads)),
                max_length_diff_vs_afvfibers_trace=worst, units='sv (4 base voxels; 2 per trace voxel)',
                villa='C:/VesuviusTemp/strips/villa_6f2 (6f2024569 + #2163 + normalization patch)',
                rule='heads of a full run: S = 24 + 16k sv (arange(12, length + 1e-9, 8) trace voxels) on the '
                     'repaired geometry; raw arclength s = interp(S, m<id>_full, m<id>_raw)')
    np.savez_compressed(out, ids=np.asarray(ids, np.int64), files=np.asarray(files), length_raw_sv=np.asarray(lraw),
                        length_full_sv=np.asarray(lfull), kink_repairs=np.asarray(kinks, np.int64),
                        heads=np.asarray(heads, np.int64), meta=np.asarray(json.dumps(meta)), **arrays)
    meta.update(breakpoints=int(sum(nbp)), seconds=round(time.time()-t0, 1))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
