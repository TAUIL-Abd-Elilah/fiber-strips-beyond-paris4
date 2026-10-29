"""Write the AFV of the planned fiber pieces (score_strips.py match / crossplan) with fibers_json_to_afv.py's
conventions, and the exact infer.py run plan.

    python make_piece_afv.py PLAN.json FIBER_DIR OUT.afv --volume S3_OR_HTTPS_ZARR [--base-shape Z Y X]
        [--run-only] --full-afv FULL.afv --map OUT.tsv --runplan OUT_runplan.tsv --report OUT_report.json

Each piece is a contiguous run of the parent JSON's line_points (inside its controlled span, the span
fibers_json_to_afv.py keeps), copied bit for bit (float64, native level-0 XYZ). Identity: uuid5(URL,
'vc3d_fiber_piece:' + parent uuid + ':' + first + '-' + last line index); AFV id = its first 13 hex digits (as the
converter derives ids from uuids). The parent uuid is the converter's (the JSON's uuid, else uuid5 of the
coordinate space and file name). ``annotation`` keeps the converter's fields plus the piece's parent, line index
range, parent arclength range and windows.

Run plan (v3). The AFV is opened as the strip inference opens it (AFVFibers grid_scale 8: trace voxels; densified
and kink-repaired). A piece's planned heads are those of a full infer_all run of its parent fiber, k = head_k[0] ..
head_k[1] (score_strips.FullRunMap: 12 + 8k trace voxels on the FULL AFV's repaired geometry, ``--full-afv``). The
first one is placed on the piece's own repaired geometry through the exact raw <-> repaired vertex map of the piece
(tools/full_run_map.fiber_map; the raw arclength of that head is the plan's ``heads_sv[0]``), and infer.py gets
``--arc START END`` in patch voxels (2 per trace voxel) with START = that position and END = START + 16 (n - 1) +
END_PAD_SV, so that infer.py's own head code (``lo, hi = max(START/2, 12), min(END/2, length)``;
``numpy.arange(lo, hi + 1e-9, 8)``) makes exactly the n planned decisions. Each of them is compared with the
full run's head k (3-D distance on the two repaired geometries, sv): ``max_head_dev_sv`` per piece. The tool
stops when a piece's decision count differs from the plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid as uuidlib

import numpy as np

HERE = Path(__file__).resolve().parent
STRIPS = HERE.parents[1]
sys.path.insert(0, str(STRIPS/'tools'))
import fibers_json_to_afv as conv  # noqa: E402  (also puts villa_6f2/vesuvius/src on sys.path)
from vc3d_fiber_format import parse_vc3d_fiber_format  # noqa: E402
from vesuvius.neural_tracing.tracers.shared.annotations.afv import (  # noqa: E402
    AFVFibers, open_data_frame, open_data_volume, read_afv_polylines, write_afv)
from vesuvius.neural_tracing.tracers.shared.geometry import interp_at  # noqa: E402
sys.path.insert(0, str(HERE))
from full_run_map import fiber_map  # noqa: E402

TOOL = 'make_piece_afv.py v1 (fibers_json_to_afv.py v1 conventions)'  # AFV conventions unchanged since v2
RUNPLAN_TOOL = 'make_piece_afv.py v3 run plan (heads on the full run grid)'
END_PAD_SV = 1.0     # --arc END beyond the last head (patch voxels = sv): half a trace voxel, well under one step (16)
MIN_HEAD_ARC = 12.   # trace voxels (infer.py)
STEP_TRACE = 8.      # --step (trace voxels)
TRACE_PER_PATCH = .5  # ct_grid_scale 4 / grid_scale 8 of the checkpoint (tests/test_piece_afv.py reads it from it)


def infer_heads(arc_start, arc_end, length_trace):
    """The decision arclengths (trace voxels) infer.py makes for --arc ARC_START ARC_END (patch voxels) on a fiber of
    ``length_trace``: its own two lines (sheet_strip/infer.py at 6f2024569, main)."""
    lo, hi = arc_start*TRACE_PER_PATCH, arc_end*TRACE_PER_PATCH
    lo, hi = max(lo, MIN_HEAD_ARC), min(hi, length_trace)
    return np.arange(lo, hi+1e-9, STEP_TRACE) if hi > lo else np.zeros(0)


def piece_uuid(parent_uuid, a, b):
    return str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, f'vc3d_fiber_piece:{parent_uuid}:{a}-{b}'))


def build_rows(plan, fiber_dir, volume, run_only):
    windows = {w['id']: w for w in plan.get('stretches', plan.get('windows', []))}
    pieces = [p for p in plan['pieces'] if p.get('run', True) or not run_only]
    rows, ids, space_seen, report = [], {}, set(), dict(pieces=len(pieces), checks=[])
    for p in pieces:
        path = Path(fiber_dir)/p['parent_file']
        data = path.read_bytes()
        raw = json.loads(data)
        parsed = parse_vc3d_fiber_format(raw, path=path)
        space = raw.get('vc_open_data_coordinate_space')
        if space is None:
            sample, vid = open_data_volume(volume)[:2]
            space = f'{sample}/{vid}@L0'
        space_seen.add(space)
        family, family_source, manual, auto = conv.family_of(raw)
        if family != p['family']:
            raise ValueError(f"{path.name}: family {family} != plan {p['family']}")
        _, (ca, cb) = conv.kept_points(parsed, 'controlled')
        A, B = p['line_index_range']
        if not (ca <= A < B <= cb):
            raise ValueError(f'{path.name}: piece {A}-{B} outside the controlled span {ca}-{cb}')
        line = np.asarray(parsed.line_points_xyz, np.float64)
        xyz = line[A:B+1]
        parent_uuid, uuid_source = conv.fiber_uuid(raw, space, path)
        if parent_uuid != p['parent_uuid']:
            raise ValueError(f"{path.name}: parent uuid {parent_uuid} != plan {p['parent_uuid']}")
        pu = piece_uuid(parent_uuid, A, B)
        fid = int(pu.replace('-', '')[:13], 16)
        if pu != p['uuid'] or fid != p['afv_id']:
            raise ValueError(f"{p['id']}: uuid/id differ from the plan")
        if fid in ids:
            raise ValueError(f"id collision: {p['id']} and {ids[fid]}")
        ids[fid] = p['id']
        raw_len_sv = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())/4.
        annotation = dict(source_file=path.name, uuid=pu, uuid_source='uuid5(parent uuid, line index range)',
                          family=family, family_source=family_source, manual_tag=manual, automatic_tag=auto,
                          span='piece', line_index_range=[A, B], line_points=len(parsed.line_points_xyz),
                          control_points=len(parsed.control_points_xyz), json_version=parsed.version,
                          json_sha256=hashlib.sha256(data).hexdigest(), username=raw.get('username'),
                          coordinate_space=space, coordinate_space_source='json' if raw.get(
                              'vc_open_data_coordinate_space') else 'volume argument', converter=TOOL,
                          piece=dict(id=p['id'], parent_uuid=parent_uuid, parent_uuid_source=uuid_source,
                                     parent_afv_id=p['parent_afv_id'], controlled_span=[ca, cb],
                                     parent_arc_sv=p['parent_arc_sv'], windows=p['windows']))
        rows.append((dict(id=fid, name=pu, family=family, point_count=len(xyz), length=0., min_x=0., max_x=0.,
                          min_y=0., max_y=0., min_z=0., max_z=0., annotation=json.dumps(annotation, sort_keys=True)),
                     xyz))
        report['checks'].append(dict(piece=p['id'], afv_id=fid, points=len(xyz), length_sv_json=raw_len_sv,
                                     length_sv_plan=p['length_sv']))
    if len(space_seen) > 1:
        raise ValueError(f'pieces from several coordinate spaces: {sorted(space_seen)}')
    return rows, windows, pieces, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('plan')
    ap.add_argument('fiber_dir')
    ap.add_argument('out')
    ap.add_argument('--volume', required=True)
    ap.add_argument('--base-shape', type=int, nargs=3, metavar=('Z', 'Y', 'X'))
    ap.add_argument('--run-only', action='store_true', help="only the pieces holding a window marked 'run'")
    ap.add_argument('--full-afv', required=True, help="the full AFV of the scroll (the full run's geometry)")
    ap.add_argument('--map', required=True)
    ap.add_argument('--runplan', required=True)
    ap.add_argument('--report', required=True)
    a = ap.parse_args(argv)
    plan = json.loads(Path(a.plan).read_text())
    rows, windows, pieces, report = build_rows(plan, a.fiber_dir, a.volume, a.run_only)
    sample, vid, um, https = open_data_volume(a.volume)
    space = f'{sample}/{vid}@L0'
    shape = list(a.base_shape) if a.base_shape else conv.volume_shape(a.volume)
    rows.sort(key=lambda r: r[0]['id'])
    frame = open_data_frame(f's3://vesuvius-challenge-open-data/{sample}/volumes/{Path(https).name}', shape)
    digest = hashlib.sha256(''.join(r[0]['name'] for r in rows).encode()).hexdigest()
    metadata = dict(frame=frame['frame'], root=frame['root'],
                    uuid=str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, f'afv:{space}:pieces:{digest}')),
                    source=f'VC3D fiber JSON pieces ({TOOL})',
                    provenance=dict(converter=TOOL, plan=str(a.plan), plan_sha256=hashlib.sha256(
                        Path(a.plan).read_bytes()).hexdigest(), fiber_dir=str(a.fiber_dir), volume=a.volume,
                                    coordinate_space=space, run_only=a.run_only))
    out = Path(a.out)
    if out.exists():
        out.unlink()
    info = write_afv(None, out, rows, metadata)
    with open(a.map, 'w') as fh:
        fh.write('afv_id\tuuid\tfamily\tparent_file\tpiece\tline_first\tline_last\tpoints\tlength_l0\n')
        for row, xyz in rows:
            ann = json.loads(row['annotation'])
            length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
            fh.write(f"{row['id']}\t{row['name']}\t{row['family']}\t{ann['source_file']}\t{ann['piece']['id']}\t"
                     f"{ann['line_index_range'][0]}\t{ann['line_index_range'][1]}\t{len(xyz)}\t{length:.3f}\n")
    # the AFV as the strip inference opens it; one infer.py run per piece
    afv = AFVFibers(str(out), 8., split='all')
    full = AFVFibers(str(a.full_afv), 8., split='all')
    by_piece = {p['id']: p for p in pieces}
    lines, dec_plan, dec_run, worst_len, worst_dev = [], 0, 0, 0., 0.
    rows_by_id = {r[0]['id']: r[1] for r in rows}
    for chk in report['checks']:
        p = by_piece[chk['piece']]
        geo = afv[afv.id_to_index[p['afv_id']]]
        chk['length_sv_afvfibers'] = float(geo.length*2.)
        chk['kink_repairs'] = int(getattr(geo, 'kink_repairs', 0) or 0)
        worst_len = max(worst_len, abs(chk['length_sv_afvfibers']-chk['length_sv_json']))
        # the piece's raw <-> repaired vertex map (what AFVFibers does, checked vertex for vertex)
        s_raw, s_rep, pts, _ = fiber_map(rows_by_id[p['afv_id']])
        if not np.array_equal(pts, np.asarray(geo.points)):
            raise SystemExit(f"{p['id']}: piece geometry differs from AFVFibers")
        k0, k1 = p['head_k']
        n = k1-k0+1
        if n != p['decisions']:
            raise SystemExit(f"{p['id']}: head_k {p['head_k']} != {p['decisions']} decisions")
        x = (p['heads_sv'][0]-p['parent_arc_sv'][0])/2.  # raw piece arclength of the first head, trace voxels
        t0 = float(np.interp(x, s_raw, s_rep))
        start = round(2.*t0, 6)
        end = round(start+16.*(n-1)+END_PAD_SV, 6)
        arcs = infer_heads(float(f'{start:.6f}'), float(f'{end:.6f}'), float(geo.length))
        if len(arcs) != n:
            raise SystemExit(f"{p['id']}: infer.py would make {len(arcs)} decisions, the plan {n}")
        parent = full[full.id_to_index[p['parent_afv_id']]]
        ref = interp_at(np.asarray(parent.points), np.asarray(parent.s), MIN_HEAD_ARC+STEP_TRACE*np.arange(k0, k1+1))
        got = interp_at(np.asarray(geo.points), np.asarray(geo.s), arcs)
        dev = np.linalg.norm(got-ref, axis=1)*2.  # trace voxels -> sv
        worst_dev = max(worst_dev, float(dev.max()))
        chk.update(arc_patch_vox=[start, end], decisions=int(len(arcs)), decisions_planned=p['decisions'],
                   head_k=[int(k0), int(k1)], first_head_trace=float(arcs[0]), max_head_dev_sv=float(dev.max()),
                   mean_head_dev_sv=float(dev.mean()), clamped_to_min_head=bool(t0 < MIN_HEAD_ARC))
        dec_plan += p['decisions']
        dec_run += int(len(arcs))
        segs = sorted({windows[w].get('segment', windows[w].get('crossing', '')) for w in p['windows']})
        lines.append((p['id'], p['afv_id'], p['family'], p['fiber'], f'{start:.6f}', f'{end:.6f}', int(len(arcs)),
                      p['decisions'], bool(p.get('run', True)), p.get('run_order'), ','.join(p['windows']),
                      ','.join(segs), p['parent_afv_id'], k0, k1, round(float(dev.max()), 4)))
    with open(a.runplan, 'w') as fh:
        fh.write('piece\tafv_id\tfamily\tfiber\tarc_start_patch_vox\tarc_end_patch_vox\tdecisions\t'
                 'decisions_planned\trun\trun_order\twindows\tsegments_or_crossings\tparent_afv_id\thead_k_first\t'
                 'head_k_last\tmax_head_dev_sv\n')
        for line in lines:
            fh.write('\t'.join(map(str, line))+'\n')
    if dec_run != dec_plan:
        raise SystemExit(f'run plan {dec_run} decisions != plan {dec_plan}')
    devs = {c['piece']: c['max_head_dev_sv'] for c in report['checks']}
    report.update(afv=str(out), write_afv=info, volume=a.volume, base_shape_zyx=shape, pieces_written=len(lines),
                  runplan_tool=RUNPLAN_TOOL, full_afv=str(a.full_afv), end_pad_sv=END_PAD_SV,
                  decisions_runplan=dec_run, decisions_plan=dec_plan,
                  max_length_diff_sv_afvfibers_vs_json=worst_len, max_head_dev_sv_vs_full_run=worst_dev,
                  pieces_head_dev_over_1sv=sorted(k for k, v in devs.items() if v > 1.),
                  kink_repaired_pieces=sum(c['kink_repairs'] > 0 for c in report['checks']),
                  infer_command=('python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer --afv AFV '
                                 '--fiber AFV_ID --arc ARC_START ARC_END --out OUT/<piece> --checkpoint '
                                 'scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e '
                                 '--ct <the volume> --umbilicus <its umbilicus, level-2 voxels> --cache-dir '
                                 'C:/VesuviusTemp/strips/ct_cache --device cpu --index none --no-render '
                                 '--step 8 --blend 16 --trim 75 --batch 32 (CUDA_VISIBLE_DEVICES=-1; '
                                 'runs/v3/run_all.ps1: 8 torch threads, BelowNormal, one process at a time)'))
    Path(a.report).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=1))


if __name__ == '__main__':
    main()
