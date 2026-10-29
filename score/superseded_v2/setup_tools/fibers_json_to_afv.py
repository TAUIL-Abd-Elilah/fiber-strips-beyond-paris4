"""Convert VC3D fiber JSON files (type vc3d_fiber, versions 3 and 4) into an AFV (VC3D's Automated Fiber Volume,
SQLite, format version 1): the input of villa's sheet-strip inference (``sheet_strip/infer.py`` and ``infer_all.py``
--afv, villa 6f2024569; the strip model reads fibers only through ``shared/annotations/afv.AFVFibers``).

    python fibers_json_to_afv.py OUT.afv JSON_OR_DIR [...] [--volume S3_OR_HTTPS_ZARR] [--span controlled|full]
        [--base-shape Z Y X] [--map OUT.tsv]
    python fibers_json_to_afv.py --to-json AFV OUT_DIR       # AFV -> minimal vc3d_fiber v4 JSONs (round trips)

Coordinates. VC3D fiber JSON stores ``line_points`` and control points as x, y, z in level-0 voxels of one open-data
volume (``vc_open_data_coordinate_space`` 'SAMPLE/VOLUME_ID@L0', e.g. 'PHerc0139/20260102150214@L0' at 2.399 um).
An AFV stores native level-0 XYZ too (AFVFibers refuses any other level), so points are copied unchanged (float64,
bit for bit). The model's CT level is applied by villa at read time: AFVFibers divides by the checkpoint's
grid_scale (8: trace voxels) and the strip CT is level ``ct_level`` 2 (ct_grid_scale 4: one patch voxel = 4 L0
voxels = 9.596 um for the 2.399 um scans, 9.6 um for PHerc Paris 4). Strips come out in level-2 voxels of the
volume given here; pass the same volume to the inference (--ct). The JSON's coordinate space, base shape and source
path are checked against ``--volume`` (default: the JSON's own vc_open_data_source_path).

Geometry (``--span``): 'controlled' (default) keeps the line points between the first and the last control point,
as villa's annotation loader does (shared/annotations/vc3d.load_fibers: the tails beyond the outer controls are
extrapolations, not annotation); 'full' keeps every line point. Fibers with fewer than two control points (a seed
and its extrapolation) or fewer than two kept points are skipped and reported.

Identity: the AFV ``id`` (INTEGER) is the first 13 hex digits of the fiber's UUID (52 bits: stable across subsets,
exact in JSON/float64; a collision is an error). The UUID is the file's ``uuid`` (version-4 files written by newer
VC3D builds) or, when absent, uuid5(NAMESPACE_URL, 'vc3d_fiber:' + coordinate space + '/' + file name), marked
``uuid_source: 'uuid5(file)'``. ``name`` is the UUID; ``annotation`` (JSON text) keeps the file name, UUID and its
source, the family and its source, both H/V tags, the span, the JSON's sha256, version, username and counts.
Family: hv_classification.manual_tag when it is H or V, else automatic_tag (villa's fiber_tag reads automatic_tag
only; the annotation records both, and the report counts the disagreements); a fiber with neither is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid as uuidlib

import numpy as np

VILLA_SRC = Path(__file__).resolve().parents[1]/'villa_6f2'/'vesuvius'/'src'
if str(VILLA_SRC) not in sys.path:
    sys.path.insert(0, str(VILLA_SRC))

from vc3d_fiber_format import parse_vc3d_fiber_format  # noqa: E402
from vesuvius.neural_tracing.tracers.shared.annotations.afv import (  # noqa: E402
    AFVFibers, open_data_frame, open_data_volume, read_afv_polylines, write_afv)

CONVERTER = 'fibers_json_to_afv.py v1'
SPACE_RE = re.compile(r'^([^/]+)/(\d+)@L0$')


def volume_shape(url):
    """Level-0 shape (z, y, x) of an open-data zarr (its 0/.zarray over HTTPS)."""
    import requests
    https = open_data_volume(url)[3]
    meta = requests.get(f'{https}/0/.zarray', timeout=60).json()
    return [int(v) for v in meta['shape']]


def fiber_uuid(raw, space, path):
    if raw.get('uuid'):
        return str(raw['uuid']).lower(), 'json'
    return str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, f'vc3d_fiber:{space}/{Path(path).name}')), 'uuid5(file)'


def family_of(raw):
    hv = raw.get('hv_classification') or {}
    manual, auto = hv.get('manual_tag') or '', hv.get('automatic_tag') or ''
    if manual in ('H', 'V'):
        return manual, 'manual_tag', manual, auto
    if auto in ('H', 'V'):
        return auto, 'automatic_tag', manual, auto
    return None, None, manual, auto


def kept_points(parsed, span):
    """(xyz float64 (N, 3), (first, last) line indices) of the kept geometry."""
    line = np.asarray(parsed.line_points_xyz, np.float64)
    if span == 'full':
        return line, (0, len(line)-1)
    controls = np.asarray(parsed.control_points_xyz, np.float64)
    if len(controls) < 2:
        return None, (None, None)
    d = np.linalg.norm(line[None]-controls[:, None], axis=2)
    anchors = d.argmin(1)
    if np.any(d[np.arange(len(controls)), anchors] > 1e-4):
        raise ValueError('control points are not anchored in line_points')
    a, b = int(anchors.min()), int(anchors.max())
    return line[a:b+1], (a, b)


def convert(paths, out, volume=None, span='controlled', base_shape=None, map_path=None):
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob('*.json')) if p.is_dir() else [p]
    rows, report, ids = [], dict(files=len(files), converted=0, skipped=[], family_disagreements=0,
                                 uuid_from_json=0, uuid_derived=0, versions={}), {}
    spaces = set()
    for path in files:
        data = path.read_bytes()
        raw = json.loads(data)
        parsed = parse_vc3d_fiber_format(raw, path=path)
        if parsed.version not in (3, 4):
            report['skipped'].append((path.name, f'version {parsed.version}'))
            continue
        report['versions'][str(parsed.version)] = report['versions'].get(str(parsed.version), 0)+1
        source = raw.get('vc_open_data_source_path')
        if volume is None and source:
            volume = source
        space = raw.get('vc_open_data_coordinate_space')
        if space is None and volume is not None:  # files written without the tag (e.g. PHerc0332): the given volume
            sample, vid = open_data_volume(volume)[:2]
            space = f'{sample}/{vid}@L0'
            space_source = 'volume argument'
        else:
            space_source = 'json'
        if raw.get('vc_open_data_source_coordinate_level', 0) != 0 or raw.get(
                'vc_open_data_source_coordinate_scale_factor', 1) != 1:
            raise ValueError(f'{path}: coordinates are not native level 0')
        spaces.add(space)
        family, family_source, manual, auto = family_of(raw)
        if family is None:
            report['skipped'].append((path.name, 'no H/V tag'))
            continue
        report['family_disagreements'] += bool(manual in ('H', 'V') and auto in ('H', 'V') and manual != auto)
        xyz, (a, b) = kept_points(parsed, span)
        if xyz is None or len(xyz) < 2 or not np.isfinite(xyz).all():
            report['skipped'].append((path.name, 'fewer than two control points' if xyz is None else 'too few points'))
            continue
        fuuid, uuid_source = fiber_uuid(raw, space, path)
        report['uuid_from_json' if uuid_source == 'json' else 'uuid_derived'] += 1
        fid = int(fuuid.replace('-', '')[:13], 16)
        if fid in ids:
            raise ValueError(f'id collision: {path.name} and {ids[fid]}')
        ids[fid] = path.name
        annotation = dict(source_file=path.name, uuid=fuuid, uuid_source=uuid_source, family=family,
                          family_source=family_source, manual_tag=manual, automatic_tag=auto,
                          span=span, line_index_range=[a, b], line_points=len(parsed.line_points_xyz),
                          control_points=len(parsed.control_points_xyz), json_version=parsed.version,
                          json_sha256=hashlib.sha256(data).hexdigest(), username=raw.get('username'),
                          coordinate_space=space, coordinate_space_source=space_source, converter=CONVERTER)
        rows.append((dict(id=fid, name=fuuid, family=family, point_count=len(xyz), length=0., min_x=0., max_x=0.,
                          min_y=0., max_y=0., min_z=0., max_z=0., annotation=json.dumps(annotation, sort_keys=True)),
                     xyz))
    if len(spaces) > 1:
        raise ValueError(f'fibers from several coordinate spaces: {sorted(spaces)}')
    if volume is None:
        raise ValueError('no volume: pass --volume (the JSONs carry no vc_open_data_source_path)')
    sample, vid, um, https = open_data_volume(volume)
    space = next(iter(spaces)) if spaces else f'{sample}/{vid}@L0'
    m = SPACE_RE.match(space)
    if not m or (m[1], m[2]) != (sample, vid):
        raise ValueError(f'coordinate space {space} is not the volume {volume}')
    shape = list(base_shape) if base_shape else volume_shape(volume)
    for path in files:  # VC3D writes coordinate_base_shape_zyx as the L0 shape rounded up (+1 seen on 0139 and 0332)
        raw_shape = json.loads(path.read_bytes()).get('coordinate_base_shape_zyx')
        if raw_shape and any(abs(int(s)-t) > 1 for s, t in zip(raw_shape, shape)):
            raise ValueError(f'{path.name}: base shape {raw_shape} does not match the volume L0 shape {shape}')
    rows.sort(key=lambda r: r[0]['id'])
    frame = open_data_frame(f's3://vesuvius-challenge-open-data/{sample}/volumes/{Path(https).name}', shape)
    digest = hashlib.sha256(''.join(sorted(json.loads(r[0]['annotation'])['json_sha256'] for r in rows)).encode())
    metadata = dict(frame=frame['frame'], root=frame['root'],
                    uuid=str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, f'afv:{space}:{span}:{digest.hexdigest()}')),
                    source=f'VC3D fiber JSON ({CONVERTER})',
                    provenance=dict(converter=CONVERTER, span=span, inputs=[str(p) for p in paths],
                                    files=len(files), volume=volume, coordinate_space=space))
    out = Path(out)
    if out.exists():
        out.unlink()
    info = write_afv(None, out, rows, metadata)
    report.update(converted=len(rows), afv=str(out), write_afv=info, volume=volume, coordinate_space=space,
                  base_shape_zyx=shape, span=span)
    if map_path:
        with open(map_path, 'w') as fh:
            fh.write('afv_id\tuuid\tuuid_source\tfamily\tfile\tpoints\tlength_l0\n')
            for row, xyz in rows:
                a = json.loads(row['annotation'])
                length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
                fh.write(f"{row['id']}\t{row['name']}\t{a['uuid_source']}\t{row['family']}\t{a['source_file']}\t"
                         f"{len(xyz)}\t{length:.3f}\n")
    return report


def placeholder_segment():
    """A valid version-3 'cspline' segment_to_next (no trace diagnostics), for JSONs written back from an AFV."""
    import vc3d_fiber_format as fmt
    config = {k: (1 if k in fmt._INTEGER_CONFIG_KEYS else .5) for k in fmt._CONFIG_KEYS_V3}
    return dict(optimizer='native_fiber_trace3d', metadata_version=3, tracer_version=2, interp_goal='cspline',
                interp_mode='cspline', metric=None, msg='afv round trip', normal_manifest='', fiber_manifest='',
                trace_to_base_scale=1.0, meeting_error_base_voxels=None, meeting_error_ratio=None, meeting_source='',
                failure_code='', failure_detail='', lasagna_failure_code='', lasagna_failure_detail='', config=config)


def afv_to_json(afv, out_dir):
    """Minimal vc3d_fiber v4 JSONs from an AFV (the kept polyline as line_points, its two ends as control points,
    each fiber's uuid and family), for round-trip tests; segment metadata is a placeholder trace."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with __import__('sqlite3').connect(f'file:{Path(afv).resolve()}?mode=ro', uri=True) as c:
        meta = {k: json.loads(v) for k, v in c.execute('SELECT key,value FROM metadata')}
    space = meta['frame']['vc_open_data_coordinate_space']
    written = []
    for row, xyz in read_afv_polylines(afv):
        a = json.loads(row['annotation'])
        seg = placeholder_segment()
        obj = dict(type='vc3d_fiber', version=4, optimization_mode='native_fiber_trace3d', generation=1,
                   uuid=a['uuid'], line_points=xyz.tolist(),
                   control_points=[dict(position=xyz[0].tolist(), segment_to_next=seg), dict(position=xyz[-1].tolist())],
                   hv_classification=dict(manual_tag=a['manual_tag'], automatic_tag=a['automatic_tag']),
                   vc_open_data_coordinate_space=space+('' if space.endswith('@L0') else '@L0'),
                   vc_open_data_source_coordinate_level=0, vc_open_data_source_coordinate_scale_factor=1,
                   vc_open_data_source_path=meta['root'].get('vc_open_data_source_path'),
                   filename=a['source_file'])
        path = out_dir/a['source_file']
        path.write_text(json.dumps(obj))
        written.append(path)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', nargs='?')
    ap.add_argument('inputs', nargs='*')
    ap.add_argument('--volume', help='open-data CT zarr (s3:// or https://) the fibers are in (default: the JSON\'s)')
    ap.add_argument('--span', choices=('controlled', 'full'), default='controlled')
    ap.add_argument('--base-shape', type=int, nargs=3, metavar=('Z', 'Y', 'X'),
                    help="the volume's level-0 shape (default: read from its 0/.zarray)")
    ap.add_argument('--map', help='write a TSV of afv id, uuid, family, file')
    ap.add_argument('--to-json', nargs=2, metavar=('AFV', 'OUT_DIR'))
    args = ap.parse_args(argv)
    if args.to_json:
        print(len(afv_to_json(*args.to_json)), 'files')
        return
    report = convert(args.inputs, args.out, args.volume, args.span, args.base_shape, args.map)
    fibers = AFVFibers(args.out, 8., split='all')  # what the strip inference opens (grid_scale 8)
    report['afvfibers_open'] = dict(fibers=len(fibers), lengths_trace_vox=[round(float(fibers[i].length), 2)
                                                                           for i in range(len(fibers))])
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
