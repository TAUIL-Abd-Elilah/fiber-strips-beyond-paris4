"""Prefetch the level-2 CT chunks of the PREREG v3 run sets into villa's remote-CT cache (CPU only, no strip output).

    python -I prefetch_v3.py --set primary|secondary|0332 [--workers 2] [--min-free-gib 10] [--log LOG]
    python -I prefetch_v3.py --init-metadata        # write each volume's cache .zarray/.zattrs (villa's own code)

The chunks of a set are those the frozen plan budgets: per run piece, the level-2 chunks within CT_RADIUS (170)
level-2 voxels of its decision path (score_strips.ct_footprint of head_path_l0, exactly as the plan computed them;
the union is checked against the plan's ``run.ct_chunks``). Chunks already in the cache are skipped (resume-safe).
Each chunk is fetched over HTTPS from the open-data bucket and written atomically in villa's cache layout
(RemoteChunkedArray: CACHE/s3/<bucket>/<zarr path>/2/<z>/<y>/<x>, decoded, uncompressed C-order bytes, edge chunks
padded with the fill value, a missing object -> a chunk of the fill value), as the setup run's tools/prefetch_ct.py
did for G0 (whose streamed and prefetched strips were bitwise identical). At most ``--workers`` (2) connections.
Before each batch of ``--batch`` chunks the free space on C: is checked: below ``--min-free-gib`` the script stops
with exit code 3. Progress (chunks, MiB, MiB/s) goes to stdout and ``--log``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import requests

SCORE = Path(__file__).resolve().parents[2]
STRIPS = SCORE.parent
sys.path.insert(0, str(SCORE))
import score_strips as S  # noqa: E402

BUCKET = 'vesuvius-challenge-open-data'
HTTPS = f'https://{BUCKET}.s3.us-east-1.amazonaws.com'
CACHE = STRIPS/'ct_cache'
VOL = {'0139': 'PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr',
       '0332': 'PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr'}
SETS = {'primary': ('results/matched_stretches.json', 'data/fibers0139', 'PHerc0139/20260102150214@L0', '0139'),
        'secondary': ('results/matched_stretches_secondary_run40.json', 'data/fibers0139',
                      'PHerc0139/20260102150214@L0', '0139'),
        '0332': ('results/crossings_0332_plan.json', 'data/fibers0332', 'PHerc0332/20251211183505@L0', '0332')}
LEVEL = S.CT_LEVEL


def say(msg, log=None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [prefetch] {msg}"
    print(line, flush=True)
    if log:
        with open(log, 'a', encoding='utf-8') as fh:
            fh.write(line+'\n')


def remote_zarray(vol):
    r = requests.get(f'{HTTPS}/{vol}/{LEVEL}/.zarray', timeout=60)
    r.raise_for_status()
    meta = r.json()
    assert meta['compressor'] is None and meta['dtype'] == '|u1' and meta.get('filters') is None, meta
    return meta


def cache_root(vol):
    return CACHE/'s3'/BUCKET/vol/str(LEVEL)


def init_metadata(log=None):
    """The cache's .zarray/.zattrs of each volume, written by villa's RemoteChunkedArray.publish_metadata (as villa
    writes them when it opens the remote store) from the remote .zarray; existing ones are left alone."""
    sys.path.insert(0, str(STRIPS/'villa_6f2'/'vesuvius'/'src'))
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import RemoteChunkedArray
    for key, vol in VOL.items():
        root = cache_root(vol)
        if (root/'.zarray').exists():
            say(f'{key}: cache metadata present ({root})', log)
            continue
        meta = remote_zarray(vol)
        md = SimpleNamespace(fill_value=meta.get('fill_value'), dimension_separator=meta.get('dimension_separator', '.'))
        shim = SimpleNamespace(shape=tuple(meta['shape']), chunks=tuple(meta['chunks']), dtype=np.dtype(meta['dtype']),
                               metadata=md)
        root.mkdir(parents=True, exist_ok=True)
        RemoteChunkedArray.publish_metadata(root, shim, f's3://{BUCKET}/{vol}', LEVEL)
        say(f'{key}: wrote cache metadata {root/".zarray"}: {(root/".zarray").read_text()}', log)


def set_keys(name):
    plan_path, fibers_dir, space, scroll = SETS[name]
    plan = json.loads((SCORE/plan_path).read_text())
    pieces = [p for p in plan['pieces'] if p.get('run', True)]
    names = {p['fiber'] for p in pieces}
    fibers, _ = S.load_fibers(SCORE/fibers_dir, names=names, space=space)
    by = {f.name: f for f in fibers}
    meta = remote_zarray(VOL[scroll])
    keys = set()
    for p in pieces:
        keys |= S.ct_footprint(S.head_path_l0(by[p['fiber']], p['heads_sv']), LEVEL, S.CT_RADIUS, meta['chunks'],
                               meta['shape'])
    want = plan['plan']['run']['ct_chunks']
    if len(keys) != want:
        raise SystemExit(f'{name}: {len(keys)} chunks, the plan has {want}')
    return sorted(keys), meta, scroll, len(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=sorted(SETS))
    ap.add_argument('--init-metadata', action='store_true')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--min-free-gib', type=float, default=10.)
    ap.add_argument('--log')
    ap.add_argument('--status', help='JSON file updated with the progress (key "prefetch")')
    a = ap.parse_args()
    if a.workers > 2:
        raise SystemExit('at most 2 connections')
    if a.init_metadata:
        init_metadata(a.log)
        return
    keys, meta, scroll, npieces = set_keys(a.set)
    vol = VOL[scroll]
    root = cache_root(vol)
    if not (root/'.zarray').exists():
        raise SystemExit(f'{root}/.zarray missing: run --init-metadata first')
    chunks, shape, sep = meta['chunks'], meta['shape'], meta.get('dimension_separator', '.')
    fill = meta.get('fill_value') or 0
    nbytes = int(np.prod(chunks))
    todo = [k for k in keys if not (root/sep.join(map(str, k))).exists()]
    say(f'{a.set}: {npieces} pieces, {len(keys)} chunks ({len(keys)*nbytes/2**30:.2f} GiB), '
        f'{len(keys)-len(todo)} cached, {len(todo)} to fetch ({len(todo)*nbytes/2**30:.2f} GiB), '
        f'{a.workers} connections', a.log)
    session = requests.Session()
    session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=a.workers, pool_maxsize=a.workers))

    def fetch(k):
        url = f'{HTTPS}/{vol}/{LEVEL}/{sep.join(map(str, k))}'
        for attempt in range(8):
            try:
                r = session.get(url, timeout=300)
                if r.status_code == 404:
                    data = np.full(chunks, fill, np.uint8).tobytes()
                elif r.status_code == 200 and len(r.content) == nbytes:
                    data = r.content
                else:
                    raise IOError(f'HTTP {r.status_code}, {len(r.content)} bytes')
                if any((ki+1)*c > s for ki, c, s in zip(k, chunks, shape)):  # villa pads the clipped edge
                    arr = np.frombuffer(data, np.uint8).reshape(chunks).copy()
                    for ax, (ki, c, s) in enumerate(zip(k, chunks, shape)):
                        if s-ki*c < c:
                            idx = [slice(None)]*3
                            idx[ax] = slice(s-ki*c, None)
                            arr[tuple(idx)] = fill
                    data = arr.tobytes()
                dest = root/sep.join(map(str, k))
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name+f'.tmp{os.getpid()}')
                tmp.write_bytes(data)
                os.replace(tmp, dest)
                return r.status_code, len(r.content)
            except Exception as error:  # noqa: BLE001
                if attempt == 7:
                    raise
                say(f'retry {k}: {error}', a.log)
                time.sleep(5*(attempt+1))

    t0, done, got, missing = time.time(), 0, 0, 0
    with ThreadPoolExecutor(a.workers) as pool:
        for i in range(0, len(todo), a.batch):
            free = shutil.disk_usage('C:/').free/2**30
            if free < a.min_free_gib:
                say(f'STOP: C: has {free:.2f} GiB free (< {a.min_free_gib:g}); {done}/{len(todo)} fetched', a.log)
                sys.exit(3)
            for status, n in pool.map(fetch, todo[i:i+a.batch]):
                done += 1
                got += n if status == 200 else 0
                missing += status == 404
            el = time.time()-t0
            say(f'{a.set}: {done}/{len(todo)} chunks, {got/2**20:.0f} MiB, {el:.0f} s, '
                f'{got/2**20/max(el, 1e-9):.2f} MiB/s, C: {free:.1f} GiB free', a.log)
            if a.status:
                try:
                    st = json.loads(Path(a.status).read_text(encoding='utf-8-sig'))
                except Exception:  # noqa: BLE001
                    st = {}
                st['prefetch'] = dict(set=a.set, fetched=done, to_fetch=len(todo), chunks=len(keys),
                                      mib=round(got/2**20, 1), mib_per_s=round(got/2**20/max(el, 1e-9), 3),
                                      updated=time.strftime('%Y-%m-%d %H:%M:%S'))
                tmp = Path(a.status).with_suffix('.tmp')
                tmp.write_text(json.dumps(st, indent=1))
                os.replace(tmp, a.status)
    left = [k for k in keys if not (root/sep.join(map(str, k))).exists()]
    say(f'{a.set}: done: {done} fetched ({missing} missing objects -> fill), {got/2**20:.0f} MiB in '
        f'{time.time()-t0:.0f} s; {len(left)} of {len(keys)} chunks not cached', a.log)
    if left:
        sys.exit(4)


if __name__ == '__main__':
    main()
