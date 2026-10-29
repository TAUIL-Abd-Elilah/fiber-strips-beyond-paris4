"""Download the PHerc0139 scoring inputs (CPU/network only; no CT):

  1. every wrap-segment tifxyz meshed on volume 20260102150214 (2.399 um): S3 PHerc0139/segments/*/mesh/
     *-on-20260102150214-2.399um.tifxyz/{meta.json,x.tif,y.tif,z.tif}, listed afresh from the bucket
     -> data/segments/<segment dir>/
  2. every human fiber JSON of dl.ash2txt.org/other/annotators/0139_fibers/ (the URL list data/fiber_urls_0139.txt)
     -> data/fibers0139/

Each file is fetched to a .part name and renamed when its size equals the listed / Content-Length size; a file
already present with that size is skipped. A manifest (name, bytes, sha256, source URL) is written for each set.

    python download_inputs.py --root C:/VesuviusTemp/strips/score [--threads 8]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

S3 = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
MESH_RE = re.compile(r'^PHerc0139/segments/([^/]+)/mesh/([^/]+-on-20260102150214-2\.399um\.tifxyz)/([^/]+)$')


def s3_list(prefix, delimiter=None):
    """(key, size) and common prefixes under ``prefix`` (paginated; retried on connection resets)."""
    token, out, prefixes = None, [], []
    while True:
        q = {'list-type': '2', 'prefix': prefix, 'max-keys': '1000'}
        if delimiter:
            q['delimiter'] = delimiter
        if token:
            q['continuation-token'] = token
        for attempt in range(6):
            try:
                xml = urllib.request.urlopen(S3+'?'+urllib.parse.urlencode(q), timeout=120).read().decode()
                break
            except Exception:  # connection resets seen on 28 Oct: retried
                if attempt == 5:
                    raise
                time.sleep(3+5*attempt)
        prefixes += re.findall(r'<CommonPrefixes><Prefix>([^<]*)</Prefix>', xml)
        out += [(k, int(s)) for k, s in re.findall(r'<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>', xml, re.S)]
        m = re.search(r'<NextContinuationToken>([^<]*)</NextContinuationToken>', xml)
        if not m:
            return out, prefixes
        token = m.group(1)


def fetch(url, dest, size=None, tries=4):
    dest = Path(dest)
    if dest.exists() and size is not None and dest.stat().st_size == size:
        return dest, 'present'
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name+'.part')
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                expect = int(r.headers.get('Content-Length', -1))
                with open(part, 'wb') as fh:
                    while True:
                        block = r.read(1 << 20)
                        if not block:
                            break
                        fh.write(block)
            got = part.stat().st_size
            if (size is not None and got != size) or (expect >= 0 and got != expect):
                raise IOError(f'size {got} != listed {size} / content-length {expect}')
            part.replace(dest)
            return dest, 'fetched'
        except Exception:  # retried, then raised
            if attempt == tries-1:
                raise
            time.sleep(2+3*attempt)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    root = Path(a.root)
    t0 = time.time()
    _, seg_prefixes = s3_list('PHerc0139/segments/', '/')
    seg_dirs = sorted(p.rstrip('/').split('/')[-1] for p in seg_prefixes)
    with ThreadPoolExecutor(8) as pool:  # each segment's mesh/ only (the segment dirs hold many other keys)
        keys = [kv for rows, _ in pool.map(lambda d: s3_list(f'PHerc0139/segments/{d}/mesh/'), seg_dirs) for kv in rows]
    mesh = [(k, s) for k, s in keys if MESH_RE.match(k)]
    jobs = []
    for k, s in mesh:
        seg, tifxyz, name = MESH_RE.match(k).groups()
        jobs.append((S3+k, root/'data'/'segments'/seg/name, s, dict(segment=seg, tifxyz=tifxyz, file=name)))
    urls = [u.strip() for u in (root/'data'/'fiber_urls_0139.txt').read_text().splitlines() if u.strip()]
    for u in urls:
        jobs.append((u, root/'data'/'fibers0139'/u.rsplit('/', 1)[1], None, dict(fiber=u.rsplit('/', 1)[1])))
    print(f'{len(seg_dirs)} segment dirs, {len(mesh)} mesh files on 20260102150214 '
          f'({sum(s for _, s in mesh)/2**20:.1f} MiB), {len(urls)} fiber JSONs', flush=True)
    done, status = 0, {}
    with ThreadPoolExecutor(a.threads) as pool:
        futs = {pool.submit(fetch, u, d, s): (u, d, s, m) for u, d, s, m in jobs}
        for f in as_completed(futs):
            u, d, s, m = futs[f]
            try:
                _, st = f.result()
            except Exception as error:
                st = f'FAILED: {error}'
            status[str(d)] = st
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f'{done}/{len(jobs)} files, {time.time()-t0:.0f} s', flush=True)
    seg_manifest, fib_manifest = [], []
    for u, d, s, m in jobs:
        rec = dict(m, source=u, bytes=d.stat().st_size if d.exists() else None, listed_bytes=s,
                   sha256=sha256(d) if d.exists() else None, status=status.get(str(d)))
        (seg_manifest if 'segment' in m else fib_manifest).append(rec)
    with_mesh = sorted({r['segment'] for r in seg_manifest})
    summary = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), segment_dirs=len(seg_dirs),
                   segment_dirs_without_2399_mesh=[d for d in seg_dirs if d not in with_mesh],
                   segments_with_2399_mesh=len(with_mesh), segment_files=len(seg_manifest),
                   segment_bytes=sum(r['bytes'] or 0 for r in seg_manifest),
                   fiber_files=len(fib_manifest), fiber_bytes=sum(r['bytes'] or 0 for r in fib_manifest),
                   failed=[k for k, v in status.items() if v.startswith('FAILED')], seconds=round(time.time()-t0))
    (root/'data'/'segments_manifest.json').write_text(json.dumps(dict(summary=summary, files=seg_manifest), indent=1))
    (root/'data'/'fibers0139_manifest.json').write_text(json.dumps(dict(summary=summary, files=fib_manifest), indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
