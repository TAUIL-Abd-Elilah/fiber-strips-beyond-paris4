"""Download the inputs this repo does not redistribute, check each file against its manifest sha256, and put it
where the frozen scripts expect it (the layout of the author's workspace, with this repo as its root).

    python fetch_inputs.py --list                      # what each set holds; no network
    python fetch_inputs.py SET [SET ...] [--threads 4]

Sets (manifest -> destination):
  fibers0139  score/data/fibers0139_manifest.json   -> score/data/fibers0139/<file>       620 JSON, 224 MiB
  fibers0332  score/data/fibers0332_manifest.json   -> score/data/fibers0332/<file>       255 JSON,  84 MiB
  segments    score/data/segments_manifest.json     -> score/data/segments/<segment>/<file>  152 files, 893 MiB
  samples     the 10 fibers tools/test_fibers_json_to_afv.py reads -> data/samples/<scroll>/<file>
  umbilicus   manifests/umbilicus.json -> score/data/umbilicus/, data/umbilicus/ (+ the level-2 copies)
  checkpoint  manifests/checkpoint_fiber_strip_small_v1.json -> hf_home/hub/... (needs huggingface_hub)

The human fiber JSONs (dl.ash2txt.org/other/annotators/0139_fibers/ and 0332_fibers/) carry no license file, so
they are linked, not copied. Those folders change over time (0332 was still being edited on 26 Oct 2026): a file
whose sha256 differs from the manifest is kept as <name>.mismatch and reported, and the frozen results only hold
for the manifest's bytes. No CT is downloaded here; villa's remote-CT cache fetches it during inference
(score/runs/v3/prefetch_v3.py prefetches the chunks of the run sets).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parent
SAMPLES = {'0139': ['dj_20260902T151037604_000001.json', 'kb_20260916T031524573_000138.json',
                    'lt_20260831T100021884_000036.json', 'lt_20260928T114319112_000202.json',
                    'lt_20261008T100739895_000358.json'],
           '0332': ['dj_20260810T113807458_000001.json', 'dj_20261026T193623205_000068.json',
                    'kb_20260824T194228123_000077.json', 'lt_20261014T070715152_000035.json',
                    'lt_20261020T112237109_000104.json']}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(rel):
    return json.loads((REPO/rel).read_text(encoding='utf-8'))


def jobs_for(name):
    """[(url, destination, bytes, sha256)] of one set."""
    if name in ('fibers0139', 'fibers0332'):
        m = load(f'score/data/{name}_manifest.json')
        return [(f['source'], REPO/'score'/'data'/name/f['fiber'], f['bytes'], f['sha256']) for f in m['files']]
    if name == 'segments':
        m = load('score/data/segments_manifest.json')
        return [(f['source'], REPO/'score'/'data'/'segments'/f['segment']/f['file'], f['bytes'], f['sha256'])
                for f in m['files']]
    if name == 'samples':
        out = []
        for scroll, names in SAMPLES.items():
            files = {f['fiber']: f for f in load(f'score/data/fibers{scroll}_manifest.json')['files']}
            out += [(files[n]['source'], REPO/'data'/'samples'/scroll/n, files[n]['bytes'], files[n]['sha256'])
                    for n in names]
        return out
    if name == 'umbilicus':
        return [(f['url'], REPO/f['copies'][0], f['bytes'], f['sha256']) for f in load('manifests/umbilicus.json')['files']]
    raise KeyError(name)


def fetch(url, dest, size, want, tries=4):
    if dest.exists() and dest.stat().st_size == size and sha256(dest) == want:
        return 'present'
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name+'.part')
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(part, 'wb') as fh:
                shutil.copyfileobj(r, fh, 1 << 20)
            break
        except Exception:
            if attempt == tries-1:
                raise
            time.sleep(2+3*attempt)
    if sha256(part) != want:
        part.replace(dest.with_name(dest.name+'.mismatch'))
        return 'MISMATCH (kept as .mismatch)'
    part.replace(dest)
    return 'fetched'


def run_set(name, threads):
    jobs = jobs_for(name)
    status = {}
    with ThreadPoolExecutor(threads) as pool:
        futs = {pool.submit(fetch, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            url, dest = futs[f][:2]
            try:
                status[dest] = f.result()
            except Exception as error:
                status[dest] = f'FAILED: {error}'
            if i % 50 == 0 or i == len(jobs):
                print(f'  {name}: {i}/{len(jobs)}', flush=True)
    bad = {d: s for d, s in status.items() if s not in ('present', 'fetched')}
    for d, s in sorted(bad.items()):
        print(f'  {d.relative_to(REPO)}: {s}')
    return bad


def umbilicus_extras():
    """The second copy and the level-2 files (tools/umbilicus_l0_to_l2.py), matched to the frozen bytes."""
    bad = {}
    for f in load('manifests/umbilicus.json')['files']:
        src = REPO/f['copies'][0]
        if not src.exists():
            bad[src] = 'missing'
            continue
        for c in f['copies'][1:]:
            (REPO/c).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, REPO/c)
        l2 = f['level2']
        dst = REPO/l2['path']
        subprocess.run([sys.executable, '-I', str(REPO/'tools'/'umbilicus_l0_to_l2.py'), str(src), str(dst)], check=True)
        if sha256(dst) != l2['sha256']:
            # the frozen copies were written on Windows (text mode: CRLF); match those bytes on other systems
            crlf = dst.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
            if hashlib.sha256(crlf).hexdigest() == l2['sha256']:
                dst.write_bytes(crlf)
            else:
                bad[dst] = 'level-2 file differs from the frozen one'
    return bad


def checkpoint():
    from huggingface_hub import hf_hub_download
    m = load('manifests/checkpoint_fiber_strip_small_v1.json')
    bad = {}
    for f in m['files']:
        p = Path(hf_hub_download(m['repo_id'], f['file'], revision=m['revision'], cache_dir=str(REPO/'hf_home'/'hub')))
        if sha256(p) != f['sha256']:
            bad[p] = 'sha256 differs from the manifest'
    return bad


def main():
    sets = ['fibers0139', 'fibers0332', 'segments', 'samples', 'umbilicus', 'checkpoint']
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('sets', nargs='*', metavar='SET', help=' '.join(sets))
    ap.add_argument('--list', action='store_true', help='show what each set holds; no network')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    unknown = [s for s in a.sets if s not in sets]
    if unknown:
        ap.error(f'unknown set(s) {unknown}; choose from {sets}')
    if a.list or not a.sets:
        for name in sets:
            if name == 'checkpoint':
                m = load('manifests/checkpoint_fiber_strip_small_v1.json')
                n, size = len(m['files']), sum(f['bytes'] for f in m['files'])
            else:
                j = jobs_for(name)
                n, size = len(j), sum(x[2] for x in j)
            print(f'{name:11s} {n:4d} files {size/2**20:9.1f} MiB')
        return 0
    bad = {}
    for name in a.sets:
        print(f'{name} ...', flush=True)
        if name == 'checkpoint':
            bad.update(checkpoint())
            continue
        bad.update(run_set(name, a.threads))
        if name == 'umbilicus':
            bad.update(umbilicus_extras())
    print('done, all files match their manifests' if not bad else f'done, {len(bad)} files need attention')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
