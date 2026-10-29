"""Verify the pre-registration freeze from the repo root.

    python verify_freeze.py                      # check every frozen file shipped in this repo
    python verify_freeze.py --data-root DIR      # also check the data files that are not shipped, in DIR
    python verify_freeze.py --data-root DIR --strict   # fail unless every frozen file was checked

What it checks:
  1. score/FREEZE.txt itself has the sha256 recorded below (the v3.2 freeze, 2026-10-29 02:10:33 UTC).
  2. Every "<sha256> *<path>" line of score/FREEZE.txt (paths relative to score/). Since v3.1 these include the
     code that makes the strips: the pinned villa worktree's changed files (villa_6f2/ at the repo root), the
     patches/ and the checkpoint files under hf_home/ (fetch_inputs.py checkpoint); since v3.2 also the relaunch
     wrapper and the prefetch-only pass (score/runs/).
  3. Every line of score/superseded_v2/ARCHIVE.txt (the archived v2 files, checked against their v2 sha256).
  4. The v3 freeze: score/superseded_v3/FREEZE_v3.txt has its recorded sha256, and each of its lines is checked
     against the archived copy (superseded_v3/, superseded_v31/) of a file a later version changed, or else against
     the file itself.
  5. The v3.1 freeze, the same way: score/superseded_v31/FREEZE_v31.txt and the archived copies in superseded_v31/.

Some frozen files are data or code that this repo does not redistribute (AFVs built from the human fiber JSONs, maps
and normals derived from them or from open data, umbilicus files, villa's files, the model checkpoint). They are
listed in NOT_SHIPPED with the reason and the way to get or rebuild them. Without --data-root they are reported as
"not shipped" and do not fail the check (a file that is present, e.g. villa_6f2/ after README step 3, is always
checked). With --data-root DIR, a missing file is looked up at DIR/score/<path> (DIR has this repo's layout, e.g.
the author's workspace C:/VesuviusTemp/strips, or this repo after fetch_inputs.py and the rebuild steps of the
README).

Exit code 0 when no checked file differs and no shipped file is missing; 1 otherwise. Standard library only.
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCORE = REPO/'score'
FREEZE_SHA256 = 'd2bf1f7b7612bee6b22925d3de123a65ec792642bfa0fd8980ac8dfda51454be'      # v3.2, 2026-10-29 02:10:33
V31_FREEZE_SHA256 = '7c8be79970c1a98d3b097c0988ea0b53d7a5354379e78f7f5e11abc851dbd559'  # v3.1, 2026-10-29 00:34:06
V3_FREEZE_SHA256 = '5309debbbeeb6ed7401badbbd163809b9c4632838c71f7eded8945d0783db8d5'   # v3, 2026-10-28 14:04:24
# v3 files that v3.1 changed, and their archived copies (score/superseded_v3/ARCHIVE.txt)
V3_ARCHIVED = {'PREREG.md': 'superseded_v3/PREREG_v3.md', 'tools/write_freeze.py': 'superseded_v3/write_freeze_v3.py'}
# v3.1 files that v3.2 changed, and their archived copies (score/superseded_v31/ARCHIVE.txt)
V31_ARCHIVED = {'PREREG.md': 'superseded_v31/PREREG_v31.md',
                'tools/write_freeze.py': 'superseded_v31/write_freeze_v31.py'}

_AFV = ('AFV (SQLite) built from the raw human fiber JSONs, which are not redistributed; '
        'rebuild: tools/fibers_json_to_afv.py, then score/runs/plan_v3/run_afv.sh')
_TSV = 'piece table written next to the piece AFV (make_piece_afv.py --map); rebuilt with it by run_afv.sh'
_MAP = 'full-run arclength map derived from the full AFV; rebuild: score/tools/full_run_map.py'
_LAS = 'Lasagna normals sampled from the open-data Lasagna predictions; rebuild: score/score_strips.py lasagna'
_UMB = 'open-data umbilicus (manifests/umbilicus.json); fetch_inputs.py umbilicus'
_UMB2 = 'level-2 copy of the open-data umbilicus; rebuild: tools/umbilicus_l0_to_l2.py'
_V2 = ('v2 piece AFV or piece table (superseded; built from the raw human fiber JSONs with '
       'superseded_v2/tools/make_piece_afv.py and the v2 plans)')
_VILLA = ('a file of the pinned villa worktree (6f2024569 + #2163 + the normalization patch), not redistributed; '
          'README, How to reproduce, step 3 (villa_6f2/ at the repo root)')
_CKPT = ('a file of the strip model checkpoint (Hugging Face scrollprize/fiber_strip_small_v1 at 7bd163a2), not '
         'redistributed; fetch_inputs.py checkpoint (hf_home/ at the repo root)')
_VILLA_FILES = ['fibers/README.md', 'fibers/sheet_strip/infer.py', 'fibers/tracing/service.py',
                'shared/crop_sampling/gpu.py', 'shared/crop_sampling/prefetch.py', 'shared/crop_sampling/volume.py',
                'shared/io.py', 'fibers/tests/test_chunk_lock.py', 'fibers/tests/test_service_transport.py']
_SNAPSHOT = ('../hf_home/hub/models--scrollprize--fiber_strip_small_v1/snapshots/'
             '7bd163a22c839ff312879ae1916195d0c9729c0e/')

# Paths as written in FREEZE.txt / ARCHIVE.txt (relative to score/).
NOT_SHIPPED = {
    '../data/human_afv/PHerc0139_human_controlled.afv': _AFV,
    '../data/human_afv/PHerc0332_human_controlled.afv': _AFV,
    'data/full_run_map_0139.npz': _MAP,
    'data/full_run_map_0332.npz': _MAP,
    'data/matched_0139.afv': _AFV,
    'data/matched_0139.tsv': _TSV,
    'data/matched_0139_secondary_run40.afv': _AFV,
    'data/matched_0139_secondary_run40.tsv': _TSV,
    'data/crossings_0332.afv': _AFV,
    'data/crossings_0332.tsv': _TSV,
    'data/lasagna_normals_0139_primary.npz': _LAS,
    'data/lasagna_normals_0139_secondary_run40.npz': _LAS,
    'data/umbilicus/PHerc0139_20260102150214_umbilicus.json': _UMB,
    '../data/umbilicus/PHerc0139_20260102150214_umbilicus_L2.json': _UMB2,
    '../data/umbilicus/PHerc0332_20251211183505_umbilicus_L2.json': _UMB2,
    # the v2 archive keeps its own copies of the same kinds of files (v2 plans, v2 tools)
    'superseded_v2/data/matched_0139.afv': _V2,
    'superseded_v2/data/matched_0139.tsv': _V2,
    'superseded_v2/data/matched_0139_secondary_run40.afv': _V2,
    'superseded_v2/data/matched_0139_secondary_run40.tsv': _V2,
    'superseded_v2/data/crossings_0332.afv': _V2,
    'superseded_v2/data/crossings_0332.tsv': _V2,
    'superseded_v2/data/lasagna_normals_0139_primary.npz': _LAS + ' (same bytes as the v3 file)',
    'superseded_v2/data/lasagna_normals_0139_secondary_run40.npz': _LAS + ' (same bytes as the v3 file)',
    'superseded_v2/data/umbilicus/PHerc0139_20260102150214_umbilicus.json': _UMB + ' (same bytes as the v3 file)',
    # v3.1 pins of the code that makes the strips
    **{f'../villa_6f2/vesuvius/src/vesuvius/neural_tracing/tracers/{p}': _VILLA for p in _VILLA_FILES},
    **{_SNAPSHOT+p: _CKPT for p in ('last.pt', 'heading_model.pt', 'config.json', 'run.json')},
}

FREEZE_LINE = re.compile(r'^\s*([0-9a-f]{64}) \*(.+?)\s*$')
ARCHIVE_LINE = re.compile(r'^([0-9a-f]{64})\s+(\S+)\s+(?:->\s+(\S+)|\(kept in place\))\s*$')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check(entries, data_root, label):
    """entries: [(sha, path relative to score/)]. Returns counts and the problem lines."""
    ok, ok_data, not_shipped, differ, missing = 0, 0, [], [], []
    for want, rel in entries:
        here = (SCORE/rel).resolve()
        if here.exists():
            if sha256(here) == want:
                ok += 1
            else:
                differ.append(f'{rel} (in repo)')
            continue
        if rel not in NOT_SHIPPED:
            missing.append(rel)
            continue
        there = (Path(data_root)/'score'/rel).resolve() if data_root else None
        if there is not None and there.exists():
            if sha256(there) == want:
                ok_data += 1
            else:
                differ.append(f'{rel} (in {data_root})')
        else:
            not_shipped.append(rel)
    print(f'{label}: {len(entries)} entries; {ok} verified in the repo, {ok_data} verified in --data-root, '
          f'{len(not_shipped)} not shipped and not checked, {len(differ)} differ, {len(missing)} missing')
    for rel in differ:
        print(f'  DIFFERS  {rel}')
    for rel in missing:
        print(f'  MISSING  {rel} (should be in this repo)')
    return not_shipped, differ, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data-root', help='a directory with this repo layout that holds the data not shipped here')
    ap.add_argument('--strict', action='store_true', help='also fail when a frozen file could not be checked')
    ap.add_argument('--quiet', action='store_true', help='do not list the files that were not checked')
    a = ap.parse_args()

    freeze = SCORE/'FREEZE.txt'
    got = sha256(freeze)
    print(f'score/FREEZE.txt sha256 {got}: ' + ('as recorded' if got == FREEZE_SHA256 else f'DIFFERS from {FREEZE_SHA256}'))
    bad = got != FREEZE_SHA256

    text = freeze.read_text(encoding='utf-8')
    entries = [(m.group(1), m.group(2)) for m in map(FREEZE_LINE.match, text.splitlines()) if m]
    first = next((ln for ln in text.splitlines() if ln.startswith('frozen_at_utc')), '').split(':', 1)[-1].strip()
    ns1, d1, m1 = check(entries, a.data_root, f'FREEZE.txt (v3.2, frozen {first})')

    archive = []
    for line in (SCORE/'superseded_v2'/'ARCHIVE.txt').read_text(encoding='utf-8').splitlines():
        m = ARCHIVE_LINE.match(line.strip())
        if m:
            archive.append((m.group(1), m.group(3) or m.group(2)))
    ns2, d2, m2 = check(archive, a.data_root, 'superseded_v2/ARCHIVE.txt (v2)')

    v3 = SCORE/'superseded_v3'/'FREEZE_v3.txt'
    got3 = sha256(v3)
    print(f'score/superseded_v3/FREEZE_v3.txt sha256 {got3}: '
          + ('as recorded' if got3 == V3_FREEZE_SHA256 else f'DIFFERS from {V3_FREEZE_SHA256}'))
    bad = bad or got3 != V3_FREEZE_SHA256
    v3_entries = [(m.group(1), V3_ARCHIVED.get(m.group(2), m.group(2)))
                  for m in map(FREEZE_LINE.match, v3.read_text(encoding='utf-8').splitlines()) if m]
    ns3, d3, m3 = check(v3_entries, a.data_root, 'superseded_v3/FREEZE_v3.txt (v3; changed files as archived)')

    v31 = SCORE/'superseded_v31'/'FREEZE_v31.txt'
    got31 = sha256(v31)
    print(f'score/superseded_v31/FREEZE_v31.txt sha256 {got31}: '
          + ('as recorded' if got31 == V31_FREEZE_SHA256 else f'DIFFERS from {V31_FREEZE_SHA256}'))
    bad = bad or got31 != V31_FREEZE_SHA256
    v31_entries = [(m.group(1), V31_ARCHIVED.get(m.group(2), m.group(2)))
                   for m in map(FREEZE_LINE.match, v31.read_text(encoding='utf-8').splitlines()) if m]
    ns4, d4, m4 = check(v31_entries, a.data_root, 'superseded_v31/FREEZE_v31.txt (v3.1; changed files as archived)')

    not_checked = sorted(set(ns1) | set(ns2) | set(ns3) | set(ns4))
    if not_checked and not a.quiet:
        print('Not shipped here and not checked (pass --data-root to check them):')
        for rel in not_checked:
            print(f'  {rel}\n      {NOT_SHIPPED[rel]}')
    bad = bad or bool(d1 or m1 or d2 or m2 or d3 or m3 or d4 or m4) or (a.strict and bool(not_checked))
    print('FAIL' if bad else 'PASS')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
