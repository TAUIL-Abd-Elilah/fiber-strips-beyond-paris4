"""Write FREEZE.txt: sha256 of the pre-registration, the scorer, its tests, the plans, AFVs, run plans, the runner
and the data manifests, with the time and the state. Run after every frozen file is final. ``--verify`` checks
every file against FREEZE.txt instead (exit 1 on any difference; the v3 runner does this before it starts).

v3.1 (erratum, PREREG 10.7) also pins the code that makes the strips: every file the villa_6f2 worktree changes
against its base commit (``git diff --name-only`` plus the two untracked #2163 test files), the setup run's
patches/*.diff and APPLIED.txt, and the checkpoint files of the Hugging Face snapshot (last.pt, heading_model.pt,
config.json, run.json at revision 7bd163a2). ``--verify`` also checks that the worktree is at its base commit with
exactly the pinned files changed (``git status``).

    python tools/write_freeze.py --tests "<pytest summary line>" [--note TEXT ...]
    python tools/write_freeze.py --verify
"""
import argparse
import hashlib
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'v3.1'
FILES = [
    'PREREG.md', 'score_strips.py', 'tests/test_score_strips.py', 'tests/test_piece_afv.py',
    'tests/test_erratum_v31.py',
    'tools/make_piece_afv.py', 'tools/full_run_map.py', 'tools/download_inputs.py', 'tools/write_freeze.py',
    'tools/archive_v2.py', 'tools/paris4_piece_check.py', 'tools/strata_reading_check.py',
    'tools/erratum_geometry_checks.py',
    '../tools/fibers_json_to_afv.py',
    'runs/v3/run_all.ps1', 'runs/v3/prefetch_v3.py', 'runs/v3/infer_cpu.py',
    'runs/plan_v3/run_match.sh', 'runs/plan_v3/run_afv.sh',
    'runs/erratum_v31/infer_cpu4.py', 'runs/erratum_v31/paris4/plan.json',
    '../data/human_afv/PHerc0139_human_controlled.afv', '../data/human_afv/PHerc0332_human_controlled.afv',
    'data/full_run_map_0139.npz', 'data/full_run_map_0332.npz',
    'results/matched_stretches.json', 'results/matched_stretches_secondary_run40.json',
    'results/crossings_0332_plan.json',
    'data/matched_0139.afv', 'data/matched_0139.tsv', 'results/runplan_0139_primary.tsv',
    'results/matched_0139_afv_report.json',
    'data/matched_0139_secondary_run40.afv', 'data/matched_0139_secondary_run40.tsv',
    'results/runplan_0139_secondary_run40.tsv', 'results/matched_0139_secondary_run40_afv_report.json',
    'data/crossings_0332.afv', 'data/crossings_0332.tsv', 'results/runplan_0332_crossings.tsv',
    'results/crossings_0332_afv_report.json',
    'results/erratum_v31/strata_reading_check.json', 'results/erratum_v31/paris4_piece_check.json',
    'results/erratum_v31/geometry_checks.json',
    'data/lasagna_normals_0139_primary.npz', 'data/lasagna_normals_0139_secondary_run40.npz',
    'data/segments_manifest.json', 'data/fibers0139_manifest.json', 'data/fibers0332_manifest.json',
    'data/umbilicus/PHerc0139_20260102150214_umbilicus.json',
    '../data/umbilicus/PHerc0139_20260102150214_umbilicus_L2.json',
    '../data/umbilicus/PHerc0332_20251211183505_umbilicus_L2.json',
    '../results/g0_errata.md',
    'superseded_v3/PREREG_v3.md', 'superseded_v3/FREEZE_v3.txt', 'superseded_v3/write_freeze_v3.py',
    'superseded_v3/ARCHIVE.txt',
    'superseded_v2/PREREG_v2.md', 'superseded_v2/FREEZE_v2.txt', 'superseded_v2/score_strips_v2.py',
    'superseded_v2/ARCHIVE.txt',
    'superseded_v1/PREREG_v1.md', 'superseded_v1/FREEZE_v1.txt', 'superseded_v1/score_strips_v1.py',
    'superseded_v1/g1_v1_result.json',
]
FIXTURES = 'tests/fixtures/paris4_real'  # every file under it (the real Paris 4 infer.py outputs of 10.7, check 5)
# the code that makes the strips (v3.1)
VILLA = ROOT.parent/'villa_6f2'
VILLA_BASE = '6f2024569404a46b0f008eee292773945c07a684'
VILLA_CHANGED = [  # git diff --name-only, then the two untracked #2163 test files
    'vesuvius/src/vesuvius/neural_tracing/tracers/fibers/README.md',
    'vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/infer.py',
    'vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tracing/service.py',
    'vesuvius/src/vesuvius/neural_tracing/tracers/shared/crop_sampling/gpu.py',
    'vesuvius/src/vesuvius/neural_tracing/tracers/shared/crop_sampling/prefetch.py',
    'vesuvius/src/vesuvius/neural_tracing/tracers/shared/crop_sampling/volume.py',
    'vesuvius/src/vesuvius/neural_tracing/tracers/shared/io.py',
    'vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tests/test_chunk_lock.py',
    'vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tests/test_service_transport.py',
]
PATCHES = ['APPLIED.txt', 'applied_pr2163_on_6f2024569.diff', 'applied_pr2163_on_6f2024569_with_tests.diff',
           'pr2163_full.diff', 'strip_bind_normalization_other_volume.diff', 'worktree_full_after_g0.diff']
SNAPSHOT = '../hf_home/hub/models--scrollprize--fiber_strip_small_v1/snapshots/7bd163a22c839ff312879ae1916195d0c9729c0e'
CHECKPOINT_FILES = ['last.pt', 'heading_model.pt', 'config.json', 'run.json']
PINS = ([f'../villa_6f2/{p}' for p in VILLA_CHANGED] + [f'../patches/{p}' for p in PATCHES]
        + [f'{SNAPSHOT}/{p}' for p in CHECKPOINT_FILES])


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def all_files():
    fixtures = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/FIXTURES).rglob('*') if p.is_file())
    return FILES+fixtures+PINS


def worktree_state():
    """(HEAD, sorted changed paths) of the villa_6f2 worktree; changed = modified or untracked (git status)."""
    head = subprocess.run(['git', '-C', str(VILLA), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()
    status = subprocess.run(['git', '-C', str(VILLA), 'status', '--porcelain', '--untracked-files=all'],
                            capture_output=True, text=True, check=True).stdout.splitlines()
    return head, sorted(line[3:].strip().strip('"') for line in status if line.strip())


def check_worktree():
    try:
        head, changed = worktree_state()
    except Exception as error:  # noqa: BLE001
        return [f'villa_6f2 worktree: git failed ({error})']
    bad = []
    if head != VILLA_BASE:
        bad.append(f'villa_6f2 HEAD {head} != {VILLA_BASE}')
    if changed != sorted(VILLA_CHANGED):
        bad.append(f'villa_6f2 changed files {changed} != the pinned {sorted(VILLA_CHANGED)}')
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tests')
    ap.add_argument('--note', action='append', default=[])
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    if a.verify:
        bad, n = [], 0
        for line in (ROOT/'FREEZE.txt').read_text().splitlines():
            line = line.strip()
            if len(line) > 66 and line[64:66] == ' *':
                n += 1
                p = (ROOT/line[66:]).resolve()
                if not p.exists() or sha(p) != line[:64]:
                    bad.append(line[66:])
        wt = check_worktree()
        print(f'FREEZE verify: {n} files, {len(bad)} differ: {bad}; villa_6f2 worktree: '
              f'{"ok (HEAD " + VILLA_BASE[:9] + ", " + str(len(VILLA_CHANGED)) + " pinned changed files)" if not wt else wt}')
        raise SystemExit(1 if bad or wt or not n else 0)
    if not a.tests:
        raise SystemExit('--tests is required')
    wt = check_worktree()
    if wt:
        raise SystemExit(f'cannot freeze: {wt}')
    lines = [f'Freeze of pre-registration {VERSION} for fiber_strip_small_v1 off Paris 4 (C:/VesuviusTemp/strips/score)',
             f"frozen_at_local: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
             f"frozen_at_utc:   {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC",
             'state: no PHerc0139 or PHerc0332 strip of the run plans has been made, scored or inspected; the only '
             'such strips are smoke tests of fibers in no scored set: the setup run\'s 0139 strip (PREREG 10.3) and a '
             'separate build\'s two 0332 strips of one fiber in no plan, not opened here (PREREG 10.7 item 8); the '
             'v3.1 checks ran on Paris 4 only (PREREG 10.7)',
             f'tests: {a.tests}',
             f'villa_6f2: HEAD {VILLA_BASE}; changed files exactly the {len(VILLA_CHANGED)} pinned below (git status)',
             'notes:'] + [f'  - {n}' for n in a.note] + ['sha256:']
    missing = []
    for rel in all_files():
        p = (ROOT/rel).resolve()
        if p.exists():
            lines.append(f'  {sha(p)} *{rel}')
        else:
            missing.append(rel)
    if missing:
        raise SystemExit(f'missing: {missing}')
    (ROOT/'FREEZE.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
