"""Write FREEZE.txt: sha256 of the pre-registration, the scorer, its tests, the plans, AFVs, run plans, the runner
and the data manifests, with the time and the state. Run after every frozen file is final. ``--verify`` checks
every file against FREEZE.txt instead (exit 1 on any difference; the v3 runner does this before it starts).

    python tools/write_freeze.py --tests "<pytest summary line>" [--note TEXT ...]
    python tools/write_freeze.py --verify
"""
import argparse
import hashlib
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'PREREG.md', 'score_strips.py', 'tests/test_score_strips.py', 'tests/test_piece_afv.py',
    'tools/make_piece_afv.py', 'tools/full_run_map.py', 'tools/download_inputs.py', 'tools/write_freeze.py',
    'tools/archive_v2.py', '../tools/fibers_json_to_afv.py',
    'runs/v3/run_all.ps1', 'runs/v3/prefetch_v3.py', 'runs/v3/infer_cpu.py',
    'runs/plan_v3/run_match.sh', 'runs/plan_v3/run_afv.sh',
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
    'data/lasagna_normals_0139_primary.npz', 'data/lasagna_normals_0139_secondary_run40.npz',
    'data/segments_manifest.json', 'data/fibers0139_manifest.json', 'data/fibers0332_manifest.json',
    'data/umbilicus/PHerc0139_20260102150214_umbilicus.json',
    '../data/umbilicus/PHerc0139_20260102150214_umbilicus_L2.json',
    '../data/umbilicus/PHerc0332_20251211183505_umbilicus_L2.json',
    '../results/g0_errata.md', '../patches/applied_pr2163_on_6f2024569_with_tests.diff',
    'superseded_v2/PREREG_v2.md', 'superseded_v2/FREEZE_v2.txt', 'superseded_v2/score_strips_v2.py',
    'superseded_v2/ARCHIVE.txt',
    'superseded_v1/PREREG_v1.md', 'superseded_v1/FREEZE_v1.txt', 'superseded_v1/score_strips_v1.py',
    'superseded_v1/g1_v1_result.json',
]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


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
        print(f'FREEZE verify: {n} files, {len(bad)} differ: {bad}')
        raise SystemExit(1 if bad or not n else 0)
    if not a.tests:
        raise SystemExit('--tests is required')
    lines = ['Freeze of pre-registration v3 for fiber_strip_small_v1 off Paris 4 (C:/VesuviusTemp/strips/score)',
             f"frozen_at_local: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
             f"frozen_at_utc:   {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC",
             'state: no PHerc0139 or PHerc0332 strip has been scored or inspected; the one 0139 smoke-test strip of the '
             'setup run (runs/smoke0139, fiber dj_20260902T151037604_000001) has no matched stretch (PREREG 10.3)',
             f'tests: {a.tests}', 'notes:'] + [f'  - {n}' for n in a.note] + ['sha256:']
    missing = []
    for rel in FILES:
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
