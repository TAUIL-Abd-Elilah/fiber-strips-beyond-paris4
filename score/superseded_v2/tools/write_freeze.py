"""Write FREEZE.txt: sha256 of the pre-registration, the scorer, its tests, the plans, AFVs, run plans and data
manifests, with the time and the state. Run after every frozen file is final.

    python tools/write_freeze.py --tests "<pytest summary line>" [--note TEXT ...]
"""
import argparse
import hashlib
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'PREREG.md', 'score_strips.py', 'tests/test_score_strips.py', 'tests/test_piece_afv.py',
    'tools/make_piece_afv.py', 'tools/download_inputs.py', 'tools/write_freeze.py',
    '../tools/fibers_json_to_afv.py',
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
    ap.add_argument('--tests', required=True)
    ap.add_argument('--note', action='append', default=[])
    a = ap.parse_args()
    lines = ['Freeze of pre-registration v2 for fiber_strip_small_v1 off Paris 4 (C:/VesuviusTemp/strips/score)',
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
