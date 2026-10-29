"""PREREG v3.1 erratum, gap 1: the strata of the seeded subsample (PREREG 4.8 rule 2) read as the code reads them and
as the v3 text read literally, on the frozen secondary plan (fibers, segments and plan geometry only; no strip).

    python tools/strata_reading_check.py OUT.json

* The code (score_strips.plan_pieces): a piece's stratum segment is the segment holding the most total stretch
  length (sum of s1 - s0 of the piece's stretches on it; ties to the first segment name in sorted order).
* v3's text read literally: "the segment of the piece's longest stretch" (ties reported).

Each piece's CT chunks are recomputed with the frozen scorer (``piece_ct``). "New CT" counts chunks not cached at plan
time: the PHerc0139 level-2 chunk files in the CT cache last written before the plan was made (the cache move kept
the file times; checked: the count must equal the plan's ``ct_chunks_cached``). The frozen scan (code's strata) must
be reproduced exactly before the literal reading is evaluated with the same ``ct_cap_scan``.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SCORE = HERE.parent
sys.path.insert(0, str(SCORE))
import score_strips as S  # noqa: E402

PLAN = SCORE/'results'/'matched_stretches_secondary_run40.json'
CACHE_LEVEL = (Path('C:/VesuviusTemp/strips/ct_cache')/'s3'/'vesuvius-challenge-open-data'/'PHerc0139'/'volumes' /
               '20260102150214-2.399um-0.2m-78keV-masked.zarr'/'2')


def main(out):
    m = json.loads(PLAN.read_text())
    plan, pieces, stretches = m['plan'], m['pieces'], {st['id']: st for st in m['stretches']}
    plan_time = datetime.strptime(m['provenance']['time'], '%Y-%m-%d %H:%M:%S')
    names = {p['fiber'] for p in pieces}
    fibers, _ = S.load_fibers(SCORE/'data'/'fibers0139', space='PHerc0139/20260102150214@L0', names=names)
    by_name = {f.name: f for f in fibers}
    zarray = dict(shape=plan['zarray']['shape'], chunks=plan['zarray']['chunks'], dtype=plan['zarray']['dtype'])
    S.piece_ct(by_name, pieces, zarray)
    allk = set().union(*[p['_ct'] for p in pieces])
    sep = json.loads((CACHE_LEVEL/'.zarray').read_text()).get('dimension_separator', '.')
    cached = set()
    for k in allk:
        f = CACHE_LEVEL/sep.join(map(str, k))
        if f.exists() and datetime.fromtimestamp(f.stat().st_mtime) < plan_time:
            cached.add(k)
    checks = dict(ct_chunks_all=len(allk), ct_chunks_all_plan=plan['all_matched']['ct_chunks'],
                  cached_at_plan_time=len(cached), cached_at_plan_time_plan=plan['all_matched']['ct_chunks_cached'],
                  ct_chunks_per_piece_equal=all(p['ct_chunks'] == len(p['_ct']) for p in pieces))
    if not (checks['ct_chunks_all'] == checks['ct_chunks_all_plan'] and checks['cached_at_plan_time'] ==
            checks['cached_at_plan_time_plan'] and checks['ct_chunks_per_piece_equal']):
        raise SystemExit(f'cannot reproduce the plan inputs: {checks}')
    caps = plan['caps']
    grid = tuple(S.CT_CAP_SCAN_0139_SECONDARY)
    code_key = (lambda u: (u['segment'], u['family']))
    cap_code, scan_code = S.ct_cap_scan(pieces, code_key, zarray, caps['cpu_hours'], grid, caps['min_run_fibers'], cached)
    run_code = S.stratified_subsample(pieces, code_key, zarray, caps['cpu_hours'], cap_code, cached)
    frozen_run = [p['id'] for p in sorted((p for p in pieces if p['run']), key=lambda p: p['run_order'])]
    checks.update(code_scan_equals_frozen=scan_code == caps['ct_cap_scan'], code_cap=cap_code,
                  code_run_equals_frozen=run_code == frozen_run)
    if not (checks['code_scan_equals_frozen'] and checks['code_run_equals_frozen']):
        raise SystemExit(f'cannot reproduce the frozen subsample: {checks}')
    # the literal reading
    literal, ties, differ = {}, [], []
    for p in pieces:
        sts = [stretches[i] for i in p['windows']]
        best = max(st['s1']-st['s0'] for st in sts)
        top = sorted({st['segment'] for st in sts if st['s1']-st['s0'] == best})
        seg_len = {}
        for st in sts:
            seg_len[st['segment']] = seg_len.get(st['segment'], 0.)+st['s1']-st['s0']
        if len(top) > 1:
            ties.append(dict(piece=p['id'], longest_sv=best, segments=top))
        literal[p['id']] = top[0]
        if top[0] != p['segment'] or len(top) > 1:
            differ.append(dict(piece=p['id'], run_v3=p['run'], code_segment=p['segment'],
                               code_segment_total_sv=round(seg_len[p['segment']], 3),
                               literal_segment=top if len(top) > 1 else top[0], longest_stretch_sv=round(best, 3),
                               stretches=len(sts)))
    lit_pieces = [dict(p, segment=literal[p['id']]) for p in pieces]
    cap_lit, scan_lit = S.ct_cap_scan(lit_pieces, code_key, zarray, caps['cpu_hours'], grid, caps['min_run_fibers'],
                                      cached)
    run_lit = S.stratified_subsample(lit_pieces, code_key, zarray, caps['cpu_hours'], cap_lit, cached)
    b_lit = S.budget([p for p in pieces if p['id'] in set(run_lit)], zarray, cached)
    b_code = S.budget([p for p in pieces if p['id'] in set(run_code)], zarray, cached)
    # the literal strata at the frozen cap (9.6 GiB)
    run_lit_96 = S.stratified_subsample(lit_pieces, code_key, zarray, caps['cpu_hours'], cap_code, cached)
    b_lit_96 = S.budget([p for p in pieces if p['id'] in set(run_lit_96)], zarray, cached)

    def brief(b):
        return {k: b[k] for k in ('pieces', 'fibers', 'decisions', 'ct_chunks', 'ct_new_gib')}
    rep = dict(tool='tools/strata_reading_check.py', plan=str(PLAN.relative_to(SCORE)),
               plan_sha256=S.sha256_file(PLAN), scorer_sha256=S.sha256_file(SCORE/'score_strips.py'),
               checks=checks, pieces=len(pieces), pieces_differing=len(differ), ties=ties, differing=differ,
               code=dict(cap_gib=cap_code, run=brief(b_code), run_ids=run_code),
               literal=dict(cap_gib=cap_lit, scan=scan_lit, run=brief(b_lit), run_ids=run_lit,
                            vs_frozen=dict(only_literal=sorted(set(run_lit)-set(run_code)),
                                           only_frozen=sorted(set(run_code)-set(run_lit)),
                                           fibers_only_literal=sorted({p['fiber'] for p in pieces if p['id'] in
                                                                       set(run_lit)}-{p['fiber'] for p in pieces
                                                                                      if p['id'] in set(run_code)}),
                                           fibers_only_frozen=sorted({p['fiber'] for p in pieces if p['id'] in
                                                                      set(run_code)}-{p['fiber'] for p in pieces
                                                                                      if p['id'] in set(run_lit)}))),
               literal_at_frozen_cap=dict(cap_gib=cap_code, run=brief(b_lit_96),
                                          same_set_as_frozen=set(run_lit_96) == set(run_code)))
    Path(out).write_text(json.dumps(S.jsonable(rep), indent=1))
    print(json.dumps(S.jsonable({k: v for k, v in rep.items() if k not in ('code',)}), indent=1)[:6000])


if __name__ == '__main__':
    main(sys.argv[1])
