"""PREREG v3.1 erratum checks 5 and 6, on Paris 4 only (three G0 fibers; no PHerc0139 or PHerc0332 strip).

    python tools/paris4_piece_check.py plan OUTDIR          # piece AFV + run plan, as tools/make_piece_afv.py does
    python tools/paris4_piece_check.py run OUTDIR           # villa's infer.py, one process at a time, CPU, 4 threads
    python tools/paris4_piece_check.py compare OUTDIR REPORT.json

Check 5 (real infer.py output through the scorer). Each piece is cut from a G0 fiber (C:/VesuviusTemp/strips/data/g0/
g0_fibers.afv) the way make_piece_afv.py cuts a PHerc0139 fiber: the window's full-run heads (score_strips.FullRunMap
on the full fiber's AFVFibers geometry, ``plan_pieces``), the fiber kept from PIECE_BEFORE_SV before the first head to
PIECE_AFTER_SV after the last, polyline points copied bit for bit, uuid5 identity, and the exact ``--arc`` (the first
head placed on the piece's own repaired geometry; END = START + 16 (n - 1) + 1). villa's ``sheet_strip.infer`` then
runs on it with run_all.ps1's arguments (runs/v3/infer_cpu.py, the frozen wrapper, with THREADS = 4 instead of 8; the
Paris 4 CT store as ``--ct``, the Paris 4 umbilicus). The strip is then read with the scorer's own reading path
(``discover_strips``, ``load_strip``, ``frames_from_decisions``) and scored end to end (``score_strips``,
``aggregate``) against the G0 full-fiber strip of the same fiber used as a stand-in "segment".

Check 6 (a piece run against the full-fiber run). The same rows of the piece strip and of the full-fiber strip are
compared node for node (rows matched by the full run's head index: piece row r = full row r + 8 k0 / df; columns in
infer.py's order; nodes valid in both): against the G0 CPU strips of runs/g0 (infer_all, 16 threads; within 0.03
voxels of the published strips), and against a full-fiber infer.py run made here with the same 4 threads (control).
Only the rows of the window (the rows a PHerc0139 stretch of that extent would score) are compared, and also every
row valid in both.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid as uuidlib

import numpy as np

HERE = Path(__file__).resolve().parent
SCORE = HERE.parent
STRIPS = SCORE.parent
sys.path.insert(0, str(SCORE))
sys.path.insert(0, str(HERE))
import score_strips as S  # noqa: E402
import make_piece_afv as MP  # noqa: E402  (puts villa_6f2/vesuvius/src on sys.path)
from full_run_map import breakpoints, fiber_map  # noqa: E402
from vesuvius.neural_tracing.tracers.shared.annotations.afv import (  # noqa: E402
    AFVFibers, read_afv_polylines, write_afv)
from vesuvius.neural_tracing.tracers.shared.geometry import interp_at  # noqa: E402

G0_AFV = STRIPS/'data'/'g0'/'g0_fibers.afv'
G0_STRIPS = STRIPS/'runs'/'g0'/'strips'/'0000'
PY = r'C:\Users\PC\miniconda3\envs\vesuvius\python.exe'
WRAPPER = SCORE/'runs'/'erratum_v31'/'infer_cpu4.py'
CHECKPOINT = 'scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e'
CT_PARIS4 = 's3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr'
UMB_PARIS4 = 'C:/VesuviusTemp/strips/data/paris4/umbilicus.json'
CACHE = 'C:/VesuviusTemp/strips/ct_cache'
INFER_ARGS = ['--device', 'cpu', '--index', 'none', '--no-render', '--step', '8', '--blend', '16', '--trim', '75',
              '--batch', '32']
# (tag, G0 fiber id, window s0, s1 in raw sv): a late window (piece cut at its start), an early one (cut at its end)
# and one on the fiber with a kink repair
WINDOWS = [('A', 778459, 400., 440.), ('B', 757072, 40., 80.), ('C', 329464, 100., 140.)]
TIMEOUT_S = 1800


def g0_fiber(fid, row, xyz):
    """A G0 fiber as a score_strips.Fiber (sv; the whole polyline is its span) and its FullRunMap."""
    line = np.asarray(xyz, np.float64)/S.SCORE_UNIT_BASE
    pts, s = S.resample(line, S.SAMPLE_STEP)
    parent_uuid = str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, f'PHercParis4/20260411134726/g0_fibers.afv:{fid}'))
    fiber = S.Fiber(f'g0_{fid}', row['family'], 'afv', pts, s, {}, line, S.arclength(line), 0, parent_uuid,
                    'uuid5(g0 afv id)', f'g0_{fid}')
    s_raw, s_full, _, runs = fiber_map(xyz)
    k = breakpoints(s_raw, s_full)
    fmap = S.FullRunMap(fiber.length, s_raw[k]*2., s_full[k]*2., float(s_full[-1])*2.)
    return fiber, fmap, len(runs)


def cmd_plan(out):
    out.mkdir(parents=True, exist_ok=True)
    polys = {row['id']: (row, xyz) for row, xyz in read_afv_polylines(G0_AFV)}
    g0 = AFVFibers(str(G0_AFV), 8., split='all')
    rows, plan = [], []
    for tag, fid, s0, s1 in WINDOWS:
        row, xyz = polys[fid]
        fiber, fmap, kinks = g0_fiber(fid, row, xyz)
        w = dict(id=f'W{tag}', s0=s0, s1=s1, segment='paris4')
        piece = S.plan_pieces(fiber, [w], fmap)[0]
        piece['id'] = f'Q{tag}'
        A, B = piece['line_index_range']
        cut = np.asarray(xyz, np.float64)[A:B+1]
        pu = MP.piece_uuid(fiber.uuid, A, B)
        afv_id = int(pu.replace('-', '')[:13], 16)
        assert pu == piece['uuid'] and afv_id == piece['afv_id']
        ann = dict(source_file=G0_AFV.name, uuid=pu, uuid_source='uuid5(parent uuid, line index range)',
                   family=row['family'], span='piece', line_index_range=[A, B], line_points=len(xyz),
                   converter='tools/paris4_piece_check.py (make_piece_afv.py conventions)',
                   piece=dict(id=piece['id'], parent_g0_id=fid, parent_uuid=fiber.uuid,
                              parent_arc_sv=piece['parent_arc_sv'], windows=piece['windows']))
        rows.append((dict(id=afv_id, name=pu, family=row['family'], point_count=len(cut), length=0., min_x=0.,
                          max_x=0., min_y=0., max_y=0., min_z=0., max_z=0., annotation=json.dumps(ann, sort_keys=True)),
                     cut))
        plan.append(dict(tag=tag, g0_id=fid, family=row['family'], window_sv=[s0, s1], fiber_length_sv=fiber.length,
                         fiber_length_full_sv=fmap.length_full, fiber_kink_repairs=kinks, piece=piece,
                         cut_line_points=[A, B], fiber_line_points=len(xyz)))
    afv_path = out/'paris4_pieces.afv'
    if afv_path.exists():
        afv_path.unlink()
    meta = AFVFibers(str(G0_AFV), 8., split='all').metadata
    info = write_afv(None, afv_path, sorted(rows, key=lambda r: r[0]['id']),
                     dict(frame=meta['frame'], root=meta['root'],
                          uuid=str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, 'paris4_pieces:' + ','.join(
                              r[0]['name'] for r in rows))),
                          source='G0 Paris 4 fiber pieces (PREREG v3.1 erratum checks 5 and 6)'))
    afv = AFVFibers(str(afv_path), 8., split='all')
    by_tag = {p['tag']: p for p in plan}
    runplan = []
    for (rowd, cut), p in zip(rows, plan):
        pc = p['piece']
        geo = afv[afv.id_to_index[pc['afv_id']]]
        s_raw, s_rep, pts, _ = fiber_map(cut)
        if not np.array_equal(pts, np.asarray(geo.points)):
            raise SystemExit(f"{pc['id']}: piece geometry differs from AFVFibers")
        k0, k1 = pc['head_k']
        n = k1-k0+1
        x = (pc['heads_sv'][0]-pc['parent_arc_sv'][0])/2.
        t0 = float(np.interp(x, s_raw, s_rep))
        start = round(2.*t0, 6)
        end = round(start+16.*(n-1)+MP.END_PAD_SV, 6)
        arcs = MP.infer_heads(float(f'{start:.6f}'), float(f'{end:.6f}'), float(geo.length))
        if len(arcs) != n:
            raise SystemExit(f"{pc['id']}: infer.py would make {len(arcs)} decisions, the plan {n}")
        parent = g0[g0.id_to_index[p['g0_id']]]
        ref = interp_at(np.asarray(parent.points), np.asarray(parent.s), MP.MIN_HEAD_ARC+MP.STEP_TRACE*np.arange(k0, k1+1))
        got = interp_at(np.asarray(geo.points), np.asarray(geo.s), arcs)
        dev = np.linalg.norm(got-ref, axis=1)*2.
        p.update(arc_patch_vox=[start, end], decisions=int(n), head_k=[int(k0), int(k1)],
                 piece_length_sv=float(geo.length*2.), piece_kink_repairs=int(geo.kink_repairs),
                 max_head_dev_sv=float(dev.max()), first_head_trace=float(arcs[0]))
        runplan.append(dict(name=f'piece_{p["tag"]}', kind='piece', tag=p['tag'], afv=str(afv_path),
                            fiber=int(pc['afv_id']), arc=[f'{start:.6f}', f'{end:.6f}']))
    for p in plan:  # the control: infer.py on the whole G0 fiber (heads arange(12, L, 8), as infer_all)
        runplan.append(dict(name=f'full_{p["tag"]}', kind='full', tag=p['tag'], afv=str(G0_AFV), fiber=p['g0_id'],
                            arc=None))
    (out/'plan.json').write_text(json.dumps(S.jsonable(dict(
        tool='tools/paris4_piece_check.py', afv=str(afv_path), write_afv=info, g0_afv=str(G0_AFV),
        g0_afv_sha256=S.sha256_file(G0_AFV), windows=plan, runplan=runplan, infer_args=INFER_ARGS,
        checkpoint=CHECKPOINT, ct=CT_PARIS4, umbilicus=UMB_PARIS4)), indent=1))
    for p in plan:
        print(p['tag'], p['g0_id'], p['family'], 'window', p['window_sv'], 'heads k', p['head_k'], 'decisions',
              p['decisions'], 'cut', p['cut_line_points'], 'of', p['fiber_line_points'], 'arc', p['arc_patch_vox'],
              'max head dev sv', round(p['max_head_dev_sv'], 6), 'kinks fiber/piece', p['fiber_kink_repairs'],
              p['piece_kink_repairs'])


def cmd_run(out):
    plan = json.loads((out/'plan.json').read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', HF_HOME=str(STRIPS/'hf_home'), HF_HUB_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONIOENCODING='utf-8',
               PYTHONUNBUFFERED='1')
    for r in plan['runplan']:
        rec = out/'records'/f"{r['name']}.json"
        dest = out/'strips'/r['name']
        if rec.exists() and json.loads(rec.read_text()).get('status') == 'ok' and (dest/'strip.npz').exists():
            print(f"{r['name']}: done already", flush=True)
            continue
        args = ['--afv', r['afv'], '--fiber', str(r['fiber'])]
        if r['arc']:
            args += ['--arc', *r['arc']]
        args += ['--out', str(dest), '--checkpoint', CHECKPOINT, '--ct', CT_PARIS4, '--umbilicus', UMB_PARIS4,
                 '--cache-dir', CACHE, *INFER_ARGS]
        (out/'logs').mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(out/'logs'/f"{r['name']}.out", 'w') as so, open(out/'logs'/f"{r['name']}.err", 'w') as se:
            p = subprocess.run([PY, '-I', '-u', str(WRAPPER), str(rec), '--', *args], env=env, stdout=so, stderr=se,
                               cwd=str(SCORE), timeout=TIMEOUT_S,
                               creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
        print(f"{r['name']}: rc {p.returncode} in {time.time()-t0:.0f} s", flush=True)


def canonical(strip_dir):
    """A strip as (rows (R, C, 3) sv, valid) in infer.py's own layout (rows along the fiber from its start, columns
    in v order): load_strip undoes infer_all's transpose; the row/column reversals of its display layout are undone
    here from trace.json."""
    st = S.load_strip(strip_dir)
    rows, valid = st.rows, st.valid
    if st.trace.get('fiber_reversed'):
        rows, valid = rows[::-1], valid[::-1]
    if st.trace.get('across_reversed'):
        rows, valid = rows[:, ::-1], valid[:, ::-1]
    return st, rows, valid


def node_stats(a, va, b, vb):
    both = va & vb
    d = np.linalg.norm(a-b, axis=-1)[both]
    if not len(d):
        return dict(nodes=0)
    return dict(nodes=int(len(d)), mean_sv=float(d.mean()), p99_sv=float(np.percentile(d, 99)), max_sv=float(d.max()),
                median_sv=float(np.median(d)))


def compare_pair(piece, full, offset, row_sel=None):
    """Node distances (sv) between the piece strip's rows and the full strip's rows ``offset`` later."""
    (pr, pv), (fr, fv) = piece, full
    R = pr.shape[0]
    lo, hi = max(0, -offset), min(R, fr.shape[0]-offset)
    a, va = pr[lo:hi], pv[lo:hi].copy()
    b, vb = fr[lo+offset:hi+offset], fv[lo+offset:hi+offset]
    if row_sel is not None:
        keep = np.zeros(R, bool)
        keep[row_sel] = True
        va &= keep[lo:hi, None]
    out = node_stats(a, va, b, vb)
    out['valid_piece_only'] = int((va & ~vb).sum())
    out['valid_full_only'] = int((vb & ~va).sum())
    return out


def cmd_compare(out, report_path):
    plan = json.loads((out/'plan.json').read_text())
    polys = {row['id']: (row, xyz) for row, xyz in read_afv_polylines(G0_AFV)}
    # check 5: the scorer's real reading path on every real infer.py output
    found = S.discover_strips(out/'strips')
    reading = {}
    for d in found:
        st = S.load_strip(d)
        z = np.load(S.find_decisions(d))
        scale = 4./S.SCORE_UNIT_BASE
        tpp = float(z['trace_per_patch'])
        oriented, oframes, _ = S.orient_decisions(z['heights'].astype(np.float64), z['frames'].astype(np.float64))
        rows, weight, _ = S.stitch(oriented, oframes, z['positions'].astype(np.float64), z['arcs'].astype(np.float64),
                                   z['f'].astype(np.float64), z['v'].astype(np.float64), float(st.trace.get('blend', 16.)))
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(rows[weight > 0]/tpp*scale).query(st.rows[st.valid], k=1)
        reading[st.name] = dict(path=str(Path(d).relative_to(out)), rows=int(st.rows.shape[0]),
                                cols=int(st.rows.shape[1]), valid_nodes=int(st.valid.sum()), notes=st.notes,
                                frame_plane_available=st.frame_rows is not None,
                                frame_plane_nodes=int(np.isfinite(st.frame_rows).all(-1).sum())
                                if st.frame_rows is not None else 0,
                                expected_error_available=st.expected is not None,
                                reproduced_within_0_05_sv=float((dist <= 0.05).mean()),
                                reproduction_max_sv=float(dist.max()), column_step_sv=st.column_step,
                                decisions=int(len(z['arcs'])), npz_keys=sorted(z.files),
                                trace_keys=sorted(st.trace))
    # check 6 and the end-to-end scoring, per window
    results = []
    for p in plan['windows']:
        tag, fid = p['tag'], p['g0_id']
        row, xyz = polys[fid]
        fiber, fmap, _ = g0_fiber(fid, row, xyz)
        piece_dir = out/'strips'/f'piece_{tag}'/'strip'
        _, pr, pv = canonical(piece_dir)
        summ = json.loads((out/'strips'/f'piece_{tag}'/'summary.json').read_text())
        k0 = p['head_k'][0]
        df_trace = 2.  # f spacing (trace voxels), checked below against strip.npz
        zf = np.load(out/'strips'/f'piece_{tag}'/'strip.npz')
        assert abs(float(zf['f'][1]-zf['f'][0])-df_trace) < 1e-9
        offset = int(round(8.*k0/df_trace))
        # rows of the window: full-run arclength of the row within [S(s0), S(s1)] (trace voxels)
        S0, S1 = (float(fmap.to_full(x))/2. for x in p['window_sv'])
        first_full = 12.+float(zf['f'][0])  # full run's row 0 (trace voxels)
        rr = np.arange(pr.shape[0])
        row_arc_full = first_full+(rr+offset)*df_trace
        in_window = rr[(row_arc_full >= S0) & (row_arc_full <= S1)]
        # rows that take no decision outside the piece's heads k0..k1 (|row arc - head| < blend, trace voxels)
        k1 = p['head_k'][1]
        lo_ok, hi_ok = 12.+8.*(k0-1)+16., 12.+8.*(k1+1)-16.
        full_set = rr[(row_arc_full > lo_ok) & (row_arc_full < hi_ok)]
        res = dict(tag=tag, g0_id=fid, family=p['family'], window_sv=p['window_sv'], head_k=p['head_k'],
                   piece_row_offset_in_full=offset, window_rows=int(len(in_window)),
                   rows_with_the_full_runs_decisions=int(len(full_set)),
                   piece_arc_trace=summ['arc_trace'], piece_decisions=summ['decisions'])
        for ref, ref_dir in (('g0_infer_all_16_threads', G0_STRIPS/f'fiber_{fid}'),
                             ('infer_py_full_4_threads', out/'strips'/f'full_{tag}'/'strip')):
            st_f, fr, fv = canonical(ref_dir)
            res[ref] = dict(window_rows=compare_pair((pr, pv), (fr, fv), offset, in_window),
                            rows_with_the_full_runs_decisions=compare_pair((pr, pv), (fr, fv), offset, full_set),
                            all_rows=compare_pair((pr, pv), (fr, fv), offset))
            prof = []
            for r_ in rr:
                if offset+r_ >= fr.shape[0]:
                    continue
                both = pv[r_] & fv[offset+r_]
                if both.any():
                    prof.append([int(r_), round(float(row_arc_full[r_])*2., 1), bool(r_ in set(in_window)),
                                 round(float(np.linalg.norm(pr[r_]-fr[offset+r_], axis=-1)[both].max()), 4)])
            res[ref]['per_row_max_sv'] = dict(columns=['piece_row', 'full_run_arc_sv', 'in_window', 'max_sv'],
                                              rows=prof)
        # does any part of the fiber outside the piece come back within the crop sphere of a head (the tube input)?
        from scipy.spatial import cKDTree
        from erratum_geometry_checks import RADIUS_SV
        hs = fmap.to_raw(fmap.head_full(np.arange(p['head_k'][0], p['head_k'][1]+1)))
        heads = fiber.points[np.clip(np.searchsorted(fiber.s, hs), 0, len(fiber.s)-1)]
        a0, b0 = p['piece']['parent_arc_sv']
        outside = (fiber.s < a0-1e-9) | (fiber.s > b0+1e-9)
        if outside.any():
            d_out, _ = cKDTree(fiber.points[outside]).query(heads, k=1)
            res['outside_points_nearest_to_a_head_sv'] = float(d_out.min())
            res['heads_with_outside_points_in_crop_sphere'] = int((d_out <= RADIUS_SV).sum())
        # the G0 strip against the 4-thread full infer.py strip (both whole fibers): thread / batch noise
        _, g, gv = canonical(G0_STRIPS/f'fiber_{fid}')
        _, c, cv = canonical(out/'strips'/f'full_{tag}'/'strip')
        res['g0_vs_infer_py_full'] = compare_pair((c, cv), (g, gv), 0)
        # end to end: the piece strip scored by score_strips against the G0 full strip as a stand-in segment
        seg = S.SegmentSource(f'g0strip_{fid}', G0_STRIPS/f'fiber_{fid}', 4.)
        stretch = dict(id=f'W{tag}', fiber=fiber.name, s0=p['window_sv'][0], s1=p['window_sv'][1],
                       segment=seg.name, _n=0, family=p['family'], annotator='g0')
        strip = S.load_strip(piece_dir)
        strip.name = f'Q{tag}'
        entries, skips = S.score_strips([strip], [fiber], [stretch], [seg], None, None, log=lambda *a: None,
                                        pieces=[dict(id=f'Q{tag}', windows=[f'W{tag}'])])
        e = entries.get(f'Q{tag}')
        res['score_end_to_end'] = dict(
            skips=skips, scored=e is not None,
            strip_nodes=int(np.isfinite(e['err']['strip']).sum()) if e else 0,
            frame_plane_nodes=int(np.isfinite(e['err']['frame_plane']).sum()) if e and 'frame_plane' in e['err'] else 0,
            strip_mean_abs_sv=float(np.nanmean(np.abs(e['err']['strip']))) if e else None,
            frame_plane_mean_abs_sv=(float(np.nanmean(np.abs(e['err']['frame_plane'])))
                                     if e and 'frame_plane' in e['err'] else None),
            assoc_median_sv=e['assoc_median_sv'] if e else None, notes=e['notes'] if e else None)
        if e:
            agg = S.aggregate(entries)
            res['score_end_to_end']['aggregate'] = dict(
                scored_fibers=agg['scored_fibers'], strip=agg['strip']['point'],
                frame_plane_available_fibers=agg['frame_plane']['available_fibers'],
                frame_plane=agg['frame_plane']['stats']['point'] if agg['frame_plane']['stats'] else None)
        results.append(res)
    records = {f.stem: json.loads(f.read_text()) for f in sorted((out/'records').glob('*.json'))}
    rep = dict(tool='tools/paris4_piece_check.py', tool_sha256=S.sha256_file(Path(__file__)),
               scorer_sha256=S.sha256_file(SCORE/'score_strips.py'), plan_sha256=S.sha256_file(out/'plan.json'),
               scope='Paris 4 only (G0 fibers); no PHerc0139 or PHerc0332 strip',
               reading_path=reading, piece_vs_full=results,
               records={k: {kk: v.get(kk) for kk in ('status', 'wall_seconds', 'peak_rss_gb', 'torch_threads',
                                                      'cuda_visible_devices', 'priority', 'infer_py')}
                        for k, v in records.items()})
    Path(report_path).write_text(json.dumps(S.jsonable(rep), indent=1))
    print(json.dumps(S.jsonable(rep), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['plan', 'run', 'compare'])
    ap.add_argument('out', type=Path)
    ap.add_argument('report', nargs='?')
    a = ap.parse_args(argv)
    if a.cmd == 'plan':
        cmd_plan(a.out)
    elif a.cmd == 'run':
        cmd_run(a.out)
    else:
        cmd_compare(a.out, a.report)


if __name__ == '__main__':
    main()
