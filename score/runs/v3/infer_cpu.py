"""Run villa's sheet_strip/infer.py (pinned worktree C:/VesuviusTemp/strips/villa_6f2) for one piece, CPU only.

    python -I infer_cpu.py RECORD.json -- <infer.py arguments>
    python -I infer_cpu.py --preflight CT UMBILICUS     # load the model, frame model and CT metadata; no inference

Refuses to run when torch sees a GPU (CUDA_VISIBLE_DEVICES must be -1) or when villa would be imported from anywhere
but the pinned worktree. Runs at BelowNormal priority with at most THREADS torch threads (one interop thread). The
record holds the status, wall time, decisions, peak RSS and thread count; it is written only after infer.py
returned, so its presence with status 'ok' marks a finished piece. Nothing of the strip is read here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time

VILLA_SRC = Path('C:/VesuviusTemp/strips/villa_6f2/vesuvius/src')
THREADS = 8
CHECKPOINT = 'scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e'


def setup():
    sys.path.insert(0, str(VILLA_SRC))
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[k] = str(THREADS)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise SystemExit('CUDA_VISIBLE_DEVICES must be -1')
    import psutil
    proc = psutil.Process()
    try:
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception as error:  # noqa: BLE001
        print(f'[infer_cpu] could not lower the priority: {error}', flush=True)
    import torch
    if torch.cuda.is_available() or torch.cuda.device_count():
        print('[infer_cpu] torch sees CUDA: refusing to run', flush=True)
        sys.exit(2)
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    import vesuvius
    if Path(vesuvius.__file__).resolve().parents[1] != VILLA_SRC.resolve():
        raise SystemExit(f'villa imported from {vesuvius.__file__}, not the pinned worktree')
    return torch, proc, psutil


def preflight(ct, umbilicus):
    torch, proc, psutil = setup()
    import argparse
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip import infer as INF
    ap = argparse.ArgumentParser()
    INF.add_checkpoint_arguments(ap)
    args = ap.parse_args(['--checkpoint', CHECKPOINT, '--ct', ct, '--cache-dir', 'C:/VesuviusTemp/strips/ct_cache',
                          '--umbilicus', umbilicus, '--device', 'cpu'])
    t0 = time.time()
    pred = INF.StripPredictor(args.checkpoint, args.device, args.umbilicus, 16, volume=args)
    print(json.dumps(dict(preflight='ok', ct=ct, checkpoint=pred.source, step=pred.ck.get('step'),
                          trace_per_patch=pred.trace_per_patch, grid_scale=pred.spec.grid_scale,
                          ct_grid_scale=pred.spec.ct_grid_scale, f_spacing=float(pred.f[1]-pred.f[0]),
                          vol_shape=[int(x) for x in getattr(pred.vol, 'shape', [])], torch=torch.__version__,
                          torch_threads=torch.get_num_threads(), cuda_available=torch.cuda.is_available(),
                          priority=str(proc.nice()), infer_py=INF.__file__, seconds=round(time.time()-t0, 1))),
          flush=True)


def run(record_path, argv):
    torch, proc, psutil = setup()
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip import infer as INF
    peak = dict(rss=0)
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                peak['rss'] = max(peak['rss'], proc.memory_info().rss)
            except Exception:  # noqa: BLE001
                pass
            stop.wait(1.)
    threading.Thread(target=sample, daemon=True).start()
    t0, cpu0 = time.time(), proc.cpu_times()
    status = 'ok'
    try:
        INF.main(argv)
    except SystemExit as error:
        status = f'SystemExit {error.code}'
    except Exception as error:  # noqa: BLE001
        status = f'{type(error).__name__}: {error}'
    stop.set()
    cpu1 = proc.cpu_times()
    rec = dict(status=status, argv=argv, wall_seconds=round(time.time()-t0, 2),
               cpu_user_seconds=round(cpu1.user-cpu0.user, 1), peak_rss_gb=round(peak['rss']/1e9, 3),
               torch_threads=torch.get_num_threads(), torch=torch.__version__,
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), priority=str(proc.nice()),
               infer_py=INF.__file__, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    Path(record_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(record_path)+'.tmp')
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, record_path)
    print(f'[infer_cpu] {status} in {rec["wall_seconds"]} s, peak RSS {rec["peak_rss_gb"]} GB', flush=True)
    sys.exit(0 if status == 'ok' else 1)


if __name__ == '__main__':
    if sys.argv[1] == '--preflight':
        preflight(sys.argv[2], sys.argv[3])
    else:
        assert sys.argv[2] == '--'
        run(sys.argv[1], sys.argv[3:])
