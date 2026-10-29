"""render.png of an existing sheet_strip/infer.py output on a scroll other than Paris 4 (CT on the predicted strip).

    CUDA_VISIBLE_DEVICES=-1 python tools/render_strip.py --villa C:/VesuviusTemp/strips/villa_6f2 --out RENDER_DIR \
        --checkpoint scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e \
        --ct s3://vesuvius-challenge-open-data/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr \
        --cache-dir C:/VesuviusTemp/strips/ct_cache

RENDER_DIR is a copy of one piece's output folder (strip/, strip.npz, summary.json); render.png is written there, so
the run's own folders are never written.

Why not ``infer.py --render-only``: that branch of infer.py's main() opens the CT with the checkpoint's normalization
record as it is, while our patch (patches/strip_bind_normalization_other_volume.diff) binds the record to the CT read
only in StripPredictor, i.e. for inference. On PHerc0139 --render-only therefore stops with "Invalid z-score
normalization record or different volume". This script does exactly what that branch does (resolve the checkpoint,
read its vol_spec, apply the CT options, render_saved) with infer.bind_normalization(spec) added before the volume is
opened, as inference does. villa is imported from the pinned worktree and is not changed. CPU only: it refuses to run
unless CUDA_VISIBLE_DEVICES is -1.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--villa', required=True, help='the pinned villa worktree (villa_6f2)')
    ap.add_argument('--out', required=True, help='a copy of one infer.py output folder; render.png is written there')
    ap.add_argument('--threads', type=int, default=4, help='torch threads (default 4)')
    args, rest = ap.parse_known_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise SystemExit('CUDA_VISIBLE_DEVICES must be -1')
    src = Path(args.villa).resolve()/'vesuvius'/'src'
    sys.path.insert(0, str(src))
    import torch
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip import infer as INF
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolume, CTVolumeSpec
    if not Path(INF.__file__).resolve().is_relative_to(src):
        raise SystemExit(f'villa imported from {INF.__file__}, not from {src}')
    torch.set_num_threads(args.threads)
    opts = argparse.ArgumentParser()
    INF.add_checkpoint_arguments(opts)  # --checkpoint, the CT options (--ct, --cache-dir, ...), --umbilicus, --device
    ns = opts.parse_args(rest)
    out = Path(args.out)
    for name in ('summary.json', 'strip.npz', 'strip/x.tif', 'strip/y.tif', 'strip/z.tif'):
        if not (out/name).is_file():
            raise SystemExit(f'{out/name} is missing')
    # infer.py main(), --render-only branch, plus bind_normalization (as StripPredictor does for inference)
    path = INF.resolve_checkpoint(ns.checkpoint)
    spec = CTVolumeSpec.from_dict(torch.load(path, map_location='cpu', weights_only=False)['vol_spec'])
    INF.configure_volume(ns, spec, argparse.Namespace(), path)
    INF.bind_normalization(spec)
    INF.render_saved(out, CTVolume(spec))
    print(f'[render_strip] wrote {out/"render.png"}', flush=True)


if __name__ == '__main__':
    main()
