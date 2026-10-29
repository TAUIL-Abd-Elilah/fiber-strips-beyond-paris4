# Real `infer.py` output on Paris 4 (test fixture, PREREG v3.1 check 5)

Paris 4 only. No PHerc0139 or PHerc0332 data.

- `pieces/QA/`: the output of villa's `sheet_strip/infer.py` (worktree `C:/VesuviusTemp/strips/villa_6f2`: 6f2024569 +
  #2163 + the normalization patch) on piece QA of G0 fiber 778459, `--arc 300.704493 413.704493`, checkpoint
  `scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e`, `--device cpu --index none --no-render
  --step 8 --blend 16 --trim 75 --batch 32`, 4 torch threads, 29 Oct 2026 (`tools/paris4_piece_check.py`). Copied
  byte for byte from `runs/erratum_v31/paris4/strips/piece_A`.
- `g0_fiber_778459/`: the G0 CPU strip of the whole fiber (`C:/VesuviusTemp/strips/runs/g0`, villa `infer_all`, 16
  threads), which matches the published Paris 4 strip within 0.03 voxels.
- `fiber_778459.json`: that fiber's polyline (level-0 xyz) from the G0 AFV, a byte-for-byte subset of
  `dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/fibers.afv`, with the piece's window, heads and line range.

The model is sean's (bruniss) `fiber_strip_small_v1` (villa #2110); the Windows port is #2163 (slchicoineau-creator,
Baz248). The strips are derived from the Vesuvius Challenge Paris 4 scan and fiber data: **CC BY-NC 4.0** derived
data (see `LICENSE-DATA.md`), not covered by the code's MIT licence.
