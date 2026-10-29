# Independent re-scoring sample (fixed before the results)

`verification_sample.json` lists what will be recomputed, after the scored run, by code that does not import
`score/score_strips.py`:
- 20 of the 284 run stretches of the PHerc0139 secondary set;
- both primary stretches;
- all 15 PHerc0332 run crossings.

The scorer's numbers for those items will be compared with the recomputed ones, and every comparison will be
reported whatever it shows.

**How the sample was drawn.** The 20 stretches were drawn from the frozen plan
(`score/results/matched_stretches_secondary_run40.json`, the stretches with `run = true`, sorted by id) with numpy
`default_rng(20261031).choice(284, 20, replace=False)`. The file records the sha256 of the three plan files it reads;
they equal the ones in `score/FREEZE.txt`.

**Timing.** It was drawn on 29 Oct 2026 at 02:27:30 UTC, while the scored run (started 02:13 UTC) was still at
inference, and committed before any result existed. This check is not part of the pre-registered gates in
`score/PREREG.md`. It is an extra check of the scorer, chosen before the scorer's output was seen.
