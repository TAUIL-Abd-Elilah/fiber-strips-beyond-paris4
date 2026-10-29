# Independent re-scoring sample (fixed before the results)

*The text down to the end of "Timing" is as committed at 02:29 UTC and pushed at 02:30 UTC on 29 Oct 2026 (commit
a2f1b56), before any result.*

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

## Result (29 Oct 2026, after the scored run)

`independent_rescore.py` was written from the text of PREREG sections 4.1, 5 and 7 without importing
`score/score_strips.py`, by AI agents: version 1.0 during the run (02:31 to 03:25 UTC), and version 1.1.0, which
added the solver bounds (tier 2), after it (about 04:00 to 05:24 UTC), when the scorer's reports existed (03:39 to
03:41 UTC), under a rule never to read them. A last edit of version 1.1.0 (06:09 UTC) was made by the author, not by
those agents, after the reports and parts of the scorer had been seen. It added the second review's results to the
docstring and exit code 2 on an input error; no reading, metric, tolerance or bound changed. The docstring lists
what was read of the scorer: file formats, units and the report structure; two rules seen there and not settled
independently (the umbilicus interpolation in z, R8; the per-piece aggregation, R9); and four facts a review
reported from the scorer (version 1.1). The code was reviewed adversarially twice, before and after version 1.1, on
synthetic and Paris 4 data only. Its tolerance and its solver bounds were fixed before it was compared with any
report. It was run on the run's outputs at 06:10 UTC:

    python verification/independent_rescore.py --workspace C:/VesuviusTemp/strips/score \
        --sample verification/verification_sample.json --out verification/rescore_report.json --compare

A final review then found that three sentences of the docstring dated version 1.1 before any result, and that its
summary of the second review left out some counts. Version 1.1.1 corrects the docstring and the version string only.
The same command was run again with it at 07:44 UTC: every verdict and every value is unchanged. The run gives
`rescore_report.json`; it differs from the 06:10 run's report only in its time, the tool's version and sha256, the
docstring text it repeats under `tolerance`, and the per-segment run times. The 06:10 run used version 1.1.0, sha256
`38fa0c90e502df5f46397cefe6556e957a5b29229ce83580b06f0d948c18e652`. Its report is shipped unchanged as
`rescore_report_v1.1.0.json`, and `independent_rescore_1.1.0_to_1.1.1.diff` is the change from 1.1.0 to 1.1.1. Run
in this folder, this command rebuilds version 1.1.0 (that sha256) without changing the current file:

    patch -R -o independent_rescore_1.1.0.py independent_rescore.py independent_rescore_1.1.0_to_1.1.1.diff

Both re-scorer runs exited with code 1, because one item differs; both reports record the workspace paths. All 224
comparisons are listed below.

- **Agree within 0.001 sv, integers exact: 213.**
  - 13 of the 15 pieces that hold a sampled stretch, on node count, mean \|d\| and signed mean d.
  - All 15 pieces on association median, stretch list and fiber floor.
  - The primary pool: fibers, nodes, mean, median, p90, nodes within 1 and 2 sv.
  - Every PHerc0332 crossing on missing status, directions, crossing point and closest approach, and 5 of its 12
    agreement values.
  - The ground-truth check for 190 of 192 stretches: the human fiber projected on its segment, against the
    numbers the frozen plan recorded before any strip existed.
- **Agree within the solver bounds: 10.** These are differences that another correct projection solver could give
  (docstring, "TOLERANCE AND VERDICTS").
  - 7 crossing agreements. The largest difference is X0000: 9.53 here against 9.37 sv in the scorer.
  - 2 ground-truth stretches: S0039 (maximum \|d\| off by 0.002 sv) and S1061 (maximum \|d\| off by 0.045 sv, mean d
    by 0.002 sv).
  - P0126: 1 node more here, and its signed mean d 0.0013 sv lower (its mean \|d\| is 0.001 sv higher, within
    tolerance).
- **Differ: 1, piece P0033** (26,664 strip nodes in the scorer, 26,650 here, of the 26,675 on its scored rows; the
  largest piece).
  - The scorer covers 14 nodes that this code leaves uncovered. Here their closest point on the mesh is more than
    1 sv sideways, off the mesh's edge. The scorer instead finds a foot elsewhere on the mesh: by the difference of
    the two sums, those 14 nodes lie about 9 sv from it there.
  - The piece's mean \|d\| differs by 0.004 sv (scorer 0.9160, here 0.9116) and its signed mean by 0.005 sv.
  - Dropping the 14 nodes would change the pooled secondary mean by about 0.001 sv (1.1986 to 1.1977).
  - The second review of this code had found this kind of case on synthetic wrinkled meshes and a spiral turn of
    more than 360 deg (two layers): the scorer's local Gauss-Newton solver can stop on another wrinkle or layer,
    farther than the closest point.
- **Side by side, not a pass/fail item: the PHerc0332 summary.** Mean 4.65 here against 4.63 sv; median 4.71
  against 4.71; p90 8.45 against 8.44. The share within 2 sv is 25 % here against 33 %, because X0046 is 2.02 sv
  here and 1.94 sv in the scorer, either side of the 2 sv line. The tight subset is 1.47 against 1.43 sv.

None of this changes a gate. The secondary pooled mean would move by about 0.001 sv, and the reported PHerc0332
share within 2 sv rests on one crossing that sits within 0.1 sv of the line.
