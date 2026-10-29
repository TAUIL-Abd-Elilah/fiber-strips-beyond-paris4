# Pre-registration: `fiber_strip_small_v1` off Paris 4

**Accuracy on PHerc0139 against wrap segments, plus a consistency check on PHerc0332 (no ground truth).**

- **Date:** 28 Oct 2026. Drafted on the evening of 27 Oct 2026.
- **Frozen before any PHerc0139 or PHerc0332 strip exists.** No strip model has run on either scroll on this
  machine, and we have seen no strip from anyone else for these scrolls.
- **Freeze record:** `FREEZE.txt` holds the sha256 of this file, of `score_strips.py`, of its tests and of the
  data listings it names. Every report written by `score_strips.py` records the scorer's own sha256 and this file's
  sha256. Any change after the freeze goes in section 10 ("Deviations"), with its date and reason.
- **Author:** TAUIL Abd Elilah. AI assistance: the scorer, the tests and this text were drafted with Claude
  (Anthropic) and checked by running the tests.

Units: **sv** ("score voxel", called "L2 vox" in the plan) is level 2 of the 2.4 um scans: 4 base voxels, which is
9.596 um on PHerc0139. This is the strip model's "patch voxel". Paris 4's is 4 x 2.400 um = 9.6 um.

---

## 1. Question

sean's (bruniss) `fiber_strip_small_v1` predicts the papyrus sheet a fiber lies on, +-60 voxels across it
(villa #2110). It is `sheet_strip_paris4_slim50k_v3`, update 15000, trained on Paris 4 verified patches only.

**Primary question.** Run unchanged on human-traced fibers of PHerc0139, a scroll it never saw, does it place the
sheet correctly? "Correctly" is measured against the PHerc0139 wrap segments, which are meshed on the same scan as the
fibers.

**Secondary question.** On PHerc0332 there is no ground truth. Are the H and V strips consistent where their fibers
cross?

Reference numbers on Paris 4 (villa `tracers/fibers/README.md` at 6f2024569):
- validation error 1.006 / 1.117 vox (full / observed tube);
- on validation fiber 469373, 1.22 vox mean error against 6.89 for the frame plane, with 82 % of points within 2 vox.

Those Paris 4 numbers are heights along the model's frame normal, measured against verified patches. Ours are normal
distances to wrap-segment meshes. The two are close but not identical (section 8).

## 2. Model and inference (fixed before any result)

- **Code.** villa `6f2024569` (23 Oct 2026). This is before #2316, which made `fiber_store.py` import the C++
  `vc_object_store` unconditionally. The strip inference therefore still takes `--afv`.
- **Windows.** The #2163 Windows fixes, by slchicoineau-creator and Baz248: an msvcrt lock instead of fcntl, an eager
  gather when Triton is missing, and AF_UNIX on Winsock. A human-fiber JSON-to-AFV converter, in base voxels, writes
  a map from AFV fiber id to JSON file.
- **Checkpoint.** `scrollprize/fiber_strip_small_v1`, with its Hugging Face commit recorded. The frame model comes from
  the same repo. The checkpoint's own CT level and grid scales are kept; only the store changes:
  - PHerc0139: `PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr`, umbilicus
    `PHerc0139/representations/umbilicus/20260102150214-umbilicus-20260826112529.json`;
  - PHerc0332: `PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr`, umbilicus
    `PHerc0332/representations/umbilicus/20251211183505-umbilicus-20260828110920.json`.
- **Inference settings.** The villa defaults: `--step 8 --blend 16 --trim 75`, `--batch 4 --threads 1`. Inference
  runs on the CPU. The README says CPU and GPU numerics differ by about 0.015 vox on average.
- **Decisions.** Every scored fiber also keeps its decisions: `strip.npz` with `arcs, positions, frames, heights, f,
  v, trace_per_patch, expected_error_vox`, as `sheet_strip/infer.py` writes it. The frame-plane baseline needs them.
  `infer_all.py` does not write them.
- **Gate G0, decided by the setup run before any 0139 strip.** The pinned Windows build must reproduce at least 9 of
  10 published Paris 4 strips within 1 vox mean (`all_strips.zip`, read by HTTP range). If it fails, no 0139 strip is
  made from this build.
- **No tuning.** No inference setting, threshold or selection rule changes after any 0139 strip has been seen.

## 3. Data and frames

| Item | Source | Frame / units |
|---|---|---|
| PHerc0139 human fibers | `dl.ash2txt.org/other/annotators/0139_fibers/` (620 JSON on 27 Oct 2026: lt 338, kb 233, dj 49; listing in `data/fibers0139_sizes.txt`) | VC3D fiber JSON v3/v4, `line_points` xyz, level 0 of `PHerc0139/20260102150214` (2.399 um) |
| PHerc0139 wrap segments | S3 `PHerc0139/segments/*/mesh/*-on-20260102150214-2.399um.tifxyz`. 38 of the 39 segment dirs have one: w023-w059 and the title; `20260130000001-w030_202601301912` has none. Listing in `data/segments0139_mesh_keys.txt`, URLs in `data/segment_urls_0139_2399.txt` | tifxyz xyz in level-0 voxels, grid step 20 base voxels (meta scale 0.05) |
| Strips | outputs of section 2 | tifxyz in patch voxels (4 base voxels), `expected_error.tif` in patch voxels |
| Lasagna normals | S3 `PHerc0139/representations/predictions/lasagna/20260102150214-lasagna-20260419180421-L2/..._nx/_ny.ome.zarr`, level 4 | uint8 hemisphere encoding (villa lasagna: (u8-128)/127, nz >= 0), 16 base voxels per voxel |
| PHerc0332 human fibers | `dl.ash2txt.org/other/annotators/0332_fibers/` (255 JSON on 27 Oct 2026) | level 0 of `PHerc0332/20251211183505` |

PHerc0332 has two 2024 segments on other scans and none on 20251211183505, so it has no ground truth.

Wrap segment z extents come from their `meta.json` bboxes:
- the 37 wraps span z of about 3,000 to 30,600 base voxels (750 to 7,650 sv);
- the title segment spans z 32,940 to 39,560.

Fibers outside those ranges cannot be matched. Five sample fibers read before the freeze lay at z 34,000 to 73,000,
so G1 (section 6) may fail on coverage alone. This is stated here so that a G1 failure is not a surprise.

## 4. Ground-truth selection (which fibers are scored)

The normal distance `d` from a point to a surface is defined in 5.1.

1. **Span.** A fiber is the line between its first and last human control points. Fibers with fewer than 2 control
   points are excluded. This matches villa `annotations/vc3d.load_fibers`.
2. **Samples.** We sample every 1 sv along the span. For each sample we take the segment with the smallest `|d|`
   among the segments whose foot (the closest point of the surface) lies on the valid mesh, within 30 sv.
3. **Candidate fiber.** At least 70 % of its samples have `|d| <= 6 sv`, about half a winding gap. Such a fiber lies
   on a meshed sheet.
4. **Matched fiber.** This is the selection rule.
   - On-segment runs are consecutive samples on the same segment with `|d| <= 2 sv`. Runs shorter than 40 sv are
     dropped (villa's README uses `--min-run 40`).
   - The runs must cover at least 70 % of the samples.
   - The fiber must be at least 2.5 mm long (260 sv), which is `infer_all.py`'s default minimum length.
   - The ±2 window matches villa's `match_fibers_to_patches.py` (`--max-inward 2 --max-outward 2` in patch voxels).
5. **Noise floor.** For each candidate fiber, take the median `|d|` of its candidate samples (`|d| <= 6`). The noise
   floor is the median of these values over all candidate fibers. It measures the agreement of human fibers with
   segments, independent of the 2 sv match window. Measured on matched fibers alone it would be at most 2 by
   construction.
6. **Scored rows.**
   - A strip row is scored when its centre (the v = 0 column) lies within 10 sv of the fiber and its fiber
     arclength falls inside a matched run.
   - Each row is scored against that run's segment only, never against the nearest segment. A strip that jumps
     to the neighbouring winding therefore counts as an error of about one winding gap, not as a hit on the next
     wrap.
7. **Strip association.**
   - A strip belongs to the fiber its centre column runs along: median distance at most 5 sv.
   - A strip that matches no fiber is rejected as a unit or association error and listed.
8. **Family.** The JSON's manual H/V tag; else its automatic tag; else geometry (V when the z extent is more than
   half the length).
9. **Sensitivity results.** These are reported but do not gate:
   - matched fibers at a 50 % threshold;
   - fibers flagged by our fold-back/jump checks excluded (the #2248 noise classes);
   - results per annotator (lt / kb / dj).

## 5. Metrics

**5.1 Normal distance.**
- A point is projected onto the bilinear surface of the segment's tifxyz grid by Gauss-Newton from its 4 nearest
  nodes. `d` is the signed distance along the surface normal at the foot, positive outward (away from the umbilicus).
- The surface normal is the cross product of the grid's column and row derivatives, as in
  `match_fibers_to_patches.py`.
- A point is uncovered, and not scored, in two cases: its foot is off the valid mesh (sideways residual above
  1 sv), or `|d| > 30 sv`. Coverage is reported.

**5.2 Primary metric.**
- For every valid strip node on a scored row we take `|d|` to the matched segment.
- Pooled over nodes, we report the mean, median, p90, the fraction within 1 sv and the fraction within 2 sv.
- The signed mean `d` is the bias: inward or outward of the mesh.
- Each is reported for all fibers and per family (H, V).
- A fiber with fewer than 25 covered nodes (one strip row) is not scored.

**5.3 Uncertainty.**
- Cluster bootstrap over fibers: resample fibers with replacement and pool their nodes.
- 2000 replicates (1000 for the median and p90), 95 % percentile intervals, seed 20261028.

**5.4 Baselines.** Both are measured on the same strip rows and columns, with the same 5.1 distance.
- **(a) Frame plane.** The u = 0 plane of each decision's frame: the frame model's sheet plane through the fiber,
  the plane the strip starts from. It is stitched exactly as the strip, with villa's `stitch` and the same weights.
  This is the 0139 analogue of the README's 6.89 vox comparison.
- **(b) Lasagna-normal plane.**
  - The plane passes through the fiber point at the row's arclength and holds the fiber tangent. Its normal is the
    Lasagna normal there, made perpendicular to the tangent.
  - The Lasagna normal is the principal axis of the trilinearly weighted outer products of the 8 neighbouring
    level-4 normals.
  - Nodes sit at the strip's column offsets across the fiber.
- **Comparison.** A paired cluster bootstrap gives the baseline mean `|d|` minus the strip mean `|d|`, on nodes
  covered by both. The strip beats a baseline when the 95 % CI lower bound is above 0. The within-2 difference is
  also reported.

**5.5 Fiber floor.** The human fiber's own `|d|` at each scored row, one point per row. This is how far the
ground truth itself is from the mesh.

**5.6 Calibration of `expected_error`.**
- Bins of predicted error, with edges `[0, .5, .75, 1, 1.25, 1.5, 2, 3, inf)` sv.
- Per bin: the node count, the mean predicted error, and the mean and median observed `|d|`.
- Also: the node-weighted mean `|observed - predicted|`, the ratio of observed to predicted, and Spearman's rho at
  node and at fiber level.
- The model predicts its error along its frame normal against Paris 4 patches; we observe a normal distance to the
  wrap mesh.

**5.7 Diagnostics.**
- The median distance of each strip's centre column from its fiber (a unit check).
- The angle between the Lasagna normal and the segment normal at the fiber.
- The coverage of every surface.

## 6. Gates and what follows from each

**G1: is the ground truth adequate?** Run on the fibers and segments alone, before any strip exists.
- Pass: **at least 30 matched fibers** and a **noise floor of at most 2 sv**.
- Fail: we drop the 0139 accuracy claim. We publish the G1 numbers, the 0332 strips and the crossing agreement,
  labelled "no ground truth".

**G1' (at scoring time).** At least 30 fibers must actually be scored (at least 25 covered nodes each). Otherwise
the outcome is the same as a G1 failure.

**G2: transfers or degrades.** The verdict is on all families pooled; per-family results are reported.
- **"transfers"** requires all three of:
  - a pooled mean `|d|` of at most 1.5 sv;
  - at least 70 % of nodes within 2 sv;
  - **both** baselines beaten, with the paired-bootstrap CI excluding 0.
- **"degrades"** in every other case.
- A baseline that cannot be computed counts as not beaten: frame planes need decisions on at least 2 fibers; the
  Lasagna plane needs normals. So the verdict cannot be "transfers" without both baselines. Missing inputs are named
  in the report.

**Publication.** Whatever the outcome ("transfers", "degrades" or "G1 failed"), we publish:
- the report JSON, this pre-registration with `FREEZE.txt`, the scorer and its tests, on GitHub under
  TAUIL-Abd-Elilah with a permissive licence;
- a short write-up, as a draft for the user to post (Claude never posts).

No result is withheld. No run is repeated with changed settings to get a different verdict.

## 7. PHerc0332: H/V crossing agreement (no ground truth)

1. **Crossings.** A crossing is an H fiber and a V fiber whose closest approach is at most 3 sv. Each pair gives
   one crossing, at its closest approach.
2. **Agreement.** For each of the two strips, take its nodes within 30 sv of the crossing point and project them
   onto the other strip's surface (5.1, unsigned); take the median `|d|`. The crossing agreement is the mean of the
   two medians.
3. **Summary.** Over all crossings: mean, median, p90, and the fraction at most 2 sv, each with a bootstrap CI over
   crossings (2000 replicates). The summary is labelled **"no ground truth: consistency, not accuracy"**.
4. **The same number on PHerc0139** (`score --crossings`), so the consistency number can be read against measured
   accuracy.

Limits of the consistency number:
- Two strips can agree and both be wrong, for example both on the neighbouring winding.
- Fibers that cross while lying on different windings (an annotation error) show a disagreement of about one gap.

## 8. Known limits

- **Meshes can be off-sheet.** Segment meshes are not the sheet. Wrap meshes can sit on the sheet surface or its
  middle, smooth over detail, or jump. The scorer measures distance to the mesh.
- **Human fiber noise.** ChaseHendrick's fiber_qa (#2248) found 72 fold-backs, 16 jumps and 8 near-duplicate pairs
  in the published 0139 fibers. The noise floor (G1), the 2 sv selection window and the sensitivity runs (4.9) bound
  this; they do not remove it.
- **Fiber vs mesh offset.** A fiber lies on a sheet surface, while a mesh may follow the sheet's middle. A constant
  offset appears as a signed bias, which is reported. It is not subtracted.
- **Metric mismatch with the README.**
  - Normal distance to a wrap mesh vs frame-normal height against verified patches.
  - 9.596 um vs 9.6 um voxels.
  - Segment cell 5 sv: the facet sagitta is about 0.02 sv at a 200 sv radius, so it is negligible.
- **Selection bias.** Only fibers on meshed sheets are scored, and meshed regions are the readable ones. The result
  does not cover unmeshed or compressed regions.
- **Coverage risk.** Wrap meshes span z of about 3,000 to 30,600 base voxels (section 3). Human fibers elsewhere
  cannot count toward G1.
- **Coarse Lasagna normals.** Level 4 is 16 base voxels (about 38 um). Its plane is a weak baseline on curved
  sheets; the frame plane is the README's own baseline.
- **CPU inference.** Small numeric differences from the GPU path are expected (about 0.015 vox on average, per the
  README).
- **Families.** The H/V tag is the JSON's automatic classification when no manual tag is set.

## 9. Credits

- **sean (bruniss):** the sheet-strip model, training and inference code and the Paris 4 strips (villa #2110), the
  fiber follower (#2045), the frame model and `match_fibers_to_patches.py`, whose rule section 4 follows.
- **slchicoineau-creator and Baz248:** #2163, the native Windows port of the follower and the strip model.
- **modaribrahim:** `fiber-tracer-heldout-eval` (MIT), the held-out evaluation of fiber tracing on the 0139/0332/1451
  human fibers, and the scoring fix in villa #2143.
- **ChaseHendrick:** #2248 fiber_qa (ground-truth noise), #2265 (sheet-normal drift of the beam tracer on 0139),
  #2320 (v4 JSON ingest).
- **The annotators** of the 0139 and 0332 human fiber scaffolds.
- **The Vesuvius Challenge data team:** scans, wrap segments, Lasagna predictions, umbilici.
- **villa** is MIT-licensed. `orient_decisions`, `stitch` and `stitch_values` are copied from
  `sheet_strip/infer.py` at 6f2024569 with attribution.

## 10. Deviations after the freeze

None at freeze time.
