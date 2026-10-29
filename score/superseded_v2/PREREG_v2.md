# Pre-registration v2: `fiber_strip_small_v1` off Paris 4

**Accuracy on PHerc0139 against wrap segments, on the fiber stretches that lie on them, plus a consistency check
on PHerc0332 (no ground truth).**

- **Date:** 28 Oct 2026.
- **Status at freeze:** no PHerc0139 or PHerc0332 strip has been scored or inspected. One 0139 smoke-test strip
  exists (section 10.3); its fiber has no matched stretch, so it can't enter any scored set.
- **Freeze record:** `FREEZE.txt` holds the sha256 of this file, of `score_strips.py`, its tests, the matched
  sets, the AFVs and run plans, and the data manifests. Every report from `score_strips.py` records the scorer's
  sha256 and this file's sha256. Any later change goes in section 10.4 with its date and reason.
- **Supersedes** v1, frozen on 27 Oct 2026 at 19:36 (`superseded_v1/`). v1 used a whole-fiber selection rule. What
  changed, and what was seen before v2 was written, is in section 10.
- **Author:** TAUIL Abd Elilah. AI assistance: the scorer, its tests and this text were drafted with Claude
  (Anthropic) and checked by running the tests.

**Units.**
- **sv** ("L2 vox") is level 2 of the 2.4 um scans: 4 base voxels, which is 9.596 um on PHerc0139 and PHerc0332.
  It equals the strip model's "patch voxel" (Paris 4: 9.6 um).
- **Trace voxels.** The strip model walks a fiber in trace voxels (AFVFibers grid_scale 8): 8 base voxels, or 2 sv.

**Decision spacing.** The setup note said one decision per 8 voxels. Checked against the code, it is one decision
per 8 *trace* voxels. That is 16 sv, or 154 um, of fiber per decision (`--step 8`, `infer_all`:
`arange(12, L, 8)` in trace voxels). The setup run's full-0139 estimate uses the same rule: 231,715 decisions for
35.7 m of fiber.

---

## 1. Question

sean's (bruniss) `fiber_strip_small_v1` predicts the papyrus sheet a fiber lies on, ±60 voxels across it
(villa #2110). It is `sheet_strip_paris4_slim50k_v3`, update 15000, trained on Paris 4 verified patches only.

**Primary question.** Run unchanged on human-traced fibers of PHerc0139, a scroll it never saw, does it place the
sheet where the PHerc0139 wrap segments are? The segments are meshed on the same scan as the fibers. It is scored
only where the human fiber itself lies on a segment.

**Secondary question (PHerc0332, no ground truth).** Are the H and V strips consistent where an H and a V fiber
cross?

Reference numbers on Paris 4 (villa `tracers/fibers/README.md` at 6f2024569):
- validation error 1.006 / 1.117 voxels (full / observed tube);
- on validation fiber 469373, 1.22 voxels mean error against 6.89 for the frame plane, with 82 % of points within
  2 voxels.

## 2. Model and inference (fixed before any result)

- **Code.** villa `6f2024569` plus #2163 (the Windows port by slchicoineau-creator and Baz248) plus the setup run's
  normalization patch, at `C:/VesuviusTemp/strips/villa_6f2`.
  - Gate G0 passed: 10 of 10 published Paris 4 strips were reproduced within 0.03 voxels.
  - Strips were matched node by node on the same grid (`results/g0.json` of the setup run).
- **Checkpoint.** `scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e`, with its frame model
  (`heading_model.pt`) from the same commit.
- **Runs.** One `sheet_strip.infer` process per AFV piece, on the CPU (`CUDA_VISIBLE_DEVICES=-1`, `--device cpu`).
  It uses villa's default settings:

  ```
  python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer --afv <AFV> --fiber <afv_id> \
    --arc <arc_start> <arc_end> --out <OUT>/<piece id> --checkpoint scrollprize/fiber_strip_small_v1@7bd163a2... \
    --ct <volume zarr> --umbilicus <level-2 umbilicus> --cache-dir C:/VesuviusTemp/strips/ct_cache \
    --device cpu --index none --no-render --step 8 --blend 16 --trim 75
  ```

  - `infer.py` writes `strip/` (tifxyz) and `strip.npz`, which holds the decisions the frame-plane baseline needs.
  - `--afv`, `--fiber` and `--arc` come from the run plans (`results/runplan_*.tsv`).
  - The volume and umbilicus for each scroll:

    | Scroll | `--ct` | `--umbilicus` |
    |---|---|---|
    | PHerc0139 | `.../PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr` | `C:/VesuviusTemp/strips/data/umbilicus/PHerc0139_20260102150214_umbilicus_L2.json` |
    | PHerc0332 | `.../PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr` | `.../PHerc0332_20251211183505_umbilicus_L2.json` |
- **Pieces, not whole fibers.** A full 0139 run would need about 48 CPU-h and 157 to 183 GiB of CT, so only the
  matched stretches are run. Section 4.6 explains why a piece with `--arc` reproduces the decisions of a full-fiber
  run on the scored rows.
- **No tuning.** No inference setting, selection rule, threshold, subsample or metric changes after any 0139 or
  0332 strip has been seen.

## 3. Data and frames

| Item | Source (fetched 28 Oct 2026) | Frame / units |
|---|---|---|
| PHerc0139 human fibers | `dl.ash2txt.org/other/annotators/0139_fibers/`: 620 JSON, 235,298,374 bytes, in `data/fibers0139/` (`data/fibers0139_manifest.json`: sha256 per file) | VC3D fiber JSON v3/v4, `line_points` xyz, level 0 of `PHerc0139/20260102150214` (2.399 um) |
| PHerc0139 wrap segments | S3 `PHerc0139/segments/*/mesh/*-on-20260102150214-2.399um.tifxyz`. 39 segment dirs, 38 with such a mesh (w023 to w059 and the title); `20260130000001-w030_202601301912` has none. 152 files, 935,849,599 bytes (892.5 MiB), in `data/segments/` (`data/segments_manifest.json`) | tifxyz xyz in level-0 voxels, grid step 20 base voxels (5 sv) |
| PHerc0332 human fibers | `dl.ash2txt.org/other/annotators/0332_fibers/`: 255 JSON, 88,068,432 bytes, downloaded by the setup run on 27 Oct 2026, copied to `data/fibers0332/` (`data/fibers0332_manifest.json`) | level 0 of `PHerc0332/20251211183505`; the JSONs carry no coordinate space, so `PHerc0332/20251211183505@L0` is used |
| Lasagna normals (0139) | S3 `PHerc0139/representations/predictions/lasagna/20260102150214-lasagna-20260419180421-L2/..._nx/_ny.ome.zarr/4` | uint8 hemisphere encoding, 16 base voxels per voxel |
| CT (for budgets only) | `.../20260102150214-...zarr/2`: shape (19239, 6628, 6628), 128³ uint8 chunks of 2 MiB. 0332 the same | level 2 = sv |

- **Unusable fibers.** 5 of the 620 0139 fibers have fewer than 2 control points, so no controlled span: 615
  remain. 2 of the 255 0332 fibers likewise: 253 remain.
- **H/V tags.** Every remaining fiber has a manual or automatic H/V tag.
- **Coverage.** Only 45.9 % of the 0139 fiber samples lie inside the z range of any wrap segment.

## 4. Ground-truth selection (PHerc0139)

### 4.1 Distance

- Fibers are taken over their controlled span, between the first and last human control points, as villa's
  `annotations/vc3d.load_fibers` and the converter do. They are sampled every 1 sv.
- For every sample and segment, the signed normal distance `d` to the segment's bilinear surface is computed
  (section 5.1). A sample counts only when its foot lies on the segment's valid mesh.
- **Cross-check.** On three segments (about 55,000 samples) the distance agrees with villa's
  `match_fibers_to_patches.py` rule (nearest node, offset along the node normal): median difference 0.01 sv, p95
  0.1 sv.

### 4.2 The primary rule (as specified for this build)

- **Stretch.** A *stretch* is a maximal run of consecutive samples on one segment with `|d| <= 2` sv (8 base
  voxels). It is the window of villa's `--max-inward 2 --max-outward 2`. The run must be at least **2.5 mm** long:
  260.5 sv, `infer_all`'s minimum fiber length.
- **Overlaps.** A sample can be on two segments when meshes overlap. Stretches are then accepted longest first; a
  later stretch keeps its longest non-overlapping part if that is still long enough, else it is dropped and
  listed.
- **Selected set.** Every stretch that passes is selected. The unit of inference is the stretch's piece (4.6); the
  bootstrap unit is the fiber.

### 4.3 Primary counts (`results/matched_stretches.json`)

**2 stretches on 2 fibers, both H, 7.42 mm in total:**

| Stretch | Fiber | Segment | Length | mean `d` | median `\|d\|` | p90 `\|d\|` |
|---|---|---|---|---|---|---|
| S0000 | `kb_20260904T021448778_000085` | w044 | 3.73 mm | +0.47 | 0.92 | 1.69 sv |
| S0001 | `lt_20260918T092404850_000160` | w054 | 3.68 mm | +0.13 | 0.61 | 1.23 sv |

- **Inference.** 2 pieces, 58 decisions, 0.03 CPU-h. CT: 138 level-2 chunks within r = 170 of the decision path
  (0.27 GiB), about 0.13 h of download. Nothing is cached yet.
- **G1 fails** at this stage: 2 fibers, against at least 30 required (section 6).

### 4.4 Noise floors

- **NF_match** = median `|d|` over all samples of the matched stretches.
  - Primary: **0.78 sv**. Secondary (4.10): 0.82 sv.
  - It can't exceed 2 sv, because the selection bounds it.
- **NF6** = median `|d|` over all samples of the *near runs*: runs of consecutive samples on one segment with
  `|d| <= 6` sv (half a winding gap), at least 2.5 mm long. NF6 doesn't depend on the 2 sv window.
  - Value: **1.76 sv**, over 287 near runs.

### 4.5 Ground-truth diagnostics

These numbers come from the fibers and segments only. They were seen before this file was written; they explain
the primary count and motivated 4.10.

**Where the fiber samples lie (35.72 m of fiber, 3,722,114 samples):**

| Samples | Share |
|---|---|
| within 2 sv of a segment | 4.09 % |
| within 6 sv of a segment | 8.51 % |

- Within 6 sv, the signed `d` has median +0.51 sv and `|d|` has median 2.09 sv.
- Only 48 % of those samples are within 2 sv. Their histogram from −6 to 6 sv is broad, with a modest peak at 0
  to +1 sv.

**Fibers with one contiguous run on one segment, by window and minimum length:**

| Window | ≥ 2.5 mm | ≥ 1 mm | ≥ 0.5 mm |
|---|---|---|---|
| `\|d\| <= 2` sv | 2 | 30 | 54 |
| `\|d\| <= 3` sv | 18 | 42 | 58 |
| `\|d\| <= 4` sv | 28 | 53 | 62 |
| `\|d\| <= 6` sv | 46 | 59 | 64 |

- Allowing gaps of up to 20 samples in a 2 sv run gives 17 fibers at 2.5 mm.

The human fibers rarely stay within 2 sv of a wrap mesh for 2.5 mm. No reasonable variant of the primary rule
reaches 30 fibers.

### 4.6 Inference plan: decision heads and pieces

**Decision heads.**
- A full-fiber run puts heads at 24 + 16k sv of fiber arclength (`MIN_HEAD_ARC` 12 and `--step` 8, in trace
  voxels).
- A row takes the decisions whose heads are within 32 sv of it (`--blend` 16).
- For each stretch [s0, s1], the planned heads are the full-run grid points in [s0 − 40, s1 + 40], clipped to the
  fiber. So every scored row gets every decision a full run would give it.

**Pieces.**
- **Extent.** The fiber is kept from 300 sv before the first head to 160 sv after the last head:
  - 300 sv covers the observed path behind (`n_history` 128 trace voxels = 256 sv) plus margin;
  - 160 sv covers the crop ahead (crop reach 58 trace voxels = 116 sv) and the prediction window.
- **Cut.** Pieces are cut at the JSON's own line points, copied bit for bit, and kept inside the controlled span.
- **Merging.** Pieces of one fiber that overlap are merged. One `infer.py` run covers all heads of a merged piece,
  from its windows' first head to their last, including any gap between them.
- **Trim.** The piece's ±75 sv end trim never reaches a scored row, except at a true fiber end, where a full run
  trims it the same way.
- **AFV conventions.** The AFV follows `tools/fibers_json_to_afv.py`:
  - native level-0 xyz;
  - uuid = uuid5(URL, `vc3d_fiber_piece:<parent uuid>:<first>-<last line index>`), AFV id = its first 13 hex
    digits;
  - the annotation keeps the parent file, uuid and id, the line range and the windows.
- **`--arc` values.** `tools/make_piece_afv.py` opens the AFV as the model does (`AFVFibers`, grid_scale 8,
  densified and kink-repaired). It places each piece's first and last head on that geometry and writes the exact
  `--arc` values and decision counts.
  - Primary: 58 decisions, as planned, and no kink repairs.
  - Secondary: 12,643 decisions against 12,764 planned. 45 of 181 pieces are kink-repaired, which shortens them by
    up to 18.5 sv.
- **Where it differs from a full run.** Head positions match a full run within about 1 sv. Inputs differ only
  where AFVFibers' kink repair acts differently near a cut.

### 4.7 Budget

- **CPU.** 1.35 decisions/s on the CPU, as measured by the setup run, plus an *assumed* 30 s per `infer.py` process
  (model, frame model, CT store; not measured).
- **CT.** The union of the level-2 chunks (2 MiB each, uncompressed) within **r = 170** level-2 voxels of each
  piece's decision path. The path is sampled every 1 voxel.
- **Download time.** At the measured 0.58 MiB/s.

### 4.8 Seeded subsample rule (applies when a set exceeds its caps)

**Caps.**
- PHerc0139: **8 CPU-h** and **8 GiB of new CT**. C: has about 21 to 29 GB free and must keep 8 GB.
- PHerc0332: 2 CPU-h, 2 GiB and at most 60 crossings.

**Rule.**
1. **Units.** The units are pieces; for PHerc0332, crossings.
2. **Strata.** Units are grouped into strata: (segment of the piece's longest stretch, family). For PHerc0332:
   (tight crossing yes/no, z tercile).
3. **Order.** Strata are taken in sorted key order. Within a stratum, units are sorted by id and permuted by
   `numpy.random.default_rng(20261028)`, drawn stratum by stratum in that order.
4. **Selection.** Round robin over the strata: each turn offers a stratum's next unit, preferring fibers not yet
   chosen.
5. **Caps.** A unit is taken if the cumulative CPU hours and new CT stay within the caps. Otherwise it is passed
   over for good.

This is `stratified_subsample` in the scorer. The primary set is within its caps, so it isn't subsampled.

### 4.9 Sensitivity results (reported, not gating)

- unflagged stretches only, without the #2248-style fold-backs or jumps flagged in the stretch;
- per annotator (lt / kb / dj);
- per family.

### 4.10 Secondary set (pre-registered now; labelled, never reported as the primary result)

**Rule.** The same as 4.2, but the minimum stretch is **40 sv** (0.384 mm). This is the `--min-run 40` with which
sean matched fibers to patches for this model's own training and validation (README, "Fibers on patches"). It was
chosen after the primary count (2 fibers) was known, from the fibers and segments only, before any strip.

Results in `results/matched_stretches_secondary_run40.json`:

**All matched:**
- **Stretches:** 57 fibers (25 H, 32 V), 1,075 stretches, 710.5 mm in total; median stretch 58 sv, p90 108 sv.
- **Overlaps:** 96 overlaps dropped.
- **Noise floors:** NF_match 0.82 sv, NF6 1.76 sv.
- **Inference:** 181 pieces, 12,764 decisions, 4.1 CPU-h.
- **CT:** 12,487 chunks, **24.3 GiB** (11.9 h of download). This is over the CT cap, so the set is subsampled.

**Run subsample (4.8), `run: true`:**
- **Stretches:** 39 pieces, 242 stretches, **30 fibers** (12 H with 188 stretches, 18 V with 54), on 25 segments,
  167.9 mm in total.
- **Inference:** 2,932 decisions, 0.93 CPU-h.
- **CT:** 4,095 chunks, 8.00 GiB (3.9 h of download).

**Verdict.** The secondary set gets the same G1, G1' and G2 rules. Its verdict is reported as
"secondary: transfers / degrades / no accuracy claim".

Primary stretch S0000 is in the secondary run subsample (as S0200); S0001 is not (S0992), so the primary pieces are
run separately.

## 5. Metrics

**5.1 Normal distance.**
- A point is projected onto the bilinear surface of the segment's tifxyz grid by Gauss-Newton from its 4 nearest
  nodes.
- `d` is the signed distance along the surface normal at the foot, positive outward (away from the umbilicus).
- A point is uncovered, and not scored, if its foot is off the valid mesh (sideways residual above 1 sv) or
  `|d| > 30` sv.

**5.2 Scored nodes.**
- A strip row is scored when two conditions hold:
  - its centre column lies within **30 sv** of its fiber;
  - its arclength on that fiber (nearest fiber sample) falls inside a matched stretch.
- Each row is scored against **that stretch's segment only**. A strip that jumps to the next winding inside a
  stretch counts as an error of about one gap (12 sv), not as a hit on the next wrap.
- A strip belongs to its piece (OUT dir named by the piece id) and must run along that fiber: median centre
  distance ≤ 5 sv. If not, it is rejected as a unit or association error.
- A fiber with fewer than 25 covered nodes is not scored.

**5.3 Primary metrics.** Pooled over nodes, for all fibers and per family (H, V):
- the mean, median and p90 of `|d|`;
- the fraction of nodes within 1 sv and within 2 sv;
- the signed mean `d` (bias).

**5.4 Uncertainty.**
- Cluster bootstrap over **fibers**: all of a fiber's stretches and strips form one cluster.
- 2000 replicates (1000 for the median and p90), 95 % percentile intervals, seed 20261028.

**5.5 Baselines.** Both use the same nodes and the same distance.

- **(a) Frame plane.**
  - The u = 0 plane of each decision's frame, stitched exactly as the strip with villa's `stitch` and the same
    weights, from `strip.npz`.
  - This is the README's 6.89-vs-1.22 comparison.
- **(b) Lasagna-normal plane.**
  - **Construction.** The plane passes through the fiber point at the row's arclength and holds the fiber tangent.
    Its normal is the Lasagna normal there (the principal axis of the 8 trilinearly weighted level-4 normals), made
    perpendicular to the tangent. Nodes sit at the strip's column offsets.
  - **Availability.** Lasagna L2 normals for 20260102150214 were cheap to reach. They are precomputed now for every
    matched stretch:
    - `data/lasagna_normals_0139_primary.npz`;
    - `data/lasagna_normals_0139_secondary_run40.npz` (57 fibers, 1,586 chunks of 32³, about 5 minutes).

    Every stretch sample has a valid normal. So the baseline is kept, not dropped.
- **Comparison.**
  - A paired cluster bootstrap gives the baseline mean `|d|` minus the strip mean `|d|`, on nodes covered by both.
  - "Beaten" means the 95 % CI lower bound is above 0.
  - The within-2 difference is also reported.

**5.6 Fiber floor.** The human fiber's own `|d|` per scored row: how far the ground truth itself is from the mesh.

**5.7 Calibration of `expected_error`.**
- Node bins with edges [0, .5, .75, 1, 1.25, 1.5, 2, 3, ∞) sv. Per bin: the node count, the mean predicted error,
  and the mean and median observed `|d|`.
- Also: the node-weighted mean `|observed − predicted|`, the ratio of observed to predicted, and Spearman's rho at
  node and at fiber level.
- `expected_error` is the model's predicted error along its frame normal against Paris 4 patches; it is compared
  with the normal distance to the wrap mesh.

**5.8 Diagnostics.**
- each strip's centre distance to its fiber;
- the coverage of each surface;
- the angle between the Lasagna normal and the segment normal at the fiber.

## 6. Gates

**G1: is the ground truth adequate?** Decided on the fibers and segments alone.
- Pass needs all three:
  - at least **30 fibers** with a matched stretch;
  - **NF_match ≤ 2 sv**, which the selection guarantees;
  - **NF6 ≤ 2 sv**, the window-independent test.
- **Primary result: G1 FAILS.** 2 fibers. NF_match 0.78 sv and NF6 1.76 sv would pass. So there is **no PHerc0139
  accuracy claim from the primary set.** Its 2 strips will still be run and reported as case studies.

**G1' (at scoring time).** At least 30 fibers must actually be scored, each with at least 25 covered nodes.
Otherwise the outcome is the same as a G1 failure.

**G2: transfers or degrades** (only when G1 and G1' pass).
- **"transfers"** requires all three:
  - a pooled mean `|d|` of at most **1.5 sv**;
  - at least **70 %** of nodes within 2 sv;
  - **both** baselines beaten, with a paired-bootstrap CI excluding 0.
- **"degrades"** in every other case.
- A baseline that can't be computed (no `strip.npz`, or no normals) counts as not beaten.

**6.3 Secondary set (4.10).**
- G1 on its own numbers would pass: 57 fibers, NF_match 0.82, NF6 1.76.
- G1' needs 30 scored fibers out of the 30 in its run subsample, so a single fiber lost to a failed run or to
  coverage makes it fail.
- Its verdict is reported with the prefix "secondary:".

**Publication.** Whatever the outcome, we publish:
- the report JSON, this file with `FREEZE.txt`, the scorer, its tests and the plans, on GitHub under
  TAUIL-Abd-Elilah with a permissive licence;
- a short write-up, as a draft for the user to post. Claude never posts.

No result is withheld, and no run is repeated with changed settings.

## 7. PHerc0332: H/V crossing agreement (no ground truth)

**Crossings** (`results/crossings_0332_plan.json`):
- **Definition.** H and V fiber pairs whose closest approach is at most **6 sv** (half a winding gap, the NF6
  window). One crossing per pair, at the closest approach.
- **Why 6 sv, not 3.** v1 used 3 sv, which finds only 4 pairs among the 253 fibers (140 H, 113 V). For calibration,
  the counts by closest approach are:

  | Closest approach | ≤ 4 sv | ≤ 6 sv | ≤ 8 sv | ≤ 10 sv |
  |---|---|---|---|---|
  | Pairs | 13 | 48 | 120 | 202 |

- **Tight subset.** The crossings at ≤ 3 sv are reported separately as the tight subset.

**Windows.**
- Each crossing gets a 2.5 mm window on each fiber, centred on the crossing and shifted to fit inside the fiber.
- 45 of the 48 crossings have both windows.
- Pieces and heads are planned as in 4.6.

**Fixed subsample (4.8).**
- **Size:** 15 crossings: all 3 tight ones, then z terciles 7 / 3 / 5.
- **Inference:** 25 pieces, 30 windows, 24 fibers, 560 planned decisions (543 on the AFV geometry), 0.32 CPU-h.
- **CT:** 1,009 chunks, 1.97 GiB (1.0 h of download).
- **All 45 crossings would need** 1,873 decisions and 6.3 GiB.

**Agreement metric.**
1. For each of the two strips, take its nodes within 30 sv of the crossing point.
2. Project them onto the other strip's surface (5.1, unsigned) and take the median `|d|`.
3. The crossing's agreement is the mean of the two medians.

**Summary.**
- Over the planned crossings: mean, median, p90, the fraction at most 2 sv, and bootstrap CIs over crossings (2000
  replicates).
- The tight subset is reported apart.
- Planned crossings without both strips are listed as missing.
- The label is **"no ground truth: consistency, not accuracy"**.

**The same number on PHerc0139** (`score --crossings`) is computed on whatever H/V crossings (≤ 3 sv) the 0139
strips contain. That may be none, because the stretches are short and sparse.

**Limits.**
- Two strips can agree and both be wrong, for example both on the next winding.
- At 3 to 6 sv the two fibers may lie on opposite faces of one sheet, or on neighbouring sheets. The agreement then
  reflects the sheet thickness or a gap rather than an error, so the distribution, not only the mean, is reported.
- Fifteen crossings are few.

## 8. Known limits

- **Meshes are not the sheet.** Wrap meshes can sit on a face or the middle of the sheet, smooth over detail, or
  jump. Section 4.5 shows the human fibers and the meshes typically disagree by about 2 sv where they are near.
  NF6 is 1.76 sv, close to G2's 1.5 sv threshold, so a perfect strip would also show about 1 to 2 sv.
  - The fiber floor (5.6) and the paired baselines are the reference for that.
  - A constant offset shows up as bias and is not subtracted.
- **Human fiber noise.** ChaseHendrick's fiber_qa (#2248) found fold-backs, jumps and near-duplicates in the 0139
  fibers. The sensitivity results (4.9) report the unflagged stretches separately.
- **Selection bias.**
  - Only fibers on meshed sheets are scored: 45.9 % of fiber samples are even within the segments' z range, and
    meshed regions are the readable ones.
  - The secondary stretches are short (median 0.56 mm), so the score is local.
- **Pieces, not whole fibers.** The scored rows get the same decisions as a full run, within about 1 sv of head
  position. Kink repair on a cut piece can differ slightly from the full fiber's near the cut (4.6).
- **CPU and Windows.** CPU and GPU numerics differ by about 0.015 voxels on average (README). Windows and Linux
  aren't bit-identical: G0 reproduced published strips within 0.03 voxels, not exactly.
- **Assumptions in the budget.** The 30 s per-process overhead is assumed, and the download rate was measured once.
- **Units and metric.**
  - We measure normal distance to a wrap mesh; the README measures frame-normal height against verified patches.
  - Voxels are 9.596 um here against 9.6 um in the README.
  - The segment cell is 5 sv, with a facet sagitta of about 0.02 sv.

## 9. Credits

- **sean (bruniss):** the sheet-strip model, its training and inference code, and the Paris 4 strips (villa #2110).
  Also the fiber follower (#2045), the frame model and `match_fibers_to_patches.py`, whose ±2 sv window and
  `--min-run 40` are used here.
- **slchicoineau-creator and Baz248:** #2163, the native Windows port of the follower and the strip model.
- **modaribrahim:** `fiber-tracer-heldout-eval` (MIT), the held-out evaluation of fiber tracing on the 0139, 0332 and
  1451 human fibers, and the scoring fix in villa #2143.
- **ChaseHendrick:** #2248 fiber_qa (ground-truth noise), #2265, #2320.
- **The annotators** of the 0139 and 0332 human fiber scaffolds (lt, kb, dj).
- **The Vesuvius Challenge data team:** scans, wrap segments, Lasagna predictions, umbilici.
- **villa** is MIT-licensed. `orient_decisions`, `stitch` and `stitch_values` are copied from
  `sheet_strip/infer.py` at 6f2024569 with attribution.

## 10. History and deviations

**10.1 v1 (27 Oct 2026, 19:36; PREREG sha256 `e30daff6…`, `superseded_v1/`).**
- v1 selected whole fibers: at least 70 % of the controlled length in runs within 2 sv, runs of at least 40 sv, and
  a fiber of at least 2.5 mm.
- Its G1 run, made after the v1 freeze, found 0 matched fibers. 22 fibers had at least 70 % within 6 sv, with a
  noise floor of 1.78 sv (`superseded_v1/g1_v1_result.json`).
- These are ground-truth-only numbers, but they were seen before v2.

**10.2 Changes in v2.** All were decided after the v1 G1 numbers and the 4.5 diagnostics, and before any 0139 or
0332 strip was scored or inspected:
- the stretch rule (4.2), as specified for this build;
- NF6 added to G1, because NF_match is bounded by construction;
- the inference protocol on pieces with `--arc` (4.6);
- budget caps and the seeded subsample (4.7, 4.8);
- the secondary set with min-run 40 (4.10);
- the row rule widened from 10 to 30 sv, so that a strip on the wrong winding counts as an error instead of being
  dropped;
- the bootstrap cluster is the fiber over all its stretches;
- the 0332 crossing tolerance raised from 3 to 6 sv, with a tight subset (7).

**10.3 The one existing 0139 strip.**
- The setup run made a smoke-test strip on 27 Oct 2026 (`C:/VesuviusTemp/strips/runs/smoke0139`): fiber
  `dj_20260902T151037604_000001`, trace arc 500 to 644, to test the pipeline.
- Neither this build nor the setup run scored it or compared it with any segment. This build read only its layout
  arrays (`f`, `v`, `arcs`, `trace_per_patch`) to confirm the strip grid; no heights or positions.
- That fiber has no matched stretch in either set, so it can't enter scoring.

**10.4 Deviations after the freeze.** None at freeze time.

## 11. Files

| Path | What |
|---|---|
| `score_strips.py` | match, crossplan, lasagna, score, crossings (CPU only) |
| `tests/test_score_strips.py`, `tests/test_piece_afv.py` | synthetic tests, and integration tests on the real AFVs |
| `tools/download_inputs.py`, `tools/make_piece_afv.py` | data download; AFV and run-plan writer |
| `results/matched_stretches.json`, `data/matched_0139.afv` (+ `.tsv`), `results/runplan_0139_primary.tsv` | the primary set |
| `results/matched_stretches_secondary_run40.json`, `data/matched_0139_secondary_run40.afv`, `results/runplan_0139_secondary_run40.tsv` | the secondary set; run = `True` rows |
| `results/crossings_0332_plan.json`, `data/crossings_0332.afv`, `results/runplan_0332_crossings.tsv` | the PHerc0332 plan |
| `data/lasagna_normals_0139_*.npz` | the Lasagna normals for baseline (b) |

**Scoring commands** (CPU, `CUDA_VISIBLE_DEVICES=-1`):

```
python score_strips.py score --fibers data/fibers0139 --segments data/segments \
  --umbilicus data/umbilicus/PHerc0139_20260102150214_umbilicus.json \
  --matches results/matched_stretches.json --strips <OUT_primary> \
  --lasagna-normals data/lasagna_normals_0139_primary.npz --crossings --out <report.json>
# secondary: --matches results/matched_stretches_secondary_run40.json --run-only, with its normals file
python score_strips.py crossings --fibers data/fibers0332 --space PHerc0332/20251211183505@L0 \
  --plan results/crossings_0332_plan.json --strips <OUT_0332> --out <report_0332.json>
```
