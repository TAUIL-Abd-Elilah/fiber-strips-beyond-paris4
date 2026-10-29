# Pre-registration v3: `fiber_strip_small_v1` off Paris 4

**Accuracy on PHerc0139 against wrap segments, on the fiber stretches that lie on them, plus a consistency check
on PHerc0332 (no ground truth).**

- **Date:** 28 Oct 2026.
- **Status at freeze:** no PHerc0139 or PHerc0332 strip has been run against the v2 or v3 run plans, and none has
  been scored or inspected. One 0139 smoke-test strip exists (section 10.3); its fiber has no matched stretch, so it
  can't enter any scored set.
- **Freeze record:** `FREEZE.txt` holds the sha256 of this file, of `score_strips.py`, its tests, the matched sets,
  the full-run maps, the AFVs and run plans, the runner, and the data manifests. Every report from `score_strips.py`
  records the scorer's sha256 and this file's sha256. Any later change goes in section 10.6 with its date and reason.
- **Supersedes** v2, frozen on 28 Oct 2026 at 12:27 (`superseded_v2/`), which superseded v1 (`superseded_v1/`). An
  independent verifier failed v2 before any strip was run; what changed, and why, is in section 10.5.
- **Author:** TAUIL Abd Elilah. AI assistance: the scorer, its tests, the runner and this text were drafted with
  Claude (Anthropic) and checked by running the tests.

**Units.**
- **sv** ("L2 vox") is level 2 of the 2.4 um scans: 4 base voxels, which is 9.596 um on PHerc0139 and PHerc0332.
  It equals the strip model's "patch voxel" (Paris 4: 9.6 um).
- **Trace voxels.** The strip model walks a fiber in trace voxels (AFVFibers grid_scale 8): 8 base voxels, or 2 sv.

**Decision spacing.** One decision per 8 *trace* voxels: 16 sv, or 154 um, of fiber per decision (`--step 8`;
`infer_all`: `arange(12, L, 8)` in trace voxels, along AFVFibers' densified, kink-repaired geometry).

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
  - Strips were matched node by node on the same grid (`results/g0.json` of the setup run; corrections to the G0
    write-up are in `C:/VesuviusTemp/strips/results/g0_errata.md`).
  - The complete #2163 diff as applied, with its two new test files, is
    `patches/applied_pr2163_on_6f2024569_with_tests.diff`.
- **Checkpoint.** `scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e`, with its frame model
  (`heading_model.pt`) from the same commit.
- **Runs.** One `sheet_strip.infer` process per AFV piece, one process at a time, on the CPU
  (`CUDA_VISIBLE_DEVICES=-1`, `--device cpu`; the wrapper `runs/v3/infer_cpu.py` exits if torch sees a GPU),
  BelowNormal priority, 8 torch threads (G0 used 16; this machine's other work needs the rest). It uses villa's
  default inference settings, as G0, and G0's batch of 32 (infer.py's own default is 16):

  ```
  python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer --afv <AFV> --fiber <afv_id> \
    --arc <arc_start> <arc_end> --out <OUT>/<piece id> --checkpoint scrollprize/fiber_strip_small_v1@7bd163a2... \
    --ct <volume zarr> --umbilicus <level-2 umbilicus> --cache-dir C:/VesuviusTemp/strips/ct_cache \
    --device cpu --index none --no-render --step 8 --blend 16 --trim 75 --batch 32
  ```

  - `infer.py` writes `strip/` (tifxyz) and `strip.npz`, which holds the decisions the frame-plane baseline needs.
  - `--afv`, `--fiber` and `--arc` come from the run plans (`results/runplan_*.tsv`).
  - The volume and umbilicus for each scroll:

    | Scroll | `--ct` | `--umbilicus` |
    |---|---|---|
    | PHerc0139 | `s3://vesuvius-challenge-open-data/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr` | `C:/VesuviusTemp/strips/data/umbilicus/PHerc0139_20260102150214_umbilicus_L2.json` |
    | PHerc0332 | `s3://vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr` | `C:/VesuviusTemp/strips/data/umbilicus/PHerc0332_20251211183505_umbilicus_L2.json` |
- **Pieces, not whole fibers.** A full run of all 0139 fibers would need about 48 CPU-h at the loaded rate (4.7) and
  183.0 GiB of CT at the r = 170 footprint (157 GiB is the r = 120 figure; see the G0 errata), so only the matched
  stretches are run. Section 4.6 says how far a piece with `--arc` reproduces the decisions of a full run.
- **No tuning.** No inference setting, selection rule, threshold, subsample or metric changes after any 0139 or
  0332 strip has been seen.

## 3. Data and frames

| Item | Source (fetched 28 Oct 2026) | Frame / units |
|---|---|---|
| PHerc0139 human fibers | `dl.ash2txt.org/other/annotators/0139_fibers/`: 620 JSON, 235,298,374 bytes, in `data/fibers0139/` (`data/fibers0139_manifest.json`: sha256 per file) | VC3D fiber JSON v3/v4, `line_points` xyz, level 0 of `PHerc0139/20260102150214` (2.399 um) |
| PHerc0139 wrap segments | S3 `PHerc0139/segments/*/mesh/*-on-20260102150214-2.399um.tifxyz`. 39 segment dirs, 38 with such a mesh (w023 to w059 and the title); `20260130000001-w030_202601301912` has none. 152 files, 935,849,599 bytes (892.5 MiB), in `data/segments/` (`data/segments_manifest.json`) | tifxyz xyz in level-0 voxels, grid step 20 base voxels (5 sv) |
| PHerc0332 human fibers | `dl.ash2txt.org/other/annotators/0332_fibers/`: 255 JSON, 88,068,432 bytes, downloaded by the setup run on 27 Oct 2026, copied to `data/fibers0332/` (`data/fibers0332_manifest.json`) | level 0 of `PHerc0332/20251211183505`; the JSONs carry no coordinate space, so `PHerc0332/20251211183505@L0` is used |
| Full AFVs (the full run's geometry) | the setup run's `C:/VesuviusTemp/strips/data/human_afv/PHerc0139_human_controlled.afv` (615 fibers) and `PHerc0332_human_controlled.afv` (253), written by `tools/fibers_json_to_afv.py` from the same JSONs (controlled span) | native level-0 xyz; AFVFibers grid_scale 8 |
| Lasagna normals (0139) | S3 `PHerc0139/representations/predictions/lasagna/20260102150214-lasagna-20260419180421-L2/..._nx/_ny.ome.zarr/4` | uint8 hemisphere encoding, 16 base voxels per voxel |
| CT | `.../20260102150214-...zarr/2`: shape (19239, 6628, 6628), 128³ uint8 chunks of 2 MiB. 0332: (8398, 3941, 3941), the same chunks | level 2 = sv |

- **Unusable fibers.** 5 of the 620 0139 fibers have fewer than 2 control points, so no controlled span: 615
  remain. 2 of the 255 0332 fibers likewise: 253 remain.
- **H/V tags.** Every remaining fiber has a manual or automatic H/V tag.
- **Coverage.** Only 45.9 % of the 0139 fiber samples lie inside the z range of any wrap segment.
- **Identity of JSONs without a coordinate space.** The scorer derives such a fiber's uuid from `--space`, as the
  converter derives it from its volume. v3 passes `--space PHerc0139/20260102150214@L0` to `match` (v2 passed none).
  This changes the AFV id of one 0139 fiber, `kb_20260827T190132947_000015`, which has no matched stretch; every
  matched fiber keeps its id, and every 0139 id now equals the full AFV's.

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

**2 stretches on 2 fibers, both H, 7.42 mm in total** (the same as v2):

| Stretch | Fiber | Segment | Length | mean `d` | median `\|d\|` | p90 `\|d\|` |
|---|---|---|---|---|---|---|
| S0000 | `kb_20260904T021448778_000085` | w044 | 3.73 mm | +0.47 | 0.92 | 1.69 sv |
| S0001 | `lt_20260918T092404850_000160` | w054 | 3.68 mm | +0.13 | 0.61 | 1.23 sv |

- **Inference.** 2 pieces, 58 decisions, 0.02 to 0.03 CPU-h (4.7). CT: 140 level-2 chunks within r = 170 of the
  decision path (0.27 GiB), about 0.13 h of download. Nothing is cached yet.
- **G1 fails** at this stage: 2 fibers, against at least 30 required (section 6).

### 4.4 Noise floors

- **NF_match** = median `|d|` over all samples of the matched stretches.
  - Primary: **0.78 sv**. Secondary (4.10): 0.82 sv.
  - It can't exceed 2 sv, because the selection bounds it.
- **NF6** = median `|d|` over all samples of the *near runs*: runs of consecutive samples on one segment with
  `|d| <= 6` sv (half a winding gap), at least 2.5 mm long. NF6 doesn't depend on the 2 sv window.
  - Value: **1.76 sv**, over 287 near runs.

### 4.5 Ground-truth diagnostics

These numbers come from the fibers and segments only. They were seen before v2 was written; they explain the
primary count and motivated 4.10.

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

**The full run.** A full `infer_all` run opens each fiber from the full AFV with AFVFibers (every segment densified
to at most 1 trace voxel, then `repair_kinks`) and puts its decision heads at `arange(12, L + 1e-9, 8)` trace
voxels of that repaired geometry: **S_k = 24 + 16k sv of the repaired arclength**. Kink repair shortens a fiber, so
S differs from the raw JSON arclength s the stretches are measured in (by up to several sv, accumulated upstream).
v2 planned heads at 24 + 16k sv of the raw arclength, which is not the full run's grid (10.5, P2).

**The full-run map** (`tools/full_run_map.py`, `data/full_run_map_0139.npz`, `data/full_run_map_0332.npz`).
`repair_kinks` keeps the vertex count of the densified line and moves only vertices inside a repaired run, so
densified vertex i before and after repair is the same annotation point. This gives the exact map s ↔ S at every
vertex. The tool recomputes the geometry with villa's own functions and checks it against AFVFibers vertex for
vertex. 313 of the 615 0139 fibers and 133 of the 253 0332 fibers have kink repairs.

**Decision heads.**
- A row takes the decisions whose heads are within 32 sv of it (`--blend` 16 trace voxels, in the run's arclength).
- For each window [s0, s1] (a stretch, or a 0332 crossing window), the planned heads are the full run's heads S_k
  within 40 sv of [S(s0), S(s1)], clipped to the repaired fiber (`head_k`: first and last k).
- So each window gets every head index k that a full run would give its rows: the test
  `test_pieces_hold_every_full_run_head_within_the_blend_of_their_windows` checks this for every window of every
  plan.

**Pieces.**
- **Extent.** The fiber is kept from 300 sv before the first head to 160 sv after the last head (raw arclength of
  those heads):
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

**`--arc` values (P1).** `tools/make_piece_afv.py` opens the piece AFV as the model does. It places the piece's first
head (full-run head k0) on the piece's own repaired geometry through the piece's own exact raw ↔ repaired vertex
map, and writes `--arc START END` (patch voxels) with END = START + 16 (n − 1) + 1 sv. infer.py's own head code,
`lo, hi = max(START/2, 12), min(END/2, length)` and `arange(lo, hi + 1e-9, 8)`, then makes exactly the n planned
decisions: the 1 sv pad is half a trace voxel, far under one step. The test
`test_infer_py_makes_exactly_the_planned_heads_on_the_full_run_grid` runs villa's `infer.py` `main()` unchanged on
every run-plan row, up to the model call (the model is replaced by a stand-in carrying the checkpoint's own
`vol_spec`), and requires exact equality for every piece of every plan. (v2 snapped both ends to the nearest
geometry sample and lost the last head in 23 of the 39 secondary run pieces and 17 of the 25 0332 pieces.)

| Set | Pieces | Decisions (plan = infer.py) | Kink-repaired pieces |
|---|---|---|---|
| Primary | 2 | 58 | 0 |
| Secondary, all matched | 182 | 12,725 | 46 |
| Secondary, run subsample (4.10) | 46 | 3,513 | |
| PHerc0332 | 25 | 558 | 9 |

**Where it differs from a full run (P2).** Each planned head is compared with the full run's head k (3-D distance
between the piece's and the full fiber's repaired geometries):
- **Within 1 sv:** both primary pieces (max 3e-7 sv), 179 of the 182 secondary pieces (max 0.67 sv) and 23 of the
  25 PHerc0332 pieces (max 0.38 sv).
- **Not within 1 sv, in 5 pieces.** AFVFibers' kink repair of the cut piece differs from the full fiber's at one
  place inside the head span: a repaired run starts or ends a little elsewhere, or is found in one but not the
  other. `repair_kinks` resamples the whole line on a unit grid and repeats up to 4 passes, so its result near a
  borderline kink depends on the rest of the line. The heads after that place are offset along the fiber, and the
  observed path differs there by the repair difference:

  | Piece | In the run set | Heads | Max head offset | Where |
  |---|---|---|---|---|
  | P0047 (secondary) | no | 135 | 3.49 sv | heads 49 to 135 offset 1.2 to 3.5 sv |
  | P0104 (secondary) | **yes** | 187 | 1.01 sv | one head at 1.01 sv; all others ≤ 0.23 sv |
  | P0114 (secondary) | no | 198 | 7.81 sv | heads 108 to 198 offset 3.6 to 7.8 sv |
  | C0001 (0332, window X0018V) | **yes** | 21 | 5.73 sv | heads 13 to 21 offset 3.2 to 5.7 sv |
  | C0004 (0332, window X0008H) | **yes** | 21 | 1.26 sv | heads 11 to 21 offset 0.9 to 1.3 sv |

  The test pins this list: every other piece must be within 1 sv, and these five must show these offsets. In the
  run sets this concerns P0104 (practically at 1 sv) and two of the 30 PHerc0332 crossing windows, X0018V and X0008H.
  No primary piece is affected.
- So the claim is: **each scored row gets every decision index a full run would give it; the head positions match
  the full run's within 1 sv except in the five pieces above.** Inputs other than the head positions (the observed
  path, the crop axis) also differ from a full run's only where the piece's kink repair differs from the full
  fiber's.
- All run fibers are longer than 2.5 mm, so `infer_all` would run each of them.

### 4.7 Budget

- **CPU (A2).** Throughput depends on the load from other work on this machine:
  - **idle machine: 3.27 to 3.56 decisions/s** (an independent rerun of G0's inputs);
  - **1.35 decisions/s** under another session's CPU load (the setup run's measurement). The caps use this rate.
  - A heavier load is slower: another session's jobs had the CPU at 100 % again on 28 Oct, and this run has
    BelowNormal priority.
  - Plus an *assumed* 30 s per `infer.py` process (model, frame model, CT store; not measured).
  - CPU time is therefore given as a range: idle (3.56 to 3.27 decisions/s) to loaded (1.35 decisions/s).
- **CT.** The union of the level-2 chunks (2 MiB each, uncompressed) within **r = 170** level-2 voxels of each
  piece's decision path (raw fiber path between its first and last head, sampled every 1 voxel).
- **Download time.** At the measured 0.58 MiB/s.

### 4.8 Seeded subsample rule (applies when a set exceeds its caps)

**Caps.**
- PHerc0139 primary: **8 CPU-h** and **8 GiB of new CT**.
- PHerc0139 secondary: 8 CPU-h and the CT cap of amendment A1 below.
- PHerc0332: 2 CPU-h, 2 GiB and at most 60 crossings.
- C: had about 28 GB free on 28 Oct and must keep 10 GiB (the runner stops below that).

**Rule.**
1. **Units.** The units are pieces; for PHerc0332, crossings.
2. **Strata.** Units are grouped into strata: (segment of the piece's longest stretch, family). For PHerc0332:
   (tight crossing yes/no, z tercile).
3. **Order.** Strata are taken in sorted key order. Within a stratum, units are sorted by id and permuted by
   `numpy.random.default_rng(20261028)`, drawn stratum by stratum in that order.
4. **Selection.** Round robin over the strata: each turn offers a stratum's next unit, preferring units of fibers
   not yet chosen. For PHerc0332 this preference has no effect: each crossing unit's fiber key is its crossing id,
   so no two units share it.
5. **Caps.** A unit is taken if the cumulative CPU hours (at 1.35 decisions/s) and new CT stay within the caps.
   Otherwise it is passed over for good.

This is `stratified_subsample` in the scorer. The primary set is within its caps, so it isn't subsampled.

**Amendment A1 (v3; decided before any strip was run).** With a CT cap of 8 GiB the secondary run subsample had
exactly 30 fibers in v2 (31 with the v3 heads), and G1' needs at least 30 scored fibers: losing a single fiber to a
failed run or to coverage would fail the gate. Only the 0139 secondary CT cap is raised: it is **the smallest cap
on the grid 8.0, 8.1, ..., 13.0 GiB whose subsample (same seeded, stratified round robin) holds at least 36
fibers**, or 13 GiB if none does (`ct_cap_scan`, `match --ct-cap-scan 8 13 0.1 --min-run-fibers 36`). The scan is
recorded in the plan (`plan.caps.ct_cap_scan`). Fibers by cap: 31 at 8.0 GiB, 28 to 33 from 8.1 to 8.9, 34 from 9.0
to 9.3, 35 at 9.4 and 9.5, **36 at 9.6 GiB** (the count is not monotone in the cap, because the greedy rule passes
units over for good). The cap is therefore **9.6 GiB**. The primary and PHerc0332 sets are unchanged (the same two
stretches; the same 15 crossings in the same order).

### 4.9 Sensitivity results (reported, not gating)

- unflagged stretches only, without the #2248-style fold-backs or jumps flagged in the stretch;
- per annotator (lt / kb / dj);
- per family.

### 4.10 Secondary set (pre-registered; labelled, never reported as the primary result)

**Rule.** The same as 4.2, but the minimum stretch is **40 sv** (0.384 mm). This is the `--min-run 40` with which
sean matched fibers to patches for this model's own training and validation (README, "Fibers on patches"). It was
chosen after the primary count (2 fibers) was known, from the fibers and segments only, before any strip.

Results in `results/matched_stretches_secondary_run40.json`:

**All matched** (stretches unchanged from v2):
- **Stretches:** 57 fibers (25 H, 32 V), 1,075 stretches, 710.5 mm in total; median stretch 58 sv, p90 108 sv.
- **Overlaps:** 96 overlaps dropped.
- **Noise floors:** NF_match 0.82 sv, NF6 1.76 sv.
- **Inference:** 182 pieces, 12,725 decisions, 2.5 to 2.6 CPU-h idle, 4.1 CPU-h at the loaded rate.
- **CT:** 12,479 chunks, **24.4 GiB** (11.9 h of download). This is over the CT cap, so the set is subsampled.

**Run subsample (4.8 with A1, cap 9.6 GiB), `run: true`:**
- **Stretches:** 46 pieces, 284 stretches, **36 fibers** (14 H with 215 stretches, 22 V with 69), on 30
  segments, 198.1 mm in total.
- **Inference:** 3,513 decisions, 0.66 to 0.68 CPU-h idle, 1.11 CPU-h at the loaded rate.
- **CT:** 4,914 chunks, 9.60 GiB (4.7 h of download).

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
  - **Rule in the code, stated here (P3 d).** A row gets a Lasagna-plane row only if a fiber sample with a valid
    Lasagna normal lies within **10 sv** of the row's arclength (the nearest valid sample is used); otherwise that
    row has no baseline nodes.
  - **Availability.** Lasagna L2 normals for 20260102150214 were cheap to reach. They are precomputed for every
    matched stretch (±30 sv):
    - `data/lasagna_normals_0139_primary.npz`;
    - `data/lasagna_normals_0139_secondary_run40.npz` (57 fibers, 1,586 chunks of 32³, about 5 minutes).

    Every stretch sample has a valid normal. So the baseline is kept, not dropped.
- **Comparison.**
  - A paired cluster bootstrap gives the baseline mean `|d|` minus the strip mean `|d|`, on nodes covered by both.
  - "Beaten" means the 95 % CI lower bound is above 0.
  - The within-2 difference is also reported.

**5.6 Fiber floor.** The human fiber's own `|d|` per scored row: how far the ground truth itself is from the mesh.

**5.7 Calibration of `expected_error`.**
- Node bins with edges [0, .5, .75, 1, 1.25, 1.5, 2, 3, ∞) sv of predicted error. Per bin: the node count, the mean
  predicted error, and the mean and median observed `|d|` (and the fraction within 2 sv).
- **The summary gap (P3 c).** `weighted_abs_gap_sv` = Σ over bins of (bin nodes) × |bin mean observed − bin mean
  predicted| / all nodes. It is a difference of bin means weighted by the bins' node counts (an expected-calibration-
  error style number), not a per-node mean of `|observed − predicted|`.
- Also: the ratio of mean observed to mean predicted, and Spearman's rho at node and at fiber level (fibers with at
  least 25 nodes).
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
- G1 on its own numbers passes: 57 fibers, NF_match 0.82, NF6 1.76.
- G1' needs 30 scored fibers out of the **36** in its run subsample (A1): up to 6 fibers can be lost to failed runs
  or coverage before the gate fails.
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
- Pieces and heads are planned as in 4.6, on the full PHerc0332 AFV's grid.

**Fixed subsample (4.8).**
- **Size (P3 a):** 15 crossings in all, by z tercile **7 / 3 / 5**. These 15 include the 3 tight ones, which the
  round robin offers first (their stratum key sorts first); the 7 / 3 / 5 already count them.
- **The same 15 crossings, in the same order, as v2.**
- **Inference:** 25 pieces, 30 windows, 24 fibers, 558 decisions (plan = infer.py), 0.25 to 0.26 CPU-h idle, 0.32
  CPU-h at the loaded rate.
- **CT:** 1,009 chunks, 1.97 GiB (1.0 h of download).
- **All 45 crossings would need** 1,867 decisions and 6.3 GiB.
- Two of the 30 windows lie in the pieces with head offsets (4.6): X0018V (C0001, up to 5.7 sv) and X0008H (C0004,
  up to 1.3 sv).

**Agreement metric.**
1. For each of the two strips, take its nodes within 30 sv of the crossing point.
2. Project them onto the other strip's surface (5.1, unsigned) and take the median `|d|`.
3. The crossing's agreement is the mean of the two medians.
- **Rule in the code, stated here (P3 d).** A direction (one strip onto the other) is skipped when the first strip
  has fewer than **5** nodes within 30 sv of the crossing, or fewer than **5** of them are covered by the other
  strip. A crossing with one usable direction uses that one median; with none it is listed as missing.

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
- **Pieces, not whole fibers.** Every scored row gets every decision index a full run would give it. The head
  positions match the full run's within 1 sv in every piece except the five of 4.6 (in the run sets: P0104 at
  1.01 sv, and the 0332 windows X0018V and X0008H at up to 5.7 and 1.3 sv), where the cut piece's kink repair
  differs from the full fiber's. The observed path and crop axis differ from a full run's only where the kink
  repair differs.
- **CPU and Windows.** CPU and GPU numerics differ by about 0.015 voxels on average (README). Windows and Linux
  aren't bit-identical: G0 reproduced published strips within 0.03 voxels, not exactly. The runs use 8 torch
  threads with G0's batch of 32; G0's strips were made with 16 threads (its 8-thread runs were timed, not compared
  node by node).
- **Assumptions in the budget.** The 30 s per-process overhead is assumed, the download rate was measured once, and
  the CPU rate depends on the other session's load (4.7).
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
  `sheet_strip/infer.py` at 6f2024569 with attribution; `tools/full_run_map.py` and `tools/make_piece_afv.py` call
  villa's `dense_line`, `repair_kinks` and `AFVFibers` to reproduce the full run's geometry.

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
- Neither this build nor the setup run scored it or compared it with any segment. The v2 build read only its layout
  arrays (`f`, `v`, `arcs`, `trace_per_patch`) to confirm the strip grid; no heights or positions. The v3 build did
  not open it.
- That fiber has no matched stretch in either set, so it can't enter scoring.

**10.4 v2 (28 Oct 2026, 12:27; PREREG sha256 `ac91035f…`, `superseded_v2/`).** v2 was frozen and then failed by an
independent verifier, before any strip was run against its plans. `superseded_v2/ARCHIVE.txt` lists its files, each
checked against the v2 FREEZE sha256. (The v2 scorer had already been edited for v3 when it was archived; its v2
text was rebuilt by undoing those edits exactly and matches the frozen sha256.)

**10.5 Changes in v3.** All decided on 28 Oct 2026 from the verifier's report, from plan geometry and from ground
truth only, before any PHerc0139 or PHerc0332 strip was run against a plan:
- **P1, the last head was dropped.** v2 snapped each piece's first and last head to the nearest geometry sample;
  when that made the `--arc` span shorter than 16 (n − 1) sv, `infer.py` dropped the last head (2,908 of 2,932
  secondary run decisions; 543 of 560 for 0332; 121 over all 181 secondary pieces). v3 writes END = START +
  16 (n − 1) + 1 sv, and the test now requires exact equality with infer.py's own head code (4.6) instead of
  tolerating max(2, 5 %).
- **P2, heads now on the full run's grid.** v2 planned 24 + 16k sv of the raw JSON arclength; a full run uses the
  kink-repaired AFVFibers arclength. 24 of the 30 v2 secondary run fibers had repairs upstream, 25 of 78 head
  endpoints were off by more than 2 sv (up to 7.5 sv), and primary P0000 by about 6 sv. v3 plans on the full run's
  own grid through the full-run map (4.6). Primary P0000's heads are now within 3e-7 sv of the full run's. The v2
  claim "within about 1 sv" is corrected: it holds for all but the five pieces named in 4.6.
- **P3, wording.** (a) the 15 crossings are 7 / 3 / 5 by tercile including the 3 tight ones (7); (b) the
  fiber preference has no effect for PHerc0332 (4.8); (c) the calibration gap is described as computed (5.7);
  (d) two rules in the code are now stated: the 5-node rule of the crossing agreement (7) and the 10 sv rule of the
  Lasagna-plane rows (5.5).
- **A1, slack for G1'.** The secondary CT cap is raised from 8 GiB to the smallest cap on a 0.1 GiB grid up to
  13 GiB giving at least 36 fibers: 9.6 GiB, 36 fibers (4.8). Decided before any strip was run, so that one lost
  fiber cannot fail G1' by itself.
- **A2, budget.** CPU time is given as a load-dependent range (4.7), after an idle rerun measured 3.27 to 3.56
  decisions/s against the 1.35 measured under load.
- **Identity fix.** `match` gets `--space PHerc0139/20260102150214@L0` (3), which changes only one unmatched fiber's
  id.
- **New pieces of the plan.** With the heads moved, the secondary set has 182 pieces (v2: 181) and 12,725 decisions;
  primary P0000 moved by about 6 sv and its line range with it; the PHerc0332 plan has 558 decisions (v2: 560
  planned). The matched stretches, G1 numbers and noise floors are unchanged.
- **Runner.** `runs/v3/run_all.ps1` (with `prefetch_v3.py` and `infer_cpu.py`) is frozen with the plan (11).
- **G0 errata.** `C:/VesuviusTemp/strips/results/g0_errata.md` corrects the G0 write-up (CT 183.0 / 36.4 GiB at
  r = 170; the AFV is 147,456 B; the complete #2163 diff with its two test files; "4 threads 1.47" was 16 threads
  with `--crop-threads 4`; RAM peak about 4.1 GB). `g0.json` is unchanged.

**10.6 Deviations after the freeze.** None at freeze time.

## 11. Files

| Path | What |
|---|---|
| `score_strips.py` | match, crossplan, lasagna, score, crossings (CPU only) |
| `tests/test_score_strips.py`, `tests/test_piece_afv.py` | synthetic tests, and integration tests on the real AFVs, plans and villa's infer.py |
| `tools/full_run_map.py`, `data/full_run_map_0139.npz`, `data/full_run_map_0332.npz` | the full run's arclength map (4.6) |
| `tools/download_inputs.py`, `tools/make_piece_afv.py` | data download; AFV and run-plan writer |
| `results/matched_stretches.json`, `data/matched_0139.afv` (+ `.tsv`), `results/runplan_0139_primary.tsv` | the primary set |
| `results/matched_stretches_secondary_run40.json`, `data/matched_0139_secondary_run40.afv`, `results/runplan_0139_secondary_run40.tsv` | the secondary set; run = `True` rows |
| `results/crossings_0332_plan.json`, `data/crossings_0332.afv`, `results/runplan_0332_crossings.tsv` | the PHerc0332 plan |
| `data/lasagna_normals_0139_*.npz` | the Lasagna normals for baseline (b) |
| `runs/plan_v3/run_match.sh`, `runs/plan_v3/run_afv.sh` | the commands that wrote the plans, AFVs and run plans |
| `runs/v3/run_all.ps1`, `runs/v3/prefetch_v3.py`, `runs/v3/infer_cpu.py` | the detached runner: prefetch, inference, scoring |
| `tools/archive_v2.py`, `superseded_v2/`, `superseded_v1/` | earlier versions |

**Planning commands** (CPU; `runs/plan_v3/run_match.sh`): `match ... --space PHerc0139/20260102150214@L0
--full-run-map data/full_run_map_0139.npz` (secondary also `--min-stretch-mm 0.38384 --analysis secondary
--ct-cap-scan 8 13 0.1 --min-run-fibers 36`); `crossplan ... --full-run-map data/full_run_map_0332.npz`; then
`make_piece_afv.py ... --full-afv <the full AFV>` (`run_afv.sh`).

**Scoring commands** (CPU, `CUDA_VISIBLE_DEVICES=-1`; run by `runs/v3/run_all.ps1`, reports in `results/v3/`):

```
python score_strips.py score --fibers data/fibers0139 --segments data/segments \
  --umbilicus data/umbilicus/PHerc0139_20260102150214_umbilicus.json --crossings \
  --matches results/matched_stretches.json --strips runs/v3/strips/primary \
  --lasagna-normals data/lasagna_normals_0139_primary.npz --out results/v3/primary_case_studies.json
# secondary: --matches results/matched_stretches_secondary_run40.json --run-only --strips runs/v3/strips/secondary
#   --lasagna-normals data/lasagna_normals_0139_secondary_run40.npz --out results/v3/secondary_report.json
python score_strips.py crossings --fibers data/fibers0332 --space PHerc0332/20251211183505@L0 \
  --plan results/crossings_0332_plan.json --strips runs/v3/strips/0332 --out results/v3/crossings_0332_report.json
```
