# fiber_strip_small_v1 beyond Paris 4

A pre-registered CPU evaluation of sean's (bruniss) sheet-strip model `fiber_strip_small_v1`
([villa #2110](https://github.com/ScrollPrize/villa/pull/2110)) on two scrolls it was not trained on:
PHerc0139, scored against its wrap segments, and PHerc0332, checked for consistency at fiber crossings.

**Status, 29 Oct 2026.** Pre-registration v3.2 was frozen on 29 Oct 2026 at 02:10:33 UTC
(`score/PREREG.md`, `score/FREEZE.txt`). It is the second erratum to v3 (frozen 28 Oct, 14:04 UTC;
`score/superseded_v3/`), after v3.1 (29 Oct, 00:34 UTC; `score/superseded_v31/`). Both errata were written before
any PHerc0139 or PHerc0332 strip of the plans was made, and neither changes the plans, the scorer or the runner. The
CT of all three run sets is downloaded, and the scored run started on 29 Oct 2026 at 02:13 UTC. This commit is
public before that run's results exist. Every number written `{{RESULT_...}}` below will be filled from the frozen
scorer's reports when they exist. Nothing below was tuned after a strip was seen.

- PREREG v3.2 sha256 `dffa97b9dffb7cd9b69d6cdc66fa247f00a57de6a208db71fdc098034b59812e`
- FREEZE.txt sha256 `d2bf1f7b7612bee6b22925d3de123a65ec792642bfa0fd8980ac8dfda51454be` (v3.1: `7c8be79970c1a98d3b097c0988ea0b53d7a5354379e78f7f5e11abc851dbd559`,
  kept as `score/superseded_v31/FREEZE_v31.txt`; v3: `5309debbbeeb6ed7401badbbd163809b9c4632838c71f7eded8945d0783db8d5`,
  kept as `score/superseded_v3/FREEZE_v3.txt`)
- Check them: `python verify_freeze.py` (see [How to reproduce](#how-to-reproduce)).

**Units.** sv = one level-2 voxel of the 2.4 µm scans = 4 base voxels = 9.596 µm on PHerc0139 and PHerc0332
(9.6 µm on Paris 4). It is the strip model's "patch voxel".

## Question

**Primary (PHerc0139).** Run unchanged on human-traced fibers of PHerc0139, does `fiber_strip_small_v1` place the
sheet where the PHerc0139 wrap segments are? The segments are meshed on the same 2.399 µm scan as the fibers. A
strip is scored only where the human fiber itself lies on a segment.

**Secondary (PHerc0332, no ground truth).** Where an H fiber and a V fiber cross, do their two strips agree?

**Reference on Paris 4** (villa `tracers/fibers/README.md` at 6f2024569): validation error 1.006 / 1.117 voxels
(full / observed tube). On validation fiber 469373: 1.22 voxels mean error against 6.89 for the frame plane, with
82 % of points within 2 voxels.

## Why

Paul's progress update of 26 Oct 2026 (our paraphrase): strips exist only for Paris 4 and have only been scored
there; the team is done with Paris 4 for now; the active work is on the human fiber scaffolds of PHerc0139,
PHerc0332 and PHerc1451.

So before strips are used on those scaffolds, it helps to know how well the model does off Paris 4. PHerc0139 has
38 wrap segments meshed on the same volume as its 620 human fibers (39 segment folders; one has no mesh on that
volume), so it gives an independent reference.
PHerc0332 has no segments; there we can only check that strips agree with each other.

On 28 Oct 2026 the organizers' `spiral_datasets` folders for PHerc0139 and PHerc0332 held fiber tracks only, no
strips.

## Design

Everything here is fixed in `score/PREREG.md` (v3.2) and was frozen before any PHerc0139 or PHerc0332 strip was
run against the plans. Section numbers below refer to it.

### Model and inference (PREREG 2)

- **Code:** villa 6f2024569, plus #2163 (the Windows port by slchicoineau-creator and Baz248, applied unchanged),
  plus our 20-line normalization patch for scrolls other than Paris 4 (`patches/`). Since v3.1, `FREEZE.txt` pins
  the sha256 of every file of the worktree that differs from 6f2024569 (9 files), of the patches and of the
  checkpoint files, and the runner's freeze check also requires the worktree to be at 6f2024569 with exactly those
  files changed. Since v3.2 it also pins the relaunch wrapper that writes the run's final status
  (`score/runs/relaunch_v3.ps1`).
- **Checkpoint:** `scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e` with its frame model
  `heading_model.pt`. Not redistributed; `fetch_inputs.py checkpoint` downloads it and checks its sha256. The pins
  were checked against the Hugging Face API at that revision (LFS sha256 of the two `.pt` files, git blob ids of
  `config.json` and `run.json`; PREREG 10.8 item 7).
- **Runs:** one `sheet_strip.infer` process per piece, one at a time, CPU only (`CUDA_VISIBLE_DEVICES=-1`,
  `--device cpu`), 8 torch threads, batch 32, villa's default settings (`--step 8 --blend 16 --trim 75`).
- **Pieces, not whole fibers.** A full run of all 0139 fibers would need about 48 CPU-h and 183 GiB of CT, so only
  the matched stretches are run, each as a piece of its fiber with `--arc`. The decision heads are placed on the
  full run's own (kink-repaired) arclength grid. A test runs villa's `infer.py` head code on every run-plan row and
  requires exactly the planned decisions. Head positions match a full run's within 1 sv in every piece except
  five; three of those are in the run sets (P0104 at 1.01 sv; 0332 pieces C0001 at up to 5.7 sv and C0004 at up to
  1.3 sv) (PREREG 4.6).
- **Is a piece run the same as a full run? (v3.1 and v3.2, PREREG 10.7 and 10.8.)** Tested on Paris 4: three G0
  fibers cut into pieces the same way and run with the same `infer.py` command. On the rows a stretch would score,
  the piece strip equals the full-fiber strip within 0.001 sv (max over 425 nodes); rows at the piece's ends, which
  take decisions from outside the piece and are never scored, differ by up to 2.9 sv. Not quite everything else is
  equal, though (v3.2): the first decision of a piece that starts mid-fiber differed from the full run's (heights by
  up to 0.42 sv, cause not identified). It can reach at most the first one or two rows of a window, with at most
  about 7 % of their weight (0.0085 sv on the Paris 4 row tested), in 2 of 2 primary, 17 of 46 secondary and 8 of 25
  PHerc0332 pieces. One input is not covered by that test: `infer.py`
  also gives the model every fiber point inside the crop, so a part of the fiber outside the piece that loops back
  into the crop changes the input. From fiber geometry, this can happen for 121 of the 2,207 secondary heads that
  feed scored rows (6 pieces) and 112 of the 531 PHerc0332 ones (6 pieces), and for none in the primary set. Its
  effect is not measured; the affected heads are listed in `score/results/erratum_v31/geometry_checks.json`.
- **The scorer on real output (v3.1).** It had only read synthetic strips. On six real Paris 4 `infer.py` outputs it
  reproduces 100 % of the strip nodes from the saved decisions, so the frame-plane baseline is available, and it
  scores them end to end. One of them is now a test fixture (`score/tests/fixtures/paris4_real`).

### Sets

| Set | Rule | Fibers | Stretches or windows | Pieces | Decisions | New CT |
|---|---|---|---|---|---|---|
| 0139 primary | on one segment, \|d\| ≤ 2 sv, ≥ 2.5 mm | 2 (both H) | 2 stretches, 7.42 mm | 2 | 58 | 0.27 GiB |
| 0139 secondary, all matched | as primary, ≥ 40 sv (0.384 mm) | 57 (25 H, 32 V) | 1,075 stretches, 710.5 mm | 182 | 12,725 | 24.3 GiB |
| 0139 secondary, run subsample | seeded subsample, CT cap 9.6 GiB | 36 (14 H, 22 V) | 284 stretches on 30 segments, 198.1 mm | 46 | 3,513 | 9.60 GiB |
| 0332 crossings | H/V pairs within 6 sv; 48 found, 45 with both windows | 24 | 15 crossings (7 / 3 / 5 by z tercile, incl. 3 tight ≤ 3 sv), 30 windows | 25 | 558 | 1.97 GiB |

Sources: `score/results/matched_stretches.json`, `matched_stretches_secondary_run40.json`,
`crossings_0332_plan.json`, and the run plans `score/results/runplan_*.tsv`.

- 5 of the 620 PHerc0139 fibers and 2 of the 255 PHerc0332 fibers have fewer than 2 control points and are not used.
- The secondary rule's 40 sv minimum is the `--min-run 40` sean used to match fibers to patches for this model's
  own training. It was chosen after the primary count (2 fibers) was known, from fibers and segments only, before
  any strip. It is labelled secondary everywhere.

### Metrics (PREREG 5)

- `d`: signed normal distance from a strip node to the matched segment's bilinear surface, positive outward.
- A strip row is scored if its centre is within 30 sv of its fiber and its arclength falls in a matched stretch.
  It is scored against that stretch's segment only, so a strip on the next winding counts as an error of about one
  gap (12 sv). A fiber needs at least 25 covered nodes.
- Reported: mean, median and p90 of \|d\|, the fraction of nodes within 1 and 2 sv, and the signed mean (bias),
  pooled and per family (H, V).
- Uncertainty: cluster bootstrap over fibers, 2000 replicates (1000 for median and p90), seed 20261028; the paired
  baseline comparison uses seed 20261029 and fibers with at least 25 nodes covered by both surfaces, the PHerc0332
  crossing summary seed 20261030.
- Also: the human fiber's own distance to the mesh (fiber floor), calibration of the model's `expected_error`,
  and sensitivity cuts (unflagged stretches, per annotator, per family).

### Baselines (PREREG 5.5)

Both use the same nodes and the same distance.

- **(a) Frame plane:** the u = 0 plane of each decision's frame, stitched like the strip. This is the comparison
  behind the README's 6.89 vs 1.22.
- **(b) Lasagna-normal plane:** the plane through the fiber point that holds the fiber tangent, with its normal
  from the open-data Lasagna L2 normals.
- "Beaten" means the paired cluster-bootstrap 95 % CI of (baseline mean \|d\| − strip mean \|d\|) lies above 0.

### Gates (PREREG 6)

- **G1, ground truth adequate** (fibers and segments only): at least 30 fibers with a matched stretch, NF_match
  ≤ 2 sv and NF6 ≤ 2 sv. NF_match is the median \|d\| of the human fiber over the matched stretches; NF6 the same
  over all runs within 6 sv of a segment, which does not depend on the 2 sv window.
  - Primary: **fails**. 2 fibers (NF_match 0.78 sv, NF6 1.76 sv). There is no PHerc0139 accuracy claim from the
    primary set; its 2 strips are reported as case studies.
  - Secondary: passes. 57 fibers, NF_match 0.82 sv, NF6 1.76 sv.
- **G1', at scoring time:** at least 30 fibers actually scored, each with at least 25 covered nodes. The secondary
  run subsample has 36 fibers, so up to 6 can be lost to failed runs or coverage.
- **G2, transfers or degrades** (only if G1 and G1' pass): "transfers" needs a pooled mean \|d\| ≤ 1.5 sv, at least
  70 % of nodes within 2 sv, and both baselines beaten. Anything else is "degrades". A baseline that cannot be
  computed counts as not beaten. The secondary verdict always carries the prefix "secondary:".
- **PHerc0332:** no gate. Per crossing, the agreement is the mean of the two strips' median \|d\| to each other
  within 30 sv of the crossing. It is labelled "no ground truth: consistency, not accuracy".

### Subsample rule (PREREG 4.8)

- **Caps:** 0139 primary 8 CPU-h and 8 GiB of new CT; 0139 secondary 8 CPU-h and the CT cap of amendment A1;
  PHerc0332 2 CPU-h, 2 GiB and at most 60 crossings.
- **Units:** pieces (PHerc0332: crossings). **Strata:** (the piece's segment, family), where the piece's segment is
  the one holding the most total stretch length in the piece; for PHerc0332, (tight crossing yes/no, z tercile).
  v3's text said "segment of the piece's longest stretch", which is not what the code did; read that way, the rule
  would have given a 9.1 GiB cap and a different set of 45 pieces. The frozen set, made by the code, stays (PREREG
  10.7, item 1).
- **Order:** strata in sorted key order; within a stratum, units sorted by id and permuted by
  `numpy.random.default_rng(20261028)`.
- **Selection:** round robin over the strata, preferring fibers not yet chosen. A unit is taken if the cumulative
  CPU time (at 1.35 decisions/s) and new CT stay within the caps; otherwise it is passed over for good.
- The primary set is within its caps and is not subsampled.

### Amendments (PREREG 10)

| Version | Frozen (UTC) | What changed |
|---|---|---|
| v1 | 27 Oct 2026, 19:36 | Whole-fiber selection (≥ 70 % of the fiber within 2 sv). Its G1 run, after the freeze, found 0 fibers (`score/superseded_v1/`). |
| v2 | 28 Oct 2026, 12:27 | Stretch rule; NF6 in G1; pieces with `--arc`; caps and the seeded subsample; the secondary set; row window 10 → 30 sv; fiber-level bootstrap clusters; 0332 crossing tolerance 3 → 6 sv. Decided after v1's G1 numbers and ground-truth diagnostics, before any strip. An independent verifier failed v2 before any strip was run against its plans (`score/superseded_v2/`). |
| v3 | 28 Oct 2026, 14:04 | Decided on 28 Oct from the verifier's report, plan geometry and ground truth only, before any 0139 or 0332 strip was run against a plan. **P1:** v2 lost the last decision head of most pieces; v3 sets END = START + 16 (n − 1) + 1 sv and tests exact equality with `infer.py`'s head code. **P2:** heads moved from the raw JSON arclength to the full run's kink-repaired grid. **P3:** wording, and two rules in the code now stated (5-node crossing rule, 10 sv Lasagna rule). **A1:** secondary CT cap from 8 GiB to 9.6 GiB, the smallest cap on 8.0 to 13.0 GiB (step 0.1) giving at least 36 fibers, so that one lost fiber cannot fail G1'. **A2:** CPU time given as a range, 3.27 to 3.56 decisions/s idle and 1.35 under load. **Identity fix:** `--space` for fibers without a coordinate space (changes one unmatched fiber's id). |
| v3.1 | 29 Oct 2026, 00:34 | Erratum, from an independent adversarial check of v3, before any strip of the plans. No plan, scorer or runner change (50 of v3's 52 frozen files keep their sha256). **Wording:** the strata rule (the segment with the most stretch length, not the segment of the longest stretch); the crossing rule (nearest V sample per H sample; an exhaustive search finds the same 48 pairs); P0104's other heads ≤ 0.235 sv. **Rules now stated:** bootstrap seeds 20261029 and 20261030, the 25-node rule of the paired comparison, the row-centre fallback, "new CT" = not cached at plan time, `--arc` of the run plan authoritative, one retry per failed piece. **G0 limit:** G0 ran before the normalization patch, which is a no-op on Paris 4. **Pins:** villa worktree, patches, checkpoint. **Checks on Paris 4:** the scorer on real `infer.py` output; a piece run against the full run. **Recorded:** the runner's stops on low disk, the CT cache moves, an accidental shutdown, the prefetch-only pass, and two PHerc0332 smoke strips of a fiber in no plan made by a separate build. |
| v3.2 | 29 Oct 2026, 02:10:33 | Erratum, from a second independent adversarial check (of v3.1), before any strip of the plans. Disclosure and wording only; no plan, scorer, runner or pin of the strip code changes. **Listed:** a byte-identical rerun of the 0139 smoke strip by the setup's verifier; every file of the separate build's smoke test (its exploratory crossing step was given no strip). **Corrected:** the piece-run claim (the first decision of a piece can differ; see Design); the shutdown history (two power cycles); the crossing point (the midpoint of the closest H and V samples). **Pinned:** the relaunch wrapper, whose code is that of its first version, and the prefetch-only pass. **Checked:** the checkpoint pins against Hugging Face itself. **Declared:** all local paths in frozen files. |
| after v3.2 | | {{RESULT_DEVIATIONS}} (PREREG 10.6: none at freeze time) |

Before the freeze exactly one PHerc0139 strip existed: a smoke test of the setup run (fiber
`dj_20260902T151037604_000001`), plus a byte-identical rerun of it by the setup's independent verifier. That fiber
has no matched stretch, and the strip was never scored (PREREG 10.3). A separate full-PHerc0332 build made two smoke
strips of one PHerc0332 fiber (`kb_20260824T164423720_000066`) that is in no plan and in none of the 48 crossings;
nobody opened them, and that build's exploratory crossing step was given no strip and computed nothing (PREREG
10.7, item 8).

## G0 reproduction (Paris 4)

Before running anything new, we checked that the pinned code on CPU and Windows reproduces the organizers'
published Paris 4 strips (`results/g0.json`).

- **Selection:** 10 fibers (5 H, 5 V) by a seeded rule written before any strip was made
  (`results/g0_selection_rule.txt`, seed 20261027): published v3-checkpoint strips of 2.5 to 6 mm, spread over z
  quintiles and radius (`results/g0_selected.json`).
- **Gate:** at least 9 of 10 strips with a mean node-to-node distance ≤ 1 patch voxel, nodes matched by grid index.
- **Result: 10 of 10 pass.** Mean of the 10 means 0.012 voxels; worst mean 0.029 voxels; worst p99 0.123 voxels.
  Every strip has the published shape and layout, and the valid-node masks agree exactly (IoU 1.0). All 10 are also
  inside #2163's tighter band (mean ≤ 0.05, p99 ≤ 0.5).
- **Setup:** CPU only, fp32 (autocast is off on CPU; the published strips were made on GPU in bf16), Windows 11,
  Python 3.14.6, torch 2.13.0+cu126 with CUDA hidden, 16 torch threads, villa 6f2024569 + #2163. A streamed pass and
  a cached pass gave bitwise identical strips.

| Fiber | Family | Length | Decisions | Nodes | Mean distance | p99 | Mask IoU |
|---|---|---|---|---|---|---|---|
| 238475 | H | 3.52 mm | 22 | 1,350 | 0.029 | 0.123 | 1.0 |
| 329464 | V | 4.35 mm | 27 | 1,875 | 0.011 | 0.043 | 1.0 |
| 435199 | H | 3.33 mm | 21 | 1,225 | 0.008 | 0.027 | 1.0 |
| 307062 | V | 3.24 mm | 20 | 1,175 | 0.009 | 0.035 | 1.0 |
| 498336 | H | 3.68 mm | 23 | 1,450 | 0.015 | 0.073 | 1.0 |
| 523953 | V | 2.96 mm | 18 | 975 | 0.010 | 0.038 | 1.0 |
| 757072 | H | 4.68 mm | 29 | 2,100 | 0.010 | 0.044 | 1.0 |
| 740130 | V | 3.02 mm | 19 | 1,025 | 0.012 | 0.041 | 1.0 |
| 838073 | H | 3.06 mm | 19 | 1,050 | 0.009 | 0.035 | 1.0 |
| 778459 | V | 4.80 mm | 30 | 2,175 | 0.008 | 0.029 | 1.0 |

Distances in patch voxels (9.6 µm). The published strips are not redistributed;
`data/g0/published/manifest.json` lists the zip members we compared (names, sizes, CRC-32) from
`https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/fiber_strips/101526/all_strips.zip`.

**Throughput** (shared machine, CT cached): 1.35 decisions/s with 16 threads and batch 32; 1.47 with 16 threads and
`--crop-threads 4`; 1.33 with 8 threads and batch 16. An idle rerun measured 3.27 to 3.56 decisions/s.

**Errata** (`results/g0_errata.md`; none changes the verdict):
1. CT for a whole scroll at the r = 170 footprint is 183.0 GiB (PHerc0139) and 36.4 GiB (PHerc0332). The 157 and
   32 GiB first stated are the r = 120 figures.
2. The 10-fiber G0 AFV is 147,456 bytes, not 0.36 MB.
3. `patches/applied_pr2163_on_6f2024569.diff` omits #2163's two new test files. The complete diff is
   `patches/applied_pr2163_on_6f2024569_with_tests.diff`.
4. "4 threads: 1.47 decisions/s" was 16 torch threads with `--crop-threads 4`.
5. The RAM peak was about 4.1 GB.

## Results

Not in yet. The detached runner (`score/runs/v3/run_all.ps1`) verifies `FREEZE.txt`, prefetches the CT chunks,
runs the pieces and then the frozen scorer. Reports will go to `score/results/v3/`.

**Run record.** Started {{RESULT_RUN_START}}, finished {{RESULT_RUN_END}}, exit code {{RESULT_RUN_RC}}. Pieces
ok / failed: primary {{RESULT_PRIMARY_PIECES_OK}} / {{RESULT_PRIMARY_PIECES_FAILED}}, secondary
{{RESULT_SEC_PIECES_OK}} / {{RESULT_SEC_PIECES_FAILED}}, PHerc0332 {{RESULT_0332_PIECES_OK}} /
{{RESULT_0332_PIECES_FAILED}}. Decisions {{RESULT_DECISIONS_TOTAL}} of 4,129 planned. CPU time
{{RESULT_CPU_HOURS}} h; CT fetched {{RESULT_CT_GIB}} GiB.

### PHerc0139 primary: 2 case studies (G1 failed, no accuracy claim)

| Stretch | Fiber | Segment | Length | Nodes | Mean \|d\| | Within 2 sv | Bias | Frame plane mean \|d\| | Lasagna plane mean \|d\| | Fiber floor |
|---|---|---|---|---|---|---|---|---|---|---|
| S0000 | `kb_20260904T021448778_000085` (H) | w044 | 3.73 mm | {{RESULT_P_S0000_NODES}} | {{RESULT_P_S0000_MEAN}} | {{RESULT_P_S0000_W2}} | {{RESULT_P_S0000_BIAS}} | {{RESULT_P_S0000_FRAME}} | {{RESULT_P_S0000_LASAGNA}} | {{RESULT_P_S0000_FLOOR}} |
| S0001 | `lt_20260918T092404850_000160` (H) | w054 | 3.68 mm | {{RESULT_P_S0001_NODES}} | {{RESULT_P_S0001_MEAN}} | {{RESULT_P_S0001_W2}} | {{RESULT_P_S0001_BIAS}} | {{RESULT_P_S0001_FRAME}} | {{RESULT_P_S0001_LASAGNA}} | {{RESULT_P_S0001_FLOOR}} |

### PHerc0139 secondary (labelled secondary)

**G1':** {{RESULT_SEC_FIBERS_SCORED}} of 36 fibers scored with at least 25 covered nodes, so G1'
{{RESULT_SEC_G1P}}.

| | Fibers | Nodes | Mean \|d\| [95 % CI] | Median | p90 | Within 1 sv | Within 2 sv | Bias |
|---|---|---|---|---|---|---|---|---|
| All | {{RESULT_SEC_ALL_FIBERS}} | {{RESULT_SEC_ALL_NODES}} | {{RESULT_SEC_ALL_MEAN}} [{{RESULT_SEC_ALL_MEAN_CI}}] | {{RESULT_SEC_ALL_MEDIAN}} | {{RESULT_SEC_ALL_P90}} | {{RESULT_SEC_ALL_W1}} | {{RESULT_SEC_ALL_W2}} | {{RESULT_SEC_ALL_BIAS}} |
| H | {{RESULT_SEC_H_FIBERS}} | {{RESULT_SEC_H_NODES}} | {{RESULT_SEC_H_MEAN}} [{{RESULT_SEC_H_MEAN_CI}}] | {{RESULT_SEC_H_MEDIAN}} | {{RESULT_SEC_H_P90}} | {{RESULT_SEC_H_W1}} | {{RESULT_SEC_H_W2}} | {{RESULT_SEC_H_BIAS}} |
| V | {{RESULT_SEC_V_FIBERS}} | {{RESULT_SEC_V_NODES}} | {{RESULT_SEC_V_MEAN}} [{{RESULT_SEC_V_MEAN_CI}}] | {{RESULT_SEC_V_MEDIAN}} | {{RESULT_SEC_V_P90}} | {{RESULT_SEC_V_W1}} | {{RESULT_SEC_V_W2}} | {{RESULT_SEC_V_BIAS}} |

| Baseline | Mean \|d\| | Baseline − strip [95 % CI] | Within-2 difference | Beaten |
|---|---|---|---|---|
| (a) Frame plane | {{RESULT_SEC_FRAME_MEAN}} | {{RESULT_SEC_FRAME_DIFF}} [{{RESULT_SEC_FRAME_DIFF_CI}}] | {{RESULT_SEC_FRAME_W2_DIFF}} | {{RESULT_SEC_FRAME_BEATEN}} |
| (b) Lasagna-normal plane | {{RESULT_SEC_LASAGNA_MEAN}} | {{RESULT_SEC_LASAGNA_DIFF}} [{{RESULT_SEC_LASAGNA_DIFF_CI}}] | {{RESULT_SEC_LASAGNA_W2_DIFF}} | {{RESULT_SEC_LASAGNA_BEATEN}} |

- **Verdict:** secondary: {{RESULT_SEC_VERDICT}}.
- **Fiber floor** (the human fiber's own \|d\| on the scored rows): {{RESULT_SEC_FIBER_FLOOR}}.
- **`expected_error` calibration:** weighted gap {{RESULT_SEC_CAL_GAP}} sv; observed / predicted
  {{RESULT_SEC_CAL_RATIO}}; Spearman rho {{RESULT_SEC_CAL_RHO_NODE}} (nodes), {{RESULT_SEC_CAL_RHO_FIBER}} (fibers).
- **Sensitivity** (not gating): unflagged stretches {{RESULT_SEC_SENS_UNFLAGGED}}; per annotator
  {{RESULT_SEC_SENS_ANNOTATOR}}.
- **Rejected strips** (unit or association errors): {{RESULT_SEC_REJECTED}}.

### PHerc0332: H/V crossing agreement (no ground truth: consistency, not accuracy)

| | Crossings with both strips | Mean | Median | p90 | ≤ 2 sv |
|---|---|---|---|---|---|
| All 15 planned | {{RESULT_0332_N}} | {{RESULT_0332_MEAN}} [{{RESULT_0332_MEAN_CI}}] | {{RESULT_0332_MEDIAN}} | {{RESULT_0332_P90}} | {{RESULT_0332_W2}} |
| Tight subset (3, ≤ 3 sv) | {{RESULT_0332_TIGHT_N}} | {{RESULT_0332_TIGHT_MEAN}} | {{RESULT_0332_TIGHT_MEDIAN}} | | |

Missing crossings: {{RESULT_0332_MISSING}}. H/V crossings found among the 0139 strips: {{RESULT_0139_CROSSINGS}}.

## What it enables

- A measured accuracy of `fiber_strip_small_v1` off Paris 4, or a documented reason why it could not be measured,
  with a ground-truth noise floor and two baselines. This is what someone needs before using strips as sheet
  evidence on the 0139, 0332 or 1451 scaffolds.
- A check of whether the model's own `expected_error` still tracks its error off Paris 4, i.e. whether it can be
  used to filter strips.
- Strips for the PHerc0332 crossing windows and the PHerc0139 run pieces, to be published as a release asset
  under CC BY-NC 4.0 once local paths are removed from their metadata.
- Tools that work on their own: a VC3D fiber JSON (v3/v4) to AFV converter that keeps points bit for bit
  (`tools/fibers_json_to_afv.py`); a piece planner that reproduces a full run's decisions
  (`score/tools/make_piece_afv.py`, `full_run_map.py`); a scorer for normal distance to tifxyz meshes with a
  fiber-clustered bootstrap (`score/score_strips.py`); and a tested CPU/Windows recipe for the strip model (G0).

## How to reproduce

1. **Check the freeze.** `python verify_freeze.py` (Python 3, standard library only). Expected: FREEZE.txt as
   recorded; 72 of its 100 files, 25 of the 34 v2 archive files and 37 of the 52 v3 files verified in the repo; the
   others are data, villa files or checkpoint files not shipped here, each listed with how to get or rebuild it;
   `PASS`. After step 3, `fetch_inputs.py` and the rebuild steps, `python verify_freeze.py --strict` checks all of
   them.
2. **Environment** used here: Windows 11, Python 3.14.6, torch 2.13.0 (CPU is enough), numpy, scipy, tifffile,
   pytest, huggingface_hub, and villa's own dependencies.
3. **villa**, pinned, in `villa_6f2/` at the repo root (the scripts put `villa_6f2/vesuvius/src` first on
   `sys.path`):
   ```
   git clone https://github.com/ScrollPrize/villa villa_6f2
   git -C villa_6f2 checkout 6f2024569404a46b0f008eee292773945c07a684
   git -C villa_6f2 apply ../patches/applied_pr2163_on_6f2024569_with_tests.diff
   git -C villa_6f2 apply ../patches/strip_bind_normalization_other_volume.diff
   ```
4. **Inputs** (about 1.2 GiB, checked against the manifests' sha256):
   `python fetch_inputs.py fibers0139 fibers0332 segments umbilicus samples checkpoint`.
   `python fetch_inputs.py --list` shows the sets without any download. The PHerc0332 fiber folder kept changing
   after our download; a changed file is kept as `.mismatch` and reported.
5. **Derived data** (not shipped): the full AFVs with `tools/fibers_json_to_afv.py` (controlled span), the
   full-run maps with `score/tools/full_run_map.py`, the piece AFVs with `score/runs/plan_v3/run_afv.sh`, the
   Lasagna normals with `score/score_strips.py lasagna` (source arrays in PREREG 3). The run plans and match files
   that these must agree with are shipped in `score/results/`.
6. **Run:** `score/runs/v3/run_all.ps1` (PowerShell, detached). Stages: verify FREEZE, CT metadata, preflight, CT
   prefetch (at most 2 connections; stops below 10 GiB free), inference one piece at a time, scoring. Budget at
   freeze: about 1 CPU-h idle to 1.5 CPU-h under load, and 11.7 GiB of new CT (about 5.7 h at 0.58 MiB/s).
7. **Tests:** `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q score/tests tools`. The 35 synthetic scorer tests and 10
   of the 12 v3.1 tests (`score/tests/test_erratum_v31.py`, with the real Paris 4 fixture) run anywhere: 45 passed
   and 4 skipped in a fresh checkout. `score/tests/test_piece_afv.py`, the v3.1 P0104 and pin tests need the data,
   the checkpoint and the pinned villa; `tools/test_fibers_json_to_afv.py` needs `data/samples`
   (`fetch_inputs.py samples`). Without them they are skipped. Use `CUDA_VISIBLE_DEVICES=-1`: on Windows an empty
   value does not hide the GPU.

The frozen scripts keep the absolute paths of the author's machine (`C:/VesuviusTemp/strips`,
`C:/Users/PC/miniconda3/envs/vesuvius/python.exe`). To run them unchanged, check this repo out to
`C:/VesuviusTemp/strips`. Anywhere else, edit those lines; `verify_freeze.py` will then report those files as
changed, as it should. Two more local paths appear in frozen text, as history only: `D:/PrizeTemp/strips_ct_cache`
(where the CT cache was on 28 Oct; PREREG 10.7 item 7) and `D:/Competition/Vesuvius progress prizes/villa` (where
the patches were made; `patches/APPLIED.txt`). No script reads them.

## Data in this repo

| Included | Not included (manifest or rebuild instead) |
|---|---|
| Code, tests, the runner, the patches | CT (never redistributed; fetched by villa's cache during inference) |
| PREREG v1 to v3.2, FREEZE records | Raw human fiber JSONs (`score/data/fibers*_manifest.json`: URL, size, sha256) |
| One real Paris 4 strip and its G0 reference, as a test fixture (`score/tests/fixtures/paris4_real`, CC BY-NC 4.0) | PHerc0139 and PHerc0332 strips (none made yet) |
| Match, plan and AFV-report tables (`score/results/`) | Wrap segment meshes (`score/data/segments_manifest.json`) |
| G0 records (`results/`) | Strip model weights (`manifests/checkpoint_fiber_strip_small_v1.json`) |
| Manifests | Umbilici (`manifests/umbilicus.json`), AFVs, maps, Lasagna normals (rebuilt, step 5) |
| | Published Paris 4 strips (`data/g0/published/manifest.json`), ink-detection images |

Code is MIT (`LICENSE`). Tables and any strips are CC BY-NC 4.0 (`LICENSE-DATA.md`). villa-derived parts keep
villa's MIT notice (`NOTICE.md`).

## Limits

- **Meshes are not the sheet.** Wrap meshes can sit on one face of the sheet, smooth over detail, or jump. Where
  the human fibers are near a mesh they disagree with it by about 2 sv: NF6 is 1.76 sv, close to G2's 1.5 sv
  threshold. A perfect strip would also show about 1 to 2 sv. The fiber floor and the baselines are the reference.
- **Ground-truth noise.** ChaseHendrick's fiber_qa (#2248) found fold-backs, jumps and near-duplicates in the 0139
  fibers. Unflagged stretches are reported separately.
- **Selection bias.** Only fibers on meshed sheets are scored, and meshed regions are the readable ones. Only
  45.9 % of the 0139 fiber samples are even within the segments' z range. The secondary stretches are short
  (median 0.56 mm), so the score is local.
- **The primary set failed G1.** It has 2 fibers. Any PHerc0139 accuracy statement rests on the secondary set,
  whose 40 sv rule was chosen after the primary count was known.
- **Pieces, not whole fibers.** Five pieces have head offsets above 1 sv from a full run (three in the run sets).
  Where a fiber loops back into a crop from outside its piece (6 secondary and 6 PHerc0332 run pieces), the
  model's tube input differs from a full run's by an amount not measured. The first decision of a piece cut
  mid-fiber can differ too (cause not identified); it reaches at most a window's first one or two rows, with at most
  about 7 % of their weight. On the Paris 4 window rows tested, a piece run equals a full run within 0.001 sv.
- **G0 does not cover the normalization patch.** G0 ran before it, and on Paris 4 it does nothing. Its effect on
  PHerc0139 and PHerc0332 is only to bind the per-crop z-score record to the store being read.
- **CPU and Windows.** fp32 on CPU against bf16 on GPU; G0 matched the published strips within 0.03 voxels, not
  exactly. The runs use 8 threads; G0's compared strips used 16.
- **Metric.** We measure normal distance to a wrap mesh. The villa README measures frame-normal height against
  verified patches. The numbers are related, not identical.
- **PHerc0332 has no ground truth.** Two strips can agree and both be wrong. At 3 to 6 sv the two fibers may lie on
  opposite faces of a sheet or on neighbouring sheets. Fifteen crossings are few.
- **Budget figures** rest on a 30 s per-process overhead that was assumed, and a download rate measured once.
- **Licensing.** The human fiber folders and the checkpoint carry no license. We link them and do not copy them.
  Treating our derived tables as CC BY-NC 4.0 is our reading of the open-data terms, not an explicit grant.
- **Local paths.** The frozen files keep four paths of the author's machine (How to reproduce); they could not be
  edited without breaking the freeze.

## Credits

- **Strip model:** Sean Johnson (bruniss): sheet_strip training and inference, ScrollPrize/villa PR #2110 (merged
  2026-10-15, 826c2f64e), and the checkpoint `scrollprize/fiber_strip_small_v1` on Hugging Face (revision
  7bd163a22c839ff312879ae1916195d0c9729c0e). Also the fiber follower (#2045), the frame model and
  `match_fibers_to_patches.py`, whose ±2 sv window and `--min-run 40` are used here. The weights are not
  redistributed here; the scripts download them from Hugging Face.
- **Windows/CPU support:** ScrollPrize/villa PR #2163 by Sebastien Chicoineau (slchicoineau-creator) and Chris
  Thomas (Baz248), applied unchanged at head bfad589 (open, unmerged at the time of writing).
- **Human fiber scaffolds:** the Vesuvius Challenge annotators whose files carry the initials lt, kb and dj
  (https://dl.ash2txt.org/other/annotators/0139_fibers/ and /0332_fibers/). The files are not redistributed here;
  `score/data/fibers0139_manifest.json` and `fibers0332_manifest.json` list each URL and sha256.
- **Wrap segments, CT scans and representations:** Vesuvius Challenge open data (s3://vesuvius-challenge-open-data),
  CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). Volumes: PHerc0139 20260102150214 (2.399 µm),
  PHerc0332 20251211183505 (2.399 µm), PHercParis4 20260411134726 (2.400 µm). CT is not redistributed.
- **Data citation** (as requested at https://scrollprize.org/data#how-to-cite): Giorgio Angelotti, Stephen Parsons,
  Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph, Paul Tafforeau, Alessandro Mirone, Paul Henderson, Hendrik
  Schilling, Forrest McDonald, David Josey, Youssef Nader, C. Seth Parker, W. Brent Seales. Vesuvius Challenge -
  CT Scans of Herculaneum Papyri. Vesuvius Challenge.
- **Prior and related work:** modaribrahim/fiber-tracer-heldout-eval (MIT) evaluated fiber_single_small on the
  human fibers of 0139, 1451 and 0332 and on held-out Paris 4, and fixed the scoring in villa #2143.
  ChaseHendrick: villa #2248 (fiber_qa), #2265 (sheet_plane_weight; drift off the sheet on PHerc0139 fibers) and
  #2320/#2321 (ingest of the published v4 fiber JSONs).
- **Fiber tracing, strip inference and VC3D code:** ScrollPrize/villa (MIT, Copyright (c) 2024 Vesuvius
  Challenge).
- **License:** code MIT. Strips, expected_error maps and result tables are derived from Vesuvius Challenge CT and
  human fiber annotations and are released under CC BY-NC 4.0. They were modified by running fiber_strip_small_v1
  on the human fibers.

Author: TAUIL Abd Elilah.

Developed with AI assistance under my direction.
