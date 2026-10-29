#!/usr/bin/env python
"""Independent re-scoring of the pre-specified verification sample (PREREG v3.2), without score/score_strips.py.

    python verification/independent_rescore.py --workspace C:/VesuviusTemp/strips/score \
        --sample verification/verification_sample.json --out verification/rescore_report.json [--compare]

Run it only AFTER the scored run (score/runs/v3/run_all.ps1) has finished. It recomputes, from the text of
score/PREREG.md (sections 4.1, 5 and 7), the numbers the scorer reports for the items of
verification/verification_sample.json (fixed and published before any result): 20 secondary stretches, the 2
primary stretches and the 15 PHerc0332 run crossings, and writes them to --out. With --compare it then reads the
scorer's reports (<workspace>/results/v3/primary_case_studies.json, secondary_report.json,
crossings_0332_report.json; --reports overrides the folder) and adds per-item differences and a summary to the same
file. --compare-only compares an existing --out report without recomputing. --gt-only computes only the
ground-truth check below: it reads no strip and no scorer report. Exit code: 0 when no compared item differs (each
agrees within 1e-3 sv, or within the solver bounds of tier 2, or under a flagged other reading: see TOLERANCE AND
VERDICTS; also when nothing was compared), 1 when an item differs or exists on one side only, 2 on a usage or input
error.

Inputs, all under --workspace (the frozen score/ tree): results/matched_stretches_secondary_run40.json,
results/matched_stretches.json and results/crossings_0332_plan.json (their sha256 are checked against the
sample's); data/fibers0139 and data/fibers0332 (VC3D fiber JSONs); the segments of the plan's segment list
(data/segments/<segment>/{x,y,z}.tif); data/umbilicus/PHerc0139_20260102150214_umbilicus.json (level 0); the strips
runs/v3/strips/{primary,secondary,0332}/<piece id>/strip (run_all.ps1's layout; --strips overrides the root).

WHAT IS COMPARED
The scorer's 'score' report has no per-stretch numbers. It has one 'per_strip' entry per strip, i.e. per piece
(strip, stretches, assoc_median_sv, strip_nodes, strip_mean_abs_sv, strip_signed_mean_sv, fiber_floor_mean_abs_sv,
baseline fields), and pooled 'accuracy' blocks. A piece holds every stretch of its windows (P0006 holds 31). So for
each of the 15 pieces that hold a sampled stretch (13 secondary, 2 primary) every stretch of the piece is recomputed
(190 + 2 stretches); the per-piece numbers are compared with the scorer's per_strip entry, and the per-stretch
numbers are reported (they have a scorer counterpart only where a piece holds one stretch: S0235 in P0035, S0000 in
P0000, S0001 in P0001). The primary report's pooled point estimates (accuracy.all: scored fibers, nodes, mean,
median, p90, nodes within 1 and 2 sv) are compared too: both primary pieces are recomputed, so that pool is
complete. For PHerc0332 each sampled crossing's missing status, directions, agreement_sv, point_sv and
closest_approach_sv are compared, and the number of planned crossings; the association of both crossing strips
(5.2, R12) is reported (the scorer's per_crossing row does not hold it).

Ground-truth check (no strip). For every recomputed stretch the fiber samples of the stretch are projected onto its
segment with the same distance code, and their count, coverage, mean d, median |d|, p90 |d| and max |d| are
compared with the values the frozen plan records for that stretch (written by the scorer's 'match' step before any
strip existed). This tests 4.1 and 5.1 (sampling, projection, sign, coverage) on the real meshes without strips.

Not recomputed: the frame-plane and Lasagna-plane baselines (5.5), the calibration (5.7), the bootstrap intervals
(5.4, 7), the gates, the secondary set's pooled numbers (only a sample is recomputed), the sensitivity blocks (4.9)
and the PHerc0139 crossings block of the score reports (score --crossings). For the crossing summary (mean,
median, p90, fraction <= 2 sv over crossings) the recomputed values and the scorer's block are put side by side
without a pass/fail, because the scorer's key names were not read.

INDEPENDENCE
This file does not import score_strips.py and copies none of its code. The metrics are written from the text of
PREREG.md. score_strips.py (sha256 d7d6d194..., line numbers of that frozen file) was read, by the author of
version 1.0, only for file formats, units and the report structure, in these places:
  1-135      module docstring, constants, two helpers: units (sv = level 2 = 4 base voxels; fiber JSONs and
             segment tifxyz in level-0 voxels; strips in patch voxels = sv) and the constants, which PREREG states;
  181-198    Umbilicus: the umbilicus JSON ('control_points' with x, y, z). These lines also show how the scorer
             interpolates the umbilicus in z (control points sorted by z, linear between them, constant beyond).
             R8 uses the same rule, so that part of R8 was not settled independently of the scorer (PREREG does not
             define it). To show what it changes, every block counts the covered nodes whose outward sign would
             differ with the umbilicus taken at the control point nearest in z instead
             (outward_sign_differs_with_nearest_control_point);
  202-221    read_tifxyz: meta.json, x/y/z.tif, a node is invalid when a coordinate is -1 or not finite;
  343-392    SegmentSource, discover_segments: segment folders and their names;
  397-534    Fiber ... load_fibers: the JSON fields 'line_points', 'control_points'[].position and
             'hv_classification' (manual tag, else automatic), the file stem as the fiber name. This block also
             holds the scorer's controlled-span rule; the rule used here is taken from tools/fibers_json_to_afv.py
             (kept_points), the converter that PREREG 4.1 names, and is the same rule. The scorer's resample() (the
             1 sv sampling) was not read; the end sample of R1 is inferred from the frozen plan's data;
  1037-1094  Strip, find_decisions, load_strip, and 1135-1138 discover_strips: OUT/<piece id>/strip,
             trace.json / summary.json, the fiber_along == 'columns' transpose, strip unit base 4;
  1766-1780  provenance, load_axis, and 2065-2124 main's defaults (--umbilicus-unit-base 1, i.e. level 0;
             --strip-unit-base 4; --span controlled);
  1999-2062  cmd_score, cmd_crossings: the report files ('per_strip' fields, 'accuracy', 'crossings'). These lines
             also show how the per_strip fields are aggregated (means over the finite per-node values, i.e. over
             the covered nodes, and for the fiber floor over the rows where it is finite). R9's per-piece
             aggregation is the same and was therefore seen in the scorer, not settled independently;
  1735-1737  the 'per_crossing' row (id, h_fiber, v_fiber, closest_approach_sv, point_sv, agreement_sv,
             directions);
  plus a text search for 'per_crossing', 'missing' and "'id'" that showed single lines elsewhere, among them
  1351, which suggests the scorer takes a strip's stretches from its piece's windows (R6: this file keeps its own
  reading and counts the rows where the two readings differ).
Not read: the projection (GridSurface), the sampling, stretch finding, row centres, association, score_strips,
aggregate, the bootstrap, the calibration, find_crossings and crossing_agreement (except its three output lines).
Version 1.1 (29 Oct 2026, about 04:00 to 05:24 UTC, after an adversarial review) read no line of score_strips.py. It
was made after the scored run had finished and its reports existed (03:39 to 03:41 UTC), by agents working under the
rule never to read the run's outputs, and before this file was compared with any report (06:10 UTC).
It used four facts the review reported from its own reading of the scorer: (a) the scorer projects by a
step-limited Gauss-Newton from the 4 nearest nodes (8 iterations, clamped at invalid cells, the nearest covered
candidate), a different solver from R7; this file keeps R7 and reads differences through tier 2 (TOLERANCE AND
VERDICTS); (b) the scorer's crossings step associates each strip with the nearest planned fiber (5.2's 5 sv rule)
and takes, per fiber, the strip with the most nodes within 30 sv of the crossing point; R12 now applies 5.2's rule
as PREREG's text states it, and that choice is computed as R12's other reading; (c) with no scored fiber the
scorer writes accuracy.all.strip.point = None; (d) a strip with no row in a matched stretch gets no per_strip entry.
Version 1.1 checked tier 2 with the review's scripts and adaptations of them (see TOLERANCE AND VERDICTS). They call
the scorer's projection, score_strips() and crossing_agreement() as black boxes and rebuild cmd_score's per_strip
means; their code was read, the scorer's was not.
The last edit of version 1.1 (06:09 UTC), made after the reports and parts of score_strips.py had been seen, added
the later review's results to TOLERANCE AND VERDICTS and exit code 2 on an input error; it changed no reading,
metric, tolerance or bound. Version 1.1.1 (29 Oct 2026, after the comparison of 06:10 UTC) corrects this docstring
only (the timing of version 1.1 and the later review's counts); apart from TOOL_VERSION no code changed.
tools/paris4_piece_check.py (not the scorer) was read to set up the Paris 4 fixture test as PREREG 10.7 item 5 ran
it. Drafted with AI assistance (Claude, Anthropic); checked by verification/test_independent_rescore.py.

READINGS OF THE PREREG TEXT (where it is ambiguous; flagged here, not settled from the scorer's code)
  R1  Fiber samples (4.1): the controlled polyline (each control point is anchored at its nearest line point; the
      span runs from the smallest to the largest anchor index, as tools/fibers_json_to_afv.py), in sv (level 0 /
      4), sampled at arclength 0, 1, 2, ... sv and at its end. (The frozen plan's stretches S0105, S0117 and
      S0770 end at their fibers' non-integer lengths, so the end sample exists.)
  R2  "Distance to its fiber" (the 30 sv row window and the 5 sv association median, 5.2) is the distance from the
      row centre to the nearest fiber sample, the sample that also gives the row's arclength. (The distance to the
      polyline would be smaller by at most 0.5 sv at 0 sv, 0.025 sv at 5 sv, 0.004 sv at 30 sv.)
  R3  Every bound is inclusive on the accepted side: a row centre exactly 30 sv from its fiber is scored, a strip
      node exactly 30 sv from the crossing point counts, |d| = 30 sv and a sideways residual of exactly 1 sv are
      covered, a median centre distance of exactly 5 sv is associated, a V sample exactly 6 sv away counts, and
      |d| = 1 or 2 sv is within 1 or 2 sv.
  R4  The row centre is the node of column C // 2 (25 columns: column 12, v = 0), or the mean of the row's valid
      nodes when that node is invalid. For an even C the column C // 2 is used and a note is written.
  R5  The association median is over every row that has a centre (at least one valid node). A rejected strip
      (median above 5 sv) scores no row.
  R6  "Falls inside a matched stretch": s0 <= s <= s1 for the arclength s of the row's nearest sample, for any
      stretch of the strip's fiber in the scored set (secondary: run = true; primary: both), on that fiber's
      whole controlled span. Rows that fall in a stretch of another piece are counted
      (rows_in_other_pieces_stretches); a row in two stretches takes the first id (listed in rows_in_two_stretches).
      Other reading, reported (own_windows_only): only the stretches of the piece's own windows. Five fibers of
      the sample have two run pieces (P0005/P0006, P0083/P0084, P0125/P0126, P0149/P0150, P0179/P0180), where the
      readings can differ.
  R7  Projection (5.1). PREREG names Gauss-Newton from the 4 nearest nodes, not its iterations, step control or
      behaviour at cell edges. Here the foot is the closest point of the bilinear surface of the valid cells (cells
      whose 4 corners are valid) that lie within two cells of the point's 4 nearest valid nodes; per cell it is
      found by Gauss-Newton inside the cell and exactly on the cell's 4 straight edges. The normal is the surface
      normal at the foot (cross product of the two grid derivatives), d = (p - foot) . n, and the sideways
      residual is the part of p - foot orthogonal to n. A point is covered when a foot exists, the sideways
      residual is <= 1 sv and |d| <= 30 sv. A foot on a cell edge or node (a crease) has one normal per valid cell
      there: the one that fits p - foot best (the smallest sideways residual) gives d, and d and the sideways
      residual over all of them are kept for tier 2.
  R8  "Positive outward (away from the umbilicus)": per point, n is turned so that its horizontal part points away
      from the umbilicus at the foot's z (the umbilicus linear in z between its control points, constant beyond
      them: the scorer's rule, see INDEPENDENCE). Without an umbilicus (crossings, the Paris 4 fixture) only |d| is
      used.
  R9  Statistics are over the covered nodes of scored rows: mean, median and p90 (numpy's linear percentile) of
      |d|, the fraction with |d| <= 1 and <= 2 sv, the signed mean of d. The fiber floor (5.6) is, per scored row,
      |d| of the fiber sample that gives the row's arclength, on the same segment; fiber_floor_mean_abs_sv is its
      mean over the scored rows where that sample is covered. (The per-piece aggregation is the scorer's, see
      INDEPENDENCE.)
  R10 The fiber rule of 5.2 (fewer than 25 covered nodes: not scored) applies to pooled numbers only; per-piece and
      per-stretch numbers are reported for every strip that is not rejected.
  R11 Crossing point (7): for every sample of the H fiber the nearest sample of any V fiber of the scroll (k = 1;
      family from the JSON tag, manual if H or V, else automatic) is taken if within 6 sv; among the H samples
      whose nearest is on the crossing's V fiber the closest gives the point, the midpoint of that H sample and its
      nearest V sample. The recomputed point, not the plan's, is used for the agreement; both are reported.
  R12 Crossing strips: the strips of the pieces that hold the crossing's H and V windows in the plan. Each must run
      along its piece's own fiber (5.2: median centre distance, R2, R4, R5, <= 5 sv); a strip that does not is
      rejected and the crossing is missing (7: planned crossings without both strips are listed as missing). The
      association medians are reported. "Nodes within 30 sv of the crossing point": valid strip nodes at Euclidean
      distance <= 30 sv. "Covered by the other strip": covered as in R7 on the other strip's surface, unsigned. The
      median is over those covered nodes. Other reading, reported (other_reading): each strip belongs to the
      planned fiber it runs closest to (median <= 5 sv), and per fiber the strip with the most valid nodes within
      30 sv of the point is used.

TOLERANCE AND VERDICTS (fixed before any comparison; revised in version 1.1 on 29 Oct 2026, after the run's reports
existed but without reading them, because a review showed that tier 1 alone calls a correct scorer different when
strips lie far off the mesh)
  Tier 1, 'agree': integers (node counts, numbers of usable directions, missing status, a piece's stretch ids,
    numbers of scored fibers, sample counts, and the fractions within 1 and 2 sv, compared as node counts) exactly;
    values in sv (means, medians, p90, max, fiber floor, association median, agreement, crossing point and closest
    approach) within 1e-3 sv. Both codes read the same float32 coordinates and compute in float64; when both find
    the same foot a distance agrees to about 1e-9 sv, while another surface model (planar facets, nearest-node
    planes) moves it by up to the facet sagitta, about 0.02 sv (PREREG 8). 1e-3 sv is 1/1500 of the 1.5 sv gate
    and below the 2-decimal precision of the reported numbers.
  Why a tier 2: PREREG 5.1 names the method, not its iterations, step control or behaviour at cell edges, so the
    scorer and R7 are two solvers of one definition. Near the mesh they find the same foot. Far from it the
    distance has several local minima, at a crease the normal is not unique, and near the 1 sv and 30 sv bounds a
    small change flips coverage: there two correct solvers differ in coverage and in |d|. (The review measured, on
    synthetic meshes, node counts off by up to 5 % and means by up to 0.40 sv for strips tilted 10 to 40 deg or on
    a mesh with 0.3 sv node noise, |d| differences up to 0.64 sv at 12 to 30 sv, and crossing agreements off by
    0.03 to 0.79 sv; good strips agreed within 2.6e-4 sv.)
  Classes (classify(); d and the sideways residual are taken over every normal at the foot, R7):
    robust: covered, |d| <= 6 sv and sideways residual <= 0.5 sv for every normal at the foot, and |d| varying by
      at most 1e-3 sv between them (6 sv: half a winding gap, the NF6 window of PREREG 4.4; 0.5 sv: half the
      coverage bound);
    sensitive: any other node whose foot another solver may cover. Its |d| can be <= 30 sv: least - delta_below <=
      30 sv, where least = sqrt(distance^2 - 1 sv^2) is the smallest |d| a covering foot can give (no foot is
      nearer than this file's closest point, and a covering one is at most 1 sv sideways). And either its foot lies
      inside the mesh (on a crease or node there the residual depends on the normal), or on the mesh edge with a
      sideways residual <= max(1.5 sv, 0.35 x the distance) (far from the mesh another solver can find an inner
      local minimum);
    firmly uncovered: the rest (no foot; |d| out of reach; a foot on the mesh edge with a larger residual).
  Tier 2 tests this hypothesis: the scorer gives every robust node this file's |d| within 0.005 sv + 2 % of |d|,
    covers no firmly uncovered node, and covers any subset of the sensitive nodes, each with |d| in [least -
    delta_below, distance + delta_above] (delta_below = 0.1 sv + 2 % of the distance: a closest point that this
    file's cells could miss; delta_above = 0.1 sv + 10 % of the distance: another local minimum farther away) and
    d on the node's side of the mesh (either side when the normals at the foot disagree). Every compared value then
    has an interval: the count of covered nodes; a mean or signed mean (the scorer's node count fixes how many
    sensitive nodes it covers, and the interval runs from that many smallest lower bounds to that many largest
    upper bounds); median, p90 and max; the within-1 and within-2 counts; the fiber floor and the crossing medians
    (over every count, since the scorer does not report it; for a crossing direction every count the 5-node rule
    allows); the 25-node rule; the number of usable directions.
  Tier 2, 'agree_within_solver_bounds': every value that fails tier 1 lies in its interval, widened by the tier-1
    tolerance. Values that do not depend on the projection must pass tier 1: the association median, the crossing
    point and closest approach, the sample count of the ground-truth check, a missing or rejected strip. Each
    interval is written with its width, which is what the item could still detect: a wide interval means the
    item cannot tell a defect from a solver difference.
  'agree_under_other_reading': the item passes tier 1 or 2 only under R6's or R12's other reading.
  'differs': none of these; neither the solver nor a flagged reading explains the difference.
  The thresholds were set from the geometry above and revised twice, before any comparison with the run's reports
    (which existed by then but were not read), after checks against the
    frozen scorer run as a black box (the review's scripts; 116,000 points on synthetic cylinders of radius 60 and
    300 sv with node noise 0 to 0.3 sv, with and without holes, and on 4 Paris 4 G0 strips as surfaces, up to 35 sv
    off the mesh; no PHerc0139 or PHerc0332 data). The first check found 369 nodes this file called firmly
    uncovered that the scorer covered, far off noisy meshes (54 still so after the inner-crease rule, all with the
    foot on the mesh edge), 5 robust nodes off by 0.007 to 0.026 sv, and 3 sensitive nodes up to 0.19 sv outside
    their interval. Hence the inner-crease rule, the edge residual of 0.35 x the distance, the relative allowances
    and the asymmetric delta. With the final rules no point of those checks violates them (worst robust difference
    0.026 sv; scorer feet up to 6.9 % of the distance farther than the closest point; covered edge feet with residuals
    up to 0.21 x the distance). A later out-of-sample review on other shapes (folds, wrinkles, a spiral turn of more
    than 360 deg, sphere caps, ragged and collapsed edges; 54,000 points) found points outside these rules: 42 of
    13,424 robust nodes outside the allowance and 1 robust node uncovered (all on wrinkled meshes), 19 firmly
    uncovered nodes that the scorer covered (8 on wrinkled meshes, 11 on the spiral turn: its local solver stopped
    on another wrinkle or layer, farther than the closest point), 574 sensitive nodes outside their interval and 2
    with the wrong sign on a 90 deg fold; and, on wrinkled meshes, 1 false 'differs' of the ground-truth check in
    295 to 620 stretches (a median |d| 0.003 sv outside its interval). Whole pieces still agreed or agreed within
    the bounds (27 of 27). So a ground-truth 'differs' of a few thousandths of a sv on a wrinkled mesh, or scorer
    coverage of nodes this file calls firmly uncovered where another layer or wrinkle is within reach, can be a
    solver difference. On the first review's 42 per-piece cases 27 agree and 15 agree within the solver bounds: the
    mean's interval is 0.07 to 0.23 sv wide on good strips and up to 8.3 sv on strips tilted 40 deg. Of its 8
    crossings, 3 agree and 5 agree within the bounds. Its three edge cases agree: a strip with no row in a window, a
    crossing strip 8 sv off its fiber, and no primary fiber scored.

Memory: each segment is read once and cropped to the box of the points projected on it plus 40 sv; the projection
runs in batches. Peak memory is a few hundred MB (CPU only; no torch, no GPU).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '2')

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

TOOL = 'verification/independent_rescore.py'
TOOL_VERSION = '1.1.1'

# ---- PREREG v3.2 numbers, from its text (section in brackets) ---------------------------------------------------
L0_PER_SV = 4.0          # [Units] sv = level 2 = 4 base (level-0) voxels
SAMPLE_STEP_SV = 1.0     # [4.1] fibers are sampled every 1 sv
K_NEAREST = 4            # [5.1] Gauss-Newton from the 4 nearest nodes
SIDEWAYS_MAX_SV = 1.0    # [5.1] the foot is off the valid mesh when the sideways residual is above 1 sv
REACH_SV = 30.0          # [5.1] |d| > 30 sv: uncovered
ROW_WINDOW_SV = 30.0     # [5.2] a row is scored only when its centre is within 30 sv of its fiber
ASSOC_MAX_SV = 5.0       # [5.2] a strip runs along its fiber when its median centre distance is <= 5 sv
MIN_FIBER_NODES = 25     # [5.2] a fiber with fewer covered nodes is not scored (pooled numbers, R10)
CROSS_RADIUS_SV = 30.0   # [7] strip nodes within 30 sv of the crossing point
CROSS_MIN_NODES = 5      # [7] the 5-node rule
CROSS_TOL_SV = 6.0       # [7] the crossing rule: nearest V sample within 6 sv
TIGHT_SV = 3.0           # [7] the tight subset
# ---- this tool (TOLERANCE AND VERDICTS) ---------------------------------------------------------------------------
TOL_SV = 1e-3            # tier 1: values in sv agree within 1e-3 sv
ROBUST_D_SV = 6.0        # robust: |d| <= 6 sv for every normal at the foot
ROBUST_SIDE_SV = 0.5     # robust: sideways residual <= 0.5 sv for every normal at the foot
ROBUST_SPREAD_SV = 1e-3  # robust: |d| varies by <= 1e-3 sv between the normals at the foot
ROBUST_SLACK_SV = 0.005  # tier 2: allowance on a robust node's |d|: 0.005 sv + 2 % of |d|
ROBUST_SLACK_REL = 0.02
SENSITIVE_SIDE_SV = 1.5  # sensitive: a foot on the mesh edge with a sideways residual <= max(1.5 sv, 0.35 x the
SENSITIVE_SIDE_REL = 0.35  # distance to the mesh), or a foot inside the mesh (an inner crease or node: any residual)
SLACK_ABS_SV = 0.1       # tier 2: a sensitive node's |d| lies in [least - delta_below, distance + delta_above],
SLACK_REL_BELOW = 0.02   # delta_below = 0.1 sv + 2 % of the distance, delta_above = 0.1 sv + 10 % of the distance
SLACK_REL_ABOVE = 0.10
EDGE_EPS = 1e-9          # a foot parameter within this of 0 or 1 lies on a cell edge
CROP_MARGIN_SV = 40.0    # a segment is cropped to the points' box plus this (30 sv reach + 2 cell diagonals)
CELL_RING = 2            # cells searched around each nearest node: index offsets -2 .. +1 (two cells each way)
GN_ITERATIONS = 60
BATCH = 4096
ABS_D_BINS = ((0., 1.), (1., 2.), (2., 6.), (6., 12.), (12., 30.))
VERDICTS = ('agree', 'agree_within_solver_bounds', 'agree_under_other_reading', 'differs')

PLAN_FILES = {'secondary': 'results/matched_stretches_secondary_run40.json',
              'primary': 'results/matched_stretches.json',
              '0332': 'results/crossings_0332_plan.json'}
REPORT_FILES = {'primary': 'primary_case_studies.json', 'secondary': 'secondary_report.json',
                '0332': 'crossings_0332_report.json'}
UMBILICUS_0139 = 'data/umbilicus/PHerc0139_20260102150214_umbilicus.json'   # level 0, as run_all.ps1 passes it
FIBER_DIRS = {'0139': 'data/fibers0139', '0332': 'data/fibers0332'}


# ---- small helpers ---------------------------------------------------------------------------------------------

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(x):
    """numpy values, tuples and paths to plain JSON values; NaN and inf to None."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def _doc_block(start, stop):
    doc = __doc__ or ''
    a = doc.find(start)
    b = doc.find(stop, a + 1)
    return doc[a:b] if a >= 0 and b > a else ''


def readings():
    """R1..R12 of the docstring, as {id: text}."""
    items, cur = {}, None
    for line in _doc_block('READINGS OF THE PREREG TEXT', 'TOLERANCE AND VERDICTS (').splitlines()[1:]:
        m = re.match(r'^  (R\d+)\s+(.*)$', line)
        if m:
            cur = m.group(1)
            items[cur] = m.group(2).strip()
        elif cur and line.strip():
            items[cur] += ' ' + line.strip()
    return items


def tolerance_text():
    lines = _doc_block('TOLERANCE AND VERDICTS (', 'Memory:').splitlines()[1:]
    return ' '.join(s.strip() for s in lines if s.strip())


def _num(mine, theirs, tol):
    """One compared value (tier 1): exact for tol 0, else |mine - scorer| <= tol. None on both sides agrees."""
    if mine is None or theirs is None:
        return dict(mine=mine, scorer=theirs, diff=None, agree=mine is None and theirs is None)
    diff = float(mine) - float(theirs)
    return dict(mine=mine, scorer=theirs, diff=diff, agree=bool(abs(diff) <= tol))


def _bounds(interval, value, tol):
    """Tier 2 of one value: its interval under the solver hypothesis, and whether the scorer's value lies in it."""
    if interval is None:
        return dict(lo=None, hi=None, width=None, within=False)
    lo, hi = float(interval[0]), float(interval[1])
    ok = value is not None and lo - tol <= float(value) <= hi + tol
    return dict(lo=lo, hi=hi, width=hi - lo, within=bool(ok))


def verdict_of(fields):
    """'agree' when every field passes tier 1; 'agree_within_solver_bounds' when every field passes tier 1 or lies in
    its tier-2 interval; else 'differs'."""
    if all(f['agree'] for f in fields.values()):
        return 'agree'
    if all(f['agree'] or (f.get('bounds') or {}).get('within', False) for f in fields.values()):
        return 'agree_within_solver_bounds'
    return 'differs'


def _set_verdict(it, verdict):
    it.update(verdict=verdict, agree=verdict == 'agree', ok=verdict != 'differs')
    return it


# ---- human fibers ------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Fiber:
    """A human fiber in sv: samples ``pts`` (N, 3) at arclengths ``s`` (N,) of its controlled span (R1)."""
    name: str
    family: str
    pts: np.ndarray
    s: np.ndarray
    line_range: tuple = None
    _tree: object = dataclasses.field(default=None, repr=False)

    @property
    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.pts)
        return self._tree

    @property
    def length(self):
        return float(self.s[-1])


def _vc3d_point(p):
    """A VC3D point: [x, y, z] or {'position': [x, y, z]}."""
    if isinstance(p, dict):
        p = p['position']
    return float(p[0]), float(p[1]), float(p[2])


def controlled_span(line, controls):
    """(first, last) line index of the controlled span, or None (R1; tools/fibers_json_to_afv.py kept_points)."""
    if len(line) < 2 or len(controls) < 2:
        return None
    anchors = []
    for c in controls:
        diff = line - c
        anchors.append(int(np.argmin(np.einsum('ij,ij->i', diff, diff))))
    first, last = min(anchors), max(anchors)
    return (first, last) if last > first else None


def sample_polyline(line, step=SAMPLE_STEP_SV):
    """Points every ``step`` of arclength from the start of a polyline, plus its end point (R1): (pts, s)."""
    line = np.asarray(line, np.float64).reshape(-1, 3)
    if len(line) < 2:
        return line.copy(), np.zeros(len(line))
    diff = np.diff(line, axis=0)
    cum = np.concatenate(([0.], np.cumsum(np.sqrt(np.einsum('ij,ij->i', diff, diff)))))
    total = float(cum[-1])
    s = np.arange(int(math.floor(total/step)) + 1, dtype=np.float64)*step
    if total - s[-1] > 1e-9:
        s = np.append(s, total)
    i = np.clip(np.searchsorted(cum, s, side='right') - 1, 0, len(line) - 2)
    span = cum[i + 1] - cum[i]
    t = np.zeros_like(s)
    np.divide(s - cum[i], span, out=t, where=span > 0)
    t = np.clip(t, 0., 1.)
    pts = line[i] + t[:, None]*(line[i + 1] - line[i])
    end = t >= 1.
    pts[end] = line[i[end] + 1]
    return pts, s


def read_fiber(path, span='controlled'):
    """A VC3D fiber JSON (v3/v4) as (Fiber in sv, None), or (None, reason)."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding='utf-8'))
    line = np.array([_vc3d_point(p) for p in raw.get('line_points') or []], np.float64).reshape(-1, 3)/L0_PER_SV
    controls = np.array([_vc3d_point(p) for p in raw.get('control_points') or []],
                        np.float64).reshape(-1, 3)/L0_PER_SV
    hv = raw.get('hv_classification') or {}
    manual, auto = hv.get('manual_tag') or '', hv.get('automatic_tag') or ''
    family = manual if manual in ('H', 'V') else auto if auto in ('H', 'V') else '?'
    if span == 'controlled':
        rng = controlled_span(line, controls)
        if rng is None:
            return None, 'no controlled span (fewer than 2 control points or line points)'
    else:
        if len(line) < 2:
            return None, 'fewer than 2 line points'
        rng = (0, len(line) - 1)
    pts, s = sample_polyline(line[rng[0]:rng[1] + 1])
    if len(s) < 2:
        return None, 'span shorter than one sample step'
    return Fiber(path.stem, family, pts, s, rng), None


def load_fiber_dir(folder, names=None):
    """{name: Fiber} for the JSONs of a folder (only ``names`` when given), and {name: reason} of skipped ones."""
    fibers, skipped = {}, {}
    for p in sorted(Path(folder).glob('*.json')):
        if names is not None and p.stem not in names:
            continue
        try:
            fib, why = read_fiber(p)
        except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
            fib, why = None, f'unreadable ({type(error).__name__}: {error})'
        if fib is None:
            skipped[p.stem] = why
        else:
            fibers[p.stem] = fib
    return fibers, skipped


class Axis:
    """The umbilicus: control points xyz (sv), linear in z between them and constant beyond (R8)."""

    def __init__(self, xyz_sv):
        xyz = np.asarray(xyz_sv, np.float64).reshape(-1, 3)
        self.xyz = xyz[np.argsort(xyz[:, 2], kind='stable')]

    @classmethod
    def from_json(cls, path, l0_per_unit=1.0):
        cps = json.loads(Path(path).read_text(encoding='utf-8'))['control_points']
        return cls(np.array([[c['x'], c['y'], c['z']] for c in cps], np.float64)*(l0_per_unit/L0_PER_SV))

    def xy(self, z):
        z = np.asarray(z, np.float64)
        zs = self.xyz[:, 2]
        return np.stack([np.interp(z, zs, self.xyz[:, 0]), np.interp(z, zs, self.xyz[:, 1])], axis=-1)

    def xy_nearest(self, z):
        """R8's check: the umbilicus at the control point nearest in z (another interpolation)."""
        z = np.asarray(z, np.float64)
        zs = self.xyz[:, 2]
        if len(zs) == 1:
            return np.broadcast_to(self.xyz[0, :2], z.shape + (2,)).copy()
        hi = np.clip(np.searchsorted(zs, z), 1, len(zs) - 1)
        lo = hi - 1
        return self.xyz[np.where(np.abs(z - zs[lo]) <= np.abs(zs[hi] - z), lo, hi), :2]


# ---- tifxyz surfaces ---------------------------------------------------------------------------------------------

def read_tifxyz(folder, l0_per_unit, bbox=None, margin=0.):
    """A tifxyz folder (x.tif, y.tif, z.tif) as (nodes (R, C, 3) in sv, NaN at invalid nodes; valid (R, C)).
    ``l0_per_unit``: base voxels per file unit (segments 1, strips 4). A node is invalid when a coordinate is -1 or
    not finite. With ``bbox`` (lo, hi in sv) the grid is cut to the index rectangle of the valid nodes inside the box
    grown by ``margin``, plus one node of context on each side."""
    import tifffile
    folder = Path(folder)
    chans = [np.asarray(tifffile.imread(str(folder/f'{c}.tif'))) for c in 'xyz']
    if chans[0].ndim != 2 or not chans[0].shape == chans[1].shape == chans[2].shape:
        raise ValueError(f'{folder}: x/y/z.tif are not three 2-D arrays of one shape')
    valid = np.ones(chans[0].shape, bool)
    for a in chans:
        valid &= np.isfinite(a) & (a != -1)
    scale = l0_per_unit/L0_PER_SV
    if bbox is not None:
        lo = np.asarray(bbox[0], np.float64) - margin
        hi = np.asarray(bbox[1], np.float64) + margin
        inside = valid.copy()
        for k in range(3):
            v = chans[k].astype(np.float64)*scale
            with np.errstate(invalid='ignore'):
                inside &= (v >= lo[k]) & (v <= hi[k])
            del v
        if not inside.any():
            return np.zeros((0, 0, 3)), np.zeros((0, 0), bool)
        rows, cols = np.flatnonzero(inside.any(axis=1)), np.flatnonzero(inside.any(axis=0))
        r0, r1 = max(int(rows[0]) - 1, 0), min(int(rows[-1]) + 2, valid.shape[0])
        c0, c1 = max(int(cols[0]) - 1, 0), min(int(cols[-1]) + 2, valid.shape[1])
        chans = [a[r0:r1, c0:c1] for a in chans]
        valid = valid[r0:r1, c0:c1]
    grid = np.stack([a.astype(np.float64)*scale for a in chans], axis=-1)
    grid[~valid] = np.nan
    return grid, valid


def strip_tifxyz_dir(piece_dir):
    """The strip tifxyz folder of an infer.py OUT dir (OUT/strip), or the folder itself when it holds x.tif."""
    d = Path(piece_dir)
    for cand in (d/'strip', d):
        if (cand/'x.tif').is_file():
            return cand
    return None


def read_strip(strip_dir):
    """A strip tifxyz (patch voxels = sv) as (nodes (R, C, 3), valid (R, C), info), rows along the fiber."""
    d = Path(strip_dir)
    grid, valid = read_tifxyz(d, L0_PER_SV)
    info = {}
    for cand in (d/'trace.json', d.parent/'summary.json'):
        if cand.is_file():
            for key, value in json.loads(cand.read_text(encoding='utf-8')).items():
                info.setdefault(key, value)
    if info.get('fiber_along') == 'columns':      # infer_all's display layout: back to rows along the fiber
        grid, valid = np.ascontiguousarray(np.swapaxes(grid, 0, 1)), np.ascontiguousarray(valid.T)
    return grid, valid, info


@dataclasses.dataclass
class Projection:
    found: np.ndarray    # a foot exists (a valid cell near the point)
    d: np.ndarray        # signed distance along the normal at the foot (oriented outward with an umbilicus)
    side: np.ndarray     # sideways residual
    foot: np.ndarray
    normal: np.ndarray
    dist: np.ndarray = None      # |p - foot|
    d_lo: np.ndarray = None      # d over the normals of every valid cell at the foot (R7): smallest ... largest
    d_hi: np.ndarray = None
    side_lo: np.ndarray = None   # the sideways residual over the same normals
    side_hi: np.ndarray = None
    crease: np.ndarray = None    # the foot lies on an edge or node shared with another valid cell
    boundary: np.ndarray = None  # the foot lies on an edge of the valid mesh (no valid cell across it)
    sign_alt: np.ndarray = None  # R8 check: the outward sign differs with the umbilicus at the nearest control point

    def __post_init__(self):
        n = len(self.found)
        for name in ('dist', 'd_lo', 'd_hi', 'side_lo', 'side_hi'):
            if getattr(self, name) is None:
                setattr(self, name, np.full(n, np.nan))
        for name in ('crease', 'boundary', 'sign_alt'):
            if getattr(self, name) is None:
                setattr(self, name, np.zeros(n, bool))

    @classmethod
    def empty(cls, n):
        return cls(np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan), np.full((n, 3), np.nan),
                   np.full((n, 3), np.nan))

    @classmethod
    def concat(cls, parts):
        if not parts:
            return cls.empty(0)
        return cls(*[np.concatenate([getattr(p, f.name) for p in parts]) for f in dataclasses.fields(cls)])

    def covered(self):
        """5.1 (R7): a foot exists, sideways residual <= 1 sv, |d| <= 30 sv."""
        with np.errstate(invalid='ignore'):
            return self.found & (self.side <= SIDEWAYS_MAX_SV) & (np.abs(self.d) <= REACH_SV)

    def take(self, sl):
        return Projection(*[getattr(self, f.name)[sl] for f in dataclasses.fields(self)])


def uncovered_reason(pr):
    """Per point: 0 covered, 1 no foot, 2 sideways residual above 1 sv, 3 |d| above 30 sv."""
    code = np.zeros(len(pr.found), np.int8)
    with np.errstate(invalid='ignore'):
        side_ok = pr.side <= SIDEWAYS_MAX_SV
        code[~pr.found] = 1
        code[pr.found & ~side_ok] = 2
        code[pr.found & side_ok & ~(np.abs(pr.d) <= REACH_SV)] = 3
    return code


def _reason_counts(code):
    code = np.asarray(code)
    return dict(no_foot=int((code == 1).sum()), sideways_over_1sv=int((code == 2).sum()),
                beyond_30sv=int((code == 3).sum()))


def closest_on_cells(q, p00, p01, p10, p11, iterations=GN_ITERATIONS):
    """Closest point to q[m] of bilinear cell m, S(a, b) = p00 + a (p10 - p00) + b (p01 - p00) + a b t,
    a, b in [0, 1]: Gauss-Newton for an interior stationary point, and the exact closest point of each of the 4
    straight edges; the nearest wins. Returns (a, b, foot, distance)."""
    ea, eb = p10 - p00, p01 - p00
    tw = p11 - p10 - p01 + p00
    m = len(q)
    a, b = np.full(m, .5), np.full(m, .5)
    conv = np.zeros(m, bool)
    for _ in range(iterations):
        sa = ea + b[:, None]*tw
        sb = eb + a[:, None]*tw
        r = q - (p00 + a[:, None]*ea + b[:, None]*eb + (a*b)[:, None]*tw)
        h11, h12, h22 = (np.einsum('ij,ij->i', sa, sa), np.einsum('ij,ij->i', sa, sb),
                         np.einsum('ij,ij->i', sb, sb))
        g1, g2 = np.einsum('ij,ij->i', sa, r), np.einsum('ij,ij->i', sb, r)
        det = h11*h22 - h12*h12
        ok = det > 1e-12*h11*h22
        safe = np.where(ok, det, 1.)
        da = np.where(ok, (h22*g1 - h12*g2)/safe, 0.)
        db = np.where(ok, (h11*g2 - h12*g1)/safe, 0.)
        a, b = np.clip(a + da, -1., 2.), np.clip(b + db, -1., 2.)
        conv = ok & (np.maximum(np.abs(da), np.abs(db)) < 1e-12)
        if conv.all():
            break
    eps = 1e-12
    inner = conv & (a >= -eps) & (a <= 1. + eps) & (b >= -eps) & (b <= 1. + eps)
    a, b = np.clip(a, 0., 1.), np.clip(b, 0., 1.)
    best_f = p00 + a[:, None]*ea + b[:, None]*eb + (a*b)[:, None]*tw
    best_d = np.where(inner, np.linalg.norm(q - best_f, axis=1), np.inf)
    best_a, best_b = a.copy(), b.copy()
    for start, stop, t_is_a, fixed in ((p00, p10, True, 0.), (p01, p11, True, 1.),
                                       (p00, p01, False, 0.), (p10, p11, False, 1.)):
        e = stop - start
        ee = np.einsum('ij,ij->i', e, e)
        t = np.zeros(m)
        np.divide(np.einsum('ij,ij->i', q - start, e), ee, out=t, where=ee > 0)
        t = np.clip(t, 0., 1.)
        f = start + t[:, None]*e
        f[t >= 1.] = stop[t >= 1.]          # the corners exactly
        f[t <= 0.] = start[t <= 0.]
        dist = np.linalg.norm(q - f, axis=1)
        better = dist < best_d
        best_d = np.where(better, dist, best_d)
        best_f[better] = f[better]
        if t_is_a:
            best_a, best_b = np.where(better, t, best_a), np.where(better, fixed, best_b)
        else:
            best_a, best_b = np.where(better, fixed, best_a), np.where(better, t, best_b)
    return best_a, best_b, best_f, best_d


class BilinearSurface:
    """The bilinear surface of a tifxyz grid: nodes (R, C, 3) in sv and their validity; a cell is valid when its
    four corners are."""

    def __init__(self, grid, valid, name=''):
        self.name = name
        g = np.asarray(grid, np.float64)
        if g.size == 0:
            g = np.zeros((0, 0, 3))
        ok = np.asarray(valid, bool).reshape(g.shape[:2]) & np.isfinite(g).all(axis=-1)
        self.grid = np.where(ok[..., None], g, 0.)
        self.valid = ok
        self.shape = ok.shape
        rows, cols = self.shape
        if rows >= 2 and cols >= 2:
            self.cell_ok = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
        else:
            self.cell_ok = np.zeros((max(rows - 1, 0), max(cols - 1, 0)), bool)
        self.nodes = np.argwhere(ok)
        self.tree = cKDTree(self.grid[ok]) if len(self.nodes) and self.cell_ok.any() else None

    def project(self, points, axis=None):
        """Projection of points (N, 3) in sv (R7); with ``axis`` the sign of d is outward (R8)."""
        p = np.asarray(points, np.float64).reshape(-1, 3)
        out = Projection.empty(len(p))
        if self.tree is None or not len(p):
            return out
        k = min(K_NEAREST, len(self.nodes))
        for b0 in range(0, len(p), BATCH):
            self._batch(p[b0:b0 + BATCH], k, out, b0)
        if axis is not None and out.found.any():
            hit = np.flatnonzero(out.found)
            foot, nh = out.foot[hit], out.normal[hit, :2]
            sign = np.sign(np.einsum('ij,ij->i', nh, foot[:, :2] - axis.xy(foot[:, 2])))
            sign[sign == 0] = 1.
            alt = np.sign(np.einsum('ij,ij->i', nh, foot[:, :2] - axis.xy_nearest(foot[:, 2])))
            alt[alt == 0] = 1.
            out.sign_alt[hit] = alt != sign
            out.normal[hit] *= sign[:, None]
            out.d[hit] *= sign
            lo, hi = out.d_lo[hit]*sign, out.d_hi[hit]*sign
            out.d_lo[hit], out.d_hi[hit] = np.minimum(lo, hi), np.maximum(lo, hi)
        return out

    def _normals(self, i, j, a, b):
        """Unit normals of cells (i, j) at (a, b), and where they exist."""
        g = self.grid
        p00, p01, p10, p11 = g[i, j], g[i, j + 1], g[i + 1, j], g[i + 1, j + 1]
        tw = p11 - p10 - p01 + p00
        n = np.cross((p10 - p00) + b[:, None]*tw, (p01 - p00) + a[:, None]*tw)
        nn = np.linalg.norm(n, axis=1)
        ok = nn > 1e-12
        return n/np.where(ok, nn, 1.)[:, None], ok

    def _batch(self, p, k, out, offset):
        rows, cols = self.shape
        _, idx = self.tree.query(p, k=k)
        idx = np.asarray(idx).reshape(len(p), k)
        nij = self.nodes[idx]                                            # (B, k, 2) the k nearest valid nodes
        off = np.arange(-CELL_RING, CELL_RING)
        oi, oj = np.meshgrid(off, off, indexing='ij')
        ci = (nij[:, :, 0, None] + oi.ravel()).reshape(len(p), -1)        # candidate cells (lower-left corner)
        cj = (nij[:, :, 1, None] + oj.ravel()).reshape(len(p), -1)
        good = (ci >= 0) & (ci < rows - 1) & (cj >= 0) & (cj < cols - 1)
        good[good] = self.cell_ok[ci[good], cj[good]]
        key = np.full(ci.shape, -1, np.int64)
        key[good] = ci[good].astype(np.int64)*(cols - 1) + cj[good]
        key.sort(axis=1)
        first = np.ones(key.shape, bool)
        first[:, 1:] = key[:, 1:] != key[:, :-1]
        pi, col = np.nonzero((key >= 0) & first)
        if not len(pi):
            return
        kk = key[pi, col]
        i, j = kk//(cols - 1), kk % (cols - 1)
        g = self.grid
        a, b, f, dist = closest_on_cells(p[pi], g[i, j], g[i, j + 1], g[i + 1, j], g[i + 1, j + 1])
        order = np.lexsort((dist, pi))
        best = order[np.r_[0, np.flatnonzero(np.diff(pi[order])) + 1]]   # nearest candidate of each point
        bp, i, j, a, b, f = pi[best], i[best], j[best], a[best], b[best], f[best]
        n, ok = self._normals(i, j, a, b)
        r = p[bp] - f
        d = np.einsum('ij,ij->i', r, n)
        side = np.linalg.norm(r - d[:, None]*n, axis=1)
        # R7: at a foot on a cell edge or node, the normals of the other valid cells there (a crease): all of them are
        # kept as a range, and the one with the smallest sideways residual gives d
        d_lo, d_hi, s_lo, s_hi = d.copy(), d.copy(), side.copy(), side.copy()
        crease, boundary = np.zeros(len(bp), bool), np.zeros(len(bp), bool)
        on_a = {-1: a <= EDGE_EPS, 1: a >= 1. - EDGE_EPS}
        on_b = {-1: b <= EDGE_EPS, 1: b >= 1. - EDGE_EPS}
        for di, dj in itertools.product((-1, 0, 1), (-1, 0, 1)):
            if di == 0 and dj == 0:
                continue
            m = ok.copy()
            if di:
                m &= on_a[di]
            if dj:
                m &= on_b[dj]
            if not m.any():
                continue
            ii, jj = i + di, j + dj
            cell = m & (ii >= 0) & (ii < rows - 1) & (jj >= 0) & (jj < cols - 1)
            cell[cell] = self.cell_ok[ii[cell], jj[cell]]
            if di == 0 or dj == 0:                                      # across an edge: none there = mesh edge
                boundary |= m & ~cell
            if not cell.any():
                continue
            crease |= cell
            sel = np.flatnonzero(cell)
            aa = a[sel] if di == 0 else np.full(len(sel), 1. if di == -1 else 0.)
            bb = b[sel] if dj == 0 else np.full(len(sel), 1. if dj == -1 else 0.)
            n2, ok2 = self._normals(ii[sel], jj[sel], aa, bb)
            n2 *= np.where(np.einsum('ij,ij->i', n2, n[sel]) < 0., -1., 1.)[:, None]
            d2 = np.einsum('ij,ij->i', r[sel], n2)
            s2 = np.linalg.norm(r[sel] - d2[:, None]*n2, axis=1)
            sel, d2, s2, n2 = sel[ok2], d2[ok2], s2[ok2], n2[ok2]
            d_lo[sel], d_hi[sel] = np.minimum(d_lo[sel], d2), np.maximum(d_hi[sel], d2)
            s_lo[sel], s_hi[sel] = np.minimum(s_lo[sel], s2), np.maximum(s_hi[sel], s2)
            swap = s2 < side[sel] - 1e-12
            sw = sel[swap]
            d[sw], side[sw], n[sw] = d2[swap], s2[swap], n2[swap]
        tgt = offset + bp[ok]
        out.found[tgt] = True
        out.d[tgt], out.side[tgt], out.foot[tgt], out.normal[tgt] = d[ok], side[ok], f[ok], n[ok]
        out.dist[tgt] = np.linalg.norm(r[ok], axis=1)
        out.d_lo[tgt], out.d_hi[tgt], out.side_lo[tgt], out.side_hi[tgt] = d_lo[ok], d_hi[ok], s_lo[ok], s_hi[ok]
        out.crease[tgt], out.boundary[tgt] = crease[ok], boundary[ok]


class Requests:
    """Points to project, grouped by segment, so that each segment is read once (cropped to its points)."""

    def __init__(self):
        self.items = {}

    def add(self, segment, key, points):
        self.items.setdefault(segment, []).append((key, np.asarray(points, np.float64).reshape(-1, 3)))

    def run(self, open_segment, axis=None, log=None):
        results, info = {}, {}
        for seg in sorted(self.items):
            group = self.items[seg]
            allp = np.concatenate([p for _, p in group])
            if not len(allp):
                for key, _ in group:
                    results[key] = Projection.empty(0)
                continue
            t0 = time.time()
            surf, note = open_segment(seg, allp)
            if surf is None:
                for key, _ in group:
                    results[key] = None
                info[seg] = dict(points=int(len(allp)), error=note)
                if log:
                    log(f'segment {seg}: {note}')
                continue
            pr = surf.project(allp, axis)
            o = 0
            for key, pts in group:
                results[key] = pr.take(slice(o, o + len(pts)))
                o += len(pts)
            info[seg] = dict(points=int(len(allp)), crop_shape=list(surf.shape),
                             crop_valid_nodes=int(surf.valid.sum()), covered=int(pr.covered().sum()), note=note,
                             seconds=round(time.time() - t0, 2))
            if log:
                log(f'segment {seg}: {len(allp)} points, crop {surf.shape}, {info[seg]["seconds"]} s')
            del surf, pr
        return results, info


def segment_opener(folders, l0_per_unit=1.0, margin=CROP_MARGIN_SV):
    """open_segment(name, points) -> (BilinearSurface cropped to the points' box + margin, note) or (None, error),
    for tifxyz folders {name: folder} (a folder holding x.tif, or a tree holding exactly one)."""
    def open_segment(name, pts):
        folder = folders.get(name)
        if folder is None:
            return None, f'segment {name} is not in the segment list'
        folder = Path(folder)
        if not (folder/'x.tif').is_file():
            found = sorted({p.parent for p in folder.rglob('x.tif')}) if folder.is_dir() else []
            if len(found) != 1:
                return None, f'segment {name}: {len(found)} tifxyz folders under {folder}'
            folder = found[0]
        grid, valid = read_tifxyz(folder, l0_per_unit, (pts.min(axis=0), pts.max(axis=0)), margin)
        note = None if valid.any() else f'no valid node within {margin} sv of the points'
        return BilinearSurface(grid, valid, name), note
    return open_segment


# ---- tier 2: solver sensitivity (TOLERANCE AND VERDICTS) ----------------------------------------------------------

def classify(pr):
    """Per point: 'robust', 'sensitive' (another solver of 5.1 may cover it or not, with |d| in [lo, hi] and d in
    [dlo, dhi]) or neither (firmly uncovered); see TOLERANCE AND VERDICTS."""
    with np.errstate(invalid='ignore'):
        alo = np.where(pr.d_lo*pr.d_hi > 0., np.minimum(np.abs(pr.d_lo), np.abs(pr.d_hi)), 0.)
        ahi = np.maximum(np.abs(pr.d_lo), np.abs(pr.d_hi))
        below = SLACK_ABS_SV + SLACK_REL_BELOW*pr.dist
        above = SLACK_ABS_SV + SLACK_REL_ABOVE*pr.dist
        cov = pr.covered()
        robust = cov & (ahi <= ROBUST_D_SV) & (pr.side_hi <= ROBUST_SIDE_SV) & (ahi - alo <= ROBUST_SPREAD_SV)
        # a covering solver's foot is no nearer than this file's (the closest point), and its residual is <= 1 sv
        least = np.sqrt(np.maximum(pr.dist**2 - SIDEWAYS_MAX_SV**2, 0.))
        edge_ok = pr.side_lo <= np.maximum(SENSITIVE_SIDE_SV, SENSITIVE_SIDE_REL*pr.dist)
        maybe = pr.found & (least - below <= REACH_SV) & (edge_ok | ~pr.boundary)
        sensitive = maybe & ~robust
        lo = np.clip(np.minimum(alo, least) - below, 0., REACH_SV)
        hi = np.clip(pr.dist + above, 0., REACH_SV)
        pos, neg = pr.d_lo > 0., pr.d_hi < 0.                 # the side of the mesh, when every normal agrees
        return dict(covered=cov, robust=robust, sensitive=sensitive, abs_d=np.abs(pr.d), d=pr.d, alo=alo, ahi=ahi,
                    lo=lo, hi=hi, dlo=np.where(pos, lo, -hi), dhi=np.where(neg, -lo, hi))


def robust_slack(abs_d):
    """Tier 2's allowance on a robust node's |d| (and d): 0.005 sv + 2 % of |d|."""
    return ROBUST_SLACK_SV + ROBUST_SLACK_REL*np.asarray(abs_d, np.float64)


def _merge_cls(parts):
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]} if parts else None


def tier2_ingredients(cls, mask=None, quantiles=False):
    """What tier 2 needs of a set of points: the robust count and sums (and sorted |d| for quantiles), and the
    sensitive intervals sorted for the subset bounds (lo, dlo ascending; hi, dhi descending)."""
    n = len(cls['covered'])
    m = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    rob, sen = cls['robust'] & m, cls['sensitive'] & m
    out = dict(n_robust=int(rob.sum()), robust_abs_sum=float(cls['abs_d'][rob].sum()),
               robust_signed_sum=float(cls['d'][rob].sum()),
               lo=np.sort(cls['lo'][sen]), hi=np.sort(cls['hi'][sen])[::-1],
               dlo=np.sort(cls['dlo'][sen]), dhi=np.sort(cls['dhi'][sen])[::-1])
    if quantiles:
        out['robust_abs'] = np.sort(cls['abs_d'][rob])
    return out


def solver_diagnostics(pr, cls, mask=None):
    """Per block: covered nodes by |d|, the 0.5-1.5 sv sideways band, feet on creases and on the mesh edge, the
    tier-2 classes, and R8's check."""
    n = len(pr.found)
    m = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    cov, a = cls['covered'] & m, cls['abs_d']
    with np.errstate(invalid='ignore'):
        bins = {f'{lo:g}-{hi:g}': int((cov & ((a >= lo) if lo == 0. else (a > lo)) & (a <= hi)).sum())
                for lo, hi in ABS_D_BINS}
        band = int((m & pr.found & (pr.side > ROBUST_SIDE_SV) & (pr.side <= SENSITIVE_SIDE_SV)).sum())
        spread = int((m & pr.found & (cls['ahi'] - cls['alo'] > ROBUST_SPREAD_SV)).sum())
    rob, sen = cls['robust'] & m, cls['sensitive'] & m
    return dict(points=int(m.sum()), covered=int(cov.sum()), covered_by_abs_d_sv=bins, sideways_0p5_to_1p5_sv=band,
                feet_on_crease=int((m & pr.crease).sum()), feet_on_mesh_edge=int((m & pr.boundary).sum()),
                crease_spread_over_0p001_sv=spread, robust=int(rob.sum()), sensitive_covered=int((sen & cov).sum()),
                sensitive_uncovered=int((sen & ~cov).sum()), firmly_uncovered=int((m & ~rob & ~sen).sum()),
                outward_sign_differs_with_nearest_control_point=int((cov & pr.sign_alt).sum()))


def _arr(x):
    return np.asarray(x if x is not None else [], np.float64)


def count_interval(ing):
    return ing['n_robust'], ing['n_robust'] + len(ing['lo'])


def mean_interval(ing, n_total, signed=False):
    """[lo, hi] of a mean over ``n_total`` covered nodes under tier 2: the robust ones and n_total - n_robust of the
    sensitive ones; None when n_total is 0 or outside the count interval."""
    lo_s, hi_s = (_arr(ing['dlo']), _arr(ing['dhi'])) if signed else (_arr(ing['lo']), _arr(ing['hi']))
    nr = ing['n_robust']
    k = int(n_total) - nr
    if n_total <= 0 or k < 0 or k > len(lo_s):
        return None
    base = ing['robust_signed_sum' if signed else 'robust_abs_sum']
    e = ROBUST_SLACK_SV*nr + ROBUST_SLACK_REL*ing['robust_abs_sum']
    return (base - e + float(lo_s[:k].sum()))/n_total, (base + e + float(hi_s[:k].sum()))/n_total


def mean_interval_any_count(ing):
    """As mean_interval, over every count of covered sensitive nodes (the scorer does not report it)."""
    lo_s, hi_s = _arr(ing['lo']), _arr(ing['hi'])
    nr = ing['n_robust']
    n = nr + np.arange(len(lo_s) + 1)
    ok = n > 0
    if not ok.any():
        return None
    e = ROBUST_SLACK_SV*nr + ROBUST_SLACK_REL*ing['robust_abs_sum']
    lo = (ing['robust_abs_sum'] - e + np.concatenate(([0.], np.cumsum(lo_s))))[ok]/n[ok]
    hi = (ing['robust_abs_sum'] + e + np.concatenate(([0.], np.cumsum(hi_s))))[ok]/n[ok]
    return float(lo.min()), float(hi.max())


def quantile_interval(ing, k, q):
    """[lo, hi] of the q-th percentile of |d| over the robust nodes and k sensitive ones under tier 2."""
    r = _arr(ing.get('robust_abs'))
    lo_s, hi_s = _arr(ing['lo']), _arr(ing['hi'])
    if k is None or k < 0 or k > len(lo_s) or len(r) + k == 0:
        return None
    lower = np.concatenate([np.maximum(r - robust_slack(r), 0.), lo_s[:k]])
    upper = np.concatenate([r + robust_slack(r), hi_s[:k]])
    return float(np.percentile(lower, q)), float(np.percentile(upper, q))


def median_interval_any_count(ing, k_min=0):
    """The median's interval over every count k >= k_min of covered sensitive nodes."""
    ivs = [quantile_interval(ing, k, 50) for k in range(max(k_min, 0), len(ing['lo']) + 1)]
    ivs = [iv for iv in ivs if iv is not None]
    return (min(iv[0] for iv in ivs), max(iv[1] for iv in ivs)) if ivs else None


def within_interval(ing, k, t):
    """[lo, hi] of the number of covered nodes with |d| <= t under tier 2 (k sensitive nodes covered)."""
    r = _arr(ing.get('robust_abs'))
    lo_s, hi_s = _arr(ing['lo']), _arr(ing['hi'])
    if k is None or k < 0 or k > len(lo_s):
        return None
    return (int((r + robust_slack(r) <= t).sum()) + max(0, k - int((hi_s > t).sum())),
            int((r - robust_slack(r) <= t).sum()) + min(k, int((lo_s <= t).sum())))


def block_sensitivity(pr, cls, mask=None, quantiles=False):
    return dict(diagnostics=solver_diagnostics(pr, cls, mask), tier2=tier2_ingredients(cls, mask, quantiles))


# ---- PHerc0139: rows, nodes, stretches, pieces (5.2) -------------------------------------------------------------

def row_centres(grid, valid):
    """R4: per row the node of column C // 2, or the mean of the row's valid nodes when that node is invalid."""
    nrow, ncol = valid.shape
    cc = ncol//2
    has = valid.any(axis=1) if ncol else np.zeros(nrow, bool)
    centre = np.full((nrow, 3), np.nan)
    fallback = np.zeros(nrow, bool)
    if ncol:
        on = has & valid[:, cc]
        centre[on] = grid[on, cc]
        fallback = has & ~valid[:, cc]
        for r in np.flatnonzero(fallback):
            centre[r] = grid[r, valid[r]].mean(axis=0)
    return centre, has, fallback, cc


def centre_distances(centre, has, fiber):
    """R2: each row centre's distance to the nearest fiber sample, that sample and its arclength; R5: their median."""
    nrow = len(has)
    dist, k, s = np.full(nrow, np.nan), np.full(nrow, -1, np.int64), np.full(nrow, np.nan)
    if has.any():
        dd, kk = fiber.tree.query(centre[has], k=1)
        dist[has], k[has], s[has] = dd, kk, fiber.s[kk]
    assoc = float(np.median(dist[has])) if has.any() else math.nan
    return dist, k, s, assoc


def prepare_rows(grid, valid, fiber, fiber_stretches):
    """5.2 for one strip: row centres (R4), distance and arclength on the fiber (R2), association median (R5), and
    for every row the stretch of the fiber its arclength falls in (R6); scored rows are those with a stretch and a
    centre within 30 sv, in a strip that is not rejected."""
    nrow, ncol = valid.shape
    centre, has, fallback, cc = row_centres(grid, valid)
    dist, k, s, assoc = centre_distances(centre, has, fiber)
    rejected = not (assoc <= ASSOC_MAX_SV)
    ordered = sorted(fiber_stretches, key=lambda st: st['id'])
    in_stretch, multi = [None]*nrow, []
    for r in np.flatnonzero(has):
        hits = [st['id'] for st in ordered if st['s0'] <= s[r] <= st['s1']]
        if hits:
            in_stretch[r] = hits[0]
            if len(hits) > 1:
                multi.append([int(r), hits])
    with np.errstate(invalid='ignore'):
        within = has & (dist <= ROW_WINDOW_SV)
    scored = np.array([bool(within[r]) and in_stretch[r] is not None for r in range(nrow)], bool) & (not rejected)
    return dict(rows=int(nrow), cols=int(ncol), centre_col=int(cc), has_centre=has, fallback=fallback,
                centre=centre, dist=dist, k=k, s=s, assoc_median=assoc, rejected=bool(rejected),
                in_stretch=in_stretch, within=within, scored=scored, multi=multi)


def node_stats(d):
    """R9 over the finite values of signed distances d."""
    d = np.asarray(d, np.float64)
    d = d[np.isfinite(d)]
    if not len(d):
        return dict(nodes=0, mean_abs_sv=None, median_abs_sv=None, p90_abs_sv=None, within1=None, within2=None,
                    within1_nodes=0, within2_nodes=0, signed_mean_sv=None, max_abs_sv=None)
    a = np.abs(d)
    return dict(nodes=int(len(d)), mean_abs_sv=float(a.mean()), median_abs_sv=float(np.median(a)),
                p90_abs_sv=float(np.percentile(a, 90)), within1=float(np.mean(a <= 1.)),
                within2=float(np.mean(a <= 2.)), within1_nodes=int((a <= 1.).sum()),
                within2_nodes=int((a <= 2.).sum()), signed_mean_sv=float(d.mean()), max_abs_sv=float(a.max()))


def _strip_fields(stats):
    """node_stats renamed like the scorer's per_strip fields (strip_nodes, strip_mean_abs_sv, ...)."""
    return {('strip_' + k): v for k, v in stats.items()}


def stretch_base(st, set_name):
    return dict(id=st['id'], set=set_name, fiber=st['fiber'], family=st.get('family'), annotator=st.get('annotator'),
                segment=st['segment'], s0=st['s0'], s1=st['s1'], length_sv=st.get('length_sv'),
                piece=st.get('piece'))


def gt_fields(gt):
    """The ground-truth check against the plan's record of the stretch: tier 1 and, from the plan's sample count
    (every sample of a stretch was covered when the plan was made), tier 2."""
    plan = gt.get('plan') or {}
    t2 = gt.get('tier2')
    n_plan = plan.get('samples')
    k = None
    if t2 is not None and n_plan is not None:
        k = int(n_plan) - t2['n_robust']
        k = k if 0 <= k <= len(t2['lo']) else None
    f = {}
    if 'samples' in plan:
        f['samples'] = _num(gt.get('samples'), plan['samples'], 0)
        f['covered'] = _num(gt.get('covered'), plan['samples'], 0)
        if t2 is not None:
            f['covered']['bounds'] = _bounds(count_interval(t2), plan['samples'], 0)
    for mine, theirs in (('mean_d_sv', 'mean_d'), ('median_abs_d_sv', 'median_abs_d'), ('p90_abs_d_sv', 'p90_abs_d'),
                         ('max_abs_d_sv', 'max_abs_d')):
        if theirs not in plan:
            continue
        f[theirs] = _num(gt.get(mine), plan[theirs], TOL_SV)
        if t2 is None:
            continue
        if theirs == 'mean_d':
            iv = mean_interval(t2, n_plan, signed=True) if n_plan else None
        else:
            iv = quantile_interval(t2, k, {'median_abs_d': 50, 'p90_abs_d': 90, 'max_abs_d': 100}[theirs])
        f[theirs]['bounds'] = _bounds(iv, plan[theirs], TOL_SV)
    return f


def gt_record(st, pr, n_samples):
    out = dict(samples=int(n_samples))
    if pr is None:
        out['error'] = 'segment could not be read'
    else:
        cov = pr.covered()
        d = pr.d[cov]
        cls = classify(pr)
        out.update(covered=int(cov.sum()), uncovered=_reason_counts(uncovered_reason(pr)),
                   solver_sensitivity=solver_diagnostics(pr, cls), tier2=tier2_ingredients(cls, quantiles=True))
        if len(d):
            a = np.abs(d)
            out.update(mean_d_sv=float(d.mean()), median_abs_d_sv=float(np.median(a)),
                       p90_abs_d_sv=float(np.percentile(a, 90)), max_abs_d_sv=float(a.max()),
                       within_2sv=int((a <= 2.).sum()))
    plan = {k: st[k] for k in ('samples', 'mean_d', 'median_abs_d', 'p90_abs_d', 'max_abs_d') if k in st}
    if plan:
        out['plan'] = plan
        out['vs_plan'] = gt_fields(out)
        out['verdict'] = verdict_of(out['vs_plan']) if out['vs_plan'] else 'agree'
        out['agree'] = out['verdict'] == 'agree'
    return out


def piece_sensitivity(npr, cls, fpr, fcls, sid_nodes, node_mask=None, row_mask=None):
    """Diagnostics and tier-2 ingredients of a piece's covered nodes and of its fiber floor rows."""
    m = np.ones(len(npr.found), bool) if node_mask is None else node_mask
    rob, pos = cls['robust'] & m, (cls['robust'] | cls['sensitive']) & m
    out = block_sensitivity(npr, cls, m)
    out['floor'] = block_sensitivity(fpr, fcls, row_mask)
    out['stretches_with_robust_nodes'] = sorted(set(sid_nodes[rob].tolist()))
    out['stretches_with_possible_nodes'] = sorted(set(sid_nodes[pos].tolist()))
    return out


def score_pieces(set_name, jobs, fibers, scored_stretches, open_segment, axis=None, strips=True, log=None):
    """Score the strips of ``jobs`` (dicts: piece, fiber, strip_dir) of one set against the stretches of the scored
    set (plan dicts: id, fiber, segment, s0, s1, piece, ...), and run the ground-truth check of every piece's own
    stretches. Returns dict(pieces, stretches, node_d (covered signed d per piece), node_cls (tier-2 classes of the
    nodes of scored rows per piece), segments)."""
    by_fiber, by_id = {}, {}
    for st in scored_stretches:
        by_fiber.setdefault(st['fiber'], []).append(st)
        by_id[st['id']] = st
    req, work = Requests(), []
    for job in jobs:
        pid, fname = job['piece'], job['fiber']
        rec = dict(set=set_name, piece=pid, fiber=fname, notes=[])
        own = sorted((st for st in by_fiber.get(fname, []) if st.get('piece') == pid), key=lambda st: st['id'])
        rec['own_stretches'] = [st['id'] for st in own]
        w = dict(rec=rec, own=own, gt=[], groups={})
        work.append(w)
        fiber = fibers.get(fname)
        if fiber is None:
            rec['error'] = 'fiber not loaded'
            continue
        rec['family'] = fiber.family
        for st in own:
            sel = np.flatnonzero((fiber.s >= st['s0']) & (fiber.s <= st['s1']))
            w['gt'].append((st, sel))
            req.add(st['segment'], ('gt', pid, st['id']), fiber.pts[sel])
        if not strips:
            continue
        sdir = job.get('strip_dir')
        if sdir is None:
            rec['error'] = 'no strip tifxyz for this piece'
            continue
        grid, valid, info = read_strip(sdir)
        prep = prepare_rows(grid, valid, fiber, by_fiber.get(fname, []))
        w['prep'] = prep
        rec.update(strip_dir=str(sdir), strip_rows=prep['rows'], strip_cols=prep['cols'], valid_nodes=int(valid.sum()),
                   transposed=info.get('fiber_along') == 'columns')
        if prep['cols'] % 2 == 0:
            rec['notes'].append(f"even column count {prep['cols']}: centre column {prep['centre_col']} used (R4)")
        if prep['rejected']:
            continue
        rows_by_seg = {}
        for r in np.flatnonzero(prep['scored']):
            rows_by_seg.setdefault(by_id[prep['in_stretch'][r]]['segment'], []).append(int(r))
        for seg, rows in rows_by_seg.items():
            rows = np.asarray(rows, np.int64)
            nr, nc = np.nonzero(valid[rows])
            nr = rows[nr]
            req.add(seg, ('nodes', pid, seg), grid[nr, nc])
            req.add(seg, ('floor', pid, seg), fiber.pts[prep['k'][rows]])
            w['groups'][seg] = (rows, nr)
    results, seg_info = req.run(open_segment, axis, log)

    pieces, stretches, node_d, node_cls = {}, {}, {}, {}
    for w in work:
        rec = w['rec']
        pid = rec['piece']
        pieces[pid] = rec
        for st, sel in w['gt']:
            srec = stretches.setdefault(st['id'], stretch_base(st, set_name))
            srec['ground_truth_check'] = gt_record(st, results.get(('gt', pid, st['id'])), len(sel))
        prep = w.get('prep')
        if prep is None:
            continue
        rows_in = [r for r in range(prep['rows']) if prep['in_stretch'][r] is not None]
        rec.update(rows_with_centre=int(prep['has_centre'].sum()), centre_fallback_rows=int(prep['fallback'].sum()),
                   assoc_median_sv=prep['assoc_median'], rejected=prep['rejected'], rows_in_stretches=len(rows_in),
                   rows_in_stretches_beyond_30sv=int(sum(1 for r in rows_in if not prep['within'][r])),
                   rows_in_two_stretches=prep['multi'])
        if prep['rejected']:
            rec['notes'].append(f"rejected (5.2): median centre distance {prep['assoc_median']:.4f} sv > "
                                f"{ASSOC_MAX_SV} sv; no row scored")
            for st in w['own']:
                stretches.setdefault(st['id'], stretch_base(st, set_name)).update(rows_scored=0, strip_rejected=True)
            continue
        node_parts, floor_parts, sid_nodes, sid_rows = [], [], [], []
        for seg, (rows, nr) in w['groups'].items():
            pr, fr = results.get(('nodes', pid, seg)), results.get(('floor', pid, seg))
            if pr is None:
                rec['notes'].append(f'segment {seg} could not be read: its {len(nr)} nodes count as uncovered')
                pr, fr = Projection.empty(len(nr)), Projection.empty(len(rows))
            node_parts.append(pr)
            floor_parts.append(fr)
            sid_nodes += [prep['in_stretch'][r] for r in nr]
            sid_rows += [prep['in_stretch'][r] for r in rows]
        npr, fpr = Projection.concat(node_parts), Projection.concat(floor_parts)
        cls, fcls = classify(npr), classify(fpr)
        code = uncovered_reason(npr)
        d = np.where(code == 0, npr.d, np.nan)
        fd = np.where(fpr.covered(), fpr.d, np.nan)
        sid_nodes, sid_rows = np.asarray(sid_nodes, dtype=object), np.asarray(sid_rows, dtype=object)
        node_d[pid] = d[np.isfinite(d)]
        node_cls[pid] = cls
        ff = np.abs(fd[np.isfinite(fd)])
        rec.update(rows_scored=int(len(sid_rows)), nodes_on_scored_rows=int(len(d)), uncovered=_reason_counts(code),
                   rows_in_other_pieces_stretches=int(sum(1 for s_ in sid_rows if by_id[s_].get('piece') != pid)),
                   stretches=sorted(set(sid_rows.tolist())),
                   stretches_with_covered_nodes=sorted(set(sid_nodes[np.isfinite(d)].tolist())),
                   fiber_floor_rows=int(len(ff)), fiber_floor_mean_abs_sv=float(ff.mean()) if len(ff) else None,
                   **_strip_fields(node_stats(d)))
        rec['solver_sensitivity'] = piece_sensitivity(npr, cls, fpr, fcls, sid_nodes)
        if rec['rows_in_other_pieces_stretches']:   # R6's other reading: only the piece's own windows
            own_ids = {st['id'] for st in w['own']}
            mo = np.array([s_ in own_ids for s_ in sid_nodes], bool)
            mro = np.array([s_ in own_ids for s_ in sid_rows], bool)
            ffo = np.abs(fd[mro][np.isfinite(fd[mro])])
            rec['own_windows_only'] = dict(
                note="R6's other reading: rows in this piece's own windows only (reported, not the primary reading)",
                rows_scored=int(mro.sum()), stretches=sorted(set(sid_rows[mro].tolist())),
                stretches_with_covered_nodes=sorted(set(sid_nodes[mo & np.isfinite(d)].tolist())),
                fiber_floor_mean_abs_sv=float(ffo.mean()) if len(ffo) else None, **_strip_fields(node_stats(d[mo])),
                solver_sensitivity=piece_sensitivity(npr, cls, fpr, fcls, sid_nodes, mo, mro))
        for s_id in sorted(set(sid_rows.tolist())):
            st = by_id[s_id]
            m, mr = sid_nodes == s_id, sid_rows == s_id
            ffs = np.abs(fd[mr][np.isfinite(fd[mr])])
            part = dict(rows_scored=int(mr.sum()), nodes_on_scored_rows=int(m.sum()), uncovered=_reason_counts(code[m]),
                        fiber_floor_rows=int(len(ffs)), fiber_floor_mean_abs_sv=float(ffs.mean()) if len(ffs) else None,
                        solver_sensitivity=solver_diagnostics(npr, cls, m), **_strip_fields(node_stats(d[m])))
            srec = stretches.setdefault(s_id, stretch_base(st, set_name))
            if st.get('piece') == pid:
                srec.update(part, scored_in_piece=pid)
            else:
                srec.setdefault('rows_from_other_pieces', {})[pid] = part
        for st in w['own']:
            srec = stretches.setdefault(st['id'], stretch_base(st, set_name))
            if 'rows_scored' not in srec:
                srec.update(rows_scored=0, nodes_on_scored_rows=0, fiber_floor_rows=0, fiber_floor_mean_abs_sv=None,
                            scored_in_piece=pid, **_strip_fields(node_stats(np.zeros(0))))
    return dict(pieces=pieces, stretches=stretches, node_d=node_d, node_cls=node_cls, segments=seg_info)


def pooled(pieces, node_d, node_cls=None):
    """Pooled point estimates over the fibers with at least 25 covered nodes (5.2, 5.3; R10), and per fiber the
    tier-2 ingredients of its nodes."""
    per_fiber, cls_by_fiber = {}, {}
    for pid, rec in sorted(pieces.items()):
        if pid in node_d:
            per_fiber.setdefault(rec['fiber'], []).append(node_d[pid])
            if node_cls is not None and pid in node_cls:
                cls_by_fiber.setdefault(rec['fiber'], []).append(node_cls[pid])
    per_fiber = {f: np.concatenate(v) for f, v in per_fiber.items()}
    scored = sorted(f for f, v in per_fiber.items() if len(v) >= MIN_FIBER_NODES)
    d = np.concatenate([per_fiber[f] for f in scored]) if scored else np.zeros(0)
    out = dict(fibers=len(per_fiber), covered_nodes_per_fiber={f: int(len(v)) for f, v in per_fiber.items()},
               scored_fibers=len(scored), scored_fiber_names=scored, strip=node_stats(d))
    if node_cls is not None:
        out['tier2_per_fiber'] = {f: tier2_ingredients(_merge_cls(v), quantiles=True)
                                  for f, v in sorted(cls_by_fiber.items())}
    return out


# ---- PHerc0332: crossings (7) ------------------------------------------------------------------------------------

def v_sample_index(fibers):
    """All samples of the V fibers, for the k = 1 crossing rule (R11)."""
    names = sorted(n for n, f in fibers.items() if f.family == 'V')
    if not names:
        return None
    pts = np.concatenate([fibers[n].pts for n in names])
    return dict(names=names, index={n: i for i, n in enumerate(names)}, pts=pts, tree=cKDTree(pts),
                owner=np.concatenate([np.full(len(fibers[n].pts), i, np.int64) for i, n in enumerate(names)]),
                s=np.concatenate([fibers[n].s for n in names]))


def crossing_point(h, v, vt):
    """R11: the closest H sample whose nearest V sample (any V fiber, k = 1) is on v within 6 sv, and the midpoint
    of the two. Falls back to the pairwise closest samples (flagged) when the rule does not find the pair."""
    if vt is not None and v.name in vt['index']:
        dist, j = vt['tree'].query(h.pts, k=1)
        own = (vt['owner'][j] == vt['index'][v.name]) & (dist <= CROSS_TOL_SV)
        if own.any():
            cand = np.flatnonzero(own)
            hk = int(cand[np.argmin(dist[cand])])
            vj = int(j[hk])
            return dict(rule='k = 1 nearest V sample within 6 sv', closest_approach_sv=float(dist[hk]),
                        point_sv=((h.pts[hk] + vt['pts'][vj])/2.).tolist(), h_s=float(h.s[hk]),
                        v_s=float(vt['s'][vj]))
    dd, jj = v.tree.query(h.pts, k=1)
    hk = int(np.argmin(dd))
    vj = int(jj[hk])
    return dict(rule='pairwise closest samples (the k = 1 rule does not find this pair)',
                closest_approach_sv=float(dd[hk]), point_sv=((h.pts[hk] + v.pts[vj])/2.).tolist(),
                h_s=float(h.s[hk]), v_s=float(v.s[vj]))


def direction_options(rec):
    """Tier 2 of one direction: can it be usable, can it be unusable (5-node rule), and its median's interval."""
    if rec['nodes_within_30sv'] < CROSS_MIN_NODES or 'solver_sensitivity' not in rec:
        return dict(can_be_usable=False, can_be_unusable=True, median_interval=None)
    t2 = rec['solver_sensitivity']['tier2']
    nr, ns = t2['n_robust'], len(t2['lo'])
    usable = nr + ns >= CROSS_MIN_NODES
    return dict(can_be_usable=usable, can_be_unusable=nr < CROSS_MIN_NODES,
                median_interval=median_interval_any_count(t2, CROSS_MIN_NODES - nr) if usable else None)


def agreement_tier2(directions):
    """Tier 2 of a crossing: the possible numbers of usable directions and the agreement's interval for each."""
    o = [direction_options(r) for r in directions]
    two = None
    if o[0]['can_be_usable'] and o[1]['can_be_usable']:
        two = [(o[0]['median_interval'][0] + o[1]['median_interval'][0])/2.,
               (o[0]['median_interval'][1] + o[1]['median_interval'][1])/2.]
    one = [o[x]['median_interval'] for x, y in ((0, 1), (1, 0)) if o[x]['can_be_usable'] and o[y]['can_be_unusable']]
    none = bool(o[0]['can_be_unusable'] and o[1]['can_be_unusable'])
    possible = [n for n, ok in ((0, none), (1, bool(one)), (2, two is not None)) if ok]
    return dict(per_direction=o, two_directions=two, one_direction=one, possible_directions=possible)


def crossing_agreement(h_grid, h_valid, v_grid, v_valid, point):
    """7 (R12): per direction the median |d| of one strip's nodes within 30 sv of the point on the other strip's
    surface, with the 5-node rule; the agreement is the mean of the usable medians. Also tier 2."""
    point = np.asarray(point, np.float64)
    sides = {'H': (np.asarray(h_grid, np.float64), np.asarray(h_valid, bool)),
             'V': (np.asarray(v_grid, np.float64), np.asarray(v_valid, bool))}
    surf = {key: BilinearSurface(g, val) for key, (g, val) in sides.items()}
    dirs = []
    for a, b in (('H', 'V'), ('V', 'H')):
        g, val = sides[a]
        nodes = g[val]
        near = nodes[np.linalg.norm(nodes - point, axis=1) <= CROSS_RADIUS_SV]
        rec = dict(direction=f'{a} strip onto {b} strip', nodes_within_30sv=int(len(near)), covered=None,
                   median_abs_d_sv=None, usable=False)
        if len(near) < CROSS_MIN_NODES:
            rec['reason'] = f'fewer than {CROSS_MIN_NODES} nodes within {CROSS_RADIUS_SV:g} sv of the crossing'
        else:
            pr = surf[b].project(near)
            cls = classify(pr)
            cov = cls['covered']
            rec['covered'] = int(cov.sum())
            rec['solver_sensitivity'] = block_sensitivity(pr, cls, quantiles=True)
            if cov.sum() < CROSS_MIN_NODES:
                rec['reason'] = f'fewer than {CROSS_MIN_NODES} of them covered by the other strip'
            else:
                rec.update(median_abs_d_sv=float(np.median(np.abs(pr.d[cov]))), usable=True)
        dirs.append(rec)
    meds = [r['median_abs_d_sv'] for r in dirs if r['usable']]
    return dict(directions=dirs, n_directions=len(meds), agreement_sv=float(np.mean(meds)) if meds else None,
                tier2=agreement_tier2(dirs))


def summary_of(values):
    v = np.asarray([x for x in values if x is not None], np.float64)
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), mean_sv=float(v.mean()), median_sv=float(np.median(v)),
                p90_sv=float(np.percentile(v, 90)), frac_le_2sv=float(np.mean(v <= 2.)))


def _near_count(s, point):
    nodes = s['grid'][s['valid']]
    return int((np.linalg.norm(nodes - np.asarray(point, np.float64), axis=1) <= CROSS_RADIUS_SV).sum())


def rescore_crossings(ws, sample, plan, strips_root, log=None):
    ws, strips_root = Path(ws), Path(strips_root)
    out = dict(sample=[c['id'] for c in sample.get('crossings', [])], per_crossing={}, notes=[])
    crossings = {c['id']: c for c in plan.get('crossings', [])}
    windows = {w['id']: w for w in plan.get('windows', [])}
    plan_pieces = {p['id']: p for p in plan.get('pieces', [])}
    fibers, skipped = load_fiber_dir(ws/FIBER_DIRS['0332'])
    vt = v_sample_index(fibers)
    planned = sorted({p.get('fiber') for p in plan_pieces.values() if p.get('fiber') in fibers})
    out.update(fibers_loaded=len(fibers), fibers_skipped=skipped, v_fibers=len(vt['names']) if vt else 0,
               h_fibers=sum(1 for f in fibers.values() if f.family == 'H'), planned_fibers=len(planned))
    cache = {}

    def strip_of(pid):
        """A crossing strip with its association (R12): to its piece's fiber (5.2) and to the nearest planned fiber."""
        if pid not in cache:
            sdir = strip_tifxyz_dir(strips_root/'0332'/pid) if pid else None
            if sdir is None:
                cache[pid] = None
            else:
                grid, valid, _ = read_strip(sdir)
                centre, has, _, _ = row_centres(grid, valid)
                med = {f: centre_distances(centre, has, fibers[f])[3] for f in planned}
                own = (plan_pieces.get(pid) or {}).get('fiber')
                own_med = med.get(own)
                if own_med is None and own in fibers:
                    own_med = centre_distances(centre, has, fibers[own])[3]
                finite = {f: m for f, m in med.items() if math.isfinite(m)}
                nearest = min(finite, key=lambda f: (finite[f], f)) if finite else None
                cache[pid] = dict(grid=grid, valid=valid, dir=str(sdir), fiber=own, assoc_median_sv=own_med,
                                  rejected=not (own_med is not None and own_med <= ASSOC_MAX_SV),
                                  nearest_planned_fiber=nearest,
                                  nearest_planned_median_sv=finite.get(nearest) if nearest else None)
        return cache[pid]

    def other_reading_piece(fiber, point):
        """R12's other reading: among the strips whose nearest planned fiber is ``fiber`` (median <= 5 sv), the one
        with the most valid nodes within 30 sv of the point."""
        best = None
        for pid in sorted(plan_pieces):
            s = strip_of(pid)
            if s is None or s['nearest_planned_fiber'] != fiber or not s['nearest_planned_median_sv'] <= ASSOC_MAX_SV:
                continue
            n = _near_count(s, point)
            if best is None or n > best[1]:
                best = (pid, n)
        return best[0] if best else None

    def assoc_info(s):
        return None if s is None else dict(strip=s['dir'], fiber=s['fiber'], assoc_median_sv=s['assoc_median_sv'],
                                           rejected=s['rejected'], nearest_planned_fiber=s['nearest_planned_fiber'],
                                           nearest_planned_median_sv=s['nearest_planned_median_sv'])

    for cid in out['sample']:
        rec = dict(id=cid, missing=True)
        out['per_crossing'][cid] = rec
        c = crossings.get(cid)
        if c is None:
            rec['missing_reason'] = 'not in the plan'
            continue
        rec.update(h_fiber=c['h_fiber'], v_fiber=c['v_fiber'], tight=bool(c['closest_approach_sv'] <= TIGHT_SV),
                   plan=dict(closest_approach_sv=c.get('closest_approach_sv'), point_sv=c.get('point_sv'),
                             h_s=c.get('h_s'), v_s=c.get('v_s'), run=c.get('run'), z_tercile=c.get('z_tercile')))
        h, v = fibers.get(c['h_fiber']), fibers.get(c['v_fiber'])
        if h is None or v is None:
            rec['missing_reason'] = 'fiber not loaded'
            continue
        if h.family != 'H' or v.family != 'V':
            rec['note'] = f'JSON tags: h fiber {h.family}, v fiber {v.family}'
        cp = crossing_point(h, v, vt)
        rec['recomputed'] = cp
        if c.get('point_sv') is not None:
            rec['point_vs_plan_max_abs_sv'] = float(np.max(np.abs(np.asarray(cp['point_sv']) -
                                                                  np.asarray(c['point_sv'], np.float64))))
        rec['closest_approach_vs_plan_sv'] = cp['closest_approach_sv'] - float(c['closest_approach_sv'])
        pieces = {}
        for wid in c.get('windows', []):
            win = windows.get(wid) or {}
            pieces[win.get('family') or wid[-1]] = win.get('piece')
        rec.update(h_piece=pieces.get('H'), v_piece=pieces.get('V'))
        hs, vs = strip_of(pieces.get('H')), strip_of(pieces.get('V'))
        rec.update(h_strip=hs and hs['dir'], v_strip=vs and vs['dir'],
                   association=dict(H=assoc_info(hs), V=assoc_info(vs)))
        alt = (other_reading_piece(c['h_fiber'], cp['point_sv']), other_reading_piece(c['v_fiber'], cp['point_sv']))
        if hs is None or vs is None:
            rec['missing_reason'] = 'no strip for ' + ' and '.join(k for k, x in (('H', hs), ('V', vs)) if x is None)
        elif hs['rejected'] or vs['rejected']:
            rec['missing_reason'] = 'strip rejected (5.2, R12): ' + ', '.join(
                f"{k} median centre distance {x['assoc_median_sv']} sv" for k, x in (('H', hs), ('V', vs))
                if x['rejected'])
        else:
            rec.update(crossing_agreement(hs['grid'], hs['valid'], vs['grid'], vs['valid'], cp['point_sv']))
            rec['missing'] = rec['agreement_sv'] is None
            if rec['missing']:
                rec['missing_reason'] = 'no usable direction (5-node rule)'
        if alt != (rec['h_piece'], rec['v_piece']):
            o = dict(h_piece=alt[0], v_piece=alt[1], missing=True)
            if alt[0] and alt[1]:
                ha, va = strip_of(alt[0]), strip_of(alt[1])
                o.update(crossing_agreement(ha['grid'], ha['valid'], va['grid'], va['valid'], cp['point_sv']))
                o['missing'] = o['agreement_sv'] is None
            rec['other_reading'] = o
        if log:
            log(f"crossing {cid}: agreement {rec.get('agreement_sv')}, directions {rec.get('n_directions')}")
    vals = [r.get('agreement_sv') for r in out['per_crossing'].values()]
    out.update(planned=len(out['sample']), missing=sorted(k for k, r in out['per_crossing'].items() if r['missing']),
               summary=summary_of(vals),
               tight_summary=summary_of([r.get('agreement_sv') for r in out['per_crossing'].values() if r.get('tight')]))
    return out


# ---- the recomputation -------------------------------------------------------------------------------------------

def recompute(ws, sample_path, strips_root=None, gt_only=False, log=None):
    ws, sample_path = Path(ws), Path(sample_path)
    strips_root = Path(strips_root) if strips_root else ws/'runs'/'v3'/'strips'
    sample = json.loads(sample_path.read_text(encoding='utf-8'))
    rep = dict(tool=TOOL, tool_version=TOOL_VERSION, tool_sha256=sha256(__file__),
               created_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), mode='gt-only' if gt_only else 'full',
               python=sys.version.split()[0], numpy=np.__version__,
               inputs=dict(workspace=str(ws), sample=str(sample_path), sample_sha256=sha256(sample_path),
                           strips_root=None if gt_only else str(strips_root), plans={}),
               readings=readings(),
               tolerance=dict(sv=TOL_SV, integers='exact', robust_d_sv=ROBUST_D_SV, robust_side_sv=ROBUST_SIDE_SV,
                              robust_spread_sv=ROBUST_SPREAD_SV,
                              robust_allowance=f'{ROBUST_SLACK_SV} sv + {ROBUST_SLACK_REL} x |d|',
                              sensitive_edge_side=f'max({SENSITIVE_SIDE_SV} sv, {SENSITIVE_SIDE_REL} x distance)',
                              delta_below=f'{SLACK_ABS_SV} sv + {SLACK_REL_BELOW} x distance',
                              delta_above=f'{SLACK_ABS_SV} sv + {SLACK_REL_ABOVE} x distance',
                              text=tolerance_text()),
               not_recomputed=['frame-plane and Lasagna-plane baselines (5.5)', 'calibration (5.7)',
                               'bootstrap intervals (5.4, 7)', 'gates G1, G1prime, G2',
                               'pooled secondary numbers (only a sample is recomputed)', 'sensitivity blocks (4.9)',
                               'the PHerc0139 crossings block of the score reports'],
               errors=[])
    plans = {}
    for key, rel in PLAN_FILES.items():
        p = ws/rel
        info = dict(path=str(p), exists=p.is_file())
        if p.is_file():
            want = (sample.get('sources') or {}).get(rel)
            info.update(sha256=sha256(p), sample_sha256=want)
            info['matches_sample'] = (want == info['sha256']) if want else None
            if want and want != info['sha256']:
                rep['errors'].append(f'{rel}: sha256 differs from the one recorded in the sample')
            plans[key] = json.loads(p.read_text(encoding='utf-8'))
        else:
            rep['errors'].append(f'{rel}: missing')
        rep['inputs']['plans'][key] = info
    axis_path = ws/UMBILICUS_0139
    axis = Axis.from_json(axis_path) if axis_path.is_file() else None
    rep['inputs']['umbilicus'] = dict(path=str(axis_path), sha256=sha256(axis_path) if axis else None)
    if axis is None:
        rep['errors'].append(f'{UMBILICUS_0139}: missing, the sign of d is not oriented')

    for set_name, sample_key, run_only in (('secondary', 'secondary_stretches', True),
                                           ('primary', 'primary_stretches', False)):
        plan = plans.get(set_name)
        if plan is None:
            continue
        scored = [st for st in plan.get('stretches', []) if (bool(st.get('run', True)) if run_only else True)]
        by_id = {st['id']: st for st in scored}
        sample_items = sample.get(sample_key, [])
        mismatch = []
        for item in sample_items:
            st = by_id.get(item['id'])
            if st is None:
                mismatch.append(f"{item['id']}: not in the scored set of the plan")
                continue
            for key in ('fiber', 'segment', 's0', 's1', 'piece'):
                if key in item and item[key] != st.get(key):
                    mismatch.append(f"{item['id']}: {key} {item[key]!r} in the sample, {st.get(key)!r} in the plan")
        rep['errors'] += [f'{set_name}: {m}' for m in mismatch]
        pieces_plan = {p['id']: p for p in plan.get('pieces', [])}
        piece_ids = sorted({by_id[i['id']]['piece'] for i in sample_items if i['id'] in by_id})
        jobs = []
        for pid in piece_ids:
            fname = (pieces_plan.get(pid) or {}).get('fiber') or next(st['fiber'] for st in scored if st['piece'] == pid)
            sdir = None if gt_only else strip_tifxyz_dir(strips_root/set_name/pid)
            jobs.append(dict(piece=pid, fiber=fname, strip_dir=sdir))
        fibers, skipped = load_fiber_dir(ws/FIBER_DIRS['0139'], {j['fiber'] for j in jobs})
        for j in jobs:
            if j['fiber'] not in fibers:
                rep['errors'].append(f"{set_name} {j['piece']}: fiber {j['fiber']} not loaded "
                                     f"({skipped.get(j['fiber'], 'no JSON')})")
        seg_dirs = {s['name']: ws/Path(str(s['source']).replace('\\', '/')) for s in plan.get('segments', [])}
        if log:
            log(f'{set_name}: {len(jobs)} pieces, {sum(1 for j in jobs if j["strip_dir"])} strips found')
        res = score_pieces(set_name, jobs, fibers, scored, segment_opener(seg_dirs, 1.0), axis,
                           strips=not gt_only, log=log)
        sample_ids = {i['id'] for i in sample_items}
        for sid, srec in res['stretches'].items():
            srec['in_sample'] = sid in sample_ids
        out = dict(sample_stretches=[i['id'] for i in sample_items], scored_order=[st['id'] for st in scored],
                   pieces=res['pieces'], stretches=dict(sorted(res['stretches'].items())), segments=res['segments'],
                   strips_missing=sorted(j['piece'] for j in jobs if j['strip_dir'] is None) if not gt_only else None)
        if set_name == 'primary' and not gt_only:
            out['pooled'] = pooled(res['pieces'], res['node_d'], res['node_cls'])
        rep[set_name] = out
    if not gt_only and '0332' in plans:
        rep['crossings_0332'] = rescore_crossings(ws, sample, plans['0332'], strips_root, log)
    gts = [st['ground_truth_check'] for s in ('secondary', 'primary') for st in (rep.get(s) or {}).get('stretches',
                                                                                                    {}).values()
           if 'ground_truth_check' in st]
    rep['ground_truth_check_summary'] = dict(
        stretches=len(gts), with_plan_values=sum(1 for g in gts if 'verdict' in g),
        agree=sum(1 for g in gts if g.get('verdict') == 'agree'),
        agree_within_solver_bounds=sum(1 for g in gts if g.get('verdict') == 'agree_within_solver_bounds'),
        differs=sum(1 for g in gts if g.get('verdict') == 'differs'))
    return rep


# ---- the comparison with the scorer's reports --------------------------------------------------------------------

def _stretch_ids(values, order):
    out = []
    for x in values or []:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, int) and 0 <= x < len(order):
            out.append(order[x])                 # an index into the scorer's (run-only) stretch list
        else:
            out.append(f'?{x}')
    return out


def piece_fields(mm, e, theirs_ids, assoc_mine):
    """Tier 1 and tier 2 of a piece's per_strip fields; ``mm`` is this file's piece record or its own_windows_only
    block."""
    ss = mm.get('solver_sensitivity') or {}
    t2, ft2 = ss.get('tier2'), (ss.get('floor') or {}).get('tier2')
    f = dict(strip_nodes=_num(mm.get('strip_nodes', 0), e.get('strip_nodes'), 0))
    ns = e.get('strip_nodes')
    if t2 is not None and ns is not None:
        f['strip_nodes']['bounds'] = _bounds(count_interval(t2), ns, 0)
    for key, signed in (('strip_mean_abs_sv', False), ('strip_signed_mean_sv', True)):
        f[key] = _num(mm.get(key), e.get(key), TOL_SV)
        if t2 is not None and ns is not None:
            if ns == 0:   # no covered node: the scorer's mean is None, possible only with no robust node
                f[key]['bounds'] = dict(lo=None, hi=None, width=None, within=t2['n_robust'] == 0 and e.get(key) is None)
            else:
                f[key]['bounds'] = _bounds(mean_interval(t2, ns, signed), e.get(key), TOL_SV)
    f['fiber_floor_mean_abs_sv'] = _num(mm.get('fiber_floor_mean_abs_sv'), e.get('fiber_floor_mean_abs_sv'), TOL_SV)
    if ft2 is not None:
        tv = e.get('fiber_floor_mean_abs_sv')
        f['fiber_floor_mean_abs_sv']['bounds'] = (
            dict(lo=None, hi=None, width=None, within=ft2['n_robust'] == 0) if tv is None else
            _bounds(mean_interval_any_count(ft2), tv, TOL_SV))
    f['assoc_median_sv'] = _num(assoc_mine, e.get('assoc_median_sv'), TOL_SV)       # tier 1 only
    mine_ids = sorted(mm.get('stretches') or [])
    f['stretches'] = dict(mine=mine_ids, scorer=theirs_ids, agree=mine_ids == theirs_ids,
                          scorer_equals_mine_with_covered_nodes=sorted(mm.get('stretches_with_covered_nodes') or [])
                          == theirs_ids)
    if ss:   # the scorer may list the stretches with covered nodes, which depend on the solver
        rob, pos = set(ss.get('stretches_with_robust_nodes') or []), set(ss.get('stretches_with_possible_nodes') or [])
        f['stretches']['bounds'] = dict(lo=sorted(rob), hi=sorted(pos), within=rob <= set(theirs_ids) <= pos)
    return f


def compare_piece(set_name, pid, m, entries, theirs, order):
    it = dict(kind='piece', set=set_name, id=pid, fields={})
    if theirs is None:
        it['status'] = 'no scorer report'
        return _set_verdict(it, 'differs')
    rows = m.get('rows_scored') or 0
    mine_scored = rows > 0 and not m.get('rejected') and not m.get('error')
    alt = m.get('own_windows_only')
    mine_status = m.get('error') or ('rejected' if m.get('rejected') else 'scored' if mine_scored else 'no scored row')
    if len(entries) > 1:
        it['note'] = f'{len(entries)} per_strip entries named {pid}; the first is compared'
    if not entries:
        skips = [s for s in theirs.get('skips') or [] if str(s.get('strip')) == pid]
        it.update(status='not in the scorer per_strip', scorer_skips=skips, mine_scored=mine_scored,
                  mine_status=mine_status)
        if not mine_scored:
            return _set_verdict(it, 'agree')
        if alt is not None and not alt.get('rows_scored'):
            it['reading'] = 'R6'
            return _set_verdict(it, 'agree_under_other_reading')
        return _set_verdict(it, 'differs')
    if m.get('rejected') or m.get('error'):
        it.update(status='in the scorer per_strip only', mine_status=mine_status)
        return _set_verdict(it, 'differs')
    e = entries[0]
    theirs_ids = sorted(_stretch_ids(e.get('stretches'), order))
    it['fields'] = piece_fields(m, e, theirs_ids, m.get('assoc_median_sv'))
    verdict = verdict_of(it['fields'])
    if verdict == 'differs' and alt is not None:
        af = piece_fields(alt, e, theirs_ids, m.get('assoc_median_sv'))
        av = verdict_of(af)
        it['own_windows_only_reading'] = dict(fields=af, verdict=av,
                                              note="R6's other reading, for diagnosis (own windows only)")
        if av != 'differs':
            it['reading'] = 'R6'
            verdict = 'agree_under_other_reading'
    return _set_verdict(it, verdict)


def pooled_candidates(t2f, theirs_fibers):
    """Tier 2 of the 25-node rule: the fiber sets the scorer may have scored (with ``theirs_fibers`` fibers)."""
    sure = [f for f, t in sorted(t2f.items()) if t['n_robust'] >= MIN_FIBER_NODES]
    maybe = [f for f, t in sorted(t2f.items()) if f not in sure and count_interval(t)[1] >= MIN_FIBER_NODES]
    out = []
    for r in (range(len(maybe) + 1) if len(maybe) <= 12 else ()):
        for extra in itertools.combinations(maybe, r):
            fs = sure + list(extra)
            if theirs_fibers is None or len(fs) == theirs_fibers:
                out.append(fs)
    return out, (len(sure), len(sure) + len(maybe))


def compare_pooled(pool, theirs):
    it = dict(kind='pooled', set='primary', id='accuracy.all', fields={})
    acc = (theirs.get('accuracy') or {}).get('all') or {}
    point = (acc.get('strip') or {}).get('point')
    empty = not point                   # with no scored fiber the scorer writes point = None
    point = point or {}
    s, f = pool.get('strip') or {}, it['fields']
    tf = acc.get('scored_fibers')
    tf = 0 if tf is None and empty else tf
    tn = 0 if empty else point.get('nodes')
    f['scored_fibers'] = _num(pool.get('scored_fibers'), tf, 0)
    f['nodes'] = _num(s.get('nodes', 0), tn, 0)
    for mine, key in (('mean_abs_sv', 'mean'), ('median_abs_sv', 'median'), ('p90_abs_sv', 'p90')):
        f[key] = _num(s.get(mine), None if empty else point.get(key), TOL_SV)
    tw = {}
    for key in ('within1', 'within2'):
        frac, n = point.get(key), point.get('nodes')
        tw[key] = 0 if empty or n == 0 else (int(round(frac*n)) if frac is not None and n else None)
        f[key + '_nodes'] = _num(s.get(key + '_nodes', 0), tw[key], 0)
    t2f = pool.get('tier2_per_fiber')
    if t2f is not None and verdict_of(f) != 'agree':
        cands, frange = pooled_candidates(t2f, tf)
        tried = []
        for fs in cands:
            ing = dict(n_robust=sum(t2f[x]['n_robust'] for x in fs),
                       robust_abs_sum=sum(t2f[x]['robust_abs_sum'] for x in fs),
                       robust_signed_sum=sum(t2f[x]['robust_signed_sum'] for x in fs),
                       robust_abs=np.sort(np.concatenate([_arr(t2f[x]['robust_abs']) for x in fs] or [np.zeros(0)])),
                       lo=np.sort(np.concatenate([_arr(t2f[x]['lo']) for x in fs] or [np.zeros(0)])),
                       hi=np.sort(np.concatenate([_arr(t2f[x]['hi']) for x in fs] or [np.zeros(0)]))[::-1])
            b = dict(scored_fibers=_bounds(frange, tf, 0) if tf is not None else _bounds(None, None, 0))
            b['nodes'] = _bounds(count_interval(ing), tn, 0) if tn is not None else _bounds(None, None, 0)
            k = None if tn is None else tn - ing['n_robust']
            if empty:
                ok = not fs
                for key in ('mean', 'median', 'p90'):
                    b[key] = dict(lo=None, hi=None, width=None, within=ok)
                for key in ('within1_nodes', 'within2_nodes'):
                    b[key] = dict(lo=0, hi=0, width=0, within=ok)
            else:
                b['mean'] = _bounds(mean_interval(ing, tn) if tn else None, point.get('mean'), TOL_SV)
                for key, q in (('median', 50), ('p90', 90)):
                    b[key] = _bounds(quantile_interval(ing, k, q), point.get(key), TOL_SV)
                for key, t in (('within1_nodes', 1.), ('within2_nodes', 2.)):
                    b[key] = _bounds(within_interval(ing, k, t), tw[key[:7]], 0)
            tried.append((fs, b))
            if all(x['within'] for x in b.values()):
                break
        if tried:
            fs, b = tried[-1]
            it['tier2_fibers'] = fs
            for key, x in b.items():
                f[key]['bounds'] = x
    return _set_verdict(it, verdict_of(f))


def _crossing_fields(m, t, theirs_missing):
    """Tier 1 and tier 2 of a crossing (``m``: this file's record or its other reading; ``t``: the scorer's row)."""
    f = {}
    mine_missing = bool(m.get('missing'))
    t2 = m.get('tier2')               # absent when a strip is missing or rejected: those must match exactly
    f['missing'] = dict(mine=mine_missing, scorer=theirs_missing, agree=mine_missing == theirs_missing)
    if t2 is not None:
        pd = t2['possible_directions']
        f['missing']['bounds'] = dict(possible_directions=pd, within=(0 in pd) if theirs_missing else
                                      any(n > 0 for n in pd))
    if t is not None and not theirs_missing:
        n_s = t.get('directions')
        f['directions'] = _num(m.get('n_directions', 0) if not mine_missing or t2 else None, n_s, 0)
        f['agreement_sv'] = _num(m.get('agreement_sv'), t.get('agreement_sv'), TOL_SV)
        if t2 is not None:
            f['directions']['bounds'] = dict(possible_directions=t2['possible_directions'],
                                             within=n_s in t2['possible_directions'])
            ivs = ([t2['two_directions']] if n_s == 2 and t2['two_directions'] else
                   t2['one_direction'] if n_s == 1 else [])
            a = t.get('agreement_sv')
            hit = [iv for iv in ivs if a is not None and iv[0] - TOL_SV <= a <= iv[1] + TOL_SV]
            show = hit[0] if hit else (ivs[0] if ivs else None)
            f['agreement_sv']['bounds'] = dict(_bounds(show, a, TOL_SV), within=bool(hit))
    return f


def compare_crossings(cr, theirs):
    items = []
    res = (theirs or {}).get('crossings') or {}
    per = {str(r.get('id')): r for r in res.get('per_crossing') or []}
    listed_missing = {str(x) for x in res.get('missing') or []}
    for cid, m in sorted(cr.get('per_crossing', {}).items()):
        it = dict(kind='crossing', set='0332', id=cid)
        t = per.get(cid)
        theirs_missing = t is None or cid in listed_missing or t.get('agreement_sv') is None
        it['scorer_row'] = dict(present=t is not None, listed_missing=cid in listed_missing)
        f = _crossing_fields(m, t, theirs_missing)
        if t is not None and not theirs_missing:
            rc = m.get('recomputed') or {}
            f['closest_approach_sv'] = _num(rc.get('closest_approach_sv'), t.get('closest_approach_sv'), TOL_SV)
            if rc.get('point_sv') is not None and t.get('point_sv') is not None:
                diff = float(np.max(np.abs(np.asarray(rc['point_sv'], np.float64) -
                                           np.asarray(t['point_sv'], np.float64))))
                f['point_sv_max_abs'] = dict(mine=rc['point_sv'], scorer=t['point_sv'], diff=diff,
                                             agree=diff <= TOL_SV)
        it['fields'] = f
        verdict = verdict_of(f)
        o = m.get('other_reading')
        if verdict == 'differs' and o is not None:
            of = _crossing_fields(o, t, theirs_missing)
            for key in ('closest_approach_sv', 'point_sv_max_abs'):
                if key in f:
                    of[key] = f[key]
            ov = verdict_of(of)
            it['other_reading'] = dict(fields=of, verdict=ov, pieces=[o.get('h_piece'), o.get('v_piece')],
                                       note="R12's other reading, for diagnosis")
            if ov != 'differs':
                it['reading'] = 'R12'
                verdict = 'agree_under_other_reading'
        items.append(_set_verdict(it, verdict))
    f = dict(planned=_num(cr.get('planned'), res.get('planned'), 0))
    items.append(_set_verdict(dict(kind='crossing_set', set='0332', id='planned', fields=f), verdict_of(f)))
    side_by_side = dict(mine=dict(summary=cr.get('summary'), tight_summary=cr.get('tight_summary'),
                                  missing=cr.get('missing')),
                        scorer={k: v for k, v in res.items() if k != 'per_crossing'},
                        note='not compared field by field: the scorer\'s summary key names were not read')
    return items, side_by_side


def compare(rep, reports_dir):
    """Per-item differences between this report and the frozen plan (ground-truth check) and, when
    ``reports_dir`` is given, the scorer's reports."""
    items = []
    out = dict(reports_dir=str(reports_dir) if reports_dir else None, tolerance_sv=TOL_SV, reports={}, items=items)
    for set_name in ('primary', 'secondary'):
        for sid, st in sorted(((rep.get(set_name) or {}).get('stretches') or {}).items()):
            gt = st.get('ground_truth_check') or {}
            if 'plan' in gt:
                f = gt_fields(gt)
                items.append(_set_verdict(dict(kind='ground_truth', set=set_name, id=sid, in_sample=st.get('in_sample'),
                                               fields=f), verdict_of(f) if f else 'agree'))
    if reports_dir is not None and rep.get('mode') != 'gt-only':
        reports_dir = Path(reports_dir)
        for set_name in ('primary', 'secondary'):
            mine = rep.get(set_name)
            if not mine:
                continue
            path = reports_dir/REPORT_FILES[set_name]
            info = dict(path=str(path), found=path.is_file())
            out['reports'][set_name] = info
            theirs = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
            if theirs is not None:
                prov = theirs.get('provenance') or {}
                info.update(sha256=sha256(path), scorer_sha256=prov.get('scorer_sha256'),
                            prereg_sha256=prov.get('prereg_sha256'), stretches_planned=theirs.get('stretches_planned'),
                            per_strip_entries=len(theirs.get('per_strip') or []))
            per = {}
            for e in (theirs or {}).get('per_strip') or []:
                per.setdefault(str(e.get('strip')), []).append(e)
            order = mine.get('scored_order') or []
            for pid, m in sorted((mine.get('pieces') or {}).items()):
                items.append(compare_piece(set_name, pid, m, per.get(pid, []), theirs, order))
            if set_name == 'primary' and theirs is not None and mine.get('pooled'):
                items.append(compare_pooled(mine['pooled'], theirs))
        cr = rep.get('crossings_0332')
        if cr:
            path = reports_dir/REPORT_FILES['0332']
            info = dict(path=str(path), found=path.is_file())
            out['reports']['0332'] = info
            theirs = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
            if theirs is None:
                items.append(_set_verdict(dict(kind='crossing_set', set='0332', id='report', status='no scorer report',
                                               fields={}), 'differs'))
            else:
                prov = theirs.get('provenance') or {}
                info.update(sha256=sha256(path), scorer_sha256=prov.get('scorer_sha256'),
                            prereg_sha256=prov.get('prereg_sha256'))
                cross_items, out['crossing_summaries_side_by_side'] = compare_crossings(cr, theirs)
                items += cross_items
    by_kind, fields = {}, {}
    for it in items:
        k = by_kind.setdefault(it['kind'], dict(items=0, **{v: 0 for v in VERDICTS}))
        k['items'] += 1
        k[it['verdict']] += 1
        for name, f in it.get('fields', {}).items():
            s = fields.setdefault(f"{it['kind']}.{name}", dict(compared=0, agree=0, within_solver_bounds=0,
                                                               max_abs_diff=None, max_bounds_width=None))
            s['compared'] += 1
            s['agree'] += bool(f['agree'])
            b = f.get('bounds') or {}
            s['within_solver_bounds'] += bool(not f['agree'] and b.get('within'))
            if f.get('diff') is not None and not isinstance(f['diff'], bool):
                s['max_abs_diff'] = max(s['max_abs_diff'] or 0., abs(float(f['diff'])))
            if isinstance(b.get('width'), (int, float)):
                s['max_bounds_width'] = max(s['max_bounds_width'] or 0., float(b['width']))

    def brief(it):
        return dict(kind=it['kind'], set=it['set'], id=it['id'], status=it.get('status'), verdict=it['verdict'],
                    reading=it.get('reading'), fields=sorted(n for n, f in it.get('fields', {}).items()
                                                             if not f['agree']))
    bad = [brief(it) for it in items if it['verdict'] == 'differs']
    explained = [brief(it) for it in items if it['verdict'] in ('agree_within_solver_bounds',
                                                                'agree_under_other_reading')]
    verdicts = {v: sum(1 for it in items if it['verdict'] == v) for v in VERDICTS}
    out['summary'] = dict(items=len(items), verdicts=verdicts,
                          all_exact=all(it['verdict'] == 'agree' for it in items), no_item_differs=not bad,
                          by_kind=by_kind, fields=fields, disagreements=bad, explained=explained)
    return out


# ---- command line ------------------------------------------------------------------------------------------------

def _print_summary(rep, out_path):
    for set_name in ('secondary', 'primary'):
        s = rep.get(set_name)
        if s:
            pcs = s.get('pieces', {})
            print(f"{set_name}: {len(pcs)} pieces, {len(s.get('stretches', {}))} stretches recomputed, "
                  f"strips missing: {s.get('strips_missing')}")
    cr = rep.get('crossings_0332')
    if cr:
        print(f"crossings: {cr['planned']} planned, missing {cr['missing']}, summary {cr['summary']}")
    g = rep.get('ground_truth_check_summary') or {}
    print(f"ground-truth check vs the plan: {g.get('agree')} of {g.get('with_plan_values')} stretches agree, "
          f"{g.get('agree_within_solver_bounds')} within the solver bounds, {g.get('differs')} differ")
    c = rep.get('comparison')
    if c:
        sm = c['summary']
        v = sm['verdicts']
        print(f"comparison: {sm['items']} items: {v['agree']} agree within {TOL_SV} sv (integers exact), "
              f"{v['agree_within_solver_bounds']} within the solver bounds (tier 2), {v['agree_under_other_reading']} "
              f"under the other reading of R6 or R12, {v['differs']} DIFFER")
        for b in sm['explained'][:40]:
            print(f"  {b['verdict']} {b['kind']} {b['set']} {b['id']}: {b['status'] or ', '.join(b['fields'])}")
        for b in sm['disagreements'][:40]:
            print(f"  DIFFERS {b['kind']} {b['set']} {b['id']}: {b['status'] or ', '.join(b['fields'])}")
    if rep.get('errors'):
        print('errors:', *rep['errors'], sep='\n  ')
    print(f'wrote {out_path}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog=TOOL, description=(__doc__ or '').split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='The module docstring states what is read, the readings R1-R12, the '
                                        'tolerance and the verdicts.')
    ap.add_argument('--workspace', required=True, type=Path, help='the frozen score/ tree')
    ap.add_argument('--sample', required=True, type=Path, help='verification/verification_sample.json')
    ap.add_argument('--out', required=True, type=Path, help='the report JSON to write')
    ap.add_argument('--compare', action='store_true', help="recompute, then compare with the scorer's reports")
    ap.add_argument('--compare-only', action='store_true', help='compare the existing --out report (no recomputing)')
    ap.add_argument('--gt-only', action='store_true', help='the ground-truth check only: no strip, no scorer report')
    ap.add_argument('--strips', type=Path, help='strip root (default <workspace>/runs/v3/strips)')
    ap.add_argument('--reports', type=Path, help="the scorer's reports (default <workspace>/results/v3)")
    a = ap.parse_args(argv)

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    if a.compare_only and a.gt_only:
        print('--compare-only and --gt-only exclude each other', file=sys.stderr)
        return 2
    if a.compare_only:
        if not a.out.is_file():
            print(f'--compare-only: {a.out} does not exist', file=sys.stderr)
            return 2
        rep = json.loads(a.out.read_text(encoding='utf-8'))
    else:
        if not a.workspace.is_dir() or not a.sample.is_file():
            print('the workspace folder or the sample file does not exist', file=sys.stderr)
            return 2
        rep = recompute(a.workspace, a.sample, a.strips, a.gt_only, log)
    code = 0
    if a.compare or a.compare_only or a.gt_only:
        reports = None if a.gt_only else (a.reports or a.workspace/'results'/'v3')
        rep['comparison'] = compare(rep, reports)
        code = 0 if rep['comparison']['summary']['no_item_differs'] else 1
    if rep.get('errors'):
        code = 2  # an input error (a plan sha256, a missing file, an unloaded fiber): the docstring's exit code 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(plain(rep), indent=1), encoding='utf-8')
    _print_summary(rep, a.out)
    return code


if __name__ == '__main__':
    sys.exit(main())
