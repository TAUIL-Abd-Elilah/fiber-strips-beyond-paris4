# Data license: CC BY-NC 4.0

The code in this repository is MIT (see `LICENSE`). The data files listed below are not code. They are derived
from Vesuvius Challenge open data and from the Vesuvius Challenge human fiber annotations, and they are released
under the **Creative Commons Attribution-NonCommercial 4.0 International** license:
https://creativecommons.org/licenses/by-nc/4.0/

## What this covers

- `score/results/*.json`, `score/results/*.tsv`: matched stretches, noise floors, crossing plan, run plans and AFV
  reports (and their v2 versions in `score/superseded_v2/results/`, and `score/superseded_v1/g1_v1_result.json` with
  its log `g1_v1_log.txt`).
- `score/data/*_manifest.json` (and `score/superseded_v2/data/*_manifest.json`), `manifests/*.json`,
  `data/g0/published/manifest.json`: file lists with URLs, sizes and checksums. They hold no annotation, scan or
  model content.
- `results/*.json` (G0 comparison numbers and the step-6 CT estimates).
- `score/results/erratum_v31/*.json` and `score/runs/erratum_v31/paris4/plan.json` (the v3.1 checks: strata,
  fiber geometry, and the Paris 4 piece runs).
- **`score/tests/fixtures/paris4_real/`** (v3.1 test fixture, Paris 4 only): a real `fiber_strip_small_v1` strip of a
  piece of Paris 4 fiber 778459 with its decisions (`pieces/QA/`: tifxyz, `strip.npz`, `summary.json`), the G0 CPU
  strip of the whole fiber (`g0_fiber_778459/`: tifxyz and `expected_error.tif`), and that fiber's polyline
  (`fiber_778459.json`, from `dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/fibers.afv`). These are model
  predictions on the PHercParis4 20260411134726 scan and a fiber track derived from it. Their metadata keep two
  local paths of the author's workspace (the run's AFV and output folder).
- `score/results/v3/*.json`: the scored run's three reports, byte for byte as the frozen scorer wrote them (their
  `provenance` keeps the run's workspace paths).
- `score/runs/v3/` run records (`log.txt`, `status.json`, `DONE*`, `V3_FINAL`, the stage `*.out`/`*.err`,
  `records/`, `pieces/`) and `score/runs/relaunch_v3.log`, as the runner wrote them (they keep the workspace
  paths).
- `analysis/exploratory_postrun.json` (exploratory numbers computed from the reports, the plans and the strips),
  `figures/` (figures made from the reports, and two example renders of PHerc0139 CT sampled on predicted strips) and
  `verification/*.json` (the independent re-scoring sample and its comparison).
- The GitHub release `strips-v3`: the 73 strips of the scored run (`PHerc0139_primary_strips.zip`,
  `PHerc0139_secondary_strips.zip`, `PHerc0332_crossing_strips.zip`), with `manifest.json`, `README.md` and
  `SHA256SUMS.txt`. Local paths in their JSON members were replaced by `<local>`; nothing else changed.
- Any strip, `expected_error` map or per-stretch table added later, in this repository or in a GitHub release.

## Attribution

- **Scans, wrap segments, Lasagna predictions, umbilici:** Vesuvius Challenge open data
  (`s3://vesuvius-challenge-open-data`), CC BY-NC 4.0. Volumes: PHerc0139 20260102150214 (2.399 µm), PHerc0332
  20251211183505 (2.399 µm), PHercParis4 20260411134726 (2.400 µm).
- **Human fiber annotations:** the Vesuvius Challenge annotators whose files carry the initials lt, kb and dj
  (https://dl.ash2txt.org/other/annotators/0139_fibers/ and /0332_fibers/). The folders carry no license file;
  we treat derived data as CC BY-NC 4.0 because every file names a CC BY-NC 4.0 source volume and the open-data
  schema says derived artifacts follow the license of their volume. This is our reading, not a grant.
- **Strip model:** `scrollprize/fiber_strip_small_v1` by Sean Johnson (bruniss), villa PR #2110.

## Changes

These files were made by us, not by Vesuvius Challenge: the stretch tables come from matching the human fibers to
the wrap segments with `score/score_strips.py`; the strips are predictions of `fiber_strip_small_v1` run on pieces
of the human fibers, and the reports, analyses and figures are computed from them. The Paris 4 fixture strips are
predictions of `fiber_strip_small_v1` run on a Paris 4 fiber track (the piece strip by us; the whole-fiber strip by
us as a CPU reproduction of the organizers' published strip).
No CT volume, raw human fiber JSON, segment mesh, ink image or model weight is included; the two example renders
(`figures/example_strip_*.png`) show PHerc0139 CT (20260102150214, Vesuvius Challenge open data, CC BY-NC 4.0)
sampled on a predicted strip.

We add no restriction beyond CC BY-NC 4.0.
