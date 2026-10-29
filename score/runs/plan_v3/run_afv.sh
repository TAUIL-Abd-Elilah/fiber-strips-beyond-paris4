#!/usr/bin/env bash
# PREREG v3: piece AFVs and exact infer.py run plans (heads on the full run's grid), after run_match.sh.
set -u
cd /c/VesuviusTemp/strips/score
export CUDA_VISIBLE_DEVICES=-1
PY=/c/Users/PC/miniconda3/envs/vesuvius/python.exe
V139=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr
V332=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr
L=runs/plan_v3
$PY -I -u tools/make_piece_afv.py results/matched_stretches.json data/fibers0139 data/matched_0139.afv \
  --volume $V139 --base-shape 76953 26511 26511 --full-afv ../data/human_afv/PHerc0139_human_controlled.afv \
  --map data/matched_0139.tsv --runplan results/runplan_0139_primary.tsv \
  --report results/matched_0139_afv_report.json > $L/log_afv_primary.txt 2>&1
echo "afv primary rc $?" >> $L/rc_afv.txt
$PY -I -u tools/make_piece_afv.py results/matched_stretches_secondary_run40.json data/fibers0139 \
  data/matched_0139_secondary_run40.afv --volume $V139 --base-shape 76953 26511 26511 \
  --full-afv ../data/human_afv/PHerc0139_human_controlled.afv --map data/matched_0139_secondary_run40.tsv \
  --runplan results/runplan_0139_secondary_run40.tsv \
  --report results/matched_0139_secondary_run40_afv_report.json > $L/log_afv_secondary.txt 2>&1
echo "afv secondary rc $?" >> $L/rc_afv.txt
$PY -I -u tools/make_piece_afv.py results/crossings_0332_plan.json data/fibers0332 data/crossings_0332.afv \
  --volume $V332 --base-shape 33592 15761 15761 --full-afv ../data/human_afv/PHerc0332_human_controlled.afv \
  --map data/crossings_0332.tsv --runplan results/runplan_0332_crossings.tsv \
  --report results/crossings_0332_afv_report.json > $L/log_afv_0332.txt 2>&1
echo "afv 0332 rc $?" >> $L/rc_afv.txt
