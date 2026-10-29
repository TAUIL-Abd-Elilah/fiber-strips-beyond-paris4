#!/usr/bin/env bash
# PREREG v3 plans: matched stretches (0139 primary, secondary) and the 0332 crossing plan, heads on the full run's grid.
set -u
cd /c/VesuviusTemp/strips/score
export CUDA_VISIBLE_DEVICES=-1
PY=/c/Users/PC/miniconda3/envs/vesuvius/python.exe
CT139=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/2
CT332=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr/2
L=runs/plan_v3
$PY -I -u score_strips.py match --fibers data/fibers0139 --space PHerc0139/20260102150214@L0 --segments data/segments \
  --umbilicus data/umbilicus/PHerc0139_20260102150214_umbilicus.json --ct-zarray $CT139 \
  --ct-cache C:/VesuviusTemp/strips/ct_cache --workers 5 --tmpdir runs/match_0139 \
  --full-run-map data/full_run_map_0139.npz --out results/matched_stretches.json > $L/log_primary.txt 2>&1
echo "primary rc $?" >> $L/rc.txt
$PY -I -u score_strips.py match --fibers data/fibers0139 --space PHerc0139/20260102150214@L0 --segments data/segments \
  --umbilicus data/umbilicus/PHerc0139_20260102150214_umbilicus.json --ct-zarray $CT139 \
  --ct-cache C:/VesuviusTemp/strips/ct_cache --workers 5 --tmpdir runs/match_0139 \
  --min-stretch-mm 0.38384 --analysis secondary --full-run-map data/full_run_map_0139.npz \
  --ct-cap-scan 8 13 0.1 --min-run-fibers 36 \
  --out results/matched_stretches_secondary_run40.json > $L/log_secondary.txt 2>&1
echo "secondary rc $?" >> $L/rc.txt
$PY -I -u score_strips.py crossplan --fibers data/fibers0332 --space PHerc0332/20251211183505@L0 \
  --ct-zarray $CT332 --ct-cache C:/VesuviusTemp/strips/ct_cache --full-run-map data/full_run_map_0332.npz \
  --out results/crossings_0332_plan.json > $L/log_crossplan.txt 2>&1
echo "crossplan rc $?" >> $L/rc.txt
