#!/usr/bin/env bash
# Keyframe query-at-end Invascal batch. Self-gating, unattended-safe:
#   1. Wait for the GPU to free (lead's job may be running) before launching.
#   2. Run the smallest combo FIRST as a smoke gate; if its mIoU < 0.05 or the
#      written .label count != keyframe count, STOP the whole batch (write a
#      BROKEN marker) rather than burning the night on a broken pipeline.
#   3. Run the remaining combos; tolerate a per-combo failure/OOM (record and
#      continue) instead of dying entirely.
#
# Reuses the per-scan *_invascal staging (staged inputs + keyframes_5m.txt), so
# there is NO stage step here: run_convbki (two-pass keyframe) -> convert ->
# compute. run_convbki writes ONLY the keyframe predictions, so convert is
# naturally gated to the keyframe set.
#
# Writes results/per_combo/<combo>_invascal_kf.json (does NOT touch the ungated
# per-scan *_invascal.json).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."   # baselines/convbki
: "${OSM_BKI_DATA_ROOT:?set OSM_BKI_DATA_ROOT}"

MASTER_LOG="${1:-logs/invascal_kf_batch_$(date +%Y%m%d-%H%M%S).log}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"   # need this much free GPU before launching
GATE_MIOU="0.05"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

# Smallest first (kth_night_05_id, 178 kf) = the smoke gate.
COMBOS=(
  mcd_kth_night_05_id_invascal_kf
  mcd_kth_night_05_ood_invascal_kf
  mcd_kth_day_09_id_invascal_kf
  mcd_kth_day_09_ood_invascal_kf
  kitti360_seq0000_id_invascal_kf
  kitti360_seq0000_ood_invascal_kf
  kitti360_seq0009_id_invascal_kf
  kitti360_seq0009_ood_invascal_kf
)

log "=== Invascal KEYFRAME (query-at-end) batch: ${#COMBOS[@]} combos ==="

# ---- 1. Wait for GPU ---------------------------------------------------- #
log "Waiting for >= ${MIN_FREE_MIB} MiB free GPU (lead's job may be running)..."
waited=0
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ "${FREE:-0}" -ge "$MIN_FREE_MIB" ]]; then
    log "GPU free = ${FREE} MiB >= ${MIN_FREE_MIB}; launching after ${waited}s wait."
    break
  fi
  sleep 60
  waited=$((waited + 60))
  if (( waited % 600 == 0 )); then
    log "  still waiting: GPU free = ${FREE} MiB after ${waited}s"
  fi
done

declare -A STATUS
gate_checked=0

for combo in "${COMBOS[@]}"; do
  CFG="configs/${combo}.yaml"
  OUT_ROOT="$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['output_root'])")"
  EVAL_DIR="$(dirname "$OUT_ROOT")/evaluations_convbki"
  JSON="results/per_combo/${combo/_invascal_kf/}_invascal_kf.json"
  # normalize: results/per_combo/<combo_base>_invascal_kf.json
  JSON="results/per_combo/${combo}.json"
  mkdir -p "$EVAL_DIR" "$(dirname "$JSON")"

  log ">>> [$combo] START"
  {
    echo "===== $combo @ $(date) ====="
    echo "--- run_convbki (two-pass keyframe) ---"
    python scripts/run_convbki.py "$CFG" || { echo "RUN FAILED"; exit 12; }
    echo "--- convert_outputs (naturally keyframe-gated) ---"
    python scripts/convert_outputs.py "$CFG" --eval-dir "$EVAL_DIR" || { echo "CONVERT FAILED"; exit 13; }
    echo "--- compute_eval_numbers ---"
    python scripts/compute_eval_numbers.py "$EVAL_DIR" --out-json "$JSON" || { echo "COMPUTE FAILED"; exit 14; }
    echo "===== $combo DONE @ $(date) ====="
  } >>"$MASTER_LOG" 2>&1
  rc=$?

  if [[ $rc -ne 0 ]]; then
    STATUS[$combo]="FAILED(rc=$rc)"
    log ">>> [$combo] FAILED (rc=$rc)"
    if [[ $gate_checked -eq 0 ]]; then
      log "!!! GATE COMBO FAILED — stopping batch. See $MASTER_LOG"
      echo "GATE FAILED: $combo rc=$rc" > logs/KF_BATCH_BROKEN.flag
      break
    fi
    log ">>> continuing despite failure (non-gate combo)"
    continue
  fi

  STATUS[$combo]="OK"
  log ">>> [$combo] DONE -> $JSON"

  # ---- 2. Smoke gate on the first (smallest) combo -------------------- #
  if [[ $gate_checked -eq 0 ]]; then
    gate_checked=1
    read -r MIOU NFILES < <(python -c "
import json,sys
j=json.load(open('$JSON'))
print(j['miou'], j['n_files'])
")
    KF_EXPECTED=$(python -c "
import yaml,os,glob
c=yaml.safe_load(open('$CFG')); st=c['staging_root']
kf=set(open(os.path.join(st,'keyframes_5m.txt')).read().split())
staged=set(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(st+'/sequences/00/velodyne/*.bin'))
print(len(kf&staged))
")
    log "GATE: mIoU=$MIOU n_files=$NFILES expected_kf=$KF_EXPECTED"
    BAD=$(python -c "print(1 if (float('$MIOU') < $GATE_MIOU or int('$NFILES') != int('$KF_EXPECTED')) else 0)")
    if [[ "$BAD" == "1" ]]; then
      log "!!! SMOKE GATE FAILED (mIoU<$GATE_MIOU or file/keyframe mismatch) — stopping batch."
      echo "SMOKE GATE FAILED: $combo mIoU=$MIOU n_files=$NFILES expected=$KF_EXPECTED" > logs/KF_BATCH_BROKEN.flag
      break
    fi
    log "SMOKE GATE PASSED — proceeding with remaining combos."
  fi
done

log "=== KEYFRAME batch complete ==="
for combo in "${COMBOS[@]}"; do
  log "   ${STATUS[$combo]:-SKIPPED}  $combo"
done
