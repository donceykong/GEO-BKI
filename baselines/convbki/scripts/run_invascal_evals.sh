#!/usr/bin/env bash
# Batch-run the 7 Invascal (lidarrv) Conv-BKI eval combos start to finish.
#
# Each combo: stage_inputs (skipped if already fully staged -> manifest.json
# present) -> run_convbki -> convert_outputs (UNGATED, all-GT-scan basis;
# S-BKI keyframe sets absent) -> compute_eval_numbers -> per-combo JSON.
#
# Continues past a failing combo (records it) so one bad combo does not kill
# the batch. All bulk artifacts land under /home/sandilya (data drive is
# read-only). Writes results/per_combo/<combo>.json (combo names already end
# in _invascal, so existing *_retrained.json CENet files are untouched).
#
# Usage: run_invascal_evals.sh <master_log_path>
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."   # baselines/convbki

: "${OSM_BKI_DATA_ROOT:?set OSM_BKI_DATA_ROOT}"

MASTER_LOG="${1:-logs/invascal_batch_$(date +%Y%m%d-%H%M%S).log}"
EVAL_BASE=/home/sandilya/convbki_train_workspace/eval

COMBOS=(
  kitti360_seq0000_id_invascal
  kitti360_seq0000_ood_invascal
  kitti360_seq0009_id_invascal
  mcd_kth_day_09_id_invascal
  mcd_kth_day_09_ood_invascal
  mcd_kth_night_05_id_invascal
  mcd_kth_night_05_ood_invascal
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

log "=== Invascal batch start: ${#COMBOS[@]} combos ==="
log "OSM_BKI_DATA_ROOT=$OSM_BKI_DATA_ROOT"
declare -A STATUS

for combo in "${COMBOS[@]}"; do
  CFG="configs/${combo}.yaml"
  EVAL_DIR="$EVAL_BASE/${combo}/evaluations_convbki"
  JSON="results/per_combo/${combo}.json"
  STAGING_ROOT="$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['staging_root'])")"
  MANIFEST="$STAGING_ROOT/manifest.json"
  mkdir -p "$EVAL_DIR" "$(dirname "$JSON")"

  log ">>> [$combo] START"
  {
    echo "===== $combo @ $(date) ====="
    if [[ -f "$MANIFEST" ]]; then
      echo "--- stage_inputs: SKIP (manifest present at $MANIFEST) ---"
    else
      echo "--- [1/4] stage_inputs ---"
      python scripts/stage_inputs.py "$CFG" || { echo "STAGE FAILED"; exit 11; }
    fi
    echo "--- [2/4] run_convbki ---"
    python scripts/run_convbki.py "$CFG" || { echo "RUN FAILED"; exit 12; }
    echo "--- [3/4] convert_outputs (ungated) ---"
    python scripts/convert_outputs.py "$CFG" --eval-dir "$EVAL_DIR" || { echo "CONVERT FAILED"; exit 13; }
    echo "--- [4/4] compute_eval_numbers ---"
    python scripts/compute_eval_numbers.py "$EVAL_DIR" --out-json "$JSON" || { echo "COMPUTE FAILED"; exit 14; }
    echo "===== $combo DONE @ $(date) ====="
  } >>"$MASTER_LOG" 2>&1

  rc=$?
  if [[ $rc -eq 0 ]]; then
    STATUS[$combo]="OK"
    log ">>> [$combo] DONE -> $JSON"
  else
    STATUS[$combo]="FAILED(rc=$rc)"
    log ">>> [$combo] FAILED (rc=$rc) — continuing"
  fi
done

log "=== Invascal batch complete ==="
for combo in "${COMBOS[@]}"; do
  log "   ${STATUS[$combo]:-?}  $combo"
done
