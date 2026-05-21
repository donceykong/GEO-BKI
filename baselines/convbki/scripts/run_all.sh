#!/bin/bash
# Conv-BKI baseline: run the 8 (sequence, CENet) combos sequentially.
#
# For each combo:
#   - stage if not already done
#   - run inference
#   - convert outputs (keyframe-gated against the matching S-BKI eval dir)
#   - compute eval numbers
#   - hard-abort the loop if mIoU < 0.03 (per the build plan's safety bar)
#
# After the loop, aggregate_results.py emits results/{in_domain.md,
# cross_domain.md, raw_numbers.json}.
#
# Run with `nohup ... &` or under screen/tmux — the 10-minute Bash tool
# timeout doesn't apply when invoked directly from a shell.

set -uo pipefail

REPO=/home/sgarimella34/OSM-BKI-ROS
cd "$REPO"
export OSM_BKI_DATA_ROOT="${OSM_BKI_DATA_ROOT:-/media/sgarimella34/hercules-collect3/datasets}"
PY=/home/sgarimella34/miniforge3/envs/Where2comm/bin/python

mkdir -p baselines/convbki/logs baselines/convbki/results

COMBOS=(
  kitti360_seq0000_id
  kitti360_seq0000_ood
  kitti360_seq0009_id
  kitti360_seq0009_ood
  mcd_kth_day_09_id
  mcd_kth_day_09_ood
  mcd_kth_night_05_id
  mcd_kth_night_05_ood
)

declare -A KEYFRAME_DIR=(
  [kitti360_seq0000_id]=static_gaussian_indomain
  [kitti360_seq0000_ood]=static_gaussian_crossdomain
  [kitti360_seq0009_id]=static_gaussian_indomain
  [kitti360_seq0009_ood]=indomain_with_height
  [mcd_kth_day_09_id]=mcdlabels_no_height
  [mcd_kth_day_09_ood]=kitti360labels_no_height
  [mcd_kth_night_05_id]=
  [mcd_kth_night_05_ood]=
)

declare -A DATASET=(
  [kitti360_seq0000_id]=kitti360  [kitti360_seq0000_ood]=kitti360
  [kitti360_seq0009_id]=kitti360  [kitti360_seq0009_ood]=kitti360
  [mcd_kth_day_09_id]=mcd         [mcd_kth_day_09_ood]=mcd
  [mcd_kth_night_05_id]=mcd       [mcd_kth_night_05_ood]=mcd
)
declare -A SEQ=(
  [kitti360_seq0000_id]=2013_05_28_drive_0000_sync
  [kitti360_seq0000_ood]=2013_05_28_drive_0000_sync
  [kitti360_seq0009_id]=2013_05_28_drive_0009_sync
  [kitti360_seq0009_ood]=2013_05_28_drive_0009_sync
  [mcd_kth_day_09_id]=kth_day_09
  [mcd_kth_day_09_ood]=kth_day_09
  [mcd_kth_night_05_id]=kth_night_05
  [mcd_kth_night_05_ood]=kth_night_05
)

OVERALL_TS=$(date +%Y%m%d-%H%M%S)
ORCHESTRATOR_LOG="baselines/convbki/logs/run_all_${OVERALL_TS}.log"
echo "===== run_all started at $(date) =====" | tee "$ORCHESTRATOR_LOG"

for combo in "${COMBOS[@]}"; do
  TS=$(date +%Y%m%d-%H%M%S)
  cfg="baselines/convbki/configs/${combo}.yaml"
  log="baselines/convbki/logs/${combo}_${TS}.log"
  staging="baselines/convbki/staging/${combo}"
  output="baselines/convbki/nbki_runs/${combo}"
  ds=${DATASET[$combo]}; seq=${SEQ[$combo]}; kf=${KEYFRAME_DIR[$combo]}
  eval_dir="$OSM_BKI_DATA_ROOT/${ds}/${seq}/evaluations/convbki"

  echo "===== START $combo $(date) =====" | tee -a "$ORCHESTRATOR_LOG"
  echo "  cfg=$cfg  log=$log  staging=$staging  output=$output" | tee -a "$ORCHESTRATOR_LOG"

  # ---- stage ------------------------------------------------------------
  if [ ! -f "$staging/manifest.json" ]; then
    echo "[$combo] staging..." | tee -a "$ORCHESTRATOR_LOG"
    $PY baselines/convbki/scripts/stage_inputs.py "$cfg" > "$log" 2>&1
    if [ $? -ne 0 ]; then
      echo "[$combo] STAGING FAILED (see $log)" | tee -a "$ORCHESTRATOR_LOG"
      continue
    fi
  else
    echo "[$combo] staging already present, skipping" | tee -a "$ORCHESTRATOR_LOG"
  fi

  # ---- inference --------------------------------------------------------
  echo "[$combo] inference..." | tee -a "$ORCHESTRATOR_LOG"
  $PY baselines/convbki/scripts/run_convbki.py "$cfg" >> "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$combo] INFERENCE FAILED (see $log)" | tee -a "$ORCHESTRATOR_LOG"
    continue
  fi

  # ---- convert ----------------------------------------------------------
  rm -rf "$eval_dir"; mkdir -p "$eval_dir"
  KF_ARGS=()
  if [ -n "$kf" ] && [ -d "$OSM_BKI_DATA_ROOT/${ds}/${seq}/evaluations/$kf" ]; then
    KF_ARGS=(--keyframe-stems-from "$OSM_BKI_DATA_ROOT/${ds}/${seq}/evaluations/$kf")
    echo "[$combo] keyframe gating: $kf" | tee -a "$ORCHESTRATOR_LOG"
  else
    echo "[$combo] no keyframe source -- writing every scan that has GT" | tee -a "$ORCHESTRATOR_LOG"
  fi
  $PY baselines/convbki/scripts/convert_outputs.py "$cfg" "${KF_ARGS[@]}" --copy-raw >> "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$combo] CONVERT FAILED (see $log)" | tee -a "$ORCHESTRATOR_LOG"
    continue
  fi

  # ---- compute eval numbers --------------------------------------------
  $PY baselines/convbki/scripts/compute_eval_numbers.py "$eval_dir" >> "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$combo] EVAL FAILED (see $log)" | tee -a "$ORCHESTRATOR_LOG"
    continue
  fi

  miou=$($PY -c "import json; d=json.load(open('$eval_dir/raw_numbers.json')); print(d['miou'])")
  echo "[$combo] mIoU = $miou" | tee -a "$ORCHESTRATOR_LOG"
  # Hard-abort threshold: mIoU < 0.03
  if $PY -c "import sys; sys.exit(0 if float(sys.argv[1]) < 0.03 else 1)" "$miou"; then
    echo "[$combo] ABORTING: mIoU $miou below 0.03 threshold." | tee -a "$ORCHESTRATOR_LOG"
    cat >> baselines/convbki/DEBUG_REPORT.md <<EOF


---

## Aborted on $combo (mIoU $miou < 3%)

See $log for details. Remaining combos were not run.
EOF
    break
  fi

  # ---- cleanup ----------------------------------------------------------
  # Predictions are archived in raw_predictions/, eval .txt files are
  # written under <seq>/evaluations/convbki/, and raw_numbers.json holds
  # the per-combo numbers. The staging tree and the raw .label predictions
  # under nbki_runs/ are now redundant and would otherwise blow out the
  # drive across 8 combos. Drop them once we've passed the floor.
  echo "[$combo] cleanup: deleting staging tree and nbki_runs/<combo>" | tee -a "$ORCHESTRATOR_LOG"
  rm -rf "$staging/sequences"
  rm -rf "$output/sequences"

  echo "===== DONE $combo $(date) =====" | tee -a "$ORCHESTRATOR_LOG"
done

# ---- aggregate everything we got ----------------------------------------
echo "===== aggregating =====" | tee -a "$ORCHESTRATOR_LOG"
$PY baselines/convbki/scripts/aggregate_results.py | tee -a "$ORCHESTRATOR_LOG"

echo "===== run_all finished at $(date) =====" | tee -a "$ORCHESTRATOR_LOG"
