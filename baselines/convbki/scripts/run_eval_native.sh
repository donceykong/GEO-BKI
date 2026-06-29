#!/usr/bin/env bash
# Drive a native-9-class Conv-BKI eval end-to-end:
#   stage_inputs (native_common) -> run_convbki -> convert_outputs -> compute.
#
# Usage:
#   run_eval_native.sh <config.yaml> <eval_out_root> <per_combo_json> [keyframe_dir]
#
#   <config.yaml>     per-experiment YAML with native_common: true
#   <eval_out_root>   dir under /home/sandilya for the eval txt files
#   <per_combo_json>  path for compute_eval_numbers.py --out-json
#   [keyframe_dir]    optional: gate convert to stems present here (else all GT scans)
set -euo pipefail

CONFIG="$1"
EVAL_DIR="$2"
PER_COMBO_JSON="$3"
KEYFRAME_DIR="${4:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."   # baselines/convbki

: "${OSM_BKI_DATA_ROOT:?set OSM_BKI_DATA_ROOT}"
mkdir -p "$EVAL_DIR" "$(dirname "$PER_COMBO_JSON")"

echo "=== [1/4] stage_inputs (native) ==="
python scripts/stage_inputs.py "$CONFIG"

echo "=== [2/4] run_convbki inference ==="
python scripts/run_convbki.py "$CONFIG"

echo "=== [3/4] convert_outputs -> $EVAL_DIR ==="
if [[ -n "$KEYFRAME_DIR" ]]; then
  python scripts/convert_outputs.py "$CONFIG" --eval-dir "$EVAL_DIR" \
    --keyframe-stems-from "$KEYFRAME_DIR"
else
  python scripts/convert_outputs.py "$CONFIG" --eval-dir "$EVAL_DIR"
fi

echo "=== [4/4] compute_eval_numbers -> $PER_COMBO_JSON ==="
python scripts/compute_eval_numbers.py "$EVAL_DIR" --out-json "$PER_COMBO_JSON"

echo "=== EVAL COMPLETE: $PER_COMBO_JSON ==="
