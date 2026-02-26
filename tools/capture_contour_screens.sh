#!/usr/bin/env bash
set -euo pipefail

# Captures a sequence of device screenshots for contour validation.
#
# Usage:
#   tools/capture_contour_screens.sh                         # capture 1 frame
#   tools/capture_contour_screens.sh 5 0.5                   # capture 5 frames, 0.5s apart
#   tools/capture_contour_screens.sh 1 0.0 straight_on        # label the capture set
#
# Output:
#   Saves PNGs into ./captures_visible/ with timestamps.

COUNT="${1:-1}"
INTERVAL_SEC="${2:-0.0}"
LABEL="${3:-}"

if [[ -n "${ADB:-}" ]]; then
  ADB="${ADB}"
elif [[ -x "./tools/platform-tools/adb" ]]; then
  ADB="./tools/platform-tools/adb"
else
  ADB="adb"
fi
OUT_DIR="captures_visible"

mkdir -p "${OUT_DIR}"

ts() { date +"%Y%m%d_%H%M%S"; }

echo "Checking ADB devices..."
"${ADB}" devices
if ! "${ADB}" devices | tail -n +2 | awk '{print $2}' | grep -q '^device$'; then
  echo "No connected/authorized devices found. Fix ADB then re-run."
  echo "Tip: ${ADB} kill-server && ${ADB} start-server"
  exit 2
fi

for ((i=0; i<COUNT; i++)); do
  if [[ -n "${LABEL}" ]]; then
    FN="${OUT_DIR}/contour_${LABEL}_$(ts)_${i}.png"
  else
    FN="${OUT_DIR}/contour_test_$(ts)_${i}.png"
  fi
  echo "Capturing ${FN}"
  "${ADB}" exec-out screencap -p > "${FN}"
  if [[ "${INTERVAL_SEC}" != "0.0" ]]; then
    sleep "${INTERVAL_SEC}"
  fi
done

echo "Done."

