#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="$ROOT_DIR/tools/platform-tools/adb"

if [[ ! -x "$ADB" ]]; then
  echo "adb not found at: $ADB" >&2
  echo "Run: (from repo root) mkdir -p tools && download platform-tools" >&2
  exit 1
fi

STATE="$("$ADB" get-state 2>/dev/null || true)"
if [[ "$STATE" != "device" ]]; then
  echo "No connected/authorized Android device found." >&2
  echo "" >&2
  echo "Do this on your phone, then re-run:" >&2
  echo "- Connect USB (or start emulator)" >&2
  echo "- Enable Developer options + USB debugging" >&2
  echo "- Accept the 'Allow USB debugging' prompt" >&2
  echo "" >&2
  echo "Current devices:" >&2
  "$ADB" devices -l >&2 || true
  exit 2
fi

REMOTE_DIR="/sdcard/Pictures/Ham"

OUT_DIR="$ROOT_DIR/captures/ham"
mkdir -p "$OUT_DIR"

stamp="$(date +"%Y%m%d_%H%M%S")"
DEST="$OUT_DIR/$stamp"
mkdir -p "$DEST"

echo "Pulling latest Ham photos from: $REMOTE_DIR"

list="$("$ADB" shell "ls -1t \"$REMOTE_DIR\"/ham_*.jpg 2>/dev/null | head -n 30" | tr -d '\r' || true)"
if [[ -z "$list" ]]; then
  echo "No ham_*.jpg found in $REMOTE_DIR" >&2
  exit 3
fi

echo "Recent files (top 30):"
echo "$list" | sed 's/^/  - /'

echo ""
echo "Pulling top 12 into: $DEST"
echo "$list" | head -n 12 | while IFS= read -r remote; do
  [[ -z "$remote" ]] && continue
  base="$(basename "$remote")"
  "$ADB" pull "$remote" "$DEST/$base" >/dev/null
  echo "  pulled: $base"
done

echo ""
echo "Done. Local files:"
ls -1 "$DEST" | sed 's/^/  - /'

