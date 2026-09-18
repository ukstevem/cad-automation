#!/usr/bin/env bash
# Placement guide loop (bd 6et), run from the laptop in Git Bash while the operator is at the rig.
#
#   tools/place_loop.sh outputs/ar_fits/place/tower          # until Ctrl+C
#   tools/place_loop.sh outputs/ar_fits/place/tower 1        # one check
#
# Each round: both rig cameras take a full-resolution shot (about 11 s - they open one at a time, which is what
# keeps the frames uncompressed), the pair is copied here, tools/place_guide.py checks it inside the running API
# container (no container start-up), and the laptop beeps: three rising notes for IN PLACE, one low note otherwise.
# Keep open: http://localhost:8000/<plan dir>/guide.html - it refreshes itself.
#
# The board must stay in both views; the part can move freely between rounds.
set -u
PLAN="${1:?usage: tools/place_loop.sh <plan dir under outputs/ar_fits> [rounds]}"
ROUNDS="${2:-0}"
RIG="${PLACE_RIG:-administrator@10.0.0.36}"
LABEL="place_live"
cd "$(dirname "$0")/.." || exit 1
LIVE="outputs/ar_captures/$LABEL"

beep() { powershell.exe -NoProfile -Command "$1" >/dev/null 2>&1 || true; }

echo "page: http://localhost:8000/${PLAN%/}/guide.html"
n=0
while :; do
  n=$((n + 1))
  start=$(date +%s)
  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$RIG" "cd ~ && python3 webcam_capture.py shot $LABEL >/dev/null 2>&1"; then
    echo "round $n: the rig did not take the shot"; beep "[console]::beep(400,600)"; sleep 3; continue
  fi
  files=$(ssh -o BatchMode=yes "$RIG" "ls -t ~/captures/${LABEL}_*.png 2>/dev/null | head -2 | xargs -n1 basename")
  mkdir -p "$LIVE" && rm -f "$LIVE"/*.png
  for f in $files; do scp -o BatchMode=yes -q "$RIG:captures/$f" "$LIVE/"; done
  ssh -o BatchMode=yes "$RIG" "rm -f ~/captures/${LABEL}_*.png"
  line=$(docker exec -w //app cad-automation-api python tools/place_guide.py check --plan "$PLAN" --captures "$LIVE" 2>&1 \
         | grep -E "^(IN PLACE|ADJUST|NOT FOUND|WRONG WAY ROUND):")
  echo "round $n ($(( $(date +%s) - start )) s): ${line:-check failed - run place_guide.py check by hand to see why}"
  case "$line" in
    "IN PLACE"*) beep "[console]::beep(1200,150);[console]::beep(1600,150);[console]::beep(2000,300)" ;;
    *) beep "[console]::beep(700,250)" ;;
  esac
  # the operator pressed Set on the page
  if [ -f "$PLAN/set.request" ]; then
    rm -f "$PLAN/set.request"
    echo "set: measuring this placement..."
    if docker exec -w //app cad-automation-api python tools/place_guide.py set --plan "$PLAN" --captures "$LIVE"; then
      docker exec -w //app cad-automation-api python tools/ar_view.py --captures "$LIVE" --fit "$PLAN/fit"         --welds "${WELDS:-outputs/welds/mainframe_ifc.json}" --scale "${SCALE:-0.2}" --out "$PLAN/fit"         | grep -E "welds shown|pose trust|typical"
      echo "weld view: http://localhost:8000/$PLAN/fit/ar_view.html"
      beep "[console]::beep(1200,150);[console]::beep(1600,150);[console]::beep(2000,150);[console]::beep(2400,400)"
      break
    fi
    echo "  not set - carry on placing"
    beep "[console]::beep(400,600)"
  fi
  if [ "$ROUNDS" -gt 0 ] && [ "$n" -ge "$ROUNDS" ]; then break; fi
done
