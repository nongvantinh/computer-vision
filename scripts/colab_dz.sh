#!/usr/bin/env bash
# Finish the Dangerzone (D5) evaluation on Colab via the CLI. You do the two
# interactive auth steps; this script does the rest and pulls the result back.
#
# Prereqs (you, interactive, in a normal terminal):
#   colab new -s cdr --gpu T4     # paste the Google auth token
#   colab drivemount -s cdr       # paste the auth token; mounts your Drive on the VM
#
# Then run this (the D5 sanitized images must already be on Drive under
# <DRIVE_DIR>/results/runs/<RUN>/dz/, produced by scripts/dangerzone_sanitize.py):
#   RUN=pilot_12 scripts/colab_dz.sh
#
# It updates the repo on the VM, runs dangerzone_finalize.py + run_analysis.py on the
# run dir (~10-15 min: the model downloads once), and downloads statistics.json here.
set -euo pipefail

SESSION="${SESSION:-cdr}"
RUN="${RUN:-pilot_12}"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/computer-vision}"
RUN_DIR="$DRIVE_DIR/results/runs/$RUN"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

command -v colab >/dev/null || { echo "Colab CLI missing: uv tool install google-colab-cli"; exit 1; }
colab sessions 2>/dev/null | grep -q . || echo "  (no sessions listed; make sure 'colab new -s $SESSION' ran)"

echo "[1/2] run D5 finalize + analysis on the VM (budget ~40 min; model loads once)"
colab exec -s "$SESSION" -f "$HERE/scripts/colab_dz_bootstrap.py" \
    --timeout 2400 --env "RUN_DIR=$RUN_DIR" --env "HF_HOME=/content/hf_cache"

echo "[2/2] download the refreshed statistics -> results/from_colab/"
mkdir -p "$HERE/results/from_colab"
colab download -s "$SESSION" "$RUN_DIR/statistics.json" \
    "$HERE/results/from_colab/${RUN}_statistics.json"
echo "done -> results/from_colab/${RUN}_statistics.json"
echo "(the full run dir, incl. plots, stays on Drive at $RUN_DIR)"
