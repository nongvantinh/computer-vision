#!/usr/bin/env bash
# Drive the CDR-VLM experiment on a Colab GPU from this terminal, via the Colab CLI.
#
# Prereq (one time, interactive, you run it): authenticate the CLI, e.g.
#     colab new --gpu T4        # first use opens the Google login
#     colab usage               # shows this account's compute-unit balance
#
# Usage:
#     GPU=A100 HOURS=3 scripts/colab_run.sh
# Env vars (all optional):
#     GPU=A100|L4|T4|H100|G4   HOURS=3   SESSION=cdr   HIGHMEM=0|1   KEEP=0|1
#     REPO_URL=...   BRANCH=master   DRIVE_DIR=/content/drive/MyDrive/computer-vision
#
# Re-run to CONTINUE: finished jobs are skipped and state lives on Drive, so a run
# stopped by a usage limit resumes here or on another account (re-auth, re-run).
set -euo pipefail

GPU="${GPU:-A100}"; HOURS="${HOURS:-3}"; SESSION="${SESSION:-cdr}"
HIGHMEM="${HIGHMEM:-0}"; KEEP="${KEEP:-0}"; BRANCH="${BRANCH:-master}"
REPO_URL="${REPO_URL:-https://github.com/nongvantinh/computer-vision.git}"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/computer-vision}"
RUN_ID="${RUN_ID:-pilot_12}"; MODE="${MODE:-pilot}"   # MODE=pilot|full
HERE="$(cd "$(dirname "$0")/.." && pwd)"

command -v colab >/dev/null || { echo "Colab CLI missing: uv tool install google-colab-cli"; exit 1; }

echo "[1/5] provision $GPU (session '$SESSION')"
new_args=(-s "$SESSION" --gpu "$GPU")
[ "$HIGHMEM" = 1 ] && new_args+=(--high-mem)
colab new "${new_args[@]}"

echo "[2/5] mount Google Drive on the VM"
colab drivemount -s "$SESSION"

echo "[3/5] run the experiment on the VM (budget ${HOURS}h; streams below)"
params=$(cat <<PY
REPO_URL = "${REPO_URL}"
BRANCH = "${BRANCH}"
DRIVE_DIR = "${DRIVE_DIR}"
SESSION_HOURS = ${HOURS}
RUN_ID = "${RUN_ID}"
MODE = "${MODE}"
PY
)
{ printf '%s\n' "$params"; cat "$HERE/scripts/colab_bootstrap.py"; } | colab exec -s "$SESSION"

echo "[4/5] pull the light results bundle -> results/from_colab/"
mkdir -p "$HERE/results/from_colab"
colab download -s "$SESSION" "/content/${RUN_ID}_light.zip" \
    "$HERE/results/from_colab/${RUN_ID}_light.zip" || echo "  (no bundle yet; check the log above)"

if [ "$KEEP" = 1 ]; then
    echo "[5/5] keeping VM '$SESSION' alive (KEEP=1); stop it with: colab stop -s $SESSION"
else
    echo "[5/5] stopping VM '$SESSION'"
    colab stop -s "$SESSION"
fi
echo "done. Durable run on Drive at ${DRIVE_DIR}/results/runs/${RUN_ID} . Re-run (same RUN_ID) to continue."
