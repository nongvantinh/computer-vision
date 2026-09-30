"""Runs INSIDE a Colab VM via `colab exec` (see scripts/colab_run.sh).

scripts/colab_run.sh prepends a small parameter header (REPO_URL, BRANCH,
DRIVE_DIR, SESSION_HOURS, RUN_ID, MODE) and pipes this file to `colab exec`, so it
executes in the VM kernel. It clones the repo, points the HF cache and the run
directory at Google Drive, and runs the experiment with a time budget. Because the
run directory lives on Drive (`--runs-base`) and jobs are idempotent, re-running
continues where a previous session (or account) stopped.

MODE=pilot (default) runs the small persisted end-to-end pilot first (12 images,
eps 8/255) so the pipeline is proven before scaling. MODE=full runs the whole
matrix. Both are resumable under the same RUN_ID.

Colab VMs ship torch+CUDA preinstalled, so we reuse that and only add the light
deps instead of reinstalling a 2.5 GB torch wheel.
"""
import os
import subprocess

REPO_URL = globals().get("REPO_URL", "https://github.com/nongvantinh/computer-vision.git")
BRANCH = globals().get("BRANCH", "master")
DRIVE_DIR = globals().get("DRIVE_DIR", "/content/drive/MyDrive/computer-vision")
SESSION_HOURS = float(globals().get("SESSION_HOURS", 3.0))
RUN_ID = globals().get("RUN_ID", "pilot_12")
MODE = globals().get("MODE", "pilot")   # pilot | full
PROJECT = "/content/cdr"


def sh(cmd, cwd=None, env=None, check=False):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd, env=env).returncode


# 1. code: clone once, then fast-forward on later sessions
if os.path.exists(PROJECT + "/.git"):
    sh(f"git -C {PROJECT} fetch --all -q && git -C {PROJECT} checkout {BRANCH} -q && "
       f"git -C {PROJECT} pull --ff-only -q")
else:
    if sh(f"git clone -q --branch {BRANCH} {REPO_URL} {PROJECT}") != 0:
        raise SystemExit(f"clone failed; for a private repo use a token URL or colab upload")

# 2. durable Drive tree (small: results + state only; model/dataset stay ephemeral)
for sub in ("results", "state", "reports"):
    os.makedirs(f"{DRIVE_DIR}/{sub}", exist_ok=True)

env = os.environ.copy()
# Free Drive is 15 GB and the model is ~14 GB, so the HF cache stays on ephemeral
# /content (re-downloads each session) and only small results/state go to Drive.
env.update({"HF_HOME": "/content/hf_cache", "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

# 3. light deps (torch/numpy already present on Colab). bitsandbytes enables the
#    4-bit load that fits LLaVA-1.5-7B + the backward pass on a 16 GB T4.
sh("pip -q install transformers accelerate bitsandbytes scipy matplotlib pyyaml pillow",
   check=True)

# Dangerzone (D5) is BLOCKED on Colab: it needs rootless Podman (--userns nomap), but
# Colab runs as root with a locked-down kernel (podman run fails rc 125/126). So D5 is
# not installed here and is excluded from the run below. See docs/implementation/
# dangerzone.md; run it on a rootless-Podman host to get the D5 data point.

# 4. dataset: fetch the free 20-class subset to ephemeral disk (deterministic, so the
#    manifest is identical every session/account). Only results/state persist to Drive.
sh("python scripts/fetch_imagenette.py --out data/imagenet/val", cwd=PROJECT, env=env)

RUNS_BASE = f"{DRIVE_DIR}/results/runs"   # run dir on Drive -> resumable across sessions
OUT = f"{RUNS_BASE}/{RUN_ID}"

# 5. pipeline. Each step is resumable; run_experiment honors the time budget.
sh("python scripts/build_dataset.py --config configs/experiment.yaml", cwd=PROJECT, env=env)
sh("python scripts/test_llava.py", cwd=PROJECT, env=env)                 # checkpoint 1
sh("python scripts/validate_attack.py --epsilon 0.0313725490", cwd=PROJECT, env=env)  # ckpt 3

pilot = "--limit 12 --epsilon 0.0313725490" if MODE == "pilot" else ""
sh(f"python scripts/run_experiment.py --run-id {RUN_ID} --runs-base '{RUNS_BASE}' "
   f"--defenses D0,jpeg90,jpeg75,jpeg50 "   # D5 excluded: Dangerzone can't run on Colab
   f"--max-hours {SESSION_HOURS} {pilot}", cwd=PROJECT, env=env)
sh(f"python scripts/run_analysis.py --run '{OUT}'", cwd=PROJECT, env=env)

# 6. light bundle for `colab download` (tables + plots, no heavy images/model)
sh(f"cd '{OUT}' && zip -qr /content/{RUN_ID}_light.zip summary.json statistics.json "
   f"plots *_results.jsonl config.json environment.json 2>/dev/null || true")
print("DONE. Durable run on Drive at", OUT)
