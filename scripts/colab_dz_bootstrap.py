"""Runs INSIDE a Colab VM via `colab exec -f` (see scripts/colab_dz.sh).

Finishes the Dangerzone (D5) evaluation on the GPU host: clones/updates the repo,
installs the light deps on top of Colab's preinstalled torch, and runs
dangerzone_finalize.py (model answers on the sanitized images) + run_analysis.py on a
run directory that lives on mounted Google Drive. The sanitized images must already be
under <RUN_DIR>/dz/ (produced locally by scripts/dangerzone_sanitize.py and synced to
Drive). The attack is never recomputed.

Env (set by colab_dz.sh via `colab exec --env`): RUN_DIR (required), HF_HOME.
"""
import os
import subprocess

RUN_DIR = os.environ["RUN_DIR"]
REPO_URL = os.environ.get("REPO_URL", "https://github.com/nongvantinh/computer-vision.git")
BRANCH = os.environ.get("BRANCH", "master")
PROJECT = "/content/cdr"


def sh(cmd: str) -> int:
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode


if os.path.exists(PROJECT + "/.git"):
    sh(f"git -C {PROJECT} fetch --all -q && git -C {PROJECT} checkout {BRANCH} -q && "
       f"git -C {PROJECT} pull --ff-only -q")
else:
    if sh(f"git clone -q --branch {BRANCH} {REPO_URL} {PROJECT}") != 0:
        raise SystemExit("clone failed (private repo? use a token URL or colab upload)")

assert os.path.isdir(f"{RUN_DIR}/dz"), (
    f"{RUN_DIR}/dz not found. Mount Drive (colab drivemount) and make sure the local "
    f"sanitize output was uploaded to <RUN_DIR>/dz/.")

sh("pip -q install transformers accelerate bitsandbytes scipy matplotlib pyyaml pillow")

env = (f"HF_HOME={os.environ.get('HF_HOME', '/content/hf_cache')} "
       "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg PYTHONUNBUFFERED=1")
rc = sh(f"cd {PROJECT} && {env} python scripts/dangerzone_finalize.py "
        f"--run '{RUN_DIR}' --config configs/experiment.yaml")
if rc == 0:
    sh(f"cd {PROJECT} && {env} python scripts/run_analysis.py --run '{RUN_DIR}'")
print("DONE" if rc == 0 else "FINALIZE FAILED", flush=True)
