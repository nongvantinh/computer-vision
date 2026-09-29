"""Runs INSIDE a Colab VM via `colab exec` (see scripts/colab_run.sh).

scripts/colab_run.sh prepends a small parameter header (REPO_URL, BRANCH,
DRIVE_DIR, SESSION_HOURS) and pipes this file to `colab exec`, so it executes in
the VM kernel. It clones the repo, points the HF cache and the experiment output
at Google Drive, and runs the experiment with a time budget. Because `--out` is
on Drive and jobs are idempotent, re-running continues where a previous session
(or account) stopped.

Colab VMs ship torch+CUDA preinstalled, so we reuse that and only add the light
deps instead of reinstalling a 2.5 GB torch wheel.
"""
import os
import subprocess

REPO_URL = globals().get("REPO_URL", "https://github.com/nongvantinh/computer-vision.git")
BRANCH = globals().get("BRANCH", "master")
DRIVE_DIR = globals().get("DRIVE_DIR", "/content/drive/MyDrive/computer-vision")
SESSION_HOURS = float(globals().get("SESSION_HOURS", 3.0))
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

# 2. durable Drive tree; HF cache on Drive so the 14 GB model downloads once
for sub in ("results", "state", "hf_cache", "imagenet", "reports"):
    os.makedirs(f"{DRIVE_DIR}/{sub}", exist_ok=True)

env = os.environ.copy()
env.update({"HF_HOME": f"{DRIVE_DIR}/hf_cache", "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

# 3. light deps (torch/numpy already present on Colab)
sh("pip -q install transformers accelerate scipy matplotlib pyyaml pillow", check=True)

# 4. ImageNet subset lives on Drive; expose it where the config expects it
val_link = f"{PROJECT}/data/imagenet/val"
os.makedirs(os.path.dirname(val_link), exist_ok=True)
if not os.path.exists(val_link):
    os.symlink(f"{DRIVE_DIR}/imagenet/val", val_link)

OUT = f"{DRIVE_DIR}/results/mvp"    # state on Drive -> resumable across sessions/accounts

# 5. pipeline. Each step is resumable; run_experiment honors the time budget.
sh("python scripts/build_dataset.py --config configs/experiment.yaml", cwd=PROJECT, env=env)
sh("python scripts/test_llava.py", cwd=PROJECT, env=env)                 # checkpoint 1
sh(f"python scripts/run_experiment.py --out '{OUT}' --max-hours {SESSION_HOURS}",
   cwd=PROJECT, env=env)
sh(f"python scripts/run_analysis.py --run '{OUT}'", cwd=PROJECT, env=env)

# 6. light bundle for `colab download` (tables + plots, no heavy images/model)
sh(f"cd '{OUT}' && zip -qr /content/mvp_light.zip statistics plots metrics clean "
   f"jobs 2>/dev/null || true")
print("DONE. Durable results on Drive at", OUT)
