"""Phase 2 stages, run INSIDE a Colab VM (exec this file in the kernel).

Parameters are read from the kernel globals (set them in the cell before exec'ing, or
leave the defaults): STAGE, BRANCH, DRIVE_DIR, SESSION_HOURS, DRY_RUN, plus the stage
parameters below. Run directories are written on the VM's local disk and copied to
Drive with rsync (many small job files are slow to write straight to Drive); every
job is idempotent, so a reclaimed VM resumes where it stopped.

STAGE values
  controls      defense-only run on the STORED baseline adversarial images (no attack):
                D0, JPEG 95/90/75/50/30/10, adapter_only(+_ll), resample_only,
                chroma_only, noise1, noise2. Run dir: controls_v1.
  dz_answers    model answers on the locally sanitized Dangerzone images, for the
                legacy (dz/) and lossless (dz_ll/) variants. Needs those folders on
                Drive under <baseline run>/. Writes into controls_v1.
  pilot         adaptive attack pilot (PILOT_IMAGES images, eps 16/255) through
                JPEG 50, Dangerzone-lossless and legacy Dangerzone. Run dirs
                pilot_adaptive_<name>.
  pilot_eval    defended-loss evaluation (gate G3) for the pilot runs.
  pilot_dz      model answers for pilot adversarial images after the local, real
                Dangerzone sanitize (dz/ or dz_ll/ uploaded into each pilot run dir).
  analysis      run_analysis.py on a run (RUN_ID).
"""
import json
import os
import subprocess

STAGE = globals().get("STAGE", "controls")
REPO_URL = globals().get("REPO_URL", "https://github.com/nongvantinh/computer-vision.git")
BRANCH = globals().get("BRANCH", "master")
DRIVE_DIR = globals().get("DRIVE_DIR", "/content/drive/MyDrive/computer-vision")
SESSION_HOURS = float(globals().get("SESSION_HOURS", 3.0))
DRY_RUN = bool(globals().get("DRY_RUN", False))
BASE_RUN = globals().get("BASE_RUN", "full_200")          # holds the stored adversarial images
PILOT_IMAGES = int(globals().get("PILOT_IMAGES", 10))
PILOT_EPS = globals().get("PILOT_EPS", "0.0627450980")
RUN_ID = globals().get("RUN_ID", "controls_v1")
PROJECT = "/content/cdr"
LOCAL_RUNS = f"{PROJECT}/results/runs"
DRIVE_RUNS = f"{DRIVE_DIR}/results/runs"

CONTROL_DEFENSES = ("D0,jpeg95,jpeg90,jpeg75,jpeg50,jpeg30,jpeg10,adapter_only,"
                    "adapter_only_ll,resample_only,chroma_only,noise1,noise2")
PILOT_RUNS = {   # run id -> (defense in the attack loop, defenses to score)
    "pilot_adaptive_jpeg50": ("jpeg50", "D0,jpeg50,jpeg75,jpeg90"),
    "pilot_adaptive_dz_ll": ("dangerzone_ll", "D0,jpeg50"),
    "pilot_adaptive_dz": ("dangerzone", "D0,jpeg75"),
}
LIGHT = ("'--include=*/' '--include=*.json' '--include=*.jsonl' '--include=*.csv' "
         "'--include=*.yaml' '--include=*.png' '--exclude=*'")
WITH_NPY = ("'--include=*/' '--include=*.json' '--include=*.jsonl' '--include=*.csv' "
            "'--include=*.yaml' '--include=*.png' '--include=*.npy' '--exclude=*'")

ENV = dict(os.environ, HF_HOME="/content/hf_cache", MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")


def sh(cmd, cwd=None, check=False):
    print("+", cmd, flush=True)
    if DRY_RUN:
        return 0
    rc = subprocess.run(cmd, shell=True, cwd=cwd, env=ENV).returncode
    if check and rc:
        raise SystemExit(f"failed ({rc}): {cmd}")
    return rc


def setup():
    if os.path.exists(PROJECT + "/.git"):
        sh(f"git -C {PROJECT} fetch --all -q && git -C {PROJECT} checkout {BRANCH} -q && "
           f"git -C {PROJECT} pull --ff-only -q", check=True)
    else:
        sh(f"git clone -q --branch {BRANCH} {REPO_URL} {PROJECT}", check=True)
    sh(f"git -C {PROJECT} rev-parse HEAD")
    sh("pip -q install transformers accelerate bitsandbytes scipy matplotlib pyyaml pillow",
       check=True)
    sh("which pdftoppm >/dev/null || (apt-get -qq update && apt-get -qq install -y poppler-utils)")
    sh("python scripts/fetch_imagenette.py --out data/imagenet/val", cwd=PROJECT, check=True)
    sh("python scripts/build_dataset.py --config configs/experiment.yaml", cwd=PROJECT, check=True)
    sh("python scripts/test_llava.py", cwd=PROJECT)          # checkpoint 1


def pull(run_id):   # Drive -> local (small files only)
    sh(f"mkdir -p {LOCAL_RUNS}/{run_id} && rsync -a {LIGHT} {DRIVE_RUNS}/{run_id}/ "
       f"{LOCAL_RUNS}/{run_id}/")


def push(run_id):   # local -> Drive
    sh(f"mkdir -p {DRIVE_RUNS}/{run_id} && rsync -a {WITH_NPY} {LOCAL_RUNS}/{run_id}/ "
       f"{DRIVE_RUNS}/{run_id}/")


def stage_controls():
    pull(RUN_ID)
    sh(f"python scripts/run_experiment.py --run-id {RUN_ID} --runs-base {LOCAL_RUNS} "
       f"--attack-source {DRIVE_RUNS}/{BASE_RUN} --experiment-id {RUN_ID} "
       f"--defenses {CONTROL_DEFENSES} --max-hours {SESSION_HOURS}", cwd=PROJECT)
    push(RUN_ID)
    sh(f"python scripts/run_analysis.py --run {LOCAL_RUNS}/{RUN_ID}", cwd=PROJECT)
    push(RUN_ID)


def stage_dz_answers():
    pull(RUN_ID)
    for variant, d in (("jpeg", "dz"), ("lossless", "dz_ll")):
        sh(f"python scripts/dangerzone_finalize.py --run {LOCAL_RUNS}/{RUN_ID} "
           f"--variant {variant} --dz-dir {DRIVE_RUNS}/{BASE_RUN}/{d} "
           f"--attack-source {DRIVE_RUNS}/{BASE_RUN}", cwd=PROJECT)
    push(RUN_ID)
    sh(f"python scripts/run_analysis.py --run {LOCAL_RUNS}/{RUN_ID}", cwd=PROJECT)
    push(RUN_ID)


def stage_pilot():
    for run_id, (adaptive, defenses) in PILOT_RUNS.items():
        pull(run_id)
        sh(f"python scripts/run_experiment.py --run-id {run_id} --runs-base {LOCAL_RUNS} "
           f"--adaptive-defense {adaptive} --limit {PILOT_IMAGES} --epsilon {PILOT_EPS} "
           f"--defenses {defenses} --experiment-id {run_id} --max-hours {SESSION_HOURS}",
           cwd=PROJECT)
        push(run_id)


def stage_pilot_eval():
    for run_id, (adaptive, _) in PILOT_RUNS.items():
        pull(run_id)
        sh(f"python scripts/eval_defended_loss.py --adaptive-run {LOCAL_RUNS}/{run_id} "
           f"--oblivious-run {DRIVE_RUNS}/{BASE_RUN} --defense {adaptive} "
           f"--out {LOCAL_RUNS}/{run_id}/defended_loss_{adaptive}.json", cwd=PROJECT)
        push(run_id)


def stage_pilot_dz():
    for run_id, (adaptive, _) in PILOT_RUNS.items():
        if not adaptive.startswith("dangerzone"):
            continue
        variant = "lossless" if adaptive.endswith("_ll") else "jpeg"
        d = "dz_ll" if variant == "lossless" else "dz"
        pull(run_id)
        sh(f"python scripts/dangerzone_finalize.py --run {LOCAL_RUNS}/{run_id} "
           f"--variant {variant} --dz-dir {DRIVE_RUNS}/{run_id}/{d}", cwd=PROJECT)
        push(run_id)


def stage_analysis():
    pull(RUN_ID)
    sh(f"python scripts/run_analysis.py --run {LOCAL_RUNS}/{RUN_ID}", cwd=PROJECT)
    push(RUN_ID)


STAGES = {"controls": stage_controls, "dz_answers": stage_dz_answers, "pilot": stage_pilot,
          "pilot_eval": stage_pilot_eval, "pilot_dz": stage_pilot_dz,
          "analysis": stage_analysis}

if STAGE not in STAGES:
    raise SystemExit(f"unknown STAGE {STAGE!r}; choose from {sorted(STAGES)}")
setup()
STAGES[STAGE]()
print("STAGE DONE:", STAGE, json.dumps({"branch": BRANCH, "dry_run": DRY_RUN}))
