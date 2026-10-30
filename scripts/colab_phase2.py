"""Phase 2 stages, run INSIDE a Colab VM (exec this file in the kernel).

Parameters are read from the kernel globals (set them in the cell before exec'ing, or
leave the defaults): STAGE, BRANCH, DRIVE_DIR, SESSION_HOURS, DRY_RUN, plus the stage
parameters below. Run directories are written on the VM's local disk and copied to
Drive with rsync (many small job files are slow to write straight to Drive); every
job is idempotent, so a reclaimed VM resumes where it stopped.

STAGE may be one name or a comma-separated list; setup() (clone, deps, dataset,
checkpoint 1) runs once and the stages then run in the order given.

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
  jpeg_ext      staged continuation of the adaptive JPEG 50 pilot (same 10 images, 16/255):
                cumulative 500, 1000, 2000 steps, each continuing from the previous
                stage's stored image; stops by the rule in docs/phase2_screen_plan.md.
                Run dirs pilot_jpeg50_s<budget>.
  screen        adaptive attacks on the 50-image screen (configs/screen_50.json), in
                priority order, SCREEN_STEPS steps each. Run dirs screen_<atk>_e<eps>.
  screen_eval   defended-loss evaluation (gate G3) for the screen runs.
  screen_dz     model answers on the locally sanitized Dangerzone images of the screen
                runs (dz/ or dz_ll/ uploaded into each screen run dir).
"""
import json
import os
import subprocess
import threading
import time

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
EXT_SCHEDULE = (200, 500, 1000, 2000)     # cumulative steps; 200 is the existing pilot
EXT_RUN = "pilot_jpeg50_s{}"
SCREEN_MANIFEST = "configs/screen_50.json"
SCREEN_STEPS = int(globals().get("SCREEN_STEPS", 200))
SCREEN_ONLY = [x for x in str(globals().get("SCREEN_ONLY", "")).split(",") if x]
SCREEN_DEFENSES = "D0,jpeg90,jpeg75,jpeg50"
SCREEN_RUNS = [   # priority order: 16/255 first, Dangerzone-lossless first
    ("screen_dz_ll_e16", "dangerzone_ll", "0.0627450980"),
    ("screen_jpeg50_e16", "jpeg50", "0.0627450980"),
    ("screen_dz_e16", "dangerzone", "0.0627450980"),
    ("screen_dz_ll_e08", "dangerzone_ll", "0.0313725490"),
    ("screen_jpeg50_e08", "jpeg50", "0.0313725490"),
    ("screen_dz_e08", "dangerzone", "0.0313725490"),
]
# Confirmatory N = 200 runs (docs/phase2_preregistration.md): the registered family attacks
# JPEG 50 and lossless Dangerzone at 8/255 and 16/255 on the full 200-image manifest, at the
# screen step budget. Priority order: 16/255 first.
CONF_STEPS = int(globals().get("CONF_STEPS", 500))
CONF_ONLY = [x for x in str(globals().get("CONF_ONLY", "")).split(",") if x]
CONF_DEFENSES = "D0,jpeg90,jpeg75,jpeg50"
CONF_RUNS = [
    ("conf_dz_ll_e16", "dangerzone_ll", "0.0627450980"),
    ("conf_jpeg50_e16", "jpeg50", "0.0627450980"),
    ("conf_dz_ll_e08", "dangerzone_ll", "0.0313725490"),
    ("conf_jpeg50_e08", "jpeg50", "0.0313725490"),
]
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


SYNC_SECONDS = int(globals().get("SYNC_SECONDS", 120))


def _sync_loop():
    """Copy every local run directory to Drive on a timer and log what Drive now holds.

    Replaces the notebook's shell autosync, which never ran (Colab's /bin/sh is dash and
    rejects `exec -a`), so a reclaimed VM lost a whole 10 hour screen run."""
    while True:
        time.sleep(SYNC_SECONDS)
        try:
            if not os.path.isdir(LOCAL_RUNS):
                continue
            for rid in sorted(os.listdir(LOCAL_RUNS)):
                if not os.path.isdir(f"{LOCAL_RUNS}/{rid}/jobs"):
                    continue
                os.makedirs(f"{DRIVE_RUNS}/{rid}", exist_ok=True)
                subprocess.run(f"rsync -a {WITH_NPY} {LOCAL_RUNS}/{rid}/ {DRIVE_RUNS}/{rid}/",
                               shell=True, check=False, capture_output=True)
                n_loc = len([f for f in os.listdir(f"{LOCAL_RUNS}/{rid}/adv") if f.endswith(".npy")]) \
                    if os.path.isdir(f"{LOCAL_RUNS}/{rid}/adv") else 0
                n_drv = len([f for f in os.listdir(f"{DRIVE_RUNS}/{rid}/adv") if f.endswith(".npy")]) \
                    if os.path.isdir(f"{DRIVE_RUNS}/{rid}/adv") else 0
                print(f"SYNC {rid}: adv local {n_loc}, on Drive {n_drv}", flush=True)
        except Exception as e:   # never let the sync thread kill the stage
            print("SYNC error:", e, flush=True)


def start_sync():
    threading.Thread(target=_sync_loop, daemon=True).start()


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


def pull_npy(run_id):   # Drive -> local, including the stored adversarial images
    sh(f"mkdir -p {LOCAL_RUNS}/{run_id} && rsync -a {WITH_NPY} {DRIVE_RUNS}/{run_id}/ "
       f"{LOCAL_RUNS}/{run_id}/")


def stage_jpeg_ext():
    prev_id, prev_cum = "pilot_adaptive_jpeg50", EXT_SCHEDULE[0]
    for budget in EXT_SCHEDULE[1:]:
        rid = EXT_RUN.format(budget)
        pull_npy(prev_id)
        pull_npy(rid)   # resume needs the stored images, or finished attack jobs are redone
        sh(f"python scripts/run_experiment.py --run-id {rid} --runs-base {LOCAL_RUNS} "
           f"--adaptive-defense jpeg50 --limit {PILOT_IMAGES} --epsilon {PILOT_EPS} "
           f"--steps {budget - prev_cum} --init-from {LOCAL_RUNS}/{prev_id} "
           f"--defenses D0,jpeg50,jpeg75,jpeg90 --experiment-id {rid} "
           f"--max-hours {SESSION_HOURS}", cwd=PROJECT)
        sh(f"python scripts/eval_defended_loss.py --adaptive-run {LOCAL_RUNS}/{rid} "
           f"--oblivious-run {DRIVE_RUNS}/{BASE_RUN} --defense jpeg50 "
           f"--out {LOCAL_RUNS}/{rid}/defended_loss_jpeg50.json", cwd=PROJECT)
        push(rid)
        rc = sh(f"python scripts/stage_decision.py --run {LOCAL_RUNS}/{rid} --defense jpeg50 "
                f"--eps {PILOT_EPS} --expect {PILOT_IMAGES} "
                f"--previous-loss {LOCAL_RUNS}/{prev_id}/defended_loss_jpeg50.json "
                f"--out {LOCAL_RUNS}/{rid}/stage_decision.json", cwd=PROJECT)
        push(rid)
        if rc != 0:
            print(f"STAGE INCOMPLETE ({rc}): {rid}; run the stage again to resume", flush=True)
            return
        d = json.load(open(f"{LOCAL_RUNS}/{rid}/stage_decision.json"))
        print("STAGE DECISION:", rid, d["action"], d["reason"], flush=True)
        if d["action"] == "stop":
            return
        prev_id, prev_cum = rid, budget


def stage_screen():
    for rid, adaptive, eps in SCREEN_RUNS:
        if SCREEN_ONLY and rid not in SCREEN_ONLY:
            continue
        pull_npy(rid)
        sh(f"python scripts/run_experiment.py --run-id {rid} --runs-base {LOCAL_RUNS} "
           f"--adaptive-defense {adaptive} --manifest {SCREEN_MANIFEST} --epsilon {eps} "
           f"--steps {SCREEN_STEPS} --defenses {SCREEN_DEFENSES} --experiment-id {rid} "
           f"--max-hours {SESSION_HOURS}", cwd=PROJECT)
        push(rid)


def stage_screen_eval():
    for rid, adaptive, _ in SCREEN_RUNS:
        if SCREEN_ONLY and rid not in SCREEN_ONLY:
            continue
        pull_npy(rid)   # the defended loss reads the stored images
        sh(f"python scripts/eval_defended_loss.py --adaptive-run {LOCAL_RUNS}/{rid} "
           f"--oblivious-run {DRIVE_RUNS}/{BASE_RUN} --defense {adaptive} "
           f"--manifest {SCREEN_MANIFEST} "
           f"--out {LOCAL_RUNS}/{rid}/defended_loss_{adaptive}.json", cwd=PROJECT)
        push(rid)


def stage_screen_dz():
    for rid, adaptive, _ in SCREEN_RUNS:
        if not adaptive.startswith("dangerzone") or (SCREEN_ONLY and rid not in SCREEN_ONLY):
            continue
        variant = "lossless" if adaptive.endswith("_ll") else "jpeg"
        d = "dz_ll" if variant == "lossless" else "dz"
        pull(rid)
        sh(f"python scripts/dangerzone_finalize.py --run {LOCAL_RUNS}/{rid} "
           f"--variant {variant} --dz-dir {DRIVE_RUNS}/{rid}/{d}", cwd=PROJECT)
        push(rid)


def _conf(rid):
    return not CONF_ONLY or rid in CONF_ONLY


def stage_conf():
    for rid, adaptive, eps in CONF_RUNS:
        if not _conf(rid):
            continue
        pull_npy(rid)
        sh(f"python scripts/run_experiment.py --run-id {rid} --runs-base {LOCAL_RUNS} "
           f"--adaptive-defense {adaptive} --epsilon {eps} "
           f"--steps {CONF_STEPS} --defenses {CONF_DEFENSES} --experiment-id {rid} "
           f"--max-hours {SESSION_HOURS}", cwd=PROJECT)
        push(rid)


def stage_conf_eval():
    for rid, adaptive, _ in CONF_RUNS:
        if not _conf(rid):
            continue
        pull_npy(rid)
        sh(f"python scripts/eval_defended_loss.py --adaptive-run {LOCAL_RUNS}/{rid} "
           f"--oblivious-run {DRIVE_RUNS}/{BASE_RUN} --defense {adaptive} "
           f"--out {LOCAL_RUNS}/{rid}/defended_loss_{adaptive}.json", cwd=PROJECT)
        push(rid)


def stage_conf_dz():
    for rid, adaptive, _ in CONF_RUNS:
        if not adaptive.startswith("dangerzone") or not _conf(rid):
            continue
        variant = "lossless" if adaptive.endswith("_ll") else "jpeg"
        d = "dz_ll" if variant == "lossless" else "dz"
        pull(rid)
        sh(f"python scripts/dangerzone_finalize.py --run {LOCAL_RUNS}/{rid} "
           f"--variant {variant} --dz-dir {DRIVE_RUNS}/{rid}/{d}", cwd=PROJECT)
        push(rid)


def stage_analysis():
    pull(RUN_ID)
    sh(f"python scripts/run_analysis.py --run {LOCAL_RUNS}/{RUN_ID}", cwd=PROJECT)
    push(RUN_ID)


STAGES = {"controls": stage_controls, "dz_answers": stage_dz_answers, "pilot": stage_pilot,
          "pilot_eval": stage_pilot_eval, "pilot_dz": stage_pilot_dz,
          "analysis": stage_analysis, "jpeg_ext": stage_jpeg_ext, "screen": stage_screen,
          "screen_eval": stage_screen_eval, "screen_dz": stage_screen_dz,
          "conf": stage_conf, "conf_eval": stage_conf_eval, "conf_dz": stage_conf_dz}

todo = [s.strip() for s in STAGE.split(",") if s.strip()]
bad = [s for s in todo if s not in STAGES]
if bad:
    raise SystemExit(f"unknown STAGE {bad}; choose from {sorted(STAGES)}")
setup()
start_sync()
for name in todo:
    print(f"=== STAGE START: {name}", flush=True)
    STAGES[name]()
    print("STAGE DONE:", name, json.dumps({"branch": BRANCH, "dry_run": DRY_RUN}), flush=True)
