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
  conv          convergence study (docs/phase3_plan.md section 2): both adaptive attacks
                at 16/255 continued from the confirmatory 500-step arrays to 1000 and 2000
                cumulative steps on configs/conv_100.json, checkpoints every 100 steps.
                Run dirs conv_<atk>_e16_s<budget>.
  conv_eval     score every stored checkpoint through real JPEG 50 (GPU).
  conv_probe    two fresh random starts (seeds 1235, 1236) and the momentum+step-decay
                optimizer variant, 25 images, fresh 500-step JPEG-50 attacks.
                Run dirs conv_jpeg50_e16_{r1,r2,mom}.
  conv_dz       score the locally sanitized Dangerzone checkpoints (coarse grid
                1000/1500/2000; needs dz_ll/ckpt on Drive from the agent's local sanitize).
"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

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


def _conf_complete(rid):
    """True once the attack finished all 200 images on Drive (scoring waits for that)."""
    try:
        prog = json.load(open(f"{DRIVE_RUNS}/{rid}/progress.json"))
        return int(prog.get("adv_images", 0)) >= 200
    except (OSError, ValueError):
        return False


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
        if not _conf(rid) or not _conf_complete(rid):
            continue
        if os.path.exists(f"{DRIVE_RUNS}/{rid}/defended_loss_{adaptive}.json"):
            continue
        pull_npy(rid)
        sh(f"python scripts/eval_defended_loss.py --adaptive-run {LOCAL_RUNS}/{rid} "
           f"--oblivious-run {DRIVE_RUNS}/{BASE_RUN} --defense {adaptive} "
           f"--out {LOCAL_RUNS}/{rid}/defended_loss_{adaptive}.json", cwd=PROJECT)
        push(rid)


def stage_conf_dz():
    for rid, adaptive, _ in CONF_RUNS:
        if not adaptive.startswith("dangerzone") or not _conf(rid) or not _conf_complete(rid):
            continue
        variant = "lossless" if adaptive.endswith("_ll") else "jpeg"
        d = "dz_ll" if variant == "lossless" else "dz"
        try:
            scored = sum(1 for l in open(f"{DRIVE_RUNS}/{rid}/defense_results.jsonl")
                         if f'"defense": "{adaptive}"' in l)
        except OSError:
            scored = 0
        if scored >= 200:
            print(f"conf_dz: {rid} already scored through real Dangerzone, skipped", flush=True)
            continue
        if not os.path.isdir(f"{DRIVE_RUNS}/{rid}/{d}"):
            print(f"conf_dz: {rid} has no sanitized {d}/ on Drive yet, skipped", flush=True)
            continue
        pull(rid)
        sh(f"python scripts/dangerzone_finalize.py --run {LOCAL_RUNS}/{rid} "
           f"--variant {variant} --dz-dir {DRIVE_RUNS}/{rid}/{d}", cwd=PROJECT)
        push(rid)


# --- matched-budget oblivious control (docs/phase3_plan.md 1.3) ---------------------------
# The stored 200-step oblivious attacks (BASE_RUN) are continued for OBL_MORE_STEPS to reach the
# same 500 steps as the adaptive attacks. One attack per (image, epsilon) is scored through every
# defense. stage_cont_validate must have passed first (continuation equals an uninterrupted run).
OBL_MORE_STEPS = int(globals().get("OBL_MORE_STEPS", 300))
OBL_DEFENSES = "D0,jpeg90,jpeg75,jpeg50"
OBL_RUNS = [("obl500_e16", "0.0627450980"), ("obl500_e08", "0.0313725490")]
# Fallback fixed in phase3_plan.md 1.2 and used because the continuation check failed: fresh
# 500-step oblivious attacks from the same random start (seed 1234), no stored arrays involved.
OBL_SCRATCH_RUNS = [("obl500s_e16", "0.0627450980"), ("obl500s_e08", "0.0313725490")]
OBL_SCRATCH_STEPS = 500


def stage_cont_validate():
    """GPU check that continuing a stored attack equals an uninterrupted run (about 1 hour)."""
    pull_npy(BASE_RUN)
    sh(f"python scripts/validate_continuation.py --phase1 {LOCAL_RUNS}/{BASE_RUN} "
       f"--out {PROJECT}/results/continuation", cwd=PROJECT)
    sh(f"mkdir -p {DRIVE_DIR}/results/continuation && rsync -a {PROJECT}/results/continuation/ "
       f"{DRIVE_DIR}/results/continuation/")


def _continuation_equivalent():
    """Gate fixed in advance: the control may only continue stored arrays if the check passed."""
    try:
        return bool(json.load(open(f"{DRIVE_DIR}/results/continuation/continuation.json"))
                    .get("continuation_equivalent"))
    except (OSError, ValueError):
        return False


def stage_obl():
    if not DRY_RUN and not _continuation_equivalent():
        print("obl: continuation equivalence not established (run cont_validate; it must pass). "
              "No control attacks started.", flush=True)
        return
    for rid, eps in OBL_RUNS:
        if not _conf(rid):
            continue
        pull_npy(BASE_RUN)
        pull_npy(rid)
        sh(f"python scripts/run_experiment.py --run-id {rid} --runs-base {LOCAL_RUNS} "
           f"--epsilon {eps} --steps {OBL_MORE_STEPS} --init-from {LOCAL_RUNS}/{BASE_RUN} "
           f"--defenses {OBL_DEFENSES} --experiment-id {rid} --max-hours {SESSION_HOURS}",
           cwd=PROJECT)
        push(rid)


def stage_obl_scratch():
    """Fresh 500-step defense-unaware attacks, one per (image, epsilon), scored by every defense."""
    for rid, eps in OBL_SCRATCH_RUNS:
        if not _conf(rid):
            continue
        pull_npy(rid)
        sh(f"python scripts/run_experiment.py --run-id {rid} --runs-base {LOCAL_RUNS} "
           f"--epsilon {eps} --steps {OBL_SCRATCH_STEPS} "
           f"--defenses {OBL_DEFENSES} --experiment-id {rid} --max-hours {SESSION_HOURS}",
           cwd=PROJECT)
        push(rid)


def stage_obl_dz():
    for rid, _ in OBL_RUNS + OBL_SCRATCH_RUNS:
        if not _conf(rid) or not _conf_complete(rid):
            continue
        try:
            scored = sum(1 for l in open(f"{DRIVE_RUNS}/{rid}/defense_results.jsonl")
                         if '"defense": "dangerzone_ll"' in l)
        except OSError:
            scored = 0
        if scored >= 200:
            print(f"obl_dz: {rid} already scored through real Dangerzone, skipped", flush=True)
            continue
        if not os.path.isdir(f"{DRIVE_RUNS}/{rid}/dz_ll"):
            print(f"obl_dz: {rid} has no sanitized dz_ll/ on Drive yet, skipped", flush=True)
            continue
        pull(rid)
        sh(f"python scripts/dangerzone_finalize.py --run {LOCAL_RUNS}/{rid} "
           f"--variant lossless --dz-dir {DRIVE_RUNS}/{rid}/dz_ll", cwd=PROJECT)
        push(rid)


# --- convergence study (docs/phase3_plan.md section 2) ----------------------------------
# Both adaptive attacks at 16/255, continued from the stored confirmatory 500-step arrays
# to 1000 and 2000 cumulative steps on the fixed 100-image subset (configs/conv_100.json;
# rule in scripts/make_conv_manifests.py, decided before any convergence run). Every 100
# steps a float32 checkpoint is stored under ckpt/ with cumulative labels and scored
# through real JPEG 50 on the GPU (stage conv_eval). Real Dangerzone is scored at the
# coarse grid 1000/1500/2000 after a LOCAL sanitize (the agent runs
# scripts/dangerzone_sanitize.py --embed lossless --ckpt-steps 1000,1500,2000 and uploads
# dz_ll/), then stage conv_dz scores those images. conv_probe adds two fresh random
# starts and the momentum+step-decay optimizer variant on 25 images (configs/conv_25.json).
CONV_MANIFEST = "configs/conv_100.json"
CONV_PROBE_MANIFEST = "configs/conv_25.json"
CONV_EPS = "0.062745098"
CONV_STEPS_0 = 500                      # the stored confirmatory budget (both conf runs)
CONV_SCHEDULE = (1000, 2000)
CONV_CKPT_EVERY = 100
CONV_DEFENSES = "D0,jpeg90,jpeg75,jpeg50"
CONV_ATTACKS = (("jpeg50", "conf_jpeg50_e16"), ("dangerzone_ll", "conf_dz_ll_e16"))
CONV_PROBES = (    # (run id, PGD seed, momentum, extra flags), fresh 500-step JPEG-50 attacks
    ("conv_jpeg50_e16_r1", 1235, 0.0, ""),
    ("conv_jpeg50_e16_r2", 1236, 0.0, ""),
    ("conv_jpeg50_e16_mom", 1234, 0.9, "--step-decay"),
)


def _attack_done(rid, n=100):
    """True once the run's attack finished all n images on Drive."""
    try:
        prog = json.load(open(f"{DRIVE_RUNS}/{rid}/progress.json"))
        return int(prog.get("adv_images", 0)) >= n
    except (OSError, ValueError):
        return False


def _conv_rid(atk, budget):
    return f"conv_{atk}_e16_s{budget}"


def _ckpt_jobs_scored(rid):
    try:
        return len(list(Path(f"{LOCAL_RUNS}/{rid}/jobs/ckpt_score").glob("*.json")))
    except OSError:
        return 0


def stage_conv():
    for atk, base_rid in CONV_ATTACKS:
        prev_id, prev_cum = base_rid, CONV_STEPS_0
        for budget in CONV_SCHEDULE:
            rid = _conv_rid(atk, budget)
            if _attack_done(rid):
                print(f"conv: {rid} attack already complete, skipped", flush=True)
                prev_id, prev_cum = rid, budget
                continue
            pull_npy(prev_id)   # the arrays this stage continues from
            pull_npy(rid)       # resume needs the stored images, or finished jobs are redone
            sh(f"python scripts/run_experiment.py --run-id {rid} --runs-base {LOCAL_RUNS} "
               f"--manifest {CONV_MANIFEST} --adaptive-defense {atk} --epsilon {CONV_EPS} "
               f"--steps {budget - prev_cum} --init-from {LOCAL_RUNS}/{prev_id} "
               f"--checkpoint-every {CONV_CKPT_EVERY} --defenses {CONV_DEFENSES} "
               f"--experiment-id {rid} --max-hours {SESSION_HOURS}", cwd=PROJECT)
            push(rid)
            if not _attack_done(rid):
                print(f"conv: {rid} still incomplete; run the notebook again to resume", flush=True)
                return
            prev_id, prev_cum = rid, budget


def stage_conv_eval():
    """Score every stored checkpoint through real JPEG 50 (GPU)."""
    for atk, _ in CONV_ATTACKS:
        for budget in CONV_SCHEDULE:
            rid = _conv_rid(atk, budget)
            if not _attack_done(rid):
                continue
            n_expect = 100 * ((budget - CONV_STEPS_0) // CONV_CKPT_EVERY)
            pull_npy(rid)
            if _ckpt_jobs_scored(rid) >= n_expect:
                print(f"conv_eval: {rid} checkpoints already scored, skipped", flush=True)
                continue
            sh(f"python scripts/score_checkpoints.py --run {LOCAL_RUNS}/{rid} "
               f"--defense jpeg50 --max-hours {SESSION_HOURS}", cwd=PROJECT)
            push(rid)


def stage_conv_probe():
    for rid, seed, momentum, extra in CONV_PROBES:
        if _attack_done(rid, n=25):
            print(f"conv_probe: {rid} already complete, skipped", flush=True)
            continue
        pull_npy(rid)
        sh(f"python scripts/run_experiment.py --run-id {rid} --runs-base {LOCAL_RUNS} "
           f"--manifest {CONV_PROBE_MANIFEST} --adaptive-defense jpeg50 "
           f"--epsilon {CONV_EPS} --steps {CONV_STEPS_0} --attack-seed {seed} "
           f"--momentum {momentum} {extra} --defenses {CONV_DEFENSES} "
           f"--experiment-id {rid} --max-hours {SESSION_HOURS}", cwd=PROJECT)
        push(rid)
        if not _attack_done(rid, n=25):
            print(f"conv_probe: {rid} still incomplete; run the notebook again to resume",
                  flush=True)
            return


def stage_conv_dz():
    """Score the locally sanitized Dangerzone checkpoints (coarse grid 1000/1500/2000).

    Each conv run only holds the checkpoints of its own stage: s1000 has step 1000,
    s2000 has 1500 and 2000. The expected count comes from that range, and the upload
    must be complete before scoring starts.
    """
    prev_of = {CONV_SCHEDULE[0]: CONV_STEPS_0, CONV_SCHEDULE[1]: CONV_SCHEDULE[0]}
    for atk, _ in CONV_ATTACKS:
        for budget in CONV_SCHEDULE:
            rid = _conv_rid(atk, budget)
            if not _attack_done(rid):
                continue
            grid = [s for s in (1000, 1500, 2000) if prev_of[budget] < s <= budget]
            n_expect = 100 * len(grid)
            n_pngs = len(list(Path(f"{DRIVE_RUNS}/{rid}/dz_ll/ckpt").glob("*.png"))) \
                if os.path.isdir(f"{DRIVE_RUNS}/{rid}/dz_ll/ckpt") else 0
            if n_pngs < n_expect:
                print(f"conv_dz: {rid} has {n_pngs} sanitized ckpt pngs on Drive, "
                      f"expects {n_expect}; skipped (agent runs the local sanitize first)",
                      flush=True)
                continue
            pull_npy(rid)
            scored = sum(1 for jp in Path(f"{LOCAL_RUNS}/{rid}/jobs/ckpt_score").glob("*.json")
                         if "dangerzone_ll" in jp.name)
            if scored >= n_expect:
                print(f"conv_dz: {rid} Dangerzone checkpoints already scored, skipped", flush=True)
                continue
            sh(f"python scripts/score_checkpoints.py --run {LOCAL_RUNS}/{rid} "
               f"--dz-dir {DRIVE_RUNS}/{rid}/dz_ll --max-hours {SESSION_HOURS}", cwd=PROJECT)
            push(rid)


def stage_analysis():
    pull(RUN_ID)
    sh(f"python scripts/run_analysis.py --run {LOCAL_RUNS}/{RUN_ID}", cwd=PROJECT)
    push(RUN_ID)


STAGES = {"controls": stage_controls, "dz_answers": stage_dz_answers, "pilot": stage_pilot,
          "pilot_eval": stage_pilot_eval, "pilot_dz": stage_pilot_dz,
          "analysis": stage_analysis, "jpeg_ext": stage_jpeg_ext, "screen": stage_screen,
          "screen_eval": stage_screen_eval, "screen_dz": stage_screen_dz,
          "conf": stage_conf, "conf_eval": stage_conf_eval, "conf_dz": stage_conf_dz,
          "cont_validate": stage_cont_validate, "obl": stage_obl, "obl_scratch": stage_obl_scratch,
          "obl_dz": stage_obl_dz, "conv": stage_conv, "conv_eval": stage_conv_eval,
          "conv_probe": stage_conv_probe, "conv_dz": stage_conv_dz}

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
