# Runbook: producing results end to end

A followable procedure for a full run. Compute splits across three places because no
single available host can do everything: the **attack** needs a GPU (Colab), the
**Dangerzone sanitize** needs rootless Podman (this workstation), and the **model
answers** need the same 4-bit LLaVA as the rest of the run (Colab again). The
attack/defense split (adversarial images persist as `adv/*.npy`) is what lets these be
separate. Steps marked **[interactive]** need you (an auth token or a password); the
rest are scripted.

Set once: `RUN=pilot_12` (or `full_200`), `DRIVE_DIR=/content/drive/MyDrive/computer-vision`.

## Phase A — attack + JPEG on Colab (GPU)

Produces the run directory on Drive with the adversarial images and the D0 + JPEG
results. Two front-ends, pick one.

- **Notebook:** open `notebooks/colab_cdr_vlm.ipynb` from GitHub, set `RUN_ID`/`RUN_MODE`
  in CELL 1, Runtime > Run all. **[interactive]** approve the Drive mount popup.
- **CLI:** `colab new -s cdr --gpu T4` **[interactive]**, then
  `RUN_ID=$RUN MODE=pilot scripts/colab_run.sh`.

Result on Drive: `results/runs/$RUN/` with `adv/*.npy`, `examples/*_clean.png`,
`jobs/`, `*_results.jsonl`, `statistics.json`. Dangerzone (D5) is intentionally skipped
here (it cannot run on Colab).

## Phase B — Dangerzone sanitize on this workstation (no GPU)

One-time install **[interactive]** (needs sudo; rootless Podman works here because the
kernel has user namespaces and `/etc/subuid` set):

```bash
sudo apt-get update
sudo apt-get install -y uidmap podman poppler-utils ca-certificates curl gnupg
sudo install -dm755 /etc/apt/keyrings
sudo rm -f /etc/apt/keyrings/fpf-apt-tools-archive-keyring.gpg
sudo gpg --keyserver hkps://keys.openpgp.org --no-default-keyring --no-permission-warning \
  --homedir "$(mktemp -d)" \
  --keyring gnupg-ring:/etc/apt/keyrings/fpf-apt-tools-archive-keyring.gpg \
  --recv-keys DE28AB241FA48260FAC9B8BAA7C9B38522604281
sudo chmod +r /etc/apt/keyrings/fpf-apt-tools-archive-keyring.gpg
. /etc/os-release
echo "deb [signed-by=/etc/apt/keyrings/fpf-apt-tools-archive-keyring.gpg] https://packages.freedom.press/apt-tools-prod ${VERSION_CODENAME} main" | sudo tee /etc/apt/sources.list.d/fpf-apt-tools.list
sudo apt-get update && sudo apt-get install -y dangerzone
dangerzone-image upgrade        # rootless, as your user; pulls ~1.6 GB, cosign-verified
dangerzone-cli --version        # expect 0.11.0
```

Then, per run:

1. Download `Drive/computer-vision/results/runs/$RUN` to `results/runs/$RUN` in this repo
   (at least `adv/`, `examples/`, `attack_results.jsonl`, `dataset_manifest.json`).
2. Sanitize (no GPU): `.venv-exp/bin/python scripts/dangerzone_sanitize.py --run results/runs/$RUN`
   → writes `results/runs/$RUN/dz/` (sanitized PNGs + `sidecar.jsonl`), ~30-40 s/image.
3. Upload `results/runs/$RUN/dz/` back to Drive at `.../results/runs/$RUN/dz/`.

## Phase C — D5 model answers on Colab (GPU)

Generates the model answer on each sanitized image with the same 4-bit LLaVA, so
preservation/restoration stay comparable, then folds D5 into `statistics.json`.

```bash
colab new -s cdr --gpu T4     # [interactive] auth token
colab drivemount -s cdr       # [interactive] auth token; mounts Drive on the VM
RUN=$RUN scripts/colab_dz.sh   # scripted: finalize + analysis + download statistics.json
```

`colab_dz.sh` runs `colab_dz_bootstrap.py` on the VM (updates the repo, installs deps,
runs `dangerzone_finalize.py` + `run_analysis.py` on the Drive run dir; the model loads
once, ~10-15 min) and downloads `statistics.json` to `results/from_colab/${RUN}_statistics.json`.

Read the result: `python scripts/make_figures.py --run results/runs/$RUN` (if you pull
the run dir back) or open `statistics.json`. It now has D0, jpeg90/75/50, and dangerzone.

## Scaling

- Full 200 at eps 8/255: Phase A with `RUN=full_200 MODE=full` (the notebook/CLI already
  exclude D5). Resume across accounts by re-running with the same `RUN` (see
  `colab-resume.md`). Then Phases B and C on `full_200`.
- Add eps 16/255: pass `--epsilon 0.0627450980` (repeat `--epsilon` for several) to the
  run. Cost scales with (images x epsilons); see `colab-t4-attack-throughput` memory.
- Reduce N before changing the attack if compute is tight: a clean N=50 beats a broken
  N=200 (report attempted/completed/failed, which `summary.json` already tracks).

## Where things land

- Durable run: `Drive/computer-vision/results/runs/$RUN/` (jobs, adv, dz, tables, plots).
- Pulled stats: `results/from_colab/${RUN}_statistics.json`.
- Interpretation and tables: `docs/implementation/results.md` (measured numbers only).
