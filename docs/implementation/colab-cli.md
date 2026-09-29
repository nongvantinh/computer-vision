# Running on Colab from the terminal (Colab CLI)

An alternative to the browser notebook (`notebooks/colab_cdr_vlm.ipynb`): drive the
run from this machine with Google's [Colab CLI](https://github.com/googlecolab/google-colab-cli).
It provisions a Colab GPU, runs the experiment headlessly, and streams the log
back. State still lives on Google Drive, so the same resume rules apply
(`docs/implementation/colab-resume.md`): a run stopped by a usage limit continues
on the next invocation, here or on another account.

Files: `scripts/colab_run.sh` (local driver) and `scripts/colab_bootstrap.py`
(runs inside the VM via `colab exec`).

## One-time setup (you)
The CLI needs an interactive Google login, which only you can do:
```bash
uv tool install google-colab-cli     # already installed in this project
colab new --gpu T4                   # first use opens the browser login
colab usage                          # this account's compute-unit balance
```
Also upload the 20-class ImageNet val subset once to
`Drive/computer-vision/imagenet/val/<wnid>/*.JPEG` (shared across accounts).

## Run
```bash
GPU=A100 HOURS=3 scripts/colab_run.sh
```
It provisions the GPU, mounts Drive, clones this repo on the VM, installs the
light deps on top of Colab's preinstalled torch, points `HF_HOME` and the
experiment `--out` at Drive, then runs build_dataset, checkpoint 1, the
experiment (with the time budget), and analysis. It downloads a light results
bundle to `results/from_colab/mvp_light.zip` and stops the VM.

Env vars: `GPU` (A100/L4/T4/H100/G4), `HOURS`, `SESSION`, `HIGHMEM=1`, `KEEP=1`
(leave the VM up), `REPO_URL`, `BRANCH`, `DRIVE_DIR`.

## Continuing across usage limits and accounts
Re-run `scripts/colab_run.sh` and finished jobs are skipped, because `--out`
points at `Drive/results/mvp` and each job writes its own file. When an account's
compute units run low (`colab usage`), switch to another Google account, run the
one-time `colab new` login there, attach the same Drive folder, and re-run. The
Drive state and job idempotency carry the progress over.

## How this relates to the notebook
The CLI path and the notebook are two front-ends to the same resumable
experiment. Use the CLI for headless, scripted runs from this machine; use the
notebook for a browser session you watch. Both write to the same Drive store, so
you can even alternate between them.

## What I can and cannot do
With the CLI authenticated, runs can be driven from this terminal via Bash. The
login itself, the choice of GPU tier, and the compute-unit spend are yours: the
OAuth flow is interactive and per account.
