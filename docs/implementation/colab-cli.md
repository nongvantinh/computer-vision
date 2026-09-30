# Running on Colab from the terminal (Colab CLI)

An alternative to the browser notebook (`notebooks/colab_cdr_vlm.ipynb`): drive the
run from this machine with Google's [Colab CLI](https://github.com/googlecolab/google-colab-cli).
It provisions a Colab GPU, runs the experiment headlessly, and streams the log
back. State still lives on Google Drive, so the same resume rules apply
(`docs/implementation/colab-resume.md`): a run stopped by a usage limit continues
on the next invocation, here or on another account.

Files: `scripts/colab_run.sh` (local driver) and `scripts/colab_bootstrap.py`
(runs inside the VM via `colab exec`).

> For a free account and switching between accounts when compute runs out, prefer
> the browser notebook (`colab_cdr_vlm.ipynb`): it mounts Drive with an in-browser
> click, whereas the CLI's `colab drivemount` needs an interactive auth the headless
> shell cannot answer. Use the CLI on a paid account or where Drive mount is scripted.

## One-time setup (you)
The CLI needs an interactive Google login, which only you can do:
```bash
uv tool install google-colab-cli     # already installed in this project
colab new --gpu T4                   # first use opens the browser login
colab usage                          # this account's compute-unit balance
```
Nothing to upload: the dataset is fetched automatically each session.

## Run
```bash
RUN_ID=pilot_12 MODE=pilot GPU=T4 HOURS=3 scripts/colab_run.sh
```
It provisions the GPU, mounts Drive, clones this repo on the VM, installs the light
deps on top of Colab's preinstalled torch plus Podman + Dangerzone, fetches the
dataset to ephemeral disk, keeps `HF_HOME` on ephemeral disk (the ~14 GB model
re-downloads each session; only small state goes to Drive), and runs build_dataset,
checkpoint 1, checkpoint 3, the experiment (`--run-id`/`--runs-base` on Drive, time
budget), and analysis. It downloads a light bundle to
`results/from_colab/<run_id>_light.zip` and stops the VM.

Env vars: `RUN_ID`, `MODE` (pilot|full), `GPU` (A100/L4/T4/H100/G4), `HOURS`,
`SESSION`, `HIGHMEM=1`, `KEEP=1` (leave the VM up), `REPO_URL`, `BRANCH`, `DRIVE_DIR`.

## Continuing across usage limits and accounts
Re-run `scripts/colab_run.sh` with the same `RUN_ID` and finished jobs are skipped,
because the run directory lives at `Drive/results/runs/<run_id>` and each job writes
its own file. When an account's compute units run low (`colab usage`), switch to
another Google account, run the one-time `colab new` login there, attach the same
Drive folder, and re-run. The Drive state and job idempotency carry the progress over.

## How this relates to the notebook
The CLI path and the notebook are two front-ends to the same resumable
experiment. Use the CLI for headless, scripted runs from this machine; use the
notebook for a browser session you watch. Both write to the same Drive store, so
you can even alternate between them.

## What I can and cannot do
With the CLI authenticated, runs can be driven from this terminal via Bash. The
login itself, the choice of GPU tier, and the compute-unit spend are yours: the
OAuth flow is interactive and per account.
