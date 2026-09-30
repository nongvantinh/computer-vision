# Colab account pool (`cdrctl`)

Turns several Google/Colab accounts into interchangeable compute workers. You do OAuth
once per account; `cdrctl` stores each credential and rotates through accounts when one
runs out of compute. The experiment never knows which account ran it: state lives in the
run directory as idempotent jobs, so any account continues the same `RUN_ID`.

Tool: `scripts/cdrctl.py`. Pool logic is unit-tested (`tests/test_cdrctl.py`).

## Credential model (discovered, not assumed)

The installed colab CLI keeps one active OAuth token at
`~/.config/colab-cli/token.json` and re-runs login when it is absent. `cdrctl` stashes
one token per account under `~/.cdr-colab/accounts/<name>/token.json` (mode 600) and
activates an account by copying its token into the active location. Pool state
(`~/.cdr-colab/pool.json`) tracks each account's state and last-seen balance.

## One-time: register accounts (the only interactive step)

```bash
python scripts/cdrctl.py accounts add acctA   # completes the Google login, stashes token
python scripts/cdrctl.py accounts add acctB
python scripts/cdrctl.py accounts add acctC
python scripts/cdrctl.py accounts list        # ACCOUNT / STATE / BALANCE
```

Manual switching (also answers "how do I switch accounts"):

```bash
python scripts/cdrctl.py accounts use acctB    # make B active
python scripts/cdrctl.py accounts who          # which account + balance now
python scripts/cdrctl.py status                # balance/state across all accounts
```

## Validate rotation before the real run (dummy workload)

```bash
python scripts/cdrctl.py run --dummy --run-id smoke
```

This exercises activate -> provision session -> trivial VM job -> stop, on one account,
so the account/session wiring is checked without spending real experiment compute.

## Run the experiment across the pool

```bash
python scripts/cdrctl.py run --run-id full_200 --mode full --gpu T4 --max-hours 3
```

The loop, per round: pick the next usable account (READY, then EXHAUSTED whose cooldown
elapsed), activate it, check balance, provision a session (retrying transient 503s),
mount Drive, run one time-budgeted resumable batch of the experiment (`colab_bootstrap.py`),
stop the session, read progress, and either finish (queue empty) or rotate. Because the
batch is resumable, a killed round loses at most one job.

Account state machine: `READY -> RUNNING -> {READY | EXHAUSTED(cooldown) | AUTH_REQUIRED}`.
Failures are classified (`classify_failure`): transient (retry same account), quota
(rotate + cooldown), or auth-expired (mark AUTH_REQUIRED, needs `accounts add` again).
When every account is EXHAUSTED/AUTH_REQUIRED the loop stops and reports how many jobs
remain; add another account or wait for quota, then re-run — it resumes.

## Honest caveats

- This sits on the unofficial Colab CLI, whose `assign` endpoint returns 503 under load
  and whose `drivemount` may need one interactive approval per session. `cdrctl` retries
  and classifies, but it is not bulletproof. The **browser notebook remains the reliable
  fallback** and writes the same `results/runs/<RUN_ID>` on Drive, so you can mix paths.
- `cdrctl` automates infrastructure only. It never changes the model, attack, defenses,
  metric, or seeds; those stay in `configs/` and the run manifest.
- Multi-account continuation needs the shared Drive folder (Path A in `colab-resume.md`)
  so every account's Colab writes to one run directory.
