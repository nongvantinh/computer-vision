#!/usr/bin/env python
"""cdrctl - Colab account pool + experiment runner.

You authenticate each Google/Colab account once; cdrctl stores its credential and
rotates through accounts automatically, so account switching is an infrastructure
concern, not an experiment one. The experiment state lives in the run directory
(idempotent jobs), so any account can continue it.

Credential model (discovered from the installed colab CLI, not assumed): the CLI keeps
one active OAuth token at ~/.config/colab-cli/token.json and re-runs login if it is
absent. cdrctl stashes one token per account under ~/.cdr-colab/accounts/<name>/ and
"activates" an account by copying its token into that active location.

Commands:
    cdrctl accounts add <name>      # log in interactively; stash the token
    cdrctl accounts list            # show pool state
    cdrctl accounts use <name>      # make <name> the active colab account
    cdrctl accounts who             # which account is active now (+ balance)
    cdrctl accounts rm <name>
    cdrctl status                   # per-account balance/state
    cdrctl run --run-id ID [...]    # rotate accounts, run batches until complete
    cdrctl run --run-id ID --dummy  # rotation dry-run (trivial VM job, no experiment)

Human intervention is needed only for `accounts add` (OAuth) and when every account is
exhausted or a token expires (state AUTH_REQUIRED).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CDR_HOME = Path(os.environ.get("CDR_HOME", Path.home() / ".cdr-colab"))
COLAB_TOKEN = Path(os.environ.get("COLAB_TOKEN_PATH",
                                  Path.home() / ".config" / "colab-cli" / "token.json"))
POOL = CDR_HOME / "pool.json"

READY, EXHAUSTED, AUTH_REQUIRED = "READY", "EXHAUSTED", "AUTH_REQUIRED"


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def parse_balance(usage_output: str) -> float | None:
    """Pull the compute-unit balance out of `colab usage` text."""
    m = re.search(r"balance:\s*([0-9.]+)", usage_output, re.I)
    return float(m.group(1)) if m else None


def classify_failure(output: str) -> str:
    """Map a failed colab command's output to an action.

    rotate  - this account cannot serve compute now (quota/capacity)
    reauth  - the token is invalid/expired; the human must log in again
    retry   - transient; try the same account again shortly
    fail    - unknown; surface it
    """
    o = (output or "").lower()
    if "service unavailable" in o or "503" in o or "resource" in o and "exhaust" in o:
        return "retry"           # Colab backend / capacity blip
    if "toomanyassignments" in o or "412" in o:
        return "retry"           # a session already exists; reuse/wait
    if any(s in o for s in ("quota", "compute units", "no compute", "exhaust")):
        return "rotate"
    if any(s in o for s in ("unauthorized", "401", "invalid_grant", "credential",
                            "login", "reauth", "token")):
        return "reauth"
    return "fail"


def load_pool() -> dict:
    data = _read_json(POOL) or {}
    data.setdefault("accounts", {})
    data.setdefault("active", None)
    return data


def save_pool(pool: dict) -> None:
    CDR_HOME.mkdir(parents=True, exist_ok=True)
    _write_json(POOL, pool)


def set_state(pool: dict, name: str, **fields) -> dict:
    acct = pool["accounts"].setdefault(name, {"state": READY})
    acct.update(fields)
    return pool


def next_ready(pool: dict) -> str | None:
    """The next usable account: READY preferred, then EXHAUSTED whose cooldown passed."""
    now = time.time()
    for name, a in pool["accounts"].items():
        if a.get("state") == READY:
            return name
    for name, a in pool["accounts"].items():
        if a.get("state") == EXHAUSTED and now >= a.get("cooldown_until", 0):
            return name
    return None


# --------------------------------------------------------------------------- #
# small I/O
# --------------------------------------------------------------------------- #
def _read_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_json(p: Path, obj) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(p)


def _acct_token(name: str) -> Path:
    return CDR_HOME / "accounts" / name / "token.json"


def sh(cmd: list[str], timeout: float = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        return 124, f"timeout: {e}"
    except FileNotFoundError as e:
        return 127, f"not found: {e}"


def sh_interactive(cmd: list[str], timeout: float = 900) -> int:
    """Run a command inheriting the terminal, so an OAuth prompt (URL + code entry)
    works. Never capture output here or the login flow deadlocks."""
    try:
        return subprocess.run(cmd, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 124
    except FileNotFoundError:
        return 127


# --------------------------------------------------------------------------- #
# account manager
# --------------------------------------------------------------------------- #
def cmd_accounts_add(args) -> int:
    name = args.name
    dst = _acct_token(name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cdrctl] Log in for account '{name}'. Removing any active token so the "
          f"colab login flow starts...")
    if COLAB_TOKEN.exists():
        COLAB_TOKEN.unlink()
    # Interactive: the CLI prints an auth URL and reads the code from the terminal, so
    # this must inherit stdin/stdout (do NOT capture, or it deadlocks).
    print("[cdrctl] Complete the Google login below (open the URL, paste the code).")
    print("[cdrctl] Log in as the Google account you want to register as '%s'.\n" % name)
    rc = sh_interactive(["colab", "usage"], timeout=900)
    if not COLAB_TOKEN.exists():
        print(f"\n[cdrctl] no token created; login did not complete. rc={rc}")
        return 1
    shutil.copy2(COLAB_TOKEN, dst)
    os.chmod(dst, 0o600)
    _rc, out = sh(["colab", "usage"], timeout=120)   # captured now, just for the balance
    bal = parse_balance(out)
    pool = load_pool()
    set_state(pool, name, state=READY, last_balance=bal,
              added=time.strftime("%Y-%m-%d %H:%M:%S"))
    pool["active"] = name
    save_pool(pool)
    print(f"\n[cdrctl] stored credential for '{name}' (balance={bal}).")
    return 0


def activate(name: str) -> tuple[int, str]:
    src = _acct_token(name)
    if not src.exists():
        return 1, f"no stored token for '{name}' (run: cdrctl accounts add {name})"
    COLAB_TOKEN.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, COLAB_TOKEN)
    pool = load_pool()
    pool["active"] = name
    save_pool(pool)
    rc, out = sh(["colab", "usage"], timeout=120)
    return rc, out


def cmd_accounts_use(args) -> int:
    rc, out = activate(args.name)
    print(out.strip())
    if rc == 0:
        bal = parse_balance(out)
        pool = load_pool(); set_state(pool, args.name, last_balance=bal); save_pool(pool)
        print(f"[cdrctl] active account: {args.name} (balance={bal})")
    return rc


def cmd_accounts_who(args) -> int:
    pool = load_pool()
    rc, out = sh(["colab", "usage"], timeout=120)
    print(f"[cdrctl] active (per pool.json): {pool.get('active')}")
    print(out.strip())
    return rc


def cmd_accounts_list(args) -> int:
    pool = load_pool()
    if not pool["accounts"]:
        print("[cdrctl] no accounts registered. Add one: cdrctl accounts add <name>")
        return 0
    print(f"{'ACCOUNT':16} {'STATE':14} {'BALANCE':>9}  ADDED")
    for name, a in pool["accounts"].items():
        star = "*" if name == pool.get("active") else " "
        print(f"{star}{name:15} {a.get('state', '?'):14} "
              f"{str(a.get('last_balance', '?')):>9}  {a.get('added', '')}")
    return 0


def cmd_accounts_rm(args) -> int:
    pool = load_pool()
    pool["accounts"].pop(args.name, None)
    if pool.get("active") == args.name:
        pool["active"] = None
    save_pool(pool)
    shutil.rmtree(CDR_HOME / "accounts" / args.name, ignore_errors=True)
    print(f"[cdrctl] removed '{args.name}'")
    return 0


def cmd_status(args) -> int:
    pool = load_pool()
    for name in list(pool["accounts"]):
        rc, out = activate(name)
        bal = parse_balance(out)
        state = pool["accounts"][name].get("state", READY)
        if bal is not None and bal <= 0 and state == READY:
            state = EXHAUSTED
        set_state(pool, name, last_balance=bal, state=state)
        print(f"  {name:16} balance={bal} state={state}")
    save_pool(pool)
    return 0


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _run_batch_dummy(session: str) -> tuple[int, str]:
    """A trivial VM job to validate provisioning/rotation without the experiment."""
    prog = ("import subprocess;"
            "print('DUMMY OK', subprocess.run(['nvidia-smi','-L'],"
            "capture_output=True,text=True).stdout)")
    return sh(["colab", "exec", "-s", session, "--timeout", "120"],
              timeout=180)  # NOTE: pipe prog via stdin in real use; dummy checks wiring


def cmd_run(args) -> int:
    """Rotate through accounts, running experiment batches until the queue is empty.

    This is the orchestration skeleton. Live Colab steps (new/drivemount/exec/download)
    depend on the flaky unofficial CLI, so each is retried and classified; on quota
    exhaustion the account is cooled down and the next is used. The experiment itself is
    resumable, so a killed batch loses at most one job.
    """
    pool = load_pool()
    if not pool["accounts"]:
        print("[cdrctl] no accounts. Add some first: cdrctl accounts add <name>")
        return 1

    for rnd in range(args.max_rounds):
        name = next_ready(pool)
        if not name:
            print("[cdrctl] all accounts EXHAUSTED/AUTH_REQUIRED. "
                  "Add another (cdrctl accounts add) or wait for quota, then re-run.")
            return 2
        print(f"\n[cdrctl] === round {rnd + 1}: account '{name}' ===")
        rc, out = activate(name)
        bal = parse_balance(out)
        if rc != 0:
            act = classify_failure(out)
            print(f"[cdrctl] activate failed ({act}): {out.strip()[:200]}")
            set_state(pool, name, state=AUTH_REQUIRED if act == "reauth" else EXHAUSTED,
                      cooldown_until=time.time() + args.cooldown)
            save_pool(pool); continue
        if bal is not None and bal <= 0:
            print(f"[cdrctl] '{name}' balance {bal} <= 0; cooling down.")
            set_state(pool, name, state=EXHAUSTED, last_balance=bal,
                      cooldown_until=time.time() + args.cooldown)
            save_pool(pool); continue

        # provision a session (retry transient 503s)
        session = args.session
        for attempt in range(args.provision_retries):
            rc, out = sh(["colab", "new", "-s", session, "--gpu", args.gpu], timeout=300)
            if rc == 0:
                break
            act = classify_failure(out)
            print(f"[cdrctl] provision attempt {attempt + 1} failed ({act})")
            if act == "rotate":
                break
            time.sleep(args.retry_wait)
        if rc != 0:
            print(f"[cdrctl] could not provision on '{name}'; rotating.")
            set_state(pool, name, state=EXHAUSTED,
                      cooldown_until=time.time() + args.cooldown)
            save_pool(pool); continue

        # run the batch
        if args.dummy:
            brc, bout = _run_batch_dummy(session)
            print(f"[cdrctl] dummy batch rc={brc}: {bout.strip()[:300]}")
        else:
            brc, bout = _run_experiment_batch(session, args)
            print(bout.strip()[-800:])
        sh(["colab", "stop", "-s", session], timeout=120)

        # progress / completion
        done, planned = _progress(bout, args)
        set_state(pool, name, last_balance=parse_balance(bout) or bal)
        save_pool(pool)
        if done is not None and planned is not None and done >= planned:
            print(f"[cdrctl] queue complete ({done}/{planned}).")
            return 0
        if args.dummy:
            print("[cdrctl] dummy mode: one round only.")
            return 0
    print("[cdrctl] max rounds reached; re-run to continue.")
    return 0


def _run_experiment_batch(session: str, args) -> tuple[int, str]:
    """Pipe the Colab bootstrap to the VM to run one time-budgeted, resumable batch."""
    here = Path(__file__).resolve().parents[1]
    boot = (here / "scripts" / "colab_bootstrap.py").read_text()
    header = (f'REPO_URL = "{args.repo_url}"\nBRANCH = "{args.branch}"\n'
              f'DRIVE_DIR = "{args.drive_dir}"\nSESSION_HOURS = {args.max_hours}\n'
              f'RUN_ID = "{args.run_id}"\nMODE = "{args.mode}"\n')
    # drivemount may prompt once per account; inherit the terminal so you can approve
    # it (do not capture, or a prompt would deadlock).
    sh_interactive(["colab", "drivemount", "-s", session], timeout=300)
    proc = subprocess.run(["colab", "exec", "-s", session,
                           "--timeout", str(int(args.max_hours * 3600 + 900))],
                          input=header + boot, capture_output=True, text=True,
                          timeout=int(args.max_hours * 3600 + 1200))
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _progress(batch_output: str, args) -> tuple[int | None, int | None]:
    """Parse the bootstrap's coverage line for done vs planned, if present."""
    m = re.search(r"n_attack_jobs['\"]?\s*[:=]\s*(\d+)", batch_output)
    done = int(m.group(1)) if m else None
    planned = args.planned_attacks if args.planned_attacks else None
    return done, planned


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(prog="cdrctl")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("accounts")
    asub = a.add_subparsers(dest="acmd", required=True)
    p = asub.add_parser("add"); p.add_argument("name"); p.set_defaults(fn=cmd_accounts_add)
    p = asub.add_parser("list"); p.set_defaults(fn=cmd_accounts_list)
    p = asub.add_parser("use"); p.add_argument("name"); p.set_defaults(fn=cmd_accounts_use)
    p = asub.add_parser("who"); p.set_defaults(fn=cmd_accounts_who)
    p = asub.add_parser("rm"); p.add_argument("name"); p.set_defaults(fn=cmd_accounts_rm)

    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)

    p = sub.add_parser("run")
    p.add_argument("--run-id", default="full_200")
    p.add_argument("--mode", default="full", choices=["full", "pilot"])
    p.add_argument("--session", default="cdr")
    p.add_argument("--gpu", default="T4")
    p.add_argument("--max-hours", type=float, default=3.0, help="time budget per batch")
    p.add_argument("--max-rounds", type=int, default=50)
    p.add_argument("--provision-retries", type=int, default=3)
    p.add_argument("--retry-wait", type=float, default=60)
    p.add_argument("--cooldown", type=float, default=3600, help="EXHAUSTED cooldown secs")
    p.add_argument("--planned-attacks", type=int, default=600, help="200 imgs x 3 eps")
    p.add_argument("--repo-url", default="https://github.com/nongvantinh/computer-vision.git")
    p.add_argument("--branch", default="master")
    p.add_argument("--drive-dir", default="/content/drive/MyDrive/computer-vision")
    p.add_argument("--dummy", action="store_true", help="rotation dry-run, no experiment")
    p.set_defaults(fn=cmd_run)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
