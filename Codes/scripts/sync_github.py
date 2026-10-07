#!/usr/bin/env python3
"""Commit and push code, results and logs to github.com/DevDaring/Learning_Tech.

    python Codes/scripts/sync_github.py ["commit message"]

Safety:
- What is committed is governed by Learning_Tech/.gitignore: no .env, no secrets, no
  markdown except README.md, no submission files, no raw licensed data.
- Before every commit, each staged file is scanned for the value of every credential-like
  entry in Codes/.env; if any is found, nothing is committed or pushed.
- The GitHub token (Github_Classic_Token) is passed to git through GIT_ASKPASS for this one
  process only; it is never written to .git/config or printed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # Learning_Tech/
sys.path.insert(0, str(REPO / "Codes" / "src"))
from tlt import secrets  # noqa: E402

REMOTE = "https://github.com/DevDaring/Learning_Tech.git"
AUTHOR = ("DevDaring", "koushikdeb2009@gmail.com")
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def git(*args, env=None, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, env=env, check=check)


def main() -> int:
    if not (REPO / ".git").exists():
        git("init", "-b", "main")
    if "origin" not in git("remote").stdout.split():
        git("remote", "add", "origin", REMOTE)
    git("config", "user.name", AUTHOR[0])
    git("config", "user.email", AUTHOR[1])
    git("add", "-A")
    staged = [l for l in git("diff", "--cached", "--name-only").stdout.splitlines() if l]
    if not staged and not git("log", "origin/main..main", "--oneline", check=False).stdout.strip() \
            and git("rev-parse", "--verify", "origin/main", check=False).returncode == 0:
        print("nothing to commit or push")
        return 0
    # secret scan of everything about to be committed
    values = secrets.all_secret_values()
    leaks = []
    for rel in staged:
        p = REPO / rel
        if p.is_file() and p.stat().st_size < 200_000_000:
            text = p.read_bytes().decode("utf-8", errors="ignore")
            if any(v in text for v in values):
                leaks.append(rel)
        if Path(rel).name == ".env" or (rel.endswith(".md") and Path(rel).name.lower() != "readme.md"):
            leaks.append(rel)
    if leaks:
        git("reset", "-q")
        print("ABORTED: secret values or excluded files staged in:", leaks)
        return 2
    if staged:
        msg = sys.argv[1] if len(sys.argv) > 1 else f"Sync results and logs {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
        git("commit", "-q", "-m", msg + TRAILER)
    token = secrets.get("Github_Classic_Token")
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as f:
        f.write("#!/bin/sh\ncase \"$1\" in *Username*) echo x-access-token ;; *) echo \"$GH_TOKEN_FOR_PUSH\" ;; esac\n")
        askpass = f.name
    os.chmod(askpass, 0o700)
    try:
        env = dict(os.environ, GIT_ASKPASS=askpass, GH_TOKEN_FOR_PUSH=token, GIT_TERMINAL_PROMPT="0")
        r = git("push", "-u", "origin", "main", env=env, check=False)
    finally:
        os.unlink(askpass)
    out = (r.stdout + r.stderr).replace(token, "***")
    print(f"committed {len(staged)} files; push exit {r.returncode}\n{out.strip()[-400:]}")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
