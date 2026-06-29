"""Monitor a Conv-BKI training run: refresh RUN.md per epoch, enforce the
divergence guard, exit when training finishes.

Guard (project lead): stop training and flag if loss goes NaN, or val loss
rises for 2+ consecutive epochs. On a trip it SIGTERMs the training PID and
writes DIVERGED.flag next to RUN.md.

Runs as a detached background process; exits when the training PID dies (normal
completion) or on a divergence trip. Intended to be launched with the Bash
tool's run_in_background so the parent is notified on exit.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCH_VAL = re.compile(r"^\[epoch (\d+)\] VAL loss=(\S+) acc=\S+ miou=(\S+)")
NAN_LINE = re.compile(r"loss=(nan|inf|-inf)", re.IGNORECASE)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def refresh(meta, log, history, out):
    subprocess.run([sys.executable, os.path.join(HERE, "write_run_md.py"),
                    "--meta", meta, "--log", log, "--history", history, "--out", out],
                   check=False)


def val_losses(log):
    out = []
    if not os.path.isfile(log):
        return out
    with open(log, errors="replace") as f:
        for line in f:
            m = EPOCH_VAL.match(line.strip())
            if m:
                try:
                    out.append((int(m.group(1)), float(m.group(2))))
                except ValueError:
                    out.append((int(m.group(1)), float("nan")))
    return out


def has_nan(log):
    if not os.path.isfile(log):
        return False
    with open(log, errors="replace") as f:
        for line in f:
            if line.startswith("[epoch") or line.startswith("[final"):
                if NAN_LINE.search(line):
                    return True
    return False


def diverging(vls):
    """True if val loss rose across 2+ consecutive epoch transitions."""
    vals = [v for _, v in vls]
    if any(math.isnan(v) for v in vals):
        return True
    rises = 0
    for i in range(1, len(vals)):
        if vals[i] > vals[i - 1] + 1e-9:
            rises += 1
            if rises >= 2:
                return True
        else:
            rises = 0
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--history", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=int, default=60)
    args = p.parse_args()

    flag_path = os.path.join(os.path.dirname(args.out), "DIVERGED.flag")
    tripped = False

    while pid_alive(args.pid):
        refresh(args.meta, args.log, args.history, args.out)
        if has_nan(args.log) or diverging(val_losses(args.log)):
            tripped = True
            with open(flag_path, "w") as f:
                f.write("Divergence guard tripped (NaN or val loss rising 2+ "
                        "consecutive epochs). SIGTERM sent to training PID "
                        f"{args.pid}.\nval losses: {val_losses(args.log)}\n")
            try:
                os.kill(args.pid, signal.SIGTERM)
            except OSError:
                pass
            break
        time.sleep(args.interval)

    # Give the process a moment to flush train_history.json, then final refresh.
    time.sleep(3)
    refresh(args.meta, args.log, args.history, args.out)

    print("MONITOR_EXIT tripped=%s" % tripped)
    print("val losses:", val_losses(args.log))
    if tripped:
        print("DIVERGENCE GUARD TRIPPED — training killed. See", flag_path)


if __name__ == "__main__":
    main()
