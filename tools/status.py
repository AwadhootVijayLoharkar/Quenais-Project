#!/usr/bin/env python
"""
Progress of every benchmark job, from its log file.

    python tools/status.py              # every *.log in $B
    python tools/status.py $B/n2_gqe_angles.log $B/fe2bpym.log

One line per log: grid, runs done/total, failures, mean time per run that
actually ran, estimated time left, and the state:

  running   the log changed in the last `--stale` minutes (default 90)
  finished  the driver wrote its [collect] summary
  STALE?    no [collect] and no change for longer than --stale: the driver
            may have been killed (allocation ended?) -- check with ps

Works from any window/node: logs live in the shared $B. A log that holds
several grids (bash -c "a; b") is reported per grid.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

HEADER = re.compile(r"^\[benchmark\] (\S+): (\d+) runs over")
RUN = re.compile(r"^\[(\d+)/(\d+)\] \S+ ")
OK = re.compile(r" ok +([0-9.]+) s")


def parse(text):
    """List of per-grid dicts from one log's text."""
    grids, cur = [], None
    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            cur = {"grid": m.group(1), "total": int(m.group(2)), "done": 0,
                   "ok_times": [], "failed": 0, "skipped": 0,
                   "collected": False, "last": ""}
            grids.append(cur)
            continue
        if cur is None:
            continue
        if RUN.match(line):
            cur["done"] += 1
            cur["last"] = line.strip()
            m = OK.search(line)
            if m:
                cur["ok_times"].append(float(m.group(1)))
            elif "FAILED" in line:
                cur["failed"] += 1
            elif "skip" in line:
                cur["skipped"] += 1
        elif line.startswith("[collect]"):
            cur["collected"] = True
    return grids


def summarise(g, age_min, stale_min):
    t = g["ok_times"]
    mean = sum(t) / len(t) if t else None
    left = g["total"] - g["done"]
    if g["collected"] and left == 0:
        state = "finished"
    elif g["collected"]:
        state = "finished*"          # stopped early (e.g. killed, restarted)
    elif age_min > stale_min:
        state = "STALE?"
    else:
        state = "running"
    eta = "-" if mean is None or left == 0 else f"~{left * mean / 3600:.1f} h"
    return {
        "grid": g["grid"], "progress": f"{g['done']}/{g['total']}",
        "failed": str(g["failed"]),
        "per_run": "-" if mean is None else f"{mean / 60:.0f} min",
        "left": eta, "state": state,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--stale", type=float, default=90,
                    help="minutes without log change before 'STALE?' (default 90)")
    ap.add_argument("--all", action="store_true",
                    help="also list finished grids (default: only unfinished "
                         "ones and those finished in the last 24 h)")
    a = ap.parse_args(argv)

    logs = [Path(x) for x in a.logs]
    if not logs:
        base = os.environ.get("B")
        if not base:
            ap.error("give log files or export B")
        logs = sorted(Path(base).glob("*.log"))
    now = time.time()
    rows = []
    for log in logs:
        try:
            text = log.read_text(errors="replace")
            age = (now - log.stat().st_mtime) / 60
        except OSError:
            continue
        for g in parse(text):
            r = summarise(g, age, a.stale)
            if not a.all and r["state"].startswith("finished") and age > 24 * 60:
                continue
            r["log"] = log.name
            r["changed"] = f"{age:.0f} min ago"
            rows.append(r)
    if not rows:
        print("no matching logs")
        return 0
    cols = ["grid", "progress", "failed", "per_run", "left", "state", "changed", "log"]
    table = [cols] + [[r[c] for c in cols] for r in rows]
    w = [max(len(t[i]) for t in table) for i in range(len(cols))]
    for t in table:
        print("  ".join(x.ljust(w[i]) for i, x in enumerate(t)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
