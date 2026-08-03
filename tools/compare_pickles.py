"""
Key-by-key comparison of a QuEnAIS stage pickle against a golden reference.

WHY THIS EXISTS
---------------
Every bug in this project's history produced output with the right shape, a
plausible magnitude and the wrong value: a bath fabricated from numerical
noise, an electron count taken from the wrong space, a cache entry belonging
to a different molecule. None of those crash, and none are visible in a
summary table. They are only caught by comparing scalars against a known-good
run, one at a time, with a tolerance chosen per quantity.

So this module is deliberately paranoid:
  - a key present in golden but missing from the candidate is a FAILURE,
    not a skip (that is how a dropped ref_occ_alpha would hide);
  - an array whose shape changed is a FAILURE before values are compared;
  - NaN/inf anywhere is a FAILURE;
  - keys the comparator does not know how to compare are reported as
    SKIPPED and counted, never silently ignored.

Usage:
    python tools/compare_pickles.py golden.pkl candidate.pkl
    python tools/compare_pickles.py golden.pkl candidate.pkl --tol 1e-9 -v
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Default absolute tolerance for float comparison.
DEFAULT_TOL = 1e-9

# Per-key tolerance overrides. Keys not listed use DEFAULT_TOL.
# Rationale for the loose ones:
#   ecore / mu / uhf_energy on ScH carry ~750 Ha of magnitude, so 1e-9
#   relative is below double-precision noise for accumulated sums.
DEFAULT_KEY_TOL = {
    "ecore": 1e-8,
    "mu": 1e-8,
    "uhf_energy": 1e-8,
    "mp2_energy": 1e-8,
    "e_cas": 1e-8,
    "corr_strength": 1e-9,
    "sv": 1e-10,
    "sv_all": 1e-10,
    "sv_gap": 1e-10,
    "sv2_cov": 1e-10,
    "ref_occ_alpha": 1e-8,
    "ref_occ_beta": 1e-8,
    "h1e": 1e-9,
    "h2e": 1e-9,
    "deviation": 1e-9,
    "no_occ": 1e-9,
    "mo_coeff": 1e-7,       # sign/phase of degenerate MOs is not unique
    "mo_coeff_uhf": 1e-7,   # ditto
    "mo_energy": 1e-9,
    "dm_ao_alpha_mp2": 1e-9,
    "dm_ao_beta_mp2": 1e-9,
    "dm_ao_total_mp2": 1e-9,
    "lowdin_weights": 1e-9,
}

# Keys whose value is expected to differ between machines/runs and which are
# reported but never failed on.
INFORMATIONAL_KEYS = {"total_time", "provenance", "timestamp"}

PASS, FAIL, SKIP, INFO = "PASS", "FAIL", "SKIP", "INFO"


@dataclass
class Finding:
    key: str
    status: str
    detail: str = ""

    def __str__(self) -> str:
        return f"  [{self.status:4}] {self.key:32} {self.detail}"


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, key: str, status: str, detail: str = "") -> None:
        self.findings.append(Finding(key, status, detail))

    def count(self, status: str) -> int:
        return sum(1 for f in self.findings if f.status == status)

    @property
    def ok(self) -> bool:
        return self.count(FAIL) == 0

    def summary(self) -> str:
        return (f"{self.count(PASS)} passed, {self.count(FAIL)} failed, "
                f"{self.count(SKIP)} skipped, {self.count(INFO)} informational")

    def render(self, verbose: bool = False) -> str:
        lines = []
        for f in self.findings:
            if verbose or f.status in (FAIL, SKIP):
                lines.append(str(f))
        lines.append("")
        lines.append(f"  {self.summary()}")
        lines.append(f"  RESULT: {'OK' if self.ok else 'MISMATCH'}")
        return "\n".join(lines)


def _tol_for(key: str, key_tol: dict[str, float], default: float) -> float:
    """
    Tolerance for a key. Full dotted paths win over bare names, so a caller
    can loosen one quantity ("methods.CASSCF.energy") without loosening
    every "energy" in the file.
    """
    if key in key_tol:
        return key_tol[key]
    return key_tol.get(key.split(".")[-1], default)


def _compare_arrays(key: str, a: np.ndarray, b: np.ndarray,
                    tol: float, report: Report) -> None:
    if a.shape != b.shape:
        report.add(key, FAIL, f"shape {b.shape} != golden {a.shape}")
        return
    if a.size == 0:
        report.add(key, PASS, "empty array")
        return
    if not np.issubdtype(a.dtype, np.number):
        same = bool(np.array_equal(a, b))
        report.add(key, PASS if same else FAIL,
                   "" if same else "non-numeric array differs")
        return
    if not np.all(np.isfinite(b)):
        report.add(key, FAIL, "candidate contains NaN or inf")
        return

    diff = np.max(np.abs(a.astype(float) - b.astype(float)))
    if diff <= tol:
        report.add(key, PASS, f"max|d|={diff:.2e} <= {tol:.0e}")
    else:
        idx = np.unravel_index(int(np.argmax(np.abs(a - b))), a.shape)
        report.add(key, FAIL,
                   f"max|d|={diff:.3e} > {tol:.0e} at {idx} "
                   f"(golden={float(a[idx]):.12g}, got={float(b[idx]):.12g})")


def _compare_scalar(key: str, a: Any, b: Any, tol: float, report: Report) -> None:
    if isinstance(a, bool) or isinstance(b, bool):
        report.add(key, PASS if a == b else FAIL,
                   "" if a == b else f"golden={a}, got={b}")
        return
    fa, fb = float(a), float(b)
    if not np.isfinite(fb):
        report.add(key, FAIL, f"candidate is {fb}")
        return
    d = abs(fa - fb)
    if d <= tol:
        report.add(key, PASS, f"|d|={d:.2e} <= {tol:.0e}")
    else:
        report.add(key, FAIL,
                   f"|d|={d:.3e} > {tol:.0e} (golden={fa:.12g}, got={fb:.12g})")


def _compare_value(key: str, a: Any, b: Any, tol: float, report: Report,
                   key_tol: dict | None = None) -> None:
    if type(a) is not type(b) and not (
        isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number))
    ):
        report.add(key, FAIL, f"type {type(b).__name__} != golden {type(a).__name__}")
        return

    if isinstance(a, dict):
        _compare_dict(a, b, tol, report, prefix=key + ".", key_tol=key_tol)
    elif isinstance(a, np.ndarray):
        _compare_arrays(key, a, np.asarray(b), tol, report)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            report.add(key, FAIL, f"length {len(b)} != golden {len(a)}")
        elif all(isinstance(x, (int, float, np.number)) for x in a):
            _compare_arrays(key, np.asarray(a, dtype=float),
                            np.asarray(b, dtype=float), tol, report)
        else:
            same = list(a) == list(b)
            report.add(key, PASS if same else FAIL,
                       "" if same else f"golden={a}, got={b}")
    elif isinstance(a, (int, float, np.number)):
        _compare_scalar(key, a, b, tol, report)
    elif isinstance(a, (str, bytes)) or a is None:
        same = a == b
        report.add(key, PASS if same else FAIL,
                   "" if same else f"golden={a!r}, got={b!r}")
    else:
        report.add(key, SKIP, f"no comparator for {type(a).__name__}")


def _compare_dict(golden: dict, cand: dict, tol: float,
                  report: Report, prefix: str = "",
                  key_tol: dict | None = None) -> None:
    key_tol = DEFAULT_KEY_TOL if key_tol is None else key_tol
    for k in golden:
        key = prefix + str(k)
        short = str(k)
        if short in INFORMATIONAL_KEYS:
            report.add(key, INFO, "not compared by design")
            continue
        if k not in cand:
            report.add(key, FAIL, "MISSING from candidate")
            continue
        _compare_value(key, golden[k], cand[k],
                       _tol_for(key, key_tol, tol), report, key_tol)

    for k in cand:
        if k not in golden:
            report.add(prefix + str(k), INFO, "new key, not in golden")


def compare(golden: Any, candidate: Any, tol: float = DEFAULT_TOL,
            key_tol: dict | None = None) -> Report:
    """
    Compare two loaded pickle payloads. Returns a Report.

    key_tol overrides DEFAULT_KEY_TOL. Entries may be bare names ("ecore")
    or full dotted paths ("methods.CASSCF.energy"); the full path wins.
    Use it to loosen an optimizer-dependent quantity without loosening the
    deterministic ones next to it.
    """
    merged = dict(DEFAULT_KEY_TOL)
    if key_tol:
        merged.update(key_tol)
    report = Report()
    if isinstance(golden, dict) and isinstance(candidate, dict):
        _compare_dict(golden, candidate, tol, report, key_tol=merged)
    else:
        _compare_value("<root>", golden, candidate, tol, report, merged)
    return report


def compare_files(golden_path: str, candidate_path: str,
                  tol: float = DEFAULT_TOL, key_tol: dict | None = None) -> Report:
    with open(golden_path, "rb") as fh:
        golden = pickle.load(fh)
    with open(candidate_path, "rb") as fh:
        candidate = pickle.load(fh)
    return compare(golden, candidate, tol=tol, key_tol=key_tol)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare a QuEnAIS stage pickle against a golden reference."
    )
    p.add_argument("golden")
    p.add_argument("candidate")
    p.add_argument("--tol", type=float, default=DEFAULT_TOL,
                   help=f"default absolute tolerance (default {DEFAULT_TOL})")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="show passing keys too")
    args = p.parse_args()

    report = compare_files(args.golden, args.candidate, tol=args.tol)
    print(f"\ngolden   : {args.golden}")
    print(f"candidate: {args.candidate}\n")
    print(report.render(verbose=args.verbose))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
