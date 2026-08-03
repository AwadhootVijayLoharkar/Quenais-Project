"""
Self-checks for the regression harness itself.

These run without any of the pipeline being ported -- they verify that the
comparator and the reference table are trustworthy BEFORE anything is
measured against them. If this file fails, no other regression result means
anything.

Three properties are asserted:
  1. reference_values.py agrees with the golden pickles (no typos in the
     hand-entered table);
  2. the comparator reports OK for identical input (no false alarms);
  3. the comparator FAILS for each of the historical bug signatures
     (no false negatives -- this is the property that matters).
"""

from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
GOLDEN = HERE / "golden"

# Explicit, so this file runs identically under pytest, under `python -m`,
# and from any working directory. Relying on pytest's rootdir insertion
# makes the suite sensitive to where it is invoked from.
for _p in (HERE, HERE.parent.parent / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from compare_pickles import compare  # noqa: E402
from reference_values import SYSTEMS  # noqa: E402

STAGES = ["step0_classical.pkl", "step1_asf.pkl", "step2_hamiltonian.pkl"]


def load(system: str, stage: str):
    with open(GOLDEN / system / stage, "rb") as fh:
        return pickle.load(fh)


# ── 1. the reference table matches the golden data ───────────────────────

@pytest.mark.parametrize("system", sorted(SYSTEMS))
def test_classical_energies_match_golden(system):
    ref = SYSTEMS[system]
    s0 = load(system, "step0_classical.pkl")
    for method, (expected, _tier) in ref["energies"].items():
        if method.startswith("DMET"):
            continue
        got = s0["methods"][method]["energy"]
        assert abs(got - expected) < 1e-9, f"{system} {method}"


@pytest.mark.parametrize("system", sorted(SYSTEMS))
def test_structure_matches_golden(system):
    ref = SYSTEMS[system]["structure"]
    s1 = load(system, "step1_asf.pkl")
    s2 = load(system, "step2_hamiltonian.pkl")
    assert s1["nel"] == ref["nel"]
    assert list(s1["mo_list"]) == ref["mo_list"]
    assert s2["n_imp"] == ref["n_imp"]
    assert s2["n_bath"] == ref["n_bath"]
    assert s2["n_emb"] == ref["n_emb"]
    assert s2["n_alpha"] == ref["n_alpha"]
    assert s2["n_beta"] == ref["n_beta"]
    assert s2["reference_density_info"]["method"] == ref["reference_density_method"]


@pytest.mark.parametrize("system", sorted(SYSTEMS))
def test_scalars_match_golden(system):
    ref = SYSTEMS[system]
    s2 = load(system, "step2_hamiltonian.pkl")
    for key, (expected, tol) in ref["scalars"].items():
        assert abs(float(s2[key]) - expected) <= tol, f"{system} {key}"
    for spin, (expected, tol) in ref["ref_occ_sums"].items():
        got = float(np.sum(s2[f"ref_occ_{spin}"]))
        assert abs(got - expected) <= tol, f"{system} ref_occ_{spin}"


def test_n2_has_no_bath():
    """
    The fake-bath signature. N2's Schmidt spectrum is numerically zero, so
    the correct answer is zero bath orbitals. The pre-fix adaptive_bath
    keeps the top singular values regardless and fabricates four, producing
    ~20 Ha errors.
    """
    s2 = load("N2", "step2_hamiltonian.pkl")
    assert s2["n_bath"] == 0
    assert float(np.max(np.abs(s2["sv_all"]))) < SYSTEMS["N2"]["max_abs_sv_all_below"]


def test_lih_electron_count_comes_from_reference_density():
    """
    The 2x-HF signature. LiH's active space holds 2 electrons, but its
    embedding space holds 4 (2 alpha + 2 beta). Deriving the count from the
    active space gives (1, 1) and roughly doubles the energy.
    """
    s1 = load("LiH", "step1_asf.pkl")
    s2 = load("LiH", "step2_hamiltonian.pkl")
    assert s2["n_bath"] > 0, "precondition: LiH must have a real bath"
    naive_alpha = s1["nel"] // 2 + s1["nel"] % 2
    assert s2["n_alpha"] == 2
    assert s2["n_alpha"] != naive_alpha, "the buggy formula must not agree here"


def test_sch_degenerate_pair_kept_together():
    """
    The split-degenerate-orbital signature. MOs 11 and 12 have identical
    entanglement entropy; a cutoff that keeps one and drops the other
    breaks the molecule's symmetry.
    """
    mo_list = load("ScH", "step1_asf.pkl")["mo_list"]
    for a, b in SYSTEMS["ScH"]["degenerate_pairs"]:
        assert (a in mo_list) == (b in mo_list), f"MOs {a},{b} split"


# ── 2. the comparator does not raise false alarms ────────────────────────

@pytest.mark.parametrize("system", sorted(SYSTEMS))
@pytest.mark.parametrize("stage", STAGES)
def test_comparator_identity(system, stage):
    data = load(system, stage)
    report = compare(data, copy.deepcopy(data))
    assert report.ok, report.render(verbose=True)
    assert report.count("SKIP") == 0, "every key must be comparable"


def test_comparator_ignores_subtolerance_noise():
    data = load("LiH", "step2_hamiltonian.pkl")
    cand = copy.deepcopy(data)
    cand["ecore"] += 1e-12
    cand["h1e"] = cand["h1e"] + 1e-13
    assert compare(data, cand).ok


# ── 3. the comparator catches every historical bug signature ─────────────

def _perturbations(data):
    def with_(fn):
        c = copy.deepcopy(data)
        fn(c)
        return c

    return {
        "ecore drift 1e-4": with_(lambda c: c.__setitem__("ecore", c["ecore"] + 1e-4)),
        "electron count halved": with_(lambda c: c.__setitem__("n_alpha", 1)),
        "bath count changed": with_(lambda c: c.__setitem__("n_bath", 0)),
        "h1e scaled by 1+1e-8": with_(lambda c: c.__setitem__("h1e", c["h1e"] * (1 + 1e-8))),
        "ref_occ_alpha dropped": with_(lambda c: c.pop("ref_occ_alpha")),
        "h2e reshaped": with_(lambda c: c.__setitem__("h2e", c["h2e"][:3, :3, :3, :3])),
        "molecule tag swapped": with_(
            lambda c: c.__setitem__("mol_info", dict(c["mol_info"], molecule="ScH"))
        ),
        "schmidt spectrum zeroed": with_(
            lambda c: c.__setitem__("sv_all", np.zeros_like(np.asarray(c["sv_all"])))
        ),
        "NaN introduced": with_(
            lambda c: (c.__setitem__("h1e", c["h1e"].copy()),
                       c["h1e"].__setitem__((0, 0), np.nan))
        ),
    }


@pytest.mark.parametrize("label", list(_perturbations(load("LiH", "step2_hamiltonian.pkl"))))
def test_comparator_detects_perturbation(label):
    data = load("LiH", "step2_hamiltonian.pkl")
    cand = _perturbations(data)[label]
    assert not compare(data, cand).ok, f"comparator missed: {label}"
