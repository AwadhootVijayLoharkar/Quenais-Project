"""
Stage 6 (GQE adapter and operator pools) tests.

Most of this file runs without CUDA-Q. That is deliberate: the point of the
lazy-import discipline is that a Qiskit-only install can still import,
inspect and test these modules.

The excitation-generator convention check is the important numerical test.
A sign or phase error there would not crash anything -- it would silently
seed the operator pool with wrong-phase excitations. It needs tequila and
OpenFermion, and skips cleanly without them.
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from quenais.quantum import gqe_adapter, gqe_pools

SHIM_DIR = Path(gqe_pools.__file__).parent / "_gqe_shims"


# ── Lazy imports: a Qiskit-only install must work ────────────────────────

@pytest.mark.parametrize("module", ["gqe_adapter", "gqe_pools"])
def test_no_heavy_imports_at_module_level(module):
    """
    cudaq, tequila, torch, pyscf and gqe_qsci must all be imported inside
    functions. A module-level import breaks CLI dispatch on an install
    that has only one of the two solver stacks.
    """
    path = Path(getattr(gqe_adapter if module == "gqe_adapter" else gqe_pools,
                        "__file__"))
    tree = ast.parse(path.read_text())
    forbidden = {"cudaq", "tequila", "torch", "pyscf", "gqe_qsci", "openfermion"}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = getattr(node, "module", None) or node.names[0].name
            assert name.split(".")[0] not in forbidden, (
                f"{module}: module-level import of {name}"
            )


def test_modules_import_with_the_cudaq_stack_absent():
    """
    Import both modules in a fresh interpreter with the CUDA-Q stack
    stubbed out as unimportable. This is the property that lets a
    Qiskit-only user run the pipeline at all.
    """
    code = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        root = name.split('.')[0]\n"
        "        return self if root in "
        "{'cudaq','tequila','torch','gqe_qsci'} else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('blocked: ' + name)\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from quenais.quantum import gqe_adapter, gqe_pools\n"
        "assert callable(gqe_pools.excitation_generator_qubit_op)\n"
        "assert callable(gqe_adapter.load_from_dmet_pickle)\n"
        "print('ok')\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_pool_classes_are_exposed_lazily():
    """
    The classes are not module attributes until requested, but __all__
    advertises them so `from ... import *` works in the shims.
    """
    for name in ("DMETExcitationPool", "DMETPauliEvolutionPool",
                 "DMETUCCSDBasedPool"):
        assert name in gqe_pools.__all__
        assert name not in vars(gqe_pools), (
            f"{name} is defined eagerly -- that would import cudaq at "
            f"module import time"
        )


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        gqe_pools.NotAThing


# ── Shim modules ─────────────────────────────────────────────────────────

def test_shim_directory_is_not_a_package():
    """
    _gqe_shims is a sys.path entry for the subprocess, not an importable
    subpackage. An __init__.py would make `import dmet_excitation_pool`
    fail from inside it.
    """
    assert SHIM_DIR.is_dir()
    assert not (SHIM_DIR / "__init__.py").exists()


@pytest.mark.parametrize(
    "shim", ["dmet_molecule_adapter.py", "dmet_excitation_pool.py"]
)
def test_shims_exist_and_re_export(shim):
    """The external repo imports these by top-level name."""
    path = SHIM_DIR / shim
    assert path.exists(), f"missing shim: {shim}"
    src = path.read_text()
    assert "import *" in src
    assert "__all__" in src, (
        "the shim must re-export __all__, otherwise `import *` silently "
        "drops names the external factory needs"
    )


def test_canonical_modules_declare_all():
    """`import *` in the shims only re-exports what __all__ lists."""
    for mod in (gqe_adapter, gqe_pools):
        assert hasattr(mod, "__all__") and mod.__all__


def test_adapter_all_covers_the_integration_contract():
    for name in ("DMETEmbeddingMolecule", "load_from_dmet_pickle"):
        assert name in gqe_adapter.__all__


def test_pools_all_covers_the_factory_contract():
    """factory.py imports exactly these two names."""
    for name in ("DMETPauliEvolutionPool", "DMETExcitationPool"):
        assert name in gqe_pools.__all__


# ── Documented invariants ────────────────────────────────────────────────

def test_generator_carries_the_minus_i_convention():
    """
    The factor that reconciles the raw anti-Hermitian UCC generator with
    tequila's stored Hermitian form. Validation found a consistent ratio of
    exactly +i between them; dropping this would silently change every
    excitation's phase.
    """
    src = inspect.getsource(gqe_pools.excitation_generator_qubit_op)
    assert "-1j" in src


def test_excitation_pool_accumulates_terms():
    """
    DMETExcitationPool must sum an excitation's Pauli terms into one
    operator. Appending them separately -- what DMETPauliEvolutionPool does
    -- cannot conserve particle number, because conservation is a property
    of the full sum.
    """
    src = inspect.getsource(gqe_pools._build_classes)
    assert "operator + weighted" in src or "weighted if operator is None" in src


def test_adapter_caches_the_ci_vector():
    """
    Without caching the CI vector alongside the energy, a cache HIT returns
    early and never sets _last_casci_civec, so every run after the first
    raises AttributeError in casci_avg_occs().
    """
    src = inspect.getsource(gqe_adapter.DMETEmbeddingMolecule.compute_casci)
    assert "civec" in src
    assert "_last_casci_civec" in src


def test_adapter_placeholders_are_inert():
    """
    geometry/basis/active_indices exist only so presence checks pass. If
    geometry ever became a real value, the stock tequila pools would
    silently accept this molecule and build gates for the wrong space.
    """
    src = inspect.getsource(gqe_adapter.DMETEmbeddingMolecule.__init__)
    assert "self.geometry = None" in src


# ── Numerical: the convention check ──────────────────────────────────────

def _tequila_generator_terms(tq_molecule, indices, angle=1.0):
    """tequila's generator as {pauli_word: coeff}."""
    gate = tq_molecule.make_excitation_gate(indices=indices, angle=angle)
    gates = gate.gates if hasattr(gate, "gates") else [gate]
    terms = {}
    for g in gates:
        for p in g.generator.paulistrings:
            word = ",".join(f"{k}{v}" for k, v in sorted(p.items()))
            terms[word] = terms.get(word, 0.0) + p._coeff
    return terms


def _our_generator_terms(indices):
    """Our generator in the same {pauli_word: coeff} format."""
    qubit_op = gqe_pools.excitation_generator_qubit_op(indices)
    terms = {}
    for term, coeff in qubit_op.terms.items():
        if len(term) == 0:
            continue
        word = ",".join(f"{idx}{letter}" for idx, letter in sorted(term))
        terms[word] = terms.get(word, 0.0) + coeff
    return terms


# Interleaved spin-orbital indexing, matching gqe-for-qsci's operator_pool:
# 2*i is alpha_i, 2*i+1 is beta_i. These are the three cases the original
# validation run covered, all of which passed with ratio exactly 1.000000.
GENERATOR_CASES = [
    ("single excitation (alpha, 0->1)", [(0, 2)]),
    ("single excitation (beta, 0->1)", [(1, 3)]),
    ("double excitation (alpha-beta)", [(1, 3), (0, 2)]),
]


@pytest.mark.needs_cudaq
@pytest.mark.parametrize("label,indices", GENERATOR_CASES,
                         ids=[c[0] for c in GENERATOR_CASES])
def test_excitation_generator_matches_tequila(label, indices):
    """
    Cross-check the hand-built Jordan-Wigner excitation generator against
    tequila's actual make_excitation_gate() output, on H2/STO-3G.

    Independent of any CCSD amplitude computation and of the adapter, so
    it isolates the one thing that cannot be reasoned about safely: does
    our generator match tequila's sign and normalisation convention?

    Two conditions must hold:
      1. identical Pauli-string sets -- a term present in only one side
         means the construction is wrong, not merely rescaled;
      2. a CONSISTENT ratio across every term. A consistent ratio that is
         not 1.0 would be a one-line fix (scale the generator); an
         inconsistent ratio means the construction is broken.

    Validated result: identical sets, ratio exactly 1.000000 on all three
    cases. Re-run after any edit to excitation_generator_qubit_op -- a
    phase error there crashes nothing and silently seeds the pool with
    wrong-phase excitations.
    """
    pytest.importorskip("openfermion")
    tq = pytest.importorskip("tequila")

    tq_molecule = tq.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 0.74",
        basis_set="sto-3g",
        active_orbitals=[0, 1],
        transformation="jordan-wigner",
    )

    ref = _tequila_generator_terms(tq_molecule, indices, angle=1.0)
    ours = _our_generator_terms(indices)

    assert set(ref) == set(ours), (
        f"{label}: Pauli-string sets differ.\n"
        f"  only in tequila: {sorted(set(ref) - set(ours))}\n"
        f"  only in ours   : {sorted(set(ours) - set(ref))}"
    )
    assert ref, f"{label}: tequila produced an empty generator"

    ratios = [ours[w] / ref[w] for w in ref if abs(ref[w]) > 1e-14]
    assert ratios, f"{label}: no comparable terms"

    spread = max(abs(r - ratios[0]) for r in ratios)
    assert spread < 1e-8, (
        f"{label}: ratios are inconsistent across terms (spread={spread:.3e}) "
        f"-- the generator construction is wrong, not merely rescaled"
    )
    assert abs(ratios[0] - 1.0) < 1e-8, (
        f"{label}: consistent ratio {ratios[0]:.6f}, expected 1.0. Multiply "
        f"excitation_generator_qubit_op()'s return by {ratios[0]:.6f} to "
        f"restore the convention."
    )


@pytest.mark.needs_cudaq
@pytest.mark.slow
def test_pool_classes_construct(golden_dir, tmp_path):
    """Smoke test: the pools build against a real embedding."""
    mol = gqe_adapter.load_from_dmet_pickle(
        str(golden_dir / "LiH" / "step2_hamiltonian.pkl"),
        cache_dir=tmp_path / ".cache",
    )
    pool = gqe_pools.DMETExcitationPool(mol, params=None, threshold=1e-8)
    assert pool.get_vocab_size() > 1


# ── Adapter against real data (PySCF only) ───────────────────────────────

@pytest.mark.needs_pyscf
@pytest.mark.slow
@pytest.mark.parametrize("system", ["LiH", "N2", "ScH"])
def test_adapter_reproduces_the_embedded_scf(system, golden_dir, tmp_path):
    """
    The adapter's own SCF must land on the full-molecule UHF energy, the
    same invariant the embedding stage checks. This confirms the adapter
    reads and reconstructs the Hamiltonian faithfully.
    """
    sys.path.insert(0, str(Path(__file__).parent / "regression"))
    from reference_values import EMBEDDED_SCF_VS_UHF_TOL

    mol = gqe_adapter.load_from_dmet_pickle(
        str(golden_dir / system / "step2_hamiltonian.pkl"),
        cache_dir=tmp_path / ".cache",
    )
    delta = mol.hf.e_tot - mol._step2_result["uhf_energy"]
    assert abs(delta) <= EMBEDDED_SCF_VS_UHF_TOL, (
        f"{system}: adapter SCF differs from full UHF by {delta:.3e} Ha"
    )


@pytest.mark.needs_pyscf
@pytest.mark.slow
def test_casci_survives_a_cache_hit(golden_dir, tmp_path):
    """
    The AttributeError this fix exists for: the second call must restore
    the CI vector from disk rather than skipping it.
    """
    path = str(golden_dir / "LiH" / "step2_hamiltonian.pkl")
    cache = tmp_path / ".cache"

    first = gqe_adapter.load_from_dmet_pickle(path, cache_dir=cache)
    e1 = first.compute_casci()
    occ_a1, _ = first.casci_avg_occs()

    second = gqe_adapter.load_from_dmet_pickle(path, cache_dir=cache)
    e2 = second.compute_casci()          # cache hit
    occ_a2, _ = second.casci_avg_occs()  # must not raise

    assert e1 == pytest.approx(e2, abs=1e-12)
    assert np.allclose(occ_a1, occ_a2, atol=1e-10)


@pytest.mark.needs_pyscf
def test_load_rejects_a_non_step2_pickle(tmp_path):
    import pickle

    bad = tmp_path / "bad.pkl"
    with open(bad, "wb") as fh:
        pickle.dump({"molecule": "LiH"}, fh)
    with pytest.raises(KeyError, match="not a valid step 2 pickle"):
        gqe_adapter.load_from_dmet_pickle(str(bad))
