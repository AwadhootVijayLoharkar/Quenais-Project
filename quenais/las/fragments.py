"""
Fragment specifications and active-orbital localisation.

SPEC FORMAT
-----------
    ATOMS:NORB:NA,NB[:2S]

    ATOMS  '+'-separated atom indices (0-based) or element symbols; a
           symbol means every atom of that element.   Sc   0   0+1   Fe+N
    NORB   active orbitals in the fragment
    NA,NB  alpha and beta electrons
    2S     optional, twice the target spin; default |NA-NB|

    'Sc:6:1,1'   'F:3:3,3'   '0:5:4,2:2'   '1:5:2,4:2'

LOCALISATION
------------
Step 1 hands over a set of active orbitals that are generally delocalised
over the whole molecule. Fragment by fragment, the part of the remaining
active space that overlaps most with the fragment's orthogonalised
(meta-Loewdin) atomic orbitals is split off: an SVD of

    M = L_K^T S C_rem

gives right singular vectors ordered by overlap with fragment K; the first
NORB_K of them span fragment K, the rest stay for the next fragment. The
singular values are returned as a diagnostic -- a value well below 1 means
the requested orbitals do not really live on that fragment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FragmentSpec", "parse_fragment_spec", "resolve_atoms",
           "fragment_ao_indices", "localize_active", "split_core_active_virtual"]


@dataclass(frozen=True)
class FragmentSpec:
    atoms: tuple          # tokens as given: ints or element symbols
    norb: int
    nelec: tuple          # (na, nb)
    spin2s: int

    @property
    def label(self):
        return "+".join(str(a) for a in self.atoms)


def parse_fragment_spec(text):
    parts = str(text).strip().split(":")
    if len(parts) not in (3, 4):
        raise ValueError(
            f"fragment spec {text!r}: expected ATOMS:NORB:NA,NB[:2S], "
            f"e.g. 'Sc:6:1,1' or '0:5:4,2:2'"
        )
    tokens = []
    for tok in parts[0].split("+"):
        tok = tok.strip()
        if not tok:
            raise ValueError(f"fragment spec {text!r}: empty atom token")
        tokens.append(int(tok) if tok.lstrip("-").isdigit() else tok)
    try:
        norb = int(parts[1])
        na, nb = (int(x) for x in parts[2].split(","))
    except ValueError as exc:
        raise ValueError(f"fragment spec {text!r}: {exc}") from exc
    spin2s = int(parts[3]) if len(parts) == 4 else abs(na - nb)
    if norb <= 0:
        raise ValueError(f"fragment spec {text!r}: NORB must be > 0")
    if not (0 <= na <= norb and 0 <= nb <= norb):
        raise ValueError(f"fragment spec {text!r}: ({na},{nb}) electrons do not "
                         f"fit in {norb} orbitals")
    if spin2s < abs(na - nb) or (spin2s - abs(na - nb)) % 2 or spin2s > na + nb:
        raise ValueError(f"fragment spec {text!r}: 2S={spin2s} is impossible "
                         f"with ({na},{nb}) electrons")
    return FragmentSpec(tuple(tokens), norb, (na, nb), spin2s)


def resolve_atoms(spec, atom_syms):
    """Atom indices of a fragment given the molecule's element symbols."""
    out = []
    for tok in spec.atoms:
        if isinstance(tok, int):
            if not 0 <= tok < len(atom_syms):
                raise ValueError(f"fragment {spec.label}: atom index {tok} out of "
                                 f"range (molecule has {len(atom_syms)} atoms)")
            out.append(tok)
        else:
            hits = [i for i, s in enumerate(atom_syms) if s.lower() == tok.lower()]
            if not hits:
                raise ValueError(f"fragment {spec.label}: no atom with element "
                                 f"{tok!r} in {atom_syms}")
            out.extend(hits)
    return sorted(set(out))


def fragment_ao_indices(mol, atoms):
    slices = mol.aoslice_by_atom()
    return [ao for a in atoms for ao in range(slices[a][2], slices[a][3])]


def check_disjoint(atom_lists, labels):
    seen = {}
    for atoms, label in zip(atom_lists, labels):
        for a in atoms:
            if a in seen:
                raise ValueError(f"atom {a} is in fragment {seen[a]} and {label}; "
                                 f"fragments must not share atoms")
            seen[a] = label


def split_core_active_virtual(n_mo, mo_list, n_electrons, n_active_electrons):
    """
    Index lists [core], [active], [virtual] from step 1's mo_list.

    The core is the lowest (N - n_active)/2 non-active orbitals in mo_coeff's
    own order. For AVAS/APC that is exactly PySCF's [core|active|virtual]
    block layout; for a forced list it is the lowest-energy UHF orbitals.
    """
    n_core_el = n_electrons - n_active_electrons
    if n_core_el < 0 or n_core_el % 2:
        raise ValueError(f"{n_core_el} core electrons: the LAS core must be "
                         f"doubly occupied (check spin and the step 1 space)")
    active = sorted(int(i) for i in mo_list)
    rest = [i for i in range(n_mo) if i not in set(active)]
    ncore = n_core_el // 2
    return rest[:ncore], active, rest[ncore:]


def localize_active(mol, C_act, frag_aos, norbs, lo_coeff=None, ovlp=None):
    """
    Split the active orbitals C_act (nao, ncas) into fragment blocks.

    Returns (C_frag list, singular-value list). Fragments are processed in
    the order given; the last one takes what is left, so its singular
    values are the most informative check.
    """
    if sum(norbs) != C_act.shape[1]:
        raise ValueError(f"fragment orbitals sum to {sum(norbs)} but the active "
                         f"space has {C_act.shape[1]}")
    if lo_coeff is None:
        from pyscf import lo

        lo_coeff = lo.orth_ao(mol, "meta_lowdin")
    s = mol.intor_symmetric("int1e_ovlp") if ovlp is None else ovlp
    rem = C_act
    blocks, svals = [], []
    for aos, n in zip(frag_aos, norbs):
        if n > len(aos):
            raise ValueError(f"cannot place {n} active orbitals on a fragment "
                             f"with only {len(aos)} basis functions")
        m = lo_coeff[:, aos].T @ s @ rem
        _u, sv, vt = np.linalg.svd(m, full_matrices=True)
        v = vt.T
        blocks.append(rem @ v[:, :n])
        svals.append(sv[:n].copy())
        rem = rem @ v[:, n:]
    return blocks, svals
