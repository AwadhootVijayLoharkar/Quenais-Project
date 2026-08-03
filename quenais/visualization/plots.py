"""
Step 4 -- figures and CSV summaries from whatever the pipeline has produced.

Reads only what is on disk:

  step0_classical.pkl    classical reference energies
  step1_asf.pkl          active space, deviation spectrum, tier
  step2_hamiltonian.pkl  Schmidt spectrum, bath quality, ecore, mu
  results/gqe_train.log  "[epoch N] {...}" lines from the GQE trainer

Every output is generated independently and skipped with a printed reason
when its input is missing. Running this after only steps 0-2, or on a
Qiskit-only install with no GQE log, produces the figures it can and says
why the rest are absent.

THE CASCI REFERENCE IS NOT THE ONE IN THE PICKLE
------------------------------------------------
reference_density_info["e_cas"] is the small CASCI computed during phase B
to help BUILD the reference density, before Schmidt decomposition even
runs. It never reflects n_bath, mu, or the actual n_emb-orbital embedded
solve, so it stayed frozen at the wrong value even after the
electron-count and dm_a_hf fixes landed. Reporting it as "DMET+CASCI" is
wrong.

The real embedding CASCI is obtained by solving CASCI on the current
h1e_emb / h2e_emb / ecore / n_alpha / n_beta -- see true_embedding_casci().
"""

from __future__ import annotations

import ast
import csv
import os
import pickle
import re
import warnings

import numpy as np

__all__ = ["main", "parse_gqe_log", "true_embedding_casci"]

#: "[epoch 12] {'GQE-optimized/energy/min': -7.88, ...}"
EPOCH_RE = re.compile(r"\[epoch (\d+)\]\s*(\{.*?\})", re.DOTALL)

#: The trainer prints numpy scalars as np.float64(-7.88); strip the wrapper
#: so the dict parses literally.
FLOAT64_RE = re.compile(r"np\.float64\(([^)]*)\)")

CLASSICAL_COLOUR = "#4C72B0"
DMET_COLOUR = "#C44E52"
GQE_COLOUR = "#55A868"


# ═════════════════════════════════════════════════════════════════════════
# Loading
# ═════════════════════════════════════════════════════════════════════════

def _load(path, label):
    if not os.path.exists(path):
        print(f"  [skip] {label}: {path} not found")
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def parse_gqe_log(log_path):
    """
    Extract the per-epoch metric dicts from a training log.

    These lines exist only because of the train_pipeline.py hunk in the
    submodule patch. A non-empty log with none of them means the checkout
    was unpatched -- which is worth saying out loud, because the training
    run itself will have looked entirely successful.

    Uses ast.literal_eval, never eval. The log is machine-generated, but
    there is no reason to execute it.
    """
    if not log_path or not os.path.exists(log_path):
        print(f"  [skip] GQE log: {log_path} not found (run the GQE solver first)")
        return []

    with open(log_path) as fh:
        text = fh.read()

    rows = []
    for match in EPOCH_RE.finditer(text):
        epoch = int(match.group(1))
        payload = FLOAT64_RE.sub(r"\1", match.group(2))
        try:
            record = ast.literal_eval(payload)
        except (ValueError, SyntaxError) as exc:
            warnings.warn(f"Could not parse epoch {epoch}: {exc}", RuntimeWarning)
            continue
        record["epoch"] = epoch
        rows.append(record)

    rows.sort(key=lambda r: r["epoch"])

    if not rows and os.path.getsize(log_path) > 0:
        warnings.warn(
            f"{log_path} exists and is non-empty but contains no "
            f"'[epoch N] {{...}}' lines. That is the signature of a "
            f"gqe-for-qsci checkout missing the train_pipeline.py patch "
            f"hunk. Run: quenais-gqe-setup --repo <checkout>",
            RuntimeWarning,
        )
    return rows


def _col(rows, key):
    """One metric as (epochs, values), skipping rows where it is absent."""
    xs, ys = [], []
    for row in rows:
        if key in row and row[key] is not None:
            xs.append(row["epoch"])
            ys.append(float(row[key]))
    return np.asarray(xs), np.asarray(ys)


# ═════════════════════════════════════════════════════════════════════════
# The real embedding CASCI
# ═════════════════════════════════════════════════════════════════════════

_casci_cache = {}


def true_embedding_casci(cfg):
    """
    Solve CASCI on the current embedding Hamiltonian.

    Needs PySCF but NOT CUDA-Q -- the adapter's solver path is pure PySCF,
    so this works on a Qiskit-only install.
    """
    key = cfg.step2_file
    if key in _casci_cache:
        return _casci_cache[key]

    if not os.path.exists(cfg.step2_file):
        _casci_cache[key] = None
        return None

    try:
        from quenais.quantum.gqe_adapter import load_from_dmet_pickle

        mol = load_from_dmet_pickle(cfg.step2_file)
        energy = float(mol.compute_casci())
    except Exception as exc:
        warnings.warn(
            f"Could not compute the true embedding CASCI energy ({exc}). "
            f"It is omitted rather than substituting the phase B "
            f"reference-density CASCI, which is a different quantity and "
            f"would look plausible while being wrong.",
            RuntimeWarning,
        )
        energy = None

    _casci_cache[key] = energy
    return energy


# ═════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════

def _save(fig, path):
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_asf_spectrum(cfg, step1):
    import matplotlib.pyplot as plt

    if step1 is None:
        return
    dev = np.asarray(step1["deviation"])
    active = set(step1["mo_list"])
    order = np.argsort(-dev)
    colours = [DMET_COLOUR if i in active else CLASSICAL_COLOUR for i in order]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(order)), dev[order], color=colours)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Natural orbital (sorted by deviation)")
    ax.set_ylabel("deviation = min(n, 2-n)")
    ax.set_title(f"ASF deviation spectrum -- {step1['mol_info']['molecule']} "
                 f"(Tier {step1['tier']}, red = selected active space)")
    fig.tight_layout()
    _save(fig, os.path.join(cfg.plots_dir, "fig1_asf_deviation_spectrum.png"))


def plot_dmet_spectrum(cfg, step2):
    """
    The FULL Schmidt spectrum, not just the kept bath -- this is the plot
    that shows why adaptive_bath chose the count it did.
    """
    import matplotlib.pyplot as plt

    if step2 is None:
        return
    sv_all = np.asarray(step2.get("sv_all", []))
    if sv_all.size == 0:
        print("  [skip] fig2: step 2 pickle has no 'sv_all' -- re-run the "
              "embedding stage to save the full spectrum.")
        return

    n_bath = step2["n_bath"]
    fig, ax = plt.subplots(figsize=(9, 4))
    colours = [DMET_COLOUR if i < n_bath else CLASSICAL_COLOUR
               for i in range(len(sv_all))]

    if np.max(sv_all) <= 0:
        # A log axis cannot render an all-zero spectrum, and that spectrum
        # is exactly the interesting case: N2, no bath, correctly.
        ax.bar(range(len(sv_all)), sv_all, color=colours)
        ax.set_ylabel("singular value (linear -- all values are zero)")
        ax.text(0.5, 0.5,
                "all singular values numerically zero\n"
                "-> no bath (correct behaviour)",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
    else:
        ax.bar(range(len(sv_all)), np.maximum(sv_all, 1e-20), color=colours)
        ax.set_yscale("log")
        ax.set_ylabel("singular value (log scale)")

    ax.set_xlabel("Schmidt singular value index (sorted descending)")
    ax.set_title(f"DMET Schmidt spectrum -- {step2['mol_info']['molecule']}  "
                 f"(red = kept as bath, n_bath={n_bath}, "
                 f"sv2_cov={step2['sv2_cov']:.4f})")
    fig.tight_layout()
    _save(fig, os.path.join(cfg.plots_dir, "fig2_dmet_schmidt_spectrum.png"))


def plot_gqe_convergence(cfg, gqe_rows):
    import matplotlib.pyplot as plt

    if not gqe_rows:
        return
    series = [
        ("GQE-optimized(best_so_far)/energy - R-CASCI",
         "GQE-optimized vs CASCI", "#DD8452"),
        ("Local-refined(best_so_far)/energy - R-CASCI",
         "Local-refined vs CASCI", GQE_COLOUR),
        ("Global-refined(best_so_far)/energy - R-CASCI",
         "Global-refined vs CASCI", CLASSICAL_COLOUR),
        ("Global-refined(best_so_far)/energy - R-CCSD",
         "Global-refined vs CCSD", "#8172B2"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for key, label, colour in series:
        x, y = _col(gqe_rows, key)
        if len(x):
            ax.plot(x, y, label=label, color=colour, linewidth=1.8)
            plotted = True

    if not plotted:
        print("  [skip] fig3: no energy-error columns found in the log")
        plt.close(fig)
        return

    ax.axhline(1.6e-3, color="gray", linestyle="--", linewidth=1,
               label="chemical accuracy (1.6 mHa)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("|energy error| (Ha, log scale)")
    ax.set_title("GQE-for-QSCI convergence vs classical references")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(cfg.plots_dir, "fig3_gqe_energy_convergence.png"))


def plot_gqe_resources(cfg, gqe_rows):
    import matplotlib.pyplot as plt

    if not gqe_rows:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for key, label in [("GQE-optimized/cx_count/max", "cx_count (max)"),
                       ("GQE-optimized/total_gates/max", "total_gates (max)")]:
        x, y = _col(gqe_rows, key)
        if len(x):
            ax1.plot(x, y, label=label, linewidth=1.5)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("gate count")
    ax1.set_title("Circuit resources per epoch")
    ax1.legend(fontsize=8)

    for key, label in [
        ("Global-refined(best_so_far)/subspace_dim", "Global-refined subspace_dim"),
        ("Local-refined(best_so_far)/subspace_dim", "Local-refined subspace_dim"),
    ]:
        x, y = _col(gqe_rows, key)
        if len(x):
            ax2.plot(x, y, label=label, linewidth=1.5)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("number of configurations")
    ax2.set_title("Subspace size growth")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, os.path.join(cfg.plots_dir, "fig4_gqe_circuit_resources.png"))


def plot_method_comparison(cfg, step0, step2, gqe_rows):
    import matplotlib.pyplot as plt

    labels, energies, colours = [], [], []

    if step0 is not None:
        for name, data in step0["methods"].items():
            energy = data.get("energy")
            if energy is not None:
                labels.append(name)
                energies.append(energy)
                colours.append(CLASSICAL_COLOUR)

    e_casci = None
    if step2 is not None:
        if step2.get("reference_density_info", {}).get("method") == "casci":
            e_casci = true_embedding_casci(cfg)
            if e_casci is not None:
                labels.append("DMET+CASCI(active)")
                energies.append(e_casci)
                colours.append(DMET_COLOUR)

    if gqe_rows and e_casci is not None:
        _x, err = _col(gqe_rows, "Global-refined(best_so_far)/energy - R-CASCI")
        if len(err):
            labels.append("DMET+GQE (Global-refined, final)")
            energies.append(e_casci + err[-1])
            colours.append(GQE_COLOUR)

    if not labels:
        print("  [skip] fig5: no energies available from any stage yet")
        return None

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 5))
    ax.bar(labels, energies, color=colours)
    ax.set_ylabel("Total energy (Ha)")
    ax.set_title(f"Method comparison -- {cfg.molecule}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, os.path.join(cfg.plots_dir, "fig5_method_comparison.png"))
    return labels, energies


# ═════════════════════════════════════════════════════════════════════════
# CSV
# ═════════════════════════════════════════════════════════════════════════

def write_results_summary(cfg, step0, step1, step2, comparison):
    """
    The file a partner is most likely to send back, so every energy carries
    its reproducibility tier. See docs/limitations.md.
    """
    from quenais.classical.runner import METHOD_TIERS

    path = os.path.join(cfg.results_dir, "results_summary.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["molecule", cfg.molecule])
        writer.writerow(["basis", cfg.basis])
        writer.writerow([])

        if step1 is not None:
            writer.writerow(["-- Step 1: ASF --"])
            writer.writerow(["tier", step1["tier"]])
            writer.writerow(["active_space_nel", step1["nel"]])
            writer.writerow(["active_space_norb", step1["n_active_orbs"]])
            writer.writerow(["mo_list", step1["mo_list"]])
            writer.writerow(["forced_active_space",
                             step1.get("forced_active_space", "unknown")])
            writer.writerow(["correlation_strength", step1["corr_strength"]])
            writer.writerow([])

        if step2 is not None:
            writer.writerow(["-- Step 2: DMET --"])
            writer.writerow(["n_imp", step2["n_imp"]])
            writer.writerow(["n_bath", step2["n_bath"]])
            writer.writerow(["n_emb", step2["n_emb"]])
            writer.writerow(["n_alpha", step2["n_alpha"]])
            writer.writerow(["n_beta", step2["n_beta"]])
            writer.writerow(["sv2_coverage", step2["sv2_cov"]])
            writer.writerow(["ecore_Ha", step2["ecore"]])
            writer.writerow(["mu_Ha", step2["mu"]])
            writer.writerow(["reference_density_method",
                             step2.get("reference_density_info", {}).get("method")])
            check = step2.get("embedded_scf_check")
            if check:
                writer.writerow(["embedded_scf_vs_uhf_Ha", check["delta"]])
            writer.writerow([])

        if comparison:
            labels, energies = comparison
            writer.writerow(["-- Method comparison (Ha) --"])
            writer.writerow(["method", "energy_Ha", "reproducibility"])
            for label, energy in zip(labels, energies):
                if label.startswith("DMET+GQE"):
                    tier = "stochastic"
                elif label.startswith("DMET+CASCI"):
                    tier = "deterministic"
                else:
                    tier = METHOD_TIERS.get(label, "unknown")
                writer.writerow([label, energy, tier])
            writer.writerow([])
            writer.writerow(["# 'optimizer-dependent' and 'stochastic' values "
                             "can differ between runs; see docs/limitations.md"])

    print(f"  Saved {path}")
    return path


def write_gqe_epoch_csv(cfg, gqe_rows):
    if not gqe_rows:
        return None
    keys = sorted({k for row in gqe_rows for k in row})
    path = os.path.join(cfg.results_dir, "gqe_epoch_log.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(gqe_rows)
    print(f"  Saved {path}  ({len(gqe_rows)} epochs, {len(keys)} columns)")
    return path


# ═════════════════════════════════════════════════════════════════════════
# Stage entry point
# ═════════════════════════════════════════════════════════════════════════

def main(cfg, force=False, no_scan=False, no_quantum_scan=False):
    """
    Produce every figure and CSV the available data supports.

    force / no_scan / no_quantum_scan are accepted for CLI signature
    compatibility; this stage only reads what earlier stages wrote.
    """
    import matplotlib

    matplotlib.use("Agg")   # no display on a compute node

    os.makedirs(cfg.plots_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[Step 4] Visualisation -- {cfg.molecule}")
    print(f"{'='*60}")

    step0 = _load(cfg.step0_file, "step 0 (classical)")
    step1 = _load(cfg.step1_file, "step 1 (ASF)")
    step2 = _load(cfg.step2_file, "step 2 (DMET)")
    gqe_rows = parse_gqe_log(cfg.gqe_log_file)
    if gqe_rows:
        print(f"  Parsed {len(gqe_rows)} epochs from {cfg.gqe_log_file}")

    plot_asf_spectrum(cfg, step1)
    plot_dmet_spectrum(cfg, step2)
    plot_gqe_convergence(cfg, gqe_rows)
    plot_gqe_resources(cfg, gqe_rows)
    comparison = plot_method_comparison(cfg, step0, step2, gqe_rows)

    summary = write_results_summary(cfg, step0, step1, step2, comparison)
    epoch_csv = write_gqe_epoch_csv(cfg, gqe_rows)

    return {
        "results_summary": summary,
        "gqe_epoch_log": epoch_csv,
        "n_epochs": len(gqe_rows),
        "comparison": comparison,
    }
