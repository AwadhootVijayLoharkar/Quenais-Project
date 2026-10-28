# Reproducing the GENIE paper (QuEnAIS v0.3.0)

This tag (`v0.3.0`) is the exact code used for every number in the paper.
Development continues on `main`; results there may differ.

## 1. Environment

    git clone --recursive https://github.com/AwadhootVijayLoharkar/Quenais-Project
    cd Quenais-Project && git checkout v0.3.0
    git submodule update --init --recursive
    bash install.sh                  # builds the environment (see README)
    pip install -r requirements-paper.txt   # exact versions used for the paper

Third-party components are NOT part of this repository and keep their own
licenses: `gqe-for-qsci` (Apache-2.0, NVIDIA CUDA-QX / R. Kemmoku; we apply
`quenais/patches/gqe_dmet_source.patch` via `quenais-gqe-setup`), theochem/PyCI
and block2 (installed from upstream by `install.sh`). The submodule commits
used for the paper are recorded in `docs/submodules-paper.txt`.

GENIE runs need an NVIDIA GPU (paper: A100 40 GB, CUDA-Q 0.15.1).
Classical baselines, SQD, LAS and the router run on CPU.

## 2. Benchmark grids -> output folders

Every run is a row of a grid; set `B=<results folder>`.

| Paper content | Grid(s) (`benchmarks/`) | `--out` |
|---|---|---|
| N2/STO-3G: equal budget, basis fix, seeds | `n2_gqe_mo`, `n2_gqe_budget`, `n2_gqe_angles` | `$B/n2_curve` |
| GENIE vs SQD, product vs list | `n2_sqd_list`, `n2_sqd_opt` | `$B/n2_curve` |
| Noise model | `n2_noise` | `$B/n2_curve` |
| N2/cc-pVDZ | `n2_dz_gqe_mo`, `n2_dz_sqd`, `n2_dz_cpu`, `n2_dz_gpu` | `$B/n2_dz` |
| H10/H14 GENIE, depth | `h_chain_gqe`, `h14_gqe_depth` | `$B/h_chain` |
| H14 DMET+SQD, LAS on chains | `h_chain_compare`, `h_chain_compare_h14sqd`, `h_chain_compare_h14halves_warm` | `$B/h_chain_cmp` |
| Blind test | `blind_test`, `blind_depth` | `$B/blind` |
| [Fe(H2O)4]2bpym4+ (LAS) | `fe2bpym_las`, `fe2bpym_lassqd_warm` | `$B/fe2bpym` |

    python tools/run_benchmark.py benchmarks/<grid>.json --out <out>
    python tools/run_benchmark.py benchmarks/<grid>.json --out <out> --collect-only

## 3. Analysis (CPU)

    export QUENAIS_SELECTION_MAX_DET=4000000
    python tools/selection_baselines.py <out> --where solver=gqe --basis both   # MO only for >1e5 dets: --basis mo --methods oracle,cipsi,asci
    python tools/regime_diagnostics.py  <out> --where solver=gqe gqe_basis=mo --exact
    python tools/paper_data.py          <out> --where solver=gqe gqe_basis=mo

## 4. Figures

The LaTeX source, `figures/make_figures.py` and the exported data
(`figures/data/`) are in the data archive (DOI in the paper).

    cd figures && python make_figures.py && pdflatex fig1_pipeline.tex

## 5. Data archive

The data archive contains, per output folder: `results.csv`,
`selection_baselines.csv`, `regime_diagnostics.csv`, `paper_data/`, and per run
`row.json` and the step-2 embedding Hamiltonian (`step2_hamiltonian.pkl`), so
the classical baselines and diagnostics can be recomputed without rerunning
DMET or GENIE. GQE training outputs (`outputs/`, ~120 GB) are not archived.
