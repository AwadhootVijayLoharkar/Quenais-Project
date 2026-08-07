"""
GQE-for-QSCI settings.

These map onto keys in the external gqe-for-qsci repo's own Hydra configs
(configs/default.yaml + configs/trainer/default.yaml). Set values here
rather than editing those YAML files: anything left as None is skipped, so
the external repo's own default applies unchanged.

Nothing in this module imports CUDA-Q, torch, tequila or hydra. It builds
command-line strings; the subprocess does the rest.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

__all__ = [
    "GqeSettings",
    "OPERATOR_POOL_SPECS",
    "DMET_POOL_SPECS",
    "CUDAQ_SIMULATOR_TARGETS",
]

#: Pool specs registered by the patched factory.py.
#: The first two are upstream's, and rebuild the molecule from its geometry
#: to derive CCSD amplitudes. A DMET embedding has no geometry -- it is
#: h1e/h2e/ecore in an abstract orbital basis -- so those crash with
#: "TypeError: 'NoneType' object is not iterable". Only the dmet_* pools,
#: which take amplitudes from the embedding's own CCSD solve, are usable here.
OPERATOR_POOL_SPECS = (
    "pauli_evolution", "excitation",
    "dmet_pauli_evolution", "dmet_excitation",
)
DMET_POOL_SPECS = ("dmet_pauli_evolution", "dmet_excitation")

#: Classical simulator targets. Hardware targets additionally require
#: shot-based energy evaluation, which is not implemented.
CUDAQ_SIMULATOR_TARGETS = ("qpp-cpu", "nvidia", "tensornet", "tensornet-mps")


def _fmt(value) -> str:
    """
    Format a Python value as a Hydra override literal.

    Hydra's override grammar is not JSON. Lists must be written [a,b] --
    json.dumps would emit ["a", "b"] with double quotes and spaces, which
    Hydra rejects. Booleans must be lowercase.
    """
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(v) for v in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


@dataclass
class GqeSettings:
    """Training hyperparameters and external-repo wiring for the GQE solver."""

    # ── External repo ────────────────────────────────────────────────────
    #: Path to the gqe-for-qsci checkout. Falls back to $GQE_QSCI_REPO_PATH.
    repo_path: str | None = None
    train_entrypoint: str = "train.py"

    #: Which configs/molecule/<name>.yaml train.py loads.
    #:
    #: MANDATORY and easy to get wrong. configs/default.yaml declares
    #: `defaults: - molecule: n2`, so without this override train.py loads
    #: N2 and never reads the DMET config. That failure is silent: the run
    #: succeeds, and visualization reports (embedding CASCI) + (N2's
    #: convergence error), which looks plausible and means nothing.
    molecule_config: str = "dmet_embedding"

    # ── Trainer (configs/trainer/default.yaml -- needs "trainer." prefix) ─
    seed: int | None = None
    max_iters: int | None = 120
    num_samples: int | None = 100
    batch_size: int | None = 100
    step_per_epoch: int | None = None
    warmup_size: int | None = 100
    buffer_size: int | None = 100

    #: True resumes from whatever checkpoint dir the repo finds under
    #: outputs/, which is not scoped per molecule -- it will happily resume
    #: another system's run. Keep False unless resuming deliberately.
    load_checkpoint: bool | None = False
    checkpoint_every_n_iters: int | None = None

    optimizer_lr: float | None = None
    optimizer_cls: str | None = None
    optimizer_weight_decay: float | None = None

    loss_type: str | None = None
    loss_clip_grpo_low: float | None = None
    loss_clip_grpo_high: float | None = None

    temp_sched_initial: float | None = None
    temp_sched_delta: float | None = None
    temp_sched_target_var: float | None = None

    # ── Top level (configs/default.yaml -- no prefix) ─────────────────────
    #: Circuit depth. Larger embeddings need more: on ScH, 10 stalled at HF,
    #: 20 plateaued, 40 recovered 60% of the correlation energy.
    ngates: int | None = 40

    #: Including "R-CASCI" triggers a full FCI over the whole embedding space
    #: purely for logging, before training starts. Intractable much above
    #: 12-16 embedding orbitals. None leaves the repo default.
    reference_keys: list | None = None

    sampler_mpi: bool | None = None
    sampler_shots: int | None = None

    #: Must be a DMET-aware pool. "dmet_excitation" accumulates the Pauli
    #: terms of each excitation generator into one operator, so the pool
    #: element conserves particle number. "dmet_pauli_evolution" appends
    #: each term separately, which cannot conserve it -- number conservation
    #: is a property of the full sum, never of a single Pauli word.
    operator_pool_spec: str = "dmet_excitation"
    operator_pool_ccsd_threshold: float | None = None

    #: The Jordan-Wigner Z-ladder encodes fermionic anticommutation.
    #: Removing it makes exp(i*theta*P) no longer particle-number conserving,
    #: so sampled states leak into the wrong electron-number sector.
    #: Note: ignored by DMETExcitationPool, which factory.py constructs
    #: without these two flags.
    operator_pool_remove_z_ladder: bool | None = False
    operator_pool_only_first_pauli: bool | None = False

    #: QSCI subspace cap. The repo default of 2000 became the binding
    #: constraint on ScH -- the subspace pinned at exactly 2000 from epoch 30.
    qsci_max_dim: int | None = 10000
    qsci_enlarge_method: str | None = None
    qsci_max_cycle: int | None = None

    # ── Backend ──────────────────────────────────────────────────────────
    #: Applied via the CUDAQ_DEFAULT_SIMULATOR environment variable, not a
    #: Hydra key: the external repo never calls cudaq.set_target().
    #:
    #: Defaults to whatever CUDAQ_DEFAULT_SIMULATOR already says, which
    #: install.sh sets from the GPU's compute capability ("nvidia" at
    #: cc>=8.0, "qpp-cpu" otherwise) and persists into the environment.
    #: Falling back to a hardcoded "qpp-cpu" meant build_env() overwrote
    #: that detection on every run, so `quenais-run --solver gqe` never
    #: used the GPU simulator no matter the hardware.
    #:
    #: An explicit value still wins -- this is a default, not an override.
    cudaq_target: str = field(
        default_factory=lambda: os.environ.get(
            "CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu")
    )

    #: Extra raw Hydra overrides, appended last so they win.
    extra_overrides: list = field(default_factory=list)

    # ── Derived ──────────────────────────────────────────────────────────
    def resolved_repo_path(self) -> str | None:
        return self.repo_path or os.environ.get("GQE_QSCI_REPO_PATH")

    def _override_map(self) -> dict:
        """
        Hydra key -> value, in emission order.

        KEY PREFIXES MATTER. The repo splits its config across two files and
        Hydra addresses each by its config-group path:
          configs/trainer/default.yaml -> "trainer." prefix
          configs/default.yaml         -> no prefix
        Getting it wrong fails loudly ("Key 'X' is not in struct") rather
        than silently ignoring the override, which is the good kind of
        failure.
        """
        return {
            # Config-GROUP selection, so no "+" prefix: the group already
            # exists in the defaults list.
            "molecule": self.molecule_config,
            "trainer.seed": self.seed,
            "trainer.max_iters": self.max_iters,
            "trainer.num_samples": self.num_samples,
            "trainer.batch_size": self.batch_size,
            "trainer.step_per_epoch": self.step_per_epoch,
            "trainer.warmup_size": self.warmup_size,
            "trainer.buffer_size": self.buffer_size,
            "trainer.load_checkpoint": self.load_checkpoint,
            "trainer.checkpoint_every_n_iters": self.checkpoint_every_n_iters,
            "trainer.optimizer.lr": self.optimizer_lr,
            "trainer.optimizer.cls": self.optimizer_cls,
            "trainer.optimizer.weight_decay": self.optimizer_weight_decay,
            "trainer.loss.type": self.loss_type,
            "trainer.loss.clip_grpo_low": self.loss_clip_grpo_low,
            "trainer.loss.clip_grpo_high": self.loss_clip_grpo_high,
            "trainer.temperature_scheduler.initial": self.temp_sched_initial,
            "trainer.temperature_scheduler.delta": self.temp_sched_delta,
            "trainer.temperature_scheduler.target_var": self.temp_sched_target_var,
            "ngates": self.ngates,
            "reference_keys": self.reference_keys,
            "sampler.mpi": self.sampler_mpi,
            "sampler.shots": self.sampler_shots,
            "operator_pool.spec": self.operator_pool_spec,
            "operator_pool.ccsd_threshold": self.operator_pool_ccsd_threshold,
            "operator_pool.remove_z_ladder": self.operator_pool_remove_z_ladder,
            "operator_pool.only_use_first_pauli": self.operator_pool_only_first_pauli,
            "qsci.max_dim": self.qsci_max_dim,
            "qsci.enlarge_method": self.qsci_enlarge_method,
            "qsci.max_cycle": self.qsci_max_cycle,
        }

    def hydra_overrides(self, step2_pickle_path: str | None = None) -> list:
        """
        Build the Hydra CLI override list. Fields left as None are skipped.

        cudaq_target is deliberately absent -- it is applied through the
        CUDAQ_DEFAULT_SIMULATOR environment variable instead, because the
        external repo has no config key for it.

        Passing step2_pickle_path adds molecule.step2_pickle_path, which
        overrides the path baked into dmet_embedding.yaml. Always pass it:
        a stale hardcoded path there once produced plausible but meaningless
        results for an extended period.
        """
        overrides = [f"{key}={_fmt(val)}"
                     for key, val in self._override_map().items()
                     if val is not None]
        if step2_pickle_path is not None:
            overrides.append(
                f"molecule.step2_pickle_path={os.path.abspath(step2_pickle_path)}"
            )
        overrides.extend(self.extra_overrides)
        return overrides

    def env_overlay(self) -> dict:
        """Environment variables for the training subprocess."""
        return {"CUDAQ_DEFAULT_SIMULATOR": self.cudaq_target}

    def validate(self) -> "GqeSettings":
        if self.operator_pool_spec not in OPERATOR_POOL_SPECS:
            raise ValueError(
                f"gqe.operator_pool_spec must be one of {OPERATOR_POOL_SPECS}, "
                f"got {self.operator_pool_spec!r}"
            )
        if self.operator_pool_spec not in DMET_POOL_SPECS:
            raise ValueError(
                f"gqe.operator_pool_spec={self.operator_pool_spec!r} is an "
                f"upstream geometry-based pool. It rebuilds the molecule from "
                f"its geometry to derive CCSD amplitudes, and a DMET embedding "
                f"has no geometry, so it fails with "
                f"\"TypeError: 'NoneType' object is not iterable\". "
                f"Use one of {DMET_POOL_SPECS}."
            )
        if not self.molecule_config:
            raise ValueError(
                "gqe.molecule_config must be set (normally 'dmet_embedding'). "
                "Without it train.py silently loads configs/molecule/n2.yaml."
            )
        if self.cudaq_target not in CUDAQ_SIMULATOR_TARGETS:
            raise ValueError(
                f"gqe.cudaq_target={self.cudaq_target!r} is not a classical "
                f"simulator target. Hardware targets need provider credentials "
                f"and shot-based energy evaluation, which is not implemented. "
                f"Choose one of {CUDAQ_SIMULATOR_TARGETS}."
            )
        if self.batch_size is not None and self.num_samples is not None:
            if self.batch_size != self.num_samples:
                raise ValueError(
                    f"gqe.batch_size ({self.batch_size}) should equal "
                    f"gqe.num_samples ({self.num_samples}) for online training."
                )
        for name in ("max_iters", "num_samples", "batch_size", "warmup_size",
                     "buffer_size", "ngates", "qsci_max_dim", "sampler_shots",
                     "step_per_epoch", "checkpoint_every_n_iters", "qsci_max_cycle"):
            val = getattr(self, name)
            if val is not None and val <= 0:
                raise ValueError(f"gqe.{name} must be > 0, got {val}")
        return self
