# Contributing

## Before you change anything

Run the tests. They are not decoration — most of them exist because
something silently produced a wrong number.

```bash
pytest -q
```

## The rule that matters

**Every bug this project has had produced output with the right shape, a
plausible magnitude and the wrong value.** None of them crashed. A bath
fabricated from rounding noise, an electron count taken from the wrong
space, a cache entry belonging to a different molecule, a patch file that
lost all its hunks.

So: when you fix something, add a test that fails without the fix. If you
cannot write one, the fix is not understood well enough yet.

Several tests deliberately assert that a *wrong* implementation would give
a different answer — see `test_degenerate_pair_would_be_split_without_the_tolerance`.
Without that, a guard can pass vacuously.

## Reference data

`tests/regression/golden/` holds validated pickles for LiH, N₂ and ScH.
They are tracked in git deliberately; `.gitignore` has a negation rule for
them. Without them the suite has nothing to compare against.

To regenerate, run the pipeline on a machine whose PySCF build you trust
and replace the directory. Note that CASSCF and NEVPT2 will not reproduce
exactly — they are optimiser-dependent. See `docs/limitations.md`.

## Adding a solver

One list and one branch:

1. add the name to `QISKIT_SOLVERS` or `GQE_SOLVERS` in `quenais/config.py`
2. add a branch to `dispatch()` in `quenais/quantum/__init__.py`

`Config.validate()` and the CLI both read from that registry, so they
cannot disagree with each other.

## Import discipline

`quenais.quantum.gqe_*` must stay importable without CUDA-Q installed, and
`quenais.quantum.solver` without Qiskit. CLI dispatch imports both.

Heavy imports go inside functions. A test enforces this by importing every
module in a subprocess with the optional stacks blocked at the import hook
— convention alone decays.

The same applies to NumPy and `quenais/_threads.py`: the thread-count
variables only take effect before NumPy's first import, so `_threads` must
import nothing but `os`.

## Style

`ruff check quenais tools` must pass. Comments should say *why*, not
*what* — the interesting part of this codebase is the reasoning, and most
of it came from debugging sessions that are otherwise lost.
