# Publishing this as its own repository

One-time setup. Everything below runs in the directory containing this
file.

## 1. Create the repository on GitHub

Empty — no README, no .gitignore, no licence. Those are already here.

## 2. Initialise and push

```bash
git init
git branch -M main

git add .
git status --short | head -40        # sanity check before the first commit
git commit -m "QuEnAIS 0.2.0: DMET embedding with Qiskit and CUDA-Q solvers"

git remote add origin https://github.com/<you>/quenais.git
git push -u origin main
```

### Check the golden data actually got committed

`.gitignore` has a blanket `*.pkl` with a negation for the reference
pickles. Confirm the negation worked — without them the test suite has
nothing to compare against:

```bash
git ls-files tests/regression/golden | wc -l      # expect 15
```

If that prints 0, force them in:

```bash
git add -f tests/regression/golden
git commit -m "add validated reference pickles"
```

## 3. Add the GQE submodule

`.gitmodules` already declares it. Register the pinned commit:

```bash
git submodule add -f https://github.com/moken20/gqe-for-qsci.git gqe-for-qsci
cd gqe-for-qsci
git checkout 0a201ea
cd ..
git add gqe-for-qsci
git commit -m "pin gqe-for-qsci at 0a201ea"
git push
```

**Note:** `0a201ea` is upstream `main` as of August 2026 and clones cleanly.

An earlier revision of this repo pinned `732c1ea`, a local commit that never
existed on the public upstream. Every fresh clone failed to resolve it while
`.gitmodules` looked perfectly correct, and `git submodule status` printed
nothing at all. If you re-pin, verify the base first:

```bash
git -C gqe-for-qsci apply --check ../patches/gqe_dmet_source.patch
```

and update `UPSTREAM_SHA` in `quenais/quantum/gqe_setup.py` in the same commit
— a test asserts the two agree. The patch and the shim modules do not care
where the checkout came from, only that it is at the right commit.

## 4. Verify a clean clone works

The real test. Clone somewhere else and run it as a new user would:

```bash
cd /tmp
git clone --recurse-submodules https://github.com/<you>/quenais.git
cd quenais

mamba env create -f environment.yml -p ./quenais-env
mamba activate ./quenais-env
bash install.sh
```

`install.sh` ends with `quenais-selftest` and exits non-zero if the
physics does not reproduce, so a silent bad install is not possible.

## 5. Before making it public

- [ ] Read `gqe-for-qsci/NOTICE` for attribution terms that carry into
      derived work. `patches/gqe_dmet_source.patch` redistributes upstream
      context lines.
- [ ] Run `quenais-run --solver gqe` end to end at least once. Every piece
      is tested, but that full path is the one unexercised link.
- [ ] Decide whether `error_logs.txt` and any run outputs belong in the
      repo. They are currently ignored.
- [ ] Update the repository URL in `README.md` (it still points at the old
      one).

## Working across two machines

The workflow that caused trouble during development, and how to avoid it:

- **One machine writes, the others pull.** Generate files in exactly one
  place.
- **Never `git add .` next to a submodule.** It sweeps in submodule
  pointer moves you did not intend. Name the path: `git add quenais/`.
- **`git diff` in a submodule reports only uncommitted changes.** If you
  regenerate the patch after committing inside the submodule, hunks vanish
  silently. `quenais-gqe-setup --create-patch` now refuses to write an
  incomplete patch, but the general lesson stands.
- **A clone that never runs the pipeline does not need the submodule:**
  `git config fetch.recurseSubmodules no`.
