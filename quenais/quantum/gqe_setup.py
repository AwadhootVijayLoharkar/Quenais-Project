"""
Prepare the external gqe-for-qsci checkout: apply the source patch and
verify the result.

WHY THIS IS A SEPARATE COMMAND, NOT A pip HOOK
----------------------------------------------
Patching a git submodule during `pip install` fails in wheel builds, in CI
without submodules, and on every editable reinstall -- and a half-applied
patch is exactly the silent-wrong-value failure this project keeps hitting.
So it is an explicit, checked, idempotent step:

    quenais-gqe-setup --repo /path/to/gqe-for-qsci

WHAT THE PATCH CONTAINS (three files, deliberately not four)
------------------------------------------------------------
  gqe_qsci/factory.py        registers the dmet_* operator pools. Without
                             it the pool specs are unregistered and the run
                             fails at pool construction.

  gqe_qsci/gqe/sampler.py    makes the sampler.mpi config flag
                             authoritative. Upstream takes the MPI branch
                             whenever MPI happens to be initialised by
                             anything in the process, ignoring the flag.
                             Affects work distribution only -- not circuit
                             sampling, not the QSCI subspace.

  gqe_qsci/train_pipeline.py prints the per-epoch metrics dict. Pure
                             instrumentation, but LOAD-BEARING: these
                             "[epoch N] {...}" lines are what the
                             visualisation stage parses. Without it,
                             training runs fine and produces nothing
                             plottable.

  configs/trainer/default.yaml  NOT SHIPPED. Those were leftover debugging
                             values (max_iters 100->50, load_checkpoint
                             true->false, target_var 1e-5->1e-3). Baking
                             them into the fork would silently override the
                             package's own settings, which are passed as
                             Hydra overrides.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

__all__ = ["main", "verify_gqe_repo", "PATCHED_FILES", "UPSTREAM_SHA"]

#: Upstream commit the patch was written against.
UPSTREAM_SHA = "0a201ea"

#: Files the patch touches. configs/trainer/default.yaml is deliberately absent.
PATCHED_FILES = (
    "gqe_qsci/factory.py",
    "gqe_qsci/gqe/sampler.py",
    "gqe_qsci/train_pipeline.py",
)

#: Written into the repo after a successful apply, holding the patch hash.
STAMP_NAME = ".quenais_patch_applied"

#: Shipped patch location.
PATCH_PATH = Path(__file__).resolve().parents[2] / "patches" / "gqe_dmet_source.patch"


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True, timeout=30)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_repo(repo=None, cfg=None):
    """Repo path from the argument, the config, or $GQE_QSCI_REPO_PATH."""
    candidate = repo
    if candidate is None and cfg is not None:
        candidate = getattr(getattr(cfg, "gqe", None), "repo_path", None)
    candidate = candidate or os.environ.get("GQE_QSCI_REPO_PATH")
    if not candidate:
        raise FileNotFoundError(
            "The gqe-for-qsci checkout was not found. Pass --repo, set "
            "cfg.gqe.repo_path, or export GQE_QSCI_REPO_PATH."
        )
    path = Path(candidate).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Not a directory: {path}")
    if not (path / "train.py").exists():
        raise FileNotFoundError(
            f"{path} does not look like a gqe-for-qsci checkout "
            f"(no train.py). Did the submodule get initialised? Try "
            f"`git submodule update --init --recursive`."
        )
    return path


def verify_gqe_repo(repo, patch_path=None, strict=False):
    """
    Check the checkout is ready to run. Returns a list of problems; empty
    means good.

    Called before launching training, so a run that could only produce an
    unparseable log fails in a second rather than after hours of GPU time.
    """
    problems = []
    repo = Path(repo)
    patch_path = Path(patch_path or PATCH_PATH)

    head = _git(["rev-parse", "--short=7", "HEAD"], cwd=repo)
    if head.returncode != 0:
        problems.append(f"not a git checkout: {repo}")
    elif not head.stdout.strip().startswith(UPSTREAM_SHA[:7]):
        problems.append(
            f"submodule HEAD is {head.stdout.strip()}, expected "
            f"{UPSTREAM_SHA}. The patch was written against {UPSTREAM_SHA}; "
            f"applying it elsewhere may fail or misapply."
        )

    stamp = repo / STAMP_NAME
    if not stamp.exists():
        problems.append(
            f"the patch has not been applied ({STAMP_NAME} is missing). "
            f"Run: quenais-gqe-setup --repo {repo}"
        )
    elif patch_path.exists():
        recorded = stamp.read_text().strip()
        current = _sha256(patch_path)
        if recorded != current:
            problems.append(
                f"the applied patch does not match the shipped one "
                f"(stamp {recorded[:12]}, shipped {current[:12]}). The "
                f"submodule was probably updated. Re-run quenais-gqe-setup."
            )

    for name in PATCHED_FILES:
        if not (repo / name).exists():
            problems.append(f"missing expected file: {name}")

    if strict and problems:
        raise RuntimeError(
            "The gqe-for-qsci checkout is not ready:\n  - "
            + "\n  - ".join(problems)
        )
    return problems


def create_patch(repo, out_path=None):
    """
    Generate the patch from a working checkout.

    Run this once on a machine where the integration is already working;
    commit the result. Only the three source files are included --
    configs/trainer/default.yaml is excluded on purpose (see the module
    docstring).
    """
    repo = Path(repo)
    out_path = Path(out_path or PATCH_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = _git(["diff", "--", *PATCHED_FILES], cwd=repo)
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed: {result.stderr}")
    if not result.stdout.strip():
        raise RuntimeError(
            f"git diff produced nothing for {list(PATCHED_FILES)} in {repo}. "
            f"Either the changes are already committed in the submodule, or "
            f"this checkout does not carry them."
        )

    # VALIDATE BEFORE WRITING. git diff reports only UNCOMMITTED changes, so
    # if any of the three files has been committed inside the submodule it
    # silently drops that hunk -- and a partial patch looks like a perfectly
    # good file. That happened once and produced a patch with zero hunks
    # that overwrote the correct one. Refuse rather than write a lie.
    touched = {ln[len("+++ b/"):].strip() for ln in result.stdout.splitlines()
               if ln.startswith("+++ b/")}
    missing = set(PATCHED_FILES) - touched
    if missing:
        raise RuntimeError(
            f"Refusing to write an incomplete patch. git diff produced hunks "
            f"for {sorted(touched) or 'nothing'}, but {sorted(missing)} "
            f"are missing.\n\n"
            f"git diff only reports UNCOMMITTED changes, so this usually "
            f"means those files were committed inside the submodule. Check "
            f"with:\n"
            f"  cd {repo} && git log --oneline -3 && git status\n\n"
            f"The existing patch at {out_path} has NOT been modified."
        )

    out_path.write_text(result.stdout)
    touched = sorted(touched)
    print(f"Wrote {out_path}")
    print(f"  files: {touched}")
    print(f"  sha256: {_sha256(out_path)}")
    return out_path


def apply_patch(repo, patch_path=None, force=False):
    """Apply the patch idempotently. Returns True if the repo ends up patched."""
    repo = Path(repo)
    patch_path = Path(patch_path or PATCH_PATH)

    if not patch_path.exists():
        raise FileNotFoundError(
            f"Patch not found: {patch_path}\n"
            f"Generate it once from a working checkout with:\n"
            f"  quenais-gqe-setup --repo {repo} --create-patch"
        )

    head = _git(["rev-parse", "--short=7", "HEAD"], cwd=repo)
    sha = head.stdout.strip() if head.returncode == 0 else "?"
    if not sha.startswith(UPSTREAM_SHA[:7]) and not force:
        raise RuntimeError(
            f"Submodule HEAD is {sha}, but the patch was written against "
            f"{UPSTREAM_SHA}. Refusing to patch a different commit. Pass "
            f"--force to override."
        )

    # Already applied? Reverse-check succeeds only if the changes are present.
    # --whitespace=nowarn: the upstream sources carry trailing whitespace on
    # the context lines, so the patch does too. That is not an error here.
    apply_flags = ["--whitespace=nowarn"]

    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", *apply_flags, str(patch_path)],
        cwd=repo, capture_output=True, text=True,
    )
    if reverse.returncode == 0:
        print(f"Patch already applied to {repo} -- nothing to do.")
        (repo / STAMP_NAME).write_text(_sha256(patch_path))
        return True

    check = subprocess.run(
        ["git", "apply", "--check", *apply_flags, str(patch_path)],
        cwd=repo, capture_output=True, text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            f"The patch does not apply cleanly to {repo}:\n{check.stderr}\n"
            f"The checkout may be modified or at the wrong commit."
        )

    applied = subprocess.run(
        ["git", "apply", *apply_flags, str(patch_path)],
        cwd=repo, capture_output=True, text=True,
    )
    if applied.returncode != 0:
        raise RuntimeError(
            f"git apply failed after --check passed, which suggests a "
            f"partial application:\n{applied.stderr}"
        )

    (repo / STAMP_NAME).write_text(_sha256(patch_path))
    print(f"Applied {patch_path.name} to {repo}")
    for name in PATCHED_FILES:
        print(f"  patched: {name}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare the external gqe-for-qsci checkout."
    )
    parser.add_argument("--repo", default=None,
                        help="path to the gqe-for-qsci checkout "
                             "(default: $GQE_QSCI_REPO_PATH)")
    parser.add_argument("--patch", default=None,
                        help=f"patch file (default: {PATCH_PATH})")
    parser.add_argument("--create-patch", action="store_true",
                        help="generate the patch from this checkout instead "
                             "of applying it")
    parser.add_argument("--verify-only", action="store_true",
                        help="report readiness without changing anything")
    parser.add_argument("--force", action="store_true",
                        help="patch even if the submodule SHA differs")
    args = parser.parse_args(argv)

    try:
        repo = resolve_repo(args.repo)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"repo: {repo}")

    if args.create_patch:
        create_patch(repo, args.patch)
        return 0

    if not args.verify_only:
        try:
            apply_patch(repo, args.patch, force=args.force)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    problems = verify_gqe_repo(repo, args.patch)
    if problems:
        print("\nNot ready:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nReady: submodule is at the expected commit and patched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
