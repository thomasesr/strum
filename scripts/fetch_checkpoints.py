#!/usr/bin/env python3
"""Download STRUM inference weights from the Hugging Face Hub.

The Hub repo groups files into `drums/`, `drums_classifier_ensemble/`,
`guitar/` and `section_classifier/`, but the pipeline loads them from flat
paths under `checkpoints/`. A plain `huggingface-cli download ... --local-dir
checkpoints/` therefore produces a tree the pipeline cannot read.

This module downloads each file and puts it where the loaders look. The path
mapping is imported from `push_to_hf.py` rather than restated, so upload and
download cannot drift apart.

Weights are never baked into the Docker image. They are fetched on first run
into a mounted volume and skipped on every later start, which is why the
functions here are importable: the web app calls them at startup.

Usage:

    python scripts/fetch_checkpoints.py              # fetch what is missing
    python scripts/fetch_checkpoints.py --force      # re-fetch everything
    python scripts/fetch_checkpoints.py --only drums_v14 --only fret_mapper_v4
    python scripts/fetch_checkpoints.py --list       # show the plan, download nothing
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from push_to_hf import PLAN, REPO_ID, ROOT  # noqa: E402

logger = logging.getLogger(__name__)

# PLAN's local paths are absolute under the source tree. A destination other
# than ROOT/checkpoints -- a mounted volume, say -- is applied by rebasing them.
DEFAULT_DEST = Path(os.environ.get("STRUM_CHECKPOINT_DIR") or (ROOT / "checkpoints"))
PLAN_ROOT = ROOT / "checkpoints"

# Benchmark and paper artifacts are not needed to chart a song.
ARTIFACT_GROUPS = {
    "paper_manifest_v4", "paper_candidates_strict",
    "paper_envelope_features", "benchmark_results",
}

Entry = tuple[str, str, Path]


def resolve_plan(
    dest: Path = DEFAULT_DEST,
    only: list[str] | None = None,
    artifacts: bool = False,
) -> list[Entry]:
    """Return (group, remote, local) triples with destinations rebased onto `dest`.

    Paper artifacts live outside checkpoints/ and keep their original location;
    they are excluded altogether unless asked for.
    """
    entries = [
        (group, remote,
         dest / local.relative_to(PLAN_ROOT) if local.is_relative_to(PLAN_ROOT) else local)
        for group, remote, local in PLAN
    ]
    if not artifacts:
        entries = [e for e in entries if e[0] not in ARTIFACT_GROUPS]
    if only:
        wanted = set(only)
        entries = [e for e in entries if e[0] in wanted]
    return entries


def missing(plan: list[Entry]) -> list[Entry]:
    """The subset of `plan` not yet on disk."""
    return [e for e in plan if not e[2].exists()]


def fetch_plan(
    plan: list[Entry],
    repo_id: str = REPO_ID,
    revision: str | None = None,
    force: bool = False,
    on_progress=None,
) -> tuple[int, int, list[str]]:
    """Download every entry in `plan`. Returns (fetched, skipped, failed groups).

    `on_progress(group, done, total)` runs after each entry, so callers can
    report progress; the web UI uses it to show what is still arriving.

    Individual failures are collected rather than raised, so one unreachable
    file cannot abandon the rest of the download.
    """
    from huggingface_hub import hf_hub_download

    fetched = skipped = 0
    failed: list[str] = []
    total = len(plan)

    for index, (group, remote, local) in enumerate(plan, start=1):
        if local.exists() and not force:
            skipped += 1
        else:
            try:
                cached = hf_hub_download(
                    repo_id=repo_id, filename=remote, revision=revision
                )
                local.parent.mkdir(parents=True, exist_ok=True)
                # Copy rather than symlink: checkpoints/ is a mounted volume,
                # where a link into the host's HF cache would dangle.
                shutil.copyfile(cached, local)
                fetched += 1
            except Exception as e:
                logger.warning(f"{group}: {e}")
                failed.append(group)
        if on_progress is not None:
            on_progress(group, index, total)

    return fetched, skipped, failed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"where to place checkpoints (default: {DEFAULT_DEST})")
    ap.add_argument("--revision", default=None, help="Hub revision (branch, tag or commit)")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--only", action="append", default=[], metavar="GROUP",
                    help="fetch only these groups; repeatable")
    ap.add_argument("--artifacts", action="store_true",
                    help="also fetch paper/benchmark artifacts (not needed for charting)")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    if args.only:
        unknown = set(args.only) - {g for g, _, _ in PLAN}
        if unknown:
            print(f"Unknown group(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"Available: {', '.join(g for g, _, _ in PLAN)}", file=sys.stderr)
            return 2

    plan = resolve_plan(args.dest, args.only or None, args.artifacts)

    if args.list:
        print(f"repo: {args.repo_id}")
        for group, remote, local in plan:
            state = "present" if local.exists() else "missing"
            print(f"  {group:34} {remote:62} -> {local}  [{state}]")
        return 0

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub is not installed. Try: pip install huggingface_hub",
              file=sys.stderr)
        return 1

    todo = len(plan) if args.force else len(missing(plan))
    print(f"repo: {args.repo_id} -> {args.dest}")
    print(f"{todo} of {len(plan)} file(s) to download")

    def report(group: str, done: int, total: int) -> None:
        print(f"  [{done}/{total}] {group}", flush=True)

    fetched, skipped, failed = fetch_plan(
        plan, repo_id=args.repo_id, revision=args.revision,
        force=args.force, on_progress=report,
    )

    print(f"\n{fetched} downloaded, {skipped} already present, {len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
