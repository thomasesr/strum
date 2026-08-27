#!/usr/bin/env python3
"""Download STRUM inference weights from the Hugging Face Hub.

The Hub repo groups files into `drums/`, `drums_classifier_ensemble/`,
`guitar/` and `section_classifier/`, but the pipeline loads them from flat
paths under `checkpoints/`. A plain `huggingface-cli download ... --local-dir
checkpoints/` therefore produces a tree the pipeline cannot read.

This script downloads each file and puts it where the loaders look. The path
mapping is imported from `push_to_hf.py` rather than restated, so upload and
download cannot drift apart.

Usage:

    python scripts/fetch_checkpoints.py              # fetch what is missing
    python scripts/fetch_checkpoints.py --force      # re-fetch everything
    python scripts/fetch_checkpoints.py --only drums_v14 --only fret_mapper_v4
    python scripts/fetch_checkpoints.py --list       # show the plan, download nothing
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from push_to_hf import PLAN, REPO_ID  # noqa: E402

# Benchmark and paper artifacts are not needed to chart a song.
ARTIFACT_GROUPS = {
    "paper_manifest_v4", "paper_candidates_strict",
    "paper_envelope_features", "benchmark_results",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--revision", default=None, help="Hub revision (branch, tag or commit)")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--only", action="append", default=[], metavar="GROUP",
                    help="fetch only these groups; repeatable")
    ap.add_argument("--artifacts", action="store_true",
                    help="also fetch paper/benchmark artifacts (not needed for charting)")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    plan = list(PLAN)
    if not args.artifacts:
        plan = [e for e in plan if e[0] not in ARTIFACT_GROUPS]
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {g for g, _, _ in PLAN}
        if unknown:
            print(f"Unknown group(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"Available: {', '.join(g for g, _, _ in PLAN)}", file=sys.stderr)
            return 2
        plan = [e for e in plan if e[0] in wanted]

    if args.list:
        print(f"repo: {args.repo_id}")
        for group, remote, local in plan:
            state = "present" if local.exists() else "missing"
            print(f"  {group:34} {remote:62} -> {local}  [{state}]")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed. Try: pip install huggingface_hub",
              file=sys.stderr)
        return 1

    fetched = skipped = 0
    failed: list[str] = []
    for group, remote, local in plan:
        if local.exists() and not args.force:
            print(f"  = {group}  (already at {local})")
            skipped += 1
            continue
        print(f"  ↓ {group}  {remote}")
        try:
            cached = hf_hub_download(
                repo_id=args.repo_id, filename=remote, revision=args.revision
            )
        except Exception as e:
            print(f"    ! failed: {e}", file=sys.stderr)
            failed.append(group)
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        # Copy rather than symlink: the checkpoints directory is bind-mounted
        # into the container, where a link into the host's HF cache would dangle.
        shutil.copyfile(cached, local)
        fetched += 1

    print(f"\n{fetched} downloaded, {skipped} already present, {len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
