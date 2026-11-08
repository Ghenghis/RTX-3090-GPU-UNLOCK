#!/usr/bin/env python3
"""Verify the source tree against SOURCE-MANIFEST.json.

Recomputes the SHA-256 (and size) of every file listed in the manifest and reports
any mismatch, missing file, or — with --strict — any tracked file missing from the
manifest. Exits non-zero on any failure so it can gate CI.

Pure standard library. Run isolated:  python -I scripts/verify_manifest.py

Exit codes:
  0  all listed files present and matching
  1  one or more integrity failures (mismatch / missing / extra under --strict)
  2  usage / environment error (manifest not found, malformed, etc.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

MANIFEST_DEFAULT = "SOURCE-MANIFEST.json"
CHUNK = 1024 * 1024


def _die(msg: str) -> NoReturn:
    """Exit 2 for usage/environment errors, per the documented contract."""
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest(manifest_path: Path) -> list[dict]:
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _die(
            f"error: manifest not found: {manifest_path}\n"
            "  This file ships with the upstream source tree. If your repository is "
            "still empty,\n  import the source first, then run this check."
        )
    except OSError as exc:
        _die(f"error: cannot read manifest {manifest_path}: {exc}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _die(f"error: manifest is not valid JSON: {exc}")

    files = data.get("files")
    if not isinstance(files, list) or not files:
        _die("error: manifest has no non-empty 'files' array")

    for i, entry in enumerate(files):
        if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
            _die(f"error: manifest entry {i} is missing 'path' or 'sha256'")
    return files


def resolve_manifest(manifest: str, root: Path, cwd: Path | None = None) -> Path:
    """Locate the manifest: absolute as given; relative under --root first, else from cwd.

    So `-m SOURCE-MANIFEST.json -r tree` and `-m tree/SOURCE-MANIFEST.json -r tree`
    name the same file.
    """
    path = Path(manifest)
    if path.is_absolute():
        return path
    under_root = root / path
    if under_root.is_file():
        return under_root
    from_cwd = (cwd if cwd is not None else Path.cwd()) / path
    return from_cwd.resolve() if from_cwd.is_file() else under_root


def _git_tracked(root: Path) -> set[str] | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {line.strip() for line in out.splitlines() if line.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--manifest",
        default=MANIFEST_DEFAULT,
        help=f"manifest path (default: {MANIFEST_DEFAULT})",
    )
    parser.add_argument(
        "-r",
        "--root",
        default=".",
        help="repository root the paths are relative to (default: .)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail on git-tracked files absent from the manifest",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="print only the summary and failures"
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    manifest_path = resolve_manifest(args.manifest, root)

    files = _load_manifest(manifest_path)
    manifest_rel = {manifest_path.name}

    ok = 0
    mismatches: list[str] = []
    missing: list[str] = []

    for entry in files:
        rel = entry["path"]
        want = str(entry["sha256"]).lower()
        target = root / rel
        if not target.is_file():
            missing.append(rel)
            continue
        got = _sha256(target)
        if got != want:
            mismatches.append(f"{rel}\n    expected {want}\n    got      {got}")
            if "bytes" in entry:
                actual = target.stat().st_size
                if actual != entry["bytes"]:
                    mismatches[-1] += (
                        f"\n    size expected {entry['bytes']}, got {actual}"
                    )
        else:
            ok += 1
            if not args.quiet:
                print(f"ok    {rel}")

    extra: list[str] = []
    if args.strict:
        tracked = _git_tracked(root)
        if tracked is None:
            print(
                "warning: --strict needs git; skipping extra-file detection",
                file=sys.stderr,
            )
        else:
            listed = {e["path"] for e in files} | manifest_rel
            extra = sorted(f for f in tracked if f not in listed)

    print()
    print(f"verified {ok}/{len(files)} files against {manifest_path.name}")
    for label, items in (
        ("MISMATCH", mismatches),
        ("MISSING", missing),
        ("NOT IN MANIFEST", extra),
    ):
        for item in items:
            print(f"  {label}: {item}")

    failed = bool(mismatches or missing or extra)
    if failed:
        print("\nRESULT: FAIL", file=sys.stderr)
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
