#!/usr/bin/env python3
"""Verify the release EFI against release/provenance.json; report the manifest as a warning.

Pass/fail, from release/provenance.json:
  * release/EFI/BOOT/BOOTX64.EFI has the recorded efi_sha256 (and efi_bytes, when given);
  * every corresponding_sources entry is a safe relative path to a regular file with the
    recorded SHA-256. These are the sources the binary was built from.

Warning only, never affects the exit code:
  * SOURCE-MANIFEST.json is upstream's published inventory and is kept as published. If
    any listed file differs or is missing it is reported as STALE (an upstream inventory
    that predates later upstream commits), with counts; use verify_manifest.py for the list.

Pure standard library. Run isolated:  python -I scripts/verify_release.py -r <tree>

Exit codes:
  0  EFI and all corresponding sources match provenance (manifest may still be STALE)
  1  EFI or a corresponding source is missing, unsafe or does not match
  2  usage / environment error (provenance missing or malformed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

PROVENANCE_DEFAULT = "release/provenance.json"
EFI_DEFAULT = "release/EFI/BOOT/BOOTX64.EFI"
MANIFEST_DEFAULT = "SOURCE-MANIFEST.json"
CHUNK = 1024 * 1024
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _die(msg: str) -> NoReturn:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root: Path, rel: str) -> Path | None:
    """The file for a repository-relative POSIX path, or None if the path is unsafe.

    Refuses absolute paths, drive letters, backslashes, '..', symlinks and anything that
    resolves outside the root.
    """
    if not rel or "\\" in rel or ":" in rel:
        return None
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts:
        return None
    path = root.joinpath(*pure.parts)
    if path.is_symlink():
        return None
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return path


def load_provenance(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _die(f"error: provenance not found: {path}")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _die(f"error: cannot read provenance {path}: {exc}")
    if not isinstance(data, dict):
        _die(f"error: provenance {path} is not a JSON object")
    efi = str(data.get("efi_sha256", "")).lower()
    sources = data.get("corresponding_sources")
    if not _HEX64.match(efi):
        _die("error: provenance has no valid efi_sha256")
    if not isinstance(sources, list) or not sources:
        _die("error: provenance has no corresponding_sources")
    for i, entry in enumerate(sources):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            _die(f"error: corresponding_sources[{i}] has no path")
        if not _HEX64.match(str(entry.get("sha256", "")).lower()):
            _die(f"error: corresponding_sources[{i}] has no valid sha256")
    return data


def check_efi(root: Path, efi_rel: str, prov: dict[str, Any]) -> dict[str, Any]:
    want = str(prov["efi_sha256"]).lower()
    want_bytes = prov.get("efi_bytes")
    out: dict[str, Any] = {
        "path": efi_rel,
        "expected_sha256": want,
        "expected_bytes": want_bytes,
    }
    path = safe_path(root, efi_rel)
    if path is None or not path.is_file():
        out["ok"] = False
        out["problem"] = "missing or unsafe"
        return out
    out["sha256"] = sha256_file(path)
    out["bytes"] = path.stat().st_size
    out["ok"] = out["sha256"] == want and (
        want_bytes is None or out["bytes"] == want_bytes
    )
    return out


def check_sources(root: Path, prov: dict[str, Any]) -> dict[str, Any]:
    mismatched: list[str] = []
    missing: list[str] = []
    unsafe: list[str] = []
    for entry in prov["corresponding_sources"]:
        rel = entry["path"]
        path = safe_path(root, rel)
        if path is None:
            unsafe.append(rel)
        elif not path.is_file():
            missing.append(rel)
        elif sha256_file(path) != str(entry["sha256"]).lower():
            mismatched.append(rel)
    total = len(prov["corresponding_sources"])
    bad = len(mismatched) + len(missing) + len(unsafe)
    return {
        "total": total,
        "ok": total - bad,
        "mismatched": mismatched,
        "missing": missing,
        "unsafe": unsafe,
    }


def manifest_status(
    root: Path, manifest_rel: str, prov: dict[str, Any]
) -> dict[str, Any]:
    """CURRENT / STALE / ABSENT / UNREADABLE, plus counts. Informational only."""
    path = safe_path(root, manifest_rel)
    if path is None or not path.is_file():
        return {"status": "ABSENT", "path": manifest_rel}
    try:
        files = json.loads(path.read_text(encoding="utf-8")).get("files")
        entries = {str(e["path"]): str(e["sha256"]).lower() for e in files}
    except (
        OSError,
        UnicodeError,
        ValueError,
        AttributeError,
        KeyError,
        TypeError,
    ) as exc:
        return {"status": "UNREADABLE", "path": manifest_rel, "error": str(exc)}
    differ = missing = 0
    for rel, want in entries.items():
        target = safe_path(root, rel)
        if target is None or not target.is_file():
            missing += 1
        elif sha256_file(target) != want:
            differ += 1
    sources = {
        str(e["path"]): str(e["sha256"]).lower() for e in prov["corresponding_sources"]
    }
    agree = sum(1 for rel, want in sources.items() if entries.get(rel) == want)
    return {
        "status": "STALE" if differ or missing else "CURRENT",
        "path": manifest_rel,
        "listed": len(entries),
        "differ": differ,
        "missing": missing,
        "sources_listed_and_agreeing": agree,
        "sources_total": len(sources),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-r", "--root", default=".", help="source tree root (default: .)"
    )
    parser.add_argument(
        "-p",
        "--provenance",
        default=PROVENANCE_DEFAULT,
        help="provenance path under the root",
    )
    parser.add_argument("--efi", default=EFI_DEFAULT, help="EFI path under the root")
    parser.add_argument(
        "-m",
        "--manifest",
        default=MANIFEST_DEFAULT,
        help="manifest path under the root",
    )
    parser.add_argument(
        "--json", action="store_true", help="print a machine-readable report"
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        _die(f"error: not a directory: {root}")
    prov_path = safe_path(root, args.provenance)
    if prov_path is None:
        _die(f"error: unsafe provenance path: {args.provenance}")
    prov = load_provenance(prov_path)

    efi = check_efi(root, args.efi, prov)
    sources = check_sources(root, prov)
    manifest = manifest_status(root, args.manifest, prov)
    passed = bool(efi["ok"]) and sources["ok"] == sources["total"]
    report = {
        "result": "PASS" if passed else "FAIL",
        "efi": efi,
        "sources": sources,
        "manifest": manifest,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if passed else 1

    print(f"EFI {efi['path']}")
    if "sha256" in efi:
        print(f"    sha256 {efi['sha256']}")
        print(
            f"    expect {efi['expected_sha256']}  {'OK' if efi['ok'] else 'MISMATCH'}"
        )
        print(f"    bytes  {efi['bytes']} (expected {efi['expected_bytes']})")
    else:
        print(f"    {efi['problem']}")
    print(
        f"corresponding sources: {sources['ok']}/{sources['total']} match {args.provenance}"
    )
    for label in ("mismatched", "missing", "unsafe"):
        for rel in sources[label]:
            print(f"    {label.upper()}: {rel}")
    status = manifest["status"]
    if status == "STALE":
        print(
            f"WARNING: {manifest['path']} is STALE: {manifest['differ']} of {manifest['listed']} listed files "
            f"differ, {manifest['missing']} missing (upstream inventory predates the tree; "
            "see verify_manifest.py). Not part of the pass/fail result."
        )
        print(
            f"    {manifest['sources_listed_and_agreeing']}/{manifest['sources_total']} corresponding sources "
            "are listed in it with the same hash"
        )
    elif status == "CURRENT":
        print(f"{manifest['path']}: current ({manifest['listed']} files match)")
    else:
        print(
            f"WARNING: {manifest['path']} is {status}. Not part of the pass/fail result."
        )
    print(f"\nRESULT: {report['result']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
