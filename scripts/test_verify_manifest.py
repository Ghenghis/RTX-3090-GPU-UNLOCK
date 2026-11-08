"""Tests for the manifest verifier (pure standard library, temp trees only)."""

from __future__ import annotations

import hashlib
import json

import verify_manifest as vm


def _tree(tmp_path, content=b"hello\n"):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.txt").write_bytes(content)
    entry = {
        "path": "a.txt",
        "bytes": 6,
        "sha256": hashlib.sha256(b"hello\n").hexdigest(),
    }
    (root / "SOURCE-MANIFEST.json").write_text(
        json.dumps({"files": [entry]}), encoding="utf-8"
    )
    return root


def test_pass_and_mismatch(tmp_path):
    root = _tree(tmp_path)
    assert vm.main(["-r", str(root), "-q"]) == 0
    (root / "a.txt").write_bytes(b"changed\n")
    assert vm.main(["-r", str(root), "-q"]) == 1


def test_missing_file_fails(tmp_path):
    root = _tree(tmp_path)
    (root / "a.txt").unlink()
    assert vm.main(["-r", str(root), "-q"]) == 1


def test_manifest_path_relative_to_root_or_cwd(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    # both spellings name the same manifest
    assert (
        vm.resolve_manifest("SOURCE-MANIFEST.json", root)
        == root / "SOURCE-MANIFEST.json"
    )
    assert (
        vm.resolve_manifest("tree/SOURCE-MANIFEST.json", root)
        == (root / "SOURCE-MANIFEST.json").resolve()
    )
    assert vm.main(["-m", "tree/SOURCE-MANIFEST.json", "-r", "tree", "-q"]) == 0


def test_absent_manifest_is_usage_error(tmp_path):
    try:
        vm.main(["-r", str(tmp_path), "-q"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected exit 2")
