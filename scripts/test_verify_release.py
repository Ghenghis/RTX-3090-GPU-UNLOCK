"""Tests for the release-provenance check (pure standard library, temp trees only)."""

from __future__ import annotations

import hashlib
import json

import pytest
import verify_release as vr

EFI = b"MZ fake efi image"
SRC = {
    "unlock/app.c": b"int main(void){return 0;}\n",
    "unlock/port.h": b"#pragma once\n",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree(tmp_path, manifest_files=None):
    root = tmp_path / "tree"
    (root / "release/EFI/BOOT").mkdir(parents=True)
    (root / "release/EFI/BOOT/BOOTX64.EFI").write_bytes(EFI)
    for rel, data in SRC.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    prov = {
        "efi_sha256": _sha(EFI),
        "efi_bytes": len(EFI),
        "corresponding_sources": [
            {"path": p, "sha256": _sha(d)} for p, d in SRC.items()
        ],
    }
    (root / "release/provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    files = (
        manifest_files
        if manifest_files is not None
        else {p: _sha(d) for p, d in SRC.items()}
    )
    manifest = {"files": [{"path": p, "sha256": s} for p, s in files.items()]}
    (root / "SOURCE-MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _report(root, capsys):
    code = vr.main(["-r", str(root), "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_pass_with_current_manifest(tmp_path, capsys):
    code, rep = _report(_tree(tmp_path), capsys)
    assert code == 0 and rep["result"] == "PASS"
    assert rep["sources"]["ok"] == 2 and rep["manifest"]["status"] == "CURRENT"


def test_stale_manifest_is_a_warning_not_a_failure(tmp_path, capsys):
    stale = {"unlock/app.c": _sha(b"old version\n"), "docs/gone.md": _sha(b"x")}
    root = _tree(tmp_path, manifest_files=stale)
    code, rep = _report(root, capsys)
    assert code == 0 and rep["result"] == "PASS"
    assert rep["manifest"]["status"] == "STALE"
    assert rep["manifest"]["differ"] == 1 and rep["manifest"]["missing"] == 1
    assert vr.main(["-r", str(root)]) == 0
    assert "STALE" in capsys.readouterr().out


def test_tampered_efi_fails(tmp_path, capsys):
    root = _tree(tmp_path)
    (root / "release/EFI/BOOT/BOOTX64.EFI").write_bytes(EFI + b"!")
    code, rep = _report(root, capsys)
    assert code == 1 and rep["efi"]["ok"] is False


def test_changed_or_missing_source_fails(tmp_path, capsys):
    root = _tree(tmp_path)
    (root / "unlock/app.c").write_bytes(b"changed\n")
    (root / "unlock/port.h").unlink()
    code, rep = _report(root, capsys)
    assert code == 1
    assert rep["sources"]["mismatched"] == ["unlock/app.c"]
    assert rep["sources"]["missing"] == ["unlock/port.h"]


def test_unsafe_source_path_fails(tmp_path, capsys):
    root = _tree(tmp_path)
    prov = json.loads((root / "release/provenance.json").read_text(encoding="utf-8"))
    prov["corresponding_sources"].append({"path": "../outside.c", "sha256": _sha(b"x")})
    (root / "release/provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    code, rep = _report(root, capsys)
    assert code == 1 and rep["sources"]["unsafe"] == ["../outside.c"]


def test_absent_manifest_is_only_a_warning(tmp_path, capsys):
    root = _tree(tmp_path)
    (root / "SOURCE-MANIFEST.json").unlink()
    code, rep = _report(root, capsys)
    assert code == 0 and rep["manifest"]["status"] == "ABSENT"


def test_malformed_provenance_is_usage_error(tmp_path):
    root = _tree(tmp_path)
    (root / "release/provenance.json").write_text(
        json.dumps({"efi_sha256": "nope"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit) as exc:
        vr.main(["-r", str(root)])
    assert exc.value.code == 2


def test_safe_path_rules(tmp_path):
    for bad in ("/etc/passwd", "C:/x", "a\\b", "../x", "a/../../x", ""):
        assert vr.safe_path(tmp_path, bad) is None
    assert (
        vr.safe_path(tmp_path, "release/provenance.json")
        == tmp_path / "release" / "provenance.json"
    )
