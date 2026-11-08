"""Unit tests for the post-boot verifier's pure classification logic.

These exercise the real decision functions (no GPU, no torch needed): the card-reference
matcher and the unlocked/stock/inconclusive classifier. Run: pytest tools/
"""

from __future__ import annotations

import json

import gpu_select as gs
import pytest
import verify_unlock as vu


# --- pick_reference: most-specific match wins -----------------------------


def test_ti_not_confused_with_non_ti():
    key, stock = vu.pick_reference("NVIDIA GeForce RTX 3090 Ti")
    assert key == "3090 ti"
    assert stock == vu.STOCK_BF16_TFLOPS["3090 ti"]


def test_plain_3090_matches_3090():
    key, _ = vu.pick_reference("NVIDIA GeForce RTX 3090")
    assert key == "3090"


def test_3060_12gb_phoenix():
    key, stock = vu.pick_reference("NVIDIA GeForce RTX 3060")
    assert key == "3060"
    assert stock == 27.0


def test_unknown_card_returns_none():
    assert vu.pick_reference("NVIDIA GeForce RTX 5090") is None


def test_case_insensitive():
    assert vu.pick_reference("nvidia geforce rtx 4090")[0] == "4090"


# --- classify: ratio thresholds -------------------------------------------


def test_clearly_unlocked():
    # 3090 Ti unlocked ~157 vs stock 87.5 -> ratio ~1.8 -> UNLOCKED (0)
    assert vu.classify(157.0, 87.5) == 0


def test_clearly_stock():
    # measured at stock -> STOCK (1)
    assert vu.classify(87.5, 87.5) == 1


def test_just_above_stock_is_inconclusive():
    # ratio 1.3 is between STOCK_RATIO (1.2) and UNLOCK_RATIO (1.5)
    assert vu.classify(87.5 * 1.3, 87.5) == 2


def test_boundary_unlock_ratio_inclusive():
    assert vu.classify(100.0 * vu.UNLOCK_RATIO, 100.0) == 0


def test_boundary_stock_ratio_inclusive():
    assert vu.classify(100.0 * vu.STOCK_RATIO, 100.0) == 1


def test_zero_or_negative_stock_is_inconclusive():
    assert vu.classify(100.0, 0.0) == 2
    assert vu.classify(100.0, -5.0) == 2


def test_3060_measured_46_is_unlocked():
    # this repo's own H610M/3060 evidence: 27 -> ~46 TFLOPS, ratio ~1.7 -> UNLOCKED
    assert vu.classify(46.3, 27.0) == 0


def test_custom_ratios_respected():
    # with a stricter unlock ratio, the same measurement is no longer "unlocked"
    assert vu.classify(40.0, 27.0, unlock_ratio=1.6, stock_ratio=1.2) == 2


# --- calibration store and reference order ---------------------------------

TI = gs.Gpu(
    index=1,
    uuid="GPU-12345678-90ab-cdef-0123-456789abcdef",
    name="NVIDIA GeForce RTX 3090 Ti",
    pci_bus_id="00000000:0A:00.0",
)
OTHER = gs.Gpu(
    index=0,
    uuid="GPU-11111111-aaaa-bbbb-cccc-000000000001",
    name="NVIDIA GeForce RTX 3060",
    pci_bus_id="00000000:01:00.0",
)


def test_calibration_path_env_override_and_defaults(tmp_path):
    assert (
        vu.default_calibration_path({vu.CALIBRATION_ENV: str(tmp_path / "c.json")})
        == tmp_path / "c.json"
    )
    path = vu.default_calibration_path(
        {"XDG_CONFIG_HOME": str(tmp_path), "APPDATA": str(tmp_path)}
    )
    assert path.name == "calibration.json" and path.parent.name == "rtx-unlock"


def test_missing_calibration_file_means_none(tmp_path):
    assert vu.load_calibration(tmp_path / "absent.json") == {}


def test_corrupt_calibration_file_is_an_error_not_a_fallback(tmp_path):
    bad = tmp_path / "c.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(vu.CalibrationError):
        vu.load_calibration(bad)
    bad.write_text(json.dumps({"schema": "other", "gpus": {}}), encoding="utf-8")
    with pytest.raises(vu.CalibrationError):
        vu.load_calibration(bad)


def test_save_and_load_roundtrip_keyed_by_uuid(tmp_path):
    path = tmp_path / "sub" / "c.json"
    vu.save_calibration(
        path, {gs.norm_uuid(TI.uuid): {"uuid": TI.uuid, "bf16_tflops": 71.2}}
    )
    cal = vu.load_calibration(path)
    assert cal[gs.norm_uuid(TI.uuid)]["bf16_tflops"] == 71.2


def test_reference_order_override_then_calibration_then_table():
    cal = {
        gs.norm_uuid(TI.uuid): {
            "uuid": TI.uuid,
            "bf16_tflops": 71.2,
            "measured_at": "t",
            "driver": "621.14",
        }
    }
    assert vu.reference_for(TI, cal, stock_override=80.0)["source"] == "override"
    ref = vu.reference_for(TI, cal)
    assert ref["source"] == "calibration" and ref["tflops"] == 71.2
    # another card with no calibration falls back to the table
    assert vu.reference_for(OTHER, cal)["source"] == "spec-table"
    assert vu.reference_for(gs.Gpu(0, "GPU-x", "RTX 5090"), cal) is None


def test_calibration_refusal_band():
    assert vu.calibration_refusal(80.0, TI.name) is None
    assert "looks unlocked" in vu.calibration_refusal(150.0, TI.name)
    assert "busy" in vu.calibration_refusal(30.0, TI.name)
    assert vu.calibration_refusal(999.0, "RTX 5090") is None  # no table entry, no band


# --- main(): selection, calibration and later runs (all hardware faked) ----


@pytest.fixture
def fake_hw(monkeypatch, tmp_path):
    """Two cards; measurements and utilisation per UUID are set by each test."""
    state = {
        "tflops": {TI.uuid: 72.0, OTHER.uuid: 27.0},
        "util": {TI.uuid: 0, OTHER.uuid: 0},
    }
    monkeypatch.setattr(vu, "_import_torch", lambda: object())
    monkeypatch.setattr(vu.gpu_select, "inventory", lambda torch=None: [OTHER, TI])
    monkeypatch.setattr(vu.gpu_select, "torch_index", lambda torch, gpu: gpu.index)
    monkeypatch.setattr(
        vu.gpu_select, "utilization_pct", lambda gpu: state["util"][gpu.uuid]
    )
    monkeypatch.setattr(vu.gpu_select, "smi_query", lambda gpu, fields: "621.14")
    by_index = {OTHER.index: OTHER.uuid, TI.index: TI.uuid}
    monkeypatch.setattr(
        vu,
        "_measure_bf16_tflops",
        lambda torch, i, n, w, it: state["tflops"][by_index[i]],
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    state["cal"] = str(tmp_path / "cal.json")
    return state


def _run(capsys, *argv):
    code = vu.main(list(argv))
    return code, json.loads(capsys.readouterr().out)


def test_main_reports_which_card_was_measured(fake_hw, capsys):
    code, out = _run(
        capsys, "--device", "GPU-12345678", "--json", "--calibration", fake_hw["cal"]
    )
    assert out["uuid"] == TI.uuid and out["pci_bus_id"] == TI.pci_bus_id
    assert out["reference_source"] == "spec-table"
    assert code == 1  # 72 vs table 87.5 -> STOCK


def test_calibrate_then_later_run_uses_calibration_not_table(fake_hw, capsys):
    code, out = _run(
        capsys,
        "--device",
        "3090 Ti",
        "--calibrate",
        "--json",
        "--calibration",
        fake_hw["cal"],
    )
    assert code == 0 and out["verdict"] == "CALIBRATED"
    stored = vu.load_calibration(vu.Path(fake_hw["cal"]))
    assert stored[gs.norm_uuid(TI.uuid)]["bf16_tflops"] == 72.0
    # unlocked boot: 130 TFLOPS is 1.49x the table (inconclusive) but 1.8x this card's baseline
    fake_hw["tflops"][TI.uuid] = 130.0
    code, out = _run(capsys, "--device", "1", "--json", "--calibration", fake_hw["cal"])
    assert (
        out["reference_source"] == "calibration"
        and out["stock_reference_tflops"] == 72.0
    )
    assert code == 0 and out["verdict"] == "UNLOCKED"


def test_calibrate_refuses_busy_card_and_unlocked_looking_value(fake_hw, capsys):
    fake_hw["util"][TI.uuid] = 95
    code, out = _run(
        capsys,
        "--device",
        "3090 Ti",
        "--calibrate",
        "--json",
        "--calibration",
        fake_hw["cal"],
    )
    assert code == 2 and "busy" in out["note"]
    fake_hw["util"][TI.uuid] = 0
    fake_hw["tflops"][TI.uuid] = 150.0
    code, out = _run(
        capsys,
        "--device",
        "3090 Ti",
        "--calibrate",
        "--json",
        "--calibration",
        fake_hw["cal"],
    )
    assert code == 2 and "looks unlocked" in out["note"]
    assert vu.load_calibration(vu.Path(fake_hw["cal"])) == {}
    code, out = _run(
        capsys,
        "--device",
        "3090 Ti",
        "--calibrate",
        "--force",
        "--json",
        "--calibration",
        fake_hw["cal"],
    )
    assert code == 0 and "warning" in out


def test_busy_card_downgrades_stock_to_inconclusive(fake_hw, capsys):
    fake_hw["util"][TI.uuid] = 80
    code, out = _run(
        capsys, "--device", "3090 Ti", "--json", "--calibration", fake_hw["cal"]
    )
    assert code == 2 and "busy" in out["warning"]


def test_corrupt_calibration_gives_inconclusive_not_table(fake_hw, capsys):
    vu.Path(fake_hw["cal"]).write_text("{oops", encoding="utf-8")
    code, out = _run(
        capsys, "--device", "3090 Ti", "--json", "--calibration", fake_hw["cal"]
    )
    assert code == 2 and "calibration file" in out["note"]
    assert "stock_reference_tflops" not in out


def test_all_measures_every_card_and_exits_with_worst(fake_hw, capsys):
    fake_hw["tflops"][OTHER.uuid] = 50.0  # 3060: 50/27 = 1.85 -> UNLOCKED
    code, out = _run(capsys, "--all", "--json", "--calibration", fake_hw["cal"])
    assert [g["uuid"] for g in out["gpus"]] == [OTHER.uuid, TI.uuid]
    assert [g["verdict"] for g in out["gpus"]] == ["UNLOCKED", "STOCK"]
    assert code == 1 and out["exit"] == 1


def test_unknown_device_cannot_measure(fake_hw, capsys):
    code, out = _run(
        capsys, "--device", "4090", "--json", "--calibration", fake_hw["cal"]
    )
    assert code == 3 and "no GPU matches" in out["error"]


def test_calibrate_with_stock_override_is_rejected(fake_hw):
    with pytest.raises(SystemExit):
        vu.main(["--calibrate", "--stock", "80"])
