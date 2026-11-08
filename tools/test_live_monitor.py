"""Unit tests for the live monitor's pure logic (no GPU, no nvidia-smi needed)."""

from __future__ import annotations

import gpu_select as gs
import live_monitor as lm


# --- nvidia-smi CSV parsing -----------------------------------------------


def test_parse_full_row():
    line = "NVIDIA GeForce RTX 3090 Ti, 621.14, 1860, 1313, 310.5, 450.0, 64, 55, 92, 8192, 24576"
    d = lm.parse_smi_csv(line, lm.SMI_FIELDS)
    assert d["name"] == "NVIDIA GeForce RTX 3090 Ti"
    assert d["power.draw"] == "310.5"
    assert d["power.limit"] == "450.0"
    assert d["memory.total"] == "24576"


def test_parse_handles_na_and_short_rows():
    line = "NVIDIA GeForce RTX 3060, 621.14, [N/A], 1875"
    d = lm.parse_smi_csv(line, lm.SMI_FIELDS)
    assert d["clocks.sm"] is None  # [N/A] -> None
    assert d["clocks.mem"] == "1875"
    assert d["power.draw"] is None  # missing trailing fields -> None


# --- live throughput verdict (shared classifier) --------------------------


def test_verdict_unlocked_3090ti():
    v = lm.throughput_verdict(157.0, "NVIDIA GeForce RTX 3090 Ti")
    assert v["verdict"] == "UNLOCKED"
    assert v["ratio"] == round(157.0 / 87.5, 2)


def test_verdict_stock_3060():
    v = lm.throughput_verdict(27.0, "NVIDIA GeForce RTX 3060")
    assert v["verdict"] == "STOCK"


def test_verdict_unknown_card():
    v = lm.throughput_verdict(200.0, "NVIDIA GeForce RTX 5090")
    assert v["verdict"] == "UNKNOWN"
    assert "note" in v


# --- render is robust to partial telemetry --------------------------------


def test_render_with_missing_fields_does_not_crash():
    snap = {
        "ts": "05:17:00",
        "telemetry": {
            "name": "RTX 3060",
            "power.draw": None,
            "power.limit": None,
            "clocks.sm": "1700",
        },
        "throughput": {"verdict": "NO PROBE", "note": "torch/CUDA unavailable"},
    }
    out = lm.render(snap)
    assert "RTX 3060" in out
    assert "NO PROBE" in out


def test_render_shows_live_ratio():
    snap = {
        "ts": "05:17:00",
        "telemetry": {
            "name": "RTX 3090 Ti",
            "power.draw": "300",
            "power.limit": "450",
            "utilization.gpu": "90",
        },
        "throughput": {
            "verdict": "UNLOCKED",
            "measured_tflops": 157.0,
            "stock_reference_tflops": 87.5,
            "ratio": 1.79,
        },
    }
    out = lm.render(snap)
    assert "1.79x vs stock" in out
    assert "UNLOCKED" in out


def test_bar_bounds():
    assert lm._bar(0.0).count("#") == 0
    assert lm._bar(1.0).count("#") == 20
    assert lm._bar(2.0).count("#") == 20  # clamped


# --- per-card selection and calibration-aware verdicts ---------------------

TI = gs.Gpu(
    index=1,
    uuid="GPU-12345678-90ab-cdef-0123-456789abcdef",
    name="NVIDIA GeForce RTX 3090 Ti",
    pci_bus_id="00000000:0A:00.0",
)


def test_verdict_prefers_card_calibration_over_table():
    cal = {gs.norm_uuid(TI.uuid): {"uuid": TI.uuid, "bf16_tflops": 72.0}}
    v = lm.throughput_verdict(130.0, TI.name, TI.uuid, cal)
    assert v["reference_source"] == "calibration"
    assert (
        v["verdict"] == "UNLOCKED"
    )  # 130/72 = 1.81; against the table it would be inconclusive
    assert lm.throughput_verdict(130.0, TI.name)["verdict"] == "INCONCLUSIVE"


def test_render_names_the_card():
    snap = {"ts": "05:17:00", "gpu": TI.as_dict(), "telemetry": {"name": TI.name}}
    out = lm.render(snap)
    assert "GPU-12345678" in out and "0A:00.0" in out


def test_collect_queries_telemetry_for_the_selected_card(monkeypatch):
    seen = []

    def fake_query(gpu, fields):
        seen.append(gpu.uuid)
        return "NVIDIA GeForce RTX 3090 Ti, 621.14"

    monkeypatch.setattr(lm.gpu_select, "smi_query", fake_query)
    snap = lm.collect(TI, None, probe=False, size=64, iters=1, calibration={})
    assert seen == [TI.uuid]
    assert snap["gpu"]["uuid"] == TI.uuid and "throughput" not in snap


def test_collect_with_broken_calibration_does_not_fall_back_to_table(monkeypatch):
    monkeypatch.setattr(lm.gpu_select, "smi_query", lambda gpu, fields: TI.name)
    monkeypatch.setattr(lm.gpu_select, "torch_index", lambda torch, gpu: 0)
    monkeypatch.setattr(lm, "_probe_bf16", lambda torch, i, size, iters: 160.0)
    snap = lm.collect(
        TI,
        object(),
        probe=True,
        size=64,
        iters=1,
        calibration={},
        calibration_error="bad file",
    )
    assert snap["throughput"]["verdict"] == "UNKNOWN"
    assert snap["throughput"]["note"] == "bad file"
