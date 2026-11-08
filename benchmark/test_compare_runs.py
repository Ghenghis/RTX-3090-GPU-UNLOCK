"""Unit tests for the locked-vs-unlocked comparison logic (no GPU needed)."""

from __future__ import annotations

import json

import compare_runs as cr


def _run(state, **modes):
    return {
        "state": state,
        "device": "RTX 3090 Ti",
        "results": [{"mode": m, "tflops": v} for m, v in modes.items()],
    }


# --- ratio / classify -----------------------------------------------------


def test_ratio_basic():
    assert cr.ratio(80.0, 160.0) == 2.0
    assert cr.ratio(0.0, 100.0) is None


def test_classify_gain_and_flat():
    assert cr.classify("bf16", 1.9) == "gain"
    assert cr.classify("bf16", 1.05) == "no-change"
    assert cr.classify("bf16", 1.25) == "partial"


def test_control_flat_vs_drift():
    assert cr.classify("fp32", 1.03) == "control-flat"
    assert cr.classify("fp32", 1.20) == "control-drift"
    assert cr.classify("fp32", 0.85) == "control-drift"


def test_classify_none():
    assert cr.classify("bf16", None) == "n/a"


# --- results_map skips failed modes ---------------------------------------


def test_results_map_skips_null():
    run = {
        "results": [
            {"mode": "bf16", "tflops": 128.0},
            {"mode": "fp16", "tflops": None, "error": "x"},
        ]
    }
    m = cr.results_map(run)
    assert m == {"bf16": 128.0}


# --- compare + overall verdicts -------------------------------------------


def test_overall_gain_confirmed():
    locked = _run("locked", bf16=87.5, fp16=88.0, tf32=40.0, fp32=22.8)
    unlocked = _run("unlocked", bf16=157.0, fp16=125.0, tf32=70.0, fp32=22.9)
    rows = cr.compare(locked, unlocked)
    code, summary = cr.overall(rows)
    assert code == 0
    assert "GAIN CONFIRMED" in summary


def test_overall_no_gain():
    locked = _run("locked", bf16=87.5, fp32=22.8)
    unlocked = _run("unlocked", bf16=88.0, fp32=22.9)
    code, summary = cr.overall(cr.compare(locked, unlocked))
    assert code == 1
    assert "NO GAIN" in summary


def test_overall_control_drift_is_inconclusive():
    # FP32 jumped 30% -> runs not comparable, even though bf16 "doubled"
    locked = _run("locked", bf16=70.0, fp32=18.0)
    unlocked = _run("unlocked", bf16=140.0, fp32=23.4)
    code, summary = cr.overall(cr.compare(locked, unlocked))
    assert code == 2
    assert "INCONCLUSIVE" in summary


def test_overall_partial_power_limited():
    locked = _run("locked", bf16=87.5, fp32=22.8)
    unlocked = _run("unlocked", bf16=110.0, fp32=22.9)  # 1.26x
    code, summary = cr.overall(cr.compare(locked, unlocked))
    assert code == 1
    assert "PARTIAL" in summary


def test_compare_only_matches_shared_modes():
    locked = _run("locked", bf16=80.0)
    unlocked = _run("unlocked", bf16=150.0, fp32=22.0)
    rows = cr.compare(locked, unlocked)
    assert [r["mode"] for r in rows] == ["bf16"]


def test_load_run_rejects_non_bench(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"hello": "world"}))
    try:
        cr.load_run(str(p))
        assert False, "should have raised"
    except ValueError:
        pass


# --- the two runs must come from the same card -----------------------------


def test_same_card_by_uuid():
    a = {"device_uuid": "GPU-12345678-x"}
    assert cr.same_card(a, {"device_uuid": "gpu-12345678-x"}) is True
    assert cr.same_card(a, {"device_uuid": "GPU-11111111-y"}) is False
    assert cr.same_card(a, {}) is None  # older runs without a UUID


def test_runs_from_different_cards_are_inconclusive(tmp_path, capsys):
    locked = _run("locked", bf16=70.0, fp32=35.0)
    unlocked = _run("unlocked", bf16=140.0, fp32=35.0)
    locked["device_uuid"], unlocked["device_uuid"] = "GPU-aaaa", "GPU-bbbb"
    lp, up = tmp_path / "l.json", tmp_path / "u.json"
    lp.write_text(json.dumps(locked), encoding="utf-8")
    up.write_text(json.dumps(unlocked), encoding="utf-8")
    assert cr.main([str(lp), str(up), "--json"]) == 2
    assert "different cards" in json.loads(capsys.readouterr().out)["summary"]
