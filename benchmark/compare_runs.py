#!/usr/bin/env python3
"""Compare two gemm_bench runs (locked vs unlocked) and report honest ratios.

The point of this tool is truth: it shows the per-mode speedup AND checks the control
mode (FP32), which the unlock does not touch. If the control moved a lot, the two runs
weren't measured under comparable conditions and the "gain" is suspect — so it says so
instead of reporting a flattering number.

    python compare_runs.py locked.json unlocked.json
    python compare_runs.py locked.json unlocked.json --json

Exit codes:
  0  gain confirmed and controls flat
  1  no meaningful gain detected (unlock may not be active / wrong workload)
  2  inconclusive (control drift, missing data, or mismatched runs)
"""

from __future__ import annotations

import argparse
import json
import sys

# FP32 must not change with the unlock; it is the soundness control.
CONTROL_MODES = {"fp32"}
# The capped row the unlock targets; the headline indicator.
PRIMARY_MODE = "bf16"
# Expected-to-change modes.
TENSOR_MODES = {"bf16", "fp16", "tf32"}

GAIN_RATIO = 1.4  # >= this on a tensor mode counts as a real gain
FLAT_RATIO = 1.1  # <= this means effectively unchanged
CONTROL_TOL = (
    0.10  # a control mode moving more than +/-10% means the runs aren't comparable
)


def load_run(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if "results" not in data:
        raise ValueError(f"{path}: not a gemm-bench run (no 'results')")
    return data


def results_map(run: dict) -> dict[str, float]:
    """mode -> tflops, skipping modes that failed to measure (tflops is null)."""
    out = {}
    for r in run.get("results", []):
        if r.get("tflops") is not None:
            out[r["mode"]] = float(r["tflops"])
    return out


def ratio(locked: float, unlocked: float) -> float | None:
    if locked <= 0:
        return None
    return unlocked / locked


def classify(mode: str, r: float | None) -> str:
    if r is None:
        return "n/a"
    if mode in CONTROL_MODES:
        return "control-drift" if abs(r - 1.0) > CONTROL_TOL else "control-flat"
    if r >= GAIN_RATIO:
        return "gain"
    if r <= FLAT_RATIO:
        return "no-change"
    return "partial"


def compare(locked: dict, unlocked: dict) -> list[dict]:
    lm, um = results_map(locked), results_map(unlocked)
    rows = []
    for mode in ("bf16", "fp16", "tf32", "fp32"):
        if mode in lm and mode in um:
            r = ratio(lm[mode], um[mode])
            rows.append(
                {
                    "mode": mode,
                    "locked": lm[mode],
                    "unlocked": um[mode],
                    "ratio": round(r, 3) if r is not None else None,
                    "verdict": classify(mode, r),
                }
            )
    return rows


def same_card(locked: dict, unlocked: dict) -> bool | None:
    """True/False when both runs record a GPU UUID; None for older runs without one."""
    a, b = locked.get("device_uuid"), unlocked.get("device_uuid")
    if not a or not b:
        return None
    return str(a).lower() == str(b).lower()


def overall(rows: list[dict]) -> tuple[int, str]:
    by = {r["mode"]: r for r in rows}
    control_bad = any(r["verdict"] == "control-drift" for r in rows)
    if control_bad:
        return 2, (
            "INCONCLUSIVE — FP32 control moved beyond tolerance; the two runs were not "
            "measured under comparable conditions (clocks/power/thermals). Re-run both "
            "back-to-back at the same power limit."
        )
    prim = by.get(PRIMARY_MODE)
    if prim is None or prim["ratio"] is None:
        return 2, "INCONCLUSIVE — no BF16 result to compare."
    if prim["verdict"] == "gain":
        return (
            0,
            f"GAIN CONFIRMED — BF16 {prim['ratio']}x with FP32 control flat. Measurement sound.",
        )
    if prim["verdict"] == "no-change":
        return 1, (
            "NO GAIN — BF16 unchanged. The unlock may not be active, or this card/driver "
            "did not accept it. Verify with tools/verify_unlock.py."
        )
    return (
        1,
        f"PARTIAL — BF16 only {prim['ratio']}x (likely power/clock limited; raise the power limit).",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("locked", help="JSON from a locked (stock) run")
    p.add_argument("unlocked", help="JSON from an unlocked run")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    try:
        locked, unlocked = load_run(args.locked), load_run(args.unlocked)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = compare(locked, unlocked)
    code, summary = overall(rows)
    if same_card(locked, unlocked) is False:
        code, summary = (
            2,
            (
                "INCONCLUSIVE — the two runs were measured on different cards "
                f"({locked.get('device_uuid')} vs {unlocked.get('device_uuid')}). "
                "Re-run both with the same --device."
            ),
        )

    if args.json:
        print(json.dumps({"rows": rows, "exit": code, "summary": summary}, indent=2))
        return code

    dev = unlocked.get("device", "?")
    uuid = unlocked.get("device_uuid")
    print(f"\nGEMM comparison on {dev}" + (f" ({uuid})" if uuid else ""))
    print(
        f"  locked: {locked.get('power_limit_w','?')} W / {locked.get('sm_clock_mhz','?')} MHz"
        f"   unlocked: {unlocked.get('power_limit_w','?')} W / {unlocked.get('sm_clock_mhz','?')} MHz"
    )
    print("-" * 58)
    print(f"{'mode':6} {'locked':>10} {'unlocked':>10} {'ratio':>8}  verdict")
    for r in rows:
        rr = f"{r['ratio']:.2f}x" if r["ratio"] is not None else "n/a"
        print(
            f"{r['mode']:6} {r['locked']:>10.1f} {r['unlocked']:>10.1f} {rr:>8}  {r['verdict']}"
        )
    print("-" * 58)
    print(summary + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
