#!/usr/bin/env python3
"""Post-boot verifier: is the tensor unlock actually active right now?

After a boot, this answers one question quickly: did the full-rate tensor selectors
take effect, or is the card running at stock? It measures BF16 (FP32-accumulate) GEMM
throughput and compares it to a stock reference for that exact card. It does NOT unlock
anything, change any setting, or touch firmware. It only measures and reports.

The stock reference, in order of preference:
  1. --stock <tflops>, if given;
  2. this card's own calibration (by GPU UUID), stored by --calibrate on a stock boot;
  3. the approximate spec table below, only when the card has no calibration.
A spec-sheet peak is a rough baseline, so calibrate once per card on a stock boot (on
Windows: F8 boot menu -> Windows Boot Manager, which skips the unlock entry).

This is a heuristic confidence check, not a benchmark of record. The definitive test
is a controlled locked-vs-unlocked run (benchmark/gemm_bench.py + compare_runs.py).

Card selection: CUDA_DEVICE_ORDER is forced to PCI_BUS_ID, and --device takes an
nvidia-smi index, a GPU UUID (GPU-...) or a unique name substring. Without --device the
first CUDA_VISIBLE_DEVICES entry is used, else index 0. Every result names the card.

Usage:
    python verify_unlock.py                         # default card, measure, classify
    python verify_unlock.py --device GPU-12345678   # one card, by UUID (prefix ok)
    python verify_unlock.py --all                   # every card, one verdict each
    python verify_unlock.py --calibrate --device GPU-12345678   # on a STOCK boot only
    python verify_unlock.py --stock 87.5            # override the stock reference
    python verify_unlock.py --json --log unlock.log

Exit codes (with --all: the highest code over all cards):
    0  UNLOCKED  (throughput clearly above stock)
    1  STOCK     (throughput at/near stock; unlock did not take)
    2  INCONCLUSIVE (between thresholds, no reference, busy card, bad calibration file)
    3  CANNOT MEASURE (no CUDA/torch, card not found or hidden, measurement failed)
With --calibrate: 0 stored, 2 refused (busy card or implausible value), 3 cannot measure.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import gpu_select
from gpu_select import Gpu, SelectionError

# Approximate stock dense BF16 (FP32-accumulate) tensor throughput, in TFLOPS, for
# consumer Ampere/Ada cards at stock selectors. These are public reference figures and
# vary with driver, clocks and board; they are a baseline for a ratio test, not exact
# truth. Unlocked throughput is typically ~1.8-2x these. Override with --stock.
STOCK_BF16_TFLOPS: dict[str, float] = {
    "3060": 27.0,
    "3060 ti": 32.0,
    "3070": 40.0,
    "3070 ti": 43.0,
    "3080": 65.0,
    "3080 ti": 68.0,
    "3090": 76.0,
    "3090 ti": 87.5,
    "4070": 59.0,
    "4070 ti": 80.0,
    "4080": 104.0,
    "4090": 165.0,
}

# A measured value >= stock * UNLOCK_RATIO is treated as unlocked; <= stock * STOCK_RATIO
# as stock. Between the two is inconclusive. The gap leaves room for clock/driver noise.
UNLOCK_RATIO = 1.5
STOCK_RATIO = 1.2

# A calibration is a stock measurement. Relative to the spec table, a value above
# STOCK_RATIO looks unlocked, and one below CAL_MIN_RATIO suggests a busy or throttled
# card. Either would make every later verdict wrong, so --calibrate refuses it unless
# --force is given.
CAL_MIN_RATIO = 0.5
CAL_REPEATS = 3
CALIBRATION_ENV = "RTX_UNLOCK_CALIBRATION"
CALIBRATION_SCHEMA = "rtx-unlock-calibration/1"

VERDICTS = {0: "UNLOCKED", 1: "STOCK", 2: "INCONCLUSIVE", 3: "CANNOT MEASURE"}


class CalibrationError(ValueError):
    """The calibration file exists but cannot be used."""


def pick_reference(
    device_name: str, table: dict[str, float] | None = None
) -> tuple[str, float] | None:
    """Return (matched_key, stock_tflops) for a GPU name, or None if unknown.

    Matches the most specific key first (e.g. '3090 ti' before '3090') so a Ti card is
    never misread as its non-Ti sibling.
    """
    table = STOCK_BF16_TFLOPS if table is None else table
    name = device_name.lower()
    best: tuple[str, float] | None = None
    for key, val in table.items():
        if key in name and (best is None or len(key) > len(best[0])):
            best = (key, val)
    return best


def classify(
    measured_tflops: float,
    stock_tflops: float,
    unlock_ratio: float = UNLOCK_RATIO,
    stock_ratio: float = STOCK_RATIO,
) -> int:
    """Classify a measurement against a stock reference. Returns an exit code (0/1/2)."""
    if stock_tflops <= 0:
        return 2
    ratio = measured_tflops / stock_tflops
    if ratio >= unlock_ratio:
        return 0  # UNLOCKED
    if ratio <= stock_ratio:
        return 1  # STOCK
    return 2  # INCONCLUSIVE


# --- calibration store -------------------------------------------------------


def default_calibration_path(env: dict[str, str] | None = None) -> Path:
    """$RTX_UNLOCK_CALIBRATION, else %APPDATA%/rtx-unlock/ (Windows) or ~/.config/rtx-unlock/."""
    env = dict(os.environ) if env is None else env
    if env.get(CALIBRATION_ENV):
        return Path(env[CALIBRATION_ENV])
    if os.name == "nt" and env.get("APPDATA"):
        base = Path(env["APPDATA"])
    elif env.get("XDG_CONFIG_HOME"):
        base = Path(env["XDG_CONFIG_HOME"])
    else:
        base = Path.home() / ".config"
    return base / "rtx-unlock" / "calibration.json"


def load_calibration(path: Path) -> dict[str, dict[str, Any]]:
    """Calibrations keyed by normalised GPU UUID. A missing file means none exist.

    A file that exists but is unreadable or malformed raises CalibrationError: falling
    back to the spec table there would hide that the card's own baseline was lost.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise CalibrationError(f"cannot read calibration file {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CalibrationError(
            f"calibration file {path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict) or data.get("schema") != CALIBRATION_SCHEMA:
        raise CalibrationError(f"calibration file {path} has an unknown format")
    gpus = data.get("gpus")
    if not isinstance(gpus, dict):
        raise CalibrationError(f"calibration file {path} has an unknown format")
    out: dict[str, dict[str, Any]] = {}
    for key, entry in gpus.items():
        value = entry.get("bf16_tflops") if isinstance(entry, dict) else None
        if not isinstance(value, (int, float)) or value <= 0:
            raise CalibrationError(f"calibration file {path}: bad entry for {key}")
        out[gpu_select.norm_uuid(key)] = entry
    return out


def save_calibration(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Write all calibrations atomically (temp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema": CALIBRATION_SCHEMA,
        "note": "Stock-boot BF16 GEMM baselines written by verify_unlock.py --calibrate.",
        "gpus": {entry["uuid"]: entry for entry in entries.values()},
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def reference_for(
    gpu: Gpu,
    calibration: dict[str, dict[str, Any]],
    stock_override: float | None = None,
) -> dict[str, Any] | None:
    """The stock reference for this card: override, then calibration, then spec table."""
    if stock_override is not None:
        return {"source": "override", "key": "override", "tflops": stock_override}
    entry = calibration.get(gpu_select.norm_uuid(gpu.uuid)) if gpu.uuid else None
    if entry is not None:
        return {
            "source": "calibration",
            "key": gpu.uuid,
            "tflops": float(entry["bf16_tflops"]),
            "calibrated_at": entry.get("measured_at"),
            "calibrated_driver": entry.get("driver"),
        }
    ref = pick_reference(gpu.name)
    if ref is None:
        return None
    return {"source": "spec-table", "key": ref[0], "tflops": ref[1]}


def calibration_refusal(measured_tflops: float, device_name: str) -> str | None:
    """Why a stock calibration value is implausible for this card, or None if it is fine."""
    ref = pick_reference(device_name)
    if ref is None:
        return None
    key, table = ref
    ratio = measured_tflops / table
    if ratio > STOCK_RATIO:
        return (
            f"{measured_tflops:.1f} TFLOPS is {ratio:.2f}x the '{key}' spec ({table}); "
            "that looks unlocked, so this is probably not a stock boot"
        )
    if ratio < CAL_MIN_RATIO:
        return (
            f"{measured_tflops:.1f} TFLOPS is only {ratio:.2f}x the '{key}' spec ({table}); "
            "the card may be busy, throttled or power-limited"
        )
    return None


# --- measurement -------------------------------------------------------------


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed; cannot measure (pip install torch)."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch; cannot measure throughput."
        )
    return torch


def _measure_bf16_tflops(
    torch: Any, ordinal: int, n: int, warmup: int, iters: int
) -> float:
    """BF16 GEMM throughput (median of `iters` runs) on CUDA device `ordinal`, in TFLOPS."""
    device = torch.device("cuda", ordinal)
    with torch.cuda.device(device):
        a = torch.randn((n, n), device=device, dtype=torch.bfloat16)
        b = torch.randn((n, n), device=device, dtype=torch.bfloat16)
        for _ in range(max(1, warmup)):
            (a @ b).float().sum()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_ms: list[float] = []
        for _ in range(max(1, iters)):
            start.record()
            c = a @ b
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(start.elapsed_time(end))
            del c
        del a, b
    times_ms.sort()
    median_s = times_ms[len(times_ms) // 2] / 1000.0
    flop = 2.0 * (n**3)  # multiply-add per output element
    return flop / median_s / 1e12


def _driver_version(gpu: Gpu) -> str | None:
    return gpu_select.smi_query(gpu, "driver_version")


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _run_one(
    torch: Any, gpu: Gpu, args: argparse.Namespace, cal_path: Path
) -> tuple[int, dict[str, Any]]:
    result: dict[str, Any] = {
        "device": gpu.name,
        "uuid": gpu.uuid or None,
        "pci_bus_id": gpu.pci_bus_id,
        "index": gpu.index,
    }
    if torch is None:
        result["error"] = args.torch_error
        return 3, result
    try:
        ordinal = gpu_select.torch_index(torch, gpu)
    except SelectionError as exc:
        result["error"] = str(exc)
        return 3, result
    busy = gpu_select.utilization_pct(gpu)
    if busy is not None:
        result["util_before_pct"] = busy
    repeats = CAL_REPEATS if args.calibrate else 1
    try:
        samples = [
            _measure_bf16_tflops(torch, ordinal, args.size, args.warmup, args.iters)
            for _ in range(repeats)
        ]
    except RuntimeError as exc:  # includes CUDA errors and out-of-memory
        result["error"] = f"measurement failed: {exc}"
        return 3, result
    measured = _median(samples)
    result["measured_tflops"] = round(measured, 1)
    is_busy = busy is not None and busy >= gpu_select.BUSY_UTIL_PCT

    if args.calibrate:
        return _calibrate(gpu, measured, samples, is_busy, busy, args, cal_path, result)

    try:
        calibration = {} if args.stock is not None else load_calibration(cal_path)
    except CalibrationError as exc:
        result["note"] = f"{exc}; fix or delete it, or pass --stock"
        return 2, result
    ref = reference_for(gpu, calibration, args.stock)
    if ref is None:
        result["note"] = (
            "no stock reference for this card; run --calibrate on a stock boot or pass --stock"
        )
        return 2, result
    stock = ref["tflops"]
    verdict = classify(measured, stock)
    result.update(
        {
            "stock_reference_tflops": round(stock, 1),
            "reference_source": ref["source"],
            "reference_key": ref["key"],
            "ratio": round(measured / stock, 2) if stock else None,
        }
    )
    if ref["source"] == "calibration":
        driver = _driver_version(gpu)
        result["reference"] = (
            f"calibration from {ref['calibrated_at']} (driver {ref['calibrated_driver']})"
        )
        if driver and ref["calibrated_driver"] and driver != ref["calibrated_driver"]:
            result["warning"] = (
                f"calibrated under driver {ref['calibrated_driver']}, now {driver}; recalibrate"
            )
    elif ref["source"] == "spec-table":
        result["reference"] = (
            f"spec table '{ref['key']}' (no calibration for this card in {cal_path})"
        )
    if is_busy and verdict == 1:
        verdict = 2
        result["warning"] = (
            f"card was {busy}% busy before measuring; a low result may be contention"
        )
    elif is_busy:
        result["warning"] = f"card was {busy}% busy before measuring"
    return verdict, result


def _calibrate(
    gpu: Gpu,
    measured: float,
    samples: list[float],
    is_busy: bool,
    busy: int | None,
    args: argparse.Namespace,
    cal_path: Path,
    result: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    result["samples_tflops"] = [round(s, 2) for s in samples]
    if not gpu.uuid:
        result["error"] = "card has no UUID; calibration is keyed by UUID"
        return 3, result
    reasons = []
    if is_busy:
        reasons.append(
            f"card was {busy}% busy before measuring; calibrate on an idle card"
        )
    implausible = calibration_refusal(measured, gpu.name)
    if implausible:
        reasons.append(implausible)
    if reasons and not args.force:
        result["note"] = (
            "; ".join(reasons) + ". Not stored (use --force to store anyway)."
        )
        return 2, result
    try:
        existing = load_calibration(cal_path)
    except CalibrationError as exc:
        if not args.force:
            result["note"] = f"{exc}; not overwritten (use --force to replace it)"
            return 2, result
        existing = {}
    existing[gpu_select.norm_uuid(gpu.uuid)] = {
        "uuid": gpu.uuid,
        "name": gpu.name,
        "pci_bus_id": gpu.pci_bus_id,
        "bf16_tflops": round(measured, 2),
        "samples_tflops": [round(s, 2) for s in samples],
        "size": args.size,
        "iters": args.iters,
        "driver": _driver_version(gpu),
        "measured_at": _now(),
        "host": platform.node(),
        "forced": bool(reasons),
        "note": "operator-asserted stock boot",
    }
    try:
        save_calibration(cal_path, existing)
    except OSError as exc:
        result["error"] = f"could not write {cal_path}: {exc}"
        return 3, result
    result["stored"] = str(cal_path)
    if reasons:
        result["warning"] = "stored with --force despite: " + "; ".join(reasons)
    return 0, result


# --- output ------------------------------------------------------------------

_TEXT_KEYS = (
    "device",
    "uuid",
    "pci_bus_id",
    "measured_tflops",
    "samples_tflops",
    "stock_reference_tflops",
    "reference",
    "ratio",
    "stored",
    "warning",
    "note",
    "error",
)


def _label(code: int, calibrate: bool) -> str:
    if calibrate:
        return {0: "CALIBRATED", 2: "CALIBRATION REFUSED", 3: "CANNOT MEASURE"}.get(
            code, VERDICTS[code]
        )
    return VERDICTS[code]


def _log(path: str, line: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        print(f"warning: could not write log {path}: {exc}", file=sys.stderr)


def _emit(args: argparse.Namespace, results: list[dict[str, Any]], code: int) -> None:
    stamp = _now()
    if args.json:
        if args.all:
            print(
                json.dumps(
                    {
                        "verdict": _label(code, args.calibrate),
                        "exit": code,
                        "gpus": results,
                    }
                )
            )
        else:
            print(json.dumps(results[0]))
    else:
        for result in results:
            print(f"[{stamp}] {result['verdict']}")
            for key in _TEXT_KEYS:
                if result.get(key) is not None:
                    print(f"    {key}: {result[key]}")
        if args.all and len(results) > 1:
            print(f"overall: {_label(code, args.calibrate)} (exit {code})")
    if args.log:
        for result in results:
            _log(args.log, f"{stamp} {result['verdict']} " + json.dumps(result))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-n", "--size", type=int, default=4096, help="GEMM matrix size (default 4096)"
    )
    p.add_argument(
        "--warmup", type=int, default=5, help="warmup iterations (default 5)"
    )
    p.add_argument(
        "--iters", type=int, default=30, help="timed iterations (default 30)"
    )
    sel = p.add_mutually_exclusive_group()
    sel.add_argument(
        "--device",
        default=None,
        help="nvidia-smi index, GPU UUID (GPU-...) or unique name substring",
    )
    sel.add_argument(
        "--all", action="store_true", help="measure every NVIDIA GPU, one verdict each"
    )
    p.add_argument(
        "--stock",
        type=float,
        default=None,
        help="stock BF16 TFLOPS reference override (skips calibration and the table)",
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="store this measurement as the card's stock baseline (run on a STOCK boot only)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="with --calibrate: store despite the plausibility checks",
    )
    p.add_argument(
        "--calibration",
        default=None,
        help=f"calibration file (default: ${CALIBRATION_ENV}, else the per-user config dir)",
    )
    p.add_argument("--log", default=None, help="append a result line to this log file")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = p.parse_args(argv)
    if args.calibrate and args.stock is not None:
        p.error(
            "--calibrate measures the stock baseline; it cannot be combined with --stock"
        )

    previous_order = gpu_select.force_pci_bus_order()
    cal_path = (
        Path(args.calibration) if args.calibration else default_calibration_path()
    )
    try:
        torch = _import_torch()
        args.torch_error = None
    except RuntimeError as exc:
        torch, args.torch_error = None, str(exc)

    try:
        targets = gpu_select.select(args.device, args.all, gpu_select.inventory(torch))
    except SelectionError as exc:
        result = {"verdict": _label(3, args.calibrate), "error": str(exc)}
        if args.torch_error:
            result["note"] = args.torch_error
        _emit(args, [result], 3)
        return 3

    codes: list[int] = []
    results: list[dict[str, Any]] = []
    for gpu in targets:
        code, result = _run_one(torch, gpu, args, cal_path)
        result = {"verdict": _label(code, args.calibrate), **result}
        if previous_order:
            result["device_order_note"] = (
                f"CUDA_DEVICE_ORDER was {previous_order}; using PCI_BUS_ID"
            )
        codes.append(code)
        results.append(result)
    code = max(codes)
    _emit(args, results, code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
