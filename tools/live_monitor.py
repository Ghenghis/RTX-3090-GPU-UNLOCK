#!/usr/bin/env python3
"""Real-time GPU monitor: live status of everything the unlock affects.

Cross-platform (Windows + Linux/WSL2): telemetry comes from `nvidia-smi`, which behaves
identically on both. Every refresh it shows live clocks, power vs limit, temperature,
fan, utilization and VRAM, and, if PyTorch+CUDA is present, runs a short BF16 GEMM
probe and prints the *measured* throughput right now plus a live UNLOCKED/STOCK verdict.

It only measures. It never unlocks, changes a setting, or touches firmware.

    python live_monitor.py                         # default card, refresh every 2s
    python live_monitor.py --device GPU-12345678   # one card by UUID, index or name
    python live_monitor.py --all                   # every card
    python live_monitor.py --once --json           # one sample, machine-readable
    python live_monitor.py --no-probe              # telemetry only (no GEMM probe)

Card selection works as in tools/verify_unlock.py: CUDA_DEVICE_ORDER is forced to
PCI_BUS_ID, the card is located by UUID, and each block names the card it shows. The
verdict uses the same stock reference: the card's calibration if one exists
(verify_unlock.py --calibrate), else the approximate spec table.

Exit codes: 0 normal exit, 3 nvidia-smi not available or the card cannot be selected.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import gpu_select
import verify_unlock as vu  # shared stock table, calibration and classifier
from gpu_select import Gpu, SelectionError

SMI_FIELDS = [
    "name",
    "driver_version",
    "clocks.sm",
    "clocks.mem",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "fan.speed",
    "utilization.gpu",
    "memory.used",
    "memory.total",
]


def parse_smi_csv(line: str, fields: list[str]) -> dict[str, str | None]:
    """Parse one CSV row from nvidia-smi --format=csv,noheader,nounits into a dict.

    Values are kept as trimmed strings; missing/'[N/A]' become None. Pure + testable.
    """
    parts = [p.strip() for p in line.split(",")]
    out: dict[str, str | None] = {}
    for i, key in enumerate(fields):
        val: str | None = parts[i] if i < len(parts) else None
        if val in (None, "", "[N/A]", "N/A"):
            val = None
        out[key] = val
    return out


def sample_telemetry(gpu: Gpu, fields: list[str] = SMI_FIELDS) -> dict[str, str | None]:
    """Telemetry for exactly this card (nvidia-smi -i <uuid>)."""
    row = gpu_select.smi_query(gpu, ",".join(fields))
    if row is None:
        raise subprocess.SubprocessError(
            f"nvidia-smi returned no data for {gpu.describe()}"
        )
    return parse_smi_csv(row, fields)


def throughput_verdict(
    measured_tflops: float,
    device_name: str,
    uuid: str = "",
    calibration: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify a live BF16 measurement against the card's stock reference. Pure + testable."""
    gpu = Gpu(index=-1, uuid=uuid, name=device_name)
    ref = vu.reference_for(gpu, calibration or {})
    if ref is None:
        return {
            "verdict": "UNKNOWN",
            "measured_tflops": round(measured_tflops, 1),
            "note": "no stock reference for this card",
        }
    stock = ref["tflops"]
    code = vu.classify(measured_tflops, stock)
    return {
        "verdict": vu.VERDICTS[code],
        "measured_tflops": round(measured_tflops, 1),
        "stock_reference_tflops": stock,
        "reference_source": ref["source"],
        "ratio": round(measured_tflops / stock, 2),
    }


def _probe_bf16(torch: Any, ordinal: int, size: int, iters: int) -> float:
    device = torch.device("cuda", ordinal)
    with torch.cuda.device(device):
        a = torch.randn((size, size), device=device, dtype=torch.bfloat16)
        b = torch.randn((size, size), device=device, dtype=torch.bfloat16)
        (a @ b).float().sum()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(max(1, iters)):
            start.record()
            c = a @ b
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end) / 1000.0)
            del c
        del a, b
    times.sort()
    return (2.0 * size**3) / times[len(times) // 2] / 1e12


def _load_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def collect(
    gpu: Gpu,
    torch: Any | None,
    probe: bool,
    size: int,
    iters: int,
    calibration: dict[str, dict[str, Any]],
    calibration_error: str | None = None,
) -> dict[str, Any]:
    tel = sample_telemetry(gpu)
    snap: dict[str, Any] = {
        "ts": time.strftime("%H:%M:%S"),
        "gpu": gpu.as_dict(),
        "telemetry": tel,
    }
    if not probe:
        return snap
    if torch is None:
        snap["throughput"] = {"verdict": "NO PROBE", "note": "torch/CUDA unavailable"}
        return snap
    try:
        ordinal = gpu_select.torch_index(torch, gpu)
        tflops = _probe_bf16(torch, ordinal, size, iters)
    except (SelectionError, RuntimeError) as exc:
        snap["throughput"] = {"verdict": "NO PROBE", "note": str(exc)}
        return snap
    if calibration_error:
        # A broken calibration file must not silently fall back to the spec table.
        snap["throughput"] = {
            "verdict": "UNKNOWN",
            "measured_tflops": round(tflops, 1),
            "note": calibration_error,
        }
        return snap
    snap["throughput"] = throughput_verdict(
        tflops, tel.get("name") or gpu.name, gpu.uuid, calibration
    )
    return snap


def _bar(frac: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, frac))
    fill = int(frac * width)
    return "[" + "#" * fill + "-" * (width - fill) + "]"


def render(snap: dict[str, Any]) -> str:
    t = snap["telemetry"]
    lines = [
        f"  RTX live monitor                                   {snap['ts']}",
        "  " + "-" * 52,
    ]
    name = t.get("name") or "?"
    lines.append(f"  {name}   driver {t.get('driver_version') or '?'}")
    gpu = snap.get("gpu")
    if gpu:
        lines.append(
            f"  card    #{gpu.get('index')} {gpu.get('uuid') or 'UUID unknown'}  PCI {gpu.get('pci_bus_id') or '?'}"
        )
    try:
        draw = float(t.get("power.draw") or 0)
        lim = float(t.get("power.limit") or 0)
        pbar = _bar(draw / lim if lim else 0)
        lines.append(f"  power   {draw:6.1f} / {lim:6.1f} W  {pbar}")
    except (TypeError, ValueError):
        lines.append(f"  power   {t.get('power.draw')} / {t.get('power.limit')} W")
    try:
        util = float(t.get("utilization.gpu") or 0)
        lines.append(f"  gpu     {util:6.1f} %        {_bar(util / 100)}")
    except (TypeError, ValueError):
        pass
    lines.append(
        f"  clocks  sm {t.get('clocks.sm') or '?'} MHz   mem {t.get('clocks.mem') or '?'} MHz"
    )
    lines.append(
        f"  temp    {t.get('temperature.gpu') or '?'} C      fan {t.get('fan.speed') or '?'} %"
    )
    lines.append(
        f"  vram    {t.get('memory.used') or '?'} / {t.get('memory.total') or '?'} MiB"
    )
    if "throughput" in snap:
        tp = snap["throughput"]
        lines.append("  " + "-" * 52)
        if "measured_tflops" in tp:
            extra = f"   {tp['ratio']}x vs stock" if "ratio" in tp else ""
            source = f" ({tp['reference_source']})" if "reference_source" in tp else ""
            lines.append(
                f"  BF16    {tp['measured_tflops']} TFLOPS now   ->  {tp['verdict']}{extra}{source}"
            )
        else:
            lines.append(f"  BF16    {tp['verdict']} ({tp.get('note', '')})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--interval", type=float, default=2.0, help="refresh seconds (default 2)"
    )
    p.add_argument("--once", action="store_true", help="single sample then exit")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument(
        "--no-probe", action="store_true", help="telemetry only, skip the GEMM probe"
    )
    sel = p.add_mutually_exclusive_group()
    sel.add_argument(
        "--device",
        default=None,
        help="nvidia-smi index, GPU UUID (GPU-...) or unique name substring",
    )
    sel.add_argument("--all", action="store_true", help="show every NVIDIA GPU")
    p.add_argument(
        "--calibration",
        default=None,
        help="calibration file (default: as verify_unlock.py)",
    )
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args(argv)

    gpu_select.force_pci_bus_order()
    gpus = gpu_select.list_gpus_smi()
    if gpus is None:
        print(
            "error: nvidia-smi not found on PATH. Cannot read telemetry.",
            file=sys.stderr,
        )
        return 3
    try:
        targets = gpu_select.select(args.device, args.all, gpus)
    except SelectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    torch = None if args.no_probe else _load_torch()
    cal_path = (
        Path(args.calibration) if args.calibration else vu.default_calibration_path()
    )
    calibration: dict[str, dict[str, Any]] = {}
    calibration_error = None
    try:
        calibration = vu.load_calibration(cal_path)
    except vu.CalibrationError as exc:
        calibration_error = f"{exc}; fix or delete it"
        print(f"warning: {calibration_error}", file=sys.stderr)

    def one() -> None:
        snaps = [
            collect(
                g,
                torch,
                not args.no_probe,
                args.size,
                args.iters,
                calibration,
                calibration_error,
            )
            for g in targets
        ]
        if args.json:
            print(json.dumps(snaps[0] if not args.all else {"gpus": snaps}))
            return
        if not args.once:
            print("\033[2J\033[H", end="")  # clear screen
        print("\n\n".join(render(s) for s in snaps))

    try:
        if args.once:
            one()
        else:
            while True:
                one()
                time.sleep(max(0.2, args.interval))
    except KeyboardInterrupt:
        pass
    except subprocess.SubprocessError as exc:
        print(f"error: nvidia-smi failed: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
