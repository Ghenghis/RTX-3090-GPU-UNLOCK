#!/usr/bin/env python3
"""Reproducible GEMM throughput benchmark: the "proof" layer.

Measures matrix-multiply throughput in the math modes that matter for the tensor
unlock, and records full provenance so two runs (locked vs unlocked) can be compared
honestly. It only measures; it never unlocks or changes any setting.

Modes:
  bf16   BF16 inputs, FP32 accumulate  -> the capped row the unlock targets (primary)
  fp16   FP16 inputs (torch default accumulate)
  tf32   TF32 tensor path (FP32 inputs, allow_tf32=True)
  fp32   full FP32 (allow_tf32=False)  -> CONTROL: should not change with the unlock

Run one pass per boot state, labelling it, then compare the two JSON files:
    python gemm_bench.py --device GPU-12345678 --state locked   --out locked.json
    python gemm_bench.py --device GPU-12345678 --state unlocked --out unlocked.json
    python compare_runs.py locked.json unlocked.json

Card selection works as in tools/verify_unlock.py: CUDA_DEVICE_ORDER is forced to
PCI_BUS_ID, --device takes an index, GPU UUID or unique name substring, and the record
names the card (device, device_uuid, pci_bus_id). With --all every card is measured;
with --out each card gets its own file, e.g. locked.json -> locked.GPU-12345678.json.

Exit codes: 0 ok, 3 cannot measure (no torch/CUDA, card not found or hidden).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import gpu_select  # noqa: E402  (shared card selection lives in tools/)
from gpu_select import Gpu, SelectionError  # noqa: E402

SCHEMA = "gemm-bench/1"


def _provenance(torch: Any, gpu: Gpu, ordinal: int) -> dict[str, Any]:
    return {
        "device": torch.cuda.get_device_name(ordinal),
        "device_uuid": gpu.uuid or None,
        "pci_bus_id": gpu.pci_bus_id,
        "smi_index": gpu.index,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": gpu_select.smi_query(gpu, "driver_version"),
        "sm_clock_mhz": gpu_select.smi_query(gpu, "clocks.sm"),
        "power_limit_w": gpu_select.smi_query(gpu, "power.limit"),
        "util_before_pct": gpu_select.utilization_pct(gpu),
        "host": platform.node(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _time_matmul(torch: Any, a: Any, b: Any, iters: int) -> float:
    """Median seconds for a single a@b over `iters` timed runs."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        c = a @ b
        end.record()
        torch.cuda.synchronize(a.device)
        times.append(start.elapsed_time(end) / 1000.0)
        del c
    times.sort()
    return times[len(times) // 2]


def _tflops(n: int, seconds: float) -> float:
    return (2.0 * n**3) / seconds / 1e12


def run(
    torch: Any, ordinal: int, size: int, warmup: int, iters: int
) -> list[dict[str, Any]]:
    dev = torch.device("cuda", ordinal)
    n = size
    results: list[dict[str, Any]] = []

    def measure(label: str, dtype: Any, allow_tf32: bool | None = None) -> None:
        prev = torch.backends.cuda.matmul.allow_tf32
        try:
            if allow_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            a = torch.randn((n, n), device=dev, dtype=dtype)
            b = torch.randn((n, n), device=dev, dtype=dtype)
            for _ in range(max(1, warmup)):
                (a @ b).float().sum()
            torch.cuda.synchronize(dev)
            sec = _time_matmul(torch, a, b, max(1, iters))
            results.append(
                {
                    "mode": label,
                    "tflops": round(_tflops(n, sec), 2),
                    "median_ms": round(sec * 1000, 4),
                }
            )
        except Exception as exc:  # a mode may be unsupported on some setups
            results.append({"mode": label, "tflops": None, "error": str(exc)})
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    with torch.cuda.device(dev):
        measure("bf16", torch.bfloat16)
        measure("fp16", torch.float16)
        measure("tf32", torch.float32, allow_tf32=True)
        measure("fp32", torch.float32, allow_tf32=False)
    return results


def per_device_out(out: str, gpu: Gpu) -> Path:
    """locked.json -> locked.GPU-12345678.json (first UUID group; index if no UUID)."""
    path = Path(out)
    parts = gpu.uuid.split("-")
    tag = "-".join(parts[:2]) if len(parts) >= 2 else f"gpu{gpu.index}"
    return path.with_name(f"{path.stem}.{tag}{path.suffix or '.json'}")


def _write(path: Path, record: dict[str, Any]) -> bool:
    try:
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"error: could not write {path}: {exc}", file=sys.stderr)
        return False
    print(f"wrote {path}")
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--state",
        choices=["locked", "unlocked", "unknown"],
        default="unknown",
        help="label this run's boot/unlock state (for honest comparison)",
    )
    sel = p.add_mutually_exclusive_group()
    sel.add_argument(
        "--device",
        default=None,
        help="nvidia-smi index, GPU UUID (GPU-...) or unique name substring",
    )
    sel.add_argument("--all", action="store_true", help="benchmark every NVIDIA GPU")
    p.add_argument("-n", "--size", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument(
        "--out",
        default=None,
        help="write the run JSON here (one file per card with --all)",
    )
    args = p.parse_args(argv)

    previous_order = gpu_select.force_pci_bus_order()
    try:
        import torch
    except ImportError:
        print(
            "error: PyTorch not installed (pip install torch). Cannot measure.",
            file=sys.stderr,
        )
        return 3
    if not torch.cuda.is_available():
        print("error: CUDA not available to PyTorch. Cannot measure.", file=sys.stderr)
        return 3

    try:
        targets = gpu_select.select(args.device, args.all, gpu_select.inventory(torch))
    except SelectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    code = 0
    for gpu in targets:
        try:
            ordinal = gpu_select.torch_index(torch, gpu)
        except SelectionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            code = 3
            continue
        print(f"measuring {gpu.describe()}", file=sys.stderr)
        provenance = _provenance(torch, gpu, ordinal)
        busy = provenance["util_before_pct"]
        if busy is not None and busy >= gpu_select.BUSY_UTIL_PCT:
            print(
                f"warning: card was {busy}% busy before measuring; results may reflect contention",
                file=sys.stderr,
            )
        record = {
            "schema": SCHEMA,
            "state": args.state,
            "size": args.size,
            "warmup": args.warmup,
            "iters": args.iters,
            **provenance,
            "results": run(torch, ordinal, args.size, args.warmup, args.iters),
        }
        if previous_order:
            record["device_order_note"] = (
                f"CUDA_DEVICE_ORDER was {previous_order}; using PCI_BUS_ID"
            )
        if args.out:
            path = per_device_out(args.out, gpu) if args.all else Path(args.out)
            if not _write(path, record):
                code = 3
        else:
            print(json.dumps(record, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
