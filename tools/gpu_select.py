"""Pick which physical GPU a measurement runs on, and say which one it was.

Shared by tools/verify_unlock.py, tools/live_monitor.py and benchmark/gemm_bench.py.

Why this exists: CUDA's default device order is FASTEST_FIRST, nvidia-smi's is PCI
bus order, and CUDA_VISIBLE_DEVICES renumbers whatever it leaves visible. A bare
"cuda:0" can therefore be a different card in each tool, or a different card after
a driver update. So:

* CUDA_DEVICE_ORDER is forced to PCI_BUS_ID before CUDA initialises, which makes
  CUDA ordinals match nvidia-smi indices.
* A --device spec is resolved against nvidia-smi's inventory (index, UUID or name
  substring), and the card is then located in torch by UUID, never by position.
* Every result names the card it measured (name, UUID, PCI bus ID).

Only reads inventory and telemetry. It never changes a GPU setting.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SMI_TIMEOUT_S = 10
# A card at or above this utilisation before a measurement is shared with other work,
# so a low reading may be contention rather than stock selectors.
BUSY_UTIL_PCT = 20

_UUID_RE = re.compile(r"^(GPU|MIG)-[0-9a-f-]+$", re.IGNORECASE)


class SelectionError(RuntimeError):
    """The requested card does not exist, is ambiguous, or is hidden from CUDA."""


@dataclass(frozen=True)
class Gpu:
    index: int  # nvidia-smi index (PCI bus order)
    uuid: str  # "GPU-xxxxxxxx-...", as nvidia-smi prints it; "" if unknown
    name: str
    pci_bus_id: str | None = None

    def describe(self) -> str:
        bus = f", PCI {self.pci_bus_id}" if self.pci_bus_id else ""
        uuid = self.uuid or "UUID unknown"
        return f"#{self.index} {self.name} ({uuid}{bus})"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "pci_bus_id": self.pci_bus_id,
        }


def norm_uuid(value: str) -> str:
    """Compare UUIDs without case or the GPU- prefix (torch omits it)."""
    value = value.strip().lower()
    return value[4:] if value.startswith("gpu-") else value


def force_pci_bus_order() -> str | None:
    """Set CUDA_DEVICE_ORDER=PCI_BUS_ID; return the previous value if it differed.

    Must run before torch initialises CUDA. All three tools call it before importing torch.
    """
    previous = os.environ.get("CUDA_DEVICE_ORDER")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return previous if previous not in (None, "PCI_BUS_ID") else None


def parse_smi_list(text: str) -> list[Gpu]:
    """Parse `nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv,noheader`."""
    gpus: list[Gpu] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        bus = (
            parts[3]
            if len(parts) > 3 and parts[3] not in ("", "[N/A]", "N/A")
            else None
        )
        gpus.append(
            Gpu(index=int(parts[0]), uuid=parts[1], name=parts[2], pci_bus_id=bus)
        )
    return sorted(gpus, key=lambda g: g.index)


def _smi(args: list[str]) -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        return subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=SMI_TIMEOUT_S,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def list_gpus_smi() -> list[Gpu] | None:
    """All NVIDIA GPUs in PCI order, or None if nvidia-smi is unavailable."""
    out = _smi(["--query-gpu=index,uuid,name,pci.bus_id", "--format=csv,noheader"])
    if out is None:
        return None
    return parse_smi_list(out)


def smi_query(gpu: Gpu, fields: str) -> str | None:
    """One nvidia-smi --query-gpu row for exactly this card (selected by UUID)."""
    target = gpu.uuid or str(gpu.index)
    out = _smi(["-i", target, f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if not out:
        return None
    lines = out.strip().splitlines()
    return lines[0].strip() if lines else None


def utilization_pct(gpu: Gpu) -> int | None:
    value = smi_query(gpu, "utilization.gpu")
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


def _torch_uuid(props: Any) -> str | None:
    value = getattr(props, "uuid", None)
    return str(value) if value is not None else None


def list_gpus_torch(torch: Any) -> list[Gpu]:
    """Fallback inventory from torch when nvidia-smi is missing (indices are CUDA ordinals)."""
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        uuid = _torch_uuid(props)
        gpus.append(
            Gpu(
                index=i,
                uuid=f"GPU-{uuid}" if uuid else "",
                name=torch.cuda.get_device_name(i),
            )
        )
    return gpus


def resolve(spec: str, gpus: list[Gpu]) -> Gpu:
    """Resolve a --device spec: an index, a UUID (or unique UUID prefix), or a name substring.

    Digits that name an existing index select it (nvidia-smi index, PCI bus order); other
    digits such as "3090" are a model number and match names. "GPU-..." is a UUID.
    Anything else is a case-insensitive name substring. Names must match exactly one card.
    """
    spec = spec.strip()
    if not gpus:
        raise SelectionError("no NVIDIA GPU found")
    listing = "; ".join(g.describe() for g in gpus)
    if not spec:
        raise SelectionError("empty --device")
    if spec.isdigit():
        for gpu in gpus:
            if gpu.index == int(spec):
                return gpu
    if _UUID_RE.match(spec):
        want = norm_uuid(spec)
        hits = [g for g in gpus if g.uuid and norm_uuid(g.uuid).startswith(want)]
    else:
        hits = [g for g in gpus if spec.lower() in g.name.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SelectionError(f"no GPU matches {spec!r}. Present: {listing}")
    raise SelectionError(
        f"{spec!r} matches {len(hits)} GPUs ({'; '.join(g.describe() for g in hits)}); "
        "use the UUID instead"
    )


def default_spec(env: Mapping[str, str] | None = None) -> str:
    """The card a run uses without --device: the first CUDA_VISIBLE_DEVICES entry, else index 0."""
    source: Mapping[str, str] = os.environ if env is None else env
    visible = source.get("CUDA_VISIBLE_DEVICES", "").strip()
    first = visible.split(",")[0].strip() if visible else ""
    return first or "0"


def select(spec: str | None, all_gpus: bool, gpus: list[Gpu]) -> list[Gpu]:
    """The cards to measure: all of them, the --device one, or the default one."""
    if all_gpus:
        if not gpus:
            raise SelectionError("no NVIDIA GPU found")
        return list(gpus)
    return [resolve(spec if spec is not None else default_spec(), gpus)]


def inventory(torch: Any | None = None) -> list[Gpu]:
    """nvidia-smi inventory, falling back to torch's view when nvidia-smi is missing."""
    gpus = list_gpus_smi()
    if gpus is not None:
        return gpus
    if torch is None:
        raise SelectionError(
            "nvidia-smi not found and torch unavailable; cannot list GPUs"
        )
    return list_gpus_torch(torch)


def torch_index(torch: Any, gpu: Gpu) -> int:
    """The CUDA ordinal of this exact card in this process, located by UUID.

    Raises SelectionError if CUDA_VISIBLE_DEVICES hides it, or if it cannot be told
    apart from the others (old torch without device UUIDs).
    """
    count = torch.cuda.device_count()
    seen_uuid = False
    for i in range(count):
        uuid = _torch_uuid(torch.cuda.get_device_properties(i))
        if uuid is None:
            continue
        seen_uuid = True
        if gpu.uuid and norm_uuid(uuid) == norm_uuid(gpu.uuid):
            return i
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if seen_uuid:
        raise SelectionError(
            f"{gpu.describe()} is not visible to CUDA (CUDA_VISIBLE_DEVICES={visible!r})"
        )
    # torch without device UUIDs: with PCI_BUS_ID order and nothing hidden, ordinal == smi index.
    if (
        visible is None
        and gpu.index < count
        and torch.cuda.get_device_name(gpu.index) == gpu.name
    ):
        return gpu.index
    raise SelectionError(
        f"cannot locate {gpu.describe()} in torch by UUID; upgrade torch or unset CUDA_VISIBLE_DEVICES"
    )
