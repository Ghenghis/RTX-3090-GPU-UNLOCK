"""Unit tests for gemm_bench's card handling (no GPU or torch needed)."""

from __future__ import annotations

import gemm_bench as gb
from gpu_select import Gpu


def test_per_device_out_names_each_card_file():
    ti = Gpu(
        index=1, uuid="GPU-12345678-90ab-cdef-0123-456789abcdef", name="RTX 3090 Ti"
    )
    assert (
        gb.per_device_out("runs/locked.json", ti).as_posix()
        == "runs/locked.GPU-12345678.json"
    )
    assert (
        gb.per_device_out("locked", Gpu(index=2, uuid="", name="x")).name
        == "locked.gpu2.json"
    )


def test_provenance_comes_from_the_measured_card(monkeypatch):
    ti = Gpu(
        index=1,
        uuid="GPU-12345678-90ab-cdef-0123-456789abcdef",
        name="RTX 3090 Ti",
        pci_bus_id="00000000:0A:00.0",
    )
    asked = []
    monkeypatch.setattr(
        gb.gpu_select, "smi_query", lambda gpu, fields: asked.append(gpu.uuid) or "1"
    )
    monkeypatch.setattr(gb.gpu_select, "utilization_pct", lambda gpu: 0)

    class FakeTorch:
        __version__ = "test"
        version = type("v", (), {"cuda": "12.8"})

        class cuda:  # mirrors torch.cuda
            @staticmethod
            def get_device_name(i):
                return {0: "RTX 3060", 1: "RTX 3090 Ti"}[i]

    prov = gb._provenance(FakeTorch, ti, 1)
    assert prov["device"] == "RTX 3090 Ti"
    assert prov["device_uuid"] == ti.uuid and prov["pci_bus_id"] == ti.pci_bus_id
    assert set(asked) == {ti.uuid}
