"""Unit tests for GPU selection (no GPU, torch or nvidia-smi needed)."""

from __future__ import annotations

from types import SimpleNamespace

import gpu_select as gs
import pytest

SMI = (
    "0, GPU-11111111-aaaa-bbbb-cccc-000000000001, NVIDIA GeForce RTX 3060, 00000000:01:00.0\n"
    "1, GPU-12345678-90ab-cdef-0123-456789abcdef, NVIDIA GeForce RTX 3090 Ti, 00000000:0A:00.0\n"
)


def _gpus():
    return gs.parse_smi_list(SMI)


class FakeTorch:
    """Just enough of torch.cuda for torch_index: visible cards, in CUDA order."""

    def __init__(self, visible, with_uuid=True):
        self._visible = visible
        self._with_uuid = with_uuid
        self.cuda = self

    def device_count(self):
        return len(self._visible)

    def get_device_properties(self, i):
        uuid = gs.norm_uuid(self._visible[i].uuid)
        return SimpleNamespace(uuid=uuid) if self._with_uuid else SimpleNamespace()

    def get_device_name(self, i):
        return self._visible[i].name


# --- parsing ---------------------------------------------------------------


def test_parse_smi_list_reads_index_uuid_name_bus():
    gpus = _gpus()
    assert [g.index for g in gpus] == [0, 1]
    assert gpus[1].uuid == "GPU-12345678-90ab-cdef-0123-456789abcdef"
    assert gpus[1].name == "NVIDIA GeForce RTX 3090 Ti"
    assert gpus[1].pci_bus_id == "00000000:0A:00.0"


def test_parse_smi_list_skips_noise_and_na_bus():
    gpus = gs.parse_smi_list("No devices were found\n0, GPU-abc, RTX A, [N/A]\n")
    assert len(gpus) == 1
    assert gpus[0].pci_bus_id is None


def test_describe_names_the_card():
    text = _gpus()[1].describe()
    assert "RTX 3090 Ti" in text
    assert "GPU-12345678" in text
    assert "0A:00.0" in text


# --- resolve ---------------------------------------------------------------


def test_resolve_by_index():
    assert gs.resolve("1", _gpus()).name.endswith("3090 Ti")


def test_resolve_by_full_uuid_and_prefix_case_insensitive():
    assert gs.resolve("GPU-12345678-90ab-cdef-0123-456789abcdef", _gpus()).index == 1
    assert gs.resolve("gpu-12345678", _gpus()).index == 1


def test_resolve_by_name_substring():
    assert gs.resolve("3090", _gpus()).index == 1
    assert gs.resolve("rtx 3060", _gpus()).index == 0


def test_resolve_ambiguous_name_is_refused():
    with pytest.raises(gs.SelectionError, match="matches 2 GPUs"):
        gs.resolve("GeForce", _gpus())


def test_resolve_unknown_index_uuid_or_name():
    for spec in ("7", "GPU-deadbeef", "4090"):
        with pytest.raises(gs.SelectionError):
            gs.resolve(spec, _gpus())


def test_resolve_with_no_gpus():
    with pytest.raises(gs.SelectionError, match="no NVIDIA GPU"):
        gs.resolve("0", [])


# --- defaults and --all ----------------------------------------------------


def test_default_spec_follows_cuda_visible_devices():
    assert gs.default_spec({}) == "0"
    assert (
        gs.default_spec({"CUDA_VISIBLE_DEVICES": "GPU-12345678, 0"}) == "GPU-12345678"
    )
    assert gs.default_spec({"CUDA_VISIBLE_DEVICES": ""}) == "0"


def test_select_all_returns_every_card_in_pci_order():
    assert [g.index for g in gs.select(None, True, _gpus())] == [0, 1]


def test_select_device_returns_one(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gs.select("GPU-1234", False, _gpus())[0].index == 1
    assert gs.select(None, False, _gpus())[0].index == 0


def test_force_pci_bus_order(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    assert gs.force_pci_bus_order() == "FASTEST_FIRST"
    assert gs.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert gs.force_pci_bus_order() is None


# --- torch_index: by UUID, never by position --------------------------------


def test_torch_index_maps_by_uuid_when_visible_devices_renumber(monkeypatch):
    gpus = _gpus()
    # CUDA_VISIBLE_DEVICES=<3090 Ti uuid>: torch sees one card, as cuda:0.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpus[1].uuid)
    assert gs.torch_index(FakeTorch([gpus[1]]), gpus[1]) == 0


def test_torch_index_hidden_card_is_an_error(monkeypatch):
    gpus = _gpus()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpus[1].uuid)
    with pytest.raises(gs.SelectionError, match="not visible to CUDA"):
        gs.torch_index(FakeTorch([gpus[1]]), gpus[0])


def test_torch_index_without_uuid_falls_back_only_when_unambiguous(monkeypatch):
    gpus = _gpus()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gs.torch_index(FakeTorch(gpus, with_uuid=False), gpus[1]) == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(gs.SelectionError, match="cannot locate"):
        gs.torch_index(FakeTorch([gpus[1]], with_uuid=False), gpus[1])
