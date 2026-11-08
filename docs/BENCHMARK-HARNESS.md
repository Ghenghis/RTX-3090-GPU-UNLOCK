# Benchmark harness — measure the real gain, honestly

Headline TFLOPS and "nearly doubles" numbers come from peak microbenchmarks and
don't tell you what *your* card does. This harness measures a controlled
**locked-vs-unlocked** comparison on your own hardware and refuses to report a
flattering number if the measurement wasn't sound.

It only measures. It never unlocks, changes a setting, or touches firmware.

## Requirements

- Python 3.10+ and PyTorch with CUDA (`python -c "import torch; print(torch.cuda.is_available())"`).
- `nvidia-smi` on PATH (for driver/clock/power provenance; optional but recommended).

## Run a controlled comparison

Measure the **same card** in both states, back-to-back, at the **same power limit**,
with the machine otherwise idle. Pick the card by UUID (`nvidia-smi -L`); the tools force
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and record `device_uuid` and `pci_bus_id` in every run, and
`compare_runs.py` reports INCONCLUSIVE if the two runs come from different cards. `--all`
measures every card, writing one file per card (`locked.json` → `locked.GPU-xxxxxxxx.json`).

The unlock is applied before the OS boots, so the two states are two boots. On Windows, the
stock boot is F8 boot menu → Windows Boot Manager, which skips the unlock entry.

```bash
# 1. Stock boot (unlock skipped):
python benchmark/gemm_bench.py --device GPU-12345678 --state locked --out locked.json

# 2. Unlocked boot (normal restart through the unlock entry):
python benchmark/gemm_bench.py --device GPU-12345678 --state unlocked --out unlocked.json

# 3. Compare:
python benchmark/compare_runs.py locked.json unlocked.json
```

On the stock boot, also store the per-card baseline the verifier compares against:
`python tools/verify_unlock.py --calibrate --device GPU-12345678`. It refuses if the card
is busy or the value looks unlocked (override with `--force`).

Example output:

```
GEMM comparison on NVIDIA GeForce RTX 3090 Ti
  locked: 450 W / 1890 MHz   unlocked: 450 W / 1860 MHz
----------------------------------------------------------
mode       locked   unlocked    ratio  verdict
bf16         87.5      157.4    1.80x  gain
fp16         88.0      125.0    1.42x  gain
tf32         40.2       69.8    1.74x  gain
fp32         22.8       22.9    1.00x  control-flat
----------------------------------------------------------
GAIN CONFIRMED — BF16 1.80x with FP32 control flat. Measurement sound.
```

## How it stays honest

- **FP32 is the control.** The unlock does not touch FP32, so FP32 throughput must be
  the same in both runs. If it moved more than 10%, the two runs weren't comparable
  (different clocks, power, or thermals) and the tool reports **INCONCLUSIVE** instead
  of a gain — re-run both back-to-back at the same power limit.
- **BF16 is the headline.** It is the capped row the unlock targets, with FP32
  accumulation inherent to the BF16 tensor path — the cleanest signal.
- **Full provenance** is recorded in each JSON: device, driver, CUDA, torch, SM clock,
  power limit, timestamp. A result you can't attribute is a result you can't trust.

## Reading the verdicts

| Verdict | Meaning |
| --- | --- |
| `GAIN CONFIRMED` | BF16 ≥ 1.4x and FP32 control flat — real and sound |
| `PARTIAL` | BF16 moved but under 1.4x — usually power/clock limited; raise the power limit |
| `NO GAIN` | BF16 unchanged — unlock not active or not accepted; check `tools/verify_unlock.py` |
| `INCONCLUSIVE` | FP32 control drifted or data missing — fix the test, not the number |

## Interpreting across your two cards

- On the **3090 Ti**, expect the full picture above once you give it power headroom.
- On the **3060 Phoenix** (170 W fixed), expect a real but smaller BF16 ratio and a
  slight clock drop — the harness will show it as `gain` or `partial` honestly rather
  than pretending the 170 W cap isn't there.

Remember: a strong GEMM ratio here does **not** mean your chat model gets faster. See
[`WHAT-BENEFITS.md`](WHAT-BENEFITS.md) for which real workloads convert this into speed.
