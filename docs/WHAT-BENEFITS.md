# What actually benefits from the unlock

The unlock speeds up exactly **one** thing: matrix math that runs in **FP16, BF16, or
TF32 with a 32-bit accumulator** — the "half row" NVIDIA caps on GeForce cards.
Everything else is unchanged. So the only question that matters is: *is my model's heavy
math running in 16-bit, or not?*

## The rule, in one table

| Runs in 16-bit tensor math → **gains** | Quantized or memory-bound → **~0%** |
| --- | --- |
| SDXL / image generation in ComfyUI (FP16/BF16) | 4-bit/8-bit GGUF in **llama.cpp / Ollama** |
| vLLM / HF Transformers in FP16 or BF16 | Token generation (words appearing one by one) |
| AWQ/GPTQ 4-bit **served through vLLM** (unpacks to 16-bit to compute) | INT8 / INT4 math (the unlock leaves it alone) |
| LoRA / full fine-tunes in BF16 | Gaming, plain FP32 CUDA |
| Long-context **prompt prefill**, FlashAttention | Single-stream chat (you wait on generation) |

Two reasons the right column doesn't move: quantized inference does its hot math on an
INT8 path the unlock doesn't touch, and **token generation is memory-bandwidth bound** —
the card re-reads the weights from memory for every token, and the unlock adds math, not
memory speed.

## What to run to get the gain

- **Image generation** — ComfyUI + SDXL in FP16. The most reliable win (a 3060 went
  21.8 s → 16.2 s per image, ~35%).
- **LLM inference** — use **vLLM** (or HF Transformers / TGI) with **FP16/BF16** weights,
  or a **4-bit AWQ/GPTQ** model **served through vLLM** (it computes in 16-bit). Avoid
  GGUF-in-llama.cpp if you want the unlock to matter.
- **Fine-tuning** — LoRA/QLoRA or full fine-tunes in **BF16** (a 3090 LoRA run was +41%
  at 420 W).
- **Where in the job** — the gain is in **prefill** (reading the prompt), not **decode**
  (writing the answer). So long prompts, RAG, document processing, and **batched**
  requests feel it; one-at-a-time chat does not.

## How to maximize it

1. **Choose a 16-bit compute path** (above). This is 90% of the result.
2. **Batch and go long-context.** More math per memory read moves you from memory-bound
   toward compute-bound, which is exactly what the unlock accelerates. FlashAttention
   gains grow with context (~3% empty → ~17% at 100k tokens).
3. **Give it power headroom — and this differs by card:**

| Your card | Power | Reality with the unlock |
| --- | --- | --- |
| **EVGA RTX 3090 Ti** (GA102, 450 W, triple-fan) | Raisable | The workhorse. Owners needed a **power-limit increase** to exceed ~25%; at its limit the card just drops clocks. Raise the limit, keep it cool → full ~1.8x on 16-bit workloads. |
| **ASUS Phoenix RTX 3060 V2** (GA106, 170 W fixed, single-fan) | **Fixed** | Math rate still ~doubles, but the 170 W cap can't be raised and the single fan forces a downclock (~1890→~1700 MHz) under sustained load. Great for SDXL; not the card for long heavy fine-tunes. |

4. **Confirm it applied** each boot with [`tools/verify_unlock.py`](../tools/verify_unlock.py),
   and **measure the real gain** on your own card with the
   [benchmark harness](BENCHMARK-HARNESS.md) — don't trust headline TFLOPS.

## Does Linux get it "all the time"?

- **Availability: effectively yes.** On Linux the module loads via initramfs and
  re-applies on **every boot** before the driver binds, and it **re-applies after
  suspend/resume** — so you don't silently lose it in daily use (Windows has no resume
  re-apply; see [`SLEEP-RESUME.md`](SLEEP-RESUME.md)).
- **But it is still volatile, not permanent.** It resets on a true power loss/FLR; the
  module just re-applies it automatically.
- **And the speedup is still workload-gated.** Linux does not make a GGUF chat model or
  token generation faster — the 16-bit-math rule applies everywhere. Linux's only
  advantage is that the unlock is reliably *present*; whether you *feel* it still depends
  on running FP16/BF16 work.

**Bottom line for your setup:** Linux + the **3090 Ti** + a power-limit bump + FP16/BF16
workloads (vLLM, ComfyUI, fine-tunes) is where the unlock becomes real speed rather than
a benchmark number.
