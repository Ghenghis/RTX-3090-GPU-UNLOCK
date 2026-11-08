# Changelog

All notable changes to this repository are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adopt [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> The upstream engineering history is preserved in `CHANGES.md`. This file tracks
> releases of **this** repository from the quality/trust foundation forward.

## [Unreleased]

### Added — card selection and per-card calibration (2026-11-08)
- `tools/gpu_select.py` — shared card selection for the verifier, monitor and benchmark:
  forces `CUDA_DEVICE_ORDER=PCI_BUS_ID`, resolves `--device` (nvidia-smi index, GPU UUID or
  unique name substring), locates the card in torch by UUID (never by position), and refuses
  a card that `CUDA_VISIBLE_DEVICES` hides or a name that matches several cards.
- `--device` and `--all` on `tools/verify_unlock.py`, `tools/live_monitor.py` and
  `benchmark/gemm_bench.py`; every result names the card measured (name, UUID, PCI bus).
  With `--all`, the verifier exits with the worst code over all cards, and the benchmark
  writes one file per card.
- `verify_unlock.py --calibrate` stores this card's stock-boot BF16 TFLOPS by GPU UUID
  (median of 3 runs; `--calibration` / `$RTX_UNLOCK_CALIBRATION` / per-user config dir).
  Later runs compare against it; the spec table is used only when the card has no
  calibration. It refuses a busy card or a value that looks unlocked unless `--force` is given.
  A corrupt calibration file gives INCONCLUSIVE, never a silent table fallback.
- `verify_unlock.py` reports INCONCLUSIVE instead of STOCK when the card was ≥20 % busy
  before measuring; `gemm_bench.py` records `util_before_pct` and warns.
- `compare_runs.py` reports INCONCLUSIVE when the two runs carry different `device_uuid`s.
- `scripts/verify_release.py` — pass/fail check of `release/EFI/BOOT/BOOTX64.EFI` (`efi_sha256`,
  `efi_bytes`) and every `corresponding_sources` entry in `release/provenance.json`, with safe-path
  checks. `SOURCE-MANIFEST.json` is reported as CURRENT / STALE / ABSENT, as a warning only.
- Tests: `tools/test_gpu_select.py`, `benchmark/test_gemm_bench.py`,
  `scripts/test_verify_manifest.py`, `scripts/test_verify_release.py`, and calibration and `main()`
  coverage in the existing test files (82 tests, up from 32).

### Fixed (2026-11-08)
- `verify_manifest.py`: a relative `-m` is resolved under `--root` first, then from the current
  directory, so `-m tree/SOURCE-MANIFEST.json -r tree` works (it used to look for
  `tree/tree/SOURCE-MANIFEST.json` and exit 2).
- `gemm_bench.py` provenance (driver, clocks, power limit) came from nvidia-smi's first GPU,
  not the measured one; it is now queried for the measured card by UUID.
- `ruff format` applied with the CI-pinned ruff 0.6.9; the five E702 errors in
  `live_monitor.py` are fixed. mypy (`scripts tools`) is now clean but stays advisory in CI.

### Added — Phase 7 (live telemetry, started)
- `tools/live_monitor.py` — cross-platform (Windows + Linux) real-time monitor: live SM/mem
  clocks, power draw vs limit, temperature, fan, utilization and VRAM from `nvidia-smi`,
  plus a live BF16 GEMM probe that prints measured TFLOPS and a continuously-updated
  UNLOCKED / STOCK verdict with the ratio vs the card's stock reference. Measurement, not
  guesses. `--once` / `--json` / `--no-probe` modes; exit 0 / 3.
- `tools/test_live_monitor.py` — 8 unit tests over the CSV parser, verdict logic and
  render robustness; all passing.

### Added — Phase 3 (proof)
- `benchmark/gemm_bench.py` — reproducible GEMM throughput benchmark across BF16 / FP16 /
  TF32 / FP32 with full provenance (device, driver, CUDA, torch, SM clock, power limit).
- `benchmark/compare_runs.py` — locked-vs-unlocked comparison with an FP32 control-mode
  honesty check: reports INCONCLUSIVE rather than a flattering gain when the two runs
  weren't measured under comparable conditions. Exit 0 gain / 1 no-gain / 2 inconclusive.
- `benchmark/test_compare_runs.py` — 11 unit tests over the comparison logic; all passing.
- `docs/WHAT-BENEFITS.md` — which workloads actually gain (16-bit math) vs don't
  (quantized / memory-bound), how to maximize it, and per-card notes (3090 Ti vs 3060).
- `docs/BENCHMARK-HARNESS.md` — how to run a controlled locked-vs-unlocked comparison.

### Added — Phase 2 (always-working)
- `tools/verify_unlock.py` — post-boot verifier: measures BF16 GEMM throughput and
  classifies the card as UNLOCKED / STOCK / INCONCLUSIVE against a per-card reference,
  with a dated log line. Read-only; exit 0/1/2/3.
- `tools/test_verify_unlock.py` — 13 unit tests over the pure reference-match and
  classification logic (no GPU required); all passing.
- `docs/RELIABILITY.md` — the per-boot re-apply specified as a verified transaction
  (COMMIT / ABORT / TIMEOUT), post-conditions, idempotency, and fail-safe handoff.
- `docs/SLEEP-RESUME.md` — honest sleep/hibernate limits and the safe habits around them.

### Added — Phase 0 (never-brick) + Phase 1 (trust) foundation
- `SECURITY.md` — threat model, the real cost of disabling Secure Boot / BitLocker,
  trust boundary, and artifact-verification steps.
- `docs/RECOVERY.md` — recovery-first ladder so no single bad boot costs the machine.
- `docs/BITLOCKER-SAFETY.md` — recovery-key and suspend/resume checklist.
- `ARCHITECTURE.md` — component map and boot-flow diagram.
- `scripts/verify_manifest.py` — SHA-256 integrity check against `SOURCE-MANIFEST.json`
  (exit 0 pass / 1 drift / 2 environment error).
- `scripts/doctor.ps1` — read-only pre-flight: prerequisites, Secure Boot / BitLocker
  state, removable-FAT32 test route, installed-GPU family check. Never modifies anything.
- `.github/workflows/quality.yml` — lint (ruff/PSScriptAnalyzer), tooling tests,
  manifest integrity, secret scan, Markdown link check. Does not build the EFI payload.
- `.github/workflows/release.yml` — signed, provenance-backed **source** releases
  (checksums + SBOM + cosign + SLSA attestation).
- `.editorconfig` — deterministic formatting across C, PowerShell, Python, Markdown.

### Known
- Upstream's `SOURCE-MANIFEST.json` is its stale v0.3.0 inventory (`72c800e`); 61 of its 924
  entries differ from upstream `main`. It is kept exactly as upstream published it and reported as
  STALE (warning). The pass/fail integrity check is `scripts/verify_release.py`: the release EFI and
  its 57 corresponding sources against `release/provenance.json`. This repository has no manifest
  of its own, so the CI `manifest-integrity` job stays non-blocking.

[Unreleased]: https://github.com/Ghenghis/rtx-3090-gpu-unlock/commits/main
