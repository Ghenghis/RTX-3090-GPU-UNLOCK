# Security Policy

This project runs **before your operating system does**. A pre-boot EFI payload
that writes to GPU hardware registers is one of the highest-trust things you can
put on a machine. This document explains, honestly, what that costs you, what the
trust boundaries are, how to verify what you run, and how to report a problem.

Read it fully before you build, flash a USB stick, or change any boot setting.

---

## 1. What this project is — and is not

- It re-applies NVIDIA's full-rate FP16/BF16/TF32 tensor selectors at boot, before
  the stock driver binds. It does **not** flash your VBIOS and does **not** patch
  the NVIDIA driver.
- The unlock is **volatile**. It is cleared by any full power-off, reboot, or
  Function-Level Reset. There is no permanent change to your card.
- Success is **per-system**, not per-model. A result on one motherboard + BIOS +
  card + boot route says nothing about a different combination.

This repository does not redistribute NVIDIA firmware, PROM/BIOS images, drivers,
or CUDA DLLs. Releases are **source-only** by design.

---

## 2. The real security cost (do not skip this)

To run the pre-boot handoff on Windows you will be asked to change two settings.
Both are genuine, lasting reductions in your machine's security posture for as long
as the unlock is active:

| Setting | What changes | Why it matters |
| --- | --- | --- |
| **Secure Boot → OFF** | The firmware stops verifying boot-stage signatures | Secure Boot exists to block exactly this class of unsigned pre-OS code. With it off, a malicious bootkit would also be free to load. |
| **BitLocker → suspended / off** | Drive-encryption key release is no longer gated on an unchanged boot measurement | A changed boot chain (TPM PCR state) can trigger a recovery-key prompt — or lock you out if you do not have the key. |

If your machine holds work you cannot lose, weigh this before converting a one-off
test into an always-on boot entry. A dedicated or secondary machine, or a
removable-USB route you boot only when you need the throughput, keeps the security
downgrade from being permanent.

> A planned hardening path (enrolling your own key via MOK/shim so Secure Boot can
> stay **on**) is tracked in the roadmap. Until that lands, Secure Boot off is the
> real cost.

---

## 3. Trust boundary

When you run a build from this repository you are trusting, in order:

1. **The source** you built from — verify it (section 4).
2. **The toolchain** that compiled it (clang/lld-link, Python, EDK2).
3. **The person or CI that produced any binary you did not build yourself.**

The safest posture is to **build from verified source on your own machine** and run
only what you built. Treat any pre-compiled `BOOTX64.EFI` offered by a third party
as untrusted until you have verified its provenance and, ideally, reproduced it from
source (see the reproducible-build workflow in the roadmap).

---

## 4. How to verify what you run

Before building or flashing anything:

```powershell
# 1. Verify the source tree matches its signed manifest (hashes every tracked file).
python -I scripts/verify_manifest.py

# 2. Check your machine meets the prerequisites and your recovery path is ready.
pwsh -File scripts/doctor.ps1
```

`verify_manifest.py` recomputes the SHA-256 of every file listed in
`SOURCE-MANIFEST.json` and fails loudly on any mismatch, missing, or unexpected
file. `doctor.ps1` is **read-only**: it reports your environment and refuses nothing
on your behalf — it does not unlock, change settings, or touch firmware.

Signed releases (when published) carry Sigstore/cosign signatures and SLSA build
provenance. Verify them before trusting any release artifact:

```bash
cosign verify-blob --bundle <artifact>.sigstore.json <artifact>
gh attestation verify <artifact> --owner Ghenghis
```

---

## 5. Recovery comes first

Do not attempt a boot-path change until you have read
[`docs/RECOVERY.md`](docs/RECOVERY.md) and
[`docs/BITLOCKER-SAFETY.md`](docs/BITLOCKER-SAFETY.md), and until you can answer
"yes" to all of:

- I have my BitLocker recovery key saved **off this machine**.
- I am testing from a **removable USB** stick I can physically pull to recover.
- I know how to select **Windows Boot Manager** from my firmware boot menu.

---

## 6. Reporting a vulnerability

If you find a security issue in this project's tooling (for example: the compatibility
collector, the profile generator, the verifier, the CI, or a way the boot flow could
fail unsafely), please report it privately rather than opening a public issue:

- Use GitHub's **private vulnerability reporting** (Security → Report a vulnerability)
  on this repository, **or**
- Open a minimal public issue that says only "security report, please enable private
  reporting" with no details.

Please include: affected file/area, the unsafe outcome, reproduction steps, and the
commit you tested. We aim to acknowledge within a reasonable window for a
community-maintained project. There is no bounty.

**Out of scope:** the fact that the tool disables Secure Boot / BitLocker (documented
and intentional), results that depend on an unsupported hardware configuration, and
anything requiring a pre-existing compromise of the machine.

---

## 7. Scope of this policy

This policy covers the source in this repository. It does not cover NVIDIA firmware,
the NVIDIA driver, EDK2, your motherboard firmware, or any third-party binary. Those
belong to their respective vendors and projects.
