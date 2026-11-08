# Reliability Spec — the per-boot re-apply as a verified transaction

"Always working" is not a permanence problem; it is a reliability problem. The unlock
is volatile and is re-applied on every cold boot, so the goal is simple to state:

> **Every boot either reaches a known-good unlocked state, or hands off to a normal
> Windows boot. It never leaves the card half-applied and never strands the machine.**

This document specifies the contract the pre-boot step should satisfy. It is a
design/verification spec at the level of observable post-conditions — the same
properties the project's own success reports already record — not an implementation
guide for the mechanism.

---

## 1. The re-apply as a transaction

Treat each boot's unlock attempt as a single transaction with three outcomes:

| Outcome | Meaning | Handoff |
| --- | --- | --- |
| **COMMIT** | All post-conditions verified | Hand off to Windows Boot Manager |
| **ABORT** | A pre-condition or post-condition failed | Fail safe: hand off to a normal Windows boot |
| **TIMEOUT** | No decisive result within the budget | Fail safe: hand off to a normal Windows boot |

There is no fourth outcome. In particular, there is no "commit anyway" path: a card
that cannot be verified as correctly released and unlocked is handed to Windows
**stock**, not half-modified.

---

## 2. Post-conditions (what "COMMIT" must verify)

A boot may only COMMIT if all of the following are observed and logged. These mirror
the properties the project already reports for a successful run:

- The target GPU's speed selectors read back their full-rate values
  (`SS0 = 0x88888888`, `SS1 = 0x8`).
- Bus-mastering / mapping state matches the expected released topology (e.g. BME as
  designed, **zero residual mappings**).
- Exactly **one** Function-Level Reset occurred — no more, no fewer.
- Cleanup completed; no buffers or ownership left dangling.
- The released topology was verified **before** control passes to Windows.

If any check cannot be made, the result is ABORT, not COMMIT.

---

## 3. Idempotency and determinism

- Running the attempt twice on the same boot must not change the outcome or leave extra
  state; a second pass either re-verifies the same COMMIT or ABORTs.
- The same machine + firmware + card + route must produce the same outcome boot to boot.
  A non-deterministic result is a defect to be root-caused, not retried blindly.
- The attempt must not depend on prior-boot state. Each boot starts from stock and must
  reach its conclusion from stock.

---

## 4. Fail-safe and watchdog

- A bounded time budget governs the whole attempt. On expiry, the flow ABORTs to a
  normal Windows boot rather than waiting.
- ABORT and TIMEOUT must route to Windows Boot Manager without requiring the operator
  to intervene. A single failed boot should self-heal on the next boot, not require a
  manual power-off and menu selection. (See roadmap item #12 and
  [`RECOVERY.md`](RECOVERY.md) for the manual ladder that always exists as a backstop.)
- The fail-safe path must not itself depend on any experimental artifact being intact.

---

## 5. Observability

Each boot should leave a record sufficient to answer "what happened and why":

- Outcome (COMMIT / ABORT / TIMEOUT) and which post-condition failed, if any.
- The verified selector/topology values on COMMIT.
- A timestamp and the active profile/route.

In Windows, [`tools/verify_unlock.py`](../tools/verify_unlock.py) provides the
independent, second confirmation: it measures BF16 throughput and classifies the card
as UNLOCKED / STOCK / INCONCLUSIVE, and appends a dated line to a log. The pre-boot
record says "I believe I committed"; the post-boot verifier says "the card is in fact
running unlocked." Both should agree.

---

## 6. What this spec deliberately does not cover

- It does not describe how the selectors are written or how the pre-driver window is
  obtained. That is the existing mechanism; this spec constrains its **outcomes and
  safety**, not its internals.
- It does not loosen any existing guard. Ownership, cold-state, immutable-selector,
  PROM and topology checks remain; this spec only formalizes that their failure means
  ABORT-to-Windows.
