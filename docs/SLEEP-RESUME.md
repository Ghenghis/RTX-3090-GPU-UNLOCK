# Sleep, Resume, and Hibernate — the honest limits

This is the most common way a working unlock "silently stops working." Read it before
you rely on the unlock across a sleep/wake cycle.

---

## 1. The core fact

The tensor selectors are **volatile**. They reset to stock whenever the GPU loses power
or takes a Function-Level Reset. Several power states do exactly that:

| Power transition | Selectors after | Re-applied automatically? |
| --- | --- | --- |
| Full shutdown → cold boot | reset to stock | **Yes** — the pre-boot step runs on the next boot |
| Reboot | reset to stock | **Yes** — pre-boot runs again |
| Sleep (suspend-to-RAM, S3/S0ix) | often reset | **No** on Windows today |
| Hibernate (S4) | reset to stock | **No** — not covered |

The pre-boot step only runs during **boot**. Sleep and hibernate resume paths do not
pass through it, so on Windows the unlock does **not** come back by itself after a
sleep/wake. You can be running stock while believing you are unlocked.

> The upstream Linux module re-applies after a suspend/resume; the Windows pre-boot
> route has no equivalent hook today. This is a real gap, not a misconfiguration.

---

## 2. What to do about it (today)

Until a supported resume re-apply exists, pick one:

1. **Use full shutdown + cold boot instead of sleep** on the machine while you rely on
   the unlock. This is the reliable path: every cold boot re-applies.
2. **Disable sleep and hibernate** for that machine/profile so you are never silently
   dropped to stock:
   - Set power plan sleep to "Never" for the relevant profile.
   - Disable hibernate: `powercfg /hibernate off` (admin).
3. **Always confirm after any wake** with the post-boot verifier before you trust the
   throughput:
   ```powershell
   python tools\verify_unlock.py --log unlock.log
   ```
   If it reports `STOCK`, reboot to re-apply.

---

## 3. Why a userland re-apply is hard (so you don't wait for a silver bullet)

The write has to happen after firmware init but **before** the NVIDIA driver binds the
card. After resume, the driver is already bound, so a normal Windows service or script
cannot reach that window — the same reason the unlock cannot be a background service in
the first place. Any future resume support would need a different, lower-level hook and
must be treated as experimental until proven on specific hardware. Setting expectations
honestly here is part of the spec; see [`RELIABILITY.md`](RELIABILITY.md).

---

## 4. Summary

- Cold boot: unlock re-applies. Reliable.
- Sleep / hibernate: unlock is lost and does **not** auto-return on Windows.
- Safe habit: disable sleep/hibernate on the unlocked machine, or verify after every
  wake and reboot if stock.
