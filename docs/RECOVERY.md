# Recovery Guide — never get stuck

This is the most important document in the repository. Read it **before** you change
any boot setting. The goal is simple: **no single bad boot should ever cost you your
machine or your data.**

If you only remember one thing: **test from a removable USB stick you can physically
pull out.** If a boot ever hangs, pulling the stick and powering off returns you to a
normal Windows boot.

---

## 0. Before you touch anything — the pre-flight

Do not proceed past this list until every box is true.

- [ ] My **BitLocker recovery key** is saved **somewhere other than this machine**
      (printed, or in another device / account). See
      [`BITLOCKER-SAFETY.md`](BITLOCKER-SAFETY.md).
- [ ] I am testing from a **removable USB** stick, not an internal partition.
- [ ] I know my motherboard's **boot-menu key** (commonly F8/F11/F12 — check your
      board) and I can select **Windows Boot Manager** from it.
- [ ] I have verified the source and my environment:
      `python -I scripts/verify_manifest.py` and `pwsh -File scripts/doctor.ps1`.
- [ ] I have a current backup of anything on this machine I cannot lose.

---

## 1. The recovery ladder (try in order)

### Rung 1 — Pull the USB and power off
If you tested from USB and the screen is black or stuck: **remove the USB stick**,
hold the power button until the machine is fully off, wait a few seconds, power on.
The firmware boots normally from your internal drive. The unlock leaves no permanent
change, so a clean power cycle returns you to stock.

### Rung 2 — Firmware boot menu → Windows Boot Manager
Power on and tap your boot-menu key. Choose **Windows Boot Manager** directly. This
bypasses the unlock entry for this boot.

### Rung 3 — Windows-first recovery entry (if you registered the boot entry)
If you had set up the automatic/registered boot entry and want Windows put first
again:

```powershell
.\internal-boot\Headless.ps1 -RestoreWindowsFirst
```

This places Windows Boot Manager first and does **not** depend on any experimental
EFI or build artifact — it works even without saved state. (Restoring a prior Fast
Startup setting does need the saved state, so keep it.)

### Rung 4 — BitLocker recovery prompt
If you see a BitLocker recovery screen, this is expected when the boot measurement
changed and protection was **not** suspended. Enter your recovery key (the one you
saved in pre-flight). Then, in Windows, suspend BitLocker properly before trying
again — see [`BITLOCKER-SAFETY.md`](BITLOCKER-SAFETY.md).

### Rung 5 — Re-enable Secure Boot and remove the boot entry
To return fully to stock security:

1. Enter firmware setup and set **Secure Boot → Enabled**.
2. Remove any added boot entry so Windows Boot Manager is the only/first option.
3. In Windows, **resume BitLocker protection** (`Resume-BitLocker -MountPoint C:`).

---

## 2. Symptom → action

| What you see | Do this |
| --- | --- |
| Black screen right after the unlock entry | Rung 1 (pull USB, power cycle) |
| Boots to firmware menu repeatedly | Rung 2, then Rung 5 |
| "Enter recovery key" (BitLocker) screen | Rung 4 |
| Windows boots but tensor speed is stock | Not a recovery problem — the unlock did not apply; re-check your profile/route. Do **not** keep retrying blindly. |
| Display never recovers after unlock | Fully power off, Rung 2, and report the exact config as a compatibility issue |

---

## 3. What recovery does **not** require

- It does **not** require the experimental EFI or any build output to be intact.
- It does **not** require the unlock to have succeeded.
- It does **not** make a permanent change you must "undo" at the hardware level — the
  selectors reset themselves on power-off.

The only thing recovery genuinely depends on is **you having your BitLocker recovery
key**. That is why it is step zero.

---

## 4. If you are about to make the unlock automatic ("always-on")

Only do this **after** a manual USB/F8 test has succeeded on your exact machine, and
only once you accept the standing security cost (Secure Boot off, BitLocker
suspended). Keep this document and your recovery key reachable from another device,
because an always-on boot entry runs on every cold boot.
