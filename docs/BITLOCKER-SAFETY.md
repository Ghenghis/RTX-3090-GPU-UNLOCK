# BitLocker Safety Checklist

The single most common "the mod bricked my PC" story is **not** a hardware failure.
It is a BitLocker recovery lockout: the tool changes the boot measurement, the TPM
refuses to release the key automatically, and the user does not have their recovery
key. This is entirely avoidable.

**Rule: never change a boot setting on a BitLocker-protected drive until you hold the
recovery key off-machine and have suspended protection.**

---

## 1. Find and save your recovery key (do this first)

Check whether the system drive is protected:

```powershell
Get-BitLockerVolume -MountPoint C:
```

If `ProtectionStatus` is `On`, get your recovery key and **save it off this machine**
(print it, or store it on a different device / your Microsoft account):

```powershell
(Get-BitLockerVolume -MountPoint C:).KeyProtector |
  Where-Object KeyProtectorType -eq 'RecoveryPassword' |
  Select-Object KeyProtectorId, RecoveryPassword
```

> If this returns nothing, your key may only live in your Microsoft account
> (https://aka.ms/myrecoverykey) or your organization's directory. Retrieve it there
> **before** continuing.

---

## 2. Suspend — do not disable — protection

Suspending lets BitLocker keep the data encrypted while temporarily not gating the
key on the boot measurement. This is what prevents the recovery prompt during your
test.

```powershell
# Suspend for the next reboot only (safest). Re-protects automatically after 1 reboot.
Suspend-BitLocker -MountPoint C: -RebootCount 1
```

`-RebootCount 1` means protection resumes by itself after a single reboot — so even
if you forget, you are not left unprotected indefinitely. If you expect several test
boots, raise the count deliberately, and resume manually when done.

Do **not** use "Turn off BitLocker" (full decryption) unless you specifically intend
to decrypt the whole drive — it is slow and leaves data unprotected.

---

## 3. After testing — resume protection

When you are done testing (or before you step away):

```powershell
Resume-BitLocker -MountPoint C:
Get-BitLockerVolume -MountPoint C:   # confirm ProtectionStatus is On again
```

---

## 4. If you reach a recovery screen anyway

- Enter the recovery key you saved in step 1.
- Boot into Windows, then **suspend properly** (step 2) before trying again.
- If you did not save the key and it is not in your Microsoft/organization account,
  stop. Do not keep rebooting. Recover the key first.

---

## 5. "Always-on" and BitLocker

An always-on unlock boot entry changes the measured boot chain on **every** cold
boot. With BitLocker on and not suspended, that can mean a recovery prompt every
time. Realistic options:

1. Keep BitLocker **suspended** while the unlock entry is active (accept reduced
   at-rest protection), **or**
2. Leave BitLocker **off** on that machine (accept no at-rest protection), **or**
3. Wait for the planned MOK/shim-signed path so the measured chain stays trusted and
   Secure Boot/BitLocker can remain on (tracked in the roadmap; not yet available).

Choose deliberately. Do not discover this on boot number two.

---

## 6. Quick reference

| Command | Purpose |
| --- | --- |
| `Get-BitLockerVolume -MountPoint C:` | Check protection status |
| `Suspend-BitLocker -MountPoint C: -RebootCount 1` | Suspend for one reboot |
| `Resume-BitLocker -MountPoint C:` | Re-enable protection |
| https://aka.ms/myrecoverykey | Retrieve a key saved to a Microsoft account |

`scripts/doctor.ps1` reports your BitLocker status read-only as part of its
pre-flight checks, but it will **never** change it for you — that decision is yours
and must be deliberate.
