#requires -Version 5.1
<#
.SYNOPSIS
    Read-only pre-flight check for Windows-GPU-UNLOCK.

.DESCRIPTION
    Reports whether this machine meets the prerequisites and whether your recovery
    path is ready. It is strictly READ-ONLY: it never unlocks, changes a setting,
    touches firmware, or modifies BitLocker/Secure Boot. It only looks and reports.

    Run it before you consider any boot-path change:
        pwsh -File scripts/doctor.ps1

.NOTES
    Exit code 0 = no blocking problems found. Exit code 1 = at least one FAIL.
    A PASS here is necessary, not sufficient: success is still per-system, and you
    must read docs/RECOVERY.md and docs/BITLOCKER-SAFETY.md before proceeding.
#>

[CmdletBinding()]
param(
    [string] $SystemDrive = $env:SystemDrive
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:Results = [System.Collections.Generic.List[object]]::new()

function Add-Check {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [ValidateSet('PASS', 'WARN', 'FAIL', 'INFO')] [string] $Status,
        [string] $Detail = '',
        [string] $Fix = ''
    )
    $script:Results.Add([pscustomobject]@{
        Name = $Name; Status = $Status; Detail = $Detail; Fix = $Fix
    })
}

function Test-Command {
    param([string] $Name)
    try { return [bool](Get-Command $Name -ErrorAction Stop) } catch { return $false }
}

# --- Python ---------------------------------------------------------------
try {
    if (Test-Command 'python') {
        $pv = (& python --version 2>&1) -replace '^Python\s*', ''
        $ok = $false
        try { $ok = [version]$pv -ge [version]'3.10' } catch { $ok = $false }
        if ($ok) { Add-Check 'Python >= 3.10' 'PASS' "found $pv" }
        else     { Add-Check 'Python >= 3.10' 'FAIL' "found $pv" 'Install Python 3.10+ from python.org and re-run.' }
    } else {
        Add-Check 'Python >= 3.10' 'FAIL' 'python not on PATH' 'Install Python 3.10+ and ensure it is on PATH.'
    }
} catch {
    Add-Check 'Python >= 3.10' 'FAIL' $_.Exception.Message 'Install Python 3.10+.'
}

# --- Toolchain (clang / lld-link 20.1.8) ----------------------------------
foreach ($tool in 'clang', 'lld-link') {
    if (Test-Command $tool) {
        $ver = ''
        try { $ver = (& $tool --version 2>&1 | Select-Object -First 1) } catch { }
        if ($ver -match '20\.1\.8') { Add-Check "$tool 20.1.8" 'PASS' "$ver" }
        else { Add-Check "$tool 20.1.8" 'WARN' "found: $ver" 'Builds are pinned to clang/lld-link 20.1.8; other versions may not reproduce.' }
    } else {
        Add-Check "$tool 20.1.8" 'WARN' 'not on PATH' 'Needed only to BUILD. Install LLVM 20.1.8 when you build locally.'
    }
}

# --- Secure Boot state (read-only) ----------------------------------------
try {
    $sb = Confirm-SecureBootUEFI -ErrorAction Stop
    if ($sb) {
        Add-Check 'Secure Boot' 'INFO' 'ENABLED' 'The unlock boot path requires Secure Boot OFF. See SECURITY.md for the cost before changing this.'
    } else {
        Add-Check 'Secure Boot' 'INFO' 'DISABLED' 'Already off. Re-enable it when you are done (SECURITY.md).'
    }
} catch {
    Add-Check 'Secure Boot' 'WARN' 'could not read (non-UEFI boot or access denied)' 'Run elevated on a UEFI system to read this.'
}

# --- BitLocker state (read-only) ------------------------------------------
try {
    if (Test-Command 'Get-BitLockerVolume') {
        $sys = $SystemDrive.TrimEnd('\')
        $vol = Get-BitLockerVolume -MountPoint $sys -ErrorAction Stop
        $ps = [string]$vol.ProtectionStatus
        if ($ps -eq 'On') {
            Add-Check "BitLocker ($sys)" 'WARN' 'protection ON' 'Save your recovery key OFF this machine and SUSPEND protection before any boot change. See docs/BITLOCKER-SAFETY.md.'
        } else {
            Add-Check "BitLocker ($sys)" 'INFO' "protection $ps" 'Protection is not active; still keep any recovery key safe.'
        }
    } else {
        Add-Check 'BitLocker' 'INFO' 'BitLocker cmdlets unavailable (edition without BitLocker)' ''
    }
} catch {
    Add-Check 'BitLocker' 'WARN' "could not read: $($_.Exception.Message)" 'Run elevated to read BitLocker status.'
}

# --- A spare FAT32 removable volume for safe USB testing -------------------
try {
    $fat = Get-Volume -ErrorAction Stop |
        Where-Object { $_.FileSystem -eq 'FAT32' -and $_.DriveType -eq 'Removable' }
    if ($fat) {
        $sz = [math]::Round(($fat | Select-Object -First 1).Size / 1GB, 1)
        Add-Check 'Removable FAT32 (test route)' 'PASS' "found ($sz GB)" 'Test from this removable stick so you can physically pull it to recover.'
    } else {
        Add-Check 'Removable FAT32 (test route)' 'WARN' 'none detected' 'Prepare a FAT32 USB stick; the USB route is the safe way to test (docs/RECOVERY.md).'
    }
} catch {
    Add-Check 'Removable FAT32 (test route)' 'INFO' 'could not enumerate volumes' ''
}

# --- Installed NVIDIA GPU(s) vs reported-working families ------------------
# Read-only inventory. "Reported family" means the project has evidence for SOME
# board with that silicon; it is NOT a guarantee for YOUR exact board.
try {
    $gpus = Get-CimInstance Win32_VideoController -ErrorAction Stop |
        Where-Object { $_.Name -match 'NVIDIA|GeForce|RTX' }
    if (-not $gpus) {
        Add-Check 'NVIDIA GPU' 'WARN' 'none detected via WMI' 'Confirm the card and driver are installed.'
    }
    foreach ($g in $gpus) {
        $name = $g.Name
        $fam = switch -Regex ($name) {
            '3090 Ti|3090|3080 Ti|3080'      { 'GA102 (reported family)'; break }
            '3070 Ti|3070|3060 Ti'           { 'GA104 (reported family)'; break }
            '3060'                           { 'GA106 (reported family)'; break }
            default                          { $null }
        }
        if ($fam) {
            Add-Check "GPU: $name" 'INFO' $fam 'Reported family != your exact board. Confirm your motherboard+BIOS profile in docs/COMPATIBILITY.en.md before any attempt.'
        } else {
            Add-Check "GPU: $name" 'WARN' 'not in a reported family' 'No reported profile matches this silicon. Report compatibility; do not attempt blindly.'
        }
    }
} catch {
    Add-Check 'NVIDIA GPU' 'INFO' 'could not enumerate video controllers' ''
}

# --- Recovery readiness (cannot be auto-verified; prompt the human) --------
Add-Check 'Recovery key saved off-machine' 'INFO' 'cannot be auto-verified' 'Confirm manually: you MUST have your BitLocker recovery key stored elsewhere. docs/RECOVERY.md'

# --- Report ---------------------------------------------------------------
$pad = ($script:Results.Name | Measure-Object -Maximum -Property Length).Maximum
Write-Host ''
Write-Host 'Windows-GPU-UNLOCK pre-flight (read-only)' -ForegroundColor Cyan
Write-Host ('-' * 60)
foreach ($r in $script:Results) {
    $color = switch ($r.Status) { 'PASS' {'Green'} 'WARN' {'Yellow'} 'FAIL' {'Red'} default {'Gray'} }
    $line = '{0}  {1}  {2}' -f $r.Status.PadRight(4), $r.Name.PadRight($pad), $r.Detail
    Write-Host $line -ForegroundColor $color
    if ($r.Fix) { Write-Host ('       -> ' + $r.Fix) -ForegroundColor DarkGray }
}
Write-Host ('-' * 60)

$fails = @($script:Results | Where-Object Status -eq 'FAIL').Count
$warns = @($script:Results | Where-Object Status -eq 'WARN').Count
Write-Host ("Summary: {0} FAIL, {1} WARN" -f $fails, $warns)
Write-Host 'This check is read-only and never modifies your system.' -ForegroundColor DarkGray
Write-Host 'Before any boot change, read docs/RECOVERY.md and docs/BITLOCKER-SAFETY.md.' -ForegroundColor DarkGray

exit ([int]($fails -gt 0))
