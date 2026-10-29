# Prefetch-only pass with the frozen helper runs\v3\prefetch_v3.py (unchanged), so the CT download continues while
# the PREREG v3.1 erratum is written. No inference, no scoring. The cache is C:\VesuviusTemp\strips\ct_cache (on D:
# by junction 28 Oct 17:12-23:35, then moved back). The helper stops below 12 GiB free on C:, and C: must have
# >= 15 GiB free before each set. Writes runs\prefetch_standalone.done.
$ErrorActionPreference = 'Continue'
$Score = 'C:\VesuviusTemp\strips\score'
$Log = Join-Path $Score 'runs\prefetch_standalone.log'
$Done = Join-Path $Score 'runs\prefetch_standalone.done'
$Py = 'C:\Users\PC\miniconda3\envs\vesuvius\python.exe'
function L($m) { Add-Content -Path $Log -Value ('[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m) -Encoding UTF8 }
if (Test-Path $Done) { Remove-Item $Done -Force }
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$env:CUDA_VISIBLE_DEVICES = '-1'
$rc = 0
Set-Location $Score
foreach ($set in 'secondary', '0332') {
    $c = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
    if ($c -lt 15) { L "STOP before ${set}: C: $c GiB free (< 15)"; $rc = 3; break }
    L "prefetch ${set}: start (C: $c GiB free)"
    $p = Start-Process $Py -ArgumentList '-I', '-u', 'runs\v3\prefetch_v3.py', '--set', $set, '--workers', '2', '--min-free-gib', '12', '--log', $Log -WorkingDirectory $Score -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput (Join-Path $Score "runs\prefetch_standalone_$set.out") -RedirectStandardError (Join-Path $Score "runs\prefetch_standalone_$set.err")
    L "prefetch ${set}: rc $($p.ExitCode)"
    if ($p.ExitCode -ne 0) { $rc = $p.ExitCode; break }
}
Set-Content -Path $Done -Value ("rc=$rc finished=" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8
L "DONE rc $rc"
