# Operational wrapper around the frozen v3 runner (runs\v3\run_all.ps1, unchanged). It only retries:
# rc 3 (C: below 10 GiB, e.g. another program briefly filling C:) and rc 4 (prefetch incomplete, e.g. S3
# connection resets). Before each attempt it waits until C: has >= 15 GiB free. The runner resumes by itself
# (cached chunks and finished pieces are skipped). After a retried attempt, its DONE is moved to DONE.retry_<n>.
# V3_FINAL is written once, with the last attempt's rc, when no further retry will run.
# History (PREREG 10.7 item 7 and 10.8): first written 28 Oct 14:44 UTC. The CT cache was on D:\PrizeTemp\strips_ct_cache
# behind a junction at C:\VesuviusTemp\strips\ct_cache from 17:12 to 23:30 on 28 Oct (robocopy logs), because the page
# file on C: grows by ~12 GB at the machine's commit limit; from 17:10 to 23:30 this wrapper also waited for D: >= 25
# GiB, and no attempt started in that time. Since 23:30 the cache is a real folder on C: again. Pinned in FREEZE.txt
# from v3.2 on; the code is that of the first version (superseded_v31/).
# Detached: Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
# '-File','C:\VesuviusTemp\strips\score\runs\relaunch_v3.ps1'

$ErrorActionPreference = 'Continue'
$Run = 'C:\VesuviusTemp\strips\score\runs\v3'
$Done = Join-Path $Run 'DONE'
$Final = Join-Path $Run 'V3_FINAL'
$Log = 'C:\VesuviusTemp\strips\score\runs\relaunch_v3.log'
$MaxAttempts = 8

function L($m) { Add-Content -Path $Log -Value ('[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m) -Encoding UTF8 }
function FreeGiB { [math]::Round((Get-PSDrive C).Free / 1GB, 2) }

if (Test-Path $Final) { Move-Item $Final ("$Final.prev_" + (Get-Date -Format 'yyyyMMdd_HHmmss')) -Force }
L "relaunch wrapper pid $PID start"
$rc = 10
try {
    for ($i = 1; $i -le $MaxAttempts; $i++) {
        $waited = 0
        while ((FreeGiB) -lt 15) {
            if ($waited % 10 -eq 0) { L "attempt ${i}: waiting, C: $(FreeGiB) GiB free (< 15)" }
            Start-Sleep -Seconds 60; $waited++
        }
        L "attempt ${i}: start run_all.ps1 (C: $(FreeGiB) GiB free)"
        $p = Start-Process powershell -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $Run 'run_all.ps1') -WindowStyle Hidden -PassThru -Wait
        $rc = $p.ExitCode
        $line = if (Test-Path $Done) { (Get-Content $Done -Raw).Trim() } else { 'no DONE file' }
        L "attempt ${i}: exit $rc; DONE: $line"
        if (($rc -eq 3 -or $rc -eq 4) -and $i -lt $MaxAttempts) {
            if (Test-Path $Done) { Move-Item $Done (Join-Path $Run "DONE.retry_$i") -Force }
            Start-Sleep -Seconds 120
            continue
        }
        break
    }
}
finally {
    Set-Content -Path $Final -Value ("rc=$rc finished=" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8
    L "V3_FINAL rc $rc"
}
