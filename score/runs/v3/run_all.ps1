# PREREG v3 runner: prefetch CT, run infer.py on every frozen v3 run-plan piece (CPU only), score with the frozen
# scorer. Detached: Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy',
# 'Bypass','-File','C:\VesuviusTemp\strips\score\runs\v3\run_all.ps1'
#
# Stages (status.json "stage"): verify (FREEZE.txt) -> metadata (CT cache .zarray) -> preflight (model + CT load, CPU)
# -> prefetch primary, secondary, 0332 (<= 2 HTTP connections, stops below 10 GiB free on C:) -> infer (one
# infer.py process at a time, BelowNormal, 8 threads, CUDA hidden; finished pieces skipped) -> score -> finished.
# DONE is always written with the final rc (0 all done; 1 some pieces or a score failed; 2 CUDA visible;
# 3 low disk; 4 prefetch incomplete; 5 FREEZE mismatch; 10 other error). Resume: run the same command again.
# The runner never reads strip outputs: infer.py's stdout goes to pieces\<set>\<piece>.out, the scorers' to
# score_<set>.out; only return codes, timings and planned counts are logged.

$ErrorActionPreference = 'Stop'
$Root = 'C:\VesuviusTemp\strips'
$Score = Join-Path $Root 'score'
$Run = Join-Path $Score 'runs\v3'
$Log = Join-Path $Run 'log.txt'
$StatusFile = Join-Path $Run 'status.json'
$DoneFile = Join-Path $Run 'DONE'
$Py = 'C:\Users\PC\miniconda3\envs\vesuvius\python.exe'
$Checkpoint = 'scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e'
$Cache = 'C:/VesuviusTemp/strips/ct_cache'
$Sets = [ordered]@{
    'primary'   = @{ runplan = 'results\runplan_0139_primary.tsv'; afv = 'data\matched_0139.afv'; scroll = '0139' }
    'secondary' = @{ runplan = 'results\runplan_0139_secondary_run40.tsv'; afv = 'data\matched_0139_secondary_run40.afv'; scroll = '0139' }
    '0332'      = @{ runplan = 'results\runplan_0332_crossings.tsv'; afv = 'data\crossings_0332.afv'; scroll = '0332' }
}
$Vol = @{
    '0139' = @{ ct = 's3://vesuvius-challenge-open-data/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr'
                umb = 'C:/VesuviusTemp/strips/data/umbilicus/PHerc0139_20260102150214_umbilicus_L2.json' }
    '0332' = @{ ct = 's3://vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr'
                umb = 'C:/VesuviusTemp/strips/data/umbilicus/PHerc0332_20251211183505_umbilicus_L2.json' }
}

$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HOME = Join-Path $Root 'hf_home'
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$env:OPENBLAS_NUM_THREADS = '8'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'

$State = [ordered]@{ stage = 'start'; pid = $PID; started = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); rc = $null
                     pieces = [ordered]@{}; score = [ordered]@{} }
$rc = 0

function Write-Log([string]$msg) {  # never fatal: a reader holding the file must not stop the run
    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    foreach ($i in 1..5) {
        try { Add-Content -Path $Log -Value $line -Encoding UTF8 -ErrorAction Stop; return } catch { Start-Sleep -Seconds 1 }
    }
}

function Save-Status {  # never fatal
    try {
        $State.updated = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
        $out = [ordered]@{}
        foreach ($k in $State.Keys) { $out[$k] = $State[$k] }
        if (Test-Path $StatusFile) {  # keep the prefetch progress the Python prefetcher writes
            try { $old = Get-Content $StatusFile -Raw | ConvertFrom-Json; if ($old.prefetch) { $out['prefetch'] = $old.prefetch } } catch { }
        }
        $tmp = "$StatusFile.tmp"
        ($out | ConvertTo-Json -Depth 6) | Set-Content -Path $tmp -Encoding UTF8 -ErrorAction Stop
        Move-Item -Path $tmp -Destination $StatusFile -Force -ErrorAction Stop
    } catch { }
}

function Invoke-Py([string[]]$ArgList, [string]$OutFile, [string]$ErrFile) {
    $p = Start-Process -FilePath $Py -ArgumentList $ArgList -NoNewWindow -PassThru -WorkingDirectory $Score `
        -RedirectStandardOutput $OutFile -RedirectStandardError $ErrFile
    $null = $p.Handle
    try { $p.PriorityClass = 'BelowNormal' } catch { }
    $p.WaitForExit()
    return [int]$p.ExitCode
}

function Free-GiB { return [math]::Round((Get-PSDrive C).Free / 1GB, 2) }

New-Item -ItemType Directory -Force -Path $Run | Out-Null
if (Test-Path $DoneFile) { Move-Item $DoneFile ("$DoneFile.prev_" + (Get-Date -Format 'yyyyMMdd_HHmmss')) -Force }
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'

try {
    Write-Log "START v3 runner pid $PID (BelowNormal), CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES, HF_HUB_OFFLINE=1, C: $(Free-GiB) GiB free"
    $git = 'unknown'
    try { $git = (& git -C (Join-Path $Root 'villa_6f2') rev-parse HEAD) } catch { }
    Write-Log "villa worktree $Root\villa_6f2 at $git (+ #2163 + normalization patch, as G0); checkpoint $Checkpoint"
    Save-Status

    # verify: every frozen file is as frozen
    $State.stage = 'verify'; Save-Status
    $r = Invoke-Py @('-I', 'tools\write_freeze.py', '--verify') (Join-Path $Run 'verify.out') (Join-Path $Run 'verify.err')
    Write-Log ("verify FREEZE.txt: rc $r; " + ((Get-Content (Join-Path $Run 'verify.out') -Raw) -replace "`r?`n", ' '))
    if ($r -ne 0) { $rc = 5; throw 'FREEZE mismatch: a frozen file differs' }

    # metadata: the CT cache .zarray/.zattrs of both volumes (villa's publish_metadata)
    $State.stage = 'metadata'; Save-Status
    $r = Invoke-Py @('-I', '-u', 'runs\v3\prefetch_v3.py', '--init-metadata', '--log', $Log) (Join-Path $Run 'metadata.out') (Join-Path $Run 'metadata.err')
    if ($r -ne 0) { $rc = 10; throw "cache metadata failed (rc $r)" }

    # preflight: torch CPU only, pinned villa, checkpoint offline, CT metadata; no inference
    $State.stage = 'preflight'; Save-Status
    foreach ($s in @('0139', '0332')) {
        $o = Join-Path $Run "preflight_$s.out"
        $r = Invoke-Py @('-I', '-u', 'runs\v3\infer_cpu.py', '--preflight', $Vol[$s].ct, $Vol[$s].umb) $o (Join-Path $Run "preflight_$s.err")
        Write-Log ("preflight $s rc $r " + ((Get-Content $o -Raw) -replace "`r?`n", ' '))
        if ($r -eq 2) { $rc = 2; throw 'torch sees CUDA' }
        if ($r -ne 0) { $rc = 10; throw "preflight $s failed (rc $r)" }
    }

    # prefetch: primary, secondary, 0332
    foreach ($set in $Sets.Keys) {
        $State.stage = "prefetch:$set"; Save-Status
        Write-Log "prefetch ${set}: start (C: $(Free-GiB) GiB free)"
        $r = Invoke-Py @('-I', '-u', 'runs\v3\prefetch_v3.py', '--set', $set, '--workers', '2', '--min-free-gib', '10', '--log', $Log, '--status', $StatusFile) (Join-Path $Run "prefetch_$set.out") (Join-Path $Run "prefetch_$set.err")
        Write-Log "prefetch ${set}: rc $r (C: $(Free-GiB) GiB free)"
        if ($r -eq 3) { $rc = 3; throw "C: below 10 GiB free during prefetch $set" }
        if ($r -ne 0) { $rc = 4; throw "prefetch $set failed (rc $r)" }
    }

    # infer: one process at a time
    $failed = 0
    foreach ($set in $Sets.Keys) {
        $cfg = $Sets[$set]
        $rows = @(Import-Csv -Path (Join-Path $Score $cfg.runplan) -Delimiter "`t" | Where-Object { $_.run -eq 'True' } | Sort-Object piece)
        $plannedDec = ($rows | Measure-Object -Property decisions -Sum).Sum
        $st = [ordered]@{ planned = $rows.Count; decisions = $plannedDec; ok = 0; skipped = 0; failed = 0; failed_pieces = @() }
        $State.pieces[$set] = $st
        $State.stage = "infer:$set"; Save-Status
        Write-Log "infer ${set}: $($rows.Count) pieces, $plannedDec decisions"
        New-Item -ItemType Directory -Force -Path (Join-Path $Run "strips\$set"), (Join-Path $Run "records\$set"), (Join-Path $Run "pieces\$set") | Out-Null
        $v = $Vol[$cfg.scroll]
        foreach ($row in $rows) {
            $piece = $row.piece
            $out = Join-Path $Run "strips\$set\$piece"
            $rec = Join-Path $Run "records\$set\$piece.json"
            $done = $false
            if ((Test-Path $rec) -and (Test-Path (Join-Path $out 'strip.npz')) -and (Test-Path (Join-Path $out 'strip\meta.json'))) {
                try { $done = ((Get-Content $rec -Raw | ConvertFrom-Json).status -eq 'ok') } catch { $done = $false }
            }
            if ($done) { $st.skipped++; Write-Log "infer $set ${piece}: done already, skipped"; Save-Status; continue }
            $args_ = @('-I', '-u', 'runs\v3\infer_cpu.py', $rec, '--', '--afv', (Join-Path $Score $cfg.afv), '--fiber', $row.afv_id,
                       '--arc', $row.arc_start_patch_vox, $row.arc_end_patch_vox, '--out', $out, '--checkpoint', $Checkpoint,
                       '--ct', $v.ct, '--umbilicus', $v.umb, '--cache-dir', $Cache, '--device', 'cpu', '--index', 'none',
                       '--no-render', '--step', '8', '--blend', '16', '--trim', '75', '--batch', '32')
            $r = 1
            foreach ($attempt in 1..2) {
                $t0 = Get-Date
                $r = Invoke-Py $args_ (Join-Path $Run "pieces\$set\$piece.out") (Join-Path $Run "pieces\$set\$piece.err")
                $sec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
                $rate = if ($sec -gt 0) { [math]::Round([double]$row.decisions / $sec, 3) } else { 0 }
                Write-Log "infer $set $piece (attempt $attempt): rc $r, $($row.decisions) decisions, $sec s, $rate decisions/s incl. start-up"
                if ($r -eq 2) { $rc = 2; throw 'torch sees CUDA' }
                if ($r -eq 0) { break }
            }
            if ($r -eq 0) { $st.ok++ } else { $st.failed++; $st.failed_pieces += $piece; $failed++ }
            Save-Status
        }
        Write-Log "infer ${set}: ok $($st.ok), skipped $($st.skipped), failed $($st.failed)"
    }

    # score with the frozen scorer (reports in score\results\v3)
    $State.stage = 'score'; Save-Status
    $res = Join-Path $Score 'results\v3'
    New-Item -ItemType Directory -Force -Path $res | Out-Null
    $common = @('--fibers', 'data\fibers0139', '--segments', 'data\segments', '--umbilicus',
                'data\umbilicus\PHerc0139_20260102150214_umbilicus.json', '--crossings')
    $jobs = [ordered]@{
        'primary'   = @('-I', '-u', 'score_strips.py', 'score') + $common + @('--matches', 'results\matched_stretches.json',
                       '--strips', (Join-Path $Run 'strips\primary'), '--lasagna-normals', 'data\lasagna_normals_0139_primary.npz',
                       '--out', (Join-Path $res 'primary_case_studies.json'))
        'secondary' = @('-I', '-u', 'score_strips.py', 'score') + $common + @('--matches', 'results\matched_stretches_secondary_run40.json',
                       '--run-only', '--strips', (Join-Path $Run 'strips\secondary'), '--lasagna-normals',
                       'data\lasagna_normals_0139_secondary_run40.npz', '--out', (Join-Path $res 'secondary_report.json'))
        '0332'      = @('-I', '-u', 'score_strips.py', 'crossings', '--fibers', 'data\fibers0332', '--space',
                       'PHerc0332/20251211183505@L0', '--plan', 'results\crossings_0332_plan.json', '--strips',
                       (Join-Path $Run 'strips\0332'), '--out', (Join-Path $res 'crossings_0332_report.json'))
    }
    $scoreFailed = 0
    foreach ($k in $jobs.Keys) {
        $t0 = Get-Date
        $r = Invoke-Py $jobs[$k] (Join-Path $Run "score_$k.out") (Join-Path $Run "score_$k.err")
        $sec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        $State.score[$k] = [ordered]@{ rc = $r; seconds = $sec }
        Write-Log "score ${k}: rc $r in $sec s (report in results\v3; not read by the runner)"
        if ($r -ne 0) { $scoreFailed++ }
        Save-Status
    }
    if ($failed -gt 0 -or $scoreFailed -gt 0) { $rc = 1 }
    $State.stage = 'finished'
}
catch {
    if ($rc -eq 0) { $rc = 10 }
    $State.stage = "failed: $($State.stage)"
    $State.error = "$_"
    Write-Log "ERROR (rc $rc): $_"
}
finally {
    $State.rc = $rc
    try { Save-Status } catch { }
    foreach ($i in 1..5) {
        try {
            Set-Content -Path $DoneFile -Value ("rc=$rc stage=$($State.stage) finished=" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8 -ErrorAction Stop
            break
        } catch { Start-Sleep -Seconds 2 }
    }
    Write-Log "DONE rc $rc"
}
exit $rc
