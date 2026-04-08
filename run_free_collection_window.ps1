param([string]$ProjectRoot = $PSScriptRoot)

$ErrorActionPreference = "Continue"

$root    = (Resolve-Path $ProjectRoot).Path
$py      = Join-Path $root ".venv\Scripts\python.exe"
$logDir  = Join-Path $root "logs\free_collection"
$logPath = Join-Path $logDir "$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Run-Step([string]$script, [string[]]$extra = @()) {
    $out = & $py $script @extra 2>&1
    [pscustomobject]@{ Exit = $LASTEXITCODE; Text = ($out | Out-String).Trim() }
}

function Try-Popup([string]$msg, [bool]$ok) {
    try {
        Add-Type -AssemblyName PresentationFramework -ErrorAction Stop
        $img = if ($ok) { [System.Windows.MessageBoxImage]::Information } else { [System.Windows.MessageBoxImage]::Error }
        [System.Windows.MessageBox]::Show($msg, "REdit Free Collection", "OK", $img) | Out-Null
    } catch {
        # Non-interactive session (Task Scheduler) - skip popup silently
    }
}

$mutex   = [System.Threading.Mutex]::new($false, "Global\REditFreeCollectionMutex")
$hasLock = $false
try {
    $hasLock = $mutex.WaitOne(0, $false)
    if (-not $hasLock) {
        "Already running - skipped." | Set-Content $logPath -Encoding UTF8
        Try-Popup "A free collection run is already active. This run was skipped." $true
        exit 0
    }
    if (-not (Test-Path $py)) { throw "Python not found: $py" }
    Set-Location $root

    $pipeline = @(
        @{ Name = "Collector";  Script = "run_free_collection_schedule.py"
           Args = @("--output-dir", "data/raw/reddit_json", "--schedule-plan", "configs/schedules/schedule_plan.csv") },
        @{ Name = "History";    Script = "build_reddit_history.py";      Args = @() },
        @{ Name = "SubHealth";  Script = "build_subreddit_health.py";   Args = @() },
        @{ Name = "Prediction"; Script = "build_prediction_dataset.py"; Args = @() },
        @{ Name = "Validation"; Script = "validate_history_data.py";    Args = @() },
        @{ Name = "SQLite";     Script = "export_history_to_sqlite.py"; Args = @() }
    )

    $results = [ordered]@{}
    $ok      = $true
    foreach ($step in $pipeline) {
        if (-not $ok) { break }
        $path = Join-Path $root $step.Script
        if (-not (Test-Path $path)) { continue }
        $r = Run-Step $path $step.Args
        $results[$step.Name] = $r
        if ($r.Exit -ne 0) { $ok = $false }
    }

    $log = @("Started: $(Get-Date -Format o)", "Root: $root", "")
    $results.GetEnumerator() | ForEach-Object {
        $log += "$($_.Key): exit $($_.Value.Exit)"
        if ($_.Value.Text) { $log += $_.Value.Text; $log += "" }
    }
    $log | Set-Content $logPath -Encoding UTF8

    $summary = ($results.Values | ForEach-Object { $_.Text } | Where-Object { $_ }) -join "`n`n"
    if (-not $summary) { $summary = "Completed without output." }
    $exitCode = if ($ok) { 0 } else { 1 }
    Try-Popup "$(if ($ok) { 'Finished OK' } else { 'Failed' })`n`n$summary`n`nLog: $logPath" $ok
    exit $exitCode
}
catch {
    @("Started: $(Get-Date -Format o)", "Root: $root", "", ($_ | Out-String)) | Set-Content $logPath -Encoding UTF8
    Try-Popup "Crashed: $($_.Exception.Message)`n`nLog: $logPath" $false
    exit 1
}
finally {
    if ($hasLock) { $mutex.ReleaseMutex() | Out-Null }
    $mutex.Dispose()
}
