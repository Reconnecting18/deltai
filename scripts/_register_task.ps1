$PythonPath = "C:\e3n\project\venv\Scripts\python.exe"
$ScriptPath = "C:\Users\ethan\OneDrive\Documents\GitHub\e3n\scripts\daily_training.py"
$LogPath    = "C:\e3n\data\training\daily_training.log"
$WorkDir    = "C:\Users\ethan\OneDrive\Documents\GitHub\e3n"
$TaskName   = "E3N-Daily-Training"

if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: Not running as Administrator. Re-launching elevated..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit 0
}

Write-Host "Running as Administrator." -ForegroundColor Green

if (-NOT (Test-Path $PythonPath)) {
    Write-Host "ERROR: Python venv not found: $PythonPath" -ForegroundColor Red
    Write-Host "Run: cd C:\e3n\project && python -m venv venv && .\venv\Scripts\activate && pip install torch transformers peft trl bitsandbytes trafilatura datasets"
    Read-Host "Press Enter to exit"
    exit 1
}

if (-NOT (Test-Path $ScriptPath)) {
    Write-Host "ERROR: Script not found: $ScriptPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$LogParent = Split-Path $LogPath
if (-NOT (Test-Path $LogParent)) {
    New-Item -ItemType Directory -Path $LogParent -Force | Out-Null
    Write-Host "Created log directory: $LogParent"
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task: $TaskName"
}

$Action   = New-ScheduledTaskAction -Execute $PythonPath -Argument "$ScriptPath >> $LogPath 2>&1" -WorkingDirectory $WorkDir
$Trigger  = New-ScheduledTaskTrigger -Daily -At 2:00AM
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 15) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)

Register-ScheduledTask `
    -TaskName    $TaskName `
    -Action      $Action `
    -Trigger     $Trigger `
    -Settings    $Settings `
    -Description "E3N daily autonomous training at 2 AM - web collection, distillation, QLoRA fine-tuning"

Write-Host ""
Write-Host "Task registered: $TaskName" -ForegroundColor Green
Write-Host "  Schedule : Daily at 2:00 AM"
Write-Host "  Python   : $PythonPath"
Write-Host "  Script   : $ScriptPath"
Write-Host "  Log      : $LogPath"
Write-Host "  WorkDir  : $WorkDir"
Write-Host ""
Write-Host "Verify:" -ForegroundColor Cyan
Write-Host "  schtasks /query /tn E3N-Daily-Training /fo LIST /v"
Write-Host ""
Write-Host "Test immediately (runs now, not at 2 AM):" -ForegroundColor Cyan
Write-Host "  Start-ScheduledTask -TaskName 'E3N-Daily-Training'"
Write-Host ""
Write-Host "Dry-run test (no training writes):" -ForegroundColor Cyan
Write-Host "  & '$PythonPath' '$ScriptPath' --dry-run --verbose"
Write-Host ""
Write-Host "Phase 11 web collector test:" -ForegroundColor Cyan
Write-Host "  & '$PythonPath' '$WorkDir\scripts\collect_training_data.py' --dry-run --report"
