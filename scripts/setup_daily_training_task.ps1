# E3N Daily Autonomous Training — Windows Task Scheduler Setup
#
# Registers a scheduled task to run the daily training cycle at 2:00 AM every night.
# Runs BEFORE the 3:00 AM S3 backup (setup_backup_task.ps1) so new training
# data and adapters are included in the nightly backup.
#
# Prerequisites:
#   1. Python venv exists at C:\e3n\project\venv\
#   2. Training deps installed: torch, transformers, peft, trl, bitsandbytes
#   3. E3N API server does NOT need to be running (daily_training.py is standalone)
#
# Usage (run as Administrator):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_daily_training_task.ps1
#
# To disable training on specific days, set DAILY_TRAIN_ENABLED=false in .env
# and re-enable before the next session you want training for.

$TaskName = "E3N-Daily-Training"
$PythonPath = "C:\e3n\project\venv\Scripts\python.exe"
$ScriptPath = "C:\e3n\scripts\daily_training.py"
$LogPath = "C:\e3n\data\training\daily_training.log"

# Check if running as admin
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: Run this script as Administrator" -ForegroundColor Red
    exit 1
}

# Check prerequisites
if (-NOT (Test-Path $PythonPath)) {
    Write-Host "ERROR: Python venv not found at $PythonPath" -ForegroundColor Red
    Write-Host "Run: cd C:\e3n\project; python -m venv venv; .\venv\Scripts\activate; pip install torch transformers peft trl bitsandbytes"
    exit 1
}

if (-NOT (Test-Path $ScriptPath)) {
    Write-Host "ERROR: daily_training.py not found at $ScriptPath" -ForegroundColor Red
    exit 1
}

# Ensure log parent directory exists
$LogParent = Split-Path $LogPath
if (-NOT (Test-Path $LogParent)) {
    New-Item -ItemType Directory -Path $LogParent -Force | Out-Null
    Write-Host "Created log directory: $LogParent"
}

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task: $TaskName"
}

# Action: run daily_training.py, append output to log
$Action = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument "$ScriptPath >> $LogPath 2>&1" `
    -WorkingDirectory "C:\e3n"

# Trigger: daily at 2:00 AM
$Trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM

# Settings: allow battery, retry on failure, max 3 hours runtime
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 15) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)

# Register the task (runs as current user)
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "E3N daily autonomous training — weakness analysis, targeted distillation, QLoRA domain adapter training, memory consolidation. Runs at 2 AM before the 3 AM S3 backup."

Write-Host ""
Write-Host "Task created: $TaskName" -ForegroundColor Green
Write-Host "  Schedule:   Daily at 2:00 AM (1 hour before S3 backup)"
Write-Host "  Script:     $ScriptPath"
Write-Host "  Log:        $LogPath"
Write-Host ""
Write-Host "Configuration (.env):" -ForegroundColor Cyan
Write-Host "  DAILY_TRAIN_ENABLED=true          # Enable/disable the cycle"
Write-Host "  DAILY_TRAIN_MIN_VRAM_MB=7000      # Minimum free VRAM to proceed"
Write-Host "  DAILY_TRAIN_AUTO_PROMOTE=false    # Auto-promote better adapters"
Write-Host "  DAILY_TRAIN_AUTO_MERGE=false      # Auto-TIES merge after promote"
Write-Host "  SESSION_SYNTHESIS_ENABLED=true    # Synthesize session knowledge"
Write-Host ""
Write-Host "To test immediately:" -ForegroundColor Cyan
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "To run manually (dry run — no actual training):" -ForegroundColor Cyan
Write-Host "  $PythonPath $ScriptPath --dry-run"
Write-Host ""
Write-Host "To view last report:" -ForegroundColor Cyan
Write-Host "  $PythonPath $ScriptPath --report-only"
Write-Host ""
Write-Host "To remove:" -ForegroundColor Cyan
Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName'"
