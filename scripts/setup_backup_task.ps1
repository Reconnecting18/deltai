# E3N Nightly Backup — Windows Task Scheduler Setup
#
# Run this script as Administrator to create a scheduled task that runs
# the S3 backup every night at 3:00 AM.
#
# Prerequisites:
#   1. pip install boto3
#   2. Set environment variable: E3N_S3_BUCKET=your-bucket-name
#   3. AWS credentials configured (aws configure or env vars)
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_backup_task.ps1

$TaskName = "E3N-Nightly-Backup"
$PythonPath = "C:\e3n\project\venv\Scripts\python.exe"
$ScriptPath = "C:\e3n\scripts\backup_s3.py"
$LogPath = "C:\e3n\data\backup.log"

# Check if running as admin
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: Run this script as Administrator" -ForegroundColor Red
    exit 1
}

# Check prerequisites
if (-NOT (Test-Path $PythonPath)) {
    Write-Host "ERROR: Python venv not found at $PythonPath" -ForegroundColor Red
    Write-Host "Run: cd C:\e3n\project; python -m venv venv; .\venv\Scripts\activate; pip install boto3"
    exit 1
}

if (-NOT $env:E3N_S3_BUCKET) {
    Write-Host "WARNING: E3N_S3_BUCKET not set. Set it as a system environment variable." -ForegroundColor Yellow
    Write-Host "  setx E3N_S3_BUCKET your-bucket-name /M"
}

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task: $TaskName"
}

# Create the action: run backup script, redirect output to log
$Action = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument "$ScriptPath >> $LogPath 2>&1" `
    -WorkingDirectory "C:\e3n"

# Trigger: daily at 3:00 AM
$Trigger = New-ScheduledTaskTrigger -Daily -At 3:00AM

# Settings: run even if on battery, don't wake from sleep, retry on failure
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# Register the task (runs as current user)
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "E3N nightly backup to AWS S3 — backs up ChromaDB, SQLite, knowledge base, training data"

Write-Host ""
Write-Host "Task created: $TaskName" -ForegroundColor Green
Write-Host "  Schedule: Daily at 3:00 AM"
Write-Host "  Script:   $ScriptPath"
Write-Host "  Log:      $LogPath"
Write-Host ""
Write-Host "To test immediately:" -ForegroundColor Cyan
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "To remove:" -ForegroundColor Cyan
Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName'"
