<#
  Lists open GitHub code scanning alerts grouped by rule (severity + rule id + count).
  Prerequisites: GitHub CLI (gh), authenticated for the repo (gh auth login).

  Usage (from repo root):
    powershell -File scripts/export-code-scanning-alerts.ps1
    powershell -File scripts/export-code-scanning-alerts.ps1 -Repo "owner/name"
#>
param(
    [string]$Repo = ""
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "Install GitHub CLI (gh) and run: gh auth login"
}

if (-not $Repo) {
    $Repo = (gh repo view --json nameWithOwner -q .nameWithOwner 2>$null)
    if (-not $Repo) {
        Write-Error "Could not detect repo. Pass -Repo 'owner/name' or run from a git clone with gh configured."
    }
}

Write-Host "Fetching open code scanning alerts for $Repo ..." -ForegroundColor Cyan

$all = @()
$page = 1
$perPage = 100

while ($true) {
    $uri = "repos/$Repo/code-scanning/alerts?state=open&per_page=$perPage&page=$page"
    $chunk = gh api $uri | ConvertFrom-Json
    if (-not $chunk) { break }
    if ($chunk -isnot [System.Array]) { $chunk = @($chunk) }
    if ($chunk.Count -eq 0) { break }
    $all += $chunk
    if ($chunk.Count -lt $perPage) { break }
    $page++
}

if ($all.Count -eq 0) {
    Write-Host "No open alerts (or no permission; token may need security_events read)." -ForegroundColor Yellow
    exit 0
}

$groups = $all | Group-Object { $_.rule.id }
$rows = foreach ($g in $groups | Sort-Object Count -Descending) {
    $first = $g.Group[0]
    [PSCustomObject]@{
        Count  = $g.Count
        Severity = $first.rule.severity
        RuleId = $first.rule.id
        RuleName = $first.rule.name
        Tool = $first.tool.name
    }
}

Write-Host ""
Write-Host "Total open alerts: $($all.Count)" -ForegroundColor Green
Write-Host ""
$rows | Format-Table -AutoSize
