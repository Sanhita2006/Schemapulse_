<#
.SYNOPSIS
    SchemaPulse hackathon demo runner — golden path only.

.DESCRIPTION
    Brings up the two demo PostgreSQL servers, seeds them, checks the
    environment, and runs the deterministic fixture demo. No API key and no
    network are required: the fixture provider is offline and labelled as such.

    Live model calls are deliberately NOT part of the default path, because an
    accidental API call should never happen on stage. Pass -Live to run the
    Gemini path explicitly, once you have decided to.

.PARAMETER Live
    Also run the live Gemma provider (Gemini). Costs one API call and requires
    GEMINI_API_KEY in .env.

.EXAMPLE
    .\scripts\demo.ps1

.EXAMPLE
    .\scripts\demo.ps1 -Live
#>
[CmdletBinding()]
param(
    [switch]$Live
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$cli = Join-Path $root '.venv\Scripts\schemapulse.exe'
$step = 0

function Invoke-Step {
    param(
        [Parameter(Mandatory)][string]$Title,
        [Parameter(Mandatory)][scriptblock]$Action,
        [int[]]$OkExit = @(0)
    )
    $script:step++
    Write-Host ''
    Write-Host "==> [$script:step] $Title" -ForegroundColor Cyan

    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $Action
    $code = $LASTEXITCODE
    $sw.Stop()

    $secs = [int]$sw.Elapsed.TotalSeconds
    if ($OkExit -contains $code) {
        Write-Host ("    ok ({0}s, exit {1})" -f $secs, $code) -ForegroundColor Green
        return $true
    }
    Write-Host ("    FAILED (exit {0} after {1}s)" -f $code, $secs) -ForegroundColor Red
    return $false
}

function Stop-Demo {
    param([string]$Why)
    Write-Host ''
    Write-Host "DEMO HALTED: $Why" -ForegroundColor Red
    Write-Host 'See docs/DEMO.md section 10 (Troubleshooting).' -ForegroundColor Yellow
    exit 1
}

Write-Host 'SchemaPulse — execution-verified PostgreSQL optimization' -ForegroundColor White
Write-Host 'Gemma proposes. PostgreSQL proves.' -ForegroundColor DarkGray

# --- preflight -------------------------------------------------------------
if (-not (Test-Path -LiteralPath $cli)) {
    Stop-Demo 'no virtualenv at .venv\Scripts\schemapulse.exe (run: pip install -e ".[dev]")'
}

& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Stop-Demo 'the Docker daemon is not responding. Start Docker Desktop and re-run.'
}

# --- golden path -----------------------------------------------------------
if (-not (Invoke-Step 'Start the two demo PostgreSQL servers' { & docker compose up -d --wait })) {
    Stop-Demo 'docker compose could not start the demo databases'
}

# Cold seeding generates 1,000,000 orders on both servers and takes ~5 minutes.
if (-not (Invoke-Step 'Seed both databases (cold start ~5 min)' { & $cli demo --seed-only })) {
    Stop-Demo 'seeding failed'
}

if (-not (Invoke-Step 'Check the environment' { & $cli doctor })) {
    Stop-Demo 'doctor reported a problem; the numbers below would not be trustworthy'
}

if (-not (Invoke-Step 'Run the deterministic golden demo' { & $cli demo --provider fixture })) {
    Write-Host ''
    Write-Host 'The golden demo did not reach VERIFIED.' -ForegroundColor Red
    Write-Host 'Exit 2 means not verified, exit 1 means the environment broke.' -ForegroundColor Yellow
    Write-Host 'Do NOT lower --min-speedup to force a pass. Re-seed and investigate.' -ForegroundColor Yellow
    Stop-Demo 'the fixture demo did not verify'
}

if (-not (Invoke-Step 'Render the evidence report' { & $cli report })) {
    Stop-Demo 'the report could not be rendered'
}

# --- opt-in live path ------------------------------------------------------
if ($Live) {
    Write-Host ''
    Write-Host 'Running the LIVE Gemma path (one API call).' -ForegroundColor Yellow
    # Exit 2 is a legitimate outcome here: a live model may propose something
    # measurably wrong, and the tool is required to say so.
    if (-not (Invoke-Step 'Diagnose with the live Gemma provider' { & $cli diagnose demo/slow_orders.sql --provider gemini } -OkExit @(0, 2))) {
        Stop-Demo 'the live provider path errored'
    }
}
else {
    Write-Host ''
    Write-Host 'Skipped the live Gemma path.' -ForegroundColor DarkGray
    Write-Host 'Run it on purpose with:  .\scripts\demo.ps1 -Live' -ForegroundColor DarkGray
}

Write-Host ''
Write-Host 'Demo ready.' -ForegroundColor Green
Write-Host 'Evidence: .schemapulse\report.html' -ForegroundColor Green
Write-Host 'Open it with:  schemapulse report --open' -ForegroundColor Green