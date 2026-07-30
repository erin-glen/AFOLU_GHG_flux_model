[CmdletBinding()]
param(
    [string]$Config,
    [string]$OutputDir,
    [string]$CacheDir,
    [string]$AwsProfile,
    [switch]$ValidateOnly,
    [switch]$Overwrite,
    [switch]$NoCache,
    [switch]$VerboseLogs,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $PSScriptRoot "input_dataset_maps.config.json"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $PSScriptRoot "outputs\maps_$stamp"
}
if ([string]::IsNullOrWhiteSpace($CacheDir)) {
    $CacheDir = Join-Path $PSScriptRoot "cache"
}

$PythonArgs = @(
    (Join-Path $PSScriptRoot "create_input_dataset_maps.py"),
    "--config", $Config,
    "--output-dir", $OutputDir,
    "--cache-dir", $CacheDir
)
if (-not [string]::IsNullOrWhiteSpace($AwsProfile)) {
    $PythonArgs += @("--aws-profile", $AwsProfile)
}
if ($ValidateOnly) {
    $PythonArgs += "--validate-only"
}
if ($Overwrite) {
    $PythonArgs += "--overwrite"
}
if ($NoCache) {
    $PythonArgs += "--no-cache"
}
if ($VerboseLogs) {
    $PythonArgs += "--verbose"
}
if ($RemainingArgs) {
    $PythonArgs += $RemainingArgs
}

Write-Host "Output directory: $OutputDir"
& python @PythonArgs
exit $LASTEXITCODE

