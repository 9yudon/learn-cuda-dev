[CmdletBinding()]
param(
  [string]$BuildDir = "build/release",
  [int]$Rows = 4096,
  [int]$Cols = 4096,
  [int]$Iterations = 200
)

$ErrorActionPreference = "Stop"
$binary = Join-Path $BuildDir "rmsnorm_bench.exe"
if (-not (Test-Path -LiteralPath $binary)) {
  throw "Benchmark not found: $binary. Build target rmsnorm_bench first."
}

$output = Join-Path $BuildDir "rmsnorm_profile"
$outputDirectory = Split-Path -Parent $output
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
& ncu --force-overwrite --set full --target-processes all --export $output `
  $binary --rows $Rows --cols $Cols --iterations $Iterations
