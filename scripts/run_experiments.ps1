# Multi-Run Experiment Runner for Self-Healing Agent
# Runs 5 configurations x 5 runs each = 25 total benchmark runs
# Uses temperature=0 for deterministic results

param(
    [int]$NumRuns = 5,
    [int]$NumProblems = 0,  # 0 = all problems (164 for HumanEval)
    [switch]$DryRun  # Use --num-problems 3 for testing
)

# Configuration definitions
$configs = @(
    @{name="flash-max"; coder="qwen-flash"; critic="qwen3-max"},
    @{name="max-max"; coder="qwen3-max"; critic="qwen3-max"},
    @{name="flash-coder"; coder="qwen-flash"; critic="qwen3-coder-flash"},
    @{name="coder-coder"; coder="qwen3-coder-flash"; critic="qwen3-coder-flash"},
    @{name="flash-flash"; coder="qwen-flash"; critic="qwen-flash"}
)

# Get script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Change to project root
Push-Location $projectRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Self-Healing Agent Multi-Run Experiment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configurations: $($configs.Count)"
Write-Host "Runs per config: $NumRuns"
Write-Host "Total runs: $($configs.Count * $NumRuns)"
if ($DryRun) {
    Write-Host "MODE: DRY RUN (3 problems only)" -ForegroundColor Yellow
}
Write-Host ""

$totalRuns = $configs.Count * $NumRuns
$completedRuns = 0
$skippedRuns = 0
$failedRuns = 0
$startTime = Get-Date

foreach ($config in $configs) {
    $configName = $config.name
    $coderModel = $config.coder
    $criticModel = $config.critic

    Write-Host "----------------------------------------" -ForegroundColor Cyan
    Write-Host "Configuration: $configName" -ForegroundColor Cyan
    Write-Host "  Coder:  $coderModel"
    Write-Host "  Critic: $criticModel"
    Write-Host "----------------------------------------" -ForegroundColor Cyan

    # Ensure output directory exists
    $outputDir = Join-Path (Join-Path $projectRoot "results") $configName
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        Write-Host "Created directory: $outputDir"
    }

    for ($run = 1; $run -le $NumRuns; $run++) {
        $outputFile = Join-Path $outputDir "run$run.json"
        $runLabel = "$configName/run$run"

        # Check if output file already exists (resume capability)
        if (Test-Path $outputFile) {
            Write-Host "[$runLabel] SKIPPED - Output file already exists" -ForegroundColor Yellow
            $skippedRuns++
            continue
        }

        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host "[$timestamp] [$runLabel] Starting..." -ForegroundColor Green

        # Build command arguments
        $cmdArgs = @(
            "src/main.py",
            "--benchmark", "humaneval",
            "--from-hub",
            "--coder-provider", "qwen",
            "--coder-model", $coderModel,
            "--critic-provider", "qwen",
            "--critic-model", $criticModel,
            "--coder-temperature", "0",
            "--critic-temperature", "0",
            "--coder-max-tokens", "1024",
            "--critic-max-tokens", "2048",
            "--max-iterations", "5",
            "--enable-stuck-detection",
            "--stuck-threshold", "2",
            "--early-termination-on-stuck",
            "--early-termination-on-truncation",
            "--output", $outputFile
        )

        # Add num-problems for dry run
        if ($DryRun) {
            $cmdArgs += "--num-problems"
            $cmdArgs += "3"
        } elseif ($NumProblems -gt 0) {
            $cmdArgs += "--num-problems"
            $cmdArgs += $NumProblems.ToString()
        }

        # Run the benchmark
        try {
            $process = Start-Process -FilePath "python" -ArgumentList $cmdArgs -NoNewWindow -Wait -PassThru

            if ($process.ExitCode -eq 0) {
                $completedRuns++
                $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
                Write-Host "[$timestamp] [$runLabel] COMPLETED" -ForegroundColor Green
            } else {
                $failedRuns++
                Write-Host "[$runLabel] FAILED with exit code $($process.ExitCode)" -ForegroundColor Red
            }
        } catch {
            $failedRuns++
            Write-Host "[$runLabel] ERROR: $_" -ForegroundColor Red
        }
    }
}

# Summary
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "EXPERIMENT SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total runs:     $totalRuns"
Write-Host "Completed:      $completedRuns" -ForegroundColor Green
Write-Host "Skipped:        $skippedRuns" -ForegroundColor Yellow
Write-Host "Failed:         $failedRuns" -ForegroundColor Red
Write-Host "Duration:       $($duration.ToString('hh\:mm\:ss'))"
Write-Host ""

# Return to original directory
Pop-Location

if ($failedRuns -gt 0) {
    exit 1
}
