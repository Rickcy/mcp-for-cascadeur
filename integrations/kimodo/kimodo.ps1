[CmdletBinding()]
param(
    [ValidateSet('start', 'stop', 'status', 'logs', 'build', 'generate')]
    [string]$Action = 'status',
    [string]$ModelsPath,
    [string]$ExchangePath,
    [ValidateRange(1024, 65535)]
    [int]$Port = 9552,
    [ValidateRange(30, 3600)]
    [int]$TimeoutSeconds = 1200,
    [string]$Prompt,
    [ValidateRange(2, 1800)]
    [int]$Frames = 120,
    [ValidateRange(1, 200)]
    [int]$Steps = 50,
    [uint32]$Seed = 42,
    [ValidateRange(1, 4)]
    [int]$Samples = 1,
    [ValidateRange(0, 3)]
    [int]$SampleIndex = 0,
    [string]$Constraints,
    [switch]$Build,
    [switch]$ReleaseGpuAfter,
    [switch]$PurgeRuntime
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-DefaultExchangePath {
    param(
        [string]$ConfigPath,
        [string]$UserProfilePath = $env:USERPROFILE
    )
    if (-not $ConfigPath -and $env:LOCALAPPDATA) {
        $ConfigPath = Join-Path $env:LOCALAPPDATA 'mcp-for-cascadeur\config.json'
    }
    if ($ConfigPath -and (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
        try {
            $settings = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
            $workspaceProperty = $settings.PSObject.Properties['workspace']
            if ($null -ne $workspaceProperty -and $workspaceProperty.Value -is [string]) {
                $workspacePath = $workspaceProperty.Value
                if (-not [string]::IsNullOrWhiteSpace($workspacePath) -and
                    $workspacePath -match '^(?:[A-Za-z]:[\\/]|\\\\[^\\]+\\[^\\]+)' -and
                    (Test-Path -LiteralPath $workspacePath -PathType Container)) {
                    return Join-Path ([IO.Path]::GetFullPath($workspacePath)) 'kimodo'
                }
            }
            Write-Warning "No valid existing workspace was found in $ConfigPath. Using the local Documents fallback."
        } catch {
            Write-Warning "Could not read the MCP workspace from $ConfigPath. Using the local Documents fallback."
        }
    }
    if ([string]::IsNullOrWhiteSpace($UserProfilePath)) {
        throw 'USERPROFILE is unavailable. Specify -ExchangePath explicitly.'
    }
    return Join-Path $UserProfilePath 'Documents\Cascadeur MCP\kimodo'
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$composeFile = Join-Path $PSScriptRoot 'compose.yaml'
if (-not $ModelsPath) {
    $ModelsPath = Join-Path $repoRoot '.local\kimodo\models'
}
if (-not $ExchangePath) {
    $ExchangePath = Resolve-DefaultExchangePath
}
$modelsFullPath = [IO.Path]::GetFullPath($ModelsPath)
$exchangeFullPath = [IO.Path]::GetFullPath($ExchangePath)
Write-Output "Exchange directory: $exchangeFullPath"
$selectedModel = 'Kimodo-SOMA-RP-v1.2'
Write-Output "Model: $selectedModel"

if ($PurgeRuntime -and $Action -ne 'stop') {
    throw '-PurgeRuntime is only available with -Action stop.'
}
if ($ReleaseGpuAfter -and $Action -ne 'generate') {
    throw '-ReleaseGpuAfter is only available with -Action generate.'
}
if ($Action -eq 'generate' -and ([string]::IsNullOrWhiteSpace($Prompt) -or $Prompt.Length -gt 4000)) {
    throw '-Action generate requires a nonempty -Prompt of at most 4000 characters.'
}
if ($SampleIndex -ge $Samples) {
    throw '-SampleIndex must be smaller than -Samples. All generated candidates are retained.'
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker CLI is missing. Install Docker Desktop and enable Linux containers with GPU support.'
}

function Invoke-Compose {
    param([Parameter(Mandatory)][string[]]$Arguments)
    & docker compose --project-name cascadeur-kimodo --file $composeFile @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose failed with exit code $LASTEXITCODE."
    }
}

function Stop-KimodoProject {
    param([switch]$RemoveRuntime)
    $arguments = @('down', '--remove-orphans')
    if ($RemoveRuntime) { $arguments += '--volumes' }
    Invoke-Compose -Arguments $arguments
}

function Release-KimodoGpu {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 15
        if ($health.ready -ne $true -or $health.busy -isnot [bool] -or $health.busy) {
            Write-Output 'Kimodo GPU memory was not released: the backend is busy or not ready. No services were stopped.'
            return
        }
    } catch {
        Write-Warning 'Kimodo GPU memory was not released because backend idle status could not be verified. No services were stopped.'
        return
    }
    Stop-KimodoProject
    Write-Output 'Stopped the cascadeur-kimodo Compose project to release model GPU memory. Generated files and models were preserved.'
}

function Assert-ModelFiles {
    $mergedDir = Join-Path $modelsFullPath 'text-encoders\generated\LLM2Vec-Meta-Llama-3-8B-Instruct-merged'
    $checkpointDir = Join-Path $modelsFullPath "checkpoints\$selectedModel"
    foreach ($file in @(
        (Join-Path $mergedDir 'config.json'),
        (Join-Path $mergedDir 'llm2vec_config.json'),
        (Join-Path $mergedDir 'tokenizer_config.json'),
        (Join-Path $checkpointDir 'config.yaml'),
        (Join-Path $checkpointDir 'model.safetensors')
    )) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
            throw "Required model file is missing: $file"
        }
    }
    foreach ($area in @('body', 'global_root', 'local_root')) {
        foreach ($stat in @('mean', 'std')) {
            $statsFile = Join-Path $checkpointDir "stats\motion\$area\$stat.npy"
            if (-not (Test-Path -LiteralPath $statsFile -PathType Leaf)) {
                throw "Required checkpoint statistics are missing: $statsFile"
            }
        }
    }
    $indexPath = Join-Path $mergedDir 'model.safetensors.index.json'
    $singleWeight = Join-Path $mergedDir 'model.safetensors'
    if (Test-Path -LiteralPath $indexPath -PathType Leaf) {
        $index = Get-Content -LiteralPath $indexPath -Raw | ConvertFrom-Json
        $shards = @($index.weight_map.PSObject.Properties.Value | Select-Object -Unique)
        if ($shards.Count -eq 0) { throw 'The merged encoder index contains no weight shards.' }
        foreach ($shard in $shards) {
            if ([IO.Path]::GetFileName($shard) -ne $shard) {
                throw "Unexpected model shard path in index: $shard"
            }
            if (-not (Test-Path -LiteralPath (Join-Path $mergedDir $shard) -PathType Leaf)) {
                throw "Merged encoder shard is missing: $shard"
            }
        }
    } elseif (-not (Test-Path -LiteralPath $singleWeight -PathType Leaf)) {
        throw "The merged encoder weights are missing from $mergedDir."
    }
}

function Resolve-ArtifactPath {
    param([Parameter(Mandatory)][string]$RelativePath)
    if ([IO.Path]::IsPathRooted($RelativePath)) {
        throw "The backend returned an absolute artifact path: $RelativePath"
    }
    $fullPath = [IO.Path]::GetFullPath((Join-Path $exchangeFullPath $RelativePath.Replace('/', '\')))
    $rootPrefix = $exchangeFullPath.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "The backend returned an artifact outside the exchange directory: $RelativePath"
    }
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
        throw "The backend reported success but the artifact is missing: $fullPath"
    }
    return $fullPath
}

function Invoke-Generation {
    $baseUrl = "http://127.0.0.1:$Port"
    $health = Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 15
    if ($health.ready -ne $true) { throw 'Start Kimodo and wait for a ready backend before generating.' }
    if ($health.model_checkpoint.name -ne $selectedModel) {
        throw 'The backend loaded a different model. Rebuild and start the v1.2 runtime before generating.'
    }
    $request = @{
        prompt = $Prompt.Trim()
        num_frames = $Frames
        diffusion_steps = $Steps
        seed = $Seed
        fps = 30
        model_name = $selectedModel
        num_samples = $Samples
        sample_index = $SampleIndex
    }
    if ($Constraints) {
        if ([IO.Path]::IsPathRooted($Constraints)) {
            $constraintPath = [IO.Path]::GetFullPath($Constraints)
        } else {
            $constraintPath = [IO.Path]::GetFullPath((Join-Path $exchangeFullPath $Constraints))
        }
        $prefix = $exchangeFullPath.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
        if (-not $constraintPath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or
            [IO.Path]::GetExtension($constraintPath) -ne '.npz' -or
            -not (Test-Path -LiteralPath $constraintPath -PathType Leaf)) {
            throw '-Constraints must identify an existing NPZ inside the exchange directory.'
        }
        $request.constraints = $constraintPath.Substring($prefix.Length).Replace('\', '/')
    }
    $payload = $request | ConvertTo-Json -Compress
    $body = [Text.Encoding]::UTF8.GetBytes($payload)
    $elapsed = [Diagnostics.Stopwatch]::StartNew()
    try {
        # Submit once. An interrupted response may still represent an accepted job.
        $accepted = Invoke-RestMethod -Uri "$baseUrl/jobs" -Method Post `
            -ContentType 'application/json; charset=utf-8' -Body $body -TimeoutSec 30
    } catch {
        throw "Generation submission failed and was not repeated. Check the backend logs before submitting again. $($_.Exception.Message)"
    }
    $jobId = $accepted.job_id
    if ($accepted.ok -ne $true -or $null -eq $jobId) { throw 'The backend did not return an accepted job ID.' }
    $jobUrl = "$baseUrl/jobs/$jobId"
    Write-Output "Job $jobId accepted: $jobUrl"
    $lastStatus = ''
    while ($elapsed.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        try {
            $snapshot = Invoke-RestMethod -Uri $jobUrl -TimeoutSec 15
        } catch {
            throw "Could not read job $jobId. The job was not resubmitted. Check $jobUrl before starting another generation. $($_.Exception.Message)"
        }
        $job = $snapshot.job
        $statusLine = "Job ${jobId}: $($job.status) [$($job.progress)%]"
        if ($statusLine -ne $lastStatus) {
            Write-Output $statusLine
            $lastStatus = $statusLine
        }
        if ($job.done) {
            if ($job.error) {
                Write-Output 'Any completed candidates remain in the exchange directory. The failed job was not repeated.'
                $job.artifacts | ConvertTo-Json -Depth 8 | Write-Output
                throw "Job $jobId failed: $($job.error)"
            }
            $candidates = @($job.artifacts.candidates)
            if ($candidates.Count -ne $Samples) { throw "Job $jobId did not return every requested candidate." }
            foreach ($candidate in $candidates) {
                $paths = @{}
                foreach ($name in @('output', 'output_fbx', 'log_path', 'manifest')) {
                    $paths[$name] = Resolve-ArtifactPath -RelativePath ([string]$candidate.$name)
                }
                $manifest = Get-Content -LiteralPath $paths.manifest -Raw | ConvertFrom-Json
                if ($manifest.format -ne 'cascadeur-kimodo-motion-v1' -or
                    $manifest.num_frames -ne $Frames -or $manifest.fps -ne 30 -or $manifest.model_name -ne $selectedModel -or
                    $manifest.seed -ne $candidate.seed -or $manifest.candidate_index -ne $candidate.index -or
                    $manifest.text_encoder.kind -ne 'llm2vec' -or $manifest.text_encoder.fallback -ne $false) {
                    throw "Job $jobId returned an unexpected motion manifest: $($paths.manifest)"
                }
                $hashAlgorithm = [Security.Cryptography.SHA256]::Create()
                $fileStream = [IO.File]::OpenRead($paths.output_fbx)
                try {
                    $hash = [BitConverter]::ToString($hashAlgorithm.ComputeHash($fileStream)).Replace('-', '').ToLowerInvariant()
                } finally {
                    $fileStream.Dispose()
                    $hashAlgorithm.Dispose()
                }
                if ($hash -ne $manifest.sha256) { throw "Job $jobId FBX checksum does not match its manifest." }
                Write-Output "Candidate $($candidate.index) (seed $($candidate.seed)):"
                Write-Output "FBX: $($paths.output_fbx)"
                Write-Output "Motion data: $($paths.output)"
                Write-Output "Manifest: $($paths.manifest)"
            }
            $indexPath = Resolve-ArtifactPath -RelativePath ([string]$job.artifacts.candidates_manifest)
            Write-Output "All candidates: $indexPath"
            Write-Output "Requested default index: $SampleIndex. This is not a quality ranking."
            if ($ReleaseGpuAfter) { Release-KimodoGpu }
            return
        }
        Start-Sleep -Seconds 1
    }
    throw "Stopped waiting after $TimeoutSeconds seconds. Job $jobId may still be running: $jobUrl. It was not cancelled or resubmitted."
}

$savedVariables = @{}
foreach ($name in @('KIMODO_MODELS_DIR', 'KIMODO_EXCHANGE_DIR', 'KIMODO_BACKEND_PORT')) {
    $savedVariables[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    [Environment]::SetEnvironmentVariable('KIMODO_MODELS_DIR', $modelsFullPath, 'Process')
    [Environment]::SetEnvironmentVariable('KIMODO_EXCHANGE_DIR', $exchangeFullPath, 'Process')
    [Environment]::SetEnvironmentVariable('KIMODO_BACKEND_PORT', [string]$Port, 'Process')

    if ($Action -eq 'start' -or $Action -eq 'build') {
        $containerType = & docker info --format '{{.OSType}}' 2>$null
        if ($LASTEXITCODE -ne 0 -or $containerType -ne 'linux') {
            throw 'Start Docker Desktop with its Linux engine before launching Kimodo.'
        }
        Assert-ModelFiles
    }
    if ($Action -eq 'build' -or ($Action -eq 'start' -and $Build)) {
        $sourceRoot = Join-Path $repoRoot '.local\kimodo\source'
        if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot 'pyproject.toml') -PathType Leaf)) {
            throw "Prepared Kimodo sources are missing from $sourceRoot."
        }
        Invoke-Compose -Arguments @('build', 'text-encoder')
    }

    switch ($Action) {
        'start' {
            New-Item -ItemType Directory -Force -Path $exchangeFullPath | Out-Null
            Invoke-Compose -Arguments @('up', '--detach', '--no-build', '--wait', '--wait-timeout', [string]$TimeoutSeconds)
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 15
            if ($health.ready -ne $true) { throw 'Kimodo started without a ready backend.' }
            if ($health.model_checkpoint.name -ne $selectedModel) { throw 'Kimodo started with a different checkpoint.' }
            Write-Output "Kimodo is ready at http://127.0.0.1:$Port. Exchange: $exchangeFullPath"
        }
        'stop' {
            Stop-KimodoProject -RemoveRuntime:$PurgeRuntime
        }
        'status' {
            Invoke-Compose -Arguments @('ps', '--all')
            try {
                Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5 | ConvertTo-Json -Depth 8
            } catch {
                Write-Output "Backend is not reachable at http://127.0.0.1:$Port."
            }
        }
        'logs' {
            Invoke-Compose -Arguments @('logs', '--tail', '100', '--no-color')
        }
        'generate' {
            Invoke-Generation
        }
    }
} finally {
    foreach ($name in $savedVariables.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedVariables[$name], 'Process')
    }
}
