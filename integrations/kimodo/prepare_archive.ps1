param(
    [string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\KimodoCascadeur_v2.rar'),
    [string]$Unrar = 'C:\Program Files\WinRAR\UnRAR.exe'
)

$ErrorActionPreference = 'Stop'
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$dataRoot = Join-Path $repository '.local\kimodo'
$sourceRoot = Join-Path $dataRoot 'source'
$modelsRoot = Join-Path $dataRoot 'models'
if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) { throw "Archive not found: $Archive" }
if (-not (Test-Path -LiteralPath $Unrar -PathType Leaf)) { throw "UnRAR not found: $Unrar" }
$archiveInfo = Get-Item -LiteralPath $Archive
$receiptPath = Join-Path $dataRoot 'archive-source.json'
$required = @(
    (Join-Path $modelsRoot 'text-encoders\generated\LLM2Vec-Meta-Llama-3-8B-Instruct-merged\model.safetensors'),
    (Join-Path $modelsRoot 'text-encoders\generated\LLM2Vec-Meta-Llama-3-8B-Instruct-merged\config.json'),
    (Join-Path $modelsRoot 'text-encoders\generated\LLM2Vec-Meta-Llama-3-8B-Instruct-merged\llm2vec_config.json'),
    (Join-Path $modelsRoot 'text-encoders\generated\LLM2Vec-Meta-Llama-3-8B-Instruct-merged\tokenizer.json'),
    (Join-Path $modelsRoot 'text-encoders\generated\LLM2Vec-Meta-Llama-3-8B-Instruct-merged\tokenizer_config.json'),
    (Join-Path $sourceRoot 'kimodo\assets\skeletons\somaskel77\skin_standard.npz'),
    (Join-Path $sourceRoot 'kimodo\assets\skeletons\somaskel77\joints.p'),
    (Join-Path $sourceRoot 'kimodo\assets\skeletons\somaskel30\joints.p'),
    (Join-Path $sourceRoot 'tools\blender-4.2.12-linux-x64\blender'),
    (Join-Path $sourceRoot 'integrations\cascadeur\Kimodo.casc'),
    (Join-Path $sourceRoot 'setup.py'),
    (Join-Path $sourceRoot 'pyproject.toml'),
    (Join-Path $sourceRoot 'scripts\cascadeur_backend_service.py'),
    (Join-Path $sourceRoot 'MotionCorrection\CMakeLists.txt')
)
if (Test-Path -LiteralPath $receiptPath -PathType Leaf) {
    try {
        $previous = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
        $sameArchive = [string]::Equals([IO.Path]::GetFullPath($previous.archive), $archiveInfo.FullName,
            [StringComparison]::OrdinalIgnoreCase) -and
            [long]$previous.archive_bytes -eq $archiveInfo.Length -and
            [DateTimeOffset]::Parse($previous.archive_modified_utc).UtcDateTime.Ticks -eq $archiveInfo.LastWriteTimeUtc.Ticks
    } catch {
        throw "The preparation receipt is invalid. Review $receiptPath before preparing again."
    }
    if (-not $sameArchive) {
        throw "The archive differs from the preparation receipt. Existing source and models were preserved; review $dataRoot."
    }
    $archivePaths = @{}
    foreach ($path in $required) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -eq 0) {
            throw "Prepared assets are incomplete: $path. Review the existing extraction; nothing was overwritten."
        }
        $base = if ($path.StartsWith($modelsRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { $modelsRoot } else { $sourceRoot }
        $archivePaths['KimodoCascadeur' + $path.Substring($base.Length)] = $path
    }
    $details = @(& $Unrar lt -p- $Archive @($archivePaths.Keys))
    if ($LASTEXITCODE -ne 0) { throw 'Cannot verify prepared components against archive metadata' }
    $sizes = @{}
    $entryName = ''
    foreach ($line in $details) {
        if ($line -match '^\s*Name:\s+(.+)$') { $entryName = $Matches[1] }
        if ($line -match '^\s*Size:\s+(\d+)\s*$' -and $archivePaths.ContainsKey($entryName)) {
            $sizes[$entryName] = [long]$Matches[1]
        }
    }
    foreach ($name in $archivePaths.Keys) {
        if (-not $sizes.ContainsKey($name) -or (Get-Item -LiteralPath $archivePaths[$name]).Length -ne $sizes[$name]) {
            throw "Prepared component has an unexpected size: $($archivePaths[$name]). Review the extraction; nothing was overwritten."
        }
    }
    Write-Host "Already prepared: $dataRoot (archive identity and required component sizes match)."
    exit 0
}

foreach ($existing in @($sourceRoot, $modelsRoot, ($sourceRoot + '-extract'), ($modelsRoot + '-extract'))) {
    if (Test-Path -LiteralPath $existing) {
        throw "An unverified extraction already exists at $existing. Review it before preparing again; extraction was not started."
    }
}

$entries = @(& $Unrar lb -p- $Archive)
if ($LASTEXITCODE -ne 0) { throw 'Cannot list the archive' }
foreach ($entry in $entries) {
    if ($entry -match '(^|[\\/])\.\.([\\/]|$)|^[\\/]|:' -or $entry -notmatch '^KimodoCascadeur(\\|$)') {
        throw "Unexpected archive path: $entry"
    }
}

$sourceEntries = @($entries | Where-Object {
    ($_ -match '^KimodoCascadeur\\(kimodo|scripts|tools\\blender-4\.2\.12-linux-x64)\\' -or
     $_ -match '^KimodoCascadeur\\MotionCorrection\\' -or
     $_ -match '^KimodoCascadeur\\integrations\\cascadeur\\Kimodo\.casc$' -or
     $_ -match '^KimodoCascadeur\\(setup\.py|pyproject\.toml|MANIFEST\.in|LICENSE|ATTRIBUTIONS\.MD|README\.md)$') -and
    $_ -notmatch '[\\/]__pycache__[\\/]|\.pyc$|^KimodoCascadeur\\MotionCorrection\\(build|python\\motion_correction\.egg-info)\\|^KimodoCascadeur\\MotionCorrection\\.*\.so$'
})
$modelEntries = @($entries | Where-Object {
    $_ -match '^KimodoCascadeur\\text-encoders\\generated\\LLM2Vec-Meta-Llama-3-8B-Instruct-merged\\' -or
    $_ -match '^KimodoCascadeur\\text-encoders\\meta-llama\\Meta-Llama-3-8B-Instruct\\(LICENSE|README\.md)$'
})
if ($sourceEntries.Count -lt 100 -or $modelEntries.Count -lt 5) { throw 'The archive layout is not supported' }

New-Item -ItemType Directory -Force -Path $dataRoot | Out-Null
function Expand-Selection([string[]]$Selection, [string]$Destination) {
    # Directory patterns avoid comparing every archive entry with thousands of names.
    $exclusions = @('-x*\__pycache__\*', '-x*.pyc', '-xKimodoCascadeur\MotionCorrection\build\*',
        '-xKimodoCascadeur\MotionCorrection\python\motion_correction.egg-info\*',
        '-xKimodoCascadeur\MotionCorrection\*.so')
    $stage = $Destination + '-extract'
    $stagedRoot = Join-Path $stage 'KimodoCascadeur'
    if ((Test-Path -LiteralPath $Destination) -or (Test-Path -LiteralPath $stage)) {
        throw "An extraction destination already exists. Review $Destination and $stage; nothing was overwritten."
    }
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    & $Unrar x -idq -p- -o- $Archive @Selection @exclusions ($stage + '\')
    if ($LASTEXITCODE -ne 0) { throw "UnRAR extraction failed: $LASTEXITCODE" }
    if (Test-Path -LiteralPath $Destination) {
        throw "Destination appeared during extraction: $Destination. Review it before continuing; nothing was overwritten."
    }
    $allowedRoot = [IO.Path]::GetFullPath($dataRoot) + '\'
    foreach ($movePath in @($stagedRoot, $Destination)) {
        if (-not [IO.Path]::GetFullPath($movePath).StartsWith($allowedRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Extraction move escaped the data directory'
        }
    }
    Move-Item -LiteralPath $stagedRoot -Destination $Destination -ErrorAction Stop
    Remove-Item -LiteralPath $stage -ErrorAction Stop
}

Write-Host 'Extracting source, skeleton assets and the FBX exporter...'
Expand-Selection @('KimodoCascadeur\kimodo\*', 'KimodoCascadeur\MotionCorrection\*',
    'KimodoCascadeur\scripts\*', 'KimodoCascadeur\tools\blender-4.2.12-linux-x64\*',
    'KimodoCascadeur\integrations\cascadeur\Kimodo.casc', 'KimodoCascadeur\setup.py',
    'KimodoCascadeur\pyproject.toml', 'KimodoCascadeur\MANIFEST.in', 'KimodoCascadeur\LICENSE',
    'KimodoCascadeur\ATTRIBUTIONS.MD', 'KimodoCascadeur\README.md') $sourceRoot
Write-Host 'Extracting the merged text encoder...'
Expand-Selection @('KimodoCascadeur\text-encoders\generated\LLM2Vec-Meta-Llama-3-8B-Instruct-merged\*',
    'KimodoCascadeur\text-encoders\meta-llama\Meta-Llama-3-8B-Instruct\LICENSE',
    'KimodoCascadeur\text-encoders\meta-llama\Meta-Llama-3-8B-Instruct\README.md') $modelsRoot
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -eq 0) {
        throw "Missing extracted component: $path"
    }
}
$receipt = [ordered]@{
    archive = $archiveInfo.FullName
    archive_bytes = $archiveInfo.Length
    archive_modified_utc = $archiveInfo.LastWriteTimeUtc.ToString('o')
    source_entries = $sourceEntries.Count
    model_entries = $modelEntries.Count
    motion_checkpoint = 'Download Kimodo-SOMA-RP-v1.2 separately; no archived motion checkpoint extracted'
    extracted_utc = [DateTime]::UtcNow.ToString('o')
}
$receipt | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding UTF8
Write-Host "Ready: $dataRoot"
