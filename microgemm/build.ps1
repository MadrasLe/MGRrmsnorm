$ErrorActionPreference = "Stop"

$cl = Get-Command cl -ErrorAction SilentlyContinue
if (-not $cl) {
    Write-Error "MSVC cl.exe was not found in PATH. Open a Developer PowerShell and run this script again."
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$outDir = Join-Path $root "build"
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$include = "/Iinclude"
$cflags = @("/nologo", "/O2", "/std:c11", "/W4", "/arch:AVX2", "/openmp", "/D_CRT_SECURE_NO_WARNINGS", "/DMICROGEMM_FORCE_AVX2=1", $include)
$cxxflags = @("/nologo", "/O2", "/std:c++17", "/W4", "/EHsc", "/arch:AVX2", "/openmp", "/D_CRT_SECURE_NO_WARNINGS", "/DMICROGEMM_FORCE_AVX2=1", $include)
$coreSources = @(
    "src\microgemm_format.c",
    "src\microgemm_runtime.c",
    "src\microgemm_ops_cpu.c",
    "src\microgemm_decode_cpu.c",
    "src\microgemm_model_i8.c"
)

$cliSources = @("src\microgemm_cli.c")
$convertSources = @("src\microgemm_convert.cpp")
$textSources = @("src\microgemm_text.cpp")

Push-Location $outDir
try {
    foreach ($src in $coreSources) {
        & cl @cflags /c (Join-Path $root $src)
    }

    & lib /nologo /OUT:microgemm.lib *.obj

    foreach ($src in $cliSources) {
        & cl @cflags /c (Join-Path $root $src)
    }

    & cl /nologo /openmp /Fe:microgemm.exe microgemm_cli.obj microgemm.lib

    foreach ($src in $convertSources) {
        & cl @cxxflags /c (Join-Path $root $src)
    }

    & cl /nologo /EHsc /openmp /Fe:microgemm-convert.exe microgemm_convert.obj microgemm.lib

    foreach ($src in $textSources) {
        & cl @cxxflags /c (Join-Path $root $src)
    }

    & cl /nologo /EHsc /openmp /Fe:microgemm-text.exe microgemm_text.obj microgemm.lib
}
finally {
    Pop-Location
}
