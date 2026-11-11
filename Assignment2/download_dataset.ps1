$ErrorActionPreference = "Stop"

$datasetDirectory = Join-Path $PSScriptRoot "dataset"
$downloadDirectory = Join-Path $PSScriptRoot ".dataset-download"
$archive = Join-Path $downloadDirectory "covid-19-nlp-text-classification.zip"
$datasetSlug = "datatattle/covid-19-nlp-text-classification"

New-Item -ItemType Directory -Force -Path $datasetDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $downloadDirectory | Out-Null

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw "Kaggle CLI was not found. Install it with: python -m pip install kaggle"
}

$kaggleArguments = @(
    "datasets", "download",
    "-d", $datasetSlug,
    "-p", $downloadDirectory,
    "--force"
)
& kaggle @kaggleArguments
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle download failed. Configure Kaggle API credentials and try again."
}

if (-not (Test-Path $archive)) {
    throw "Kaggle did not create the expected archive: $archive"
}

Expand-Archive -Path $archive -DestinationPath $datasetDirectory -Force

$expectedFiles = @("Corona_NLP_train.csv", "Corona_NLP_test.csv")
foreach ($fileName in $expectedFiles) {
    $path = Join-Path $datasetDirectory $fileName
    if (-not (Test-Path $path)) {
        throw "Expected dataset file was not found after extraction: $path"
    }

    $sample = Import-Csv -Path $path | Select-Object -First 1
    if ($null -eq $sample.PSObject.Properties["OriginalTweet"] -or
        $null -eq $sample.PSObject.Properties["Sentiment"]) {
        throw "$fileName does not contain the expected OriginalTweet and Sentiment columns."
    }

    $rows = (Import-Csv -Path $path | Measure-Object).Count
    Write-Host "$fileName`: $rows rows"
}

Write-Host "Dataset is ready in $datasetDirectory"
