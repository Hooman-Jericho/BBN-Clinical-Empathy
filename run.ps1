# BBN Annotation Tool - PowerShell Launcher
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Launching BBN Clinical Empathy Annotation Tool..." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path "$ScriptDir\venv\Scripts\Activate.ps1") {
    & "$ScriptDir\venv\Scripts\Activate.ps1"
}

try {
    streamlit run "$ScriptDir\app\main.py" --server.port 8501
} catch {
    python -m streamlit run "$ScriptDir\app\main.py" --server.port 8501
}
