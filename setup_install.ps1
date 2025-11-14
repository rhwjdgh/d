$BASE = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $BASE

if (-not (Test-Path "$BASE\\.venv")) {
    Write-Output "Creating virtual environment .venv..."
    python -m venv .venv
} else {
    Write-Output ".venv already exists"
}

# Use the venv python directly to avoid activation issues in different shells
$venvPython = Join-Path $BASE ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Output "ERROR: venv python not found at $venvPython"
} else {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
}

# remove stray token-only lines from secom_analysis.py
$scriptPath = Join-Path $BASE 'secom_analysis.py'
if (Test-Path $scriptPath) {
    (Get-Content $scriptPath) | Where-Object { $_ -notmatch '^[ \t]*(?:numpy|pandas|matplotlib|seaborn)[ \t]*$' } | Set-Content $scriptPath -Encoding UTF8
    Write-Output "sanitized secom_analysis.py"
} else {
    Write-Output "secom_analysis.py not found"
}

Write-Output "Setup complete. In VS Code select Interpreter: $BASE\\.venv\\Scripts\\python.exe and reload window."
