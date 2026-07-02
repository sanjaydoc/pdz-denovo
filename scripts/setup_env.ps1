# =============================================================================
# setup_env.ps1  --  One-time environment setup for pdz-denovo (Windows)
#
# Creates a project-local virtual environment (.venv), upgrades pip, installs
# the CUDA build of PyTorch (cu121, suitable for the Quadro RTX 3000), then
# installs the remaining fully open-source requirements.
#
# Usage:   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
# =============================================================================
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPy  = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "== pdz-denovo environment setup ==" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"

if (-not (Test-Path $Python)) {
    throw "Python 3.10 not found at $Python. Install it first."
}

# 1. Create venv
if (-not (Test-Path $VenvPy)) {
    Write-Host "Creating virtual environment at $VenvDir ..." -ForegroundColor Yellow
    & $Python -m venv $VenvDir
} else {
    Write-Host ".venv already exists; reusing it." -ForegroundColor Green
}

# 2. Upgrade pip tooling
Write-Host "Upgrading pip / setuptools / wheel ..." -ForegroundColor Yellow
& $VenvPy -m pip install --upgrade pip setuptools wheel

# 3. Install CUDA PyTorch (cu121)
Write-Host "Installing PyTorch (CUDA cu121) ..." -ForegroundColor Yellow
& $VenvPy -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining requirements
Write-Host "Installing project requirements ..." -ForegroundColor Yellow
& $VenvPy -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

# 5. Install the package itself (editable)
Write-Host "Installing pdz-denovo (editable) ..." -ForegroundColor Yellow
& $VenvPy -m pip install -e $ProjectRoot

# 6. Verify torch + CUDA
Write-Host "Verifying PyTorch / CUDA ..." -ForegroundColor Yellow
& $VenvPy -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host "== Setup complete. Activate with: .\.venv\Scripts\Activate.ps1 ==" -ForegroundColor Cyan
