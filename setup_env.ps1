# Unified Environment Setup Script for Windows PowerShell
# This script creates a single virtual environment for all LLM projects

Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host "  Unified LLM Environment Setup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Setting up environment for:" -ForegroundColor White
Write-Host "  - Falcon-7B (all modes)" -ForegroundColor Green
Write-Host "  - Mistral-7B (base + fine-tuned)" -ForegroundColor Green
Write-Host "  - API Server" -ForegroundColor Green
Write-Host "  - Training scripts" -ForegroundColor Green
Write-Host "=============================================`n" -ForegroundColor Cyan

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  Found: $pythonVersion" -ForegroundColor White

if ($pythonVersion -notmatch "Python 3\.(10|11)") {
    Write-Host "  ⚠️  Warning: Python 3.10 or 3.11 recommended" -ForegroundColor Yellow
}

# Create virtual environment
Write-Host "`nCreating virtual environment 'llm-env'..." -ForegroundColor Yellow
if (Test-Path "llm-env") {
    Write-Host "  ⚠️  Directory 'llm-env' already exists!" -ForegroundColor Yellow
    $response = Read-Host "  Delete and recreate? (y/n)"
    if ($response -eq "y") {
        Write-Host "  Removing old environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force llm-env
    } else {
        Write-Host "  Keeping existing environment." -ForegroundColor White
        Write-Host "  Skipping to dependency installation..." -ForegroundColor White
        $skipVenvCreation = $true
    }
}

if (-not $skipVenvCreation) {
    python -m venv llm-env
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ❌ Failed to create virtual environment!" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✅ Virtual environment created" -ForegroundColor Green
}

# Activate environment
Write-Host "`nActivating environment..." -ForegroundColor Yellow
& .\llm-env\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Failed to activate environment!" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ Environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "  ✅ pip upgraded" -ForegroundColor Green

# Install PyTorch with CUDA
Write-Host "`nInstalling PyTorch with CUDA support..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor White
pip install torch --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Failed to install PyTorch!" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ PyTorch installed" -ForegroundColor Green

# Verify CUDA
Write-Host "`nVerifying CUDA availability..." -ForegroundColor Yellow
$cudaCheck = python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>&1
if ($cudaCheck -match "CUDA") {
    $gpuName = python -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
    Write-Host "  ✅ CUDA is available!" -ForegroundColor Green
    Write-Host "  GPU: $gpuName" -ForegroundColor White
} else {
    Write-Host "  ⚠️  CUDA not available - will use CPU (slow)" -ForegroundColor Yellow
}

# Install other dependencies
Write-Host "`nInstalling remaining dependencies..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor White
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Failed to install dependencies!" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ All dependencies installed" -ForegroundColor Green

# Verify installations
Write-Host "`nVerifying installations..." -ForegroundColor Yellow
$verifyScript = @"
import torch
import transformers
import peft
import trl
import flask
import bitsandbytes
print('✅ All key packages imported successfully')
"@

$verifyResult = python -c $verifyScript 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  $verifyResult" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Some imports failed (this may be okay)" -ForegroundColor Yellow
}

# Summary
Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host "  ✅ Setup Complete!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor White
Write-Host "  1. Activate environment:" -ForegroundColor Yellow
Write-Host "     .\llm-env\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "`n  2. Run API server:" -ForegroundColor Yellow
Write-Host "     python api_server.py --model both" -ForegroundColor White
Write-Host "`n  3. Or run Falcon chat:" -ForegroundColor Yellow
Write-Host "     cd falcon-7b-various-experiments" -ForegroundColor White
Write-Host "     python chat_base.py" -ForegroundColor White
Write-Host "`n  4. Or run Mistral chat:" -ForegroundColor Yellow
Write-Host "     cd mistral-7b-finetune" -ForegroundColor White
Write-Host "     python chat_finetuned.py" -ForegroundColor White
Write-Host "`n=============================================" -ForegroundColor Cyan

# Optional: Clean up old environments
Write-Host "`nOptional: Remove old separate environments?" -ForegroundColor Yellow
Write-Host "  This will delete:" -ForegroundColor White
Write-Host "    - falcon-7b-various-experiments\falcon-env" -ForegroundColor Gray
Write-Host "    - mistral-7b-finetune\mistral-env" -ForegroundColor Gray
$cleanup = Read-Host "  Remove old environments to save disk space? (y/n)"
if ($cleanup -eq "y") {
    if (Test-Path "falcon-7b-various-experiments\falcon-env") {
        Write-Host "  Removing falcon-env..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force falcon-7b-various-experiments\falcon-env
        Write-Host "  ✅ Removed falcon-env" -ForegroundColor Green
    }
    if (Test-Path "mistral-7b-finetune\mistral-env") {
        Write-Host "  Removing mistral-env..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force mistral-7b-finetune\mistral-env
        Write-Host "  ✅ Removed mistral-env" -ForegroundColor Green
    }
    Write-Host "  💾 Disk space saved!" -ForegroundColor Green
}

Write-Host "`n🎉 You're all set! Happy coding!" -ForegroundColor Green
