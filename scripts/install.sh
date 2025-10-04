#!/usr/bin/env bash
set -euo pipefail

# MR_AUT installation script
# Based on proven patterns from psychopy_linux_installer
# Uses UV as primary package manager, falls back to system packages only when needed

VENV_DIR=${VENV_DIR:-.venv}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
RECREATE=0

# Parse arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --recreate) RECREATE=1; shift ;;
    --venv|-v) VENV_DIR="$2"; shift 2 ;;
    --python) PYTHON_VERSION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; echo "Usage: $0 [--recreate] [--venv DIR] [--python VERSION]"; exit 1 ;;
  esac
done

echo "========================================"
echo "MR_AUT Installation Script"
echo "========================================"
echo "Platform: $(uname -s)"
echo "Python version: ${PYTHON_VERSION}"
echo "Venv directory: ${VENV_DIR}"
echo ""

# Check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
  echo "❌ ERROR: 'uv' not found."
  echo ""
  echo "Please install UV first:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo ""
  echo "Then reload your shell or run:"
  echo "  source \$HOME/.cargo/env"
  exit 1
fi

echo "✓ UV found: $(uv --version)"

# Recreate venv if requested
if [ -d "${VENV_DIR}" ] && [ "${RECREATE}" -eq 1 ]; then
  echo ""
  echo "🗑️  Recreating virtual environment..."
  rm -rf "${VENV_DIR}"
fi

# Create venv using uv with specific Python version
if [ ! -d "${VENV_DIR}" ]; then
  echo ""
  echo "📦 Creating virtual environment with Python ${PYTHON_VERSION}..."
  uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
else
  echo ""
  echo "✓ Using existing virtual environment at ${VENV_DIR}"
fi

# Activate venv for uv operations
export VIRTUAL_ENV="${PWD}/${VENV_DIR}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

echo ""
echo "📥 Installing Python packages via UV..."
echo ""

# Read requirements.txt and install packages one by one to track failures
failed_packages=()

if [ ! -f "requirements.txt" ]; then
  echo "❌ ERROR: requirements.txt not found"
  exit 1
fi

# Install packages one by one
while IFS= read -r line || [ -n "$line" ]; do
  # Skip comments and empty lines
  pkg=$(echo "$line" | sed 's/#.*//' | xargs)
  if [ -z "$pkg" ]; then
    continue
  fi
  
  echo "  Installing: $pkg"
  if uv pip install "$pkg"; then
    echo "    ✓ OK"
  else
    echo "    ✗ FAILED"
    failed_packages+=("$pkg")
  fi
done < requirements.txt

echo ""
echo "========================================"
echo "Installation Summary"
echo "========================================"

if [ ${#failed_packages[@]} -eq 0 ]; then
  echo "✓ All packages installed successfully!"
else
  echo "⚠️  Some packages failed to install:"
  for pkg in "${failed_packages[@]}"; do
    echo "  - $pkg"
  done
  echo ""
  echo "Installing system dependencies and retrying..."
  
  # Install system dependencies based on platform
  case "$(uname -s)" in
    Darwin)
      echo ""
      echo "📦 Installing macOS system dependencies via Homebrew..."
      if ! command -v brew >/dev/null 2>&1; then
        echo "❌ ERROR: Homebrew not found. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
      fi
      
      # Minimal system packages needed for PsychoPy
      brew install hdf5 portaudio pkg-config libsndfile || true
      ;;
      
    Linux)
      echo ""
      echo "📦 Installing Pop!_OS/Ubuntu system dependencies via apt..."
      if ! command -v apt-get >/dev/null 2>&1; then
        echo "⚠️  WARNING: apt-get not found. Please install system dependencies manually."
      else
        sudo apt-get update
        sudo apt-get install -y \
          build-essential \
          libhdf5-dev \
          libsndfile1-dev \
          libportaudio2 \
          portaudio19-dev \
          pkg-config \
          python3-dev \
          libffi-dev || true
      fi
      ;;
  esac
  
  echo ""
  echo "🔄 Retrying failed packages..."
  for pkg in "${failed_packages[@]}"; do
    echo "  Retrying: $pkg"
    if uv pip install "$pkg"; then
      echo "    ✓ OK"
    else
      echo "    ✗ Still failed"
    fi
  done
fi

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Activate the virtual environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Run the experiment with:"
echo "  ${VENV_DIR}/bin/python MR_AUT_enhanced.py --ui --count 3"
echo ""
echo "Or with environment guard for macOS:"
echo "  MR_AUT_ALLOW_RUN=1 ${VENV_DIR}/bin/python MR_AUT_enhanced.py --ui --count 3"
echo ""

exit 0
