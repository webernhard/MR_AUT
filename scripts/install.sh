#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=${VENV_DIR:-.venv}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
RECREATE=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --recreate) RECREATE=1; shift ;;
    --venv|-v) VENV_DIR="$2"; shift 2 ;;
    --python) PYTHON_VERSION="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

echo "========================================" 
echo "MR_AUT Installation"
echo "========================================"
echo "Platform: $(uname -s)"
echo "Python: ${PYTHON_VERSION}"
echo "Venv: ${VENV_DIR}"
echo ""

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found"
  echo "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "✓ UV: $(uv --version)"

if [ -d "${VENV_DIR}" ] && [ "${RECREATE}" -eq 1 ]; then
  echo "Recreating venv..."
  rm -rf "${VENV_DIR}"
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating venv..."
  uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
fi

export VIRTUAL_ENV="${PWD}/${VENV_DIR}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

echo "Installing packages..."
failed_packages=()

while IFS= read -r line || [ -n "$line" ]; do
  pkg=$(echo "$line" | sed 's/#.*//' | xargs)
  [ -z "$pkg" ] && continue
  
  echo "  $pkg"
  if ! uv pip install "$pkg" 2>&1 | grep -q "Installed\|Audited"; then
    failed_packages+=("$pkg")
  fi
done < requirements.txt

if [ ${#failed_packages[@]} -gt 0 ]; then
  echo "Some packages failed, installing system deps..."
  case "$(uname -s)" in
    Darwin) brew install hdf5 portaudio pkg-config libsndfile 2>/dev/null || true ;;
    Linux) sudo apt-get install -y build-essential libhdf5-dev libsndfile1-dev portaudio19-dev 2>/dev/null || true ;;
  esac
  
  for pkg in "${failed_packages[@]}"; do
    uv pip install "$pkg" || echo "Still failed: $pkg"
  done
fi

echo "Setting up ffmpeg..."
FFMPEG_PATH=$("${VENV_DIR}/bin/python" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || echo "")
if [ -n "$FFMPEG_PATH" ] && [ -f "$FFMPEG_PATH" ]; then
  ln -sf "$FFMPEG_PATH" "${VENV_DIR}/bin/ffmpeg"
  echo "✓ ffmpeg linked"
fi

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo "Run: MR_AUT_ALLOW_RUN=1 ${VENV_DIR}/bin/python MR_AUT_enhanced.py --ui --count 3"
echo ""
