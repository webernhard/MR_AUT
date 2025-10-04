# MR_AUT Installation Guide

## Prerequisites

1. **Install UV package manager** (required):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.cargo/env
   ```

## Installation

Run the installation script:

```bash
./scripts/install.sh --recreate
```

This will:
- ✅ Create a Python 3.10 virtual environment using UV
- ✅ Install all required packages (PsychoPy, Whisper, audio tools, etc.)
- ✅ Set up ffmpeg symlink for Whisper transcription
- ✅ Install minimal system dependencies if needed (Homebrew on macOS, apt on Pop!_OS)

## Usage

### Activate the environment
```bash
source .venv/bin/activate
```

### Run the experiment
```bash
# Interactive UI mode with 3 items (recommended for testing)
MR_AUT_ALLOW_RUN=1 .venv/bin/python MR_AUT_enhanced.py --ui --count 3

# Full experiment with all items
MR_AUT_ALLOW_RUN=1 .venv/bin/python MR_AUT_enhanced.py
```

### Transcribe recordings
```bash
# Transcribe with Whisper (German, small model)
.venv/bin/python transcribe_wavs.py --wav-dir data/YOURFOLDER --model small --language de

# Use larger model for better accuracy
.venv/bin/python transcribe_wavs.py --wav-dir data/YOURFOLDER --model medium --language de
```

## Troubleshooting

### "uv: command not found"
Install UV as shown in Prerequisites, then reload your shell or run:
```bash
source $HOME/.cargo/env
```

### Transcription fails with "ffmpeg not found"
The installer should create the symlink automatically. If it's missing, run:
```bash
ln -sf $(.venv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") .venv/bin/ffmpeg
```

### PsychoPy fails to start on macOS
Make sure to set the environment variable:
```bash
export MR_AUT_ALLOW_RUN=1
```

## Platform Support

- ✅ macOS (Apple Silicon & Intel)
- ✅ Pop!_OS / Ubuntu / Debian Linux

