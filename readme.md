\*to use with PsychoyPy v2025.1.1; (02.10.2025) - downloadlink: https://github.com/psychopy/psychopy/releases/tag/2025.1.1

## Installation

### For Linux
This project includes a dedicated PsychoPy installer for Linux, based on [psychopy_linux_installer](https://github.com/wieluk/psychopy_linux_installer).

To install PsychoPy on Linux, run:
```bash
./install/psychopy_linux_installer.sh --psychopy-version=2025.1.1 --additional-packages=psychopy-bids,sounddevice,soundfile
```

### For macOS
This project includes a dedicated PsychoPy installer for macOS, based on [psychopy_linux_installer](https://github.com/wieluk/psychopy_linux_installer) adapted for macOS.

To install PsychoPy on macOS, ensure you have Homebrew installed, then run:
```bash
# Use the adapted installer (requires bash 5.x, install via brew if needed)
brew install bash
/opt/homebrew/opt/bash/bin/bash ./install/psychopy_linux_installer.sh --psychopy-version=2025.1.1 --additional-packages=psychopy-bids,sounddevice,soundfile
```

Alternatively, install via pip in a virtual environment using UV:
```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
# Create and activate virtual environment
uv venv
source .venv/bin/activate
# Install PsychoPy and dependencies
uv pip install psychopy==2025.1.1
uv pip install -r requirements.txt
```

Additional dependencies:
```bash
uv pip install -r requirements.txt
```

## Hardware Setup

PsychoPy requires specific hardware configurations to work properly across different machines. Run the hardware setup script before running the experiment:

```bash
# Using the PsychoPy virtual environment
/path/to/psychopy/.venv/bin/python hardware_setup.py
```

This script:
- **Checks** hardware availability (audio devices, keyboard, mouse)
- **Tests** backend functionality 
- **Configures** preferences based on what works
- **Reports** available devices and any issues

It configures:
- Audio backend preferences (automatically detects working backends)
- Keyboard backend (event backend to avoid issues with ptb on macOS)
- Microphone backend (for audio recording)
- Speaker backend (for audio output)
- Mouse backend (event backend)

## Running the Experiment

1. Ensure PsychoPy is installed as described above.
2. Run the hardware setup script:
   ```bash
   # Using the PsychoPy virtual environment
   /path/to/psychopy/.venv/bin/python hardware_setup.py
   ```
3. Run the experiment:
   ```bash
   # From PsychoPy GUI
   /path/to/psychopy/start_psychopy MR_AUT.psyexp
   
   # Or from command line
   /path/to/psychopy/.venv/bin/python MR_AUT_lastrun.py
   ```

## Troubleshooting

- If you encounter audio issues, ensure sounddevice is installed and configured.
- On macOS, if keyboard input fails, the hardware setup script should resolve this.
- For parallel port issues on Linux, ensure appropriate permissions are set (handled by the installer).
- On macOS, parallel port is not supported - warnings are expected for MRI trigger functionality.
- If using additional hardware (eyetracker, button box, etc.), ensure appropriate drivers and permissions are set.

## Hardware Components

The MR_AUT experiment uses the following hardware components:
- **Audio**: Microphone recording via sounddevice, audio output
- **Keyboard**: Response input
- **Mouse**: Alternative response input
- **Parallel Port**: MRI triggers (Linux only)

The hardware setup script configures the essential components. For advanced hardware like eyetrackers or specialized input devices, additional configuration may be needed.

- - - - - - - 
g(o4[g])it