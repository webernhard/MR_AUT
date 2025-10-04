# MR_AUT_enhanced — compact enhanced AUT script

This repository contains a compact enhanced version of the MR_AUT experiment: `MR_AUT_enhanced.py`.

Features:
- Waits for scanner trigger key '5'
- Displays items from `stim/MR_AUT_items.csv`
- Sends parallel-port markers when available (no-op if not)
- Records audio between SPACE (start) and SPACE (stop), saves WAV per participant/item
- Simple numeric creativity rating (0-9)
- Optional BIDS event logging if `psychopy-bids` is installed

Safety:
- On macOS the script is disabled by default; set `MR_AUT_ALLOW_RUN=1` in the environment to opt in.

Quick start (macOS, if you opt in):
```bash
source .venv/bin/activate
MR_AUT_ALLOW_RUN=1 python MR_AUT_enhanced.py
```

Install dependencies (recommended inside a venv):
```bash
python -m pip install -r requirements.txt
```

Notes:
- If you don't have a parallel port, the marker writes are skipped (logged as warnings).
- If `sounddevice`/`soundfile`/`numpy` are not available, audio recording/saving will be skipped.
- The script uses Psychopy for windowing and keyboard input; ensure your venv has it installed.
