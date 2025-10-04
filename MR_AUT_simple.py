#!/usr/bin/env python3
"""
MR_AUT_simple.py

Minimal AUT paradigm:
- wait for MRI trigger key '5'
- show items from stim/MR_AUT_items.csv (column: MR_AUTitem)
- for each item: display it; start recording on space press; stop/save on next space press

Safe defaults: on macOS the script will exit unless MR_AUT_ALLOW_RUN=1 is set.
"""
import os
import sys
import csv
import time
import platform
from datetime import datetime

try:
    from psychopy import visual, core, event
except Exception as e:
    print('PsychoPy import failed:', e)
    raise

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except Exception:
    sd = None
    sf = None
    np = None


DATA_DIR = os.path.join(os.getcwd(), 'data')
WAV_DIR = os.path.join(DATA_DIR, 'wav')
CSV_PATH = os.path.join(os.getcwd(), 'stim', 'MR_AUT_items.csv')


def safe_makedirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def read_items(csv_path):
    items = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'MR_AUTitem' in row:
                items.append(row['MR_AUTitem'])
            else:
                # fallback: take first column
                items.append(next(iter(row.values())))
    return items


def save_wav(audio_frames, samplerate, filename):
    if sf is None or np is None:
        print('soundfile/numpy not available; skipping save')
        return False
    if not audio_frames:
        print('No audio captured; not saving')
        return False
    arr = np.concatenate(audio_frames, axis=0)
    sf.write(filename, arr, samplerate)
    return True


def record_until_toggle(samplerate=44100, channels=1, start_callback=None, stop_callback=None,
                        start_key='space', stop_key='space', abort_key='escape'):
    """Record when user presses start_key, stop when presses stop_key again.
    Returns path to saved file (or None).
    """
    if sd is None:
        print('sounddevice not available; skipping recording')
        return None

    audio_frames = []
    stream = None

    # wait for start
    while True:
        keys = event.getKeys(keyList=[start_key, abort_key])
        if keys:
            if abort_key in keys:
                return None
            if start_key in keys:
                break
        core.wait(0.01)

    if start_callback:
        start_callback()

    try:
        stream = sd.InputStream(samplerate=samplerate, channels=channels,
                                callback=lambda indata, frames, time, status: audio_frames.append(indata.copy()))
        stream.start()
    except Exception as e:
        print('Failed to start recording:', e)
        return None

    # now wait for stop
    while True:
        keys = event.getKeys(keyList=[stop_key, abort_key])
        if keys:
            if abort_key in keys:
                try:
                    stream.stop(); stream.close()
                except Exception:
                    pass
                return None
            if stop_key in keys:
                break
        core.wait(0.01)

    # stop
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass

    if stop_callback:
        stop_callback()

    # save wav
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(WAV_DIR, f'recording_{timestamp}.wav')
    saved = save_wav(audio_frames, samplerate, filename)
    return filename if saved else None


def main():
    # safety guard on macOS
    if platform.system() == 'Darwin' and os.environ.get('MR_AUT_ALLOW_RUN', '0') != '1':
        print('MR_AUT_simple: disabled on macOS by default. Set MR_AUT_ALLOW_RUN=1 to proceed.')
        return

    if not os.path.isfile(CSV_PATH):
        print(f'Missing stimulus CSV: {CSV_PATH}')
        return

    safe_makedirs(WAV_DIR)
    items = read_items(CSV_PATH)
    win = visual.Window(fullscr=True, color='black', units='height')

    instr = visual.TextStim(win, text="Waiting for scanner ('5')...", color='white', height=0.04)
    instr.draw(); win.flip()

    # wait for scanner trigger
    event.clearEvents()
    event.waitKeys(keyList=['5', 'escape'])

    # run items
    for item in items:
        # show fixation
        fix = visual.TextStim(win, text='+', color='white', height=0.15)
        fix.draw(); win.flip()
        core.wait(0.5)

        # show item
        stim = visual.TextStim(win, text=item, color='white', height=0.07)
        stim.draw(); win.flip()

        # record on space toggles
        def on_start():
            stim.text = 'RECORDING... (press SPACE to stop)'
            stim.color = 'green'
            stim.draw(); win.flip()

        def on_stop():
            stim.text = item
            stim.color = 'white'
            stim.draw(); win.flip()

        saved = record_until_toggle(start_callback=on_start, stop_callback=on_stop)
        print('Saved audio:', saved)

        # short inter-trial
        core.wait(0.2)

        # allow escape to quit
        if event.getKeys(keyList=['escape']):
            break

    # goodbye
    bye = visual.TextStim(win, text='Done. Thanks!', color='white', height=0.05)
    bye.draw(); win.flip()
    core.wait(1.0)
    win.close()


if __name__ == '__main__':
    main()
