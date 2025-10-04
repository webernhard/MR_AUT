"""Utility helpers for MR_AUT that avoid importing heavy dependencies.

Keep small, pure-Python helpers here so tests can run in CI without PsychoPy or audio libs.
"""
import os
import csv
import logging
from datetime import datetime

try:
    import numpy as np
    import soundfile as sf
except Exception:
    np = None
    sf = None


def safe_makedirs(p):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass


def read_items(csv_path):
    items = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'MR_AUTitem' in row and row['MR_AUTitem'] is not None:
                items.append(row['MR_AUTitem'])
            else:
                items.append(next(iter(row.values())))
    return items


def sanitize_filename(s):
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    return ''.join(c if c in keep else '_' for c in str(s))[:80]


def save_wav(audio_frames, samplerate, filename):
    """Save concatenated audio frames to filename. Returns True on success, False otherwise.

    This function is resilient when numpy/soundfile are missing and will return False.
    """
    if sf is None or np is None:
        logging.warning('soundfile/numpy missing; cannot save audio')
        return False
    if not audio_frames:
        logging.warning('No audio frames to save')
        return False
    try:
        recording = np.concatenate(audio_frames, axis=0)
        sf.write(filename, recording, samplerate)
        return True
    except Exception as e:
        logging.warning(f'Failed writing wav: {e}')
        return False
