#!/usr/bin/env python3
"""
MR_AUT_enhanced.py

Compact, enhanced AUT paradigm that restores key features from the original:
- wait for scanner trigger ('5')
- show items from stim/MR_AUT_items.csv
- send parallel-port markers when available
- record audio between SPACE press and next SPACE press (safe handling)
- save WAV per participant/item with timestamp
- simple numeric creativity rating (0-9) via keyboard
- optional BIDS events if psychopy-bids is installed

This script is compact (~300 lines) but includes the essential features.
It will not run on macOS unless MR_AUT_ALLOW_RUN=1 is set (safety guard).
"""
import os
import sys
import csv
import time
import json
import platform
from datetime import datetime

from psychopy import visual, core, event, logging
from psychopy.hardware import keyboard

# Monkey-patch PsychoPy KeyboardDevice methods to avoid crashes on macOS
try:
    from psychopy.hardware.keyboard import KeyboardDevice
    _orig_kbd_stop = KeyboardDevice.stop
    _orig_kbd_clear = KeyboardDevice.clearEvents
    _orig_kbd_dispatch = KeyboardDevice.dispatchMessages
    _orig_kbd_close = getattr(KeyboardDevice, 'close', None)

    def _safe_stop(self):
        try:
            return _orig_kbd_stop(self)
        except AttributeError as e:
            if '_buffers' in str(e):
                logging.warning('KeyboardDevice.stop skipped: _buffers missing')
                return
            raise

    def _safe_clear(self, eventType=None):
        try:
            return _orig_kbd_clear(self, eventType=eventType)
        except AttributeError as e:
            if '_buffers' in str(e):
                logging.warning('KeyboardDevice.clearEvents skipped: _buffers missing')
                return []
            raise

    def _safe_dispatch(self):
        try:
            return _orig_kbd_dispatch(self)
        except AttributeError as e:
            if '_buffers' in str(e):
                logging.warning('KeyboardDevice.dispatchMessages skipped: _buffers missing')
                return
            raise

    def _safe_close(self):
        try:
            if _orig_kbd_close is not None:
                return _orig_kbd_close(self)
            else:
                # fallback: try stop
                return _safe_stop(self)
        except AttributeError as e:
            if '_buffers' in str(e):
                logging.warning('KeyboardDevice.close skipped: _buffers missing')
                return
            raise

    KeyboardDevice.stop = _safe_stop
    KeyboardDevice.clearEvents = _safe_clear
    KeyboardDevice.dispatchMessages = _safe_dispatch
    KeyboardDevice.close = _safe_close
except Exception:
    # If anything goes wrong patching, don't fail startup
    pass

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except Exception:
    sd = sf = np = None

# Optional features (parallel, bids)
try:
    from psychopy import parallel
    PARALLEL_AVAILABLE = True
except Exception:
    parallel = None
    PARALLEL_AVAILABLE = False

try:
    from psychopy_bids import BIDSHandler, BIDSTaskEvent, BIDSError
    BIDS_AVAILABLE = True
except Exception:
    BIDS_AVAILABLE = False


ROOT = os.getcwd()
CSV_PATH = os.path.join(ROOT, 'stim', 'MR_AUT_items.csv')
DATA_DIR = os.path.join(ROOT, 'data')
WAV_DIR = os.path.join(DATA_DIR, 'wav')


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
            if 'MR_AUTitem' in row:
                items.append(row['MR_AUTitem'])
            else:
                items.append(next(iter(row.values())))
    return items


def sanitize_filename(s):
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    return ''.join(c if c in keep else '_' for c in s)[:80]


def init_parallel():
    # Parallel port is disabled on purpose (platform-incompatible). Return None.
    return None


def safe_setData(port, val):
    # Parallel port disabled: no-op to avoid platform warnings and failures.
    return


def _sanitize_participant(p):
    if p is None:
        return 'sub-unknown'
    s = str(p)
    if s.startswith('sub-'):
        return s
    return 'sub-' + s


def save_wav(audio_frames, samplerate, filename):
    if sf is None or np is None:
        logging.warning('soundfile/numpy missing; cannot save audio')
        return False
    if not audio_frames:
        logging.warning('No audio frames to save')
        return False
    recording = np.concatenate(audio_frames, axis=0)
    try:
        sf.write(filename, recording, samplerate)
        return True
    except Exception as e:
        logging.warning(f'Failed writing wav: {e}')
        return False


def record_space_toggle(win, background=None, start_key='1', stop_key='1', samplerate=44100, channels=1, prompt_text=None):
    """Wait for start_key to start, stop_key to stop. Returns list of frames or None if aborted.

    If `background` is provided, it will be redrawn under prompts so the item stays visible.
    """
    if sd is None:
        logging.warning('sounddevice not available; skipping recording')
        return None

    audio_frames = []
    stream = None

    # show start prompt (redraw background first if available)
    if prompt_text:
        prompt = visual.TextStim(win, text=prompt_text, color='white', height=0.05)
        if background is not None:
            background.draw()
        prompt.draw(); win.flip()

    event.clearEvents()

    # wait for start_key to start or ESC to abort
    while True:
        keys = event.getKeys(keyList=[start_key, 'escape'])
        if 'escape' in keys:
            return None
        if start_key in keys:
            break
        core.wait(0.01)

    # start audio
    try:
        stream = sd.InputStream(samplerate=samplerate, channels=channels,
                                callback=lambda indata, frames, time, status: audio_frames.append(indata.copy()))
        stream.start()
    except Exception as e:
        logging.warning(f'Failed to open audio stream: {e}')
        return None

    # indicate recording on-screen (redraw background if present)
    rec_text = visual.TextStim(win, text=f'RECORDING... press {stop_key} to stop', color='green', height=0.06)
    if background is not None:
        background.draw()
    rec_text.draw(); win.flip()

    # wait for stop_key
    while True:
        keys = event.getKeys(keyList=[stop_key, 'escape'])
        if 'escape' in keys:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass
            return None
        if stop_key in keys:
            break
        core.wait(0.01)

    try:
        stream.stop(); stream.close()
    except Exception:
        pass

    return audio_frames


def get_rating(win, prompt='Rate creativity 0-9 (press digit)'):
    txt = visual.TextStim(win, text=prompt, color='white', height=0.05)
    txt.draw(); win.flip()
    while True:
        keys = event.getKeys(keyList=[str(i) for i in range(10)] + ['escape'])
        if keys:
            if 'escape' in keys:
                return None
            k = keys[0]
            return int(k)
        core.wait(0.01)


def main():
    # parse CLI args early so quick-test can bypass macOS guard
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--quick-test', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--duration', type=float, default=2.0)
    parser.add_argument('--force-run', action='store_true')
    parser.add_argument('--count', type=int, default=5, help='Number of items for quick-test')
    parser.add_argument('--ui', action='store_true', help='Run UI test (opens PsychoPy window)')
    args, _ = parser.parse_known_args()

    # safety guard on macOS; allow quick-test or explicit force-run to bypass
    if platform.system() == 'Darwin' and os.environ.get('MR_AUT_ALLOW_RUN', '0') != '1' and not args.quick_test and not args.force_run:
        print('MR_AUT_enhanced: disabled on macOS by default. Set MR_AUT_ALLOW_RUN=1 to proceed.')
        return

    # ask for participant id
    participant = os.environ.get('MR_AUT_SUBJECT') or input('Participant ID (or press Enter to use "sub-XX"): ') or 'sub-XX'

    if not os.path.isfile(CSV_PATH):
        print('Missing stim CSV:', CSV_PATH)
        return

    safe_makedirs(WAV_DIR)

    # optional BIDS
    bids = None
    if BIDS_AVAILABLE:
        try:
            bids = BIDSHandler(dataset='MR_AUT_BIDS', subject=participant, task='MR_AUT')
            bids.createDataset()
            bids.addTaskCode(force=True)
            bids.addEnvironment()
        except Exception as e:
            logging.warning(f'BIDS init failed: {e}')
            bids = None

    # parallel port
    t_mrk = init_parallel()

    items = read_items(CSV_PATH)

    # if running UI test, limit items to the requested count
    if args.ui:
        items = items[: args.count if args.count > 0 else 1]
        print(f'UI test mode: running {len(items)} item(s)')

    # reuse args from earlier parsing (quick-test handled below)

    # quick-test: simulate multiple items without opening the UI (dry-run by default)
    if args.quick_test:
        num = args.count if args.count > 0 else 5
        num = min(num, len(items)) if items else num
        base_dir = globals().get('_thisDir', ROOT)
        exp_name = globals().get('expName', 'MR_AUT')
        wavDirName = os.path.join(base_dir, 'data', f"{participant}_{exp_name}_{datetime.now().strftime('%Y-%m-%d_%H%M')}_wav")
        safe_makedirs(wavDirName)
        print(f'Quick-test: running {num} items (dry-run={args.dry_run})')
        DEBUG = True
        itemDur = 5 if DEBUG else 15
        for i in range(num):
            item = items[i] if i < len(items) else f'TESTITEM_{i+1}'
            # jittered fixation between 3 and 9s
            try:
                rand_dur = float(np.random.uniform(3, 9)) if np is not None else float(__import__('random').uniform(3, 9))
            except Exception:
                import random
                rand_dur = random.uniform(3, 9)
            print(f'Item {i+1}/{num}: "{item}", fixation {rand_dur:.2f}s, itemDur {itemDur}s')
            # simulate fixation and item display (short sleeps in dry-run)
            time.sleep(min(rand_dur, 0.5) if args.dry_run else rand_dur)
            time.sleep(min(itemDur, 0.5) if args.dry_run else itemDur)
            if args.dry_run:
                print(f'Quick-test: simulated recording for item {i+1} (no audio saved)')
            else:
                simulated_file = os.path.join(wavDirName, f"{participant}_{sanitize_filename(item)}_idea{i+1:02d}_simulated.wav")
                try:
                    import numpy as _np
                    import soundfile as _sf
                    sr = 44100
                    samples = int(sr * float(args.duration))
                    data = _np.zeros((samples, 1), dtype=_np.float32)
                    _sf.write(simulated_file, data, sr)
                    print('Quick-test: wrote simulated wav ->', simulated_file)
                except Exception as e:
                    print('Quick-test: could not write wav (missing libs?)', e)
        print('Quick-test complete')
        return

    win = visual.Window(fullscr=True, color='black', units='height')

    # wait for scanner trigger
    instr = visual.TextStim(win, text="Waiting for scanner ('5')...", color='white', height=0.04)
    instr.draw(); win.flip()
    event.clearEvents()
    event.waitKeys(keyList=['5', 'escape'])

    # prepare wav directory like original
    base_dir = globals().get('_thisDir', ROOT)
    exp_name = globals().get('expName', 'MR_AUT')
    wavDirName = os.path.join(base_dir, 'data', f"{participant}_{exp_name}_{datetime.now().strftime('%Y-%m-%d_%H%M')}_wav")
    safe_makedirs(wavDirName)

    # experiment-relative zero: start counting onsets from scanner trigger
    exp_start = time.time()

    # keyboard for recording stop detection
    t_kb = keyboard.Keyboard(backend='ptb')

    recording_count = 1

    # events collected for BIDS-style output
    bids_events = []

    for item in items:
        # fixation + marker with jitter like original (3-9s)
        try:
            rand_dur = float(np.random.uniform(3, 9)) if np is not None else float(__import__('random').uniform(3, 9))
        except Exception:
            import random
            rand_dur = random.uniform(3, 9)
        fix = visual.TextStim(win, text='+', color='white', height=0.15)
        fix.draw(); win.flip()
        safe_setData(t_mrk, 10)  # fixation ON
        # show fixation for randomized duration
        core.wait(rand_dur)
        safe_setData(t_mrk, 0)

        # display item and send marker for itemDur (DEBUG short by default)
        DEBUG = True
        itemDur = 5 if DEBUG else 15
        stim = visual.TextStim(win, text=item, color='white', height=0.07)
        stim.draw(); win.flip()
        safe_setData(t_mrk, 20)  # item ON
        core.wait(itemDur)
        safe_setData(t_mrk, 0)

        # prepare wav filename base (use folder wavDirName, like original)
        MR_AUTitem = sanitize_filename(item)
        audio_data = []
        stream = None

        # send idea-phase marker (49)
        safe_setData(t_mrk, 49); core.wait(0.1); safe_setData(t_mrk, 0)
        trial_onset = time.time()

        # WAIT for participant to press '1' to START recording
        prompt_start = visual.TextStim(win, text="Press '1' to start recording", color='white', height=0.05)
        prompt_start.draw(); win.flip()
        event.clearEvents()
        while True:
            keys = event.getKeys(keyList=['1', 'escape'])
            if 'escape' in keys:
                break
            if '1' in keys:
                # start audio
                try:
                    stream = sd.InputStream(samplerate=44100, channels=1,
                                            callback=lambda indata, frames, time_, status: audio_data.append(indata.copy()))
                    stream.start()
                    # send recording start marker (48)
                    safe_setData(t_mrk, 48); core.wait(0.1); safe_setData(t_mrk, 0)
                except Exception as e:
                    logging.warning(f'Failed to start recording for item "{item}": {e}')
                    stream = None
                break
            core.wait(0.01)

        # show recording indicator while waiting for '1' to stop
        rec_text = visual.TextStim(win, text="RECORDING... press '1' to stop", color='green', height=0.06)
        stim.draw(); rec_text.draw(); win.flip()

        # wait for '1' to stop (or 'escape')
        event.clearEvents()
        while True:
            keys = event.getKeys(keyList=['1', 'escape'])
            if 'escape' in keys:
                break
            if '1' in keys:
                break
            core.wait(0.01)

        # stop stream safely
        if stream is not None:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass

        # save recording if present
        wav_file = None
        if audio_data:
            wav_file = os.path.join(wavDirName, f"{participant}_{MR_AUTitem}_idea{recording_count:02d}.wav")
            try:
                import numpy as _np
                import soundfile as _sf
                recording = _np.concatenate(audio_data, axis=0)
                _sf.write(wav_file, recording, 44100)
                logging.info(f'Saved: {wav_file}')
                recording_count += 1
            except Exception as e:
                logging.warning(f'Failed saving wav for item "{item}": {e}')
        else:
            logging.info('No audio collected for this item')

        # send stop marker (47)
        safe_setData(t_mrk, 47); core.wait(0.1); safe_setData(t_mrk, 0)

        # small inter-marker
        safe_setData(t_mrk, 21)
        core.wait(0.05)
        safe_setData(t_mrk, 0)

        # present analog slider and accept mouse click (trackball)
        # match the original Builder slider styling and labels
        header_pos_y = 0.40
        main_pos_y = 0.0
        btn_pos_y = -0.25
        r_header_txt = visual.TextStim(win=win, text='Wie kreativ findest Du Deine Antwort', pos=(0, header_pos_y),
                                       color='white', height=0.045)
        slider = visual.Slider(win=win, startValue=5, size=(1.0, 0.03), pos=(0, main_pos_y), units='height',
                               labels=None, ticks=[0,1,2,3,4,5,6,7,8,9,10], granularity=0,
                               style=['slider'], labelColor='black', markerColor='Red', lineColor='White', colorSpace='rgb')
        l_label = visual.TextStim(win=win, text='gar nicht kreativ', pos=(-.5, main_pos_y - .075), color='darkgreen', height=0.03)
        r_label = visual.TextStim(win=win, text='sehr kreativ', pos=(.5, main_pos_y - .075), color='darkgreen', height=0.03)
        rating_hint = visual.TextStim(win=win, text='Weiter', pos=(0, btn_pos_y), color='black', height=0.025)

        mouse = event.Mouse(win=win)
        mouse.mouseClock = core.Clock()

        # draw and interact
        r_header_txt.draw(); slider.draw(); l_label.draw(); r_label.draw(); rating_hint.draw(); win.flip()

        # wait for mouse button press to confirm rating
        rating = None
        while True:
            # redraw to show slider movement
            slider.draw(); r_header_txt.draw(); l_label.draw(); r_label.draw(); rating_hint.draw(); win.flip()
            buttons = mouse.getPressed()
            if any(buttons):
                # wait for release to avoid multiple triggers
                while any(mouse.getPressed()):
                    core.wait(0.01)
                rating = slider.getRating()
                break
            core.wait(0.01)
        logging.info(f'Rating for {item}: {rating}')

        # collect event for BIDS TSV (experiment-relative onset)
        try:
            duration = time.time() - trial_onset
        except Exception:
            duration = None
        try:
            onset_rel = float(trial_onset - exp_start)
        except Exception:
            onset_rel = None
        bids_events.append({
            'onset': onset_rel,
            'duration': duration,
            'trial_type': 'AUTitem',
            'response_time': None,
            'rating': rating,
            'wav_file': os.path.basename(wav_file) if wav_file else ''
        })

        # BIDS event (if psychopy_bids available) — use relative onset
        if bids:
            try:
                event_obj = BIDSTaskEvent(onset=onset_rel if onset_rel is not None else time.time(), duration=duration or 0, trial_type='AUTitem')
                bids.addEvent(event_obj)
            except Exception as e:
                logging.warning(f'Failed to add BIDS event: {e}')

        # brief pause
        core.wait(0.2)

        if event.getKeys(keyList=['escape']):
            break

    # write BIDS-style events TSV and minimal dataset_description.json (only when not dry-run/quick-test)
    try:
        if bids_events and not args.dry_run and not args.quick_test:
            bids_out_dir = os.path.join(base_dir, 'data')
            safe_makedirs(bids_out_dir)
            events_tsv = os.path.join(bids_out_dir, f"{participant}_task-{exp_name}_events.tsv")
            # write header and rows
            with open(events_tsv, 'w', encoding='utf-8') as et:
                header = ['onset', 'duration', 'trial_type', 'response_time', 'rating', 'wav_file']
                et.write('\t'.join(header) + '\n')
                for ev in bids_events:
                    row = [str(ev.get(h, '')) for h in header]
                    et.write('\t'.join(row) + '\n')
            logging.info(f'Wrote BIDS events TSV: {events_tsv}')

            # minimal dataset_description.json
            ds_json = os.path.join(bids_out_dir, 'dataset_description.json')
            dd = {
                'Name': 'MR_AUT dataset',
                'BIDSVersion': '1.8.0',
                'GeneratedBy': 'MR_AUT_enhanced.py'
            }
            with open(ds_json, 'w', encoding='utf-8') as dj:
                json.dump(dd, dj, indent=2)
            logging.info(f'Wrote dataset_description.json: {ds_json}')
    except Exception as e:
        logging.warning(f'Failed writing BIDS files: {e}')

    # finish
    bye = visual.TextStim(win, text='Done. Thank you!', color='white', height=0.05)
    bye.draw(); win.flip()
    core.wait(1.0)
    win.close()

    # optionally run transcription on saved wavs (use venv python to call helper script)
    try:
        if not args.dry_run and not args.quick_test:
            # choose python executable from venv if available
            venv_py = os.environ.get('VIRTUAL_ENV') and os.path.join(os.environ.get('VIRTUAL_ENV'), 'bin', 'python')
            if not venv_py or not os.path.exists(venv_py):
                # fallback to sys.executable
                venv_py = sys.executable
            trans_script = os.path.join(ROOT, 'transcribe_wavs.py')
            if os.path.exists(trans_script):
                logging.info('Starting post-run transcription (background)')
                # run as a background process to avoid blocking the UI close
                import subprocess
                subprocess.Popen([venv_py, trans_script, '--wav-dir', wavDirName, '--backend', 'whisper', '--model', 'small', '--device', 'cpu'])
            else:
                logging.info('No transcribe_wavs.py script found; skipping transcription')
    except Exception as e:
        logging.warning(f'Failed to launch transcription: {e}')


if __name__ == '__main__':
    main()
