#!/usr/bin/env python3
"""
MR_AUT_enhanced.py

Compact AUT paradigm (enhanced & configurable)

Default behaviour (can be overridden by a JSON paradigm spec):
- Welcome/trigger screen (wait for '5')
- For each trial:
    * Fixation with jitter (default 3–7 s)
    * Randomly pick an unused word from stim CSV (without replacement)
    * Display word immediately (no fixed pre-display period)
    * Press key '1' to START recording (word turns green while recording)
    * Press key '1' again to STOP recording
    * Rating phase (slider / creativity)
    * (Optional markers + BIDS logging preserved)

You can define an external spec file (JSON) to adjust timings / keys / selection policy.
Example (save as paradigm_spec.json):
{
    "trigger_key": "5",
    "fixation_jitter": {"min": 3.0, "max": 7.0},
    "item_selection": "random_without_replacement",
    "recording": {
        "start_key": "1",
        "stop_key": "1",
        "item_color_recording": "green",
        "indicator_text": "RECORDING... (press 1 to stop)"
    },
    "pre_record_display": {"mode": "wait_for_start"},
    "rating": {"method": "slider", "label_left": "gar nicht kreativ", "label_right": "sehr kreativ"}
}

Run with:  python MR_AUT_enhanced.py --paradigm paradigm_spec.json --ui --count 3

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
import random

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
    """Return list of item strings from CSV (column 'MR_AUTitem' or first column)."""
    items = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get('MR_AUTitem')
            if val is None:
                # fall back to first value
                try:
                    val = next(iter(row.values()))
                except Exception:
                    continue
            if val:
                items.append(val)
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

def get_rating_slider(win, prompt, labels):
    """Presents a visual analogue scale and waits for a mouse click."""
    header_pos_y = 0.40
    main_pos_y = 0.0
    
    header_txt = visual.TextStim(win=win, text=prompt, pos=(0, header_pos_y), color='white', height=0.045)
    slider = visual.Slider(win=win, startValue=5, size=(1.0, 0.03), pos=(0, main_pos_y), units='height',
                           labels=None, ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], granularity=0,
                           style=['slider'], labelColor='black', markerColor='Red', lineColor='White')
    
    l_label = visual.TextStim(win=win, text=labels[0], pos=(-.5, main_pos_y - .075), color='darkgreen', height=0.03)
    r_label = visual.TextStim(win=win, text=labels[1], pos=(.5, main_pos_y - .075), color='darkgreen', height=0.03)
    
    mouse = event.Mouse(win=win)
    
    while True:
        header_txt.draw()
        slider.draw()
        l_label.draw()
        r_label.draw()
        win.flip()
        
        if mouse.getPressed()[0]:
            # Wait for release to avoid multiple triggers
            while mouse.getPressed()[0]:
                core.wait(0.01)
            return slider.getRating()
        
        if event.getKeys(keyList=['escape']):
            return None
        
        core.wait(0.01)


def draw_timing_bar(win, timing_cfg, elapsed):
    """
    Draws a horizontal timing bar with n_segments, filling as time passes.
    timing_cfg: dict from JSON
    elapsed: seconds elapsed in current event
    """
    n = timing_cfg.get('n_segments', 5)
    seg_dur = timing_cfg.get('segment_duration', 3)
    position = timing_cfg.get('position', 'bottom')
    shape = timing_cfg.get('shape', 'rect')
    size = timing_cfg.get('size', [0.08, 0.01])
    color_empty = timing_cfg.get('color_empty', 'darkgray')
    color_filled = timing_cfg.get('color_filled', 'lightgray')
    filled = int(elapsed // seg_dur)
    # Calculate y position
    y = -0.45 if position == 'bottom' else 0.45
    # Draw bars
    for i in range(n):
        x = -0.2 + i * (size[0] + 0.01)
        bar_color = color_filled if i < filled else color_empty
        if shape == 'rect':
            bar = visual.Rect(win, width=size[0], height=size[1], pos=(x, y), fillColor=bar_color, lineColor=None)
            bar.draw()
        elif shape == 'circle':
            circ = visual.Circle(win, radius=size[0]/2, pos=(x, y), fillColor=bar_color, lineColor=None)
            circ.draw()

def run_paradigm(paradigm_spec, items, win, t_mrk, participant, wavDirName, bids_handler=None):
    """
    Executes the experiment based on the paradigm specification.
    """
    # Experiment-relative zero: start counting onsets from scanner trigger
    exp_start = time.time()
    recording_count = 1
    bids_events = []
    item_iterator = iter(items)
    current_visual_stim = None  # To hold the last displayed visual stimulus
    
    # Determine response device
    settings = paradigm_spec.get('settings', {})
    response_device = settings.get('response_device', 'keyboard')
    mouse = event.Mouse(win=win) if response_device == 'mouse' else None

    # Process each block in the paradigm
    for block_idx, block in enumerate(paradigm_spec.get('paradigm', [])):
        logging.info(f"Starting block {block_idx + 1}: {block.get('block_type', 'untitled')}")

        # Determine number of trials for this block
        n_trials_val = block.get('n_trials', 1)
        if isinstance(n_trials_val, str) and n_trials_val.startswith('from_stimuli:'):
            stim_key = n_trials_val.split(':')[-1]
            n_trials = paradigm_spec.get('stimuli', {}).get(stim_key, {}).get('n_trials', 1)
        else:
            n_trials = int(n_trials_val)

        # Loop through trials
        for trial_num in range(n_trials):
            logging.info(f"  Trial {trial_num + 1}/{n_trials}")
            trial_onset_time = time.time()
            current_item = next(item_iterator, None)
            if current_item is None:
                logging.warning("  No more items left in stimulus list.")
                break
            
            rating = None
            wav_file = None

            # Execute trial sequence
            for event_name in block.get('trial_sequence', []):
                if 'escape' in event.getKeys(keyList=['escape']):
                    logging.warning("Experiment aborted by user during trial.")
                    return bids_events, True # Return collected events and abort flag

                event_def = paradigm_spec.get('events', {}).get(event_name)
                if not event_def:
                    logging.warning(f"    Event '{event_name}' not found in spec.")
                    continue

                logging.info(f"    Executing event: {event_name}")

                # Resolve stimulus
                stimulus_val = event_def.get('stimulus')
                if stimulus_val == 'from_stimuli:items':
                    display_text = current_item
                else:
                    display_text = stimulus_val

                # Resolve duration
                duration = event_def.get('duration')
                # Support both float/int and dict duration
                if isinstance(duration, dict):
                    wait_duration = duration.get('max')
                elif isinstance(duration, (float, int)):
                    wait_duration = duration
                elif duration is not None:
                    try:
                        wait_duration = float(duration)
                    except Exception:
                        wait_duration = None
                else:
                    wait_duration = None

                # --- Event Execution Logic ---
                event_type = event_def.get('type')

                if event_type == 'visual':
                    # Special handling for show_item with timing bar
                    if event_name == 'show_item' and event_def.get('timing_bar', {}).get('enabled', False):
                        timing_cfg = event_def.get('timing_bar', {})
                        duration_cfg = event_def.get('duration', {})
                        max_dur = duration_cfg.get('max', 15)
                        end_on_button = duration_cfg.get('end_on_button', True)
                        start_time = core.getTime()
                        response_device = paradigm_spec.get('settings', {}).get('response_device', 'keyboard')
                        mouse = event.Mouse(win=win) if response_device == 'mouse' else None
                        response_keys = paradigm_spec.get('settings', {}).get('keyboard_keys', ['1'])
                        stim = visual.TextStim(win, text=display_text, color=event_def.get('color', 'yellow'), height=0.07)
                        responded = False
                        while True:
                            elapsed = core.getTime() - start_time
                            stim.draw()
                            draw_timing_bar(win, timing_cfg, elapsed)
                            win.flip()
                            if end_on_button and not responded:
                                if response_device == 'mouse':
                                    mouse_button = paradigm_spec.get('settings', {}).get('mouse_button_indices', [0])[0]
                                    if mouse.getPressed()[mouse_button]:
                                        while mouse.getPressed()[mouse_button]:
                                            core.wait(0.01)
                                        responded = True
                                        break
                                else:
                                    if event.getKeys(keyList=response_keys + ['escape']):
                                        responded = True
                                        break
                            if elapsed >= max_dur:
                                break
                            core.wait(0.01)
                        # After response or timeout, start recording phase with word still visible
                        # Get recording color from next event (record_response)
                        rec_event = paradigm_spec.get('events', {}).get('record_response', {})
                        rec_color = rec_event.get('recording_color', 'green')
                        stim.color = rec_color
                        stim.draw()
                        win.flip()
                        # Start recording
                        audio_frames = []
                        stream = None
                        try:
                            import sounddevice as sd
                            stream = sd.InputStream(samplerate=44100, channels=1,
                                                    callback=lambda indata, frames, time, status: audio_frames.append(indata.copy()))
                            stream.start()
                        except Exception as e:
                            logging.warning(f"Failed to start audio stream: {e}")
                        # Wait for stop response
                        if response_device == 'mouse':
                            mouse_button = paradigm_spec.get('settings', {}).get('mouse_button_indices', [0])[0]
                            while not mouse.getPressed()[mouse_button]:
                                if event.getKeys(keyList=['escape']):
                                    logging.warning("Experiment aborted by user.")
                                    if stream:
                                        stream.stop(); stream.close()
                                    return bids_events, True
                                core.wait(0.01)
                            while mouse.getPressed()[mouse_button]:
                                core.wait(0.01)
                        else:
                            while True:
                                keys = event.getKeys(keyList=response_keys + ['escape'])
                                if 'escape' in keys:
                                    logging.warning("Experiment aborted by user.")
                                    if stream:
                                        stream.stop(); stream.close()
                                    return bids_events, True
                                if response_keys[0] in keys:
                                    break
                                core.wait(0.01)
                        # Stop stream
                        if stream:
                            stream.stop(); stream.close()
                        # Save audio
                        if audio_frames:
                            wav_file_path = os.path.join(wavDirName, f"{participant}_{sanitize_filename(current_item)}_idea{recording_count:02d}.wav")
                            if save_wav(audio_frames, 44100, wav_file_path):
                                logging.info(f'Saved: {wav_file_path}')
                                wav_file = os.path.basename(wav_file_path)
                                recording_count += 1
                        else:
                            logging.info('No audio collected for this item')
                        # After recording, show rating scale before continuing
                        rate_event = paradigm_spec.get('events', {}).get('rate_creativity', {})
                        if rate_event.get('method') == 'slider':
                            rating = get_rating_slider(win, 
                                                       prompt=rate_event.get('prompt', 'Rate creativity'),
                                                       labels=rate_event.get('labels', ['low', 'high']))
                        else:
                            rating = get_rating(win, prompt=rate_event.get('prompt', 'Rate creativity 0-9'))
                        logging.info(f'Rating for {current_item}: {rating}')
                        # After rating, continue to next event
                    elif event_name == 'fixation' and isinstance(event_def.get('duration'), dict):
                        # Fixation with jitter
                        fj = event_def.get('duration', {})
                        fj_min = float(fj.get('min', 0.8))
                        fj_max = float(fj.get('max', 1.2))
                        rand_dur = random.uniform(fj_min, fj_max)
                        stim = visual.TextStim(win, text=display_text, color=event_def.get('color', 'white'), height=0.15)
                        stim.draw(); win.flip()
                        core.wait(rand_dur)
                    else:
                        stim = visual.TextStim(win, text=display_text, color=event_def.get('color', 'white'), height=0.07)
                        stim.draw()
                        win.flip()
                        current_visual_stim = stim  # Store the visual stimulus
                        if wait_duration:
                            core.wait(wait_duration)

                elif event_type == 'audio_recording':
                    # Resolve keys from spec
                    if response_device == 'keyboard':
                        response_keys = settings.get('keyboard_keys', ['1'])
                        start_key = response_keys[0]
                        stop_key = response_keys[0]
                    else: # mouse
                        start_key = 'mouse'
                        stop_key = 'mouse'
                        mouse_buttons = settings.get('mouse_button_indices', [0])


                    # Resolve color
                    rec_color = event_def.get('recording_color', 'green')

                    # Wait for start response
                    if current_visual_stim:
                        current_visual_stim.draw()
                    win.flip()
                    
                    if response_device == 'keyboard':
                        keys = event.waitKeys(keyList=[start_key, 'escape'])
                        if 'escape' in keys:
                            logging.warning("Experiment aborted by user.")
                            return bids_events, True
                    else: # mouse
                        while not mouse.getPressed()[mouse_buttons[0]]:
                            if event.getKeys(keyList=['escape']):
                                logging.warning("Experiment aborted by user.")
                                return bids_events, True
                            core.wait(0.01)
                        # Wait for release
                        while mouse.getPressed()[mouse_buttons[0]]:
                            core.wait(0.01)


                    # Start recording and update visuals
                    audio_frames = []
                    stream = None
                    try:
                        stream = sd.InputStream(samplerate=44100, channels=1,
                                                callback=lambda indata, frames, time, status: audio_frames.append(indata.copy()))
                        stream.start()
                        
                        # Change color of the item to indicate recording
                        if current_visual_stim:
                            current_visual_stim.color = rec_color
                            current_visual_stim.draw()
                        
                        win.flip()

                    except Exception as e:
                        logging.warning(f"Failed to start audio stream: {e}")

                    # Wait for stop response
                    if response_device == 'keyboard':
                        keys = event.waitKeys(keyList=[stop_key, 'escape'])
                        if 'escape' in keys:
                            logging.warning("Experiment aborted by user.")
                            if stream:
                                stream.stop()
                                stream.close()
                            return bids_events, True
                    else: # mouse
                        while not mouse.getPressed()[mouse_buttons[0]]:
                            if event.getKeys(keyList=['escape']):
                                logging.warning("Experiment aborted by user.")
                                if stream:
                                    stream.stop()
                                    stream.close()
                                return bids_events, True
                            core.wait(0.01)
                        # Wait for release
                        while mouse.getPressed()[mouse_buttons[0]]:
                            core.wait(0.01)
                    
                    # Stop stream
                    if stream:
                        stream.stop()
                        stream.close()

                    # Save audio
                    if audio_frames:
                        wav_file_path = os.path.join(wavDirName, f"{participant}_{sanitize_filename(current_item)}_idea{recording_count:02d}.wav")
                        if save_wav(audio_frames, 44100, wav_file_path):
                            logging.info(f'Saved: {wav_file_path}')
                            wav_file = os.path.basename(wav_file_path)
                            recording_count += 1
                    else:
                        logging.info('No audio collected for this item')

                elif event_type == 'rating_scale':
                    if event_def.get('method') == 'slider':
                        rating = get_rating_slider(win, 
                                                   prompt=event_def.get('prompt', 'Rate creativity'),
                                                   labels=event_def.get('labels', ['low', 'high']))
                    else: # fallback to keypress
                        rating = get_rating(win, prompt=event_def.get('prompt', 'Rate creativity 0-9'))
                    
                    logging.info(f'Rating for {current_item}: {rating}')
                elif event_type == 'visual' and event_name == 'show_item':
                    # Timing bar logic
                    timing_cfg = event_def.get('timing_bar', {})
                    duration_cfg = event_def.get('duration', {})
                    max_dur = duration_cfg.get('max', 15)
                    end_on_button = duration_cfg.get('end_on_button', True)
                    start_time = core.getTime()
                    response_device = paradigm_spec.get('settings', {}).get('response_device', 'keyboard')
                    mouse = event.Mouse(win=win) if response_device == 'mouse' else None
                    response_keys = paradigm_spec.get('settings', {}).get('keyboard_keys', ['1'])
                    stim = visual.TextStim(win, text=display_text, color=event_def.get('color', 'yellow'), height=0.07)
                    responded = False
                    while True:
                        elapsed = core.getTime() - start_time
                        stim.draw()
                        if timing_cfg.get('enabled', False):
                            draw_timing_bar(win, timing_cfg, elapsed)
                        win.flip()
                        if end_on_button and not responded:
                            if response_device == 'mouse':
                                if mouse.getPressed()[0]:
                                    while mouse.getPressed()[0]:
                                        core.wait(0.01)
                                    responded = True
                                    break
                            else:
                                if event.getKeys(keyList=response_keys + ['escape']):
                                    responded = True
                                    break
                        if elapsed >= max_dur:
                            break
                        core.wait(0.01)
                    # After response or timeout, continue to next event

            # --- End of Trial ---
            # BIDS event logging
            try:
                trial_duration = time.time() - trial_onset_time
                onset_rel = float(trial_onset_time - exp_start)
                bids_events.append({
                    'onset': onset_rel,
                    'duration': trial_duration,
                    'trial_type': 'AUT_item',
                    'stimulus': current_item,
                    'rating': rating,
                    'wav_file': wav_file if wav_file else 'n/a'
                })
                if bids_handler:
                    bids_handler.addEvent(
                        onset=onset_rel,
                        duration=trial_duration,
                        trial_type='AUT_item',
                        stimulus=current_item,
                        rating=rating
                    )
            except Exception as e:
                logging.warning(f"Failed to log BIDS event: {e}")

            if event.getKeys(keyList=['escape']):
                logging.warning("Experiment aborted by user.")
                return bids_events, True # Return collected events so far

    return bids_events, False

def main():
    # parse CLI args early so quick-test can bypass macOS guard
    import argparse
    import random
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--quick-test', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--duration', type=float, default=2.0)
    parser.add_argument('--force-run', action='store_true')
    parser.add_argument('--count', type=int, default=5, help='Number of items for quick-test')
    parser.add_argument('--ui', action='store_true', help='Run UI test (opens PsychoPy window)')
    parser.add_argument('--paradigm', type=str, default=None, help='Path to paradigm spec JSON file')
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

    # Load paradigm specification (JSON)
    default_spec = {
        "settings": {
            "response_device": "keyboard",
            "keyboard_keys": ["1"],
            "mouse_button_indices": [0],
            "bids_logging": False
        },
        "stimuli": {
            "items": {
                "file": "stim/MR_AUT_items.csv",
                "sampling": "random",
                "n_trials": 5
            }
        },
        "events": {
            "fixation": {
                "type": "visual",
                "stimulus": "+",
                "duration": {"min": 0.8, "max": 1.2},
                "color": "white"
            },
            "show_item": {
                "type": "visual",
                "stimulus": "from_stimuli:items",
                "duration": 5.0,
                "color": "yellow"
            },
            "record_response": {
                "type": "audio_recording",
                "duration": "button_press"
            },
            "rate_creativity": {
                "type": "rating_scale",
                "method": "slider",
                "duration": "button_press",
                "prompt": "Wie kreativ war die Idee?",
                "labels": ["gar nicht kreativ", "sehr kreativ"],
                "color": "green"
            }
        },
        "paradigm": [
            {
                "block_type": "task",
                "n_trials": "from_stimuli:items",
                "trial_sequence": [
                    "fixation",
                    "show_item",
                    "record_response",
                    "rate_creativity"
                ]
            }
        ]
    }
    paradigm_spec = default_spec
    if args.paradigm and os.path.isfile(args.paradigm):
        try:
            with open(args.paradigm, 'r', encoding='utf-8') as pj:
                paradigm_spec = json.load(pj)
            logging.info(f'Loaded paradigm spec from {args.paradigm}')
        except Exception as e:
            logging.warning(f'Failed loading paradigm spec {args.paradigm}: {e}')

    # Load items based on spec
    items_spec = paradigm_spec.get('stimuli', {}).get('items', {})
    items_file = items_spec.get('file', CSV_PATH)
    items = read_items(items_file)

    # Handle item sampling
    if items_spec.get('sampling') == 'random':
        random.shuffle(items)

    num_trials = items_spec.get('n_trials', len(items))
    items = items[:num_trials]

    # if running UI test, limit items to requested count after shuffle
    if args.ui:
        count = args.count if args.count > 0 else 1
        items = items[:count]
        print(f'UI test mode: running {len(items)} item(s)')

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

    trigger_key = paradigm_spec.get("settings", {}).get("trigger_key", "5")
    # wait for scanner trigger / welcome screen
    instr = visual.TextStim(win, text=f"Waiting for trigger ('{trigger_key}')...", color='white', height=0.04)
    instr.draw(); win.flip()
    event.clearEvents()
    event.waitKeys(keyList=[trigger_key, 'escape'])

    # prepare wav directory like original
    base_dir = globals().get('_thisDir', ROOT)
    exp_name = globals().get('expName', 'MR_AUT')
    wavDirName = os.path.join(base_dir, 'data', f"{participant}_{exp_name}_{datetime.now().strftime('%Y-%m-%d_%H%M')}_wav")
    safe_makedirs(wavDirName)

    # Run the paradigm
    bids_events, aborted = run_paradigm(paradigm_spec, items, win, t_mrk, participant, wavDirName, bids_handler=bids)
    
    if aborted:
        logging.warning("Run was aborted. Writing partial BIDS data.")

    # write BIDS-style events TSV and minimal dataset_description.json (only when not dry-run/quick-test)
    try:
        if bids_events and not args.dry_run and not args.quick_test:
            bids_out_dir = os.path.join(base_dir, 'data')
            safe_makedirs(bids_out_dir)
            events_tsv = os.path.join(bids_out_dir, f"{participant}_task-{exp_name}_events.tsv")
            # write header and rows
            with open(events_tsv, 'w', encoding='utf-8') as et:
                header = list(bids_events[0].keys()) if bids_events else []
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
