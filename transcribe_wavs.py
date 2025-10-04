#!/usr/bin/env python3
"""
transcribe_wavs.py

Simple speech-to-text post-processing tool for MR_AUT WAV recordings.

Features:
- Scan a folder for .wav files and transcribe each file.
- Supports two backends: `whisper` (OpenAI Whisper local package) and `vosk` (offline VOSK).
- Writes a `transcripts.tsv` with columns: wav_file, duration, transcript
- Writes per-file .txt transcripts and optional JSON segments (when backend provides them).

Usage examples:
  # whisper (local) - install with: pip install -U openai-whisper
  python transcribe_wavs.py --wav-dir data/sub-01_MR_AUT_2025-10-04_1200_wav --backend whisper --model small

  # vosk (offline) - install with: pip install vosk
  # download a VOSK model and point --vosk-model-dir to it
  python transcribe_wavs.py --wav-dir data/... --backend vosk --vosk-model-dir /path/to/vosk-model

Note: Transcription can be slow on CPU. For Whisper, use smaller models (tiny, base) for speed.
"""
import argparse
import os
import sys
import json
import wave
import tempfile
from pathlib import Path

# optional audio processing imports
try:
    import noisereduce as nr
except Exception:
    nr = None

try:
    from deepmultilingualpunctuation import PunctuationModel
except Exception:
    PunctuationModel = None

# lazy cache for punctuation model
_PUNCT_MODEL = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import numpy as _np
    from scipy import signal
except Exception:
    _np = None
    signal = None



def list_wavs(wav_dir):
    p = Path(wav_dir)
    if not p.exists():
        raise FileNotFoundError(f'wav dir not found: {wav_dir}')
    return sorted([str(x) for x in p.glob('*.wav')])


def wav_duration(path):
    try:
        with wave.open(path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return None


def transcribe_whisper(wav_path, model_name='small', language=None, device='cpu'):
    try:
        import whisper
    except Exception as e:
        raise RuntimeError('Whisper package not installed. pip install -U openai-whisper') from e

    # load model (caching handled by whisper)
    model = whisper.load_model(model_name, device=device)
    opts = {}
    if language:
        opts['language'] = language
        opts['task'] = 'transcribe'
    # use whisper's transcribe API; on CPU set fp16=False via environ if needed
    result = model.transcribe(wav_path, **opts)
    # result typically contains 'text' and 'segments'
    return result.get('text', ''), result.get('segments', None)


def transcribe_vosk(wav_path, vosk_model_dir, language=None):
    try:
        from vosk import Model, KaldiRecognizer
    except Exception as e:
        raise RuntimeError('vosk package not installed. pip install vosk') from e

    if not os.path.isdir(vosk_model_dir):
        raise RuntimeError(f'VOSK model dir not found: {vosk_model_dir}')

    wf = wave.open(wav_path, "rb")
    model = Model(vosk_model_dir)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    results.append(json.loads(rec.FinalResult()))
    # concatenate texts
    text = ' '.join([r.get('text', '') for r in results])
    # build simple segments when available
    segments = []
    for r in results:
        if 'result' in r:
            for w in r['result']:
                segments.append({'start': w.get('start'), 'end': w.get('end'), 'word': w.get('word')})
    return text, segments


def denoise_wav(in_path, out_path=None):
    if nr is None or _np is None or sf is None:
        raise RuntimeError('noisereduce, numpy or soundfile not available')
    data, sr = sf.read(in_path)
    # if stereo, convert to mono
    if data.ndim > 1:
        data = _np.mean(data, axis=1)
    # estimate noise from first 0.5s
    noise_sample = data[:int(0.5 * sr)] if len(data) > sr//2 else data[:int(0.1 * sr)]
    reduced = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)
    out_path = out_path or in_path
    sf.write(out_path, reduced, sr)
    return out_path


def punctuate_text(text):
    global _PUNCT_MODEL
    if PunctuationModel is None:
        # fallback: lightweight heuristic
        t = text.strip()
        if not t:
            return t
        if t[-1] not in '.?!':
            t = t[0].upper() + t[1:] + '.'
        else:
            t = t[0].upper() + t[1:]
        return t
    try:
        if _PUNCT_MODEL is None:
            _PUNCT_MODEL = PunctuationModel()
        return _PUNCT_MODEL.predict(text)
    except Exception as e:
        # graceful fallback: simple capitalization + period
        t = text.strip()
        if not t:
            return t
        if t[-1] not in '.?!':
            t = t[0].upper() + t[1:] + '.'
        else:
            t = t[0].upper() + t[1:]
        return t



def write_outputs(out_dir, wav_path, transcript, segments=None):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    base = Path(wav_path).stem
    txt_fn = p / f"{base}.txt"
    with open(txt_fn, 'w', encoding='utf-8') as f:
        f.write(transcript or '')
    if segments is not None:
        seg_fn = p / f"{base}.segments.json"
        with open(seg_fn, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-dir', required=True, help='Directory with .wav files to transcribe')
    parser.add_argument('--backend', choices=['whisper', 'vosk'], default='whisper')
    parser.add_argument('--model', default='small', help='Whisper model name (tiny, base, small, medium, large)')
    parser.add_argument('--device', default='cpu', help='Device for whisper (cpu or cuda)')
    parser.add_argument('--language', default=None, help='Language code (e.g., en, de)')
    parser.add_argument('--vosk-model-dir', default=None, help='Path to VOSK model directory (if using vosk)')
    parser.add_argument('--out-dir', default=None, help='Directory to write transcripts (default: wav-dir/transcripts)')
    parser.add_argument('--denoise', action='store_true', help='Apply noise reduction before transcription (uses noisereduce)')
    parser.add_argument('--punctuate', action='store_true', help='Apply punctuation restoration after transcription (uses deepmultilingualpunctuation)')
    parser.add_argument('--diarize', action='store_true', help='Attempt speaker diarization (pyannote.io required)')
    args = parser.parse_args()

    wavs = list_wavs(args.wav_dir)
    if not wavs:
        print('No wav files found in', args.wav_dir)
        return

    out_dir = args.out_dir or os.path.join(args.wav_dir, 'transcripts')
    os.makedirs(out_dir, exist_ok=True)

    tsv_path = os.path.join(out_dir, 'transcripts.tsv')
    with open(tsv_path, 'w', encoding='utf-8') as tsv:
        header = ['wav_file', 'duration_s', 'transcript']
        tsv.write('\t'.join(header) + '\n')
        for wav in wavs:
            print('Transcribing:', wav)
            dur = wav_duration(wav) or ''
            try:
                target_wav = wav
                if args.denoise:
                    try:
                        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        tmp.close()
                        target_wav = denoise_wav(wav, out_path=tmp.name)
                    except Exception as e:
                        print('Denoise failed, continuing with original wav:', e)

                if args.backend == 'whisper':
                    transcript, segments = transcribe_whisper(target_wav, model_name=args.model, language=args.language, device=args.device)
                else:
                    transcript, segments = transcribe_vosk(target_wav, vosk_model_dir=args.vosk_model_dir, language=args.language)

                if args.punctuate:
                    try:
                        transcript = punctuate_text(transcript)
                    except Exception as e:
                        print('Punctuation failed:', e)

                # diarization placeholder: warn if requested
                if args.diarize:
                    print('Diarization requested but not implemented automatically. See pyannote.audio docs for setup.')

            except Exception as e:
                print('Failed to transcribe', wav, e)
                transcript = ''
                segments = None

            # write per-file outputs
            write_outputs(out_dir, wav, transcript, segments)

            # write TSV row (tab-escape transcript)
            safe_trans = transcript.replace('\t', ' ').replace('\n', ' ')
            tsv.write(f"{os.path.basename(wav)}\t{dur}\t{safe_trans}\n")

    print('Transcription complete. Outputs in', out_dir)


if __name__ == '__main__':
    main()
