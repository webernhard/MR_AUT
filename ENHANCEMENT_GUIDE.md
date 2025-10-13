# Training a Speech Enhancement Model for MRI Scanner Noise

## Quick Start: Traditional Enhancement (No Training Needed!)

I've created `enhance_mri_audio.py` which uses traditional DSP techniques perfect for repetitive MRI noise:

```bash
# 1. Analyze your scanner noise (see what frequencies to suppress)
python enhance_mri_audio.py --noise-wav /Users/karl/work/voice2speech_test/test/noise.wav --analyze --output-dir ./analysis

# 2. Enhance a single file
python enhance_mri_audio.py \
  --input /path/to/noisy.wav \
  --noise-wav /Users/karl/work/voice2speech_test/test/noise.wav \
  --output /path/to/clean.wav \
  --method both

# 3. Batch process entire directory
python enhance_mri_audio.py \
  --input-dir /Users/karl/work/voice2speech_test/test \
  --noise-wav /Users/karl/work/voice2speech_test/test/noise.wav \
  --output-dir /Users/karl/work/voice2speech_test/test_enhanced \
  --method both
```

This uses:
- **Adaptive notch filtering**: Identifies dominant scanner frequencies and suppresses them
- **Spectral subtraction**: Removes the stationary noise profile
- **Combined approach**: Both methods together (recommended)

---

## Advanced: Neural Enhancement (Requires Training Data)

For best quality, train a neural network. Here's how:

### Option 1: Use Pretrained Models (Fastest)

Use an existing speech enhancement model and fine-tune on your data:

```bash
# Install dependencies
uv pip install asteroid speechbrain torch torchaudio

# Use Demucs (Facebook's model) - works out of the box
python -m demucs.separate --two-stems=vocals your_noisy_file.wav
```

### Option 2: Train Your Own Model (Best Quality)

#### Step 1: Collect Training Data

You need:
- **Clean speech**: 30+ minutes of speech WITHOUT scanner noise
  - Record same speakers outside scanner, OR
  - Use public German speech datasets (e.g., CommonVoice)
- **Scanner noise**: Your `noise.wav` (already have ✓)

#### Step 2: Generate Synthetic Training Data

```python
# create_training_data.py
import soundfile as sf
import numpy as np
from pathlib import Path

def mix_speech_and_noise(clean_speech_dir, noise_file, output_dir, snr_range=(0, 10)):
    """Mix clean speech with scanner noise at various SNR levels."""
    
    noise, noise_sr = sf.read(noise_file)
    output_dir = Path(output_dir)
    (output_dir / 'noisy').mkdir(parents=True, exist_ok=True)
    (output_dir / 'clean').mkdir(parents=True, exist_ok=True)
    
    for clean_file in Path(clean_speech_dir).glob('*.wav'):
        clean, sr = sf.read(clean_file)
        
        # Resample noise if needed
        if noise_sr != sr:
            from scipy import signal as sig
            noise = sig.resample(noise, int(len(noise) * sr / noise_sr))
        
        # Tile noise to match clean length
        if len(noise) < len(clean):
            repeats = int(np.ceil(len(clean) / len(noise)))
            noise_segment = np.tile(noise, repeats)[:len(clean)]
        else:
            start = np.random.randint(0, len(noise) - len(clean))
            noise_segment = noise[start:start+len(clean)]
        
        # Mix at random SNR
        snr_db = np.random.uniform(*snr_range)
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise_segment ** 2)
        scaling = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
        
        noisy = clean + scaling * noise_segment
        
        # Normalize
        noisy = noisy / np.max(np.abs(noisy)) * 0.95
        
        # Save
        sf.write(output_dir / 'clean' / clean_file.name, clean, sr)
        sf.write(output_dir / 'noisy' / clean_file.name, noisy, sr)
        print(f"Created: {clean_file.name} (SNR: {snr_db:.1f} dB)")
```

#### Step 3: Train Enhancement Model

**Using Asteroid (Easiest):**

```bash
# Install
uv pip install asteroid pytorch-lightning

# Train Conv-TasNet
python -m asteroid.models.conv_tasnet \
  --train_dir ./training_data/noisy \
  --valid_dir ./validation_data/noisy \
  --train_targets ./training_data/clean \
  --valid_targets ./validation_data/clean \
  --epochs 100 \
  --batch_size 4
```

**Using SpeechBrain:**

```python
# train_enhancement.py using SpeechBrain
from speechbrain.processing.speech_augmentation import AddNoise
from speechbrain.lobes.models.MetricGAN import MetricGAN

# Configure and train
# See: https://speechbrain.github.io/tutorials/enhancement/
```

#### Step 4: Use Trained Model

```python
# enhance_with_model.py
import torch
import soundfile as sf
from asteroid.models import ConvTasNet

# Load trained model
model = ConvTasNet.from_pretrained('path/to/checkpoint')

# Enhance audio
noisy, sr = sf.read('noisy.wav')
enhanced = model(torch.from_numpy(noisy).unsqueeze(0)).squeeze().numpy()

sf.write('enhanced.wav', enhanced, sr)
```

---

## Recommended Workflow for Your Case

### Immediate (Today):
1. **Use the `enhance_mri_audio.py` script I created**
   - Analyze your noise profile
   - Enhance a few test files
   - Compare transcription quality (Whisper/VOSK) before/after enhancement

### Short-term (This week):
2. **If traditional methods aren't good enough:**
   - Try pretrained Demucs model (no training needed)
   - Or use Facebook's Wav2Vec2 fine-tuning on noisy examples

### Long-term (If building production system):
3. **Train custom model:**
   - Collect 1-2 hours of clean speech from your subjects
   - Generate synthetic training data (clean + your scanner noise)
   - Train Conv-TasNet or DCCRN model
   - Fine-tune over a weekend

---

## Comparison: Traditional vs Neural

| Method | Pros | Cons | When to Use |
|--------|------|------|-------------|
| **Traditional (enhance_mri_audio.py)** | Fast, no training, interpretable, works now | May leave artifacts, less adaptive | Repetitive noise, quick results |
| **Neural (trained model)** | Best quality, learns complex patterns | Needs training data & GPU, slower | Production system, have data |

---

## Next Steps

Run this now to see immediate improvement:

```bash
# In your venv
cd /Users/karl/work/github/MR_AUT
source .venv/bin/activate

# Analyze scanner noise
python enhance_mri_audio.py \
  --noise-wav /Users/karl/work/voice2speech_test/test/noise.wav \
  --analyze \
  --output-dir ./noise_analysis

# Enhance test files
python enhance_mri_audio.py \
  --input-dir /Users/karl/work/voice2speech_test/test \
  --noise-wav /Users/karl/work/voice2speech_test/test/noise.wav \
  --output-dir /Users/karl/work/voice2speech_test/test_enhanced \
  --method both

# Then transcribe enhanced files
python transcribe_wavs.py \
  --wav-dir /Users/karl/work/voice2speech_test/test_enhanced \
  --backend whisper \
  --model medium \
  --language de \
  --out-dir ./transcripts_enhanced
```

Compare the transcripts before/after enhancement to see improvement!
