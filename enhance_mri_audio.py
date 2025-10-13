#!/usr/bin/env python3
"""
enhance_mri_audio.py

Advanced speech enhancement for MRI scanner recordings using multiple techniques:
1. Spectral analysis of repetitive noise
2. Adaptive notch filtering for periodic noise
3. Optional neural enhancement (if model trained)

For perfectly repetitive MRI noise, uses frequency-domain techniques to identify
and suppress scanner harmonics while preserving speech.

Usage:
    # Analyze noise profile
    python enhance_mri_audio.py --noise-wav noise.wav --analyze

    # Enhance with adaptive filtering
    python enhance_mri_audio.py --input noisy.wav --noise-wav noise.wav --output clean.wav --method adaptive

    # Batch process a directory
    python enhance_mri_audio.py --input-dir /path/to/wavs --noise-wav noise.wav --output-dir /path/to/enhanced --method adaptive
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal
from scipy.fft import rfft, rfftfreq, irfft
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def analyze_noise_profile(noise_path, output_dir=None):
    """Analyze noise file to identify dominant frequencies (scanner harmonics)."""
    print(f"Analyzing noise profile: {noise_path}")
    
    data, sr = sf.read(noise_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    # Compute FFT
    fft = rfft(data)
    freqs = rfftfreq(len(data), 1/sr)
    magnitudes = np.abs(fft)
    
    # Find dominant peaks (scanner harmonics)
    # Use prominence to find significant peaks above noise floor
    peaks, properties = signal.find_peaks(magnitudes, prominence=np.max(magnitudes)*0.01)
    peak_freqs = freqs[peaks]
    peak_mags = magnitudes[peaks]
    
    # Sort by magnitude
    sorted_idx = np.argsort(peak_mags)[::-1]
    top_peaks = sorted_idx[:20]  # Top 20 harmonics
    
    print("\n=== Top 20 Dominant Frequencies (Scanner Harmonics) ===")
    for i, idx in enumerate(top_peaks, 1):
        freq = peak_freqs[idx]
        mag = peak_mags[idx]
        print(f"{i:2d}. {freq:8.2f} Hz  (magnitude: {mag:.2e})")
    
    # Check if fundamental frequency is detectable (repetition rate)
    fundamental_candidates = peak_freqs[top_peaks][:5]
    print(f"\nLikely fundamental frequency (scanner repetition): {fundamental_candidates[0]:.2f} Hz")
    
    # Plot spectrum
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(freqs[:len(freqs)//10], magnitudes[:len(magnitudes)//10])  # Low freq detail
        plt.scatter(peak_freqs[top_peaks], peak_mags[top_peaks], c='red', marker='x', s=100, label='Detected peaks')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Noise Spectrum (Low Frequencies)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(freqs, magnitudes)
        plt.scatter(peak_freqs[top_peaks], peak_mags[top_peaks], c='red', marker='x', s=50)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Full Noise Spectrum')
        plt.grid(True)
        plt.tight_layout()
        
        plot_path = output_dir / 'noise_spectrum.png'
        plt.savefig(plot_path, dpi=150)
        print(f"\nSpectrum plot saved to: {plot_path}")
    
    return {
        'sample_rate': sr,
        'peak_frequencies': peak_freqs[top_peaks],
        'peak_magnitudes': peak_mags[top_peaks],
        'fundamental': fundamental_candidates[0] if len(fundamental_candidates) > 0 else None
    }


def adaptive_notch_filter(data, sr, noise_freqs, Q=30):
    """Apply adaptive notch filters at detected noise frequencies.
    
    Args:
        data: Audio signal
        sr: Sample rate
        noise_freqs: Array of frequencies to suppress (Hz)
        Q: Quality factor (higher = narrower notch)
    """
    filtered = data.copy()
    
    for freq in noise_freqs:
        if freq > 0 and freq < sr/2:  # Valid frequency range
            # Design notch filter
            b, a = signal.iirnotch(freq, Q, sr)
            filtered = signal.filtfilt(b, a, filtered)
    
    return filtered


def spectral_subtraction_stationary(data, noise_profile, sr, alpha=2.0):
    """Enhanced spectral subtraction for stationary noise.
    
    Args:
        data: Noisy audio
        noise_profile: Noise-only audio
        sr: Sample rate
        alpha: Over-subtraction factor (higher = more aggressive)
    """
    # Ensure same length by padding/truncating noise profile
    if len(noise_profile) < len(data):
        # Tile noise profile to match data length
        repeats = int(np.ceil(len(data) / len(noise_profile)))
        noise_profile = np.tile(noise_profile, repeats)[:len(data)]
    else:
        noise_profile = noise_profile[:len(data)]
    
    # Compute power spectra
    data_fft = rfft(data)
    noise_fft = rfft(noise_profile)
    
    data_power = np.abs(data_fft) ** 2
    noise_power = np.abs(noise_fft) ** 2
    
    # Spectral subtraction with over-subtraction
    enhanced_power = np.maximum(data_power - alpha * noise_power, 0.01 * data_power)
    
    # Reconstruct phase from noisy signal
    phase = np.angle(data_fft)
    enhanced_fft = np.sqrt(enhanced_power) * np.exp(1j * phase)
    
    # Inverse FFT
    enhanced = irfft(enhanced_fft, n=len(data))
    
    return enhanced


def enhance_audio(input_path, noise_wav, output_path, method='adaptive', noise_profile=None):
    """Enhance a single audio file.
    
    Args:
        input_path: Path to noisy WAV
        noise_wav: Path to noise-only WAV (or None if noise_profile provided)
        output_path: Path to save enhanced WAV
        method: 'adaptive' (notch filter) or 'spectral' (spectral subtraction) or 'both'
        noise_profile: Pre-analyzed noise profile dict (optional, to avoid re-analysis)
    """
    print(f"Enhancing: {input_path}")
    
    # Load noisy audio
    data, sr = sf.read(input_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    # Load noise profile if needed
    if noise_wav and not noise_profile:
        noise_data, noise_sr = sf.read(noise_wav)
        if noise_data.ndim > 1:
            noise_data = np.mean(noise_data, axis=1)
        if noise_sr != sr:
            # Resample
            num = int(len(noise_data) * float(sr) / noise_sr)
            noise_data = signal.resample(noise_data, num)
        
        # Analyze noise
        noise_profile = analyze_noise_profile(noise_wav)
    
    enhanced = data.copy()
    
    if method in ['adaptive', 'both']:
        if noise_profile and 'peak_frequencies' in noise_profile:
            print("  Applying adaptive notch filtering...")
            enhanced = adaptive_notch_filter(enhanced, sr, noise_profile['peak_frequencies'], Q=30)
    
    if method in ['spectral', 'both']:
        if noise_wav:
            print("  Applying spectral subtraction...")
            noise_data, _ = sf.read(noise_wav)
            if noise_data.ndim > 1:
                noise_data = np.mean(noise_data, axis=1)
            enhanced = spectral_subtraction_stationary(enhanced, noise_data, sr, alpha=2.0)
    
    # Normalize to avoid clipping
    enhanced = enhanced / np.max(np.abs(enhanced)) * 0.95
    
    # Save
    sf.write(output_path, enhanced, sr)
    print(f"  Saved: {output_path}")
    
    return enhanced


def main():
    parser = argparse.ArgumentParser(description='MRI audio enhancement')
    parser.add_argument('--input', help='Input WAV file')
    parser.add_argument('--input-dir', help='Input directory (batch mode)')
    parser.add_argument('--noise-wav', required=True, help='Noise-only WAV file')
    parser.add_argument('--output', help='Output WAV file (single file mode)')
    parser.add_argument('--output-dir', help='Output directory (batch mode)')
    parser.add_argument('--method', choices=['adaptive', 'spectral', 'both'], default='both',
                        help='Enhancement method')
    parser.add_argument('--analyze', action='store_true', help='Only analyze noise profile (no enhancement)')
    
    args = parser.parse_args()
    
    # Analyze mode
    if args.analyze:
        analyze_noise_profile(args.noise_wav, output_dir=args.output_dir or './analysis')
        return
    
    # Single file mode
    if args.input and args.output:
        enhance_audio(args.input, args.noise_wav, args.output, method=args.method)
        return
    
    # Batch mode
    if args.input_dir and args.output_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-analyze noise once
        print("Pre-analyzing noise profile...")
        noise_profile = analyze_noise_profile(args.noise_wav, output_dir=output_dir)
        
        # Process all WAVs
        wav_files = sorted(input_dir.glob('*.wav'))
        for i, wav_file in enumerate(wav_files, 1):
            if wav_file.name == Path(args.noise_wav).name:
                continue  # Skip noise file itself
            
            output_path = output_dir / wav_file.name
            print(f"\n[{i}/{len(wav_files)}]", end=" ")
            enhance_audio(str(wav_file), args.noise_wav, str(output_path), 
                         method=args.method, noise_profile=noise_profile)
        
        print(f"\nBatch enhancement complete. Output: {output_dir}")
        return
    
    parser.error("Specify either --input/--output or --input-dir/--output-dir")


if __name__ == '__main__':
    main()
