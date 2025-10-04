#!/usr/bin/env python
"""
Hardware Setup and Check Script for MR_AUT PsychoPy Experiment

This script sets up and verifies the necessary hardware preferences for PsychoPy
to work correctly on different machines, especially macOS.

Run this script before running the experiment to ensure proper hardware configuration
and functionality.
"""

import sys
import os

# Add the PsychoPy venv to path if running from outside
# Assuming the script is run from the project directory
psychopy_path = os.path.expanduser("~/psychopy_test/PsychoPy-2025.1.1-Python3.10/.venv/lib/python3.10/site-packages")
if psychopy_path not in sys.path:
    sys.path.insert(0, psychopy_path)

def check_audio():
    """Check audio backends and devices."""
    print("Checking audio backends...")
    try:
        import sounddevice as sd
        print("✓ sounddevice available")
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        print(f"  Found {len(input_devices)} input devices, {len(output_devices)} output devices")
        if input_devices:
            print(f"  Default input: {sd.query_devices(kind='input')['name']}")
        if output_devices:
            print(f"  Default output: {sd.query_devices(kind='output')['name']}")
        return True
    except ImportError:
        print("✗ sounddevice not available")
        return False

def check_psychtoolbox():
    """Check PsychToolbox availability."""
    print("Checking PsychToolbox...")
    try:
        import psychtoolbox as ptb
        print("✓ psychtoolbox available")
        return True
    except ImportError as e:
        print(f"✗ psychtoolbox not available: {e}")
        return False

def check_keyboard():
    """Check keyboard backend."""
    print("Checking keyboard...")
    try:
        from psychopy.hardware import keyboard
        kb = keyboard.Keyboard()
        print("✓ keyboard initialized successfully")
        return True
    except Exception as e:
        print(f"✗ keyboard initialization failed: {e}")
        return False

def check_mouse():
    """Check mouse backend."""
    print("Checking mouse...")
    try:
        from psychopy.hardware import mouse
        m = mouse.Mouse()
        print("✓ mouse initialized successfully")
        return True
    except Exception as e:
        print(f"✗ mouse initialization failed: {e}")
        return False

def setup_preferences():
    """Set hardware preferences."""
    print("\nSetting hardware preferences...")
    try:
        import psychopy
        from psychopy import prefs

        # Check what's available and set accordingly
        audio_backends = []
        if check_audio():
            audio_backends.append('sounddevice')
        if check_psychtoolbox():
            audio_backends.append('ptb')

        if not audio_backends:
            print("⚠ No audio backends available!")
        else:
            prefs.hardware['audioLib'] = audio_backends
            prefs.hardware['microphone'] = audio_backends[0] if audio_backends else 'default'
            prefs.hardware['speaker'] = audio_backends[0] if audio_backends else 'default'

        # Set keyboard and mouse to event backend (more reliable)
        prefs.hardware['keyboardBackend'] = 'event'
        prefs.hardware['mouseBackend'] = 'event'

        # Save the preferences
        prefs.saveUserPrefs()

        print("Hardware preferences set successfully:")
        print(f"  Audio Library: {prefs.hardware['audioLib']}")
        print(f"  Keyboard Backend: {prefs.hardware['keyboardBackend']}")
        print(f"  Microphone Backend: {prefs.hardware['microphone']}")
        print(f"  Speaker Backend: {prefs.hardware['speaker']}")
        print(f"  Mouse Backend: {prefs.hardware['mouseBackend']}")
        print("\nPreferences saved to user config.")

        return True
    except Exception as e:
        print(f"Error setting preferences: {e}")
        return False

def main():
    """Main setup and check function."""
    print("MR_AUT Hardware Setup and Check")
    print("=" * 40)

    # Check hardware availability
    print("\n1. Checking Hardware Availability:")
    audio_ok = check_audio()
    ptb_ok = check_psychtoolbox()
    keyboard_ok = check_keyboard()
    mouse_ok = check_mouse()

    # Setup preferences based on what's available
    print("\n2. Configuring Preferences:")
    prefs_ok = setup_preferences()

    # Summary
    print("\n3. Summary:")
    print(f"Audio backends available: {audio_ok or ptb_ok}")
    print(f"Keyboard functional: {keyboard_ok}")
    print(f"Mouse functional: {mouse_ok}")
    print(f"Preferences set: {prefs_ok}")

    if audio_ok and keyboard_ok and mouse_ok and prefs_ok:
        print("\n✓ Hardware setup completed successfully!")
        print("You can now run the MR_AUT experiment.")
        return 0
    else:
        print("\n⚠ Some hardware checks failed. The experiment may not work properly.")
        print("Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())