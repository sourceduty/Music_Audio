import numpy as np
import pyaudio
import scipy.fftpack
import time

# Constants for tuning
TUNING_FREQUENCIES = {
    "E4": 329.63,  # Standard tuning for guitar strings
    "A4": 440.00,
    "D4": 293.66,
    "G4": 392.00,
    "B4": 493.88,
    "E5": 659.26,
}

# Note names corresponding to frequencies for feedback
NOTE_NAMES = {value: key for key, value in TUNING_FREQUENCIES.items()}

# Audio stream parameters
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Frequency detection function using FFT
def detect_frequency(data):
    # Perform Fast Fourier Transform on audio data
    fourier_transform = np.abs(scipy.fftpack.fft(data))
    frequencies = np.fft.fftfreq(len(fourier_transform), 1 / SAMPLE_RATE)
    
    # Find the peak frequency
    peak_freq = np.abs(frequencies[np.argmax(fourier_transform)])
    return peak_freq

# Function to get the closest note
def get_closest_note(frequency):
    closest_note = min(TUNING_FREQUENCIES.values(), key=lambda x: abs(x - frequency))
    return NOTE_NAMES[closest_note], closest_note

# Function to determine tuning status
def get_tuning_status(target_frequency, actual_frequency):
    if abs(actual_frequency - target_frequency) < 1:
        return "In Tune!"
    elif actual_frequency < target_frequency:
        return "Too Flat!"
    else:
        return "Too Sharp!"

# Function to tune the instrument
def tune_instrument():
    # Set up the audio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=BUFFER_SIZE)

    print("Starting the tuner... Please play a string.")
    try:
        while True:
            # Read audio data from stream
            audio_data = np.frombuffer(stream.read(BUFFER_SIZE), dtype=np.int16)
            frequency = detect_frequency(audio_data)

            # Find the closest note
            note, target_frequency = get_closest_note(frequency)

            # Get tuning feedback
            status = get_tuning_status(target_frequency, frequency)

            # Print the results
            print(f"Detected Frequency: {frequency:.2f} Hz -> {note}")
            print(f"Tuning Status: {status}")
            time.sleep(1)  # Delay to avoid flooding terminal
    except KeyboardInterrupt:
        print("\nTuner stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# Run the tuner
if __name__ == "__main__":
    tune_instrument()
