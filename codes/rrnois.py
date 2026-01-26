import time
import numpy as np
import os
import sys
from scipy.io import wavfile
from pyrnnoise import RNNoise

# --- Configuration ---
# RNNoise expects 48000 Hz and 16-bit integer data.
SAMPLE_RATE = 48000 
DATA_TYPE = np.int16

# --- Input/Output Paths ---
INPUT_FILE = "output.wav" 
OUTPUT_FILE = "output_denoised.wav"

if not os.path.exists(INPUT_FILE):
    print(f"Error: Input file '{INPUT_FILE}' not found.")
    sys.exit(1)

# --- 1. Load, Convert, and Prepare Audio Data ---
try:
    # Load the file, retaining the actual sample rate and data
    actual_sample_rate, audio_data = wavfile.read(INPUT_FILE)
except Exception as e:
    print(f"Error reading WAV file: {e}")
    sys.exit(1)


# --- CRITICAL SAFETY CHECKS ---
if actual_sample_rate != SAMPLE_RATE:
    print(f"\n--- FATAL ERROR ---")
    print(f"Input file is {actual_sample_rate} Hz. RNNoise requires {SAMPLE_RATE} Hz.")
    print("Please manually resample your 'input.wav' file to 48000 Hz using an external tool (like Audacity or FFmpeg) and retry.")
    sys.exit(1)

# Ensure data is int16 (crucial for RNNoise stability)
if audio_data.dtype.kind == 'f':
    # Convert float data (assumed to be [-1, 1]) to int16 range
    audio_data = (audio_data * 32767).astype(DATA_TYPE)
else:
    audio_data = audio_data.astype(DATA_TYPE)

# Ensure MONO data: Reshape from (samples,) to (1, samples)
if len(audio_data.shape) == 1:
    audio_data = np.expand_dims(audio_data, axis=0)
elif audio_data.shape[0] > audio_data.shape[1]:
    # Handles (samples, channels) format (common for stereo WAVs)
    audio_data = audio_data.T
    if audio_data.shape[0] > 1:
        # If it's stereo, convert to mono by taking one channel (simple averaging is better but complex)
        audio_data = audio_data[0:1, :]
else:
    # Assumes (channels, samples) which is correct, but check channels
    if audio_data.shape[0] > 1:
        audio_data = audio_data[0:1, :]


# --- 2. Process Audio Chunk by Chunk & Measure Performance ---
print("\n--- Starting RNNoise Latency Test ---")
print(f"Processing shape: {audio_data.shape} (Must be [1, N])")
total_frames = 0
denoised_frames = []

start_time = time.time()
denoiser = RNNoise(sample_rate=SAMPLE_RATE)

try:
    for speech_prob, denoised_frame in denoiser.denoise_chunk(audio_data):
        denoised_frames.append(denoised_frame)
        total_frames += 1
except Exception as e:
    print(f"\n--- FATAL ERROR ---")
    print(f"Error during RNNoise processing: {e}")
    print("Diagnosis: The input buffer or sample size is likely causing a low-level memory allocation failure.")
    print("Action: Double-check your input file for corruption or unusual sample blocks.")
    sys.exit(1)

end_time = time.time()

# --- 3. Calculation and Reporting ---
total_runtime = end_time - start_time
total_audio_samples = audio_data.shape[-1]
total_audio_time = total_audio_samples / SAMPLE_RATE 

# Reconstruct and Save Denoised Audio
denoised_audio = np.concatenate(denoised_frames, axis=1)
final_output = np.squeeze(denoised_audio)
wavfile.write(OUTPUT_FILE, SAMPLE_RATE, final_output)

# --- Results ---
rtf = total_runtime / total_audio_time
avg_frame_time = (total_runtime / total_frames) * 1000 

print(f"\nTotal audio processed: {total_audio_time:.2f} seconds")
print(f"Total runtime: {total_runtime:.4f} seconds")
print(f"Processed frames: {total_frames}")
print("---------------------------------------")
print(f"✅ Real-Time Factor (RTF): {rtf:.3f} (Target must be < 1.0)")
print(f"✅ Avg Inference Time per frame: {avg_frame_time:.2f} ms")
print("---------------------------------------")
print(f"Denoised audio saved to: {OUTPUT_FILE}")