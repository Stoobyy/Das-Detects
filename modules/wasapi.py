import soundcard as sc
import numpy as np
from scipy.io.wavfile import write

samplerate = 48000

with sc.get_microphone(id=sc.default_speaker().name, include_loopback=True).recorder(
    samplerate=samplerate
) as mic:

    print("Recording dawg...")
    data = mic.record(numframes=samplerate * 2)
    print("Done.")

write("output.wav", samplerate, data)
print("Saved output.wav")
