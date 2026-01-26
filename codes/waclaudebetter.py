import os
import random
import subprocess
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
import tempfile

RAW_DIRS = {
    "human": Path("dataset/human_raw"),
    "ai": Path("dataset/ai_raw")
}

OUT_DIRS = {
    "human": Path("dataset/human_processed"),
    "ai": Path("dataset/ai_processed")
}

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: {result.stderr}")
    return result.returncode == 0

def simulate_packet_loss(audio_data, sr, loss_rate=0.03):
    """Simulate packet loss with bursty loss patterns (more realistic)"""
    packet_size_ms = 20  # Typical VoIP packet size
    packet_samples = int(sr * packet_size_ms / 1000)
    
    num_packets = len(audio_data) // packet_samples
    result = audio_data.copy()
    
    i = 0
    while i < num_packets:
        if random.random() < loss_rate:
            # Bursty loss: lose 1-3 consecutive packets
            burst_length = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
            
            for j in range(burst_length):
                if i + j >= num_packets:
                    break
                    
                pkt_idx = i + j
                start = pkt_idx * packet_samples
                end = start + packet_samples
                
                # Packet Loss Concealment (PLC) with interpolation
                if start > 0 and end < len(result):
                    result[start:end] = np.linspace(
                        result[start-1], 
                        result[min(end, len(result)-1)], 
                        min(packet_samples, len(result)-start)
                    )
            
            i += burst_length
        else:
            i += 1
    
    return result

def add_jitter(audio_data, sr, max_jitter_ms=30):
    """Simulate network jitter with more realistic buffer behavior"""
    chunk_size = int(sr * 0.02)  # 20ms chunks
    num_chunks = len(audio_data) // chunk_size
    
    result = []
    buffer_delay = 0
    
    for i in range(num_chunks):
        chunk = audio_data[i*chunk_size:(i+1)*chunk_size]
        
        # Simulate jitter buffer - occasionally add/remove samples
        if random.random() < 0.15:  # 15% of chunks affected
            jitter = random.randint(-10, 10)
            
            if jitter > 0:
                # Delay (stretch)
                chunk = np.pad(chunk, (jitter, 0), mode='edge')[:-jitter]
            elif jitter < 0:
                # Speed up (compress)
                chunk = chunk[-jitter:]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='edge')
        
        result.append(chunk)
    
    # Add remaining samples
    remainder = audio_data[num_chunks*chunk_size:]
    if len(remainder) > 0:
        result.append(remainder)
    
    return np.concatenate(result)

def add_codec_artifacts(audio_data, sr):
    """Add subtle artifacts from codec quantization"""
    # Add very subtle quantization noise
    quantization_noise = np.random.uniform(-0.0005, 0.0005, len(audio_data))
    audio_data = audio_data + quantization_noise
    
    # Simulate slight spectral coloring from codec
    # Add very subtle high-frequency rolloff
    b, a = signal.butter(4, 3800, fs=sr, btype='low')
    audio_data = signal.filtfilt(b, a, audio_data)
    
    return audio_data

def add_background_noise(audio_data, noise_level='low'):
    """Add realistic background noise with multiple components"""
    noise_factors = {'low': 0.001, 'medium': 0.003, 'high': 0.007}
    factor = noise_factors.get(noise_level, 0.001)
    
    # White noise component
    white_noise = np.random.normal(0, factor * 0.5, len(audio_data))
    
    # Pink noise component (1/f noise - more natural)
    pink_noise = np.random.normal(0, factor * 0.5, len(audio_data))
    # Simple pink noise approximation
    pink_noise = np.cumsum(pink_noise)
    pink_noise = pink_noise - np.mean(pink_noise)
    pink_noise = pink_noise / (np.std(pink_noise) + 1e-10) * factor * 0.5
    
    # Occasional clicks/pops (network artifacts)
    if noise_level in ['medium', 'high']:
        num_clicks = random.randint(1, 5 if noise_level == 'high' else 2)
        for _ in range(num_clicks):
            click_pos = random.randint(0, len(audio_data) - 100)
            click_amplitude = random.uniform(0.01, 0.03)
            audio_data[click_pos:click_pos+10] += click_amplitude * np.random.randn(10)
    
    return audio_data + white_noise + pink_noise

def apply_telephony_filtering(audio_data, sr):
    """Apply bandpass filter typical of telephony systems"""
    # WhatsApp/VoIP typically uses 300-3400 Hz for voice
    # Design bandpass filter
    nyquist = sr / 2
    low = 300 / nyquist
    high = 3400 / nyquist
    
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, audio_data)
    
    return filtered

def apply_agc(audio_data, target_level=0.15):
    """Apply Automatic Gain Control (AGC) - typical in VoIP"""
    # Calculate RMS in windows
    window_size = 1600  # ~100ms at 16kHz
    hop = 800
    
    result = audio_data.copy()
    
    for i in range(0, len(audio_data) - window_size, hop):
        window = audio_data[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        
        if rms > 1e-6:  # Avoid division by zero
            gain = target_level / rms
            # Limit gain to avoid amplifying noise too much
            gain = np.clip(gain, 0.5, 3.0)
            result[i:i+window_size] *= gain
    
    return result

def whatsapp_compress(in_path: Path, out_path: Path, quality='medium'):
    """Apply Opus codec compression with varying quality"""
    temp_opus = out_path.with_suffix(".opus")
    
    # WhatsApp uses different bitrates based on network conditions
    bitrates = {
        'poor': '12k',
        'medium': '16k', 
        'good': '24k'
    }
    bitrate = bitrates.get(quality, '16k')
    
    # Encode to Opus with VoIP optimizations
    success = run(
        f'ffmpeg -y -i "{in_path}" -ar 16000 -ac 1 '
        f'-c:a libopus -b:a {bitrate} -application voip '
        f'-vbr on -compression_level 10 "{temp_opus}"'
    )
    
    if not success:
        print(f"Failed to encode {in_path}")
        return False
    
    # Decode back to WAV
    success = run(f'ffmpeg -y -i "{temp_opus}" -ar 16000 -ac 1 "{out_path}"')
    
    if temp_opus.exists():
        temp_opus.unlink()
    
    return success

def apply_voip_artifacts(wav_path: Path, network_quality='medium'):
    """Apply comprehensive VoIP artifacts"""
    try:
        sr, audio = wavfile.read(wav_path)
        
        # Normalize to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        # Apply effects based on network quality
        if network_quality == 'poor':
            audio = add_background_noise(audio, 'high')
            audio = simulate_packet_loss(audio, sr, loss_rate=0.08)
            audio = add_jitter(audio, sr, max_jitter_ms=50)
            audio = apply_agc(audio, target_level=0.12)
        elif network_quality == 'medium':
            audio = add_background_noise(audio, 'medium')
            audio = simulate_packet_loss(audio, sr, loss_rate=0.03)
            audio = add_jitter(audio, sr, max_jitter_ms=30)
            audio = apply_agc(audio, target_level=0.15)
        else:  # good
            audio = add_background_noise(audio, 'low')
            audio = simulate_packet_loss(audio, sr, loss_rate=0.01)
            audio = add_jitter(audio, sr, max_jitter_ms=15)
            audio = apply_agc(audio, target_level=0.18)
        
        # Apply telephony bandpass filtering (all qualities)
        audio = apply_telephony_filtering(audio, sr)
        
        # Add subtle codec artifacts (all qualities)
        audio = add_codec_artifacts(audio, sr)
        
        # Clip and convert back to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
        
        # Overwrite the file
        wavfile.write(wav_path, sr, audio)
        return True
        
    except Exception as e:
        print(f"Error applying VoIP artifacts: {e}")
        return False

def process_file(file: Path, out_dir: Path, create_variants=True):
    """Process a single audio file"""
    clean_name = file.stem.replace(" ", "_")
    
    if create_variants:
        # Create variants with different network qualities
        qualities = ['good', 'medium', 'poor']
        
        for quality in qualities:
            out_path = out_dir / f"{clean_name}_{quality}.wav"
            
            # Step 1: Apply Opus codec compression
            if not whatsapp_compress(file, out_path, quality=quality):
                continue
            
            # Step 2: Apply VoIP network artifacts
            apply_voip_artifacts(out_path, network_quality=quality)
            
            print(f"✓ Processed: {file.name} [{quality}]")
    else:
        # Single file with random quality
        quality = random.choice(['good', 'medium', 'poor'])
        out_path = out_dir / f"{clean_name}_wa.wav"
        
        if whatsapp_compress(file, out_path, quality=quality):
            apply_voip_artifacts(out_path, network_quality=quality)
            print(f"✓ Processed: {file.name} [{quality}]")

def process_folder(kind, create_variants=True):
    """Process all files in a folder"""
    in_dir = RAW_DIRS[kind]
    out_dir = OUT_DIRS[kind]
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    audio_files = [f for f in in_dir.iterdir() 
                   if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']]
    
    print(f"\n{'='*60}")
    print(f"Processing {kind.upper()} audio files")
    print(f"Found {len(audio_files)} files")
    print(f"Output: {'3 variants per file' if create_variants else '1 file per input'}")
    print(f"{'='*60}\n")
    
    for file in audio_files:
        try:
            process_file(file, out_dir, create_variants=create_variants)
        except Exception as e:
            print(f"✗ Failed to process {file.name}: {e}")

# Main execution
if __name__ == "__main__":
    # Check if ffmpeg is available
    if subprocess.run("ffmpeg -version", shell=True, capture_output=True).returncode != 0:
        print("ERROR: ffmpeg not found. Please install ffmpeg first.")
        exit(1)
    
    print("Enhanced WhatsApp VoIP Simulation Pipeline")
    print("="*60)
    print("Features:")
    print("- Opus codec compression (12k/16k/24k)")
    print("- Bursty packet loss patterns")
    print("- Network jitter simulation")
    print("- Telephony bandpass filtering (300-3400 Hz)")
    print("- Automatic Gain Control (AGC)")
    print("- Background noise (white + pink + clicks)")
    print("- Codec quantization artifacts")
    print("="*60)
    print()
    
    # Configuration
    CREATE_VARIANTS = True  # Set to False for single random quality per file
    
    for k in RAW_DIRS:
        if RAW_DIRS[k].exists():
            process_folder(k, create_variants=CREATE_VARIANTS)
        else:
            print(f"Warning: {RAW_DIRS[k]} does not exist, skipping...")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)