import os
import random
import subprocess
import numpy as np
from pathlib import Path
from scipy.io import wavfile
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
    """Simulate packet loss by zeroing out random chunks"""
    packet_size_ms = 20  # Typical VoIP packet size
    packet_samples = int(sr * packet_size_ms / 1000)
    
    num_packets = len(audio_data) // packet_samples
    num_lost = int(num_packets * loss_rate)
    
    lost_packets = random.sample(range(num_packets), num_lost)
    
    result = audio_data.copy()
    for pkt_idx in lost_packets:
        start = pkt_idx * packet_samples
        end = start + packet_samples
        # Simulate packet loss concealment (PLC) with simple interpolation
        if start > 0 and end < len(result):
            # Linear interpolation instead of zeroing
            result[start:end] = np.linspace(result[start-1], result[end], packet_samples)
    
    return result

def add_jitter(audio_data, sr, max_jitter_ms=30):
    """Simulate network jitter by introducing small timing variations"""
    chunk_size = int(sr * 0.02)  # 20ms chunks
    num_chunks = len(audio_data) // chunk_size
    
    result = []
    for i in range(num_chunks):
        chunk = audio_data[i*chunk_size:(i+1)*chunk_size]
        
        # Occasionally add small delays or speed variations
        if random.random() < 0.1:  # 10% of chunks affected
            jitter = random.randint(-5, 5)
            if jitter > 0:
                chunk = np.pad(chunk, (jitter, 0), mode='edge')[:-jitter]
            elif jitter < 0:
                chunk = chunk[-jitter:]
                chunk = np.pad(chunk, (0, -jitter), mode='edge')
        
        result.append(chunk)
    
    # Add remaining samples
    remainder = audio_data[num_chunks*chunk_size:]
    if len(remainder) > 0:
        result.append(remainder)
    
    return np.concatenate(result)

def add_background_noise(audio_data, noise_level='low'):
    """Add realistic background noise"""
    noise_factors = {'low': 0.002, 'medium': 0.005, 'high': 0.01}
    factor = noise_factors.get(noise_level, 0.002)
    
    noise = np.random.normal(0, factor, len(audio_data))
    return audio_data + noise

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
    
    # Encode to Opus
    success = run(f'ffmpeg -y -i "{in_path}" -ar 16000 -ac 1 -c:a libopus -b:a {bitrate} -application voip "{temp_opus}"')
    
    if not success:
        print(f"Failed to encode {in_path}")
        return False
    
    # Decode back to WAV
    success = run(f'ffmpeg -y -i "{temp_opus}" -ar 16000 -ac 1 "{out_path}"')
    
    if temp_opus.exists():
        temp_opus.unlink()
    
    return success

def apply_voip_artifacts(wav_path: Path, network_quality='medium'):
    """Apply packet loss, jitter, and noise to simulate VoIP"""
    try:
        sr, audio = wavfile.read(wav_path)
        
        # Normalize to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        # Apply effects based on network quality
        if network_quality == 'poor':
            audio = simulate_packet_loss(audio, sr, loss_rate=0.08)
            audio = add_jitter(audio, sr, max_jitter_ms=50)
            audio = add_background_noise(audio, 'high')
        elif network_quality == 'medium':
            audio = simulate_packet_loss(audio, sr, loss_rate=0.03)
            audio = add_jitter(audio, sr, max_jitter_ms=30)
            audio = add_background_noise(audio, 'medium')
        else:  # good
            audio = simulate_packet_loss(audio, sr, loss_rate=0.01)
            audio = add_jitter(audio, sr, max_jitter_ms=15)
            audio = add_background_noise(audio, 'low')
        
        # Clip and convert back to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
        
        # Overwrite the file
        wavfile.write(wav_path, sr, audio)
        return True
        
    except Exception as e:
        print(f"Error applying VoIP artifacts: {e}")
        return False

def process_file(file: Path, out_dir: Path):
    """Process a single audio file with multiple quality variants"""
    clean_name = file.stem.replace(" ", "_")
    
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

def process_folder(kind):
    """Process all files in a folder"""
    in_dir = RAW_DIRS[kind]
    out_dir = OUT_DIRS[kind]
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    audio_files = [f for f in in_dir.iterdir() 
                   if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']]
    
    print(f"\n{'='*60}")
    print(f"Processing {kind.upper()} audio files")
    print(f"Found {len(audio_files)} files")
    print(f"{'='*60}\n")
    
    for file in audio_files:
        try:
            process_file(file, out_dir)
        except Exception as e:
            print(f"✗ Failed to process {file.name}: {e}")

# Main execution
if __name__ == "__main__":
    # Check if ffmpeg is available
    if subprocess.run("ffmpeg -version", shell=True, capture_output=True).returncode != 0:
        print("ERROR: ffmpeg not found. Please install ffmpeg first.")
        exit(1)
    
    print("Starting WhatsApp VoIP simulation pipeline...")
    print("This will create 3 variants per file: good, medium, poor quality\n")
    
    for k in RAW_DIRS:
        if RAW_DIRS[k].exists():
            process_folder(k)
        else:
            print(f"Warning: {RAW_DIRS[k]} does not exist, skipping...")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)