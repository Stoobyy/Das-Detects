"""
Audio Recorder Module
=====================
WASAPI loopback recording for capturing system audio during VoIP calls.
Records in configurable frame sizes (default 3 seconds) and stores in a queue.
"""

import threading
import queue
import time
import warnings
import numpy as np

# Suppress soundcard warnings about data discontinuity (normal for loopback)
warnings.filterwarnings("ignore", message="data discontinuity")

try:
    import soundcard as sc
except ImportError:
    raise ImportError("soundcard is required: pip install soundcard")


class AudioRecorder:
    """
    Records system audio via WASAPI loopback in fixed-size frames.
    Thread-safe and designed for continuous recording during calls.
    """
    
    def __init__(self, frame_duration: float = 3.0, samplerate: int = 48000, max_frames: int = 20):
        """
        Args:
            frame_duration: Duration of each audio frame in seconds
            samplerate: Audio sample rate in Hz
            max_frames: Maximum number of frames to keep in queue (oldest dropped if full)
        """
        self.frame_duration = frame_duration
        self.samplerate = samplerate
        self.max_frames = max_frames
        
        self._frames: queue.Queue = queue.Queue(maxsize=max_frames)
        self._recording = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
    
    def start(self) -> bool:
        """Start recording in a background thread. Returns True if started successfully."""
        with self._lock:
            if self._recording:
                return False
            
            self._recording = True
            self._thread = threading.Thread(target=self._record_loop, daemon=True)
            self._thread.start()
            return True
    
    def stop(self):
        """Stop recording and wait for thread to finish."""
        with self._lock:
            self._recording = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording
    
    def get_frame(self, timeout: float = 0.1) -> np.ndarray | None:
        """
        Get the next recorded frame from the queue.
        Returns None if no frame available within timeout.
        """
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_frames(self) -> list[np.ndarray]:
        """Get all available frames from the queue (non-blocking)."""
        frames = []
        while True:
            try:
                frames.append(self._frames.get_nowait())
            except queue.Empty:
                break
        return frames
    
    def clear_frames(self):
        """Clear all frames from the queue."""
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
    
    @property
    def frame_count(self) -> int:
        """Number of frames currently in the queue."""
        return self._frames.qsize()
    
    def _record_loop(self):
        """Internal recording loop - runs in background thread."""
        num_frames = int(self.samplerate * self.frame_duration)
        
        try:
            # Get default speaker for loopback recording
            speaker = sc.default_speaker()
            mic = sc.get_microphone(id=speaker.name, include_loopback=True)
            
            with mic.recorder(samplerate=self.samplerate) as recorder:
                while self._recording:
                    try:
                        # Record one frame
                        data = recorder.record(numframes=num_frames)
                        
                        # Convert to mono if stereo
                        if len(data.shape) > 1 and data.shape[1] > 1:
                            data = np.mean(data, axis=1)
                        
                        # Add to queue, drop oldest if full
                        if self._frames.full():
                            try:
                                self._frames.get_nowait()
                            except queue.Empty:
                                pass
                        
                        self._frames.put(data)
                        
                    except Exception as e:
                        print(f"[AudioRecorder] Recording error: {e}")
                        time.sleep(0.1)
                        
        except Exception as e:
            print(f"[AudioRecorder] Failed to initialize: {e}")
            self._recording = False


# Quick test
if __name__ == "__main__":
    print("Testing AudioRecorder...")
    recorder = AudioRecorder(frame_duration=3.0)
    
    recorder.start()
    print("Recording started. Capturing 2 frames...")
    
    time.sleep(7)  # Wait for ~2 frames
    
    recorder.stop()
    print(f"Recording stopped. Frames captured: {recorder.frame_count}")
    
    frames = recorder.get_all_frames()
    for i, frame in enumerate(frames):
        print(f"  Frame {i+1}: {len(frame)} samples, {len(frame)/48000:.2f}s")
