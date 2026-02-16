"""
Configuration constants for AI Voice Detection System.

All hyperparameters and paths are centralized here for easy modification.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data_files"
MODEL_DIR = PROJECT_ROOT / "saved_models"
LOG_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# Audio Parameters
# =============================================================================
SAMPLE_RATE = 16000  # 16 kHz for VoIP compatibility
AUDIO_DURATION = 2.5  # seconds per window
AUDIO_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)  # 40000 samples

# =============================================================================
# Audio Preprocessing
# =============================================================================
NORMALIZATION_METHOD = "rms"   # 'peak' or 'rms'
NORMALIZATION_TARGET_DB = -20.0  # Target dB for RMS normalization

# =============================================================================
# Feature Extraction - LFCC
# =============================================================================
N_FFT = 512  # FFT window size
HOP_LENGTH = 160  # 10ms hop at 16kHz
WIN_LENGTH = 400  # 25ms window at 16kHz
N_LFCC = 60  # Number of LFCC coefficients
N_FILTERS = 80  # Number of linear filters

# =============================================================================
# Feature Extraction - Mel Spectrogram (alternative)
# =============================================================================
N_MELS = 60  # Number of mel bands
FMIN = 0  # Minimum frequency
FMAX = 8000  # Maximum frequency (Nyquist for 16kHz)

# =============================================================================
# Model Input Shape
# =============================================================================
# LFCC output: (n_lfcc, time_frames, 1)
# Time frames = (audio_samples - n_fft) / hop_length + 1 ≈ 247
# After processing: (60, 247, 1) -> but we use fixed 79 frames for 2.5s
TIME_FRAMES = 79  # Calculated for 2.5s window
INPUT_SHAPE = (N_LFCC, TIME_FRAMES, 1)

# =============================================================================
# Model Architecture
# =============================================================================
MODEL_FILTERS = [16, 32, 64, 128]  # Channel progression
DROPOUT_RATE = 0.3
DENSE_UNITS = 64

# =============================================================================
# Training Parameters
# =============================================================================
BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 10
LR_REDUCE_PATIENCE = 5
LR_REDUCE_FACTOR = 0.5

# Label smoothing for regularization
LABEL_SMOOTHING = 0.1

# Class weights (adjust if dataset is imbalanced)
# None means auto-compute from data
CLASS_WEIGHTS = None

# =============================================================================
# Validation
# =============================================================================
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# =============================================================================
# Augmentation Parameters
# =============================================================================
AUGMENT_PROBABILITY = 0.5

# Noise injection
NOISE_MIN_SNR_DB = 5
NOISE_MAX_SNR_DB = 20

# Codec simulation (bitrate range for VoIP simulation)
CODEC_MIN_BITRATE = 8000
CODEC_MAX_BITRATE = 32000

# Time shifting
MAX_TIME_SHIFT_SAMPLES = 1600  # 100ms at 16kHz

# =============================================================================
# TFLite Conversion
# =============================================================================
QUANTIZE_MODEL = True
REPRESENTATIVE_DATASET_SIZE = 100

# =============================================================================
# Inference
# =============================================================================
DETECTION_THRESHOLD = 0.5  # Probability threshold for AI classification
MIN_AUDIO_LENGTH = 1.0  # Minimum audio length in seconds

# =============================================================================
# Labels
# =============================================================================
LABEL_HUMAN = 0
LABEL_AI = 1
LABEL_NAMES = {LABEL_HUMAN: "Human", LABEL_AI: "AI-Generated"}
