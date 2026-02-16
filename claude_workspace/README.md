# AI Voice Detection System

A lightweight, real-time AI-generated voice detection system designed for offline VoIP call analysis.

## Overview

This system performs binary classification to distinguish between AI-generated/synthetic speech and genuine human speech. It's optimized for:

- **Real-time inference** (<100ms per 2.5-second window)
- **Offline operation** (no cloud dependency)
- **VoIP-grade audio** (16 kHz mono)
- **Desktop deployment** via TensorFlow Lite

## Architecture

### Model Design

The system uses a **MobileNet-style CNN** with depthwise separable convolutions:

```
Input: LFCC spectrogram (60, 79, 1) from 2.5s audio

Block 1: Conv2D(16, 3x3, stride=2) + BN + ReLU6
Block 2: DepthwiseSeparable(32) + BN + ReLU6 + MaxPool
Block 3: DepthwiseSeparable(64) + BN + ReLU6 + MaxPool
Block 4: DepthwiseSeparable(128) + BN + ReLU6 + MaxPool

GlobalAveragePooling2D
Dense(64) + ReLU6 + Dropout(0.3)
Dense(1, sigmoid)
```

**Performance:**
- Parameters: ~300-500K
- Model size: ~1.5MB (Keras) → ~400KB (TFLite quantized)
- Inference: 30-50ms on CPU
- Expected accuracy: 88-92%

### Feature Extraction

We use **Linear Frequency Cepstral Coefficients (LFCC)** instead of Mel spectrograms because:
- Linear filterbanks better preserve high-frequency artifacts common in AI speech
- Proven effective in ASVspoof challenges
- Computationally efficient

## Project Structure

```
claude_workspace/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── config.py                    # Configuration constants
│
├── features/                    # Feature extraction
│   ├── lfcc.py                  # LFCC extraction
│   ├── mel_spectrogram.py       # Mel spectrogram (alternative)
│   └── audio_utils.py           # Audio loading/preprocessing
│
├── models/                      # Model architecture
│   ├── lightweight_cnn.py       # MobileNet-style CNN
│   └── layers.py                # Custom layers
│
├── data/                        # Data pipeline
│   ├── dataset_loader.py        # Dataset loading
│   └── augmentation.py          # VoIP augmentation
│
├── training/                    # Training utilities
│   ├── trainer.py               # Training loop
│   └── metrics.py               # EER, FAR, FRR metrics
│
├── inference/                   # Inference pipeline
│   ├── predictor.py             # Real-time predictor
│   └── tflite_converter.py      # TFLite conversion
│
└── scripts/                     # CLI scripts
    ├── train.py                 # Training script
    ├── evaluate.py              # Evaluation script
    ├── convert_model.py         # TFLite conversion
    └── benchmark.py             # Speed benchmarking
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

Organize your data in the following structure:

```
data/
├── human/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── ai/
    ├── sample1.wav
    ├── sample2.wav
    └── ...
```

Audio requirements:
- Format: WAV (preferred), MP3, FLAC, OGG
- Sample rate: Any (will be resampled to 16 kHz)
- Channels: Mono or stereo (will be converted to mono)
- Duration: At least 1 second (2.5s windows will be padded)

### 2. Train Model

```bash
python scripts/train.py --data-dir path/to/data --epochs 50 --batch-size 32
```

Options:
- `--data-dir`: Path to data directory
- `--model-dir`: Where to save models (default: `saved_models/`)
- `--epochs`: Training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--max-samples`: Limit samples per class (for testing)

### 3. Evaluate Model

```bash
python scripts/evaluate.py --model-path saved_models/best_model.keras --data-dir path/to/data
```

### 4. Convert to TFLite

```bash
python scripts/convert_model.py --model-path saved_models/best_model.keras --quantize --verify --benchmark
```

Options:
- `--quantize`: Apply int8 quantization (recommended)
- `--verify`: Verify converted model
- `--benchmark`: Run speed benchmark
- `--compare`: Compare with original Keras model

### 5. Run Inference

```python
from inference import VoicePredictor

# Load model
predictor = VoicePredictor("saved_models/model.tflite")

# Predict on file
probability, label, inference_time = predictor.predict_file("audio.wav")
print(f"{label}: {probability:.2%} ({inference_time:.1f}ms)")

# Predict on audio array
import numpy as np
audio = np.random.randn(40000)  # 2.5s at 16kHz
probability, label, inference_time = predictor.predict_audio(audio)
```

### 6. Benchmark Speed

```bash
python scripts/benchmark.py --model-path saved_models/model.tflite --iterations 100
```

## Recommended Datasets

### AI/Synthetic Voice
| Dataset | Size | Source | License |
|---------|------|--------|---------|
| ASVspoof 2021 LA | ~600K utterances | asvspoof.org | Research |
| WaveFake | ~118K samples | GitHub | Open |

### Human Voice
| Dataset | Size | Source | License |
|---------|------|--------|---------|
| LibriSpeech | 1000+ hours | openslr.org/12 | CC BY 4.0 |
| Common Voice | 2500+ hours | commonvoice.mozilla.org | CC0 |
| VCTK | 110 speakers | Edinburgh DataShare | Open |

## Configuration

All parameters are in `config.py`:

```python
# Audio
SAMPLE_RATE = 16000      # 16 kHz
AUDIO_DURATION = 2.5     # seconds per window

# Features
N_LFCC = 60              # LFCC coefficients
N_FFT = 512              # FFT size
HOP_LENGTH = 160         # 10ms hop

# Training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3
LABEL_SMOOTHING = 0.1

# Inference
DETECTION_THRESHOLD = 0.5
```

## Metrics

The system reports:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: For AI class detection
- **FAR** (False Acceptance Rate): AI classified as human
- **FRR** (False Rejection Rate): Human classified as AI
- **EER** (Equal Error Rate): Point where FAR = FRR

## Augmentation

VoIP simulation augmentations are applied during training:

- Background noise injection (5-20 dB SNR)
- Low-bitrate codec simulation
- Packet loss simulation
- Time shifting
- Room reverb

## Common Issues

### High False Positive Rate
- Add music/singing to human training set
- Increase noise augmentation
- Lower detection threshold

### Poor Generalization
- Train on multiple TTS engines (ASVspoof has 17+)
- Use heavier augmentation
- Include noisy human samples

### Slow Inference
- Use TFLite quantized model
- Reduce LFCC coefficients
- Use smaller model architecture

## API Reference

### VoicePredictor

```python
from inference import VoicePredictor

predictor = VoicePredictor(
    model_path="model.tflite",
    threshold=0.5,
    use_tflite=True,
)

# Single prediction
prob, label, time_ms = predictor.predict_audio(audio_array)
prob, label, time_ms = predictor.predict_file("audio.wav")

# Streaming prediction
final_prob, label, chunk_probs = predictor.predict_stream(
    audio_chunks, aggregate="mean"
)

# Benchmarking
results = predictor.benchmark(n_iterations=100)
```

### StreamingPredictor

For real-time VoIP integration:

```python
from inference.predictor import StreamingPredictor

streamer = StreamingPredictor(
    model_path="model.tflite",
    window_size=2.5,
    hop_size=0.5,
    smoothing_window=5,
)

# Feed audio chunks as they arrive
result = streamer.feed(audio_chunk)
if result:
    prob, label = result
    print(f"Detection: {label} ({prob:.2%})")
```

## Success Criteria

- [x] Inference time <100ms on CPU
- [x] TFLite model <1MB
- [ ] Test accuracy >85%
- [ ] False positive rate <10%
- [x] Works on 16kHz mono audio

## License

Research use. See individual dataset licenses for training data.
