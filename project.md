# Voice AI Detection (Das-Detects)

## Overview
Das-Detects is a real-time desktop application designed to monitor VoIP calls (like WhatsApp) and detect whether the caller is using an AI-generated voice or a real human voice. It protects users from AI voice cloning scams by actively listening to the call, running inferences using local machine learning models (CNN and GMM), and checking the caller's ID against a community-flagged blocklist stored in Supabase.

## Core Functionality & Architecture
The system operates through a series of multi-threaded, coordinated modules to achieve real-time monitoring without interrupting the user's system performance.

### 1. Call Detection (`voip_monitor.py` & `check_app.py`)
A background thread continuously polls the Windows Audio Session API (WASAPI) using `pycaw` and `comtypes` to detect active VoIP applications (e.g., WhatsApp). 
- It uses a **heuristic hysteresis** pattern: a call is only marked "Active" when both VoIP audio reproduction AND microphone activity are detected. This prevents false positives when a user is merely listening to a voice note.
- It fires asynchronous callbacks (`on_call_start`, `on_call_end`) to wake up the audio recording and UI subsystems only when needed.

### 2. Audio Capture (`audio_recorder.py`)
Once triggered by the VoIP Monitor, the audio recorder loops over the system's loopback interface using `soundcard`.
- Audio is captured in fixed temporal frames (default 2.5 to 3.0 seconds) at a 48kHz sample rate.
- Frames are placed into a non-blocking queue for immediate processing, minimizing memory footprint and latency.

### 3. Feature Extraction & AI Inference (`tflite_inferencer.py` & `gmm_inferencer.py`)
The system processes audio frames using two parallel machine learning models to maximize accuracy and robustness against noise.
- **Silence Pruning**: Audio chunks exhibiting >40% silence (based on RMS thresholds) are fast-discarded to save compute cycles.
- **CNN Classifier (`tflite_inferencer.py`)**: 
  - Extracts fixed-length Mel-spectrogram features (60 mel bands, 512 FFT) representing 2.5s of audio.
  - Normalizes the input to -20 dB RMS.
  - Runs a quantized TensorFlow Lite (`.tflite`) model in a dedicated background worker thread to yield an AI-likelihood probability.
- **GMM Classifier (`gmm_inferencer.py`)**: 
  - Extracts Linear Frequency Cepstral Coefficients (LFCC) from the audio stream.
  - Computes the log-likelihood scores across two pre-trained Gaussian Mixture Models (`.pkl`): one modeling human speech and one modeling AI speech.
  - Converts the likelihood differential into a bounded confidence score via a sigmoid mapping.

### 4. Ensemble Decision Logic (`decision_engine.py`)
To prevent the UI from flickering wildly due to instantaneous false classifications, the `DecisionEngine` acts as the smoothing layer.
- Uses a `deque`-based rolling buffer (default capacity of 5 frames).
- Computes a weighted ensemble average of the CNN score (default weight ~60-70%) and the GMM score (default weight ~30-40%).
- Emits three possible classifications: `Human` (< 40% AI probability), `Suspicious` (40%–70%), and `AI Generated` (> 70%).
- Throws targeted notification flags ONLY on confirmed state transitions to avoid notification spam.

### 5. Caller Blocklist & Cloud Integration (`caller_id_extractor.py` & `supabase_client.py`)
Provides crowd-sourced scammer protection.
- **Extraction**: Uses the `uiautomation` library to scan the Windows UI tree specifically searching for active WhatsApp call windows, traversing up to 15 layers deep to extract the unformatted text.
- **Validation & Privacy**: Cleans the text and asserts it is a pure E.164 phone number. It is then hashed (SHA-256) locally. Pure contact names (saved numbers) are ignored.
- **Supabase Cloud Sync**: Checks the hash against a remote Supabase Postgres instance via REST. If a match is found (`strike_count > 0`), the UI throws an immediate red warning. If the *current* call is deemed AI by the local Decision Engine, the system automatically upserts the hash to the cloud server, incrementing its `strike_count` at call termination.

### 6. Modern User Interface (`voice_call_detection_ui.py`)
Tying the backend together is a premium, dark-mode `customtkinter` interface.
- Utilizes smooth frame-rate regulated `after()` events to animate a faux-waveform visualization driven by a sine-phase math generator.
- Animates color transitions using linear interpolation (`lerp`) when buttons are clicked or when confidence states traverse safe (green), suspicious (yellow), or danger (red) zones.
- Includes OS-level push notifications (`plyer`) summarizing the call once disconnected.

## Directory Structure
- `voice_call_detection_ui.py`: Main app entry point.
- `requirements.txt`: Python package dependencies.
- `project.md`: Project summary and documentation.

### `modules/`
Detailed sub-modules handling discrete business logic as described above (`audio_recorder.py`, `voip_monitor.py`, `decision_engine.py`, etc.).

### `models/`
- Root directory contains the latest `.tflite` CNN models.
- `models/gmm/`: Contains `gmm_ai.pkl` and `gmm_human.pkl`.

### ML Training Pipeline
- `train_pipeline.py`, `augment.py`, `diagnose_dataset.py`, `cnn_classifier.py`: Standalone scripts to preprocess audio data, apply synthetic augmentations, and train the master `.h5` Keras models.
- `convert_tflite.py`: Compresses and quantizes the trained models into the production `.tflite` format.
- `batch_predict.py`: CLI testing script simulating bulk evaluations.

## Setup and Installation
1. Ensure Python 3.9+ is installed (Windows recommended).
2. Clone the repository.
3. Install dependencies via `pip install -r requirements.txt`.
4. Ensure `models/` directory contains at least one `.tflite` and the required `.pkl` files in `models/gmm/`.
5. Set `SUPABASE_URL` and `SUPABASE_KEY` in your `.env` (optional, for blocklist integration).
6. Run `python voice_call_detection_ui.py`.
