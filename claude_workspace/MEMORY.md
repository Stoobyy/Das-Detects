# Das-Detects Project Memory

## Project Goal
Train a model to detect AI-generated vs human voices on WhatsApp VoIP calls.
Lightweight, real-time, offline, TFLite-deployable on Android.

## Decision Engine Thresholds
- **0-40**: Human
- **40-70**: Suspicious
- **70-100**: AI

## Current Best Model: Mel v4 (100% Opus, Expanded Data)
- **Path**: `D:\datasets\models_mel_custom_v4\best_model.keras`
- **Features**: Mel Spectrogram, 100% Opus, RMS -20dB
- **Dataset**: 32k total (16k/class), 67% VoIP
  - Human: 1164 unique clips × 10x + 5280 LibriSpeech
  - AI: 324 unique clips × 34x + 5280 ASVspoof
- **Test Accuracy**: 98.2%, **EER: 1.75%**
- **Real VoIP Human** (61 clips): **91.8%** (56/61) — 3-tier: only 3 hard failures
- **Real VoIP AI** (55 clips): **100%** (55/55) — all scored 70%+

## Previous Models (for reference)
| Model | Human VoIP | AI Detection |
|-------|-----------|-------------|
| v2 LFCC (20x, 30k) | 37.7% | 98.2% |
| v3 LFCC (35x, 28k, 67% VoIP) | 55.7% | 96.4% |
| v3 Mel (same as LFCC v3) | 70.5% | 90.9% |
| **v4 Mel (expanded, 100% Opus)** | **91.8%** | **100%** |

## Backups
- `D:\datasets\backups\models_custom_v3_backup_20260216` (LFCC v3)
- `D:\datasets\backups\models_mel_custom_v3_backup_20260216` (Mel v3)

## Pipeline
1. `scripts/create_merged_manifest_custom.py` → manifest CSV
2. `scripts/prepare_mel_dataset_fast.py --opus-ratio 1.0` → Mel features
3. `scripts/train_from_features.py` → trained model

## Test Clip Locations
- Human VoIP: `C:\Users\amrit\Downloads\aaApp\clipz` (61 WAV)
- AI: `C:\Users\amrit\Downloads\final_Dataset\unused_ai` (55 WAV)
- Training data: `C:\Users\amrit\Downloads\final_Dataset\boosting_selection`

## Next Steps
- Consider GMM-LFCC ensemble for further improvement
- TFLite conversion for Android deployment
