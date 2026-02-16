# Das-Detects Project Handoff

Copy this to a new Claude session to continue work.

---

## Project Overview
Training a voice classifier to detect AI-generated vs human voices for WhatsApp VoIP call scenarios.

## Current Best Model: `models_custom_rms`
- **Path**: `D:\datasets\models_custom_rms\best_model.keras`
- **Accuracy on Test Set**: ~99% (Human) / ~99% (AI)
- **Key Fixes**:
  1. **RMS Normalization**: Normalized to -20dB (fixed volume mismatch with quiet inputs).
  2. **Custom Data**: Added "Annabel" and "Bemp" files to training set (boosted 50x).
  3. **Dataset**: Switched from VCTK to **LibriSpeech** (better generalization).

## Inference
To run inference with the best model:
```bash
python scripts/run_inference_opus08_rms.py
```
*Note: Pointing to `D:\datasets\models_custom_rms\best_model.keras`*

## Dataset Configuration
- **Dataset folder**: `D:\datasets`
- **Training Manifest**: `D:\datasets\prepared\manifest_custom_balanced.csv` (LibriSpeech + ASVspoof + Custom 50x)
- **Prepared Data**: `D:\datasets\prepared_opus_custom_rms` (Opus 0.8 + RMS Norm)

## Working Directory
`C:\Users\amrit\OneDrive\Documents\GitHub\Das-Detects\claude_workspace`

## Next Steps
- [ ] **TFLite Conversion**: Convert `models_custom_rms` to TFLite for deployment.
- [ ] **Aggressive Augmentation**: Add noise/reverb to fix the remaining borderline false positive (0.52).
- [ ] **Threshold Tuning**: Consider raising detection threshold to 0.6.

---

**Paste everything above to continue where we left off!**
