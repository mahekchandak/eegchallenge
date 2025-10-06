# NeurIPS EEG Challenge Submission Guide

## What I've Created

I've converted your EEGNet model to be compatible with the NeurIPS EEG Challenge submission format:

### Key Files:
- `submission.py` - Main submission file with Submission class
- `models/eegnet_pytorch.py` - PyTorch implementation of EEGNet  
- `scripts/train_pytorch_eegnet.py` - Training script for PyTorch version
- `create_submission.py` - Script to create submission zip

## Quick Start

### 1. Train Your Model

Train on your BIDS dataset with more epochs:

```bash
source .venv311/bin/activate

# Train for challenge 1 (adjust parameters as needed)
python scripts/train_pytorch_eegnet.py \
  --bids_root ./R1_mini_L100_bdf \
  --epochs 50 \
  --batch_size 32 \
  --tmin 0.0 --tmax 2.0 \
  --resample_hz 100 \
  --save_weights weights_challenge_1.pt

# Train on ALL subjects (default behavior)
python scripts/train_pytorch_eegnet.py \
  --bids_root ./R1_mini_L100_bdf \
  --epochs 50 \
  --save_weights weights_challenge_1.pt

# Train on specific tasks only
python scripts/train_pytorch_eegnet.py \
  --bids_root ./R1_mini_L100_bdf \
  --tasks symbolSearch contrastChangeDetection \
  --epochs 50 \
  --save_weights weights_challenge_1.pt
```

### 2. Create Submission Zip

```bash
python create_submission.py
```

This creates `my_submission.zip` containing:
- `submission.py` - Main submission class
- `eegnet_pytorch.py` - Model implementation
- `weights_challenge_1.pt` - Trained weights for challenge 1
- `weights_challenge_2.pt` - Trained weights for challenge 2

### 3. Submit

Upload `my_submission.zip` to the NeurIPS EEG Challenge platform.

## Model Details

**Architecture**: EEGNet (PyTorch implementation)
- Input: (batch_size, 129, 200) - 129 channels, 200 time points (2s @ 100Hz)
- Output: (batch_size, 2) - Binary classification logits
- Parameters: ~6,834 trainable parameters

**Key Adaptations**:
- Kernel length scaled for 100Hz sampling rate (challenge requirement)
- Compatible with both Challenge 1 and Challenge 2 formats
- Robust channel picking and preprocessing pipeline
- Automatic fallback for missing weights

## Training Tips

For better performance:
- **Default**: Uses ALL subjects in the dataset (no `--max_subjects` limit)
- Tune `--epochs`, `--lr`, `--batch_size` 
- Try different tasks: `symbolSearch`, `contrastChangeDetection`, `seqLearning6target`, etc.
- Experiment with `--tmin`/`--tmax` epoch windows
- Use `--max_subjects N` only if you want to limit for faster testing

## Validation

The submission format has been tested to ensure:
- ✅ Correct PyTorch model structure
- ✅ Compatible with challenge evaluation (129 channels, 200 samples)
- ✅ Proper weight loading/saving
- ✅ No folder structure (challenge requirement)
- ✅ Required Submission class with get_model_challenge_1/2 methods

Ready to submit! 🚀
