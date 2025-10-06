## NeurIPS EEG Challenge - Quick Doc

### What I set up

- Created a reproducible environment for macOS/Apple Silicon using TensorFlow macOS + Metal.
- Implemented a BIDS-compatible loader for `.bdf` EEG with MNE:
  - Picks EEG channels robustly and excludes status/stim.
  - Average reference (projection) added.
  - Notch/band-pass filtering with Nyquist safeguards.
  - Resamples to 128 Hz to match EEGNet defaults.
  - Epochs around all rows in each `_events.tsv` and derives labels from `trial_type` (fallback to `value`).
  - Aligns labels to the epochs actually kept by MNE.
- Trained the provided `EEGNet` from `arl-eegmodels/EEGModels.py` with a stratified split (falls back to random split if a class has < 2 samples).

Key files:
- `scripts/train_eegnet_bids.py`: end-to-end load → preprocess → epoch → train → evaluate.
- `arl-eegmodels/EEGModels.py`: EEGNet model definition (unchanged).

### How to run (quick start)

1) Create and activate Python 3.11 venv (recommended):

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install --upgrade pip setuptools wheel
```

2) Install packages:

```bash
pip install 'tensorflow-macos==2.16.2' 'tensorflow-metal==1.2.0' \
  mne==1.6.1 mne-bids==0.15.0 pyEDFlib==0.1.38 pyriemann==0.6 \
  scikit-learn==1.5.2 numpy scipy pandas matplotlib tqdm==4.66.5
```

3) Sanity train on one subject/task:

```bash
python scripts/train_eegnet_bids.py \
  --bids_root ./R1_mini_L100_bdf \
  --tasks symbolSearch \
  --epochs 50 \
  --batch_size 64 \
  --tmin 0.0 --tmax 1.0

```

Outputs:
- Metrics printed to stdout.
- CSV written to `eegnet_metrics.csv` in the repo root.

### Customization tips

- Use multiple tasks or all tasks: drop `--tasks` or list several.
- Scale subjects: increase or remove `--max_subjects`.
- Epoch window: adjust `--tmin/--tmax`.
- Training config: tune `--epochs`, `--batch_size`, `--lr`.

If you hit class-imbalance on tiny subsets, the script will fall back from stratified to random split automatically.


