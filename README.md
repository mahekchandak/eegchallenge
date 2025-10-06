## EEGNet training on BIDS .bdf (NeurIPS EEG Challenge)

### Environment

macOS arm64 with Apple Silicon is supported via tensorflow-macos + tensorflow-metal.

1) Create venv (Python 3.11 recommended):

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install --upgrade pip setuptools wheel
```

2) Install deps:

```bash
pip install 'tensorflow-macos==2.16.2' 'tensorflow-metal==1.2.0' \
  mne==1.6.1 mne-bids==0.15.0 pyEDFlib==0.1.38 pyriemann==0.6 \
  scikit-learn==1.5.2 numpy scipy pandas matplotlib tqdm==4.66.5
```

### Data

Place the BIDS dataset under `R1_mini_L100_bdf/` (already attached in this repo layout). Each subject has `eeg/` with `.bdf`, `_events.tsv`, and sidecar `.json` files.

### Train

Run a quick sanity training on one task/subject:

```bash
source .venv311/bin/activate
python scripts/train_eegnet_bids.py \
  --bids_root ./R1_mini_L100_bdf \
  --tasks symbolSearch \
  --max_subjects 1 \
  --epochs 3 \
  --batch_size 64 \
  --tmin 0.0 --tmax 1.0
```

Outputs:
- Metrics JSON printed to stdout
- CSV saved to `eegnet_metrics.csv` at the repo root

### Notes

- The script resamples to 128 Hz, applies notch/band-pass with Nyquist safeguards, average reference projection, and epochs around all events in the events.tsv. Labels are derived from `trial_type` if present, else `value`.
- If a class has < 2 samples, stratified split falls back to a random split.
- To train across more data, remove or increase `--max_subjects`, and pass multiple `--tasks` or omit to include all tasks.


