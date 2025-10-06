import os
import sys
import json
import glob
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ensure we can import EEGNet from the provided repo path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
EEGMODELS_DIR = os.path.join(ROOT_DIR, "arl-eegmodels")
if EEGMODELS_DIR not in sys.path:
    sys.path.insert(0, EEGMODELS_DIR)

from EEGModels import EEGNet  # type: ignore

import mne
from mne import Epochs
from mne.io.constants import FIFF
from mne.io import read_raw_bdf
from mne_bids import BIDSPath
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


def find_bids_runs(bids_root: str, subjects: List[str] | None = None, tasks: List[str] | None = None) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    subject_dirs = sorted([d for d in glob.glob(os.path.join(bids_root, "sub-*")) if os.path.isdir(d)])
    for sub_dir in subject_dirs:
        subject = os.path.basename(sub_dir).replace("sub-", "")
        if subjects and (f"sub-{subject}" not in subjects and subject not in subjects):
            continue
        eeg_dir = os.path.join(sub_dir, "eeg")
        if not os.path.isdir(eeg_dir):
            continue
        for bdf in sorted(glob.glob(os.path.join(eeg_dir, "*.bdf"))):
            fname = os.path.basename(bdf)
            # Expect BIDS-like: sub-<id>_task-<task>[_run-<run>]_eeg.bdf
            parts = fname.split("_")
            task = None
            run = None
            for part in parts:
                if part.startswith("task-"):
                    task = part[len("task-") :]
                if part.startswith("run-"):
                    run = part[len("run-") :]
            if tasks and task not in tasks:
                continue
            events_tsv = bdf.replace("_eeg.bdf", "_events.tsv")
            if not os.path.exists(events_tsv):
                continue
            records.append({
                "subject": subject,
                "task": task or "",
                "run": run or "",
                "bdf": bdf,
                "events": events_tsv,
            })
    return records


def build_event_id(events_df: pd.DataFrame, trial_type_col: str = "trial_type") -> Tuple[Dict[str, int], np.ndarray]:
    if trial_type_col not in events_df.columns:
        # fallback: use value column if present else all ones
        labels = events_df.get("value", pd.Series(np.ones(len(events_df), dtype=int))).astype(str)
    else:
        labels = events_df[trial_type_col].astype(str).fillna("n/a")
    le = LabelEncoder()
    y = le.fit_transform(labels.values)
    event_id = {cls: int(i) for i, cls in enumerate(le.classes_)}
    return event_id, y


def load_epochs_from_bdf(bdf_path: str, events_tsv: str, tmin: float = 0.0, tmax: float = 1.0,
                         l_freq: float | None = 1.0, h_freq: float | None = 40.0,
                         notch: float | None = 60.0, resample_hz: int = 128) -> Tuple[Epochs, np.ndarray, Dict[str, int]]:
    raw = read_raw_bdf(bdf_path, preload=True, verbose=False)
    # Robust channel picking: prefer EEG type, exclude stim/status
    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, ecg=False, seeg=False, misc=False)
    if len(picks) == 0:
        # Fallback: drop known status/stim channels heuristically
        picks = [i for i, name in enumerate(raw.ch_names) if "status" not in name.lower() and "stim" not in name.lower()]
    if len(picks) == 0:
        raise ValueError("No EEG-like channels found to pick.")
    raw.pick(picks)
    # Set average reference (projection) to stabilize
    try:
        raw.set_eeg_reference('average', projection=True)
    except Exception:
        pass
    # Basic filtering with safeguards vs Nyquist
    sfreq = float(raw.info["sfreq"])
    nyq = sfreq / 2.0
    safe_h_freq = None
    if h_freq is not None:
        safe_h = min(h_freq, max(1.0, nyq - 1.0))
        safe_h_freq = safe_h
    safe_l_freq = l_freq
    if notch and notch >= nyq:
        notch = None
    if notch:
        raw.notch_filter(freqs=[notch], verbose=False)
    if safe_l_freq is not None or safe_h_freq is not None:
        raw.filter(l_freq=safe_l_freq, h_freq=safe_h_freq, verbose=False)
    # Resample to target Hz (100Hz for challenge)
    if resample_hz and abs(raw.info["sfreq"] - resample_hz) > 1e-3:
        raw.resample(resample_hz, npad="auto")
        print(f"Resampled from {raw.info['sfreq']} Hz to {resample_hz} Hz")

    # Read events.tsv (BIDS), build MNE events
    ev = pd.read_csv(events_tsv, sep="\t")
    # Ensure 'onset' in seconds
    if "onset" not in ev.columns:
        raise ValueError(f"No onset column in {events_tsv}")
    # Drop NaN onsets and keep within bounds
    ev = ev[ev["onset"].notna()].copy()
    onset_samp_all = (ev["onset"].astype(float) * raw.info["sfreq"]).astype(int).values
    mask_inrange = (onset_samp_all >= 0) & (onset_samp_all < raw.n_times)
    ev = ev.loc[mask_inrange].reset_index(drop=True)
    onset_samp = onset_samp_all[mask_inrange]
    # Use a placeholder id first; actual labels via event_id mapping
    temp_ids = np.arange(1, len(ev) + 1, dtype=int)
    mne_events = np.c_[onset_samp, np.zeros_like(onset_samp), temp_ids]
    event_id_map, y_all = build_event_id(ev)

    # Map temp_ids to consolidated event ids according to label encoder order
    # We reindex events to contiguous integers [0..n_classes-1]
    # MNE requires event_id dict mapping label->code; we will pass a generic mapping
    # and then use y separately as labels for model training.
    # Here, set all events to code 1 to allow epoching over all events; labels come from y.
    mne_events[:, 2] = 1
    mne_event_id = {"stim": 1}

    epochs = mne.Epochs(raw, mne_events, event_id=mne_event_id, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False, event_repeated='drop')
    # Align labels to kept epochs
    kept_idx = np.array(epochs.selection, dtype=int)
    y = y_all[kept_idx] if len(kept_idx) > 0 else y_all
    return epochs, y, event_id_map


def prepare_data(bids_roots: List[str] | str, subjects: List[str] | None, tasks: List[str] | None,
                 max_subjects: int | None = None, tmin: float = 0.0, tmax: float = 1.0, 
                 resample_hz: int = 128) -> Tuple[np.ndarray, np.ndarray, int, int, Dict[str, int]]:
    all_records: List[Dict[str, str]] = []
    if isinstance(bids_roots, str):
        bids_roots = [bids_roots]

    for bids_root in bids_roots:
        all_records.extend(find_bids_runs(bids_root, subjects=subjects, tasks=tasks))

    if not all_records:
        raise RuntimeError("No BDF runs found under any of the provided BIDS roots.")

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    event_id_agg: Dict[str, int] | None = None
    subjects_seen: set[str] = set()

    for rec in all_records:
        subj = rec["subject"]
        if max_subjects is not None and subj not in subjects_seen and len(subjects_seen) >= max_subjects:
            continue

        epochs, y, event_id = load_epochs_from_bdf(rec["bdf"], rec["events"], tmin=tmin, tmax=tmax, resample_hz=resample_hz)
        data = epochs.get_data(copy=False)  # shape: (n_epochs, n_chans, n_times)
        # Ensure consistent channel order across runs
        if data.ndim != 3:
            continue
        X_list.append(data)
        y_list.append(y)
        subjects_seen.add(subj)
        if event_id_agg is None:
            event_id_agg = event_id

    if not X_list:
        raise RuntimeError("No epochs assembled.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    n_trials, n_chans, n_samples = X.shape
    # Add channel-last singleton
    X = X[..., np.newaxis]
    assert X.shape == (n_trials, n_chans, n_samples, 1)
    n_classes = int(np.unique(y).size)
    return X.astype(np.float32), y.astype(int), n_chans, n_samples, (event_id_agg or {})


from sklearn.model_selection import train_test_split

def train_eegnet(X: np.ndarray, y: np.ndarray, n_chans: int, n_samples: int,
                 lr: float = 1e-3, batch_size: int = 64, epochs: int = 20,
                 seed: int = 42) -> Dict[str, float]:

    # one-hot encode
    y_cat = to_categorical(y)

    # 70% train, 30% temp (val+test)
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y_cat, test_size=0.50, random_state=seed, stratify=None
    
    )

    # split temp → 15% val, 15% test
    X_va, X_te, y_va, y_te = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=None
    )

    # build and train model
    model = EEGNet(nb_classes=y_cat.shape[1], Chans=n_chans, Samples=n_samples,
                   dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, dropoutType='Dropout')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

    hist = model.fit(X_tr, y_tr, batch_size=batch_size, epochs=epochs, verbose=2,
                     validation_data=(X_va, y_va))

    eval_loss, eval_acc = model.evaluate(X_va, y_va, verbose=0)
    test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)

    return {
        "val_accuracy": float(eval_acc),
        "val_loss": float(eval_loss),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "train_accuracy": float(hist.history.get('accuracy', [np.nan])[-1]),
        "train_loss": float(hist.history.get('loss', [np.nan])[-1]),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train EEGNet on BIDS .bdf EEG data (NeurIPS EEG Challenge)")
    parser.add_argument("--bids_roots", type=str, nargs='+', default=[os.path.join(ROOT_DIR, "R1_mini_L100_bdf")], help="One or more BIDS root directories.")
    parser.add_argument("--tasks", type=str, nargs="*", default=None, help="Optional: restrict to these tasks (e.g., symbolSearch)")
    parser.add_argument("--subjects", type=str, nargs="*", default=None, help="Optional: restrict to these subjects (e.g., sub-XXX)")
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_csv", type=str, default=os.path.join(ROOT_DIR, "eegnet_metrics.csv"))
    args = parser.parse_args()

    X, y, n_chans, n_samples, event_id = prepare_data(
        bids_roots=args.bids_roots,
        subjects=args.subjects,
        tasks=args.tasks,
        max_subjects=None,  # Use all subjects by default
        tmin=args.tmin,
        tmax=args.tmax,
    )

    metrics = train_eegnet(
        X, y, n_chans, n_samples,
        lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(args.out_csv, index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
