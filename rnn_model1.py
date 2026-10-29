# -*- coding: utf-8 -*-
"""HighD lane-change classification with recording-level train/val/test split.

Labels inherited from process_data.py:
    0 = keep, 1 = left lane change, 2 = right lane change

Run from the project root or from any working directory:
    python rnn_model_group_split.py
"""

import os
import json
import pickle
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "model_output")
os.makedirs(MODEL_DIR, exist_ok=True)

NUM_RECORDINGS = 60
INPUT_SIZE = 16
SEQUENCE_LENGTH = 50
HIDDEN_SIZE = 128
NUM_LSTM_LAYERS = 2
NUM_CLASSES = 3
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
RANDOM_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, validation, test
CLASS_IDS = [0, 1, 2]
CLASS_NAMES = ["keep", "left", "right"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LaneChangeLSTM(nn.Module):
    """Input shape: [batch, 50 time steps, 16 features]."""

    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LSTM_LAYERS, num_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        sequence_outputs, _ = self.lstm(x)
        last_time_step = sequence_outputs[:, -1, :]
        return self.classifier(last_time_step)  # raw logits for CrossEntropyLoss


def load_recording_data():
    """Load each recording separately so recording IDs remain available for splitting."""
    recording_data = {}
    bad_samples = 0

    for recording_id in range(1, NUM_RECORDINGS + 1):
        filename = os.path.join(DATA_DIR, "result{:02d}.pickle".format(recording_id))
        if not os.path.isfile(filename):
            print("[Warning] Missing file, skipped:", filename)
            continue

        with open(filename, "rb") as f:
            samples = pickle.load(f)

        valid_samples = []
        for item in samples:
            try:
                sequence, label = item
                sequence = np.asarray(sequence, dtype=np.float32)
                label = int(label)
                if sequence.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
                    bad_samples += 1
                    continue
                if label not in CLASS_IDS:
                    bad_samples += 1
                    continue
                # Replace non-finite values safely before fitting the scaler.
                sequence = np.nan_to_num(
                    sequence, nan=0.0, posinf=99999.0, neginf=-99999.0
                )
                valid_samples.append((sequence, label))
            except (TypeError, ValueError, IndexError):
                bad_samples += 1

        if valid_samples:
            recording_data[recording_id] = valid_samples
            print("Loaded recording {:02d}: {:,} valid samples".format(
                recording_id, len(valid_samples)))

    if bad_samples:
        print("[Warning] Skipped malformed samples:", bad_samples)
    if len(recording_data) < 10:
        raise RuntimeError(
            "Fewer than 10 recordings with valid samples were found. "
            "Check that main.py has generated output/result01.pickle ... result60.pickle."
        )
    return recording_data


def counts_for_recordings(recording_ids, recording_data):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total = 0
    for rid in recording_ids:
        for _, label in recording_data[rid]:
            counts[label] += 1
            total += 1
    return counts, total


def choose_recording_split(recording_data, trials=12000):
    """Choose a deterministic group split, optimizing class mix and sample proportions.

    The group is the recording ID: no recording can occur in more than one split.
    Target split is approximately 70/15/15 by recording count, while the search
    favors class distributions and sample counts close to those targets.
    """
    rng = random.Random(RANDOM_SEED)
    recording_ids = sorted(recording_data.keys())
    n_groups = len(recording_ids)

    n_train = int(round(n_groups * SPLIT_RATIOS[0]))
    n_val = int(round(n_groups * SPLIT_RATIOS[1]))
    n_test = n_groups - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train -= 1

    per_record_counts = {}
    global_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_samples = 0
    for rid in recording_ids:
        c = np.zeros(NUM_CLASSES, dtype=np.float64)
        for _, label in recording_data[rid]:
            c[label] += 1
        per_record_counts[rid] = c
        global_counts += c
        total_samples += int(c.sum())

    global_class_mix = global_counts / max(global_counts.sum(), 1.0)
    target_sizes = np.asarray(SPLIT_RATIOS, dtype=np.float64)
    best_score = float("inf")
    best_split = None

    for _ in range(trials):
        shuffled = recording_ids[:]
        rng.shuffle(shuffled)
        candidate = {
            "train": shuffled[:n_train],
            "validation": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:],
        }

        split_counts = []
        split_totals = []
        missing_class_penalty = 0.0
        class_mix_error = 0.0
        size_error = 0.0

        for name in ("train", "validation", "test"):
            c = np.zeros(NUM_CLASSES, dtype=np.float64)
            for rid in candidate[name]:
                c += per_record_counts[rid]
            total = float(c.sum())
            split_counts.append(c)
            split_totals.append(total)
            if total <= 0:
                missing_class_penalty += 1000.0
                continue
            if np.any(c == 0):
                missing_class_penalty += 100.0 * int(np.sum(c == 0))
            mix = c / total
            class_mix_error += float(np.mean((mix - global_class_mix) ** 2))

        actual_sizes = np.asarray(split_totals) / max(float(total_samples), 1.0)
        size_error = float(np.mean(((actual_sizes - target_sizes) / target_sizes) ** 2))
        # Missing a class in validation/test is heavily penalized.
        score = 4.0 * size_error + 12.0 * class_mix_error + missing_class_penalty
        if score < best_score:
            best_score = score
            best_split = candidate
            if score < 1e-8:
                break

    if best_split is None:
        raise RuntimeError("Could not create recording-level data split.")

    # Sanity checks: disjoint recording groups and all classes represented.
    train_ids = set(best_split["train"])
    val_ids = set(best_split["validation"])
    test_ids = set(best_split["test"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print("\nRecording-level split (recording IDs never cross splits):")
    for name in ("train", "validation", "test"):
        ids = sorted(best_split[name])
        c, n = counts_for_recordings(ids, recording_data)
        print("{}: {} recordings, {:,} samples, class counts keep/left/right={}, IDs={}".format(
            name, len(ids), n, c.tolist(), ids))
        if np.any(c == 0):
            raise RuntimeError(
                "The {} split is missing at least one class. "
                "Try changing RANDOM_SEED or increasing split-search trials.".format(name)
            )

    split_file = os.path.join(MODEL_DIR, "recording_split.json")
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump({k: sorted(v) for k, v in best_split.items()}, f, indent=2)
    print("Saved recording split to:", split_file)
    return best_split


def flatten_recordings(recording_ids, recording_data):
    xs, ys = [], []
    for rid in sorted(recording_ids):
        for sequence, label in recording_data[rid]:
            xs.append(sequence)
            ys.append(label)
    if not xs:
        raise RuntimeError("A split has no samples.")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_pred.extend(pred.tolist())
            all_true.extend(y_batch.numpy().tolist())

    macro_f1 = f1_score(
        all_true, all_pred, labels=CLASS_IDS, average="macro", zero_division=0
    )
    balanced_acc = balanced_accuracy_score(all_true, all_pred)
    accuracy = accuracy_score(all_true, all_pred)
    return {
        "y_true": np.asarray(all_true, dtype=np.int64),
        "y_pred": np.asarray(all_pred, dtype=np.int64),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "accuracy": float(accuracy),
    }


def make_loader(x, y, batch_size, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Data directory:", DATA_DIR)
    print("Model output directory:", MODEL_DIR)

    recording_data = load_recording_data()
    split = choose_recording_split(recording_data)

    x_train, y_train = flatten_recordings(split["train"], recording_data)
    x_val, y_val = flatten_recordings(split["validation"], recording_data)
    x_test, y_test = flatten_recordings(split["test"], recording_data)

    # Fit feature scaling on TRAINING data only, then apply it to val/test.
    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, INPUT_SIZE))

    def scale_sequences(x):
        scaled = scaler.transform(x.reshape(-1, INPUT_SIZE))
        return scaled.reshape(x.shape).astype(np.float32)

    x_train = scale_sequences(x_train)
    x_val = scale_sequences(x_val)
    x_test = scale_sequences(x_test)

    print("\nSample counts: train={}, validation={}, test={}".format(
        len(y_train), len(y_val), len(y_test)))
    print("Train labels:", dict(Counter(y_train.tolist())))
    print("Validation labels:", dict(Counter(y_val.tolist())))
    print("Test labels:", dict(Counter(y_test.tolist())))

    train_loader = make_loader(x_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(x_val, y_val, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(x_test, y_test, BATCH_SIZE, shuffle=False)

    # Inverse-frequency class weights are calculated from training data only.
    train_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = len(y_train) / (NUM_CLASSES * np.maximum(train_counts, 1.0))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print("Training class weights keep/left/right:", class_weights.tolist())

    model = LaneChangeLSTM().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_macro_f1 = -1.0
    best_val_bal_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = os.path.join(MODEL_DIR, "best_lstm_model.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_seen += len(y_batch)

        train_loss = total_loss / max(total_seen, 1)
        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        val_bal_acc = val_metrics["balanced_accuracy"]

        print(
            "Epoch {:02d}/{:02d} | train_loss={:.5f} | val_Macro-F1={:.4f} | "
            "val_Balanced-Accuracy={:.4f}".format(
                epoch, NUM_EPOCHS, train_loss, val_f1, val_bal_acc
            )
        )

        # Select the checkpoint using validation Macro-F1, never test metrics.
        improved = (val_f1 > best_macro_f1) or (
            np.isclose(val_f1, best_macro_f1) and val_bal_acc > best_val_bal_acc
        )
        if improved:
            best_macro_f1 = val_f1
            best_val_bal_acc = val_bal_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_macro_f1,
                    "best_val_balanced_accuracy": best_val_bal_acc,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "train_recording_ids": sorted(split["train"]),
                    "validation_recording_ids": sorted(split["validation"]),
                    "test_recording_ids": sorted(split["test"]),
                    "config": {
                        "input_size": INPUT_SIZE,
                        "sequence_length": SEQUENCE_LENGTH,
                        "hidden_size": HIDDEN_SIZE,
                        "num_lstm_layers": NUM_LSTM_LAYERS,
                        "num_classes": NUM_CLASSES,
                        "class_names": CLASS_NAMES,
                    },
                },
                checkpoint_path,
            )
            print("  -> Saved best model (validation Macro-F1={:.4f})".format(best_macro_f1))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping: validation Macro-F1 did not improve for {} epochs.".format(
                EARLY_STOPPING_PATIENCE))
            break

    # Load the best checkpoint and evaluate the independent test set ONCE.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "=" * 68)
    print("FINAL INDEPENDENT TEST RESULTS (best epoch = {})".format(checkpoint["best_epoch"]))
    print("=" * 68)
    print("Accuracy:          {:.4f}".format(test_metrics["accuracy"]))
    print("Macro-F1:          {:.4f}".format(test_metrics["macro_f1"]))
    print("Balanced Accuracy: {:.4f}".format(test_metrics["balanced_accuracy"]))
    print("\nClassification report (keep / left / right):")
    print(classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    ))
    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], labels=CLASS_IDS)
    print("Confusion matrix (rows=true, columns=predicted; keep/left/right):")
    print(cm)

    report_path = os.path.join(MODEL_DIR, "final_test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Best epoch: {}\n".format(checkpoint["best_epoch"]))
        f.write("Best validation Macro-F1: {:.6f}\n".format(checkpoint["best_val_macro_f1"]))
        f.write("Best validation Balanced Accuracy: {:.6f}\n\n".format(
            checkpoint["best_val_balanced_accuracy"]))
        f.write("Test Accuracy: {:.6f}\n".format(test_metrics["accuracy"]))
        f.write("Test Macro-F1: {:.6f}\n".format(test_metrics["macro_f1"]))
        f.write("Test Balanced Accuracy: {:.6f}\n\n".format(
            test_metrics["balanced_accuracy"]))
        f.write("Classification report:\n")
        f.write(classification_report(
            test_metrics["y_true"], test_metrics["y_pred"],
            labels=CLASS_IDS, target_names=CLASS_NAMES,
            digits=4, zero_division=0,
        ))
        f.write("\nConfusion matrix (rows=true, columns=predicted):\n")
        f.write(np.array2string(cm))
        f.write("\n")
    print("\nBest model saved to:", checkpoint_path)
    print("Final test report saved to:", report_path)


if __name__ == "__main__":
    main()
