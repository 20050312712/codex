# -*- coding: utf-8 -*-
"""Generate future-3-second lane-change samples for all highD recordings."""

import os
import pickle
from collections import Counter

from process_data1 import run

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
NUM_RECORDINGS = 60
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_counts = Counter()
total_samples = 0

for recording_id in range(1, NUM_RECORDINGS + 1):
    number = "{:02d}".format(recording_id)
    samples, positive_windows = run(number)

    filename = os.path.join(OUTPUT_DIR, "result{}.pickle".format(number))
    with open(filename, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    counts = Counter(int(label) for _, label in samples)
    all_counts.update(counts)
    total_samples += len(samples)
    print(
        "Saved {}: samples={} | keep={} left={} right={} | positive_windows={}".format(
            filename, len(samples), counts[0], counts[1], counts[2], positive_windows
        )
    )

print("\n" + "=" * 60)
print("ALL RECORDINGS SUMMARY")
print("=" * 60)
print("Total samples:", total_samples)
print("keep (0):", all_counts[0])
print("left (1):", all_counts[1])
print("right (2):", all_counts[2])
print("Positive prediction windows (left + right):", all_counts[1] + all_counts[2])
print("Output directory:", OUTPUT_DIR)
print("\nNext step: run python rnn_model.py")
