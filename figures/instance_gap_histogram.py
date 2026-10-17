#!/usr/bin/env python3
"""
Gaps between chronologically adjacent curated instances.

For each user, sorts the selected train (and test) instances by start time,
computes the gaps between consecutive instances, prints summary statistics
(median gap, share of gaps below 4 hours) and plots a histogram of the gap
distribution for train vs. test.

By default all curated users are included. Pass --user-ids (a text file with
one user id per line) or --metrics-df (a pickled DataFrame with a `uid`
column, e.g. the evaluation output of the forecasting experiments) to restrict
the figure to the users that were actually evaluated.

Usage:
    python figures/instance_gap_histogram.py [--metrics-df metrics_df.pkl] [--out fig.pdf]
"""
import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DATA_PATH, FILTERING_OUTPUT_PATH, INSTANCE_FOLDER  # noqa: E402

DATA_DIR = os.path.join(OUTPUT_DATA_PATH, INSTANCE_FOLDER)
CURATED_DIR = os.path.join(FILTERING_OUTPUT_PATH, INSTANCE_FOLDER)
PER_USER_DIR = os.path.join(CURATED_DIR, "per_user")
NON_EXERCISE_CSV = os.path.join(DATA_DIR, "non_exercise.csv")


def load_user_ids(user_ids_file=None, metrics_df=None):
    """User ids to include: from --metrics-df, --user-ids, or all curated users."""
    if metrics_df:
        mdf = pd.read_pickle(metrics_df)
        return sorted(mdf["uid"].astype(int).unique())
    if user_ids_file:
        with open(user_ids_file) as f:
            return sorted(int(line.strip()) for line in f if line.strip())
    return sorted(int(f[len("user_"):-len(".pkl")])
                  for f in os.listdir(PER_USER_DIR) if f.endswith(".pkl"))


def load_curated_ids(uids):
    """Returns dict: user_id -> {'train': [eid...], 'test': [eid...]}."""
    uids = set(uids)
    out = {}
    for uid in uids:
        fp = os.path.join(PER_USER_DIR, f"user_{uid}.pkl")
        with open(fp, "rb") as f:
            r = pickle.load(f)
        out[uid] = {"train": r["train_ids"], "test": r["test_ids"]}
    return out


def load_eid_times(uids):
    """Returns Series indexed by (id, ex) -> time_series_start for the given
    users (chunked scan of the candidate-window CSV)."""
    uids = set(uids)
    chunks = []
    for chunk in pd.read_csv(NON_EXERCISE_CSV, usecols=["ex", "id", "time_series_start"],
                              chunksize=2_000_000):
        chunk = chunk[chunk["id"].isin(uids)]
        if len(chunk):
            chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df["time_series_start"] = pd.to_datetime(df["time_series_start"])
    return df.set_index(["id", "ex"])["time_series_start"]


def compute_gaps(curated, lut):
    """Hours between consecutive selected instances, per split."""
    gaps = {"train": [], "test": []}
    n_users_with_pair = {"train": 0, "test": 0}
    for uid, splits in curated.items():
        for split in ("train", "test"):
            eids = splits[split]
            times = [lut[(uid, eid)] for eid in eids if (uid, eid) in lut.index]
            if len(times) < 2:
                continue
            times = np.sort(np.array(times))
            dt_hours = np.diff(times).astype("timedelta64[s]").astype(float) / 3600.0
            gaps[split].append(dt_hours)
            n_users_with_pair[split] += 1
    for split in gaps:
        gaps[split] = np.concatenate(gaps[split]) if gaps[split] else np.array([])
    return gaps, n_users_with_pair


def print_summary(gaps, n_users, n_users_with_pair):
    """Print the median gap and share of gaps below 4 hours per split."""
    for split in ("train", "test"):
        g = gaps[split]
        pct_below_4h = 100.0 * np.mean(g < 4.0)
        print(f"{split}: n_gaps={len(g)}  median={np.median(g):.2f}h  "
              f"min={np.min(g)*60:.1f}min  pct_below_4h={pct_below_4h:.1f}%  "
              f"users_contributing={n_users_with_pair[split]}/{n_users}")


def plot_histogram(gaps, out_path):
    """Plot train and test gap histograms (log count) with the 4-hour target."""
    fig, axes = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True)
    bins = np.linspace(0, 12, 145)  # 5-min bins up to 12h
    specs = [
        ("train", "tab:blue", "Training instances", axes[0]),
        ("test", "tab:orange", "Test instances", axes[1]),
    ]
    for split, color, label, ax in specs:
        g = gaps[split]
        pct_below_4h = 100.0 * np.mean(g < 4.0)
        median = np.median(g)
        median_str = f"{median:.2f} h" if median >= 1 else f"{median * 60:.0f} min"
        ax.hist(g, bins=bins, color=color,
                label=f"{label}\nmedian {median_str}, {pct_below_4h:.0f}% $<$4h")
        ax.axvline(4.0, color="k", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].text(4.05, axes[0].get_ylim()[1] * 0.5, "4-hour\nminimum-gap\ntarget", fontsize=7, va="top")
    axes[-1].set_xlabel("Gap between chronologically-adjacent selected instances (hours)")
    axes[-1].set_xlim(0, 12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user-ids", default=None, help="text file, one user id per line")
    parser.add_argument("--metrics-df", default=None, help="pickled DataFrame with a uid column")
    parser.add_argument("--out", default="fig_instance_gap_histogram.pdf")
    args = parser.parse_args()

    uids = load_user_ids(args.user_ids, args.metrics_df)
    print(f"{len(uids)} users")

    curated = load_curated_ids(uids)
    lut = load_eid_times(uids)
    print(f"{len(lut)} (id, ex) rows loaded")

    gaps, n_users_with_pair = compute_gaps(curated, lut)
    print_summary(gaps, len(uids), n_users_with_pair)
    plot_histogram(gaps, args.out)


if __name__ == "__main__":
    main()
