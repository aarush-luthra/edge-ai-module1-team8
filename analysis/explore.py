"""Mission 1 exploration: histograms of every column and days_left against each input.

    python analysis/explore.py --team 8
Saves PNGs into analysis/. Not used by the server or the grader.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
a = ap.parse_args()

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")
inputs = ["battery_v", "temp_c", "rings_per_day", "wifi_rssi_dbm", "age_months"]

# Raw histograms: glitches show up as lonely bars far from the rest.
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, df.columns):
    ax.hist(df[col], bins=60)
    ax.set_title(col)
fig.tight_layout()
fig.savefig("analysis/hist_raw.png", dpi=110)

# days_left against each input. Glitch rows are drawn in red so they don't hide the shape.
glitch = (df.battery_v <= 0.5) | (df.temp_c <= -50) | (df.wifi_rssi_dbm >= -10)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, inputs):
    ax.scatter(df.loc[~glitch, col], df.loc[~glitch, "days_left"], s=4, alpha=0.4)
    ax.scatter(df.loc[glitch, col], df.loc[glitch, "days_left"], s=12, c="red", label="glitch")
    ax.set_xlabel(col)
    ax.set_ylabel("days_left")
axes.flat[0].legend()
axes.flat[-1].axis("off")
fig.tight_layout()
fig.savefig("analysis/scatter_raw.png", dpi=110)

# Same scatter with glitches removed, so the real shape of each relationship is visible.
clean = df[~glitch]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, inputs):
    ax.scatter(clean[col], clean.days_left, s=4, alpha=0.4)
    ax.set_xlabel(col)
    ax.set_ylabel("days_left")
axes.flat[-1].axis("off")
fig.tight_layout()
fig.savefig("analysis/scatter_clean.png", dpi=110)

print(f"{len(df)} rows; candidate glitch rows: {glitch.sum()}")
print("  battery_v <= 0.5 :", (df.battery_v <= 0.5).sum())
print("  temp_c <= -50    :", (df.temp_c <= -50).sum())
print("  wifi_rssi >= -10 :", (df.wifi_rssi_dbm >= -10).sum())
print("Correlation of days_left with each input (clean rows):")
print(clean.corr()["days_left"].drop("days_left").round(3).to_string())
