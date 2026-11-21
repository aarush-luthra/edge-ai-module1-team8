"""Mission 1: plot the linear baseline's predicted against true days_left, and show where it is wrong.

    python analysis/linear_plot.py --team 8
Uses the same glitch rule and the same validation split (seed 0) as train.py.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURES = ["battery_v", "temp_c", "rings_per_day", "wifi_rssi_dbm", "age_months"]
ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
a = ap.parse_args()

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")
df = df[~((df.battery_v <= 0.5) | (df.temp_c <= -50) | (df.wifi_rssi_dbm >= -10))]
idx = np.random.default_rng(0).permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]

A = np.hstack([tr[FEATURES].to_numpy(), np.ones((len(tr), 1))])
w, *_ = np.linalg.lstsq(A, tr.days_left.to_numpy(), rcond=None)
pred = np.hstack([val[FEATURES].to_numpy(), np.ones((len(val), 1))]) @ w
true = val.days_left.to_numpy()
err = pred - true

print("weights:", dict(zip(FEATURES + ["intercept"], np.round(w, 3))))
print(f"val MAE {np.abs(err).mean():.2f} days; predictions below 0: {(pred < 0).sum()} of {len(pred)}")
for lo, hi in [(0, 20), (20, 100), (100, 200), (200, 600)]:
    m = (true >= lo) & (true < hi)
    print(f"  true days {lo:3d}-{hi:3d}: n={m.sum():3d}  MAE {np.abs(err[m]).mean():6.1f}  mean signed err {err[m].mean():+6.1f}")
for lo, hi in [(3.3, 3.6), (3.6, 3.9), (3.9, 4.3)]:
    m = (val.battery_v >= lo).to_numpy() & (val.battery_v < hi).to_numpy()
    print(f"  battery {lo}-{hi} V: n={m.sum():3d}  MAE {np.abs(err[m]).mean():6.1f}  mean signed err {err[m].mean():+6.1f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
ax.scatter(true, pred, s=6, alpha=0.5)
lim = [min(pred.min(), 0), max(true.max(), pred.max())]
ax.plot(lim, lim, "k--", lw=1, label="perfect")
ax.axhline(0, color="red", lw=0.8)
ax.set_xlabel("true days_left"); ax.set_ylabel("linear prediction"); ax.legend()
ax.set_title("Linear baseline: predicted vs true (validation)")
ax = axes[1]
ax.scatter(val.battery_v, err, s=6, alpha=0.5)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("battery_v"); ax.set_ylabel("prediction - true (days)")
ax.set_title("Linear error against battery_v")
fig.tight_layout()
fig.savefig("analysis/linear_pred_vs_true.png", dpi=110)
