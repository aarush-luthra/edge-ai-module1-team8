"""Mission 2 size sweep: train every width with one and two hidden layers, plot val MAE against params.

    python analysis/sweep.py --team 8
Runs train.py once per size and reads its printed MAEs. Each run overwrites model/, so retrain
your chosen size afterwards. Saves analysis/sweep.csv and analysis/sweep.png.
"""
import argparse
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--epochs", type=int, default=1500)
a = ap.parse_args()


def n_params(hidden):
    sizes = [5, *hidden, 1]
    return sum(i * o + o for i, o in zip(sizes, sizes[1:]))


rows = []
for depth in (1, 2):
    for w in (4, 8, 16, 32, 64):
        hidden = ",".join([str(w)] * depth)
        out = subprocess.run([sys.executable, "train.py", "--team", str(a.team), "--hidden", hidden,
                              "--epochs", str(a.epochs)], capture_output=True, text=True, check=True).stdout
        m = re.search(r"network .*?:\s+([\d.]+) days\s+\(train MAE ([\d.]+)\)", out)
        val, tr = float(m.group(1)), float(m.group(2))
        rows.append(dict(hidden=hidden, depth=depth, width=w, params=n_params([w] * depth),
                         val_mae=val, train_mae=tr))
        print(f"  hidden {hidden:6s} params {rows[-1]['params']:5d}  val {val:6.2f}  train {tr:6.2f}")

df = pd.DataFrame(rows)
df.to_csv("analysis/sweep.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
for depth, g in df.groupby("depth"):
    ax.plot(g.params, g.val_mae, "o-", label=f"{depth} hidden layer{'s' if depth > 1 else ''} (val)")
    for _, r in g.iterrows():
        ax.annotate(str(r.width), (r.params, r.val_mae), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("parameters (log scale)")
ax.set_ylabel("validation MAE (days)")
ax.set_title("Validation MAE against network size (labels = width)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("analysis/sweep.png", dpi=110)
