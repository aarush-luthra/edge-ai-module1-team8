"""
train.py -- STARTER. Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 7 --hidden 32,32
Writes model/model.pt and model/config.json, which app/main.py loads.

Look for TODO. The training loop is given; the decisions are yours.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=1500)
ap.add_argument("--no-norm", action="store_true", help="skip normalisation (for the Mission 2 comparison)")
ap.add_argument("--seed", type=int, default=0, help="random start of the weights (the split stays fixed)")
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")
# TODO Mission 1: find the sensor glitches (plot every column first!) and write a rule that
# marks them. Say in your report how many you found and why each rule is safe.
# Each glitch is a single sentinel value far outside anything a working sensor reports, so each
# threshold sits in an empty gap and cannot remove a real reading:
#   battery_v = 0.0   -> a running Li-ion doorbell reads ~3.3-4.2 V; 0 V means no reading
#   temp_c    = -127  -> the "sensor disconnected" code of digital temperature sensors
#   wifi_rssi = 0 dBm -> real RSSI is always negative (~-30 next to the router); 0 means no reading
rules = {
    "battery_v <= 0.5": df.battery_v <= 0.5,
    "temp_c <= -50": df.temp_c <= -50,
    "wifi_rssi_dbm >= -10": df.wifi_rssi_dbm >= -10,
}
bad = np.zeros(len(df), dtype=bool)
for name, hit in rules.items():
    print(f"  glitch rule {name:22s} {int(hit.sum()):3d} rows")
    bad |= hit.to_numpy()
print(f"{len(df)} rows, {bad.sum()} glitch rows dropped")
df = df[~bad]

rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)
# TODO Mission 2: normalisation. Why must mu and sd come from the TRAINING rows only?
# Because the validation rows stand in for unseen data: if their values shaped mu and sd, facts
# about the "unseen" data would leak into training and the validation MAE would look too good.
# The server applies these same saved mu and sd to every live reading.
if a.no_norm:
    mu, sd = np.zeros(X.shape[1], np.float32), np.ones(X.shape[1], np.float32)
else:
    mu, sd = X.mean(axis=0), X.std(axis=0)

mae = lambda p, t: float(np.mean(np.abs(p - t)))
print(f"baseline (predict the mean): {mae(np.full_like(yv, y.mean()), yv):.2f} days")
# TODO Mission 1: a straight-line model (linear regression) as the second baseline.
# Use np.linalg.lstsq on [X, 1]. Print its validation MAE the same way as the line above.
# A column of ones lets the fit learn an intercept. Fit on training rows, score on validation rows.
A = np.hstack([X, np.ones((len(X), 1), np.float32)])
w, *_ = np.linalg.lstsq(A, y, rcond=None)
lin_pred = np.hstack([Xv, np.ones((len(Xv), 1), np.float32)]) @ w
print(f"baseline (linear regression): {mae(lin_pred, yv):.2f} days")

torch.manual_seed(a.seed)
net = BatteryNet(hidden)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
Xvt = torch.tensor((Xv - mu) / sd)
for ep in range(a.epochs):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 300 == 0 or ep == a.epochs - 1:
        with torch.no_grad():
            print(f"  epoch {ep:4d}  val MAE {mae(net(Xvt).numpy(), yv):.2f} days")
with torch.no_grad():
    val_mae = mae(net(Xvt).numpy(), yv)
    train_mae = mae(net(Xt).numpy(), y)  # far below val_mae would mean overfitting
print(f"network {hidden}:            {val_mae:.2f} days  (train MAE {train_mae:.2f})")

os.makedirs("model", exist_ok=True)
torch.save(net.state_dict(), "model/model.pt")
json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(),
               features=FEATURES, val_mae=round(val_mae, 2)), open("model/config.json", "w"), indent=2)
print("saved model/model.pt and model/config.json")
