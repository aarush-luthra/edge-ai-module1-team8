"""Mission 3 self-check: glitches refused, edge readings answered, batch == singles, 400 singles vs 1 batch.

    python analysis/check_api.py --url http://localhost:8000
    python analysis/check_api.py --url https://<name>-8000.app.github.dev
"""
import argparse
import time

import numpy as np
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
url = ap.parse_args().url.rstrip("/")
s = requests.Session()
good = {"battery_v": 3.86, "temp_c": 28.0, "rings_per_day": 5, "wifi_rssi_dbm": -66, "age_months": 22}

print("== broken readings (all must be 4xx)")
broken = {
    "battery 0 V": {**good, "battery_v": 0.0},
    "temp -127 C": {**good, "temp_c": -127},
    "temp 85 C (DS18B20 reset)": {**good, "temp_c": 85},
    "rssi 0 dBm": {**good, "wifi_rssi_dbm": 0},
    "negative rings": {**good, "rings_per_day": -3},
    "negative age": {**good, "age_months": -1},
    "missing field": {k: v for k, v in good.items() if k != "temp_c"},
    "text instead of number": {**good, "battery_v": "abc"},
    "null value": {**good, "battery_v": None},
}
ok = True
for name, body in broken.items():
    r = s.post(url + "/predict", json=body)
    ok &= 400 <= r.status_code < 500
    print(f"  {name:28s} -> {r.status_code}")
r = s.post(url + "/predict", data='{"battery_v": NaN, "temp_c": 28, "rings_per_day": 5, "wifi_rssi_dbm": -66, "age_months": 22}',
           headers={"content-type": "application/json"})
ok &= 400 <= r.status_code < 500
print(f"  {'NaN':28s} -> {r.status_code}")
r = s.post(url + "/predict_batch", json={"rows": [good, {**good, "temp_c": -127}]})
ok &= 400 <= r.status_code < 500
print(f"  {'batch with one glitch':28s} -> {r.status_code}")

print("== real readings just outside the training data (all must be 200)")
edge = {
    "battery 3.30 V": {**good, "battery_v": 3.30},
    "battery 4.25 V": {**good, "battery_v": 4.25},
    "temp 0 C": {**good, "temp_c": 0},
    "temp 50 C": {**good, "temp_c": 50},
    "rssi -92 dBm": {**good, "wifi_rssi_dbm": -92},
    "rssi -40 dBm": {**good, "wifi_rssi_dbm": -40},
    "45 rings": {**good, "rings_per_day": 45},
    "age 40 months": {**good, "age_months": 40},
}
for name, body in edge.items():
    r = s.post(url + "/predict", json=body)
    ok &= r.status_code == 200
    print(f"  {name:28s} -> {r.status_code} {r.json().get('days_left') if r.ok else r.text[:80]}")

print("== batch of 50 == 50 single calls")
rng = np.random.default_rng(7)
rows = [{"battery_v": round(rng.uniform(3.35, 4.2), 3), "temp_c": round(rng.uniform(8, 46), 1),
         "rings_per_day": int(rng.integers(0, 30)), "wifi_rssi_dbm": int(rng.integers(-88, -45)),
         "age_months": int(rng.integers(0, 37))} for _ in range(400)]
singles = [s.post(url + "/predict", json=r).json()["days_left"] for r in rows[:50]]
batch = s.post(url + "/predict_batch", json={"rows": rows[:50]}).json()["days_left"]
same = np.allclose(singles, batch, atol=0.011)
ok &= same
print(f"  identical: {same}  (max diff {np.max(np.abs(np.array(singles) - np.array(batch))):.3f} days)")

print("== 400 single /predict calls vs one /predict_batch of 400")
t0 = time.perf_counter()
for r in rows:
    s.post(url + "/predict", json=r)
t_single = time.perf_counter() - t0
t0 = time.perf_counter()
resp = s.post(url + "/predict_batch", json={"rows": rows}).json()
t_batch = time.perf_counter() - t0
print(f"  400 singles: {t_single * 1000:8.1f} ms total, {t_single * 1000 / 400:.2f} ms each")
print(f"  1 batch:     {t_batch * 1000:8.1f} ms total  (server stages: {resp['timings_ms']})")
print(f"  batch is {t_single / t_batch:.0f}x faster")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
