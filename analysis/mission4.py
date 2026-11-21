"""Mission 4: would the model fit inside the doorbell (ESP32-S3)?

    python analysis/mission4.py          (run from the top folder)
Uses the SAME layer_table() function as the server's /model_info, so the numbers always agree.
Saves analysis/mission4_table.csv.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import pandas as pd

from app.main import SRAM_BUDGET, layer_table, net
from app.model import BatteryNet

PSRAM = 8 * 1024 * 1024
CLOCK_HZ = 240e6  # ESP32-S3 maximum CPU clock (Espressif ESP32-S3 datasheet)


def peak_activations(model):
    """Most values alive at once: one Linear layer's input and output are held together."""
    return max(l["in"] + l["out"] for l in layer_table(model)["layers"])


def row(hidden):
    m = BatteryNet(hidden)
    t = layer_table(m)
    act = peak_activations(m)
    return {"hidden": ",".join(map(str, hidden)), "params": t["total_params"],
            "f32_bytes": t["total_bytes_f32"], "int8_bytes": t["total_bytes_int8"],
            "fits_sram_f32": t["fits_300kb_sram_f32"], "fits_sram_int8": t["fits_300kb_sram_int8"],
            "fits_psram_f32": t["total_bytes_f32"] <= PSRAM, "fits_psram_int8": t["total_bytes_int8"] <= PSRAM,
            "macs": t["total_macs"], "peak_act_values": act, "peak_act_f32_bytes": 4 * act}


# 1. Our deployed model
t = layer_table(net)
act = peak_activations(net)
print("== Our model:", " -> ".join([str(t["layers"][0]["in"])] + [str(l["out"]) for l in t["layers"]]))
print(f"  weights+biases: {t['total_params']} params = {t['total_bytes_f32']} B float32, {t['total_bytes_int8']} B int8")
print(f"  peak activations: {act} values = {4 * act} B float32, {act} B int8")
print(f"  share of the 300 KB budget (f32 weights + f32 activations): "
      f"{100 * (t['total_bytes_f32'] + 4 * act) / SRAM_BUDGET:.1f}%")

# 2. Scaling table for 5 -> w -> w -> 1
df = pd.DataFrame([row((w, w)) for w in (32, 128, 256, 512, 2048)])
df.to_csv("analysis/mission4_table.csv", index=False)
print("\n== 5 -> w -> w -> 1")
print(df.drop(columns=["peak_act_values"]).to_string(index=False))


# 3. Exact width where the weights stop fitting in 300 KB
def widest(fits):
    w = 1
    while fits(layer_table(BatteryNet((w + 1, w + 1)))):
        w += 1
    return w


for label, budget in [("300 KB = 300*1024 B", SRAM_BUDGET), ("300 KB = 300,000 B", 300_000)]:
    wf = widest(lambda t: t["total_bytes_f32"] <= budget)
    wi = widest(lambda t: t["total_bytes_int8"] <= budget)
    print(f"\n== Largest width that fits ({label})")
    for name, w, key in [("float32", wf, "total_bytes_f32"), ("int8", wi, "total_bytes_int8")]:
        a, b = layer_table(BatteryNet((w, w)))[key], layer_table(BatteryNet((w + 1, w + 1)))[key]
        print(f"  {name:7s}: w = {w} fits ({a:,} B); w = {w + 1} does not ({b:,} B)")

# 4. Time per prediction on the board: MACs / (MACs per second), for a range of assumptions
print("\n== Estimated time per prediction at 240 MHz")
cases = [("float32, plain C loop (~10 cycles/MAC)", 10), ("float32, tight FPU loop (~2 cycles/MAC)", 2),
         ("int8, SIMD via ESP-NN (~0.25 cycles/MAC)", 0.25)]
for hidden in [tuple(t_["out"] for t_ in t["layers"][:-1]), (2048, 2048)]:
    macs = layer_table(BatteryNet(hidden))["total_macs"]
    print(f"  {'->'.join(map(str, hidden))} ({macs:,} MACs)")
    for name, cpm in cases:
        print(f"    {name:42s} {macs * cpm / CLOCK_HZ * 1e6:10.1f} us")
