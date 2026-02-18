import os
import matplotlib.pyplot as plt
from utils import load_data, compute_derivative, find_tc
from plotting import plot_rt, plot_derivative, finalize_plot

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

FILES = {
    "SC1 (22nF)": "SC1_22nF_raw_data.csv",
    "SC2 (39nF)": "SC2_39nF_raw_data.csv",
    "SC3 (62nF)": "SC3_62nF_raw_data.csv",
}

results = {}

for label, filename in FILES.items():
    path = os.path.join(DATA_DIR, filename)
    df = load_data(path)
    T = df["Temperature_C"].values
    V = df["Vdc_V"].values
    V_smooth, dVdT = compute_derivative(T, V)
    Tc = find_tc(T, dVdT)
    results[label] = (T, V_smooth, dVdT, Tc)
    print(f"{label} → Tc ≈ {Tc:.2f} °C")

plt.figure()
for label, (T, V, _, Tc) in results.items():
    plot_rt(T, V, f"{label} (Tc={Tc:.1f}°C)")
finalize_plot("Resistance vs Temperature", "Temperature (°C)", "Voltage (V)")
plt.savefig("R_vs_T.png")
plt.show()

plt.figure()
for label, (T, _, dVdT, Tc) in results.items():
    plot_derivative(T, dVdT, f"{label} (Tc={Tc:.1f}°C)")
finalize_plot("dR/dT vs Temperature", "Temperature (°C)", "dV/dT")
plt.savefig("dR_dT_vs_T.png")
plt.show()
