from scripts.utils import load_config, generate_k
from scripts.monatomic import monatomic_dispersion
from scripts.diatomic import diatomic_dispersion
from scripts.dispersion import save_data, plot_monatomic, plot_diatomic
import pandas as pd

config = load_config()

a = config["lattice"]["a"]
points = config["simulation"]["k_points"]

K = config["monatomic"]["spring"]
m = config["monatomic"]["mass"]

m1 = config["diatomic"]["mass1"]
m2 = config["diatomic"]["mass2"]

k = generate_k(a, points)

# Monatomic
omega_mono = monatomic_dispersion(k*a, K, m)
save_data(k, omega_mono, "data/simulated_monatomic.csv")
plot_monatomic(k, omega_mono)

# Diatomic
omega_a, omega_o = diatomic_dispersion(k*a, K, m1, m2)
save_data(k, omega_a, "data/simulated_acoustic.csv")
save_data(k, omega_o, "data/simulated_optical.csv")
plot_diatomic(k, omega_a, omega_o)

# Load experimental data
exp = pd.read_csv("data/experimental_data.csv")
print("\\nExperimental data:")
print(exp)

print("\\nSimulation complete.")
print("Plots saved in plots/")
