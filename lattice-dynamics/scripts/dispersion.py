import pandas as pd
import matplotlib.pyplot as plt

def save_data(k, omega, path):
    df = pd.DataFrame({"k": k, "omega": omega})
    df.to_csv(path, index=False)

def plot_monatomic(k, omega):
    plt.figure()
    plt.plot(k, omega)
    plt.xlabel("k")
    plt.ylabel("ω")
    plt.title("Monatomic Lattice Dispersion")
    plt.grid()
    plt.savefig("plots/monatomic_dispersion.png", dpi=300)
    plt.close()

def plot_diatomic(k, omega_a, omega_o):
    plt.figure()
    plt.plot(k, omega_a, label="Acoustic")
    plt.plot(k, omega_o, label="Optical")
    plt.xlabel("k")
    plt.ylabel("ω")
    plt.title("Diatomic Lattice Dispersion")
    plt.legend()
    plt.grid()
    plt.savefig("plots/diatomic_dispersion.png", dpi=300)
    plt.close()
