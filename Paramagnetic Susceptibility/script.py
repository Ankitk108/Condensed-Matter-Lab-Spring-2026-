import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as sc
from scipy.optimize import curve_fit
from uncertainties import ufloat

plt.rcParams.update(
  {
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
  }
)


def get_paths():
  base_path = Path(__file__).parent
  paths = {
    "data": base_path / "data",
    "plots": base_path / "plots",
    "final": base_path / "final",
    "sample": base_path / "sample_data",
    "config": base_path / "config.json",
  }
  for name, path in paths.items():
    if name != "config":
      path.mkdir(parents=True, exist_ok=True)
  return paths


def setup_experiment(paths):
  if not paths["config"].exists():
    config = {
      "chi_water_mass_SI": -9.0e-9,
      "g_m_s2": sc.g,
      "mu_0_SI": sc.mu_0,
      "experiments": [
        {"file_name": "experiment_1.csv", "density_g_mL": 1.15, "h0_cm": 10.0},
        {"file_name": "experiment_2.csv", "density_g_mL": 1.20, "h0_cm": 10.0},
      ],
    }
    with open(paths["config"], "w") as f:
      json.dump(config, f, indent=2)

  calib_path = paths["sample"] / "calibration.csv"
  if not calib_path.exists():
    cal_data = pd.DataFrame(
      {
        "current_A": [],
        "field_G": [],
      }
    )
    cal_data.to_csv(calib_path, index=False)

  exp_path = paths["sample"] / "experiment_1.csv"
  if not exp_path.exists():
    exp_data = pd.DataFrame(
      {
        "current_A": [],
        "h_observed_cm": [],
        "B0_Gauss": [],
      }
    )
    exp_data.to_csv(exp_path, index=False)


def linear_func(x, m, c):
  return m * x + c


def fit_calibration(paths):
  file_path = paths["data"] / "calibration.csv"
  if not file_path.exists():
    print(f"Error: Calibration file not found at {file_path}")
    print("Please copy calibration.csv from sample_data/ to data/ and fill it.")
    return None

  df = pd.read_csv(file_path)
  field_T = df["field_Gauss"].to_numpy() * 1e-4
  popt, pcov = curve_fit(linear_func, df["current_A"], field_T)
  perr = np.sqrt(np.diag(pcov))

  m_u = ufloat(popt[0], perr[0])
  c_u = ufloat(popt[1], perr[1])

  fig, ax = plt.subplots()
  ax.scatter(df["current_A"], df["field_Gauss"].to_numpy() * 1e-4, color="black", marker="o", label="Data")
  x_range = np.linspace(df["current_A"].min(), df["current_A"].max(), 100)
  ax.plot(
    x_range, linear_func(x_range, *popt), "k--", label=f"Fit: $B = {popt[0]:.3f}I + {popt[1]:.3f}$"
  )
  ax.set_xlabel(r"Current $I$ [A]")
  ax.set_ylabel(r"Magnetic Field $B$ [T]")
  ax.legend()
  plt.savefig(paths["plots"] / "calibration_curve.png")
  plt.close()

  return m_u, c_u


def analyze_susceptibility(paths, config, m_cal, c_cal):
  results = []
  mu0 = config["mu_0_SI"]
  g = config["g_m_s2"]
  chi_w_mass = config["chi_water_mass_SI"]

  for entry in config["experiments"]:
    filename = entry["file_name"]
    rho_lab = entry["density_g_mL"]
    rho_si = rho_lab * 1000.0
    h0_m = entry["h0_cm"] * 0.01

    file_path = paths["data"] / filename
    if not file_path.exists():
      print(
        f"Warning: Experiment file '{filename}' listed in config was not found in data/ folder. Skipping."
      )
      continue

    df = pd.read_csv(file_path)
    b_fields = np.array([m_cal * i + c_cal for i in df["current_A"]])
    b_vals = np.array([val.n for val in b_fields])

    b0_t = df["B0_Gauss"].to_numpy() * 1e-4
    h_obs_m = df["h_observed_cm"].to_numpy() * 0.01
    h_diff_m = np.abs(h_obs_m - h0_m)

    x_data = b_vals**2 - b0_t**2
    popt, pcov = curve_fit(linear_func, x_data, h_diff_m)
    slope_u = ufloat(popt[0], np.sqrt(pcov[0, 0]))

    chi_vol = slope_u * (4 * mu0 * rho_si * g)
    chi_mass_total = chi_vol / rho_si
    chi_mass_mn = chi_mass_total - chi_w_mass

    results.append(
      {
        "file": filename,
        "rho_lab": rho_lab,
        "rho_si": rho_si,
        "chi_vol": chi_vol,
        "chi_mass_mn": chi_mass_mn,
        "fit_params": popt,
        "x_data": x_data,
        "y_data": h_diff_m,
      }
    )

  return results


def plot_individual_results(paths, results):
  for res in results:
    fig, ax = plt.subplots()
    x, y = res["x_data"], res["y_data"]
    popt = res["fit_params"]

    ax.scatter(x, y, color="black", marker="s", label="Measured Points")
    x_fit = np.linspace(min(x), max(x), 100)
    ax.plot(x_fit, linear_func(x_fit, *popt), "k-", linewidth=1, label="Linear Fit")

    ax.set_title(f"Quincke Analysis: {res['file']}\n(Density: {res['rho_lab']} g/mL)")
    ax.set_xlabel(r"$(B^2 - B_0^2)$ [$\text{T}^2$]")
    ax.set_ylabel(r"Displacement $h$ [m]")

    stats_text = (
      f"$\\chi_{{vol}} = {res['chi_vol'].n:.3e} \\pm {res['chi_vol'].s:.3e}$\n"
      rf"$\chi_{{mass}} = {res['chi_mass_mn'].n:.3e} \pm {res['chi_mass_mn'].s:.3e}$ m$^3$/kg"
    )
    ax.text(
      0.05,
      0.95,
      stats_text,
      transform=ax.transAxes,
      verticalalignment="top",
      bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    ax.legend(loc="lower right")
    plt.tight_layout()
    clean_name = Path(res["file"]).stem
    plt.savefig(paths["plots"] / f"fit_{clean_name}.png")
    plt.close()


def main():
  paths = get_paths()
  setup_experiment(paths)

  with open(paths["config"]) as f:
    config = json.load(f)

  cal_params = fit_calibration(paths)
  if cal_params is None:
    print("Calibration phase failed. Aborting analysis.")
    return

  m_cal, c_cal = cal_params
  results = analyze_susceptibility(paths, config, m_cal, c_cal)

  if not results:
    print("Error: No experimental data files were successfully processed.")
    print("Check if filenames in config.json match the files in the data/ folder.")
    return

  plot_individual_results(paths, results)
  with (paths["final"] / "report.txt").open("w", encoding="utf-8") as f:
    f.write("Quincke's Method: Paramagnetic Susceptibility Report\n" + "=" * 60 + "\n")
    for r in results:
      f.write(f"Dataset: {r['file']}\n")
      f.write(f"  Input Density: {r['rho_lab']:.4f} g/mL\n")
      f.write(f"  SI Density:    {r['rho_si']:.2f} kg/m^3\n")
      f.write(f"  Volume Susc:   {r['chi_vol']:.4e}\n")
      f.write(f"  Mass Susc Mn:  {r['chi_mass_mn']:.4e} m^3/kg\n")
      f.write("-" * 60 + "\n")

  print(f"Successfully processed {len(results)} datasets.")
  print("Check plots/ for individual graphs and final/report.txt for numerical results.")


if __name__ == "__main__":
  main()
