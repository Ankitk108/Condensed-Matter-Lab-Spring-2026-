import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as sc
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
from uncertainties import ufloat

plt.rcParams.update(
  {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 200,
    "axes.labelsize": 11,
  }
)


def get_paths():
  base_path = Path(__file__).parent
  paths = {
    "data": base_path / "data",
    "plots": base_path / "plots",
    "final": base_path / "final",
    "config": base_path / "config.json",
  }
  for name, path in paths.items():
    if name != "config":
      path.mkdir(parents=True, exist_ok=True)
  return paths


def linear_func(x, m, c):
  return m * x + c


def get_fit_metrics(x, y, popt, pcov):
  residuals = y - linear_func(x, *popt)
  ss_res = np.sum(residuals**2)
  ss_tot = np.sum((y - np.mean(y)) ** 2)
  r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

  slope_error = np.sqrt(pcov[0, 0])
  rel_error_pct = (slope_error / abs(popt[0])) * 100

  return r_squared, rel_error_pct


def fit_calibration(paths):
  file_path = paths["data"] / "calibration.csv"
  if not file_path.exists():
    return None

  df = pd.read_csv(file_path)
  x = df["current_A"].to_numpy()
  y = df["field_Gauss"].to_numpy() * 1e-4  # Tesla

  popt, pcov = curve_fit(linear_func, x, y)
  r2, _ = get_fit_metrics(x, y, popt, pcov)

  return (
    ufloat(popt[0], np.sqrt(pcov[0, 0])),
    ufloat(popt[1], np.sqrt(pcov[1, 1])),
    r2,
  )


def analyze_susceptibility(paths, config, m_cal, c_cal):
  results = []

  mu0 = config["mu_0_SI"]
  g = config["g_m_s2"]
  chi_w_mass = config["chi_water_mass_SI"]

  for entry in config["experiments"]:
    file_path = paths["data"] / entry["file_name"]
    if not file_path.exists():
      continue

    df = pd.read_csv(file_path)

    # ---- density from composition ----
    m_solute = entry["solute_mass_g"]          # g
    V_water = entry["water_volume_mL"]         # mL
    rho_water = entry["water_density_g_mL"]    # g/mL

    m_water = rho_water * V_water
    rho_solution = (m_solute + m_water) / V_water   # g/mL
    rho_si = rho_solution * 1000.0                  # kg/m^3

    h0_m = entry["h0_cm"] * 0.01

    B = np.array([(m_cal * i + c_cal).n for i in df["current_A"]])
    B0 = df["B0_Gauss"].to_numpy() * 1e-4

    x_data = B**2 - B0**2
    y_data = np.abs(df["h_observed_cm"].to_numpy() * 0.01 - h0_m)

    popt, pcov = curve_fit(linear_func, x_data, y_data)
    r2, rel_err = get_fit_metrics(x_data, y_data, popt, pcov)

    slope = ufloat(popt[0], np.sqrt(pcov[0, 0]))
    intercept = ufloat(popt[1], np.sqrt(pcov[1, 1]))

    chi_vol = slope * (2 * mu0 * rho_si * g)
    chi_mass = (chi_vol / rho_si) - chi_w_mass

    results.append(
      {
        "file": entry["file_name"],
        "solute_mass_g": m_solute,
        "water_volume_mL": V_water,
        "rho_solution": rho_solution,
        "slope": slope,
        "intercept": intercept,
        "chi_vol": chi_vol,
        "chi_mass": chi_mass,
        "r2": r2,
        "rel_err": rel_err,
        "x": x_data,
        "y_mm": y_data * 1000,
        "fit_line_mm": linear_func(
          x_data,
          popt[0] * 1000,
          popt[1] * 1000,
        ),
      }
    )

  return results


def plot_quincke(paths, res):
  fig, ax = plt.subplots(figsize=(6, 4.5))

  ax.scatter(res["x"], res["y_mm"], color="black", s=20, label="Observed")
  ax.plot(res["x"], res["fit_line_mm"], color="red", label="Linear Fit")

  ax.set_ylabel(r"Displacement $\Delta h$ [mm]")
  ax.set_xlabel(r"$(B^2 - B_0^2)$ [T$^2$]")

  ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
  ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

  info = (
    f"$R^2 = {res['r2']:.4f}$\n"
    f"Slope = {res['slope'].n:.3e} ± {res['slope'].s:.1e}"
  )

  ax.text(
    0.05,
    0.95,
    info,
    transform=ax.transAxes,
    va="top",
    bbox=dict(boxstyle="round", fc="white", alpha=0.8),
  )

  ax.set_title(f"Solution density: {res['rho_solution']:.4f} g/mL")
  ax.legend()
  plt.tight_layout()
  plt.savefig(paths["plots"] / f"plot_{Path(res['file']).stem}.png")
  plt.close()


def main():
  paths = get_paths()

  # ---- correct default config ----
  if not paths["config"].exists():
    with open(paths["config"], "w") as f:
      json.dump(
        {
          "chi_water_mass_SI": -9.0e-9,
          "g_m_s2": sc.g,
          "mu_0_SI": sc.mu_0,
          "experiments": [
            {
              "file_name": "experiment_1.csv",
              "solute_mass_g": 2.5,
              "water_volume_mL": 100.0,
              "water_density_g_mL": 0.997,
              "h0_cm": 10.0
            }
          ],
        },
        f,
        indent=2,
      )

  with open(paths["config"]) as f:
    config = json.load(f)

  cal = fit_calibration(paths)
  if not cal:
    print("Missing calibration.csv")
    return

  m_cal, c_cal, cal_r2 = cal
  results = analyze_susceptibility(paths, config, m_cal, c_cal)

  with open(paths["final"] / "report.txt", "w") as f:
    f.write("QUINCKE'S METHOD – ANALYSIS REPORT\n")
    f.write("=" * 50 + "\n")
    f.write(f"Calibration R²: {cal_r2:.5f}\n\n")

    for r in results:
      plot_quincke(paths, r)
      f.write(f"Dataset: {r['file']}\n")
      f.write(f"Solute mass: {r['solute_mass_g']} g\n")
      f.write(f"Water volume: {r['water_volume_mL']} mL\n")
      f.write(f"Solution density: {r['rho_solution']:.4f} g/mL\n")
      f.write(f"Slope: {r['slope']:.4e} m/T²\n")
      f.write(f"Volume susceptibility: {r['chi_vol']:.4e}\n")
      f.write(f"Mass susceptibility:   {r['chi_mass']:.4e} m³/kg\n")
      f.write(f"R²: {r['r2']:.5f}\n")
      f.write(f"Relative uncertainty: {r['rel_err']:.2f}%\n")
      f.write("-" * 50 + "\n")

  print("Analysis complete.")


if __name__ == "__main__":
  main()
