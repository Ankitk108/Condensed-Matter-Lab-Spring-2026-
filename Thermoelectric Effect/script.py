import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as sc
from scipy.optimize import curve_fit
from scipy.stats import t as student_t
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


def get_paths() -> dict[str, Path]:
  """Initializes the directory structure for the experiment."""
  base = Path(__file__).parent
  paths = {
    "data": base / "data",
    "plots": base / "plots",
    "final": base / "final",
    "sample": base / "sample_data",
    "config": base / "config.json",
  }
  for name, path in paths.items():
    if name != "config":
      path.mkdir(parents=True, exist_ok=True)
  return paths


def setup_experiment(paths: dict[str, Path]):
  """Sets up default configuration and sample data files."""
  if not paths["config"].exists():
    config = {
      "t_ref_celsius": 0.0,
      "peltier_calc_temp_celsius": 25.0,
      "thermocouple_type": "K (Chromel-Alumel)",
      "input_units": {"temperature": "C", "voltage": "mV"},
    }
    with open(paths["config"], "w", encoding="utf-8") as f:
      json.dump(config, f, indent=2)

  sample_path = paths["sample"] / "readings.csv"
  if not sample_path.exists():
    pd.DataFrame({"temp_celsius": [], "t0_celsius": [], "emf_millivolts": []}).to_csv(
      sample_path, index=False
    )


def linear_model(𝓍, 𝓂, 𝒸):
  """E = α * ΔT + c"""
  return 𝓂 * 𝓍 + 𝒸


def analyze_thermoelectric(paths: dict[str, Path], config: dict):
  """Performs data analysis, unit conversion, and curve fitting."""
  data_file = paths["data"] / "readings.csv"
  if not data_file.exists():
    print(f"Data file missing at {data_file}. Please copy sample data to begin.")
    return None

  df = pd.read_csv(data_file)
  if len(df) < 3:
    print("Insufficient data. Please record at least 3 temperature steps.")
    return None

  v_factor = 1e-3 if config["input_units"]["voltage"] == "mV" else 1.0

  if "temp_celsius" in df.columns and "emf_millivolts" in df.columns:
    𝒯_raw = df["temp_celsius"].to_numpy()
    ℰ_raw = df["emf_millivolts"].to_numpy()
  else:
    𝒯_raw = df.iloc[:, 0].to_numpy()
    ℰ_raw = df.iloc[:, 1].to_numpy()

  if "t0_celsius" in df.columns:
    𝒯0_raw = df["t0_celsius"].to_numpy()
    Δ𝒯 = 𝒯_raw - 𝒯0_raw
  else:
    t_ref = config["t_ref_celsius"]
    𝒯0_raw = np.full_like(𝒯_raw, t_ref, dtype=float)
    Δ𝒯 = 𝒯_raw - t_ref

  ℰ_si = ℰ_raw * v_factor

  popt, pcov = curve_fit(linear_model, Δ𝒯, ℰ_si)
  perr = np.sqrt(np.diag(pcov))

  n = len(Δ𝒯)
  p = len(popt)
  dof = n - p

  ℰ_fit = linear_model(Δ𝒯, *popt)
  residuals = ℰ_si - ℰ_fit
  ss_res = float(np.sum(residuals**2))
  ss_tot = float(np.sum((ℰ_si - np.mean(ℰ_si)) ** 2))
  r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
  adj_r2 = float("nan") if (dof <= 0 or ss_tot == 0.0) else 1.0 - (1.0 - r2) * (n - 1) / dof

  t_stats = np.full(p, np.nan, dtype=float)
  p_values = np.full(p, np.nan, dtype=float)
  if dof > 0:
    with np.errstate(divide="ignore", invalid="ignore"):
      t_stats = popt / perr
    p_values = 2.0 * student_t.sf(np.abs(t_stats), dof)

  α_μ = ufloat(popt[0], perr[0])
  𝒸_μ = ufloat(popt[1], perr[1])

  𝒯_abs_k = config["peltier_calc_temp_celsius"] + sc.zero_Celsius
  π_μ = α_μ * 𝒯_abs_k

  return {
    "delta_t": Δ𝒯,
    "t0_celsius": 𝒯0_raw,
    "emf_v": ℰ_si,
    "popt": popt,
    "α": α_μ,
    "𝒸": 𝒸_μ,
    "π": π_μ,
    "t_eval_k": 𝒯_abs_k,
    "r2": r2,
    "adj_r2": adj_r2,
    "p_slope": float(p_values[0]) if len(p_values) > 0 else float("nan"),
    "p_intercept": float(p_values[1]) if len(p_values) > 1 else float("nan"),
  }


def generate_latex_table(paths: dict[str, Path], res: dict):
  output_dir = paths["final"]
  tex_file = output_dir / "detailed_data_table.tex"
  df = pd.DataFrame(
    {
      "T": res["t0_celsius"] + res["delta_t"],
      "T0": res["t0_celsius"],
      "DT": res["delta_t"],
      "EMF": res["emf_v"] * 1000,
    }
  )

  n = len(df)
  half = (n + 1) // 2
  df1 = df.iloc[:half].reset_index(drop=True)
  df2 = df.iloc[half:].reset_index(drop=True)

  latex_code = [
    r"\documentclass{standalone}",
    r"\usepackage{booktabs}",
    r"\usepackage{siunitx}",
    r"\sisetup{round-mode=figures, round-precision=4, table-format=3.3}",
    r"\begin{document}",
    r"\begin{tabular}{SSSS @{\hspace{1cm}} SSSS}",
    r"\toprule",
    r"{$T$ (\unit{\celsius})} & {$T_0$ (\unit{\celsius})} & {$\Delta T$ (\unit{\kelvin})} & {$E$ (mV)} & "
    r"{$T$ (\unit{\celsius})} & {$T_0$ (\unit{\celsius})} & {$\Delta T$ (\unit{\kelvin})} & {$E$ (mV)} \\",
    r"\midrule",
  ]

  for i in range(half):
    r1 = df1.iloc[i]
    left_cols = f"{r1['T']:.2f} & {r1['T0']:.2f} & {r1['DT']:.2f} & {r1['EMF']:.3f}"

    if i < len(df2):
      r2 = df2.iloc[i]
      right_cols = f"{r2['T']:.2f} & {r2['T0']:.2f} & {r2['DT']:.2f} & {r2['EMF']:.3f}"
    else:
      right_cols = r"{} & {} & {} & {}"

    latex_code.append(f"{left_cols} & {right_cols} \\\\")

  latex_code.extend([r"\bottomrule", r"\end{tabular}", r"\end{document}"])

  with open(tex_file, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_code))

  if shutil.which("pdflatex"):
    subprocess.run(
      ["pdflatex", "-interaction=nonstopmode", tex_file.name],
      cwd=output_dir,
      stdout=subprocess.DEVNULL,
    )
    for ext in [".aux", ".log"]:
      (output_dir / tex_file.stem).with_suffix(ext).unlink(missing_ok=True)
    print(f"Detailed PDF table generated: {output_dir / 'detailed_data_table.pdf'}")


def plot_seebeck_effect(paths: dict[str, Path], res: dict):
  """Plots data and fit with directional axis labels and statistical summary."""
  fig, ax = plt.subplots(figsize=(8, 5))

  ax.scatter(res["delta_t"], res["emf_v"] * 1e3, color="black", marker="s", s=25, label="Measured Data")

  𝓍_fit = np.linspace(min(res["delta_t"]), max(res["delta_t"]), 100)
  𝓎_fit = linear_model(𝓍_fit, *res["popt"]) * 1e3

  fit_label = rf"$E = ({res['popt'][0] * 1e3:.3e})\Delta T + ({res['popt'][1] * 1e3:.3e})$"
  ax.plot(𝓍_fit, 𝓎_fit, "r-", linewidth=1.5, label=f"Linear Fit: {fit_label}")

  ax.set_xlabel(r"Temperature Difference $\Delta T$ [$^\circ$C] $\longrightarrow$")
  ax.set_ylabel(r"Thermoelectric EMF $E$ [mV] $\longrightarrow$")
  ax.set_title("Thermoelectric Effect: Determination of $\\alpha$ and $\\pi$")

  α, 𝒸, π = res["α"], res["𝒸"], res["π"]
  stats = (
    rf"$\alpha = {α.n:.3e} \pm {α.s:.3e}$ V/K"
    "\n"
    rf"$c = {𝒸.n:.3e} \pm {𝒸.s:.3e}$ V"
    "\n"
    rf"$\pi_{{{res['t_eval_k']:.1f}K}} = {π.n:.3e} \pm {π.s:.3e}$ V"
    "\n"
    rf"$R^2 = {res['r2']:.4f}$"
  )
  ax.text(
    0.05,
    0.95,
    stats,
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
  )

  ax.legend(loc="lower right")
  plt.tight_layout()
  plt.savefig(paths["plots"] / "seebeck_analysis.png")
  plt.close()


def main():
  """Main control flow."""
  paths = get_paths()
  setup_experiment(paths)

  with open(paths["config"], encoding="utf-8") as f:
    config = json.load(f)

  results = analyze_thermoelectric(paths, config)

  if results:
    plot_seebeck_effect(paths, results)
    generate_latex_table(paths, results)

    with (paths["final"] / "analysis_report.txt").open("w", encoding="utf-8") as f:
      f.write("Thermoelectric Effect Final Report\n" + "=" * 40 + "\n")
      f.write(f"Thermocouple Type:   {config['thermocouple_type']}\n")
      f.write(f"Seebeck Coeff (α):   {results['α']:.4e} V/K\n")
      f.write(f"Systematic Offset (c): {results['𝒸']:.4e} V\n")
      f.write(f"Peltier Coeff (π):    {results['π']:.4e} V (at {results['t_eval_k']:.1f} K)\n")
      f.write(f"R^2:                 {results['r2']:.6f}\n")
      f.write(f"Adjusted R^2:        {results['adj_r2']:.6f}\n")
      f.write(f"p-value (slope m):   {results['p_slope']:.6e}\n")
      f.write(f"p-value (intercept): {results['p_intercept']:.6e}\n")
      f.write("-" * 40 + "\n")

    print(f"Seebeck Coefficient: {results['α']:.3e} V/K")
    print(f"R^2: {results['r2']:.6f} | Adjusted R^2: {results['adj_r2']:.6f}")
    print(
      f"p-value slope: {results['p_slope']:.6e} | p-value intercept: {results['p_intercept']:.6e}"
    )


if __name__ == "__main__":
  main()
