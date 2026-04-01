import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
  {
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
  }
)

E_CHARGE = 1.602176634e-19


def get_paths() -> dict[str, Path]:
  base = Path(__file__).resolve().parent
  paths = {
    "base": base,
    "data": base / "data",
    "plots": base / "plots",
    "final": base / "final",
    "config": base / "config.json",
  }
  paths["plots"].mkdir(parents=True, exist_ok=True)
  paths["final"].mkdir(parents=True, exist_ok=True)
  return paths


def read_config(config_path: Path) -> dict:
  with open(config_path, "r", encoding="utf-8") as f:
    return json.load(f)


def write_text(path: Path, content: str) -> None:
  with open(path, "w", encoding="utf-8") as f:
    f.write(content)


def compile_latex_if_available(tex_path: Path) -> None:
  if shutil.which("pdflatex") is None:
    return
  result = subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", tex_path.name],
    cwd=tex_path.parent,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
  )
  pdf_path = tex_path.with_suffix(".pdf")
  if result.returncode == 0 and pdf_path.exists() and shutil.which("pdftoppm") is not None:
    subprocess.run(
      ["pdftoppm", "-png", "-singlefile", pdf_path.name, pdf_path.stem],
      cwd=tex_path.parent,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      check=False,
    )
  if result.returncode == 0 and pdf_path.exists():
    for ext in [".aux", ".log"]:
      tex_path.with_suffix(ext).unlink(missing_ok=True)
    tex_path.unlink(missing_ok=True)


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  slope, intercept = np.polyfit(x, y, 1)
  y_fit = slope * x + intercept
  ss_res = float(np.sum((y - y_fit) ** 2))
  ss_tot = float(np.sum((y - np.mean(y)) ** 2))
  r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
  n = len(x)
  if n > 2:
    s2 = ss_res / (n - 2)
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    slope_se = float(np.sqrt(s2 / sxx))
    intercept_se = float(np.sqrt(s2 * (1.0 / n + np.mean(x) ** 2 / sxx)))
  else:
    slope_se = float("nan")
    intercept_se = float("nan")
  return {
    "slope": float(slope),
    "intercept": float(intercept),
    "slope_se": slope_se,
    "intercept_se": intercept_se,
    "r2": r2,
  }


def infer_least_count(series: pd.Series, unit_scale: float = 1.0) -> float:
  max_decimals = 0
  for value in series.astype(float):
    text = format(float(value), ".10f").rstrip("0").rstrip(".")
    if "." in text:
      max_decimals = max(max_decimals, len(text.split(".")[1]))
  return float((10 ** (-max_decimals)) * unit_scale)


def analyze_resistivity(paths: dict[str, Path], config: dict) -> dict:
  df = pd.read_csv(paths["data"] / "resistivity.csv").copy()
  thickness_m = float(config["sample_thickness_mm"]) * 1e-3
  width_m = float(config["sample_width_mm"]) * 1e-3
  spacing_m = float(config["probe_spacing_mm"]) * 1e-3
  geometry_factor = width_m * thickness_m / spacing_m

  df["sr_no"] = np.arange(1, len(df) + 1)
  df["I_A"] = df["I(mA)"].astype(float) * 1e-3
  df["V_V"] = df["V(mV)"].astype(float) * 1e-3
  df["R_ohm"] = df["V_V"] / df["I_A"]
  df["rho_ohm_m"] = df["R_ohm"] * geometry_factor

  fit = linear_fit(df["I_A"].to_numpy(), df["V_V"].to_numpy())
  rho_fit = fit["slope"] * geometry_factor
  rho_fit_se = fit["slope_se"] * geometry_factor

  current_lc_a = infer_least_count(df["I(mA)"], 1e-3)
  voltage_lc_v = infer_least_count(df["V(mV)"], 1e-3)
  current_abs_unc_a = current_lc_a / 2.0
  voltage_abs_unc_v = voltage_lc_v / 2.0

  mean_i = float(df["I_A"].mean())
  mean_v = float(df["V_V"].mean())
  resistance_lc_unc = fit["slope"] * np.sqrt((voltage_abs_unc_v / mean_v) ** 2 + (current_abs_unc_a / mean_i) ** 2)
  rho_lc_unc = resistance_lc_unc * geometry_factor
  rho_total_unc = float(np.sqrt(rho_fit_se**2 + rho_lc_unc**2))

  return {
    "data": df,
    "fit": fit,
    "resistance_ohm": fit["slope"],
    "resistance_ohm_se": fit["slope_se"],
    "resistivity_ohm_m": rho_fit,
    "resistivity_ohm_m_fit_se": rho_fit_se,
    "resistivity_ohm_m_lc_se": rho_lc_unc,
    "resistivity_ohm_m_total_se": rho_total_unc,
    "least_counts": {
      "current_mA": current_lc_a * 1e3,
      "voltage_mV": voltage_lc_v * 1e3,
    },
  }


def analyze_hall(paths: dict[str, Path], config: dict, resistivity: dict) -> dict:
  field_scale = float(config["magnetic_field_column_scale_to_tesla"])
  thickness_m = float(config["sample_thickness_mm"]) * 1e-3

  straight = pd.read_csv(paths["data"] / "straight.csv").sort_values("I(mA)").reset_index(drop=True)
  reverse = pd.read_csv(paths["data"] / "reverse.csv").sort_values("I(mA)").reset_index(drop=True)
  merged = straight.merge(reverse, on="I(mA)", suffixes=("_one", "_reverse"))

  merged["sr_no"] = np.arange(1, len(merged) + 1)
  merged["I_A"] = merged["I(mA)"].astype(float) * 1e-3
  merged["V_one_mV"] = merged["V(mV)_one"].astype(float)
  merged["V_reverse_signed_mV"] = -merged["V(mV)_reverse"].astype(float)
  merged["Vh_mean_mV"] = (merged["V_one_mV"] - merged["V_reverse_signed_mV"]) / 2.0
  merged["Vh_mean_V"] = merged["Vh_mean_mV"] * 1e-3
  merged["B_forward_T"] = merged["B(gauss/10)_one"].astype(float) * field_scale
  merged["B_reverse_T"] = merged["B(gauss/10)_reverse"].astype(float) * field_scale
  merged["B_mean_T"] = (
    (merged["B(gauss/10)_one"].astype(float) + merged["B(gauss/10)_reverse"].astype(float)) / 2.0
  ) * field_scale
  merged["BI_AT"] = merged["I_A"] * merged["B_mean_T"]
  merged["Vh_over_BI"] = merged["Vh_mean_V"] / merged["BI_AT"]

  fit = linear_fit(merged["BI_AT"].to_numpy(), merged["Vh_mean_V"].to_numpy())
  rh_fit = fit["slope"] * thickness_m
  rh_fit_se = fit["slope_se"] * thickness_m

  current_lc_a = infer_least_count(merged["I(mA)"], 1e-3)
  voltage_lc_v = infer_least_count(pd.concat([merged["V_one_mV"], merged["V(mV)_reverse"]]), 1e-3)
  field_lc_t = infer_least_count(
    pd.concat([merged["B(gauss/10)_one"], merged["B(gauss/10)_reverse"]]), field_scale
  )

  current_abs_unc_a = current_lc_a / 2.0
  single_voltage_abs_unc_v = voltage_lc_v / 2.0
  mean_vh_abs_unc_v = single_voltage_abs_unc_v / np.sqrt(2.0)
  field_abs_unc_t = field_lc_t / 2.0

  mean_vh = float(merged["Vh_mean_V"].mean())
  mean_i = float(merged["I_A"].mean())
  mean_b = float(merged["B_mean_T"].mean())
  rh_lc_unc = abs(rh_fit) * np.sqrt(
    (mean_vh_abs_unc_v / mean_vh) ** 2 + (current_abs_unc_a / mean_i) ** 2 + (field_abs_unc_t / mean_b) ** 2
  )
  rh_total_unc = float(np.sqrt(rh_fit_se**2 + rh_lc_unc**2))

  mobility = abs(rh_fit) / resistivity["resistivity_ohm_m"]
  carrier_density = 1.0 / (E_CHARGE * abs(rh_fit))

  return {
    "merged": merged,
    "fit": fit,
    "hall_coefficient_m3_per_c": rh_fit,
    "hall_coefficient_fit_se": rh_fit_se,
    "hall_coefficient_lc_se": rh_lc_unc,
    "hall_coefficient_total_se": rh_total_unc,
    "carrier_density_m3": carrier_density,
    "mobility_m2_per_vs": mobility,
    "least_counts": {
      "current_mA": current_lc_a * 1e3,
      "voltage_mV": voltage_lc_v * 1e3,
      "field_T": field_lc_t,
    },
  }


def plot_resistivity(paths: dict[str, Path], resistivity: dict) -> None:
  df = resistivity["data"]
  fit = resistivity["fit"]
  x = df["I(mA)"].to_numpy()
  y = df["V(mV)"].to_numpy()

  fig, ax = plt.subplots(figsize=(7.0, 4.8))
  ax.scatter(x, y, color="black", marker="s", s=28, label="Measured data")
  x_line = np.linspace(np.min(x), np.max(x), 200)
  y_line = (fit["slope"] * (x_line * 1e-3) + fit["intercept"]) * 1e3
  ax.plot(x_line, y_line, color="firebrick", linewidth=1.6, label="Linear fit")
  ax.set_xlabel(r"Current $I$ (mA)")
  ax.set_ylabel(r"Voltage $V$ (mV)")
  ax.set_title("Resistivity Measurement: Voltage vs Current")
  ax.legend(loc="best")
  fig.tight_layout()
  fig.savefig(paths["plots"] / "resistivity_voltage_vs_current.png", bbox_inches="tight")
  plt.close(fig)


def plot_hall(paths: dict[str, Path], hall: dict) -> None:
  df = hall["merged"]
  fit = hall["fit"]

  fig, ax = plt.subplots(figsize=(7.2, 4.8))
  ax.scatter(df["BI_AT"], df["Vh_mean_mV"], color="black", marker="s", s=28, label="Merged data")
  x_line = np.linspace(df["BI_AT"].min(), df["BI_AT"].max(), 200)
  y_line = (fit["slope"] * x_line + fit["intercept"]) * 1e3
  ax.plot(x_line, y_line, color="firebrick", linewidth=1.8, label="Best-fit line")
  ax.set_xlabel(r"Product $BI$ (A T)")
  ax.set_ylabel(r"Mean Hall Voltage $V_H$ (mV)")
  ax.set_title(r"Hall Effect Measurement: Mean $V_H$ vs $BI$")
  ax.legend(loc="best")
  fig.tight_layout()
  fig.savefig(paths["plots"] / "hall_voltage_vs_bi_merged_fit.png", bbox_inches="tight")
  plt.close(fig)


def generate_hall_observation_table(paths: dict[str, Path], hall: dict) -> None:
  df = hall["merged"]
  lines = [
    r"\documentclass[varwidth=24cm,border=8pt]{standalone}",
    r"\usepackage{booktabs}",
    r"\usepackage{amsmath}",
    r"\usepackage{tabularx}",
    r"\usepackage{array}",
    r"\begin{document}",
    r"\begin{minipage}{23cm}",
    r"\centering",
    r"{\Large \textbf{Hall Observation Table}}\\[0.6em]",
    r"{\small",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\setlength{\tabcolsep}{5pt}",
    r"\begin{tabularx}{\linewidth}{>{\centering\arraybackslash}p{1.1cm} >{\centering\arraybackslash}p{1.6cm} >{\centering\arraybackslash}p{2.4cm} >{\centering\arraybackslash}p{1.8cm} >{\centering\arraybackslash}p{2.4cm} >{\centering\arraybackslash}p{1.8cm} >{\centering\arraybackslash}p{2.3cm} >{\centering\arraybackslash}p{2.3cm}}",
    r"\toprule",
    r"Sr No & Current (mA) & Hall reading in forward field (mV) & $B$ in forward field (T) & Hall reading in reverse field (mV) & $B$ in reverse field (T) & Mean Hall voltage $V_H$ (mV) & $V_H/(BI)$ ($\mathrm{m^2/C}$) \\",
    r"\midrule",
  ]
  for _, row in df.iterrows():
    lines.append(
      f"{int(row['sr_no'])} & {row['I(mA)']:.2f} & {row['V_one_mV']:.1f} & {row['B_forward_T']:.3f} & "
      f"{row['V_reverse_signed_mV']:.1f} & {row['B_reverse_T']:.3f} & {row['Vh_mean_mV']:.3f} & "
      f"{row['Vh_over_BI']:.3f} \\\\"
    )
  mean_ratio = float(df["Vh_over_BI"].mean())
  lines.extend(
    [
      r"\bottomrule",
      r"\end{tabularx}",
      r"}",
      rf"\\[0.6em] Mean $\dfrac{{V_H}}{{BI}} = {mean_ratio:.3f}\ \mathrm{{m^2/C}}$",
      r"\end{minipage}",
      r"\end{document}",
    ]
  )
  tex_path = paths["final"] / "hall_observation_table.tex"
  write_text(tex_path, "\n".join(lines) + "\n")
  compile_latex_if_available(tex_path)


def generate_resistivity_observation_table(paths: dict[str, Path], resistivity: dict) -> None:
  df = resistivity["data"]
  lines = [
    r"\documentclass[varwidth=22cm,border=8pt]{standalone}",
    r"\usepackage{booktabs}",
    r"\usepackage{amsmath}",
    r"\begin{document}",
    r"\begin{minipage}{21cm}",
    r"\centering",
    r"{\Large \textbf{Resistivity Observation Table}}\\[0.6em]",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\begin{tabular}{rrrrr}",
    r"\toprule",
    r"Sr No & Current (mA) & Voltage (mV) & Resistance $R=V/I$ ($\Omega$) & Resistivity $\rho$ ($\Omega\,\mathrm{m}$) \\",
    r"\midrule",
  ]
  for _, row in df.iterrows():
    lines.append(
      f"{int(row['sr_no'])} & {row['I(mA)']:.2f} & {row['V(mV)']:.1f} & {row['R_ohm']:.3f} & {row['rho_ohm_m']:.6e} \\\\"
    )
  lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{minipage}", r"\end{document}"])
  tex_path = paths["final"] / "resistivity_observation_table.tex"
  write_text(tex_path, "\n".join(lines) + "\n")
  compile_latex_if_available(tex_path)


def generate_uncertainty_table(paths: dict[str, Path], resistivity: dict, hall: dict) -> None:
  lines = [
    r"\documentclass[varwidth=20cm,border=8pt]{standalone}",
    r"\usepackage{booktabs}",
    r"\usepackage{amsmath}",
    r"\begin{document}",
    r"\begin{minipage}{19cm}",
    r"\centering",
    r"{\Large \textbf{Measurement Uncertainty Analysis}}\\[0.6em]",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\begin{tabular}{llll}",
    r"\toprule",
    r"Quantity & Least count contribution & Fit contribution & Total uncertainty \\",
    r"\midrule",
    rf"Resistivity $\rho$ & ${resistivity['resistivity_ohm_m_lc_se']:.6e}\ \Omega\,\mathrm{{m}}$ & ${resistivity['resistivity_ohm_m_fit_se']:.6e}\ \Omega\,\mathrm{{m}}$ & ${resistivity['resistivity_ohm_m_total_se']:.6e}\ \Omega\,\mathrm{{m}}$ \\",
    rf"Hall coefficient $R_H$ & ${hall['hall_coefficient_lc_se']:.6e}\ \mathrm{{m^3/C}}$ & ${hall['hall_coefficient_fit_se']:.6e}\ \mathrm{{m^3/C}}$ & ${hall['hall_coefficient_total_se']:.6e}\ \mathrm{{m^3/C}}$ \\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\vspace{0.8em}",
    r"\begin{tabular}{ll}",
    r"\toprule",
    r"Instrument & Least count \\",
    r"\midrule",
    rf"Current meter (Hall) & ${hall['least_counts']['current_mA']:.2f}\ \mathrm{{mA}}$ \\",
    rf"Voltmeter (Hall) & ${hall['least_counts']['voltage_mV']:.1f}\ \mathrm{{mV}}$ \\",
    rf"Gauss meter & ${hall['least_counts']['field_T']:.4f}\ \mathrm{{T}}$ \\",
    rf"Current meter (Resistivity) & ${resistivity['least_counts']['current_mA']:.2f}\ \mathrm{{mA}}$ \\",
    rf"Voltmeter (Resistivity) & ${resistivity['least_counts']['voltage_mV']:.1f}\ \mathrm{{mV}}$ \\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{minipage}",
    r"\end{document}",
  ]
  tex_path = paths["final"] / "uncertainty_analysis_table.tex"
  write_text(tex_path, "\n".join(lines) + "\n")
  compile_latex_if_available(tex_path)


def generate_summary_table(paths: dict[str, Path], resistivity: dict, hall: dict) -> None:
  lines = [
    r"\documentclass[varwidth=18cm,border=8pt]{standalone}",
    r"\usepackage{booktabs}",
    r"\usepackage{amsmath}",
    r"\begin{document}",
    r"\begin{minipage}{17cm}",
    r"\centering",
    r"{\Large \textbf{Hall Effect Results Summary}}\\[0.6em]",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\begin{tabular}{ll}",
    r"\toprule",
    r"Quantity & Value \\",
    r"\midrule",
    rf"Resistance $R$ & ${resistivity['resistance_ohm']:.3f} \pm {resistivity['resistance_ohm_se']:.3f}\ \Omega$ \\",
    rf"Resistivity $\rho$ & ${resistivity['resistivity_ohm_m']:.6e} \pm {resistivity['resistivity_ohm_m_total_se']:.6e}\ \Omega\,\mathrm{{m}}$ \\",
    rf"Hall coefficient $R_H$ & ${hall['hall_coefficient_m3_per_c']:.6e} \pm {hall['hall_coefficient_total_se']:.6e}\ \mathrm{{m^3/C}}$ \\",
    rf"Carrier density $n$ & ${hall['carrier_density_m3']:.6e}\ \mathrm{{m^{{-3}}}}$ \\",
    rf"Mobility $\mu$ & ${hall['mobility_m2_per_vs']:.6e}\ \mathrm{{m^2/(V\,s)}}$ \\",
    rf"Hall fit $R^2$ & ${hall['fit']['r2']:.6f}$ \\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{minipage}",
    r"\end{document}",
  ]
  tex_path = paths["final"] / "hall_effect_summary_table.tex"
  write_text(tex_path, "\n".join(lines) + "\n")
  compile_latex_if_available(tex_path)


def build_report(resistivity: dict, hall: dict) -> str:
  lines = [
    "HALL EFFECT ANALYSIS REPORT",
    "",
    "Observation tables",
    "  1. Merged Hall observation table using forward and reverse readings with mean Hall voltage.",
    "  2. Resistivity observation table with current, voltage, resistance, and resistivity.",
    "",
    "Final results",
    f"  Resistance R = {resistivity['resistance_ohm']:.3f} +/- {resistivity['resistance_ohm_se']:.3f} ohm",
    f"  Resistivity rho = {resistivity['resistivity_ohm_m']:.6e} +/- {resistivity['resistivity_ohm_m_total_se']:.6e} ohm m",
    f"  Hall coefficient R_H = {hall['hall_coefficient_m3_per_c']:.6e} +/- {hall['hall_coefficient_total_se']:.6e} m^3/C",
    f"  Carrier density n = {hall['carrier_density_m3']:.6e} m^-3",
    f"  Mobility mu = {hall['mobility_m2_per_vs']:.6e} m^2/(V s)",
    "",
    "Least counts used",
    f"  Hall current meter = {hall['least_counts']['current_mA']:.2f} mA",
    f"  Hall voltmeter = {hall['least_counts']['voltage_mV']:.1f} mV",
    f"  Gauss meter = {hall['least_counts']['field_T']:.4f} T",
    f"  Resistivity current meter = {resistivity['least_counts']['current_mA']:.2f} mA",
    f"  Resistivity voltmeter = {resistivity['least_counts']['voltage_mV']:.1f} mV",
  ]
  return "\n".join(lines) + "\n"


def main() -> None:
  paths = get_paths()
  config = read_config(paths["config"])
  resistivity = analyze_resistivity(paths, config)
  hall = analyze_hall(paths, config, resistivity)

  plot_resistivity(paths, resistivity)
  plot_hall(paths, hall)
  generate_hall_observation_table(paths, hall)
  generate_resistivity_observation_table(paths, resistivity)
  generate_uncertainty_table(paths, resistivity, hall)
  generate_summary_table(paths, resistivity, hall)
  write_text(paths["final"] / "analysis_report.txt", build_report(resistivity, hall))

  print(f"Saved report to {paths['final'] / 'analysis_report.txt'}")
  print(f"Saved plots to {paths['plots']}")
  print(f"Saved tables to {paths['final']}")


if __name__ == "__main__":
  main()
