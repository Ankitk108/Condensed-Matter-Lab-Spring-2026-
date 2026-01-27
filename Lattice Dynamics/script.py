import json
import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def get_paths() -> dict[str, Path]:
  base = Path(__file__).parent
  paths = {
    "data": base / "data",
    "plots": base / "plots",
    "final": base / "final",
    "config": base / "config.json",
  }
  for path in paths.values():
    if path.suffix == "":
      path.mkdir(parents=True, exist_ok=True)
  return paths


def 𝜈_mono_theory(theta, L, C):
  omega_sq = (2.0 / (L * C)) * (1.0 - np.cos(theta))
  return np.sqrt(np.maximum(omega_sq, 0)) / (2.0 * np.pi)


def analyze_mono(paths: dict[str, Path], config: dict) -> dict[str, Any] | None:
  data_path = paths["data"] / "mono_readings.csv"
  if not data_path.exists():
    return None
  df = pd.read_csv(data_path)
  f_scale = 1e3 if config["input_units"]["frequency"] == "kHz" else 1.0
  v_exp = df.iloc[:, 0].to_numpy() * f_scale
  theta_exp = np.radians(df.iloc[:, 1].to_numpy()) / config["num_sections"]

  popt, pcov = curve_fit(𝜈_mono_theory, theta_exp, v_exp, p0=[1e-3, 4e-8])
  ss_res = np.sum((v_exp - 𝜈_mono_theory(theta_exp, *popt)) ** 2)
  ss_tot = np.sum((v_exp - np.mean(v_exp)) ** 2)
  r2 = 1.0 - (ss_res / ss_tot)

  L_fit, C_fit = popt
  v_max_theory = (1.0 / np.pi) * np.sqrt(1.0 / (L_fit * C_fit))

  return {
    "theta": theta_exp,
    "freq": v_exp,
    "r2": r2,
    "popt": popt,
    "L": ufloat(L_fit, np.sqrt(pcov[0, 0])),
    "C": ufloat(C_fit, np.sqrt(pcov[1, 1])),
    "max_theory": v_max_theory,
    "max_exp": np.max(v_exp),
  }


def plot_results(paths, mono, di_readings):
  fig1, ax1 = plt.subplots(figsize=(6, 4))
  ax1.scatter(mono["theta"], mono["freq"] / 1e3, c="k", marker="o", label="Observed Data")
  t_grid = np.linspace(min(mono["theta"]), max(mono["theta"]), 200)
  ax1.plot(
    t_grid,
    𝜈_mono_theory(t_grid, *mono["popt"]) / 1e3,
    "k--",
    label=f"Theory Fit (R²={mono['r2']:.4f})",
  )
  ax1.set_title("Monatomic Lattice Dispersion")
  ax1.set_xlabel(r"Phase $\theta$ [rad]")
  ax1.set_ylabel(r"Frequency $\nu$ [kHz]")
  ax1.legend()
  fig1.tight_layout()
  fig1.savefig(paths["plots"] / "mono_dispersion.png")

  if di_readings is not None:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    branch_configs = {
      "acoustic": {"marker": "o", "label": "Acoustic"},
      "optical": {"marker": "^", "label": "Optical"},
    }
    for b, config in branch_configs.items():
      b_df = di_readings[di_readings.iloc[:, 0].str.lower() == b]
      if b_df.empty:
        continue
      t = np.radians(b_df.iloc[:, 2].to_numpy()) / 10.0
      v = b_df.iloc[:, 1].to_numpy()
      idx = np.argsort(t)
      t, v = t[idx], v[idx]
      ax2.scatter(t, v, c="k", marker=config["marker"], label=config["label"])
      ax2.plot(t, v, "k-", alpha=0.8)
    ax2.set_title("Diatomic Lattice Dispersion")
    ax2.set_xlabel(r"Phase $\theta$ [rad]")
    ax2.set_ylabel(r"Frequency $\nu$ [kHz]")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(paths["plots"] / "di_dispersion.png")


def generate_output(paths, cfg, mono, di_df):
  txt_path = paths["final"] / "analysis_summary.txt"
  nom_C1 = cfg["nominal_C1_uF"] * 1e-6

  with open(txt_path, "w") as f:
    f.write("LATTICE DYNAMICS ANALYSIS REPORT\n")
    f.write("=" * 35 + "\n")
    f.write(f"Fitted L: {mono['L'].n:.4e} H\n")
    f.write(f"Fitted C: {mono['C'].n:.4e} F\n")
    f.write(f"Monatomic R^2: {mono['r2']:.4f}\n")
    f.write("-" * 20 + "\n")
    f.write(f"Max Freq (Experimental): {mono['max_exp'] / 1e3:.2f} kHz\n")
    f.write(f"Max Freq (Theoretical): {mono['max_theory'] / 1e3:.2f} kHz\n")
    f.write(f"Nominal C1: {nom_C1:.4e} F\n")

  tex_path = paths["final"] / "data_table.tex"

  # Using raw degrees directly for the table display
  # mono['theta'] was stored as radians, converting back to total degrees for display
  m_deg = np.degrees(mono["theta"] * 10.0)
  m_v = mono["freq"]

  ac = di_df[di_df.iloc[:, 0].str.lower() == "acoustic"] if di_df is not None else pd.DataFrame()
  op = di_df[di_df.iloc[:, 0].str.lower() == "optical"] if di_df is not None else pd.DataFrame()

  a_deg = ac.iloc[:, 2].to_numpy()
  a_v = ac.iloc[:, 1].to_numpy()
  o_deg = op.iloc[:, 2].to_numpy()
  o_v = op.iloc[:, 1].to_numpy()

  mono_list = [(m_deg[i], m_v[i] / 1e3) for i in range(len(m_deg))]
  di_list = [(a_deg[i], a_v[i]) for i in range(len(a_deg))] + [
    (o_deg[i], o_v[i]) for i in range(len(o_deg))
  ]

  max_len = max(len(mono_list), len(di_list))

  content = [
    r"\documentclass[varwidth,border=10pt]{standalone}",
    r"\usepackage{booktabs, array}",
    r"\begin{document}",
    r"\centering \small \textbf{Experimental Data}\\[1ex]",
    r"\begin{tabular}{cc|cc}",
    r"\toprule",
    r"\multicolumn{2}{c}{Monatomic} & \multicolumn{2}{c}{Diatomic} \\",
    r"$\Phi$ [in $^\circ$] & $\nu$ [in kHz] & $\Phi$ [in $^\circ$] & $\nu$ [in kHz] \\",
    r"\midrule",
  ]

  for i in range(max_len):
    m_str = f"{mono_list[i][0]:.1f} & {mono_list[i][1]:.2f}" if i < len(mono_list) else "&"
    d_str = f"{di_list[i][0]:.1f} & {di_list[i][1]:.2f}" if i < len(di_list) else "&"
    content.append(f"{m_str} & {d_str} \\\\")

  content.extend([r"\bottomrule", r"\end{tabular}", r"\end{document}"])

  with open(tex_path, "w") as f:
    f.write("\n".join(content))

  try:
    subprocess.run(
      [
        "pdflatex",
        "-interaction=nonstopmode",
        "-output-directory",
        str(paths["final"]),
        str(tex_path),
      ],
      check=True,
      capture_output=True,
    )

    subprocess.run(
      [
        "magick",
        "-density",
        "300",
        str(tex_path.with_suffix(".pdf")),
        str(paths["final"] / "table_output.png"),
      ],
      check=True,
    )

    for ext in [".aux", ".log", ".pdf", ".tex"]:
      if (tex_path.with_suffix(ext)).exists():
        os.remove(tex_path.with_suffix(ext))
  except:
    pass


def main():
  paths = get_paths()
  with open(paths["config"]) as f:
    cfg = json.load(f)
  mono = analyze_mono(paths, cfg)
  di_df = (
    pd.read_csv(paths["data"] / "di_readings.csv")
    if (paths["data"] / "di_readings.csv").exists()
    else None
  )
  if mono:
    plot_results(paths, mono, di_df)
    generate_output(paths, cfg, mono, di_df)


if __name__ == "__main__":
  main()
