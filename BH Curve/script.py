import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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


@dataclass
class TraceData:
  time: np.ndarray
  value: np.ndarray
  source: str
  vertical_scale: float
  vertical_offset: float
  file_name: str


def get_paths() -> dict[str, Path]:
  base = Path(__file__).parent
  paths = {
    "base": base,
    "raw": base / "raw",
    "plots": base / "plots",
    "final": base / "final",
    "config": base / "config.json",
  }
  for key in ("plots", "final"):
    paths[key].mkdir(parents=True, exist_ok=True)
  return paths


def load_config(config_path: Path) -> dict[str, Any]:
  with open(config_path, encoding="utf-8") as f:
    return json.load(f)


def read_tek_csv(path: Path) -> TraceData:
  df = pd.read_csv(path, header=None)
  source = str(df.iloc[6, 1]).strip()
  vertical_scale = float(df.iloc[8, 1])
  vertical_offset = float(df.iloc[9, 1])
  data = df.iloc[18:, 3:5].copy()
  data.columns = ["time", "value"]
  data = data.apply(pd.to_numeric, errors="coerce").dropna()
  return TraceData(
    time=data["time"].to_numpy(),
    value=data["value"].to_numpy(),
    source=source,
    vertical_scale=vertical_scale,
    vertical_offset=vertical_offset,
    file_name=path.name,
  )


def pair_traces(raw_dir: Path) -> list[dict[str, TraceData]]:
  files = sorted(raw_dir.glob("TEK*.CSV"))
  if len(files) % 2 != 0:
    raise ValueError("Expected an even number of TEK files for CH1/CH2 pairing.")

  pairs: list[dict[str, TraceData]] = []
  for idx in range(0, len(files), 2):
    first = read_tek_csv(files[idx])
    second = read_tek_csv(files[idx + 1])
    pair = {first.source: first, second.source: second}
    if set(pair) != {"CH1", "CH2"}:
      raise ValueError(
        f"Pairing failed for {files[idx].name} and {files[idx + 1].name}; "
        "expected one CH1 and one CH2 file."
      )
    pairs.append(pair)
  return pairs


def estimate_frequency(time: np.ndarray, value: np.ndarray) -> float:
  centered = value - np.mean(value)
  dt = float(np.median(np.diff(time)))
  spectrum = np.fft.rfft(centered)
  freqs = np.fft.rfftfreq(len(centered), dt)
  valid = (freqs > 1.0) & (freqs < 500.0)
  if not np.any(valid):
    return 50.0
  freq_guess = float(freqs[valid][np.argmax(np.abs(spectrum[valid]))])
  return freq_guess


def trig_design_matrix(time: np.ndarray, omega: float, order: int) -> np.ndarray:
  cols = [np.ones_like(time)]
  for n in range(1, order + 1):
    cols.append(np.sin(n * omega * time))
    cols.append(np.cos(n * omega * time))
  return np.column_stack(cols)


def sine_model(time: np.ndarray, offset: float, amplitude: float, omega: float, phase: float) -> np.ndarray:
  return offset + amplitude * np.sin(omega * time + phase)


def fit_sine_model(time: np.ndarray, value: np.ndarray) -> dict[str, Any]:
  omega_guess = 2.0 * np.pi * estimate_frequency(time, value)
  offset_guess = float(np.mean(value))
  amplitude_guess = float(0.5 * (np.max(value) - np.min(value)))
  phase_guess = 0.0

  popt, _ = curve_fit(
    sine_model,
    time,
    value,
    p0=[offset_guess, amplitude_guess, omega_guess, phase_guess],
    maxfev=20000,
  )
  fitted = sine_model(time, *popt)
  residual = value - fitted
  ss_res = float(np.sum(residual**2))
  ss_tot = float(np.sum((value - np.mean(value)) ** 2))
  r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
  return {
    "offset": float(popt[0]),
    "amplitude": float(popt[1]),
    "omega": float(popt[2]),
    "phase": float(popt[3]),
    "frequency_hz": float(popt[2] / (2.0 * np.pi)),
    "fitted": fitted,
    "r2": r2,
  }


def evaluate_sine_model(
  time: np.ndarray, offset: float, amplitude: float, omega: float, phase: float
) -> np.ndarray:
  return sine_model(time, offset, amplitude, omega, phase)


def sine_equation_string(
  offset: float, amplitude: float, omega: float, phase: float, var_name: str
) -> str:
  return (
    rf"${var_name}(t) = {offset:.4g} {amplitude:+.4g}\sin\!\left({omega:.4g}\,t {phase:+.4g}\right)$"
    "\n"
    + rf"$f = {omega / (2.0 * np.pi):.4g}\,\mathrm{{Hz}}$"
  )


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
  return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def signed_polygon_area(x: np.ndarray, y: np.ndarray) -> float:
  return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def crossings(x: np.ndarray, y: np.ndarray, target: float = 0.0) -> list[float]:
  shifted = y - target
  values: list[float] = []
  for i in range(len(shifted) - 1):
    y1 = shifted[i]
    y2 = shifted[i + 1]
    if y1 == 0:
      values.append(float(x[i]))
      continue
    if y1 * y2 < 0:
      frac = abs(y1) / (abs(y1) + abs(y2))
      values.append(float(x[i] + frac * (x[i + 1] - x[i])))
  return values


def split_by_sign(values: list[float]) -> tuple[list[float], list[float]]:
  pos = [v for v in values if v >= 0]
  neg = [v for v in values if v <= 0]
  return pos, neg


def compute_loop_metrics(h: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
  area = polygon_area(h, b)
  signed_area = signed_polygon_area(h, b)

  b_at_h0 = crossings(b, h, 0.0)
  h_at_b0 = crossings(h, b, 0.0)
  b_pos, b_neg = split_by_sign(b_at_h0)
  h_pos, h_neg = split_by_sign(h_at_b0)

  remanence = None
  if b_pos and b_neg:
    remanence = 0.5 * (np.mean(np.abs(b_pos)) + np.mean(np.abs(b_neg)))

  coercivity = None
  if h_pos and h_neg:
    coercivity = 0.5 * (np.mean(np.abs(h_pos)) + np.mean(np.abs(h_neg)))

  mask = np.abs(h) <= 0.15 * np.max(np.abs(h))
  mu_initial = None
  if np.count_nonzero(mask) >= 5:
    slope, _ = np.polyfit(h[mask], b[mask], 1)
    mu_initial = float(abs(slope))

  return {
    "loop_area_bh": float(area),
    "signed_loop_area_bh": float(signed_area),
    "energy_loss_per_cycle_per_volume": float(area),
    "remanence_B": None if remanence is None else float(remanence),
    "coercivity_H": None if coercivity is None else float(coercivity),
    "initial_slope_dBdH": mu_initial,
    "H_max": float(np.max(h)),
    "H_min": float(np.min(h)),
    "B_max": float(np.max(b)),
    "B_min": float(np.min(b)),
  }


def build_time_grid(time: np.ndarray, cycles: int = 1, samples: int = 4000) -> np.ndarray:
  duration = time.max() - time.min()
  t0 = time.min()
  return np.linspace(t0, t0 + duration * cycles, samples, endpoint=False)


def analyze_pair(
  pair: dict[str, TraceData],
  meta: dict[str, Any],
  config: dict[str, Any],
  blank_slope: float | None,
) -> dict[str, Any]:
  ch1 = pair["CH1"]
  ch2 = pair["CH2"]

  fit_v = fit_sine_model(ch1.time, ch1.value)
  fit_b = fit_sine_model(ch2.time, ch2.value)

  mean_frequency = 0.5 * (fit_v["frequency_hz"] + fit_b["frequency_hz"])
  omega = 2.0 * np.pi * mean_frequency
  grid = build_time_grid(ch1.time)
  vx_fit = evaluate_sine_model(grid, fit_v["offset"], fit_v["amplitude"], omega, fit_v["phase"])
  vy_fit = evaluate_sine_model(grid, fit_b["offset"], fit_b["amplitude"], omega, fit_b["phase"])

  resistance = float(config["instrument"]["series_resistance_ohm"])
  turns = float(config["instrument"]["coil_turns"])
  coil_length = float(config["instrument"]["coil_length_m"])

  current_fit = vx_fit / resistance
  h_fit = turns * vx_fit / (resistance * coil_length)
  b_raw_fit = 0.5 * vy_fit
  b_corrected = b_raw_fit.copy()
  if blank_slope is not None and meta["specimen_key"] != config["blank"]["specimen_key"]:
    b_corrected = b_corrected - blank_slope * h_fit

  metrics = compute_loop_metrics(h_fit, b_corrected)
  display_area_v = polygon_area(vx_fit, vy_fit)

  return {
    "specimen": meta["specimen"],
    "specimen_key": meta["specimen_key"],
    "drive_label": meta["drive_label"],
    "run_index": meta["run_index"],
    "files": {"CH1": ch1.file_name, "CH2": ch2.file_name},
    "frequency_hz": float(mean_frequency),
    "omega_rad_s": float(omega),
    "vx_fit_r2": float(fit_v["r2"]),
    "vy_fit_r2": float(fit_b["r2"]),
    "vx_equation": sine_equation_string(
      fit_v["offset"], fit_v["amplitude"], omega, fit_v["phase"], "V_x"
    ),
    "vy_equation": sine_equation_string(
      fit_b["offset"], fit_b["amplitude"], omega, fit_b["phase"], "V_y"
    ),
    "display_area_v2": float(display_area_v),
    "current_peak_A": float(np.max(np.abs(current_fit))),
    "blank_corrected": bool(
      blank_slope is not None and meta["specimen_key"] != config["blank"]["specimen_key"]
    ),
    "blank_slope_T_per_Am": blank_slope,
    "power_loss_density_W_per_m3": float(
      metrics["energy_loss_per_cycle_per_volume"] * mean_frequency
    ),
    **metrics,
    "time_grid_s": grid.tolist(),
    "vx_fit_V": vx_fit.tolist(),
    "vy_fit_V": vy_fit.tolist(),
    "H_fit_Am": h_fit.tolist(),
    "B_fit_T": b_corrected.tolist(),
  }


def plot_time_fits(results: list[dict[str, Any]], out_dir: Path) -> None:
  for result in results:
    time = np.array(result["time_grid_s"])
    vx = np.array(result["vx_fit_V"])
    vy = np.array(result["vy_fit_V"])

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, constrained_layout=True)
    axes[0].plot(time, vx, color="black")
    axes[0].set_ylabel(r"$V_x~[\mathrm{V}]$")
    axes[0].set_title(
      f"{result['specimen']} ({result['drive_label']}): Current-channel sinusoidal fit"
    )
    axes[0].text(
      0.02,
      0.98,
      result["vx_equation"],
      transform=axes[0].transAxes,
      va="top",
      ha="left",
      fontsize=8,
      bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )

    axes[1].plot(time, vy, color="black")
    axes[1].set_ylabel(r"$V_y~[\mathrm{V}]$")
    axes[1].set_xlabel(r"$t~[\mathrm{s}]$")
    axes[1].set_title("Probe-channel sinusoidal fit")
    axes[1].text(
      0.02,
      0.98,
      result["vy_equation"],
      transform=axes[1].transAxes,
      va="top",
      ha="left",
      fontsize=8,
      bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )

    fig.savefig(out_dir / f"fit_{result['specimen_key']}_{result['drive_label']}.png")
    plt.close(fig)


def plot_bh_loops(results: list[dict[str, Any]], out_dir: Path) -> None:
  specimens = []
  for result in results:
    if result["specimen_key"] not in specimens:
      specimens.append(result["specimen_key"])

  for specimen_key in specimens:
    group = [r for r in results if r["specimen_key"] == specimen_key]
    if not group:
      continue

    fig = plt.figure(figsize=(11.5, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_eq = fig.add_subplot(gs[0, 1])
    for result in group:
      h = np.array(result["H_fit_Am"])
      b = np.array(result["B_fit_T"])
      label = (
        rf"{result['drive_label']}: "
        rf"$A_{{\mathrm{{loop}}}}={result['energy_loss_per_cycle_per_volume']:.1f}\,\mathrm{{J\,m^{{-3}}}}$"
      )
      ax.plot(h, b, linewidth=2.2, label=label)

    ax.set_title(f"{group[0]['specimen']}: $B$-$H$ loops from sinusoidal fits")
    ax.set_xlabel(r"Magnetic field, $H~[\mathrm{A\,m^{-1}}]$")
    ax.set_ylabel(r"Flux density, $B~[\mathrm{T}]$")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.axhline(0.0, color="0.82", linewidth=0.8)
    ax.axvline(0.0, color="0.82", linewidth=0.8)
    eq_text = "\n\n".join(
      [
        "\n".join(
          [
            rf"$\mathbf{{{result['drive_label']}}}$",
            result["vx_equation"],
            result["vy_equation"],
          ]
        )
        for result in group
      ]
    )
    ax_eq.axis("off")
    ax_eq.set_title("Fitted Equations", pad=10)
    ax_eq.text(
      0.02,
      0.98,
      eq_text,
      transform=ax_eq.transAxes,
      va="top",
      ha="left",
      fontsize=10,
      bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.95, "edgecolor": "0.6"},
    )
    fig.savefig(out_dir / f"bh_loop_{specimen_key}.png")
    plt.close(fig)


def plot_comparison(results: list[dict[str, Any]], out_dir: Path) -> None:
  highest_drive = [r for r in results if r["drive_label"] == "C-V3"]
  if not highest_drive:
    return

  fig, ax = plt.subplots(figsize=(6, 5))
  for result in highest_drive:
    h = np.array(result["H_fit_Am"])
    b = np.array(result["B_fit_T"])
    ax.plot(h, b, label=result["specimen"])
  ax.set_xlabel(r"$H$ [A/m]")
  ax.set_ylabel(r"$B$ [T]")
  ax.set_title("B-H loop comparison at highest drive")
  ax.legend()
  fig.tight_layout()
  fig.savefig(out_dir / "bh_comparison_v3.png")
  plt.close(fig)


def write_csv(results: list[dict[str, Any]], out_path: Path) -> None:
  rows = []
  for result in results:
    row = {
      key: value
      for key, value in result.items()
      if not isinstance(value, (list, dict))
    }
    rows.append(row)
  pd.DataFrame(rows).to_csv(out_path, index=False)


def write_tables(results: list[dict[str, Any]], final_dir: Path) -> None:
  important_columns = [
    "specimen",
    "drive_label",
    "frequency_hz",
    "energy_loss_per_cycle_per_volume",
    "power_loss_density_W_per_m3",
    "remanence_B",
    "coercivity_H",
    "initial_slope_dBdH",
    "H_max",
    "B_max",
    "vx_fit_r2",
    "vy_fit_r2",
  ]
  df = pd.DataFrame(results)[important_columns].copy()
  df = df.rename(
    columns={
      "specimen": "Specimen",
      "drive_label": "Drive",
      "frequency_hz": "f_Hz",
      "energy_loss_per_cycle_per_volume": "EnergyLossPerCycle_J_per_m3",
      "power_loss_density_W_per_m3": "PowerLoss_W_per_m3",
      "remanence_B": "Br_T",
      "coercivity_H": "Hc_A_per_m",
      "initial_slope_dBdH": "dBdH_0",
      "H_max": "Hmax_A_per_m",
      "B_max": "Bmax_T",
      "vx_fit_r2": "R2_Vx",
      "vy_fit_r2": "R2_Vy",
    }
  )
  df.to_csv(final_dir / "important_results_table.csv", index=False)

  summary = (
    df[df["Drive"] == "C-V3"]
    .sort_values("EnergyLossPerCycle_J_per_m3", ascending=False)
    .reset_index(drop=True)
  )
  summary.to_csv(final_dir / "comparison_table_v3.csv", index=False)

  md_lines = []
  md_lines.append("| Specimen | Drive | f (Hz) | Energy loss per cycle (J/m^3) | Power loss (W/m^3) | Br (T) | Hc (A/m) | dB/dH near 0 |")
  md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
  for _, row in df.iterrows():
    md_lines.append(
      f"| {row['Specimen']} | {row['Drive']} | {row['f_Hz']:.3f} | "
      f"{row['EnergyLossPerCycle_J_per_m3']:.3f} | {row['PowerLoss_W_per_m3']:.3f} | "
      f"{row['Br_T']:.5f} | {row['Hc_A_per_m']:.3f} | {row['dBdH_0']:.6g} |"
    )
  (final_dir / "important_results_table.md").write_text("\n".join(md_lines), encoding="utf-8")


def format_table_value(value: float, digits: int = 3) -> str:
  if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-3):
    return f"{value:.3e}"
  return f"{value:.{digits}f}"


def write_latex_table(results: list[dict[str, Any]], final_dir: Path) -> Path:
  df = pd.DataFrame(results)[
    [
      "specimen",
      "drive_label",
      "frequency_hz",
      "energy_loss_per_cycle_per_volume",
      "power_loss_density_W_per_m3",
      "remanence_B",
      "coercivity_H",
      "initial_slope_dBdH",
    ]
  ].copy()

  rows = []
  for _, row in df.iterrows():
    rows.append(
      " {} & {} & {} & {} & {} & {} & {} \\\\".format(
        row["specimen"],
        row["drive_label"],
        format_table_value(row["frequency_hz"], 2),
        format_table_value(row["energy_loss_per_cycle_per_volume"], 2),
        format_table_value(row["power_loss_density_W_per_m3"], 2),
        format_table_value(row["remanence_B"], 5),
        format_table_value(row["coercivity_H"], 2),
        format_table_value(row["initial_slope_dBdH"], 3),
      )
    )

  tex = "\n".join(
    [
      r"\documentclass[varwidth=20cm,border=8pt]{standalone}",
      r"\usepackage{booktabs}",
      r"\usepackage{amsmath}",
      r"\begin{document}",
      r"\begin{minipage}{19cm}",
      r"\centering",
      r"{\Large \textbf{Important B-H Curve Results}}\\[0.6em]",
      r"\renewcommand{\arraystretch}{1.2}",
      r"\begin{tabular}{llrrrrrr}",
      r"\toprule",
      r"Specimen & Drive & $f~(\mathrm{Hz})$ & $E_{\mathrm{loss}}~(\mathrm{J\,m^{-3}\,cycle^{-1}})$ & $P_{\mathrm{loss}}~(\mathrm{W\,m^{-3}})$ & $B_r~(\mathrm{T})$ & $H_c~(\mathrm{A\,m^{-1}})$ & $\left.\dfrac{dB}{dH}\right|_{H=0}$ \\",
      r"\midrule",
      *rows,
      r"\bottomrule",
      r"\end{tabular}",
      r"\end{minipage}",
      r"\end{document}",
    ]
  )
  tex_path = final_dir / "important_results_table.tex"
  tex_path.write_text(tex, encoding="utf-8")
  return tex_path


def render_latex_table(tex_path: Path) -> None:
  try:
    subprocess.run(
      [
        "pdflatex",
        "-interaction=nonstopmode",
        "-output-directory",
        str(tex_path.parent),
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
        str(tex_path.with_suffix(".png")),
      ],
      check=True,
      capture_output=True,
    )
  except Exception:
    return


def write_summary(
  results: list[dict[str, Any]],
  blank_slope: float | None,
  out_path: Path,
  config: dict[str, Any],
) -> None:
  lines = []
  lines.append("B-H CURVE ANALYSIS REPORT")
  lines.append("=" * 30)
  lines.append("")
  lines.append("Method")
  lines.append("-" * 6)
  lines.append(
    "Each CH1 and CH2 trace was fitted with a simple sinusoid of the form "
    "V(t) = C + A sin(wt + phi), "
    "converted to H and B using the instrument equations from the manual, and "
    "parameterized through time to form the closed B-H loop."
  )
  lines.append(
    f"H = N*Vx/(R*L), with N={config['instrument']['coil_turns']}, "
    f"R={config['instrument']['series_resistance_ohm']} ohm, "
    f"L={config['instrument']['coil_length_m']} m"
  )
  lines.append("B = 0.5*Vy [tesla]")
  lines.append("")

  if blank_slope is not None:
    lines.append("Blank Correction")
    lines.append("-" * 16)
    lines.append(
      "A linear air-core background B = alpha*H was estimated from the blank run and "
      "subtracted from all specimen runs."
    )
    lines.append(f"alpha = {blank_slope:.6e} T m/A")
    lines.append("")

  grouped: dict[str, list[dict[str, Any]]] = {}
  for result in results:
    grouped.setdefault(result["specimen"], []).append(result)

  for specimen, group in grouped.items():
    lines.append(specimen)
    lines.append("-" * len(specimen))
    for result in group:
      lines.append(
        f"{result['drive_label']}: "
        f"f={result['frequency_hz']:.3f} Hz, "
        f"R2(Vx)={result['vx_fit_r2']:.5f}, "
        f"R2(Vy)={result['vy_fit_r2']:.5f}, "
        f"loop area={result['energy_loss_per_cycle_per_volume']:.6g} J/m^3, "
        f"power density={result['power_loss_density_W_per_m3']:.6g} W/m^3, "
        f"Br={result['remanence_B']:.6g} T, "
        f"Hc={result['coercivity_H']:.6g} A/m, "
        f"dB/dH|0={result['initial_slope_dBdH']:.6g}"
      )
    best = max(group, key=lambda item: item["energy_loss_per_cycle_per_volume"])
    lines.append(
      f"Largest loss for {specimen}: {best['drive_label']} with "
      f"{best['energy_loss_per_cycle_per_volume']:.6g} J/m^3 per cycle."
    )
    lines.append("")

  v3_runs = [r for r in results if r["drive_label"] == "C-V3"]
  if v3_runs:
    lines.append("Material Comparison at Highest Drive")
    lines.append("-" * 33)
    area_rank = sorted(
      v3_runs, key=lambda item: item["energy_loss_per_cycle_per_volume"], reverse=True
    )
    slope_rank = sorted(v3_runs, key=lambda item: item["initial_slope_dBdH"], reverse=True)
    ret_rank = sorted(v3_runs, key=lambda item: item["remanence_B"], reverse=True)
    hc_rank = sorted(v3_runs, key=lambda item: item["coercivity_H"], reverse=True)
    lines.append("Energy-loss ranking: " + " > ".join(r["specimen"] for r in area_rank))
    lines.append("Initial permeability ranking: " + " > ".join(r["specimen"] for r in slope_rank))
    lines.append("Retentivity ranking: " + " > ".join(r["specimen"] for r in ret_rank))
    lines.append("Coercivity ranking: " + " > ".join(r["specimen"] for r in hc_rank))

  out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
  paths = get_paths()
  config = load_config(paths["config"])
  pairs = pair_traces(paths["raw"])

  if len(config["runs"]) != len(pairs):
    raise ValueError(
      f"Config has {len(config['runs'])} runs but raw directory produced {len(pairs)} pairs."
    )

  blank_index = int(config["blank"]["run_index"])
  blank_meta = config["runs"][blank_index]
  blank_pair = pairs[blank_index]
  blank_result = analyze_pair(blank_pair, blank_meta, config, blank_slope=None)

  h_blank = np.array(blank_result["H_fit_Am"])
  b_blank = 0.5 * np.array(blank_result["vy_fit_V"])
  blank_slope, _ = np.polyfit(h_blank, b_blank, 1)

  results = []
  for meta, pair in zip(config["runs"], pairs):
    result = analyze_pair(pair, meta, config, blank_slope=float(blank_slope))
    if meta["specimen_key"] != config["blank"]["specimen_key"]:
      results.append(result)

  plot_time_fits(results, paths["plots"])
  plot_bh_loops(results, paths["plots"])
  plot_comparison(results, paths["plots"])
  write_csv(results, paths["final"] / "bh_analysis_results.csv")
  write_tables(results, paths["final"])
  tex_path = write_latex_table(results, paths["final"])
  render_latex_table(tex_path)
  write_summary(results, float(blank_slope), paths["final"] / "analysis_summary.txt", config)


if __name__ == "__main__":
  main()
