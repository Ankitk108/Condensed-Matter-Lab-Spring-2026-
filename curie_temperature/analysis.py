import os
import json
import re
import shutil
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from config import (
    SMOOTH_WINDOW,
    SMOOTH_POLYORDER,
    TEMP_LEAST_COUNT_C,
    VOLTAGE_LEAST_COUNT_V,
    MC_SAMPLES,
)
from utils import (
    load_data,
    apply_vtotal_method,
    compute_capacitance,
    compute_transition_temperature,
)
from plotting import start_figure, plot_rt, plot_derivative, finalize_plot, plot_vtotal_fit

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

FILES = {
    "SC1 (22 nF)": "SC1_22nF_raw_data.csv",
    "SC2 (39 nF)": "SC2_39nF_raw_data.csv",
    "SC3 (62 nF)": "SC3_62nF_raw_data.csv",
}

results = {}
vtotal_fit_points = {}
vtotal_fit_values = {}
transition_results = {}
summary_rows = []
vtotal_rows = []


def slugify(label):
    return re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_").lower()


def clear_output_dir(folder):
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            os.remove(path)


def cleanup_legacy_root_outputs():
    legacy_files = [
        "epsilon_r_vs_T_all.png",
        "d_epsilon_r_dT_vs_T_all.png",
        "vtotal_constant_fit.png",
        "transition_temperature_report.txt",
    ]
    for filename in legacy_files:
        legacy_path = os.path.join(BASE_DIR, filename)
        if os.path.exists(legacy_path):
            os.remove(legacy_path)


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def rename_columns(df, pretty_map):
    out = df.copy()
    out.columns = [pretty_map.get(c, c.replace("_", " ")) for c in out.columns]
    return out


def prettify_columns(df):
    pretty_map = {
        "Temperature_C": "Temperature (degC)",
        "Vdc_V": "Vdc (V)",
        "Vsc_V": "Vsc (V)",
        "C": "C = Vsc / Vdc",
        "epsilon_r": "Relative permittivity",
        "epsilon_r_smooth": "Relative permittivity (smoothed)",
        "d_epsilon_r_dT": "d(relative permittivity)/dT (1/degC)",
        "sample": "Sample",
        "Tc_peak_degC": "Tc from permittivity peak (degC)",
        "Tc_derivative_degC": "Tc from max d(permittivity)/dT (degC)",
        "Tc_combined_degC": "Combined Tc (degC)",
        "Tc_peak_unc_degC": "Uncertainty in Tc from permittivity peak (degC)",
        "Tc_derivative_unc_degC": "Uncertainty in Tc from max d(permittivity)/dT (degC)",
        "Tc_combined_unc_degC": "Combined Tc uncertainty (degC)",
        "Tc_combined_unc_pct": "Combined uncertainty (%)",
        "Tc_method_difference_degC": "|Tc(permittivity peak) - Tc(max d(permittivity)/dT)| (degC)",
        "epsilon_noise_std": "Std dev of permittivity smoothing residual",
        "epsilon_instr_unc_mean": "Mean propagated permittivity uncertainty",
        "vtotal_fit_V": "Vtotal fit (V)",
        "C_ref": "Reference C",
        "n_points_total": "Total data points",
        "n_points_vtotal_fit": "Points used in Vtotal fit",
        "Tc_peak_mc_std": "MC std of Tc from permittivity peak (degC)",
        "Tc_derivative_mc_std": "MC std of Tc from max d(permittivity)/dT (degC)",
    }
    return rename_columns(df, pretty_map)


def latexify_columns(df):
    pretty_map = {
        "Temperature_C": r"Temperature ($^\circ$C)",
        "Vdc_V": r"$V_{\mathrm{dc}}$ (V)",
        "Vsc_V": r"$V_{\mathrm{sc}}$ (V)",
        "C": r"$C = V_{\mathrm{sc}}/V_{\mathrm{dc}}$",
        "epsilon_r": r"$\epsilon_r$",
        "epsilon_r_smooth": r"$\epsilon_r$ (smoothed)",
        "d_epsilon_r_dT": r"$d\epsilon_r/dT$ (1/$^\circ$C)",
        "sample": "Sample",
        "Tc_peak_degC": r"$T_c$ from peak ($^\circ$C)",
        "Tc_derivative_degC": r"$T_c$ from deriv. ($^\circ$C)",
        "Tc_combined_degC": r"Combined $T_c$ ($^\circ$C)",
        "Tc_peak_unc_degC": r"Peak unc. ($^\circ$C)",
        "Tc_derivative_unc_degC": r"Deriv. unc. ($^\circ$C)",
        "Tc_combined_unc_degC": r"Combined unc. ($^\circ$C)",
        "Tc_combined_unc_pct": r"Combined unc. (\%)",
        "Tc_method_difference_degC": r"Method diff. ($^\circ$C)",
        "epsilon_noise_std": r"Noise std.",
        "epsilon_instr_unc_mean": r"Mean instr. unc.",
        "vtotal_fit_V": r"$V_{\mathrm{total}}$ fit (V)",
        "C_ref": r"Reference $C$",
        "n_points_total": r"Total data points",
        "n_points_vtotal_fit": r"Points used in $V_{\mathrm{total}}$ fit",
        "Tc_peak_mc_std": r"MC std. peak ($^\circ$C)",
        "Tc_derivative_mc_std": r"MC std. deriv. ($^\circ$C)",
    }
    return rename_columns(df, pretty_map)


def latex_escape(text):
    text = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def format_latex_cell(value):
    if isinstance(value, str):
        return latex_escape(value)
    return latex_escape(f"{value}")


def build_table_document(df, title, a4_layout=False):
    pretty_df = latexify_columns(df)
    n_cols = len(pretty_df.columns)
    col_spec = "l" + "c" * (n_cols - 1)
    header = " & ".join(pretty_df.columns) + r" \\"
    rows = [" & ".join(format_latex_cell(value) for value in row) + r" \\" for row in pretty_df.to_numpy()]
    title_tex = latex_escape(title)
    is_data_table = a4_layout
    document_class = r"\documentclass[10pt]{article}" if is_data_table else r"\documentclass[varwidth=80cm,border=8pt]{standalone}"
    geometry_line = r"\usepackage[a4paper,landscape,margin=0.45in]{geometry}" if is_data_table else None
    font_size_line = r"\footnotesize" if is_data_table else r"\scriptsize"
    resize_line = r"\resizebox{!}{0.86\textheight}{%" if is_data_table else None
    lines = [
        document_class,
        r"\usepackage{booktabs,array,amsmath,graphicx}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        *( [geometry_line] if geometry_line else [] ),
        *( [r"\pagestyle{empty}"] if is_data_table else [] ),
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.88}",
        r"\begin{document}",
        r"\centering",
        rf"{{\large\bfseries {title_tex}\par}}",
        r"\vspace{0.4em}",
        font_size_line,
        *( [resize_line] if resize_line else [] ),
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        *( [r"}"] if is_data_table else [] ),
        r"\end{document}",
    ]
    return "\n".join(lines) + "\n"


def build_a4_wrapper_document(title, embedded_pdf_name):
    title_tex = latex_escape(title)
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,landscape,margin=0.45in]{geometry}",
        r"\usepackage{graphicx}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        r"\centering",
        rf"{{\large\bfseries {title_tex}\par}}",
        r"\vspace{0.4em}",
        rf"\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{{{embedded_pdf_name}}}",
        r"\end{document}",
    ]
    return "\n".join(lines) + "\n"


def compile_latex_to_pdf(tex_path):
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", os.path.basename(tex_path)],
        cwd=os.path.dirname(tex_path),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def convert_pdf_to_png(pdf_path):
    stem = os.path.splitext(pdf_path)[0]
    if shutil.which("pdftocairo"):
        subprocess.run(
            ["pdftocairo", "-png", "-r", "300", "-singlefile", pdf_path, stem],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        return
    subprocess.run(
        ["pdftoppm", "-png", "-r", "300", "-singlefile", pdf_path, stem],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def convert_pdf_to_svg(pdf_path):
    stem = os.path.splitext(pdf_path)[0]
    svg_path = f"{stem}.svg"
    if shutil.which("pdftocairo"):
        subprocess.run(
            ["pdftocairo", "-svg", pdf_path, svg_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        return
    raise RuntimeError("SVG export requires pdftocairo.")


def cleanup_latex_artifacts(directory, stem):
    for suffix in (".tex", ".aux", ".log"):
        remove_if_exists(os.path.join(directory, f"{stem}{suffix}"))
    remove_if_exists(os.path.join(directory, stem))


def save_table_outputs(df, title, stem):
    tex_path = os.path.join(TABLES_DIR, f"{stem}.tex")
    pdf_path = os.path.join(TABLES_DIR, f"{stem}.pdf")
    png_path = os.path.join(TABLES_DIR, f"{stem}.png")
    svg_path = os.path.join(TABLES_DIR, f"{stem}.svg")
    remove_if_exists(pdf_path)
    remove_if_exists(png_path)
    remove_if_exists(svg_path)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(build_table_document(df, title, a4_layout=False))
    try:
        compile_latex_to_pdf(tex_path)
        convert_pdf_to_png(pdf_path)
        convert_pdf_to_svg(pdf_path)
    finally:
        cleanup_latex_artifacts(TABLES_DIR, stem)


def save_plot_outputs(stem):
    plt.savefig(os.path.join(PLOTS_DIR, f"{stem}.png"))
    plt.savefig(os.path.join(PLOTS_DIR, f"{stem}.pdf"))


clear_output_dir(RESULTS_DIR)
clear_output_dir(PLOTS_DIR)
clear_output_dir(TABLES_DIR)
cleanup_legacy_root_outputs()

for label, filename in FILES.items():

    path = os.path.join(DATA_DIR, filename)

    df = load_data(path)

    df, V_total_fit, valid = apply_vtotal_method(df)
    df, C_ref = compute_capacitance(df)
    transition = compute_transition_temperature(
        df,
        smooth_window=SMOOTH_WINDOW,
        smooth_polyorder=SMOOTH_POLYORDER,
        temp_least_count_c=TEMP_LEAST_COUNT_C,
        voltage_least_count_v=VOLTAGE_LEAST_COUNT_V,
        mc_samples=MC_SAMPLES,
    )

    results[label] = df
    vtotal_fit_points[label] = valid
    vtotal_fit_values[label] = V_total_fit
    transition_results[label] = transition
    vtotal_rows.append(
        {
            "sample": label,
            "vtotal_fit_V": float(V_total_fit),
            "C_ref": float(C_ref),
            "n_points_total": int(len(df)),
            "n_points_vtotal_fit": int(len(valid)),
        }
    )

    sample_slug = slugify(label)
    processed_df = df.sort_values("Temperature_C").reset_index(drop=True).copy()
    processed_df["d_epsilon_r_dT"] = transition["d_eps_dT"]
    processed_df = processed_df[
        ["Temperature_C", "Vdc_V", "Vsc_V", "C", "epsilon_r", "d_epsilon_r_dT"]
    ].copy()
    numeric_cols = processed_df.select_dtypes(include="number").columns
    processed_df[numeric_cols] = processed_df[numeric_cols].round(6)
    save_table_outputs(
        processed_df,
        title=f"{label} Data Table",
        stem=f"{sample_slug}_processed_data_table",
    )

    summary_rows.append(
        {
            "sample": label,
            "Tc_peak_degC": transition["Tc_peak"],
            "Tc_peak_unc_degC": transition["Tc_peak_unc"],
            "Tc_derivative_degC": transition["Tc_derivative"],
            "Tc_derivative_unc_degC": transition["Tc_derivative_unc"],
            "Tc_combined_degC": transition["Tc_combined"],
            "Tc_combined_unc_degC": transition["Tc_combined_unc"],
            "Tc_combined_unc_pct": transition["Tc_combined_unc_pct"],
            "Tc_method_difference_degC": transition["Tc_method_difference"],
            "epsilon_noise_std": transition["epsilon_noise_std"],
            "epsilon_instr_unc_mean": transition["epsilon_instr_unc_mean"],
            "Tc_peak_mc_std": transition["Tc_peak_mc_std"],
            "Tc_derivative_mc_std": transition["Tc_derivative_mc_std"],
        }
    )

    print(f"\n{label}")
    print("V_total_fit =", V_total_fit)
    print("Reference C =", C_ref)
    print(
        f"Transition temperature Tc (peak epsilon_r) = "
        f"{transition['Tc_peak']:.2f} +/- {transition['Tc_peak_unc']:.2f} degC"
    )
    print(
        f"Transition temperature Tc (max d epsilon_r/dT) = "
        f"{transition['Tc_derivative']:.2f} +/- {transition['Tc_derivative_unc']:.2f} degC"
    )
    print(
        f"Combined Tc (with method spread) = "
        f"{transition['Tc_combined']:.2f} +/- {transition['Tc_combined_unc']:.2f} degC"
    )
    print(
        f"Instrument settings: dT = {TEMP_LEAST_COUNT_C:.2f} degC, "
        f"dV = {VOLTAGE_LEAST_COUNT_V:.2f} V, MC samples = {MC_SAMPLES}"
    )

# -----------------------
# Plot epsilon_r vs T
# -----------------------

start_figure(7.4, 4.8)
styles = [
    {"marker": "o", "linestyle": "-", "color": "#1f77b4"},
    {"marker": "s", "linestyle": "--", "color": "#d62728"},
    {"marker": "^", "linestyle": "-.", "color": "#2ca02c"},
]

for (label, df), style in zip(results.items(), styles):
    plot_rt(
        df["Temperature_C"],
        df["epsilon_r"],
        label=label,
        marker=style["marker"],
        linestyle=style["linestyle"],
        color=style["color"],
    )
    transition = transition_results[label]
    plt.scatter(
        transition["Tc_peak"],
        transition["epsilon_at_peak"],
        marker="X",
        s=60,
        color=style["color"],
        label=(
            rf"{label} $T_c$ ($\epsilon_r$ peak) = {transition['Tc_peak']:.2f} "
            rf"$\pm$ {transition['Tc_peak_unc']:.2f}$^\circ$C"
        ),
    )

finalize_plot(
    title=r"Relative Permittivity vs. Temperature",
    xlabel=r"Temperature, $T$ ($^\circ$C)",
    ylabel=r"Relative permittivity, $\epsilon_r$",
    legend_loc="best",
    legend_ncol=1,
)
save_plot_outputs("epsilon_r_vs_T_all")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# -----------------------
# Plot epsilon_r vs T (offset for visual separation)
# -----------------------

start_figure(7.4, 4.8)
offset_step = 0.35

for idx, ((label, df), style) in enumerate(zip(results.items(), styles)):
    offset = idx * offset_step
    y_vals = df["epsilon_r"] + offset
    plot_rt(
        df["Temperature_C"],
        y_vals,
        label=f"{label} (+{offset:.2f})",
        marker=style["marker"],
        linestyle=style["linestyle"],
        color=style["color"],
    )
    transition = transition_results[label]
    plt.scatter(
        transition["Tc_peak"],
        transition["epsilon_at_peak"] + offset,
        marker="X",
        s=60,
        color=style["color"],
        label=(
            rf"{label} offset $T_c$ = {transition['Tc_peak']:.2f} "
            rf"$\pm$ {transition['Tc_peak_unc']:.2f}$^\circ$C"
        ),
    )

finalize_plot(
    title=r"Relative Permittivity vs. Temperature (Offset for Clarity)",
    xlabel=r"Temperature, $T$ ($^\circ$C)",
    ylabel=r"Relative permittivity, $\epsilon_r$ + offset",
    legend_loc="best",
    legend_ncol=1,
)
save_plot_outputs("epsilon_r_vs_T_all_offset")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# -----------------------
# Plot d(epsilon_r)/dT vs T with Tc markers
# -----------------------

start_figure(7.4, 4.8)
for (label, transition), style in zip(transition_results.items(), styles):
    plot_derivative(
        transition["temperature"],
        transition["d_eps_dT"],
        label=label,
        marker=style["marker"],
        linestyle=style["linestyle"],
        color=style["color"],
    )
    plt.axvline(
        transition["Tc_derivative"],
        color=style["color"],
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
        label=(
            rf"{label} $T_c$ (max $d\epsilon_r/dT$) = {transition['Tc_derivative']:.2f} "
            rf"$\pm$ {transition['Tc_derivative_unc']:.2f}$^\circ$C"
        ),
    )
    plt.scatter(
        transition["Tc_derivative"],
        transition["d_eps_dT_at_peak"],
        marker="D",
        s=40,
        color=style["color"],
    )

finalize_plot(
    title=r"Derivative of Relative Permittivity vs. Temperature",
    xlabel=r"Temperature, $T$ ($^\circ$C)",
    ylabel=r"$d\epsilon_r/dT$ (1/$^\circ$C)",
    legend_loc="best",
    legend_ncol=1,
)
save_plot_outputs("d_epsilon_r_dT_vs_T_all")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# -----------------------
# Plot V_total constant fit
# -----------------------

start_figure(7.2, 4.6)
for (label, valid), style in zip(vtotal_fit_points.items(), styles):
    vtotal_values = valid["Vdc_V"] + valid["Vsc_V"]
    vtotal_fit = vtotal_fit_values[label]
    plot_vtotal_fit(
        valid["Temperature_C"],
        vtotal_values,
        vtotal_fit,
        label,
        marker=style["marker"],
        linestyle=style["linestyle"],
        color=style["color"],
    )

finalize_plot(
    title=r"Estimated Constant $V_{\mathrm{total}}$",
    xlabel=r"Temperature, $T$ ($^\circ$C)",
    ylabel=r"$V_{\mathrm{total}}$ (V)",
    legend_loc="best",
    legend_ncol=1,
)
save_plot_outputs("vtotal_constant_fit")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

summary_df = pd.DataFrame(summary_rows)
summary_image_df = summary_df[
    [
        "sample",
        "Tc_peak_degC",
        "Tc_derivative_degC",
        "Tc_combined_degC",
    ]
].copy()
summary_image_df = summary_image_df.round(3)
save_table_outputs(
    summary_image_df,
    title="Curie Temperature Summary",
    stem="curie_temperature_summary_table",
)

error_df = summary_df[
    [
        "sample",
        "Tc_peak_unc_degC",
        "Tc_derivative_unc_degC",
        "Tc_combined_unc_degC",
        "Tc_combined_unc_pct",
        "Tc_method_difference_degC",
        "epsilon_noise_std",
        "epsilon_instr_unc_mean",
        "Tc_peak_mc_std",
        "Tc_derivative_mc_std",
    ]
].copy()
error_df = error_df.round(4)
save_table_outputs(
    error_df,
    title="Curie Temperature Error Analysis",
    stem="curie_temperature_error_analysis_table",
)

vtotal_df = pd.DataFrame(vtotal_rows).round(6)
save_table_outputs(
    vtotal_df,
    title="Vtotal Fit Summary",
    stem="vtotal_fit_summary_table",
)

with open(os.path.join(RESULTS_DIR, "transition_temperature_report.txt"), "w", encoding="utf-8") as f:
    f.write("Transition Temperature (Curie) Report\n")
    f.write("=" * 42 + "\n")
    f.write(
        f"Uncertainty inputs: dT = {TEMP_LEAST_COUNT_C:.3f} degC, "
        f"dVdc = dVsc = {VOLTAGE_LEAST_COUNT_V:.3f} V, MC samples = {MC_SAMPLES}\n"
    )
    f.write("=" * 42 + "\n")
    for label, transition in transition_results.items():
        f.write(f"{label}\n")
        f.write(
            f"Tc from peak epsilon_r: {transition['Tc_peak']:.2f} +/- "
            f"{transition['Tc_peak_unc']:.2f} degC\n"
        )
        f.write(
            f"Tc from max d(epsilon_r)/dT: {transition['Tc_derivative']:.2f} +/- "
            f"{transition['Tc_derivative_unc']:.2f} degC\n"
        )
        f.write(
            f"Combined Tc: {transition['Tc_combined']:.2f} +/- "
            f"{transition['Tc_combined_unc']:.2f} degC "
            f"({transition['Tc_combined_unc_pct']:.2f}% relative)\n"
        )
        f.write(
            f"Method difference |Tc_peak - Tc_derivative|: "
            f"{transition['Tc_method_difference']:.2f} degC\n"
        )
        f.write(
            f"Epsilon smoothing residual std: {transition['epsilon_noise_std']:.4e}\n"
        )
        f.write(
            f"Mean propagated epsilon instrument uncertainty: "
            f"{transition['epsilon_instr_unc_mean']:.4e}\n"
        )
        f.write(
            f"MC Tc std (peak epsilon_r): {transition['Tc_peak_mc_std']:.4f} degC\n"
        )
        f.write(
            f"MC Tc std (max d(epsilon_r)/dT): {transition['Tc_derivative_mc_std']:.4f} degC\n"
        )
        f.write("-" * 42 + "\n")

json_payload = {
    label: {
        "Tc_peak_degC": transition["Tc_peak"],
        "Tc_peak_unc_degC": transition["Tc_peak_unc"],
        "Tc_derivative_degC": transition["Tc_derivative"],
        "Tc_derivative_unc_degC": transition["Tc_derivative_unc"],
        "Tc_combined_degC": transition["Tc_combined"],
        "Tc_combined_unc_degC": transition["Tc_combined_unc"],
        "Tc_combined_unc_pct": transition["Tc_combined_unc_pct"],
        "Tc_method_difference_degC": transition["Tc_method_difference"],
        "epsilon_noise_std": transition["epsilon_noise_std"],
        "epsilon_instr_unc_mean": transition["epsilon_instr_unc_mean"],
        "Tc_peak_mc_std": transition["Tc_peak_mc_std"],
        "Tc_derivative_mc_std": transition["Tc_derivative_mc_std"],
    }
    for label, transition in transition_results.items()
}

with open(os.path.join(RESULTS_DIR, "transition_temperature_report.json"), "w", encoding="utf-8") as f:
    json.dump(json_payload, f, indent=2)
