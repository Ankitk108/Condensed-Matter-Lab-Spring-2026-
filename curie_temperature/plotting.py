import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.grid": True,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.unicode_minus": False,
    }
)


def start_figure(width=7.2, height=4.6):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def plot_rt(T, V, label, marker="o", linestyle="-", color="black"):
    plt.plot(
        T,
        V,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.4,
        markersize=4.2,
        label=label,
    )


def plot_derivative(T, dVdT, label, marker="s", linestyle="--", color="black"):
    plt.plot(
        T,
        dVdT,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.3,
        markersize=3.2,
        label=label,
    )


def plot_vtotal_fit(temperature, vtotal, vtotal_fit, label, marker="o", linestyle="--", color="black"):
    plt.scatter(
        temperature,
        vtotal,
        color=color,
        marker=marker,
        s=24,
        label=f"{label} data",
    )
    plt.plot(
        [min(temperature), max(temperature)],
        [vtotal_fit, vtotal_fit],
        color=color,
        linestyle=linestyle,
        linewidth=1.4,
        label=rf"{label} fit: $V_{{\mathrm{{total}}}} = {vtotal_fit:.4f}\,\mathrm{{V}}$",
    )

def finalize_plot(
    title,
    xlabel,
    ylabel,
    legend_loc="best",
    legend_ncol=1,
    legend_outside=False,
    legend_group_columns=False,
):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    handles, labels = plt.gca().get_legend_handles_labels()
    if legend_group_columns and legend_ncol == 2 and len(handles) % 2 == 0:
        mid = len(handles) // 2
        reordered = []
        for left, right in zip(range(mid), range(mid, len(handles))):
            reordered.extend([left, right])
        handles = [handles[i] for i in reordered]
        labels = [labels[i] for i in reordered]
    if legend_outside:
        plt.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=legend_ncol,
            frameon=False,
            columnspacing=1.2,
            handletextpad=0.5,
        )
    else:
        plt.legend(handles, labels, loc=legend_loc, ncol=legend_ncol, frameon=True)
    plt.grid(True)
    plt.tight_layout(pad=0.6)
