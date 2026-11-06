from pathlib import Path


MODEL_LABELS = {
    "zero_excess": "Zero excess return",
    "price_only": "Price controls",
    "network": "Network features",
    "event_conditioned": "Disclosure features",
    "adaptive": "Training-selected",
}
COLORS = {
    "zero_excess": "#292d33",
    "price_only": "#8a959e",
    "network": "#687f91",
    "event_conditioned": "#405c78",
    "adaptive": "#987650",
}


def plot_diagnostics(report: dict, output: Path) -> None:
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    metrics = {row["model_id"]: row for row in report["metrics"]}
    comparisons = report["ablations"]
    annual = sorted(
        (row for row in report["slices"] if row["dimension"] == "year"),
        key=lambda row: int(row["value"]),
    )
    style = {
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#34383d", "text.color": "#252a2f",
        "xtick.color": "#4e555b", "ytick.color": "#4e555b",
        "axes.edgecolor": "#bec3c7", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with mpl.rc_context(style):
        figure = Figure(figsize=(13, 8.8), facecolor="white")
        FigureCanvasAgg(figure)
        grid = figure.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.65, wspace=0.55)
        errors = figure.add_subplot(grid[0, 0])
        intervals = figure.add_subplot(grid[0, 1])
        years = figure.add_subplot(grid[1, :])
        figure.subplots_adjust(left=0.21, right=0.97, top=0.86, bottom=0.15)
        figure.text(0.06, 0.955, "Peer-return forecasts after earnings disclosures", fontsize=17)
        figure.text(0.06, 0.92, "Corrected numerical study · inspected historical sample", fontsize=10, color="#59636b")

        short_names = {
            "zero_excess": "zero", "price_only": "price", "network": "network",
            "event_conditioned": "disclosure", "adaptive": "selected",
        }
        labels = []
        for position, row in enumerate(comparisons):
            color = COLORS[row["model_id"]]
            center = 100 * row["mae_improvement"]
            if row["ci_lower"] is not None and row["ci_upper"] is not None:
                lower, upper = 100 * row["ci_lower"], 100 * row["ci_upper"]
                errors.hlines(position, lower, upper, color=color, linewidth=1.7)
                errors.vlines([lower, upper], position - 0.07, position + 0.07, color=color, linewidth=1)
            errors.scatter(center, position, color=color, s=28, zorder=3)
            labels.append(f"{short_names[row['model_id']].capitalize()} vs {short_names[row['baseline_model_id']]}")
        errors.axvline(0, color="#a9afb4", linewidth=1, linestyle="--")
        errors.set_yticks(range(len(labels)), labels)
        errors.invert_yaxis()
        errors.set_xlabel("MAE improvement (pp)\nPositive favors the listed model", labelpad=8)
        errors.set_title("A   Paired feature comparisons", loc="left", pad=13, fontsize=11)
        errors.grid(axis="x", color="#eceef0", linewidth=0.7)
        errors.set_axisbelow(True)
        errors.margins(x=0.14, y=0.13)

        markers = ["o", "s", "^", "D", "P"]
        for marker, (model_id, row) in zip(markers, metrics.items()):
            intervals.scatter(
                100 * row["mean_interval_width"], 100 * row["interval_coverage"],
                label=MODEL_LABELS[model_id], color=COLORS[model_id], marker=marker,
                s=48, edgecolors="white", linewidths=0.5, zorder=3,
            )
        intervals.axhline(80, color="#8e969d", linewidth=1, linestyle="--")
        intervals.text(0.98, 80, "Nominal 80%", transform=intervals.get_yaxis_transform(), ha="right", va="bottom", fontsize=9)
        intervals.set_xlabel("Mean interval width (pp)", labelpad=8)
        intervals.set_ylabel("Observed coverage (%)")
        intervals.set_title("B   Interval coverage and width", loc="left", pad=13, fontsize=11)
        intervals.grid(color="#eceef0", linewidth=0.7)
        intervals.set_axisbelow(True)
        intervals.margins(x=0.22, y=0.15)
        intervals.legend(loc="upper left", bbox_to_anchor=(0, 0.84), fontsize=8, frameon=False, ncols=2, columnspacing=1, handletextpad=0.5)

        styles = ["--", ":", "-.", "-"]
        for line_style, model_id in zip(styles, (key for key in metrics if key != "zero_excess")):
            values = []
            for group in annual:
                by_model = {row["model_id"]: row for row in group["metrics"]}
                values.append(100 * (by_model["zero_excess"]["mae"] - by_model[model_id]["mae"]))
            years.plot(
                [int(group["value"]) for group in annual], values, linestyle=line_style,
                marker="o", markersize=3.5, linewidth=1.5, color=COLORS[model_id],
                label=MODEL_LABELS[model_id],
            )
        years.axhline(0, color="#a9afb4", linewidth=1)
        years.set_xticks([int(group["value"]) for group in annual])
        years.set_xlabel("Decision year · boundary years may be partial")
        years.set_ylabel("MAE improvement\nover zero return (pp)")
        years.set_title("C   Stability across years", loc="left", pad=13, fontsize=11)
        years.grid(axis="y", color="#eceef0", linewidth=0.7)
        years.legend(loc="best", fontsize=8, frameon=False, ncols=2)
        years.set_axisbelow(True)

        cluster_count = comparisons[0]["n_clusters"]
        draws = comparisons[0]["bootstrap_draws"]
        figure.text(0.06, 0.065, f"A: 95% paired bootstrap intervals; {cluster_count} connected holding-window blocks, {draws:,} resamples.", fontsize=9, color="#59636b")
        figure.text(0.06, 0.039, "Remaining serial dependence and model development are not accounted for. Panels B–C are descriptive; no causal or trading-edge claim.", fontsize=9, color="#59636b")
        output.mkdir(parents=True, exist_ok=True)
        figure.savefig(output / "diagnostics.png", dpi=180)
        figure.savefig(output / "diagnostics.pdf", metadata={"CreationDate": None, "ModDate": None})
