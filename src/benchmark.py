"""
Run predict.py first to generate data/predictions.csv, then:
    uv run src/benchmark.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jiwer import wer, cer
from matplotlib.patches import Circle
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes


def bootstrap_ci(ref: list, hyp: list, metric_fn, n_boot=1000, ci=0.95) -> float:
    """Return half-width of bootstrap CI for a metric."""
    rng = np.random.default_rng(42)
    n = len(ref)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = [ref[i] for i in idx]
        h = [hyp[i] for i in idx]
        scores.append(metric_fn(r, h))
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(scores, [alpha, 1 - alpha])
    return (hi - lo) / 2


def score_group_with_ci(df: pd.DataFrame, model: str) -> dict:
    ref = df["gt"].tolist()
    hyp = df[model].tolist()
    wer_val = round(1 - wer(ref, hyp), 4)
    cer_val = round(1 - cer(ref, hyp), 4)
    wer_ci = round(bootstrap_ci(ref, hyp, wer), 4)
    cer_ci = round(bootstrap_ci(ref, hyp, cer), 4)
    return {
        "1-WER": wer_val, "1-WER_CI": wer_ci,
        "1-CER": cer_val, "1-CER_CI": cer_ci,
    }


def radar_factory(num_vars):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return Circle((0.5, 0.5), 0.5)

        def _gen_axes_spines(self):
            return super()._gen_axes_spines()

    register_projection(RadarAxes)
    return theta


def plot_radar(results: pd.DataFrame, categories: list):
    """Radar chart comparing WER and CER across categories (excluding OVERALL)."""
    cat_results = results[results["category"] != "OVERALL"]
    cats = [c for c in categories if c != "OVERALL"]
    num_vars = len(cats)

    theta = radar_factory(num_vars)

    fig, axes = plt.subplots(figsize=(12, 5), ncols=2,
                             subplot_kw=dict(projection="radar"))
    fig.subplots_adjust(wspace=0.4)

    colors = {"renikud": "steelblue", "phonikud": "tomato"}
    models = cat_results["model"].unique()

    for ax, metric in zip(axes, ["1-WER", "1-CER"]):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8], angle=0)
        ax.set_title(f"{metric}  (↑ higher is better)", weight="bold",
                     position=(0.5, 1.15), ha="center")
        for model in models:
            vals = [
                cat_results.loc[
                    (cat_results["model"] == model) &
                    (cat_results["category"] == c), metric
                ].values[0]
                for c in cats
            ]
            ax.plot(theta, vals, color=colors[model], label=model)
            ax.fill(theta, vals, facecolor=colors[model], alpha=0.15)
        ax.set_varlabels(cats)

    axes[0].legend(loc=(1.05, 0.9), fontsize="small")
    fig.suptitle("Radar Chart — 1-WER & 1-CER per Category", weight="bold")
    plt.tight_layout()
    plt.savefig("data/radar_chart.png", dpi=150)
    print("Saved data/radar_chart.png")


def main():
    df = pd.read_csv("data/predictions.csv")

    models = ["renikud", "phonikud"]
    categories = df["category"].unique()

    records = []
    for model in models:
        for cat in categories:
            subset = df[df["category"] == cat]
            scores = score_group_with_ci(subset, model)
            records.append({"model": model, "category": cat, **scores})
        overall = score_group_with_ci(df, model)
        records.append({"model": model, "category": "OVERALL", **overall})

    results = pd.DataFrame(records)
    print(results[["model", "category", "1-WER", "1-CER"]].to_string(index=False))
    results.to_csv("data/benchmark_results.csv", index=False)

    # Bar plots with confidence interval error bars
    bar_width = 0.35
    colors = {"renikud": "steelblue", "phonikud": "tomato"}
    all_cats = list(categories) + ["OVERALL"]

    for metric in ["1-WER", "1-CER"]:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(all_cats))

        for i, model in enumerate(models):
            subset = results[results["model"] == model].set_index("category")
            vals = [subset.loc[c, metric] for c in all_cats]
            cis  = [subset.loc[c, f"{metric}_CI"] for c in all_cats]
            offset = (i - 0.5) * bar_width
            ax.bar(x + offset, vals, width=bar_width,
                   color=colors[model], edgecolor="black",
                   yerr=cis, capsize=6, label=model, alpha=0.85)

        ax.set_title(f"{metric} per Category  (↑ higher is better, error bars = 95% bootstrap CI)")
        ax.set_ylabel(metric)
        ax.set_xlabel("Category")
        ax.set_xticks(x)
        ax.set_xticklabels(all_cats, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        fname = metric.lower().replace("-", "_")
        plt.savefig(f"data/{fname}_per_category.png", dpi=150)
        print(f"Saved data/{fname}_per_category.png")

    plot_radar(results, all_cats)


if __name__ == "__main__":
    main()
