"""
Run predict.py first to generate data/predictions.csv, then:
    uv run src/benchmark.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from jiwer import wer, cer


def score_group(df: pd.DataFrame, model: str) -> dict:
    ref = df["gt"].tolist()
    hyp = df[model].tolist()
    return {
        "WER": round(wer(ref, hyp), 4),
        "CER": round(cer(ref, hyp), 4),
    }


def main():
    df = pd.read_csv("data/predictions.csv")

    models = ["renikud", "phonikud"]
    categories = df["category"].unique()

    records = []
    for model in models:
        for cat in categories:
            subset = df[df["category"] == cat]
            scores = score_group(subset, model)
            records.append({"model": model, "category": cat, **scores})
        overall = score_group(df, model)
        records.append({"model": model, "category": "OVERALL", **overall})

    results = pd.DataFrame(records)
    print(results.to_string(index=False))
    results.to_csv("data/benchmark_results.csv", index=False)

    # Plot WER and CER per category per model
    for metric in ["WER", "CER"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot = results.pivot(index="category", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"{metric} per Category  (↓ lower is better)")
        ax.set_ylabel(metric)
        ax.set_xlabel("Category")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(f"data/{metric.lower()}_per_category.png", dpi=150)
        print(f"Saved data/{metric.lower()}_per_category.png")


if __name__ == "__main__":
    main()
