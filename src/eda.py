"""Development-train-only descriptive analysis; no fitted preprocessing/models."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PROJECT_ROOT
from src.data import sha256
from src.schema import CATEGORICAL_FEATURES, CLASSES, NUMERIC_FEATURES, TARGET
from src.splits import PROCESSED_DIR, load_development_split

LABELS = {
    "air_humidity_pct": "Air humidity (%)",
    "soil_humidity_pct": "Soil moisture (%)",
    "ethylene_ppm": "Ethylene (source unit uncertain)",
    "co2_ppm": "CO2 (source unit uncertain)",
    "o2_ppm": "O2 (source unit uncertain)",
    "ldr_on_plant": "Simulated LDR reading (0-1023)",
    "leaf_color_index": "Leaf color index (0-100)",
    "temperature_c": "Temperature (degrees C)",
}


def run_eda(output_dir=PROJECT_ROOT / "reports"):
    """Generate summaries, figures, and interpretations from the saved fit rows."""
    frame = load_development_split("development_train")
    numeric = frame.loc[:, list(NUMERIC_FEATURES)]
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Nonfinite/missing sensor values require explicit audit handling before plotting.")
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    counts = frame[TARGET].value_counts().reindex(CLASSES, fill_value=0)
    proportions = counts / len(frame)
    correlations = numeric.corr()
    pairs = sorted([
        {"feature_a": a, "feature_b": b, "pearson_r": float(correlations.loc[a, b])}
        for i, a in enumerate(NUMERIC_FEATURES) for b in NUMERIC_FEATURES[i + 1:]
    ], key=lambda pair: abs(pair["pearson_r"]), reverse=True)
    quartiles = numeric.quantile([0.25, 0.75])
    spread = quartiles.loc[0.75] - quartiles.loc[0.25]
    flags = (numeric < quartiles.loc[0.25] - 1.5 * spread) | (numeric > quartiles.loc[0.75] + 1.5 * spread)
    # Only bounds supported by the source card/percentage definitions are used.
    bounds = {
        "air_humidity_pct": (0, 100), "soil_humidity_pct": (0, 100),
        "ldr_on_plant": (0, 1023), "leaf_color_index": (0, 100), "health_score": (0, 100),
    }
    violations = {name: int(((frame[name] < lo) | (frame[name] > hi)).sum())
                  for name, (lo, hi) in bounds.items()}
    score_ranges = {
        label: {"min": float(frame.loc[frame[TARGET] == label, "health_score"].min()),
                "max": float(frame.loc[frame[TARGET] == label, "health_score"].max())}
        for label in CLASSES
    }
    score_table = pd.crosstab(frame.health_score, frame[TARGET]).reindex(columns=CLASSES, fill_value=0)
    thresholds_match = np.where(frame.health_score >= 80, "healthy",
                                np.where(frame.health_score >= 55, "moderate_stress", "high_stress"))
    split_metadata = json.loads((PROCESSED_DIR / "split_metadata.json").read_text(encoding="utf-8"))
    summary = {
        "scope": "development_train only", "rows": len(frame),
        "dataset_revision": split_metadata["revision"],
        "source_sha256": split_metadata["source_sha256"],
        "membership_sha256": sha256(PROCESSED_DIR / "split_membership.csv"),
        "class_counts": {name: int(counts[name]) for name in CLASSES},
        "class_proportions": {name: float(proportions[name]) for name in CLASSES},
        "majority_class": str(counts.idxmax()), "majority_accuracy": float(proportions.max()),
        "missing_counts": {name: int(count) for name, count in frame.isna().sum().items()},
        "category_counts": {name: {str(k): int(v) for k, v in frame[name].value_counts().items()}
                            for name in CATEGORICAL_FEATURES},
        "numeric_summary": numeric.describe().to_dict(),
        "sensor_pearson_correlations": correlations.to_dict(), "strongest_sensor_pairs": pairs[:5],
        "sensor_score_pearson_r": frame[[*NUMERIC_FEATURES, "health_score"]].corr()["health_score"].drop("health_score").to_dict(),
        "iqr_flag_counts": {name: int(value) for name, value in flags.sum().items()},
        "documented_bound_violations": violations,
        "unknown_bounds": [name for name in NUMERIC_FEATURES if name not in bounds],
        "score_ranges_by_class": score_ranges,
        "score_values_with_multiple_classes": int((score_table.gt(0).sum(axis=1) > 1).sum()),
        "score_threshold_agreement": float(np.mean(thresholds_match == frame[TARGET].to_numpy())),
        "corrections": [], "rows_removed": 0,
        "threshold_audit_warning": "Post-hoc label-generation evidence on development-train only; not a validated predictive model. Score remains excluded.",
    }
    figures = []

    def save(fig, filename, title, interpretation):
        fig.suptitle(f"{title} — development-train only", fontsize=14)
        fig.tight_layout(rect=(0, 0.12 if fig.legends else 0, 1, 0.96))
        path = figures_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append({"file": f"figures/{filename}", "title": title, "interpretation": interpretation})

    sns.set_theme(style="whitegrid", context="notebook")
    palette = dict(zip(CLASSES, sns.color_palette("colorblind", 3)))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(list(CLASSES), counts.to_numpy(), color=[palette[c] for c in CLASSES])
    ax.bar_label(bars, labels=[f"{counts[c]} ({proportions[c]:.2%})" for c in CLASSES], padding=4)
    ax.set(xlabel="Health class", ylabel="Records", ylim=(0, counts.max() * 1.2))
    save(fig, "eda_class_balance.png", "Class balance",
         f"Moderate stress is the majority class ({counts['moderate_stress']} of {len(frame)}; {proportions['moderate_stress']:.2%}), while healthy has only {counts['healthy']} records ({proportions['healthy']:.2%}). A constant-majority prediction therefore gives {proportions.max():.2%} development-training accuracy; macro F1 and per-class recall are needed to expose minority-class failures.")

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, name in zip(axes.flat, NUMERIC_FEATURES):
        ax.hist(frame[name], bins=25, color="#3672a3", edgecolor="white")
        ax.set(xlabel=LABELS[name], ylabel="Records")
    save(fig, "eda_numeric_distributions.png", "Numeric sensor distributions",
         "The broad, fairly flat sensor distributions are consistent with simulated ranges, rather than evidence of measured natural variability. Their scales differ substantially, motivating training-fitted standardization for linear models; uncertain gas units remain unconverted.")

    fig, ax = plt.subplots(figsize=(11, 5))
    missing = frame.isna().sum()
    bars = ax.bar(missing.index, missing.to_numpy(), color="#3672a3")
    ax.bar_label(bars, padding=2)
    ax.set(xlabel="Raw column (including excluded labels and ID)", ylabel="Missing records", ylim=(0, max(1, int(missing.max()) + 1)))
    ax.tick_params(axis="x", rotation=65)
    save(fig, "eda_missing_values.png", "Missing values",
         "All 14 development-training fields have zero missing values. No imputation or row deletion is needed for this snapshot, although a training-fitted imputer remains planned for the reusable pipeline.")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name in zip(axes, CATEGORICAL_FEATURES):
        frequencies = frame[name].value_counts().sort_index()
        ax.bar(frequencies.index, frequencies.to_numpy(), color="#3672a3")
        ax.set(xlabel=name, ylabel="Records")
        ax.tick_params(axis="x", rotation=65)
    save(fig, "eda_category_frequencies.png", "Categorical frequencies",
         "Plant types and seasons are relatively evenly represented, whereas habitats are uneven: greenhouse and indoor are common and arid_garden/hydroponic are much smaller groups. These counts describe coverage in synthetic development data, not real-world prevalence.")

    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax, name in zip(axes.flat, NUMERIC_FEATURES):
        sns.boxplot(data=frame, x=TARGET, y=name, order=CLASSES, ax=ax, color="#8eb9d5")
        ax.set(xlabel="Health class", ylabel=LABELS[name])
        ax.tick_params(axis="x", rotation=35)
    save(fig, "eda_sensors_by_class.png", "Sensor distributions by class",
         "Leaf color, ethylene, and oxygen show visible class shifts, but distributions overlap and the synthetic scoring recipe can create these associations without establishing biological causation. Boxplot fliers use within-class IQR fences, which differ from the overall-development IQR screen reported in the range audit; no fliers were removed.")

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, name in zip(axes, CATEGORICAL_FEATURES):
        table = pd.crosstab(frame[name], frame[TARGET], normalize="index").reindex(columns=CLASSES, fill_value=0)
        table.plot.bar(stacked=True, ax=ax, color=[palette[c] for c in CLASSES], legend=False)
        ax.set(xlabel=name, ylabel="Within-category class proportion", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=65)
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[c]) for c in CLASSES]
    fig.legend(handles, CLASSES, loc="lower center", ncol=3)
    fig.subplots_adjust(bottom=0.3)
    save(fig, "eda_categories_by_class.png", "Class proportions by context",
         "Class proportions vary across context categories, so categorical inputs may encode part of the synthetic label-generation process. Each bar is normalized within its category; inspect the preceding frequency plot before interpreting differences in smaller groups.")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0,
                cmap="vlag", ax=ax, cbar_kws={"label": "Pearson correlation r"})
    ax.set(xlabel="Numeric predictor", ylabel="Numeric predictor")
    strongest = pairs[0]
    save(fig, "eda_sensor_correlations.png", "Sensor correlation matrix",
         f"The largest absolute off-diagonal Pearson correlation is only {abs(strongest['pearson_r']):.3f} ({strongest['feature_a']} versus {strongest['feature_b']}); no strong linear sensor redundancy appears. Near-zero correlation does not prove independence: nonlinear dependence may remain, and correlated predictors, if present, can destabilize individual coefficients even when predictions are stable.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label in CLASSES:
        subset = frame.loc[frame[TARGET] == label]
        axes[0].scatter(subset.leaf_color_index, subset.ethylene_ppm, s=10, alpha=0.4, color=palette[label], label=label)
        axes[1].scatter(subset.temperature_c, subset.soil_humidity_pct, s=10, alpha=0.4, color=palette[label], label=label)
    axes[0].set(xlabel=LABELS['leaf_color_index'], ylabel=LABELS['ethylene_ppm'])
    axes[1].set(xlabel=LABELS['temperature_c'], ylabel=LABELS['soil_humidity_pct'])
    axes[0].legend(title="Health class", fontsize=8)
    save(fig, "eda_pairwise_sensors.png", "Selected sensor pairs",
         "The leaf-color/ethylene panel shows class structure with substantial overlap; temperature/soil-moisture does not show an obvious single linear class boundary. These are descriptive views of two selected pairs, not evidence that remaining features are unimportant or that any classifier will generalize.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.arange(-0.5, 101.5, 1)
    for label in CLASSES:
        subset = frame.loc[frame[TARGET] == label, 'health_score']
        axes[0].hist(subset, bins=bins, alpha=0.65, color=palette[label], label=label)
    for threshold in [54.5, 79.5]:
        axes[0].axvline(threshold, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Excluded synthetic health score", ylabel="Records")
    axes[0].legend(title="Health class", fontsize=8)
    sns.heatmap(score_table.gt(0).astype(int).T, ax=axes[1], cmap="Blues", vmin=0, vmax=1,
                cbar_kws={"label": "Class present at score (0/1)"}, xticklabels=10)
    axes[1].set(xlabel="Observed health score", ylabel="Health class")
    save(fig, "eda_score_leakage.png", "Excluded score and label-generation audit",
         "In these development-training records, high stress occupies scores 0-54, moderate stress 55-79, and healthy 80-100; no observed score maps to multiple classes. Thresholds at 55 and 80 reproduce all observed labels, providing post-hoc leakage evidence rather than an independently evaluated model; health_score remains excluded from all predictors.")

    summary["figures"] = figures
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Development-training exploration — Step 5", "",
        f"All summaries and plots use only the **{len(frame):,} saved development-training rows**. Validation and test rows are not summarized, and no preprocessing or model is fitted. See [data boundaries](./DATA_BOUNDARIES.md), [machine-readable results](./eda_summary.json), and [the executed notebook](../notebooks/01_data_audit.ipynb).", "",
        "## Class balance", "", "| Class | Records | Proportion |", "| --- | ---: | ---: |",
        *[f"| {label} | {counts[label]} | {proportions[label]:.2%} |" for label in CLASSES], "",
        f"The majority class is `{counts.idxmax()}`; always predicting it gives **{proportions.max():.2%} accuracy on development-train**. This is a descriptive reference, not held-out baseline performance; the fitted DummyClassifier comparison remains Step 7.", "",
        "## Range and outlier audit", "",
        "All numeric predictors are finite, and all raw columns have zero missing values. No records violate the documented percentage/index/score bounds. No sensor records are outside the overall development-training 1.5-IQR fences; within-class boxplot fences differ and can flag observations. Neither heuristic establishes measurement validity.", "",
        "| Sensor | Observed minimum | Observed maximum | 1.5-IQR flags |", "| --- | ---: | ---: | ---: |",
        *[f"| {name} | {numeric[name].min():g} | {numeric[name].max():g} | {int(flags[name].sum())} |" for name in NUMERIC_FEATURES], "",
        "Gas-reading units remain uncertain, and the source supplies no validated admissible gas/temperature ranges. The LDR maximum of 1023 and leaf-index maximum of 100 are allowed synthetic endpoints, not reasons to delete rows. **Correction log: no edits, conversions, imputations, or removals; all 1,600 records retained.**", "",
        "## Figures and interpretations", "",
    ]
    for figure in figures:
        lines.extend([f"### {figure['title']}", "", f"![{figure['title']}]({figure['file']})", "", figure['interpretation'], ""])
    lines.extend([
        "## Implications for the next step", "",
        "Keep the existing 11-feature allowlist and the exclusion of both labels and record IDs. Use a training-fitted imputer/scaler/encoder in Step 6; no exploratory statistic here is reused as fitted preprocessing. Retain macro F1 as the primary selection metric and report per-class recall because healthy is the smallest class. Sensor/score associations and clean score boundaries reflect an intentionally learnable synthetic task; they do not validate real-plant diagnosis.", "",
        "The correlation discussion connects to [Assignment 2, Q6 and Q11](../Assigments/assignment_2_1.py): uncorrelated variables can be dependent, and correlated sensors can make individual coefficients unstable. Read figures as predictive associations rather than causal biological mechanisms.", "",
        "## Reproduction", "", "Run from the repository root:", "", "```powershell",
        "& ../.venv/Scripts/python.exe -m src.eda", "```", "",
        "The [EDA module](../src/eda.py) loads only `development_train`, writes the summary/report, and saves plots locally. The notebook embeds the saved plots and their interpretations so they remain viewable even when ignored figure files are not committed. No model has been trained; next is Step 6, the preprocessing pipeline.", "",
    ])
    (output_dir / "EDA_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    results = run_eda()
    print(json.dumps({name: results[name] for name in ["scope", "rows", "class_counts", "majority_accuracy", "iqr_flag_counts", "score_threshold_agreement"]}, indent=2))
