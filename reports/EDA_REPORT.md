# Development-training exploration — Step 5

All summaries and plots use only the **1,600 saved development-training rows**. Validation and test rows are not summarized, and no preprocessing or model is fitted. See [data boundaries](./DATA_BOUNDARIES.md), [machine-readable results](./eda_summary.json), and [the executed notebook](../notebooks/01_data_audit.ipynb).

## Class balance

| Class | Records | Proportion |
| --- | ---: | ---: |
| healthy | 192 | 12.00% |
| moderate_stress | 743 | 46.44% |
| high_stress | 665 | 41.56% |

The majority class is `moderate_stress`; always predicting it gives **46.44% accuracy on development-train**. This is a descriptive reference, not held-out baseline performance; the fitted DummyClassifier comparison remains Step 7.

## Range and outlier audit

All numeric predictors are finite, and all raw columns have zero missing values. No records violate the documented percentage/index/score bounds. No sensor records are outside the overall development-training 1.5-IQR fences; within-class boxplot fences differ and can flag observations. Neither heuristic establishes measurement validity.

| Sensor | Observed minimum | Observed maximum | 1.5-IQR flags |
| --- | ---: | ---: | ---: |
| air_humidity_pct | 25 | 93.9 | 0 |
| soil_humidity_pct | 8.1 | 90 | 0 |
| ethylene_ppm | 0.001 | 2.496 | 0 |
| co2_ppm | 330.5 | 1249.7 | 0 |
| o2_ppm | 18.2 | 21.5 | 0 |
| ldr_on_plant | 30 | 1023 | 0 |
| leaf_color_index | 20.1 | 100 | 0 |
| temperature_c | 8 | 38 | 0 |

Gas-reading units remain uncertain, and the source supplies no validated admissible gas/temperature ranges. The LDR maximum of 1023 and leaf-index maximum of 100 are allowed synthetic endpoints, not reasons to delete rows. **Correction log: no edits, conversions, imputations, or removals; all 1,600 records retained.**

## Figures and interpretations

### Class balance

![Class balance](figures/eda_class_balance.png)

Moderate stress is the majority class (743 of 1600; 46.44%), while healthy has only 192 records (12.00%). A constant-majority prediction therefore gives 46.44% development-training accuracy; macro F1 and per-class recall are needed to expose minority-class failures.

### Numeric sensor distributions

![Numeric sensor distributions](figures/eda_numeric_distributions.png)

The broad, fairly flat sensor distributions are consistent with simulated ranges, rather than evidence of measured natural variability. Their scales differ substantially, motivating training-fitted standardization for linear models; uncertain gas units remain unconverted.

### Missing values

![Missing values](figures/eda_missing_values.png)

All 14 development-training fields have zero missing values. No imputation or row deletion is needed for this snapshot, although a training-fitted imputer remains planned for the reusable pipeline.

### Categorical frequencies

![Categorical frequencies](figures/eda_category_frequencies.png)

Plant types and seasons are relatively evenly represented, whereas habitats are uneven: greenhouse and indoor are common and arid_garden/hydroponic are much smaller groups. These counts describe coverage in synthetic development data, not real-world prevalence.

### Sensor distributions by class

![Sensor distributions by class](figures/eda_sensors_by_class.png)

Leaf color, ethylene, and oxygen show visible class shifts, but distributions overlap and the synthetic scoring recipe can create these associations without establishing biological causation. Boxplot fliers use within-class IQR fences, which differ from the overall-development IQR screen reported in the range audit; no fliers were removed.

### Class proportions by context

![Class proportions by context](figures/eda_categories_by_class.png)

Class proportions vary across context categories, so categorical inputs may encode part of the synthetic label-generation process. Each bar is normalized within its category; inspect the preceding frequency plot before interpreting differences in smaller groups.

### Sensor correlation matrix

![Sensor correlation matrix](figures/eda_sensor_correlations.png)

The largest absolute off-diagonal Pearson correlation is only 0.044 (soil_humidity_pct versus ethylene_ppm); no strong linear sensor redundancy appears. Near-zero correlation does not prove independence: nonlinear dependence may remain, and correlated predictors, if present, can destabilize individual coefficients even when predictions are stable.

### Selected sensor pairs

![Selected sensor pairs](figures/eda_pairwise_sensors.png)

The leaf-color/ethylene panel shows class structure with substantial overlap; temperature/soil-moisture does not show an obvious single linear class boundary. These are descriptive views of two selected pairs, not evidence that remaining features are unimportant or that any classifier will generalize.

### Excluded score and label-generation audit

![Excluded score and label-generation audit](figures/eda_score_leakage.png)

In these development-training records, high stress occupies scores 0-54, moderate stress 55-79, and healthy 80-100; no observed score maps to multiple classes. Thresholds at 55 and 80 reproduce all observed labels, providing post-hoc leakage evidence rather than an independently evaluated model; health_score remains excluded from all predictors.

## Implications for the next step

Keep the existing 11-feature allowlist and the exclusion of both labels and record IDs. Use a training-fitted imputer/scaler/encoder in Step 6; no exploratory statistic here is reused as fitted preprocessing. Retain macro F1 as the primary selection metric and report per-class recall because healthy is the smallest class. Sensor/score associations and clean score boundaries reflect an intentionally learnable synthetic task; they do not validate real-plant diagnosis.

The correlation discussion connects to [Assignment 2, Q6 and Q11](../Assigments/assignment_2_1.py): uncorrelated variables can be dependent, and correlated sensors can make individual coefficients unstable. Read figures as predictive associations rather than causal biological mechanisms.

## Reproduction

Run from the repository root:

```powershell
& ../.venv/Scripts/python.exe -m src.eda
```

The [EDA module](../src/eda.py) loads only `development_train`, writes the summary/report, and saves plots locally. The notebook embeds the saved plots and their interpretations so they remain viewable even when ignored figure files are not committed. No model has been trained; next is Step 6, the preprocessing pipeline.
