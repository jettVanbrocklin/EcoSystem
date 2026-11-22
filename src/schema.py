"""Explicit modeling boundary: raw numeric columns also contain a label."""

NUMERIC_FEATURES = (
    "air_humidity_pct", "soil_humidity_pct", "ethylene_ppm", "co2_ppm",
    "o2_ppm", "ldr_on_plant", "leaf_color_index", "temperature_c",
)
CATEGORICAL_FEATURES = ("plant_type", "habitat", "season")
FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
TARGET = "health_class"
CLASSES = ("healthy", "moderate_stress", "high_stress")
EXCLUDED_COLUMNS = ("record_id", "health_score", TARGET)


def features_target_ids(frame):
    """Return independent predictor, target, and traceability objects."""
    required = (*FEATURES, TARGET, "record_id")
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing modeling columns: {sorted(missing)}")
    if frame[TARGET].isna().any() or not frame[TARGET].isin(CLASSES).all():
        raise ValueError("Training targets must be nonmissing expected class labels.")
    return (
        frame.loc[:, list(FEATURES)].copy(),
        frame[TARGET].copy(),
        frame["record_id"].copy(),
    )
