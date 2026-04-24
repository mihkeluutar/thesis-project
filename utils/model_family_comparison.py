from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:  # pragma: no cover - runtime dependent
    tf = None
    keras = None
    layers = None

try:
    from .xgboost_architecture_lock import PRESET_CANDIDATES, require_xgboost
except Exception:  # pragma: no cover - import style depends on caller
    from xgboost_architecture_lock import PRESET_CANDIDATES, require_xgboost


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_DIR = DATA_DIR / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURE_METADATA_FILE = FEATURE_DIR / "feature_metadata.csv"
FEATURE_CATALOG_FILE = FEATURE_DIR / "feature_catalog.csv"

DEFAULT_BUILDINGS = ("U05", "U06", "LIB", "U02B", "SOC", "U03")
DEFAULT_HORIZONS = (1, 2, 4, 6, 8, 12, 16, 20, 24, 36)
DEFAULT_REGIMES = ("per_building", "pooled_same_buildings")
DEFAULT_WEATHER_MODES = ("FW0", "FW1", "FW2")
DEFAULT_MODES = ("M0", "M1", "M2", "M4")
DEFAULT_MODEL_FAMILIES = ("lstm", "xgboost")
CANONICAL_MODE_ORDER = ("M0", "M1", "M2", "M3", "M4")
CANONICAL_WEATHER_MODE_ORDER = ("FW0", "FW1", "FW2")
COMPARISON_MANIFEST_KEY_COLS = ["regime", "building", "model_family", "mode", "weather_mode", "horizon_h"]
COMPARISON_PAIR_KEY_COLS = ["regime", "building", "mode", "weather_mode", "horizon_h"]

HEATING_TEMP_THRESHOLD_C = 15.0
XGB_FIXED_PARAMS = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "tree_method": "hist",
    "n_estimators": 2000,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_lambda": 5.0,
    "early_stopping_rounds": 50,
}

LSTM_BASE_TEMPORAL_FEATURES = [
    "feat_heat_obs",
    "feat_outdoor_temp_c",
    "feat_wind_ms",
    "feat_solar_irradiance_wm2",
    "feat_hour_sin",
    "feat_hour_cos",
    "feat_dow_sin",
    "feat_dow_cos",
]

LSTM_WEATHER_MEMORY_FEATURES = [
    "feat_temp_roll24h",
]

LSTM_SYSTEM_DYNAMIC_FEATURES = [
    "feat_space_heat_active",
    "feat_space_deltaT_c",
    "feat_space_low_deltaT_flag",
    "feat_vent_heat_active",
    "feat_vent_deltaT_c",
    "feat_vent_low_deltaT_flag",
]

LSTM_STATIC_FEATURES_SETB = [
    "stat_heated_area_m2",
    "stat_usage_non_res_share_of_heated",
    "stat_building_age_years",
    "stat_n_points",
    "stat_n_heat_points",
    "stat_n_vent_points",
    "stat_n_dhw_points",
    "stat_vent_class_basic",
    "stat_vent_class_none",
    "stat_vent_class_rich",
    "stat_ventilation_has_heat_recovery",
    "ehr_compactness_ratio",
    "ehr_max_floors",
    "ehr_volume_per_heated_area",
    "stat_missing_heated_area_m2",
    "stat_missing_usage_non_res_share_of_heated",
    "stat_missing_building_age_years",
    "ehr_missing_compactness_ratio",
    "ehr_missing_max_floors",
    "ehr_missing_volume_per_heated_area",
]

MODE_TEMPORAL_FEATURES = {
    "M0": LSTM_BASE_TEMPORAL_FEATURES.copy(),
    "M1": LSTM_BASE_TEMPORAL_FEATURES + LSTM_WEATHER_MEMORY_FEATURES,
    "M2": LSTM_BASE_TEMPORAL_FEATURES + LSTM_SYSTEM_DYNAMIC_FEATURES,
    "M3": LSTM_BASE_TEMPORAL_FEATURES + LSTM_WEATHER_MEMORY_FEATURES + LSTM_SYSTEM_DYNAMIC_FEATURES,
    "M4": LSTM_BASE_TEMPORAL_FEATURES + LSTM_WEATHER_MEMORY_FEATURES + LSTM_SYSTEM_DYNAMIC_FEATURES,
}
for _mode_name, _cols in list(MODE_TEMPORAL_FEATURES.items()):
    seen: set[str] = set()
    MODE_TEMPORAL_FEATURES[_mode_name] = [c for c in _cols if not (c in seen or seen.add(c))]

ARCHITECTURES = {
    "A6": {
        "architecture_id": "A6",
        "architecture_label": "single 64 | L72",
        "lookback_hours": 72,
        "lstm_stack": [64],
        "dropout": 0.0,
        "dense_units": 16,
        "notes": "Best overall long-horizon planning architecture from 07.",
    },
    "A3": {
        "architecture_id": "A3",
        "architecture_label": "single 64 | L48",
        "lookback_hours": 48,
        "lstm_stack": [64],
        "dropout": 0.0,
        "dense_units": 16,
        "notes": "Lean single-layer near-tied option from 07.",
    },
}

XGB_PRESET_LOOKUP = {preset["preset_id"]: preset for preset in PRESET_CANDIDATES}


MODE_DESCRIPTIONS = {
    "M0": "Base temporal core only: observed heat, current outdoor weather, wind, solar, and cyclic calendar features.",
    "M1": "M0 plus historical weather memory via the 24h rolling outdoor-temperature feature.",
    "M2": "M0 plus system / inertia proxies from space-heating and ventilation loop activity and deltaT state.",
    "M3": "M1 plus M2, still dynamic-only and still sourced from setA.",
    "M4": "M3 plus the static building-context branch from setB; forecasts still remain a weather-mode choice, not a mode property.",
}

WEATHER_MODE_DESCRIPTIONS = {
    "FW0": {
        "description": "No future weather is appended.",
        "future_weather_origin": "none",
        "operational_status": "baseline",
        "upper_bound_only": False,
        "forecast_like": False,
        "exported_in_regular_feature_csv": False,
        "built_in_memory_at_runtime": False,
    },
    "FW1": {
        "description": "Oracle future weather is built in memory from actual future temperature / RH and should be treated as an upper bound only.",
        "future_weather_origin": "oracle future weather",
        "operational_status": "upper_bound_only",
        "upper_bound_only": True,
        "forecast_like": False,
        "exported_in_regular_feature_csv": False,
        "built_in_memory_at_runtime": True,
    },
    "FW2": {
        "description": "Forecast-like proxy future weather comes from exported feat_fw_* columns in the regular setA / setB feature files.",
        "future_weather_origin": "forecast-like proxy future weather",
        "operational_status": "operational_analogue",
        "upper_bound_only": False,
        "forecast_like": True,
        "exported_in_regular_feature_csv": True,
        "built_in_memory_at_runtime": False,
    },
}


@dataclass(frozen=True)
class ExperimentConfig:
    buildings: tuple[str, ...] = DEFAULT_BUILDINGS
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    regimes: tuple[str, ...] = DEFAULT_REGIMES
    weather_modes: tuple[str, ...] = DEFAULT_WEATHER_MODES
    modes: tuple[str, ...] = DEFAULT_MODES
    train_end: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2023-12-31 23:00:00"))
    test_start: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2024-01-01 00:00:00"))
    lookback_hours: int = 72
    validation_fraction: float = 0.10
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR / "model_family_comparison_23032026")
    lstm_architecture_id: str = "A6"
    xgb_preset_id: str = "P01_md3_lr003_mc5"
    random_seed: int = 42
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 8
    learning_rate: float = 1e-3
    deterministic_ops: bool = True
    model_families: tuple[str, ...] = DEFAULT_MODEL_FAMILIES


@dataclass(frozen=True)
class RunSpec:
    regime: str
    building: str
    model_family: str
    mode: str
    weather_mode: str
    horizon_h: int


@dataclass
class FeatureBundle:
    raw_frame: pd.DataFrame
    target_name: str
    dynamic_feature_cols: list[str]
    static_feature_cols: list[str]
    valid_issue_mask: pd.Series
    model_ready: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitSpec:
    train_issue_end: pd.Timestamp
    feature_train_mask: pd.Series
    train_issue_mask: pd.Series
    test_issue_mask: pd.Series


@dataclass
class TrainingArtifacts:
    run_spec: RunSpec
    history_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    prediction_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonOutputs:
    manifest_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison_predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison_metrics_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison_coverage_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    run_log_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    lstm_training_history_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    xgb_eval_history_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def ensure_results_dirs(config: ExperimentConfig) -> dict[str, Path]:
    plot_dir = config.results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return {
        "results": config.results_dir,
        "plots": plot_dir,
        "manifest": config.results_dir / "comparison_manifest.csv",
        "predictions": config.results_dir / "comparison_predictions.csv",
        "metrics": config.results_dir / "comparison_metrics.csv",
        "summary": config.results_dir / "comparison_summary.csv",
        "coverage": config.results_dir / "comparison_coverage.csv",
        "run_log": config.results_dir / "run_log.csv",
        "lstm_history": config.results_dir / "lstm_training_history.csv",
        "xgb_history": config.results_dir / "xgb_eval_history.csv",
    }


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def _sort_reset(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    use_cols = [col for col in sort_cols if col in work.columns]
    if use_cols:
        work.sort_values(use_cols, inplace=True)
    work.reset_index(drop=True, inplace=True)
    return work


def _dedupe(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    use_cols = [col for col in subset if col in df.columns]
    if not use_cols:
        return df.copy().reset_index(drop=True)
    return df.drop_duplicates(subset=use_cols, keep="last").reset_index(drop=True)


def _normalize_outputs(outputs: ComparisonOutputs) -> ComparisonOutputs:
    manifest_df = _dedupe(
        _sort_reset(
            outputs.manifest_df,
            ["regime", "building", "model_family", "mode", "weather_mode", "horizon_h"],
        ),
        ["run_id"],
    )
    predictions_df = _dedupe(
        _sort_reset(
            outputs.comparison_predictions_df,
            ["regime", "building", "model_family", "mode", "weather_mode", "horizon_h", "datetime"],
        ),
        ["run_id", "building", "datetime"],
    )
    metrics_df = _dedupe(
        _sort_reset(
            outputs.comparison_metrics_df,
            ["regime", "building", "mode", "weather_mode", "horizon_h", "model_family"],
        ),
        ["regime", "building", "model_family", "mode", "weather_mode", "horizon_h"],
    )
    coverage_df = _dedupe(
        _sort_reset(
            outputs.comparison_coverage_df,
            ["regime", "building", "mode", "weather_mode", "horizon_h"],
        ),
        ["regime", "building", "mode", "weather_mode", "horizon_h"],
    )
    run_log_df = _sort_reset(
        outputs.run_log_df,
        ["regime", "building", "model_family", "mode", "weather_mode", "horizon_h"],
    )
    lstm_history_df = _dedupe(
        _sort_reset(
            outputs.lstm_training_history_df,
            ["run_id", "epoch"],
        ),
        ["run_id", "epoch"],
    )
    xgb_history_df = _dedupe(
        _sort_reset(
            outputs.xgb_eval_history_df,
            ["run_id", "iteration", "dataset", "metric"],
        ),
        ["run_id", "dataset", "metric", "iteration"],
    )
    summary_df = build_comparison_summary(metrics_df)
    return ComparisonOutputs(
        manifest_df=manifest_df,
        comparison_predictions_df=predictions_df,
        comparison_metrics_df=metrics_df,
        comparison_summary_df=summary_df,
        comparison_coverage_df=coverage_df,
        run_log_df=run_log_df,
        lstm_training_history_df=lstm_history_df,
        xgb_eval_history_df=xgb_history_df,
    )


def load_saved_outputs(
    config: ExperimentConfig,
    *,
    manifest_df: pd.DataFrame | None = None,
) -> ComparisonOutputs:
    paths = ensure_results_dirs(config)
    outputs = ComparisonOutputs(
        manifest_df=manifest_df.copy() if manifest_df is not None else _read_csv_if_exists(paths["manifest"]),
        comparison_predictions_df=_read_csv_if_exists(paths["predictions"]),
        comparison_metrics_df=_read_csv_if_exists(paths["metrics"]),
        comparison_summary_df=_read_csv_if_exists(paths["summary"]),
        comparison_coverage_df=_read_csv_if_exists(paths["coverage"]),
        run_log_df=_read_csv_if_exists(paths["run_log"]),
        lstm_training_history_df=_read_csv_if_exists(paths["lstm_history"]),
        xgb_eval_history_df=_read_csv_if_exists(paths["xgb_history"]),
    )
    if outputs.manifest_df.empty and manifest_df is not None:
        outputs.manifest_df = manifest_df.copy()
    return _normalize_outputs(outputs)


def build_artifact_status_table(config: ExperimentConfig) -> pd.DataFrame:
    paths = ensure_results_dirs(config)
    rows = []
    for name, path in paths.items():
        if name in {"results", "plots"}:
            rows.append({"artifact": name, "exists": path.exists(), "path": str(path), "rows": np.nan})
            continue
        n_rows = np.nan
        if path.exists():
            try:
                n_rows = int(len(pd.read_csv(path)))
            except Exception:
                n_rows = np.nan
        rows.append({"artifact": path.name, "exists": path.exists(), "path": str(path), "rows": n_rows})
    return pd.DataFrame(rows)


def _write_outputs(paths: dict[str, Path], outputs: ComparisonOutputs) -> None:
    normalized = _normalize_outputs(outputs)
    normalized.manifest_df.to_csv(paths["manifest"], index=False)
    normalized.comparison_predictions_df.to_csv(paths["predictions"], index=False)
    normalized.comparison_metrics_df.to_csv(paths["metrics"], index=False)
    normalized.comparison_summary_df.to_csv(paths["summary"], index=False)
    normalized.comparison_coverage_df.to_csv(paths["coverage"], index=False)
    normalized.run_log_df.to_csv(paths["run_log"], index=False)
    normalized.lstm_training_history_df.to_csv(paths["lstm_history"], index=False)
    normalized.xgb_eval_history_df.to_csv(paths["xgb_history"], index=False)


def build_defended_comparison_contract_metadata(
    *,
    rmse_metric_label: str = "Cumulative RMSE (kWh)",
    rmse_context_csv: str = "rmse_load_context_per_building.csv",
) -> dict[str, Any]:
    return {
        "reported_target_column": "heat_kwh",
        "reported_error_unit": "kWh",
        "headline_metric": "WAPE_pct",
        "rmse_metric_label": rmse_metric_label,
        "heating_rule_name": "weather_defined_heating_slice",
        "heating_threshold_c": float(HEATING_TEMP_THRESHOLD_C),
        "heating_flag_column": "feat_is_heating_weather",
        "heating_flag_rule": "flag > 0.5 when present",
        "heating_fallback_rule": f"feat_outdoor_temp_c < {HEATING_TEMP_THRESHOLD_C:.1f}",
        "evaluation_population": "matched_valid_heating_rows",
        "no_heating_row_fallback": "all_rows_emergency_fallback",
        "rmse_context_stat": "median_observed_load_kwh",
        "rmse_context_csv": rmse_context_csv,
    }


def attach_defended_comparison_contract(
    df: pd.DataFrame,
    *,
    rmse_metric_label: str = "Cumulative RMSE (kWh)",
    rmse_context_csv: str = "rmse_load_context_per_building.csv",
) -> pd.DataFrame:
    work = df.copy()
    metadata = build_defended_comparison_contract_metadata(
        rmse_metric_label=rmse_metric_label,
        rmse_context_csv=rmse_context_csv,
    )
    for key, value in metadata.items():
        work[key] = value
    return work


def build_rmse_load_context_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "regime",
        "building",
        "mode",
        "weather_mode",
        "horizon_h",
        "n_eval_rows",
        "median_observed_load_kwh",
        "mean_observed_load_kwh",
        "evaluation_population",
    ]
    if predictions_df.empty:
        return pd.DataFrame(columns=cols)

    dedupe_cols = ["regime", "building", "mode", "weather_mode", "horizon_h", "datetime"]
    base = (
        predictions_df.copy()
        .drop_duplicates(subset=dedupe_cols)
        .sort_values(dedupe_cols)
        .reset_index(drop=True)
    )
    group_cols = ["regime", "building", "mode", "weather_mode", "horizon_h"]
    rows: list[dict[str, Any]] = []
    for keys, group in base.groupby(group_cols, dropna=False):
        heating_group = group.loc[group["is_heating_eval"].fillna(False)].copy()
        chosen = heating_group if not heating_group.empty else group
        evaluation_population = "matched_valid_heating_rows" if not heating_group.empty else "all_rows_emergency_fallback"
        rows.append(
            {
                "regime": keys[0],
                "building": keys[1],
                "mode": keys[2],
                "weather_mode": keys[3],
                "horizon_h": int(keys[4]),
                "n_eval_rows": int(len(chosen)),
                "median_observed_load_kwh": float(chosen["y_true"].median()),
                "mean_observed_load_kwh": float(chosen["y_true"].mean()),
                "evaluation_population": evaluation_population,
            }
        )
    return pd.DataFrame(rows, columns=cols).sort_values(group_cols).reset_index(drop=True)


def write_defended_comparison_contract_metadata(
    *,
    report_dir: Path,
    rmse_metric_label: str = "Cumulative RMSE (kWh)",
    rmse_context_csv: str = "rmse_load_context_per_building.csv",
) -> Path:
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_defended_comparison_contract_metadata(
        rmse_metric_label=rmse_metric_label,
        rmse_context_csv=rmse_context_csv,
    )
    metadata_path = report_dir / "comparison_contract_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def clone_experiment_config(
    config: ExperimentConfig,
    **overrides: Any,
) -> ExperimentConfig:
    config_kwargs = {name: getattr(config, name) for name in config.__dataclass_fields__.keys()}
    config_kwargs.update(overrides)
    return ExperimentConfig(**config_kwargs)


def build_mode_semantics_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_set = set(MODE_TEMPORAL_FEATURES["M0"])
    weather_memory_set = set(LSTM_WEATHER_MEMORY_FEATURES)
    system_dynamic_set = set(LSTM_SYSTEM_DYNAMIC_FEATURES)
    static_set = set(LSTM_STATIC_FEATURES_SETB)
    for mode in CANONICAL_MODE_ORDER:
        dynamic_cols = MODE_TEMPORAL_FEATURES[mode]
        dynamic_set = set(dynamic_cols)
        rows.append(
            {
                "mode": mode,
                "feature_frame": "setB" if mode == "M4" else "setA",
                "dynamic_feature_count": int(len(dynamic_cols)),
                "base_temporal_core_count": int(len(dynamic_set & base_set)),
                "weather_memory_feature_count": int(len(dynamic_set & weather_memory_set)),
                "system_dynamic_feature_count": int(len(dynamic_set & system_dynamic_set)),
                "static_feature_count": int(len(static_set) if mode == "M4" else 0),
                "uses_weather_memory": bool(len(dynamic_set & weather_memory_set) > 0),
                "uses_system_inertia": bool(len(dynamic_set & system_dynamic_set) > 0),
                "uses_static_branch": bool(mode == "M4"),
                "description": MODE_DESCRIPTIONS[mode],
            }
        )
    return pd.DataFrame(rows)


def build_weather_mode_semantics_table(
    config: ExperimentConfig | None = None,
    *,
    base_frames: dict[str, dict[str, pd.DataFrame]] | None = None,
    sample_building: str = "U05",
    sample_horizon_h: int = 24,
) -> pd.DataFrame:
    config = config or ExperimentConfig()
    if sample_building not in config.buildings:
        sample_building = str(config.buildings[0])

    if base_frames is None:
        sample_config = clone_experiment_config(
            config,
            buildings=(sample_building,),
            horizons=(int(sample_horizon_h),),
        )
        base_frames = build_base_frame_cache(sample_config)

    sample_frame = base_frames[sample_building]["setA"].copy()
    rows: list[dict[str, Any]] = []
    for weather_mode in CANONICAL_WEATHER_MODE_ORDER:
        frame_out, fw_cols = apply_weather_mode(sample_frame.copy(), weather_mode, int(sample_horizon_h))
        desc = WEATHER_MODE_DESCRIPTIONS[weather_mode]
        rows.append(
            {
                "weather_mode": weather_mode,
                "sample_building": sample_building,
                "sample_horizon_h": int(sample_horizon_h),
                "future_weather_feature_count": int(len(fw_cols)),
                "sample_feature_example": ", ".join(fw_cols[:6]),
                "future_weather_origin": desc["future_weather_origin"],
                "operational_status": desc["operational_status"],
                "upper_bound_only": bool(desc["upper_bound_only"]),
                "forecast_like": bool(desc["forecast_like"]),
                "exported_in_regular_feature_csv": bool(desc["exported_in_regular_feature_csv"]),
                "built_in_memory_at_runtime": bool(desc["built_in_memory_at_runtime"]),
                "description": desc["description"],
                "sample_frame_rows": int(len(frame_out)),
            }
        )
    return pd.DataFrame(rows)


def annotate_comparison_scope(
    df: pd.DataFrame,
    *,
    aggregated: bool = False,
    artifact_kind: str = "evaluation",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    regime = work["regime"].astype(str)
    artifact_kind = str(artifact_kind)
    if aggregated or artifact_kind == "portfolio_summary" or "building" not in work.columns:
        work["evaluation_building"] = pd.NA
        work["training_scope"] = np.where(regime.eq("pooled_same_buildings"), "POOLED", "PER_BUILDING_MODEL")
        work["evaluation_level"] = "portfolio_summary"
        work["scope_label"] = np.where(
            regime.eq("pooled_same_buildings"),
            "pooled training, summary aggregated across evaluation buildings",
            "per-building training, summary aggregated across evaluation buildings",
        )
    elif artifact_kind == "training_run":
        building = work["building"].astype(str)
        work["evaluation_building"] = np.where(regime.eq("pooled_same_buildings"), pd.NA, building)
        work["training_scope"] = np.where(regime.eq("pooled_same_buildings"), "POOLED", building)
        work["evaluation_level"] = "training_run"
        work["scope_label"] = np.where(
            regime.eq("pooled_same_buildings"),
            "pooled training run",
            "per-building training run",
        )
    else:
        building = work["building"].astype(str)
        work["evaluation_building"] = building
        work["training_scope"] = np.where(regime.eq("pooled_same_buildings"), "POOLED", building)
        work["evaluation_level"] = "building"
        work["scope_label"] = np.where(
            regime.eq("pooled_same_buildings"),
            "pooled training, building-level evaluation",
            "per-building training, same-building evaluation",
        )
    work["training_scope_kind"] = np.where(regime.eq("pooled_same_buildings"), "pooled_same_buildings", "per_building")
    return work


def build_mode_delta_tables(
    metrics_df: pd.DataFrame,
    *,
    mode_pairs: tuple[tuple[str, str], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    detailed_rows: list[dict[str, Any]] = []
    group_cols = ["regime", "building", "model_family", "weather_mode", "horizon_h"]
    for keys, group_df in metrics_df.groupby(group_cols, dropna=False):
        mode_lookup = {
            str(mode): row
            for mode, row in group_df.set_index("mode").iterrows()
        }
        for lhs_mode, rhs_mode in mode_pairs:
            if lhs_mode not in mode_lookup or rhs_mode not in mode_lookup:
                continue
            lhs_row = mode_lookup[lhs_mode]
            rhs_row = mode_lookup[rhs_mode]
            row = {col: value for col, value in zip(group_cols, keys)}
            row["comparison"] = f"{lhs_mode} - {rhs_mode}"
            row["lhs_mode"] = lhs_mode
            row["rhs_mode"] = rhs_mode
            row["training_scope"] = "POOLED" if str(row["regime"]) == "pooled_same_buildings" else str(row["building"])
            row["delta_wape_pct"] = float(lhs_row["wape_pct"] - rhs_row["wape_pct"])
            row["delta_rmse"] = float(lhs_row["rmse"] - rhs_row["rmse"])
            row["delta_r2"] = float(lhs_row["r2"] - rhs_row["r2"])
            row["delta_mae"] = float(lhs_row["mae"] - rhs_row["mae"])
            detailed_rows.append(row)

    detailed_df = pd.DataFrame(detailed_rows)
    if detailed_df.empty:
        return detailed_df, pd.DataFrame()

    summary_df = (
        detailed_df.groupby(
            ["regime", "model_family", "weather_mode", "horizon_h", "comparison", "lhs_mode", "rhs_mode"],
            as_index=False,
        )
        .agg(
            n_buildings=("building", "nunique"),
            mean_delta_wape_pct=("delta_wape_pct", "mean"),
            median_delta_wape_pct=("delta_wape_pct", "median"),
            buildings_improved_wape=("delta_wape_pct", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
            mean_delta_rmse=("delta_rmse", "mean"),
            mean_delta_r2=("delta_r2", "mean"),
            mean_delta_mae=("delta_mae", "mean"),
        )
        .sort_values(["regime", "model_family", "weather_mode", "horizon_h", "lhs_mode", "rhs_mode"])
        .reset_index(drop=True)
    )
    summary_df["scope_label"] = np.where(
        summary_df["regime"].astype(str).eq("pooled_same_buildings"),
        "pooled training, building-level evaluation",
        "per-building training, same-building evaluation",
    )
    return detailed_df.sort_values(group_cols + ["lhs_mode", "rhs_mode"]).reset_index(drop=True), summary_df


def merge_comparison_outputs(*outputs: ComparisonOutputs) -> ComparisonOutputs:
    merged = ComparisonOutputs()
    for output in outputs:
        if output is None:
            continue
        merged.manifest_df = _append_frame(merged.manifest_df, output.manifest_df)
        merged.comparison_predictions_df = _append_frame(merged.comparison_predictions_df, output.comparison_predictions_df)
        merged.comparison_metrics_df = _append_frame(merged.comparison_metrics_df, output.comparison_metrics_df)
        merged.comparison_coverage_df = _append_frame(merged.comparison_coverage_df, output.comparison_coverage_df)
        merged.run_log_df = _append_frame(merged.run_log_df, output.run_log_df)
        merged.lstm_training_history_df = _append_frame(merged.lstm_training_history_df, output.lstm_training_history_df)
        merged.xgb_eval_history_df = _append_frame(merged.xgb_eval_history_df, output.xgb_eval_history_df)
    return _normalize_outputs(merged)


def latest_comparison_run_log(run_log_df: pd.DataFrame) -> pd.DataFrame:
    if run_log_df.empty:
        return run_log_df.copy()
    key_cols = ["regime", "building", "model_family", "mode", "weather_mode", "horizon_h"]
    work = run_log_df.copy()
    work["slot_index"] = pd.to_numeric(work.get("slot_index", np.nan), errors="coerce")
    if "elapsed_s" in work.columns:
        work["elapsed_s"] = pd.to_numeric(work["elapsed_s"], errors="coerce")
    status_rank = work["status"].astype(str).map({"ok": 3, "skipped_insufficient_sequences": 2, "skipped_insufficient_rows": 2, "error": 1}).fillna(0)
    work["_status_rank"] = status_rank.astype(int)
    sort_cols = [col for col in ["slot_index", "_status_rank", "elapsed_s"] if col in work.columns]
    work = work.sort_values(key_cols + sort_cols)
    return work.drop_duplicates(subset=key_cols, keep="last").drop(columns=["_status_rank"]).reset_index(drop=True)


def export_report_ready_model_family_artifacts(
    outputs: ComparisonOutputs,
    config: ExperimentConfig,
    report_dir: Path,
    *,
    main_modes: tuple[str, ...],
    transition_mode_pairs: tuple[tuple[str, str], ...],
    focus_weather_mode: str = "FW2",
    sample_building: str = "U05",
    sample_horizon_h: int = 24,
    desired_manifest_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_config = clone_experiment_config(config, results_dir=report_dir)
    normalized = _normalize_outputs(outputs)
    _write_outputs(ensure_results_dirs(report_config), normalized)
    rmse_context_df = build_rmse_load_context_table(normalized.comparison_predictions_df)
    rmse_context_path = report_dir / "rmse_load_context_per_building.csv"
    rmse_context_df.to_csv(rmse_context_path, index=False)
    write_defended_comparison_contract_metadata(
        report_dir=report_dir,
        rmse_metric_label="Cumulative RMSE (kWh)",
        rmse_context_csv=rmse_context_path.name,
    )

    base_frames = build_base_frame_cache(
        clone_experiment_config(config, buildings=(sample_building,), horizons=(int(sample_horizon_h),))
    )
    mode_semantics_df = build_mode_semantics_table()
    weather_mode_semantics_df = build_weather_mode_semantics_table(
        config,
        base_frames=base_frames,
        sample_building=sample_building,
        sample_horizon_h=sample_horizon_h,
    )
    manifest_report_df = annotate_comparison_scope(normalized.manifest_df, artifact_kind="training_run")
    metrics_report_df = annotate_comparison_scope(normalized.comparison_metrics_df, artifact_kind="evaluation")
    coverage_report_df = annotate_comparison_scope(normalized.comparison_coverage_df, artifact_kind="evaluation")
    summary_report_df = annotate_comparison_scope(normalized.comparison_summary_df, artifact_kind="portfolio_summary", aggregated=True)
    summary_report_df = attach_defended_comparison_contract(
        summary_report_df,
        rmse_metric_label="Cumulative RMSE (kWh)",
        rmse_context_csv=rmse_context_path.name,
    )
    run_log_latest_df = annotate_comparison_scope(
        latest_comparison_run_log(normalized.run_log_df),
        artifact_kind="training_run",
    )
    desired_manifest_scope_df = (
        _normalize_comparison_manifest_frame(desired_manifest_df)
        if desired_manifest_df is not None
        else normalized.manifest_df.copy()
    )
    inventory_df = build_comparison_matrix_inventory(
        desired_manifest_scope_df,
        normalized.manifest_df,
        source_dir=report_dir,
        notebook="12",
        layer="model_family_comparison",
    )
    missing_manifest_df = build_missing_comparison_manifest(
        desired_manifest_scope_df,
        normalized.manifest_df,
    )

    thesis_main_summary_df = summary_report_df.loc[
        (summary_report_df["regime"].astype(str) == "per_building")
        & (summary_report_df["mode"].astype(str).isin(main_modes))
        & (summary_report_df["weather_mode"].astype(str) == str(focus_weather_mode))
    ].copy()
    thesis_pooled_summary_df = summary_report_df.loc[
        (summary_report_df["regime"].astype(str) == "pooled_same_buildings")
        & (summary_report_df["mode"].astype(str).isin(main_modes))
        & (summary_report_df["weather_mode"].astype(str) == str(focus_weather_mode))
    ].copy()
    mode_delta_detail_df, mode_delta_summary_df = build_mode_delta_tables(
        normalized.comparison_metrics_df,
        mode_pairs=transition_mode_pairs,
    )

    exports = {
        "comparison_manifest_report.csv": manifest_report_df,
        "mode_semantics.csv": mode_semantics_df,
        "weather_mode_semantics.csv": weather_mode_semantics_df,
        "comparison_metrics_report.csv": metrics_report_df,
        "comparison_coverage_report.csv": coverage_report_df,
        "comparison_summary_report.csv": summary_report_df,
        "run_log_latest_report.csv": run_log_latest_df,
        "thesis_per_building_summary.csv": thesis_main_summary_df,
        "thesis_pooled_summary.csv": thesis_pooled_summary_df,
        "mode_delta_detail.csv": mode_delta_detail_df,
        "mode_delta_summary.csv": mode_delta_summary_df,
        "comparison_matrix_inventory.csv": inventory_df,
        "comparison_manifest_missing.csv": missing_manifest_df,
    }
    for filename, df in exports.items():
        df.to_csv(report_dir / filename, index=False)
    return exports


def validate_saved_metric_recompute(
    outputs: ComparisonOutputs,
    *,
    report_dir: Path,
    building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
) -> pd.DataFrame:
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    predictions_df = outputs.comparison_predictions_df.copy()
    metrics_df = outputs.comparison_metrics_df.copy()
    if predictions_df.empty or metrics_df.empty:
        validation_df = pd.DataFrame()
        validation_df.to_csv(report_dir / "metric_recompute_validation.csv", index=False)
        return validation_df

    rows: list[dict[str, Any]] = []
    regimes = [
        regime
        for regime in DEFAULT_REGIMES
        if regime in set(metrics_df["regime"].astype(str))
    ]
    model_families = sorted(metrics_df["model_family"].astype(str).unique().tolist())

    for regime in regimes:
        for model_family in model_families:
            pred_slice = predictions_df.loc[
                (predictions_df["regime"].astype(str) == str(regime))
                & (predictions_df["building"].astype(str) == str(building))
                & (predictions_df["model_family"].astype(str) == str(model_family))
                & (predictions_df["mode"].astype(str) == str(mode))
                & (predictions_df["weather_mode"].astype(str) == str(weather_mode))
                & (pd.to_numeric(predictions_df["horizon_h"], errors="coerce") == int(horizon_h))
            ].copy()
            metric_slice = metrics_df.loc[
                (metrics_df["regime"].astype(str) == str(regime))
                & (metrics_df["building"].astype(str) == str(building))
                & (metrics_df["model_family"].astype(str) == str(model_family))
                & (metrics_df["mode"].astype(str) == str(mode))
                & (metrics_df["weather_mode"].astype(str) == str(weather_mode))
                & (pd.to_numeric(metrics_df["horizon_h"], errors="coerce") == int(horizon_h))
            ].copy()
            if pred_slice.empty or metric_slice.empty:
                continue

            eval_slice = pred_slice.copy()
            if "is_heating_eval" in eval_slice.columns:
                eval_mask = eval_slice["is_heating_eval"].fillna(False).astype(bool)
                if eval_mask.any():
                    eval_slice = eval_slice.loc[eval_mask].copy()
            recomputed = compute_regression_metrics(
                eval_slice["y_true"].to_numpy(dtype=float),
                eval_slice["y_pred"].to_numpy(dtype=float),
            )
            metric_row = metric_slice.iloc[0]
            row = {
                "regime": regime,
                "building": str(building),
                "model_family": model_family,
                "mode": str(mode),
                "weather_mode": str(weather_mode),
                "horizon_h": int(horizon_h),
                "n_prediction_rows": int(len(pred_slice)),
                "n_eval_rows": int(len(eval_slice)),
            }
            max_abs_diff = 0.0
            for metric_name in ("rmse", "wape_pct", "r2", "mae"):
                saved_value = float(metric_row[metric_name])
                recomputed_value = float(recomputed[metric_name])
                diff = recomputed_value - saved_value
                row[f"saved_{metric_name}"] = saved_value
                row[f"recomputed_{metric_name}"] = recomputed_value
                row[f"delta_{metric_name}"] = diff
                max_abs_diff = max(max_abs_diff, abs(diff))
            row["max_abs_diff"] = max_abs_diff
            row["matches_within_1e-9"] = bool(max_abs_diff <= 1e-9)
            rows.append(row)

    validation_df = (
        pd.DataFrame(rows).sort_values(["regime", "model_family"]).reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )
    validation_df.to_csv(report_dir / "metric_recompute_validation.csv", index=False)
    return validation_df


def _pair_mask_common(
    df: pd.DataFrame,
    regime: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    mask = (
        df["regime"].astype(str).eq(regime)
        & df["mode"].astype(str).eq(mode)
        & df["weather_mode"].astype(str).eq(weather_mode)
        & pd.to_numeric(df["horizon_h"], errors="coerce").eq(int(horizon_h))
    )
    return mask


def _pair_mask_eval_artifact(
    df: pd.DataFrame,
    regime: str,
    scope_building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
) -> pd.Series:
    mask = _pair_mask_common(df, regime, mode, weather_mode, horizon_h)
    if df.empty:
        return mask
    if regime == "per_building" and "building" in df.columns:
        mask &= df["building"].astype(str).eq(scope_building)
    return mask


def _pair_mask_run_artifact(
    df: pd.DataFrame,
    regime: str,
    scope_building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
) -> pd.Series:
    mask = _pair_mask_common(df, regime, mode, weather_mode, horizon_h)
    if df.empty:
        return mask
    run_building = scope_building if regime == "per_building" else "POOLED"
    if "building" in df.columns:
        mask &= df["building"].astype(str).eq(run_building)
    return mask


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy() if not df.empty else pd.DataFrame()


def _drop_pair_rows(
    outputs: ComparisonOutputs,
    *,
    regime: str,
    scope_building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
) -> ComparisonOutputs:
    predictions_df = outputs.comparison_predictions_df.loc[
        ~_pair_mask_eval_artifact(outputs.comparison_predictions_df, regime, scope_building, mode, weather_mode, horizon_h)
    ].copy() if not outputs.comparison_predictions_df.empty else _empty_like(outputs.comparison_predictions_df)
    metrics_df = outputs.comparison_metrics_df.loc[
        ~_pair_mask_eval_artifact(outputs.comparison_metrics_df, regime, scope_building, mode, weather_mode, horizon_h)
    ].copy() if not outputs.comparison_metrics_df.empty else _empty_like(outputs.comparison_metrics_df)
    coverage_df = outputs.comparison_coverage_df.loc[
        ~_pair_mask_eval_artifact(outputs.comparison_coverage_df, regime, scope_building, mode, weather_mode, horizon_h)
    ].copy() if not outputs.comparison_coverage_df.empty else _empty_like(outputs.comparison_coverage_df)
    run_log_df = outputs.run_log_df.loc[
        ~_pair_mask_run_artifact(outputs.run_log_df, regime, scope_building, mode, weather_mode, horizon_h)
    ].copy() if not outputs.run_log_df.empty else _empty_like(outputs.run_log_df)
    lstm_history_df = outputs.lstm_training_history_df.loc[
        ~_pair_mask_run_artifact(outputs.lstm_training_history_df, regime, scope_building, mode, weather_mode, horizon_h)
    ].copy() if not outputs.lstm_training_history_df.empty else _empty_like(outputs.lstm_training_history_df)
    xgb_history_df = outputs.xgb_eval_history_df.loc[
        ~_pair_mask_run_artifact(outputs.xgb_eval_history_df, regime, scope_building, mode, weather_mode, horizon_h)
    ].copy() if not outputs.xgb_eval_history_df.empty else _empty_like(outputs.xgb_eval_history_df)
    return ComparisonOutputs(
        manifest_df=outputs.manifest_df.copy(),
        comparison_predictions_df=predictions_df,
        comparison_metrics_df=metrics_df,
        comparison_summary_df=pd.DataFrame(),
        comparison_coverage_df=coverage_df,
        run_log_df=run_log_df,
        lstm_training_history_df=lstm_history_df,
        xgb_eval_history_df=xgb_history_df,
    )


def _append_frame(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new_df.copy()
    if new_df.empty:
        return existing.copy()
    return pd.concat([existing, new_df], ignore_index=True)


def _expected_pair_artifact_rows(config: ExperimentConfig, regime: str) -> tuple[int, int]:
    n_buildings = 1 if regime == "per_building" else len(config.buildings)
    return 2 * n_buildings, n_buildings


def _pair_slices_complete(
    metrics_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    run_log_df: pd.DataFrame,
    config: ExperimentConfig,
    *,
    regime: str,
) -> bool:
    expected_metric_rows, expected_coverage_rows = _expected_pair_artifact_rows(config, regime)
    if not run_log_df.empty:
        terminal_statuses = {"ok", "skipped_insufficient_sequences", "skipped_insufficient_rows"}
        terminal_logs = run_log_df[
            run_log_df["model_family"].astype(str).isin(config.model_families)
            & run_log_df["status"].astype(str).isin(terminal_statuses)
        ]
        if terminal_logs["model_family"].astype(str).nunique() == len(config.model_families) and not terminal_logs["status"].astype(str).eq("ok").all():
            return True

    if len(metrics_df) != expected_metric_rows:
        return False
    if len(coverage_df) != expected_coverage_rows:
        return False
    if predictions_df.empty:
        return False
    expected_families = set(config.model_families)
    if set(metrics_df["model_family"].astype(str).unique()) != expected_families:
        return False
    if set(predictions_df["model_family"].astype(str).unique()) != expected_families:
        return False
    ok_logs = run_log_df[
        run_log_df["model_family"].astype(str).isin(config.model_families)
        & run_log_df["status"].astype(str).eq("ok")
    ] if not run_log_df.empty else pd.DataFrame()
    if ok_logs["model_family"].astype(str).nunique() < len(expected_families):
        return False
    if "n_test_rows_common" in coverage_df.columns and (pd.to_numeric(coverage_df["n_test_rows_common"], errors="coerce") <= 0).any():
        return False
    return True


def _comparison_slot_id(regime: str, scope_building: str, mode: str, weather_mode: str, horizon_h: int) -> str:
    h = int(horizon_h)
    if regime == "per_building":
        return f"pb|{scope_building}|{mode}|{weather_mode}|{h}"
    return f"pool|{mode}|{weather_mode}|{h}"


def _assign_comparison_slot_id(df: pd.DataFrame) -> pd.Series:
    if df.empty or "regime" not in df.columns:
        return pd.Series(dtype=str, index=df.index)
    r = df["regime"].astype(str)
    mode = df["mode"].astype(str)
    wm = df["weather_mode"].astype(str)
    h = pd.to_numeric(df["horizon_h"], errors="coerce").fillna(-1).astype(int).astype(str)
    b = df["building"].astype(str) if "building" in df.columns else pd.Series("", index=df.index)
    is_pb = r.eq("per_building")
    slot_pb = "pb|" + b + "|" + mode + "|" + wm + "|" + h
    slot_pool = "pool|" + mode + "|" + wm + "|" + h
    return slot_pb.where(is_pb, slot_pool)


def _artifact_frames_by_comparison_slot(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    work = df.copy()
    work["_cmp_slot"] = _assign_comparison_slot_id(work)
    out: dict[str, pd.DataFrame] = {}
    for key, grp in work.groupby("_cmp_slot", sort=False):
        out[str(key)] = grp.drop(columns=["_cmp_slot"])
    return out


def _comparison_pair_plan_df(
    config: ExperimentConfig,
    *,
    manifest_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    plan_df = manifest_df.copy() if manifest_df is not None else build_comparison_manifest(config)
    if plan_df.empty:
        return pd.DataFrame(columns=COMPARISON_PAIR_KEY_COLS)
    work = plan_df.copy()
    for col in COMPARISON_PAIR_KEY_COLS:
        if col not in work.columns:
            work[col] = pd.NA
    work["horizon_h"] = pd.to_numeric(work["horizon_h"], errors="coerce")
    work = work.loc[work["horizon_h"].notna(), COMPARISON_PAIR_KEY_COLS].copy()
    if work.empty:
        return pd.DataFrame(columns=COMPARISON_PAIR_KEY_COLS)
    work["regime"] = work["regime"].astype(str)
    work["building"] = work["building"].astype(str)
    work["mode"] = work["mode"].astype(str)
    work["weather_mode"] = work["weather_mode"].astype(str)
    work["horizon_h"] = work["horizon_h"].astype(int)
    return work.drop_duplicates().sort_values(COMPARISON_PAIR_KEY_COLS).reset_index(drop=True)


def _normalize_comparison_manifest_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COMPARISON_MANIFEST_KEY_COLS + ["run_id"])
    work = df.copy()
    for col in COMPARISON_MANIFEST_KEY_COLS:
        if col not in work.columns:
            work[col] = pd.NA
    if "run_id" not in work.columns:
        work["run_id"] = pd.NA
    work["horizon_h"] = pd.to_numeric(work["horizon_h"], errors="coerce")
    work = work.loc[work["horizon_h"].notna(), COMPARISON_MANIFEST_KEY_COLS + ["run_id"]].copy()
    if work.empty:
        return pd.DataFrame(columns=COMPARISON_MANIFEST_KEY_COLS + ["run_id"])
    for col in ("regime", "building", "model_family", "mode", "weather_mode"):
        work[col] = work[col].astype(str)
    work["horizon_h"] = work["horizon_h"].astype(int)
    return (
        work.drop_duplicates(subset=COMPARISON_MANIFEST_KEY_COLS, keep="last")
        .sort_values(COMPARISON_MANIFEST_KEY_COLS)
        .reset_index(drop=True)
    )


def build_missing_comparison_manifest(
    desired_manifest_df: pd.DataFrame,
    current_manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    desired = _normalize_comparison_manifest_frame(desired_manifest_df)
    current = _normalize_comparison_manifest_frame(current_manifest_df)
    if desired.empty:
        return desired.copy()
    current_keys = current.loc[:, COMPARISON_MANIFEST_KEY_COLS].drop_duplicates()
    missing = desired.merge(
        current_keys.assign(_present=True),
        on=COMPARISON_MANIFEST_KEY_COLS,
        how="left",
    )
    missing = missing.loc[missing["_present"].fillna(False) == False].drop(columns="_present")
    return missing.reset_index(drop=True)


def build_comparison_matrix_inventory(
    desired_manifest_df: pd.DataFrame,
    current_manifest_df: pd.DataFrame,
    *,
    source_dir: Path | str,
    notebook: str = "12",
    layer: str = "model_family_comparison",
) -> pd.DataFrame:
    desired = _normalize_comparison_manifest_frame(desired_manifest_df)
    current = _normalize_comparison_manifest_frame(current_manifest_df)
    if desired.empty:
        return pd.DataFrame(
            columns=[
                "notebook",
                "layer",
                "source_dir",
                "model_family_scope",
                "regime",
                "building_scope",
                "mode",
                "weather_mode",
                "horizon_h",
                "seed",
                "target_kind",
                "status",
                "completed_now",
                "desired_full_matrix",
                "notes",
                "building",
                "model_family",
                "run_id",
            ]
        )
    source_path = str(Path(source_dir).resolve())
    inventory = desired.merge(
        current.loc[:, COMPARISON_MANIFEST_KEY_COLS].drop_duplicates().assign(completed_now=True),
        on=COMPARISON_MANIFEST_KEY_COLS,
        how="left",
    )
    inventory["completed_now"] = inventory["completed_now"].fillna(False).astype(bool)
    inventory["notebook"] = str(notebook)
    inventory["layer"] = str(layer)
    inventory["source_dir"] = source_path
    inventory["model_family_scope"] = inventory["model_family"].astype(str)
    inventory["building_scope"] = inventory["building"].astype(str)
    inventory["seed"] = pd.NA
    inventory["target_kind"] = "comparison"
    inventory["status"] = np.where(inventory["completed_now"], "complete", "missing")
    inventory["desired_full_matrix"] = True
    inventory["notes"] = np.where(
        inventory["completed_now"],
        "Present in the current comparison manifest.",
        "Missing from the current comparison manifest; safe to schedule for resume-only execution.",
    )
    ordered_cols = [
        "notebook",
        "layer",
        "source_dir",
        "model_family_scope",
        "regime",
        "building_scope",
        "mode",
        "weather_mode",
        "horizon_h",
        "seed",
        "target_kind",
        "status",
        "completed_now",
        "desired_full_matrix",
        "notes",
        "building",
        "model_family",
        "run_id",
    ]
    return inventory.loc[:, ordered_cols].sort_values(
        ["regime", "building_scope", "model_family_scope", "mode", "weather_mode", "horizon_h"]
    ).reset_index(drop=True)


def comparison_manifest_overlap_report(
    subset_manifest_df: pd.DataFrame,
    superset_manifest_df: pd.DataFrame,
    *,
    subset_label: str,
    superset_label: str,
) -> pd.DataFrame:
    subset = _normalize_comparison_manifest_frame(subset_manifest_df)
    superset = _normalize_comparison_manifest_frame(superset_manifest_df)
    missing = build_missing_comparison_manifest(subset, superset)
    return pd.DataFrame(
        [
            {
                "subset_label": str(subset_label),
                "superset_label": str(superset_label),
                "subset_rows": int(len(subset)),
                "superset_rows": int(len(superset)),
                "missing_rows": int(len(missing)),
                "is_subset": bool(missing.empty),
            }
        ]
    )


def _completed_comparison_slots(
    outputs: ComparisonOutputs,
    config: ExperimentConfig,
    *,
    manifest_df: pd.DataFrame | None = None,
) -> set[str]:
    metrics_by = _artifact_frames_by_comparison_slot(outputs.comparison_metrics_df)
    coverage_by = _artifact_frames_by_comparison_slot(outputs.comparison_coverage_df)
    pred_by = _artifact_frames_by_comparison_slot(outputs.comparison_predictions_df)
    run_by = _artifact_frames_by_comparison_slot(outputs.run_log_df)

    completed_slots: set[str] = set()
    pair_plan_df = _comparison_pair_plan_df(config, manifest_df=manifest_df)
    for row in pair_plan_df.itertuples(index=False):
        slot_id = _comparison_slot_id(
            str(row.regime),
            str(row.building),
            str(row.mode),
            str(row.weather_mode),
            int(row.horizon_h),
        )
        if _pair_slices_complete(
            metrics_by.get(slot_id, pd.DataFrame()),
            coverage_by.get(slot_id, pd.DataFrame()),
            pred_by.get(slot_id, pd.DataFrame()),
            run_by.get(slot_id, pd.DataFrame()),
            config,
            regime=str(row.regime),
        ):
            completed_slots.add(slot_id)
    return completed_slots


def _csv_data_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        n_lines = sum(1 for _ in f)
    return max(0, n_lines - 1)


def print_resume_diagnostics(
    config: ExperimentConfig,
    *,
    manifest_df: pd.DataFrame | None = None,
) -> None:
    """Print where artifacts live, CSV sizes, and pair-level resume progress (fast path)."""
    paths = ensure_results_dirs(config)
    print("--- resume diagnostics ---")
    print(f"cwd: {Path.cwd()}")
    print(f"results_dir (resolved): {config.results_dir.resolve()}")
    for key in ("manifest", "metrics", "predictions", "coverage", "run_log"):
        p = paths[key]
        print(f"  {paths[key].name}: exists={p.exists()} data_rows={_csv_data_row_count(p)}")

    manifest_scope_df = manifest_df.copy() if manifest_df is not None else build_comparison_manifest(config)
    outputs = load_saved_outputs(config, manifest_df=manifest_scope_df)
    metrics_by = _artifact_frames_by_comparison_slot(outputs.comparison_metrics_df)
    coverage_by = _artifact_frames_by_comparison_slot(outputs.comparison_coverage_df)
    pred_by = _artifact_frames_by_comparison_slot(outputs.comparison_predictions_df)
    run_by = _artifact_frames_by_comparison_slot(outputs.run_log_df)

    pair_plan_df = _comparison_pair_plan_df(config, manifest_df=manifest_scope_df)
    total_pairs = int(len(pair_plan_df))
    n_complete = 0
    first_bad: tuple[int, str, str, str, str, int] | None = None
    for idx, row in enumerate(pair_plan_df.itertuples(index=False), start=1):
        sid = _comparison_slot_id(
            str(row.regime),
            str(row.building),
            str(row.mode),
            str(row.weather_mode),
            int(row.horizon_h),
        )
        ok = _pair_slices_complete(
            metrics_by.get(sid, pd.DataFrame()),
            coverage_by.get(sid, pd.DataFrame()),
            pred_by.get(sid, pd.DataFrame()),
            run_by.get(sid, pd.DataFrame()),
            config,
            regime=str(row.regime),
        )
        if ok:
            n_complete += 1
        elif first_bad is None:
            first_bad = (
                idx,
                str(row.regime),
                str(row.building),
                str(row.mode),
                str(row.weather_mode),
                int(row.horizon_h),
            )
    print(f"pair slots complete: {n_complete}/{total_pairs} (same rules as pair_is_complete)")
    if first_bad is None:
        print("first incomplete slot: (none — all pairs complete)")
    else:
        i, r, s, m, w, h = first_bad
        print(
            f"first incomplete slot: [{i}/{total_pairs}] regime={r} scope={s} mode={m} weather={w} h={h}"
        )
    print("--- end resume diagnostics ---")


def pair_is_complete(
    outputs: ComparisonOutputs,
    config: ExperimentConfig,
    *,
    regime: str,
    scope_building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
) -> bool:
    metrics_df = outputs.comparison_metrics_df.loc[
        _pair_mask_eval_artifact(outputs.comparison_metrics_df, regime, scope_building, mode, weather_mode, horizon_h)
    ].copy() if not outputs.comparison_metrics_df.empty else pd.DataFrame()
    coverage_df = outputs.comparison_coverage_df.loc[
        _pair_mask_eval_artifact(outputs.comparison_coverage_df, regime, scope_building, mode, weather_mode, horizon_h)
    ].copy() if not outputs.comparison_coverage_df.empty else pd.DataFrame()
    predictions_df = outputs.comparison_predictions_df.loc[
        _pair_mask_eval_artifact(outputs.comparison_predictions_df, regime, scope_building, mode, weather_mode, horizon_h)
    ].copy() if not outputs.comparison_predictions_df.empty else pd.DataFrame()
    run_log_df = outputs.run_log_df.loc[
        _pair_mask_run_artifact(outputs.run_log_df, regime, scope_building, mode, weather_mode, horizon_h)
    ].copy() if not outputs.run_log_df.empty else pd.DataFrame()
    return _pair_slices_complete(metrics_df, coverage_df, predictions_df, run_log_df, config, regime=regime)


def _upsert_pair_outputs(
    outputs: ComparisonOutputs,
    *,
    regime: str,
    scope_building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    run_log_df: pd.DataFrame,
    lstm_history_df: pd.DataFrame,
    xgb_history_df: pd.DataFrame,
    normalize: bool = True,
) -> ComparisonOutputs:
    cleared = _drop_pair_rows(
        outputs,
        regime=regime,
        scope_building=scope_building,
        mode=mode,
        weather_mode=weather_mode,
        horizon_h=horizon_h,
    )
    updated = ComparisonOutputs(
        manifest_df=outputs.manifest_df.copy(),
        comparison_predictions_df=_append_frame(cleared.comparison_predictions_df, predictions_df),
        comparison_metrics_df=_append_frame(cleared.comparison_metrics_df, metrics_df),
        comparison_summary_df=pd.DataFrame(),
        comparison_coverage_df=_append_frame(cleared.comparison_coverage_df, coverage_df),
        run_log_df=_append_frame(cleared.run_log_df, run_log_df),
        lstm_training_history_df=_append_frame(cleared.lstm_training_history_df, lstm_history_df),
        xgb_eval_history_df=_append_frame(cleared.xgb_eval_history_df, xgb_history_df),
    )
    return _normalize_outputs(updated) if normalize else updated


def run_id_from_spec(run_spec: RunSpec) -> str:
    return (
        f"{run_spec.regime}__{run_spec.building}__{run_spec.model_family}__"
        f"{run_spec.mode}__{run_spec.weather_mode}__h{int(run_spec.horizon_h):02d}"
    )


def htag(horizon_h: int) -> str:
    return f"h{int(horizon_h):02d}"


def feature_set_for_mode(mode: str) -> str:
    return "setB" if mode == "M4" else "setA"


def load_feature_metadata(path: Path = FEATURE_METADATA_FILE) -> pd.DataFrame:
    meta = pd.read_csv(path)
    required = {"building", "path_setA", "path_setB"}
    missing = required.difference(meta.columns)
    if missing:
        raise KeyError(f"Feature metadata missing required columns: {sorted(missing)}")
    return meta


def load_feature_catalog(path: Path = FEATURE_CATALOG_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def feature_path_for_building(building: str, set_name: str, meta: pd.DataFrame | None = None) -> Path:
    meta = meta if meta is not None else load_feature_metadata()
    row = meta.loc[meta["building"] == building]
    if row.empty:
        raise KeyError(f"Building {building} missing from feature metadata")
    rel_path = str(row.iloc[0][f"path_{set_name}"])
    csv_path = PROJECT_ROOT / rel_path
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return csv_path


def load_feature_frame(building: str, set_name: str, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    csv_path = feature_path_for_building(building, set_name, meta=meta)
    df = pd.read_csv(csv_path, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return pd.concat([df, pd.Series(building, index=df.index, name="building")], axis=1)


def add_cumulative_targets(df: pd.DataFrame, horizons: Iterable[int], source_col: str = "heat_kwh") -> pd.DataFrame:
    out = df.copy()
    heat = pd.to_numeric(out[source_col], errors="coerce")
    target_cols: dict[str, pd.Series] = {}
    for horizon_h in horizons:
        target_cols[f"target_cum_h{int(horizon_h)}"] = sum(heat.shift(-i) for i in range(int(horizon_h)))
    return pd.concat([out, pd.DataFrame(target_cols, index=out.index)], axis=1)


def build_base_frame_cache(config: ExperimentConfig, meta: pd.DataFrame | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    meta = meta if meta is not None else load_feature_metadata()
    cache: dict[str, dict[str, pd.DataFrame]] = {}
    for building in config.buildings:
        frame_a = add_cumulative_targets(load_feature_frame(building, "setA", meta=meta), config.horizons)
        frame_b = add_cumulative_targets(load_feature_frame(building, "setB", meta=meta), config.horizons)
        cache[building] = {"setA": frame_a, "setB": frame_b}
    return cache


def build_future_summary_columns(series: pd.Series, horizon_h: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    future_mat = pd.concat([series.shift(-k) for k in range(1, int(horizon_h) + 1)], axis=1)
    return future_mat.mean(axis=1), future_mat.min(axis=1), series.shift(-int(horizon_h))


def build_future_path_columns(series: pd.Series, horizon_h: int, prefix: str) -> dict[str, pd.Series]:
    tag = htag(horizon_h)
    return {
        f"{prefix}_tplus{step:02d}_{tag}": series.shift(-step)
        for step in range(1, int(horizon_h) + 1)
    }


def build_oracle_future_weather_columns(df: pd.DataFrame, horizon_h: int) -> dict[str, pd.Series]:
    required = {"feat_outdoor_temp_c", "feat_rh_pct"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Oracle future weather requires columns: {missing}")
    tag = htag(horizon_h)
    out: dict[str, pd.Series] = {}
    temp_mean, temp_min, temp_end = build_future_summary_columns(df["feat_outdoor_temp_c"], horizon_h)
    rh_mean, _, rh_end = build_future_summary_columns(df["feat_rh_pct"], horizon_h)
    out[f"feat_fw_temp_mean_{tag}"] = temp_mean
    out[f"feat_fw_temp_min_{tag}"] = temp_min
    out[f"feat_fw_temp_end_{tag}"] = temp_end
    out[f"feat_fw_rh_mean_{tag}"] = rh_mean
    out[f"feat_fw_rh_end_{tag}"] = rh_end
    out.update(build_future_path_columns(df["feat_outdoor_temp_c"], horizon_h, "feat_fw_temp"))
    out.update(build_future_path_columns(df["feat_rh_pct"], horizon_h, "feat_fw_rh"))
    return out


def future_weather_feature_cols(columns: Iterable[str], horizon_h: int) -> list[str]:
    suffix = f"_{htag(horizon_h)}"
    return sorted([col for col in columns if str(col).startswith("feat_fw_") and str(col).endswith(suffix)])


def apply_weather_mode(frame: pd.DataFrame, weather_mode: str, horizon_h: int) -> tuple[pd.DataFrame, list[str]]:
    if weather_mode == "FW0":
        return frame, []

    fw_cols = future_weather_feature_cols(frame.columns, horizon_h)
    if weather_mode == "FW2":
        if not fw_cols:
            raise KeyError(f"No proxy future-weather columns found for horizon {horizon_h}")
        return frame, fw_cols

    if weather_mode == "FW1":
        out = frame.copy()
        oracle_cols = build_oracle_future_weather_columns(out, horizon_h)
        for col_name, values in oracle_cols.items():
            out[col_name] = values
        fw_cols = future_weather_feature_cols(out.columns, horizon_h)
        if not fw_cols:
            raise KeyError(f"No oracle future-weather columns created for horizon {horizon_h}")
        return out, fw_cols

    raise ValueError(f"Unknown weather mode: {weather_mode}")


def build_split_spec(frame: pd.DataFrame, horizon_h: int, config: ExperimentConfig) -> SplitSpec:
    dt = pd.to_datetime(frame["datetime"])
    train_issue_end = config.train_end - pd.Timedelta(hours=int(horizon_h) - 1)
    return SplitSpec(
        train_issue_end=train_issue_end,
        feature_train_mask=dt <= config.train_end,
        train_issue_mask=dt <= train_issue_end,
        test_issue_mask=dt >= config.test_start,
    )


def heating_mask(frame: pd.DataFrame) -> pd.Series:
    if "feat_is_heating_weather" in frame.columns:
        return pd.to_numeric(frame["feat_is_heating_weather"], errors="coerce").fillna(0.0) > 0.5
    if "feat_outdoor_temp_c" in frame.columns:
        return pd.to_numeric(frame["feat_outdoor_temp_c"], errors="coerce") < HEATING_TEMP_THRESHOLD_C
    return pd.Series(True, index=frame.index)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "wape_pct": np.nan}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    denom = float(np.sum(np.abs(y_true)))
    wape = float(100.0 * np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2, "wape_pct": wape}


def _metrics_from_prediction_frame(frame: pd.DataFrame) -> tuple[dict[str, float], int]:
    if frame.empty:
        metrics = compute_regression_metrics(np.array([]), np.array([]))
        return metrics, 0
    eval_mask = frame["is_heating_eval"].astype(bool).to_numpy()
    if not eval_mask.any():
        eval_mask = np.ones(len(frame), dtype=bool)
    y_true = frame.loc[eval_mask, "y_true"].to_numpy(dtype=float)
    y_pred = frame.loc[eval_mask, "y_pred"].to_numpy(dtype=float)
    return compute_regression_metrics(y_true, y_pred), int(eval_mask.sum())


def architecture_spec(config: ExperimentConfig) -> dict[str, Any]:
    if config.lstm_architecture_id not in ARCHITECTURES:
        raise KeyError(f"Unknown LSTM architecture id: {config.lstm_architecture_id}")
    spec = ARCHITECTURES[config.lstm_architecture_id].copy()
    spec["lookback_hours"] = int(config.lookback_hours)
    return spec


def xgb_preset(config: ExperimentConfig) -> dict[str, Any]:
    if config.xgb_preset_id not in XGB_PRESET_LOOKUP:
        raise KeyError(f"Unknown XGBoost preset id: {config.xgb_preset_id}")
    return XGB_PRESET_LOOKUP[config.xgb_preset_id].copy()


def require_tensorflow() -> None:
    if tf is None or keras is None or layers is None:
        raise ImportError("TensorFlow/Keras is not available in the current environment.")


def set_all_seeds(seed: int, *, deterministic_ops: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
        if deterministic_ops:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass


def validate_feature_contract(
    config: ExperimentConfig,
    base_frames: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    base_frames = base_frames if base_frames is not None else build_base_frame_cache(config)
    rows: list[dict[str, Any]] = []
    for building in config.buildings:
        frame_a = base_frames[building]["setA"]
        frame_b = base_frames[building]["setB"]
        missing_a = sorted(set(MODE_TEMPORAL_FEATURES["M0"] + MODE_TEMPORAL_FEATURES["M2"] + ["heat_kwh"]) - set(frame_a.columns))
        missing_b = sorted(set(MODE_TEMPORAL_FEATURES["M4"] + LSTM_STATIC_FEATURES_SETB + ["heat_kwh"]) - set(frame_b.columns))
        future_missing = []
        for horizon_h in config.horizons:
            expected_fw_cols = future_weather_feature_cols(frame_a.columns, horizon_h)
            if not expected_fw_cols:
                future_missing.append(str(horizon_h))
        static_complete = not frame_b[LSTM_STATIC_FEATURES_SETB].iloc[[0]].isna().any().any()
        status = "ok" if not missing_a and not missing_b and not future_missing and static_complete else "warn"
        rows.append(
            {
                "building": building,
                "setA_missing_cols": ", ".join(missing_a),
                "setB_missing_cols": ", ".join(missing_b),
                "missing_fw_horizons": ", ".join(future_missing),
                "static_complete_M4": bool(static_complete),
                "rows_setA": int(len(frame_a)),
                "rows_setB": int(len(frame_b)),
                "status": status,
            }
        )
    return pd.DataFrame(rows).sort_values("building").reset_index(drop=True)


def run_sanity_check(
    config: ExperimentConfig,
    *,
    base_frames: dict[str, dict[str, pd.DataFrame]] | None = None,
    run_smoke: bool = False,
    smoke_building: str | None = None,
    smoke_horizon_h: int | None = None,
    smoke_mode: str = "M0",
    smoke_weather_mode: str = "FW0",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paths = ensure_results_dirs(config)
    base_frames = base_frames if base_frames is not None else build_base_frame_cache(config)
    manifest_df = build_comparison_manifest(config)
    expected_runs = int(len(manifest_df))

    rows.append(
        {
            "check": "feature_metadata_exists",
            "status": "ok" if FEATURE_METADATA_FILE.exists() else "block",
            "detail": str(FEATURE_METADATA_FILE),
        }
    )
    rows.append(
        {
            "check": "feature_catalog_exists",
            "status": "ok" if FEATURE_CATALOG_FILE.exists() else "warn",
            "detail": str(FEATURE_CATALOG_FILE),
        }
    )
    try:
        probe_file = paths["results"] / ".write_probe"
        probe_file.write_text("ok\n", encoding="utf-8")
        probe_file.unlink()
        rows.append({"check": "results_dir_writable", "status": "ok", "detail": str(paths["results"])})
    except Exception as exc:
        rows.append({"check": "results_dir_writable", "status": "block", "detail": f"{paths['results']} | {exc}"})

    contract_df = validate_feature_contract(config, base_frames=base_frames)
    bad_contract = contract_df.loc[contract_df["status"] != "ok"]
    rows.append(
        {
            "check": "feature_contract",
            "status": "ok" if bad_contract.empty else "block",
            "detail": "all required columns available" if bad_contract.empty else bad_contract.to_dict(orient="records"),
        }
    )
    rows.append(
        {
            "check": "manifest_expected_runs",
            "status": "ok",
            "detail": f"{expected_runs} runs across {len(config.regimes)} regimes, {len(config.modes)} modes, {len(config.weather_modes)} weather modes, {len(config.horizons)} horizons",
        }
    )

    tf_ok = tf is not None and keras is not None and layers is not None
    rows.append(
        {
            "check": "tensorflow_available",
            "status": "ok" if tf_ok else "block",
            "detail": "TensorFlow/Keras import succeeded" if tf_ok else "TensorFlow/Keras is not available in the current environment.",
        }
    )
    try:
        require_xgboost()
        rows.append({"check": "xgboost_available", "status": "ok", "detail": "xgboost import succeeded"})
        xgb_ok = True
    except Exception as exc:
        rows.append({"check": "xgboost_available", "status": "block", "detail": repr(exc)})
        xgb_ok = False

    sample_building = smoke_building or config.buildings[0]
    sample_horizon = int(smoke_horizon_h or config.horizons[0])
    try:
        bundle, split_spec, preview_df = preview_feature_bundle(
            config=config,
            building=sample_building,
            mode=smoke_mode if smoke_mode in config.modes else config.modes[0],
            weather_mode=smoke_weather_mode if smoke_weather_mode in config.weather_modes else config.weather_modes[0],
            horizon_h=sample_horizon,
            base_frames=base_frames,
        )
        rows.append(
            {
                "check": "sample_bundle_preview",
                "status": "ok",
                "detail": {
                    "building": sample_building,
                    "horizon_h": sample_horizon,
                    "mode": smoke_mode,
                    "weather_mode": smoke_weather_mode,
                    "n_dynamic_features": len(bundle.dynamic_feature_cols),
                    "n_static_features": len(bundle.static_feature_cols),
                    "train_issue_end": str(split_spec.train_issue_end),
                    "preview_rows": int(len(preview_df)),
                },
            }
        )
    except Exception as exc:
        rows.append({"check": "sample_bundle_preview", "status": "block", "detail": repr(exc)})

    if run_smoke:
        if not tf_ok or not xgb_ok:
            rows.append(
                {
                    "check": "smoke_run",
                    "status": "block",
                    "detail": "Smoke run skipped because TensorFlow or xgboost is unavailable.",
                }
            )
        else:
            smoke_results_dir = config.results_dir / "_sanity_smoke"
            smoke_config = ExperimentConfig(
                buildings=(sample_building,),
                horizons=(sample_horizon,),
                regimes=("per_building",),
                weather_modes=(smoke_weather_mode,),
                modes=(smoke_mode,),
                train_end=config.train_end,
                test_start=config.test_start,
                lookback_hours=config.lookback_hours,
                validation_fraction=config.validation_fraction,
                results_dir=smoke_results_dir,
                lstm_architecture_id=config.lstm_architecture_id,
                xgb_preset_id=config.xgb_preset_id,
                random_seed=config.random_seed,
                batch_size=min(config.batch_size, 64),
                epochs=1,
                early_stopping_patience=max(1, min(config.early_stopping_patience, 2)),
                learning_rate=config.learning_rate,
                deterministic_ops=config.deterministic_ops,
                model_families=config.model_families,
            )
            try:
                smoke_outputs = run_full_comparison(
                    smoke_config,
                    save_artifacts=True,
                    verbose=False,
                    resume_existing=False,
                    save_after_each_pair=True,
                    continue_on_error=False,
                )
                rows.append(
                    {
                        "check": "smoke_run",
                        "status": "ok",
                        "detail": {
                            "results_dir": str(smoke_results_dir),
                            "metrics_rows": int(len(smoke_outputs.comparison_metrics_df)),
                            "prediction_rows": int(len(smoke_outputs.comparison_predictions_df)),
                            "run_log_rows": int(len(smoke_outputs.run_log_df)),
                        },
                    }
                )
            except Exception as exc:
                rows.append({"check": "smoke_run", "status": "block", "detail": repr(exc)})

    return pd.DataFrame(rows)


def build_feature_bundle(
    config: ExperimentConfig,
    building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    base_frames: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> tuple[FeatureBundle, SplitSpec]:
    base_frames = base_frames if base_frames is not None else build_base_frame_cache(config)
    set_name = feature_set_for_mode(mode)
    frame = base_frames[building][set_name].copy()
    frame, fw_cols = apply_weather_mode(frame, weather_mode, horizon_h)
    target_name = f"target_cum_h{int(horizon_h)}"
    dynamic_cols = MODE_TEMPORAL_FEATURES[mode].copy() + fw_cols
    static_cols = LSTM_STATIC_FEATURES_SETB.copy() if mode == "M4" else []
    split_spec = build_split_spec(frame, horizon_h, config)
    valid_issue_mask = (frame[target_name].notna()) & (split_spec.train_issue_mask | split_spec.test_issue_mask)
    bundle = FeatureBundle(
        raw_frame=frame,
        target_name=target_name,
        dynamic_feature_cols=dynamic_cols,
        static_feature_cols=static_cols,
        valid_issue_mask=valid_issue_mask,
        model_ready={
            "feature_set": set_name,
            "weather_mode": weather_mode,
            "horizon_h": int(horizon_h),
            "mode": mode,
        },
    )
    return bundle, split_spec


def preview_feature_bundle(
    config: ExperimentConfig,
    building: str = "U05",
    mode: str = "M4",
    weather_mode: str = "FW2",
    horizon_h: int = 24,
    base_frames: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> tuple[FeatureBundle, SplitSpec, pd.DataFrame]:
    bundle, split_spec = build_feature_bundle(
        config=config,
        building=building,
        mode=mode,
        weather_mode=weather_mode,
        horizon_h=horizon_h,
        base_frames=base_frames,
    )
    preview_cols = ["datetime", "building", "heat_kwh", bundle.target_name, "feat_is_heating_weather"]
    preview_cols += bundle.dynamic_feature_cols[: min(6, len(bundle.dynamic_feature_cols))]
    preview_df = bundle.raw_frame.loc[:, [c for c in preview_cols if c in bundle.raw_frame.columns]].head(8).copy()
    preview_df["valid_issue"] = bundle.valid_issue_mask.head(8).to_numpy()
    return bundle, split_spec, preview_df


def build_comparison_manifest(config: ExperimentConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime in config.regimes:
        if regime == "per_building":
            scope_buildings = config.buildings
        else:
            scope_buildings = ("POOLED",)
        for building in scope_buildings:
            for model_family in config.model_families:
                for mode in config.modes:
                    for weather_mode in config.weather_modes:
                        for horizon_h in config.horizons:
                            rows.append(
                                {
                                    "regime": regime,
                                    "building": building,
                                    "model_family": model_family,
                                    "mode": mode,
                                    "weather_mode": weather_mode,
                                    "horizon_h": int(horizon_h),
                                    "run_id": run_id_from_spec(
                                        RunSpec(
                                            regime=regime,
                                            building=building,
                                            model_family=model_family,
                                            mode=mode,
                                            weather_mode=weather_mode,
                                            horizon_h=int(horizon_h),
                                        )
                                    ),
                                }
                            )
    return pd.DataFrame(rows).sort_values(
        ["regime", "building", "model_family", "mode", "weather_mode", "horizon_h"]
    ).reset_index(drop=True)


def fit_static_scaler(
    config: ExperimentConfig,
    base_frames: dict[str, dict[str, pd.DataFrame]],
) -> StandardScaler:
    rows = []
    for building in config.buildings:
        row = base_frames[building]["setB"][LSTM_STATIC_FEATURES_SETB].iloc[[0]].apply(pd.to_numeric, errors="coerce")
        if not row.isna().any().any():
            rows.append(row.values)
    if not rows:
        raise ValueError("No complete static rows available for fitting the M4 static scaler.")
    scaler = StandardScaler()
    scaler.fit(np.vstack(rows))
    return scaler


def build_static_vector(
    frame: pd.DataFrame,
    static_scaler: StandardScaler,
    static_cols: list[str],
) -> np.ndarray:
    row = frame[static_cols].iloc[[0]].apply(pd.to_numeric, errors="coerce")
    if row.isna().any().any():
        raise ValueError("Static feature row contains NaNs.")
    return static_scaler.transform(row.values).astype("float32")


def _fit_standard_scaler(values: pd.DataFrame) -> StandardScaler:
    fit_values = values.apply(pd.to_numeric, errors="coerce").dropna()
    if fit_values.empty:
        raise ValueError("No non-null rows available for scaler fitting.")
    scaler = StandardScaler()
    scaler.fit(fit_values.values)
    return scaler


def scale_frame_for_lstm(
    frame: pd.DataFrame,
    dynamic_cols: list[str],
    target_name: str,
    feature_train_mask: pd.Series,
    train_issue_mask: pd.Series,
) -> tuple[pd.DataFrame, StandardScaler, StandardScaler]:
    out = frame.copy()
    feature_scaler = _fit_standard_scaler(out.loc[feature_train_mask, dynamic_cols])
    target_scaler = _fit_standard_scaler(out.loc[train_issue_mask & out[target_name].notna(), [target_name]])

    dynamic_numeric = out.loc[:, dynamic_cols].apply(pd.to_numeric, errors="coerce")
    complete_feature_rows = dynamic_numeric.notna().all(axis=1)
    out.loc[:, dynamic_cols] = np.nan
    if complete_feature_rows.any():
        out.loc[complete_feature_rows, dynamic_cols] = feature_scaler.transform(dynamic_numeric.loc[complete_feature_rows].values)

    out[f"{target_name}_scaled"] = np.nan
    target_mask = out[target_name].notna()
    if target_mask.any():
        out.loc[target_mask, f"{target_name}_scaled"] = target_scaler.transform(
            out.loc[target_mask, [target_name]].apply(pd.to_numeric, errors="coerce").values
        ).reshape(-1)
    return out, feature_scaler, target_scaler


def make_internal_fit_split(X: np.ndarray, y: np.ndarray, meta_df: pd.DataFrame, frac: float) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    n = X.shape[0]
    if n == 0:
        return X, y, meta_df.copy(), X, y, meta_df.copy()
    if frac <= 0 or n < 50:
        empty_meta = meta_df.iloc[0:0].copy()
        return X, y, meta_df.copy(), np.empty((0,) + X.shape[1:], dtype=X.dtype), np.empty((0,), dtype=y.dtype), empty_meta
    n_val = max(1, int(round(n * frac)))
    n_val = min(n_val, max(1, n - 1))
    split_at = n - n_val
    return (
        X[:split_at],
        y[:split_at],
        meta_df.iloc[:split_at].reset_index(drop=True),
        X[split_at:],
        y[split_at:],
        meta_df.iloc[split_at:].reset_index(drop=True),
    )


def build_sequences(
    df_scaled: pd.DataFrame,
    dynamic_cols: list[str],
    target_col: str,
    split_mask: pd.Series | np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    lookback = int(lookback)
    mask = np.asarray(split_mask, dtype=bool)
    n_rows = len(df_scaled)
    if n_rows <= lookback:
        return (
            np.empty((0, lookback, len(dynamic_cols)), dtype="float32"),
            np.empty((0,), dtype="float32"),
            pd.DataFrame(columns=["building", "datetime", "is_heating_eval"]),
        )

    dynamic_values = df_scaled.loc[:, dynamic_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32", copy=False)
    target_values = pd.to_numeric(df_scaled[target_col], errors="coerce").to_numpy(dtype=float)
    heating_values = heating_mask(df_scaled).to_numpy(dtype=bool)
    building_values = df_scaled["building"].astype(str).to_numpy()
    datetime_values = pd.to_datetime(df_scaled["datetime"]).to_numpy()

    end_indices = np.arange(lookback, n_rows, dtype=int)
    current_mask = mask[end_indices]
    target_valid = np.isfinite(target_values[end_indices])

    row_valid = np.isfinite(dynamic_values).all(axis=1)
    history_mask_valid = np.lib.stride_tricks.sliding_window_view(mask, lookback)[: n_rows - lookback].all(axis=1)
    history_feature_valid = np.lib.stride_tricks.sliding_window_view(row_valid, lookback)[: n_rows - lookback].all(axis=1)
    valid_endpoints = current_mask & target_valid & history_mask_valid & history_feature_valid

    if not valid_endpoints.any():
        return (
            np.empty((0, lookback, len(dynamic_cols)), dtype="float32"),
            np.empty((0,), dtype="float32"),
            pd.DataFrame(columns=["building", "datetime", "is_heating_eval"]),
        )

    sequence_windows = np.lib.stride_tricks.sliding_window_view(dynamic_values, lookback, axis=0)[: n_rows - lookback]
    sequence_windows = np.transpose(sequence_windows, (0, 2, 1))
    X = np.ascontiguousarray(sequence_windows[valid_endpoints], dtype="float32")
    y = target_values[end_indices][valid_endpoints].astype("float32", copy=False)
    meta_df = pd.DataFrame(
        {
            "building": building_values[end_indices][valid_endpoints],
            "datetime": pd.to_datetime(datetime_values[end_indices][valid_endpoints]),
            "is_heating_eval": heating_values[end_indices][valid_endpoints],
        }
    )
    return X, y, meta_df


def build_xgb_model_frames(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_name: str,
    split_spec: SplitSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_cols = ["building", "datetime", target_name] + feature_cols
    if "feat_is_heating_weather" in frame.columns:
        keep_cols.append("feat_is_heating_weather")
    train_df = frame.loc[split_spec.train_issue_mask, keep_cols].copy()
    test_df = frame.loc[split_spec.test_issue_mask, keep_cols].copy()
    for work_df in (train_df, test_df):
        for col in feature_cols + [target_name]:
            work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
        work_df["is_heating_eval"] = heating_mask(work_df).to_numpy(dtype=bool)
        work_df.dropna(subset=[target_name], inplace=True)
        work_df.sort_values(["datetime", "building"], inplace=True)
        work_df.reset_index(drop=True, inplace=True)
    return train_df, test_df


def _split_xgb_train_validation(train_df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_df.empty or frac <= 0:
        return train_df.copy(), train_df.iloc[0:0].copy()
    n_val = max(1, int(round(len(train_df) * frac)))
    n_val = min(n_val, len(train_df) - 1) if len(train_df) > 1 else 0
    split_at = len(train_df) - n_val
    if split_at <= 0:
        return train_df.iloc[0:0].copy(), train_df.copy()
    return train_df.iloc[:split_at].copy(), train_df.iloc[split_at:].copy()


def make_optimizer(learning_rate: float):
    try:
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    except Exception:
        return keras.optimizers.Adam(learning_rate=learning_rate)


def build_lstm_temporal_only_from_spec(n_timesteps: int, n_features: int, spec: dict[str, Any], learning_rate: float) -> keras.Model:
    temporal_in = keras.Input(shape=(n_timesteps, n_features), name="temporal_input")
    x = temporal_in
    stack = spec["lstm_stack"]
    for layer_idx, units in enumerate(stack, start=1):
        return_sequences = layer_idx < len(stack)
        x = layers.LSTM(units, return_sequences=return_sequences, name=f"lstm_{layer_idx}")(x)
        if return_sequences and spec["dropout"] > 0:
            x = layers.Dropout(spec["dropout"], name=f"dropout_{layer_idx}")(x)
    x = layers.Dense(spec["dense_units"], activation="relu", name="dense_1")(x)
    out = layers.Dense(1, name="output")(x)
    model = keras.Model(inputs=temporal_in, outputs=out, name=f"{spec['architecture_id']}_temporal_only")
    model.compile(optimizer=make_optimizer(learning_rate), loss="mse", metrics=["mse"])
    return model


def build_lstm_temporal_plus_static_from_spec(
    n_timesteps: int,
    n_dynamic: int,
    n_static_features: int,
    spec: dict[str, Any],
    learning_rate: float,
) -> keras.Model:
    temporal_in = keras.Input(shape=(n_timesteps, n_dynamic), name="temporal_input")
    static_in = keras.Input(shape=(n_static_features,), name="static_input")

    x = temporal_in
    stack = spec["lstm_stack"]
    for layer_idx, units in enumerate(stack, start=1):
        return_sequences = layer_idx < len(stack)
        x = layers.LSTM(units, return_sequences=return_sequences, name=f"lstm_{layer_idx}")(x)
        if return_sequences and spec["dropout"] > 0:
            x = layers.Dropout(spec["dropout"], name=f"dropout_{layer_idx}")(x)

    s = layers.Dense(32, activation="relu", name="static_dense_1")(static_in)
    s = layers.Dense(16, activation="relu", name="static_dense_2")(s)
    merged = layers.Concatenate(name="concat")([x, s])
    h = layers.Dense(spec["dense_units"], activation="relu", name="merged_dense_1")(merged)
    out = layers.Dense(1, name="output")(h)

    model = keras.Model(inputs=[temporal_in, static_in], outputs=out, name=f"{spec['architecture_id']}_temporal_plus_static")
    model.compile(optimizer=make_optimizer(learning_rate), loss="mse", metrics=["mse"])
    return model


if keras is not None:
    class LearningRateHistory(keras.callbacks.Callback):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.rows: list[dict[str, float]] = []

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            learning_rate = np.nan
            optimizer = getattr(self.model, "optimizer", None)
            if optimizer is not None:
                try:
                    learning_rate = float(tf.keras.backend.get_value(optimizer.learning_rate))
                except Exception:
                    try:
                        learning_rate = float(tf.keras.backend.get_value(optimizer.lr))
                    except Exception:
                        learning_rate = np.nan
            row = {"epoch": int(epoch) + 1, "learning_rate": learning_rate}
            for key, value in logs.items():
                row[key] = float(value) if value is not None else np.nan
            self.rows.append(row)
else:  # pragma: no cover - only used when tensorflow is unavailable
    class LearningRateHistory:  # type: ignore[override]
        def __init__(self) -> None:
            self.rows: list[dict[str, float]] = []


def _safe_eval_history(model) -> pd.DataFrame:
    try:
        result = model.evals_result()
    except Exception:
        return pd.DataFrame()
    rows = []
    for dataset_name, metrics in result.items():
        for metric_name, values in metrics.items():
            for idx, value in enumerate(values, start=1):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "metric": metric_name,
                        "iteration": int(idx),
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _fit_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    preset: dict[str, Any],
    config: ExperimentConfig,
):
    XGBRegressor = require_xgboost()
    model = XGBRegressor(
        objective=XGB_FIXED_PARAMS["objective"],
        booster=XGB_FIXED_PARAMS["booster"],
        tree_method=XGB_FIXED_PARAMS["tree_method"],
        n_estimators=XGB_FIXED_PARAMS["n_estimators"],
        max_depth=int(preset["max_depth"]),
        learning_rate=float(preset["learning_rate"]),
        min_child_weight=float(preset["min_child_weight"]),
        subsample=float(XGB_FIXED_PARAMS["subsample"]),
        colsample_bytree=float(XGB_FIXED_PARAMS["colsample_bytree"]),
        reg_lambda=float(XGB_FIXED_PARAMS["reg_lambda"]),
        random_state=int(config.random_seed),
        n_jobs=-1,
        eval_metric="rmse",
    )
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=int(XGB_FIXED_PARAMS["early_stopping_rounds"]),
            verbose=False,
        )
    except TypeError:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
    return model


def _predict_with_best_iteration(model, X: pd.DataFrame) -> np.ndarray:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return np.asarray(model.predict(X), dtype=float)
    try:
        return np.asarray(model.predict(X, iteration_range=(0, int(best_iteration) + 1)), dtype=float)
    except TypeError:
        return np.asarray(model.predict(X), dtype=float)


def _inverse_transform_by_building(values_scaled: np.ndarray, meta_df: pd.DataFrame, target_scalers: dict[str, StandardScaler]) -> np.ndarray:
    out = np.empty((len(values_scaled),), dtype=float)
    for building, idx in meta_df.groupby("building").groups.items():
        scaler = target_scalers[str(building)]
        transformed = scaler.inverse_transform(values_scaled[np.asarray(list(idx))].reshape(-1, 1)).reshape(-1)
        out[np.asarray(list(idx))] = transformed
    return out


def _prepare_lstm_sequences_for_building(
    bundle: FeatureBundle,
    split_spec: SplitSpec,
    config: ExperimentConfig,
) -> dict[str, Any]:
    df_scaled, feature_scaler, target_scaler = scale_frame_for_lstm(
        frame=bundle.raw_frame,
        dynamic_cols=bundle.dynamic_feature_cols,
        target_name=bundle.target_name,
        feature_train_mask=split_spec.feature_train_mask,
        train_issue_mask=split_spec.train_issue_mask,
    )
    X_train_full, y_train_full, train_meta_full = build_sequences(
        df_scaled,
        bundle.dynamic_feature_cols,
        f"{bundle.target_name}_scaled",
        split_spec.train_issue_mask,
        config.lookback_hours,
    )
    X_test, y_test, test_meta = build_sequences(
        df_scaled,
        bundle.dynamic_feature_cols,
        f"{bundle.target_name}_scaled",
        split_spec.test_issue_mask,
        config.lookback_hours,
    )
    return {
        "frame": df_scaled,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "train_meta_full": train_meta_full,
        "X_test": X_test,
        "y_test": y_test,
        "test_meta": test_meta,
    }


def _run_lstm_regime(
    config: ExperimentConfig,
    regime: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    static_scaler: StandardScaler,
) -> tuple[TrainingArtifacts, pd.DataFrame]:
    require_tensorflow()
    set_all_seeds(config.random_seed, deterministic_ops=config.deterministic_ops)
    tf.keras.backend.clear_session()

    spec = architecture_spec(config)
    start_time = perf_counter()
    if regime == "pooled_same_buildings":
        run_building = "POOLED"
    elif len(config.buildings) == 1:
        run_building = str(config.buildings[0])
    else:
        run_building = "MULTI"
    run_spec = RunSpec(
        regime=regime,
        building=run_building,
        model_family="lstm",
        mode=mode,
        weather_mode=weather_mode,
        horizon_h=int(horizon_h),
    )
    run_id = run_id_from_spec(run_spec)

    prepared_by_building: dict[str, dict[str, Any]] = {}
    static_vectors: dict[str, np.ndarray] = {}
    per_building_counts = []
    buildings_to_use = config.buildings
    for building in buildings_to_use:
        bundle, split_spec = build_feature_bundle(
            config=config,
            building=building,
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=int(horizon_h),
            base_frames=base_frames,
        )
        prepared = _prepare_lstm_sequences_for_building(bundle, split_spec, config)
        if mode == "M4":
            static_vectors[building] = build_static_vector(bundle.raw_frame, static_scaler, bundle.static_feature_cols)
        prepared_by_building[building] = prepared
        per_building_counts.append(
            {
                "building": building,
                "n_train_seq_raw": int(prepared["X_train_full"].shape[0]),
                "n_test_seq_raw": int(prepared["X_test"].shape[0]),
            }
        )

    fit_X_list: list[np.ndarray] = []
    fit_y_list: list[np.ndarray] = []
    fit_meta_list: list[pd.DataFrame] = []
    val_X_list: list[np.ndarray] = []
    val_y_list: list[np.ndarray] = []
    val_meta_list: list[pd.DataFrame] = []
    test_X_list: list[np.ndarray] = []
    test_y_list: list[np.ndarray] = []
    test_meta_list: list[pd.DataFrame] = []
    fit_static_list: list[np.ndarray] = []
    val_static_list: list[np.ndarray] = []
    test_static_list: list[np.ndarray] = []
    target_scalers = {building: prepared_by_building[building]["target_scaler"] for building in buildings_to_use}

    for building in buildings_to_use:
        prepared = prepared_by_building[building]
        X_fit, y_fit, meta_fit, X_val, y_val, meta_val = make_internal_fit_split(
            prepared["X_train_full"],
            prepared["y_train_full"],
            prepared["train_meta_full"],
            config.validation_fraction,
        )
        if X_fit.shape[0] > 0:
            fit_X_list.append(X_fit)
            fit_y_list.append(y_fit)
            fit_meta_list.append(meta_fit)
            if mode == "M4":
                fit_static_list.append(np.repeat(static_vectors[building], repeats=X_fit.shape[0], axis=0))
        if X_val.shape[0] > 0:
            val_X_list.append(X_val)
            val_y_list.append(y_val)
            val_meta_list.append(meta_val)
            if mode == "M4":
                val_static_list.append(np.repeat(static_vectors[building], repeats=X_val.shape[0], axis=0))
        if prepared["X_test"].shape[0] > 0:
            test_X_list.append(prepared["X_test"])
            test_y_list.append(prepared["y_test"])
            test_meta_list.append(prepared["test_meta"])
            if mode == "M4":
                test_static_list.append(np.repeat(static_vectors[building], repeats=prepared["X_test"].shape[0], axis=0))

    if not fit_X_list or not test_X_list:
        run_log_row = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "regime": regime,
                    "building": run_building,
                    "model_family": "lstm",
                    "mode": mode,
                    "weather_mode": weather_mode,
                    "horizon_h": int(horizon_h),
                    "status": "skipped_insufficient_sequences",
                    "elapsed_s": float(perf_counter() - start_time),
                }
            ]
        )
        return TrainingArtifacts(run_spec=run_spec), run_log_row

    X_fit_all = np.concatenate(fit_X_list, axis=0)
    y_fit_all = np.concatenate(fit_y_list, axis=0)
    fit_meta_all = pd.concat(fit_meta_list, ignore_index=True)
    X_test_all = np.concatenate(test_X_list, axis=0)
    y_test_all = np.concatenate(test_y_list, axis=0)
    test_meta_all = pd.concat(test_meta_list, ignore_index=True)

    X_val_all = np.concatenate(val_X_list, axis=0) if val_X_list else np.empty((0,) + X_fit_all.shape[1:], dtype=X_fit_all.dtype)
    y_val_all = np.concatenate(val_y_list, axis=0) if val_y_list else np.empty((0,), dtype=y_fit_all.dtype)
    val_meta_all = pd.concat(val_meta_list, ignore_index=True) if val_meta_list else pd.DataFrame(columns=fit_meta_all.columns)

    if mode == "M4":
        X_fit_static_all = np.concatenate(fit_static_list, axis=0)
        X_test_static_all = np.concatenate(test_static_list, axis=0)
        X_val_static_all = np.concatenate(val_static_list, axis=0) if val_static_list else np.empty((0, X_fit_static_all.shape[1]), dtype=X_fit_static_all.dtype)
        model = build_lstm_temporal_plus_static_from_spec(
            spec["lookback_hours"],
            X_fit_all.shape[-1],
            X_fit_static_all.shape[-1],
            spec,
            config.learning_rate,
        )
    else:
        X_fit_static_all = None
        X_test_static_all = None
        X_val_static_all = None
        model = build_lstm_temporal_only_from_spec(
            spec["lookback_hours"],
            X_fit_all.shape[-1],
            spec,
            config.learning_rate,
        )

    lr_callback = LearningRateHistory()
    callbacks = [lr_callback]
    validation_data = None
    if X_val_all.shape[0] > 0:
        callbacks += [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0),
        ]
        validation_data = ([X_val_all, X_val_static_all], y_val_all) if mode == "M4" else (X_val_all, y_val_all)

    if mode == "M4":
        history = model.fit(
            [X_fit_all, X_fit_static_all],
            y_fit_all,
            validation_data=validation_data,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        y_pred_scaled = model.predict([X_test_all, X_test_static_all], verbose=0).reshape(-1)
    else:
        history = model.fit(
            X_fit_all,
            y_fit_all,
            validation_data=validation_data,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        y_pred_scaled = model.predict(X_test_all, verbose=0).reshape(-1)

    y_true = _inverse_transform_by_building(y_test_all, test_meta_all, target_scalers)
    y_pred = _inverse_transform_by_building(y_pred_scaled, test_meta_all, target_scalers)
    pred_df = test_meta_all.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["abs_error"] = np.abs(pred_df["y_true"] - pred_df["y_pred"])
    pred_df["run_id"] = run_id
    pred_df["regime"] = regime
    pred_df["mode"] = mode
    pred_df["weather_mode"] = weather_mode
    pred_df["horizon_h"] = int(horizon_h)
    pred_df["model_family"] = "lstm"
    pred_df = pred_df[
        ["run_id", "regime", "building", "model_family", "mode", "weather_mode", "horizon_h", "datetime", "y_true", "y_pred", "abs_error", "is_heating_eval"]
    ]

    history_rows = lr_callback.rows.copy()
    if not history_rows and history.history.get("loss"):
        for epoch_idx, loss_value in enumerate(history.history["loss"], start=1):
            history_rows.append(
                {
                    "epoch": int(epoch_idx),
                    "loss": float(loss_value),
                    "val_loss": float(history.history.get("val_loss", [np.nan] * len(history.history["loss"]))[epoch_idx - 1]),
                    "learning_rate": np.nan,
                }
            )
    history_df = pd.DataFrame(history_rows)
    if not history_df.empty:
        history_df["run_id"] = run_id
        history_df["regime"] = regime
        history_df["building"] = run_building
        history_df["model_family"] = "lstm"
        history_df["mode"] = mode
        history_df["weather_mode"] = weather_mode
        history_df["horizon_h"] = int(horizon_h)
        history_df = history_df[
            ["run_id", "regime", "building", "model_family", "mode", "weather_mode", "horizon_h", "epoch", "loss", "val_loss", "learning_rate"]
        ]

    elapsed = float(perf_counter() - start_time)
    run_log_row = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "regime": regime,
                "building": run_building,
                "model_family": "lstm",
                "mode": mode,
                "weather_mode": weather_mode,
                "horizon_h": int(horizon_h),
                "status": "ok",
                "elapsed_s": elapsed,
                "n_train_rows": int(X_fit_all.shape[0]),
                "n_val_rows": int(X_val_all.shape[0]),
                "n_test_rows": int(X_test_all.shape[0]),
                "best_epoch": int(np.nanargmin(history_df["val_loss"].to_numpy())) + 1 if not history_df.empty and history_df["val_loss"].notna().any() else np.nan,
            }
        ]
    )
    model_summary = {
        "architecture_id": spec["architecture_id"],
        "lookback_hours": int(spec["lookback_hours"]),
        "n_train_rows": int(X_fit_all.shape[0]),
        "n_val_rows": int(X_val_all.shape[0]),
        "n_test_rows": int(X_test_all.shape[0]),
        "per_building_counts": per_building_counts,
    }
    return TrainingArtifacts(run_spec=run_spec, history_df=history_df, prediction_df=pred_df, model_summary=model_summary), run_log_row


def _run_xgb_regime(
    config: ExperimentConfig,
    regime: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    base_frames: dict[str, dict[str, pd.DataFrame]],
) -> tuple[TrainingArtifacts, pd.DataFrame]:
    start_time = perf_counter()
    preset = xgb_preset(config)
    run_building = "POOLED" if regime == "pooled_same_buildings" else "MULTI"
    run_spec = RunSpec(
        regime=regime,
        building=run_building,
        model_family="xgboost",
        mode=mode,
        weather_mode=weather_mode,
        horizon_h=int(horizon_h),
    )
    run_id = run_id_from_spec(run_spec)

    train_fit_frames: list[pd.DataFrame] = []
    train_val_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    feature_cols: list[str] | None = None

    buildings_to_use = config.buildings if regime == "pooled_same_buildings" else ()
    target_buildings = config.buildings if regime == "pooled_same_buildings" else ()
    if regime == "per_building":
        target_buildings = ()

    for building in config.buildings:
        if regime == "per_building":
            work_buildings = [building]
        else:
            work_buildings = []
        if regime == "per_building" and building not in config.buildings:
            continue
        bundle, split_spec = build_feature_bundle(
            config=config,
            building=building,
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=int(horizon_h),
            base_frames=base_frames,
        )
        feature_cols_here = bundle.dynamic_feature_cols + bundle.static_feature_cols
        if feature_cols is None:
            feature_cols = feature_cols_here
        train_df_full, test_df = build_xgb_model_frames(bundle.raw_frame, feature_cols_here, bundle.target_name, split_spec)
        fit_df, val_df = _split_xgb_train_validation(train_df_full, config.validation_fraction)
        if regime == "per_building":
            run_building = building
            run_spec = RunSpec(
                regime=regime,
                building=run_building,
                model_family="xgboost",
                mode=mode,
                weather_mode=weather_mode,
                horizon_h=int(horizon_h),
            )
            run_id = run_id_from_spec(run_spec)
            if fit_df.empty or test_df.empty:
                run_log_row = pd.DataFrame(
                    [
                        {
                            "run_id": run_id,
                            "regime": regime,
                            "building": building,
                            "model_family": "xgboost",
                            "mode": mode,
                            "weather_mode": weather_mode,
                            "horizon_h": int(horizon_h),
                            "status": "skipped_insufficient_rows",
                            "elapsed_s": float(perf_counter() - start_time),
                        }
                    ]
                )
                return TrainingArtifacts(run_spec=run_spec), run_log_row
            model = _fit_xgb_model(
                fit_df[feature_cols_here],
                fit_df[bundle.target_name],
                val_df[feature_cols_here],
                val_df[bundle.target_name],
                preset,
                config,
            )
            y_pred = _predict_with_best_iteration(model, test_df[feature_cols_here])
            pred_df = test_df.loc[:, ["building", "datetime", "is_heating_eval", bundle.target_name]].copy()
            pred_df.rename(columns={bundle.target_name: "y_true"}, inplace=True)
            pred_df["y_pred"] = y_pred
            pred_df["abs_error"] = np.abs(pred_df["y_true"] - pred_df["y_pred"])
            pred_df["run_id"] = run_id
            pred_df["regime"] = regime
            pred_df["mode"] = mode
            pred_df["weather_mode"] = weather_mode
            pred_df["horizon_h"] = int(horizon_h)
            pred_df["model_family"] = "xgboost"
            pred_df = pred_df[
                ["run_id", "regime", "building", "model_family", "mode", "weather_mode", "horizon_h", "datetime", "y_true", "y_pred", "abs_error", "is_heating_eval"]
            ]
            eval_history_df = _safe_eval_history(model)
            if not eval_history_df.empty:
                eval_history_df["run_id"] = run_id
                eval_history_df["regime"] = regime
                eval_history_df["building"] = building
                eval_history_df["model_family"] = "xgboost"
                eval_history_df["mode"] = mode
                eval_history_df["weather_mode"] = weather_mode
                eval_history_df["horizon_h"] = int(horizon_h)
                eval_history_df = eval_history_df[
                    ["run_id", "regime", "building", "model_family", "mode", "weather_mode", "horizon_h", "dataset", "metric", "iteration", "value"]
                ]
            elapsed = float(perf_counter() - start_time)
            run_log_row = pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "regime": regime,
                        "building": building,
                        "model_family": "xgboost",
                        "mode": mode,
                        "weather_mode": weather_mode,
                        "horizon_h": int(horizon_h),
                        "status": "ok",
                        "elapsed_s": elapsed,
                        "n_train_rows": int(len(fit_df)),
                        "n_val_rows": int(len(val_df)),
                        "n_test_rows": int(len(test_df)),
                        "best_iteration": int(getattr(model, "best_iteration", XGB_FIXED_PARAMS["n_estimators"] - 1) or XGB_FIXED_PARAMS["n_estimators"] - 1),
                    }
                ]
            )
            model_summary = {
                "preset_id": preset["preset_id"],
                "n_train_rows": int(len(fit_df)),
                "n_val_rows": int(len(val_df)),
                "n_test_rows": int(len(test_df)),
                "feature_cols": feature_cols_here,
            }
            return TrainingArtifacts(run_spec=run_spec, history_df=eval_history_df, prediction_df=pred_df, model_summary=model_summary), run_log_row

        train_fit_frames.append(fit_df)
        train_val_frames.append(val_df)
        test_frames.append(test_df)

    feature_cols = feature_cols or []
    fit_all = pd.concat(train_fit_frames, ignore_index=True) if train_fit_frames else pd.DataFrame()
    val_all = pd.concat(train_val_frames, ignore_index=True) if train_val_frames else pd.DataFrame()
    test_all = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    if fit_all.empty or test_all.empty:
        run_log_row = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "regime": regime,
                    "building": run_building,
                    "model_family": "xgboost",
                    "mode": mode,
                    "weather_mode": weather_mode,
                    "horizon_h": int(horizon_h),
                    "status": "skipped_insufficient_rows",
                    "elapsed_s": float(perf_counter() - start_time),
                }
            ]
        )
        return TrainingArtifacts(run_spec=run_spec), run_log_row

    target_name = f"target_cum_h{int(horizon_h)}"
    model = _fit_xgb_model(
        fit_all[feature_cols],
        fit_all[target_name],
        val_all[feature_cols],
        val_all[target_name],
        preset,
        config,
    )
    y_pred = _predict_with_best_iteration(model, test_all[feature_cols])
    pred_df = test_all.loc[:, ["building", "datetime", "is_heating_eval", target_name]].copy()
    pred_df.rename(columns={target_name: "y_true"}, inplace=True)
    pred_df["y_pred"] = y_pred
    pred_df["abs_error"] = np.abs(pred_df["y_true"] - pred_df["y_pred"])
    pred_df["run_id"] = run_id
    pred_df["regime"] = regime
    pred_df["mode"] = mode
    pred_df["weather_mode"] = weather_mode
    pred_df["horizon_h"] = int(horizon_h)
    pred_df["model_family"] = "xgboost"
    pred_df = pred_df[
        ["run_id", "regime", "building", "model_family", "mode", "weather_mode", "horizon_h", "datetime", "y_true", "y_pred", "abs_error", "is_heating_eval"]
    ]
    eval_history_df = _safe_eval_history(model)
    if not eval_history_df.empty:
        eval_history_df["run_id"] = run_id
        eval_history_df["regime"] = regime
        eval_history_df["building"] = run_building
        eval_history_df["model_family"] = "xgboost"
        eval_history_df["mode"] = mode
        eval_history_df["weather_mode"] = weather_mode
        eval_history_df["horizon_h"] = int(horizon_h)
        eval_history_df = eval_history_df[
            ["run_id", "regime", "building", "model_family", "mode", "weather_mode", "horizon_h", "dataset", "metric", "iteration", "value"]
        ]
    elapsed = float(perf_counter() - start_time)
    run_log_row = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "regime": regime,
                "building": run_building,
                "model_family": "xgboost",
                "mode": mode,
                "weather_mode": weather_mode,
                "horizon_h": int(horizon_h),
                "status": "ok",
                "elapsed_s": elapsed,
                "n_train_rows": int(len(fit_all)),
                "n_val_rows": int(len(val_all)),
                "n_test_rows": int(len(test_all)),
                "best_iteration": int(getattr(model, "best_iteration", XGB_FIXED_PARAMS["n_estimators"] - 1) or XGB_FIXED_PARAMS["n_estimators"] - 1),
            }
        ]
    )
    model_summary = {
        "preset_id": preset["preset_id"],
        "n_train_rows": int(len(fit_all)),
        "n_val_rows": int(len(val_all)),
        "n_test_rows": int(len(test_all)),
        "feature_cols": feature_cols,
    }
    return TrainingArtifacts(run_spec=run_spec, history_df=eval_history_df, prediction_df=pred_df, model_summary=model_summary), run_log_row


def _align_predictions_and_metrics(
    config: ExperimentConfig,
    regime: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    lstm_artifacts: TrainingArtifacts,
    xgb_artifacts: TrainingArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lstm_pred = lstm_artifacts.prediction_df.copy()
    xgb_pred = xgb_artifacts.prediction_df.copy()
    if lstm_pred.empty or xgb_pred.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    key_cols = ["building", "datetime"]
    common_keys = lstm_pred[key_cols].merge(xgb_pred[key_cols], on=key_cols, how="inner").drop_duplicates()
    if common_keys.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    aligned_predictions = []
    coverage_rows = []
    metric_rows = []

    for building in sorted(common_keys["building"].astype(str).unique()):
        common_b = common_keys.loc[common_keys["building"].astype(str) == building]
        lstm_b = lstm_pred.merge(common_b, on=key_cols, how="inner").sort_values("datetime").reset_index(drop=True)
        xgb_b = xgb_pred.merge(common_b, on=key_cols, how="inner").sort_values("datetime").reset_index(drop=True)
        aligned_predictions.extend([lstm_b, xgb_b])

        lstm_metrics, lstm_eval_rows = _metrics_from_prediction_frame(lstm_b)
        xgb_metrics, xgb_eval_rows = _metrics_from_prediction_frame(xgb_b)
        metric_rows.extend(
            [
                {
                    "regime": regime,
                    "building": building,
                    "model_family": "lstm",
                    "mode": mode,
                    "weather_mode": weather_mode,
                    "horizon_h": int(horizon_h),
                    "rmse": lstm_metrics["rmse"],
                    "wape_pct": lstm_metrics["wape_pct"],
                    "r2": lstm_metrics["r2"],
                    "mae": lstm_metrics["mae"],
                    "n_test_rows": int(lstm_eval_rows),
                },
                {
                    "regime": regime,
                    "building": building,
                    "model_family": "xgboost",
                    "mode": mode,
                    "weather_mode": weather_mode,
                    "horizon_h": int(horizon_h),
                    "rmse": xgb_metrics["rmse"],
                    "wape_pct": xgb_metrics["wape_pct"],
                    "r2": xgb_metrics["r2"],
                    "mae": xgb_metrics["mae"],
                    "n_test_rows": int(xgb_eval_rows),
                },
            ]
        )
        coverage_rows.append(
            {
                "regime": regime,
                "building": building,
                "mode": mode,
                "weather_mode": weather_mode,
                "horizon_h": int(horizon_h),
                "n_test_rows_lstm_raw": int((lstm_pred["building"].astype(str) == building).sum()),
                "n_test_rows_xgb_raw": int((xgb_pred["building"].astype(str) == building).sum()),
                "n_test_rows_common": int(len(common_b)),
                "n_eval_rows_common": int(lstm_eval_rows),
            }
        )

    pred_df = pd.concat(aligned_predictions, ignore_index=True) if aligned_predictions else pd.DataFrame()
    return pred_df, pd.DataFrame(metric_rows), pd.DataFrame(coverage_rows)


def build_comparison_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "regime",
                "mode",
                "model_family",
                "weather_mode",
                "horizon_h",
                "n_buildings",
                "rmse_mean",
                "wape_mean",
                "r2_mean",
                "mae_mean",
            ]
        )
    summary_df = (
        metrics_df.groupby(["regime", "mode", "model_family", "weather_mode", "horizon_h"], as_index=False)
        .agg(
            n_buildings=("building", "nunique"),
            rmse_mean=("rmse", "mean"),
            wape_mean=("wape_pct", "mean"),
            r2_mean=("r2", "mean"),
            mae_mean=("mae", "mean"),
        )
        .sort_values(["regime", "mode", "weather_mode", "horizon_h", "model_family"])
        .reset_index(drop=True)
    )
    return summary_df


def _build_pair_error_log(
    *,
    regime: str,
    scope_building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    elapsed_s: float,
    error: Exception,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": f"{regime}__{scope_building}__pair__{mode}__{weather_mode}__h{int(horizon_h):02d}",
                "regime": regime,
                "building": scope_building if regime == "per_building" else "POOLED",
                "model_family": "pair",
                "mode": mode,
                "weather_mode": weather_mode,
                "horizon_h": int(horizon_h),
                "status": "error",
                "elapsed_s": float(elapsed_s),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        ]
    )


def run_full_comparison(
    config: ExperimentConfig,
    *,
    manifest_df: pd.DataFrame | None = None,
    save_artifacts: bool = True,
    verbose: bool = True,
    resume_existing: bool = True,
    save_after_each_pair: bool = True,
    continue_on_error: bool = True,
) -> ComparisonOutputs:
    paths = ensure_results_dirs(config)
    base_frames = build_base_frame_cache(config)
    contract_df = validate_feature_contract(config, base_frames=base_frames)
    if (contract_df["status"] != "ok").any():
        failures = contract_df.loc[contract_df["status"] != "ok", ["building", "setA_missing_cols", "setB_missing_cols", "missing_fw_horizons", "static_complete_M4"]]
        raise ValueError(f"Feature contract failed for one or more buildings:\n{failures.to_string(index=False)}")

    static_scaler = fit_static_scaler(config, base_frames)
    manifest_scope_df = (
        _normalize_comparison_manifest_frame(manifest_df)
        if manifest_df is not None
        else build_comparison_manifest(config)
    )
    outputs = (
        load_saved_outputs(config, manifest_df=manifest_scope_df)
        if resume_existing
        else ComparisonOutputs(manifest_df=manifest_scope_df.copy())
    )
    outputs.manifest_df = manifest_scope_df.copy()
    outputs = _normalize_outputs(outputs)
    completed_slots = _completed_comparison_slots(outputs, config, manifest_df=manifest_scope_df)

    if save_artifacts:
        _write_outputs(paths, outputs)

    pair_plan_df = _comparison_pair_plan_df(config, manifest_df=manifest_scope_df)
    total_pairs = int(len(pair_plan_df))

    for completed, row in enumerate(pair_plan_df.itertuples(index=False), start=1):
        regime = str(row.regime)
        scope_building = str(row.building)
        mode = str(row.mode)
        weather_mode = str(row.weather_mode)
        horizon_h = int(row.horizon_h)
        slot_id = _comparison_slot_id(regime, scope_building, mode, weather_mode, horizon_h)
        if slot_id in completed_slots:
            if verbose:
                print(
                    f"[{completed:>3}/{total_pairs}] regime={regime} scope={scope_building} "
                    f"mode={mode} weather={weather_mode} h={horizon_h} | resume-skip"
                )
            continue
        if verbose:
            print(
                f"[{completed:>3}/{total_pairs}] regime={regime} scope={scope_building} "
                f"mode={mode} weather={weather_mode} h={horizon_h}"
            )
        pair_buildings = (scope_building,) if regime == "per_building" else tuple(config.buildings)
        pair_config = clone_experiment_config(
            config,
            buildings=pair_buildings,
            horizons=(horizon_h,),
            regimes=(regime,),
            weather_modes=(weather_mode,),
            modes=(mode,),
        )
        pair_base_frames = base_frames if regime != "per_building" else {scope_building: base_frames[scope_building]}
        pair_static_scaler = static_scaler
        pair_start = perf_counter()
        try:
            lstm_artifacts, lstm_log = _run_lstm_regime(
                config=pair_config,
                regime=regime,
                mode=mode,
                weather_mode=weather_mode,
                horizon_h=horizon_h,
                base_frames=pair_base_frames,
                static_scaler=pair_static_scaler,
            )
            xgb_artifacts, xgb_log = _run_xgb_regime(
                config=pair_config,
                regime=regime,
                mode=mode,
                weather_mode=weather_mode,
                horizon_h=horizon_h,
                base_frames=pair_base_frames,
            )
            aligned_pred_df, metrics_df, coverage_df = _align_predictions_and_metrics(
                config=pair_config,
                regime=regime,
                mode=mode,
                weather_mode=weather_mode,
                horizon_h=horizon_h,
                lstm_artifacts=lstm_artifacts,
                xgb_artifacts=xgb_artifacts,
            )
            outputs = _upsert_pair_outputs(
                outputs,
                regime=regime,
                scope_building=scope_building,
                mode=mode,
                weather_mode=weather_mode,
                horizon_h=horizon_h,
                predictions_df=aligned_pred_df,
                metrics_df=metrics_df,
                coverage_df=coverage_df,
                run_log_df=pd.concat([lstm_log, xgb_log], ignore_index=True),
                lstm_history_df=lstm_artifacts.history_df,
                xgb_history_df=xgb_artifacts.history_df,
                normalize=False,
            )
            if _pair_slices_complete(
                metrics_df,
                coverage_df,
                aligned_pred_df,
                pd.concat([lstm_log, xgb_log], ignore_index=True),
                pair_config,
                regime=regime,
            ):
                completed_slots.add(slot_id)
            else:
                completed_slots.discard(slot_id)
        except Exception as exc:
            error_log_df = _build_pair_error_log(
                regime=regime,
                scope_building=scope_building,
                mode=mode,
                weather_mode=weather_mode,
                horizon_h=horizon_h,
                elapsed_s=float(perf_counter() - pair_start),
                error=exc,
            )
            outputs = _upsert_pair_outputs(
                outputs,
                regime=regime,
                scope_building=scope_building,
                mode=mode,
                weather_mode=weather_mode,
                horizon_h=horizon_h,
                predictions_df=pd.DataFrame(),
                metrics_df=pd.DataFrame(),
                coverage_df=pd.DataFrame(),
                run_log_df=error_log_df,
                lstm_history_df=pd.DataFrame(),
                xgb_history_df=pd.DataFrame(),
                normalize=False,
            )
            completed_slots.discard(slot_id)
            if save_artifacts and save_after_each_pair:
                _write_outputs(paths, outputs)
            if verbose:
                print(
                    f"    pair failed | regime={regime} scope={scope_building} "
                    f"mode={mode} weather={weather_mode} h={horizon_h} | {type(exc).__name__}: {exc}"
                )
            if not continue_on_error:
                raise
            continue

        if save_artifacts and save_after_each_pair:
            _write_outputs(paths, outputs)

    outputs = _normalize_outputs(outputs)
    outputs.manifest_df = manifest_scope_df.copy()
    if save_artifacts:
        _write_outputs(paths, outputs)
    return outputs


def _save_fig(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_metric_bars(
    metrics_df: pd.DataFrame,
    *,
    metric: str = "wape_pct",
    regime: str = "per_building",
    mode: str = "M0",
    weather_mode: str = "FW0",
    horizon_h: int = 24,
    save_path: str | Path | None = None,
) -> plt.Figure:
    plot_df = metrics_df[
        (metrics_df["regime"] == regime)
        & (metrics_df["mode"] == mode)
        & (metrics_df["weather_mode"] == weather_mode)
        & (metrics_df["horizon_h"] == int(horizon_h))
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.barplot(data=plot_df, x="building", y=metric, hue="model_family", ax=ax, errorbar=None)
    ax.set_title(f"{metric} by building | {regime} | {mode} | {weather_mode} | h={int(horizon_h)}")
    ax.set_xlabel("building")
    ax.set_ylabel(metric)
    ax.legend(title="model")
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_horizon_curves(
    summary_df: pd.DataFrame,
    *,
    metric: str = "wape_mean",
    regime: str = "per_building",
    mode: str = "M0",
    weather_mode: str = "FW0",
    save_path: str | Path | None = None,
) -> plt.Figure:
    plot_df = summary_df[
        (summary_df["regime"] == regime)
        & (summary_df["mode"] == mode)
        & (summary_df["weather_mode"] == weather_mode)
    ].copy()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.lineplot(data=plot_df, x="horizon_h", y=metric, hue="model_family", marker="o", ax=ax)
    ax.set_title(f"{metric} across horizons | {regime} | {mode} | {weather_mode}")
    ax.set_xlabel("horizon [h]")
    ax.set_ylabel(metric)
    ax.legend(title="model")
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_training_curves_lstm(
    lstm_history_df: pd.DataFrame,
    *,
    run_id: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    plot_df = lstm_history_df[lstm_history_df["run_id"] == run_id].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    if not plot_df.empty:
        ax.plot(plot_df["epoch"], plot_df["loss"], label="train_loss")
        if "val_loss" in plot_df:
            ax.plot(plot_df["epoch"], plot_df["val_loss"], label="val_loss")
    ax.set_title(f"LSTM loss curves | {run_id}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_training_curves_xgb(
    xgb_eval_history_df: pd.DataFrame,
    *,
    run_id: str,
    metric: str = "rmse",
    save_path: str | Path | None = None,
) -> plt.Figure:
    plot_df = xgb_eval_history_df[(xgb_eval_history_df["run_id"] == run_id) & (xgb_eval_history_df["metric"] == metric)].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    if not plot_df.empty:
        for dataset_name, sub_df in plot_df.groupby("dataset"):
            ax.plot(sub_df["iteration"], sub_df["value"], label=dataset_name)
    ax.set_title(f"XGBoost {metric} curves | {run_id}")
    ax.set_xlabel("boosting round")
    ax.set_ylabel(metric)
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_learning_rate_curve_lstm(
    lstm_history_df: pd.DataFrame,
    *,
    run_id: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    plot_df = lstm_history_df[lstm_history_df["run_id"] == run_id].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    if not plot_df.empty and "learning_rate" in plot_df:
        ax.plot(plot_df["epoch"], plot_df["learning_rate"], label="learning_rate")
    ax.set_title(f"LSTM learning-rate curve | {run_id}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, save_path)


def _select_prediction_rows(
    predictions_df: pd.DataFrame,
    *,
    regime: str,
    model_family: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    building: str,
) -> pd.DataFrame:
    return predictions_df[
        (predictions_df["regime"] == regime)
        & (predictions_df["model_family"] == model_family)
        & (predictions_df["mode"] == mode)
        & (predictions_df["weather_mode"] == weather_mode)
        & (predictions_df["horizon_h"] == int(horizon_h))
        & (predictions_df["building"] == building)
    ].sort_values("datetime").reset_index(drop=True)


def plot_predictions_vs_actual(
    predictions_df: pd.DataFrame,
    *,
    regime: str,
    model_family: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    building: str,
    max_points: int = 600,
    save_path: str | Path | None = None,
) -> plt.Figure:
    plot_df = _select_prediction_rows(
        predictions_df,
        regime=regime,
        model_family=model_family,
        mode=mode,
        weather_mode=weather_mode,
        horizon_h=horizon_h,
        building=building,
    )
    if len(plot_df) > max_points:
        idx = np.linspace(0, len(plot_df) - 1, max_points).astype(int)
        plot_df = plot_df.iloc[idx].copy()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(plot_df["datetime"], plot_df["y_true"], label="actual", alpha=0.9)
    ax.plot(plot_df["datetime"], plot_df["y_pred"], label="predicted", alpha=0.8)
    ax.set_title(f"Predictions vs actual | {model_family} | {building} | {mode} | {weather_mode} | h={int(horizon_h)}")
    ax.set_xlabel("datetime")
    ax.set_ylabel("heat demand")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_residual_scatter(
    predictions_df: pd.DataFrame,
    *,
    regime: str,
    model_family: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    building: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    plot_df = _select_prediction_rows(
        predictions_df,
        regime=regime,
        model_family=model_family,
        mode=mode,
        weather_mode=weather_mode,
        horizon_h=horizon_h,
        building=building,
    ).copy()
    plot_df["residual"] = plot_df["y_pred"] - plot_df["y_true"]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(plot_df["y_true"], plot_df["residual"], s=8, alpha=0.35)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title(f"Residual scatter | {model_family} | {building} | {mode} | {weather_mode} | h={int(horizon_h)}")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted - actual")
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_residual_histogram(
    predictions_df: pd.DataFrame,
    *,
    regime: str,
    model_family: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    building: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    plot_df = _select_prediction_rows(
        predictions_df,
        regime=regime,
        model_family=model_family,
        mode=mode,
        weather_mode=weather_mode,
        horizon_h=horizon_h,
        building=building,
    ).copy()
    plot_df["residual"] = plot_df["y_pred"] - plot_df["y_true"]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sns.histplot(data=plot_df, x="residual", bins=30, kde=True, ax=ax)
    ax.set_title(f"Residual histogram | {model_family} | {building} | {mode} | {weather_mode} | h={int(horizon_h)}")
    ax.set_xlabel("predicted - actual")
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_parity_scatter(
    predictions_df: pd.DataFrame,
    *,
    regime: str,
    model_family: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    building: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    plot_df = _select_prediction_rows(
        predictions_df,
        regime=regime,
        model_family=model_family,
        mode=mode,
        weather_mode=weather_mode,
        horizon_h=horizon_h,
        building=building,
    )
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(plot_df["y_true"], plot_df["y_pred"], s=8, alpha=0.35)
    if not plot_df.empty:
        lim = max(float(plot_df["y_true"].max()), float(plot_df["y_pred"].max()))
        ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1.0)
    ax.set_title(f"Parity scatter | {model_family} | {building} | {mode} | {weather_mode} | h={int(horizon_h)}")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    fig.tight_layout()
    return _save_fig(fig, save_path)
