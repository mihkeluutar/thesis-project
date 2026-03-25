from __future__ import annotations

from dataclasses import dataclass, field
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
DEFAULT_WEATHER_MODES = ("FW0", "FW2", "FW1")
DEFAULT_MODES = ("M0", "M2", "M4")
DEFAULT_MODEL_FAMILIES = ("lstm", "xgboost")

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
    "M2": LSTM_BASE_TEMPORAL_FEATURES + LSTM_SYSTEM_DYNAMIC_FEATURES,
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


def _csv_data_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        n_lines = sum(1 for _ in f)
    return max(0, n_lines - 1)


def print_resume_diagnostics(config: ExperimentConfig) -> None:
    """Print where artifacts live, CSV sizes, and pair-level resume progress (fast path)."""
    paths = ensure_results_dirs(config)
    print("--- resume diagnostics ---")
    print(f"cwd: {Path.cwd()}")
    print(f"results_dir (resolved): {config.results_dir.resolve()}")
    for key in ("manifest", "metrics", "predictions", "coverage", "run_log"):
        p = paths[key]
        print(f"  {paths[key].name}: exists={p.exists()} data_rows={_csv_data_row_count(p)}")

    manifest_df = build_comparison_manifest(config)
    outputs = load_saved_outputs(config, manifest_df=manifest_df)
    metrics_by = _artifact_frames_by_comparison_slot(outputs.comparison_metrics_df)
    coverage_by = _artifact_frames_by_comparison_slot(outputs.comparison_coverage_df)
    pred_by = _artifact_frames_by_comparison_slot(outputs.comparison_predictions_df)
    run_by = _artifact_frames_by_comparison_slot(outputs.run_log_df)

    total_pairs = sum(
        (len(config.buildings) if regime == "per_building" else 1)
        * len(config.modes)
        * len(config.weather_modes)
        * len(config.horizons)
        for regime in config.regimes
    )
    n_complete = 0
    first_bad: tuple[int, str, str, str, str, int] | None = None
    idx = 0
    for regime in config.regimes:
        scope_buildings = config.buildings if regime == "per_building" else ("POOLED",)
        for scope_building in scope_buildings:
            for mode in config.modes:
                for weather_mode in config.weather_modes:
                    for horizon_h in config.horizons:
                        idx += 1
                        sid = _comparison_slot_id(regime, scope_building, mode, weather_mode, int(horizon_h))
                        ok = _pair_slices_complete(
                            metrics_by.get(sid, pd.DataFrame()),
                            coverage_by.get(sid, pd.DataFrame()),
                            pred_by.get(sid, pd.DataFrame()),
                            run_by.get(sid, pd.DataFrame()),
                            config,
                            regime=regime,
                        )
                        if ok:
                            n_complete += 1
                        elif first_bad is None:
                            first_bad = (idx, regime, scope_building, mode, weather_mode, int(horizon_h))
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
    return _normalize_outputs(updated)


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
    out = frame.copy()
    if weather_mode == "FW0":
        return out, []

    fw_cols = future_weather_feature_cols(out.columns, horizon_h)
    if weather_mode == "FW2":
        if not fw_cols:
            raise KeyError(f"No proxy future-weather columns found for horizon {horizon_h}")
        return out, fw_cols

    if weather_mode == "FW1":
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


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
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
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    meta_rows: list[dict[str, Any]] = []
    mask = np.asarray(split_mask, dtype=bool)
    heating = heating_mask(df_scaled)
    for end_idx in range(int(lookback), len(df_scaled)):
        if not mask[end_idx]:
            continue
        start_idx = end_idx - int(lookback)
        if not mask[start_idx:end_idx].all():
            continue
        window = df_scaled.iloc[start_idx:end_idx]
        target_val = df_scaled.iloc[end_idx][target_col]
        if window[dynamic_cols].isna().any().any() or pd.isna(target_val):
            continue
        X_list.append(window[dynamic_cols].values.astype("float32"))
        y_list.append(float(target_val))
        meta_rows.append(
            {
                "building": str(df_scaled.iloc[end_idx]["building"]),
                "datetime": pd.Timestamp(df_scaled.iloc[end_idx]["datetime"]),
                "is_heating_eval": bool(heating.iloc[end_idx]),
            }
        )
    if not X_list:
        return (
            np.empty((0, int(lookback), len(dynamic_cols)), dtype="float32"),
            np.empty((0,), dtype="float32"),
            pd.DataFrame(columns=["building", "datetime", "is_heating_eval"]),
        )
    return np.stack(X_list, axis=0), np.array(y_list, dtype="float32"), pd.DataFrame(meta_rows)


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
    set_all_seeds(config.random_seed)
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
    manifest_df = build_comparison_manifest(config)
    outputs = load_saved_outputs(config, manifest_df=manifest_df) if resume_existing else ComparisonOutputs(manifest_df=manifest_df.copy())
    outputs.manifest_df = manifest_df.copy()
    outputs = _normalize_outputs(outputs)

    if save_artifacts:
        _write_outputs(paths, outputs)

    total_pairs = sum((len(config.buildings) if regime == "per_building" else 1) * len(config.modes) * len(config.weather_modes) * len(config.horizons) for regime in config.regimes)
    completed = 0

    for regime in config.regimes:
        scope_buildings = config.buildings if regime == "per_building" else ("POOLED",)
        for scope_building in scope_buildings:
            for mode in config.modes:
                for weather_mode in config.weather_modes:
                    for horizon_h in config.horizons:
                        completed += 1
                        if pair_is_complete(
                            outputs,
                            config,
                            regime=regime,
                            scope_building=scope_building,
                            mode=mode,
                            weather_mode=weather_mode,
                            horizon_h=int(horizon_h),
                        ):
                            if verbose:
                                print(
                                    f"[{completed:>3}/{total_pairs}] regime={regime} scope={scope_building} "
                                    f"mode={mode} weather={weather_mode} h={int(horizon_h)} | resume-skip"
                                )
                            continue
                        if verbose:
                            print(
                                f"[{completed:>3}/{total_pairs}] regime={regime} scope={scope_building} "
                                f"mode={mode} weather={weather_mode} h={int(horizon_h)}"
                            )
                        pair_config = config if regime != "per_building" else ExperimentConfig(
                            buildings=(scope_building,),
                            horizons=config.horizons,
                            regimes=(regime,),
                            weather_modes=config.weather_modes,
                            modes=config.modes,
                            train_end=config.train_end,
                            test_start=config.test_start,
                            lookback_hours=config.lookback_hours,
                            validation_fraction=config.validation_fraction,
                            results_dir=config.results_dir,
                            lstm_architecture_id=config.lstm_architecture_id,
                            xgb_preset_id=config.xgb_preset_id,
                            random_seed=config.random_seed,
                            batch_size=config.batch_size,
                            epochs=config.epochs,
                            early_stopping_patience=config.early_stopping_patience,
                            learning_rate=config.learning_rate,
                            model_families=config.model_families,
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
                                horizon_h=int(horizon_h),
                                base_frames=pair_base_frames,
                                static_scaler=pair_static_scaler,
                            )
                            xgb_artifacts, xgb_log = _run_xgb_regime(
                                config=pair_config,
                                regime=regime,
                                mode=mode,
                                weather_mode=weather_mode,
                                horizon_h=int(horizon_h),
                                base_frames=pair_base_frames,
                            )
                            aligned_pred_df, metrics_df, coverage_df = _align_predictions_and_metrics(
                                config=pair_config,
                                regime=regime,
                                mode=mode,
                                weather_mode=weather_mode,
                                horizon_h=int(horizon_h),
                                lstm_artifacts=lstm_artifacts,
                                xgb_artifacts=xgb_artifacts,
                            )
                            outputs = _upsert_pair_outputs(
                                outputs,
                                regime=regime,
                                scope_building=scope_building,
                                mode=mode,
                                weather_mode=weather_mode,
                                horizon_h=int(horizon_h),
                                predictions_df=aligned_pred_df,
                                metrics_df=metrics_df,
                                coverage_df=coverage_df,
                                run_log_df=pd.concat([lstm_log, xgb_log], ignore_index=True),
                                lstm_history_df=lstm_artifacts.history_df,
                                xgb_history_df=xgb_artifacts.history_df,
                            )
                        except Exception as exc:
                            error_log_df = _build_pair_error_log(
                                regime=regime,
                                scope_building=scope_building,
                                mode=mode,
                                weather_mode=weather_mode,
                                horizon_h=int(horizon_h),
                                elapsed_s=float(perf_counter() - pair_start),
                                error=exc,
                            )
                            outputs = _upsert_pair_outputs(
                                outputs,
                                regime=regime,
                                scope_building=scope_building,
                                mode=mode,
                                weather_mode=weather_mode,
                                horizon_h=int(horizon_h),
                                predictions_df=pd.DataFrame(),
                                metrics_df=pd.DataFrame(),
                                coverage_df=pd.DataFrame(),
                                run_log_df=error_log_df,
                                lstm_history_df=pd.DataFrame(),
                                xgb_history_df=pd.DataFrame(),
                            )
                            if save_artifacts and save_after_each_pair:
                                _write_outputs(paths, outputs)
                            if verbose:
                                print(
                                    f"    pair failed | regime={regime} scope={scope_building} "
                                    f"mode={mode} weather={weather_mode} h={int(horizon_h)} | {type(exc).__name__}: {exc}"
                                )
                            if not continue_on_error:
                                raise
                            continue

                        if save_artifacts and save_after_each_pair:
                            _write_outputs(paths, outputs)

    outputs = _normalize_outputs(outputs)
    outputs.manifest_df = manifest_df.copy()
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
