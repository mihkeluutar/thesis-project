from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import importlib.util
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_DIR = DATA_DIR / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURE_METADATA_FILE = FEATURE_DIR / "feature_metadata.csv"

DEFAULT_HORIZONS = (1, 2, 3, 6, 8, 12, 16, 20, 24, 36)
DEFAULT_AUDIT_HORIZONS = (6, 12, 24, 36)
DEFAULT_AUDIT_BUILDINGS = ("U05", "U06")
DEFAULT_AUDIT_MODES = ("M0", "M3")
DEFAULT_MODES = ("M0", "M1", "M2", "M3")
DEFAULT_TRAIN_END = pd.Timestamp("2023-12-31 23:00:00")
DEFAULT_TEST_START = DEFAULT_TRAIN_END + pd.Timedelta(hours=1)
HEATING_TEMP_THRESHOLD_C = 15.0

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

MODE_FEATURES = {
    "M0": LSTM_BASE_TEMPORAL_FEATURES.copy(),
    "M1": LSTM_BASE_TEMPORAL_FEATURES + LSTM_WEATHER_MEMORY_FEATURES,
    "M2": LSTM_BASE_TEMPORAL_FEATURES + LSTM_SYSTEM_DYNAMIC_FEATURES,
    "M3": LSTM_BASE_TEMPORAL_FEATURES + LSTM_SYSTEM_DYNAMIC_FEATURES + LSTM_WEATHER_MEMORY_FEATURES,
}
for mode_name, cols in list(MODE_FEATURES.items()):
    seen: set[str] = set()
    MODE_FEATURES[mode_name] = [c for c in cols if not (c in seen or seen.add(c))]

SAFE_GLOBAL_STATIC_FEATURES = [
    "stat_n_points",
    "stat_n_vent_points",
    "stat_n_heat_points",
    "stat_n_dhw_points",
    "stat_has_multiple_hoone_labels",
    "stat_heated_area_m2",
    "stat_usage_non_res_share_of_heated",
    "stat_building_age_years",
    "stat_heated_area_missing",
    "stat_has_energy_class",
    "stat_ventilation_has_heat_recovery",
    "stat_has_cooling_system",
    "stat_vent_class_basic",
    "stat_vent_class_none",
    "stat_vent_class_rich",
    "stat_energy_class_d",
    "stat_energy_class_e",
    "stat_building_type_muu_erihoone",
    "stat_building_type_muu_haridus_voi_teadushoone",
    "stat_building_type_raamatukogu",
    "stat_building_type_ulikooli_rakenduskorgkooli_oppehoone",
    "stat_ventilation_type_soojustagastusega_ventilatsioon",
    "stat_cooling_system_lokaalne_jahutus",
    "stat_missing_heated_area_m2",
    "stat_missing_usage_non_res_share_of_heated",
    "stat_missing_building_age_years",
    "stat_missing_heated_area_missing",
    "stat_missing_has_energy_class",
]

PRESET_CANDIDATES = (
    {"preset_id": "P01_md3_lr003_mc5", "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5},
    {"preset_id": "P02_md3_lr005_mc1", "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 1},
    {"preset_id": "P03_md5_lr005_mc5", "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 5},
    {"preset_id": "P04_md5_lr010_mc1", "max_depth": 5, "learning_rate": 0.10, "min_child_weight": 1},
    {"preset_id": "P05_md7_lr005_mc5", "max_depth": 7, "learning_rate": 0.05, "min_child_weight": 5},
)


def htag(horizon_h: int) -> str:
    return f"h{int(horizon_h):02d}"


def build_future_summary_columns(series: pd.Series, horizon_h: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    future_mat = pd.concat([series.shift(-k) for k in range(1, int(horizon_h) + 1)], axis=1)
    mean_col = future_mat.mean(axis=1)
    min_col = future_mat.min(axis=1)
    end_col = series.shift(-int(horizon_h))
    return mean_col, min_col, end_col


def add_oracle_future_weather_features(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    required = {"feat_outdoor_temp_c", "feat_rh_pct"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Future-weather features require source columns: {missing}")
    out = df.copy()
    future_cols: dict[str, pd.Series] = {}
    for horizon_h in horizons:
        tag = htag(horizon_h)
        temp_mean, temp_min, temp_end = build_future_summary_columns(df["feat_outdoor_temp_c"], int(horizon_h))
        rh_mean, _, _ = build_future_summary_columns(df["feat_rh_pct"], int(horizon_h))
        future_cols[f"feat_fw_temp_mean_{tag}"] = temp_mean
        future_cols[f"feat_fw_temp_min_{tag}"] = temp_min
        future_cols[f"feat_fw_temp_end_{tag}"] = temp_end
        future_cols[f"feat_fw_rh_mean_{tag}"] = rh_mean
    return pd.concat([out, pd.DataFrame(future_cols, index=out.index)], axis=1)


def future_weather_feature_cols(horizon_h: int) -> list[str]:
    tag = htag(horizon_h)
    return [
        f"feat_fw_temp_mean_{tag}",
        f"feat_fw_temp_min_{tag}",
        f"feat_fw_temp_end_{tag}",
        f"feat_fw_rh_mean_{tag}",
    ]


def mode_feature_cols(mode: str, horizon_h: int, config: "NotebookConfig") -> list[str]:
    cols = list(MODE_FEATURES[mode])
    if config.use_future_weather:
        cols.extend(future_weather_feature_cols(horizon_h))
    seen: set[str] = set()
    return [c for c in cols if not (c in seen or seen.add(c))]


@dataclass(frozen=True)
class NotebookConfig:
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    audit_horizons: tuple[int, ...] = DEFAULT_AUDIT_HORIZONS
    audit_buildings: tuple[str, ...] = DEFAULT_AUDIT_BUILDINGS
    audit_modes: tuple[str, ...] = DEFAULT_AUDIT_MODES
    modes: tuple[str, ...] = DEFAULT_MODES
    buildings: tuple[str, ...] | None = None
    train_end: pd.Timestamp = DEFAULT_TRAIN_END
    test_start: pd.Timestamp = DEFAULT_TEST_START
    validation_fraction: float = 0.10
    seed: int = 42
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    subsample: float = 0.80
    colsample_bytree: float = 0.80
    reg_lambda: float = 5.0
    objective: str = "reg:squarederror"
    booster: str = "gbtree"
    tree_method: str = "hist"
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)
    min_train_rows: int = 96
    min_validation_rows: int = 24
    min_test_rows: int = 24
    preset_margin_wape_pp: float = 0.20
    outputs_date_tag: str = "20260322"
    use_future_weather: bool = False


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except Exception:
        return False
    return True


def require_xgboost():
    try:
        from xgboost import XGBRegressor  # type: ignore
    except Exception as e:
        if importlib.util.find_spec("xgboost") is None:
            raise ImportError(
                "xgboost is not installed in the current Python environment. "
                "Install it in the thesis environment before running the XGBoost notebook."
            ) from e
        hint = ""
        if sys.platform == "darwin":
            hint = (
                " On macOS, install OpenMP with the Homebrew that matches your machine "
                "(Apple Silicon: `/opt/homebrew/bin/brew install libomp`; Intel: `/usr/local/bin/brew install libomp`)."
            )
        raise ImportError(
            "xgboost is installed but its native library failed to load (often missing libomp on macOS)."
            + hint
            + f" Detail: {e!r}"
        ) from e
    return XGBRegressor


def artifact_paths(config: NotebookConfig) -> dict[str, Path]:
    tag = config.outputs_date_tag
    variant_suffix = "_fw1" if config.use_future_weather else ""
    return {
        "audit": config.results_dir / f"xgboost_architecture_lock{variant_suffix}_audit_{tag}.csv",
        "per_building": config.results_dir / f"xgboost_architecture_lock{variant_suffix}_per_building_{tag}.csv",
        "global": config.results_dir / f"xgboost_architecture_lock{variant_suffix}_global_{tag}.csv",
        "summary": config.results_dir / f"xgboost_architecture_lock{variant_suffix}_summary_{tag}.csv",
        "matched": config.results_dir / f"xgboost_vs_lstm_matched_summary{variant_suffix}_{tag}.csv",
    }


def load_feature_metadata(path: Path = FEATURE_METADATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"building", "path_setA", "path_setB"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Feature metadata is missing required columns: {sorted(missing)}")
    return df


def discover_buildings(meta: pd.DataFrame | None = None) -> list[str]:
    meta = meta if meta is not None else load_feature_metadata()
    buildings: list[str] = []
    for _, row in meta.iterrows():
        path_a = PROJECT_ROOT / str(row["path_setA"])
        path_b = PROJECT_ROOT / str(row["path_setB"])
        if path_a.exists() and path_b.exists():
            buildings.append(str(row["building"]))
    return sorted(buildings)


def selected_buildings(config: NotebookConfig, meta: pd.DataFrame | None = None) -> list[str]:
    available = discover_buildings(meta)
    if config.buildings is None:
        return available
    available_set = set(available)
    return [building for building in config.buildings if building in available_set]


def feature_path_for_building(building: str, set_name: str, meta: pd.DataFrame | None = None) -> Path:
    meta = meta if meta is not None else load_feature_metadata()
    row = meta.loc[meta["building"] == building]
    if row.empty:
        raise KeyError(f"Building {building} not present in feature metadata")
    path_col = f"path_{set_name}"
    if path_col not in row.columns:
        raise KeyError(f"Feature metadata missing column {path_col}")
    csv_path = PROJECT_ROOT / str(row.iloc[0][path_col])
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return csv_path


def load_feature_frame(building: str, set_name: str, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    csv_path = feature_path_for_building(building, set_name, meta=meta)
    df = pd.read_csv(csv_path, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df.assign(building=building)


def add_cumulative_targets(df: pd.DataFrame, horizons: tuple[int, ...], source_col: str = "heat_kwh") -> pd.DataFrame:
    out = df.copy()
    heat = out[source_col].astype(float)
    target_cols: dict[str, pd.Series] = {}
    for horizon in horizons:
        target_cols[f"target_cum_h{int(horizon)}"] = sum(heat.shift(-i) for i in range(int(horizon)))
    return pd.concat([out, pd.DataFrame(target_cols, index=out.index)], axis=1)


def heating_mask(df: pd.DataFrame) -> pd.Series:
    if "feat_is_heating_weather" in df.columns:
        return df["feat_is_heating_weather"].fillna(0).astype(float) > 0.5
    if "feat_outdoor_temp_c" in df.columns:
        return df["feat_outdoor_temp_c"].astype(float) < HEATING_TEMP_THRESHOLD_C
    return pd.Series(True, index=df.index)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "wape_pct": np.nan, "r2": np.nan}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = float(np.sum(np.abs(y_true)))
    wape = float(100.0 * np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    return {"rmse": rmse, "mae": mae, "wape_pct": wape, "r2": r2}


def _build_issue_mask(dt: pd.Series, horizon_h: int, train_end: pd.Timestamp, test_start: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    train_issue_end = train_end - pd.Timedelta(hours=int(horizon_h) - 1)
    train_mask = (dt <= train_issue_end).to_numpy(dtype=bool)
    test_mask = (dt >= test_start).to_numpy(dtype=bool)
    return train_mask, test_mask, train_issue_end


def _numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _build_model_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_name: str,
    row_mask: np.ndarray,
    horizon_h: int,
) -> pd.DataFrame:
    keep_cols = ["datetime", "building", "heat_kwh", target_name] + feature_cols
    if "feat_is_heating_weather" in df.columns:
        keep_cols.append("feat_is_heating_weather")
    out = df.loc[row_mask, keep_cols].copy()
    out = _numeric_frame(out, [c for c in feature_cols if c in out.columns] + ["heat_kwh", target_name])
    out = out.loc[out[target_name].notna()].copy()
    out["horizon_h"] = int(horizon_h)
    out["baseline_pred"] = float(horizon_h) * out["heat_kwh"].astype(float)
    out["is_heating_eval"] = heating_mask(out).to_numpy(dtype=bool)
    out = out.sort_values(["datetime", "building"]).reset_index(drop=True)
    return out


def _split_fit_validation(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or frac <= 0:
        return df.copy(), df.iloc[0:0].copy()
    n_val = max(1, int(round(len(df) * frac)))
    n_val = min(n_val, len(df) - 1) if len(df) > 1 else 0
    split_at = len(df) - n_val
    if split_at <= 0:
        return df.iloc[0:0].copy(), df.copy()
    return df.iloc[:split_at].copy(), df.iloc[split_at:].copy()


def _predict_with_best_iteration(model, X: pd.DataFrame) -> np.ndarray:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return model.predict(X)
    try:
        return model.predict(X, iteration_range=(0, int(best_iteration) + 1))
    except TypeError:
        return model.predict(X)


def _fit_xgb_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, preset: dict[str, Any], config: NotebookConfig):
    XGBRegressor = require_xgboost()
    model = XGBRegressor(
        objective=config.objective,
        booster=config.booster,
        tree_method=config.tree_method,
        n_estimators=config.n_estimators,
        max_depth=int(preset["max_depth"]),
        learning_rate=float(preset["learning_rate"]),
        min_child_weight=float(preset["min_child_weight"]),
        subsample=float(config.subsample),
        colsample_bytree=float(config.colsample_bytree),
        reg_lambda=float(config.reg_lambda),
        random_state=int(config.seed),
        n_jobs=-1,
        eval_metric="rmse",
    )
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=int(config.early_stopping_rounds),
            verbose=False,
        )
    except TypeError:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    return model


def _evaluate_on_heating_rows(frame: pd.DataFrame, preds: np.ndarray, target_name: str) -> dict[str, float]:
    eval_mask = frame["is_heating_eval"].to_numpy(dtype=bool)
    if not eval_mask.any():
        eval_mask = np.ones(len(frame), dtype=bool)
    y_true = frame.loc[eval_mask, target_name].astype(float).to_numpy()
    y_pred = np.asarray(preds, dtype=float)[eval_mask]
    return compute_metrics(y_true, y_pred)


def _run_single_case(
    frame: pd.DataFrame,
    scope: str,
    building_or_global: str,
    mode: str,
    horizon_h: int,
    feature_cols: list[str],
    preset: dict[str, Any],
    config: NotebookConfig,
    target_name: str,
) -> dict[str, Any]:
    dt = pd.to_datetime(frame["datetime"])
    train_mask, test_mask, train_issue_end = _build_issue_mask(dt, horizon_h, config.train_end, config.test_start)
    train_df_all = _build_model_frame(frame, feature_cols, target_name, train_mask, horizon_h)
    test_df = _build_model_frame(frame, feature_cols, target_name, test_mask, horizon_h)
    fit_df, val_df = _split_fit_validation(train_df_all, config.validation_fraction)

    row: dict[str, Any] = {
        "scope": scope,
        "building_or_global": building_or_global,
        "mode": mode,
        "horizon_h": int(horizon_h),
        "target_name": target_name,
        "preset_id": preset["preset_id"],
        "n_features": len(feature_cols),
        "feature_list": ",".join(feature_cols),
        "n_train_rows": int(len(fit_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "train_issue_end": str(train_issue_end),
        "status": "ok",
    }

    if len(fit_df) < config.min_train_rows:
        row["status"] = "skipped_insufficient_train_rows"
        return row
    if len(val_df) < config.min_validation_rows:
        row["status"] = "skipped_insufficient_validation_rows"
        return row
    if len(test_df) < config.min_test_rows:
        row["status"] = "skipped_insufficient_test_rows"
        return row

    X_train = fit_df[feature_cols]
    y_train = fit_df[target_name].astype(float)
    X_val = val_df[feature_cols]
    y_val = val_df[target_name].astype(float)
    X_test = test_df[feature_cols]

    model = _fit_xgb_model(X_train, y_train, X_val, y_val, preset, config)
    val_pred = _predict_with_best_iteration(model, X_val)
    test_pred = _predict_with_best_iteration(model, X_test)
    baseline_pred = test_df["baseline_pred"].astype(float).to_numpy()

    val_metrics = _evaluate_on_heating_rows(val_df, val_pred, target_name)
    test_metrics = _evaluate_on_heating_rows(test_df, test_pred, target_name)
    baseline_metrics = _evaluate_on_heating_rows(test_df, baseline_pred, target_name)

    row.update(
        {
            "best_iteration": int(getattr(model, "best_iteration", config.n_estimators - 1) or config.n_estimators - 1),
            "val_wape": val_metrics["wape_pct"],
            "val_rmse": val_metrics["rmse"],
            "rmse": test_metrics["rmse"],
            "mae": test_metrics["mae"],
            "wape_pct": test_metrics["wape_pct"],
            "r2": test_metrics["r2"],
            "baseline_wape_pct": baseline_metrics["wape_pct"],
            "delta_wape_vs_baseline": test_metrics["wape_pct"] - baseline_metrics["wape_pct"]
            if pd.notna(test_metrics["wape_pct"]) and pd.notna(baseline_metrics["wape_pct"])
            else np.nan,
            "n_test_eval_rows": int(test_df["is_heating_eval"].sum()) if int(test_df["is_heating_eval"].sum()) > 0 else int(len(test_df)),
        }
    )
    return row


def _load_frames_for_buildings(buildings: list[str], horizons: tuple[int, ...], meta: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    frames_a: dict[str, pd.DataFrame] = {}
    frames_b: dict[str, pd.DataFrame] = {}
    for building in buildings:
        frame_a = add_cumulative_targets(load_feature_frame(building, "setA", meta=meta), horizons)
        frame_b = add_cumulative_targets(load_feature_frame(building, "setB", meta=meta), horizons)
        frames_a[building] = add_oracle_future_weather_features(frame_a, horizons)
        frames_b[building] = add_oracle_future_weather_features(frame_b, horizons)
    return frames_a, frames_b


def _global_static_cols(frames_b: dict[str, pd.DataFrame]) -> list[str]:
    available = set(SAFE_GLOBAL_STATIC_FEATURES)
    for frame in frames_b.values():
        available &= set(frame.columns)
    return [c for c in SAFE_GLOBAL_STATIC_FEATURES if c in available]


def _load_existing_frame(path: Path, key_cols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or key_cols is None:
        return df
    keep_cols = [col for col in key_cols if col in df.columns]
    if not keep_cols:
        return df
    return df.drop_duplicates(subset=keep_cols, keep="last").reset_index(drop=True)


def _frame_to_rows_and_keys(df: pd.DataFrame, key_cols: list[str]) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    if df.empty:
        return [], set()
    rows = df.to_dict("records")
    keys = {tuple(row.get(col) for col in key_cols) for row in rows}
    return rows, keys


def _rows_to_frame(rows: list[dict[str, Any]], sort_cols: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    use_sort_cols = [col for col in sort_cols if col in df.columns]
    if use_sort_cols:
        df = df.sort_values(use_sort_cols, ignore_index=True)
    return df


def _persist_rows(path: Path, rows: list[dict[str, Any]], sort_cols: list[str]) -> pd.DataFrame:
    df = _rows_to_frame(rows, sort_cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _write_partial_summary_artifacts(config: NotebookConfig) -> None:
    paths = artifact_paths(config)
    per_df = _load_existing_frame(paths["per_building"])
    global_df = _load_existing_frame(paths["global"])
    if per_df.empty and global_df.empty:
        return
    build_summary_artifacts(config, per_df=per_df, global_df=global_df)


def build_mode_catalog_df(config: NotebookConfig | None = None, frames_b: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    config = config or NotebookConfig()
    if frames_b is None:
        meta = load_feature_metadata()
        buildings = selected_buildings(config, meta)
        _, frames_b = _load_frames_for_buildings(buildings, config.horizons, meta)
    global_static = _global_static_cols(frames_b) if frames_b else []
    rows = []
    for mode in config.modes:
        fw_cols = future_weather_feature_cols(config.horizons[0]) if config.use_future_weather and config.horizons else []
        rows.append(
            {
                "mode": mode,
                "dynamic_feature_count": len(MODE_FEATURES[mode]),
                "dynamic_feature_list": ", ".join(MODE_FEATURES[mode]),
                "future_weather_feature_count": len(fw_cols),
                "future_weather_feature_list": ", ".join(fw_cols),
                "global_static_feature_count": len(global_static),
                "global_static_feature_list": ", ".join(global_static),
            }
        )
    return pd.DataFrame(rows)


def build_target_preview(config: NotebookConfig, building: str = "U05", horizon_h: int = 6) -> pd.DataFrame:
    meta = load_feature_metadata()
    df = add_cumulative_targets(load_feature_frame(building, "setA", meta=meta), config.horizons)
    df = add_oracle_future_weather_features(df, config.horizons)
    target_name = f"target_cum_h{int(horizon_h)}"
    dt = pd.to_datetime(df["datetime"])
    train_mask, _, train_issue_end = _build_issue_mask(dt, horizon_h, config.train_end, config.test_start)
    first_valid = df.loc[df[target_name].notna(), ["datetime", "heat_kwh", target_name]].head(3)
    last_train = df.loc[train_mask & df[target_name].notna(), ["datetime", "heat_kwh", target_name]].tail(3)
    preview = pd.concat([first_valid, last_train], axis=0).drop_duplicates().reset_index(drop=True)
    if config.use_future_weather:
        fw_cols = future_weather_feature_cols(horizon_h)
        preview = preview.merge(
            df.loc[:, ["datetime"] + fw_cols].drop_duplicates(subset=["datetime"]),
            on="datetime",
            how="left",
        )
    preview["train_issue_end"] = str(train_issue_end)
    return preview


def build_building_catalog_df(config: NotebookConfig) -> pd.DataFrame:
    meta = load_feature_metadata()
    buildings = selected_buildings(config, meta)
    if not buildings:
        raise ValueError("No buildings available for the current notebook config")
    frames_a, frames_b = _load_frames_for_buildings(buildings, config.horizons, meta)
    static_cols = _global_static_cols(frames_b)
    rows: list[dict[str, Any]] = []
    for building in buildings:
        frame_a = frames_a[building]
        frame_b = frames_b[building]
        dt = pd.to_datetime(frame_a["datetime"])
        row = {
            "building": building,
            "rows_setA": int(len(frame_a)),
            "rows_setB": int(len(frame_b)),
            "datetime_start": str(dt.min()),
            "datetime_end": str(dt.max()),
            "heating_share_setA_pct": float(100.0 * heating_mask(frame_a).mean()),
            "global_static_complete": bool(not frame_b[static_cols].iloc[[0]].isna().any().any()) if static_cols else False,
        }
        for horizon_h in config.horizons:
            target_name = f"target_cum_h{int(horizon_h)}"
            train_mask, test_mask, _ = _build_issue_mask(dt, int(horizon_h), config.train_end, config.test_start)
            row[f"{target_name}_nonnull_train"] = int(frame_a.loc[train_mask, target_name].notna().sum())
            row[f"{target_name}_nonnull_test"] = int(frame_a.loc[test_mask, target_name].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("building").reset_index(drop=True)


def _preset_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    ok = audit_df.loc[audit_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    summary = (
        ok.groupby("preset_id", as_index=False)
        .agg(
            n_runs=("preset_id", "size"),
            mean_val_wape=("val_wape", "mean"),
            mean_val_rmse=("val_rmse", "mean"),
            mean_best_iteration=("best_iteration", "mean"),
            max_depth=("max_depth", "first"),
            learning_rate=("learning_rate", "first"),
            min_child_weight=("min_child_weight", "first"),
        )
        .sort_values(["mean_val_wape", "mean_val_rmse", "max_depth", "learning_rate"], ignore_index=True)
    )
    return summary


def select_best_preset(audit_df: pd.DataFrame, config: NotebookConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    summary = _preset_summary(audit_df)
    if summary.empty:
        raise ValueError("No successful preset-audit rows available")
    best_mean = float(summary["mean_val_wape"].min())
    pool = summary.loc[summary["mean_val_wape"] <= best_mean + config.preset_margin_wape_pp].copy()
    pool = pool.sort_values(
        ["mean_val_wape", "mean_val_rmse", "max_depth", "min_child_weight", "learning_rate"],
        ascending=[True, True, True, False, True],
        ignore_index=True,
    )
    selected = pool.iloc[0].to_dict()
    preset = next(p for p in PRESET_CANDIDATES if p["preset_id"] == selected["preset_id"])
    return preset, summary


def run_preset_audit(
    config: NotebookConfig,
    *,
    resume: bool = True,
    save_every: int = 1,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    require_xgboost()

    meta = load_feature_metadata()
    available_buildings = set(selected_buildings(config, meta))
    audit_buildings = [b for b in config.audit_buildings if b in available_buildings]
    if not audit_buildings:
        raise ValueError("No audit buildings available for the current notebook config")
    frames_a, frames_b = _load_frames_for_buildings(audit_buildings, config.horizons, meta)
    global_static_cols = _global_static_cols(frames_b)
    global_frame = pd.concat([frames_b[b] for b in audit_buildings], ignore_index=True).sort_values(["datetime", "building"]).reset_index(drop=True)

    audit_path = artifact_paths(config)["audit"]
    key_cols = ["preset_id", "scope", "building_or_global", "mode", "horizon_h"]
    sort_cols = ["preset_id", "scope", "building_or_global", "mode", "horizon_h"]
    rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[Any, ...]] = set()
    if resume and audit_path.exists():
        existing_df = _load_existing_frame(audit_path, key_cols=key_cols)
        rows, completed_keys = _frame_to_rows_and_keys(existing_df, key_cols)

    total_cases = len(PRESET_CANDIDATES) * len(config.audit_modes) * len(config.audit_horizons) * (len(audit_buildings) + 1)
    completed = len(completed_keys)
    if verbose:
        print(f"Preset audit scope: {len(audit_buildings)} buildings x {len(config.audit_modes)} modes x {len(config.audit_horizons)} horizons x {len(PRESET_CANDIDATES)} presets")
        if completed > 0:
            print(f"Resuming preset audit: {completed}/{total_cases} rows already available at {audit_path}")

    for preset in PRESET_CANDIDATES:
        for mode in config.audit_modes:
            for horizon_h in config.audit_horizons:
                target_features = mode_feature_cols(mode, int(horizon_h), config)
                global_features = target_features + global_static_cols
                target_name = f"target_cum_h{int(horizon_h)}"
                for building in audit_buildings:
                    key = (preset["preset_id"], "per_building", building, mode, int(horizon_h))
                    if key in completed_keys:
                        continue
                    row = _run_single_case(
                        frame=frames_a[building],
                        scope="per_building",
                        building_or_global=building,
                        mode=mode,
                        horizon_h=int(horizon_h),
                        feature_cols=target_features,
                        preset=preset,
                        config=config,
                        target_name=target_name,
                    )
                    row["max_depth"] = preset["max_depth"]
                    row["learning_rate"] = preset["learning_rate"]
                    row["min_child_weight"] = preset["min_child_weight"]
                    rows.append(row)
                    completed_keys.add(key)
                    completed += 1
                    if verbose:
                        print(
                            f"[{completed:>3}/{total_cases}] audit | {preset['preset_id']} | per_building | {building} | "
                            f"{mode} | h={int(horizon_h):>2} | status={row['status']} | val_wape={row.get('val_wape', np.nan):.3f}"
                        )
                    if save_every > 0 and (completed % save_every == 0 or completed == total_cases):
                        _persist_rows(audit_path, rows, sort_cols)

                key = (preset["preset_id"], "global", "GLOBAL_AUDIT", mode, int(horizon_h))
                if key in completed_keys:
                    continue
                row = _run_single_case(
                    frame=global_frame,
                    scope="global",
                    building_or_global="GLOBAL_AUDIT",
                    mode=mode,
                    horizon_h=int(horizon_h),
                    feature_cols=global_features,
                    preset=preset,
                    config=config,
                    target_name=target_name,
                )
                row["max_depth"] = preset["max_depth"]
                row["learning_rate"] = preset["learning_rate"]
                row["min_child_weight"] = preset["min_child_weight"]
                rows.append(row)
                completed_keys.add(key)
                completed += 1
                if verbose:
                    print(
                        f"[{completed:>3}/{total_cases}] audit | {preset['preset_id']} | global | GLOBAL_AUDIT | "
                        f"{mode} | h={int(horizon_h):>2} | status={row['status']} | val_wape={row.get('val_wape', np.nan):.3f}"
                    )
                if save_every > 0 and (completed % save_every == 0 or completed == total_cases):
                    _persist_rows(audit_path, rows, sort_cols)

    audit_df = _persist_rows(audit_path, rows, sort_cols)
    selected_preset, preset_summary = select_best_preset(audit_df, config)
    return audit_df, selected_preset, preset_summary


def _run_matrix_for_scope(
    config: NotebookConfig,
    scope: str,
    frames: dict[str, pd.DataFrame],
    feature_builder,
    preset: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode in config.modes:
        for horizon_h in config.horizons:
            target_name = f"target_cum_h{int(horizon_h)}"
            if scope == "per_building":
                for building, frame in frames.items():
                    row = _run_single_case(
                        frame=frame,
                        scope="per_building",
                        building_or_global=building,
                        mode=mode,
                        horizon_h=int(horizon_h),
                        feature_cols=feature_builder(mode, building),
                        preset=preset,
                        config=config,
                        target_name=target_name,
                    )
                    rows.append(row)
            else:
                frame = frames["GLOBAL"]
                row = _run_single_case(
                    frame=frame,
                    scope="global",
                    building_or_global="GLOBAL",
                    mode=mode,
                    horizon_h=int(horizon_h),
                    feature_cols=feature_builder(mode, "GLOBAL"),
                    preset=preset,
                    config=config,
                    target_name=target_name,
                )
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["scope", "building_or_global", "mode", "horizon_h"], ignore_index=True)


def run_full_matrix(config: NotebookConfig, preset: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    require_xgboost()

    paths = artifact_paths(config)
    if preset is None:
        audit_df = pd.read_csv(paths["audit"])
        preset, _ = select_best_preset(audit_df, config)

    meta = load_feature_metadata()
    buildings = selected_buildings(config, meta)
    frames_a, frames_b = _load_frames_for_buildings(buildings, config.horizons, meta)
    global_static_cols = _global_static_cols(frames_b)
    global_frame = pd.concat([frames_b[b] for b in buildings], ignore_index=True).sort_values(["datetime", "building"]).reset_index(drop=True)

    sort_cols = ["scope", "building_or_global", "mode", "horizon_h"]
    key_cols = sort_cols.copy()
    per_path = paths["per_building"]
    global_path = paths["global"]
    per_rows, per_completed = _frame_to_rows_and_keys(_load_existing_frame(per_path, key_cols), key_cols)
    global_rows, global_completed = _frame_to_rows_and_keys(_load_existing_frame(global_path, key_cols), key_cols)

    total_per = len(buildings) * len(config.modes) * len(config.horizons)
    total_global = len(config.modes) * len(config.horizons)
    per_done = len(per_completed)
    global_done = len(global_completed)

    print(f"Full matrix scope: {len(buildings)} buildings x {len(config.modes)} modes x {len(config.horizons)} horizons")
    if per_done > 0:
        print(f"Resuming per-building matrix: {per_done}/{total_per} rows already available at {per_path}")
    if global_done > 0:
        print(f"Resuming global matrix: {global_done}/{total_global} rows already available at {global_path}")

    for mode in config.modes:
        for horizon_h in config.horizons:
            feature_cols = mode_feature_cols(mode, int(horizon_h), config)
            target_name = f"target_cum_h{int(horizon_h)}"
            for building, frame in frames_a.items():
                key = ("per_building", building, mode, int(horizon_h))
                if key in per_completed:
                    continue
                row = _run_single_case(
                    frame=frame,
                    scope="per_building",
                    building_or_global=building,
                    mode=mode,
                    horizon_h=int(horizon_h),
                    feature_cols=feature_cols,
                    preset=preset,
                    config=config,
                    target_name=target_name,
                )
                per_rows.append(row)
                per_completed.add(key)
                per_done += 1
                print(
                    f"[{per_done:>3}/{total_per}] per_building | {building} | {mode} | h={int(horizon_h):>2} | "
                    f"status={row['status']} | wape={row.get('wape_pct', np.nan):.3f}"
                )
                per_df = _persist_rows(per_path, per_rows, sort_cols)
                if per_done > 0:
                    _write_partial_summary_artifacts(config)

    for mode in config.modes:
        for horizon_h in config.horizons:
            feature_cols = mode_feature_cols(mode, int(horizon_h), config) + global_static_cols
            key = ("global", "GLOBAL", mode, int(horizon_h))
            if key in global_completed:
                continue
            target_name = f"target_cum_h{int(horizon_h)}"
            row = _run_single_case(
                frame=global_frame,
                scope="global",
                building_or_global="GLOBAL",
                mode=mode,
                horizon_h=int(horizon_h),
                feature_cols=feature_cols,
                preset=preset,
                config=config,
                target_name=target_name,
            )
            global_rows.append(row)
            global_completed.add(key)
            global_done += 1
            print(
                f"[{global_done:>3}/{total_global}] global | GLOBAL | {mode} | h={int(horizon_h):>2} | "
                f"status={row['status']} | wape={row.get('wape_pct', np.nan):.3f}"
            )
            global_df = _persist_rows(global_path, global_rows, sort_cols)
            _write_partial_summary_artifacts(config)

    per_df = _persist_rows(per_path, per_rows, sort_cols)
    global_df = _persist_rows(global_path, global_rows, sort_cols)
    _write_partial_summary_artifacts(config)
    return per_df, global_df, preset


def build_summary_artifacts(
    config: NotebookConfig,
    per_df: pd.DataFrame | None = None,
    global_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = artifact_paths(config)
    if per_df is None:
        per_df = _load_existing_frame(paths["per_building"])
    if global_df is None:
        global_df = _load_existing_frame(paths["global"])

    ok_per = per_df.loc[per_df["status"] == "ok"].copy() if not per_df.empty and "status" in per_df.columns else pd.DataFrame()
    ok_global = global_df.loc[global_df["status"] == "ok"].copy() if not global_df.empty and "status" in global_df.columns else pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    if not ok_per.empty:
        per_summary = (
            ok_per.groupby(["mode", "horizon_h", "target_name", "preset_id"], as_index=False)
            .agg(
                n_cases=("building_or_global", "size"),
                mean_wape_pct=("wape_pct", "mean"),
                mean_rmse=("rmse", "mean"),
                mean_mae=("mae", "mean"),
                mean_r2=("r2", "mean"),
                mean_baseline_wape_pct=("baseline_wape_pct", "mean"),
                mean_delta_wape_vs_baseline=("delta_wape_vs_baseline", "mean"),
            )
        )
        for _, row in per_summary.iterrows():
            summary_rows.append(
                {
                    "scope": "portfolio_mean",
                    "building_or_global": "PORTFOLIO",
                    **row.to_dict(),
                }
            )
    if not ok_global.empty:
        for _, row in ok_global.iterrows():
            summary_rows.append(
                {
                    "scope": "global",
                    "building_or_global": row["building_or_global"],
                    "mode": row["mode"],
                    "horizon_h": int(row["horizon_h"]),
                    "target_name": row["target_name"],
                    "preset_id": row["preset_id"],
                    "n_cases": 1,
                    "mean_wape_pct": row["wape_pct"],
                    "mean_rmse": row["rmse"],
                    "mean_mae": row["mae"],
                    "mean_r2": row["r2"],
                    "mean_baseline_wape_pct": row["baseline_wape_pct"],
                    "mean_delta_wape_vs_baseline": row["delta_wape_vs_baseline"],
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["scope", "horizon_h", "mode"], ignore_index=True)
    summary_df.to_csv(paths["summary"], index=False)

    matched_df = build_matched_lstm_summary(config, per_df=per_df, global_df=global_df)
    matched_df.to_csv(paths["matched"], index=False)
    return summary_df, matched_df


def _load_lstm_match_rows(config: NotebookConfig) -> list[dict[str, Any]]:
    if config.use_future_weather:
        return []
    rows: list[dict[str, Any]] = []

    per_building_path = config.results_dir / "long_horizon_inertia_run_log.csv"
    if per_building_path.exists():
        df = pd.read_csv(per_building_path)
        if {"status", "target_family", "building", "mode", "horizon_hours", "wape_pct", "rmse"}.issubset(df.columns):
            ok = df.loc[(df["status"] == "ok") & (df["target_family"] == "cumulative")].copy()
            for _, row in ok.iterrows():
                rows.append(
                    {
                        "scope": "per_building",
                        "building": str(row["building"]),
                        "mode": str(row["mode"]),
                        "horizon_h": int(row["horizon_hours"]),
                        "lstm_wape": float(row["wape_pct"]),
                        "lstm_rmse": float(row["rmse"]),
                    }
                )

    portfolio_path = config.results_dir / "cumulative_m0_overnight_architecture_summary_by_horizon_20032026.csv"
    if portfolio_path.exists():
        df = pd.read_csv(portfolio_path)
        required = {"target_name", "horizon_hours", "architecture_id", "wape_mean", "rmse_mean"}
        if required.issubset(df.columns):
            use = df.loc[df["architecture_id"] == "A6"].copy() if "A6" in set(df["architecture_id"]) else df.copy()
            use = use.sort_values(["horizon_hours", "wape_mean"]).drop_duplicates(["horizon_hours"], keep="first")
            for _, row in use.iterrows():
                rows.append(
                    {
                        "scope": "portfolio_mean",
                        "building": "PORTFOLIO",
                        "mode": "M0",
                        "horizon_h": int(row["horizon_hours"]),
                        "lstm_wape": float(row["wape_mean"]),
                        "lstm_rmse": float(row["rmse_mean"]),
                    }
                )

    return rows


def build_matched_lstm_summary(
    config: NotebookConfig,
    per_df: pd.DataFrame | None = None,
    global_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    paths = artifact_paths(config)
    if per_df is None:
        per_df = _load_existing_frame(paths["per_building"])
    if global_df is None:
        global_df = _load_existing_frame(paths["global"])

    xgb_rows: list[dict[str, Any]] = []
    ok_per = per_df.loc[per_df["status"] == "ok"].copy() if not per_df.empty and "status" in per_df.columns else pd.DataFrame()
    for _, row in ok_per.iterrows():
        xgb_rows.append(
            {
                "scope": "per_building",
                "building": row["building_or_global"],
                "mode": row["mode"],
                "horizon_h": int(row["horizon_h"]),
                "xgb_wape": float(row["wape_pct"]),
                "xgb_rmse": float(row["rmse"]),
            }
        )

    if not ok_per.empty:
        portfolio = (
            ok_per.groupby(["mode", "horizon_h"], as_index=False)[["wape_pct", "rmse"]]
            .mean()
            .rename(columns={"wape_pct": "xgb_wape", "rmse": "xgb_rmse"})
        )
        for _, row in portfolio.iterrows():
            xgb_rows.append(
                {
                    "scope": "portfolio_mean",
                    "building": "PORTFOLIO",
                    "mode": row["mode"],
                    "horizon_h": int(row["horizon_h"]),
                    "xgb_wape": float(row["xgb_wape"]),
                    "xgb_rmse": float(row["xgb_rmse"]),
                }
            )

    ok_global = global_df.loc[global_df["status"] == "ok"].copy() if not global_df.empty and "status" in global_df.columns else pd.DataFrame()
    for _, row in ok_global.iterrows():
        xgb_rows.append(
            {
                "scope": "global",
                "building": "GLOBAL",
                "mode": row["mode"],
                "horizon_h": int(row["horizon_h"]),
                "xgb_wape": float(row["wape_pct"]),
                "xgb_rmse": float(row["rmse"]),
            }
        )

    if not xgb_rows:
        return pd.DataFrame(
            columns=[
                "scope",
                "building",
                "mode",
                "horizon_h",
                "xgb_wape",
                "xgb_rmse",
                "lstm_wape",
                "lstm_rmse",
                "delta_wape_xgb_minus_lstm",
            ]
        )

    xgb_df = pd.DataFrame(xgb_rows).drop_duplicates(["scope", "building", "mode", "horizon_h"])
    lstm_df = pd.DataFrame(_load_lstm_match_rows(config)).drop_duplicates(["scope", "building", "mode", "horizon_h"])
    if lstm_df.empty:
        merged = xgb_df.copy()
        merged["lstm_wape"] = np.nan
        merged["lstm_rmse"] = np.nan
    else:
        merged = xgb_df.merge(lstm_df, on=["scope", "building", "mode", "horizon_h"], how="left")
    merged["delta_wape_xgb_minus_lstm"] = merged["xgb_wape"] - merged["lstm_wape"]
    return merged.sort_values(["scope", "building", "horizon_h", "mode"], ignore_index=True)


def notebook_runtime_note(config: NotebookConfig) -> str:
    buildings = selected_buildings(config)
    audit_buildings = [building for building in config.audit_buildings if building in set(buildings)]
    audit_runs = len(PRESET_CANDIDATES) * len(config.audit_modes) * len(config.audit_horizons) * (len(audit_buildings) + 1)
    full_matrix_runs = len(buildings) * len(config.modes) * len(config.horizons) + len(config.modes) * len(config.horizons)
    return (
        f"Selected buildings: {', '.join(buildings)}\n"
        f"Audit buildings: {', '.join(audit_buildings)}\n"
        f"Horizons: {list(config.horizons)}\n"
        f"Audit horizons: {list(config.audit_horizons)}\n"
        f"Modes: {list(config.modes)}\n"
        f"Oracle future weather: {config.use_future_weather}\n"
        f"Planned audit rows: {audit_runs}\n"
        f"Planned full-matrix rows: {full_matrix_runs}\n"
        f"xgboost available: {xgboost_available()}\n"
        f"Results dir: {config.results_dir}"
    )
