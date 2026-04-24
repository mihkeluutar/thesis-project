from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
from time import perf_counter
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import model_family_comparison as mfc


HEAT_HISTORY_FEATURES = [
    "feat_heat_obs",
    "feat_heat_lag1",
    "feat_heat_lag2",
    "feat_heat_lag3",
    "feat_heat_lag24",
    "feat_heat_lag168",
    "feat_heat_roll6h",
    "feat_heat_roll24h",
]

WEATHER_HISTORY_FEATURES = [
    "feat_temp_roll6h",
    "feat_temp_roll12h",
    "feat_temp_roll24h",
    "feat_temp_diff1h",
    "feat_temp_diff3h",
]

DEFAULT_CONTEXT_FALLBACKS = [
    "feat_outdoor_temp_c",
    "feat_heat_obs",
    "feat_space_deltaT_c",
    "feat_vent_deltaT_c",
    "feat_rh_pct",
    "feat_wind_ms",
]

THESIS_MODE_GRAMMAR = ("M0", "M1", "M2", "M3", "M4")
THESIS_WEATHER_MODE_ORDER = ("FW0", "FW2", "FW1")
THESIS_BROAD_FEATURE_GROUP_ORDER = (
    "recent_demand_history",
    "historical_weather_memory",
    "future_weather",
    "current_weather",
    "system_dynamic_proxies",
    "calendar",
    "static_building_context",
)
THESIS_FEATURE_GROUP_ORDER = THESIS_BROAD_FEATURE_GROUP_ORDER
THESIS_FINE_FEATURE_GROUP_ORDER = (
    "demand_signal",
    "current_weather",
    "weather_memory",
    "future_weather_summaries",
    "future_weather_paths",
    "space_heating_dynamics",
    "ventilation_dynamics",
    "calendar",
    "static_profile_and_inventory",
    "static_ehr_morphology",
)
THESIS_TAXONOMIES = ("broad", "fine")
PRIMARY_MODE_TRANSITION_COLUMNS = (
    "M1_minus_M0_wape_pct_pp",
    "M2_minus_M0_wape_pct_pp",
    "M3_minus_M2_wape_pct_pp",
    "M4_minus_M3_wape_pct_pp",
)
SUPPORTING_MODE_TRANSITION_COLUMNS = (
    "M3_minus_M1_wape_pct_pp",
    "M4_minus_M0_wape_pct_pp",
)
BROAD_FEATURE_GROUP_LABELS = {
    "recent_demand_history": "Recent demand history",
    "historical_weather_memory": "Historical weather memory",
    "future_weather": "Future weather",
    "current_weather": "Current weather",
    "system_dynamic_proxies": "System dynamics",
    "calendar": "Calendar",
    "static_building_context": "Static building context",
}
BROAD_FEATURE_GROUP_PALETTE = {
    "recent_demand_history": "#2a9d8f",
    "historical_weather_memory": "#e76f51",
    "future_weather": "#457b9d",
    "current_weather": "#264653",
    "system_dynamic_proxies": "#f4a261",
    "calendar": "#8ab17d",
    "static_building_context": "#6d597a",
}
FINE_FEATURE_GROUP_LABELS = {
    "demand_signal": "Demand signal",
    "current_weather": "Current weather",
    "weather_memory": "Weather memory",
    "future_weather_summaries": "Future weather summaries",
    "future_weather_paths": "Future weather paths",
    "space_heating_dynamics": "Space-heating dynamics",
    "ventilation_dynamics": "Ventilation and DHW dynamics",
    "calendar": "Calendar",
    "static_profile_and_inventory": "Static profile and inventory",
    "static_ehr_morphology": "Static EHR morphology",
}
FINE_FEATURE_GROUP_PALETTE = {
    "demand_signal": "#2a9d8f",
    "current_weather": "#264653",
    "weather_memory": "#e76f51",
    "future_weather_summaries": "#457b9d",
    "future_weather_paths": "#1d4e89",
    "space_heating_dynamics": "#f4a261",
    "ventilation_dynamics": "#f7b267",
    "calendar": "#8ab17d",
    "static_profile_and_inventory": "#7b5ea7",
    "static_ehr_morphology": "#5f0f40",
}
FEATURE_GROUP_LABELS = BROAD_FEATURE_GROUP_LABELS
FEATURE_GROUP_PALETTE = BROAD_FEATURE_GROUP_PALETTE
FINE_TO_BROAD_FEATURE_GROUP = {
    "demand_signal": "recent_demand_history",
    "current_weather": "current_weather",
    "weather_memory": "historical_weather_memory",
    "future_weather_summaries": "future_weather",
    "future_weather_paths": "future_weather",
    "space_heating_dynamics": "system_dynamic_proxies",
    "ventilation_dynamics": "system_dynamic_proxies",
    "calendar": "calendar",
    "static_profile_and_inventory": "static_building_context",
    "static_ehr_morphology": "static_building_context",
}


@dataclass
class FittedXGBExperiment:
    building: str
    mode: str
    weather_mode: str
    horizon_h: int
    target_kind: str
    target_name: str
    feature_cols: list[str]
    raw_frame: pd.DataFrame
    train_df: pd.DataFrame
    fit_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    predictions_df: pd.DataFrame
    metrics: dict[str, float]
    model: Any
    model_summary: dict[str, Any]
    regime: str = "per_building"
    seed: int = 42
    training_scope: str = ""
    model_family: str = "xgboost"


@dataclass
class FittedLSTMExperiment:
    building: str
    mode: str
    weather_mode: str
    horizon_h: int
    target_kind: str
    target_name: str
    feature_cols: list[str]
    dynamic_feature_cols: list[str]
    static_feature_cols: list[str]
    raw_frame: pd.DataFrame
    frame_scaled: pd.DataFrame
    fit_meta: pd.DataFrame
    validation_meta: pd.DataFrame
    test_meta: pd.DataFrame
    predictions_df: pd.DataFrame
    metrics: dict[str, float]
    model: Any
    model_summary: dict[str, Any]
    X_fit_dynamic: np.ndarray
    X_validation_dynamic: np.ndarray
    X_test_dynamic: np.ndarray
    X_fit_static: np.ndarray | None
    X_validation_static: np.ndarray | None
    X_test_static: np.ndarray | None
    y_fit_scaled: np.ndarray
    y_validation_scaled: np.ndarray
    y_test_scaled: np.ndarray
    target_scaler: Any
    regime: str = "per_building"
    seed: int = 42
    training_scope: str = ""
    model_family: str = "lstm"


def build_xai_artifact_paths(results_dir: Path) -> dict[str, Path]:
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return {
        "results": results_dir,
        "plots": plot_dir,
        "run_log": results_dir / "xai_run_log.csv",
        "seed_metrics": results_dir / "xai_seed_metrics.csv",
        "metrics": results_dir / "xai_metrics.csv",
        "predictions": results_dir / "xai_predictions.csv",
        "seed_grouped_pfi": results_dir / "xai_seed_grouped_pfi.csv",
        "grouped_pfi": results_dir / "xai_grouped_pfi.csv",
        "seed_grouped_pfi_fine": results_dir / "xai_seed_grouped_pfi_fine.csv",
        "grouped_pfi_fine": results_dir / "xai_grouped_pfi_fine.csv",
        "seed_grouped_shap": results_dir / "xai_seed_grouped_shap.csv",
        "grouped_shap": results_dir / "xai_grouped_shap.csv",
        "seed_grouped_shap_fine": results_dir / "xai_seed_grouped_shap_fine.csv",
        "grouped_shap_fine": results_dir / "xai_grouped_shap_fine.csv",
        "group_share_summary": results_dir / "xai_group_share_summary.csv",
        "group_share_summary_fine": results_dir / "xai_group_share_summary_fine.csv",
        "seed_agreement": results_dir / "xai_seed_pfi_shap_agreement.csv",
        "agreement": results_dir / "xai_pfi_shap_agreement.csv",
        "seed_agreement_fine": results_dir / "xai_seed_pfi_shap_agreement_fine.csv",
        "agreement_fine": results_dir / "xai_pfi_shap_agreement_fine.csv",
        "stability_summary": results_dir / "xai_stability_summary.csv",
        "stability_by_group": results_dir / "xai_stability_by_group.csv",
        "stability_summary_fine": results_dir / "xai_stability_summary_fine.csv",
        "stability_by_group_fine": results_dir / "xai_stability_by_group_fine.csv",
        "seed_mode_deltas": results_dir / "xai_seed_mode_delta_summary.csv",
        "mode_deltas": results_dir / "xai_mode_delta_summary.csv",
        "plot_manifest": results_dir / "xai_plot_manifest.csv",
        "cases": results_dir / "xai_selected_cases.csv",
        "local_figures": results_dir / "xai_local_figure_manifest.csv",
        "scenarios": results_dir / "xai_scenarios.csv",
        "scenario_summary": results_dir / "xai_scenario_directional_summary.csv",
        "rq1_summary": results_dir / "xai_rq1_accuracy_summary.csv",
        "rq2_summary": results_dir / "xai_rq2_xai_stability_summary.csv",
        "rq2_summary_fine": results_dir / "xai_rq2_xai_stability_summary_fine.csv",
        "rq3_summary": results_dir / "xai_rq3_operational_scenarios_summary.csv",
        "matrix_manifest": results_dir / "xai_matrix_manifest.csv",
        "cross_family_metrics": results_dir / "xai_cross_family_metrics.csv",
        "cross_family_grouped_pfi": results_dir / "xai_cross_family_grouped_pfi.csv",
        "cross_family_grouped_shap": results_dir / "xai_cross_family_grouped_shap.csv",
        "cross_family_agreement": results_dir / "xai_cross_family_pfi_shap_agreement.csv",
    }


def feature_group_order_for_taxonomy(taxonomy: str = "broad") -> tuple[str, ...]:
    taxonomy = str(taxonomy).strip().lower()
    if taxonomy == "broad":
        return THESIS_BROAD_FEATURE_GROUP_ORDER
    if taxonomy == "fine":
        return THESIS_FINE_FEATURE_GROUP_ORDER
    raise ValueError(f"Unknown feature-group taxonomy: {taxonomy}")


def feature_group_labels_for_taxonomy(taxonomy: str = "broad") -> dict[str, str]:
    taxonomy = str(taxonomy).strip().lower()
    if taxonomy == "broad":
        return BROAD_FEATURE_GROUP_LABELS
    if taxonomy == "fine":
        return FINE_FEATURE_GROUP_LABELS
    raise ValueError(f"Unknown feature-group taxonomy: {taxonomy}")


def feature_group_palette_for_taxonomy(taxonomy: str = "broad") -> dict[str, str]:
    taxonomy = str(taxonomy).strip().lower()
    if taxonomy == "broad":
        return BROAD_FEATURE_GROUP_PALETTE
    if taxonomy == "fine":
        return FINE_FEATURE_GROUP_PALETTE
    raise ValueError(f"Unknown feature-group taxonomy: {taxonomy}")


def _fine_feature_group_for_feature(feature_name: str) -> str | None:
    feature_name = str(feature_name)
    if feature_name.startswith("feat_heat_"):
        return "demand_signal"
    if feature_name in {
        "feat_outdoor_temp_c",
        "feat_windchill_c",
        "feat_hdh_15c",
        "feat_solar_irradiance_wm2",
        "feat_sunshine_min",
        "feat_rh_pct",
        "feat_wind_ms",
    }:
        return "current_weather"
    if feature_name in WEATHER_HISTORY_FEATURES:
        return "weather_memory"
    if feature_name.startswith("feat_fw_"):
        if "_tplus" in feature_name:
            return "future_weather_paths"
        if any(token in feature_name for token in ("_mean_", "_min_", "_end_")):
            return "future_weather_summaries"
    if feature_name.startswith("feat_space_"):
        return "space_heating_dynamics"
    if feature_name.startswith("feat_vent_") or feature_name.startswith("feat_dhw_"):
        return "ventilation_dynamics"
    if (
        feature_name.startswith("feat_hour_")
        or feature_name.startswith("feat_dow_")
        or feature_name.startswith("feat_month_")
        or feature_name in {"feat_is_weekend", "feat_is_night", "feat_is_daytime"}
    ):
        return "calendar"
    if feature_name.startswith("ehr_missing_") or feature_name.startswith("ehr_"):
        return "static_ehr_morphology"
    if feature_name.startswith("stat_"):
        return "static_profile_and_inventory"
    return None


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


def mode_temporal_features(mode: str) -> list[str]:
    return list(mfc.MODE_TEMPORAL_FEATURES[str(mode)])


def mode_static_features(mode: str) -> list[str]:
    return list(mfc.LSTM_STATIC_FEATURES_SETB) if str(mode) == "M4" else []


def feature_set_for_mode(mode: str) -> str:
    return "setB" if str(mode) == "M4" else "setA"


def add_point_targets(df: pd.DataFrame, horizons: tuple[int, ...], source_col: str = "heat_kwh") -> pd.DataFrame:
    out = df.copy()
    source = pd.to_numeric(out[source_col], errors="coerce")
    target_cols = {
        f"target_point_h{int(h)}": source.shift(-int(h))
        for h in horizons
    }
    return pd.concat([out, pd.DataFrame(target_cols, index=out.index)], axis=1)


def augment_base_frames_with_point_targets(
    base_frames: dict[str, dict[str, pd.DataFrame]],
    horizons: tuple[int, ...],
) -> dict[str, dict[str, pd.DataFrame]]:
    augmented: dict[str, dict[str, pd.DataFrame]] = {}
    for building, frame_map in base_frames.items():
        augmented[building] = {}
        for set_name, frame in frame_map.items():
            augmented[building][set_name] = add_point_targets(frame, horizons)
    return augmented


def target_name_for(target_kind: str, horizon_h: int) -> str:
    target_kind = str(target_kind).strip().lower()
    if target_kind == "cum":
        return f"target_cum_h{int(horizon_h)}"
    if target_kind == "point":
        return f"target_point_h{int(horizon_h)}"
    raise ValueError(f"Unknown target kind: {target_kind}")


def ensure_target_column(df: pd.DataFrame, target_kind: str, horizon_h: int) -> pd.DataFrame:
    target_name = target_name_for(target_kind, int(horizon_h))
    if target_name in df.columns:
        return df
    if str(target_kind).strip().lower() == "cum":
        return mfc.add_cumulative_targets(df, [int(horizon_h)])
    if str(target_kind).strip().lower() == "point":
        return add_point_targets(df, (int(horizon_h),))
    raise ValueError(f"Unknown target kind: {target_kind}")


def normalize_horizons_for_target_kind(
    horizons: tuple[int, ...] | list[int],
    target_kind: str,
) -> tuple[int, ...]:
    normalized = tuple(sorted({int(h) for h in horizons}))
    target_kind = str(target_kind).strip().lower()
    if target_kind == "cum":
        return tuple(h for h in normalized if h > 1)
    if target_kind == "point":
        return normalized
    raise ValueError(f"Unknown target kind: {target_kind}")


def make_feature_groups(
    feature_cols: list[str],
    *,
    taxonomy: str = "broad",
    strict: bool = True,
) -> dict[str, list[str]]:
    feature_cols = [str(col) for col in feature_cols]
    fine_groups = {name: [] for name in THESIS_FINE_FEATURE_GROUP_ORDER}
    unmapped: list[str] = []
    for feature_name in feature_cols:
        fine_group = _fine_feature_group_for_feature(feature_name)
        if fine_group is None:
            unmapped.append(feature_name)
            continue
        fine_groups[fine_group].append(feature_name)
    if strict and unmapped:
        raise ValueError(
            "Unmapped feature columns in XAI taxonomy: " + ", ".join(sorted(unmapped))
        )

    taxonomy = str(taxonomy).strip().lower()
    if taxonomy == "fine":
        return {name: cols for name, cols in fine_groups.items() if cols}
    if taxonomy == "broad":
        broad_groups = {name: [] for name in THESIS_BROAD_FEATURE_GROUP_ORDER}
        for fine_group, cols in fine_groups.items():
            if not cols:
                continue
            broad_groups[FINE_TO_BROAD_FEATURE_GROUP[fine_group]].extend(cols)
        return {name: cols for name, cols in broad_groups.items() if cols}
    raise ValueError(f"Unknown feature-group taxonomy: {taxonomy}")


def feature_group_table(
    feature_groups: dict[str, list[str]],
    *,
    taxonomy: str = "broad",
    include_rollup: bool = False,
) -> pd.DataFrame:
    rows = []
    for group_name, cols in feature_groups.items():
        row = {
            "feature_group": group_name,
            "feature_group_label": feature_group_labels_for_taxonomy(taxonomy).get(group_name, str(group_name)),
            "n_features": int(len(cols)),
            "features": ", ".join(cols),
        }
        if include_rollup and str(taxonomy).strip().lower() == "fine":
            row["broad_rollup"] = FINE_TO_BROAD_FEATURE_GROUP.get(group_name, "")
            row["broad_rollup_label"] = BROAD_FEATURE_GROUP_LABELS.get(row["broad_rollup"], row["broad_rollup"])
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    order_map = {name: idx for idx, name in enumerate(feature_group_order_for_taxonomy(taxonomy))}
    return (
        table.assign(_feature_sort=lambda df: df["feature_group"].map(order_map).fillna(len(order_map)))
        .sort_values(["_feature_sort", "feature_group"])
        .drop(columns="_feature_sort")
        .reset_index(drop=True)
    )


def build_broad_taxonomy_definition_table() -> pd.DataFrame:
    rows = [
        {
            "feature_group": "recent_demand_history",
            "headline_role": "Persistence anchor",
            "description": "Observed demand and lagged demand structure that anchors short- and medium-horizon persistence.",
        },
        {
            "feature_group": "historical_weather_memory",
            "headline_role": "Past weather inertia",
            "description": "Lagged or rolling weather terms that reflect slower thermal-memory effects rather than immediate forcing.",
        },
        {
            "feature_group": "future_weather",
            "headline_role": "Look-ahead weather information",
            "description": "Future temperature and humidity information introduced through the `FW1` and `FW2` regimes.",
        },
        {
            "feature_group": "current_weather",
            "headline_role": "Immediate forcing",
            "description": "Current outdoor weather conditions at issue time.",
        },
        {
            "feature_group": "system_dynamic_proxies",
            "headline_role": "Operational dynamics",
            "description": "Subsystem state and delta-T proxies for heating, ventilation, and DHW behavior.",
        },
        {
            "feature_group": "calendar",
            "headline_role": "Schedule structure",
            "description": "Hour, weekday, month, and indicator features that encode timing and occupancy rhythm.",
        },
        {
            "feature_group": "static_building_context",
            "headline_role": "Cross-building context",
            "description": "Static building profile, system inventory, and EHR-derived morphology descriptors.",
        },
    ]
    order_map = {name: idx for idx, name in enumerate(THESIS_BROAD_FEATURE_GROUP_ORDER)}
    return (
        pd.DataFrame(rows)
        .assign(_feature_sort=lambda df: df["feature_group"].map(order_map).fillna(len(order_map)))
        .sort_values("_feature_sort")
        .drop(columns="_feature_sort")
        .reset_index(drop=True)
    )


def build_fine_taxonomy_definition_table() -> pd.DataFrame:
    rows = [
        {
            "feature_group": "demand_signal",
            "broad_rollup": "recent_demand_history",
            "description": "Observed demand and its lagged or rolled temporal structure.",
        },
        {
            "feature_group": "current_weather",
            "broad_rollup": "current_weather",
            "description": "Current weather forcing at forecast issue time.",
        },
        {
            "feature_group": "weather_memory",
            "broad_rollup": "historical_weather_memory",
            "description": "Lagged and rolling historical weather memory terms.",
        },
        {
            "feature_group": "future_weather_summaries",
            "broad_rollup": "future_weather",
            "description": "Summary look-ahead descriptors such as future mean, minimum, and endpoint values.",
        },
        {
            "feature_group": "future_weather_paths",
            "broad_rollup": "future_weather",
            "description": "Hour-by-hour future weather path columns across the cumulative window.",
        },
        {
            "feature_group": "space_heating_dynamics",
            "broad_rollup": "system_dynamic_proxies",
            "description": "Space-heating loop activity and delta-T state proxies.",
        },
        {
            "feature_group": "ventilation_dynamics",
            "broad_rollup": "system_dynamic_proxies",
            "description": "Ventilation and DHW activity or delta-T proxy terms.",
        },
        {
            "feature_group": "calendar",
            "broad_rollup": "calendar",
            "description": "Cyclic calendar and schedule indicator features.",
        },
        {
            "feature_group": "static_profile_and_inventory",
            "broad_rollup": "static_building_context",
            "description": "Static building size, usage, and system inventory descriptors from the regular static profile.",
        },
        {
            "feature_group": "static_ehr_morphology",
            "broad_rollup": "static_building_context",
            "description": "EHR-derived morphology and EHR-specific missingness descriptors.",
        },
    ]
    order_map = {name: idx for idx, name in enumerate(THESIS_FINE_FEATURE_GROUP_ORDER)}
    return (
        pd.DataFrame(rows)
        .assign(
            broad_rollup_label=lambda df: df["broad_rollup"].map(BROAD_FEATURE_GROUP_LABELS),
            _feature_sort=lambda df: df["feature_group"].map(order_map).fillna(len(order_map)),
        )
        .sort_values("_feature_sort")
        .drop(columns="_feature_sort")
        .reset_index(drop=True)
    )


def build_thesis_mode_definition_table() -> pd.DataFrame:
    rows = [
        {
            "mode": "M0",
            "feature_increment": "Lean temporal core",
            "thesis_reading": "Reference mode with recent demand, current weather, and calendar structure.",
        },
        {
            "mode": "M1",
            "feature_increment": "Add historical weather memory",
            "thesis_reading": "Tests whether lagged weather-memory features help beyond the lean temporal core.",
        },
        {
            "mode": "M2",
            "feature_increment": "Add system dynamics",
            "thesis_reading": "Tests whether subsystem delta-T and activity proxies help beyond the lean temporal core.",
        },
        {
            "mode": "M3",
            "feature_increment": "Add historical weather memory on top of M2",
            "thesis_reading": "Separates the incremental value of historical weather memory once system dynamics are already available.",
        },
        {
            "mode": "M4",
            "feature_increment": "Add static building context on top of M3",
            "thesis_reading": "Tests whether static building descriptors add information beyond the richer temporal stack.",
        },
    ]
    return pd.DataFrame(rows)


def build_thesis_weather_role_table() -> pd.DataFrame:
    rows = [
        {
            "weather_mode": "FW0",
            "thesis_role": "Clean inertia identification",
            "interpretation": "No future weather is available, so weather-related importance can be read as current forcing or historical weather memory only.",
        },
        {
            "weather_mode": "FW2",
            "thesis_role": "Realistic forecast-like weather",
            "interpretation": "Uses the exported feat_fw_* proxy future weather and represents the realistic forecast-with-noise setting for the thesis.",
        },
        {
            "weather_mode": "FW1",
            "thesis_role": "Oracle upper bound",
            "interpretation": "Uses actual future weather and should be read as an upper-bound reference rather than an operational setting.",
        },
    ]
    order_map = {mode: idx for idx, mode in enumerate(THESIS_WEATHER_MODE_ORDER)}
    return pd.DataFrame(rows).assign(
        _sort_key=lambda df: df["weather_mode"].map(order_map).fillna(len(order_map))
    ).sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)


def _eval_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if eval_mask is None:
        return y_true.astype(float), y_pred.astype(float)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    if eval_mask.any():
        return y_true[eval_mask].astype(float), y_pred[eval_mask].astype(float)
    return y_true.astype(float), y_pred.astype(float)


def _prepare_xgb_building_case(
    config: mfc.ExperimentConfig,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    target_kind: str,
    prepared_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cache_key = (
        str(building),
        str(mode),
        str(weather_mode),
        int(horizon_h),
        str(target_kind),
    )
    if prepared_cache is not None and cache_key in prepared_cache:
        return prepared_cache[cache_key]

    set_name = feature_set_for_mode(mode)
    frame = ensure_target_column(base_frames[str(building)][set_name], target_kind, int(horizon_h))
    frame, fw_cols = mfc.apply_weather_mode(frame, weather_mode, int(horizon_h))
    dynamic_cols = mode_temporal_features(mode) + list(fw_cols)
    static_cols = mode_static_features(mode)
    feature_cols = dynamic_cols + static_cols
    target_name = target_name_for(target_kind, int(horizon_h))
    split_spec = mfc.build_split_spec(frame, int(horizon_h), config)
    train_df, test_df = mfc.build_xgb_model_frames(frame, feature_cols, target_name, split_spec)
    fit_df, val_df = mfc._split_xgb_train_validation(train_df, config.validation_fraction)

    prepared = {
        "frame": frame,
        "feature_cols": feature_cols,
        "target_name": target_name,
        "train_df": train_df,
        "fit_df": fit_df,
        "val_df": val_df,
        "test_df": test_df,
    }
    if prepared_cache is not None:
        prepared_cache[cache_key] = prepared
    return prepared


def fit_xgb_experiment(
    config: mfc.ExperimentConfig,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    target_kind: str,
    prepared_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> FittedXGBExperiment:
    prepared = _prepare_xgb_building_case(
        config,
        base_frames,
        building=building,
        mode=mode,
        weather_mode=weather_mode,
        horizon_h=int(horizon_h),
        target_kind=target_kind,
        prepared_cache=prepared_cache,
    )
    frame = prepared["frame"]
    feature_cols = list(prepared["feature_cols"])
    target_name = str(prepared["target_name"])
    train_df = prepared["train_df"]
    fit_df = prepared["fit_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]
    if fit_df.empty or test_df.empty:
        raise ValueError(
            f"Insufficient rows for building={building}, mode={mode}, weather={weather_mode}, "
            f"h={int(horizon_h)}, target_kind={target_kind}"
        )

    model = mfc._fit_xgb_model(
        fit_df[feature_cols],
        fit_df[target_name],
        val_df[feature_cols],
        val_df[target_name],
        mfc.xgb_preset(config),
        config,
    )
    y_pred = mfc._predict_with_best_iteration(model, test_df[feature_cols])
    y_true = test_df[target_name].to_numpy(dtype=float)
    eval_mask = test_df["is_heating_eval"].to_numpy(dtype=bool)
    eval_true, eval_pred = _eval_arrays(y_true, y_pred, eval_mask)
    metrics = mfc.compute_regression_metrics(eval_true, eval_pred)

    predictions_df = test_df.loc[:, ["building", "datetime", "is_heating_eval", target_name]].copy()
    predictions_df.rename(columns={target_name: "y_true"}, inplace=True)
    predictions_df["y_pred"] = y_pred
    predictions_df["abs_error"] = np.abs(predictions_df["y_true"] - predictions_df["y_pred"])
    predictions_df["hour"] = pd.to_datetime(predictions_df["datetime"]).dt.hour
    predictions_df["target_kind"] = str(target_kind)
    predictions_df["mode"] = mode
    predictions_df["weather_mode"] = weather_mode
    predictions_df["horizon_h"] = int(horizon_h)
    predictions_df["row_idx"] = np.arange(len(predictions_df), dtype=int)
    for helper_col in [
        "feat_outdoor_temp_c",
        "feat_heat_obs",
        "feat_temp_diff3h",
        "feat_space_deltaT_c",
        "feat_vent_deltaT_c",
    ]:
        if helper_col in test_df.columns:
            predictions_df[helper_col] = test_df[helper_col].to_numpy()

    return FittedXGBExperiment(
        building=building,
        mode=mode,
        weather_mode=weather_mode,
        horizon_h=int(horizon_h),
        target_kind=str(target_kind),
        target_name=target_name,
        feature_cols=feature_cols,
        raw_frame=frame,
        train_df=train_df,
        fit_df=fit_df,
        validation_df=val_df,
        test_df=test_df.reset_index(drop=True),
        predictions_df=predictions_df.reset_index(drop=True),
        metrics=metrics,
        model=model,
        model_summary={
            "n_train_rows": int(len(fit_df)),
            "n_val_rows": int(len(val_df)),
            "n_test_rows": int(len(test_df)),
            "best_iteration": int(
                getattr(model, "best_iteration", mfc.XGB_FIXED_PARAMS["n_estimators"] - 1)
                or mfc.XGB_FIXED_PARAMS["n_estimators"] - 1
            ),
            "feature_cols": feature_cols,
        },
    )


def _inverse_single_target(values_scaled: np.ndarray, target_scaler: Any) -> np.ndarray:
    values_scaled = np.asarray(values_scaled, dtype=float).reshape(-1, 1)
    return target_scaler.inverse_transform(values_scaled).reshape(-1).astype(float)


def _predict_lstm_original_scale(
    experiment: FittedLSTMExperiment,
    X_dynamic: np.ndarray,
    X_static: np.ndarray | None = None,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The structure of `inputs` doesn't match the expected structure\.",
            category=UserWarning,
        )
        if experiment.mode == "M4":
            if X_static is None:
                raise ValueError("M4 prediction requires static inputs.")
            y_pred_scaled = experiment.model.predict([X_dynamic, X_static], verbose=0).reshape(-1)
        else:
            y_pred_scaled = experiment.model.predict(X_dynamic, verbose=0).reshape(-1)
    return _inverse_single_target(y_pred_scaled, experiment.target_scaler)


def fit_lstm_experiment(
    config: mfc.ExperimentConfig,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    target_kind: str,
    static_scaler: Any | None = None,
) -> FittedLSTMExperiment:
    if str(target_kind).strip().lower() != "cum":
        raise NotImplementedError("The current LSTM XAI sweep supports only cumulative targets.")

    mfc.require_tensorflow()
    mfc.set_all_seeds(config.random_seed, deterministic_ops=config.deterministic_ops)
    mfc.tf.keras.backend.clear_session()

    set_name = feature_set_for_mode(mode)
    frame = ensure_target_column(base_frames[building][set_name].copy(), target_kind, int(horizon_h))
    frame, fw_cols = mfc.apply_weather_mode(frame, weather_mode, int(horizon_h))
    target_name = target_name_for(target_kind, int(horizon_h))
    dynamic_cols = mode_temporal_features(mode) + list(fw_cols)
    static_cols = mode_static_features(mode)
    feature_cols = dynamic_cols + static_cols

    split_spec = mfc.build_split_spec(frame, int(horizon_h), config)
    frame_scaled, feature_scaler, target_scaler = mfc.scale_frame_for_lstm(
        frame=frame,
        dynamic_cols=dynamic_cols,
        target_name=target_name,
        feature_train_mask=split_spec.feature_train_mask,
        train_issue_mask=split_spec.train_issue_mask,
    )
    X_train_full, y_train_full, train_meta_full = mfc.build_sequences(
        frame_scaled,
        dynamic_cols,
        f"{target_name}_scaled",
        split_spec.train_issue_mask,
        config.lookback_hours,
    )
    X_test, y_test, test_meta = mfc.build_sequences(
        frame_scaled,
        dynamic_cols,
        f"{target_name}_scaled",
        split_spec.test_issue_mask,
        config.lookback_hours,
    )
    X_fit, y_fit, fit_meta, X_val, y_val, val_meta = mfc.make_internal_fit_split(
        X_train_full,
        y_train_full,
        train_meta_full,
        config.validation_fraction,
    )
    if X_fit.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError(
            f"Insufficient LSTM sequences for building={building}, mode={mode}, "
            f"weather={weather_mode}, h={int(horizon_h)}, target_kind={target_kind}"
        )

    spec = mfc.architecture_spec(config)
    if mode == "M4":
        if static_scaler is None:
            static_scaler = mfc.fit_static_scaler(config, base_frames)
        static_vector = mfc.build_static_vector(frame, static_scaler, static_cols)
        X_fit_static = np.repeat(static_vector, repeats=X_fit.shape[0], axis=0)
        X_val_static = np.repeat(static_vector, repeats=X_val.shape[0], axis=0) if X_val.shape[0] > 0 else np.empty((0, static_vector.shape[1]), dtype="float32")
        X_test_static = np.repeat(static_vector, repeats=X_test.shape[0], axis=0)
        model = mfc.build_lstm_temporal_plus_static_from_spec(
            spec["lookback_hours"],
            X_fit.shape[-1],
            X_fit_static.shape[-1],
            spec,
            config.learning_rate,
        )
    else:
        X_fit_static = None
        X_val_static = None
        X_test_static = None
        model = mfc.build_lstm_temporal_only_from_spec(
            spec["lookback_hours"],
            X_fit.shape[-1],
            spec,
            config.learning_rate,
        )

    lr_callback = mfc.LearningRateHistory()
    callbacks: list[Any] = [lr_callback]
    validation_data = None
    if X_val.shape[0] > 0:
        callbacks += [
            mfc.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                restore_best_weights=True,
            ),
            mfc.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                verbose=0,
            ),
        ]
        validation_data = ([X_val, X_val_static], y_val) if mode == "M4" else (X_val, y_val)

    if mode == "M4":
        history = model.fit(
            [X_fit, X_fit_static],
            y_fit,
            validation_data=validation_data,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0,
            callbacks=callbacks,
        )
    else:
        history = model.fit(
            X_fit,
            y_fit,
            validation_data=validation_data,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0,
            callbacks=callbacks,
        )

    y_true = _inverse_single_target(y_test, target_scaler)
    y_pred = _predict_lstm_original_scale(
        FittedLSTMExperiment(
            building=building,
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=int(horizon_h),
            target_kind=str(target_kind),
            target_name=target_name,
            feature_cols=feature_cols,
            dynamic_feature_cols=dynamic_cols,
            static_feature_cols=static_cols,
            raw_frame=frame,
            frame_scaled=frame_scaled,
            fit_meta=fit_meta,
            validation_meta=val_meta,
            test_meta=test_meta,
            predictions_df=pd.DataFrame(),
            metrics={},
            model=model,
            model_summary={},
            X_fit_dynamic=X_fit,
            X_validation_dynamic=X_val,
            X_test_dynamic=X_test,
            X_fit_static=X_fit_static,
            X_validation_static=X_val_static,
            X_test_static=X_test_static,
            y_fit_scaled=y_fit,
            y_validation_scaled=y_val,
            y_test_scaled=y_test,
            target_scaler=target_scaler,
        ),
        X_test,
        X_test_static,
    )
    eval_mask = test_meta["is_heating_eval"].to_numpy(dtype=bool)
    eval_true, eval_pred = _eval_arrays(y_true, y_pred, eval_mask)
    metrics = mfc.compute_regression_metrics(eval_true, eval_pred)

    predictions_df = test_meta.copy()
    predictions_df["y_true"] = y_true
    predictions_df["y_pred"] = y_pred
    predictions_df["abs_error"] = np.abs(predictions_df["y_true"] - predictions_df["y_pred"])
    predictions_df["target_kind"] = str(target_kind)
    predictions_df["mode"] = mode
    predictions_df["weather_mode"] = weather_mode
    predictions_df["horizon_h"] = int(horizon_h)
    predictions_df["row_idx"] = np.arange(len(predictions_df), dtype=int)

    best_epoch = np.nan
    val_loss_hist = history.history.get("val_loss", [])
    if len(val_loss_hist) > 0:
        best_epoch = int(np.nanargmin(np.asarray(val_loss_hist, dtype=float))) + 1

    return FittedLSTMExperiment(
        building=building,
        mode=mode,
        weather_mode=weather_mode,
        horizon_h=int(horizon_h),
        target_kind=str(target_kind),
        target_name=target_name,
        feature_cols=feature_cols,
        dynamic_feature_cols=dynamic_cols,
        static_feature_cols=static_cols,
        raw_frame=frame,
        frame_scaled=frame_scaled,
        fit_meta=fit_meta.reset_index(drop=True),
        validation_meta=val_meta.reset_index(drop=True),
        test_meta=test_meta.reset_index(drop=True),
        predictions_df=predictions_df.reset_index(drop=True),
        metrics=metrics,
        model=model,
        model_summary={
            "n_train_rows": int(X_fit.shape[0]),
            "n_val_rows": int(X_val.shape[0]),
            "n_test_rows": int(X_test.shape[0]),
            "best_epoch": best_epoch,
            "lookback_hours": int(spec["lookback_hours"]),
            "feature_scaler": feature_scaler,
            "feature_cols": feature_cols,
        },
        X_fit_dynamic=X_fit,
        X_validation_dynamic=X_val,
        X_test_dynamic=X_test,
        X_fit_static=X_fit_static,
        X_validation_static=X_val_static,
        X_test_static=X_test_static,
        y_fit_scaled=np.asarray(y_fit, dtype=float),
        y_validation_scaled=np.asarray(y_val, dtype=float),
        y_test_scaled=np.asarray(y_test, dtype=float),
        target_scaler=target_scaler,
    )


def run_experiment_grid(
    config: mfc.ExperimentConfig,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    buildings: tuple[str, ...],
    modes: tuple[str, ...],
    weather_mode: str,
    horizon_h: int,
    target_kinds: tuple[str, ...],
) -> tuple[dict[tuple[str, str, str], FittedXGBExperiment], pd.DataFrame, pd.DataFrame]:
    fitted: dict[tuple[str, str, str], FittedXGBExperiment] = {}
    metric_rows = []
    prediction_frames = []
    for target_kind in target_kinds:
        for building in buildings:
            for mode in modes:
                experiment = fit_xgb_experiment(
                    config,
                    base_frames,
                    building=building,
                    mode=mode,
                    weather_mode=weather_mode,
                    horizon_h=int(horizon_h),
                    target_kind=target_kind,
                )
                key = (target_kind, building, mode)
                fitted[key] = experiment
                metric_rows.append(
                    {
                        "target_kind": target_kind,
                        "building": building,
                        "mode": mode,
                        "weather_mode": weather_mode,
                        "horizon_h": int(horizon_h),
                        "rmse": experiment.metrics["rmse"],
                        "mae": experiment.metrics["mae"],
                        "r2": experiment.metrics["r2"],
                        "wape_pct": experiment.metrics["wape_pct"],
                        "n_eval_rows": int(experiment.predictions_df["is_heating_eval"].astype(bool).sum()),
                        "n_test_rows": int(len(experiment.predictions_df)),
                        "n_features": int(len(experiment.feature_cols)),
                        "best_iteration": int(experiment.model_summary["best_iteration"]),
                    }
                )
                prediction_frames.append(experiment.predictions_df.copy())
    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["target_kind", "building", "mode"]
    ).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return fitted, metrics_df, predictions_df


def run_cross_family_horizon_grid(
    config: mfc.ExperimentConfig,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    buildings: tuple[str, ...],
    modes: tuple[str, ...],
    weather_mode: str,
    horizons: tuple[int, ...],
    target_kind: str,
    model_families: tuple[str, ...] = ("xgboost", "lstm"),
    pfi_repeats: int = 10,
    horizons_by_family: dict[str, tuple[int, ...]] | None = None,
    pfi_repeats_by_family: dict[str, int] | None = None,
    shap_background_size: int = 128,
    shap_explain_size: int = 256,
    lstm_pfi_max_rows: int | None = None,
    random_seed: int = 42,
) -> tuple[dict[tuple[str, str, str, int], Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fitted: dict[tuple[str, str, str, int], Any] = {}
    metrics_rows: list[dict[str, Any]] = []
    pfi_frames: list[pd.DataFrame] = []
    shap_frames: list[pd.DataFrame] = []
    agreement_frames: list[pd.DataFrame] = []
    horizons = normalize_horizons_for_target_kind(horizons, target_kind)
    if not horizons:
        raise ValueError(
            f"No valid horizons remain for target_kind={target_kind}. "
            "For cumulative targets, the sweep excludes h=1 because point and cumulative are identical there."
        )

    static_scaler = None
    if "lstm" in model_families and "M4" in modes:
        static_scaler = mfc.fit_static_scaler(config, base_frames)

    for model_family in model_families:
        family_horizons = horizons
        if horizons_by_family is not None and model_family in horizons_by_family:
            family_horizons = normalize_horizons_for_target_kind(horizons_by_family[model_family], target_kind)
        if not family_horizons:
            continue
        family_pfi_repeats = int(pfi_repeats_by_family.get(model_family, pfi_repeats)) if pfi_repeats_by_family else int(pfi_repeats)
        for building in buildings:
            for mode in modes:
                for horizon_h in family_horizons:
                    if model_family == "xgboost":
                        experiment = fit_xgb_experiment(
                            config,
                            base_frames,
                            building=building,
                            mode=mode,
                            weather_mode=weather_mode,
                            horizon_h=int(horizon_h),
                            target_kind=target_kind,
                        )
                        pfi_df = grouped_permutation_importance(
                            experiment.model,
                            experiment.test_df[experiment.feature_cols],
                            experiment.test_df[experiment.target_name].to_numpy(dtype=float),
                            make_feature_groups(experiment.feature_cols),
                            eval_mask=experiment.test_df["is_heating_eval"].to_numpy(dtype=bool),
                            n_repeats=family_pfi_repeats,
                            random_seed=random_seed,
                        )
                        shap_payload = compute_tree_shap(experiment)
                        shap_group_df = summarize_grouped_shap(
                            shap_payload["shap_values"],
                            shap_payload["feature_names"],
                            make_feature_groups(experiment.feature_cols),
                        )
                    elif model_family == "lstm":
                        experiment = fit_lstm_experiment(
                            config,
                            base_frames,
                            building=building,
                            mode=mode,
                            weather_mode=weather_mode,
                            horizon_h=int(horizon_h),
                            target_kind=target_kind,
                            static_scaler=static_scaler,
                        )
                        feature_groups = make_feature_groups(experiment.feature_cols)
                        pfi_df = grouped_permutation_importance_lstm(
                            experiment,
                            feature_groups,
                            n_repeats=family_pfi_repeats,
                            random_seed=random_seed,
                            max_rows=lstm_pfi_max_rows,
                        )
                        shap_payload = compute_lstm_gradient_shap(
                            experiment,
                            n_background=shap_background_size,
                            n_explain=shap_explain_size,
                            random_seed=random_seed,
                        )
                        shap_group_df = summarize_grouped_lstm_shap(
                            shap_payload["shap_values"],
                            shap_payload["feature_names"],
                            feature_groups,
                            static_shap_values=shap_payload.get("static_shap_values"),
                            static_feature_names=shap_payload.get("static_feature_names"),
                        )
                    else:
                        raise ValueError(f"Unsupported model family: {model_family}")

                    agreement_df = build_pfi_shap_agreement_table(pfi_df, shap_group_df)
                    key = (model_family, building, mode, int(horizon_h))
                    fitted[key] = {
                        "experiment": experiment,
                        "shap": shap_payload,
                    }
                    metrics_rows.append(
                        {
                            "model_family": model_family,
                            "target_kind": target_kind,
                            "building": building,
                            "mode": mode,
                            "weather_mode": weather_mode,
                            "horizon_h": int(horizon_h),
                            "rmse": experiment.metrics["rmse"],
                            "mae": experiment.metrics["mae"],
                            "r2": experiment.metrics["r2"],
                            "wape_pct": experiment.metrics["wape_pct"],
                            "n_test_rows": int(len(experiment.predictions_df)),
                            "n_features": int(len(experiment.feature_cols)),
                        }
                    )

                    for frame in (pfi_df, shap_group_df, agreement_df):
                        if frame.empty:
                            continue
                        frame["model_family"] = model_family
                        frame["target_kind"] = target_kind
                        frame["building"] = building
                        frame["mode"] = mode
                        frame["weather_mode"] = weather_mode
                        frame["horizon_h"] = int(horizon_h)
                    pfi_frames.append(pfi_df)
                    shap_frames.append(shap_group_df)
                    agreement_frames.append(agreement_df)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["model_family", "building", "horizon_h", "mode"]
    ).reset_index(drop=True)
    pfi_df = pd.concat([df for df in pfi_frames if not df.empty], ignore_index=True) if pfi_frames else pd.DataFrame()
    shap_df = pd.concat([df for df in shap_frames if not df.empty], ignore_index=True) if shap_frames else pd.DataFrame()
    agreement_df = pd.concat([df for df in agreement_frames if not df.empty], ignore_index=True) if agreement_frames else pd.DataFrame()
    return fitted, metrics_df, pfi_df, shap_df, agreement_df


def grouped_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y_true: np.ndarray,
    feature_groups: dict[str, list[str]],
    *,
    eval_mask: np.ndarray | None = None,
    n_repeats: int = 30,
    random_seed: int = 42,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    baseline_pred = mfc._predict_with_best_iteration(model, X)
    base_true_eval, base_pred_eval = _eval_arrays(y_true, baseline_pred, eval_mask)
    baseline_metrics = mfc.compute_regression_metrics(base_true_eval, base_pred_eval)
    rng = np.random.default_rng(int(random_seed))
    rows = []

    for group_name, group_cols in feature_groups.items():
        group_cols = [col for col in group_cols if col in X.columns]
        if not group_cols:
            continue
        delta_rmse = []
        delta_wape = []
        delta_mae = []
        for _ in range(int(n_repeats)):
            X_perm = X.copy()
            perm = rng.permutation(len(X_perm))
            for col in group_cols:
                X_perm[col] = X_perm[col].to_numpy()[perm]
            perm_pred = mfc._predict_with_best_iteration(model, X_perm)
            perm_true_eval, perm_pred_eval = _eval_arrays(y_true, perm_pred, eval_mask)
            perm_metrics = mfc.compute_regression_metrics(perm_true_eval, perm_pred_eval)
            delta_rmse.append(perm_metrics["rmse"] - baseline_metrics["rmse"])
            delta_wape.append(perm_metrics["wape_pct"] - baseline_metrics["wape_pct"])
            delta_mae.append(perm_metrics["mae"] - baseline_metrics["mae"])
        rows.append(
            {
                "feature_group": group_name,
                "n_features": int(len(group_cols)),
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_wape_pct": baseline_metrics["wape_pct"],
                "baseline_mae": baseline_metrics["mae"],
                "delta_rmse_mean": float(np.mean(delta_rmse)),
                "delta_rmse_std": float(np.std(delta_rmse, ddof=0)),
                "delta_wape_mean": float(np.mean(delta_wape)),
                "delta_wape_std": float(np.std(delta_wape, ddof=0)),
                "delta_mae_mean": float(np.mean(delta_mae)),
                "delta_mae_std": float(np.std(delta_mae, ddof=0)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("delta_wape_mean", ascending=False).reset_index(drop=True)


def grouped_permutation_importance_lstm(
    experiment: FittedLSTMExperiment,
    feature_groups: dict[str, list[str]],
    *,
    n_repeats: int = 10,
    random_seed: int = 42,
    max_rows: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_seed))
    row_idx = np.arange(experiment.X_test_dynamic.shape[0], dtype=int)
    if max_rows is not None and len(row_idx) > int(max_rows):
        row_idx = np.sort(rng.choice(row_idx, size=int(max_rows), replace=False))

    X_test_dynamic = np.asarray(experiment.X_test_dynamic[row_idx], dtype="float32")
    X_test_static = None if experiment.X_test_static is None else np.asarray(experiment.X_test_static[row_idx], dtype="float32")
    y_true = experiment.predictions_df["y_true"].to_numpy(dtype=float)[row_idx]
    eval_mask = experiment.predictions_df["is_heating_eval"].to_numpy(dtype=bool)[row_idx]

    baseline_pred = _predict_lstm_original_scale(
        experiment,
        X_test_dynamic,
        X_test_static,
    )
    base_true_eval, base_pred_eval = _eval_arrays(y_true, baseline_pred, eval_mask)
    baseline_metrics = mfc.compute_regression_metrics(base_true_eval, base_pred_eval)
    rows = []

    dynamic_index = {col: idx for idx, col in enumerate(experiment.dynamic_feature_cols)}
    static_index = {col: idx for idx, col in enumerate(experiment.static_feature_cols)}

    for group_name, group_cols in feature_groups.items():
        dyn_cols = [col for col in group_cols if col in dynamic_index]
        sta_cols = [col for col in group_cols if col in static_index]
        if not dyn_cols and not sta_cols:
            continue
        delta_rmse = []
        delta_wape = []
        delta_mae = []
        for _ in range(int(n_repeats)):
            perm = rng.permutation(X_test_dynamic.shape[0])
            X_dyn_perm = np.array(X_test_dynamic, copy=True)
            X_sta_perm = None if X_test_static is None else np.array(X_test_static, copy=True)

            for col in dyn_cols:
                col_idx = dynamic_index[col]
                X_dyn_perm[:, :, col_idx] = X_dyn_perm[perm, :, col_idx]
            if X_sta_perm is not None:
                for col in sta_cols:
                    col_idx = static_index[col]
                    X_sta_perm[:, col_idx] = X_sta_perm[perm, col_idx]

            perm_pred = _predict_lstm_original_scale(experiment, X_dyn_perm, X_sta_perm)
            perm_true_eval, perm_pred_eval = _eval_arrays(y_true, perm_pred, eval_mask)
            perm_metrics = mfc.compute_regression_metrics(perm_true_eval, perm_pred_eval)
            delta_rmse.append(perm_metrics["rmse"] - baseline_metrics["rmse"])
            delta_wape.append(perm_metrics["wape_pct"] - baseline_metrics["wape_pct"])
            delta_mae.append(perm_metrics["mae"] - baseline_metrics["mae"])

        rows.append(
            {
                "feature_group": group_name,
                "n_features": int(len(dyn_cols) + len(sta_cols)),
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_wape_pct": baseline_metrics["wape_pct"],
                "baseline_mae": baseline_metrics["mae"],
                "delta_rmse_mean": float(np.mean(delta_rmse)),
                "delta_rmse_std": float(np.std(delta_rmse, ddof=0)),
                "delta_wape_mean": float(np.mean(delta_wape)),
                "delta_wape_std": float(np.std(delta_wape, ddof=0)),
                "delta_mae_mean": float(np.mean(delta_mae)),
                "delta_mae_std": float(np.std(delta_mae, ddof=0)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("delta_wape_mean", ascending=False).reset_index(drop=True)


def require_shap():
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise ImportError(
            "The `shap` package is required for the XAI notebook. "
            "Run the notebook setup cell to install missing runtime packages."
        ) from exc
    return shap


def compute_tree_shap(experiment: FittedXGBExperiment) -> dict[str, Any]:
    X = experiment.test_df[experiment.feature_cols].copy()
    if hasattr(experiment.model, "get_booster"):
        try:
            import xgboost as xgb  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise ImportError(
                "The `xgboost` package is required for the XAI notebook."
            ) from exc

        booster = experiment.model.get_booster()
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        best_iteration = getattr(experiment.model, "best_iteration", None)
        predict_kwargs: dict[str, Any] = {"pred_contribs": True}
        if best_iteration is not None:
            predict_kwargs["iteration_range"] = (0, int(best_iteration) + 1)
        contrib = np.asarray(booster.predict(dmatrix, **predict_kwargs), dtype=float)
        if contrib.ndim != 2 or contrib.shape[1] != len(X.columns) + 1:
            raise ValueError(
                "Unexpected contribution matrix shape from XGBoost pred_contribs: "
                f"{contrib.shape}"
            )
        shap_values = contrib[:, :-1]
        expected_value = float(np.mean(contrib[:, -1]))
        explainer = None
    else:
        shap = require_shap()
        explainer = shap.TreeExplainer(experiment.model)
        shap_values = np.asarray(explainer.shap_values(X), dtype=float)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            expected_value = float(np.asarray(expected_value).reshape(-1)[0])
    return {
        "explainer": explainer,
        "X": X,
        "feature_names": list(X.columns),
        "shap_values": shap_values,
        "expected_value": float(expected_value),
    }


def _coerce_gradient_shap_output(raw_values: Any) -> np.ndarray | list[np.ndarray]:
    if isinstance(raw_values, list):
        coerced = [np.asarray(value, dtype=float) for value in raw_values]
        return [value[..., 0] if value.ndim > 0 and value.shape[-1] == 1 else value for value in coerced]
    values = np.asarray(raw_values, dtype=float)
    return values[..., 0] if values.ndim > 0 and values.shape[-1] == 1 else values


def compute_lstm_gradient_shap(
    experiment: FittedLSTMExperiment,
    *,
    n_background: int = 128,
    n_explain: int = 256,
    random_seed: int = 42,
    explain_idx: np.ndarray | list[int] | None = None,
) -> dict[str, Any]:
    shap = require_shap()
    rng = np.random.default_rng(int(random_seed))

    n_fit = experiment.X_fit_dynamic.shape[0]
    n_test = experiment.X_test_dynamic.shape[0]
    if n_fit == 0 or n_test == 0:
        raise ValueError("LSTM SHAP requires non-empty fit and test sequence arrays.")

    bg_size = min(int(n_background), n_fit)
    bg_idx = np.sort(rng.choice(n_fit, size=bg_size, replace=False))
    if explain_idx is None:
        explain_size = min(int(n_explain), n_test)
        explain_idx_arr = np.sort(rng.choice(n_test, size=explain_size, replace=False))
    else:
        explain_idx_arr = np.asarray(sorted({int(idx) for idx in explain_idx}), dtype=int)
        explain_idx_arr = explain_idx_arr[(explain_idx_arr >= 0) & (explain_idx_arr < n_test)]
        if explain_idx_arr.size == 0:
            raise ValueError("No valid explain_idx values remain after bounds filtering.")

    background_dyn = np.asarray(experiment.X_fit_dynamic[bg_idx], dtype="float32")
    explain_dyn = np.asarray(experiment.X_test_dynamic[explain_idx_arr], dtype="float32")

    if experiment.mode == "M4":
        if experiment.X_fit_static is None or experiment.X_test_static is None:
            raise ValueError("M4 SHAP requires static sequence inputs.")
        background_sta = np.asarray(experiment.X_fit_static[bg_idx], dtype="float32")
        explain_sta = np.asarray(experiment.X_test_static[explain_idx_arr], dtype="float32")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The structure of `inputs` doesn't match the expected structure\.",
                category=UserWarning,
            )
            explainer = shap.GradientExplainer(experiment.model, [background_dyn, background_sta])
            raw_values = explainer.shap_values([explain_dyn, explain_sta])
        shap_values = _coerce_gradient_shap_output(raw_values)
        if not isinstance(shap_values, list) or len(shap_values) != 2:
            raise ValueError("Unexpected SHAP output structure for multi-input LSTM.")
        dynamic_shap, static_shap = shap_values
        X_payload: Any = {
            "dynamic": explain_dyn,
            "static": explain_sta,
            "sample_idx": explain_idx_arr,
        }
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The structure of `inputs` doesn't match the expected structure\.",
                category=UserWarning,
            )
            explainer = shap.GradientExplainer(experiment.model, background_dyn)
            raw_values = explainer.shap_values(explain_dyn)
        dynamic_shap = _coerce_gradient_shap_output(raw_values)
        if isinstance(dynamic_shap, list):
            dynamic_shap = np.asarray(dynamic_shap[0], dtype=float)
        static_shap = None
        X_payload = {
            "dynamic": explain_dyn,
            "sample_idx": explain_idx_arr,
        }

    return {
        "explainer": explainer,
        "X": X_payload,
        "feature_names": list(experiment.dynamic_feature_cols),
        "static_feature_names": list(experiment.static_feature_cols),
        "shap_values": np.asarray(dynamic_shap, dtype=float),
        "static_shap_values": None if static_shap is None else np.asarray(static_shap, dtype=float),
        "sample_idx": explain_idx_arr,
        "expected_value": np.nan,
    }


def summarize_grouped_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 2:
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
    elif shap_values.ndim == 3:
        if shap_values.shape[-1] != len(feature_names):
            raise ValueError(
                "The trailing SHAP dimension must match the provided feature names. "
                f"Got shap_values.shape={shap_values.shape} and {len(feature_names)} feature names."
            )
        rows = []
        for group_name, cols in feature_groups.items():
            idx = [feature_names.index(col) for col in cols if col in feature_names]
            if not idx:
                continue
            grouped_signed = shap_values[:, :, idx].sum(axis=(1, 2))
            grouped_abs = np.abs(shap_values[:, :, idx]).sum(axis=(1, 2))
            rows.append(
                {
                    "feature_group": group_name,
                    "n_features": int(len(idx)),
                    "mean_abs_group_shap": float(np.mean(grouped_abs)),
                    "mean_signed_group_shap": float(np.mean(grouped_signed)),
                    "median_abs_group_shap": float(np.median(grouped_abs)),
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values("mean_abs_group_shap", ascending=False).reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported SHAP array rank: {shap_values.ndim}")

    rows = []
    for group_name, cols in feature_groups.items():
        cols = [col for col in cols if col in shap_df.columns]
        if not cols:
            continue
        grouped_signed = shap_df[cols].sum(axis=1)
        grouped_abs = shap_df[cols].abs().sum(axis=1)
        rows.append(
            {
                "feature_group": group_name,
                "n_features": int(len(cols)),
                "mean_abs_group_shap": float(grouped_abs.mean()),
                "mean_signed_group_shap": float(grouped_signed.mean()),
                "median_abs_group_shap": float(grouped_abs.median()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("mean_abs_group_shap", ascending=False).reset_index(drop=True)


def summarize_grouped_lstm_shap(
    dynamic_shap_values: np.ndarray,
    dynamic_feature_names: list[str],
    feature_groups: dict[str, list[str]],
    *,
    static_shap_values: np.ndarray | None = None,
    static_feature_names: list[str] | None = None,
) -> pd.DataFrame:
    base = summarize_grouped_shap(dynamic_shap_values, dynamic_feature_names, feature_groups)
    if static_shap_values is None or not static_feature_names:
        return base

    rows = []
    static_df = pd.DataFrame(np.asarray(static_shap_values, dtype=float), columns=static_feature_names)
    for group_name, cols in feature_groups.items():
        cols = [col for col in cols if col in static_df.columns]
        if not cols:
            continue
        grouped_signed = static_df[cols].sum(axis=1)
        grouped_abs = static_df[cols].abs().sum(axis=1)
        rows.append(
            {
                "feature_group": group_name,
                "n_features_static": int(len(cols)),
                "mean_abs_group_shap_static": float(grouped_abs.mean()),
                "mean_signed_group_shap_static": float(grouped_signed.mean()),
                "median_abs_group_shap_static": float(grouped_abs.median()),
            }
        )
    static_group_df = pd.DataFrame(rows)
    if static_group_df.empty:
        return base

    if base.empty:
        out = static_group_df.rename(
            columns={
                "n_features_static": "n_features",
                "mean_abs_group_shap_static": "mean_abs_group_shap",
                "mean_signed_group_shap_static": "mean_signed_group_shap",
                "median_abs_group_shap_static": "median_abs_group_shap",
            }
        )
        return out.sort_values("mean_abs_group_shap", ascending=False).reset_index(drop=True)

    merged = base.merge(static_group_df, on="feature_group", how="outer")
    merged["n_features"] = merged["n_features"].fillna(0).astype(int) + merged["n_features_static"].fillna(0).astype(int)
    merged["mean_abs_group_shap"] = merged["mean_abs_group_shap"].fillna(0.0) + merged["mean_abs_group_shap_static"].fillna(0.0)
    merged["mean_signed_group_shap"] = merged["mean_signed_group_shap"].fillna(0.0) + merged["mean_signed_group_shap_static"].fillna(0.0)
    merged["median_abs_group_shap"] = merged["median_abs_group_shap"].fillna(0.0) + merged["median_abs_group_shap_static"].fillna(0.0)
    keep_cols = [
        "feature_group",
        "n_features",
        "mean_abs_group_shap",
        "mean_signed_group_shap",
        "median_abs_group_shap",
    ]
    return merged.loc[:, keep_cols].sort_values("mean_abs_group_shap", ascending=False).reset_index(drop=True)


def build_pfi_shap_agreement_table(
    pfi_df: pd.DataFrame,
    shap_group_df: pd.DataFrame,
) -> pd.DataFrame:
    if pfi_df.empty or shap_group_df.empty:
        return pd.DataFrame()
    work_pfi = pfi_df.copy()
    work_shap = shap_group_df.copy()
    work_pfi["pfi_rank_wape"] = work_pfi["delta_wape_mean"].rank(method="dense", ascending=False).astype(int)
    work_shap["shap_rank"] = work_shap["mean_abs_group_shap"].rank(method="dense", ascending=False).astype(int)
    merged = work_pfi.merge(work_shap, on=["feature_group", "n_features"], how="inner")
    if merged.empty:
        return merged
    merged["rank_gap"] = (merged["pfi_rank_wape"] - merged["shap_rank"]).abs()
    return merged.sort_values(["pfi_rank_wape", "shap_rank"]).reset_index(drop=True)


def _rollup_grouped_frame_to_broad(
    df: pd.DataFrame,
    *,
    value_cols: list[str],
    passthrough_cols: list[str],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work = work.loc[work["feature_group"].isin(FINE_TO_BROAD_FEATURE_GROUP)].copy()
    if work.empty:
        return work
    work["feature_group"] = work["feature_group"].map(FINE_TO_BROAD_FEATURE_GROUP)
    count_cols = [col for col in ["n_features"] if col in work.columns]
    group_cols = [
        col
        for col in work.columns
        if col not in set(value_cols + passthrough_cols + count_cols)
    ]
    agg_spec: dict[str, Any] = {col: "sum" for col in count_cols + value_cols if col in work.columns}
    agg_spec.update({col: "first" for col in passthrough_cols if col in work.columns})
    rolled = work.groupby(group_cols, as_index=False).agg(agg_spec)
    return rolled.sort_values(group_cols).reset_index(drop=True)


def rollup_seed_grouped_pfi_to_broad(seed_pfi_fine_df: pd.DataFrame) -> pd.DataFrame:
    return _rollup_grouped_frame_to_broad(
        seed_pfi_fine_df,
        value_cols=[
            "delta_rmse_mean",
            "delta_rmse_std",
            "delta_wape_mean",
            "delta_wape_std",
            "delta_mae_mean",
            "delta_mae_std",
        ],
        passthrough_cols=["baseline_rmse", "baseline_wape_pct", "baseline_mae"],
    )


def rollup_seed_grouped_shap_to_broad(seed_shap_fine_df: pd.DataFrame) -> pd.DataFrame:
    return _rollup_grouped_frame_to_broad(
        seed_shap_fine_df,
        value_cols=[
            "mean_abs_group_shap",
            "mean_signed_group_shap",
        ],
        passthrough_cols=["median_abs_group_shap"],
    )


def rollup_seed_agreement_to_broad(seed_agreement_fine_df: pd.DataFrame) -> pd.DataFrame:
    if seed_agreement_fine_df.empty:
        return seed_agreement_fine_df.copy()
    pfi_cols = [
        "feature_group",
        "n_features",
        "baseline_rmse",
        "baseline_wape_pct",
        "baseline_mae",
        "delta_rmse_mean",
        "delta_wape_mean",
        "delta_mae_mean",
    ]
    shap_cols = [
        "feature_group",
        "n_features",
        "mean_abs_group_shap",
        "mean_signed_group_shap",
        "median_abs_group_shap",
    ]
    key_cols = [
        col
        for col in seed_agreement_fine_df.columns
        if col not in {"feature_group", "n_features", "pfi_rank_wape", "shap_rank", "rank_gap"}
        and col not in {
            "baseline_rmse",
            "baseline_wape_pct",
            "baseline_mae",
            "delta_rmse_mean",
            "delta_wape_mean",
            "delta_mae_mean",
            "mean_abs_group_shap",
            "mean_signed_group_shap",
            "median_abs_group_shap",
        }
    ]
    pfi_df = rollup_seed_grouped_pfi_to_broad(seed_agreement_fine_df.loc[:, [col for col in key_cols + pfi_cols if col in seed_agreement_fine_df.columns]])
    shap_df = rollup_seed_grouped_shap_to_broad(seed_agreement_fine_df.loc[:, [col for col in key_cols + shap_cols if col in seed_agreement_fine_df.columns]])
    return build_pfi_shap_agreement_table(pfi_df, shap_df)


def select_representative_cases(experiment: FittedXGBExperiment) -> pd.DataFrame:
    pred_df = experiment.predictions_df.copy()
    if pred_df.empty:
        return pd.DataFrame()

    pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])
    pred_df["hour"] = pred_df["datetime"].dt.hour
    if "feat_outdoor_temp_c" not in pred_df.columns and "feat_outdoor_temp_c" in experiment.test_df.columns:
        pred_df["feat_outdoor_temp_c"] = experiment.test_df["feat_outdoor_temp_c"].to_numpy()
    if "feat_heat_obs" not in pred_df.columns and "feat_heat_obs" in experiment.test_df.columns:
        pred_df["feat_heat_obs"] = experiment.test_df["feat_heat_obs"].to_numpy()
    if "feat_temp_diff3h" not in pred_df.columns and "feat_temp_diff3h" in experiment.test_df.columns:
        pred_df["feat_temp_diff3h"] = experiment.test_df["feat_temp_diff3h"].to_numpy()

    pred_df["heat_ramp_delta"] = pred_df["y_true"] - pred_df.get("feat_heat_obs", pd.Series(np.nan, index=pred_df.index))
    pred_df["abs_temp_shift"] = pred_df.get("feat_temp_diff3h", pd.Series(np.nan, index=pred_df.index)).abs()
    pred_df["cold_rank_temp"] = pred_df.get("feat_outdoor_temp_c", pd.Series(np.nan, index=pred_df.index))

    used: set[int] = set()
    selected_rows: list[pd.Series] = []

    def pick_row(case_type: str, mask: pd.Series, sort_cols: list[str], ascending: list[bool]) -> None:
        subset = pred_df.loc[mask & (~pred_df["row_idx"].isin(used))].copy()
        if subset.empty:
            return
        subset = subset.sort_values(sort_cols, ascending=ascending)
        row = subset.iloc[0].copy()
        row["case_type"] = case_type
        selected_rows.append(row)
        used.add(int(row["row_idx"]))

    y_q95 = float(pred_df["y_true"].quantile(0.95))
    temp_q15 = float(pred_df["feat_outdoor_temp_c"].quantile(0.15)) if "feat_outdoor_temp_c" in pred_df else np.nan
    ramp_q90 = float(pred_df["heat_ramp_delta"].quantile(0.90))
    ramp_q10 = float(pred_df["heat_ramp_delta"].quantile(0.10))
    weather_q95 = float(pred_df["abs_temp_shift"].quantile(0.95)) if "abs_temp_shift" in pred_df else np.nan

    cold_mask = (pred_df["y_true"] >= y_q95) & (pred_df["feat_outdoor_temp_c"] <= temp_q15)
    morning_mask = pred_df["hour"].between(5, 9) & (pred_df["heat_ramp_delta"] >= ramp_q90)
    coasting_mask = pred_df["hour"].between(10, 18) & (pred_df["heat_ramp_delta"] <= ramp_q10)
    weather_mask = pred_df["abs_temp_shift"] >= weather_q95

    pick_row("cold_peak", cold_mask, ["y_true", "feat_outdoor_temp_c"], [False, True])
    pick_row("morning_recovery", morning_mask, ["heat_ramp_delta", "y_true"], [False, False])
    pick_row("coasting_decline", coasting_mask, ["heat_ramp_delta", "y_true"], [True, False])
    pick_row("abrupt_weather_shift", weather_mask, ["abs_temp_shift", "abs_error"], [False, False])
    pick_row("high_error", pd.Series(True, index=pred_df.index), ["abs_error"], [False])

    stability_candidates = pred_df.loc[
        (~pred_df["row_idx"].isin(used))
        & pred_df["hour"].between(5, 9)
        & pred_df["feat_outdoor_temp_c"].notna()
        & pred_df["feat_heat_obs"].notna()
    ].copy()
    if len(stability_candidates) >= 2:
        pair_features = ["feat_outdoor_temp_c", "feat_heat_obs"]
        if "feat_rh_pct" in experiment.test_df.columns:
            stability_candidates["feat_rh_pct"] = experiment.test_df.loc[
                stability_candidates["row_idx"], "feat_rh_pct"
            ].to_numpy()
            pair_features.append("feat_rh_pct")
        feature_mat = stability_candidates[pair_features].apply(pd.to_numeric, errors="coerce").dropna()
        if len(feature_mat) >= 2:
            feature_norm = (feature_mat - feature_mat.mean()) / feature_mat.std(ddof=0).replace(0.0, 1.0)
            candidate_idx = feature_norm.index.to_numpy()
            values = feature_norm.to_numpy(dtype=float)
            dist = np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))
            np.fill_diagonal(dist, np.inf)
            best_pair = np.unravel_index(np.argmin(dist), dist.shape)
            first_row = stability_candidates.loc[candidate_idx[best_pair[0]]].copy()
            second_row = stability_candidates.loc[candidate_idx[best_pair[1]]].copy()
            first_row["case_type"] = "stability_anchor"
            second_row["case_type"] = "stability_match"
            selected_rows.extend([first_row, second_row])

    if not selected_rows:
        return pd.DataFrame()
    case_df = pd.DataFrame(selected_rows)
    cols = [
        "case_type",
        "row_idx",
        "datetime",
        "hour",
        "y_true",
        "y_pred",
        "abs_error",
        "feat_outdoor_temp_c",
        "feat_heat_obs",
        "heat_ramp_delta",
        "abs_temp_shift",
    ]
    keep_cols = [col for col in cols if col in case_df.columns]
    return case_df.loc[:, keep_cols].reset_index(drop=True)


def _context_feature_candidates(top_features: list[str], available_cols: list[str]) -> list[str]:
    context_features = []
    blocked_prefixes = ("feat_hour_", "feat_dow_", "feat_month_")
    blocked_names = {"feat_is_weekend", "feat_is_night", "feat_is_daytime"}
    for feat in top_features:
        if feat in blocked_names or feat.startswith(blocked_prefixes):
            continue
        if feat not in context_features:
            context_features.append(feat)
        if len(context_features) >= 3:
            break
    for feat in DEFAULT_CONTEXT_FALLBACKS:
        if feat in available_cols and feat not in context_features:
            context_features.append(feat)
        if len(context_features) >= 3:
            break
    return context_features[:3]


def plot_group_importance_bars(
    df: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    if df.empty:
        raise ValueError("Cannot plot an empty importance table.")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = df.sort_values(value_col, ascending=True)
    ax.barh(plot_df["feature_group"], plot_df[value_col], color="#33658A")
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_horizon_mode_group_heatmap(
    df: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    save_path: Path | None = None,
    fill_value: float = 0.0,
    taxonomy: str = "broad",
) -> plt.Figure:
    if df.empty:
        raise ValueError("Cannot plot an empty horizon heatmap table.")
    pivot_df = df.pivot_table(
        index="feature_group",
        columns=["mode", "horizon_h"],
        values=value_col,
        aggfunc="mean",
    ).fillna(fill_value)
    order = [group for group in feature_group_order_for_taxonomy(taxonomy) if group in pivot_df.index]
    if order:
        pivot_df = pivot_df.loc[order]
    fig_width = max(8.0, 1.2 * pivot_df.shape[1])
    fig_height = max(4.5, 0.6 * pivot_df.shape[0] + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(pivot_df, cmap="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Mode / horizon")
    ax.set_ylabel("")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_group_trend_lines(
    df: pd.DataFrame,
    *,
    feature_groups: list[str],
    value_col: str,
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    plot_df = df.loc[df["feature_group"].isin(feature_groups)].copy()
    if plot_df.empty:
        raise ValueError("Cannot plot trend lines for an empty feature-group slice.")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.lineplot(
        data=plot_df,
        x="horizon_h",
        y=value_col,
        hue="feature_group",
        style="mode",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel(value_col.replace("_", " "))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def build_manifest_coverage_summary(
    manifest_df: pd.DataFrame,
    seed_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    coverage_keys = MATRIX_KEY_COLS + ["seed", "training_scope"]
    if manifest_df.empty or not all(col in manifest_df.columns for col in coverage_keys):
        return pd.DataFrame()
    observed_keys = seed_metrics_df[coverage_keys].drop_duplicates().assign(observed=True) if not seed_metrics_df.empty else pd.DataFrame(columns=coverage_keys + ["observed"])
    coverage_df = manifest_df[coverage_keys].drop_duplicates().merge(observed_keys, on=coverage_keys, how="left")
    coverage_df["observed"] = coverage_df["observed"].fillna(False).astype(bool)
    return (
        coverage_df.groupby(["regime", "weather_mode", "horizon_h"], as_index=False)
        .agg(
            completed_slots=("observed", "sum"),
            total_slots=("observed", "size"),
        )
        .assign(coverage_pct=lambda df: df["completed_slots"] / df["total_slots"])
        .reset_index(drop=True)
    )


def plot_manifest_coverage_heatmap(
    coverage_summary_df: pd.DataFrame,
    *,
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.5, 3.8), constrained_layout=True)
    if coverage_summary_df.empty:
        ax.text(0.5, 0.5, "No coverage data available", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
    else:
        pivot_df = coverage_summary_df.pivot(index=["regime", "weather_mode"], columns="horizon_h", values="coverage_pct")
        label_df = pivot_df.apply(lambda col: col.map(lambda value: f"{value:.0%}" if pd.notna(value) else ""))
        sns.heatmap(
            pivot_df,
            annot=label_df,
            fmt="",
            cmap="YlGn",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Coverage share"},
            ax=ax,
        )
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel("Regime / weather mode")
    ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_feature_share_horizon_lines(
    share_overview_df: pd.DataFrame,
    *,
    method: str,
    title: str,
    save_path: Path | None = None,
    model_family: str | None = None,
    regime: str | None = None,
    weather_mode: str | None = None,
    modes: tuple[str, ...] | list[str] | None = None,
    feature_groups: tuple[str, ...] | list[str] | None = None,
    taxonomy: str = "broad",
) -> plt.Figure:
    plot_df = share_overview_df.loc[share_overview_df["method"] == str(method)].copy()
    if model_family is not None:
        plot_df = plot_df.loc[plot_df["model_family"] == str(model_family)]
    if regime is not None:
        plot_df = plot_df.loc[plot_df["regime"] == str(regime)]
    if weather_mode is not None:
        plot_df = plot_df.loc[plot_df["weather_mode"] == str(weather_mode)]
    if modes is not None:
        plot_df = plot_df.loc[plot_df["mode"].isin([str(mode) for mode in modes])]
    if feature_groups is not None:
        plot_df = plot_df.loc[plot_df["feature_group"].isin([str(group) for group in feature_groups])]
    if plot_df.empty:
        raise ValueError("Cannot plot feature-share trends for an empty slice.")
    summary_df = (
        plot_df.groupby(["horizon_h", "feature_group"], as_index=False)
        .agg(mean_share=("mean_share", "mean"))
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    labels = feature_group_labels_for_taxonomy(taxonomy)
    palette = feature_group_palette_for_taxonomy(taxonomy)
    for feature_group in feature_group_order_for_taxonomy(taxonomy):
        feature_slice = summary_df.loc[summary_df["feature_group"] == feature_group].sort_values("horizon_h")
        if feature_slice.empty:
            continue
        ax.plot(
            feature_slice["horizon_h"],
            feature_slice["mean_share"],
            marker="o",
            linewidth=2.1,
            color=palette.get(feature_group, "#4c4c4c"),
            label=labels.get(feature_group, str(feature_group)),
        )
    ax.set_title(title)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Average within-model share")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_mode_feature_share_lines(
    share_overview_df: pd.DataFrame,
    *,
    method: str,
    regime: str,
    weather_mode: str,
    modes: tuple[str, ...] | list[str],
    title: str,
    save_path: Path | None = None,
    model_family: str | None = None,
    feature_groups: tuple[str, ...] | list[str] | None = None,
    taxonomy: str = "broad",
) -> plt.Figure:
    plot_df = share_overview_df.loc[
        (share_overview_df["method"] == str(method))
        & (share_overview_df["regime"] == str(regime))
        & (share_overview_df["weather_mode"] == str(weather_mode))
        & (share_overview_df["mode"].isin([str(mode) for mode in modes]))
    ].copy()
    if model_family is not None:
        plot_df = plot_df.loc[plot_df["model_family"] == str(model_family)]
    if feature_groups is not None:
        plot_df = plot_df.loc[plot_df["feature_group"].isin([str(group) for group in feature_groups])]
    if plot_df.empty:
        raise ValueError("Cannot plot mode-share trends for an empty slice.")
    mode_list = [str(mode) for mode in modes if str(mode) in set(plot_df["mode"])]
    fig, axes = plt.subplots(
        1,
        len(mode_list),
        figsize=(5.2 * max(len(mode_list), 1), 4.2),
        sharey=True,
        constrained_layout=True,
    )
    if len(mode_list) == 1:
        axes = [axes]
    labels = feature_group_labels_for_taxonomy(taxonomy)
    palette = feature_group_palette_for_taxonomy(taxonomy)
    for ax, mode in zip(axes, mode_list):
        mode_slice = plot_df.loc[plot_df["mode"] == mode]
        for feature_group in feature_group_order_for_taxonomy(taxonomy):
            feature_slice = mode_slice.loc[mode_slice["feature_group"] == feature_group].sort_values("horizon_h")
            if feature_slice.empty:
                continue
            ax.plot(
                feature_slice["horizon_h"],
                feature_slice["mean_share"],
                marker="o",
                linewidth=2.0,
                color=palette.get(feature_group, "#4c4c4c"),
                label=labels.get(feature_group, str(feature_group)),
            )
        ax.set_title(mode)
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Average within-model share")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    fig.suptitle(title, y=1.16)
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    *,
    max_display: int = 12,
    save_path: Path | None = None,
) -> plt.Figure:
    shap = require_shap()
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_local_case_figure(
    experiment: FittedXGBExperiment,
    shap_values: np.ndarray,
    *,
    row_idx: int,
    case_type: str,
    save_path: Path | None = None,
    context_hours: int = 24,
    top_k: int = 8,
) -> plt.Figure:
    pred_row = experiment.predictions_df.loc[experiment.predictions_df["row_idx"] == int(row_idx)]
    if pred_row.empty:
        raise KeyError(f"row_idx {row_idx} missing from predictions.")
    pred_row = pred_row.iloc[0]
    test_row = experiment.test_df.iloc[int(row_idx)]
    issue_time = pd.Timestamp(pred_row["datetime"])
    start_time = issue_time - pd.Timedelta(hours=int(context_hours))
    context_df = experiment.raw_frame.loc[
        (pd.to_datetime(experiment.raw_frame["datetime"]) >= start_time)
        & (pd.to_datetime(experiment.raw_frame["datetime"]) <= issue_time)
    ].copy()

    feature_names = list(experiment.feature_cols)
    shap_row = np.asarray(shap_values[int(row_idx)], dtype=float)
    shap_abs = pd.Series(np.abs(shap_row), index=feature_names).sort_values(ascending=False)
    top_features = shap_abs.head(int(top_k)).index.tolist()
    context_features = _context_feature_candidates(top_features, list(context_df.columns))

    n_left = 1 + len(context_features)
    fig, axes = plt.subplots(
        n_left,
        2,
        figsize=(14, max(5.5, 2.4 * n_left)),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
        squeeze=False,
    )

    heat_ax = axes[0, 0]
    heat_ax.plot(context_df["datetime"], context_df["heat_kwh"], color="#1b4965", linewidth=1.8)
    heat_ax.axvline(issue_time, color="#c1121f", linestyle="--", linewidth=1.2)
    heat_ax.scatter([issue_time], [test_row.get("feat_heat_obs", np.nan)], color="#1b4965", s=24, zorder=3)
    heat_ax.set_ylabel("Heat kWh")
    heat_ax.set_title(
        f"{experiment.building} | {experiment.mode} | {experiment.target_kind}_h{experiment.horizon_h:02d} | {case_type}\n"
        f"issue={issue_time} | pred={pred_row['y_pred']:.1f} | actual={pred_row['y_true']:.1f}"
    )
    heat_ax.grid(alpha=0.25)

    for ax, feature_name in zip(axes[1:, 0], context_features):
        ax.plot(context_df["datetime"], context_df[feature_name], linewidth=1.6, color="#5f0f40")
        ax.axvline(issue_time, color="#c1121f", linestyle="--", linewidth=1.0)
        ax.set_ylabel(feature_name)
        ax.grid(alpha=0.25)

    for ax in axes[:, 1]:
        ax.axis("off")

    bar_ax = axes[:, 1][0]
    bar_ax.axis("on")
    local_df = pd.DataFrame(
        {
            "feature": top_features[::-1],
            "shap_value": shap_row[[feature_names.index(feat) for feat in top_features[::-1]]],
        }
    )
    colors = np.where(local_df["shap_value"] >= 0, "#c1121f", "#1d3557")
    bar_ax.barh(local_df["feature"], local_df["shap_value"], color=colors)
    bar_ax.axvline(0.0, color="black", linewidth=0.9)
    bar_ax.set_title("Top local SHAP values")
    bar_ax.set_xlabel("SHAP contribution")
    bar_ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def scenario_history_columns(feature_cols: list[str]) -> dict[str, list[str]]:
    heat_cols = [col for col in HEAT_HISTORY_FEATURES if col in feature_cols]
    weather_cols = [col for col in WEATHER_HISTORY_FEATURES if col in feature_cols]
    system_cols = [
        col
        for col in feature_cols
        if col.endswith("_lag1")
        or col.endswith("_roll6h")
        or col.endswith("_roll24h")
        or col.endswith("_low_deltaT_flag")
        or col.endswith("_low_deltaT_roll24h")
    ]
    return {
        "heat_history": heat_cols,
        "historical_weather_memory": weather_cols,
        "system_history": system_cols,
    }


def run_shift_scenarios(
    experiment: FittedXGBExperiment,
    *,
    row_idx: int,
    shift_hours: tuple[int, ...] = (1, 2),
    include_weather_memory: bool = True,
) -> pd.DataFrame:
    pred_row = experiment.predictions_df.loc[experiment.predictions_df["row_idx"] == int(row_idx)]
    if pred_row.empty:
        raise KeyError(f"row_idx {row_idx} missing from predictions.")
    pred_row = pred_row.iloc[0]
    issue_time = pd.Timestamp(pred_row["datetime"])
    frame = experiment.raw_frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    current_rows = frame.index[frame["datetime"] == issue_time]
    if len(current_rows) != 1:
        raise KeyError(f"Expected one frame row at {issue_time}, found {len(current_rows)}")
    current_row_idx = int(current_rows[0])

    feature_blocks = scenario_history_columns(experiment.feature_cols)
    history_cols = list(feature_blocks["heat_history"])
    if include_weather_memory:
        history_cols.extend(feature_blocks["historical_weather_memory"])
        history_cols.extend(feature_blocks["system_history"])
    history_cols = [col for col in history_cols if col in experiment.feature_cols]

    baseline_features = experiment.test_df.loc[int(row_idx), experiment.feature_cols].copy()
    baseline_pred = float(pred_row["y_pred"])
    rows = [
        {
            "scenario": "baseline",
            "shift_hours": 0,
            "prediction": baseline_pred,
            "delta_prediction": 0.0,
            "source_time": issue_time,
            "n_changed_features": 0,
        }
    ]

    for shift in shift_hours:
        for direction, sign in (("earlier", -1), ("later", 1)):
            source_idx = current_row_idx + sign * int(shift)
            if source_idx < 0 or source_idx >= len(frame):
                continue
            source_row = frame.iloc[source_idx]
            scenario_features = baseline_features.copy()
            for col in history_cols:
                if col in source_row.index:
                    scenario_features[col] = source_row[col]
            scenario_df = pd.DataFrame([scenario_features], columns=experiment.feature_cols)
            scenario_pred = float(mfc._predict_with_best_iteration(experiment.model, scenario_df)[0])
            rows.append(
                {
                    "scenario": f"history_{direction}_{int(shift)}h",
                    "shift_hours": int(shift),
                    "prediction": scenario_pred,
                    "delta_prediction": scenario_pred - baseline_pred,
                    "source_time": pd.Timestamp(source_row["datetime"]),
                    "n_changed_features": int(len(history_cols)),
                }
            )
    return pd.DataFrame(rows)


def plot_scenario_deltas(
    scenario_df: pd.DataFrame,
    *,
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    if scenario_df.empty:
        raise ValueError("Scenario dataframe is empty.")
    plot_df = scenario_df.loc[scenario_df["scenario"] != "baseline"].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = np.where(plot_df["delta_prediction"] >= 0, "#c1121f", "#1d3557")
    ax.barh(plot_df["scenario"], plot_df["delta_prediction"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("Prediction delta vs baseline")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


MATRIX_KEY_COLS = [
    "model_family",
    "regime",
    "building",
    "mode",
    "weather_mode",
    "horizon_h",
    "target_kind",
]


def clone_config_with_overrides(
    config: mfc.ExperimentConfig,
    **overrides: Any,
) -> mfc.ExperimentConfig:
    config_kwargs = {name: getattr(config, name) for name in config.__dataclass_fields__.keys()}
    config_kwargs.update(overrides)
    return mfc.ExperimentConfig(**config_kwargs)


def build_xai_matrix_manifest(
    *,
    model_families: tuple[str, ...],
    regimes: tuple[str, ...],
    buildings: tuple[str, ...],
    modes: tuple[str, ...],
    weather_modes: tuple[str, ...],
    horizons: tuple[int, ...],
    target_kind: str,
    seed_list: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    horizons = normalize_horizons_for_target_kind(horizons, target_kind)
    for model_family in model_families:
        for regime in regimes:
            for building in buildings:
                for mode in modes:
                    for weather_mode in weather_modes:
                        for horizon_h in horizons:
                            for seed in seed_list:
                                rows.append(
                                    {
                                        "model_family": model_family,
                                        "regime": regime,
                                        "building": building,
                                        "mode": mode,
                                        "weather_mode": weather_mode,
                                        "horizon_h": int(horizon_h),
                                        "target_kind": str(target_kind),
                                        "seed": int(seed),
                                        "training_scope": "POOLED" if regime == "pooled_same_buildings" else str(building),
                                    }
                                )
    return pd.DataFrame(rows).sort_values(MATRIX_KEY_COLS + ["seed"]).reset_index(drop=True)


def _normalize_xai_manifest_frame(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = MATRIX_KEY_COLS + ["seed", "training_scope"]
    if df.empty:
        return pd.DataFrame(columns=key_cols)
    work = df.copy()
    for col in key_cols:
        if col not in work.columns:
            work[col] = pd.NA
    work["horizon_h"] = pd.to_numeric(work["horizon_h"], errors="coerce")
    work["seed"] = pd.to_numeric(work["seed"], errors="coerce")
    work = work.loc[work["horizon_h"].notna() & work["seed"].notna(), key_cols].copy()
    if work.empty:
        return pd.DataFrame(columns=key_cols)
    for col in ("model_family", "regime", "building", "mode", "weather_mode", "target_kind"):
        work[col] = work[col].astype(str)
    work["horizon_h"] = work["horizon_h"].astype(int)
    work["seed"] = work["seed"].astype(int)
    inferred_scope = np.where(
        work["regime"].astype(str).eq("pooled_same_buildings"),
        "POOLED",
        work["building"].astype(str),
    )
    work["training_scope"] = work["training_scope"].where(work["training_scope"].notna(), inferred_scope)
    work["training_scope"] = work["training_scope"].astype(str)
    return (
        work.drop_duplicates(subset=key_cols, keep="last")
        .sort_values(key_cols)
        .reset_index(drop=True)
    )


def build_missing_xai_manifest(
    desired_manifest_df: pd.DataFrame,
    current_manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    desired = _normalize_xai_manifest_frame(desired_manifest_df)
    current = _normalize_xai_manifest_frame(current_manifest_df)
    if desired.empty:
        return desired.copy()
    key_cols = MATRIX_KEY_COLS + ["seed", "training_scope"]
    current_keys = current.loc[:, key_cols].drop_duplicates()
    missing = desired.merge(
        current_keys.assign(_present=True),
        on=key_cols,
        how="left",
    )
    missing = missing.loc[missing["_present"].fillna(False) == False].drop(columns="_present")
    return missing.reset_index(drop=True)


def build_completed_xai_manifest_from_outputs(
    outputs_or_results_dir: dict[str, Any] | Path | str,
    desired_manifest_df: pd.DataFrame,
    *,
    buildings: tuple[str, ...],
    require_fine_groups: bool = False,
    allowed_model_families: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    desired = _normalize_xai_manifest_frame(desired_manifest_df)
    if desired.empty:
        return desired.copy()
    if isinstance(outputs_or_results_dir, (str, Path)):
        outputs = load_saved_xai_outputs(
            Path(outputs_or_results_dir),
            allowed_model_families=allowed_model_families,
            use_latest_run_log=True,
        )
    else:
        outputs = _normalize_xai_outputs(outputs_or_results_dir)
    combo_cols = [col for col in MATRIX_KEY_COLS if col != "building"] + ["seed"]
    completed_slots = _completed_xai_combo_slots(
        outputs.get("seed_metrics_df", pd.DataFrame()),
        buildings=tuple(str(building) for building in buildings),
        manifest_df=desired,
        seed_grouped_pfi_fine_df=outputs.get("seed_grouped_pfi_fine_df", pd.DataFrame()),
        seed_grouped_shap_fine_df=outputs.get("seed_grouped_shap_fine_df", pd.DataFrame()),
        require_fine_groups=require_fine_groups,
    )
    if not completed_slots:
        return desired.iloc[0:0].copy()
    completed = desired.copy()
    completed["_slot_id"] = completed.apply(
        lambda row: tuple(row[col] for col in combo_cols),
        axis=1,
    )
    completed = completed.loc[completed["_slot_id"].isin(completed_slots)].drop(columns="_slot_id")
    return completed.reset_index(drop=True)


def build_missing_xai_resume_manifest(
    outputs_or_results_dir: dict[str, Any] | Path | str,
    desired_manifest_df: pd.DataFrame,
    *,
    buildings: tuple[str, ...],
    require_fine_groups: bool = False,
    allowed_model_families: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    completed_manifest_df = build_completed_xai_manifest_from_outputs(
        outputs_or_results_dir,
        desired_manifest_df,
        buildings=buildings,
        require_fine_groups=require_fine_groups,
        allowed_model_families=allowed_model_families,
    )
    return build_missing_xai_manifest(desired_manifest_df, completed_manifest_df)


def build_xai_matrix_inventory(
    desired_manifest_df: pd.DataFrame,
    current_manifest_df: pd.DataFrame,
    *,
    source_dir: Path | str,
    notebook: str = "13",
    layer: str = "xai_matrix",
) -> pd.DataFrame:
    desired = _normalize_xai_manifest_frame(desired_manifest_df)
    current = _normalize_xai_manifest_frame(current_manifest_df)
    key_cols = MATRIX_KEY_COLS + ["seed", "training_scope"]
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
                "training_scope",
            ]
        )
    inventory = desired.merge(
        current.loc[:, key_cols].drop_duplicates().assign(completed_now=True),
        on=key_cols,
        how="left",
    )
    inventory["completed_now"] = inventory["completed_now"].fillna(False).astype(bool)
    inventory["notebook"] = str(notebook)
    inventory["layer"] = str(layer)
    inventory["source_dir"] = str(Path(source_dir).resolve())
    inventory["model_family_scope"] = inventory["model_family"].astype(str)
    inventory["building_scope"] = inventory["training_scope"].astype(str)
    inventory["status"] = np.where(inventory["completed_now"], "complete", "missing")
    inventory["desired_full_matrix"] = True
    inventory["notes"] = np.where(
        inventory["completed_now"],
        "Present in the current XAI manifest.",
        "Missing from the current XAI manifest; safe to schedule for resume-only execution.",
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
        "training_scope",
    ]
    return inventory.loc[:, ordered_cols].sort_values(
        ["model_family_scope", "regime", "building_scope", "mode", "weather_mode", "horizon_h", "seed"]
    ).reset_index(drop=True)


def xai_manifest_overlap_report(
    subset_manifest_df: pd.DataFrame,
    superset_manifest_df: pd.DataFrame,
    *,
    subset_label: str,
    superset_label: str,
) -> pd.DataFrame:
    subset = _normalize_xai_manifest_frame(subset_manifest_df)
    superset = _normalize_xai_manifest_frame(superset_manifest_df)
    missing = build_missing_xai_manifest(subset, superset)
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


def _normalize_xai_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    manifest_df = _dedupe(
        _sort_reset(
            outputs.get("manifest_df", pd.DataFrame()),
            MATRIX_KEY_COLS + ["seed", "training_scope"],
        ),
        MATRIX_KEY_COLS + ["seed", "training_scope"],
    )
    seed_metrics_df = _dedupe(
        _sort_reset(
            outputs.get("seed_metrics_df", pd.DataFrame()),
            MATRIX_KEY_COLS + ["seed", "training_scope"],
        ),
        MATRIX_KEY_COLS + ["seed", "training_scope"],
    )
    seed_grouped_pfi_df = _dedupe(
        _sort_reset(
            outputs.get("seed_grouped_pfi_df", pd.DataFrame()),
            MATRIX_KEY_COLS + ["seed", "training_scope", "feature_group"],
        ),
        MATRIX_KEY_COLS + ["seed", "training_scope", "feature_group"],
    )
    seed_grouped_pfi_fine_df = _dedupe(
        _sort_reset(
            outputs.get("seed_grouped_pfi_fine_df", pd.DataFrame()),
            MATRIX_KEY_COLS + ["seed", "training_scope", "feature_group"],
        ),
        MATRIX_KEY_COLS + ["seed", "training_scope", "feature_group"],
    )
    seed_grouped_shap_df = _dedupe(
        _sort_reset(
            outputs.get("seed_grouped_shap_df", pd.DataFrame()),
            MATRIX_KEY_COLS + ["seed", "training_scope", "feature_group"],
        ),
        MATRIX_KEY_COLS + ["seed", "training_scope", "feature_group"],
    )
    seed_grouped_shap_fine_df = _dedupe(
        _sort_reset(
            outputs.get("seed_grouped_shap_fine_df", pd.DataFrame()),
            MATRIX_KEY_COLS + ["seed", "training_scope", "feature_group"],
        ),
        MATRIX_KEY_COLS + ["seed", "training_scope", "feature_group"],
    )
    seed_agreement_df = _dedupe(
        _sort_reset(
            outputs.get("seed_agreement_df", pd.DataFrame()),
            MATRIX_KEY_COLS + ["seed", "training_scope", "feature_group"],
        ),
        MATRIX_KEY_COLS + ["seed", "training_scope", "feature_group"],
    )
    seed_agreement_fine_df = _dedupe(
        _sort_reset(
            outputs.get("seed_agreement_fine_df", pd.DataFrame()),
            MATRIX_KEY_COLS + ["seed", "training_scope", "feature_group"],
        ),
        MATRIX_KEY_COLS + ["seed", "training_scope", "feature_group"],
    )
    run_log_df = _sort_reset(
        outputs.get("run_log_df", pd.DataFrame()),
        ["slot_index"] + MATRIX_KEY_COLS + ["seed", "status"],
    )
    detail_cache = dict(outputs.get("detail_cache", {}))
    return {
        "manifest_df": manifest_df,
        "seed_metrics_df": seed_metrics_df,
        "seed_grouped_pfi_df": seed_grouped_pfi_df,
        "seed_grouped_pfi_fine_df": seed_grouped_pfi_fine_df,
        "seed_grouped_shap_df": seed_grouped_shap_df,
        "seed_grouped_shap_fine_df": seed_grouped_shap_fine_df,
        "seed_agreement_df": seed_agreement_df,
        "seed_agreement_fine_df": seed_agreement_fine_df,
        "run_log_df": run_log_df,
        "detail_cache": detail_cache,
    }


def load_saved_xai_outputs(
    results_dir: Path,
    *,
    manifest_df: pd.DataFrame | None = None,
    scope_to_manifest: bool = False,
    allowed_model_families: tuple[str, ...] | None = None,
    use_latest_run_log: bool = False,
) -> dict[str, Any]:
    paths = build_xai_artifact_paths(results_dir)
    outputs = {
        "manifest_df": manifest_df.copy() if manifest_df is not None else _read_csv_if_exists(paths["matrix_manifest"]),
        "seed_metrics_df": _read_csv_if_exists(paths["seed_metrics"]),
        "seed_grouped_pfi_df": _read_csv_if_exists(paths["seed_grouped_pfi"]),
        "seed_grouped_pfi_fine_df": _read_csv_if_exists(paths["seed_grouped_pfi_fine"]),
        "seed_grouped_shap_df": _read_csv_if_exists(paths["seed_grouped_shap"]),
        "seed_grouped_shap_fine_df": _read_csv_if_exists(paths["seed_grouped_shap_fine"]),
        "seed_agreement_df": _read_csv_if_exists(paths["seed_agreement"]),
        "seed_agreement_fine_df": _read_csv_if_exists(paths["seed_agreement_fine"]),
        "run_log_df": _read_csv_if_exists(paths["run_log"]),
        "detail_cache": {},
    }
    if outputs["manifest_df"].empty and manifest_df is not None:
        outputs["manifest_df"] = manifest_df.copy()
    normalized = _normalize_xai_outputs(outputs)
    if scope_to_manifest:
        return scope_xai_outputs_to_manifest(
            normalized,
            normalized["manifest_df"],
            allowed_model_families=allowed_model_families,
            use_latest_run_log=use_latest_run_log,
        )
    if allowed_model_families is not None:
        return scope_xai_outputs_to_manifest(
            normalized,
            normalized["manifest_df"],
            allowed_model_families=allowed_model_families,
            use_latest_run_log=use_latest_run_log,
        )
    if use_latest_run_log:
        normalized["run_log_df"] = latest_xai_run_log(normalized["run_log_df"])
    return normalized


def build_xai_artifact_status_table(results_dir: Path) -> pd.DataFrame:
    paths = build_xai_artifact_paths(results_dir)
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


def _write_xai_outputs(paths: dict[str, Path], outputs: dict[str, Any]) -> None:
    normalized = _normalize_xai_outputs(outputs)
    normalized["manifest_df"].to_csv(paths["matrix_manifest"], index=False)
    normalized["seed_metrics_df"].to_csv(paths["seed_metrics"], index=False)
    normalized["seed_grouped_pfi_df"].to_csv(paths["seed_grouped_pfi"], index=False)
    normalized["seed_grouped_pfi_fine_df"].to_csv(paths["seed_grouped_pfi_fine"], index=False)
    normalized["seed_grouped_shap_df"].to_csv(paths["seed_grouped_shap"], index=False)
    normalized["seed_grouped_shap_fine_df"].to_csv(paths["seed_grouped_shap_fine"], index=False)
    normalized["seed_agreement_df"].to_csv(paths["seed_agreement"], index=False)
    normalized["seed_agreement_fine_df"].to_csv(paths["seed_agreement_fine"], index=False)
    normalized["run_log_df"].to_csv(paths["run_log"], index=False)


def _filter_df_by_keys(
    df: pd.DataFrame,
    allowed_keys_df: pd.DataFrame,
    *,
    key_cols: list[str],
) -> pd.DataFrame:
    if df.empty or allowed_keys_df.empty:
        return df.iloc[0:0].copy()
    use_key_cols = [col for col in key_cols if col in df.columns and col in allowed_keys_df.columns]
    if not use_key_cols:
        return df.copy()
    allowed = allowed_keys_df.loc[:, use_key_cols].drop_duplicates().assign(_allowed=True)
    merged = df.merge(allowed, on=use_key_cols, how="left")
    return merged.loc[merged["_allowed"].fillna(False)].drop(columns=["_allowed"]).reset_index(drop=True)


def latest_xai_run_log(run_log_df: pd.DataFrame) -> pd.DataFrame:
    if run_log_df.empty:
        return run_log_df.copy()
    key_cols = MATRIX_KEY_COLS + ["seed", "training_scope"]
    work = run_log_df.copy()
    work["slot_index"] = pd.to_numeric(work.get("slot_index", np.nan), errors="coerce")
    if "elapsed_s" in work.columns:
        work["elapsed_s"] = pd.to_numeric(work["elapsed_s"], errors="coerce")
    status_rank = work["status"].astype(str).map({"ok": 3, "skipped": 2, "skipped_insufficient_rows": 2, "skipped_insufficient_sequences": 2, "error": 1}).fillna(0)
    work["_status_rank"] = status_rank.astype(int)
    sort_cols = [col for col in ["slot_index", "_status_rank", "elapsed_s"] if col in work.columns]
    work = work.sort_values(key_cols + sort_cols)
    return work.drop_duplicates(subset=key_cols, keep="last").drop(columns=["_status_rank"]).reset_index(drop=True)


def scope_xai_outputs_to_manifest(
    outputs: dict[str, Any],
    manifest_df: pd.DataFrame,
    *,
    allowed_model_families: tuple[str, ...] | None = None,
    use_latest_run_log: bool = False,
) -> dict[str, Any]:
    normalized = _normalize_xai_outputs(outputs)
    scoped_manifest_df = manifest_df.copy()
    if allowed_model_families is not None:
        allowed_families = {str(family) for family in allowed_model_families}
        if not scoped_manifest_df.empty and "model_family" in scoped_manifest_df.columns:
            scoped_manifest_df = scoped_manifest_df.loc[
                scoped_manifest_df["model_family"].astype(str).isin(allowed_families)
            ].reset_index(drop=True)

    metric_key_cols = MATRIX_KEY_COLS + ["seed", "training_scope"]
    scoped_outputs = {
        "manifest_df": scoped_manifest_df.copy(),
        "seed_metrics_df": _filter_df_by_keys(normalized["seed_metrics_df"], scoped_manifest_df, key_cols=metric_key_cols),
        "seed_grouped_pfi_df": _filter_df_by_keys(normalized["seed_grouped_pfi_df"], scoped_manifest_df, key_cols=metric_key_cols),
        "seed_grouped_pfi_fine_df": _filter_df_by_keys(normalized["seed_grouped_pfi_fine_df"], scoped_manifest_df, key_cols=metric_key_cols),
        "seed_grouped_shap_df": _filter_df_by_keys(normalized["seed_grouped_shap_df"], scoped_manifest_df, key_cols=metric_key_cols),
        "seed_grouped_shap_fine_df": _filter_df_by_keys(normalized["seed_grouped_shap_fine_df"], scoped_manifest_df, key_cols=metric_key_cols),
        "seed_agreement_df": _filter_df_by_keys(normalized["seed_agreement_df"], scoped_manifest_df, key_cols=metric_key_cols),
        "seed_agreement_fine_df": _filter_df_by_keys(normalized["seed_agreement_fine_df"], scoped_manifest_df, key_cols=metric_key_cols),
        "run_log_df": _filter_df_by_keys(normalized["run_log_df"], scoped_manifest_df, key_cols=metric_key_cols),
        "detail_cache": {},
    }
    if use_latest_run_log:
        scoped_outputs["run_log_df"] = latest_xai_run_log(scoped_outputs["run_log_df"])
    return _normalize_xai_outputs(scoped_outputs)


def _completed_xai_combo_slots(
    seed_metrics_df: pd.DataFrame,
    *,
    buildings: tuple[str, ...],
    manifest_df: pd.DataFrame | None = None,
    seed_grouped_pfi_fine_df: pd.DataFrame | None = None,
    seed_grouped_shap_fine_df: pd.DataFrame | None = None,
    require_fine_groups: bool = False,
) -> set[tuple[Any, ...]]:
    if seed_metrics_df.empty:
        return set()
    combo_cols = [col for col in MATRIX_KEY_COLS if col != "building"] + ["seed"]
    counts_df = (
        seed_metrics_df.groupby(combo_cols, dropna=False)["building"]
        .nunique()
        .reset_index(name="n_buildings")
    )
    if manifest_df is not None and not manifest_df.empty:
        expected_counts_df = (
            _normalize_xai_manifest_frame(manifest_df)
            .groupby(combo_cols, dropna=False)["building"]
            .nunique()
            .reset_index(name="expected_n_buildings")
        )
        counts_df = counts_df.merge(expected_counts_df, on=combo_cols, how="left")
        counts_df["expected_n_buildings"] = pd.to_numeric(
            counts_df["expected_n_buildings"], errors="coerce"
        ).fillna(int(len(buildings)))
    else:
        counts_df["expected_n_buildings"] = int(len(buildings))
    completion_mask = counts_df["n_buildings"] >= counts_df["expected_n_buildings"]

    if require_fine_groups:
        fine_specs = [
            ("n_buildings_fine_pfi", seed_grouped_pfi_fine_df),
            ("n_buildings_fine_shap", seed_grouped_shap_fine_df),
        ]
        for col_name, frame in fine_specs:
            if frame is None or frame.empty:
                counts_df[col_name] = 0
                completion_mask &= False
                continue
            fine_counts_df = (
                frame.groupby(combo_cols, dropna=False)["building"]
                .nunique()
                .reset_index(name=col_name)
            )
            counts_df = counts_df.merge(fine_counts_df, on=combo_cols, how="left")
            counts_df[col_name] = pd.to_numeric(counts_df[col_name], errors="coerce").fillna(0)
            completion_mask &= counts_df[col_name] >= counts_df["expected_n_buildings"]

    completed = counts_df.loc[completion_mask, combo_cols]
    return {
        tuple(row[col] for col in combo_cols)
        for _, row in completed.iterrows()
    }


def print_xai_resume_diagnostics(
    results_dir: Path,
    *,
    manifest_df: pd.DataFrame,
    buildings: tuple[str, ...],
    require_fine_groups: bool = False,
) -> None:
    paths = build_xai_artifact_paths(results_dir)
    outputs = load_saved_xai_outputs(results_dir, manifest_df=manifest_df)
    print("--- xai resume diagnostics ---")
    print(f"cwd: {Path.cwd()}")
    print(f"results_dir (resolved): {results_dir.resolve()}")
    for key in ("matrix_manifest", "seed_metrics", "seed_grouped_pfi", "seed_grouped_shap", "seed_agreement", "run_log"):
        p = paths[key]
        print(f"  {p.name}: exists={p.exists()} data_rows={len(_read_csv_if_exists(p))}")
    completed_slots = _completed_xai_combo_slots(
        outputs["seed_metrics_df"],
        buildings=buildings,
        manifest_df=manifest_df,
        seed_grouped_pfi_fine_df=outputs.get("seed_grouped_pfi_fine_df", pd.DataFrame()),
        seed_grouped_shap_fine_df=outputs.get("seed_grouped_shap_fine_df", pd.DataFrame()),
        require_fine_groups=require_fine_groups,
    )
    combo_cols = [col for col in MATRIX_KEY_COLS if col != "building"] + ["seed"]
    total_slots = 0 if manifest_df.empty else int(len(manifest_df[combo_cols].drop_duplicates()))
    print(f"combo slots complete: {len(completed_slots)}/{total_slots}")
    if len(completed_slots) < total_slots:
        remaining = manifest_df[combo_cols].drop_duplicates()
        remaining["slot_id"] = remaining.apply(lambda row: tuple(row[col] for col in combo_cols), axis=1)
        next_rows = remaining.loc[~remaining["slot_id"].isin(completed_slots)]
        if not next_rows.empty:
            first = next_rows.iloc[0]
            print(
                "first incomplete slot: "
                f"family={first['model_family']} regime={first['regime']} mode={first['mode']} "
                f"weather={first['weather_mode']} h={int(first['horizon_h'])} seed={int(first['seed'])}"
            )
    print("--- end xai resume diagnostics ---")


def _augment_prediction_helpers(
    predictions_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_kind: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
) -> pd.DataFrame:
    out = predictions_df.copy()
    out["target_kind"] = str(target_kind)
    out["mode"] = str(mode)
    out["weather_mode"] = str(weather_mode)
    out["horizon_h"] = int(horizon_h)
    out["row_idx"] = np.arange(len(out), dtype=int)
    out["hour"] = pd.to_datetime(out["datetime"]).dt.hour
    helper_cols = [
        "feat_outdoor_temp_c",
        "feat_heat_obs",
        "feat_temp_diff3h",
        "feat_temp_roll24h",
        "feat_space_deltaT_c",
        "feat_vent_deltaT_c",
        "feat_rh_pct",
        "feat_wind_ms",
    ]
    merge_cols = ["building", "datetime"]
    if all(col in test_df.columns for col in merge_cols):
        available_helper_cols = [col for col in helper_cols if col in test_df.columns]
        if available_helper_cols:
            helper_df = test_df.loc[:, merge_cols + available_helper_cols].copy()
            helper_df["datetime"] = pd.to_datetime(helper_df["datetime"])
            out = out.merge(helper_df, on=merge_cols, how="left")
    for helper_col in [
        "feat_outdoor_temp_c",
        "feat_heat_obs",
        "feat_temp_diff3h",
        "feat_temp_roll24h",
        "feat_space_deltaT_c",
        "feat_vent_deltaT_c",
        "feat_rh_pct",
        "feat_wind_ms",
    ]:
        if helper_col in test_df.columns and helper_col not in out.columns:
            out[helper_col] = test_df[helper_col].to_numpy()
    return out.reset_index(drop=True)


def fit_xgb_regime_experiments(
    config: mfc.ExperimentConfig,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    regime: str,
    buildings: tuple[str, ...],
    mode: str,
    weather_mode: str,
    horizon_h: int,
    target_kind: str,
    prepared_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> dict[str, FittedXGBExperiment]:
    if regime == "per_building":
        output = {}
        for building in buildings:
            experiment = fit_xgb_experiment(
                config,
                base_frames,
                building=building,
                mode=mode,
                weather_mode=weather_mode,
                horizon_h=int(horizon_h),
                target_kind=target_kind,
                prepared_cache=prepared_cache,
            )
            experiment.regime = "per_building"
            experiment.seed = int(config.random_seed)
            experiment.training_scope = str(building)
            experiment.model_family = "xgboost"
            output[str(building)] = experiment
        return output

    if str(target_kind).strip().lower() not in {"cum", "point"}:
        raise ValueError(f"Unsupported target kind for pooled XGBoost: {target_kind}")

    target_name = target_name_for(target_kind, int(horizon_h))
    feature_cols: list[str] | None = None
    pooled_train_frames: list[pd.DataFrame] = []
    pooled_fit_frames: list[pd.DataFrame] = []
    pooled_val_frames: list[pd.DataFrame] = []
    pooled_test_frames: list[pd.DataFrame] = []
    raw_frames: dict[str, pd.DataFrame] = {}

    for building in buildings:
        prepared = _prepare_xgb_building_case(
            config,
            base_frames,
            building=str(building),
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=int(horizon_h),
            target_kind=target_kind,
            prepared_cache=prepared_cache,
        )
        feat_here = list(prepared["feature_cols"])
        if feature_cols is None:
            feature_cols = feat_here
        train_df = prepared["train_df"]
        fit_df = prepared["fit_df"]
        val_df = prepared["val_df"]
        test_df = prepared["test_df"]
        raw_frames[str(building)] = prepared["frame"]
        if not fit_df.empty:
            pooled_fit_frames.append(fit_df)
        if not val_df.empty:
            pooled_val_frames.append(val_df)
        if not train_df.empty:
            pooled_train_frames.append(train_df)
        if not test_df.empty:
            pooled_test_frames.append(test_df)

    if feature_cols is None or not pooled_fit_frames or not pooled_test_frames:
        raise ValueError(
            f"Insufficient pooled XGBoost rows for mode={mode}, weather={weather_mode}, "
            f"h={int(horizon_h)}, target_kind={target_kind}"
        )

    fit_all = pd.concat(pooled_fit_frames, ignore_index=True)
    val_all = pd.concat(pooled_val_frames, ignore_index=True) if pooled_val_frames else pd.DataFrame(columns=fit_all.columns)
    train_all = pd.concat(pooled_train_frames, ignore_index=True)
    test_all = pd.concat(pooled_test_frames, ignore_index=True)
    model = mfc._fit_xgb_model(
        fit_all[feature_cols],
        fit_all[target_name],
        val_all[feature_cols],
        val_all[target_name],
        mfc.xgb_preset(config),
        config,
    )

    output: dict[str, FittedXGBExperiment] = {}
    best_iteration = int(
        getattr(model, "best_iteration", mfc.XGB_FIXED_PARAMS["n_estimators"] - 1)
        or mfc.XGB_FIXED_PARAMS["n_estimators"] - 1
    )
    for building in buildings:
        test_slice = test_all.loc[test_all["building"].astype(str) == str(building)].copy().reset_index(drop=True)
        if test_slice.empty:
            continue
        y_pred = mfc._predict_with_best_iteration(model, test_slice[feature_cols])
        y_true = test_slice[target_name].to_numpy(dtype=float)
        eval_mask = test_slice["is_heating_eval"].to_numpy(dtype=bool)
        eval_true, eval_pred = _eval_arrays(y_true, y_pred, eval_mask)
        metrics = mfc.compute_regression_metrics(eval_true, eval_pred)
        predictions_df = test_slice.loc[:, ["building", "datetime", "is_heating_eval", target_name]].copy()
        predictions_df.rename(columns={target_name: "y_true"}, inplace=True)
        predictions_df["y_pred"] = y_pred
        predictions_df["abs_error"] = np.abs(predictions_df["y_true"] - predictions_df["y_pred"])
        predictions_df = _augment_prediction_helpers(
            predictions_df,
            test_slice,
            target_kind=target_kind,
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=int(horizon_h),
        )
        output[str(building)] = FittedXGBExperiment(
            building=str(building),
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=int(horizon_h),
            target_kind=str(target_kind),
            target_name=target_name,
            feature_cols=list(feature_cols),
            raw_frame=raw_frames[str(building)],
            train_df=train_all.copy(),
            fit_df=fit_all.copy(),
            validation_df=val_all.copy(),
            test_df=test_slice.reset_index(drop=True),
            predictions_df=predictions_df,
            metrics=metrics,
            model=model,
            model_summary={
                "n_train_rows": int(len(fit_all)),
                "n_val_rows": int(len(val_all)),
                "n_test_rows": int(len(test_slice)),
                "best_iteration": best_iteration,
                "feature_cols": list(feature_cols),
                "training_buildings": list(buildings),
            },
            regime="pooled_same_buildings",
            seed=int(config.random_seed),
            training_scope="POOLED",
            model_family="xgboost",
        )
    return output


def _prepare_lstm_building_inputs(
    config: mfc.ExperimentConfig,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    target_kind: str,
    static_scaler: Any | None,
) -> dict[str, Any]:
    set_name = feature_set_for_mode(mode)
    frame = ensure_target_column(base_frames[building][set_name].copy(), target_kind, int(horizon_h))
    frame, fw_cols = mfc.apply_weather_mode(frame, weather_mode, int(horizon_h))
    target_name = target_name_for(target_kind, int(horizon_h))
    dynamic_cols = mode_temporal_features(mode) + list(fw_cols)
    static_cols = mode_static_features(mode)
    split_spec = mfc.build_split_spec(frame, int(horizon_h), config)
    frame_scaled, feature_scaler, target_scaler = mfc.scale_frame_for_lstm(
        frame=frame,
        dynamic_cols=dynamic_cols,
        target_name=target_name,
        feature_train_mask=split_spec.feature_train_mask,
        train_issue_mask=split_spec.train_issue_mask,
    )
    X_train_full, y_train_full, train_meta_full = mfc.build_sequences(
        frame_scaled,
        dynamic_cols,
        f"{target_name}_scaled",
        split_spec.train_issue_mask,
        config.lookback_hours,
    )
    X_test, y_test, test_meta = mfc.build_sequences(
        frame_scaled,
        dynamic_cols,
        f"{target_name}_scaled",
        split_spec.test_issue_mask,
        config.lookback_hours,
    )
    X_fit, y_fit, fit_meta, X_val, y_val, val_meta = mfc.make_internal_fit_split(
        X_train_full,
        y_train_full,
        train_meta_full,
        config.validation_fraction,
    )
    static_vector = None
    if mode == "M4":
        if static_scaler is None:
            raise ValueError("M4 pooled LSTM preparation requires a fitted static scaler.")
        static_vector = mfc.build_static_vector(frame, static_scaler, static_cols)
    return {
        "raw_frame": frame,
        "frame_scaled": frame_scaled,
        "test_feature_frame": frame.merge(
            test_meta.loc[:, ["building", "datetime"]],
            on=["building", "datetime"],
            how="right",
        ),
        "target_name": target_name,
        "dynamic_cols": dynamic_cols,
        "static_cols": static_cols,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "X_fit": X_fit,
        "y_fit": y_fit,
        "fit_meta": fit_meta,
        "X_val": X_val,
        "y_val": y_val,
        "val_meta": val_meta,
        "X_test": X_test,
        "y_test": y_test,
        "test_meta": test_meta,
        "static_vector": static_vector,
    }


def fit_lstm_regime_experiments(
    config: mfc.ExperimentConfig,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    regime: str,
    buildings: tuple[str, ...],
    mode: str,
    weather_mode: str,
    horizon_h: int,
    target_kind: str,
    static_scaler: Any | None = None,
) -> dict[str, FittedLSTMExperiment]:
    if str(target_kind).strip().lower() != "cum":
        raise NotImplementedError("The broad LSTM XAI matrix supports cumulative targets only.")
    if regime == "per_building":
        output = {}
        for building in buildings:
            experiment = fit_lstm_experiment(
                config,
                base_frames,
                building=building,
                mode=mode,
                weather_mode=weather_mode,
                horizon_h=int(horizon_h),
                target_kind=target_kind,
                static_scaler=static_scaler,
            )
            experiment.regime = "per_building"
            experiment.seed = int(config.random_seed)
            experiment.training_scope = str(building)
            output[str(building)] = experiment
        return output

    mfc.require_tensorflow()
    mfc.set_all_seeds(config.random_seed, deterministic_ops=config.deterministic_ops)
    mfc.tf.keras.backend.clear_session()
    spec = mfc.architecture_spec(config)

    prepared_by_building = {
        str(building): _prepare_lstm_building_inputs(
            config,
            base_frames,
            building=str(building),
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=int(horizon_h),
            target_kind=target_kind,
            static_scaler=static_scaler,
        )
        for building in buildings
    }
    sample_prepared = next(iter(prepared_by_building.values()))
    dynamic_cols = list(sample_prepared["dynamic_cols"])
    static_cols = list(sample_prepared["static_cols"])
    feature_cols = dynamic_cols + static_cols

    fit_dyn, fit_y, fit_meta = [], [], []
    val_dyn, val_y, val_meta = [], [], []
    test_dyn, test_y, test_meta = [], [], []
    fit_static, val_static, test_static = [], [], []
    for building in buildings:
        prepared = prepared_by_building[str(building)]
        if prepared["X_fit"].shape[0] > 0:
            fit_dyn.append(prepared["X_fit"])
            fit_y.append(prepared["y_fit"])
            fit_meta.append(prepared["fit_meta"])
            if mode == "M4":
                fit_static.append(np.repeat(prepared["static_vector"], repeats=prepared["X_fit"].shape[0], axis=0))
        if prepared["X_val"].shape[0] > 0:
            val_dyn.append(prepared["X_val"])
            val_y.append(prepared["y_val"])
            val_meta.append(prepared["val_meta"])
            if mode == "M4":
                val_static.append(np.repeat(prepared["static_vector"], repeats=prepared["X_val"].shape[0], axis=0))
        if prepared["X_test"].shape[0] > 0:
            test_dyn.append(prepared["X_test"])
            test_y.append(prepared["y_test"])
            test_meta.append(prepared["test_meta"])
            if mode == "M4":
                test_static.append(np.repeat(prepared["static_vector"], repeats=prepared["X_test"].shape[0], axis=0))

    if not fit_dyn or not test_dyn:
        raise ValueError(
            f"Insufficient pooled LSTM sequences for mode={mode}, weather={weather_mode}, "
            f"h={int(horizon_h)}, target_kind={target_kind}"
        )

    X_fit_all = np.concatenate(fit_dyn, axis=0)
    y_fit_all = np.concatenate(fit_y, axis=0)
    fit_meta_all = pd.concat(fit_meta, ignore_index=True)
    X_val_all = np.concatenate(val_dyn, axis=0) if val_dyn else np.empty((0,) + X_fit_all.shape[1:], dtype=X_fit_all.dtype)
    y_val_all = np.concatenate(val_y, axis=0) if val_y else np.empty((0,), dtype=y_fit_all.dtype)
    val_meta_all = pd.concat(val_meta, ignore_index=True) if val_meta else pd.DataFrame(columns=fit_meta_all.columns)

    if mode == "M4":
        X_fit_static_all = np.concatenate(fit_static, axis=0)
        X_val_static_all = np.concatenate(val_static, axis=0) if val_static else np.empty((0, X_fit_static_all.shape[1]), dtype=X_fit_static_all.dtype)
        model = mfc.build_lstm_temporal_plus_static_from_spec(
            spec["lookback_hours"],
            X_fit_all.shape[-1],
            X_fit_static_all.shape[-1],
            spec,
            config.learning_rate,
        )
    else:
        X_fit_static_all = None
        X_val_static_all = None
        model = mfc.build_lstm_temporal_only_from_spec(
            spec["lookback_hours"],
            X_fit_all.shape[-1],
            spec,
            config.learning_rate,
        )

    lr_callback = mfc.LearningRateHistory()
    callbacks: list[Any] = [lr_callback]
    validation_data = None
    if X_val_all.shape[0] > 0:
        callbacks += [
            mfc.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                restore_best_weights=True,
            ),
            mfc.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                verbose=0,
            ),
        ]
        validation_data = ([X_val_all, X_val_static_all], y_val_all) if mode == "M4" else (X_val_all, y_val_all)

    if mode == "M4":
        model.fit(
            [X_fit_all, X_fit_static_all],
            y_fit_all,
            validation_data=validation_data,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0,
            callbacks=callbacks,
        )
    else:
        model.fit(
            X_fit_all,
            y_fit_all,
            validation_data=validation_data,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0,
            callbacks=callbacks,
        )

    output: dict[str, FittedLSTMExperiment] = {}
    for building in buildings:
        prepared = prepared_by_building[str(building)]
        X_test = np.asarray(prepared["X_test"], dtype="float32")
        if X_test.shape[0] == 0:
            continue
        X_test_static = None
        if mode == "M4":
            X_test_static = np.repeat(prepared["static_vector"], repeats=X_test.shape[0], axis=0)
        predictions_template = pd.DataFrame()
        temp_experiment = FittedLSTMExperiment(
            building=str(building),
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=int(horizon_h),
            target_kind=str(target_kind),
            target_name=prepared["target_name"],
            feature_cols=list(feature_cols),
            dynamic_feature_cols=dynamic_cols,
            static_feature_cols=static_cols,
            raw_frame=prepared["raw_frame"],
            frame_scaled=prepared["frame_scaled"],
            fit_meta=fit_meta_all.reset_index(drop=True),
            validation_meta=val_meta_all.reset_index(drop=True),
            test_meta=prepared["test_meta"].reset_index(drop=True),
            predictions_df=predictions_template,
            metrics={},
            model=model,
            model_summary={
                "n_train_rows": int(X_fit_all.shape[0]),
                "n_val_rows": int(X_val_all.shape[0]),
                "n_test_rows": int(X_test.shape[0]),
                "lookback_hours": int(spec["lookback_hours"]),
                "feature_cols": list(feature_cols),
                "feature_scaler": prepared["feature_scaler"],
                "training_buildings": list(buildings),
            },
            X_fit_dynamic=X_fit_all,
            X_validation_dynamic=X_val_all,
            X_test_dynamic=X_test,
            X_fit_static=X_fit_static_all,
            X_validation_static=X_val_static_all,
            X_test_static=X_test_static,
            y_fit_scaled=np.asarray(y_fit_all, dtype=float),
            y_validation_scaled=np.asarray(y_val_all, dtype=float),
            y_test_scaled=np.asarray(prepared["y_test"], dtype=float),
            target_scaler=prepared["target_scaler"],
            regime="pooled_same_buildings",
            seed=int(config.random_seed),
            training_scope="POOLED",
            model_family="lstm",
        )
        y_true = _inverse_single_target(prepared["y_test"], prepared["target_scaler"])
        y_pred = _predict_lstm_original_scale(temp_experiment, X_test, X_test_static)
        eval_mask = prepared["test_meta"]["is_heating_eval"].to_numpy(dtype=bool)
        eval_true, eval_pred = _eval_arrays(y_true, y_pred, eval_mask)
        metrics = mfc.compute_regression_metrics(eval_true, eval_pred)

        predictions_df = prepared["test_meta"].copy()
        predictions_df["y_true"] = y_true
        predictions_df["y_pred"] = y_pred
        predictions_df["abs_error"] = np.abs(predictions_df["y_true"] - predictions_df["y_pred"])
        predictions_df = _augment_prediction_helpers(
            predictions_df,
            prepared["test_feature_frame"],
            target_kind=target_kind,
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=int(horizon_h),
        )
        temp_experiment.predictions_df = predictions_df.reset_index(drop=True)
        temp_experiment.metrics = metrics
        output[str(building)] = temp_experiment
    return output


def run_broad_xai_matrix(
    config: mfc.ExperimentConfig,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    manifest_df: pd.DataFrame | None = None,
    regimes: tuple[str, ...],
    buildings: tuple[str, ...],
    modes: tuple[str, ...],
    weather_modes: tuple[str, ...],
    horizons_by_target_kind: dict[str, tuple[int, ...]],
    target_kind: str,
    seed_list: tuple[int, ...],
    model_families: tuple[str, ...] = ("xgboost", "lstm"),
    pfi_repeats_by_family: dict[str, int] | None = None,
    shap_background_size: int = 128,
    shap_explain_size: int = 256,
    lstm_pfi_max_rows: int | None = None,
    detail_requests: set[tuple[str, str, str, str, str, int, int]] | None = None,
    save_artifacts: bool = False,
    resume_existing: bool = False,
    save_after_each_slot: bool = False,
    save_every_n_slots: int | None = None,
    continue_on_error: bool = False,
    verbose: bool = False,
    require_fine_groups_for_resume: bool = False,
) -> dict[str, Any]:
    horizons = normalize_horizons_for_target_kind(horizons_by_target_kind[target_kind], target_kind)
    manifest_scope_df = (
        _normalize_xai_manifest_frame(manifest_df)
        if manifest_df is not None
        else build_xai_matrix_manifest(
            model_families=model_families,
            regimes=regimes,
            buildings=buildings,
            modes=modes,
            weather_modes=weather_modes,
            horizons=horizons,
            target_kind=target_kind,
            seed_list=seed_list,
        )
    )
    paths = build_xai_artifact_paths(config.results_dir)
    outputs = (
        load_saved_xai_outputs(config.results_dir, manifest_df=manifest_scope_df)
        if resume_existing
        else _normalize_xai_outputs(
            {
                "manifest_df": manifest_scope_df.copy(),
                "seed_metrics_df": pd.DataFrame(),
                "seed_grouped_pfi_df": pd.DataFrame(),
                "seed_grouped_pfi_fine_df": pd.DataFrame(),
                "seed_grouped_shap_df": pd.DataFrame(),
                "seed_grouped_shap_fine_df": pd.DataFrame(),
                "seed_agreement_df": pd.DataFrame(),
                "seed_agreement_fine_df": pd.DataFrame(),
                "run_log_df": pd.DataFrame(),
                "detail_cache": {},
            }
        )
    )
    outputs["manifest_df"] = manifest_scope_df.copy()
    manifest_buildings = (
        tuple(sorted(manifest_scope_df["building"].dropna().astype(str).unique().tolist()))
        if not manifest_scope_df.empty
        else buildings
    )
    completed_slots = _completed_xai_combo_slots(
        outputs["seed_metrics_df"],
        buildings=manifest_buildings,
        manifest_df=manifest_scope_df,
        seed_grouped_pfi_fine_df=outputs.get("seed_grouped_pfi_fine_df", pd.DataFrame()),
        seed_grouped_shap_fine_df=outputs.get("seed_grouped_shap_fine_df", pd.DataFrame()),
        require_fine_groups=require_fine_groups_for_resume,
    )
    combo_cols = [col for col in MATRIX_KEY_COLS if col != "building"] + ["seed"]
    total_slots = int(len(manifest_scope_df[combo_cols].drop_duplicates()))

    static_scaler = None
    if "lstm" in model_families and "M4" in modes:
        static_scaler = mfc.fit_static_scaler(
            clone_config_with_overrides(config, buildings=manifest_buildings),
            base_frames,
        )

    slot_index = 0
    dirty_slots = 0
    checkpoint_interval = max(0, int(save_every_n_slots or 0))
    combo_plan_df = manifest_scope_df[combo_cols].drop_duplicates().sort_values(combo_cols).reset_index(drop=True)
    for combo in combo_plan_df.itertuples(index=False):
        model_family = str(combo.model_family)
        regime = str(combo.regime)
        mode = str(combo.mode)
        weather_mode = str(combo.weather_mode)
        horizon_h = int(combo.horizon_h)
        combo_target_kind = str(combo.target_kind)
        seed = int(combo.seed)
        slot_index += 1
        combo_slot = (
            model_family,
            regime,
            mode,
            weather_mode,
            horizon_h,
            combo_target_kind,
            seed,
        )
        combo_buildings = tuple(
            sorted(
                manifest_scope_df.loc[
                    (manifest_scope_df["model_family"].astype(str) == model_family)
                    & (manifest_scope_df["regime"].astype(str) == regime)
                    & (manifest_scope_df["mode"].astype(str) == mode)
                    & (manifest_scope_df["weather_mode"].astype(str) == weather_mode)
                    & (pd.to_numeric(manifest_scope_df["horizon_h"], errors="coerce") == horizon_h)
                    & (manifest_scope_df["target_kind"].astype(str) == combo_target_kind)
                    & (pd.to_numeric(manifest_scope_df["seed"], errors="coerce") == seed),
                    "building",
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )
        family_pfi_repeats = int((pfi_repeats_by_family or {}).get(model_family, 10))
        if combo_slot in completed_slots:
            if verbose:
                print(
                    f"[{slot_index:>3}/{total_slots}] family={model_family} regime={regime} "
                    f"mode={mode} weather={weather_mode} h={horizon_h} seed={seed} | resume-skip"
                )
            continue
        if verbose:
            print(
                f"[{slot_index:>3}/{total_slots}] family={model_family} regime={regime} "
                f"mode={mode} weather={weather_mode} h={horizon_h} seed={seed}"
            )
        seed_config = clone_config_with_overrides(
            config,
            buildings=combo_buildings,
            horizons=(horizon_h,),
            regimes=(regime,),
            weather_modes=(weather_mode,),
            modes=(mode,),
            model_families=(model_family,),
            random_seed=seed,
        )
        xgb_prepared_cache = {} if model_family == "xgboost" else None
        slot_start = perf_counter()
        try:
            if model_family == "xgboost":
                experiment_map = fit_xgb_regime_experiments(
                    seed_config,
                    base_frames,
                    regime=regime,
                    buildings=combo_buildings,
                    mode=mode,
                    weather_mode=weather_mode,
                    horizon_h=horizon_h,
                    target_kind=combo_target_kind,
                    prepared_cache=xgb_prepared_cache,
                )
            elif model_family == "lstm":
                experiment_map = fit_lstm_regime_experiments(
                    seed_config,
                    base_frames,
                    regime=regime,
                    buildings=combo_buildings,
                    mode=mode,
                    weather_mode=weather_mode,
                    horizon_h=horizon_h,
                    target_kind=combo_target_kind,
                    static_scaler=static_scaler,
                )
            else:
                raise ValueError(f"Unsupported model family: {model_family}")

            metric_rows: list[dict[str, Any]] = []
            pfi_frames: list[pd.DataFrame] = []
            pfi_frames_fine: list[pd.DataFrame] = []
            shap_frames: list[pd.DataFrame] = []
            shap_frames_fine: list[pd.DataFrame] = []
            agreement_frames: list[pd.DataFrame] = []
            agreement_frames_fine: list[pd.DataFrame] = []
            run_log_rows: list[dict[str, Any]] = []

            for building, experiment in experiment_map.items():
                feature_groups_fine = make_feature_groups(experiment.feature_cols, taxonomy="fine")
                feature_groups = make_feature_groups(experiment.feature_cols, taxonomy="broad")
                if model_family == "xgboost":
                    pfi_df_fine = grouped_permutation_importance(
                        experiment.model,
                        experiment.test_df[experiment.feature_cols],
                        experiment.test_df[experiment.target_name].to_numpy(dtype=float),
                        feature_groups_fine,
                        eval_mask=experiment.test_df["is_heating_eval"].to_numpy(dtype=bool),
                        n_repeats=family_pfi_repeats,
                        random_seed=seed,
                    )
                    shap_payload = compute_tree_shap(experiment)
                    shap_group_df_fine = summarize_grouped_shap(
                        shap_payload["shap_values"],
                        shap_payload["feature_names"],
                        feature_groups_fine,
                    )
                else:
                    pfi_df_fine = grouped_permutation_importance_lstm(
                        experiment,
                        feature_groups_fine,
                        n_repeats=family_pfi_repeats,
                        random_seed=seed,
                        max_rows=lstm_pfi_max_rows,
                    )
                    shap_payload = compute_lstm_gradient_shap(
                        experiment,
                        n_background=shap_background_size,
                        n_explain=shap_explain_size,
                        random_seed=seed,
                    )
                    shap_group_df_fine = summarize_grouped_lstm_shap(
                        shap_payload["shap_values"],
                        shap_payload["feature_names"],
                        feature_groups_fine,
                        static_shap_values=shap_payload.get("static_shap_values"),
                        static_feature_names=shap_payload.get("static_feature_names"),
                    )

                pfi_df = rollup_seed_grouped_pfi_to_broad(pfi_df_fine)
                shap_group_df = rollup_seed_grouped_shap_to_broad(shap_group_df_fine)
                agreement_df_fine = build_pfi_shap_agreement_table(pfi_df_fine, shap_group_df_fine)
                agreement_df = rollup_seed_agreement_to_broad(agreement_df_fine)
                meta = {
                    "model_family": model_family,
                    "regime": regime,
                    "building": str(building),
                    "mode": mode,
                    "weather_mode": weather_mode,
                    "horizon_h": horizon_h,
                    "target_kind": combo_target_kind,
                    "seed": seed,
                    "training_scope": experiment.training_scope or ("POOLED" if regime == "pooled_same_buildings" else str(building)),
                }

                metric_row = {
                    **meta,
                    "rmse": float(experiment.metrics["rmse"]),
                    "mae": float(experiment.metrics["mae"]),
                    "r2": float(experiment.metrics["r2"]),
                    "wape_pct": float(experiment.metrics["wape_pct"]),
                    "n_eval_rows": int(experiment.predictions_df["is_heating_eval"].astype(bool).sum()),
                    "n_test_rows": int(len(experiment.predictions_df)),
                    "n_features": int(len(experiment.feature_cols)),
                }
                if model_family == "xgboost":
                    metric_row["best_iteration"] = int(experiment.model_summary.get("best_iteration", np.nan))
                else:
                    metric_row["best_epoch"] = float(experiment.model_summary.get("best_epoch", np.nan))
                metric_rows.append(metric_row)

                for frame in (pfi_df, pfi_df_fine, shap_group_df, shap_group_df_fine, agreement_df, agreement_df_fine):
                    if frame.empty:
                        continue
                    for col_name, value in meta.items():
                        frame[col_name] = value
                if not pfi_df.empty:
                    pfi_frames.append(pfi_df)
                if not pfi_df_fine.empty:
                    pfi_frames_fine.append(pfi_df_fine)
                if not shap_group_df.empty:
                    shap_frames.append(shap_group_df)
                if not shap_group_df_fine.empty:
                    shap_frames_fine.append(shap_group_df_fine)
                if not agreement_df.empty:
                    agreement_frames.append(agreement_df)
                if not agreement_df_fine.empty:
                    agreement_frames_fine.append(agreement_df_fine)

                detail_key = (
                    model_family,
                    regime,
                    str(building),
                    mode,
                    weather_mode,
                    horizon_h,
                    seed,
                )
                if detail_requests is not None and detail_key in detail_requests:
                    outputs["detail_cache"][detail_key] = {
                        "experiment": experiment,
                        "feature_groups": feature_groups,
                        "feature_groups_fine": feature_groups_fine,
                        "shap": shap_payload,
                    }

                run_log_rows.append(
                    {
                        **meta,
                        "slot_index": int(slot_index),
                        "status": "ok",
                        "elapsed_s": float(perf_counter() - slot_start),
                        "error_type": "",
                        "error_message": "",
                    }
                )

            if metric_rows:
                outputs["seed_metrics_df"] = pd.concat(
                    [outputs["seed_metrics_df"], pd.DataFrame(metric_rows)],
                    ignore_index=True,
                )
            if pfi_frames:
                outputs["seed_grouped_pfi_df"] = pd.concat(
                    [outputs["seed_grouped_pfi_df"], *pfi_frames],
                    ignore_index=True,
                )
            if pfi_frames_fine:
                outputs["seed_grouped_pfi_fine_df"] = pd.concat(
                    [outputs["seed_grouped_pfi_fine_df"], *pfi_frames_fine],
                    ignore_index=True,
                )
            if shap_frames:
                outputs["seed_grouped_shap_df"] = pd.concat(
                    [outputs["seed_grouped_shap_df"], *shap_frames],
                    ignore_index=True,
                )
            if shap_frames_fine:
                outputs["seed_grouped_shap_fine_df"] = pd.concat(
                    [outputs["seed_grouped_shap_fine_df"], *shap_frames_fine],
                    ignore_index=True,
                )
            if agreement_frames:
                outputs["seed_agreement_df"] = pd.concat(
                    [outputs["seed_agreement_df"], *agreement_frames],
                    ignore_index=True,
                )
            if agreement_frames_fine:
                outputs["seed_agreement_fine_df"] = pd.concat(
                    [outputs["seed_agreement_fine_df"], *agreement_frames_fine],
                    ignore_index=True,
                )
            if run_log_rows:
                outputs["run_log_df"] = pd.concat(
                    [outputs["run_log_df"], pd.DataFrame(run_log_rows)],
                    ignore_index=True,
                )
            outputs = _normalize_xai_outputs(outputs)
            completed_slots = _completed_xai_combo_slots(
                outputs["seed_metrics_df"],
                buildings=manifest_buildings,
                manifest_df=manifest_scope_df,
                seed_grouped_pfi_fine_df=outputs.get("seed_grouped_pfi_fine_df", pd.DataFrame()),
                seed_grouped_shap_fine_df=outputs.get("seed_grouped_shap_fine_df", pd.DataFrame()),
                require_fine_groups=require_fine_groups_for_resume,
            )
            dirty_slots += 1
        except Exception as exc:
            error_log_rows = []
            for building in combo_buildings:
                error_log_rows.append(
                    {
                        "slot_index": int(slot_index),
                        "model_family": model_family,
                        "regime": regime,
                        "building": str(building),
                        "mode": mode,
                        "weather_mode": weather_mode,
                        "horizon_h": horizon_h,
                        "target_kind": combo_target_kind,
                        "seed": seed,
                        "training_scope": "POOLED" if regime == "pooled_same_buildings" else str(building),
                        "status": "error",
                        "elapsed_s": float(perf_counter() - slot_start),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            outputs["run_log_df"] = pd.concat(
                [outputs["run_log_df"], pd.DataFrame(error_log_rows)],
                ignore_index=True,
            )
            outputs = _normalize_xai_outputs(outputs)
            completed_slots.discard(combo_slot)
            dirty_slots += 1
            if save_artifacts and (
                save_after_each_slot
                or (checkpoint_interval > 0 and dirty_slots >= checkpoint_interval)
                or not continue_on_error
            ):
                _write_xai_outputs(paths, outputs)
                dirty_slots = 0
            if verbose:
                print(
                    f"    slot failed | family={model_family} regime={regime} "
                    f"mode={mode} weather={weather_mode} h={horizon_h} seed={seed} | "
                    f"{type(exc).__name__}: {exc}"
                )
            if not continue_on_error:
                raise
            continue

        if save_artifacts and (
            save_after_each_slot
            or (checkpoint_interval > 0 and dirty_slots >= checkpoint_interval)
        ):
            _write_xai_outputs(paths, outputs)
            dirty_slots = 0

    outputs = _normalize_xai_outputs(outputs)
    outputs["manifest_df"] = manifest_scope_df.copy()
    if save_artifacts and (dirty_slots > 0 or not paths["seed_metrics"].exists()):
        _write_xai_outputs(paths, outputs)
    return outputs


def aggregate_seed_metrics(seed_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if seed_metrics_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = MATRIX_KEY_COLS + ["training_scope"]
    for keys, group_df in seed_metrics_df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        row["seed_count"] = int(group_df["seed"].nunique())
        for col in ["rmse", "mae", "r2", "wape_pct", "n_eval_rows", "n_test_rows", "n_features"]:
            if col in group_df.columns:
                row[col] = float(group_df[col].mean())
                row[f"{col}_seed_std"] = float(group_df[col].std(ddof=0))
        if "best_iteration" in group_df.columns:
            row["best_iteration"] = float(group_df["best_iteration"].mean())
        if "best_epoch" in group_df.columns:
            row["best_epoch"] = float(group_df["best_epoch"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def aggregate_seed_grouped_pfi(seed_pfi_df: pd.DataFrame) -> pd.DataFrame:
    if seed_pfi_df.empty:
        return pd.DataFrame()
    group_cols = MATRIX_KEY_COLS + ["training_scope", "feature_group", "n_features"]
    rows = []
    for keys, group_df in seed_pfi_df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        row["seed_count"] = int(group_df["seed"].nunique())
        for col in [
            "baseline_rmse",
            "baseline_wape_pct",
            "baseline_mae",
            "delta_rmse_mean",
            "delta_rmse_std",
            "delta_wape_mean",
            "delta_wape_std",
            "delta_mae_mean",
            "delta_mae_std",
        ]:
            row[col] = float(group_df[col].mean())
            row[f"{col}_seed_std"] = float(group_df[col].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(MATRIX_KEY_COLS + ["feature_group"]).reset_index(drop=True)


def aggregate_seed_grouped_shap(seed_shap_df: pd.DataFrame) -> pd.DataFrame:
    if seed_shap_df.empty:
        return pd.DataFrame()
    group_cols = MATRIX_KEY_COLS + ["training_scope", "feature_group", "n_features"]
    rows = []
    for keys, group_df in seed_shap_df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        row["seed_count"] = int(group_df["seed"].nunique())
        for col in [
            "mean_abs_group_shap",
            "mean_signed_group_shap",
            "median_abs_group_shap",
        ]:
            row[col] = float(group_df[col].mean())
            row[f"{col}_seed_std"] = float(group_df[col].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(MATRIX_KEY_COLS + ["feature_group"]).reset_index(drop=True)


def aggregate_seed_agreement(seed_agreement_df: pd.DataFrame) -> pd.DataFrame:
    if seed_agreement_df.empty:
        return pd.DataFrame()
    group_cols = MATRIX_KEY_COLS + ["training_scope", "feature_group", "n_features"]
    rows = []
    for keys, group_df in seed_agreement_df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        row["seed_count"] = int(group_df["seed"].nunique())
        for col in [
            "baseline_rmse",
            "baseline_wape_pct",
            "baseline_mae",
            "delta_rmse_mean",
            "delta_wape_mean",
            "delta_mae_mean",
            "pfi_rank_wape",
            "mean_abs_group_shap",
            "mean_signed_group_shap",
            "median_abs_group_shap",
            "shap_rank",
            "rank_gap",
        ]:
            row[col] = float(group_df[col].mean())
            row[f"{col}_seed_std"] = float(group_df[col].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(MATRIX_KEY_COLS + ["feature_group"]).reset_index(drop=True)


def _ordered_mode_label(column_name: str) -> str:
    return column_name.replace("_wape_pct_pp", "").replace("_minus_", " - ")


def build_xai_group_share_summary(
    grouped_pfi_df: pd.DataFrame,
    grouped_shap_df: pd.DataFrame,
    *,
    pfi_value_col: str = "delta_rmse_mean",
    shap_value_col: str = "mean_abs_group_shap",
    taxonomy: str = "broad",
) -> pd.DataFrame:
    key_cols = MATRIX_KEY_COLS + ["training_scope"]
    frames: list[pd.DataFrame] = []
    method_specs = (
        ("pfi", grouped_pfi_df, pfi_value_col),
        ("shap", grouped_shap_df, shap_value_col),
    )
    for method, df, value_col in method_specs:
        if df.empty or value_col not in df.columns:
            continue
        work_df = df.copy()
        work_df[value_col] = pd.to_numeric(work_df[value_col], errors="coerce")
        work_df["share_basis_value"] = work_df[value_col].clip(lower=0.0)
        total_basis = work_df.groupby(key_cols)["share_basis_value"].transform("sum")
        work_df["share"] = np.where(total_basis > 0, work_df["share_basis_value"] / total_basis, np.nan)
        work_df["method"] = method
        work_df["value_col"] = value_col
        work_df["value_mean"] = work_df[value_col]
        frames.append(
            work_df.loc[
                :,
                key_cols
                + [
                    "feature_group",
                    "n_features",
                    "method",
                    "value_col",
                    "value_mean",
                    "share_basis_value",
                    "share",
                ],
            ]
        )
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    feature_order = {name: idx for idx, name in enumerate(feature_group_order_for_taxonomy(taxonomy))}
    out["_feature_sort"] = out["feature_group"].map(feature_order).fillna(len(feature_order))
    return (
        out.sort_values(MATRIX_KEY_COLS + ["training_scope", "method", "_feature_sort", "feature_group"])
        .drop(columns="_feature_sort")
        .reset_index(drop=True)
    )


def summarize_group_shares_for_thesis(
    group_share_df: pd.DataFrame,
    *,
    taxonomy: str = "broad",
) -> pd.DataFrame:
    if group_share_df.empty:
        return pd.DataFrame()
    group_cols = [
        "model_family",
        "regime",
        "weather_mode",
        "mode",
        "horizon_h",
        "target_kind",
        "method",
        "feature_group",
    ]
    summary_df = (
        group_share_df.groupby(group_cols, as_index=False)
        .agg(
            mean_share=("share", "mean"),
            mean_value=("value_mean", "mean"),
            share_basis_mean=("share_basis_value", "mean"),
            n_buildings=("building", "nunique"),
            n_rows=("building", "size"),
        )
        .reset_index(drop=True)
    )
    feature_order = {name: idx for idx, name in enumerate(feature_group_order_for_taxonomy(taxonomy))}
    mode_order = {name: idx for idx, name in enumerate(THESIS_MODE_GRAMMAR)}
    weather_order = {name: idx for idx, name in enumerate(THESIS_WEATHER_MODE_ORDER)}
    summary_df["_mode_sort"] = summary_df["mode"].map(mode_order).fillna(len(mode_order))
    summary_df["_weather_sort"] = summary_df["weather_mode"].map(weather_order).fillna(len(weather_order))
    summary_df["_feature_sort"] = summary_df["feature_group"].map(feature_order).fillna(len(feature_order))
    return (
        summary_df.sort_values(
            ["model_family", "regime", "_weather_sort", "_mode_sort", "horizon_h", "method", "_feature_sort", "feature_group"]
        )
        .drop(columns=["_mode_sort", "_weather_sort", "_feature_sort"])
        .reset_index(drop=True)
    )


def build_xai_mode_completion_table(
    manifest_df: pd.DataFrame,
    seed_metrics_df: pd.DataFrame,
    *,
    requested_modes: tuple[str, ...] = THESIS_MODE_GRAMMAR,
) -> pd.DataFrame:
    metric_keys = MATRIX_KEY_COLS + ["seed", "training_scope"]
    manifest_counts: dict[str, int] = {}
    completed_counts: dict[str, int] = {}
    if not manifest_df.empty and all(col in manifest_df.columns for col in metric_keys):
        manifest_counts = (
            manifest_df[metric_keys]
            .drop_duplicates()
            .groupby("mode")
            .size()
            .astype(int)
            .to_dict()
        )
    if not seed_metrics_df.empty and all(col in seed_metrics_df.columns for col in metric_keys):
        completed_counts = (
            seed_metrics_df[metric_keys]
            .drop_duplicates()
            .groupby("mode")
            .size()
            .astype(int)
            .to_dict()
        )
    reading_map = build_thesis_mode_definition_table().set_index("mode")["thesis_reading"].to_dict()
    rows: list[dict[str, Any]] = []
    for mode in requested_modes:
        total_slots = int(manifest_counts.get(mode, 0))
        completed_slots = int(completed_counts.get(mode, 0))
        if total_slots <= 0:
            status = "pending"
            detail = "This mode is part of the thesis grammar but is not present in the current manifest yet."
            coverage_pct = np.nan
        elif completed_slots <= 0:
            status = "not_started"
            detail = "This mode is present in the current manifest but has no saved XAI metric slots yet."
            coverage_pct = 0.0
        elif completed_slots < total_slots:
            status = "partial"
            detail = f"{completed_slots}/{total_slots} manifest slots currently have saved XAI metrics."
            coverage_pct = completed_slots / total_slots
        else:
            status = "complete"
            detail = f"{completed_slots}/{total_slots} manifest slots currently have saved XAI metrics."
            coverage_pct = 1.0
        rows.append(
            {
                "mode": mode,
                "status": status,
                "completed_slots": completed_slots,
                "total_slots": total_slots,
                "coverage_pct": coverage_pct,
                "thesis_reading": reading_map.get(mode, ""),
                "detail": detail,
            }
        )
    mode_order = {name: idx for idx, name in enumerate(requested_modes)}
    return (
        pd.DataFrame(rows)
        .assign(_mode_sort=lambda df: df["mode"].map(mode_order).fillna(len(mode_order)))
        .sort_values("_mode_sort")
        .drop(columns="_mode_sort")
        .reset_index(drop=True)
    )


def build_xai_mode_weather_completion_table(
    manifest_df: pd.DataFrame,
    seed_metrics_df: pd.DataFrame,
    *,
    requested_modes: tuple[str, ...] = THESIS_MODE_GRAMMAR,
    weather_modes: tuple[str, ...] = THESIS_WEATHER_MODE_ORDER,
) -> pd.DataFrame:
    metric_keys = MATRIX_KEY_COLS + ["seed", "training_scope"]
    manifest_counts: dict[tuple[str, str, str], int] = {}
    completed_counts: dict[tuple[str, str, str], int] = {}
    if not manifest_df.empty and all(col in manifest_df.columns for col in metric_keys):
        manifest_counts = (
            manifest_df[metric_keys]
            .drop_duplicates()
            .groupby(["regime", "weather_mode", "mode"])
            .size()
            .astype(int)
            .to_dict()
        )
    if not seed_metrics_df.empty and all(col in seed_metrics_df.columns for col in metric_keys):
        completed_counts = (
            seed_metrics_df[metric_keys]
            .drop_duplicates()
            .groupby(["regime", "weather_mode", "mode"])
            .size()
            .astype(int)
            .to_dict()
        )
    regimes = sorted(
        {
            str(regime)
            for regime in list(manifest_df.get("regime", pd.Series(dtype=object)).dropna().astype(str).unique())
            + list(seed_metrics_df.get("regime", pd.Series(dtype=object)).dropna().astype(str).unique())
        }
    )
    rows: list[dict[str, Any]] = []
    for regime in regimes:
        for weather_mode in weather_modes:
            for mode in requested_modes:
                total_slots = int(manifest_counts.get((regime, weather_mode, mode), 0))
                completed_slots = int(completed_counts.get((regime, weather_mode, mode), 0))
                if total_slots <= 0:
                    status = "pending"
                    coverage_pct = np.nan
                elif completed_slots <= 0:
                    status = "not_started"
                    coverage_pct = 0.0
                elif completed_slots < total_slots:
                    status = "partial"
                    coverage_pct = completed_slots / total_slots
                else:
                    status = "complete"
                    coverage_pct = 1.0
                rows.append(
                    {
                        "regime": regime,
                        "weather_mode": weather_mode,
                        "mode": mode,
                        "status": status,
                        "completed_slots": completed_slots,
                        "total_slots": total_slots,
                        "coverage_pct": coverage_pct,
                    }
                )
    mode_order = {name: idx for idx, name in enumerate(requested_modes)}
    weather_order = {name: idx for idx, name in enumerate(weather_modes)}
    return (
        pd.DataFrame(rows)
        .assign(
            _weather_sort=lambda df: df["weather_mode"].map(weather_order).fillna(len(weather_order)),
            _mode_sort=lambda df: df["mode"].map(mode_order).fillna(len(mode_order)),
        )
        .sort_values(["regime", "_weather_sort", "_mode_sort"])
        .drop(columns=["_weather_sort", "_mode_sort"])
        .reset_index(drop=True)
    )


def build_xai_stability_overview_table(rq2_summary_df: pd.DataFrame) -> pd.DataFrame:
    if rq2_summary_df.empty:
        return pd.DataFrame()

    def first_mode(series: pd.Series) -> str:
        non_null = series.dropna().astype(str)
        if non_null.empty:
            return ""
        modes = non_null.mode()
        return str(modes.iloc[0]) if not modes.empty else str(non_null.iloc[0])

    key_cols = ["model_family", "regime", "weather_mode", "mode", "horizon_h", "target_kind"]
    summary_df = (
        rq2_summary_df.groupby(key_cols, as_index=False)
        .agg(
            n_buildings=("building", "nunique"),
            consensus_top_pfi_group=("top_pfi_group", first_mode),
            consensus_top_shap_group=("top_shap_group", first_mode),
            mean_rank_gap=("mean_rank_gap", "mean"),
            max_rank_gap=("max_rank_gap", "max"),
            pairwise_spearman_mean_pfi=("pairwise_spearman_mean_pfi", "mean"),
            pairwise_spearman_mean_shap=("pairwise_spearman_mean_shap", "mean"),
            topk_overlap_mean_pfi=("topk_overlap_mean_pfi", "mean"),
            topk_overlap_mean_shap=("topk_overlap_mean_shap", "mean"),
        )
        .reset_index(drop=True)
    )
    mode_order = {name: idx for idx, name in enumerate(THESIS_MODE_GRAMMAR)}
    weather_order = {name: idx for idx, name in enumerate(THESIS_WEATHER_MODE_ORDER)}
    return (
        summary_df.assign(
            _mode_sort=lambda df: df["mode"].map(mode_order).fillna(len(mode_order)),
            _weather_sort=lambda df: df["weather_mode"].map(weather_order).fillna(len(weather_order)),
        )
        .sort_values(["model_family", "regime", "_weather_sort", "_mode_sort", "horizon_h"])
        .drop(columns=["_mode_sort", "_weather_sort"])
        .reset_index(drop=True)
    )


def split_xai_mode_transition_summary(
    mode_transition_summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode_transition_summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    primary_labels = {_ordered_mode_label(name) for name in PRIMARY_MODE_TRANSITION_COLUMNS}
    supporting_labels = {_ordered_mode_label(name) for name in SUPPORTING_MODE_TRANSITION_COLUMNS}
    primary_df = mode_transition_summary_df.loc[
        mode_transition_summary_df["comparison"].isin(primary_labels)
    ].copy()
    supporting_df = mode_transition_summary_df.loc[
        mode_transition_summary_df["comparison"].isin(supporting_labels)
    ].copy()
    return (
        primary_df.reset_index(drop=True),
        supporting_df.reset_index(drop=True),
    )


def build_xai_thesis_packets(
    *,
    manifest_df: pd.DataFrame,
    seed_metrics_df: pd.DataFrame,
    grouped_pfi_df: pd.DataFrame,
    grouped_shap_df: pd.DataFrame,
    grouped_pfi_fine_df: pd.DataFrame | None = None,
    grouped_shap_fine_df: pd.DataFrame | None = None,
    mode_transition_summary_df: pd.DataFrame,
    rq2_summary_df: pd.DataFrame,
    rq2_summary_fine_df: pd.DataFrame | None = None,
    requested_modes: tuple[str, ...] = THESIS_MODE_GRAMMAR,
    weather_modes: tuple[str, ...] = THESIS_WEATHER_MODE_ORDER,
) -> dict[str, pd.DataFrame]:
    group_share_summary_df = build_xai_group_share_summary(grouped_pfi_df, grouped_shap_df)
    group_share_overview_df = summarize_group_shares_for_thesis(group_share_summary_df)
    grouped_pfi_fine_df = grouped_pfi_fine_df if grouped_pfi_fine_df is not None else pd.DataFrame()
    grouped_shap_fine_df = grouped_shap_fine_df if grouped_shap_fine_df is not None else pd.DataFrame()
    rq2_summary_fine_df = rq2_summary_fine_df if rq2_summary_fine_df is not None else pd.DataFrame()
    group_share_summary_fine_df = build_xai_group_share_summary(
        grouped_pfi_fine_df,
        grouped_shap_fine_df,
        taxonomy="fine",
    )
    group_share_overview_fine_df = summarize_group_shares_for_thesis(group_share_summary_fine_df, taxonomy="fine")
    primary_transition_df, supporting_transition_df = split_xai_mode_transition_summary(mode_transition_summary_df)
    return {
        "mode_definition_df": build_thesis_mode_definition_table(),
        "weather_role_df": build_thesis_weather_role_table(),
        "broad_taxonomy_definition_df": build_broad_taxonomy_definition_table(),
        "fine_taxonomy_definition_df": build_fine_taxonomy_definition_table(),
        "mode_completion_df": build_xai_mode_completion_table(
            manifest_df,
            seed_metrics_df,
            requested_modes=requested_modes,
        ),
        "mode_weather_completion_df": build_xai_mode_weather_completion_table(
            manifest_df,
            seed_metrics_df,
            requested_modes=requested_modes,
            weather_modes=weather_modes,
        ),
        "group_share_summary_df": group_share_summary_df,
        "group_share_overview_df": group_share_overview_df,
        "group_share_summary_fine_df": group_share_summary_fine_df,
        "group_share_overview_fine_df": group_share_overview_fine_df,
        "primary_transition_df": primary_transition_df,
        "supporting_transition_df": supporting_transition_df,
        "stability_overview_df": build_xai_stability_overview_table(rq2_summary_df),
        "stability_overview_fine_df": build_xai_stability_overview_table(rq2_summary_fine_df),
    }


def compute_mode_delta_summaries(
    seed_metrics_df: pd.DataFrame,
    *,
    metric_col: str = "wape_pct",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if seed_metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    compare_pairs = [
        ("M1", "M0"),
        ("M2", "M0"),
        ("M3", "M1"),
        ("M3", "M2"),
        ("M4", "M3"),
        ("M4", "M0"),
    ]
    pivot_df = seed_metrics_df.pivot_table(
        index=[col for col in MATRIX_KEY_COLS if col != "mode"] + ["training_scope", "seed"],
        columns="mode",
        values=metric_col,
    ).reset_index()
    for lhs, rhs in compare_pairs:
        if lhs in pivot_df.columns and rhs in pivot_df.columns:
            pivot_df[f"{lhs}_minus_{rhs}_{metric_col}_pp"] = pivot_df[lhs] - pivot_df[rhs]
    seed_level = pivot_df.copy()

    agg_rows = []
    group_cols = [col for col in MATRIX_KEY_COLS if col != "mode"] + ["training_scope"]
    delta_cols = [col for col in seed_level.columns if col.endswith(f"_{metric_col}_pp")]
    for keys, group_df in seed_level.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        row["seed_count"] = int(group_df["seed"].nunique())
        for col in delta_cols:
            row[col] = float(group_df[col].mean())
            row[f"{col}_seed_std"] = float(group_df[col].std(ddof=0))
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows).sort_values(group_cols).reset_index(drop=True)
    return seed_level, agg_df


def compute_stability_tables(
    seed_pfi_df: pd.DataFrame,
    seed_shap_df: pd.DataFrame,
    *,
    top_k: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    method_specs = [
        ("pfi", seed_pfi_df, "delta_wape_mean", None),
        ("shap", seed_shap_df, "mean_abs_group_shap", "mean_signed_group_shap"),
    ]
    base_group_cols = MATRIX_KEY_COLS + ["training_scope"]

    for method, df, value_col, signed_col in method_specs:
        if df.empty:
            continue
        for keys, group_df in df.groupby(base_group_cols, dropna=False):
            seeds = tuple(sorted(int(seed) for seed in group_df["seed"].dropna().unique().tolist()))
            if len(seeds) < 2:
                continue
            value_pivot = group_df.pivot_table(index="feature_group", columns="seed", values=value_col, aggfunc="mean").fillna(0.0)
            rank_pivot = value_pivot.rank(axis=0, method="dense", ascending=False)
            pairwise_spearman = []
            pairwise_overlap = []
            for seed_a, seed_b in itertools.combinations(seeds, 2):
                rho = rank_pivot[seed_a].corr(rank_pivot[seed_b], method="spearman")
                pairwise_spearman.append(float(rho) if pd.notna(rho) else np.nan)
                top_a = set(value_pivot[seed_a].sort_values(ascending=False).head(int(top_k)).index.tolist())
                top_b = set(value_pivot[seed_b].sort_values(ascending=False).head(int(top_k)).index.tolist())
                pairwise_overlap.append(float(len(top_a & top_b)) / float(top_k))
            summary_row = {col: value for col, value in zip(base_group_cols, keys)}
            summary_row["method"] = method
            summary_row["seed_count"] = int(len(seeds))
            summary_row["pairwise_spearman_mean"] = float(np.nanmean(pairwise_spearman))
            summary_row["pairwise_spearman_min"] = float(np.nanmin(pairwise_spearman))
            summary_row["topk_overlap_mean"] = float(np.nanmean(pairwise_overlap))
            summary_row["topk_overlap_min"] = float(np.nanmin(pairwise_overlap))
            summary_rows.append(summary_row)

            signed_pivot = None
            if signed_col is not None and signed_col in group_df.columns:
                signed_pivot = group_df.pivot_table(index="feature_group", columns="seed", values=signed_col, aggfunc="mean").fillna(0.0)

            for feature_group, rank_row in rank_pivot.iterrows():
                value_row = value_pivot.loc[feature_group]
                group_row = {col: value for col, value in zip(base_group_cols, keys)}
                group_row["method"] = method
                group_row["feature_group"] = str(feature_group)
                group_row["seed_count"] = int(len(seeds))
                group_row["rank_mean"] = float(rank_row.mean())
                group_row["rank_std"] = float(rank_row.std(ddof=0))
                group_row["rank_span"] = float(rank_row.max() - rank_row.min())
                group_row["value_mean"] = float(value_row.mean())
                group_row["value_std"] = float(value_row.std(ddof=0))
                if signed_pivot is not None and feature_group in signed_pivot.index:
                    sign_row = np.sign(signed_pivot.loc[feature_group].to_numpy(dtype=float))
                    positive_share = float(np.mean(sign_row > 0))
                    negative_share = float(np.mean(sign_row < 0))
                    zero_share = float(np.mean(sign_row == 0))
                    group_row["positive_seed_share"] = positive_share
                    group_row["negative_seed_share"] = negative_share
                    group_row["zero_seed_share"] = zero_share
                    group_row["sign_consistency"] = float(max(positive_share, negative_share, zero_share))
                group_rows.append(group_row)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(base_group_cols + ["method"]).reset_index(drop=True)
    group_df = pd.DataFrame(group_rows)
    if not group_df.empty:
        group_df = group_df.sort_values(base_group_cols + ["method", "rank_mean", "feature_group"]).reset_index(drop=True)
    return summary_df, group_df


def build_rq1_accuracy_summary(
    metrics_df: pd.DataFrame,
    mode_delta_df: pd.DataFrame,
) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    key_cols = [col for col in MATRIX_KEY_COLS if col != "mode"] + ["training_scope"]
    best_df = metrics_df.sort_values("wape_pct").groupby(key_cols, as_index=False).first()
    delta_cols = [col for col in mode_delta_df.columns if col.endswith("_wape_pct_pp")]
    if not delta_cols:
        return best_df
    return best_df.merge(mode_delta_df[key_cols + delta_cols], on=key_cols, how="left")


def build_rq2_xai_stability_summary(
    grouped_pfi_df: pd.DataFrame,
    grouped_shap_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    stability_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    if grouped_pfi_df.empty or grouped_shap_df.empty:
        return pd.DataFrame()
    key_cols = MATRIX_KEY_COLS + ["training_scope"]
    pfi_top = grouped_pfi_df.sort_values("delta_wape_mean", ascending=False).groupby(key_cols, as_index=False).first()
    shap_top = grouped_shap_df.sort_values("mean_abs_group_shap", ascending=False).groupby(key_cols, as_index=False).first()
    pfi_top = pfi_top[key_cols + ["feature_group"]].rename(columns={"feature_group": "top_pfi_group"})
    shap_top = shap_top[key_cols + ["feature_group"]].rename(columns={"feature_group": "top_shap_group"})
    agreement_summary = agreement_df.groupby(key_cols, as_index=False).agg(
        mean_rank_gap=("rank_gap", "mean"),
        max_rank_gap=("rank_gap", "max"),
    )
    stability_wide = pd.DataFrame()
    if not stability_summary_df.empty:
        stability_wide = stability_summary_df.pivot_table(
            index=key_cols,
            columns="method",
            values=["pairwise_spearman_mean", "topk_overlap_mean"],
            aggfunc="first",
        )
        stability_wide.columns = ["_".join(str(part) for part in col if part) for col in stability_wide.columns.to_flat_index()]
        stability_wide = stability_wide.reset_index()
    out = pfi_top.merge(shap_top, on=key_cols, how="outer").merge(agreement_summary, on=key_cols, how="left")
    if not stability_wide.empty:
        out = out.merge(stability_wide, on=key_cols, how="left")
    return out.sort_values(key_cols).reset_index(drop=True)


def build_rq3_operational_summary(scenario_summary_df: pd.DataFrame) -> pd.DataFrame:
    if scenario_summary_df.empty:
        return pd.DataFrame()
    key_cols = ["model_family", "regime", "building", "mode", "weather_mode", "horizon_h", "case_type", "scenario"]
    value_cols = ["prediction", "delta_prediction", "n_changed_features"]
    agg_map = {col: "mean" for col in value_cols if col in scenario_summary_df.columns}
    return (
        scenario_summary_df.groupby(key_cols, as_index=False)
        .agg(agg_map)
        .sort_values(key_cols)
        .reset_index(drop=True)
    )


def merge_xai_outputs(*outputs_list: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "manifest_df": pd.DataFrame(),
        "seed_metrics_df": pd.DataFrame(),
        "seed_grouped_pfi_df": pd.DataFrame(),
        "seed_grouped_pfi_fine_df": pd.DataFrame(),
        "seed_grouped_shap_df": pd.DataFrame(),
        "seed_grouped_shap_fine_df": pd.DataFrame(),
        "seed_agreement_df": pd.DataFrame(),
        "seed_agreement_fine_df": pd.DataFrame(),
        "run_log_df": pd.DataFrame(),
        "detail_cache": {},
    }
    for outputs in outputs_list:
        if not outputs:
            continue
        normalized = _normalize_xai_outputs(outputs)
        for key in (
            "manifest_df",
            "seed_metrics_df",
            "seed_grouped_pfi_df",
            "seed_grouped_pfi_fine_df",
            "seed_grouped_shap_df",
            "seed_grouped_shap_fine_df",
            "seed_agreement_df",
            "seed_agreement_fine_df",
            "run_log_df",
        ):
            merged[key] = pd.concat([merged[key], normalized[key]], ignore_index=True) if not merged[key].empty else normalized[key].copy()
        merged["detail_cache"].update(dict(normalized.get("detail_cache", {})))
    return _normalize_xai_outputs(merged)


def build_xai_mode_transition_summary(
    mode_delta_df: pd.DataFrame,
    *,
    preferred_pairs: tuple[str, ...] = (
        *PRIMARY_MODE_TRANSITION_COLUMNS,
        *SUPPORTING_MODE_TRANSITION_COLUMNS,
    ),
) -> pd.DataFrame:
    if mode_delta_df.empty:
        return pd.DataFrame()
    key_cols = ["model_family", "regime", "weather_mode", "horizon_h"]
    value_cols = [col for col in preferred_pairs if col in mode_delta_df.columns]
    if not value_cols:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for column_name in value_cols:
        label = _ordered_mode_label(column_name)
        subset = mode_delta_df.loc[:, [col for col in key_cols + ["building", "training_scope", column_name] if col in mode_delta_df.columns]].copy()
        subset["comparison"] = label
        subset.rename(columns={column_name: "delta_wape_pct_pp"}, inplace=True)
        rows.append(subset)
    long_df = pd.concat(rows, ignore_index=True)
    return (
        long_df.groupby(key_cols + ["comparison"], as_index=False)
        .agg(
            n_buildings=("building", "nunique"),
            mean_delta_wape_pct_pp=("delta_wape_pct_pp", "mean"),
            median_delta_wape_pct_pp=("delta_wape_pct_pp", "median"),
            buildings_improved_wape=("delta_wape_pct_pp", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
        )
        .sort_values(key_cols + ["comparison"])
        .reset_index(drop=True)
    )


def export_report_ready_xai_artifacts(
    outputs: dict[str, Any],
    report_dir: Path,
    *,
    manifest_df: pd.DataFrame,
    allowed_model_families: tuple[str, ...],
    desired_manifest_df: pd.DataFrame | None = None,
    preferred_mode_pairs: tuple[str, ...] = (
        *PRIMARY_MODE_TRANSITION_COLUMNS,
        *SUPPORTING_MODE_TRANSITION_COLUMNS,
    ),
) -> dict[str, pd.DataFrame]:
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    scoped_outputs = scope_xai_outputs_to_manifest(
        outputs,
        manifest_df,
        allowed_model_families=allowed_model_families,
        use_latest_run_log=True,
    )

    paths = build_xai_artifact_paths(report_dir)
    _write_xai_outputs(paths, scoped_outputs)

    metrics_df = aggregate_seed_metrics(scoped_outputs["seed_metrics_df"])
    grouped_pfi_df = aggregate_seed_grouped_pfi(scoped_outputs["seed_grouped_pfi_df"])
    grouped_pfi_fine_df = aggregate_seed_grouped_pfi(scoped_outputs["seed_grouped_pfi_fine_df"])
    grouped_shap_df = aggregate_seed_grouped_shap(scoped_outputs["seed_grouped_shap_df"])
    grouped_shap_fine_df = aggregate_seed_grouped_shap(scoped_outputs["seed_grouped_shap_fine_df"])
    group_share_summary_df = build_xai_group_share_summary(grouped_pfi_df, grouped_shap_df)
    group_share_summary_fine_df = build_xai_group_share_summary(
        grouped_pfi_fine_df,
        grouped_shap_fine_df,
        taxonomy="fine",
    )
    agreement_df = aggregate_seed_agreement(scoped_outputs["seed_agreement_df"])
    agreement_fine_df = aggregate_seed_agreement(scoped_outputs["seed_agreement_fine_df"])
    seed_mode_delta_df, mode_delta_df = compute_mode_delta_summaries(scoped_outputs["seed_metrics_df"])
    stability_summary_df, stability_by_group_df = compute_stability_tables(
        scoped_outputs["seed_grouped_pfi_df"],
        scoped_outputs["seed_grouped_shap_df"],
    )
    stability_summary_fine_df, stability_by_group_fine_df = compute_stability_tables(
        scoped_outputs["seed_grouped_pfi_fine_df"],
        scoped_outputs["seed_grouped_shap_fine_df"],
    )
    rq1_summary_df = build_rq1_accuracy_summary(metrics_df, mode_delta_df)
    rq2_summary_df = build_rq2_xai_stability_summary(grouped_pfi_df, grouped_shap_df, agreement_df, stability_summary_df)
    rq2_summary_fine_df = build_rq2_xai_stability_summary(
        grouped_pfi_fine_df,
        grouped_shap_fine_df,
        agreement_fine_df,
        stability_summary_fine_df,
    )
    transition_summary_df = build_xai_mode_transition_summary(
        mode_delta_df,
        preferred_pairs=preferred_mode_pairs,
    )
    desired_manifest_scope_df = (
        _normalize_xai_manifest_frame(desired_manifest_df)
        if desired_manifest_df is not None
        else scoped_outputs["manifest_df"].copy()
    )
    coverage_df, missing_df = validate_matrix_completeness(
        desired_manifest_scope_df,
        scoped_outputs["seed_metrics_df"],
    )
    inventory_df = build_xai_matrix_inventory(
        desired_manifest_scope_df,
        scoped_outputs["manifest_df"],
        source_dir=report_dir,
        notebook="13",
        layer="xai_matrix",
    )

    exports = {
        "xai_metrics.csv": metrics_df,
        "xai_grouped_pfi.csv": grouped_pfi_df,
        "xai_grouped_pfi_fine.csv": grouped_pfi_fine_df,
        "xai_grouped_shap.csv": grouped_shap_df,
        "xai_grouped_shap_fine.csv": grouped_shap_fine_df,
        "xai_group_share_summary.csv": group_share_summary_df,
        "xai_group_share_summary_fine.csv": group_share_summary_fine_df,
        "xai_pfi_shap_agreement.csv": agreement_df,
        "xai_pfi_shap_agreement_fine.csv": agreement_fine_df,
        "xai_seed_mode_delta_summary.csv": seed_mode_delta_df,
        "xai_mode_delta_summary.csv": mode_delta_df,
        "xai_stability_summary.csv": stability_summary_df,
        "xai_stability_by_group.csv": stability_by_group_df,
        "xai_stability_summary_fine.csv": stability_summary_fine_df,
        "xai_stability_by_group_fine.csv": stability_by_group_fine_df,
        "xai_rq1_accuracy_summary.csv": rq1_summary_df,
        "xai_rq2_xai_stability_summary.csv": rq2_summary_df,
        "xai_rq2_xai_stability_summary_fine.csv": rq2_summary_fine_df,
        "xai_mode_transition_summary.csv": transition_summary_df,
        "xai_manifest_coverage.csv": coverage_df,
        "xai_manifest_missing.csv": missing_df,
        "xai_matrix_inventory.csv": inventory_df,
        "xai_run_log_latest.csv": scoped_outputs["run_log_df"],
    }
    for filename, df in exports.items():
        df.to_csv(report_dir / filename, index=False)
    return exports


def collapse_lstm_shap_payload(
    experiment: FittedLSTMExperiment,
    shap_payload: dict[str, Any],
    *,
    value_source: str = "last",
) -> tuple[np.ndarray, pd.DataFrame]:
    dynamic_shap = np.asarray(shap_payload["shap_values"], dtype=float)
    dynamic_values = np.asarray(shap_payload["X"]["dynamic"], dtype=float)
    if value_source == "last":
        dynamic_feature_values = dynamic_values[:, -1, :]
    elif value_source == "mean":
        dynamic_feature_values = dynamic_values.mean(axis=1)
    else:
        raise ValueError(f"Unsupported LSTM SHAP value_source: {value_source}")

    shap_2d = dynamic_shap.sum(axis=1)
    value_df = pd.DataFrame(dynamic_feature_values, columns=shap_payload["feature_names"])

    static_shap = shap_payload.get("static_shap_values")
    static_feature_names = shap_payload.get("static_feature_names") or []
    if static_shap is not None and len(static_feature_names) > 0:
        static_shap_arr = np.asarray(static_shap, dtype=float)
        shap_2d = np.concatenate([shap_2d, static_shap_arr], axis=1)
        static_values = np.asarray(shap_payload["X"]["static"], dtype=float)
        value_df = pd.concat(
            [
                value_df,
                pd.DataFrame(static_values, columns=list(static_feature_names)),
            ],
            axis=1,
        )
    return shap_2d, value_df


def prepare_shap_plot_payload(
    experiment: FittedXGBExperiment | FittedLSTMExperiment,
    shap_payload: dict[str, Any],
    *,
    value_source: str = "last",
) -> tuple[np.ndarray, pd.DataFrame]:
    if isinstance(experiment, FittedXGBExperiment):
        return np.asarray(shap_payload["shap_values"], dtype=float), shap_payload["X"].copy()
    return collapse_lstm_shap_payload(experiment, shap_payload, value_source=value_source)


def plot_shap_violin_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    *,
    max_display: int = 12,
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    shap_df = pd.DataFrame(np.asarray(shap_values, dtype=float), columns=list(X.columns))
    top_features = shap_df.abs().mean().sort_values(ascending=False).head(int(max_display)).index.tolist()
    plot_df = shap_df.loc[:, top_features].melt(var_name="feature", value_name="shap_value")
    fig, ax = plt.subplots(figsize=(10, max(5.5, 0.45 * len(top_features) + 2.0)))
    sns.violinplot(data=plot_df, x="shap_value", y="feature", orient="h", inner="quart", cut=0, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("SHAP contribution")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_shap_dependence_scatter(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    *,
    feature_name: str,
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    shap_df = pd.DataFrame(np.asarray(shap_values, dtype=float), columns=list(X.columns))
    if feature_name not in shap_df.columns or feature_name not in X.columns:
        raise KeyError(f"Feature {feature_name} missing from SHAP scatter inputs.")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(X[feature_name], shap_df[feature_name], s=16, alpha=0.45, color="#33658A", edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(feature_name)
    ax.set_ylabel(f"SHAP contribution: {feature_name}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_faceted_group_horizon_bars(
    df: pd.DataFrame,
    *,
    feature_groups: list[str],
    value_col: str,
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    plot_df = df.loc[df["feature_group"].isin(feature_groups)].copy()
    if plot_df.empty:
        raise ValueError("Cannot plot empty faceted horizon bars.")
    feature_groups = [group for group in feature_groups if group in plot_df["feature_group"].unique()]
    fig, axes = plt.subplots(1, len(feature_groups), figsize=(max(7.5, 4.4 * len(feature_groups)), 4.8), squeeze=False, sharey=True)
    for ax, feature_group in zip(axes[0], feature_groups):
        group_slice = plot_df.loc[plot_df["feature_group"] == feature_group].copy()
        sns.barplot(
            data=group_slice,
            x="horizon_h",
            y=value_col,
            hue="mode",
            ax=ax,
            errorbar=None,
        )
        ax.set_title(feature_group)
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel(value_col.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
        if ax is not axes[0, 0]:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), title="mode")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def _local_feature_contribution_series(
    experiment: FittedXGBExperiment | FittedLSTMExperiment,
    shap_payload: dict[str, Any] | None,
    *,
    row_idx: int,
    random_seed: int = 42,
) -> pd.Series:
    if isinstance(experiment, FittedXGBExperiment):
        payload = shap_payload if shap_payload is not None else compute_tree_shap(experiment)
        return pd.Series(
            np.asarray(payload["shap_values"], dtype=float)[int(row_idx)],
            index=list(payload["feature_names"]),
            dtype=float,
        )

    payload = shap_payload
    if payload is None or int(row_idx) not in set(np.asarray(payload.get("sample_idx", []), dtype=int).tolist()):
        payload = compute_lstm_gradient_shap(
            experiment,
            n_background=min(128, max(32, experiment.X_fit_dynamic.shape[0])),
            random_seed=random_seed,
            explain_idx=[int(row_idx)],
        )
    shap_2d, value_df = collapse_lstm_shap_payload(experiment, payload)
    sample_idx = np.asarray(payload["sample_idx"], dtype=int)
    position = int(np.where(sample_idx == int(row_idx))[0][0])
    return pd.Series(np.asarray(shap_2d[position], dtype=float), index=list(value_df.columns), dtype=float)


def plot_local_case_figure_any(
    experiment: FittedXGBExperiment | FittedLSTMExperiment,
    *,
    row_idx: int,
    case_type: str,
    shap_payload: dict[str, Any] | None = None,
    save_path: Path | None = None,
    context_hours: int = 24,
    top_k: int = 8,
    random_seed: int = 42,
) -> plt.Figure:
    pred_row = experiment.predictions_df.loc[experiment.predictions_df["row_idx"] == int(row_idx)]
    if pred_row.empty:
        raise KeyError(f"row_idx {row_idx} missing from predictions.")
    pred_row = pred_row.iloc[0]
    issue_time = pd.Timestamp(pred_row["datetime"])
    start_time = issue_time - pd.Timedelta(hours=int(context_hours))
    context_df = experiment.raw_frame.loc[
        (pd.to_datetime(experiment.raw_frame["datetime"]) >= start_time)
        & (pd.to_datetime(experiment.raw_frame["datetime"]) <= issue_time)
    ].copy()

    shap_series = _local_feature_contribution_series(
        experiment,
        shap_payload,
        row_idx=int(row_idx),
        random_seed=random_seed,
    )
    top_features = shap_series.abs().sort_values(ascending=False).head(int(top_k)).index.tolist()
    context_features = _context_feature_candidates(top_features, list(context_df.columns))

    n_left = 1 + len(context_features)
    fig, axes = plt.subplots(
        n_left,
        2,
        figsize=(14, max(5.5, 2.4 * n_left)),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
        squeeze=False,
    )

    heat_ax = axes[0, 0]
    heat_ax.plot(context_df["datetime"], context_df["heat_kwh"], color="#1b4965", linewidth=1.8)
    heat_ax.axvline(issue_time, color="#c1121f", linestyle="--", linewidth=1.2)
    if "feat_heat_obs" in context_df.columns:
        latest_heat_obs = context_df.loc[pd.to_datetime(context_df["datetime"]) == issue_time, "feat_heat_obs"]
        if not latest_heat_obs.empty:
            heat_ax.scatter([issue_time], [float(latest_heat_obs.iloc[0])], color="#1b4965", s=24, zorder=3)
    heat_ax.set_ylabel("Heat kWh")
    heat_ax.set_title(
        f"{experiment.model_family.upper()} | {experiment.building} | {experiment.mode} | "
        f"{experiment.target_kind}_h{experiment.horizon_h:02d} | {case_type}\n"
        f"issue={issue_time} | pred={pred_row['y_pred']:.1f} | actual={pred_row['y_true']:.1f}"
    )
    heat_ax.grid(alpha=0.25)

    for ax, feature_name in zip(axes[1:, 0], context_features):
        ax.plot(context_df["datetime"], context_df[feature_name], linewidth=1.6, color="#5f0f40")
        ax.axvline(issue_time, color="#c1121f", linestyle="--", linewidth=1.0)
        ax.set_ylabel(feature_name)
        ax.grid(alpha=0.25)

    for ax in axes[:, 1]:
        ax.axis("off")

    bar_ax = axes[:, 1][0]
    bar_ax.axis("on")
    local_df = pd.DataFrame(
        {
            "feature": top_features[::-1],
            "shap_value": shap_series.loc[top_features[::-1]].to_numpy(dtype=float),
        }
    )
    colors = np.where(local_df["shap_value"] >= 0, "#c1121f", "#1d3557")
    bar_ax.barh(local_df["feature"], local_df["shap_value"], color=colors)
    bar_ax.axvline(0.0, color="black", linewidth=0.9)
    bar_ax.set_title("Top local SHAP values")
    bar_ax.set_xlabel("SHAP contribution")
    bar_ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def run_operational_scenarios(
    experiment: FittedXGBExperiment,
    *,
    row_idx: int,
    shift_hours: tuple[int, ...] = (1, 2),
    include_weather_memory: bool = True,
) -> pd.DataFrame:
    pred_row = experiment.predictions_df.loc[experiment.predictions_df["row_idx"] == int(row_idx)]
    if pred_row.empty:
        raise KeyError(f"row_idx {row_idx} missing from predictions.")
    pred_row = pred_row.iloc[0]
    issue_time = pd.Timestamp(pred_row["datetime"])
    frame = experiment.raw_frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    current_rows = frame.index[frame["datetime"] == issue_time]
    if len(current_rows) != 1:
        raise KeyError(f"Expected one frame row at {issue_time}, found {len(current_rows)}")
    current_row_idx = int(current_rows[0])

    feature_blocks = scenario_history_columns(experiment.feature_cols)
    block_map = {
        "heat_history": list(feature_blocks["heat_history"]),
        "historical_weather_memory": list(feature_blocks["historical_weather_memory"]) if include_weather_memory else [],
    }
    baseline_features = experiment.test_df.loc[int(row_idx), experiment.feature_cols].copy()
    baseline_pred = float(pred_row["y_pred"])
    rows = [
        {
            "scenario": "baseline",
            "scenario_block": "baseline",
            "shift_hours": 0,
            "prediction": baseline_pred,
            "delta_prediction": 0.0,
            "source_time": issue_time,
            "n_changed_features": 0,
        }
    ]

    for block_name, block_cols in block_map.items():
        block_cols = [col for col in block_cols if col in experiment.feature_cols]
        if not block_cols:
            continue
        for shift in shift_hours:
            for direction, sign in (("earlier", -1), ("later", 1)):
                source_idx = current_row_idx + sign * int(shift)
                if source_idx < 0 or source_idx >= len(frame):
                    continue
                source_row = frame.iloc[source_idx]
                scenario_features = baseline_features.copy()
                for col in block_cols:
                    if col in source_row.index:
                        scenario_features[col] = source_row[col]
                scenario_df = pd.DataFrame([scenario_features], columns=experiment.feature_cols)
                scenario_pred = float(mfc._predict_with_best_iteration(experiment.model, scenario_df)[0])
                rows.append(
                    {
                        "scenario": f"{block_name}_{direction}_{int(shift)}h",
                        "scenario_block": block_name,
                        "shift_hours": int(shift),
                        "prediction": scenario_pred,
                        "delta_prediction": scenario_pred - baseline_pred,
                        "source_time": pd.Timestamp(source_row["datetime"]),
                        "n_changed_features": int(len(block_cols)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_scenario_directionality(scenario_df: pd.DataFrame) -> pd.DataFrame:
    if scenario_df.empty:
        return pd.DataFrame()
    plot_df = scenario_df.loc[scenario_df["scenario"] != "baseline"].copy()
    key_cols = [
        col
        for col in [
            "model_family",
            "regime",
            "building",
            "mode",
            "weather_mode",
            "horizon_h",
            "case_type",
            "scenario_block",
            "shift_hours",
        ]
        if col in plot_df.columns
    ]
    summary_df = (
        plot_df.groupby(key_cols, as_index=False)
        .agg(
            mean_delta_prediction=("delta_prediction", "mean"),
            mean_abs_delta_prediction=("delta_prediction", lambda s: float(np.mean(np.abs(s)))),
            min_delta_prediction=("delta_prediction", "min"),
            max_delta_prediction=("delta_prediction", "max"),
            n_changed_features=("n_changed_features", "mean"),
        )
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    return summary_df


def validate_matrix_completeness(
    manifest_df: pd.DataFrame,
    seed_metrics_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if manifest_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    metric_keys = MATRIX_KEY_COLS + ["seed", "training_scope"]
    observed_df = seed_metrics_df.loc[:, [col for col in metric_keys if col in seed_metrics_df.columns]].drop_duplicates()
    merged = manifest_df.merge(
        observed_df.assign(observed=True),
        on=metric_keys,
        how="left",
    )
    merged["observed"] = merged["observed"].fillna(False).astype(bool)
    missing_df = merged.loc[~merged["observed"]].copy().reset_index(drop=True)
    return merged.reset_index(drop=True), missing_df
