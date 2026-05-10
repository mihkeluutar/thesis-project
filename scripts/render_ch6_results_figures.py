#!/usr/bin/env python3
"""Render Chapter 6 thesis figures from saved report-ready artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent.parent / ".mplconfig")))

import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

try:
    pd.get_option("mode.use_inf_as_na")
except pd.errors.OptionError:
    from pandas._config.config import register_option

    register_option(
        "mode.use_inf_as_na",
        False,
        doc="Compatibility option for seaborn line plots with newer pandas.",
    )

SCRIPT_PATH = Path(__file__).resolve()
THESIS_ROOT = SCRIPT_PATH.parents[2]
PROJECT_ROOT = SCRIPT_PATH.parents[1]
FIGURES_DIR = THESIS_ROOT / "masters-thesis" / "figures"

TRAIN_TEST_SPLIT = pd.Timestamp("2024-01-01")
WEATHER_CLEAN_CSV = PROJECT_ROOT / "data" / "clean" / "weather_hourly_clean.csv"
U05_CLEAN_CSV = PROJECT_ROOT / "data" / "clean" / "U05_hourly_clean.csv"
OBSERVED_BUILDING_CSVS = {
    "U05": PROJECT_ROOT / "data" / "clean" / "U05_hourly_clean.csv",
    "U06": PROJECT_ROOT / "data" / "clean" / "U06_hourly_clean.csv",
    "LIB": PROJECT_ROOT / "data" / "clean" / "LIB_hourly_clean.csv",
    "U02B": PROJECT_ROOT / "data" / "clean" / "U02B_hourly_clean.csv",
    "SOC": PROJECT_ROOT / "data" / "clean" / "SOC_hourly_clean.csv",
    "U03": PROJECT_ROOT / "data" / "clean" / "U03_hourly_clean.csv",
}
BASELINE_POINT_SUMMARY_CSV = PROJECT_ROOT / "results" / "baseline_point_horizon_summary.csv"
BASELINE_POINT_TRACE_CSV = PROJECT_ROOT / "results" / "baseline_point_horizon_U05_trace_weeks.csv"
BAYES_RESULTS_DIR = PROJECT_ROOT / "results" / "bayesian_es_baselines_20042026"
BAYES_ANALYSIS_DIR = BAYES_RESULTS_DIR / "analysis"
BAYES_FW2_BASELINE_WAPE_CSV = BAYES_ANALYSIS_DIR / "fw2_baseline_wape_summary.csv"
BAYES_FW2_BEST_VS_NONLINEAR_M4_WAPE_CSV = (
    BAYES_ANALYSIS_DIR / "fw2_best_baseline_vs_nonlinear_m4_wape.csv"
)
BASE_COMPARISON_CSV = (
    PROJECT_ROOT
    / "results"
    / "report_ready_20260405"
    / "model_family_base"
    / "comparison_summary_report.csv"
)
BASE_CONTEXT_CSV = (
    PROJECT_ROOT
    / "results"
    / "report_ready_20260405"
    / "model_family_base"
    / "rmse_load_context_per_building.csv"
)
MODEL_FAMILY_COMPARISON_CSV = (
    PROJECT_ROOT
    / "results"
    / "report_ready_20260405"
    / "model_family"
    / "comparison_summary.csv"
)
SUPPLEMENT_PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "results"
    / "report_ready_20260405"
    / "model_family_m3_supplement"
    / "comparison_predictions.csv"
)
SUPPLEMENT_COMPARISON_CSV = (
    PROJECT_ROOT
    / "results"
    / "report_ready_20260405"
    / "model_family_m3_supplement"
    / "comparison_summary_report.csv"
)
SUPPLEMENT_METRICS_CSV = (
    PROJECT_ROOT
    / "results"
    / "report_ready_20260405"
    / "model_family_m3_supplement"
    / "comparison_metrics_report.csv"
)
XAI_GROUP_SHARE_CSV = (
    PROJECT_ROOT
    / "results"
    / "xai_matrix_work_20260405"
    / "_report_ready"
    / "xai_group_share_summary.csv"
)
XAI_GROUP_SHARE_FINE_CSV = (
    PROJECT_ROOT
    / "results"
    / "xai_matrix_work_20260405"
    / "_report_ready"
    / "xai_group_share_summary_fine.csv"
)

HORIZONS = [1, 2, 4, 6, 8, 12, 16, 20, 24, 36]
HORIZON_POSITIONS = list(range(len(HORIZONS)))
HORIZON_POSITION_MAP = {h: idx for idx, h in enumerate(HORIZONS)}
BASELINE_POINT_HORIZONS = [1, 3, 4, 6, 8]
BASELINE_MODEL_ORDER = [
    "Persistence_1h",
    "Persistence_week",
    "Static_ES",
    "ARX_ES",
    "ARMAX_ES",
]
BASELINE_MODEL_LABELS = {
    "Persistence_1h": "Persistence 1h",
    "Persistence_week": "Persistence week",
    "Static_ES": "Static ES",
    "ARX_ES": "ARX-ES",
    "ARMAX_ES": "ARMAX-ES",
}
BASELINE_MODEL_COLORS = {
    "Persistence 1h": "#1f77b4",
    "Persistence week": "#7f7f7f",
    "Static ES": "#2ca02c",
    "ARX-ES": "#ff7f0e",
    "ARMAX-ES": "#d62728",
}
BASELINE_MODEL_MARKERS = {
    "Persistence 1h": "o",
    "Persistence week": "s",
    "Static ES": "^",
    "ARX-ES": "D",
    "ARMAX-ES": "P",
}
WEATHER_ORDER = ["FW0", "FW2", "FW1"]
WEATHER_LABELS = {
    "FW0": "FW0: no future weather",
    "FW2": "FW2: forecast-like future weather",
    "FW1": "FW1: oracle future weather",
}
WEATHER_COLORS = {
    "FW0": "#8c510a",
    "FW2": "#1b9e77",
    "FW1": "#5e3c99",
}
MODEL_FAMILY_COLORS = {
    "LSTM": "#c46a2d",
    "XGBoost": "#2a6f8f",
}
MODEL_FAMILY_MARKERS = {
    "LSTM": "o",
    "XGBoost": "s",
}
BAYES_BASELINE_ORDER = ["Bayes_ES", "Bayes_ARX_ES", "Bayes_ARMAX_ES"]
BAYES_BASELINE_LABELS = {
    "Bayes_ES": "Bayes ES",
    "Bayes_ARX_ES": "Bayes ARX-ES",
    "Bayes_ARMAX_ES": "Bayes ARMAX-ES",
}
BAYES_BASELINE_COLORS = {
    "Bayes_ES": "#7a7f87",
    "Bayes_ARX_ES": "#4c78a8",
    "Bayes_ARMAX_ES": "#c46a2d",
}
BAYES_BASELINE_MARKERS = {
    "Bayes_ES": "o",
    "Bayes_ARX_ES": "s",
    "Bayes_ARMAX_ES": "D",
}
BAYES_OVERLAY_ORDER = ["Bayes_ARX_ES", "LSTM M4", "XGBOOST M4"]
BAYES_OVERLAY_LABELS = {
    "Bayes_ARX_ES": "Bayes ARX-ES",
    "LSTM M4": "LSTM M4",
    "XGBOOST M4": "XGBoost M4",
}
BAYES_OVERLAY_COLORS = {
    "Bayes_ARX_ES": "#6c5b7b",
    "LSTM M4": MODEL_FAMILY_COLORS["LSTM"],
    "XGBOOST M4": MODEL_FAMILY_COLORS["XGBoost"],
}
BAYES_OVERLAY_MARKERS = {
    "Bayes_ARX_ES": "o",
    "LSTM M4": "s",
    "XGBOOST M4": "D",
}
BAYES_OVERLAY_LINESTYLES = {
    "Bayes_ARX_ES": "--",
    "LSTM M4": "-",
    "XGBOOST M4": "-",
}
FINE_GROUP_ORDER = [
    "demand_signal",
    "current_weather",
    "calendar",
    "weather_memory",
    "space_heating_dynamics",
    "ventilation_dynamics",
    "future_weather_paths",
    "future_weather_summaries",
    "static_ehr_morphology",
    "static_profile_and_inventory",
]
FINE_GROUP_LABELS = {
    "demand_signal": "Demand signal",
    "current_weather": "Current weather",
    "calendar": "Calendar",
    "weather_memory": "Weather memory",
    "space_heating_dynamics": "Space-heating dynamics",
    "ventilation_dynamics": "Ventilation dynamics",
    "future_weather_paths": "Future weather paths",
    "future_weather_summaries": "Future weather summaries",
    "static_ehr_morphology": "Static EHR morphology",
    "static_profile_and_inventory": "Static profile & inventory",
}
GROUP_ORDER = [
    "recent_demand_history",
    "current_weather",
    "future_weather",
    "historical_weather_memory",
    "system_dynamic_proxies",
    "static_building_context",
    "calendar",
]
GROUP_LABELS = {
    "recent_demand_history": "Recent demand history",
    "current_weather": "Current weather",
    "future_weather": "Future weather",
    "historical_weather_memory": "Historical weather memory",
    "system_dynamic_proxies": "System dynamic proxies",
    "static_building_context": "Static building context",
    "calendar": "Calendar",
}
GROUP_COLORS = {
    "recent_demand_history": "#2a9d8f",
    "current_weather": "#264653",
    "future_weather": "#577590",
    "historical_weather_memory": "#e76f51",
    "system_dynamic_proxies": "#f4a261",
    "static_building_context": "#9c89b8",
    "calendar": "#8ab17d",
}
WRAPPED_GROUP_LABELS = {
    "demand_signal": "Demand\nsignal",
    "current_weather": "Current\nweather",
    "calendar": "Calendar",
    "weather_memory": "Weather\nmemory",
    "space_heating_dynamics": "Space-heating\ndynamics",
    "ventilation_dynamics": "Ventilation\ndynamics",
    "future_weather_paths": "Future weather\npaths",
    "future_weather_summaries": "Future weather\nsummaries",
    "static_ehr_morphology": "Static EHR\nmorphology",
    "static_profile_and_inventory": "Static profile\n& inventory",
}


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9.5,
            "axes.titlesize": 12.5,
            "axes.titlepad": 10,
            "font.size": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.75,
        }
    )


def best_family_rows(
    df: pd.DataFrame,
    weather_mode: str,
    family: str,
    modes: list[str],
    regime: str = "per_building",
) -> pd.DataFrame:
    subset = df[
        (df["regime"] == regime)
        & (df["weather_mode"] == weather_mode)
        & (df["model_family"] == family)
        & (df["mode"].isin(modes))
    ].copy()
    subset["horizon_h"] = subset["horizon_h"].astype(int)
    subset["wape_mean"] = subset["wape_mean"].astype(float)
    subset["rmse_mean"] = subset["rmse_mean"].astype(float)
    subset = subset.sort_values(["horizon_h", "wape_mean", "rmse_mean"])
    return subset.groupby("horizon_h", as_index=False).first().sort_values("horizon_h")


def aggregated_load_context() -> pd.DataFrame:
    df = pd.read_csv(BASE_CONTEXT_CSV)
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["median_observed_load_kwh"] = df["median_observed_load_kwh"].astype(float)
    return (
        df.groupby("horizon_h", as_index=False)["median_observed_load_kwh"]
        .median()
        .sort_values("horizon_h")
    )


def figure_path(name: str) -> Path:
    return FIGURES_DIR / f"{name}.png"


def add_baseline_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Model_label"] = df["Model"].map(BASELINE_MODEL_LABELS).fillna(df["Model"])
    return df


def add_panel_label(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, fontsize=13, fontweight="bold", pad=10)


def add_heatmap_annotations(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    dark_threshold: float,
    fontsize: float = 8.0,
    threshold: float = 0.0,
) -> None:
    values = matrix.to_numpy()
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = float(values[row_idx, col_idx])
            label = f"{value:.2f}" if abs(value) >= threshold else ""
            if not label:
                continue
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white" if value >= dark_threshold else "#222222",
            )


def add_all_heatmap_labels(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    signed: bool,
    dark_threshold: float,
    fontsize: float,
) -> None:
    values = matrix.to_numpy()
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = float(values[row_idx, col_idx])
            if signed:
                label = "0.00" if abs(value) < 0.005 else f"{value:+.2f}"
                is_dark = abs(value) >= dark_threshold
            else:
                label = f"{value:.2f}"
                is_dark = value >= dark_threshold
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white" if is_dark else "#222222",
            )


def observed_daily_demand_matrix() -> pd.DataFrame:
    frames: list[pd.Series] = []
    for building, path in OBSERVED_BUILDING_CSVS.items():
        df = pd.read_csv(path, parse_dates=["datetime"])
        daily = (
            df.set_index("datetime")["heat_target_mwh"]
            .resample("D")
            .sum(min_count=1)
            .rename(building)
        )
        frames.append(daily)
    return pd.concat(frames, axis=1).sort_index()


def plot_observed_temperature_demand_context() -> None:
    weather = pd.read_csv(WEATHER_CLEAN_CSV, parse_dates=["datetime"])
    temp_daily = (
        weather.set_index("datetime")["wx_outdoor_temp_c"]
        .resample("D")
        .mean()
        .sort_index()
    )
    demand_daily = observed_daily_demand_matrix().sum(axis=1, min_count=1)

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.1), sharex=True)
    panels = [
        (axes[0], temp_daily, "Outdoor air temperature", "Temp. (C)", "#2a6f8f"),
        (axes[1], demand_daily, "Six-building portfolio heat demand", "Demand (MWh/day)", "#c46a2d"),
    ]

    for ax, series, title, ylabel, color in panels:
        rolling = series.rolling(14, min_periods=3).mean()
        ax.plot(series.index, series.values, color="#b8b8b8", linewidth=0.8, alpha=0.65, label="Daily value")
        ax.plot(rolling.index, rolling.values, color=color, linewidth=2.1, label="14-day rolling mean")
        ax.axvline(TRAIN_TEST_SPLIT, color="#222222", linestyle="--", linewidth=1.2)
        ax.grid(True, color="#d8d8d8", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="both", labelsize=10)
        add_panel_label(ax, title)

    axes[0].legend(loc="upper right", ncol=2, fontsize=10)
    axes[1].set_xlim(pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31"))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.subplots_adjust(left=0.10)
    fig.savefig(figure_path("observed_temperature_demand_context"), bbox_inches="tight")
    plt.close(fig)


def plot_observed_demand_building_distribution() -> None:
    daily = observed_daily_demand_matrix()
    stats = pd.DataFrame(
        {
            "min": daily.min(axis=1, skipna=True),
            "q1": daily.quantile(0.25, axis=1),
            "median": daily.median(axis=1, skipna=True),
            "mean": daily.mean(axis=1, skipna=True),
            "q3": daily.quantile(0.75, axis=1),
            "max": daily.max(axis=1, skipna=True),
        }
    ).rolling(14, min_periods=3).mean()

    fig, ax = plt.subplots(figsize=(11.2, 4.1))
    ax.fill_between(stats.index, stats["min"], stats["max"], color="#d3d3d3", alpha=0.75, label="Min-max")
    ax.fill_between(stats.index, stats["q1"], stats["q3"], color="#8a8a8a", alpha=0.85, label="Q1-Q3")
    ax.plot(stats.index, stats["median"], color="#111111", linewidth=1.9, label="Median")
    ax.plot(stats.index, stats["mean"], color="#1f77b4", linewidth=2.1, label="Mean")
    ax.axvline(TRAIN_TEST_SPLIT, color="#222222", linestyle="--", linewidth=1.2, label="2024 test start")
    ax.set_ylabel("Daily building heat demand (MWh)")
    ax.set_xlabel("Date")
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, color="#d8d8d8", linewidth=0.8)
    ax.legend(loc="upper right", ncol=5, fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(left=0.10)
    fig.savefig(figure_path("observed_demand_building_distribution"), bbox_inches="tight")
    plt.close(fig)


def plot_baseline_point_horizon_portfolio_curves() -> None:
    df = pd.read_csv(BASELINE_POINT_SUMMARY_CSV)
    df = df[
        (df["Scope"] == "common_core")
        & (df["Model"].isin(BASELINE_MODEL_ORDER))
        & (df["Horizon_h"].isin(BASELINE_POINT_HORIZONS))
    ].copy()
    df["Horizon_h"] = df["Horizon_h"].astype(int)
    df["Model"] = pd.Categorical(df["Model"], BASELINE_MODEL_ORDER, ordered=True)
    df = add_baseline_labels(df).sort_values(["Model", "Horizon_h"])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), sharex=True)
    metrics = [
        ("WAPE_pct_mean", "Mean WAPE (%)"),
        ("RMSE_mean", "Mean RMSE (kWh)"),
    ]
    for ax, (metric, y_label) in zip(axes, metrics):
        sns.lineplot(
            data=df,
            x="Horizon_h",
            y=metric,
            hue="Model_label",
            style="Model_label",
            hue_order=[BASELINE_MODEL_LABELS[m] for m in BASELINE_MODEL_ORDER],
            style_order=[BASELINE_MODEL_LABELS[m] for m in BASELINE_MODEL_ORDER],
            palette=BASELINE_MODEL_COLORS,
            markers=BASELINE_MODEL_MARKERS,
            dashes=False,
            linewidth=2.1,
            markersize=5.8,
            ax=ax,
            legend=False,
        )
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_ylabel(y_label)
        ax.set_xticks(BASELINE_POINT_HORIZONS)
        ax.grid(True, color="#d8d8d8", linewidth=0.8)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=BASELINE_MODEL_COLORS[BASELINE_MODEL_LABELS[model]],
            marker=BASELINE_MODEL_MARKERS[BASELINE_MODEL_LABELS[model]],
            linewidth=2.1,
            markersize=5.8,
            label=BASELINE_MODEL_LABELS[model],
        )
        for model in BASELINE_MODEL_ORDER
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(figure_path("baseline_point_horizon_portfolio_curves"), bbox_inches="tight")
    plt.close(fig)


def plot_baseline_point_horizon_u05_traces() -> None:
    df = pd.read_csv(BASELINE_POINT_TRACE_CSV, parse_dates=["datetime"])
    df = df[
        df["Horizon_h"].isin(BASELINE_POINT_HORIZONS)
        & df["Model"].isin(BASELINE_MODEL_ORDER)
    ].copy()
    df["Model"] = pd.Categorical(df["Model"], BASELINE_MODEL_ORDER, ordered=True)
    df = add_baseline_labels(df).sort_values(["Horizon_h", "Model", "datetime"])

    actual_df = (
        df[["Horizon_h", "datetime", "actual_kwh"]]
        .drop_duplicates()
        .assign(Model_label="Actual")
        .rename(columns={"actual_kwh": "load_kwh"})
    )
    pred_df = df.rename(columns={"pred_kwh": "load_kwh"})

    fig, axes = plt.subplots(
        len(BASELINE_POINT_HORIZONS),
        2,
        figsize=(12.8, 11.8),
        sharex="col",
    )
    for row_idx, horizon in enumerate(BASELINE_POINT_HORIZONS):
        horizon_pred = pred_df[pred_df["Horizon_h"] == horizon]
        horizon_actual = actual_df[actual_df["Horizon_h"] == horizon]
        trace_ax = axes[row_idx, 0]
        error_ax = axes[row_idx, 1]

        sns.lineplot(
            data=horizon_actual,
            x="datetime",
            y="load_kwh",
            color="black",
            linewidth=2.1,
            ax=trace_ax,
            label="Actual",
            legend=False,
        )
        sns.lineplot(
            data=horizon_pred,
            x="datetime",
            y="load_kwh",
            hue="Model_label",
            style="Model_label",
            hue_order=[BASELINE_MODEL_LABELS[m] for m in BASELINE_MODEL_ORDER],
            style_order=[BASELINE_MODEL_LABELS[m] for m in BASELINE_MODEL_ORDER],
            palette=BASELINE_MODEL_COLORS,
            dashes=False,
            linewidth=1.45,
            alpha=0.86,
            ax=trace_ax,
            legend=False,
        )
        sns.lineplot(
            data=horizon_pred,
            x="datetime",
            y="abs_error_kwh",
            hue="Model_label",
            style="Model_label",
            hue_order=[BASELINE_MODEL_LABELS[m] for m in BASELINE_MODEL_ORDER],
            style_order=[BASELINE_MODEL_LABELS[m] for m in BASELINE_MODEL_ORDER],
            palette=BASELINE_MODEL_COLORS,
            dashes=False,
            linewidth=1.45,
            alpha=0.86,
            ax=error_ax,
            legend=False,
        )

        add_panel_label(trace_ax, f"{horizon}h actual and predicted")
        add_panel_label(error_ax, f"{horizon}h absolute error")
        trace_ax.set_ylabel("Load (kWh)")
        error_ax.set_ylabel("Absolute error (kWh)")
        trace_ax.grid(True, color="#d8d8d8", linewidth=0.8)
        error_ax.grid(True, color="#d8d8d8", linewidth=0.8)
        trace_ax.tick_params(axis="x", labelrotation=25)
        error_ax.tick_params(axis="x", labelrotation=25)
        if row_idx < len(BASELINE_POINT_HORIZONS) - 1:
            trace_ax.set_xlabel("")
            error_ax.set_xlabel("")
        else:
            trace_ax.set_xlabel("Datetime")
            error_ax.set_xlabel("Datetime")

    legend_handles = [
        mlines.Line2D([], [], color="black", linewidth=2.1, label="Actual"),
        *[
            mlines.Line2D(
                [],
                [],
                color=BASELINE_MODEL_COLORS[BASELINE_MODEL_LABELS[model]],
                linewidth=1.8,
                label=BASELINE_MODEL_LABELS[model],
            )
            for model in BASELINE_MODEL_ORDER
        ],
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figure_path("baseline_point_horizon_U05_traces"), bbox_inches="tight")
    plt.close(fig)


def plot_baseline_train_test_weekly_overlay_u05() -> None:
    demand = pd.read_csv(
        U05_CLEAN_CSV,
        usecols=["datetime", "heat_target_mwh"],
        parse_dates=["datetime"],
    )
    df = demand.copy()
    df["actual_demand_kwh"] = df["heat_target_mwh"].astype(float) * 1000.0
    df["split"] = "Training weeks"
    df.loc[df["datetime"] >= TRAIN_TEST_SPLIT, "split"] = "Test weeks"
    df["week_start"] = df["datetime"].dt.to_period("W-SUN").dt.start_time
    df["week_hour"] = df["datetime"].dt.dayofweek * 24 + df["datetime"].dt.hour

    value_col = "actual_demand_kwh"
    fig, ax = plt.subplots(figsize=(12.4, 4.7))
    train_color = "#c44e52"
    test_color = "#1f77b4"
    tick_positions = [0, 24, 48, 72, 96, 120, 144, 167]
    tick_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", ""]

    train = df[df["datetime"] < TRAIN_TEST_SPLIT].copy()
    test = df[df["datetime"] >= TRAIN_TEST_SPLIT].copy()

    for week_df, color in [
        *[(week_df, train_color) for _, week_df in train.dropna(subset=[value_col]).groupby("week_start", sort=True)],
        *[(week_df, test_color) for _, week_df in test.dropna(subset=[value_col]).groupby("week_start", sort=True)],
    ]:
        if week_df[value_col].notna().sum() < 24:
            continue
        sns.lineplot(
            data=week_df.sort_values("week_hour"),
            x="week_hour",
            y=value_col,
            ax=ax,
            color=color,
            linewidth=0.65,
            alpha=0.10,
            estimator=None,
            legend=False,
        )

    train_mean = (
        train.dropna(subset=[value_col])
        .groupby("week_hour", as_index=False)[value_col]
        .mean()
        .sort_values("week_hour")
    )
    test_mean = (
        test.dropna(subset=[value_col])
        .groupby("week_hour", as_index=False)[value_col]
        .mean()
        .sort_values("week_hour")
    )
    sns.lineplot(
        data=train_mean,
        x="week_hour",
        y=value_col,
        ax=ax,
        color=train_color,
        linewidth=2.8,
        estimator=None,
        legend=False,
    )
    sns.lineplot(
        data=test_mean,
        x="week_hour",
        y=value_col,
        ax=ax,
        color=test_color,
        linewidth=2.8,
        estimator=None,
        legend=False,
    )

    ax.set_ylabel("Actual demand (kWh)")
    ax.set_xlabel("Hour of week")
    ax.set_xlim(0, 167)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, color="#d8d8d8", linewidth=0.8)
    legend_handles = [
        mlines.Line2D([], [], color=train_color, alpha=0.22, linewidth=1.2, label="Training weeks"),
        mlines.Line2D([], [], color=test_color, alpha=0.22, linewidth=1.2, label="2024 test weeks"),
        mlines.Line2D([], [], color=train_color, linewidth=2.8, label="Training mean"),
        mlines.Line2D([], [], color=test_color, linewidth=2.8, label="2024 test mean"),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(figure_path("baseline_train_test_weekly_overlay_u05"), bbox_inches="tight")
    plt.close(fig)


def best_metric_rows(
    df: pd.DataFrame,
    weather_mode: str,
    family: str,
    modes: list[str],
    regime: str = "per_building",
) -> pd.DataFrame:
    subset = df[
        (df["regime"] == regime)
        & (df["weather_mode"] == weather_mode)
        & (df["model_family"] == family)
        & (df["mode"].isin(modes))
    ].copy()
    subset["horizon_h"] = subset["horizon_h"].astype(int)
    subset["wape_pct"] = subset["wape_pct"].astype(float)
    subset["rmse"] = subset["rmse"].astype(float)
    subset = subset.sort_values(["building", "horizon_h", "wape_pct", "rmse"])
    return (
        subset.groupby(["building", "horizon_h"], as_index=False)
        .first()
        .sort_values(["building", "horizon_h"])
    )


def plot_model_family_best_wape(regime: str = "per_building", output_name: str = "model_family_best_wape_by_weather") -> None:
    df = pd.read_csv(BASE_COMPARISON_CSV)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), sharey=True)

    colors = MODEL_FAMILY_COLORS
    markers = MODEL_FAMILY_MARKERS

    for ax, weather in zip(axes, WEATHER_ORDER):
        lstm = best_family_rows(df, weather, "lstm", ["M0", "M2", "M4"], regime=regime)
        xgb = best_family_rows(df, weather, "xgboost", ["M0", "M2", "M4"], regime=regime)
        plot_df = pd.concat(
            [
                lstm.assign(family_label="LSTM"),
                xgb.assign(family_label="XGBoost"),
            ],
            ignore_index=True,
        )
        sns.lineplot(
            data=plot_df,
            x="horizon_h",
            y="wape_mean",
            hue="family_label",
            style="family_label",
            hue_order=["LSTM", "XGBoost"],
            style_order=["LSTM", "XGBoost"],
            palette=colors,
            markers=markers,
            dashes=False,
            linewidth=2.2,
            markersize=6.5,
            ax=ax,
            legend=False,
        )
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Horizon (h)")
        ax.grid(True, color="#d8d8d8", linewidth=0.8)
        add_panel_label(ax, weather)

    axes[0].set_ylabel("Mean WAPE (%)")
    legend_handles = [
        mlines.Line2D([], [], color=colors["LSTM"], marker=markers["LSTM"], linewidth=2.2, label="LSTM"),
        mlines.Line2D([], [], color=colors["XGBoost"], marker=markers["XGBoost"], linewidth=2.2, label="XGBoost"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figure_path(output_name), bbox_inches="tight")
    plt.close(fig)


def plot_model_family_weather_scope_comparison(weather: str) -> None:
    df = pd.read_csv(BASE_COMPARISON_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)

    colors = MODEL_FAMILY_COLORS
    markers = MODEL_FAMILY_MARKERS
    scopes = [
        ("per_building", "Per-building"),
        ("pooled_same_buildings", "Pooled"),
    ]

    for ax, (regime, panel_label) in zip(axes, scopes):
        lstm = best_family_rows(df, weather, "lstm", ["M0", "M2", "M4"], regime=regime)
        xgb = best_family_rows(df, weather, "xgboost", ["M0", "M2", "M4"], regime=regime)
        plot_df = pd.concat(
            [
                lstm.assign(family_label="LSTM"),
                xgb.assign(family_label="XGBoost"),
            ],
            ignore_index=True,
        )
        plot_df["horizon_pos"] = plot_df["horizon_h"].map(HORIZON_POSITION_MAP)
        sns.lineplot(
            data=plot_df,
            x="horizon_pos",
            y="wape_mean",
            hue="family_label",
            style="family_label",
            hue_order=["LSTM", "XGBoost"],
            style_order=["LSTM", "XGBoost"],
            palette=colors,
            markers=markers,
            dashes=False,
            linewidth=2.2,
            markersize=6.2,
            ax=ax,
            legend=False,
        )
        ax.set_xticks(HORIZON_POSITIONS)
        ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=10)
        ax.set_xlabel("Horizon (h)")
        ax.grid(True, color="#d8d8d8", linewidth=0.8)
        add_panel_label(ax, panel_label)

    axes[0].set_ylabel("Mean WAPE (%)")
    legend_handles = [
        mlines.Line2D([], [], color=colors["LSTM"], marker=markers["LSTM"], linewidth=2.2, label="LSTM"),
        mlines.Line2D([], [], color=colors["XGBoost"], marker=markers["XGBoost"], linewidth=2.2, label="XGBoost"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figure_path(f"model_family_wape_scope_compare_{weather.lower()}"), bbox_inches="tight")
    plt.close(fig)


def plot_model_family_weather_regime_grid() -> None:
    df = pd.read_csv(MODEL_FAMILY_COMPARISON_CSV)
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["horizon_pos"] = df["horizon_h"].map(HORIZON_POSITION_MAP)
    scopes = [
        ("per_building", "Per-building"),
        ("pooled_same_buildings", "Pooled same-buildings"),
    ]
    families = [("lstm", "LSTM"), ("xgboost", "XGBoost")]

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.4), sharex=True, sharey=True)
    for row_idx, (regime, scope_label) in enumerate(scopes):
        for col_idx, weather in enumerate(WEATHER_ORDER):
            ax = axes[row_idx, col_idx]
            for family, family_label in families:
                subset = df[
                    (df["regime"] == regime)
                    & (df["weather_mode"] == weather)
                    & (df["model_family"] == family)
                ].copy()
                best = (
                    subset.sort_values(["horizon_h", "wape_mean", "rmse_mean"])
                    .groupby("horizon_h", as_index=False)
                    .first()
                    .sort_values("horizon_h")
                )
                ax.plot(
                    best["horizon_pos"],
                    best["wape_mean"],
                    color=MODEL_FAMILY_COLORS[family_label],
                    marker=MODEL_FAMILY_MARKERS[family_label],
                    linewidth=2.1,
                    markersize=5.4,
                    label=family_label,
                )
            ax.set_xticks(HORIZON_POSITIONS)
            ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=9)
            ax.tick_params(axis="both", labelsize=9)
            ax.grid(True, color="#d8d8d8", linewidth=0.8)
            title = f"{scope_label}; {weather}"
            add_panel_label(ax, title)
            ax.set_xlabel("")
            ax.set_ylabel("")

    legend_handles = [
        mlines.Line2D([], [], color=MODEL_FAMILY_COLORS["LSTM"], marker=MODEL_FAMILY_MARKERS["LSTM"], linewidth=2.1, label="LSTM"),
        mlines.Line2D([], [], color=MODEL_FAMILY_COLORS["XGBoost"], marker=MODEL_FAMILY_MARKERS["XGBoost"], linewidth=2.1, label="XGBoost"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=11)
    fig.supxlabel("Horizon (h)", fontsize=12)
    fig.supylabel("Best-mode WAPE (%)", fontsize=12)
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.95))
    fig.savefig(figure_path("model_family_weather_regime_grid"), bbox_inches="tight")
    plt.close(fig)


def plot_model_family_weather_regime_pair(weather: str, output_name: str) -> None:
    df = pd.read_csv(MODEL_FAMILY_COMPARISON_CSV)
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["horizon_pos"] = df["horizon_h"].map(HORIZON_POSITION_MAP)
    scopes = [
        ("per_building", "Per-building"),
        ("pooled_same_buildings", "Pooled same-buildings"),
    ]
    families = [("lstm", "LSTM"), ("xgboost", "XGBoost")]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), sharey=True)
    for ax, (regime, scope_label) in zip(axes, scopes):
        for family, family_label in families:
            subset = df[
                (df["regime"] == regime)
                & (df["weather_mode"] == weather)
                & (df["model_family"] == family)
            ].copy()
            best = (
                subset.sort_values(["horizon_h", "wape_mean", "rmse_mean"])
                .groupby("horizon_h", as_index=False)
                .first()
                .sort_values("horizon_h")
            )
            ax.plot(
                best["horizon_pos"],
                best["wape_mean"],
                color=MODEL_FAMILY_COLORS[family_label],
                marker=MODEL_FAMILY_MARKERS[family_label],
                linewidth=2.2,
                markersize=6.0,
                label=family_label,
            )
        ax.set_xticks(HORIZON_POSITIONS)
        ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=10)
        ax.set_xlabel("Horizon (h)")
        ax.grid(True, color="#d8d8d8", linewidth=0.8)
        add_panel_label(ax, scope_label)

    axes[0].set_ylabel("Best-mode WAPE (%)")
    legend_handles = [
        mlines.Line2D([], [], color=MODEL_FAMILY_COLORS["LSTM"], marker=MODEL_FAMILY_MARKERS["LSTM"], linewidth=2.2, label="LSTM"),
        mlines.Line2D([], [], color=MODEL_FAMILY_COLORS["XGBoost"], marker=MODEL_FAMILY_MARKERS["XGBoost"], linewidth=2.2, label="XGBoost"),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figure_path(output_name), bbox_inches="tight")
    plt.close(fig)


def plot_model_family_weather_regime_fw0() -> None:
    plot_model_family_weather_regime_pair("FW0", "model_family_weather_regime_fw0")


def plot_model_family_weather_regime_fw2() -> None:
    plot_model_family_weather_regime_pair("FW2", "model_family_weather_regime_fw2")


def plot_model_family_weather_regime_fw1() -> None:
    plot_model_family_weather_regime_pair("FW1", "model_family_weather_regime_fw1")


def plot_model_family_best_rmse() -> None:
    df = pd.read_csv(BASE_COMPARISON_CSV)
    context = aggregated_load_context()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), sharey=True)

    colors = MODEL_FAMILY_COLORS
    markers = MODEL_FAMILY_MARKERS

    for ax, weather in zip(axes, WEATHER_ORDER):
        lstm = best_family_rows(df, weather, "lstm", ["M0", "M2", "M4"], regime="per_building")
        xgb = best_family_rows(df, weather, "xgboost", ["M0", "M2", "M4"], regime="per_building")
        plot_df = pd.concat(
            [
                lstm.assign(family_label="LSTM", plot_value=lstm["rmse_mean"].astype(float)),
                xgb.assign(family_label="XGBoost", plot_value=xgb["rmse_mean"].astype(float)),
                context.assign(family_label="Median observed cumulative load", plot_value=context["median_observed_load_kwh"]),
            ],
            ignore_index=True,
        )
        sns.lineplot(
            data=plot_df[plot_df["family_label"].isin(["LSTM", "XGBoost"])],
            x="horizon_h",
            y="plot_value",
            hue="family_label",
            style="family_label",
            hue_order=["LSTM", "XGBoost"],
            style_order=["LSTM", "XGBoost"],
            palette=colors,
            markers=markers,
            dashes=False,
            linewidth=2.2,
            markersize=6.5,
            ax=ax,
            legend=False,
        )
        sns.lineplot(
            data=plot_df[plot_df["family_label"] == "Median observed cumulative load"],
            x="horizon_h",
            y="plot_value",
            color="#666666",
            linestyle="--",
            linewidth=1.8,
            ax=ax,
            legend=False,
        )
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Horizon (h)")
        ax.grid(True, color="#d8d8d8", linewidth=0.8)
        add_panel_label(ax, weather)

    axes[0].set_ylabel("Cumulative RMSE (kWh)")
    legend_handles = [
        mlines.Line2D([], [], color="#c66b22", marker="o", linewidth=2.2, label="LSTM"),
        mlines.Line2D([], [], color="#17739f", marker="s", linewidth=2.2, label="XGBoost"),
        mlines.Line2D([], [], color="#666666", linestyle="--", linewidth=1.8, label="Median observed cumulative load"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figure_path("model_family_best_rmse_by_weather"), bbox_inches="tight")
    plt.close(fig)


def plot_fw2_mode_extension() -> None:
    df = pd.read_csv(SUPPLEMENT_COMPARISON_CSV)
    df = df[(df["regime"] == "per_building") & (df["weather_mode"] == "FW2")].copy()
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["wape_mean"] = df["wape_mean"].astype(float)

    mode_styles = {
        "M0": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "M1": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
        "M2": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
        "M3": {"color": "#d62728", "marker": "D", "linestyle": ":"},
        "M4": {"color": "#9467bd", "marker": "P", "linestyle": "-"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), sharey=True)
    for ax, family, panel_label in zip(axes, ["lstm", "xgboost"], ["LSTM", "XGBoost"]):
        fam = df[df["model_family"] == family]
        for mode in ["M0", "M1", "M2", "M3", "M4"]:
            series = fam[fam["mode"] == mode].sort_values("horizon_h")
            style = mode_styles[mode]
            sns.lineplot(
                data=series,
                x="horizon_h",
                y="wape_mean",
                label=mode,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                markersize=6.2,
                ax=ax,
                legend=False,
            )
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Horizon (h)")
        ax.grid(True, color="#d8d8d8", linewidth=0.8)
        add_panel_label(ax, panel_label)

    axes[0].set_ylabel("Mean WAPE (%)")
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=mode_styles[mode]["color"],
            marker=mode_styles[mode]["marker"],
            linestyle=mode_styles[mode]["linestyle"],
            linewidth=2.0,
            markersize=6.2,
            label=mode,
        )
        for mode in ["M0", "M1", "M2", "M3", "M4"]
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figure_path("model_family_fw2_mode_extension"), bbox_inches="tight")
    plt.close(fig)


def plot_fw2_mode_delta_heatmap() -> None:
    df = pd.read_csv(SUPPLEMENT_COMPARISON_CSV)
    df = df[(df["regime"] == "per_building") & (df["weather_mode"] == "FW2")].copy()
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["wape_mean"] = df["wape_mean"].astype(float)

    rows = []
    labels = []
    for family, family_label in [("lstm", "LSTM"), ("xgboost", "XGBoost")]:
        fam = df[df["model_family"] == family]
        base = (
            fam[fam["mode"] == "M0"]
            .set_index("horizon_h")["wape_mean"]
            .reindex(HORIZONS)
        )
        for mode in ["M1", "M2", "M3", "M4"]:
            series = (
                fam[fam["mode"] == mode]
                .set_index("horizon_h")["wape_mean"]
                .reindex(HORIZONS)
            )
            rows.append(series - base)
            labels.append(f"{family_label} {mode} - M0")

    matrix = pd.DataFrame(rows, index=labels, columns=HORIZONS)
    vmax = max(abs(float(matrix.min().min())), abs(float(matrix.max().max())))
    vmax = max(vmax, 0.1)

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "WAPE change vs M0 (percentage points)"},
    )
    add_all_heatmap_labels(ax, matrix, signed=True, dark_threshold=max(vmax * 0.55, 0.35), fontsize=8.2)
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("Mode comparison")
    ax.set_xticklabels([str(h) for h in HORIZONS], rotation=0)
    ax.set_yticklabels(matrix.index, rotation=0)
    fig.tight_layout()
    fig.savefig(figure_path("model_family_fw2_mode_delta_heatmap"), bbox_inches="tight")
    plt.close(fig)


def plot_fw2_mode_delta_lines() -> None:
    df = pd.read_csv(SUPPLEMENT_COMPARISON_CSV)
    df = df[(df["regime"] == "per_building") & (df["weather_mode"] == "FW2")].copy()
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["wape_mean"] = df["wape_mean"].astype(float)

    mode_colors = {
        "M1": "#ff7f0e",
        "M2": "#2ca02c",
        "M3": "#d62728",
        "M4": "#9467bd",
    }

    records = []
    for family, family_label in [("lstm", "LSTM"), ("xgboost", "XGBoost")]:
        fam = df[df["model_family"] == family]
        base = (
            fam[fam["mode"] == "M0"]
            .set_index("horizon_h")["wape_mean"]
            .reindex(HORIZONS)
        )
        for mode in ["M1", "M2", "M3", "M4"]:
            series = (
                fam[fam["mode"] == mode]
                .set_index("horizon_h")["wape_mean"]
                .reindex(HORIZONS)
            )
            delta = series - base
            for horizon, value in delta.items():
                records.append(
                    {
                        "family": family_label,
                        "mode": mode,
                        "horizon_h": horizon,
                        "delta_wape": float(value),
                    }
                )

    delta_df = pd.DataFrame(records)
    y_abs = max(abs(float(delta_df["delta_wape"].min())), abs(float(delta_df["delta_wape"].max())))
    y_abs = max(y_abs * 1.12, 0.35)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8), sharey=True)
    for ax, family in zip(axes, ["LSTM", "XGBoost"]):
        fam = delta_df[delta_df["family"] == family]
        ax.axhspan(-y_abs, 0, color="#dcefdc", alpha=0.45, zorder=0)
        ax.axhspan(0, y_abs, color="#f5d9d9", alpha=0.35, zorder=0)
        ax.axhline(0, color="#333333", linewidth=1.2)
        for mode in ["M1", "M2", "M3", "M4"]:
            series = fam[fam["mode"] == mode].sort_values("horizon_h")
            sns.lineplot(
                data=series,
                x="horizon_h",
                y="delta_wape",
                marker="o",
                linewidth=2.0,
                markersize=5.8,
                color=mode_colors[mode],
                label=mode,
                ax=ax,
                legend=False,
            )
        ax.set_title(family, fontsize=13.5, fontweight="bold")
        ax.set_xlabel("Horizon (h)")
        ax.set_xticks(HORIZONS)
        ax.set_ylim(-y_abs, y_abs)
        ax.grid(True, color="#d8d8d8", linewidth=0.8)
        ax.text(
            0.02,
            0.04,
            "Better than M0",
            transform=ax.transAxes,
            fontsize=9.5,
            color="#2f6b2f",
            ha="left",
            va="bottom",
        )
        ax.text(
            0.02,
            0.96,
            "Worse than M0",
            transform=ax.transAxes,
            fontsize=9.5,
            color="#8a3a3a",
            ha="left",
            va="top",
        )

    axes[0].set_ylabel("WAPE difference from M0 (percentage points)")
    legend_handles = [
        mlines.Line2D([], [], color=mode_colors[mode], marker="o", linewidth=2.0, markersize=5.8, label=mode)
        for mode in ["M1", "M2", "M3", "M4"]
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figure_path("model_family_fw2_mode_delta_lines"), bbox_inches="tight")
    plt.close(fig)


def plot_weather_mode_example() -> None:
    usecols = [
        "regime",
        "building",
        "model_family",
        "mode",
        "weather_mode",
        "horizon_h",
        "datetime",
        "y_true",
        "y_pred",
        "is_heating_eval",
    ]
    chunks = []
    for chunk in pd.read_csv(
        SUPPLEMENT_PREDICTIONS_CSV,
        usecols=usecols,
        parse_dates=["datetime"],
        chunksize=250000,
    ):
        mask = (
            (chunk["regime"] == "per_building")
            & (chunk["building"] == "U02B")
            & (chunk["model_family"] == "xgboost")
            & (chunk["mode"] == "M4")
            & (chunk["horizon_h"].astype(int) == 16)
            & (chunk["is_heating_eval"] == True)
        )
        filtered = chunk.loc[mask]
        if not filtered.empty:
            chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)
    df = df[
        (df["datetime"] >= "2024-01-04")
        & (df["datetime"] <= "2024-01-15")
    ]

    pred = (
        df.pivot(index="datetime", columns="weather_mode", values="y_pred")
        .reindex(columns=["FW0", "FW2", "FW1"])
        .dropna()
    )
    actual = (
        df.drop_duplicates("datetime")
        .set_index("datetime")["y_true"]
        .reindex(pred.index)
    )

    fig, ax = plt.subplots(figsize=(11.2, 4.9))
    actual_plot_df = actual.rename("load_kwh").reset_index()
    sns.lineplot(
        data=actual_plot_df,
        x="datetime",
        y="load_kwh",
        color="black",
        linewidth=2.5,
        ax=ax,
        legend=False,
    )
    pred_plot_df = (
        pred.reset_index()
        .melt(id_vars="datetime", var_name="weather_mode", value_name="load_kwh")
        .assign(weather_label=lambda frame: frame["weather_mode"].map(WEATHER_LABELS))
    )
    sns.lineplot(
        data=pred_plot_df,
        x="datetime",
        y="load_kwh",
        hue="weather_label",
        style="weather_label",
        hue_order=[WEATHER_LABELS[w] for w in ["FW0", "FW2", "FW1"]],
        style_order=[WEATHER_LABELS[w] for w in ["FW0", "FW2", "FW1"]],
        palette={WEATHER_LABELS[w]: WEATHER_COLORS[w] for w in ["FW0", "FW2", "FW1"]},
        dashes=False,
        linewidth=2.0,
        ax=ax,
        legend=False,
    )

    legend_handles = [
        mlines.Line2D([], [], color="black", linewidth=2.5, label="Actual load"),
        *[
            mlines.Line2D([], [], color=WEATHER_COLORS[weather_mode], linewidth=2.0, label=WEATHER_LABELS[weather_mode])
            for weather_mode in ["FW0", "FW2", "FW1"]
        ],
    ]
    ax.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper left",
        fontsize=9.5,
        ncol=2,
    )

    ax.set_xlabel("Datetime")
    ax.set_ylabel("Cumulative heat target (kWh)")
    ax.grid(True, color="#d8d8d8", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(figure_path("model_family_weather_mode_example"), bbox_inches="tight")
    plt.close(fig)


def plot_bayes_fw2_baseline_families_wape() -> None:
    df = pd.read_csv(BAYES_FW2_BASELINE_WAPE_CSV)
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["horizon_pos"] = df["horizon_h"].map(HORIZON_POSITION_MAP)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)

    for ax, regime_label in zip(axes, ["Per-building", "Pooled"]):
        panel_df = df[df["regime_label"] == regime_label].copy()
        for baseline in BAYES_BASELINE_ORDER:
            series = panel_df[panel_df["baseline"] == baseline].sort_values("horizon_h")
            ax.plot(
                series["horizon_pos"],
                series["mean_wape_pct"],
                color=BAYES_BASELINE_COLORS[baseline],
                marker=BAYES_BASELINE_MARKERS[baseline],
                linewidth=2.2,
                markersize=6.0,
                label=BAYES_BASELINE_LABELS[baseline],
            )
        ax.set_xticks(HORIZON_POSITIONS)
        ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=10)
        ax.set_xlabel("Horizon (h)")
        ax.grid(True, color="#d8d8d8", linewidth=0.8)
        add_panel_label(ax, regime_label)

    axes[0].set_ylabel("Mean WAPE (%)")
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=BAYES_BASELINE_COLORS[baseline],
            marker=BAYES_BASELINE_MARKERS[baseline],
            linewidth=2.2,
            markersize=6.0,
            label=BAYES_BASELINE_LABELS[baseline],
        )
        for baseline in BAYES_BASELINE_ORDER
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figure_path("bayes_fw2_baseline_families_wape"), bbox_inches="tight")
    plt.close(fig)


def plot_bayes_fw2_best_baseline_vs_nonlinear_m4_wape() -> None:
    df = pd.read_csv(BAYES_FW2_BEST_VS_NONLINEAR_M4_WAPE_CSV)
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["horizon_pos"] = df["horizon_h"].map(HORIZON_POSITION_MAP)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)

    for ax, regime_label in zip(axes, ["Per-building", "Pooled"]):
        panel_df = df[df["regime_label"] == regime_label].copy()
        for series_label in BAYES_OVERLAY_ORDER:
            series = panel_df[panel_df["series_label"] == series_label].sort_values("horizon_h")
            ax.plot(
                series["horizon_pos"],
                series["mean_wape_pct"],
                color=BAYES_OVERLAY_COLORS[series_label],
                marker=BAYES_OVERLAY_MARKERS[series_label],
                linestyle=BAYES_OVERLAY_LINESTYLES[series_label],
                linewidth=2.2,
                markersize=6.0,
                label=BAYES_OVERLAY_LABELS[series_label],
            )
        ax.set_xticks(HORIZON_POSITIONS)
        ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=10)
        ax.set_xlabel("Horizon (h)")
        ax.grid(True, color="#d8d8d8", linewidth=0.8)
        add_panel_label(ax, regime_label)

    axes[0].set_ylabel("Mean WAPE (%)")
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=BAYES_OVERLAY_COLORS[series_label],
            marker=BAYES_OVERLAY_MARKERS[series_label],
            linestyle=BAYES_OVERLAY_LINESTYLES[series_label],
            linewidth=2.2,
            markersize=6.0,
            label=BAYES_OVERLAY_LABELS[series_label],
        )
        for series_label in BAYES_OVERLAY_ORDER
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figure_path("bayes_fw2_best_baseline_vs_nonlinear_m4_wape"), bbox_inches="tight")
    plt.close(fig)


def xai_share_matrix(method: str, weather_mode: str | None = None, regime: str = "per_building") -> pd.DataFrame:
    df = pd.read_csv(XAI_GROUP_SHARE_FINE_CSV)
    mask = (
        (df["regime"] == regime)
        & (df["model_family"] == "xgboost")
        & (df["mode"] == "M0")
        & (df["method"] == method)
    )
    if weather_mode is not None:
        mask &= df["weather_mode"] == weather_mode
    df = df[mask].copy()
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["share"] = df["share"].astype(float)
    grouped = (
        df.groupby(["feature_group", "horizon_h"], as_index=False)["share"]
        .mean()
    )
    matrix = grouped.pivot(index="feature_group", columns="horizon_h", values="share")
    matrix = matrix.reindex(FINE_GROUP_ORDER)
    matrix = matrix[[h for h in HORIZONS if h in matrix.columns]]
    return matrix.fillna(0.0)


def xai_group_share_matrix(
    method: str,
    weather_mode: str,
    mode: str,
    regime: str = "per_building",
) -> pd.DataFrame:
    df = pd.read_csv(XAI_GROUP_SHARE_CSV)
    mask = (
        (df["regime"] == regime)
        & (df["model_family"] == "xgboost")
        & (df["method"] == method)
        & (df["weather_mode"] == weather_mode)
        & (df["mode"] == mode)
    )
    df = df[mask].copy()
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["share"] = df["share"].astype(float)
    grouped = (
        df.groupby(["feature_group", "horizon_h"], as_index=False)["share"]
        .mean()
    )
    matrix = grouped.pivot(index="feature_group", columns="horizon_h", values="share")
    present_groups = [group for group in GROUP_ORDER if group in matrix.index]
    matrix = matrix.reindex(present_groups)
    matrix = matrix[[h for h in HORIZONS if h in matrix.columns]]
    return matrix.fillna(0.0)


def xai_group_share_mode_overview(method: str, mode: str, regime: str = "per_building") -> pd.DataFrame:
    df = pd.read_csv(XAI_GROUP_SHARE_CSV)
    df = df[
        (df["regime"] == regime)
        & (df["model_family"] == "xgboost")
        & (df["method"] == method)
        & (df["mode"] == mode)
    ].copy()
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["share"] = df["share"].astype(float)
    grouped = (
        df.groupby(["feature_group", "horizon_h"], as_index=False)["share"]
        .mean()
    )
    matrix = grouped.pivot(index="feature_group", columns="horizon_h", values="share")
    present_groups = [group for group in GROUP_ORDER if group in matrix.index]
    matrix = matrix.reindex(present_groups)
    matrix = matrix[[h for h in HORIZONS if h in matrix.columns]]
    return matrix.fillna(0.0)


def plot_xai_driver_shares(method: str, output_name: str) -> None:
    matrix = xai_share_matrix(method)
    fig, ax = plt.subplots(figsize=(9.3, 5.2))
    vmax = float(matrix.max().max())
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlGnBu",
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Mean share"},
        linewidths=0.35,
        linecolor="white",
    )
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("Fine feature group")
    ax.set_yticklabels([FINE_GROUP_LABELS[g] for g in matrix.index], rotation=0)
    ax.set_xticklabels([str(int(v.get_text())) for v in ax.get_xticklabels()], rotation=0)
    add_heatmap_annotations(ax, matrix, dark_threshold=max(vmax * 0.55, 0.30), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(figure_path(output_name), bbox_inches="tight")
    plt.close(fig)


def plot_xai_weather_role_single(weather: str) -> None:
    matrix = xai_share_matrix("pfi", weather, regime="per_building")
    vmax = max(float(xai_share_matrix("pfi", w, regime="per_building").max().max()) for w in WEATHER_ORDER)

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="rocket_r",
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Mean PFI share"},
        linewidths=0.35,
        linecolor="white",
    )
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("Fine feature group")
    ax.set_yticklabels([FINE_GROUP_LABELS[g] for g in matrix.index], rotation=0, fontsize=10)
    ax.set_xticklabels([str(int(v.get_text())) for v in ax.get_xticklabels()], rotation=0)
    add_heatmap_annotations(ax, matrix, dark_threshold=max(vmax * 0.55, 0.30), fontsize=7.5, threshold=0.0)
    fig.tight_layout()
    fig.savefig(figure_path(f"xai_weather_roles_fine_pfi_{weather.lower()}"), bbox_inches="tight")
    plt.close(fig)


def plot_xai_mode_transition() -> None:
    df = pd.read_csv(XAI_GROUP_SHARE_FINE_CSV)
    df = df[
        (df["regime"] == "per_building")
        & (df["model_family"] == "xgboost")
        & (df["weather_mode"] == "FW0")
        & (df["method"] == "pfi")
    ].copy()
    df["share"] = df["share"].astype(float)

    grouped = (
        df.groupby(["mode", "feature_group"], as_index=False)["share"]
        .mean()
    )
    pivot = (
        grouped.pivot(index="mode", columns="feature_group", values="share")
        .reindex(columns=FINE_GROUP_ORDER)
        .fillna(0.0)
    )
    transitions = [
        ("M1", "M0", "M1 - M0"),
        ("M2", "M0", "M2 - M0"),
        ("M3", "M2", "M3 - M2"),
        ("M4", "M3", "M4 - M3"),
    ]

    records = []
    for mode_a, mode_b, transition_label in transitions:
        delta = pivot.loc[mode_a] - pivot.loc[mode_b]
        for group, value in delta.items():
            records.append(
                {
                    "transition": transition_label,
                    "feature_group": group,
                    "feature_label": WRAPPED_GROUP_LABELS.get(group, FINE_GROUP_LABELS[group]),
                    "delta_pp": float(value) * 100.0,
                    "abs_delta_pp": abs(float(value) * 100.0),
                }
            )
    plot_df = pd.DataFrame(records)
    x_limit = max(float(plot_df["abs_delta_pp"].max()) * 1.22, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4), sharex=True)
    axes = axes.flatten()
    increase_color = "#b2182b"
    decrease_color = "#2166ac"

    for idx, (ax, (_, _, transition_label)) in enumerate(zip(axes, transitions)):
        panel_df = (
            plot_df[plot_df["transition"] == transition_label]
            .sort_values("abs_delta_pp", ascending=False)
            .reset_index(drop=True)
        )
        label_order = panel_df["feature_label"].tolist()
        sns.barplot(
            data=panel_df,
            x="delta_pp",
            y="feature_label",
            order=label_order,
            ax=ax,
            color="#888888",
            errorbar=None,
        )
        for patch, value in zip(ax.patches, panel_df["delta_pp"]):
            patch.set_facecolor(increase_color if value >= 0 else decrease_color)
            patch.set_alpha(0.88)

        ax.axvline(0, color="#333333", linewidth=1.0)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_xlabel("PFI share change (pp)" if idx >= 2 else "")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.0f}"))
        ax.grid(True, axis="x", color="#d8d8d8", linewidth=0.8)
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", labelsize=8.4)
        ax.text(
            0.03,
            0.93,
            transition_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11.2,
            fontweight="bold",
        )

        label_offset = x_limit * 0.025
        for y_idx, value in enumerate(panel_df["delta_pp"]):
            if abs(value) < 0.05:
                continue
            ax.text(
                value + (label_offset if value >= 0 else -label_offset),
                y_idx,
                f"{value:+.1f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8.8,
                color="#222222",
            )

    legend_handles = [
        mlines.Line2D([], [], color=increase_color, linewidth=7, label="Higher share in later mode"),
        mlines.Line2D([], [], color=decrease_color, linewidth=7, label="Lower share in later mode"),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figure_path("xai_mode_transition_fine_fw0_pfi"), bbox_inches="tight")
    plt.close(fig)


def plot_fw2_mode_group_shares(method: str, output_name: str) -> None:
    modes = ["M0", "M1", "M2", "M3", "M4"]
    matrices = {mode: xai_group_share_matrix(method, "FW2", mode, regime="per_building") for mode in modes}
    vmax = max(float(matrix.max().max()) for matrix in matrices.values())

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2))
    axes = axes.flatten()

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        matrix = matrices[mode]
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="YlGnBu" if method == "pfi" else "YlGnBu",
            vmin=0,
            vmax=vmax,
            cbar=(idx == len(modes) - 1),
            cbar_kws={"label": f"Mean {method.upper()} share"} if idx == len(modes) - 1 else None,
            linewidths=0.35,
            linecolor="white",
        )
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("Feature group")
        ax.set_yticklabels([GROUP_LABELS[g] for g in matrix.index], rotation=0, fontsize=9)
        ax.set_xticklabels([str(int(v.get_text())) for v in ax.get_xticklabels()], rotation=0, fontsize=9)
        add_heatmap_annotations(ax, matrix, dark_threshold=max(vmax * 0.55, 0.25), fontsize=7.3, threshold=0.0)
        add_panel_label(ax, mode)

    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(figure_path(output_name), bbox_inches="tight")
    plt.close(fig)


def plot_mode_feature_share_heatmaps(mode: str) -> None:
    regimes = [
        ("per_building", "Per-building"),
        ("pooled_same_buildings", "Pooled"),
    ]
    matrices = {
        (regime, method): xai_group_share_mode_overview(method, mode, regime=regime)
        for regime, _ in regimes
        for method in ["pfi", "shap"]
    }
    vmax = max(float(m.max().max()) for m in matrices.values())

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.1), sharey=False)
    for row_idx, (regime, row_label) in enumerate(regimes):
        for col_idx, (method, panel_label) in enumerate([("pfi", "PFI"), ("shap", "SHAP")]):
            ax = axes[row_idx, col_idx]
            matrix = matrices[(regime, method)]
            sns.heatmap(
                matrix,
                ax=ax,
                cmap="YlGnBu",
                vmin=0,
                vmax=vmax,
                cbar=(row_idx == 1 and col_idx == 1),
                cbar_kws={"label": f"Mean {method.upper()} share"} if (row_idx == 1 and col_idx == 1) else None,
                linewidths=0.35,
                linecolor="white",
            )
            ax.set_xlabel("Horizon (h)")
            ax.set_ylabel("Feature group")
            ax.set_xticklabels([str(int(v.get_text())) for v in ax.get_xticklabels()], rotation=0, fontsize=9)
            ax.set_yticklabels([GROUP_LABELS[g] for g in matrix.index], rotation=0, fontsize=9)
            add_heatmap_annotations(ax, matrix, dark_threshold=max(vmax * 0.60, 0.35), fontsize=7.6, threshold=0.0)
            ax.set_title(f"{row_label} {panel_label}", fontsize=13.5, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(figure_path(f"xai_mode_feature_shares_{mode.lower()}"), bbox_inches="tight")
    plt.close(fig)


def plot_xai_mode_scope_compare_pfi(mode: str) -> None:
    regimes = [
        ("per_building", "Per-building"),
        ("pooled_same_buildings", "Pooled same-buildings"),
    ]
    matrices = {
        regime: xai_group_share_mode_overview("pfi", mode, regime=regime)
        for regime, _ in regimes
    }
    vmax = max(float(m.max().max()) for m in matrices.values())

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), sharey=False)
    for ax, (regime, panel_label) in zip(axes, regimes):
        matrix = matrices[regime]
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="YlGnBu",
            vmin=0,
            vmax=vmax,
            cbar=(regime == "pooled_same_buildings"),
            cbar_kws={"label": "Mean PFI share"} if regime == "pooled_same_buildings" else None,
            linewidths=0.35,
            linecolor="white",
        )
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("Feature group")
        ax.set_xticklabels([str(int(v.get_text())) for v in ax.get_xticklabels()], rotation=0, fontsize=9)
        ax.set_yticklabels([GROUP_LABELS[g] for g in matrix.index], rotation=0, fontsize=9)
        add_heatmap_annotations(ax, matrix, dark_threshold=max(vmax * 0.60, 0.35), fontsize=7.8, threshold=0.0)
        ax.set_title(panel_label, fontsize=13.5, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(figure_path(f"xai_mode_scope_compare_pfi_{mode.lower()}"), bbox_inches="tight")
    plt.close(fig)


def plot_xai_fw2_m4_pfi_shap_driver_share_for_regime(regime: str, output_name: str, panel_title: str) -> None:
    df = pd.read_csv(XAI_GROUP_SHARE_FINE_CSV)
    df = df[
        (df["regime"] == regime)
        & (df["model_family"] == "xgboost")
        & (df["mode"] == "M4")
        & (df["weather_mode"] == "FW2")
    ].copy()
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["share_pct"] = df["share"].astype(float) * 100.0
    matrices: dict[str, pd.DataFrame] = {}
    for method in ["pfi", "shap"]:
        grouped = (
            df[df["method"] == method]
            .groupby(["feature_group", "horizon_h"], as_index=False)["share_pct"]
            .mean()
        )
        matrix = grouped.pivot(index="feature_group", columns="horizon_h", values="share_pct")
        matrices[method] = (
            matrix.reindex(FINE_GROUP_ORDER)
            .reindex(columns=[h for h in HORIZONS if h in matrix.columns])
            .fillna(0.0)
        )

    vmax = max(float(matrix.max().max()) for matrix in matrices.values())
    vmax = max(vmax, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), sharey=False)
    for ax, method, label in zip(axes, ["pfi", "shap"], ["PFI", "SHAP"]):
        matrix = matrices[method]
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="YlGnBu",
            vmin=0,
            vmax=vmax,
            linewidths=0.35,
            linecolor="white",
            cbar=(method == "shap"),
            cbar_kws={"label": "Mean share (%)"} if method == "shap" else None,
        )
        add_all_heatmap_labels(ax, matrix, signed=False, dark_threshold=max(vmax * 0.55, 20.0), fontsize=6.8)
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("Fine feature group")
        ax.set_yticklabels([FINE_GROUP_LABELS[g] for g in matrix.index], rotation=0, fontsize=9.5)
        ax.set_xticklabels([str(h) for h in matrix.columns], rotation=0, fontsize=9.5)
        ax.set_title(label, fontsize=13.5, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(figure_path(output_name), bbox_inches="tight")
    plt.close(fig)


def xai_fw2_m4_driver_share_matrix(regime: str, method: str) -> pd.DataFrame:
    df = pd.read_csv(XAI_GROUP_SHARE_FINE_CSV)
    df = df[
        (df["regime"] == regime)
        & (df["model_family"] == "xgboost")
        & (df["mode"] == "M4")
        & (df["weather_mode"] == "FW2")
        & (df["method"] == method)
    ].copy()
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["share_pct"] = df["share"].astype(float) * 100.0
    grouped = df.groupby(["feature_group", "horizon_h"], as_index=False)["share_pct"].mean()
    matrix = grouped.pivot(index="feature_group", columns="horizon_h", values="share_pct")
    return (
        matrix.reindex(FINE_GROUP_ORDER)
        .reindex(columns=[h for h in HORIZONS if h in matrix.columns])
        .fillna(0.0)
    )


def plot_xai_fw2_m4_driver_share(regime: str, method: str, output_name: str) -> None:
    matrix = xai_fw2_m4_driver_share_matrix(regime, method)
    comparison_method = "shap" if method == "pfi" else "pfi"
    comparison = xai_fw2_m4_driver_share_matrix(regime, comparison_method)
    vmax = max(float(matrix.max().max()), float(comparison.max().max()), 1.0)

    fig, ax = plt.subplots(figsize=(8.7, 5.55))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlGnBu",
        vmin=0,
        vmax=vmax,
        linewidths=0.35,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": "Mean share (%)", "shrink": 0.88},
    )
    add_all_heatmap_labels(ax, matrix, signed=False, dark_threshold=max(vmax * 0.55, 20.0), fontsize=7.2)
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("Fine feature group")
    ax.set_yticklabels([FINE_GROUP_LABELS[g] for g in matrix.index], rotation=0, fontsize=9.6)
    ax.set_xticklabels([str(h) for h in matrix.columns], rotation=0, fontsize=9.6)
    fig.tight_layout()
    fig.savefig(figure_path(output_name), bbox_inches="tight")
    plt.close(fig)


def plot_xai_fw2_m4_pfi_shap_driver_share() -> None:
    plot_xai_fw2_m4_pfi_shap_driver_share_for_regime(
        "per_building",
        "xai_fw2_m4_pfi_shap_driver_share",
        "FW2 M4 grouped driver shares: per-building",
    )


def plot_xai_fw2_m4_pooled_pfi_shap_driver_share() -> None:
    plot_xai_fw2_m4_pfi_shap_driver_share_for_regime(
        "pooled_same_buildings",
        "xai_fw2_m4_pooled_pfi_shap_driver_share",
        "FW2 M4 grouped driver shares: pooled same-buildings",
    )


def plot_xai_fw2_m4_per_building_pfi_driver_share() -> None:
    plot_xai_fw2_m4_driver_share(
        "per_building",
        "pfi",
        "xai_fw2_m4_per_building_pfi_driver_share",
    )


def plot_xai_fw2_m4_per_building_shap_driver_share() -> None:
    plot_xai_fw2_m4_driver_share(
        "per_building",
        "shap",
        "xai_fw2_m4_per_building_shap_driver_share",
    )


def plot_xai_fw2_m4_pooled_pfi_driver_share() -> None:
    plot_xai_fw2_m4_driver_share(
        "pooled_same_buildings",
        "pfi",
        "xai_fw2_m4_pooled_pfi_driver_share",
    )


def plot_xai_fw2_m4_pooled_shap_driver_share() -> None:
    plot_xai_fw2_m4_driver_share(
        "pooled_same_buildings",
        "shap",
        "xai_fw2_m4_pooled_shap_driver_share",
    )


def plot_xai_fw2_m4_scope_method_driver_share() -> None:
    df = pd.read_csv(XAI_GROUP_SHARE_FINE_CSV)
    df = df[
        (df["model_family"] == "xgboost")
        & (df["mode"] == "M4")
        & (df["weather_mode"] == "FW2")
    ].copy()
    df["horizon_h"] = df["horizon_h"].astype(int)
    df["share_pct"] = df["share"].astype(float) * 100.0

    regimes = [
        ("per_building", "Per-building"),
        ("pooled_same_buildings", "Pooled same-buildings"),
    ]
    methods = [("pfi", "PFI"), ("shap", "SHAP")]
    matrices: dict[tuple[str, str], pd.DataFrame] = {}
    for regime, _ in regimes:
        for method, _ in methods:
            grouped = (
                df[(df["regime"] == regime) & (df["method"] == method)]
                .groupby(["feature_group", "horizon_h"], as_index=False)["share_pct"]
                .mean()
            )
            matrix = grouped.pivot(index="feature_group", columns="horizon_h", values="share_pct")
            matrices[(regime, method)] = (
                matrix.reindex(FINE_GROUP_ORDER)
                .reindex(columns=[h for h in HORIZONS if h in matrix.columns])
                .fillna(0.0)
            )

    vmax = max(float(matrix.max().max()) for matrix in matrices.values())
    vmax = max(vmax, 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.3), sharey=False)
    for row_idx, (regime, regime_label) in enumerate(regimes):
        for col_idx, (method, method_label) in enumerate(methods):
            ax = axes[row_idx, col_idx]
            matrix = matrices[(regime, method)]
            sns.heatmap(
                matrix,
                ax=ax,
                cmap="YlGnBu",
                vmin=0,
                vmax=vmax,
                linewidths=0.35,
                linecolor="white",
                cbar=(row_idx == 1 and col_idx == 1),
                cbar_kws={"label": "Mean share (%)", "shrink": 0.82} if (row_idx == 1 and col_idx == 1) else None,
            )
            ax.set_xlabel("Horizon (h)" if row_idx == 1 else "", fontsize=10)
            ax.set_ylabel("Fine feature group" if col_idx == 0 else "", fontsize=10)
            ax.set_xticklabels([str(h) for h in matrix.columns], rotation=0, fontsize=8)
            ax.set_yticklabels([FINE_GROUP_LABELS[g] for g in matrix.index], rotation=0, fontsize=8)
            ax.set_title(f"{regime_label}; {method_label}", fontsize=11.5, fontweight="bold", pad=8)
            add_all_heatmap_labels(ax, matrix, signed=False, dark_threshold=max(vmax * 0.55, 20.0), fontsize=5.8)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.34, wspace=0.34)
    fig.savefig(figure_path("xai_fw2_m4_scope_method_driver_share"), bbox_inches="tight")
    plt.close(fig)


FIGURE_BUILDERS = {
    "observed_temperature_demand_context": plot_observed_temperature_demand_context,
    "model_family_weather_regime_fw0": plot_model_family_weather_regime_fw0,
    "model_family_weather_regime_fw2": plot_model_family_weather_regime_fw2,
    "model_family_weather_regime_fw1": plot_model_family_weather_regime_fw1,
    "bayes_fw2_baseline_families_wape": plot_bayes_fw2_baseline_families_wape,
    "bayes_fw2_best_baseline_vs_nonlinear_m4_wape": plot_bayes_fw2_best_baseline_vs_nonlinear_m4_wape,
    "xai_fw2_m4_per_building_pfi_driver_share": plot_xai_fw2_m4_per_building_pfi_driver_share,
    "xai_fw2_m4_per_building_shap_driver_share": plot_xai_fw2_m4_per_building_shap_driver_share,
    "xai_fw2_m4_pooled_pfi_driver_share": plot_xai_fw2_m4_pooled_pfi_driver_share,
    "xai_fw2_m4_pooled_shap_driver_share": plot_xai_fw2_m4_pooled_shap_driver_share,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        default="all",
        choices=["all", *FIGURE_BUILDERS.keys()],
        help="Figure name to render, or 'all'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.figure == "all":
        for builder in FIGURE_BUILDERS.values():
            builder()
        return

    FIGURE_BUILDERS[args.figure]()


if __name__ == "__main__":
    main()
