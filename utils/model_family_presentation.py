from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FAMILY_COLORS = {
    "lstm": "#c2410c",
    "xgboost": "#0f766e",
}

MODE_COLORS = {
    "M0": "#2563eb",
    "M1": "#7c3aed",
    "M2": "#ea580c",
    "M3": "#b45309",
    "M4": "#059669",
}

SEASON_COLORS = {
    "winter": "#2563eb",
    "spring": "#16a34a",
    "summer": "#dc2626",
    "fall": "#f59e0b",
}

SEASON_ORDER = ["winter", "spring", "summer", "fall"]


def season_windows(year: int = 2024) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {
        "winter": (pd.Timestamp(f"{year}-01-08 00:00:00"), pd.Timestamp(f"{year}-01-14 23:00:00")),
        "spring": (pd.Timestamp(f"{year}-04-08 00:00:00"), pd.Timestamp(f"{year}-04-14 23:00:00")),
        "summer": (pd.Timestamp(f"{year}-07-08 00:00:00"), pd.Timestamp(f"{year}-07-14 23:00:00")),
        "fall": (pd.Timestamp(f"{year}-10-07 00:00:00"), pd.Timestamp(f"{year}-10-13 23:00:00")),
    }


def _save_fig(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def _month_to_season(month: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                month.isin([12, 1, 2]),
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8]),
            ],
            ["winter", "spring", "summer"],
            default="fall",
        ),
        index=month.index,
    )


def _with_time_columns(df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[dt_col])
    out[dt_col] = dt
    out["hour"] = dt.dt.hour
    out["month"] = dt.dt.month
    out["season"] = pd.Categorical(_month_to_season(out["month"]), categories=SEASON_ORDER, ordered=True)
    return out


def _history_source_frame(base_frames: dict[str, dict[str, pd.DataFrame]], building: str) -> pd.DataFrame:
    frame_map = base_frames[building]
    if "setA" in frame_map:
        return frame_map["setA"]
    return next(iter(frame_map.values()))


def _history_slice(
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    building: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    history_hours: int = 168,
) -> pd.DataFrame:
    frame = _history_source_frame(base_frames, building)[["datetime", "heat_kwh"]].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    hist_start = pd.Timestamp(start) - pd.Timedelta(hours=int(history_hours))
    hist_end = pd.Timestamp(end)
    return frame.loc[(frame["datetime"] >= hist_start) & (frame["datetime"] <= hist_end)].copy()


def _prediction_slice(
    predictions_df: pd.DataFrame,
    *,
    regime: str,
    building: str,
    model_family: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    out = predictions_df[
        (predictions_df["regime"].astype(str) == regime)
        & (predictions_df["building"].astype(str) == building)
        & (predictions_df["model_family"].astype(str) == model_family)
        & (predictions_df["mode"].astype(str) == mode)
        & (predictions_df["weather_mode"].astype(str) == weather_mode)
        & (predictions_df["horizon_h"] == int(horizon_h))
    ].copy()
    if out.empty:
        return out
    out["datetime"] = pd.to_datetime(out["datetime"])
    if start is not None:
        out = out.loc[out["datetime"] >= pd.Timestamp(start)].copy()
    if end is not None:
        out = out.loc[out["datetime"] <= pd.Timestamp(end)].copy()
    return out.sort_values("datetime").reset_index(drop=True)


def _resolve_window(
    predictions_df: pd.DataFrame,
    *,
    regime: str,
    building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    model_family: str = "lstm",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    default_days: int = 7,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if start is not None:
        start_ts = pd.Timestamp(start)
    else:
        probe = _prediction_slice(
            predictions_df,
            regime=regime,
            building=building,
            model_family=model_family,
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=horizon_h,
        )
        if probe.empty:
            raise ValueError("No prediction rows available for the requested window.")
        start_ts = pd.Timestamp(probe["datetime"].min())
    if end is not None:
        end_ts = pd.Timestamp(end)
    else:
        end_ts = start_ts + pd.Timedelta(days=int(default_days)) - pd.Timedelta(hours=1)
    return start_ts, end_ts


def _target_label(horizon_h: int) -> str:
    return f"Next {int(horizon_h)}h cumulative heat"


def _smooth_hourly_profile(df: pd.DataFrame, value_cols: Iterable[str], smoothing_window: int) -> pd.DataFrame:
    out_parts = []
    for season, sub in df.groupby("season", observed=True):
        cur = sub.sort_values("hour").copy()
        for col in value_cols:
            cur[col] = cur[col].rolling(int(smoothing_window), center=True, min_periods=1).mean()
        out_parts.append(cur)
    if not out_parts:
        return df.copy()
    return pd.concat(out_parts, ignore_index=True)


def build_model_intro_tables(
    config: Any,
    *,
    architectures: dict[str, dict[str, Any]],
    xgb_fixed_params: dict[str, Any],
    xgb_preset_lookup: dict[str, dict[str, Any]],
    mode_temporal_features: dict[str, list[str]],
    static_features_setb: list[str],
    summary_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    lstm_spec = architectures[config.lstm_architecture_id].copy()
    xgb_preset = xgb_preset_lookup[config.xgb_preset_id].copy()

    family_cards = pd.DataFrame(
        [
            {
                "model_family": "lstm",
                "role": "sequence model",
                "locked_id": config.lstm_architecture_id,
                "architecture": f"L{int(config.lookback_hours)} -> LSTM({', '.join(str(x) for x in lstm_spec['lstm_stack'])}) -> Dense({int(lstm_spec['dense_units'])}) -> 1",
                "training": f"Adam lr={config.learning_rate}, batch={config.batch_size}, epochs={config.epochs}, early_stop={config.early_stopping_patience}",
                "what_it_does": "Learns temporal state from ordered sequences and can add a separate static branch in M4.",
            },
            {
                "model_family": "xgboost",
                "role": "gradient-boosted trees",
                "locked_id": config.xgb_preset_id,
                "architecture": f"gbtree, n_estimators={xgb_fixed_params['n_estimators']}, max_depth={xgb_preset['max_depth']}",
                "training": (
                    f"lr={xgb_preset['learning_rate']}, min_child_weight={xgb_preset['min_child_weight']}, "
                    f"subsample={xgb_fixed_params['subsample']}, colsample={xgb_fixed_params['colsample_bytree']}, "
                    f"reg_lambda={xgb_fixed_params['reg_lambda']}, early_stop={xgb_fixed_params['early_stopping_rounds']}"
                ),
                "what_it_does": "Learns nonlinear interactions in tabular feature rows and usually converges much faster than the LSTM.",
            },
        ]
    )

    mode_cards = pd.DataFrame(
        [
            {
                "mode": "M0",
                "dynamic_feature_count": len(mode_temporal_features["M0"]),
                "uses_static_branch": False,
                "description": "Lean temporal core: observed heat, current weather, and calendar only.",
            },
            {
                "mode": "M1",
                "dynamic_feature_count": len(mode_temporal_features["M1"]),
                "uses_static_branch": False,
                "description": "M0 plus historical weather memory via the 24h rolling outdoor-temperature feature.",
            },
            {
                "mode": "M2",
                "dynamic_feature_count": len(mode_temporal_features["M2"]),
                "uses_static_branch": False,
                "description": "M0 plus system / inertia features from space and ventilation loops.",
            },
            {
                "mode": "M3",
                "dynamic_feature_count": len(mode_temporal_features["M3"]),
                "uses_static_branch": False,
                "description": "M1 plus M2, still dynamic-only and still sourced from setA.",
            },
            {
                "mode": "M4",
                "dynamic_feature_count": len(mode_temporal_features["M4"]),
                "uses_static_branch": True,
                "description": "M3 plus the static setB branch; future forecasts remain a weather-mode choice, not a mode property.",
            },
        ]
    )

    feature_blocks = pd.DataFrame(
        [
            {
                "feature_block": "base_temporal_core",
                "count": len(mode_temporal_features["M0"]),
                "used_in": "M0, M1, M2, M3, M4",
                "description": "Observed heat, outdoor weather, and cyclic time features.",
            },
            {
                "feature_block": "system_inertia_block",
                "count": len(set(mode_temporal_features["M2"]) - set(mode_temporal_features["M0"])),
                "used_in": "M2, M3, M4",
                "description": "Space / ventilation activity, deltaT, and low-deltaT state indicators.",
            },
            {
                "feature_block": "weather_memory_block",
                "count": len(set(mode_temporal_features["M1"]) - set(mode_temporal_features["M0"])),
                "used_in": "M1, M3, M4",
                "description": "Longer temperature memory via the 24h rolling outdoor-temperature feature.",
            },
            {
                "feature_block": "static_setB_branch",
                "count": len(static_features_setb),
                "used_in": "M4",
                "description": "Static building descriptors, topology, EHR proxies, and missingness flags.",
            },
        ]
    )

    weather_mode_cards = pd.DataFrame(
        [
            {
                "weather_mode": "FW0",
                "description": "No future weather is appended.",
                "future_weather_origin": "none",
                "operational_status": "baseline",
            },
            {
                "weather_mode": "FW1",
                "description": "Oracle future weather is created in memory from actual future temperature / RH. It is an upper bound, not an operational input.",
                "future_weather_origin": "oracle future weather",
                "operational_status": "upper_bound_only",
            },
            {
                "weather_mode": "FW2",
                "description": "Forecast-like proxy future weather comes from exported feat_fw_* columns in the regular setA / setB feature files.",
                "future_weather_origin": "forecast-like proxy future weather",
                "operational_status": "operational_analogue",
            },
        ]
    )

    partial_highlights = pd.DataFrame()
    if summary_df is not None and not summary_df.empty:
        focus = summary_df.loc[summary_df["weather_mode"].astype(str) == "FW2"].copy()
        rows = []
        for regime, regime_df in focus.groupby("regime", observed=True):
            for horizon_h, sub in regime_df.groupby("horizon_h", observed=True):
                best = sub.loc[sub["wape_mean"].idxmin()]
                rows.append(
                    {
                        "regime": regime,
                        "horizon_h": int(horizon_h),
                        "best_mode": str(best["mode"]),
                        "best_family": str(best["model_family"]),
                        "best_wape_mean": float(best["wape_mean"]),
                    }
                )
        partial_highlights = pd.DataFrame(rows).sort_values(["regime", "horizon_h"], ignore_index=True)

    return {
        "family_cards": family_cards,
        "mode_cards": mode_cards,
        "weather_mode_cards": weather_mode_cards,
        "feature_blocks": feature_blocks,
        "partial_highlights": partial_highlights,
    }


def plot_short_window_family_comparison(
    predictions_df: pd.DataFrame,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    regime: str = "per_building",
    model_families: tuple[str, ...] = ("lstm", "xgboost"),
    history_hours: int = 168,
    save_path: str | Path | None = None,
) -> plt.Figure:
    start_ts, end_ts = _resolve_window(
        predictions_df,
        regime=regime,
        building=building,
        mode=mode,
        weather_mode=weather_mode,
        horizon_h=horizon_h,
        model_family=model_families[0],
        start=start,
        end=end,
    )

    history = _history_slice(base_frames, building=building, start=start_ts, end=end_ts, history_hours=history_hours)
    target_actual = None
    family_preds: dict[str, pd.DataFrame] = {}
    for family in model_families:
        sub = _prediction_slice(
            predictions_df,
            regime=regime,
            building=building,
            model_family=family,
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=horizon_h,
            start=start_ts,
            end=end_ts,
        )
        if not sub.empty:
            family_preds[family] = sub
            if target_actual is None:
                target_actual = sub[["datetime", "y_true"]].drop_duplicates().copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True, gridspec_kw={"height_ratios": [1, 2]})

    axes[0].plot(history["datetime"], history["heat_kwh"], color="#334155", linewidth=1.3, label="historic hourly heat")
    axes[0].axvspan(start_ts, end_ts, color="#fde68a", alpha=0.35)
    axes[0].set_title(f"{building} historical context before forecast window")
    axes[0].set_ylabel("Hourly heat [kWh]")
    axes[0].legend(loc="upper left")

    if target_actual is not None and not target_actual.empty:
        axes[1].plot(target_actual["datetime"], target_actual["y_true"], color="black", linewidth=2.0, label=f"actual target ({_target_label(horizon_h)})")
    for family, sub in family_preds.items():
        axes[1].plot(
            sub["datetime"],
            sub["y_pred"],
            linewidth=1.9,
            label=f"{family} predicted",
            color=FAMILY_COLORS.get(family, None),
        )
    axes[1].axvspan(start_ts, end_ts, color="#fde68a", alpha=0.15)
    axes[1].set_title(f"Short-window forecast comparison | {building} | {mode} | {weather_mode} | h={int(horizon_h)}")
    axes[1].set_ylabel("Forecast target [kWh]")
    axes[1].set_xlabel("Datetime")
    axes[1].legend(loc="upper left", ncol=max(1, len(family_preds) + 1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_multi_horizon_window(
    predictions_df: pd.DataFrame,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    building: str,
    model_family: str,
    mode: str,
    weather_mode: str,
    horizons: tuple[int, ...] = (1, 4, 8, 24),
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    regime: str = "per_building",
    history_hours: int = 168,
    save_path: str | Path | None = None,
) -> plt.Figure:
    start_ts, end_ts = _resolve_window(
        predictions_df,
        regime=regime,
        building=building,
        mode=mode,
        weather_mode=weather_mode,
        horizon_h=int(horizons[0]),
        model_family=model_family,
        start=start,
        end=end,
    )

    history = _history_slice(base_frames, building=building, start=start_ts, end=end_ts, history_hours=history_hours)
    nrows = len(horizons) + 1
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 3.1 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    axes[0].plot(history["datetime"], history["heat_kwh"], color="#334155", linewidth=1.3, label="historic hourly heat")
    axes[0].axvspan(start_ts, end_ts, color="#dbeafe", alpha=0.35)
    axes[0].set_title(f"{building} historical context")
    axes[0].set_ylabel("Hourly heat [kWh]")
    axes[0].legend(loc="upper left")

    for ax, horizon_h in zip(axes[1:], horizons):
        sub = _prediction_slice(
            predictions_df,
            regime=regime,
            building=building,
            model_family=model_family,
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=int(horizon_h),
            start=start_ts,
            end=end_ts,
        )
        if sub.empty:
            ax.text(0.5, 0.5, f"No rows for h={int(horizon_h)}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.plot(sub["datetime"], sub["y_true"], color="black", linewidth=1.8, label="actual")
        ax.plot(sub["datetime"], sub["y_pred"], color=FAMILY_COLORS.get(model_family, "#9333ea"), linewidth=1.8, label=f"{model_family} predicted")
        ax.axvspan(start_ts, end_ts, color="#dbeafe", alpha=0.15)
        ax.set_title(f"{_target_label(horizon_h)} | {model_family} | {mode} | {weather_mode}")
        ax.set_ylabel("Target [kWh]")
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Datetime")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_feature_set_effect_window(
    predictions_df: pd.DataFrame,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    building: str,
    model_family: str,
    weather_mode: str,
    horizon_h: int,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    modes: tuple[str, ...] = ("M0", "M1", "M2", "M4"),
    regime: str = "per_building",
    history_hours: int = 168,
    save_path: str | Path | None = None,
) -> plt.Figure:
    start_ts, end_ts = _resolve_window(
        predictions_df,
        regime=regime,
        building=building,
        mode=modes[0],
        weather_mode=weather_mode,
        horizon_h=horizon_h,
        model_family=model_family,
        start=start,
        end=end,
    )

    history = _history_slice(base_frames, building=building, start=start_ts, end=end_ts, history_hours=history_hours)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    axes[0].plot(history["datetime"], history["heat_kwh"], color="#334155", linewidth=1.3, label="historic hourly heat")
    axes[0].axvspan(start_ts, end_ts, color="#dcfce7", alpha=0.35)
    axes[0].set_title(f"{building} historical context before mode comparison")
    axes[0].set_ylabel("Hourly heat [kWh]")
    axes[0].legend(loc="upper left")

    actual_drawn = False
    for mode in modes:
        sub = _prediction_slice(
            predictions_df,
            regime=regime,
            building=building,
            model_family=model_family,
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=horizon_h,
            start=start_ts,
            end=end_ts,
        )
        if sub.empty:
            continue
        if not actual_drawn:
            axes[1].plot(sub["datetime"], sub["y_true"], color="black", linewidth=2.0, label="actual")
            actual_drawn = True
        axes[1].plot(
            sub["datetime"],
            sub["y_pred"],
            linewidth=1.9,
            label=f"{mode} predicted",
            color=MODE_COLORS.get(mode, None),
        )
    axes[1].axvspan(start_ts, end_ts, color="#dcfce7", alpha=0.15)
    axes[1].set_title(f"Feature-set effect on forecast | {building} | {model_family} | {weather_mode} | h={int(horizon_h)}")
    axes[1].set_ylabel("Forecast target [kWh]")
    axes[1].set_xlabel("Datetime")
    axes[1].legend(loc="upper left", ncol=max(1, len(modes) + 1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_historic_and_forecast_profiles_by_season(
    predictions_df: pd.DataFrame,
    base_frames: dict[str, dict[str, pd.DataFrame]],
    *,
    building: str,
    model_family: str,
    mode: str,
    weather_mode: str,
    horizons: tuple[int, ...] = (1, 24),
    regime: str = "per_building",
    test_start: str | pd.Timestamp | None = None,
    smoothing_window: int = 3,
    save_path: str | Path | None = None,
) -> plt.Figure:
    history = _history_source_frame(base_frames, building)[["datetime", "heat_kwh"]].copy()
    history = _with_time_columns(history)
    if test_start is not None:
        history = history.loc[history["datetime"] >= pd.Timestamp(test_start)].copy()

    hist_profile = history.groupby(["season", "hour"], observed=True, as_index=False)["heat_kwh"].mean()
    hist_profile = _smooth_hourly_profile(hist_profile, ["heat_kwh"], smoothing_window)

    fig, axes = plt.subplots(len(horizons), 2, figsize=(15, 4.5 * len(horizons)), sharex=True)
    if len(horizons) == 1:
        axes = np.asarray([axes])

    for row_idx, horizon_h in enumerate(horizons):
        ax_hist = axes[row_idx, 0]
        for season in SEASON_ORDER:
            sub = hist_profile.loc[hist_profile["season"].astype(str) == season].copy()
            if sub.empty:
                continue
            ax_hist.plot(sub["hour"], sub["heat_kwh"], label=season, color=SEASON_COLORS[season], linewidth=2.0)
        ax_hist.set_title(f"Historic hourly load profile | {building}")
        ax_hist.set_ylabel("Hourly heat [kWh]")
        ax_hist.set_xlabel("Hour of day")
        ax_hist.legend(loc="upper left")

        pred = _prediction_slice(
            predictions_df,
            regime=regime,
            building=building,
            model_family=model_family,
            mode=mode,
            weather_mode=weather_mode,
            horizon_h=int(horizon_h),
        )
        pred = _with_time_columns(pred) if not pred.empty else pred
        if pred.empty:
            ax_fore = axes[row_idx, 1]
            ax_fore.text(0.5, 0.5, f"No prediction rows for h={int(horizon_h)}", ha="center", va="center", transform=ax_fore.transAxes)
            ax_fore.set_axis_off()
            continue

        pred_profile = pred.groupby(["season", "hour"], observed=True, as_index=False)[["y_true", "y_pred"]].mean()
        pred_profile = _smooth_hourly_profile(pred_profile, ["y_true", "y_pred"], smoothing_window)

        ax_fore = axes[row_idx, 1]
        for season in SEASON_ORDER:
            sub = pred_profile.loc[pred_profile["season"].astype(str) == season].copy()
            if sub.empty:
                continue
            color = SEASON_COLORS[season]
            ax_fore.plot(sub["hour"], sub["y_true"], color=color, linewidth=2.0, label=f"{season} actual")
            ax_fore.plot(sub["hour"], sub["y_pred"], color=color, linewidth=2.0, linestyle="--", label=f"{season} predicted")
        ax_fore.set_title(f"Forecast profile by season | {model_family} | {mode} | {weather_mode} | h={int(horizon_h)}")
        ax_fore.set_ylabel("Forecast target [kWh]")
        ax_fore.set_xlabel("Issue hour")
        ax_fore.legend(loc="upper left", ncol=2)

    fig.tight_layout()
    return _save_fig(fig, save_path)
