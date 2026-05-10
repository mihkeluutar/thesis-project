from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent.parent / ".mplconfig")))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
THESIS_ROOT = PROJECT_ROOT.parent / "masters-thesis"
FIGURES_DIR = THESIS_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results" / "ch4_support_figures_20260507"

WEATHER_SOURCE = PROJECT_ROOT / "data" / "combined_weather_2022_2025.csv"
FEATURE_SOURCE = PROJECT_ROOT / "data" / "features" / "U05_features_setA.csv"

FUTURE_WEATHER_HORIZON = 36
PREFERRED_ISSUE_TIME = pd.Timestamp("2024-01-03 06:00:00")
RNG_SEED = 20260322


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9.5,
            "axes.titlesize": 12.5,
            "axes.titlepad": 8,
            "font.size": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    result_path = RESULTS_DIR / filename
    thesis_path = FIGURES_DIR / filename
    fig.savefig(result_path, bbox_inches="tight")
    shutil.copy2(result_path, thesis_path)
    plt.close(fig)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#d8d8d8", linewidth=0.75, alpha=0.75)


def render_weather_source_scatter() -> plt.Figure:
    df = pd.read_csv(WEATHER_SOURCE, parse_dates=["datetime"]).sort_values("datetime")
    panels = [
        {
            "x": "COP_temp_c",
            "y": "KKP_temp_c",
            "title": "Temperature",
            "xlabel": "COP temperature (C)",
            "ylabel": "KKP temperature (C)",
            "identity": True,
        },
        {
            "x": "COP_wind_speed_ms",
            "y": "KKP_wind_speed_ms",
            "title": "Wind speed",
            "xlabel": "COP wind speed (m/s)",
            "ylabel": "KKP wind speed (m/s)",
            "identity": True,
        },
        {
            "x": "COP_ssrd_W_per_m2",
            "y": "KKP_sunshine_duration_min",
            "title": "Solar proxy",
            "xlabel": "COP SSRD (W/m2)",
            "ylabel": "KKP sunshine (min/h)",
            "identity": False,
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.45))
    for ax, panel in zip(axes, panels, strict=True):
        sub = df[[panel["x"], panel["y"]]].dropna()
        ax.scatter(
            sub[panel["x"]],
            sub[panel["y"]],
            s=3.6,
            color="#2a6f8f",
            alpha=0.18,
            edgecolors="none",
            rasterized=True,
        )
        if panel["identity"]:
            lo = float(min(sub[panel["x"]].min(), sub[panel["y"]].min()))
            hi = float(max(sub[panel["x"]].max(), sub[panel["y"]].max()))
            pad = (hi - lo) * 0.04
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#111111", linewidth=1.05, linestyle="--")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(panel["title"])
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel(panel["ylabel"])
        _style_axis(ax)

    fig.legend(
        handles=[Line2D([0], [0], color="#111111", linestyle="--", linewidth=1.05, label="1:1 reference")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=1,
        handlelength=2.4,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.78, wspace=0.36)
    return fig


def _future_weather_columns(prefix: str) -> list[str]:
    return [f"feat_fw_{prefix}_tplus{hour:02d}_h{FUTURE_WEATHER_HORIZON}" for hour in range(1, FUTURE_WEATHER_HORIZON + 1)]


def _select_issue_row(df: pd.DataFrame, required_columns: list[str]) -> pd.Series:
    preferred = df[df["datetime"].eq(PREFERRED_ISSUE_TIME)]
    if not preferred.empty and preferred[required_columns].notna().all(axis=1).iloc[0]:
        return preferred.iloc[0]

    valid = df[
        (df["datetime"] >= "2024-01-01")
        & (df["datetime"] < "2024-04-01")
        & df[required_columns].notna().all(axis=1)
    ].copy()
    if valid.empty:
        raise ValueError("No valid 2024 U05 rows found for the 36-hour future-weather path.")

    temp_cols = _future_weather_columns("temp")
    rh_cols = _future_weather_columns("rh")
    valid["path_range_score"] = (
        valid[temp_cols].max(axis=1)
        - valid[temp_cols].min(axis=1)
        + 0.12 * (valid[rh_cols].max(axis=1) - valid[rh_cols].min(axis=1))
    )
    return valid.sort_values(["path_range_score", "datetime"], ascending=[False, True]).iloc[0]


def _forecast_proxy(
    realized: pd.Series,
    sigma: pd.Series,
    *,
    seed_offset: int = 0,
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED + seed_offset)
    noise = rng.normal(0.0, sigma.fillna(0.0).to_numpy(dtype=float))
    proxy = realized.to_numpy(dtype=float) + noise
    if lower is not None or upper is not None:
        proxy = np.clip(proxy, -np.inf if lower is None else lower, np.inf if upper is None else upper)
    return proxy


def render_future_weather_path() -> plt.Figure:
    df = pd.read_csv(FEATURE_SOURCE, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    temp_cols = _future_weather_columns("temp")
    rh_cols = _future_weather_columns("rh")
    required = temp_cols + rh_cols
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing future-weather columns in {FEATURE_SOURCE}: {missing[:4]}")

    row = _select_issue_row(df, required)
    train_mask = df["datetime"] < "2024-01-01"
    residuals = df[required].astype(float) - df[required].shift(24).astype(float)
    sigma = residuals.loc[train_mask].std(skipna=True)

    realized_temp = row[temp_cols].astype(float)
    realized_rh = row[rh_cols].astype(float)
    proxy_temp = _forecast_proxy(realized_temp, sigma[temp_cols])
    proxy_rh = _forecast_proxy(realized_rh, sigma[rh_cols], seed_offset=1, lower=0.0, upper=100.0)
    hours = np.arange(1, FUTURE_WEATHER_HORIZON + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 4.45), sharex=True)
    series = [
        (axes[0], realized_temp.to_numpy(dtype=float), proxy_temp, "Temperature (C)"),
        (axes[1], realized_rh.to_numpy(dtype=float), proxy_rh, "Relative humidity (%)"),
    ]

    for ax, realized, proxy, ylabel in series:
        ax.plot(hours, realized, color="#111111", linewidth=1.65, label="Realized future")
        ax.plot(hours, proxy, color="#c46a2d", linewidth=1.45, label="FW2 proxy")
        ax.fill_between(hours, realized, proxy, color="#c46a2d", alpha=0.10, linewidth=0)
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, FUTURE_WEATHER_HORIZON)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _style_axis(ax)

    axes[1].set_xlabel("Look-ahead hour after forecast issue")
    axes[1].set_xticks([1, 6, 12, 18, 24, 30, 36])
    axes[1].set_ylim(0, 100)

    fig.legend(
        handles=[
            Line2D([0], [0], color="#111111", linewidth=1.65, label="Realized future"),
            Line2D([0], [0], color="#c46a2d", linewidth=1.45, label="FW2 proxy"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        handlelength=2.4,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.84, hspace=0.16)
    return fig


FIGURE_BUILDERS = {
    "weather_source_scatter": (render_weather_source_scatter, "weather_source_scatter.png"),
    "future_weather_fw2_path_u05_h36": (render_future_weather_path, "future_weather_fw2_path_u05_h36.png"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render thesis-referenced Chapter 4 support figures.")
    parser.add_argument("--figure", choices=["all", *FIGURE_BUILDERS.keys()], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    names = list(FIGURE_BUILDERS) if args.figure == "all" else [args.figure]
    for name in names:
        builder, filename = FIGURE_BUILDERS[name]
        save_figure(builder(), filename)
        print(f"Wrote {FIGURES_DIR / filename}")


if __name__ == "__main__":
    main()
