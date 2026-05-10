#!/usr/bin/env python3
"""Render the Chapter 4 raw campus-meter illustration."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import warnings

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent.parent / ".mplconfig")))
warnings.filterwarnings("ignore", message="Pandas requires version")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
THESIS_ROOT = PROJECT_ROOT.parent / "masters-thesis"
FIGURES_DIR = THESIS_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results" / "ch4_raw_data_context"

WINDOW_START = pd.Timestamp("2024-01-01 00:00:00")
WINDOW_END = pd.Timestamp("2024-01-08 00:00:00")
HOURLY_INDEX = pd.date_range(WINDOW_START, WINDOW_END - pd.Timedelta(hours=1), freq="h")

RAW_DIR = PROJECT_ROOT / "data" / "campus-data" / "U06" / "2024"
HEAT_CSV = RAW_DIR / "U06_BHB04_n\u00e4it_2024.csv"
SUPPLY_CSV = RAW_DIR / "U06_BHB04_pealevoolu temp_2024.csv"
RETURN_CSV = RAW_DIR / "U06_BHB04_tagasivoolu temp_2024.csv"

MOCK_FIGURE_NAME = "raw_meter_design_mock.png"
THESIS_FIGURE_NAME = "raw_u06_vent_meter_slice.png"


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )


def load_raw(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Time"])
    df = df.loc[(df["Time"] >= WINDOW_START) & (df["Time"] < WINDOW_END), ["Time", "Value"]]
    return df.dropna().sort_values("Time").set_index("Time")["Value"]


def interpolate_to_hourly(series: pd.Series) -> pd.Series:
    hourly = (
        series.reindex(series.index.union(HOURLY_INDEX))
        .interpolate(method="time")
        .reindex(HOURLY_INDEX)
    )
    return hourly


def make_real_frame() -> pd.DataFrame:
    cumulative_heat = interpolate_to_hourly(load_raw(HEAT_CSV))
    supply = interpolate_to_hourly(load_raw(SUPPLY_CSV))
    ret = interpolate_to_hourly(load_raw(RETURN_CSV))

    frame = pd.DataFrame(
        {
            "timestamp": HOURLY_INDEX,
            "heat": cumulative_heat.diff().to_numpy(),
            "supply": supply.to_numpy(),
            "return": ret.to_numpy(),
        }
    )
    return frame.dropna().reset_index(drop=True)


def make_mock_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    timestamp = HOURLY_INDEX
    hour = timestamp.hour.to_numpy()
    day = np.arange(len(timestamp)) / 24

    heat = 0.11 + 0.035 * np.sin((hour - 7) / 24 * 2 * np.pi)
    heat += 0.025 * np.sin(day / 1.4 * 2 * np.pi)
    heat += 0.035 * np.exp(-((day - 2.7) / 0.55) ** 2)
    heat += rng.normal(0, 0.004, len(timestamp))
    heat = np.clip(heat, 0.04, None)

    supply = 48 + 5.5 * np.sin((day - 0.7) / 7 * 2 * np.pi)
    supply += 2.0 * np.sin(day / 1.8 * 2 * np.pi)
    supply += rng.normal(0, 0.45, len(timestamp))

    ret = supply - 12.5 - 1.8 * np.sin(day / 1.2 * 2 * np.pi)
    ret += rng.normal(0, 0.35, len(timestamp))

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "heat": heat,
            "supply": supply,
            "return": ret,
        }
    )


def plot_meter_frame(frame: pd.DataFrame, output_path: Path) -> None:
    fig, heat_ax = plt.subplots(figsize=(10.2, 3.85))
    temp_ax = heat_ax.twinx()

    lines = []
    lines += heat_ax.plot(
        frame["timestamp"],
        frame["heat"],
        color="#111111",
        linewidth=1.65,
        label="Hourly heat use",
    )
    lines += temp_ax.plot(
        frame["timestamp"],
        frame["supply"],
        color="#2a6f8f",
        linewidth=1.45,
        label="Supply temperature",
    )
    lines += temp_ax.plot(
        frame["timestamp"],
        frame["return"],
        color="#c46a2d",
        linewidth=1.45,
        label="Return temperature",
    )

    heat_ax.set_ylabel("Heat use (MWh/h)")
    temp_ax.set_ylabel("Temperature (C)")
    heat_ax.set_xlabel("Timestamp")

    heat_ax.set_xlim(WINDOW_START, WINDOW_END - pd.Timedelta(hours=1))
    heat_ax.set_ylim(0, float(frame["heat"].max()) * 1.18)
    temp_ax.set_ylim(float(frame["return"].min()) - 2.0, float(frame["supply"].max()) + 2.0)

    heat_ax.grid(True, color="#d8d8d8", linewidth=0.75)
    temp_ax.grid(False)
    temp_ax.spines["right"].set_visible(True)

    heat_ax.xaxis.set_major_locator(mdates.DayLocator())
    heat_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    fig.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=2.8,
    )
    fig.subplots_adjust(left=0.075, right=0.915, bottom=0.18, top=0.83)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mock_path = RESULTS_DIR / MOCK_FIGURE_NAME
    result_path = RESULTS_DIR / THESIS_FIGURE_NAME
    thesis_path = FIGURES_DIR / THESIS_FIGURE_NAME

    plot_meter_frame(make_mock_frame(), mock_path)
    plot_meter_frame(make_real_frame(), result_path)
    shutil.copy2(result_path, thesis_path)


def main() -> None:
    apply_style()
    render()


if __name__ == "__main__":
    main()
