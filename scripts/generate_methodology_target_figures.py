from __future__ import annotations

import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent.parent / ".mplconfig")))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
THESIS_ROOT = PROJECT_ROOT.parent / "masters-thesis"
RESULTS_DIR = PROJECT_ROOT / "results" / "methodology_target_figures_20260405"
THESIS_FIG_DIR = THESIS_ROOT / "figures"

TRACE_SOURCE = PROJECT_ROOT / "results" / "baseline_predictions_U05.csv"
SEMANTICS_SOURCE = PROJECT_ROOT / "results" / "architecture_audit_input_semantics_20032026.csv"

ISSUE_TIME = pd.Timestamp("2024-01-03 06:00:00")
LOOKBACK_HOURS = 24
POINT_HORIZON = 8
CUMULATIVE_HORIZON = 24


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def load_trace() -> pd.DataFrame:
    df = pd.read_csv(TRACE_SOURCE, parse_dates=["timestamp"])
    df = df.rename(columns={"timestamp": "datetime", "actual_kwh": "heat_kwh"})
    df = df[["datetime", "heat_kwh"]].dropna().sort_values("datetime").reset_index(drop=True)
    if ISSUE_TIME not in set(df["datetime"]):
        raise ValueError(f"Issue time {ISSUE_TIME} not found in {TRACE_SOURCE}")
    return df


def load_semantics() -> pd.DataFrame:
    df = pd.read_csv(
        SEMANTICS_SOURCE,
        parse_dates=["first_target_datetime", "input_window_start", "input_window_end"],
    )
    return df


def save_figure(fig: plt.Figure, filename: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    out_result = RESULTS_DIR / filename
    out_thesis = THESIS_FIG_DIR / filename
    fig.savefig(out_result, bbox_inches="tight")
    shutil.copy2(out_result, out_thesis)
    plt.close(fig)


def _annotate_common(ax: plt.Axes, ymin: float, ymax: float) -> None:
    input_start = ISSUE_TIME - pd.Timedelta(hours=LOOKBACK_HOURS)
    point_time = ISSUE_TIME + pd.Timedelta(hours=POINT_HORIZON)
    cumulative_end = ISSUE_TIME + pd.Timedelta(hours=CUMULATIVE_HORIZON - 1)

    ax.axvspan(input_start, ISSUE_TIME, color="#cbd5e1", alpha=0.45, lw=0)
    ax.axvspan(ISSUE_TIME, cumulative_end, color="#fdba74", alpha=0.22, lw=0)
    ax.axvline(ISSUE_TIME, color="#b91c1c", linestyle="--", linewidth=1.8)
    ax.axvline(point_time, color="#2563eb", linestyle=":", linewidth=1.6)

    ax.text(
        input_start + (ISSUE_TIME - input_start) / 2,
        ymax,
        "Past input window\nused by the model",
        ha="center",
        va="top",
        color="#334155",
        fontsize=10,
    )
    ax.text(
        ISSUE_TIME,
        ymin,
        "Issue time $t$",
        ha="left",
        va="bottom",
        color="#991b1b",
        fontsize=10,
    )
    ax.text(
        point_time,
        ymax,
        f"Point target\n$y_{{t+{POINT_HORIZON}}}$",
        ha="center",
        va="top",
        color="#1d4ed8",
        fontsize=10,
    )
    ax.text(
        ISSUE_TIME + (cumulative_end - ISSUE_TIME) / 2,
        ymin,
        f"Cumulative window\n$Y_t^{{({CUMULATIVE_HORIZON})}} = \\sum_{{k=0}}^{{{CUMULATIVE_HORIZON-1}}} y_{{t+k}}$",
        ha="center",
        va="bottom",
        color="#9a3412",
        fontsize=10,
    )


def _format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.set_xlabel("Forecast timeline")


def make_trace_example(df: pd.DataFrame) -> plt.Figure:
    input_start = ISSUE_TIME - pd.Timedelta(hours=LOOKBACK_HOURS)
    point_time = ISSUE_TIME + pd.Timedelta(hours=POINT_HORIZON)
    cumulative_end = ISSUE_TIME + pd.Timedelta(hours=CUMULATIVE_HORIZON - 1)
    window_start = ISSUE_TIME - pd.Timedelta(hours=36)
    window_end = ISSUE_TIME + pd.Timedelta(hours=CUMULATIVE_HORIZON + 14)

    sub = df[(df["datetime"] >= window_start) & (df["datetime"] <= window_end)].copy()
    point_value = float(df.loc[df["datetime"] == point_time, "heat_kwh"].iloc[0])
    fill_df = df[(df["datetime"] >= ISSUE_TIME) & (df["datetime"] <= cumulative_end)].copy()

    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.plot(sub["datetime"], sub["heat_kwh"], color="#0f172a", linewidth=2.3, label="Observed hourly heat demand")
    ax.fill_between(fill_df["datetime"], 0, fill_df["heat_kwh"], color="#fb923c", alpha=0.28)
    ax.scatter([point_time], [point_value], s=70, color="#2563eb", zorder=5)

    ymin = max(0.0, float(sub["heat_kwh"].min()) * 0.88)
    ymax = float(sub["heat_kwh"].max()) * 1.08
    ax.set_ylim(ymin, ymax)
    _annotate_common(ax, ymin + 0.02 * (ymax - ymin), ymax - 0.02 * (ymax - ymin))

    ax.set_title("Candidate A: Real heat-demand trace with point and cumulative targets")
    ax.set_ylabel("Heat demand (kWh)")
    ax.set_xlabel("Datetime")
    ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return fig


def make_point_target_example(df: pd.DataFrame) -> plt.Figure:
    point_time = ISSUE_TIME + pd.Timedelta(hours=POINT_HORIZON)
    window_start = ISSUE_TIME - pd.Timedelta(hours=30)
    window_end = ISSUE_TIME + pd.Timedelta(hours=30)
    sub = df[(df["datetime"] >= window_start) & (df["datetime"] <= window_end)].copy()
    point_value = float(df.loc[df["datetime"] == point_time, "heat_kwh"].iloc[0])

    fig, ax = plt.subplots(figsize=(11.5, 3.9))
    ax.plot(sub["datetime"], sub["heat_kwh"], color="#111827", linewidth=2.15)

    ymin = max(0.0, float(sub["heat_kwh"].min()) * 0.88)
    ymax = float(sub["heat_kwh"].max()) * 1.08
    ax.set_ylim(ymin, ymax)
    ax.axvspan(ISSUE_TIME - pd.Timedelta(hours=LOOKBACK_HOURS), ISSUE_TIME, color="#cbd5e1", alpha=0.45, lw=0)
    ax.axvline(ISSUE_TIME, color="#b91c1c", linestyle="--", linewidth=1.8)
    ax.axvline(point_time, color="#2563eb", linestyle=":", linewidth=1.6)
    ax.scatter([point_time], [point_value], s=80, color="#2563eb", zorder=5)
    ax.set_ylabel("Heat demand (kWh)")
    _format_time_axis(ax)
    fig.autofmt_xdate(rotation=25, ha="right")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#111827", linewidth=2.15, label="Observed hourly heat demand"),
            Patch(facecolor="#cbd5e1", edgecolor="none", alpha=0.45, label="Past input window"),
            Line2D([0], [0], color="#b91c1c", linestyle="--", linewidth=1.8, label="Forecast issue time"),
            Line2D([0], [0], color="#2563eb", linestyle=":", marker="o", markersize=7, linewidth=1.6, label="Point-horizon target hour"),
        ],
        loc="upper left",
        frameon=True,
    )
    fig.tight_layout()
    return fig

def make_cumulative_target_example(df: pd.DataFrame) -> plt.Figure:
    cumulative_end = ISSUE_TIME + pd.Timedelta(hours=CUMULATIVE_HORIZON - 1)
    window_start = ISSUE_TIME - pd.Timedelta(hours=30)
    window_end = ISSUE_TIME + pd.Timedelta(hours=30)
    sub = df[(df["datetime"] >= window_start) & (df["datetime"] <= window_end)].copy()
    fill_df = df[(df["datetime"] >= ISSUE_TIME) & (df["datetime"] <= cumulative_end)].copy()

    fig, ax = plt.subplots(figsize=(11.5, 3.9))
    ax.plot(sub["datetime"], sub["heat_kwh"], color="#111827", linewidth=2.15)

    ymin = max(0.0, float(sub["heat_kwh"].min()) * 0.88)
    ymax = float(sub["heat_kwh"].max()) * 1.08
    ax.set_ylim(ymin, ymax)
    ax.axvspan(ISSUE_TIME - pd.Timedelta(hours=LOOKBACK_HOURS), ISSUE_TIME, color="#cbd5e1", alpha=0.45, lw=0)
    ax.axvline(ISSUE_TIME, color="#b91c1c", linestyle="--", linewidth=1.8)
    ax.axvspan(ISSUE_TIME, cumulative_end, color="#fdba74", alpha=0.22, lw=0)
    ax.fill_between(fill_df["datetime"], 0, fill_df["heat_kwh"], color="#fb923c", alpha=0.30)
    ax.set_ylabel("Heat demand (kWh)")
    _format_time_axis(ax)
    fig.autofmt_xdate(rotation=25, ha="right")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#111827", linewidth=2.15, label="Observed hourly heat demand"),
            Patch(facecolor="#cbd5e1", edgecolor="none", alpha=0.45, label="Past input window"),
            Line2D([0], [0], color="#b91c1c", linestyle="--", linewidth=1.8, label="Forecast issue time"),
            Patch(facecolor="#fb923c", edgecolor="none", alpha=0.30, label="Cumulative target window"),
        ],
        loc="upper left",
        frameon=True,
    )
    fig.tight_layout()
    return fig


def make_schematic_example(semantics_df: pd.DataFrame) -> plt.Figure:
    row = semantics_df[(semantics_df["mode"] == "M0") & (semantics_df["lookback_hours"] == LOOKBACK_HOURS)].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True)
    x_min = -LOOKBACK_HOURS - 2
    x_max = CUMULATIVE_HORIZON + 6
    y_mid = 0.5

    titles = [
        f"Point-horizon contract: predict one future hour at $t+{POINT_HORIZON}$",
        f"Cumulative contract: aggregate all hours from $t$ to $t+{CUMULATIVE_HORIZON-1}$",
    ]

    for idx, ax in enumerate(axes):
        ax.hlines(y_mid, x_min, x_max, color="#475569", linewidth=1.3)
        ax.axvspan(-LOOKBACK_HOURS, 0, color="#cbd5e1", alpha=0.55, lw=0)
        for hour in range(-LOOKBACK_HOURS, 0):
            ax.add_patch(Rectangle((hour, 0.32), 1.0, 0.36, fill=False, edgecolor="#94a3b8", linewidth=0.8))
        ax.axvline(0, color="#b91c1c", linestyle="--", linewidth=1.8)
        ax.text(-LOOKBACK_HOURS / 2, 0.93, "Past engineered rows", ha="center", va="top", color="#334155")
        ax.text(0.1, 0.08, "Issue time $t$", ha="left", va="bottom", color="#991b1b")
        ax.text(x_max - 0.2, 0.08, "Future", ha="right", va="bottom", color="#475569")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(titles[idx])

    axes[0].scatter([POINT_HORIZON], [y_mid], s=90, color="#2563eb", zorder=5)
    axes[0].text(POINT_HORIZON, 0.86, f"$y_{{t+{POINT_HORIZON}}}$", ha="center", va="top", color="#1d4ed8", fontsize=12)

    for hour in range(0, CUMULATIVE_HORIZON):
        axes[1].add_patch(Rectangle((hour, 0.34), 1.0, 0.32, facecolor="#fdba74", edgecolor="#ea580c", alpha=0.75))
    axes[1].text(
        CUMULATIVE_HORIZON / 2,
        0.86,
        f"$Y_t^{{({CUMULATIVE_HORIZON})}} = \\sum_{{k=0}}^{{{CUMULATIVE_HORIZON-1}}} y_{{t+k}}$",
        ha="center",
        va="top",
        color="#9a3412",
        fontsize=12,
    )
    axes[1].text(
        x_min,
        -0.08,
        (
            "Input semantics anchor from notebook 06: "
            f"lookback={LOOKBACK_HOURS}h, first target issued at "
            f"{pd.Timestamp(row['first_target_datetime']).strftime('%Y-%m-%d %H:%M')}"
        ),
        ha="left",
        va="top",
        fontsize=9,
        color="#475569",
        transform=axes[1].transData,
    )

    axes[1].set_xlim(x_min, x_max)
    axes[1].set_xlabel("Hours relative to forecast issue time")
    axes[1].set_xticks([-24, -12, -1, 0, 8, 12, 24])
    axes[1].set_xticklabels(["$t-24$", "$t-12$", "$t-1$", "$t$", "$t+8$", "$t+12$", "$t+24$"])
    fig.suptitle("Candidate C: Clean schematic of the forecast-origin logic", y=0.98, fontsize=16)
    fig.tight_layout()
    return fig


def main() -> None:
    configure_style()
    trace_df = load_trace()

    save_figure(make_point_target_example(trace_df), "methodology_target_point_example.png")
    save_figure(make_cumulative_target_example(trace_df), "methodology_target_cumulative_example.png")

    manifest = pd.DataFrame(
        [
            {
                "filename": "methodology_target_point_example.png",
                "role": "Candidate B1",
                "description": "Point-horizon target figure with a separate target hour, past input window, and forecast issue time.",
            },
            {
                "filename": "methodology_target_cumulative_example.png",
                "role": "Candidate B2",
                "description": "Cumulative-target figure with the same forecast issue time and a shaded future demand window.",
            },
        ]
    )
    manifest.to_csv(RESULTS_DIR / "methodology_target_figures_manifest.csv", index=False)


if __name__ == "__main__":
    main()
