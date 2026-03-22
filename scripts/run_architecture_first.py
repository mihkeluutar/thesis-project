#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _bootstrap_src() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "thesis-project":
        project_root = cwd
    elif (cwd / "thesis-project").exists():
        project_root = (cwd / "thesis-project").resolve()
    else:
        project_root = cwd
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root


PROJECT_ROOT = _bootstrap_src()

from forecasting.config import (  # noqa: E402
    ARCHITECTURE_COHORT,
    DEFAULT_RESULTS_DIR,
    MAIN_COMPARISON_BUILDINGS,
    MODE_FEATURES,
    SEED,
    build_target_specs,
)
from forecasting.data import add_target_columns, list_available_buildings, load_feature_frame  # noqa: E402
from forecasting.modeling import (  # noqa: E402
    choose_architecture,
    monthly_rolling_origin_folds,
    run_architecture_subexperiment,
    run_main_comparison,
    write_architecture_choice_note,
)
from forecasting.quality import build_quality_matrix, write_quality_note  # noqa: E402


QUALITY_POINT_HORIZONS = [8, 10, 12, 16, 20, 24, 36]
QUALITY_CUM_HORIZONS = [10, 12, 16, 20, 24, 36]

ARCHITECTURE_POINT_HORIZONS = [8, 24, 36]
ARCHITECTURE_CUM_HORIZONS = [24]

MAIN_POINT_HORIZONS = [10, 12, 16, 20, 24, 36]
MAIN_CUM_HORIZONS = [10, 12, 16, 20, 24, 36]


def load_frames(buildings: list[str], target_specs):
    frames = {}
    for building in buildings:
        df = load_feature_frame(building, set_name="setB")
        frames[building] = add_target_columns(df, target_specs)
    return frames


def cmd_quality_audit(args: argparse.Namespace) -> None:
    buildings = list_available_buildings(set_name="setB")
    target_specs = build_target_specs(QUALITY_POINT_HORIZONS, QUALITY_CUM_HORIZONS)
    quality_df = build_quality_matrix(buildings=buildings, target_specs=target_specs, set_name="setB")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "data_quality_building_horizon.csv"
    note_path = output_dir / "data_quality_note.md"

    quality_df.to_csv(csv_path, index=False)
    write_quality_note(quality_df, note_path)

    print(f"Wrote quality matrix to {csv_path}")
    print(f"Wrote quality note to {note_path}")


def cmd_architecture_subexperiment(args: argparse.Namespace) -> None:
    target_specs = build_target_specs(ARCHITECTURE_POINT_HORIZONS, ARCHITECTURE_CUM_HORIZONS)
    frames = load_frames(ARCHITECTURE_COHORT, target_specs)
    run_df, summary_df = run_architecture_subexperiment(
        frames_by_building=frames,
        buildings=ARCHITECTURE_COHORT,
        target_specs=target_specs,
        modes=["M0", "M1"],
        folds=monthly_rolling_origin_folds(2024),
        seed=SEED,
        epochs=args.epochs,
        smoke=args.smoke,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "architecture_subexperiment_run_log.csv"
    summary_path = output_dir / "architecture_subexperiment_summary.csv"
    choice_path = output_dir / "architecture_choice.md"

    run_df.to_csv(run_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_architecture_choice_note(summary_df, choice_path)

    selected = choose_architecture(summary_df) if not summary_df.empty else None
    print(f"Wrote architecture run log to {run_path}")
    print(f"Wrote architecture summary to {summary_path}")
    print(f"Wrote architecture choice note to {choice_path}")
    if selected:
        print(f"Selected architecture: {selected}")


def cmd_main_comparison(args: argparse.Namespace) -> None:
    target_specs = build_target_specs(MAIN_POINT_HORIZONS, MAIN_CUM_HORIZONS)
    load_specs = build_target_specs([1] + MAIN_POINT_HORIZONS, MAIN_CUM_HORIZONS)
    frames = load_frames(MAIN_COMPARISON_BUILDINGS, load_specs)
    direct_specs = [spec for spec in target_specs if spec.family == "point" and spec.horizon_hours in MAIN_POINT_HORIZONS]

    result_df, path_df = run_main_comparison(
        frames_by_building=frames,
        buildings=MAIN_COMPARISON_BUILDINGS,
        direct_target_specs=direct_specs,
        recursive_horizons=MAIN_CUM_HORIZONS,
        architecture_name=args.architecture,
        modes=args.modes,
        seed=SEED,
        epochs=args.epochs,
        smoke=args.smoke,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "main_comparison_results.csv"
    path_path = output_dir / "recursive_path_exports.csv"

    result_df.to_csv(result_path, index=False)
    path_df.to_csv(path_path, index=False)

    ok_df = result_df.loc[result_df["status"] == "ok"].copy() if not result_df.empty else pd.DataFrame()
    if not ok_df.empty:
        summary = (
            ok_df.groupby(["strategy", "target_kind", "horizon_h", "mode"], observed=True)[["wape_heating_pct", "rmse_heating"]]
            .mean()
            .reset_index()
            .sort_values(["strategy", "target_kind", "horizon_h", "mode"])
        )
        summary_path = output_dir / "main_comparison_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote main comparison summary to {summary_path}")

    print(f"Wrote main comparison results to {result_path}")
    print(f"Wrote recursive paths to {path_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Architecture-first thesis forecasting runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    quality = subparsers.add_parser("quality-audit", help="Build the data-quality gate outputs")
    quality.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    quality.set_defaults(func=cmd_quality_audit)

    architecture = subparsers.add_parser("architecture-subexperiment", help="Run the architecture sweep")
    architecture.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    architecture.add_argument("--epochs", type=int, default=35)
    architecture.add_argument("--smoke", action="store_true")
    architecture.set_defaults(func=cmd_architecture_subexperiment)

    comparison = subparsers.add_parser("main-comparison", help="Run direct vs recursive comparison")
    comparison.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    comparison.add_argument("--architecture", default="A0")
    comparison.add_argument("--modes", nargs="+", default=["M0", "M1"])
    comparison.add_argument("--epochs", type=int, default=35)
    comparison.add_argument("--smoke", action="store_true")
    comparison.set_defaults(func=cmd_main_comparison)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
