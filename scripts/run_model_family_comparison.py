#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


DEFAULT_BUILDINGS = ("U05", "U06", "LIB", "U02B", "SOC", "U03")
DEFAULT_HORIZONS = (1, 2, 4, 6, 8, 12, 16, 20, 24, 36)
DEFAULT_REGIMES = ("per_building", "pooled_same_buildings")
DEFAULT_WEATHER_MODES = ("FW0", "FW1", "FW2")
DEFAULT_MODES = ("M0", "M1", "M2", "M4")
DEFAULT_MODEL_FAMILIES = ("lstm", "xgboost")


def _bootstrap_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "thesis-project":
        project_root = cwd
    elif (cwd / "thesis-project").exists():
        project_root = (cwd / "thesis-project").resolve()
    else:
        project_root = cwd
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("MPLCONFIGDIR", str((project_root / ".mplconfig").resolve()))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_project_root()


def _comparison_api():
    return importlib.import_module("utils.model_family_comparison")


def _pd():
    return importlib.import_module("pandas")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the thesis model-family comparison outside the notebook."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--buildings", nargs="+", default=list(DEFAULT_BUILDINGS))
        subparser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
        subparser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES))
        subparser.add_argument("--weather-modes", nargs="+", default=list(DEFAULT_WEATHER_MODES))
        subparser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
        subparser.add_argument("--lookback-hours", type=int, default=72)
        subparser.add_argument(
            "--results-dir",
            default=str(PROJECT_ROOT / "results" / "model_family_comparison_29032026"),
        )
        subparser.add_argument("--lstm-architecture-id", default="A6")
        subparser.add_argument("--xgb-preset-id", default="P01_md3_lr003_mc5")
        subparser.add_argument("--random-seed", type=int, default=42)
        subparser.add_argument("--batch-size", type=int, default=64)
        subparser.add_argument("--epochs", type=int, default=50)
        subparser.add_argument("--early-stopping-patience", type=int, default=8)
        subparser.add_argument("--learning-rate", type=float, default=1e-3)
        subparser.add_argument("--model-families", nargs="+", default=list(DEFAULT_MODEL_FAMILIES))

    resume_parser = subparsers.add_parser(
        "resume",
        help="Print artifact locations and pair-level resume status.",
    )
    add_config_args(resume_parser)

    sanity_parser = subparsers.add_parser(
        "sanity",
        help="Run preflight checks, with an optional one-pair smoke test.",
    )
    add_config_args(sanity_parser)
    sanity_parser.add_argument("--smoke", action="store_true")
    sanity_parser.add_argument("--require-pass", action="store_true")
    sanity_parser.add_argument("--smoke-building", default="U05")
    sanity_parser.add_argument("--smoke-horizon", type=int, default=8)
    sanity_parser.add_argument("--smoke-mode", default="M0")
    sanity_parser.add_argument("--smoke-weather-mode", default="FW0")

    run_parser = subparsers.add_parser(
        "run",
        help="Run the full comparison matrix from the terminal.",
    )
    add_config_args(run_parser)
    run_parser.add_argument("--sanity-check", action="store_true")
    run_parser.add_argument("--sanity-smoke", action="store_true")
    run_parser.add_argument("--require-sanity-pass", action="store_true")
    run_parser.add_argument("--smoke-building", default="U05")
    run_parser.add_argument("--smoke-horizon", type=int, default=8)
    run_parser.add_argument("--smoke-mode", default="M0")
    run_parser.add_argument("--smoke-weather-mode", default="FW0")
    run_parser.add_argument("--no-resume", action="store_true")
    run_parser.add_argument("--no-save-after-each-pair", action="store_true")
    run_parser.add_argument("--stop-on-error", action="store_true")
    run_parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def _build_config(args: argparse.Namespace):
    api = _comparison_api()
    return api.ExperimentConfig(
        buildings=tuple(args.buildings),
        horizons=tuple(int(h) for h in args.horizons),
        regimes=tuple(args.regimes),
        weather_modes=tuple(args.weather_modes),
        modes=tuple(args.modes),
        lookback_hours=int(args.lookback_hours),
        results_dir=Path(args.results_dir).resolve(),
        lstm_architecture_id=args.lstm_architecture_id,
        xgb_preset_id=args.xgb_preset_id,
        random_seed=int(args.random_seed),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        early_stopping_patience=int(args.early_stopping_patience),
        learning_rate=float(args.learning_rate),
        model_families=tuple(args.model_families),
    )


def _print_heading(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_df(df) -> None:
    if df.empty:
        print("(empty)")
        return
    pd = _pd()
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))


def _run_sanity(
    config,
    *,
    smoke: bool,
    require_pass: bool,
    smoke_building: str,
    smoke_horizon: int,
    smoke_mode: str,
    smoke_weather_mode: str,
):
    api = _comparison_api()
    sanity_df = api.run_sanity_check(
        config,
        run_smoke=smoke,
        smoke_building=smoke_building,
        smoke_horizon_h=smoke_horizon,
        smoke_mode=smoke_mode,
        smoke_weather_mode=smoke_weather_mode,
    )
    _print_heading("Sanity Check")
    _print_df(sanity_df)
    if require_pass and not sanity_df.empty and "status" in sanity_df.columns:
        blocked = sanity_df["status"].astype(str).eq("block")
        if blocked.any():
            blocked_df = sanity_df.loc[blocked, ["check", "detail"]].copy()
            raise RuntimeError(
                "Sanity check has blocking issues:\n"
                + blocked_df.to_string(index=False)
            )
    return sanity_df


def _print_artifact_status(config) -> None:
    _print_heading("Artifact Status")
    api = _comparison_api()
    _print_df(api.build_artifact_status_table(config))


def _print_run_summary(config, outputs) -> None:
    api = _comparison_api()
    paths = api.ensure_results_dirs(config)
    _print_heading("Run Summary")
    print(f"results_dir: {paths['results']}")
    print(f"metrics_rows: {len(outputs.comparison_metrics_df)}")
    print(f"prediction_rows: {len(outputs.comparison_predictions_df)}")
    print(f"summary_rows: {len(outputs.comparison_summary_df)}")
    print(f"coverage_rows: {len(outputs.comparison_coverage_df)}")
    print(f"run_log_rows: {len(outputs.run_log_df)}")
    _print_heading("Recent Run Log")
    _print_df(outputs.run_log_df.tail(12))


def main() -> None:
    args = _parse_args()
    config = _build_config(args)

    _print_heading("Configuration")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"results_dir: {config.results_dir}")
    print(f"buildings: {', '.join(config.buildings)}")
    print(f"horizons: {', '.join(str(h) for h in config.horizons)}")
    print(f"regimes: {', '.join(config.regimes)}")
    print(f"modes: {', '.join(config.modes)}")
    print(f"weather_modes: {', '.join(config.weather_modes)}")
    print(f"model_families: {', '.join(config.model_families)}")

    if args.command == "resume":
        api = _comparison_api()
        _print_artifact_status(config)
        _print_heading("Resume Diagnostics")
        api.print_resume_diagnostics(config)
        return

    if args.command == "sanity":
        _run_sanity(
            config,
            smoke=bool(args.smoke),
            require_pass=bool(args.require_pass),
            smoke_building=args.smoke_building,
            smoke_horizon=int(args.smoke_horizon),
            smoke_mode=args.smoke_mode,
            smoke_weather_mode=args.smoke_weather_mode,
        )
        _print_artifact_status(config)
        return

    if args.command == "run":
        api = _comparison_api()
        if args.sanity_check:
            _run_sanity(
                config,
                smoke=bool(args.sanity_smoke),
                require_pass=bool(args.require_sanity_pass),
                smoke_building=args.smoke_building,
                smoke_horizon=int(args.smoke_horizon),
                smoke_mode=args.smoke_mode,
                smoke_weather_mode=args.smoke_weather_mode,
            )
        _print_heading("Resume Diagnostics")
        api.print_resume_diagnostics(config)
        outputs = api.run_full_comparison(
            config,
            save_artifacts=True,
            verbose=not args.quiet,
            resume_existing=not args.no_resume,
            save_after_each_pair=not args.no_save_after_each_pair,
            continue_on_error=not args.stop_on_error,
        )
        _print_run_summary(config, outputs)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
