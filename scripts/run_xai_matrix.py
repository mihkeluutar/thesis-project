#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


DEFAULT_BUILDINGS = ("U05", "U06", "LIB", "U02B", "SOC", "U03")
DEFAULT_HORIZONS = (2, 4, 6, 8, 12, 16, 20, 24, 36)
DEFAULT_REGIMES = ("per_building", "pooled_same_buildings")
DEFAULT_WEATHER_MODES = ("FW0", "FW1", "FW2")
DEFAULT_MODES = ("M0", "M1", "M2", "M4")
DEFAULT_MODEL_FAMILIES = ("xgboost",)
DEFAULT_SEEDS = (42, 52, 62)


def _bootstrap_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "thesis-project":
        project_root = cwd
    elif (cwd / "thesis-project").exists():
        project_root = (cwd / "thesis-project").resolve()
    else:
        project_root = cwd
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("MPLCONFIGDIR", str((project_root / ".mplconfig").resolve()))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_project_root()


def _mfc():
    return importlib.import_module("utils.model_family_comparison")


def _xns():
    return importlib.import_module("utils.xai_notebook_support")


def _pd():
    return importlib.import_module("pandas")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or inspect the thesis XAI matrix outside the notebook.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--buildings", nargs="+", default=list(DEFAULT_BUILDINGS))
        subparser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
        subparser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES))
        subparser.add_argument("--weather-modes", nargs="+", default=list(DEFAULT_WEATHER_MODES))
        subparser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
        subparser.add_argument("--model-families", nargs="+", default=list(DEFAULT_MODEL_FAMILIES))
        subparser.add_argument("--seed-list", nargs="+", type=int, default=list(DEFAULT_SEEDS))
        subparser.add_argument("--target-kind", default="cum", choices=("cum", "point"))
        subparser.add_argument(
            "--results-dir",
            default=str(PROJECT_ROOT / "results" / "xai_final_31032026"),
        )
        subparser.add_argument("--xgb-preset-id", default="P01_md3_lr003_mc5")
        subparser.add_argument("--pfi-repeats-xgboost", type=int, default=10)
        subparser.add_argument("--pfi-repeats-lstm", type=int, default=3)
        subparser.add_argument("--shap-background-size", type=int, default=96)
        subparser.add_argument("--shap-explain-size", type=int, default=192)
        subparser.add_argument("--lstm-pfi-max-rows", type=int, default=1024)
        subparser.add_argument("--detail-buildings", nargs="*", default=[])
        subparser.add_argument("--detail-horizons", nargs="*", type=int, default=[])
        subparser.add_argument("--detail-weather-mode", default="FW0")

    resume_parser = subparsers.add_parser(
        "resume",
        help="Print saved-artifact status and manifest completeness.",
    )
    add_common_args(resume_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the XAI matrix from the terminal.",
    )
    add_common_args(run_parser)
    run_parser.add_argument("--save-artifacts", action="store_true")
    run_parser.add_argument("--no-resume", action="store_true")
    run_parser.add_argument("--save-after-each-slot", action="store_true")
    run_parser.add_argument("--save-every-n-slots", type=int, default=12)
    run_parser.add_argument("--continue-on-error", action="store_true")
    run_parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def _build_config(args: argparse.Namespace):
    mfc = _mfc()
    xns = _xns()
    horizons = xns.normalize_horizons_for_target_kind(tuple(int(h) for h in args.horizons), args.target_kind)
    return mfc.ExperimentConfig(
        buildings=tuple(args.buildings),
        horizons=horizons,
        regimes=tuple(args.regimes),
        weather_modes=tuple(args.weather_modes),
        modes=tuple(args.modes),
        results_dir=Path(args.results_dir).resolve(),
        xgb_preset_id=args.xgb_preset_id,
        random_seed=int(args.seed_list[0]),
        model_families=tuple(args.model_families),
    )


def _build_detail_requests(args: argparse.Namespace, config) -> set[tuple[str, str, str, str, str, int, int]]:
    detail_buildings = [building for building in args.detail_buildings if building in set(config.buildings)]
    detail_horizons = [int(h) for h in args.detail_horizons if int(h) in set(config.horizons)]
    if not detail_buildings or not detail_horizons:
        return set()
    return {
        (model_family, "per_building", building, mode, str(args.detail_weather_mode), int(horizon_h), int(args.seed_list[0]))
        for model_family in config.model_families
        for building in detail_buildings
        for mode in config.modes
        for horizon_h in detail_horizons
    }


def _print_heading(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_df(df) -> None:
    if df.empty:
        print("(empty)")
        return
    pd = _pd()
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))


def _manifest_for_args(config, args: argparse.Namespace):
    xns = _xns()
    return xns.build_xai_matrix_manifest(
        model_families=tuple(args.model_families),
        regimes=tuple(args.regimes),
        buildings=tuple(args.buildings),
        modes=tuple(args.modes),
        weather_modes=tuple(args.weather_modes),
        horizons=tuple(config.horizons),
        target_kind=str(args.target_kind),
        seed_list=tuple(int(seed) for seed in args.seed_list),
    )


def cmd_resume(args: argparse.Namespace) -> None:
    mfc = _mfc()
    xns = _xns()
    config = _build_config(args)
    manifest_df = _manifest_for_args(config, args)

    _print_heading("Configuration")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"results_dir: {config.results_dir}")
    print(f"target_kind: {args.target_kind}")
    print(f"buildings: {', '.join(config.buildings)}")
    print(f"horizons: {', '.join(str(h) for h in config.horizons)}")
    print(f"modes: {', '.join(config.modes)}")
    print(f"weather_modes: {', '.join(config.weather_modes)}")
    print(f"model_families: {', '.join(config.model_families)}")

    _print_heading("Artifact Status")
    _print_df(xns.build_xai_artifact_status_table(config.results_dir))
    _print_heading("Resume Diagnostics")
    xns.print_xai_resume_diagnostics(
        config.results_dir,
        manifest_df=manifest_df,
        buildings=config.buildings,
    )


def cmd_run(args: argparse.Namespace) -> None:
    mfc = _mfc()
    xns = _xns()
    config = _build_config(args)
    base_frames = mfc.build_base_frame_cache(config)
    detail_requests = _build_detail_requests(args, config)

    pfi_repeats = {
        "xgboost": int(args.pfi_repeats_xgboost),
        "lstm": int(args.pfi_repeats_lstm),
    }
    _print_heading("Configuration")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"results_dir: {config.results_dir}")
    print(f"target_kind: {args.target_kind}")
    print(f"buildings: {', '.join(config.buildings)}")
    print(f"horizons: {', '.join(str(h) for h in config.horizons)}")
    print(f"regimes: {', '.join(config.regimes)}")
    print(f"modes: {', '.join(config.modes)}")
    print(f"weather_modes: {', '.join(config.weather_modes)}")
    print(f"model_families: {', '.join(config.model_families)}")
    print(f"seed_list: {', '.join(str(seed) for seed in args.seed_list)}")
    print(f"detail_requests: {len(detail_requests)}")

    manifest_df = _manifest_for_args(config, args)
    _print_heading("Resume Diagnostics")
    xns.print_xai_resume_diagnostics(
        config.results_dir,
        manifest_df=manifest_df,
        buildings=config.buildings,
    )

    outputs = xns.run_broad_xai_matrix(
        config,
        base_frames,
        regimes=config.regimes,
        buildings=config.buildings,
        modes=config.modes,
        weather_modes=config.weather_modes,
        horizons_by_target_kind={str(args.target_kind): tuple(config.horizons)},
        target_kind=str(args.target_kind),
        seed_list=tuple(int(seed) for seed in args.seed_list),
        model_families=config.model_families,
        pfi_repeats_by_family={family: pfi_repeats.get(family, 10) for family in config.model_families},
        shap_background_size=int(args.shap_background_size),
        shap_explain_size=int(args.shap_explain_size),
        lstm_pfi_max_rows=int(args.lstm_pfi_max_rows) if args.lstm_pfi_max_rows is not None else None,
        detail_requests=detail_requests or None,
        save_artifacts=bool(args.save_artifacts),
        resume_existing=not bool(args.no_resume),
        save_after_each_slot=bool(args.save_after_each_slot),
        save_every_n_slots=int(args.save_every_n_slots) if args.save_every_n_slots is not None else None,
        continue_on_error=bool(args.continue_on_error),
        verbose=bool(args.verbose),
    )

    _print_heading("Run Summary")
    print(f"manifest_rows: {len(outputs['manifest_df'])}")
    print(f"seed_metrics_rows: {len(outputs['seed_metrics_df'])}")
    print(f"seed_grouped_pfi_rows: {len(outputs['seed_grouped_pfi_df'])}")
    print(f"seed_grouped_shap_rows: {len(outputs['seed_grouped_shap_df'])}")
    print(f"seed_agreement_rows: {len(outputs['seed_agreement_df'])}")
    print(f"run_log_rows: {len(outputs['run_log_df'])}")
    _print_heading("Recent Run Log")
    _print_df(outputs["run_log_df"].tail(12))


def main() -> None:
    args = _parse_args()
    if args.command == "resume":
        cmd_resume(args)
        return
    if args.command == "run":
        cmd_run(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
