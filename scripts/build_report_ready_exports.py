#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


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


def _mfc():
    return importlib.import_module("utils.model_family_comparison")


def _xns():
    return importlib.import_module("utils.xai_notebook_support")


def _ordered_unique(values: list[Any], preferred_order: tuple[Any, ...] = ()) -> tuple[Any, ...]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in preferred_order:
        if value in values and value not in seen:
            ordered.append(value)
            seen.add(value)
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return tuple(ordered)


def _normalize_source_dirs(source_dirs: list[str]) -> tuple[Path, ...]:
    return tuple(Path(path).resolve() for path in source_dirs)


def _parse_transition_pairs(values: list[str]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        token = str(value).strip()
        if "-" not in token:
            raise ValueError(f"Expected transition pair like M1-M0, got: {value}")
        lhs, rhs = token.split("-", 1)
        pairs.append((lhs.strip(), rhs.strip()))
    return tuple(pairs)


def _infer_comparison_config_from_outputs(outputs, *, results_dir: Path):
    mfc = _mfc()
    frames = [
        outputs.manifest_df,
        outputs.comparison_metrics_df,
        outputs.comparison_coverage_df,
        outputs.run_log_df,
    ]

    def collect_values(column: str) -> list[Any]:
        values: list[Any] = []
        for frame in frames:
            if column not in frame.columns or frame.empty:
                continue
            series = frame[column].dropna()
            if column == "building":
                series = series.astype(str)
                series = series.loc[series != "POOLED"]
            values.extend(series.tolist())
        return values

    buildings = tuple(sorted({str(value) for value in collect_values("building")})) or tuple(mfc.DEFAULT_BUILDINGS)
    horizons = tuple(sorted({int(value) for value in collect_values("horizon_h")})) or tuple(mfc.DEFAULT_HORIZONS)
    regimes = _ordered_unique(
        [str(value) for value in collect_values("regime")],
        preferred_order=tuple(mfc.DEFAULT_REGIMES),
    ) or tuple(mfc.DEFAULT_REGIMES)
    weather_modes = _ordered_unique(
        [str(value) for value in collect_values("weather_mode")],
        preferred_order=tuple(mfc.CANONICAL_WEATHER_MODE_ORDER),
    ) or tuple(mfc.DEFAULT_WEATHER_MODES)
    modes = _ordered_unique(
        [str(value) for value in collect_values("mode")],
        preferred_order=tuple(mfc.CANONICAL_MODE_ORDER),
    ) or tuple(mfc.DEFAULT_MODES)
    model_families = _ordered_unique(
        [str(value) for value in collect_values("model_family")],
        preferred_order=tuple(mfc.DEFAULT_MODEL_FAMILIES),
    ) or tuple(mfc.DEFAULT_MODEL_FAMILIES)

    return mfc.ExperimentConfig(
        buildings=buildings,
        horizons=horizons,
        regimes=tuple(str(value) for value in regimes),
        weather_modes=tuple(str(value) for value in weather_modes),
        modes=tuple(str(value) for value in modes),
        results_dir=results_dir,
        model_families=tuple(str(value) for value in model_families),
    )


def _load_and_merge_comparison_outputs(source_dirs: tuple[Path, ...]):
    mfc = _mfc()
    source_outputs = []
    for source_dir in source_dirs:
        config = mfc.ExperimentConfig(results_dir=source_dir)
        source_outputs.append(mfc.load_saved_outputs(config))
    merged = mfc.merge_comparison_outputs(*source_outputs)
    config = _infer_comparison_config_from_outputs(merged, results_dir=source_dirs[0])
    return config, merged


def _validate_model_family_metrics(
    outputs,
    *,
    report_dir: Path,
    building: str,
    mode: str,
    weather_mode: str,
    horizon_h: int,
) -> pd.DataFrame:
    mfc = _mfc()
    rows: list[dict[str, Any]] = []
    predictions_df = outputs.comparison_predictions_df.copy()
    metrics_df = outputs.comparison_metrics_df.copy()
    if predictions_df.empty or metrics_df.empty:
        validation_df = pd.DataFrame()
        validation_df.to_csv(report_dir / "metric_recompute_validation.csv", index=False)
        return validation_df

    regimes = [regime for regime in ("per_building", "pooled_same_buildings") if regime in set(metrics_df["regime"].astype(str))]
    model_families = sorted(metrics_df["model_family"].astype(str).unique().tolist())
    for regime in regimes:
        for model_family in model_families:
            pred_slice = predictions_df.loc[
                (predictions_df["regime"].astype(str) == str(regime))
                & (predictions_df["building"].astype(str) == str(building))
                & (predictions_df["model_family"].astype(str) == str(model_family))
                & (predictions_df["mode"].astype(str) == str(mode))
                & (predictions_df["weather_mode"].astype(str) == str(weather_mode))
                & (pd.to_numeric(predictions_df["horizon_h"], errors="coerce") == int(horizon_h))
            ].copy()
            metric_slice = metrics_df.loc[
                (metrics_df["regime"].astype(str) == str(regime))
                & (metrics_df["building"].astype(str) == str(building))
                & (metrics_df["model_family"].astype(str) == str(model_family))
                & (metrics_df["mode"].astype(str) == str(mode))
                & (metrics_df["weather_mode"].astype(str) == str(weather_mode))
                & (pd.to_numeric(metrics_df["horizon_h"], errors="coerce") == int(horizon_h))
            ].copy()
            if pred_slice.empty or metric_slice.empty:
                continue
            eval_slice = pred_slice.copy()
            if "is_heating_eval" in eval_slice.columns:
                eval_mask = eval_slice["is_heating_eval"].fillna(False).astype(bool)
                if eval_mask.any():
                    eval_slice = eval_slice.loc[eval_mask].copy()
            recomputed = mfc.compute_regression_metrics(
                eval_slice["y_true"].to_numpy(dtype=float),
                eval_slice["y_pred"].to_numpy(dtype=float),
            )
            metric_row = metric_slice.iloc[0]
            row = {
                "regime": regime,
                "building": building,
                "model_family": model_family,
                "mode": mode,
                "weather_mode": weather_mode,
                "horizon_h": int(horizon_h),
                "n_prediction_rows": int(len(pred_slice)),
                "n_eval_rows": int(len(eval_slice)),
            }
            max_abs_diff = 0.0
            for metric_name in ("rmse", "wape_pct", "r2", "mae"):
                saved_value = float(metric_row[metric_name])
                recomputed_value = float(recomputed[metric_name])
                diff = recomputed_value - saved_value
                row[f"saved_{metric_name}"] = saved_value
                row[f"recomputed_{metric_name}"] = recomputed_value
                row[f"delta_{metric_name}"] = diff
                max_abs_diff = max(max_abs_diff, abs(diff))
            row["max_abs_diff"] = max_abs_diff
            row["matches_within_1e-9"] = bool(max_abs_diff <= 1e-9)
            rows.append(row)

    validation_df = pd.DataFrame(rows).sort_values(["regime", "model_family"]).reset_index(drop=True) if rows else pd.DataFrame()
    validation_df.to_csv(report_dir / "metric_recompute_validation.csv", index=False)
    return validation_df


def _manifest_key_frame(manifest_df: pd.DataFrame, *, key_cols: list[str]) -> pd.DataFrame:
    if manifest_df.empty:
        return pd.DataFrame(columns=key_cols)
    use_cols = [col for col in key_cols if col in manifest_df.columns]
    return manifest_df.loc[:, use_cols].drop_duplicates().assign(_allowed=True)


def _validate_xai_scope(report_dir: Path) -> pd.DataFrame:
    xns = _xns()
    paths = xns.build_xai_artifact_paths(report_dir)
    manifest_df = xns._read_csv_if_exists(paths["matrix_manifest"])
    key_cols = list(xns.MATRIX_KEY_COLS) + ["seed", "training_scope"]
    allowed = _manifest_key_frame(manifest_df, key_cols=key_cols)
    rows: list[dict[str, Any]] = []
    for artifact_name, filename in (
        ("seed_metrics_df", "seed_metrics"),
        ("seed_grouped_pfi_df", "seed_grouped_pfi"),
        ("seed_grouped_shap_df", "seed_grouped_shap"),
        ("seed_agreement_df", "seed_agreement"),
        ("run_log_df", "run_log"),
    ):
        frame = xns._read_csv_if_exists(paths[filename])
        if frame.empty:
            rows.append(
                {
                    "artifact": artifact_name,
                    "rows": 0,
                    "rows_outside_manifest": 0,
                    "valid": True,
                }
            )
            continue
        use_cols = [col for col in key_cols if col in frame.columns and col in allowed.columns]
        merged = frame.loc[:, use_cols].drop_duplicates().merge(allowed, on=use_cols, how="left")
        outside_count = int(merged["_allowed"].isna().sum())
        rows.append(
            {
                "artifact": artifact_name,
                "rows": int(len(frame)),
                "rows_outside_manifest": outside_count,
                "valid": bool(outside_count == 0),
            }
        )
    validation_df = pd.DataFrame(rows).sort_values("artifact").reset_index(drop=True)
    validation_df.to_csv(report_dir / "xai_scope_validation.csv", index=False)
    return validation_df


def cmd_model_family(args: argparse.Namespace) -> None:
    mfc = _mfc()
    source_dirs = _normalize_source_dirs(args.source_dirs)
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    config, outputs = _load_and_merge_comparison_outputs(source_dirs)
    sample_building = args.sample_building if args.sample_building in set(config.buildings) else config.buildings[0]
    sample_horizon = int(args.sample_horizon if args.sample_horizon in set(config.horizons) else config.horizons[0])
    exports = mfc.export_report_ready_model_family_artifacts(
        outputs,
        config,
        report_dir,
        main_modes=tuple(args.main_modes),
        transition_mode_pairs=_parse_transition_pairs(args.transition_pairs),
        focus_weather_mode=str(args.focus_weather_mode),
        sample_building=sample_building,
        sample_horizon_h=sample_horizon,
    )

    validation_df = pd.DataFrame()
    if not args.skip_validation:
        validation_building = args.validate_building if args.validate_building in set(config.buildings) else config.buildings[0]
        validation_horizon = int(args.validate_horizon if args.validate_horizon in set(config.horizons) else config.horizons[-1])
        validation_df = _validate_model_family_metrics(
            outputs,
            report_dir=report_dir,
            building=validation_building,
            mode=str(args.validate_mode),
            weather_mode=str(args.validate_weather_mode),
            horizon_h=validation_horizon,
        )

    summary_df = pd.DataFrame(
        [
            {"artifact": "source_dirs", "value": "; ".join(str(path) for path in source_dirs)},
            {"artifact": "report_dir", "value": str(report_dir)},
            {"artifact": "manifest_rows", "value": int(len(outputs.manifest_df))},
            {"artifact": "metric_rows", "value": int(len(outputs.comparison_metrics_df))},
            {"artifact": "coverage_rows", "value": int(len(outputs.comparison_coverage_df))},
            {"artifact": "run_log_rows", "value": int(len(outputs.run_log_df))},
            {"artifact": "thesis_per_building_summary_rows", "value": int(len(exports["thesis_per_building_summary.csv"]))},
            {"artifact": "thesis_pooled_summary_rows", "value": int(len(exports["thesis_pooled_summary.csv"]))},
            {"artifact": "metric_recompute_rows", "value": int(len(validation_df))},
        ]
    )
    summary_df.to_csv(report_dir / "report_build_summary.csv", index=False)
    print(summary_df.to_string(index=False))


def _load_and_merge_xai_outputs(source_dirs: tuple[Path, ...]) -> dict[str, Any]:
    xns = _xns()
    outputs = []
    for source_dir in source_dirs:
        outputs.append(xns.load_saved_xai_outputs(source_dir))
    return xns.merge_xai_outputs(*outputs)


def cmd_xai(args: argparse.Namespace) -> None:
    xns = _xns()
    source_dirs = _normalize_source_dirs(args.source_dirs)
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    merged_outputs = _load_and_merge_xai_outputs(source_dirs)
    manifest_df = merged_outputs["manifest_df"].copy()
    exports = xns.export_report_ready_xai_artifacts(
        merged_outputs,
        report_dir,
        manifest_df=manifest_df,
        allowed_model_families=tuple(args.allowed_model_families),
        preferred_mode_pairs=tuple(args.preferred_mode_pairs),
    )
    scope_validation_df = _validate_xai_scope(report_dir)
    coverage_df = exports.get("xai_manifest_coverage.csv", pd.DataFrame())
    missing_df = exports.get("xai_manifest_missing.csv", pd.DataFrame())

    summary_df = pd.DataFrame(
        [
            {"artifact": "source_dirs", "value": "; ".join(str(path) for path in source_dirs)},
            {"artifact": "report_dir", "value": str(report_dir)},
            {"artifact": "manifest_rows", "value": int(len(manifest_df))},
            {"artifact": "xai_metrics_rows", "value": int(len(exports["xai_metrics.csv"]))},
            {"artifact": "xai_grouped_pfi_rows", "value": int(len(exports["xai_grouped_pfi.csv"]))},
            {"artifact": "xai_grouped_shap_rows", "value": int(len(exports["xai_grouped_shap.csv"]))},
            {"artifact": "xai_run_log_latest_rows", "value": int(len(exports["xai_run_log_latest.csv"]))},
            {"artifact": "xai_manifest_coverage_rows", "value": int(len(coverage_df))},
            {"artifact": "xai_manifest_missing_rows", "value": int(len(missing_df))},
            {"artifact": "xai_scope_validation_rows", "value": int(len(scope_validation_df))},
        ]
    )
    summary_df.to_csv(report_dir / "report_build_summary.csv", index=False)
    print(summary_df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build report-ready thesis CSV exports for notebook 12 and notebook 13 in stepwise phases.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    model_parser = subparsers.add_parser(
        "model-family",
        help="Export and validate report-ready model-family comparison artifacts from one or more saved result directories.",
    )
    model_parser.add_argument("--source-dirs", nargs="+", required=True)
    model_parser.add_argument("--report-dir", required=True)
    model_parser.add_argument("--focus-weather-mode", default="FW2")
    model_parser.add_argument("--main-modes", nargs="+", default=["M0", "M1", "M2", "M4"])
    model_parser.add_argument("--transition-pairs", nargs="+", default=["M1-M0", "M2-M0", "M4-M2"])
    model_parser.add_argument("--sample-building", default="U05")
    model_parser.add_argument("--sample-horizon", type=int, default=24)
    model_parser.add_argument("--validate-building", default="U05")
    model_parser.add_argument("--validate-mode", default="M4")
    model_parser.add_argument("--validate-weather-mode", default="FW2")
    model_parser.add_argument("--validate-horizon", type=int, default=24)
    model_parser.add_argument("--skip-validation", action="store_true")
    model_parser.set_defaults(func=cmd_model_family)

    xai_parser = subparsers.add_parser(
        "xai",
        help="Export scoped report-ready XAI artifacts from one or more saved result directories.",
    )
    xai_parser.add_argument("--source-dirs", nargs="+", required=True)
    xai_parser.add_argument("--report-dir", required=True)
    xai_parser.add_argument("--allowed-model-families", nargs="+", default=["xgboost"])
    xai_parser.add_argument(
        "--preferred-mode-pairs",
        nargs="+",
        default=[
            "M1_minus_M0_wape_pct_pp",
            "M2_minus_M0_wape_pct_pp",
            "M3_minus_M2_wape_pct_pp",
            "M4_minus_M3_wape_pct_pp",
        ],
    )
    xai_parser.set_defaults(func=cmd_xai)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
