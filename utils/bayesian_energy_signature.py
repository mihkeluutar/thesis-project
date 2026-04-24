from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import contextlib
import importlib
import io
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "bayesian_es_baselines_20042026"
COMPARISON_DIR = PROJECT_ROOT / "results" / "report_ready_20260405" / "model_family"
FEATURE_METADATA = PROJECT_ROOT / "data" / "features" / "feature_metadata.csv"

os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(RESULTS_DIR / ".cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    from . import model_family_comparison as mfc
except Exception:  # pragma: no cover - notebook import fallback
    import model_family_comparison as mfc


BUILDINGS = ("U05", "U06", "LIB", "U02B", "SOC", "U03")
HORIZONS = (1, 2, 4, 6, 8, 12, 16, 20, 24, 36)
WEATHER_MODES = ("FW0", "FW1", "FW2")
REGIMES = ("per_building", "pooled_same_buildings")
BASELINES = ("Bayes_ES", "Bayes_ARX_ES", "Bayes_ARMAX_ES")
BASE_FEATURE_COLUMNS = {
    "datetime",
    "heat_kwh",
    "heat_kwh_m2",
    "feat_outdoor_temp_c",
    "feat_rh_pct",
    "feat_wind_ms",
    "feat_solar_irradiance_wm2",
    "feat_heat_obs",
    "feat_heat_lag1",
    "feat_heat_roll24h",
    "feat_is_heating_weather",
    "stat_heated_area_m2",
}
FW_SUMMARY_PREFIXES = (
    "feat_fw_temp_mean_",
    "feat_fw_temp_min_",
    "feat_fw_temp_end_",
    "feat_fw_rh_mean_",
    "feat_fw_rh_end_",
)
_FEATURE_FRAME_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


class BayesianEnvironmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    buildings: tuple[str, ...] = BUILDINGS
    horizons: tuple[int, ...] = HORIZONS
    weather_modes: tuple[str, ...] = WEATHER_MODES
    baselines: tuple[str, ...] = BASELINES
    train_end: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2023-12-31 23:00:00"))
    test_start: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2024-01-01 00:00:00"))
    results_dir: Path = RESULTS_DIR
    comparison_dir: Path = COMPARISON_DIR
    feature_metadata: Path = FEATURE_METADATA
    max_train_rows: int | None = 2500
    draws: int = 100
    tune: int = 100
    chains: int = field(default_factory=lambda: min(2, max(1, os.cpu_count() or 1)))
    target_accept: float = 0.92
    cores: int = field(default_factory=lambda: min(2, max(1, os.cpu_count() or 1)))
    random_seed: int = 42
    interval_prob: float = 0.90
    posterior_samples: int = 100
    compute_loo: bool = False
    rerun_failed: bool = False


@dataclass(frozen=True)
class Spec:
    regime: str
    fit_building: str
    baseline: str
    weather_mode: str
    horizon_h: int

    @property
    def run_id(self) -> str:
        return (
            f"{self.regime}__{self.fit_building}__{self.baseline}__"
            f"{self.weather_mode}__h{self.horizon_h:02d}__cumulative"
        )

    @property
    def pooled(self) -> bool:
        return self.regime == "pooled_same_buildings"


def paths(results_dir: Path = RESULTS_DIR) -> dict[str, Path]:
    return {
        "manifest": results_dir / "bayes_es_manifest.csv",
        "predictions": results_dir / "bayes_es_predictions.csv",
        "run_metrics": results_dir / "bayes_es_run_metrics.csv",
        "run_overview": results_dir / "bayes_es_run_overview.csv",
        "metrics_native": results_dir / "bayes_es_metrics_native.csv",
        "metrics_aligned": results_dir / "bayes_es_metrics_aligned.csv",
        "summary": results_dir / "bayes_es_summary.csv",
        "vs_model_family": results_dir / "bayes_es_vs_model_family.csv",
        "diagnostics": results_dir / "bayes_es_trace_diagnostics.csv",
        "metadata": results_dir / "bayes_es_contract_metadata.json",
    }


def ensure_results(config: Config) -> dict[str, Path]:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    (config.results_dir / ".mplconfig").mkdir(exist_ok=True)
    (config.results_dir / ".cache").mkdir(exist_ok=True)
    return paths(config.results_dir)


def reset_outputs(config: Config) -> dict[str, Path]:
    out = ensure_results(config)
    _FEATURE_FRAME_CACHE.clear()
    for name, path in out.items():
        if path.exists():
            path.unlink()
    return out


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def check_environment(config: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check: str, ok: bool, version: str | None = None, detail: str = "") -> None:
        rows.append({"check": check, "ok": bool(ok), "version": version, "detail": detail})

    for name in ("pytensor", "arviz", "pymc"):
        try:
            module = importlib.import_module(name)
            add(name, True, getattr(module, "__version__", "unknown"), "import ok")
        except Exception as exc:
            add(name, False, None, f"{type(exc).__name__}: {exc}")

    mpl_dir = config.results_dir / ".mplconfig"
    try:
        mpl_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=mpl_dir, prefix="write_test_", delete=True) as handle:
            handle.write(b"ok")
        add("MPLCONFIGDIR_writable", True, None, str(mpl_dir))
    except Exception as exc:
        add("MPLCONFIGDIR_writable", False, None, f"{type(exc).__name__}: {exc}")

    return pd.DataFrame(rows)


def require_environment(config: Config) -> None:
    checks = check_environment(config)
    failed = checks.loc[~checks["ok"].astype(bool)]
    if not failed.empty:
        msg = "; ".join(f"{r.check}: {r.detail}" for r in failed.itertuples())
        raise BayesianEnvironmentError(msg)


def build_manifest(config: Config, scope: str = "full_grid") -> pd.DataFrame:
    if scope not in {"per_building", "pooled", "full_grid"}:
        raise ValueError("scope must be one of: per_building, pooled, full_grid")

    specs: list[Spec] = []
    if scope in {"per_building", "full_grid"}:
        for building in config.buildings:
            for horizon_h in config.horizons:
                for weather_mode in config.weather_modes:
                    for baseline in config.baselines:
                        specs.append(Spec("per_building", building, baseline, weather_mode, horizon_h))

    if scope in {"pooled", "full_grid"}:
        for horizon_h in config.horizons:
            for weather_mode in config.weather_modes:
                for baseline in config.baselines:
                    specs.append(Spec("pooled_same_buildings", "POOLED", baseline, weather_mode, horizon_h))

    return pd.DataFrame(
        [
            {
                "run_id": spec.run_id,
                "regime": spec.regime,
                "fit_building": spec.fit_building,
                "baseline": spec.baseline,
                "weather_mode": spec.weather_mode,
                "horizon_h": spec.horizon_h,
                "target": f"target_cum_h{spec.horizon_h}",
                "status": "pending",
                "started_at": "",
                "completed_at": "",
                "error": "",
            }
            for spec in specs
        ]
    )


def merge_manifest(planned: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    if existing.empty or "run_id" not in existing.columns:
        return planned
    keep = existing[["run_id", "status", "started_at", "completed_at", "error"]].drop_duplicates("run_id", keep="last")
    out = planned.drop(columns=["status", "started_at", "completed_at", "error"]).merge(keep, on="run_id", how="left")
    out["status"] = out["status"].fillna("pending")
    for col in ("started_at", "completed_at", "error"):
        out[col] = out[col].fillna("")
    return out[planned.columns]


def read_manifest(config: Config) -> pd.DataFrame:
    path = paths(config.results_dir)["manifest"]
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_manifest(config: Config, manifest: pd.DataFrame) -> None:
    ensure_results(config)["manifest"].parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(paths(config.results_dir)["manifest"], index=False)


def spec_from_row(row: pd.Series) -> Spec:
    return Spec(
        regime=str(row["regime"]),
        fit_building=str(row["fit_building"]),
        baseline=str(row["baseline"]),
        weather_mode=str(row["weather_mode"]),
        horizon_h=int(row["horizon_h"]),
    )


def feature_metadata(config: Config) -> pd.DataFrame:
    return pd.read_csv(config.feature_metadata)


def _feature_csv_path(building: str, meta: pd.DataFrame) -> Path:
    row = meta.loc[meta["building"].astype(str) == str(building)]
    if row.empty:
        raise KeyError(f"Building not found in feature metadata: {building}")
    return (PROJECT_ROOT / str(row.iloc[0]["path_setB"])).resolve()


def _use_feature_column(col: str) -> bool:
    return col in BASE_FEATURE_COLUMNS or any(str(col).startswith(prefix) for prefix in FW_SUMMARY_PREFIXES)


def feature_frame(building: str, config: Config, meta: pd.DataFrame) -> pd.DataFrame:
    cache_key = (str(config.feature_metadata), str(building))
    if cache_key not in _FEATURE_FRAME_CACHE:
        csv_path = _feature_csv_path(building, meta)
        frame = pd.read_csv(
            csv_path,
            parse_dates=["datetime"],
            usecols=lambda col: _use_feature_column(str(col)),
        ).sort_values("datetime").reset_index(drop=True)
        frame["building"] = str(building)
        frame = mfc.add_cumulative_targets(frame, config.horizons)
        _FEATURE_FRAME_CACHE[cache_key] = frame
    return _FEATURE_FRAME_CACHE[cache_key].copy()


def heated_area(frame: pd.DataFrame, building: str) -> float:
    values = pd.to_numeric(frame.get("stat_heated_area_m2"), errors="coerce").dropna()
    values = values.loc[values > 0]
    if not values.empty:
        return float(values.iloc[0])
    ratio = pd.to_numeric(frame["heat_kwh"], errors="coerce") / pd.to_numeric(frame["heat_kwh_m2"], errors="coerce").replace(0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio.loc[ratio > 0]
    if ratio.empty:
        raise ValueError(f"No heated area for {building}")
    return float(ratio.median())


def temp_col(frame: pd.DataFrame, weather_mode: str, horizon_h: int) -> str:
    if weather_mode == "FW0":
        return "feat_outdoor_temp_c"
    tag = f"h{horizon_h:02d}"
    for col in (f"feat_fw_temp_mean_{tag}", f"feat_fw_temp_end_{tag}", f"feat_fw_temp_min_{tag}", "feat_outdoor_temp_c"):
        if col in frame.columns:
            return col
    raise KeyError(f"No temperature feature for {weather_mode} h={horizon_h}")


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").astype(float)


def scale(train: pd.Series, full: pd.Series) -> tuple[pd.Series, float, float]:
    valid = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    mean = float(valid.mean()) if not valid.empty else 0.0
    sd = float(valid.std(ddof=0)) if len(valid) > 1 else 1.0
    if not np.isfinite(sd) or sd < 1e-6:
        sd = 1.0
    return ((full.fillna(mean) - mean) / sd).astype(float), mean, sd


def residual_proxy(frame: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    hdh = np.maximum(0.0, 16.0 - out["es_temp_c"].to_numpy(dtype=float))
    x = np.column_stack([np.ones(len(out)), hdh, hdh * out["es_wind_ms"], out["es_solar_wm2"]])
    y = out["heat_obs_wm2"].to_numpy(dtype=float)
    valid = np.asarray(train_mask) & np.isfinite(y) & np.isfinite(x).all(axis=1)
    coef = np.zeros(x.shape[1])
    if valid.sum() > x.shape[1] + 5:
        coef = np.linalg.solve(x[valid].T @ x[valid] + 10.0 * np.eye(x.shape[1]), x[valid].T @ y[valid])
    resid = pd.Series(y - x @ coef, index=out.index)
    out["ma_resid_lag1_wm2"] = resid.shift(1)
    out["ma_resid_roll24_wm2"] = resid.shift(1).rolling(24, min_periods=3).mean()
    return out


def sample_train_rows(train: pd.DataFrame, config: Config) -> pd.DataFrame:
    if config.max_train_rows is None or len(train) <= config.max_train_rows:
        return train.reset_index(drop=True)
    rng = np.random.default_rng(config.random_seed)
    work = train.copy()
    work["_stratum"] = pd.to_datetime(work["datetime"]).dt.month.astype(str) + "_" + work["is_heating_eval"].astype(str)
    counts = work["_stratum"].value_counts()
    chosen: list[int] = []
    for stratum, count in counts.items():
        quota = max(1, round(config.max_train_rows * count / len(work)))
        idx = work.index[work["_stratum"] == stratum].to_numpy()
        chosen.extend(rng.choice(idx, min(quota, len(idx)), replace=False).tolist())
    if len(chosen) > config.max_train_rows:
        chosen = rng.choice(np.array(chosen), config.max_train_rows, replace=False).tolist()
    return work.loc[sorted(chosen)].drop(columns="_stratum").reset_index(drop=True)


def prepare_building(building: str, spec: Spec, config: Config, meta: pd.DataFrame) -> pd.DataFrame:
    frame = feature_frame(building, config, meta)
    frame, _ = mfc.apply_weather_mode(frame, spec.weather_mode, spec.horizon_h)
    # Consolidate blocks before adding derived columns to avoid fragmentation overhead.
    frame = frame.copy()
    if frame.columns.duplicated().any():
        frame = frame.loc[:, ~frame.columns.duplicated(keep="last")].copy()
    area = heated_area(frame, building)
    target = f"target_cum_h{spec.horizon_h}"
    tcol = temp_col(frame, spec.weather_mode, spec.horizon_h)

    derived = pd.DataFrame(
        {
            "building": building,
            "heated_area_m2": area,
            "target_wm2": num(frame, target) / spec.horizon_h / area * 1000.0,
            "es_temp_c": num(frame, tcol),
            "es_temp_source_col": tcol,
            "es_wind_ms": num(frame, "feat_wind_ms"),
            "es_solar_wm2": num(frame, "feat_solar_irradiance_wm2"),
            "heat_obs_wm2": num(frame, "feat_heat_obs") / area * 1000.0,
            "heat_lag1_wm2": num(frame, "feat_heat_lag1") / area * 1000.0,
            "heat_roll24_wm2": num(frame, "feat_heat_roll24h") / area * 1000.0,
            "is_heating_eval": mfc.heating_mask(frame).astype(bool),
        },
        index=frame.index,
    )
    overwrite_cols = [col for col in derived.columns if col in frame.columns]
    if overwrite_cols:
        frame = frame.drop(columns=overwrite_cols)
    frame = pd.concat([frame, derived], axis=1, copy=False)

    split = mfc.build_split_spec(
        frame,
        spec.horizon_h,
        mfc.ExperimentConfig(
            buildings=config.buildings,
            horizons=config.horizons,
            train_end=config.train_end,
            test_start=config.test_start,
        ),
    )
    split_flags = pd.DataFrame(
        {
            "is_train": split.train_issue_mask.astype(bool).to_numpy(),
            "is_eval": split.test_issue_mask.astype(bool).to_numpy(),
        },
        index=frame.index,
    )
    overwrite_split_cols = [col for col in split_flags.columns if col in frame.columns]
    if overwrite_split_cols:
        frame = frame.drop(columns=overwrite_split_cols)
    frame = pd.concat([frame, split_flags], axis=1, copy=False)
    if frame.columns.duplicated().any():
        frame = frame.loc[:, ~frame.columns.duplicated(keep="last")].copy()
    frame = residual_proxy(frame, frame["is_train"])
    return frame


def prepare_dataset(spec: Spec, config: Config, meta: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    meta = feature_metadata(config) if meta is None else meta
    buildings = config.buildings if spec.pooled else (spec.fit_building,)
    frame = pd.concat([prepare_building(b, spec, config, meta) for b in buildings], ignore_index=True)
    if frame.columns.duplicated().any():
        frame = frame.loc[:, ~frame.columns.duplicated(keep="last")].copy()
    blevels = sorted(frame["building"].unique())
    frame["building_idx"] = frame["building"].map({b: i for i, b in enumerate(blevels)}).astype(int)

    needed = ["target_wm2", f"target_cum_h{spec.horizon_h}", "es_temp_c", "es_wind_ms", "es_solar_wm2", "heat_obs_wm2"]
    if spec.baseline in {"Bayes_ARX_ES", "Bayes_ARMAX_ES"}:
        needed += ["heat_lag1_wm2", "heat_roll24_wm2"]
    if spec.baseline == "Bayes_ARMAX_ES":
        needed += ["ma_resid_lag1_wm2", "ma_resid_roll24_wm2"]
    valid = np.ones(len(frame), dtype=bool)
    for col in needed:
        valid &= num(frame, col).replace([np.inf, -np.inf], np.nan).notna().to_numpy()
    frame = frame.loc[valid].sort_values(["building", "datetime"]).reset_index(drop=True)

    train_mask = frame["is_train"].astype(bool)
    for col in ("heat_obs_wm2", "heat_lag1_wm2", "heat_roll24_wm2", "ma_resid_lag1_wm2", "ma_resid_roll24_wm2"):
        frame[f"{col}_z"], _, _ = scale(frame.loc[train_mask, col], num(frame, col))

    train = sample_train_rows(frame.loc[frame["is_train"]].copy(), config)
    eval_frame = frame.loc[frame["is_eval"]].copy().reset_index(drop=True)
    return train, eval_frame, blevels


def contract_checks(config: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check: str, ok: bool, detail: str = "") -> None:
        rows.append({"check": check, "ok": bool(ok), "detail": detail})

    meta = feature_metadata(config)
    add("six_buildings_in_feature_metadata", set(BUILDINGS).issubset(set(meta["building"].astype(str))))
    sample = feature_frame(config.buildings[0], config, meta)
    missing_targets = [f"target_cum_h{h}" for h in HORIZONS if f"target_cum_h{h}" not in sample.columns]
    add("all_cumulative_targets_created", not missing_targets, ",".join(missing_targets))

    missing_area = []
    for building in BUILDINGS:
        try:
            heated_area(feature_frame(building, config, meta), building)
        except Exception:
            missing_area.append(building)
    add("heated_area_for_six_buildings", not missing_area, ",".join(missing_area))

    failures = []
    for weather_mode in WEATHER_MODES:
        try:
            test_spec = Spec("per_building", config.buildings[0], "Bayes_ES", weather_mode, 8)
            frame = prepare_building(config.buildings[0], test_spec, config, meta)
            temp_col(frame, weather_mode, 8)
        except Exception as exc:
            failures.append(f"{weather_mode}: {type(exc).__name__}: {exc}")
    add("weather_modes_select_features", not failures, " | ".join(failures))

    missing_files = [name for name in ("comparison_predictions.csv", "comparison_metrics.csv") if not (config.comparison_dir / name).exists()]
    add("notebook12_artifacts_exist", not missing_files, ",".join(missing_files))
    return pd.DataFrame(rows)


def _pm() -> tuple[Any, Any, Any]:
    try:
        import arviz as az
        import pymc as pm
        import pytensor.tensor as pt
        return pm, az, pt
    except Exception as exc:
        raise BayesianEnvironmentError(f"{type(exc).__name__}: {exc}") from exc


@contextlib.contextmanager
def quiet_bayesian_output() -> Iterable[None]:
    logger_names = [
        "pymc",
        "pymc.sampling",
        "pymc.sampling.mcmc",
        "arviz",
        "pytensor",
    ]
    previous_levels = {name: logging.getLogger(name).level for name in logger_names}
    previous_disable = logging.root.manager.disable
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        logging.disable(logging.CRITICAL)
        for name in logger_names:
            logging.getLogger(name).setLevel(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                yield
    finally:
        logging.disable(previous_disable)
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)


def param(pm: Any, name: str, n: int, loc: float, scale_: float, positive: bool) -> Any:
    if n == 1:
        if positive:
            return pm.LogNormal(name, mu=np.log(max(loc, 1e-4)), sigma=scale_, shape=n)
        return pm.Normal(name, mu=loc, sigma=scale_, shape=n)
    if positive:
        mu = pm.Normal(f"{name}_log_mu", np.log(max(loc, 1e-4)), scale_)
        sd = pm.HalfNormal(f"{name}_log_sd", scale_)
        return pm.LogNormal(name, mu=mu, sigma=sd, shape=n)
    mu = pm.Normal(f"{name}_mu", loc, scale_)
    sd = pm.HalfNormal(f"{name}_sd", scale_)
    return pm.Normal(name, mu=mu, sigma=sd, shape=n)


def coef(pm: Any, name: str, n_buildings: int, n_cols: int, hierarchical: bool) -> Any:
    if hierarchical and n_buildings > 1:
        mu = pm.Normal(f"{name}_mu", 0.0, 1.5, shape=n_cols)
        sd = pm.HalfNormal(f"{name}_sd", 1.5, shape=n_cols)
        return pm.Normal(name, mu=mu, sigma=sd, shape=(n_buildings, n_cols))
    return pm.Normal(name, 0.0, 1.5, shape=(n_buildings, n_cols))


def build_model(spec: Spec, train: pd.DataFrame, buildings: list[str], config: Config) -> tuple[Any, Any]:
    pm, az, pt = _pm()
    n_b = len(buildings)
    idx = train["building_idx"].to_numpy(dtype=int)
    temp = train["es_temp_c"].to_numpy(dtype=float)
    wind = train["es_wind_ms"].to_numpy(dtype=float)
    solar = train["es_solar_wm2"].to_numpy(dtype=float)
    y = train["target_wm2"].to_numpy(dtype=float)

    with pm.Model() as model:
        ua0 = param(pm, "ua0", n_b, 2.0, 0.8, True)
        uawind = param(pm, "uawind", n_b, 0.05, 0.9, True)
        tbase = param(pm, "tbase", n_b, 16.0, 3.0, False)
        solar_gain = param(pm, "solar_gain", n_b, 0.01, 0.8, True)
        phi_base = param(pm, "phi_base", n_b, 1.0, 1.0, True)
        k = param(pm, "smooth_k", n_b, 0.25, 0.7, True) + 0.03

        winter = (ua0[idx] + uawind[idx] * wind) * (tbase[idx] - temp) - solar_gain[idx] * solar + phi_base[idx]
        base = phi_base[idx]
        mu = pt.logsumexp(pt.stack([k[idx] * winter, k[idx] * base], axis=0), axis=0) / k[idx]

        if spec.baseline in {"Bayes_ARX_ES", "Bayes_ARMAX_ES"}:
            ar_cols = ["heat_obs_wm2_z", "heat_lag1_wm2_z", "heat_roll24_wm2_z"]
            ar = train[ar_cols].to_numpy(dtype=float)
            beta_ar = coef(pm, "beta_ar", n_b, len(ar_cols), spec.pooled)
            mu = mu + pt.sum(ar * beta_ar[idx], axis=1)

        if spec.baseline == "Bayes_ARMAX_ES":
            ma_cols = ["ma_resid_lag1_wm2_z", "ma_resid_roll24_wm2_z"]
            ma = train[ma_cols].to_numpy(dtype=float)
            beta_ma = coef(pm, "beta_ma", n_b, len(ma_cols), spec.pooled)
            mu = mu + pt.sum(ma * beta_ma[idx], axis=1)

        pm.Deterministic("mu", pt.maximum(mu, 0.0))
        sigma = pm.HalfNormal("sigma", max(2.0, float(np.nanstd(y))))
        pm.Normal("y", mu=pt.maximum(mu, 0.0), sigma=sigma, observed=y)
    return model, az


def flat(idata: Any, name: str) -> np.ndarray:
    da = idata.posterior[name].stack(sample=("chain", "draw"))
    dims = [d for d in da.dims if d != "sample"]
    return np.asarray(da.transpose("sample", *dims).values)


def subset(values: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if len(values) <= max_n:
        return values
    rng = np.random.default_rng(seed)
    return values[np.sort(rng.choice(np.arange(len(values)), max_n, replace=False))]


def mu_samples(spec: Spec, idata: Any, frame: pd.DataFrame, config: Config) -> np.ndarray:
    n = config.posterior_samples
    ua0 = subset(flat(idata, "ua0"), n, config.random_seed)
    take = len(ua0)
    uawind = subset(flat(idata, "uawind"), take, config.random_seed)
    tbase = subset(flat(idata, "tbase"), take, config.random_seed)
    solar_gain = subset(flat(idata, "solar_gain"), take, config.random_seed)
    phi_base = subset(flat(idata, "phi_base"), take, config.random_seed)
    k = np.maximum(subset(flat(idata, "smooth_k"), take, config.random_seed), 1e-4)
    b = frame["building_idx"].to_numpy(dtype=int)

    temp = frame["es_temp_c"].to_numpy(dtype=float)[None, :]
    wind = frame["es_wind_ms"].to_numpy(dtype=float)[None, :]
    solar = frame["es_solar_wm2"].to_numpy(dtype=float)[None, :]
    winter = (ua0[:, b] + uawind[:, b] * wind) * (tbase[:, b] - temp) - solar_gain[:, b] * solar + phi_base[:, b]
    base = phi_base[:, b]
    mu = np.logaddexp(k[:, b] * winter, k[:, b] * base) / k[:, b]

    if spec.baseline in {"Bayes_ARX_ES", "Bayes_ARMAX_ES"}:
        ar = frame[["heat_obs_wm2_z", "heat_lag1_wm2_z", "heat_roll24_wm2_z"]].to_numpy(dtype=float)
        beta_ar = subset(flat(idata, "beta_ar"), take, config.random_seed)
        mu = mu + np.einsum("nf,snf->sn", ar, beta_ar[:, b, :])
    if spec.baseline == "Bayes_ARMAX_ES":
        ma = frame[["ma_resid_lag1_wm2_z", "ma_resid_roll24_wm2_z"]].to_numpy(dtype=float)
        beta_ma = subset(flat(idata, "beta_ma"), take, config.random_seed)
        mu = mu + np.einsum("nf,snf->sn", ma, beta_ma[:, b, :])
    return np.maximum(mu, 0.0)


def predictions_from_idata(spec: Spec, idata: Any, eval_frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    mu = mu_samples(spec, idata, eval_frame, config)
    sigma = subset(flat(idata, "sigma"), len(mu), config.random_seed)
    rng = np.random.default_rng(config.random_seed)
    y_wm2 = np.maximum(rng.normal(mu, sigma[:, None]), 0.0)
    y_kwh = y_wm2 * eval_frame["heated_area_m2"].to_numpy(dtype=float)[None, :] * spec.horizon_h / 1000.0
    lo = (1.0 - config.interval_prob) / 2.0
    hi = 1.0 - lo
    target = f"target_cum_h{spec.horizon_h}"

    out = pd.DataFrame(
        {
            "run_id": spec.run_id,
            "regime": spec.regime,
            "fit_building": spec.fit_building,
            "building": eval_frame["building"].astype(str).to_numpy(),
            "model_family": "bayesian_es",
            "baseline": spec.baseline,
            "weather_mode": spec.weather_mode,
            "horizon_h": spec.horizon_h,
            "datetime": pd.to_datetime(eval_frame["datetime"]).astype(str).to_numpy(),
            "y_true": eval_frame[target].to_numpy(dtype=float),
            "y_pred": y_kwh.mean(axis=0),
            "y_pred_lower": np.quantile(y_kwh, lo, axis=0),
            "y_pred_upper": np.quantile(y_kwh, hi, axis=0),
            "target_wm2": eval_frame["target_wm2"].to_numpy(dtype=float),
            "pred_wm2": y_wm2.mean(axis=0),
            "is_heating_eval": eval_frame["is_heating_eval"].astype(bool).to_numpy(),
        }
    )
    out["abs_error"] = (out["y_true"] - out["y_pred"]).abs()
    return out


def run_metric_row(spec: Spec, pred: pd.DataFrame) -> pd.DataFrame:
    eval_pred = pred.loc[pred["is_heating_eval"].astype(bool)].copy()
    metrics = mfc.compute_regression_metrics(eval_pred["y_true"].to_numpy(float), eval_pred["y_pred"].to_numpy(float))
    coverage = np.nan
    if len(eval_pred):
        coverage = ((eval_pred["y_true"] >= eval_pred["y_pred_lower"]) & (eval_pred["y_true"] <= eval_pred["y_pred_upper"])).mean()
    return pd.DataFrame(
        [
            {
                "run_id": spec.run_id,
                "regime": spec.regime,
                "fit_building": spec.fit_building,
                "model_family": "bayesian_es",
                "baseline": spec.baseline,
                "weather_mode": spec.weather_mode,
                "horizon_h": spec.horizon_h,
                "n_eval_buildings": int(eval_pred["building"].astype(str).nunique()) if len(eval_pred) else 0,
                "n_test_rows": int(len(eval_pred)),
                "interval_coverage": coverage,
                **metrics,
            }
        ]
    )


def metric_rows(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["run_id", "regime", "fit_building", "building", "model_family", "baseline", "weather_mode", "horizon_h"]
    for key, group in pred.groupby(keys, dropna=False):
        eval_group = group.loc[group["is_heating_eval"].astype(bool)]
        metrics = mfc.compute_regression_metrics(eval_group["y_true"].to_numpy(float), eval_group["y_pred"].to_numpy(float))
        coverage = np.nan
        if len(eval_group):
            coverage = ((eval_group["y_true"] >= eval_group["y_pred_lower"]) & (eval_group["y_true"] <= eval_group["y_pred_upper"])).mean()
        rows.append(dict(zip(keys, key, strict=True), n_test_rows=len(eval_group), interval_coverage=coverage, **metrics))
    return pd.DataFrame(rows)


def diagnostics(spec: Spec, idata: Any, train: pd.DataFrame, eval_frame: pd.DataFrame, pred: pd.DataFrame, config: Config) -> pd.DataFrame:
    _, az, _ = _pm()
    row: dict[str, Any] = {
        "run_id": spec.run_id,
        "regime": spec.regime,
        "fit_building": spec.fit_building,
        "baseline": spec.baseline,
        "weather_mode": spec.weather_mode,
        "horizon_h": spec.horizon_h,
        "n_train_rows": len(train),
        "n_eval_rows": len(eval_frame),
        "max_r_hat": np.nan,
        "min_ess_bulk": np.nan,
        "divergences": np.nan,
        "bayesian_r2_approx": np.nan,
        "loo_elpd": np.nan,
        "loo_p": np.nan,
        "posterior_interval_coverage": np.nan,
    }
    try:
        row["divergences"] = int(np.asarray(idata.sample_stats["diverging"].values).sum())
    except Exception:
        pass
    try:
        n_chains = int(idata.posterior.sizes.get("chain", 0))
    except Exception:
        n_chains = 0
    if n_chains >= 2:
        try:
            row["max_r_hat"] = float(np.nanmax(az.rhat(idata).to_array().values))
            row["min_ess_bulk"] = float(np.nanmin(az.ess(idata, method="bulk").to_array().values))
        except Exception:
            pass
    try:
        mu = mu_samples(spec, idata, train, config)
        y = train["target_wm2"].to_numpy(float)[None, :]
        row["bayesian_r2_approx"] = float(np.mean(np.var(mu, axis=1) / (np.var(mu, axis=1) + np.var(y - mu, axis=1) + 1e-9)))
    except Exception:
        pass
    if config.compute_loo:
        try:
            loo = az.loo(idata)
            row["loo_elpd"] = float(loo.elpd_loo)
            row["loo_p"] = float(loo.p_loo)
        except Exception:
            pass
    eval_pred = pred.loc[pred["is_heating_eval"].astype(bool)]
    if len(eval_pred):
        row["posterior_interval_coverage"] = float(((eval_pred["y_true"] >= eval_pred["y_pred_lower"]) & (eval_pred["y_true"] <= eval_pred["y_pred_upper"])).mean())
    return pd.DataFrame([row])


def fit_spec(spec: Spec, config: Config, meta: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_environment(config)
    meta = feature_metadata(config) if meta is None else meta
    train, eval_frame, buildings = prepare_dataset(spec, config, meta)
    if train.empty:
        raise ValueError(f"No training rows after filtering for {spec.run_id}")
    if eval_frame.empty:
        raise ValueError(f"No evaluation rows after filtering for {spec.run_id}")
    model, _ = build_model(spec, train, buildings, config)
    pm, _, _ = _pm()
    with quiet_bayesian_output():
        with model:
            idata = pm.sample(
                draws=config.draws,
                tune=config.tune,
                chains=config.chains,
                target_accept=config.target_accept,
                cores=config.cores,
                random_seed=config.random_seed,
                init="adapt_diag",
                progressbar=False,
                compute_convergence_checks=False,
                idata_kwargs={"log_likelihood": config.compute_loo},
            )
    pred = predictions_from_idata(spec, idata, eval_frame, config)
    with quiet_bayesian_output():
        diag = diagnostics(spec, idata, train, eval_frame, pred, config)
    return pred, run_metric_row(spec, pred), diag


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def upsert_csv(path: Path, frame: pd.DataFrame, key_cols: list[str] | tuple[str, ...]) -> None:
    if frame.empty:
        return

    key_cols = list(key_cols)
    out = frame.copy()
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.empty and all(col in existing.columns for col in key_cols):
            keys = frame[key_cols].drop_duplicates().assign(_drop=1)
            existing = existing.merge(keys, on=key_cols, how="left")
            existing = existing.loc[existing["_drop"].isna()].drop(columns="_drop")
            out = pd.concat([existing, frame], ignore_index=True)
    out.to_csv(path, index=False)


def read_run_metrics(config: Config) -> pd.DataFrame:
    path = paths(config.results_dir)["run_metrics"]
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def backfill_run_metrics_from_predictions(
    config: Config,
    overwrite: bool = False,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    p = ensure_results(config)
    run_metrics_path = p["run_metrics"]
    if run_metrics_path.exists() and not overwrite:
        return pd.read_csv(run_metrics_path)

    pred_path = p["predictions"]
    if not pred_path.exists():
        return pd.DataFrame()

    usecols = [
        "run_id",
        "regime",
        "fit_building",
        "baseline",
        "weather_mode",
        "horizon_h",
        "building",
        "y_true",
        "y_pred",
        "y_pred_lower",
        "y_pred_upper",
        "is_heating_eval",
    ]
    accum: dict[str, dict[str, Any]] = {}
    for chunk in pd.read_csv(pred_path, usecols=usecols, chunksize=chunksize):
        if chunk.empty:
            continue
        chunk = chunk.loc[chunk["is_heating_eval"].fillna(False).astype(bool)].copy()
        if chunk.empty:
            continue
        chunk["abs_error"] = (chunk["y_true"] - chunk["y_pred"]).abs()
        chunk["sq_error"] = np.square(chunk["y_true"] - chunk["y_pred"])
        chunk["abs_y_true"] = chunk["y_true"].abs()
        chunk["interval_hit"] = ((chunk["y_true"] >= chunk["y_pred_lower"]) & (chunk["y_true"] <= chunk["y_pred_upper"])).astype(int)

        for run_id, group in chunk.groupby("run_id", dropna=False):
            first = group.iloc[0]
            state = accum.setdefault(
                str(run_id),
                {
                    "run_id": str(run_id),
                    "regime": str(first["regime"]),
                    "fit_building": str(first["fit_building"]),
                    "model_family": "bayesian_es",
                    "baseline": str(first["baseline"]),
                    "weather_mode": str(first["weather_mode"]),
                    "horizon_h": int(first["horizon_h"]),
                    "n_test_rows": 0,
                    "sum_abs_error": 0.0,
                    "sum_sq_error": 0.0,
                    "sum_y": 0.0,
                    "sum_y2": 0.0,
                    "sum_abs_y": 0.0,
                    "interval_hits": 0,
                    "buildings": set(),
                },
            )
            state["n_test_rows"] += int(len(group))
            state["sum_abs_error"] += float(group["abs_error"].sum())
            state["sum_sq_error"] += float(group["sq_error"].sum())
            state["sum_y"] += float(group["y_true"].sum())
            state["sum_y2"] += float(np.square(group["y_true"]).sum())
            state["sum_abs_y"] += float(group["abs_y_true"].sum())
            state["interval_hits"] += int(group["interval_hit"].sum())
            state["buildings"].update(group["building"].astype(str).unique().tolist())

    rows: list[dict[str, Any]] = []
    for state in accum.values():
        n = int(state["n_test_rows"])
        sse = float(state["sum_sq_error"])
        sae = float(state["sum_abs_error"])
        sum_y = float(state["sum_y"])
        sum_y2 = float(state["sum_y2"])
        sum_abs_y = float(state["sum_abs_y"])
        if n:
            rmse = float(np.sqrt(sse / n))
            mae = float(sae / n)
            coverage = float(state["interval_hits"] / n)
            denom = sum_abs_y
            wape = float(100.0 * sae / denom) if denom > 0 else np.nan
        else:
            rmse = np.nan
            mae = np.nan
            coverage = np.nan
            wape = np.nan
        if n < 2:
            r2 = np.nan
        else:
            sst = sum_y2 - (sum_y**2 / n)
            if abs(sst) <= 1e-12:
                r2 = 1.0 if sse <= 1e-12 else 0.0
            else:
                r2 = float(1.0 - (sse / sst))
        rows.append(
            {
                "run_id": state["run_id"],
                "regime": state["regime"],
                "fit_building": state["fit_building"],
                "model_family": state["model_family"],
                "baseline": state["baseline"],
                "weather_mode": state["weather_mode"],
                "horizon_h": state["horizon_h"],
                "n_eval_buildings": int(len(state["buildings"])),
                "n_test_rows": n,
                "interval_coverage": coverage,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "wape_pct": wape,
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["regime", "fit_building", "baseline", "weather_mode", "horizon_h"],
        kind="stable",
    ).reset_index(drop=True)
    if not out.empty:
        out.to_csv(run_metrics_path, index=False)
    return out


def write_run_overview(config: Config) -> pd.DataFrame:
    p = ensure_results(config)
    overview = read_manifest(config).copy()
    if overview.empty:
        return overview

    run_metrics = read_run_metrics(config)
    if not run_metrics.empty:
        metric_cols = [col for col in run_metrics.columns if col not in {"regime", "fit_building", "baseline", "weather_mode", "horizon_h"}]
        overview = overview.merge(run_metrics[metric_cols], on="run_id", how="left")

    diagnostics_df = pd.read_csv(p["diagnostics"]) if p["diagnostics"].exists() else pd.DataFrame()
    if not diagnostics_df.empty:
        diag_cols = [col for col in diagnostics_df.columns if col not in {"regime", "fit_building", "baseline", "weather_mode", "horizon_h"}]
        overview = overview.merge(diagnostics_df[diag_cols], on="run_id", how="left")

    overview.to_csv(p["run_overview"], index=False)
    return overview


def run_manifest(config: Config, manifest: pd.DataFrame, resume: bool = True) -> pd.DataFrame:
    out = merge_manifest(manifest, read_manifest(config)) if resume else manifest.copy()
    save_manifest(config, out)
    p = ensure_results(config)
    meta = feature_metadata(config)
    total = len(out)
    for idx, row in out.iterrows():
        status = str(row.get("status", "pending"))
        if resume and status in {"completed", "skipped"}:
            continue
        if resume and status == "failed" and not config.rerun_failed:
            continue
        spec = spec_from_row(row)
        out.loc[idx, ["status", "started_at", "error"]] = ["running", now_utc(), ""]
        save_manifest(config, out)
        try:
            pred, metrics, diag = fit_spec(spec, config, meta)
            upsert_csv(p["run_metrics"], metrics, key_cols=["run_id"])
            upsert_csv(p["diagnostics"], diag, key_cols=["run_id"])
            out.loc[idx, ["status", "completed_at", "error"]] = ["completed", now_utc(), ""]
            processed = int(out["status"].isin(["completed", "failed", "skipped"]).sum())
            print(f"[{processed}/{total}] completed {spec.run_id}", flush=True)
        except ValueError as exc:
            out.loc[idx, ["status", "completed_at", "error"]] = ["skipped", now_utc(), str(exc)]
            processed = int(out["status"].isin(["completed", "failed", "skipped"]).sum())
            print(f"[{processed}/{total}] skipped {spec.run_id}: {exc}", flush=True)
        except Exception as exc:
            out.loc[idx, ["status", "completed_at", "error"]] = ["failed", now_utc(), f"{type(exc).__name__}: {exc}"]
            processed = int(out["status"].isin(["completed", "failed", "skipped"]).sum())
            print(f"[{processed}/{total}] failed {spec.run_id}: {type(exc).__name__}: {exc}", flush=True)
        save_manifest(config, out)
    write_run_overview(config)
    return out


def load_artifacts(config: Config) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path) if path.exists() else pd.DataFrame() for name, path in paths(config.results_dir).items() if path.suffix == ".csv"}


def comparison_inventory(config: Config) -> pd.DataFrame:
    metrics = pd.read_csv(config.comparison_dir / "comparison_metrics.csv")
    cols = ["regime", "building", "model_family", "mode", "weather_mode", "horizon_h", "n_test_rows"]
    return metrics[cols].drop_duplicates().sort_values(cols[:-1]).reset_index(drop=True)


def aligned_metrics(config: Config, predictions: pd.DataFrame | None = None, chunksize: int = 500_000) -> pd.DataFrame:
    p = ensure_results(config)
    if predictions is None:
        usecols = [
            "run_id",
            "regime",
            "fit_building",
            "building",
            "baseline",
            "weather_mode",
            "horizon_h",
            "datetime",
            "y_true",
            "y_pred",
            "y_pred_lower",
            "y_pred_upper",
        ]
        predictions = (
            pd.read_csv(p["predictions"], usecols=usecols, parse_dates=["datetime"])
            if p["predictions"].exists()
            else pd.DataFrame()
        )
    if predictions.empty:
        return pd.DataFrame()

    base = predictions.copy()
    base["datetime"] = pd.to_datetime(base["datetime"])
    base_key = ["regime", "building", "weather_mode", "horizon_h", "datetime"]
    base = base[base_key + ["run_id", "fit_building", "baseline", "y_true", "y_pred", "y_pred_lower", "y_pred_upper"]]
    base = base.drop_duplicates(["run_id", *base_key], keep="last")
    slots = base[["regime", "building", "weather_mode", "horizon_h"]].drop_duplicates()

    chunks = []
    usecols = ["regime", "building", "model_family", "mode", "weather_mode", "horizon_h", "datetime", "y_true", "is_heating_eval"]
    for chunk in pd.read_csv(config.comparison_dir / "comparison_predictions.csv", usecols=usecols, parse_dates=["datetime"], chunksize=chunksize):
        chunk = chunk.merge(slots, on=["regime", "building", "weather_mode", "horizon_h"], how="inner")
        if not chunk.empty:
            merged = chunk.merge(base, on=base_key, how="inner", suffixes=("_comparison", "_baseline"))
            if not merged.empty:
                chunks.append(merged)
    if not chunks:
        return pd.DataFrame()
    aligned = pd.concat(chunks, ignore_index=True)
    aligned["y_true_abs_delta"] = (aligned["y_true_comparison"] - aligned["y_true_baseline"]).abs()
    aligned["y_true_matches_comparison"] = aligned["y_true_abs_delta"] <= 1e-6

    rows = []
    keys = ["regime", "building", "model_family", "mode", "weather_mode", "horizon_h", "run_id", "fit_building", "baseline"]
    for key, group in aligned.groupby(keys, dropna=False):
        eval_group = group.loc[group["is_heating_eval"].astype(bool)]
        metrics = mfc.compute_regression_metrics(eval_group["y_true_comparison"].to_numpy(float), eval_group["y_pred"].to_numpy(float))
        coverage = np.nan
        if len(eval_group):
            coverage = ((eval_group["y_true_comparison"] >= eval_group["y_pred_lower"]) & (eval_group["y_true_comparison"] <= eval_group["y_pred_upper"])).mean()
        row = dict(zip(keys, key, strict=True))
        row["comparison_model_family"] = row.pop("model_family")
        row["comparison_mode"] = row.pop("mode")
        row.update(
            n_aligned_rows=len(group),
            n_test_rows=len(eval_group),
            max_y_true_abs_delta=float(group["y_true_abs_delta"].max()),
            y_true_matches_comparison=bool(group["y_true_matches_comparison"].all()),
            interval_coverage=coverage,
            **metrics,
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(p["metrics_aligned"], index=False)
    return out


def write_summary(config: Config, aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = ensure_results(config)
    if aligned.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = (
        aligned.groupby(["regime", "baseline", "weather_mode", "horizon_h"], dropna=False)
        .agg(mean_wape_pct=("wape_pct", "mean"), mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"), mean_r2=("r2", "mean"), total_test_rows=("n_test_rows", "sum"))
        .reset_index()
    )
    actual = pd.read_csv(config.comparison_dir / "comparison_metrics.csv").rename(
        columns={
            "model_family": "comparison_model_family",
            "mode": "comparison_mode",
            "rmse": "actual_rmse",
            "mae": "actual_mae",
            "r2": "actual_r2",
            "wape_pct": "actual_wape_pct",
            "n_test_rows": "actual_n_test_rows",
        }
    )
    base = aligned.rename(columns={"rmse": "baseline_rmse", "mae": "baseline_mae", "r2": "baseline_r2", "wape_pct": "baseline_wape_pct", "n_test_rows": "baseline_n_test_rows"})
    join = ["regime", "building", "comparison_model_family", "comparison_mode", "weather_mode", "horizon_h"]
    vs = base.merge(actual, on=join, how="left")
    vs["delta_wape_pct_baseline_minus_actual"] = vs["baseline_wape_pct"] - vs["actual_wape_pct"]
    vs["delta_rmse_baseline_minus_actual"] = vs["baseline_rmse"] - vs["actual_rmse"]
    summary.to_csv(p["summary"], index=False)
    vs.to_csv(p["vs_model_family"], index=False)
    return summary, vs


def write_metadata(config: Config, scope: str, env: pd.DataFrame) -> None:
    p = ensure_results(config)
    config_payload = {}
    for key, value in asdict(config).items():
        if isinstance(value, Path):
            config_payload[key] = str(value)
        elif isinstance(value, pd.Timestamp):
            config_payload[key] = value.isoformat()
        else:
            config_payload[key] = value
    payload = {
        "created_at_utc": now_utc(),
        "scope": scope,
        "contract": {
            "buildings": list(BUILDINGS),
            "horizons": list(HORIZONS),
            "weather_modes": list(WEATHER_MODES),
            "regimes": list(REGIMES),
            "target": "target_cum_h{H}",
            "metrics": ["wape_pct", "rmse", "r2", "mae"],
        },
        "config": config_payload,
        "environment": env.to_dict(orient="records"),
    }
    p["metadata"].write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
