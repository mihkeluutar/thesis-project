from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as ARIMA
    ARIMA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-specific import failure
    ARIMA = None
    ARIMA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


MODEL_ORDER = [
    "Persistence_1h",
    "Persistence_week",
    "Static_ES",
    "ARX_ES",
    "ARMAX_ES",
]

MODEL_STYLES = {
    "Persistence_1h": ("#8ecae6", "--", 1.3),
    "Persistence_week": ("#219ebc", ":", 1.3),
    "Static_ES": ("#ffb703", "-.", 1.5),
    "ARX_ES": ("#fb8500", "-", 1.8),
    "ARMAX_ES": ("#e63946", "-", 1.8),
}

HOUR = pd.Timedelta(hours=1)


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    peak_mask: pd.Series,
    mape_min_kwh: float,
) -> dict:
    eps = 1e-5
    y_t = y_true.values
    y_p = y_pred.values
    e = y_t - y_p
    n = len(y_t)

    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    wape = float(np.sum(np.abs(e)) / (np.sum(y_t) + eps)) * 100
    smape = float(np.mean(2 * np.abs(e) / (np.abs(y_t) + np.abs(y_p) + eps))) * 100

    valid = y_t > mape_min_kwh
    mape_g = float(np.mean(np.abs(e[valid]) / y_t[valid])) * 100 if valid.any() else np.nan
    r2 = float(r2_score(y_t, y_p)) if (np.var(y_t) > 1e-6 and n >= 10) else np.nan
    resid_ac1 = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if n >= 20 else np.nan
    resid_std = float(np.std(e))

    pm = peak_mask.values.astype(bool)
    if pm.sum() >= 5:
        mae_pk = float(np.mean(np.abs(e[pm])))
        rmse_pk = float(np.sqrt(np.mean(e[pm] ** 2)))
    else:
        mae_pk = np.nan
        rmse_pk = np.nan

    return {
        "n": n,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "WAPE_pct": round(wape, 3),
        "sMAPE_pct": round(smape, 3),
        "MAPE_g_pct": round(mape_g, 3) if not np.isnan(mape_g) else np.nan,
        "R2": round(r2, 4) if not np.isnan(r2) else np.nan,
        "ResidAC1": round(resid_ac1, 4) if not np.isnan(resid_ac1) else np.nan,
        "ResidStd": round(resid_std, 3),
        "MAE_peak": round(mae_pk, 3) if not np.isnan(mae_pk) else np.nan,
        "RMSE_peak": round(rmse_pk, 3) if not np.isnan(rmse_pk) else np.nan,
        "n_peak": int(pm.sum()),
    }


def add_point_horizon_targets(mdf: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = mdf.copy()
    for horizon in horizons:
        target_col = f"target_point_h{horizon}"
        shift_steps = horizon - 1
        out[target_col] = out["heat_kwh"].shift(-shift_steps)
        out[f"{target_col}_prevweek"] = out[target_col].shift(168)
    return out


def parse_point_horizon(target_col: str) -> int:
    return int(target_col.rsplit("h", 1)[-1])


def build_point_horizon_masks(
    mdf: pd.DataFrame,
    horizon: int,
    train_end: str,
    heating_threshold_c: float,
    peak_quantile: float,
    exog_cols: list[str],
) -> dict:
    target_col = f"target_point_h{horizon}"
    prevweek_col = f"{target_col}_prevweek"

    train_mask = mdf.index <= pd.Timestamp(train_end)
    test_mask = mdf.index > pd.Timestamp(train_end)

    train = mdf.loc[train_mask].copy()
    test = mdf.loc[test_mask].copy()

    train_hs = train["wx_outdoor_temp_c"] < heating_threshold_c
    test_hs = test["wx_outdoor_temp_c"] < heating_threshold_c

    train_target_hs = train.loc[train_hs, target_col].dropna()
    peak_threshold = float(train_target_hs.quantile(peak_quantile)) if len(train_target_hs) else np.nan
    test_peak = test[target_col] >= peak_threshold if not np.isnan(peak_threshold) else pd.Series(False, index=test.index)

    eval_bool = (
        test_hs
        & test[target_col].notna()
        & test["heat_lag1"].notna()
        & test["heat_lag2"].notna()
        & test[prevweek_col].notna()
        & test[exog_cols].notna().all(axis=1)
    )
    eval_index = test.index[eval_bool]

    return {
        "train": train,
        "test": test,
        "train_hs": train_hs,
        "test_hs": test_hs,
        "peak_threshold": peak_threshold,
        "test_peak": test_peak,
        "eval_index": eval_index,
        "n_test_hs": int(test_hs.sum()),
        "n_eval": len(eval_index),
        "n_peak_eval": int((test_peak & eval_bool).sum()),
        "target_col": target_col,
        "prevweek_col": prevweek_col,
    }


def fit_persistence_last(test_df: pd.DataFrame, **_) -> tuple[pd.Series, str]:
    return test_df["heat_lag1"], "ok"


def fit_persistence_week(test_df: pd.DataFrame, prevweek_col: str, **_) -> tuple[pd.Series, str]:
    return test_df[prevweek_col], "ok"


def fit_static_es(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    exog_cols: list[str],
    min_train_hours: int,
    **_,
) -> tuple[pd.Series, str]:
    tr = train_df.dropna(subset=exog_cols + [target_col])
    if len(tr) < min_train_hours:
        return pd.Series(np.nan, index=test_df.index), "insufficient_train"

    model = LinearRegression()
    model.fit(tr[exog_cols], tr[target_col])

    te = test_df[exog_cols].ffill().bfill()
    if te.isna().any().any():
        te = te.fillna(tr[exog_cols].median())

    y_pred = pd.Series(model.predict(te), index=test_df.index)
    return y_pred, "ok"


def fit_arx_es(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    exog_cols: list[str],
    min_train_hours: int,
    **_,
) -> tuple[pd.Series, str]:
    """Recursive h-step ARX forecast from a one-step model.

    This uses only information available at the issue timestamp:
    observed past demand, recursively predicted intermediate demand,
    and the exogenous weather path on the forecast horizon.
    """

    horizon = parse_point_horizon(target_col)
    arx_cols = exog_cols + ["heat_lag1", "heat_lag2"]
    tr = train_df.dropna(subset=arx_cols + ["heat_kwh"])
    if len(tr) < min_train_hours:
        return pd.Series(np.nan, index=test_df.index), "insufficient_train"

    model = LinearRegression()
    model.fit(tr[arx_cols], tr["heat_kwh"])

    full_df = pd.concat([train_df, test_df]).sort_index()
    y_pred = pd.Series(np.nan, index=test_df.index, dtype=float)

    for issue_time in test_df.index:
        recursive_heat: dict[pd.Timestamp, float] = {}
        valid_forecast = True

        for step in range(horizon):
            forecast_ts = pd.Timestamp(issue_time + step * HOUR)
            if forecast_ts not in full_df.index:
                valid_forecast = False
                break

            lag1_ts = forecast_ts - HOUR
            lag2_ts = forecast_ts - 2 * HOUR

            lag1 = recursive_heat.get(lag1_ts, full_df["heat_kwh"].get(lag1_ts, np.nan))
            lag2 = recursive_heat.get(lag2_ts, full_df["heat_kwh"].get(lag2_ts, np.nan))
            exog_vals = full_df.loc[forecast_ts, exog_cols]

            if pd.isna(lag1) or pd.isna(lag2) or exog_vals.isna().any():
                valid_forecast = False
                break

            x_row = pd.DataFrame(
                [[*exog_vals.tolist(), lag1, lag2]],
                columns=arx_cols,
                index=[forecast_ts],
            )
            recursive_heat[forecast_ts] = float(model.predict(x_row)[0])

        if valid_forecast:
            final_ts = pd.Timestamp(issue_time + (horizon - 1) * HOUR)
            y_pred.at[issue_time] = recursive_heat.get(final_ts, np.nan)

    return y_pred, "ok"


def fit_armax_es(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    exog_cols: list[str],
    min_train_hours: int,
    armax_order: tuple[int, int, int],
    **_,
) -> tuple[pd.Series, str]:
    """Recursive h-step SARIMAX forecast with rolling observed-history updates.

    Crucially, this does not condition on the future target values in the test set.
    For each issue timestamp t, the model state is updated only through t-1, then
    a forecast is produced for the requested horizon using the exogenous weather path.
    """

    if ARIMA is None:
        short_reason = (ARIMA_IMPORT_ERROR or "ARMAX backend unavailable")[:120]
        return pd.Series(np.nan, index=test_df.index), f"armax_unavailable: {short_reason}"

    horizon = parse_point_horizon(target_col)
    tr = train_df.copy().sort_index().asfreq("h")
    if tr["heat_kwh"].notna().sum() < min_train_hours:
        return pd.Series(np.nan, index=test_df.index), "insufficient_train"

    try:
        tr_exog = tr[exog_cols].ffill().bfill()
        if tr_exog.isna().any().any():
            tr_exog = tr_exog.fillna(tr_exog.median())

        armax_res = ARIMA(
            endog=tr["heat_kwh"].values,
            exog=tr_exog.values,
            order=armax_order,
        ).fit(disp=False)

        full_df = pd.concat([train_df, test_df]).sort_index().asfreq("h")
        full_exog = full_df[exog_cols].ffill().bfill()
        if full_exog.isna().any().any():
            full_exog = full_exog.fillna(tr_exog.median())
        y_pred = pd.Series(np.nan, index=test_df.index, dtype=float)
        rolling_res = armax_res

        for issue_time in test_df.index:
            future_idx = [pd.Timestamp(issue_time + step * HOUR) for step in range(horizon)]
            if not all(ts in full_df.index for ts in future_idx):
                continue

            exog_future = full_exog.loc[future_idx].copy()
            if exog_future.isna().any().any():
                continue

            fc = rolling_res.get_forecast(steps=horizon, exog=exog_future.values).predicted_mean
            y_pred.at[issue_time] = float(np.asarray(fc)[-1])

            obs_val = full_df.at[issue_time, "heat_kwh"] if issue_time in full_df.index else np.nan
            if pd.notna(obs_val):
                obs_now = np.array([obs_val], dtype=float)
                exog_now = full_exog.loc[[issue_time], exog_cols].values
                rolling_res = rolling_res.append(endog=obs_now, exog=exog_now, refit=False)

        return y_pred, "ok"
    except Exception as exc:  # pragma: no cover - model fit/runtime specific
        return pd.Series(np.nan, index=test_df.index), f"armax_failed: {str(exc)[:80]}"


POINT_MODELS = {
    "Persistence_1h": fit_persistence_last,
    "Persistence_week": fit_persistence_week,
    "Static_ES": fit_static_es,
    "ARX_ES": fit_arx_es,
    "ARMAX_ES": fit_armax_es,
}


def build_portfolio_summaries(
    metrics_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    horizons: list[int],
    core_min_eval_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    ok_df = metrics_df[metrics_df["Status"] == "ok"].copy()
    eligible_by_horizon = (
        mask_df[mask_df["n_eval"] >= core_min_eval_rows]
        .groupby("Horizon_h")["Building"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )

    common_core_buildings = sorted(
        set.intersection(*(set(eligible_by_horizon.get(h, [])) for h in horizons))
    ) if horizons else []

    scope_map = {
        "extended_all": sorted(mask_df["Building"].unique().tolist()),
        "common_core": common_core_buildings,
    }

    summary_rows = []
    for scope_name, scope_buildings in scope_map.items():
        if not scope_buildings:
            continue
        scoped = ok_df[ok_df["Building"].isin(scope_buildings)].copy()
        for (horizon, model), sub in scoped.groupby(["Horizon_h", "Model"], observed=True):
            summary_rows.append({
                "Scope": scope_name,
                "Horizon_h": int(horizon),
                "Model": model,
                "n_buildings": int(sub["Building"].nunique()),
                "MAE_mean": round(float(sub["MAE"].mean()), 3),
                "RMSE_mean": round(float(sub["RMSE"].mean()), 3),
                "WAPE_pct_mean": round(float(sub["WAPE_pct"].mean()), 3),
                "R2_mean": round(float(sub["R2"].mean()), 4),
                "MAE_peak_mean": round(float(sub["MAE_peak"].mean()), 3),
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(["Scope", "Horizon_h", "Model"]).reset_index(drop=True)

    delta_rows = []
    for (scope_name, model), sub in summary_df.groupby(["Scope", "Model"], observed=True):
        sub = sub.sort_values("Horizon_h").reset_index(drop=True)
        base = sub[sub["Horizon_h"] == 1]
        if len(base):
            base = base.iloc[0]
            for _, row in sub.iterrows():
                delta_rows.append({
                    "Scope": scope_name,
                    "Model": model,
                    "Horizon_h": int(row["Horizon_h"]),
                    "prev_horizon_h": np.nan,
                    "delta_MAE_vs_1h": round(float(row["MAE_mean"] - base["MAE_mean"]), 3),
                    "delta_RMSE_vs_1h": round(float(row["RMSE_mean"] - base["RMSE_mean"]), 3),
                    "delta_WAPE_pct_vs_1h": round(float(row["WAPE_pct_mean"] - base["WAPE_pct_mean"]), 3),
                })
        for idx in range(1, len(sub)):
            prev = sub.iloc[idx - 1]
            row = sub.iloc[idx]
            delta_rows.append({
                "Scope": scope_name,
                "Model": model,
                "Horizon_h": int(row["Horizon_h"]),
                "prev_horizon_h": int(prev["Horizon_h"]),
                "delta_MAE_vs_1h": np.nan,
                "delta_RMSE_vs_1h": np.nan,
                "delta_WAPE_pct_vs_1h": np.nan,
                "delta_MAE_vs_prev_h": round(float(row["MAE_mean"] - prev["MAE_mean"]), 3),
                "delta_RMSE_vs_prev_h": round(float(row["RMSE_mean"] - prev["RMSE_mean"]), 3),
                "delta_WAPE_pct_vs_prev_h": round(float(row["WAPE_pct_mean"] - prev["WAPE_pct_mean"]), 3),
            })

    delta_df = pd.DataFrame(delta_rows)
    if len(delta_df):
        delta_df = delta_df.sort_values(["Scope", "Model", "Horizon_h", "prev_horizon_h"], na_position="first").reset_index(drop=True)
    return summary_df, delta_df, common_core_buildings


def plot_portfolio_error_curves(summary_df: pd.DataFrame, common_core_buildings: list[str], results_dir: Path) -> Path:
    plot_df = summary_df[summary_df["Scope"] == "common_core"].copy()
    if plot_df.empty:
        plot_df = summary_df[summary_df["Scope"] == "extended_all"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=True)
    metrics = [("WAPE_pct_mean", "Mean WAPE (%)"), ("RMSE_mean", "Mean RMSE (kWh)")]

    for ax, (metric_col, ylabel) in zip(axes, metrics):
        for model in MODEL_ORDER:
            sub = plot_df[plot_df["Model"] == model].sort_values("Horizon_h")
            if sub.empty:
                continue
            color, _, _ = MODEL_STYLES[model]
            ax.plot(
                sub["Horizon_h"],
                sub[metric_col],
                marker="o",
                markersize=4,
                lw=1.8,
                color=color,
                label=model,
            )
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(plot_df["Horizon_h"].unique().tolist()))
        ax.grid(alpha=0.2)

    scope_label = (
        f"common core ({len(common_core_buildings)} buildings)"
        if common_core_buildings
        else "all valid buildings"
    )
    axes[0].set_title(f"Portfolio error vs forecast horizon\nScope: {scope_label}")
    axes[1].set_title("Error growth profile by model family")
    axes[1].legend(fontsize=7, loc="best")
    plt.tight_layout()

    out_path = results_dir / "baseline_point_horizon_portfolio_curves.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    return out_path


def build_u05_trace_plot(
    horizon_frames: dict[str, pd.DataFrame],
    mask_store: dict[tuple[str, int], dict],
    pred_store: dict[tuple[str, int], dict[str, pd.Series]],
    horizons: list[int],
    results_dir: Path,
    building: str = "U05",
) -> tuple[pd.DataFrame, Path]:
    trace_rows = []
    fig, axes = plt.subplots(len(horizons), 2, figsize=(18, 3.6 * len(horizons)), sharex=False)
    if len(horizons) == 1:
        axes = np.array([axes])

    ref_horizon = horizons[0]
    ref_target_col = f"target_point_h{ref_horizon}"
    ref_mdf = horizon_frames[building]
    ref_masks = mask_store[(building, ref_horizon)]
    ref_eval = ref_mdf.loc[ref_masks["eval_index"], ref_target_col].dropna()
    shared_best_start = None
    shared_best_var = -1.0

    if len(ref_eval) >= 24:
        scan_starts = ref_eval.index[:-167] if len(ref_eval) > 167 else [ref_eval.index[0]]
        for ts in scan_starts:
            week = ref_eval.loc[ts: ts + pd.Timedelta(hours=167)]
            if len(week) >= min(72, len(ref_eval)) and float(week.var()) > shared_best_var:
                shared_best_var = float(week.var())
                shared_best_start = ts

    if shared_best_start is None and len(ref_eval):
        shared_best_start = ref_eval.index[0]
        shared_best_var = float(ref_eval.var())

    shared_best_end = shared_best_start + pd.Timedelta(hours=167) if shared_best_start is not None else None

    for row_idx, horizon in enumerate(horizons):
        target_col = f"target_point_h{horizon}"
        mdf = horizon_frames[building]
        masks = mask_store[(building, horizon)]
        preds = pred_store.get((building, horizon), {})
        ev_idx = masks["eval_index"]

        ax_pred = axes[row_idx, 0]
        ax_err = axes[row_idx, 1]

        if len(ev_idx) == 0 or not preds:
            ax_pred.set_title(f"{building} — h={horizon}: no eval data")
            ax_err.set_title("No residual view available")
            continue

        if shared_best_start is None:
            y_eval = mdf.loc[ev_idx, target_col].dropna()
            best_start = y_eval.index[0]
            best_end = best_start + pd.Timedelta(hours=167)
            best_var = float(y_eval.var()) if len(y_eval) else np.nan
        else:
            best_start = shared_best_start
            best_end = shared_best_end
            best_var = shared_best_var

        week_idx = mdf.loc[best_start:best_end].index
        actual = mdf.loc[week_idx, target_col]

        ax_pred.plot(actual.index, actual.values, color="k", lw=2.0, label="Actual", zorder=10)
        for model in MODEL_ORDER:
            if model not in preds:
                continue
            color, ls, lw = MODEL_STYLES[model]
            y_pred = preds[model].reindex(week_idx)
            ax_pred.plot(week_idx, y_pred.values, color=color, ls=ls, lw=lw, alpha=0.85, label=model)

            abs_err = np.abs(actual.values - y_pred.values)
            ax_err.plot(week_idx, abs_err, color=color, ls=ls, lw=lw, alpha=0.85, label=model)

            for ts, actual_v, pred_v, abs_err_v in zip(week_idx, actual.values, y_pred.values, abs_err):
                trace_rows.append({
                    "Building": building,
                    "Horizon_h": horizon,
                    "datetime": pd.Timestamp(ts),
                    "Model": model,
                    "actual_kwh": actual_v,
                    "pred_kwh": pred_v,
                    "abs_error_kwh": abs_err_v,
                    "week_start": best_start,
                    "week_end": best_end,
                    "week_variance_kwh2": round(best_var, 3) if not np.isnan(best_var) else np.nan,
                })

        ax_pred.set_ylabel(f"h={horizon}\nTarget (kWh)")
        ax_pred.set_title(
            f"{building} — point horizon {horizon}h\n"
            f"{best_start.strftime('%d %b %Y')} to {best_end.strftime('%d %b %Y')}"
        )
        ax_err.set_ylabel("|Residual| (kWh)")
        ax_err.set_title(f"{building} — point horizon {horizon}h residuals")
        ax_pred.legend(fontsize=6, ncol=3, loc="upper left")
        ax_err.legend(fontsize=6, ncol=3, loc="upper left")

        for ax in (ax_pred, ax_err):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
            for tick in ax.get_xticklabels():
                tick.set_rotation(25)
                tick.set_ha("right")

    axes[-1, 0].set_xlabel("Issue timestamp")
    axes[-1, 1].set_xlabel("Issue timestamp")
    plt.tight_layout()

    out_path = results_dir / "baseline_point_horizon_U05_traces.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    trace_df = pd.DataFrame(trace_rows)
    return trace_df, out_path


def run_multi_horizon_point_baselines(
    building_dfs: dict[str, pd.DataFrame],
    campus_buildings: list[str],
    results_dir: Path,
    exog_cols: list[str],
    train_end: str,
    heating_threshold_c: float,
    peak_quantile: float,
    min_train_hours: int,
    mape_min_kwh: float,
    core_min_eval_rows: int,
    armax_order: tuple[int, int, int],
    horizons: list[int],
    trace_building: str = "U05",
) -> dict:
    horizon_frames = {bldg: add_point_horizon_targets(mdf, horizons) for bldg, mdf in building_dfs.items()}

    all_metrics = []
    mask_rows = []
    pred_store: dict[tuple[str, int], dict[str, pd.Series]] = {}
    skip_log: dict[tuple[str, int], dict[str, str]] = {}

    for horizon in horizons:
        print(f"\nPoint baselines — horizon {horizon}h")
        print("-" * 72)
        for bldg in campus_buildings:
            if bldg not in horizon_frames:
                continue

            mdf = horizon_frames[bldg]
            masks = build_point_horizon_masks(
                mdf=mdf,
                horizon=horizon,
                train_end=train_end,
                heating_threshold_c=heating_threshold_c,
                peak_quantile=peak_quantile,
                exog_cols=exog_cols,
            )
            mask_store_key = (bldg, horizon)
            target_col = masks["target_col"]
            prevweek_col = masks["prevweek_col"]

            mask_rows.append({
                "Building": bldg,
                "Horizon_h": horizon,
                "target_col": target_col,
                "n_test_hs": masks["n_test_hs"],
                "n_eval": masks["n_eval"],
                "n_peak_eval": masks["n_peak_eval"],
                "peak_threshold": round(float(masks["peak_threshold"]), 3) if not np.isnan(masks["peak_threshold"]) else np.nan,
            })

            ev_idx = masks["eval_index"]
            if len(ev_idx) == 0:
                print(f"  {bldg}: SKIPPED (empty eval_index)")
                skip_log[mask_store_key] = {model: "empty_eval_index" for model in POINT_MODELS}
                pred_store[mask_store_key] = {}
                continue

            train = masks["train"]
            test = masks["test"]
            pred_store[mask_store_key] = {}
            bldg_skip = {}

            for model_name, fit_fn in POINT_MODELS.items():
                y_pred, status = fit_fn(
                    train_df=train,
                    test_df=test,
                    target_col=target_col,
                    prevweek_col=prevweek_col,
                    exog_cols=exog_cols,
                    min_train_hours=min_train_hours,
                    armax_order=armax_order,
                )
                bldg_skip[model_name] = status

                y_true_ev = mdf.loc[ev_idx, target_col]
                y_pred_ev = y_pred.reindex(ev_idx)
                peak_ev = masks["test_peak"].reindex(ev_idx).fillna(False)

                if status != "ok":
                    metrics = {k: np.nan for k in [
                        "n", "MAE", "RMSE", "WAPE_pct", "sMAPE_pct", "MAPE_g_pct",
                        "R2", "ResidAC1", "ResidStd", "MAE_peak", "RMSE_peak", "n_peak",
                    ]}
                else:
                    valid = y_true_ev.notna() & y_pred_ev.notna()
                    if valid.sum() < 10:
                        metrics = {k: np.nan for k in [
                            "n", "MAE", "RMSE", "WAPE_pct", "sMAPE_pct", "MAPE_g_pct",
                            "R2", "ResidAC1", "ResidStd", "MAE_peak", "RMSE_peak", "n_peak",
                        ]}
                        bldg_skip[model_name] = "insufficient_eval"
                    else:
                        metrics = compute_metrics(
                            y_true=y_true_ev[valid],
                            y_pred=y_pred_ev[valid],
                            peak_mask=peak_ev[valid],
                            mape_min_kwh=mape_min_kwh,
                        )

                row = {
                    "Building": bldg,
                    "Horizon_h": horizon,
                    "target_col": target_col,
                    "Model": model_name,
                    "Status": bldg_skip[model_name],
                }
                row.update(metrics)
                all_metrics.append(row)
                pred_store[mask_store_key][model_name] = y_pred

            skip_log[mask_store_key] = bldg_skip
            ok_models = sum(1 for status in bldg_skip.values() if status == "ok")
            print(f"  {bldg}: {ok_models}/{len(POINT_MODELS)} models OK")

    metrics_df = pd.DataFrame(all_metrics)
    mask_df = pd.DataFrame(mask_rows).sort_values(["Horizon_h", "Building"]).reset_index(drop=True)
    summary_df, delta_df, common_core_buildings = build_portfolio_summaries(
        metrics_df=metrics_df,
        mask_df=mask_df,
        horizons=horizons,
        core_min_eval_rows=core_min_eval_rows,
    )

    trace_df, trace_plot_path = build_u05_trace_plot(
        horizon_frames=horizon_frames,
        mask_store={
            (bldg, horizon): build_point_horizon_masks(
                mdf=horizon_frames[bldg],
                horizon=horizon,
                train_end=train_end,
                heating_threshold_c=heating_threshold_c,
                peak_quantile=peak_quantile,
                exog_cols=exog_cols,
            )
            for bldg in campus_buildings if bldg in horizon_frames
            for horizon in horizons
        },
        pred_store=pred_store,
        horizons=horizons,
        results_dir=results_dir,
        building=trace_building,
    )
    curve_plot_path = plot_portfolio_error_curves(summary_df, common_core_buildings, results_dir)

    metrics_path = results_dir / "baseline_point_horizon_metrics.csv"
    mask_path = results_dir / "baseline_point_horizon_eval_masks.csv"
    summary_path = results_dir / "baseline_point_horizon_summary.csv"
    delta_path = results_dir / "baseline_point_horizon_delta_vs_1h.csv"
    trace_path = results_dir / f"baseline_point_horizon_{trace_building}_trace_weeks.csv"

    metrics_df.to_csv(metrics_path, index=False)
    mask_df.to_csv(mask_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    trace_df.to_csv(trace_path, index=False)

    return {
        "metrics_df": metrics_df,
        "mask_df": mask_df,
        "summary_df": summary_df,
        "delta_df": delta_df,
        "trace_df": trace_df,
        "common_core_buildings": common_core_buildings,
        "paths": {
            "metrics": metrics_path,
            "masks": mask_path,
            "summary": summary_path,
            "delta": delta_path,
            "trace": trace_path,
            "trace_plot": trace_plot_path,
            "curve_plot": curve_plot_path,
        },
    }
