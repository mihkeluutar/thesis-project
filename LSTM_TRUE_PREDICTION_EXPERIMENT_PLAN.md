# True Prediction Experiment Plan — LSTM 1h + Multi-Horizon

## Summary
- Objective: run a true out-of-sample forecasting benchmark for hourly heat demand.
- Target: canonical comfort-heating source column (`heat_target_mwh`) exported by `01_thesis_data_pipeline_09032026.ipynb`, reported here as `heat_kwh`.
- Scope: all eligible campus buildings, with deep dives for U06 and U05.
- Models: LSTM + existing baselines (Persistence, Static_ES, ARX_ES, ARMAX_ES for 1h).

## Forecast Setup
- Horizons: 1, 3, 6, 12, 24 hours ahead.
- Multi-step strategy: direct per horizon (separate target per horizon).
- Lookback window: 24 hours.
- Weather policy: oracle (actual future weather for horizon timestamp).

## Backtest Protocol
- Period: 2024.
- Fold design: monthly rolling-origin.
- Train window for each fold: all data up to fold start (expanding window).
- Test window for each fold: the corresponding month in 2024.
- Strict split rule: train timestamps <= fold_train_end, test timestamps > fold_train_end.

## Evaluation and Reporting
- Primary window: weather-defined heating rows (`T_out < 15.0°C`) on the matched valid evaluation index.
- Secondary window: all-hours.
- Lead metric: WAPE.
- Absolute-error companion: RMSE (kWh).
- Secondary metrics: R², MAE.
- Peak metrics: RMSE_peak, MAE_peak using threshold from training quantile (0.90).
- Portfolio reporting:
  - Core: buildings with >= 1000 eval rows per horizon.
  - Extended: all evaluated buildings, clearly labeled non-headline.

## Required Outputs
- `results/true_prediction/metrics_long.csv`
  - Columns: building, horizon_h, fold_id, model, window, n, rmse, r2, mae, wape, rmse_peak, mae_peak, status.
- `results/true_prediction/predictions_{building}.csv`
  - Columns: timestamp_origin, timestamp_target, horizon_h, actual_kwh, pred_{model}, fold_id, is_heating, is_peak.
- `results/true_prediction/summary.md`
  - Core vs extended portfolio tables, model ranking by WAPE (+RMSE companion), and U06/U05 deep-dive figures.

## Validation Checks
- No train/test overlap per fold.
- No feature uses target from future timestamps.
- Horizon target alignment is exact (`y(t+h)`).
- Reproducible LSTM runs with fixed seed.
