# Thesis Project

This repository contains the code-side materials for the master's thesis on TalTech campus heat-demand forecasting. The thesis PDF keeps the readable methodology, selected tables, and interpreted results; this project repository keeps the executable notebooks, utility modules, scripts, exported artifacts, and README notes needed to inspect the full workflow.

## Environment

- **Python:** use **3.10 or 3.11** (64-bit). The project was developed primarily on **3.10**; some dependency wheels differ by version and platform.
- **Dependencies:** from the repository root, create a virtual environment, then run:

  ```bash
  python -m pip install -U pip
  python -m pip install -r requirements.txt
  ```

  On Apple Silicon, if `tensorflow` fails to install, check the [TensorFlow install guide](https://www.tensorflow.org/install) for the current supported Python version and pip package name.

## Licensing

- **This repository (code and notebooks):** licensed under the **MIT License** ([LICENSE](LICENSE)). That is a common choice for open thesis companion code: others may reuse the code with attribution, while your thesis document may still be covered separately by TalTech / Digikogu rules (non-exclusive licence, Creative Commons, etc.). If your supervisor or school policy requires a different code licence, replace `LICENSE` accordingly.

## Main Materials

- `00_data_analysis_13032026.ipynb` and `01_thesis_data_pipeline_09032026.ipynb`: source-data inspection, cleaning, and canonical hourly data preparation.
- `02_establishing_baselines_20042026.ipynb`: Bayesian Energy Signature baseline experiments.
- `03_feature_engineering_10032026.ipynb` and `03b_future_weather_lookahead_22032026.ipynb`: model-ready feature exports and future-weather proxy construction.
- `04_lstm_experiments_10032026.ipynb` to `10_lstm_feature_selection_screen_22032026.ipynb`: LSTM architecture and feature-mode experiments.
- `11_xgboost_architecture_lock_22032026.ipynb` and `12_model_family_comparison_23032026.ipynb`: locked XGBoost setup and shared LSTM--XGBoost comparison.
- `13_xai_31032026.ipynb`: grouped PFI and SHAP analysis.
- `notebooks/exploratory/`: early notebooks used to understand the raw datasets (structure, coverage, and quality). They are not part of the main thesis experiment chain; see `data-exploration.ipynb` and `ehr-data-exploration.ipynb`.
- `utils/`: reusable modelling, data-loading, XAI, and reporting helpers.
- `scripts/`: report-ready rendering and batch execution helpers.

## Thesis Appendix Role

The thesis appendices summarize the data and feature contracts in a readable form. Large generated lists, especially the many `feat_fw_*` future-weather columns, are intentionally kept in the project files instead of being printed line by line in the thesis.

The repository is available at:

<https://github.com/mihkeluutar/thesis-project>
