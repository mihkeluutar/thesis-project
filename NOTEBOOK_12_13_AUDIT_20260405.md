# Notebook 12/13 Audit Outcome

Date: 2026-04-05

Scope:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/12_model_family_comparison_23032026.ipynb`
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/13_xai_31032026.ipynb`

This note records the current audit outcome for the model-family and XAI notebooks, the fixes already implemented, the validation status, and the remaining manual notebook steps required before thesis write-up.

## Executive Outcome

The audit materially improved both notebooks.

- Notebook 12 is now reload-first, report-oriented, and uses a defendable modeling taxonomy that separates `M0-M4` from `FW0-FW2`.
- Notebook 13 is now scoped to the current manifest by default, so thesis-facing XAI summaries no longer need to mix legacy rows into the current result set.
- Both notebooks now support notebook-native smoke checks, notebook-native export of report-ready CSVs, and notebook-native validation cells.
- The current thesis-ready base state is strong for:
  - notebook 12 base results
  - notebook 13 XGBoost-only base XAI results
- The main missing work is still computational:
  - notebook 12 `M1` extension is only partially run
  - notebook 12 `M3` supplementary decomposition has not been run
  - notebook 13 `M1` XAI extension has not been run

## Main Audit Findings

### 1. Modeling taxonomy was previously too compressed

The previous presentation logic blurred feature-mode semantics and future-weather semantics.

What is now enforced:
- `M0`: base temporal core only
- `M1`: `M0 +` historical weather memory
- `M2`: `M0 +` system / inertia proxies
- `M3`: `M1 + M2`, still dynamic-only
- `M4`: `M3 +` static building-context branch from `setB`
- `FW0`: no future weather
- `FW1`: oracle future weather built at run time from actual future temperature / RH
- `FW2`: forecast-like proxy future weather from exported `feat_fw_*` columns in the regular feature files

Reporting rule now used:
- weather memory is a mode property
- forecasts are a weather-mode property

This matters because:
- `M4` should not be described as "the weather-memory mode"
- future weather should not be described as part of `M0-M4`
- `M1` is important and should be part of the headline notebook-12 story

### 2. Notebook 12 pooled reporting needed clearer semantics

The raw model-family outputs contain a training/evaluation mismatch in pooled runs:
- pooled training is represented in some places using `building=POOLED`
- metrics and coverage are still evaluated per real building

This did not necessarily make the saved results wrong, but it made report-facing joins and interpretations easier to misread.

What is now fixed:
- report-facing exports distinguish pooled training scope from per-building evaluation
- `per_building` is positioned as the main story
- `pooled_same_buildings` is positioned as supplementary

### 3. Notebook 13 needed strict manifest scoping

The raw XAI result store contained mixed historical rows:
- legacy `M1/M3` XGBoost rows
- partial LSTM rows
- retried run-log entries that could overstate failure if read naively

This created a thesis-reporting risk because summary tables could accidentally reflect a wider or dirtier scope than the notebook intended.

What is now fixed:
- notebook-13 exports can be scoped to the current manifest
- report-facing XAI summaries can be restricted to allowed model families
- a latest-status run log is available for report-facing review

## What Was Implemented

### Notebook-level improvements

Both notebooks were changed so they can be used directly in the notebook environment without requiring the CLI as the primary interface.

Implemented in both notebooks:
- project venv kernelspec wired in
- a cheap Quick Smoke Check near the top
- saved-results-first loading flow
- notebook-native export of report-ready CSVs
- notebook-native validation cells
- separation between:
  - currently loaded result scope
  - manually runnable experiment scope
- support for merging multiple saved result directories inside the notebook

Additional notebook changes:
- notebook 12 no longer defaults into notebook-side smoke training
- notebook 13 notebook cells were corrected to use `rebuild_detail_cache(...)` instead of `ensure_detail_cache(...)`

### Helper and export pipeline improvements

The audit also added reusable reporting/export support in the Python utilities and scripts:
- canonical mode and weather-mode semantics tables
- report-ready model-family export helpers
- manifest-scoped XAI export helpers
- latest-status run-log views
- mode-delta summaries for reporting
- notebook-compatible merged-result export flow

## Current Validated State

### Notebook 12 base state

Authoritative current base source:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/model_family_comparison_29032026`

Current report-ready base export:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/report_ready_20260405/model_family_base`

Validated base artifacts include:
- `mode_semantics.csv`
- `weather_mode_semantics.csv`
- `thesis_per_building_summary.csv`
- `thesis_pooled_summary.csv`
- `mode_delta_summary.csv`
- `metric_recompute_validation.csv`
- `run_log_latest_report.csv`

Validation outcome:
- sampled metric recomputation is available and matches for the checked slices
- pooled semantics are made explicit in report-facing artifacts
- base export is thesis-usable for the current base mode set

Current limitation:
- this base still does not complete the planned headline mode set because `M1` extension coverage is not yet complete

### Notebook 13 base state

Authoritative current base source:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/xai_final_31032026`

Current report-ready base export:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/report_ready_20260405/xai_base`

Validated base artifacts include:
- `xai_metrics.csv`
- `xai_grouped_pfi.csv`
- `xai_grouped_shap.csv`
- `xai_pfi_shap_agreement.csv`
- `xai_rq1_accuracy_summary.csv`
- `xai_rq2_xai_stability_summary.csv`
- `xai_mode_transition_summary.csv`
- `xai_scope_validation.csv`
- `xai_run_log_latest.csv`
- `xai_manifest_coverage.csv`
- `xai_manifest_missing.csv`

Validation outcome:
- report-facing XAI exports are scoped to the current manifest
- report-facing XAI exports are XGBoost-first
- scope validation shows zero rows outside the current manifest in the cleaned base export

Current limitation:
- the cleaned base is not yet a complete `M0/M1/M2/M4` XAI notebook result set because `M1` coverage is still incomplete in the raw XAI store

## Phase Status

Source:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/report_ready_20260405/implementation_status_20260405.csv`

Status summary:
- Phase 1 taxonomy: complete
- Phase 2 notebook-12 base cleaning/export: complete
- Phase 3 notebook-12 `M1` extension: partial
- Phase 4 notebook-12 `M3` supplementary decomposition: not started
- Phase 5 notebook-13 base scoping/export: complete
- Phase 6 notebook-13 `M1` XAI extension: not started

Current partial-run note:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/model_family_comparison_m1_extension_20260405`
- this extension directory contains partial artifacts only and should be resumed rather than discarded

## Notebook Audit Outcome By Notebook

### Notebook 12 outcome

Current judgement:
- defendable as a base comparison notebook after the audit changes
- not yet complete as the final headline thesis notebook until `M1` is fully added

What is now defendable:
- the mode/weather taxonomy shown in the notebook
- the base `per_building` reporting flow
- the base report-ready CSV export path
- the pooled-training supplementary framing

What is still missing:
- full `M1` completion across the intended notebook-12 grid
- final merged report-ready output using `M0/M1/M2/M4`
- targeted `M3` supplementary run if you want to defend any static-context claim via `M4 - M3`

### Notebook 13 outcome

Current judgement:
- defendable as a scoped XGBoost-base XAI notebook after the audit changes
- not yet complete as the final thesis XAI notebook if `M1` is part of the intended mode story

What is now defendable:
- cleaned XGBoost-only base XAI summaries
- manifest-scoped aggregation
- report-ready CSV export path
- notebook-side validation of scope and export coverage

What is still missing:
- a proper `M1` XAI completion run
- a final merged XAI export containing the approved thesis mode set
- optional supplementary `M3` interpretation only if notebook-12 results show it is necessary

## Manual Notebook Workflow From Here

Important:
- notebooks should be run manually by the user
- notebook execution was intentionally not performed automatically as part of this audit write-up

### Step 1. Audit notebook 12 base in-place

Open:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/12_model_family_comparison_23032026.ipynb`

Run only the reload/export/validation path first:
- imports/config
- Quick Smoke Check
- setup and saved-output load
- report export
- notebook validation

Keep:
- `RUN_FULL_MATRIX_IN_NOTEBOOK = False`

Expected base export location:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/report_ready_20260405/model_family_base`

Review especially:
- `metric_recompute_validation.csv`
- `thesis_per_building_summary.csv`
- `thesis_pooled_summary.csv`
- `mode_delta_summary.csv`

### Step 2. Run notebook-12 `M1` extension manually

In notebook 12 set:
- `RUN_FULL_MATRIX_IN_NOTEBOOK = True`
- `FULL_RUN_RESULTS_DIR = PROJECT_ROOT / "results" / "model_family_comparison_m1_extension_20260405"`
- `RUN_MODES = ("M1",)`
- `RUN_WEATHER_MODES = ("FW0", "FW1", "FW2")`
- `RUN_REGIMES = ("per_building", "pooled_same_buildings")`

Then run:
- sanity/smoke section
- full matrix section
- report export
- notebook validation

Goal:
- complete the missing `M1` coverage without changing the base `M0/M2/M4` source

### Step 3. Merge notebook-12 base + `M1`

In notebook 12 set:
- `RUN_FULL_MATRIX_IN_NOTEBOOK = False`
- `MERGE_SOURCE_RESULTS_DIRS = (RAW_RESULTS_DIR, PROJECT_ROOT / "results" / "model_family_comparison_m1_extension_20260405")`

Then re-run:
- saved-output load
- report export
- notebook validation

Expected merged report location:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/report_ready_20260405/model_family`

Review especially:
- `thesis_per_building_summary.csv`
- `mode_delta_summary.csv`

### Step 4. Run notebook-12 `M3` supplement manually

In notebook 12 set:
- `RUN_FULL_MATRIX_IN_NOTEBOOK = True`
- `FULL_RUN_RESULTS_DIR = PROJECT_ROOT / "results" / "model_family_comparison_m3_supplement_20260405"`
- `RUN_MODES = ("M3",)`
- `RUN_WEATHER_MODES = ("FW0", "FW2")`
- `RUN_REGIMES = ("per_building",)`
- `RUN_HORIZONS = (8, 24, 36)`

Then run:
- sanity/smoke section
- full matrix section
- report export
- notebook validation

Goal:
- support interpretation of:
  - `M3 - M2`
  - `M4 - M3`

### Step 5. Audit notebook 13 base in-place

Open:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/13_xai_31032026.ipynb`

Run only the reload/export/validation path first:
- imports/config
- Quick Smoke Check
- setup and saved-output load
- report export
- notebook validation

Keep:
- `RUN_FULL_MATRIX_IN_NOTEBOOK = False`

Expected base export location:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/report_ready_20260405/xai_base`

Review especially:
- `xai_scope_validation.csv`
- `xai_rq1_accuracy_summary.csv`
- `xai_rq2_xai_stability_summary.csv`
- `xai_mode_transition_summary.csv`

### Step 6. Run notebook-13 `M1` XAI extension manually

Do this only after notebook-12 `M1` is approved.

In notebook 13 set:
- `RUN_FULL_MATRIX_IN_NOTEBOOK = True`
- `FULL_MATRIX_RESULTS_DIR = PROJECT_ROOT / "results" / "xai_m1_extension_20260405"`
- `RUN_MODES = ("M1",)`
- `RUN_MODEL_FAMILIES = ("xgboost",)`
- `RUN_WEATHER_MODES = ("FW0", "FW1", "FW2")`
- `RUN_REGIMES = ("per_building", "pooled_same_buildings")`
- `RUN_SEED_LIST = (42, 52, 62)`

Then run:
- matrix generation section
- export section
- notebook validation section
- downstream aggregation/plot sections

Goal:
- complete XAI coverage for the approved thesis mode set

### Step 7. Merge notebook-13 base + `M1`

In notebook 13 set:
- `RUN_FULL_MATRIX_IN_NOTEBOOK = False`
- `MERGE_SOURCE_RESULTS_DIRS = (RAW_XAI_RESULTS_DIR, PROJECT_ROOT / "results" / "xai_m1_extension_20260405")`

Then re-run:
- saved-output load
- report export
- notebook validation
- summary/plot sections

Expected merged report location:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/report_ready_20260405/xai`

Review especially:
- `xai_rq1_accuracy_summary.csv`
- `xai_rq2_xai_stability_summary.csv`
- `xai_mode_transition_summary.csv`

## What Can Already Be Used In The Thesis

Already usable now:
- notebook-12 base taxonomy and base result framing
- notebook-12 base report-ready CSVs
- notebook-13 scoped XGBoost-base XAI CSVs
- notebook-side validation checks in both notebooks

Not yet safe to finalize as headline claims:
- notebook-12 full mode comparison unless `M1` is completed and merged
- any strong static-context interpretation unless the `M3` supplement is run and reviewed
- notebook-13 full mode-transition XAI story unless the `M1` XAI extension is completed

## Recommended Reporting Position Right Now

Until the remaining runs are completed, the safest thesis wording is:
- notebook 12 currently supports a defendable base comparison with corrected taxonomy and clean export logic
- notebook 13 currently supports a defendable scoped XGBoost-base XAI analysis
- final headline mode-level claims remain provisional until:
  - notebook-12 `M1` is completed and merged
  - notebook-12 `M3` supplement is reviewed if static-context claims are needed
  - notebook-13 `M1` XAI extension is completed and merged

## Related Artifacts

Primary status file:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/report_ready_20260405/implementation_status_20260405.csv`

Notebook-12 base export:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/report_ready_20260405/model_family_base`

Notebook-13 base export:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/report_ready_20260405/xai_base`

Partial notebook-12 `M1` extension:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/model_family_comparison_m1_extension_20260405`

Planned final merged export locations:
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/report_ready_20260405/model_family`
- `/Users/mihkeluutar/Documents/TalTech/Thesis/thesis-project/results/report_ready_20260405/xai`
