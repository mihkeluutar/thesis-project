# Architecture-First Notebook Plan

## Purpose

This plan resets the next stage of the thesis around three practical constraints:

1. The LSTM setup must be understood before we scale experiments.
2. Cumulative demand is the practically important output.
3. Work should remain notebook-first, using the existing clean data pipeline and feature exports rather than rebuilding the pipeline again.

The goal is to move forward in small, reviewable pieces, with one clear question per notebook and one decision gate before the next phase.

## Starting Point

This plan is meant to bridge from the **current thesis state** to the **next cumulative-forecasting stage**.

The current state is:

- data cleaning and target construction already exist,
- feature tables already exist,
- baseline experiments already exist,
- earlier LSTM notebooks already exist,
- long-horizon LSTM work already exists, but is still too direct-target-focused and not yet architecture-audited in a defensible way.

So this plan is **not** starting from raw data.

It is starting from:

- a usable clean pipeline,
- a usable feature export layer,
- existing notebooks that define the current comparison logic,
- and a need to reorganize the next experiments into a more thesis-defensible sequence.

## End Goal

The target end state is:

- one justified LSTM architecture,
- one clearly defined cumulative forecasting setup,
- one clear direct-vs-recursive cumulative comparison,
- one explainability phase focused on the best cumulative models,
- one optional XGBoost comparison,
- and a thesis narrative that can explain both the logic and the experimental choices without ambiguity.

In other words, this plan should take us from:

- "I have working experiments, but they are broad and partially exploratory"

to:

- "I have a staged, explainable, cumulative-first experiment program that I can defend in the thesis."

## Core Decisions

- The **headline neural output is cumulative heat demand**, not point demand.
- Recursive 1-hour rollouts may still be used **internally** to build cumulative forecasts, but the thesis-facing neural comparison should emphasize cumulative planning horizons.
- The existing clean data and feature pipeline are the source of truth.
- New work should be organized primarily in **Jupyter notebooks with markdown explanations**, plots, and clearly separated experiment sections.
- Reusable helper code is still allowed in `src/` if it reduces copy-paste, but the **main workflow should be runnable and understandable from notebooks**.
- We do **not** try to solve architecture tuning, recursive logic, XAI, XGBoost, and consumption-profile interpretation all at once.

## Stable Assumptions

These assumptions should stay fixed unless we explicitly decide to change them.

- The clean pipeline outputs are the source of truth.
- Forecasting remains chronological.
- Fixed seed is `42`.
- Robustness should come primarily from **time-based validation**, not from many seeds.
- Heating-season evaluation remains important for primary reporting.
- Recursive forecasting uses **oracle future weather** unless explicitly changed.
- Cumulative horizons are the thesis-facing long-horizon outputs.
- `M2/M3/M4` should not be treated as headline winners unless the quality gate shows they are fair to compare.

## Existing Inputs We Intend To Reuse

The plan assumes the next notebooks will directly reuse:

- `01_thesis_data_pipeline_09032026.ipynb`
- `03_feature_engineering_10032026.ipynb`
- `04_lstm_experiments_10032026.ipynb`
- `05_long_horizon_inertia_experiments_16032026.ipynb`
- `data/clean/`
- `data/features/`
- `data/clean/campus_building_features_for_models.csv`

The purpose of the new notebook sequence is **not** to replace these files, but to build on top of them with cleaner scope and clearer markdown narration.

## Existing Outputs Already Produced

The repo already contains useful outputs that should inform the next phase rather than be forgotten:

- baseline summaries and trace plots in `results/`
- long-horizon LSTM summaries in `results/`
- the architecture-first quality audit outputs in `results/architecture_first/`

Those are not the final thesis answer, but they are part of the context and should be cited in markdown when relevant.

## What This Means In Practice

We should stop thinking of the next week as "one large experiment".

Instead, we should think in terms of **five linked notebook phases**:

1. Architecture audit
2. Cumulative target design
3. Recursive cumulative forecasting
4. Feature usefulness and explainability
5. XGBoost and consumption-profile interpretation

Each phase should end with a short written conclusion that answers:

- what was tested,
- what changed,
- what was learned,
- what should happen next.

## Working Principles

### 1. Use the Existing Pipeline

The clean hourly data, canonical targets, and feature tables already exist. The next stage should rely on:

- `data/clean/`
- `data/features/`
- current building proxy tables
- existing thesis-ready split logic where relevant

We should only add lightweight derivations on top of these, such as:

- new cumulative targets,
- recursive rollout views,
- architecture comparison tables,
- grouped feature diagnostics.

That means the next notebooks should begin with a short markdown reminder of:

- where the data come from,
- which prior notebook produced the relevant tables,
- and which columns are being reused rather than rebuilt.

### 2. Notebook-First, Not Script-First

The main experimental interface should be notebooks, because that makes it easier to:

- inspect intermediate tables,
- visualize behavior,
- write markdown explanations next to code,
- check assumptions before running long jobs,
- keep thesis logic visible while experimenting.

That means new work should preferably appear as new notebooks rather than as more notebook-local sprawl inside `05_long_horizon_inertia_experiments_16032026.ipynb`.

### 3. Cumulative-First Evaluation

Point forecasts can still exist as internal machinery, but they are not the main thesis deliverable for the next stage.

The key practical question is:

> How much heat is needed over the next `x` hours?

So the main reported neural targets should be:

- cumulative 10h
- cumulative 12h
- cumulative 16h
- cumulative 20h
- cumulative 24h
- cumulative 36h

If recursive rollout is used, the recursive 1h path is mainly a way to obtain these cumulative totals.

## Fresh-Context Handoff Notes

If we revisit this plan later from a fresh context, the first things to recover should be:

1. the current thesis question,
2. the current clean data sources,
3. the current feature tables,
4. the existing long-horizon notebook logic,
5. the architecture-first quality gate and roadmap.

To make that easy, each new notebook should include a top markdown block called:

`Context and Starting Assumptions`

That block should briefly state:

- what earlier notebook or table it depends on,
- what decisions are already fixed,
- what this notebook is allowed to change,
- and what is outside its scope.

## Notebook Roadmap

## Notebook 06: Architecture Audit

### Proposed file

`06_lstm_architecture_audit_cumulative_20260318.ipynb`

### Main question

Which LSTM architecture is defensible before we scale to larger cumulative experiments?

### Scope

- buildings: `U05`, `U06`, `U03`, `LIB`
- modes: `M0`, `M1`
- fixed seed: `42`
- evaluation: monthly rolling-origin over 2024
- primary scoring: heating-season rows
- key targets for selection:
  - `cum_h24`
  - optionally use `point_h8` and `point_h24` only as internal diagnostics, not as headline outputs

### Candidate architectures

- `A0`: lookback 24, stacked LSTM `64 -> 32`, dropout `0.2`, dense `16`
- `A1`: lookback 48, stacked LSTM `64 -> 32`, dropout `0.2`, dense `16`
- `A2`: lookback 72, stacked LSTM `64 -> 32`, dropout `0.2`, dense `16`
- `A3`: lookback 48, single LSTM `64`, no dropout, dense `16`
- `A4`: lookback 48, single LSTM `128`, no dropout, dense `16`
- `A5`: lookback 48, stacked LSTM `128 -> 64`, dropout `0.2`, dense `32`

### Notebook sections

- motivation and architecture question
- data sources and split rules
- cumulative target definition
- monthly rolling-origin fold construction
- architecture matrix
- result tables
- result plots
- decision note: chosen architecture and why

### Inputs

- clean feature exports for `U05`, `U06`, `U03`, `LIB`
- current cumulative target logic
- current `M0` and `M1` feature blocks
- architecture candidate list

### Outputs

- architecture comparison table
- one saved summary CSV
- one markdown-style conclusion inside the notebook
- one explicit selected architecture

### Decision gate

Do not move to recursive cumulative comparison until one architecture is chosen and justified.

## Notebook 07: Cumulative Target Design

### Proposed file

`07_cumulative_target_design_20260318.ipynb`

### Main question

How exactly should cumulative neural targets be defined and evaluated so the comparison is fair and useful?

### Scope

- define cumulative horizons `10, 12, 16, 20, 24, 36`
- confirm target alignment on the hourly grid
- confirm training/test truncation logic
- compare:
  - direct cumulative target construction
  - recursive cumulative sums from a 1h rollout

### Deliverables

- one clear target-definition table
- one timeline diagram or table showing issue time, internal path, and cumulative target window
- one "no leakage" explanation in markdown

### Inputs

- chosen architecture from Notebook 06
- current long-horizon target logic from existing notebooks
- current chronological split assumptions

### Outputs

- one notebook-local target-definition reference section
- one reusable target convention for later notebooks
- one written statement of forecast-origin semantics

### Decision gate

Before running large experiments, we should be able to answer:

- what timestamp indexes the forecast?
- what hours are being summed?
- how is recursive cumulative output aligned against the direct target?

## Notebook 08: Recursive Cumulative Forecasting

### Proposed file

`08_recursive_cumulative_lstm_20260318.ipynb`

### Main question

Does recursive cumulative forecasting help more than the current direct setup for planning-oriented horizons?

### Scope

- use the chosen architecture from Notebook 06
- use only clean, quality-passed buildings for the main comparison
- start with `M0`
- add `M1` if the architecture audit says it is still worth carrying
- use oracle weather for recursive rollout
- report cumulative horizons only as the main output

### Minimum comparison set

- direct cumulative LSTM
- recursive cumulative LSTM
- persistence-style cumulative reference if useful

### Notebook sections

- reminder of architecture choice
- direct cumulative setup
- recursive cumulative setup
- example path visualizations
- building-level tables
- portfolio summaries
- discussion: where recursive helps, where it drifts

### Inputs

- chosen architecture from Notebook 06
- target semantics from Notebook 07
- quality-passed building cohort

### Outputs

- direct cumulative results table
- recursive cumulative results table
- path examples
- short conclusion on whether recursive is worth keeping

### Decision gate

Before moving to explainability, we should know:

- whether recursive is worth keeping,
- which cumulative horizons are strongest,
- whether one mode is enough or both `M0` and `M1` are needed.

## Notebook 09: Feature Utility And Explainability

### Proposed file

`09_cumulative_feature_utility_xai_20260318.ipynb`

### Main question

Which feature groups actually help cumulative forecasting, and where?

### Scope

- only the best validated cumulative model(s)
- grouped ablation first
- grouped PFI second
- keep feature groups simple:
  - demand memory
  - weather
  - calendar
  - system dynamics
  - static/profile

### Important note

Do not try to explain unstable or obviously inferior models just because they exist.

### Deliverables

- grouped performance-drop table
- grouped PFI plot
- short interpretation paragraph per feature group

### Inputs

- best cumulative model choice from Notebook 08
- existing thesis feature-group framing
- building profile tables where useful

### Outputs

- one grouped ablation result set
- one grouped PFI result set
- one interpretation section that can be reused in the thesis

## Notebook 10: XGBoost And Consumption Profiles

### Proposed file

`10_xgboost_cumulative_and_building_profiles_20260318.ipynb`

### Main question

Can a tabular model match or beat the cumulative LSTM, and what building characteristics explain gains or failures?

### Scope

- same cumulative horizons
- same quality-gated cohort
- use current profile proxies such as:
  - `ac24`
  - `night_day_ratio`
  - `n_vent_points`
  - `vent_class`
  - `energy_class`
  - normalized demand indicators where available

### Deliverables

- LSTM vs XGBoost cumulative comparison
- gain-vs-profile tables or scatter plots
- short conclusion on when building profile matters

### Inputs

- same cumulative targets as Notebook 08
- same quality-gated cohort
- building profile proxies from current clean metadata tables

### Outputs

- comparison tables
- profile-diagnostic plots
- short conclusion on when simple/tabular models are competitive

## Notebook Style Template

Each new notebook should follow roughly the same structure:

1. Title and question
2. Why this notebook exists
3. Data and assumptions
4. Target definition
5. Split logic
6. Model setup
7. Evaluation logic
8. Results
9. Interpretation
10. Decision for the next notebook

Markdown should explain:

- what is being tested,
- why it is being tested,
- what assumptions are being made,
- what would count as a useful result,
- what is intentionally deferred.

Each notebook should also end with a markdown section called:

`What This Means For The Next Notebook`

That section should make handoff easier if work pauses between phases.

## What We Should Not Do Yet

- Do not add many seeds unless two architectures are nearly tied.
- Do not expand to every feature mode before the cumulative logic is stable.
- Do not jump into SHAP before the winning cumulative model is known.
- Do not write thesis claims from exploratory plots alone.
- Do not keep overloading old notebooks with unrelated sections.

## Open Questions That Are Still Allowed

The plan is structured, but not everything is fixed yet. These are still valid open questions:

- whether direct cumulative or recursive cumulative will be stronger,
- whether `M1` deserves to stay after the architecture phase,
- whether cumulative 36h remains practically strong enough to keep as a headline result,
- whether XGBoost becomes a real competitor or just a reference baseline,
- which building-profile proxies best explain gains.

These questions belong to later notebooks; they are not blockers for starting Notebook 06.

## Immediate Next Step

The next concrete task should be:

### Start Notebook 06 first

That notebook should:

- load the existing clean feature tables,
- define cumulative targets cleanly,
- run the architecture subexperiment on the chosen cohort,
- and end with one explicit architecture decision.

Only after that should we create Notebook 07 for the direct-vs-recursive cumulative target logic.

## Relation To The Current Code

Some reusable helper code now exists under `src/forecasting/` and in `scripts/run_architecture_first.py`.

That code can still be useful as backend support for:

- target construction,
- quality checks,
- architecture loops,
- recursive rollouts.

But the main user-facing workflow should shift back into notebooks, where the logic, plots, and markdown discussion stay visible.

## Why This Plan Is Resume-Friendly

This plan should now be usable from a fresh context because it records:

- the current starting state,
- the target end state,
- the fixed assumptions,
- the notebook order,
- the decision gates,
- the expected inputs and outputs for each phase,
- and the intentionally deferred questions.

If we later reopen the project with little context, this file should be enough to answer:

- where we are,
- what comes next,
- why that step is next,
- and what files or data the next notebook should depend on.

## Summary

The right next move is **not** "run everything".

The right next move is:

1. choose the architecture in a controlled notebook,
2. define cumulative targets clearly,
3. compare direct vs recursive cumulative forecasting,
4. then analyze feature usefulness,
5. then add XGBoost and building-profile interpretation.

That keeps the thesis logic understandable and prevents the next stage from turning into another large, hard-to-interpret experiment pile.
