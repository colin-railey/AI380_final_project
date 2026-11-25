# Requirements — Next-Year Governance Score Prediction (cross-country + Morocco case study)
 
**Status:** IPR1 baseline. These are proposed classroom thresholds, chosen before any model was evaluated.
**Notation:** the targets are WGI PV and GE governance scores (0–100) at t+1. The final test covers outcome years 2021–2025. MAE means mean absolute error in WGI points, computed **separately for PV and GE**. Unless stated otherwise, confidence intervals are 95% bootstrap intervals that resample whole countries (1,000 resamples).
 
| ID | Type | Requirement | Metric | Acceptance criterion | Rationale | Evidence method |
|---|---|---|---|---|---|---|
| **R1** | Functional | The system shall build a traceable country-year dataset that pairs each country's year-*t* inputs with its year-*t+1* PV and GE scores. | Key duplicates; lag errors; unit, filter, or provenance errors; documented row counts | 0 duplicate (ISO3, input_year) keys. Outcome year = input year + 1 in 100% of rows. Every value has a recorded source and unit. Dropped rows and missing input values are counted by country and year. No values are imputed in the stored dataset. | Correct lags and consistent measures are needed for any valid result. Gaps in coverage must be visible, because they are worst in fragile countries. | **V1:** independent audit of three sample pairs for Morocco plus two other countries; automated checks of keys and lags |
| **R2** | Reliability | The system shall keep later outcome years out of fitting, preprocessing (imputation and scaling), and model selection in every chronological evaluation. | Isolation violations | 0 violations. In validation year *v*, training outcomes end at *v−1*. The imputer and scaler are fitted on training rows only. Outcomes from 2021–2025 are never used in training or tuning. | Using evaluation-period information would invalidate every comparison. | **V2:** inspect saved fold masks. Tamper test: changing final-test labels leaves transformations, settings, and parameters unchanged. |
| **R3** | Functional | The system shall produce separately labeled PV and GE predictions from the MLP, Ridge, and persistence for every final-test country-year with an observed outcome, and report Morocco's results for every test year. | Completeness; finiteness; correct target mapping | 100% of eligible rows have two finite predictions per model (MLP for each of 5 seeds). Values outside 0–100 are flagged, never clipped or dropped. All 5 Morocco test years are reported. | Complete reporting stops averages from hiding missing years, swapped columns, or large errors. | **V3:** shape and column-order checks; metric fixture (expected MAEs 3.5 and 4.0) |
| **R4** | Performance | The system shall show a practically noticeable reduction in final-test MAE compared with both simple baselines, for each outcome. | Mean MAE of the 5 MLP seeds vs. Ridge MAE and persistence MAE, per target, across all test countries | MLP MAE ≤ **0.90 × min(Ridge, persistence)** for PV and for GE. The paired-bootstrap CI and the absolute differences are reported. | A 10% gain is a predeclared target that would justify the extra complexity. It is not an operational tolerance. The baselines receive the same current scores, so any gain reflects what the indicators add. Failure is a reportable result. | **E1:** freeze the configuration, evaluate once, and report per-year errors and the decision |
| **R5** | Robustness | The system shall handle invalid inputs and extrapolation transparently, without hiding difficult cases. | Correct responses to a predefined suite of invalid, boundary, and out-of-training-range inputs | Pass every check. Reject infinite values, GDP ≤ 0, and unemployment or water percentages outside 0–100. Accept 0% and 100% and negative inflation. Missing values are imputed and counted, not rejected. Valid values outside the training minimum or maximum are flagged and kept. | Invalid values must not produce scores that look credible. Legitimate unusual values, such as deflation, must not be discarded to improve results. | **V4:** deterministic edge-case tests and Hypothesis property tests |
| **R6** | Robustness | The system shall degrade gracefully when socioeconomic inputs are missing or unreliable. | Relative increase in final-test MAE per target, mean ± SD over 20 seeds | **30% of indicator values masked at random:** MLP MAE rises by **≤ 15%**. **±10% relative noise on inputs:** MLP MAE rises by **≤ 10%**. Ridge's result under the same conditions is reported for comparison. | This keeps the pitch's question about missing and unreliable indicators. 30% masking reflects realistic coverage gaps. The budgets are classroom thresholds set before testing. | **E3:** masking and noise experiments; **V5:** property test of the masking harness |
| **R7** | Reliability | The system shall report and limit how sensitive the neural network is to random initialization. | Sample SD (ddof = 1) of final-test MAE across seeds 0–4, per target; completeness of runs and warnings | SD ≤ **5% of the mean MLP MAE** for each target. All 5 runs and every warning are reported. If the limit is exceeded, mark R7 unmet and never pick the best test seed. | A relative budget fits any error scale. 5% is half of the R4 margin, so the R4 conclusion cannot depend on the choice of seed. | **E2:** five final fits; table of individual MAEs, mean, SD, min/max, and warnings |
| **R8** | Explanation | The system shall evaluate each input's predictive contribution using preplanned comparisons limited to the development period. | Coverage of the 4 leave-one-input-out variants × 5 folds × 5 seeds; variability of the MAE change | All variants are run with the selected configuration. The MAE change is reported by target, year, and seed, including sign changes. No features are selected using final-test results, and findings are labeled exploratory. | Repeated comparisons can show whether an input helps consistently. Correlated time series cannot establish causal effects. | **E4:** refit the imputer, scaler, and model within each fold; compare against matched full-model predictions |
| **R9** | Reliability | The system shall let another team member reproduce the reported results from the preserved sources and configuration. | Completeness of provenance and configuration; differences on rerun | File hashes, filters, ISO3 crosswalk, library versions, settings, seeds, and commands are all recorded. A rerun in the same environment with the same seeds matches within **1e-6** WGI points. | Reproducibility makes the evidence auditable. | **V7:** a second member reruns from the README and records any mismatch |
 
## Traceability
| Req. | Proposal § | Test / experiment |
|---|---|---|
| R1 | 3.1–3.2, App. B | `tests/test_data.py` (V1) |
| R2 | 3.2 | `tests/test_isolation.py` (V2) |
| R3 | 3.3 | `tests/test_outputs.py` (V3) |
| R4 | 3.3, 6.2 | `experiments/final_eval.py` (E1) |
| R5 | 3.1 | `tests/test_input_validation.py` (V4) |
| R6 | 6.2 | `experiments/robustness.py` (E3), `tests/test_masking.py` (V5) |
| R7 | 5.4 | `experiments/seed_study.py` (E2) |
| R8 | 6.2 | `experiments/ablation.py` (E4) |
| R9 | 6.1 | `README.md` rerun log (V7) |
 
_The file paths are planned and will be created during implementation._
 
## Change log
Any revision records the old criterion, the new criterion, the date, the reason, and **whether final results had been inspected**. Unmet requirements stay listed.
 
| Date | ID | Previous | New | Reason | Final results inspected? |
|---|---|---|---|---|---|
| _[date]_ | — | — | Initial version (IPR1) | — | No |
 
