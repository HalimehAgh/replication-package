# Replication Package
## Maintainability Associations of ML-Specific Code Smell Lifecycles: An Empirical Study

This replication package contains all data, results, scripts, and documentation
needed to independently verify and extend the findings reported in the paper.


## Data Description

### data/projects/

| File | Description |
|---|---|
| `final_181_projects.csv` | The 181 repositories used in the study. Columns: project name, GitHub URL, domain, size category, stratum, total smell events. |
| `selected_231_projects.csv` | The 231 projects selected after stratified sampling, before metric extraction filtering. |
| `stratum_distribution.csv` | Number of projects per domain × size stratum showing proportional allocation across 21 strata. |

### data/smell_events/

| File | Description |
|---|---|
| `clone_log.csv` | Log of git clone operations. All repositories were cloned on January 26, 2026. |
| `smell_event_summary.csv` | Summary of 10,323 smell events. Columns: smell_type, introduction_count, removal_count, total, percentage. Corresponds to Table 2 in the paper. |

### data/manual_inspection/

| File | Description |
|---|---|
| `coding_scheme.md` | Full codebook: definitions of the 4 confound types and 7 binary change categories, with examples drawn from the inspection sample and inter-rater protocol. |
| `final_coding_commits.csv` | All 88 smell-dimension observations used for analysis (80 commits, 58 projects), after merging and resolving disagreements. |
| `supervisor_final.csv` | Final agreed codings from the supervisor rater after discussion. Columns: project, after_commit, smell_type, change_type, dimension, confound_type, pure_smell_fix, method_extraction, error_handling, type_hints, documentation, refactored_nearby, defensive_checks, explanation, evidence_notes. |
| `student_final.csv` | Raw independent codings from the student rater before discussion. Same schema as supervisor_final.csv. |
| `disagreements_combined_final.csv` | All discordant coding decisions (n=111) with the agreed resolution and a brief justification note. |

### results/rq1/

| File | Description |
|---|---|
| `rq1_all_results.csv` | Full results for all 12 RQ1 mixed-effects models. Columns: dimension, transition_type, effect_size_beta, ci_lower, ci_upper, r_squared, f_statistic, p_value, p_value_adjusted, effect_direction. Corresponds to Table 4 in the paper. |
| `rq1_summary.csv` | Human-readable summary of RQ1 findings including significance counts. |
| `rq1_heterogenity.csv` | Wald test statistics for smell-level heterogeneity across dimensions. |
| `rq1_diagnostics.json` | Model diagnostics: VIF per predictor, Durbin-Watson statistic, Shapiro-Wilk test result for each of the 12 models. |
| `rq1_robustness_analysis.json` | Results of alpha-sensitivity check across thresholds 0.01, 0.05, 0.10. |

### results/rq2/

| File | Description |
|---|---|
| `rq2_stage1_aggregate.csv` | Stage 1 aggregate screening results for all 8 eligible smell types × 2 directions (16 models). |
| `rq2_stage1_qualified.json` | Smell types qualifying for Stage 2 with their Stage 1 p-values. |
| `rq2_stage2_dimension_results.csv` | Full results for all 72 Stage 2 dimension-level models. Corresponds to Table 5 Panel A. |
| `rq2_stage3_symmetry.csv` | Stage 3 symmetry z-test results for CDTNE dimension pairs. Corresponds to Table 5 Panel B. |
| `rq2_summary.json` | Overall RQ2 summary including Stage 1, 2, and 3 counts. |
| `rq2_skipped_tests.csv` | Models excluded from Stage 2 with reasons (MAP and MNF insufficient observations). |
| `rq2_robustness_analysis.json` | PCA-based aggregation sensitivity: all 4 CDTNE Stage 2 cells refit with first principal component scores. |
| `rq2_assumption_warnings.json` | VIF and diagnostic warnings per model, including the GNC introduction VIF=15.0 finding and orthogonalization refit results. |

### slr/

| File | Description |
|---|---|
| `SLR_Documentation.pdf` | Full SLR documentation: research question, database-specific search strings, PRISMA flow (530→450→85→26 retained), inclusion/exclusion criteria (IC1-IC6, EC1-EC6), metric catalog, mapping of 17 study metrics to SLR evidence, smell-metric relationships, and limitations. |
| `selected_studies.csv` | The 26 studies retained after full-text screening. Columns: title, authors, year, venue, category (empirical/review/framework/technical_debt). |

### src/

The `src/` folder contains the full analysis pipeline organized into five steps.

| Step | Folder | Description |
|---|---|---|
| 1 | `step1_project_selection/` | Stratified sampling from NICHE dataset, repository cloning |
| 2 | `step2_smell_detection/` | Commit-by-commit smell detection using PyDriller and CodeSmile |
| 3 | `step3_metrics_calculation/` | Extraction of 17 maintainability metrics, normalization, dimension aggregation |
| 4 | `step4_analysis/` | Data preparation, RQ1 mixed-effects models, RQ2 three-stage hierarchical testing |
| 5 | `step5_manual_inspection/` | Cohen's kappa computation for manual inspection inter-rater agreement |

---

## Reproducing Key Results

### Table 2 — Smell event distribution
Load `data/smell_events/smell_event_summary.csv`. All counts are directly readable.

### Table 4 — RQ1 results
Load `results/rq1/rq1_all_results.csv`. The columns `effect_size_beta`, `ci_lower`, `ci_upper`, `r_squared`, `p_value_adjusted` correspond directly to the table columns.

### Table 5 — RQ2 CDTNE results
Load `results/rq2/rq2_stage2_dimension_results.csv` filtered to `smell_type == "columns_and_datatype_not_explicitly_set"` for Panel A. Load `results/rq2/rq2_stage3_symmetry.csv` for Panel B.

### Inter-rater agreement (κ = 0.690)
```bash
cd src/step5_manual_inspection
python calculate_kappa.py \
  --supervisor ../../data/manual_inspection/supervisor_final.csv \
  --student ../../data/manual_inspection/student_final.csv
```

### Running the full analysis pipeline
```bash
# Step 1: Project selection (requires NICHE dataset access)
python src/step1_project_selection/stratified_project_selection.py

# Step 2: Smell detection (requires cloned repositories and CodeSmile)
python src/step2_smell_detection/historical_smell_detection.py

# Step 3: Metric extraction
python src/step3_metrics_calculation/calculate_metrics.py

# Step 4: Statistical analysis
python src/step4_analysis/hypothesis_testing/test_rq1.py
python src/step4_analysis/hypothesis_testing/test_rq2.py
```

---

## Requirements

```
Python >= 3.9
pandas
numpy
scipy
statsmodels
matplotlib
pydriller
```

Install with: `pip install pandas numpy scipy statsmodels matplotlib pydriller`

**CodeSmile** is required for Step 2 (smell detection). See the original CodeSmile repository for installation instructions.

---

## Notes on Reproducibility

- All repositories were cloned on **January 26, 2026**. Re-running Step 1 after this date may yield different results due to repository updates.
- The analysis in Steps 3 and 4 is fully reproducible from the cloned repositories and does not require re-running Step 2.
- All result files in `results/` were generated with a fixed random seed where applicable.

---

## Notes on Anonymization

This package does not contain author-identifying information. Repository URLs for the 181 analyzed projects are included in `data/projects/final_181_projects.csv` since these are publicly available open-source repositories. No proprietary or confidential data is included.

---

## Contact

For questions about this replication package, please open an issue in this repository.
