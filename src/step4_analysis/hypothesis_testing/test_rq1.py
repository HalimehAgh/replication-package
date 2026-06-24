"""
RQ1 Hypothesis Testing Module (Revised)

Tests the overall impact research question: Do ML-specific code smells
negatively impact software maintainability?

Reference: rq1_revised.md specification

This module tests introduction, removal, and heterogeneity effects:
- H1 (Introduction): A-1 -> A transitions across 6 dimensions
- H2 (Removal): B-1 -> B transitions across 6 dimensions
- H1.3 (Heterogeneity): Wald test for β₁ = β₂ = ... = β₁₂

Performs 12 hypothesis tests + heterogeneity tests:
- H1.1 - H1.6: Introduction impact on 6 maintainability dimensions
- H2.1 - H2.6: Removal impact on 6 maintainability dimensions

Multi-indicator regression model (per spec §3A):
    Δ_Dimension_d = β₀ + β₁·CIDX + β₂·CDTNE + ... + β₁₂·TA +
                    γ₁·Dimension_d_before +
                    γ₂·Total_Churn_before +
                    γ₃·Commit_Count_before +
                    γ₄·Bug_Fix_Ratio_before +
                    δ₁·Lines_Added +
                    δ₂·Lines_Deleted +
                    δ₃·Is_Feature_Commit +
                    δ₄·Is_Bug_Fix +
                    δ₅·Log_LOC + ε

Note: The spec calls for u_project random effects. Since statsmodels
mixed-effects has numpy compatibility issues, we use project-level
clustered standard errors as an approximation.

Note: δ₅·Log_LOC is an additional control beyond the spec (file size).

Note: Historical context variables (total churn, commit count, bug fix ratio)
are included as control variables, not as a tested dimension.

Hypotheses:
Introduction (H1):
- H1.1: Introduction of ML code smells increases structural complexity
- H1.2: Introduction of ML code smells increases code size
- H1.3: Introduction of ML code smells increases defect proneness
- H1.4: Introduction of ML code smells decreases understandability
- H1.5: Introduction of ML code smells decreases testing quality
- H1.6: Introduction of ML code smells increases ML-specific issues

Removal (H2):
- H2.1: Removal of ML code smells decreases structural complexity
- H2.2: Removal of ML code smells decreases code size
- H2.3: Removal of ML code smells decreases defect proneness
- H2.4: Removal of ML code smells increases understandability
- H2.5: Removal of ML code smells increases testing quality
- H2.6: Removal of ML code smells improves ML-specific quality

Heterogeneity (H1.3 in spec):
- Tests H₀: β₁ = β₂ = ... = β₁₂ (all smell effects are equal)

Note: RQ2 (test_rq2.py) performs 3-stage smell-specific analysis with H2.1, H2.2, and H2.3.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.step4_analysis.hypothesis_testing.regression_utils import (
    run_hypothesis_test,
    benjamini_hochberg_correction,
    results_to_dataframe,
    HypothesisTestResult,
    check_regression_assumptions,
    calculate_post_hoc_power
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Maintainability dimensions (6 dimensions; historical context is a control variable)
DIMENSIONS = [
    'structural_complexity',
    'code_size',
    'defect_proneness',
    'understandability',
    'testing_quality',
    'ml_specific',
]

# Expected direction of impact for each dimension
# positive = higher score means worse maintainability
# negative = higher score means better maintainability (inverted)
DIMENSION_DIRECTION = {
    'structural_complexity': 'positive',  # Higher = worse
    'code_size': 'positive',              # Higher = worse
    'defect_proneness': 'positive',       # Higher = worse
    'understandability': 'negative',      # Higher = better (inverted for hypothesis)
    'testing_quality': 'negative',        # Higher = better (inverted for hypothesis)
    'ml_specific': 'positive',            # Higher = worse (ML-specific issues)
}

# Hypothesis statements for introduction (H1)
INTRO_HYPOTHESIS_STATEMENTS = {
    'structural_complexity': 'Introduction of ML code smells increases structural complexity',
    'code_size': 'Introduction of ML code smells increases code size',
    'defect_proneness': 'Introduction of ML code smells increases defect proneness',
    'understandability': 'Introduction of ML code smells decreases understandability',
    'testing_quality': 'Introduction of ML code smells decreases testing quality',
    'ml_specific': 'Introduction of ML code smells increases ML-specific issues',
}

# Hypothesis statements for removal (H2)
REMOVAL_HYPOTHESIS_STATEMENTS = {
    'structural_complexity': 'Removal of ML code smells decreases structural complexity',
    'code_size': 'Removal of ML code smells decreases code size',
    'defect_proneness': 'Removal of ML code smells decreases defect proneness',
    'understandability': 'Removal of ML code smells increases understandability',
    'testing_quality': 'Removal of ML code smells increases testing quality',
    'ml_specific': 'Removal of ML code smells improves ML-specific quality',
}

# 12 ML smell type indicator columns (multi-indicator model for RQ1 revision)
# Each smell type is a separate predictor with its own coefficient
SMELL_INDICATOR_COLS = [
    'smell_CIDX',   # Chain Indexing
    'smell_CDTNE',  # Columns and Datatype Not Explicitly Set
    'smell_DCA',    # Dataframe Conversion API Misused
    'smell_IPA',    # In-Place APIs Misused
    'smell_MMA',    # Matrix Multiplication API Misused
    'smell_MAP',    # Merge API Parameter Not Explicitly Set
    'smell_NAN',    # NaN Equivalence Comparison Misused
    'smell_UI',     # Unnecessary Iteration
    'smell_GNC',    # Gradients Not Cleared Before Backward Propagation
    'smell_MNF',    # Memory Not Freed
    'smell_PC',     # PyTorch Call Method Misused
    'smell_TA',     # Tensor Array Not Used
]

# Control variables for regression (aligned with RQ revision specification §3A)
# Formula: Δ_Dimension_d = β₀ + β₁·CIDX + ... + β₁₂·TA +
#          γ₁·Dimension_d_before + γ₂·Total_Churn_before +
#          γ₃·Commit_Count_before + γ₄·Bug_Fix_Ratio_before +
#          δ₁·Lines_Added + δ₂·Lines_Deleted + δ₃·Is_Feature_Commit + δ₄·Is_Bug_Fix +
#          δ₅·Log_LOC + u_project + ε
#
# Variable mapping:
#   β₁-β₁₂: SmellType indicators - Count of each smell type introduced/removed (treatment variables)
#   γ₁: baseline_score           - Baseline dimension score at A-1 (or B-1 for removal)
#   γ₂: total_churn              - Cumulative code churn before the event
#   γ₃: commit_count             - Number of commits in time window before event
#   γ₄: bug_fix_ratio            - Historical ratio of bug fix commits
#   δ₁: lines_added              - Lines added in the commit
#   δ₂: lines_deleted            - Lines deleted in the commit
#   δ₃: is_feature_commit        - Binary flag for feature commits
#   δ₄: is_bug_fix               - Binary flag for bug fix commits
#   δ₅: log_loc                  - Log of lines of code at baseline (additional control beyond spec)
#
# Note on clustering: Spec calls for u_project random effects. Implementation uses
# project-level clustered standard errors as an approximation:
# - Statsmodels mixed-effects has numpy compatibility issues
# - Clustered SE provides robust inference under heteroskedasticity
# - Project-level clustering matches the spec's u_project grouping
CONTROL_COLS = ['baseline_score', 'total_churn', 'commit_count', 'bug_fix_ratio',
                'lines_added', 'lines_deleted', 'is_feature_commit', 'is_bug_fix', 'log_loc']

# Threshold for adequate sample size (same as RQ2 for consistency)
# Tests with n < ADEQUATE_SAMPLE_SIZE are flagged as exploratory/underpowered
ADEQUATE_SAMPLE_SIZE = 30


def generate_hypothesis_id(dimension: str, transition: str) -> str:
    """
    Generate hypothesis ID.

    H1.1 - H1.6 for introduction
    H2.1 - H2.6 for removal
    """
    dim_idx = DIMENSIONS.index(dimension)
    h_prefix = "H1" if transition == 'introduction' else "H2"
    return f"{h_prefix}.{dim_idx + 1}"


def generate_hypothesis_text(dimension: str, transition: str) -> str:
    """Generate hypothesis statement text."""
    if transition == 'introduction':
        return INTRO_HYPOTHESIS_STATEMENTS.get(
            dimension,
            f"Introduction of ML code smells affects {dimension.replace('_', ' ')}"
        )
    else:
        return REMOVAL_HYPOTHESIS_STATEMENTS.get(
            dimension,
            f"Removal of ML code smells improves {dimension.replace('_', ' ')}"
        )


def run_transition_tests(
    data: pd.DataFrame,
    transition: str,
    alpha: float = 0.05,
    min_samples: int = 10
) -> List[HypothesisTestResult]:
    """
    Run hypothesis tests for a specific transition type using multi-indicator model.

    Uses 12 smell type indicators as treatment variables, each with its own coefficient.
    This allows testing the aggregate effect of all smell types on each dimension.

    Args:
        data: Prepared DataFrame with regression variables (wide format with 12 smell columns)
        transition: 'introduction' or 'removal'
        alpha: Significance level
        min_samples: Minimum sample size for a test

    Returns:
        List of HypothesisTestResult objects
    """
    results = []

    # Filter to specific transition
    transition_data = data[data['change_type'] == transition].copy()
    n_events = len(transition_data)

    h_label = "H1" if transition == 'introduction' else "H2"
    logger.info(f"\n=== {transition.upper()} IMPACT ({h_label}, {n_events} events) ===")
    logger.info(f"Using multi-indicator model with {len(SMELL_INDICATOR_COLS)} smell type columns")

    # Flag if sample size is adequate (>= 30) or exploratory
    sample_adequate = n_events >= ADEQUATE_SAMPLE_SIZE
    if not sample_adequate and n_events >= min_samples:
        logger.info(f"  (Note: n={n_events} < {ADEQUATE_SAMPLE_SIZE}, flagged as exploratory)")

    if n_events < min_samples:
        logger.warning(f"Insufficient samples for {transition}: {n_events} < {min_samples}")
        return results

    # Check which smell indicator columns exist in data
    available_smell_cols = [col for col in SMELL_INDICATOR_COLS if col in transition_data.columns]
    missing_smell_cols = [col for col in SMELL_INDICATOR_COLS if col not in transition_data.columns]

    if missing_smell_cols:
        logger.warning(f"Missing smell columns: {missing_smell_cols}")
        # Fill missing with 0
        for col in missing_smell_cols:
            transition_data[col] = 0
        available_smell_cols = SMELL_INDICATOR_COLS

    # Filter out constant columns (zero variance) to avoid singular matrix
    non_constant_cols = [
        col for col in available_smell_cols
        if transition_data[col].nunique() > 1
    ]
    constant_cols = set(available_smell_cols) - set(non_constant_cols)
    if constant_cols:
        logger.warning(f"Excluding constant smell columns: {constant_cols}")
    available_smell_cols = non_constant_cols

    logger.info(f"  Available smell indicators: {len(available_smell_cols)}")

    for dimension in DIMENSIONS:
        hypothesis_id = generate_hypothesis_id(dimension, transition)
        hypothesis_text = generate_hypothesis_text(dimension, transition)

        logger.info(f"\n{hypothesis_id}: {hypothesis_text}")

        # Prepare variables for this test
        y_col = f'{dimension}_delta'
        baseline_col = f'{dimension}_baseline'

        if y_col not in transition_data.columns:
            logger.warning(f"  Missing column: {y_col}")
            continue

        # Create test-specific control columns
        test_data = transition_data.copy()
        test_data['baseline_score'] = test_data.get(baseline_col, 0)

        # Ensure control variables exist
        for ctrl in CONTROL_COLS:
            if ctrl not in test_data.columns:
                if ctrl == 'baseline_score':
                    test_data[ctrl] = test_data.get(baseline_col, 0)
                elif ctrl == 'log_loc':
                    test_data[ctrl] = np.log1p(test_data.get('loc', 100))
                elif ctrl in ['is_bug_fix', 'is_feature_commit']:
                    test_data[ctrl] = False
                elif ctrl in ['lines_added', 'lines_deleted']:
                    test_data[ctrl] = 0
                else:
                    test_data[ctrl] = 0

        # Add before/after scores for effect size calculation
        test_data['before_score'] = test_data.get(f'{dimension}_before', 0)
        test_data['after_score'] = test_data.get(f'{dimension}_after', 0)

        # For multi-indicator model, we use all smell types as treatment variables
        # The combined treatment columns are the smell indicators
        # We run the regression with all predictors and extract the joint F-test

        # Run hypothesis test with multi-indicator treatment
        # We use the first smell indicator as the "treatment_col" for API compatibility,
        # but pass all smell indicators as additional predictors
        all_predictors = available_smell_cols + CONTROL_COLS

        result = run_hypothesis_test(
            data=test_data,
            y_col=y_col,
            treatment_col=available_smell_cols[0] if available_smell_cols else 'total_smell_types',
            control_cols=all_predictors[1:] if available_smell_cols else CONTROL_COLS,  # All except first treatment
            cluster_col='project',
            hypothesis_id=hypothesis_id,
            dimension=dimension,
            transition_type=transition,
            smell_type=None,
            alpha=alpha,
            min_samples=min_samples,
            multi_treatment_cols=available_smell_cols  # Pass all smell indicators for joint test
        )

        if result:
            # Add sample size adequacy flag to result
            result.sample_size_adequate = sample_adequate
            results.append(result)

            # Log results
            sig_marker = "***" if result.is_significant else ""
            exploratory_marker = " [exploratory]" if not sample_adequate else ""

            # Log F-test for joint significance (multi-indicator model)
            if result.regression.f_statistic is not None:
                logger.info(f"  Joint F-test: F={result.regression.f_statistic:.2f}, "
                          f"p={result.regression.f_p_value:.4f}{sig_marker}")
            else:
                logger.info(f"  p-value: {result.raw_p_value:.4f}{sig_marker}")

            logger.info(f"  Effect size: {result.effect_size_interpretation} "
                      f"(avg|β|={result.effect_size_beta:.4f}){exploratory_marker}")
            logger.info(f"  N={result.regression.n_observations}, "
                      f"clusters={result.regression.n_clusters}, R²={result.regression.r_squared:.4f}")

            # Log individual smell coefficients if available
            if result.regression.multi_coefficients:
                sig_smells = []
                for smell_col, coef in result.regression.multi_coefficients.items():
                    pval = result.regression.multi_p_values.get(smell_col, 1.0)
                    if pval < alpha:
                        sig_smells.append(f"{smell_col}(β={coef:.4f}, p={pval:.4f})")
                if sig_smells:
                    logger.info(f"  Significant smell types: {', '.join(sig_smells)}")
        else:
            logger.warning(f"  Test failed or insufficient data")

    return results


def run_rq1_tests(
    data: pd.DataFrame,
    alpha: float = 0.05,
    min_samples: int = 10
) -> Tuple[List[HypothesisTestResult], List[HypothesisTestResult]]:
    """
    Run all RQ1 hypothesis tests (both introduction and removal).

    Args:
        data: Prepared DataFrame with regression variables
        alpha: Significance level
        min_samples: Minimum sample size for a test

    Returns:
        Tuple of (introduction_results, removal_results)
    """
    logger.info("=" * 60)
    logger.info("RQ1: Do ML-specific code smells negatively impact")
    logger.info("     software maintainability?")
    logger.info("=" * 60)

    # Run introduction tests (H1)
    intro_results = run_transition_tests(data, 'introduction', alpha, min_samples)

    # Run removal tests (H2)
    removal_results = run_transition_tests(data, 'removal', alpha, min_samples)

    return intro_results, removal_results


def run_heterogeneity_tests(
    all_results: List[HypothesisTestResult],
    output_dir: Path,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Run H1.3 heterogeneity tests: Are smell type effects equal?

    Tests H₀: β₁ = β₂ = ... = β₁₂ for each dimension × transition.
    Uses the Wald test statistic computed during regression.

    This tests RQ1.3: whether different smell types have heterogeneous effects.

    Args:
        all_results: List of hypothesis test results (from H1/H2 tests)
        output_dir: Directory to save results
        alpha: Significance level

    Returns:
        DataFrame with heterogeneity test results
    """
    logger.info("\n=== H1.3: HETEROGENEITY TESTS ===")
    logger.info("Testing H₀: β₁ = β₂ = ... = β₁₂ (all smell effects equal)")

    rows = []
    for r in all_results:
        if r.regression.wald_statistic is not None:
            is_sig = r.regression.wald_p_value < alpha
            rows.append({
                'hypothesis_id': f"H1.3_{r.hypothesis_id}",
                'dimension': r.dimension,
                'transition': r.transition_type,
                'wald_statistic': r.regression.wald_statistic,
                'wald_p_value': r.regression.wald_p_value,
                'wald_df': r.regression.wald_df,
                'significant': is_sig,
                'interpretation': 'Heterogeneous effects' if is_sig else 'Homogeneous effects',
                'n_observations': r.regression.n_observations,
                'n_clusters': r.regression.n_clusters
            })

            sig_marker = "***" if is_sig else ""
            logger.info(f"  {r.hypothesis_id} ({r.dimension}, {r.transition_type}): "
                       f"W={r.regression.wald_statistic:.2f}, "
                       f"p={r.regression.wald_p_value:.4f}{sig_marker}, "
                       f"df={r.regression.wald_df}")

    if rows:
        het_df = pd.DataFrame(rows)
        het_file = output_dir / 'rq1_heterogeneity.csv'
        het_df.to_csv(het_file, index=False)
        logger.info(f"Saved: {het_file}")

        sig_count = het_df['significant'].sum()
        logger.info(f"\nHeterogeneity summary: {sig_count}/{len(rows)} "
                   f"dimension-transition pairs show heterogeneous effects")
        return het_df
    else:
        logger.warning("No Wald test results available for heterogeneity analysis")
        return pd.DataFrame()


def apply_multiple_testing_correction(
    results: List[HypothesisTestResult],
    alpha: float = 0.05
) -> List[HypothesisTestResult]:
    """
    Apply Benjamini-Hochberg FDR correction to all test results.

    Args:
        results: List of hypothesis test results
        alpha: Family-wise error rate

    Returns:
        Updated results with adjusted p-values
    """
    if not results:
        return results

    # Extract raw p-values
    p_values = [r.raw_p_value for r in results]

    # Apply BH correction
    adjusted_p, is_significant = benjamini_hochberg_correction(p_values, alpha)

    # Update results
    for i, result in enumerate(results):
        result.adjusted_p_value = adjusted_p[i]
        result.is_significant = is_significant[i]

    return results


def check_hypothesis_support(result: HypothesisTestResult) -> bool:
    """
    Check if a hypothesis result supports the expected direction.

    For introduction (H1):
    - "positive" dimensions (higher=worse): expect POSITIVE coefficient
    - "negative" dimensions (higher=better): expect NEGATIVE coefficient

    For removal (H2):
    - "positive" dimensions (higher=worse): expect NEGATIVE coefficient
    - "negative" dimensions (higher=better): expect POSITIVE coefficient
    """
    direction = DIMENSION_DIRECTION[result.dimension]
    coef = result.regression.coefficient

    if result.transition_type == 'introduction':
        # Introduction should worsen metrics
        if direction == 'positive':
            return coef > 0  # Increase in "bad" metrics
        else:
            return coef < 0  # Decrease in "good" metrics
    else:
        # Removal should improve metrics
        if direction == 'positive':
            return coef < 0  # Decrease in "bad" metrics
        else:
            return coef > 0  # Increase in "good" metrics


def generate_rq1_summary(
    intro_results: List[HypothesisTestResult],
    removal_results: List[HypothesisTestResult]
) -> pd.DataFrame:
    """
    Generate combined summary table for RQ1 results.

    Args:
        intro_results: List of introduction test results
        removal_results: List of removal test results

    Returns:
        Summary DataFrame
    """
    all_results = intro_results + removal_results
    summary_rows = []

    for r in all_results:
        hypothesis_supported = check_hypothesis_support(r)
        sample_adequate = getattr(r, 'sample_size_adequate', True)

        summary_rows.append({
            'Hypothesis': r.hypothesis_id,
            'Transition': r.transition_type.title(),
            'Dimension': r.dimension.replace('_', ' ').title(),
            'Statement': generate_hypothesis_text(r.dimension, r.transition_type),
            'Coefficient': r.regression.coefficient,
            'Std_Error': r.regression.std_error,
            't_Statistic': r.regression.t_statistic,
            'F_Statistic': r.regression.f_statistic,
            'F_p_Value': r.regression.f_p_value,
            'p_Value_Raw': r.raw_p_value,
            'p_Value_Adj': r.adjusted_p_value,
            'CI_Lower': r.regression.ci_lower,
            'CI_Upper': r.regression.ci_upper,
            'R_Squared': r.regression.r_squared,
            'N': r.regression.n_observations,
            'Clusters': r.regression.n_clusters,
            'Effect_Size_Beta': r.effect_size_beta,
            'Effect_Size': r.effect_size_interpretation,
            'Significant': r.is_significant,
            'Sample_Adequate': sample_adequate,
            'Direction_Expected': hypothesis_supported,
            'Conclusion': 'Supported' if (r.is_significant and hypothesis_supported) else
                         'Not Supported (wrong direction)' if (r.is_significant and not hypothesis_supported) else
                         'Not Significant',
            'Notes': '' if sample_adequate else f'Exploratory (n < {ADEQUATE_SAMPLE_SIZE})'
        })

    return pd.DataFrame(summary_rows)


def save_rq1_results(
    intro_results: List[HypothesisTestResult],
    removal_results: List[HypothesisTestResult],
    output_dir: Path
):
    """
    Save RQ1 results to files.

    Args:
        intro_results: List of introduction test results
        removal_results: List of removal test results
        output_dir: Output directory for tables
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = intro_results + removal_results

    # Main results table (all)
    results_df = results_to_dataframe(all_results)
    results_df.to_csv(output_dir / 'rq1_all_results.csv', index=False)
    logger.info(f"Saved: {output_dir / 'rq1_all_results.csv'}")

    # Introduction results only
    if intro_results:
        intro_df = results_to_dataframe(intro_results)
        intro_df.to_csv(output_dir / 'rq1_introduction_results.csv', index=False)
        logger.info(f"Saved: {output_dir / 'rq1_introduction_results.csv'}")

    # Removal results only
    if removal_results:
        removal_df = results_to_dataframe(removal_results)
        removal_df.to_csv(output_dir / 'rq1_removal_results.csv', index=False)
        logger.info(f"Saved: {output_dir / 'rq1_removal_results.csv'}")

    # Combined summary table
    summary_df = generate_rq1_summary(intro_results, removal_results)
    summary_df.to_csv(output_dir / 'rq1_summary.csv', index=False)
    logger.info(f"Saved: {output_dir / 'rq1_summary.csv'}")

    # Results by dimension (all transitions)
    by_dim = summary_df[['Hypothesis', 'Transition', 'Dimension', 'Coefficient',
                         'p_Value_Adj', 'Effect_Size_Beta', 'Effect_Size',
                         'Significant', 'Direction_Expected', 'Conclusion']]
    by_dim.to_csv(output_dir / 'rq1_by_dimension.csv', index=False)
    logger.info(f"Saved: {output_dir / 'rq1_by_dimension.csv'}")

    # Diagnostics with assumption checking and power analysis
    diagnostics = []
    assumption_warnings = []
    power_results = []

    # Number of predictors (12 smell indicators + 6 control variables)
    n_predictors = len(SMELL_INDICATOR_COLS) + len(CONTROL_COLS)

    for r in all_results:
        diag = check_regression_assumptions(r.regression)
        diag['hypothesis_id'] = r.hypothesis_id
        diag['transition'] = r.transition_type
        diag['dimension'] = r.dimension

        # Calculate post-hoc power
        power_info = calculate_post_hoc_power(
            n=r.regression.n_observations,
            k=n_predictors,
            r_squared=r.regression.r_squared,
            alpha=0.05
        )
        diag['power_analysis'] = power_info
        power_results.append({
            'hypothesis_id': r.hypothesis_id,
            'dimension': r.dimension,
            'transition': r.transition_type,
            'n': r.regression.n_observations,
            'r_squared': r.regression.r_squared,
            'power': power_info.get('power'),
            'power_interpretation': power_info.get('power_interpretation'),
            'cohens_f_squared': power_info.get('cohens_f_squared'),
            'effect_size': power_info.get('effect_size_interpretation')
        })

        diagnostics.append(diag)

        # Check for assumption violations and log warnings
        warnings_for_test = []

        # Multicollinearity check
        if diag.get('multicollinearity', {}).get('severe', False):
            warnings_for_test.append(
                f"SEVERE multicollinearity detected (VIF > 10)"
            )
        elif diag.get('multicollinearity', {}).get('problematic', False):
            warnings_for_test.append(
                f"Potential multicollinearity (VIF > 5)"
            )

        # Normality check
        if diag.get('normality', {}).get('is_normal') == False:
            warnings_for_test.append(
                f"Residuals may not be normally distributed (p={diag['normality'].get('shapiro_p_value', 'N/A'):.4f})"
            )

        # Autocorrelation check
        if diag.get('autocorrelation', {}).get('has_autocorrelation', False):
            dw = diag['autocorrelation'].get('durbin_watson', 'N/A')
            warnings_for_test.append(
                f"Possible autocorrelation detected (DW={dw:.2f})"
            )

        if warnings_for_test:
            assumption_warnings.append({
                'hypothesis_id': r.hypothesis_id,
                'dimension': r.dimension,
                'transition': r.transition_type,
                'warnings': warnings_for_test
            })
            for warning in warnings_for_test:
                logger.warning(f"  {r.hypothesis_id}: {warning}")

    if diagnostics:
        diag_file = output_dir / 'rq1_diagnostics.json'
        with open(diag_file, 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        logger.info(f"Saved: {diag_file}")

    # Save power analysis results
    if power_results:
        power_df = pd.DataFrame(power_results)
        power_file = output_dir / 'rq1_power_analysis.csv'
        power_df.to_csv(power_file, index=False)
        logger.info(f"Saved: {power_file}")

        # Log power summary
        adequate_power = sum(1 for p in power_results if p.get('power', 0) and p['power'] >= 0.80)
        underpowered = sum(1 for p in power_results if p.get('power', 0) and p['power'] < 0.80)
        logger.info(f"\nPower analysis summary:")
        logger.info(f"  Adequately powered (≥0.80): {adequate_power}/{len(power_results)}")
        logger.info(f"  Underpowered (<0.80): {underpowered}/{len(power_results)}")

    # Save assumption warnings summary
    if assumption_warnings:
        warnings_file = output_dir / 'rq1_assumption_warnings.json'
        with open(warnings_file, 'w') as f:
            json.dump({
                'total_tests_with_warnings': len(assumption_warnings),
                'warnings': assumption_warnings
            }, f, indent=2)
        logger.info(f"Saved: {warnings_file}")
        logger.warning(f"Found {len(assumption_warnings)} tests with assumption violations - see {warnings_file}")

    # Summary statistics JSON
    intro_adequate = sum(1 for r in intro_results if getattr(r, 'sample_size_adequate', True))
    intro_exploratory = len(intro_results) - intro_adequate
    removal_adequate = sum(1 for r in removal_results if getattr(r, 'sample_size_adequate', True))
    removal_exploratory = len(removal_results) - removal_adequate

    summary_stats = {
        'introduction': {
            'total_tests': len(intro_results),
            'significant_tests': sum(1 for r in intro_results if r.is_significant),
            'supported_hypotheses': sum(1 for r in intro_results
                                       if r.is_significant and check_hypothesis_support(r)),
            'adequate_sample_tests': intro_adequate,
            'exploratory_tests': intro_exploratory,
            'avg_effect_size_beta': np.mean([abs(r.effect_size_beta) for r in intro_results]) if intro_results else 0
        },
        'removal': {
            'total_tests': len(removal_results),
            'significant_tests': sum(1 for r in removal_results if r.is_significant),
            'supported_hypotheses': sum(1 for r in removal_results
                                       if r.is_significant and check_hypothesis_support(r)),
            'adequate_sample_tests': removal_adequate,
            'exploratory_tests': removal_exploratory,
            'avg_effect_size_beta': np.mean([abs(r.effect_size_beta) for r in removal_results]) if removal_results else 0
        },
        'combined': {
            'total_tests': len(all_results),
            'significant_tests': sum(1 for r in all_results if r.is_significant),
            'supported_hypotheses': sum(1 for r in all_results
                                       if r.is_significant and check_hypothesis_support(r)),
            'adequate_sample_tests': intro_adequate + removal_adequate,
            'exploratory_tests': intro_exploratory + removal_exploratory
        },
        'sample_size_threshold': ADEQUATE_SAMPLE_SIZE,
        'exploratory_note': f'Tests with n < {ADEQUATE_SAMPLE_SIZE} are flagged as exploratory/underpowered'
    }

    with open(output_dir / 'rq1_summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    logger.info(f"Saved: {output_dir / 'rq1_summary_stats.json'}")


def run_robustness_analysis(
    all_results: List[HypothesisTestResult],
    output_dir: Path,
    alpha_levels: List[float] = [0.01, 0.05, 0.10]
) -> Dict[str, any]:
    """
    Run robustness/sensitivity analysis at different alpha levels.

    Tests how conclusions change across different significance thresholds.

    Args:
        all_results: List of hypothesis test results
        output_dir: Directory to save robustness results
        alpha_levels: List of alpha values to test

    Returns:
        Dictionary with robustness analysis results
    """
    logger.info("\n=== ROBUSTNESS ANALYSIS ===")
    logger.info(f"Testing conclusions across alpha levels: {alpha_levels}")

    robustness_data = {
        'alpha_levels': alpha_levels,
        'by_alpha': {},
        'stable_conclusions': [],
        'unstable_conclusions': []
    }

    # For each alpha level, re-apply BH correction and count significant results
    for alpha in alpha_levels:
        p_values = [r.raw_p_value for r in all_results]
        _, is_significant = benjamini_hochberg_correction(p_values, alpha)

        significant_count = sum(is_significant)
        supported_count = sum(
            1 for i, r in enumerate(all_results)
            if is_significant[i] and check_hypothesis_support(r)
        )

        robustness_data['by_alpha'][str(alpha)] = {
            'significant_tests': significant_count,
            'supported_hypotheses': supported_count,
            'significant_rate': significant_count / len(all_results) if all_results else 0
        }

        logger.info(f"  α = {alpha}: {significant_count}/{len(all_results)} significant, "
                   f"{supported_count} supported")

    # Identify stable vs unstable conclusions
    for i, r in enumerate(all_results):
        sig_at_levels = []
        for alpha in alpha_levels:
            p_values = [res.raw_p_value for res in all_results]
            _, is_sig = benjamini_hochberg_correction(p_values, alpha)
            sig_at_levels.append(is_sig[i])

        # Stable: same conclusion at all alpha levels
        if all(sig_at_levels) or not any(sig_at_levels):
            robustness_data['stable_conclusions'].append({
                'hypothesis_id': r.hypothesis_id,
                'dimension': r.dimension,
                'transition': r.transition_type,
                'conclusion': 'Significant at all levels' if all(sig_at_levels) else 'Not significant at any level',
                'p_value': r.raw_p_value
            })
        else:
            # Unstable: conclusion changes depending on alpha
            robustness_data['unstable_conclusions'].append({
                'hypothesis_id': r.hypothesis_id,
                'dimension': r.dimension,
                'transition': r.transition_type,
                'significant_at': [alpha_levels[j] for j, s in enumerate(sig_at_levels) if s],
                'not_significant_at': [alpha_levels[j] for j, s in enumerate(sig_at_levels) if not s],
                'p_value': r.raw_p_value
            })

    # Summary
    robustness_data['summary'] = {
        'total_tests': len(all_results),
        'stable_conclusions': len(robustness_data['stable_conclusions']),
        'unstable_conclusions': len(robustness_data['unstable_conclusions']),
        'stability_rate': len(robustness_data['stable_conclusions']) / len(all_results) if all_results else 0
    }

    logger.info(f"\nConclusion stability:")
    logger.info(f"  Stable: {robustness_data['summary']['stable_conclusions']}/{len(all_results)} "
               f"({robustness_data['summary']['stability_rate']:.1%})")
    logger.info(f"  Unstable: {robustness_data['summary']['unstable_conclusions']}/{len(all_results)}")

    # Save robustness analysis
    robustness_file = output_dir / 'rq1_robustness_analysis.json'
    with open(robustness_file, 'w') as f:
        json.dump(robustness_data, f, indent=2, default=str)
    logger.info(f"Saved: {robustness_file}")

    return robustness_data


def main(
    data_path: str = None,
    output_dir: str = None,
    alpha: float = 0.05,
    min_samples: int = 10
):
    """
    Main function to run all RQ1 hypothesis tests.

    Args:
        data_path: Path to analysis-ready directory (contains all_events.csv)
        output_dir: Path to output directory
        alpha: Significance level
        min_samples: Minimum sample size per test
    """
    # Default paths
    base_dir = Path(__file__).parent.parent.parent.parent

    if data_path is None:
        data_path = base_dir / "data" / "analysis-ready"
    else:
        data_path = Path(data_path)

    if output_dir is None:
        output_dir = base_dir / "outputs" / "rq1" / "tables"
    else:
        output_dir = Path(output_dir)

    logger.info("=" * 60)
    logger.info("RQ1: Do ML-specific code smells negatively impact")
    logger.info("     software maintainability?")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Testing both introduction (H1) and removal (H2) effects")
    logger.info("")

    # Load RQ1-specific data (wide format with 12 smell indicator columns)
    # These files have individual smell type columns as treatment variables
    rq1_dir = data_path / "rq1"
    intro_wide_file = rq1_dir / "introduction_wide.csv"
    removal_wide_file = rq1_dir / "removal_wide.csv"

    # Fall back to old format if wide format doesn't exist
    intro_file = rq1_dir / "introduction_data.csv"
    removal_file = rq1_dir / "removal_data.csv"

    # Check for wide format first (RQ revision)
    use_wide_format = intro_wide_file.exists() and removal_wide_file.exists()

    if not use_wide_format and not (intro_file.exists() and removal_file.exists()):
        logger.info("RQ1 data not found. Running data preparation...")
        from src.step4_analysis.data_preparation.prepare_data import main as prepare_main
        prepare_main()
        use_wide_format = intro_wide_file.exists() and removal_wide_file.exists()

    if use_wide_format:
        logger.info("Using wide-format data with 12 smell indicator columns")
        intro_data = pd.read_csv(intro_wide_file)
        removal_data = pd.read_csv(removal_wide_file)
        data = pd.concat([intro_data, removal_data], ignore_index=True)
        logger.info(f"Loaded RQ1 wide-format data: {len(intro_data)} introductions, {len(removal_data)} removals")

        # Show smell columns available
        smell_cols = [c for c in data.columns if c.startswith('smell_')]
        logger.info(f"  - Smell indicator columns: {len(smell_cols)}")
        logger.info(f"  - Introduction observations: {len(intro_data)}")
        logger.info(f"  - Removal observations: {len(removal_data)}")
    elif intro_file.exists() and removal_file.exists():
        logger.warning("Wide-format data not found, using legacy format (single treatment variable)")
        intro_data = pd.read_csv(intro_file)
        removal_data = pd.read_csv(removal_file)
        data = pd.concat([intro_data, removal_data], ignore_index=True)
        logger.info(f"Loaded RQ1 data: {len(intro_data)} introductions, {len(removal_data)} removals")

        # Show breakdown
        intro_count = len(intro_data)
        removal_count = len(removal_data)
        logger.info(f"  - Introduction observations: {intro_count}")
        logger.info(f"  - Removal observations: {removal_count}")
    else:
        logger.error(f"RQ1 data files not found in: {rq1_dir}")
        logger.info("Please run data preparation first: python -m src.step4_analysis.data_preparation.prepare_data")
        return None

    # Run tests
    intro_results, removal_results = run_rq1_tests(data, alpha=alpha, min_samples=min_samples)

    all_results = intro_results + removal_results
    if not all_results:
        logger.error("No tests completed successfully")
        return None

    # Apply multiple testing correction (across all 12 tests: 6 dimensions × 2 transitions)
    all_results = apply_multiple_testing_correction(all_results, alpha=alpha)

    # Split back after correction
    intro_results = [r for r in all_results if r.transition_type == 'introduction']
    removal_results = [r for r in all_results if r.transition_type == 'removal']

    # Save results
    save_rq1_results(intro_results, removal_results, output_dir)

    # Print summary
    summary = generate_rq1_summary(intro_results, removal_results)

    logger.info("\n" + "=" * 60)
    logger.info("RQ1 SUMMARY")
    logger.info("=" * 60)

    # Introduction summary
    logger.info("\n--- H1: INTRODUCTION IMPACT ---")
    intro_sig = sum(1 for r in intro_results if r.is_significant)
    intro_supported = sum(1 for r in intro_results if r.is_significant and check_hypothesis_support(r))
    logger.info(f"Tests completed: {len(intro_results)}/{len(DIMENSIONS)}")
    logger.info(f"Significant results (p < {alpha}): {intro_sig}")
    logger.info(f"Hypotheses supported: {intro_supported}")

    for r in intro_results:
        sig = "***" if r.is_significant else ""
        direction = "as expected" if check_hypothesis_support(r) else "opposite"
        logger.info(f"  {r.hypothesis_id} ({r.dimension}): "
                   f"β={r.regression.coefficient:.4f}, p={r.adjusted_p_value:.4f}{sig}, "
                   f"effect={r.effect_size_interpretation}, {direction}")

    # Removal summary
    logger.info("\n--- H2: REMOVAL IMPACT ---")
    removal_sig = sum(1 for r in removal_results if r.is_significant)
    removal_supported = sum(1 for r in removal_results if r.is_significant and check_hypothesis_support(r))
    logger.info(f"Tests completed: {len(removal_results)}/{len(DIMENSIONS)}")
    logger.info(f"Significant results (p < {alpha}): {removal_sig}")
    logger.info(f"Hypotheses supported: {removal_supported}")

    for r in removal_results:
        sig = "***" if r.is_significant else ""
        direction = "improved" if check_hypothesis_support(r) else "worsened"
        logger.info(f"  {r.hypothesis_id} ({r.dimension}): "
                   f"β={r.regression.coefficient:.4f}, p={r.adjusted_p_value:.4f}{sig}, "
                   f"effect={r.effect_size_interpretation}, {direction}")

    # Overall conclusion
    logger.info("\n" + "-" * 40)
    total_supported = intro_supported + removal_supported
    total_sig = intro_sig + removal_sig
    logger.info(f"OVERALL: {total_supported}/{total_sig} significant effects in expected direction")
    logger.info("-" * 40)

    # Run H1.3 heterogeneity tests (Wald test: β₁ = β₂ = ... = β₁₂)
    run_heterogeneity_tests(all_results, output_dir, alpha=alpha)

    # Run robustness analysis at multiple alpha levels
    run_robustness_analysis(all_results, output_dir)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RQ1 hypothesis tests (introduction and removal)")
    parser.add_argument("--data", type=str, help="Path to analysis-ready data")
    parser.add_argument("--output", type=str, help="Path to output directory")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum samples per test")

    args = parser.parse_args()

    main(
        data_path=args.data,
        output_dir=args.output,
        alpha=args.alpha,
        min_samples=args.min_samples
    )
