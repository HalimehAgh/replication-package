"""
Regression Utilities for Hypothesis Testing

Implements OLS regression with clustered standard errors for analyzing
the impact of ML code smells on software maintainability.

Uses scipy and numpy for pure Python implementation without statsmodels
dependency issues.

Effect sizes are measured using beta coefficients from the OLS regression,
interpreted as: "For each additional smell, the maintainability dimension
score changes by β units."

Reference:
    - Cameron, A.C., Gelbach, J.B., & Miller, D.L. (2008). Bootstrap-Based
      Improvements for Inference with Clustered Errors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy import linalg

# Import statistical utilities from shared module
from src.utils.stat_utils import (
    benjamini_hochberg_correction,
)

# Import clustering utilities for cluster diagnostics
from src.step4_analysis.hypothesis_testing.clustering import (
    cluster_summary,
    diagnose_cluster_effects,
)


@dataclass
class RegressionResult:
    """Container for regression results."""
    coefficient: float
    std_error: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    r_squared: float
    adj_r_squared: float
    n_observations: int
    n_clusters: int
    vif_values: Optional[Dict[str, float]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    # For multi-indicator models: store all treatment coefficients
    multi_coefficients: Optional[Dict[str, float]] = None
    multi_std_errors: Optional[Dict[str, float]] = None
    multi_t_statistics: Optional[Dict[str, float]] = None
    multi_p_values: Optional[Dict[str, float]] = None
    # F-test for joint significance of all treatment variables
    f_statistic: Optional[float] = None
    f_p_value: Optional[float] = None
    # Wald test for equality of treatment coefficients (H1.3 heterogeneity)
    wald_statistic: Optional[float] = None
    wald_p_value: Optional[float] = None
    wald_df: Optional[int] = None
    # Full coefficient vector and clustered variance matrix (for Wald test)
    full_beta: Optional[np.ndarray] = None
    full_vcov: Optional[np.ndarray] = None
    var_names: Optional[List[str]] = None


# Beta coefficient effect size thresholds for NORMALIZED (0-1 scale) metrics
# Calibrated based on empirical analysis of maintainability dimension deltas:
#   - 0.005 normalized ≈ 9 raw cyclomatic complexity units
#   - 0.01 normalized ≈ 20 raw cyclomatic complexity units
#   - 0.02 normalized ≈ 40 raw cyclomatic complexity units
# Thresholds set relative to practical significance in code quality:
BETA_EFFECT_THRESHOLDS = {
    "negligible": 0.002,  # < 4 CC units - barely noticeable
    "small": 0.005,       # 4-9 CC units - minor but detectable
    "medium": 0.015,      # 9-30 CC units - meaningful change
}
# >= 0.015 is "large" (30+ CC units - substantial change)


def interpret_beta_coefficient(beta: float, se: float = None) -> str:
    """
    Interpret the magnitude of a beta coefficient for normalized metrics.

    Thresholds calibrated for 0-1 normalized maintainability scores where:
        - 0.005 normalized ≈ 9 raw cyclomatic complexity units
        - |β| < 0.002: negligible (< 4 CC units)
        - 0.002 <= |β| < 0.005: small (4-9 CC units)
        - 0.005 <= |β| < 0.015: medium (9-30 CC units)
        - |β| >= 0.015: large (30+ CC units)

    Args:
        beta: The beta coefficient value
        se: Standard error (optional, for future standardization)

    Returns:
        Interpretation string: "negligible", "small", "medium", or "large"
    """
    abs_beta = abs(beta)
    if abs_beta < BETA_EFFECT_THRESHOLDS["negligible"]:
        return "negligible"
    elif abs_beta < BETA_EFFECT_THRESHOLDS["small"]:
        return "small"
    elif abs_beta < BETA_EFFECT_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "large"


@dataclass
class HypothesisTestResult:
    """Container for hypothesis test results."""
    hypothesis_id: str
    dimension: str
    transition_type: str
    smell_type: Optional[str]
    regression: RegressionResult
    effect_size_beta: float
    effect_size_interpretation: str
    effect_direction: str
    is_significant: bool
    raw_p_value: float
    adjusted_p_value: Optional[float] = None


def calculate_vif(X: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Variance Inflation Factor for each predictor.

    VIF > 5 indicates potential multicollinearity issues.
    VIF > 10 indicates serious multicollinearity problems.

    Args:
        X: DataFrame of predictors (without constant)

    Returns:
        Dictionary mapping variable names to VIF values
    """
    vif_data = {}
    X_matrix = X.values.astype(float)

    for i, col in enumerate(X.columns):
        try:
            # For each variable, regress it on all others
            y_col = X_matrix[:, i]
            X_others = np.delete(X_matrix, i, axis=1)

            if X_others.shape[1] == 0:
                vif_data[col] = 1.0
                continue

            # Add constant
            X_others = np.column_stack([np.ones(len(y_col)), X_others])

            # Compute R-squared
            try:
                beta = np.linalg.lstsq(X_others, y_col, rcond=None)[0]
                y_pred = X_others @ beta
                ss_res = np.sum((y_col - y_pred) ** 2)
                ss_tot = np.sum((y_col - np.mean(y_col)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
                vif_data[col] = vif
            except (np.linalg.LinAlgError, ZeroDivisionError):
                vif_data[col] = np.inf
        except Exception:
            vif_data[col] = np.inf

    return vif_data


def wald_test_equality(
    beta_vec: np.ndarray,
    V_cluster: np.ndarray,
    treatment_indices: List[int]
) -> Tuple[float, float, int]:
    """
    Wald test for equality of treatment coefficients.

    Tests H₀: β₁ = β₂ = ... = βₘ (all treatment effects are equal)
    using (m-1) linear restrictions.

    The contrast matrix R encodes pairwise differences:
        R = [[1, -1, 0, ..., 0],
             [0,  1, -1, ..., 0],
             ...
             [0,  0, ..., 1, -1]]

    W = (Rβ)' [R V R']^{-1} (Rβ) ~ χ²(m-1)

    Args:
        beta_vec: Full coefficient vector (including constant)
        V_cluster: Clustered variance-covariance matrix
        treatment_indices: Indices of treatment coefficients in beta_vec

    Returns:
        Tuple of (wald_statistic, p_value, degrees_of_freedom)
    """
    m = len(treatment_indices)
    if m < 2:
        return 0.0, 1.0, 0

    k = len(beta_vec)
    R = np.zeros((m - 1, k))
    for i in range(m - 1):
        R[i, treatment_indices[i]] = 1
        R[i, treatment_indices[i + 1]] = -1

    Rb = R @ beta_vec
    RVRt = R @ V_cluster @ R.T

    try:
        W = float(Rb.T @ np.linalg.inv(RVRt) @ Rb)
        p_value = float(1 - stats.chi2.cdf(W, m - 1))
    except np.linalg.LinAlgError:
        W = 0.0
        p_value = 1.0

    return W, p_value, m - 1


def run_ols_clustered(
    y: pd.Series,
    X: pd.DataFrame,
    cluster_var: pd.Series,
    alpha: float = 0.05,
    include_cluster_diagnostics: bool = True,
    treatment_cols: Optional[List[str]] = None
) -> RegressionResult:
    """
    Run OLS regression with clustered standard errors.

    Uses clustered standard errors to account for within-cluster
    correlation of errors (Cameron & Miller, 2015).

    Model: Y = beta_0 + beta_1*X1 + ... + beta_n*Xn + epsilon

    Args:
        y: Dependent variable (e.g., delta_dimension_score)
        X: Independent variables DataFrame (predictors)
        cluster_var: Cluster identifier (e.g., file_id)
        alpha: Significance level for confidence intervals
        include_cluster_diagnostics: Include ICC and cluster statistics (default True)

    Returns:
        RegressionResult with coefficient estimates, clustered SEs, and diagnostics
    """
    # Convert to numpy arrays
    y_arr = y.values.astype(float)
    X_arr = X.values.astype(float)
    var_names = X.columns.tolist()

    # Add constant term
    n = len(y_arr)
    X_with_const = np.column_stack([np.ones(n), X_arr])
    k = X_with_const.shape[1]  # Number of parameters (including constant)

    # OLS estimation: beta = (X'X)^(-1) X'y
    try:
        XtX = X_with_const.T @ X_with_const
        XtX_inv = np.linalg.inv(XtX)
        beta = XtX_inv @ (X_with_const.T @ y_arr)
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix - cannot compute OLS estimates")

    # Residuals
    y_pred = X_with_const @ beta
    residuals = y_arr - y_pred

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k) if n > k else 0

    # Clustered standard errors
    clusters = cluster_var.values
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Compute cluster-robust variance matrix
    # V_cluster = (X'X)^(-1) * B * (X'X)^(-1)
    # where B = sum_g (X_g' * e_g * e_g' * X_g)

    meat = np.zeros((k, k))
    for cluster in unique_clusters:
        mask = clusters == cluster
        X_g = X_with_const[mask]
        e_g = residuals[mask]

        # Outer product of residuals weighted by X
        score_g = X_g.T @ e_g  # k x 1
        meat += np.outer(score_g, score_g)

    # Small sample adjustment: (G / (G-1)) * ((n-1) / (n-k))
    # where G is number of clusters
    if n_clusters > 1:
        adjustment = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    else:
        adjustment = 1.0

    # Clustered variance-covariance matrix
    V_cluster = adjustment * XtX_inv @ meat @ XtX_inv

    # Standard errors
    se = np.sqrt(np.diag(V_cluster))

    # Create mapping of variable names to indices (including constant)
    all_var_names = ['const'] + var_names
    var_to_idx = {name: i for i, name in enumerate(all_var_names)}

    # Extract coefficient for the main predictor (first non-constant variable)
    # Index 0 is the constant, index 1 is the main predictor
    main_idx = 1
    coef = beta[main_idx]
    se_coef = se[main_idx]

    # t-statistic and p-value (using t-distribution with G-1 degrees of freedom)
    df = max(n_clusters - 1, 1)
    t_stat = coef / se_coef if se_coef > 0 else 0
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    # Confidence interval
    t_critical = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = coef - t_critical * se_coef
    ci_upper = coef + t_critical * se_coef

    # Calculate VIF for multicollinearity check
    vif_values = calculate_vif(X)

    # Multi-indicator handling: compute individual statistics for treatment variables
    multi_coefficients = None
    multi_std_errors = None
    multi_t_statistics = None
    multi_p_values = None
    f_statistic = None
    f_p_value = None

    if treatment_cols and len(treatment_cols) > 0:
        multi_coefficients = {}
        multi_std_errors = {}
        multi_t_statistics = {}
        multi_p_values = {}

        for col in treatment_cols:
            if col in var_to_idx:
                idx = var_to_idx[col]
                b = beta[idx]
                se_b = se[idx]
                t_b = b / se_b if se_b > 0 else 0
                p_b = 2 * (1 - stats.t.cdf(abs(t_b), df))

                multi_coefficients[col] = b
                multi_std_errors[col] = se_b
                multi_t_statistics[col] = t_b
                multi_p_values[col] = p_b

        # Compute F-test for joint significance of all treatment variables
        # H0: β₁ = β₂ = ... = βₘ = 0 (all treatment coefficients are zero)
        # F = (R² / m) / ((1 - R²) / (n - k - 1))
        # where m = number of treatment variables, k = total predictors
        if len(treatment_cols) > 0 and r_squared > 0:
            m = len([c for c in treatment_cols if c in var_to_idx])  # Number of restrictions
            if m > 0 and n > k:
                # Compute restricted model R² (model without treatment variables)
                control_indices = [i for i, name in enumerate(all_var_names)
                                   if name != 'const' and name not in treatment_cols]
                if control_indices:
                    # Restricted model: only controls
                    X_restricted = np.column_stack([np.ones(n)] +
                                                   [X_with_const[:, i] for i in control_indices])
                    try:
                        beta_r = np.linalg.lstsq(X_restricted, y_arr, rcond=None)[0]
                        y_pred_r = X_restricted @ beta_r
                        ss_res_r = np.sum((y_arr - y_pred_r) ** 2)
                        r_squared_r = 1 - (ss_res_r / ss_tot) if ss_tot > 0 else 0
                    except np.linalg.LinAlgError:
                        r_squared_r = 0
                else:
                    # No controls - restricted model is just the mean
                    r_squared_r = 0

                # F-statistic: incremental R² from treatment variables
                delta_r2 = r_squared - r_squared_r
                if delta_r2 > 0:
                    df_num = m  # Number of treatment variables
                    df_denom = n - k  # Residual degrees of freedom
                    if df_denom > 0:
                        f_statistic = (delta_r2 / m) / ((1 - r_squared) / df_denom)
                        f_p_value = 1 - stats.f.cdf(f_statistic, df_num, df_denom)

    # Wald test for equality of treatment coefficients (H1.3 heterogeneity)
    wald_statistic = None
    wald_p_value = None
    wald_df = None
    if treatment_cols and len(treatment_cols) >= 2:
        treatment_indices = [var_to_idx[col] for col in treatment_cols if col in var_to_idx]
        if len(treatment_indices) >= 2:
            wald_statistic, wald_p_value, wald_df = wald_test_equality(
                beta, V_cluster, treatment_indices
            )

    # Diagnostics
    diagnostics = {
        'residuals': residuals,
        'fitted_values': y_pred,
        'all_coefficients': dict(zip(['const'] + var_names, beta)),
        'all_std_errors': dict(zip(['const'] + var_names, se)),
    }

    # Add cluster diagnostics if requested
    if include_cluster_diagnostics:
        try:
            # Build temporary DataFrame for clustering functions
            cluster_df = pd.DataFrame({
                'y': y_arr,
                'cluster_id': cluster_var.values
            })
            # Get cluster summary statistics
            diagnostics['cluster_summary'] = cluster_summary(cluster_df, 'cluster_id')
            # Get cluster effect diagnostics (ICC, design effect)
            diagnostics['cluster_effects'] = diagnose_cluster_effects(
                cluster_df, 'y', 'cluster_id'
            )
        except Exception:
            # Silently skip if clustering diagnostics fail
            pass

    return RegressionResult(
        coefficient=coef,
        std_error=se_coef,
        t_statistic=t_stat,
        p_value=p_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        n_observations=int(n),
        n_clusters=int(n_clusters),
        vif_values=vif_values,
        diagnostics=diagnostics,
        multi_coefficients=multi_coefficients,
        multi_std_errors=multi_std_errors,
        multi_t_statistics=multi_t_statistics,
        multi_p_values=multi_p_values,
        f_statistic=f_statistic,
        f_p_value=f_p_value,
        wald_statistic=wald_statistic,
        wald_p_value=wald_p_value,
        wald_df=wald_df,
        full_beta=beta,
        full_vcov=V_cluster,
        var_names=['const'] + var_names
    )


def run_hypothesis_test(
    data: pd.DataFrame,
    y_col: str,
    treatment_col: str,
    control_cols: List[str],
    cluster_col: str,
    hypothesis_id: str,
    dimension: str,
    transition_type: str,
    smell_type: Optional[str] = None,
    alpha: float = 0.05,
    min_samples: int = 10,
    multi_treatment_cols: Optional[List[str]] = None
) -> Optional[HypothesisTestResult]:
    """
    Run a single hypothesis test with regression and effect size calculation.

    Regression Model (aligned with specification):
        ΔMaintainability_d = β₀ + β₁*smell_change + β₂*total_smells_at_commit +
                             β₃*baseline + β₄*total_churn + β₅*bug_fix_ratio +
                             β₆*log_loc + ε

    For multi-indicator models (RQ1 revision):
        ΔMaintainability_d = β₀ + Σᵢβᵢ*SmellTypeᵢ + controls + ε

    Args:
        data: DataFrame with all variables
        y_col: Name of dependent variable column
        treatment_col: Name of treatment/main predictor column
        control_cols: List of control variable column names
        cluster_col: Name of cluster identifier column
        hypothesis_id: Unique identifier for this hypothesis
        dimension: Maintainability dimension being tested
        transition_type: 'introduction' or 'removal'
        smell_type: Specific smell type (None for RQ1 aggregate tests)
        alpha: Significance level
        min_samples: Minimum sample size required
        multi_treatment_cols: List of treatment columns for multi-indicator model (RQ1 revision)
                             When provided, performs F-test for joint significance

    Returns:
        HypothesisTestResult or None if insufficient data
    """
    # Check sample size
    if len(data) < min_samples:
        warnings.warn(f"Insufficient samples for {hypothesis_id}: {len(data)} < {min_samples}")
        return None

    # Prepare variables
    y = data[y_col]

    # For multi-indicator model, use all treatment columns
    if multi_treatment_cols:
        treatment_cols_list = multi_treatment_cols
        X_cols = treatment_cols_list + [c for c in control_cols if c not in treatment_cols_list]
    else:
        treatment_cols_list = [treatment_col]
        X_cols = [treatment_col] + control_cols

    X = data[X_cols].copy()
    cluster_var = data[cluster_col]

    # Handle missing values
    valid_mask = ~(y.isna() | X.isna().any(axis=1) | cluster_var.isna())
    y = y[valid_mask]
    X = X[valid_mask]
    cluster_var = cluster_var[valid_mask]

    if len(y) < min_samples:
        warnings.warn(f"Insufficient valid samples for {hypothesis_id} after removing NAs")
        return None

    # Run regression with treatment columns for multi-indicator F-test
    try:
        regression_result = run_ols_clustered(
            y, X, cluster_var, alpha,
            treatment_cols=treatment_cols_list if multi_treatment_cols else None
        )
    except Exception as e:
        warnings.warn(f"Regression failed for {hypothesis_id}: {str(e)}")
        return None

    # For multi-indicator model, use F-test for joint significance
    if multi_treatment_cols and len(multi_treatment_cols) > 1:
        # Get all treatment coefficients from the multi_coefficients
        if regression_result.multi_coefficients:
            treatment_coefs = list(regression_result.multi_coefficients.values())
            # Effect size: use average of absolute treatment coefficients
            avg_beta = np.mean([abs(c) for c in treatment_coefs]) if treatment_coefs else 0
        else:
            avg_beta = abs(regression_result.coefficient)

        beta = avg_beta
        beta_interpretation = interpret_beta_coefficient(beta)

        # For p-value, use the F-test p-value for joint significance
        # If F-test failed, fall back to the first treatment coefficient p-value
        if regression_result.f_p_value is not None:
            p_value_for_significance = regression_result.f_p_value
        else:
            p_value_for_significance = regression_result.p_value
    else:
        # Single treatment variable
        beta = regression_result.coefficient
        beta_interpretation = interpret_beta_coefficient(beta)
        p_value_for_significance = regression_result.p_value

    # Determine effect direction based on coefficient sign
    # For multi-indicator, we look at the sum of coefficients to determine overall direction
    if multi_treatment_cols and regression_result.multi_coefficients:
        coef_sum = sum(regression_result.multi_coefficients.values())
        if coef_sum > 0:
            effect_direction = "increases" if transition_type == "introduction" else "improves"
        else:
            effect_direction = "decreases" if transition_type == "introduction" else "worsens"
    else:
        if beta > 0:
            effect_direction = "increases" if transition_type == "introduction" else "improves"
        else:
            effect_direction = "decreases" if transition_type == "introduction" else "worsens"

    return HypothesisTestResult(
        hypothesis_id=hypothesis_id,
        dimension=dimension,
        transition_type=transition_type,
        smell_type=smell_type,
        regression=regression_result,
        effect_size_beta=beta,
        effect_size_interpretation=beta_interpretation,
        effect_direction=effect_direction,
        is_significant=p_value_for_significance < alpha,
        raw_p_value=p_value_for_significance
    )


def results_to_dataframe(results: List[HypothesisTestResult]) -> pd.DataFrame:
    """
    Convert list of HypothesisTestResult objects to a DataFrame.

    Args:
        results: List of test results

    Returns:
        DataFrame with one row per test
    """
    rows = []
    for r in results:
        if r is None:
            continue
        row = {
            'hypothesis_id': r.hypothesis_id,
            'dimension': r.dimension,
            'transition_type': r.transition_type,
            'smell_type': r.smell_type or 'all',
            'coefficient': r.regression.coefficient,
            'std_error': r.regression.std_error,
            't_statistic': r.regression.t_statistic,
            'p_value_raw': r.raw_p_value,
            'p_value_adjusted': r.adjusted_p_value,
            'ci_lower': r.regression.ci_lower,
            'ci_upper': r.regression.ci_upper,
            'r_squared': r.regression.r_squared,
            'adj_r_squared': r.regression.adj_r_squared,
            'n_observations': r.regression.n_observations,
            'n_clusters': r.regression.n_clusters,
            'effect_size_beta': r.effect_size_beta,
            'effect_size_interpretation': r.effect_size_interpretation,
            'effect_direction': r.effect_direction,
            'is_significant': r.is_significant,
        }

        # Add F-test results for multi-indicator models
        if r.regression.f_statistic is not None:
            row['f_statistic'] = r.regression.f_statistic
            row['f_p_value'] = r.regression.f_p_value

        # Add individual smell coefficients if available
        if r.regression.multi_coefficients:
            for smell_col, coef in r.regression.multi_coefficients.items():
                row[f'{smell_col}_coef'] = coef
            for smell_col, pval in r.regression.multi_p_values.items():
                row[f'{smell_col}_pval'] = pval

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_post_hoc_power(
    n: int,
    k: int,
    r_squared: float,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Calculate post-hoc statistical power for OLS regression.

    Uses Cohen's f² effect size measure for regression:
    f² = R² / (1 - R²)

    Interpretation of f² (Cohen, 1988):
    - 0.02 = small effect
    - 0.15 = medium effect
    - 0.35 = large effect

    Args:
        n: Sample size (number of observations)
        k: Number of predictors (excluding constant)
        r_squared: R² value from regression
        alpha: Significance level

    Returns:
        Dictionary with power analysis results
    """
    # Calculate Cohen's f² effect size
    if r_squared >= 1.0:
        r_squared = 0.999  # Cap to avoid division by zero
    f_squared = r_squared / (1 - r_squared)

    # Interpret effect size
    if f_squared < 0.02:
        effect_interpretation = "negligible"
    elif f_squared < 0.15:
        effect_interpretation = "small"
    elif f_squared < 0.35:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    # Degrees of freedom
    df_num = k  # Number of predictors
    df_denom = n - k - 1  # Residual df

    if df_denom <= 0:
        return {
            'power': None,
            'cohens_f_squared': f_squared,
            'effect_size_interpretation': effect_interpretation,
            'error': 'Insufficient degrees of freedom'
        }

    # Non-centrality parameter
    lambda_nc = f_squared * (n)

    # Calculate power using F distribution
    # Power = P(F > F_crit | H1 is true)
    try:
        f_critical = stats.f.ppf(1 - alpha, df_num, df_denom)
        # Power is 1 - beta, where beta = P(F < F_crit | lambda)
        # Using non-central F distribution
        power = 1 - stats.ncf.cdf(f_critical, df_num, df_denom, lambda_nc)
    except Exception:
        power = None

    # Power interpretation
    if power is not None:
        if power < 0.50:
            power_interpretation = "underpowered"
        elif power < 0.80:
            power_interpretation = "marginally powered"
        else:
            power_interpretation = "adequately powered"
    else:
        power_interpretation = "unknown"

    return {
        'power': power,
        'power_interpretation': power_interpretation,
        'cohens_f_squared': f_squared,
        'effect_size_interpretation': effect_interpretation,
        'sample_size': n,
        'predictors': k,
        'alpha': alpha,
        'recommended_interpretation': (
            'Power >= 0.80 is conventionally considered adequate. '
            f'Current power: {power:.3f}' if power else 'Could not calculate power'
        )
    }


def check_regression_assumptions(result: RegressionResult) -> Dict[str, Any]:
    """
    Check regression assumptions and return diagnostic summary.

    Checks:
    1. Multicollinearity (VIF)
    2. Normality of residuals (Shapiro-Wilk)
    3. Autocorrelation (Durbin-Watson)

    Args:
        result: RegressionResult with diagnostics

    Returns:
        Dictionary with assumption check results
    """
    checks = {}

    # Multicollinearity
    if result.vif_values:
        max_vif = max(result.vif_values.values())
        checks['multicollinearity'] = {
            'max_vif': max_vif,
            'problematic': max_vif > 5,
            'severe': max_vif > 10,
            'vif_values': result.vif_values
        }

    # Normality of residuals
    if result.diagnostics and 'residuals' in result.diagnostics:
        residuals = result.diagnostics['residuals']
        if len(residuals) >= 3:
            try:
                stat, p = stats.shapiro(residuals[:5000])  # Shapiro limited to 5000
                checks['normality'] = {
                    'shapiro_statistic': stat,
                    'shapiro_p_value': p,
                    'is_normal': p > 0.05
                }
            except Exception:
                checks['normality'] = {'error': 'Could not compute Shapiro-Wilk test'}

    # Durbin-Watson approximation
    if result.diagnostics and 'residuals' in result.diagnostics:
        residuals = result.diagnostics['residuals']
        if len(residuals) > 1:
            diff = np.diff(residuals)
            dw = np.sum(diff ** 2) / np.sum(residuals ** 2)
            checks['autocorrelation'] = {
                'durbin_watson': dw,
                'has_autocorrelation': dw < 1.5 or dw > 2.5
            }

    return checks


if __name__ == "__main__":
    # Simple test
    print("Regression utilities module loaded successfully.")

    # Test beta coefficient interpretation
    test_betas = [0.02, 0.07, 0.15, 0.35]
    print("\nBeta coefficient interpretation test:")
    for beta in test_betas:
        interp = interpret_beta_coefficient(beta)
        print(f"  β={beta:.2f} -> {interp}")

    # Test BH correction
    p_vals = [0.01, 0.04, 0.03, 0.05, 0.10]
    adj, sig = benjamini_hochberg_correction(p_vals)
    print(f"\nBH correction test: adjusted={[f'{p:.3f}' for p in adj]}")

    # Test OLS regression
    np.random.seed(42)
    n = 100
    X_data = pd.DataFrame({
        'treatment': np.random.randn(n),
        'control1': np.random.randn(n),
        'control2': np.random.randn(n)
    })
    y_data = pd.Series(0.5 * X_data['treatment'] + 0.2 * X_data['control1'] + np.random.randn(n) * 0.5)
    clusters = pd.Series(np.random.choice(['A', 'B', 'C', 'D', 'E'], n))

    result = run_ols_clustered(y_data, X_data, clusters)
    print(f"\nOLS test: coef={result.coefficient:.4f}, SE={result.std_error:.4f}, "
          f"p={result.p_value:.4f}, R2={result.r_squared:.4f}")
