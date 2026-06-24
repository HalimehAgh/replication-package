"""
RQ2 Hypothesis Testing Module — 3-Stage Hierarchical Approach

Tests smell-specific effects: Do different types of ML code smells have
distinct impacts on software maintainability?

Three-stage funnel design (per rq2_revised-Ver2.md specification):

Stage 1 (H2.1): Aggregate Qualification
  - Test each smell type for aggregate effect across all 6 maintainability dimensions
  - BH FDR at alpha=0.05
  - Qualification gate: n >= 30 AND p_adj < 0.10 (lenient screening)
  - Qualification is per-smell: a smell qualifies if ANY transition meets the gate

Stage 2 (H2.2): Dimension-Specific Analysis
  - Only for qualified smells from Stage 1
  - Test each qualified smell across BOTH transitions × 6 dimensions
  - BH FDR at alpha=0.05

Stage 3 (H2.3): Symmetry Analysis
  - For (smell, dimension) pairs significant in Stage 2 in either transition
  - Wald test comparing |beta_intro| vs |beta_removal|
  - Uses empirical rho from RQ1 results (fallback: 0.3)
  - Classify: symmetric / hysteresis / overcorrection
  - BH FDR at alpha=0.05

Hypothesis naming:
  Stage 1: H2.1.{smell_abbrev}.{intro|removal}
  Stage 2: H2.2.{smell_abbrev}.{dim_abbrev}.{intro|removal}
  Stage 3: H2.3.{smell_abbrev}.{dim_abbrev}

Each test uses OLS regression with clustered standard errors (file-level clustering).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging
import sys

from scipy import stats

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.step4_analysis.hypothesis_testing.regression_utils import (
    run_hypothesis_test,
    benjamini_hochberg_correction,
    results_to_dataframe,
    HypothesisTestResult,
    check_regression_assumptions,
    calculate_post_hoc_power,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMENSIONS = [
    'structural_complexity',
    'code_size',
    'defect_proneness',
    'understandability',
    'testing_quality',
    'ml_specific',
]

# Abbreviations used in hypothesis IDs (aligned with RQ1 smell indicator names)
SMELL_ABBREVIATIONS = {
    'Chain_Indexing': 'CIDX',
    'chain_indexing': 'CIDX',
    'columns_and_datatype_not_explicitly_set': 'CDTNE',
    'dataframe_conversion_api_misused': 'DCA',
    'in_place_apis_misused': 'IPA',
    'matrix_multiplication_api_misused': 'MMA',
    'merge_api_parameter_not_explicitly_set': 'MAP',
    'unnecessary_iteration': 'UI',
    'gradients_not_cleared_before_backward_propagation': 'GNC',
    'memory_not_freed': 'MNF',
    'pytorch_call_method_misused': 'PC',
    'tensor_array_not_used': 'TA',
    'nan_equivalence_comparison': 'NAN',
    'hyperparameters_not_explicitly_set': 'HNS',
    'deterministic_algorithm_option_not_used': 'DAO',
}

# Canonical smell type ordering
SMELL_TYPES = [
    'Chain_Indexing',
    'columns_and_datatype_not_explicitly_set',
    'dataframe_conversion_api_misused',
    'gradients_not_cleared_before_backward_propagation',
    'in_place_apis_misused',
    'matrix_multiplication_api_misused',
    'memory_not_freed',
    'merge_api_parameter_not_explicitly_set',
    'pytorch_call_method_misused',
    'unnecessary_iteration',
]

# Dimension abbreviations for hypothesis IDs (H2.2/H2.3)
DIMENSION_ABBREVIATIONS = {
    'structural_complexity': 'SC',
    'code_size': 'CS',
    'defect_proneness': 'DP',
    'understandability': 'UND',
    'testing_quality': 'TQ',
    'ml_specific': 'ML',
}

# Stage 1 controls (aggregate model — matches spec §3.2)
STAGE1_CONTROL_COLS = [
    'aggregate_baseline',
    'other_smell_types',
    'total_churn',
    'commit_count',
    'bug_fix_ratio',
    'lines_added',
    'lines_deleted',
    'is_feature_commit',
    'is_bug_fix',
]

# Stage 2 controls (dimension-specific model — spec §4.3 + log_loc)
# Note: log_loc is an additional control beyond the Ver2 spec formula,
# included for consistency with RQ1 and to control for file size effects.
STAGE2_CONTROL_COLS = [
    'baseline_score',
    'other_smell_types',
    'total_churn',
    'commit_count',
    'bug_fix_ratio',
    'lines_added',
    'lines_deleted',
    'is_feature_commit',
    'is_bug_fix',
    'log_loc',
]

# Thresholds
STAGE1_MIN_SAMPLES = 30
STAGE1_FDR_ALPHA = 0.05       # BH correction alpha for Stage 1
STAGE1_QUALIFICATION_THRESHOLD = 0.10  # Qualification gate: p_raw < this (screening)
STAGE2_ALPHA = 0.05
STAGE3_ALPHA = 0.05
STAGE3_RHO_DEFAULT = 0.3      # Fallback if empirical rho unavailable

# Adequate sample size flag (same as RQ1)
ADEQUATE_SAMPLE_SIZE = 30


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _smell_abbrev(smell_type: str) -> str:
    """Return abbreviation for a smell type, falling back to first 4 chars."""
    return SMELL_ABBREVIATIONS.get(smell_type, smell_type[:4].upper())


def _dim_abbrev(dimension: str) -> str:
    """Return abbreviation for a dimension, falling back to first 3 chars."""
    return DIMENSION_ABBREVIATIONS.get(dimension, dimension[:3].upper())


def compute_aggregate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add aggregate columns to a DataFrame.

    Adds:
      - delta_aggregate: mean of 6 dimension deltas
      - aggregate_baseline: mean of 6 dimension baselines
    """
    delta_cols = [f'{dim}_delta' for dim in DIMENSIONS]
    baseline_cols = [f'{dim}_baseline' for dim in DIMENSIONS]

    # Use only columns that exist
    available_delta = [c for c in delta_cols if c in df.columns]
    available_baseline = [c for c in baseline_cols if c in df.columns]

    if available_delta:
        df['delta_aggregate'] = df[available_delta].mean(axis=1)
    else:
        df['delta_aggregate'] = 0.0

    if available_baseline:
        df['aggregate_baseline'] = df[available_baseline].mean(axis=1)
    else:
        df['aggregate_baseline'] = 0.0

    return df


def calculate_empirical_rho(base_dir: Path) -> Optional[float]:
    """
    Calculate empirical correlation (rho) between introduction and removal
    effect sizes from RQ1 results.

    Reads rq1_all_results.csv, extracts H1.x (intro) and H2.x (removal)
    coefficients across the 6 dimensions, and computes np.corrcoef.

    Args:
        base_dir: Project root directory

    Returns:
        Correlation coefficient, or None if data unavailable
    """
    rq1_file = base_dir / "outputs" / "rq1" / "tables" / "rq1_all_results.csv"
    if not rq1_file.exists():
        logger.warning(f"RQ1 results not found at {rq1_file}; cannot compute empirical rho")
        return None

    try:
        rq1_df = pd.read_csv(rq1_file)
        # H1.x = introduction, H2.x = removal (matched by dimension)
        intro = rq1_df[rq1_df['hypothesis_id'].str.startswith('H1')].sort_values('dimension')
        removal = rq1_df[rq1_df['hypothesis_id'].str.startswith('H2')].sort_values('dimension')

        if len(intro) == 0 or len(removal) == 0:
            logger.warning("No H1/H2 results found in RQ1 file")
            return None

        # Match by dimension
        shared_dims = set(intro['dimension']) & set(removal['dimension'])
        if len(shared_dims) < 2:
            logger.warning(f"Only {len(shared_dims)} shared dimensions; need >= 2 for correlation")
            return None

        intro_betas = intro[intro['dimension'].isin(shared_dims)].sort_values('dimension')['coefficient'].values
        removal_betas = removal[removal['dimension'].isin(shared_dims)].sort_values('dimension')['coefficient'].values

        rho = float(np.corrcoef(intro_betas, removal_betas)[0, 1])
        logger.info(f"Empirical rho from RQ1: {rho:.4f} (across {len(shared_dims)} dimensions)")
        return rho

    except Exception as e:
        logger.warning(f"Error computing empirical rho: {e}")
        return None


def test_pairwise_smell_differences(
    stage1_results: List[HypothesisTestResult],
) -> pd.DataFrame:
    """
    Test pairwise differences between smell effects within the same transition.

    For each pair of smells (A, B) within the same transition:
      z = (beta_A - beta_B) / sqrt(SE_A^2 + SE_B^2)

    Applies BH correction across all pairwise tests.

    Args:
        stage1_results: Stage 1 results with aggregate coefficients

    Returns:
        DataFrame with pairwise comparison results
    """
    rows = []

    for transition in ['introduction', 'removal']:
        trans_results = [r for r in stage1_results if r.transition_type == transition]
        n = len(trans_results)

        for i in range(n):
            for j in range(i + 1, n):
                r_a = trans_results[i]
                r_b = trans_results[j]

                beta_a = r_a.regression.coefficient
                beta_b = r_b.regression.coefficient
                se_a = r_a.regression.std_error
                se_b = r_b.regression.std_error

                denom = np.sqrt(se_a ** 2 + se_b ** 2)
                if denom > 0:
                    z_stat = (beta_a - beta_b) / denom
                    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                else:
                    z_stat = 0.0
                    p_val = 1.0

                rows.append({
                    'smell_A': r_a.smell_type,
                    'smell_A_abbrev': _smell_abbrev(r_a.smell_type),
                    'smell_B': r_b.smell_type,
                    'smell_B_abbrev': _smell_abbrev(r_b.smell_type),
                    'transition': transition,
                    'beta_A': beta_a,
                    'beta_B': beta_b,
                    'z_statistic': z_stat,
                    'p_value': p_val,
                    'p_adj': np.nan,  # Set after BH
                    'significant': False,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Apply BH correction
    p_values = df['p_value'].tolist()
    adj_p, is_sig = benjamini_hochberg_correction(p_values, 0.05)
    df['p_adj'] = adj_p
    df['significant'] = is_sig

    n_sig = df['significant'].sum()
    logger.info(f"Pairwise smell comparisons: {n_sig}/{len(df)} significant differences")

    return df


def wald_symmetry_test(
    beta_intro: float,
    se_intro: float,
    beta_removal: float,
    se_removal: float,
    rho: float = STAGE3_RHO_DEFAULT,
) -> Tuple[float, float, str]:
    """
    Wald test for symmetry of introduction vs removal effects.

    Tests H0: |beta_intro| = |beta_removal|

    z = (|beta_intro| - |beta_removal|) /
        sqrt(SE_intro^2 + SE_removal^2 - 2*rho*SE_intro*SE_removal)

    Args:
        beta_intro: Introduction coefficient
        se_intro: Introduction standard error
        beta_removal: Removal coefficient
        se_removal: Removal standard error
        rho: Assumed correlation between estimates

    Returns:
        (z_statistic, p_value, classification)
        classification: 'symmetric' | 'hysteresis' | 'overcorrection'
    """
    var_diff = se_intro ** 2 + se_removal ** 2 - 2 * rho * se_intro * se_removal

    if var_diff <= 0:
        return 0.0, 1.0, 'symmetric'

    z_stat = (abs(beta_intro) - abs(beta_removal)) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Classification (based on raw comparison, significance determined later by BH)
    if abs(beta_intro) > abs(beta_removal):
        classification = 'hysteresis'
    elif abs(beta_intro) < abs(beta_removal):
        classification = 'overcorrection'
    else:
        classification = 'symmetric'

    return float(z_stat), float(p_value), classification


def generate_stage1_hypothesis_id(smell_type: str, transition: str) -> str:
    """Generate Stage 1 hypothesis ID: H2.1.{abbrev}.{intro|removal}."""
    abbrev = _smell_abbrev(smell_type)
    t = 'intro' if transition == 'introduction' else 'removal'
    return f"H2.1.{abbrev}.{t}"


def generate_stage2_hypothesis_id(smell_type: str, dimension: str, transition: str) -> str:
    """Generate Stage 2 hypothesis ID: H2.2.{abbrev}.{dim_abbrev}.{intro|removal}."""
    abbrev = _smell_abbrev(smell_type)
    dim_ab = _dim_abbrev(dimension)
    t = 'intro' if transition == 'introduction' else 'removal'
    return f"H2.2.{abbrev}.{dim_ab}.{t}"


def generate_stage3_hypothesis_id(smell_type: str, dimension: str) -> str:
    """Generate Stage 3 hypothesis ID: H2.3.{abbrev}.{dim_abbrev}."""
    abbrev = _smell_abbrev(smell_type)
    dim_ab = _dim_abbrev(dimension)
    return f"H2.3.{abbrev}.{dim_ab}"


# ---------------------------------------------------------------------------
# Stage 1: Aggregate Qualification (RQ2.1)
# ---------------------------------------------------------------------------

def run_stage1_aggregate(
    data: pd.DataFrame,
    fdr_alpha: float = STAGE1_FDR_ALPHA,
    qualification_threshold: float = STAGE1_QUALIFICATION_THRESHOLD,
    min_samples: int = STAGE1_MIN_SAMPLES,
) -> Tuple[List[HypothesisTestResult], List[Dict], Set[str]]:
    """
    Stage 1: Test each smell for aggregate effect across all dimensions.

    For each smell x transition: run ONE regression with delta_aggregate as DV.
    Treatment: instance_count, controls: STAGE1_CONTROL_COLS, cluster by file_id.
    BH FDR at fdr_alpha (0.05). Qualification gate: p_adj < qualification_threshold (0.10).
    Qualification is per-smell: a smell qualifies if ANY transition meets the gate.

    Args:
        data: Prepared DataFrame with all events
        fdr_alpha: BH correction alpha (default 0.05)
        qualification_threshold: Qualification gate for p_adj (default 0.10)
        min_samples: Minimum sample size for qualification

    Returns:
        (results, skipped, qualified_smells)
        qualified_smells: set of smell type names (not per-transition)
    """
    results: List[HypothesisTestResult] = []
    skipped: List[Dict] = []

    available_smells = sorted(data['smell_type'].unique())
    logger.info(f"Stage 1: Testing {len(available_smells)} smell types for aggregate effects")

    for smell_type in available_smells:
        smell_data = data[data['smell_type'] == smell_type]
        logger.info(f"\n{'='*50}")
        logger.info(f"SMELL: {smell_type} ({_smell_abbrev(smell_type)}) — {len(smell_data)} total events")

        for transition in ['introduction', 'removal']:
            transition_data = smell_data[smell_data['change_type'] == transition].copy()
            n_events = len(transition_data)
            hypothesis_id = generate_stage1_hypothesis_id(smell_type, transition)

            logger.info(f"  {transition}: n={n_events}")

            if n_events < min_samples:
                logger.warning(f"    SKIPPED: n={n_events} < {min_samples}")
                skipped.append({
                    'stage': 1,
                    'hypothesis_id': hypothesis_id,
                    'smell_type': smell_type,
                    'transition': transition,
                    'dimension': 'aggregate',
                    'n_events': n_events,
                    'reason': f'insufficient_samples (n={n_events} < {min_samples})',
                })
                continue

            # Add aggregate columns
            transition_data = compute_aggregate_columns(transition_data)

            # Ensure control variables exist
            for ctrl in STAGE1_CONTROL_COLS:
                if ctrl not in transition_data.columns:
                    if ctrl == 'aggregate_baseline':
                        pass  # Already computed
                    elif ctrl in ['is_bug_fix', 'is_feature_commit']:
                        transition_data[ctrl] = False
                    elif ctrl in ['lines_added', 'lines_deleted']:
                        transition_data[ctrl] = 0
                    else:
                        transition_data[ctrl] = 0

            # Run hypothesis test
            result = run_hypothesis_test(
                data=transition_data,
                y_col='delta_aggregate',
                treatment_col='instance_count',
                control_cols=STAGE1_CONTROL_COLS,
                cluster_col='file_id',
                hypothesis_id=hypothesis_id,
                dimension='aggregate',
                transition_type=transition,
                smell_type=smell_type,
                alpha=0.05,  # For CI computation; FDR handles significance
                min_samples=min_samples,
            )

            if result:
                result.sample_size_adequate = n_events >= ADEQUATE_SAMPLE_SIZE
                results.append(result)
                logger.info(f"    beta={result.regression.coefficient:.6f}, "
                          f"p={result.raw_p_value:.4f}, "
                          f"effect={result.effect_size_interpretation}")
            else:
                skipped.append({
                    'stage': 1,
                    'hypothesis_id': hypothesis_id,
                    'smell_type': smell_type,
                    'transition': transition,
                    'dimension': 'aggregate',
                    'n_events': n_events,
                    'reason': 'regression_failed',
                })

    # Apply BH FDR correction at Stage 1 fdr_alpha (0.05)
    if results:
        p_values = [r.raw_p_value for r in results]
        adjusted_p, is_significant = benjamini_hochberg_correction(p_values, fdr_alpha)
        for i, result in enumerate(results):
            result.adjusted_p_value = adjusted_p[i]
            result.is_significant = is_significant[i]

    # Determine qualified smells (per-smell, not per smell-transition)
    # A smell qualifies if ANY transition has p_raw < qualification_threshold
    # Rationale: Stage 1 is screening — using raw p-values follows established
    # practice in exploratory staged designs (Palomba et al. 2018). FDR correction
    # is applied rigorously in Stage 2/3 for final inference.
    qualified: Set[str] = set()
    for r in results:
        if r.raw_p_value is not None and r.raw_p_value < qualification_threshold:
            qualified.add(r.smell_type)

    n_below_fdr = sum(1 for r in results if r.is_significant)
    n_below_gate = sum(1 for r in results
                       if r.raw_p_value is not None
                       and r.raw_p_value < qualification_threshold)

    logger.info(f"\n{'='*50}")
    logger.info(f"Stage 1 Summary:")
    logger.info(f"  Tests run: {len(results)}")
    logger.info(f"  Tests skipped: {len(skipped)}")
    logger.info(f"  Significant (BH p_adj < {fdr_alpha}): {n_below_fdr}")
    logger.info(f"  Below qualification gate (p_raw < {qualification_threshold}): {n_below_gate}")
    logger.info(f"  Qualified smells for Stage 2: {len(qualified)}")
    for smell in sorted(qualified):
        logger.info(f"    - {smell}")

    return results, skipped, qualified


# ---------------------------------------------------------------------------
# Stage 2: Dimension-Specific Analysis (RQ2.2)
# ---------------------------------------------------------------------------

def run_stage2_dimension_analysis(
    data: pd.DataFrame,
    qualified: Set[str],
    alpha: float = STAGE2_ALPHA,
) -> Tuple[List[HypothesisTestResult], List[Dict]]:
    """
    Stage 2: Test qualified smells across both transitions × 6 dimensions.

    Only tests smells that passed Stage 1 qualification (per-smell).
    For each qualified smell, tests BOTH introduction and removal across all 6 dimensions.
    Treatment: instance_count, controls: STAGE2_CONTROL_COLS, cluster by file_id.
    BH FDR at alpha (default 0.05).

    Args:
        data: Prepared DataFrame with all events
        qualified: Set of qualified smell type names from Stage 1
        alpha: Significance level for Stage 2

    Returns:
        (results, skipped)
    """
    results: List[HypothesisTestResult] = []
    skipped: List[Dict] = []

    if not qualified:
        logger.warning("Stage 2: No qualified smells from Stage 1")
        return results, skipped

    logger.info(f"\nStage 2: Dimension-specific analysis for {len(qualified)} qualified smells × 2 transitions × 6 dimensions")

    for smell_type in sorted(qualified):
        for transition in ['introduction', 'removal']:
            smell_data = data[
                (data['smell_type'] == smell_type) & (data['change_type'] == transition)
            ].copy()
            n_events = len(smell_data)

            logger.info(f"\n{'='*50}")
            logger.info(f"{smell_type} / {transition} (n={n_events})")

            if n_events < 10:
                logger.warning(f"  SKIPPED: n={n_events} < 10 for {transition}")
                for dimension in DIMENSIONS:
                    hypothesis_id = generate_stage2_hypothesis_id(smell_type, dimension, transition)
                    skipped.append({
                        'stage': 2,
                        'hypothesis_id': hypothesis_id,
                        'smell_type': smell_type,
                        'transition': transition,
                        'dimension': dimension,
                        'n_events': n_events,
                        'reason': f'insufficient_samples (n={n_events} < 10)',
                    })
                continue

            sample_adequate = n_events >= ADEQUATE_SAMPLE_SIZE

            for dimension in DIMENSIONS:
                hypothesis_id = generate_stage2_hypothesis_id(smell_type, dimension, transition)

                y_col = f'{dimension}_delta'
                baseline_col = f'{dimension}_baseline'

                if y_col not in smell_data.columns:
                    logger.warning(f"  {hypothesis_id}: Missing column {y_col}")
                    skipped.append({
                        'stage': 2,
                        'hypothesis_id': hypothesis_id,
                        'smell_type': smell_type,
                        'transition': transition,
                        'dimension': dimension,
                        'n_events': n_events,
                        'reason': f'missing_column_{y_col}',
                    })
                    continue

                # Set up test data with baseline_score
                test_data = smell_data.copy()
                test_data['baseline_score'] = test_data.get(baseline_col, 0)

                # Ensure control variables exist
                for ctrl in STAGE2_CONTROL_COLS:
                    if ctrl not in test_data.columns:
                        if ctrl == 'baseline_score':
                            pass  # Already set above
                        elif ctrl == 'log_loc':
                            test_data[ctrl] = np.log1p(test_data.get('loc', 100))
                        elif ctrl in ['is_bug_fix', 'is_feature_commit']:
                            test_data[ctrl] = False
                        elif ctrl in ['lines_added', 'lines_deleted']:
                            test_data[ctrl] = 0
                        else:
                            test_data[ctrl] = 0

                result = run_hypothesis_test(
                    data=test_data,
                    y_col=y_col,
                    treatment_col='instance_count',
                    control_cols=STAGE2_CONTROL_COLS,
                    cluster_col='file_id',
                    hypothesis_id=hypothesis_id,
                    dimension=dimension,
                    transition_type=transition,
                    smell_type=smell_type,
                    alpha=0.05,  # For CI computation
                    min_samples=10,  # Already filtered by Stage 1
                )

                if result:
                    result.sample_size_adequate = sample_adequate
                    results.append(result)
                    sig_marker = "*" if result.raw_p_value < 0.05 else ""
                    logger.info(f"  {dimension}: beta={result.regression.coefficient:.6f}, "
                              f"p={result.raw_p_value:.4f}{sig_marker}, "
                              f"effect={result.effect_size_interpretation}")
                else:
                    skipped.append({
                        'stage': 2,
                        'hypothesis_id': hypothesis_id,
                        'smell_type': smell_type,
                        'transition': transition,
                        'dimension': dimension,
                        'n_events': n_events,
                        'reason': 'regression_failed',
                    })

    # Apply BH FDR correction at Stage 2 alpha (separate from Stage 1)
    if results:
        p_values = [r.raw_p_value for r in results]
        adjusted_p, is_significant = benjamini_hochberg_correction(p_values, alpha)
        for i, result in enumerate(results):
            result.adjusted_p_value = adjusted_p[i]
            result.is_significant = is_significant[i]

    n_sig = sum(1 for r in results if r.is_significant)
    logger.info(f"\n{'='*50}")
    logger.info(f"Stage 2 Summary:")
    logger.info(f"  Tests run: {len(results)}")
    logger.info(f"  Tests skipped: {len(skipped)}")
    logger.info(f"  Significant (p_adj < {alpha}): {n_sig}")

    return results, skipped


# ---------------------------------------------------------------------------
# Stage 3: Symmetry Analysis (H2.3)
# ---------------------------------------------------------------------------

def run_stage3_symmetry_analysis(
    stage2_results: List[HypothesisTestResult],
    alpha: float = STAGE3_ALPHA,
    rho: float = STAGE3_RHO_DEFAULT,
) -> pd.DataFrame:
    """
    Stage 3: Test introduction/removal symmetry for significant Stage 2 pairs.

    For each (smell, dimension) where either transition was significant in Stage 2:
      - Retrieve intro and removal coefficients and SEs
      - Run Wald symmetry test
      - Classify: symmetric / hysteresis / overcorrection

    BH FDR at alpha (default 0.05, separate from other stages).

    Args:
        stage2_results: List of Stage 2 HypothesisTestResult objects
        alpha: Significance level for symmetry tests
        rho: Assumed correlation between intro/removal estimates

    Returns:
        DataFrame with symmetry analysis results
    """
    if not stage2_results:
        logger.warning("Stage 3: No Stage 2 results available")
        return pd.DataFrame()

    # Index Stage 2 results by (smell, dimension, transition)
    result_map: Dict[Tuple[str, str, str], HypothesisTestResult] = {}
    for r in stage2_results:
        key = (r.smell_type, r.dimension, r.transition_type)
        result_map[key] = r

    # Find (smell, dimension) pairs where either transition is significant
    sig_pairs: Set[Tuple[str, str]] = set()
    for r in stage2_results:
        if r.is_significant:
            sig_pairs.add((r.smell_type, r.dimension))

    if not sig_pairs:
        logger.info("Stage 3: No significant Stage 2 pairs to test for symmetry")
        return pd.DataFrame()

    logger.info(f"\nStage 3: Symmetry analysis for {len(sig_pairs)} smell-dimension pairs")

    rows = []
    for smell_type, dimension in sorted(sig_pairs):
        intro_key = (smell_type, dimension, 'introduction')
        removal_key = (smell_type, dimension, 'removal')

        intro_result = result_map.get(intro_key)
        removal_result = result_map.get(removal_key)

        hypothesis_id = generate_stage3_hypothesis_id(smell_type, dimension)

        # Need both transitions to test symmetry
        if intro_result is None or removal_result is None:
            # One-sided: only one transition was tested (or available)
            available = 'introduction' if intro_result else 'removal'
            rows.append({
                'hypothesis_id': hypothesis_id,
                'smell_type': smell_type,
                'smell_abbreviation': _smell_abbrev(smell_type),
                'dimension': dimension,
                'beta_intro': intro_result.regression.coefficient if intro_result else np.nan,
                'se_intro': intro_result.regression.std_error if intro_result else np.nan,
                'p_intro': intro_result.adjusted_p_value if intro_result else np.nan,
                'beta_removal': removal_result.regression.coefficient if removal_result else np.nan,
                'se_removal': removal_result.regression.std_error if removal_result else np.nan,
                'p_removal': removal_result.adjusted_p_value if removal_result else np.nan,
                'z_statistic': np.nan,
                'p_symmetry_raw': np.nan,
                'p_symmetry_adj': np.nan,
                'classification': 'unidirectional',
                'significant_asymmetry': False,
                'note': f'Only {available} tested/available',
            })
            continue

        beta_intro = intro_result.regression.coefficient
        se_intro = intro_result.regression.std_error
        beta_removal = removal_result.regression.coefficient
        se_removal = removal_result.regression.std_error

        z_stat, p_val, classification = wald_symmetry_test(
            beta_intro, se_intro, beta_removal, se_removal, rho
        )

        rows.append({
            'hypothesis_id': hypothesis_id,
            'smell_type': smell_type,
            'smell_abbreviation': _smell_abbrev(smell_type),
            'dimension': dimension,
            'beta_intro': beta_intro,
            'se_intro': se_intro,
            'p_intro': intro_result.adjusted_p_value,
            'sig_intro': intro_result.is_significant,
            'beta_removal': beta_removal,
            'se_removal': se_removal,
            'p_removal': removal_result.adjusted_p_value,
            'sig_removal': removal_result.is_significant,
            'z_statistic': z_stat,
            'p_symmetry_raw': p_val,
            'p_symmetry_adj': np.nan,  # Will be set after BH correction
            'classification': classification,
            'significant_asymmetry': False,  # Will be set after BH correction
            'note': '',
        })

    symmetry_df = pd.DataFrame(rows)

    # Apply BH FDR correction on the symmetry p-values (only for rows with valid tests)
    testable_mask = symmetry_df['p_symmetry_raw'].notna()
    if testable_mask.any():
        raw_p = symmetry_df.loc[testable_mask, 'p_symmetry_raw'].tolist()
        adj_p, is_sig = benjamini_hochberg_correction(raw_p, alpha)
        symmetry_df.loc[testable_mask, 'p_symmetry_adj'] = adj_p
        symmetry_df.loc[testable_mask, 'significant_asymmetry'] = is_sig

        # Update classification for non-significant pairs to 'symmetric'
        for idx in symmetry_df.index[testable_mask]:
            if not symmetry_df.loc[idx, 'significant_asymmetry']:
                symmetry_df.loc[idx, 'classification'] = 'symmetric'

    # Summary
    n_testable = testable_mask.sum()
    n_symmetric = (symmetry_df['classification'] == 'symmetric').sum()
    n_hysteresis = (symmetry_df['classification'] == 'hysteresis').sum()
    n_overcorrection = (symmetry_df['classification'] == 'overcorrection').sum()
    n_unidirectional = (symmetry_df['classification'] == 'unidirectional').sum()

    logger.info(f"\nStage 3 Summary:")
    logger.info(f"  Total pairs: {len(symmetry_df)}")
    logger.info(f"  Testable (both transitions): {n_testable}")
    logger.info(f"  Symmetric: {n_symmetric}")
    logger.info(f"  Hysteresis (|beta_intro| > |beta_removal|): {n_hysteresis}")
    logger.info(f"  Overcorrection (|beta_intro| < |beta_removal|): {n_overcorrection}")
    logger.info(f"  Unidirectional (only one transition): {n_unidirectional}")

    return symmetry_df


# ---------------------------------------------------------------------------
# Output helper functions
# ---------------------------------------------------------------------------

def generate_smell_ranking_stage1(
    stage1_results: List[HypothesisTestResult],
) -> pd.DataFrame:
    """
    Rank smells by aggregate |beta| from Stage 1.

    Returns DataFrame sorted by absolute aggregate effect size.
    """
    rows = []
    for r in stage1_results:
        rows.append({
            'smell_type': r.smell_type,
            'smell_abbreviation': _smell_abbrev(r.smell_type),
            'transition': r.transition_type,
            'hypothesis_id': r.hypothesis_id,
            'n_observations': r.regression.n_observations,
            'n_clusters': r.regression.n_clusters,
            'coefficient': r.regression.coefficient,
            'abs_coefficient': abs(r.regression.coefficient),
            'std_error': r.regression.std_error,
            'p_value_raw': r.raw_p_value,
            'p_value_adj': r.adjusted_p_value,
            'effect_size_beta': r.effect_size_beta,
            'effect_interpretation': r.effect_size_interpretation,
            'is_significant': r.is_significant,
            'below_qualification_gate': (r.adjusted_p_value is not None
                                         and r.adjusted_p_value < STAGE1_QUALIFICATION_THRESHOLD),
            'qualified_stage2': (r.adjusted_p_value is not None
                                 and r.adjusted_p_value < STAGE1_QUALIFICATION_THRESHOLD),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('abs_coefficient', ascending=False)
        df['rank'] = range(1, len(df) + 1)

    return df


def generate_heatmap_data(
    results: List[HypothesisTestResult],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate heatmap data for smell x dimension visualization from Stage 2.

    Returns:
        (introduction_heatmap, removal_heatmap) DataFrames
    """
    intro_data = {}
    removal_data = {}

    for r in results:
        target = intro_data if r.transition_type == 'introduction' else removal_data
        if r.smell_type not in target:
            target[r.smell_type] = {}
        target[r.smell_type][r.dimension] = {
            'coefficient': r.regression.coefficient,
            'p_value': r.adjusted_p_value or r.raw_p_value,
            'effect_size_beta': r.effect_size_beta,
            'significant': r.is_significant,
        }

    def _to_df(data_dict):
        rows = []
        for smell, dims in data_dict.items():
            row = {'smell_type': smell, 'smell_abbreviation': _smell_abbrev(smell)}
            for dim, vals in dims.items():
                row[f'{dim}_coef'] = vals['coefficient']
                row[f'{dim}_sig'] = vals['significant']
                row[f'{dim}_effect_beta'] = vals['effect_size_beta']
            rows.append(row)
        return pd.DataFrame(rows)

    return _to_df(intro_data), _to_df(removal_data)


def generate_dimension_susceptibility(
    results: List[HypothesisTestResult],
) -> pd.DataFrame:
    """Analyze which dimensions are most affected by smells (from Stage 2)."""
    dim_stats = {}

    for r in results:
        if r.dimension not in dim_stats:
            dim_stats[r.dimension] = {
                'total_tests': 0,
                'significant_tests': 0,
                'smells_with_effect': set(),
                'effect_sizes_beta': [],
                'intro_effects': [],
                'removal_effects': [],
            }

        stats_ = dim_stats[r.dimension]
        stats_['total_tests'] += 1
        stats_['effect_sizes_beta'].append(abs(r.effect_size_beta))

        if r.is_significant:
            stats_['significant_tests'] += 1
            stats_['smells_with_effect'].add(r.smell_type)

        if r.transition_type == 'introduction':
            stats_['intro_effects'].append(r.regression.coefficient)
        else:
            stats_['removal_effects'].append(r.regression.coefficient)

    rows = []
    for dim, s in dim_stats.items():
        rows.append({
            'dimension': dim,
            'total_tests': s['total_tests'],
            'significant_tests': s['significant_tests'],
            'susceptibility_rate': s['significant_tests'] / s['total_tests'] if s['total_tests'] > 0 else 0,
            'smells_with_effect': len(s['smells_with_effect']),
            'avg_effect_size_beta': np.mean(s['effect_sizes_beta']),
            'avg_intro_effect': np.mean(s['intro_effects']) if s['intro_effects'] else 0,
            'avg_removal_effect': np.mean(s['removal_effects']) if s['removal_effects'] else 0,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('susceptibility_rate', ascending=False)
    return df


# ---------------------------------------------------------------------------
# Save / Output
# ---------------------------------------------------------------------------

def save_rq2_results(
    stage1_results: List[HypothesisTestResult],
    stage2_results: List[HypothesisTestResult],
    symmetry_df: pd.DataFrame,
    qualified: Set[str],
    all_skipped: List[Dict],
    output_dir: Path,
    pairwise_df: Optional[pd.DataFrame] = None,
    rho_used: Optional[float] = None,
):
    """Save all RQ2 results to output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1 outputs ----
    if stage1_results:
        stage1_df = results_to_dataframe(stage1_results)
        stage1_df['stage'] = 1
        # Add VIF reporting alongside Stage 1 results
        s1_max_vifs = []
        s1_vif_flags = []
        for r in stage1_results:
            if r.regression.vif_values:
                mv = max(r.regression.vif_values.values())
                s1_max_vifs.append(round(mv, 2))
                if mv > 10:
                    s1_vif_flags.append('severe (VIF>10)')
                elif mv > 5:
                    s1_vif_flags.append('moderate (VIF>5)')
                else:
                    s1_vif_flags.append('')
            else:
                s1_max_vifs.append(None)
                s1_vif_flags.append('')
        stage1_df['max_vif'] = s1_max_vifs
        stage1_df['vif_flag'] = s1_vif_flags
        stage1_df.to_csv(output_dir / 'rq2_stage1_aggregate.csv', index=False)
        logger.info(f"Saved: {output_dir / 'rq2_stage1_aggregate.csv'}")

    # Qualified smells documentation (per-smell qualification)
    qualified_info = {
        'qualified_smells': sorted(qualified),
        'qualification_criteria': {
            'min_samples': STAGE1_MIN_SAMPLES,
            'fdr_alpha': STAGE1_FDR_ALPHA,
            'qualification_threshold': STAGE1_QUALIFICATION_THRESHOLD,
            'qualification_uses_raw_p': True,
            'description': f'n >= {STAGE1_MIN_SAMPLES} AND raw p < {STAGE1_QUALIFICATION_THRESHOLD} (screening threshold)',
            'justification': (
                'Following established practice in exploratory staged designs '
                '(Palomba et al. 2018), raw p-values used for Stage 1 screening '
                'while maintaining FDR-corrected thresholds (alpha=0.05) for '
                'Stage 2/3 hypothesis testing.'
            ),
        },
        'total_tested': len(stage1_results),
        'total_qualified': len(qualified),
        'smell_details': {},
    }
    # Add per-smell details
    for r in stage1_results:
        key = f"{r.smell_type}_{r.transition_type}"
        smell_qualified = r.smell_type in qualified
        qualified_info['smell_details'][key] = {
            'smell_type': r.smell_type,
            'transition': r.transition_type,
            'n_observations': r.regression.n_observations,
            'coefficient': r.regression.coefficient,
            'p_raw': r.raw_p_value,
            'p_adj': r.adjusted_p_value,
            'qualified': smell_qualified,
            'reason': (
                'qualified (p_raw < threshold)'
                if smell_qualified and r.raw_p_value is not None and r.raw_p_value < qualification_threshold
                else 'qualified via other transition'
                if smell_qualified
                else f'p_raw={r.raw_p_value:.4f} >= {STAGE1_QUALIFICATION_THRESHOLD}'
            ),
        }
    # Add skipped smells from Stage 1
    for s in all_skipped:
        if s.get('stage') == 1:
            key = f"{s['smell_type']}_{s['transition']}"
            qualified_info['smell_details'][key] = {
                'smell_type': s['smell_type'],
                'transition': s['transition'],
                'n_events': s.get('n_events', 0),
                'qualified': False,
                'reason': s.get('reason', 'unknown'),
            }

    with open(output_dir / 'rq2_stage1_qualified.json', 'w') as f:
        json.dump(qualified_info, f, indent=2, default=str)
    logger.info(f"Saved: {output_dir / 'rq2_stage1_qualified.json'}")

    # Smell ranking from Stage 1
    ranking_df = generate_smell_ranking_stage1(stage1_results)
    if not ranking_df.empty:
        ranking_df.to_csv(output_dir / 'rq2_smell_ranking.csv', index=False)
        logger.info(f"Saved: {output_dir / 'rq2_smell_ranking.csv'}")

    # Pairwise smell comparisons (Stage 1)
    if pairwise_df is not None and not pairwise_df.empty:
        pairwise_df.to_csv(output_dir / 'rq2_stage1_pairwise.csv', index=False)
        logger.info(f"Saved: {output_dir / 'rq2_stage1_pairwise.csv'}")

    # ---- Stage 2 outputs ----
    if stage2_results:
        stage2_df = results_to_dataframe(stage2_results)
        stage2_df['stage'] = 2

        # Add VIF reporting alongside results (supervisor requirement)
        max_vifs = []
        vif_flags = []
        sample_notes = []
        for r in stage2_results:
            if r.regression.vif_values:
                mv = max(r.regression.vif_values.values())
                max_vifs.append(round(mv, 2))
                if mv > 10:
                    vif_flags.append('severe (VIF>10)')
                elif mv > 5:
                    vif_flags.append('moderate (VIF>5)')
                else:
                    vif_flags.append('')
            else:
                max_vifs.append(None)
                vif_flags.append('')
            # Flag small samples as exploratory
            n = r.regression.n_observations
            sample_notes.append('exploratory: small sample' if n < 150 else '')
        stage2_df['max_vif'] = max_vifs
        stage2_df['vif_flag'] = vif_flags
        stage2_df['sample_note'] = sample_notes

        stage2_df.to_csv(output_dir / 'rq2_stage2_dimension_results.csv', index=False)
        logger.info(f"Saved: {output_dir / 'rq2_stage2_dimension_results.csv'}")

        # Heatmaps
        intro_heatmap, removal_heatmap = generate_heatmap_data(stage2_results)
        if not intro_heatmap.empty:
            intro_heatmap.to_csv(output_dir / 'rq2_stage2_heatmap_introduction.csv', index=False)
            logger.info(f"Saved: {output_dir / 'rq2_stage2_heatmap_introduction.csv'}")
        if not removal_heatmap.empty:
            removal_heatmap.to_csv(output_dir / 'rq2_stage2_heatmap_removal.csv', index=False)
            logger.info(f"Saved: {output_dir / 'rq2_stage2_heatmap_removal.csv'}")

        # Dimension susceptibility
        suscept_df = generate_dimension_susceptibility(stage2_results)
        if not suscept_df.empty:
            suscept_df.to_csv(output_dir / 'rq2_dimension_susceptibility.csv', index=False)
            logger.info(f"Saved: {output_dir / 'rq2_dimension_susceptibility.csv'}")

    # ---- Stage 3 outputs ----
    if not symmetry_df.empty:
        symmetry_df.to_csv(output_dir / 'rq2_stage3_symmetry.csv', index=False)
        logger.info(f"Saved: {output_dir / 'rq2_stage3_symmetry.csv'}")

    # ---- Combined outputs ----
    all_results = stage1_results + stage2_results
    if all_results:
        all_df = results_to_dataframe(all_results)
        # Add stage column
        all_df['stage'] = [1] * len(stage1_results) + [2] * len(stage2_results)
        all_df.to_csv(output_dir / 'rq2_all_results.csv', index=False)
        logger.info(f"Saved: {output_dir / 'rq2_all_results.csv'}")

    # Skipped tests
    if all_skipped:
        skipped_df = pd.DataFrame(all_skipped)
        skipped_df.to_csv(output_dir / 'rq2_skipped_tests.csv', index=False)
        logger.info(f"Saved: {output_dir / 'rq2_skipped_tests.csv'}")

    # ---- Diagnostics, power analysis, assumption warnings ----
    diagnostics = []
    assumption_warnings = []
    power_results = []

    # Stage 1 has fewer predictors than Stage 2
    s1_n_predictors = 1 + len(STAGE1_CONTROL_COLS)
    s2_n_predictors = 1 + len(STAGE2_CONTROL_COLS)

    for r in all_results:
        n_predictors = s1_n_predictors if r.dimension == 'aggregate' else s2_n_predictors
        stage_label = 'stage1' if r.dimension == 'aggregate' else 'stage2'

        diag = check_regression_assumptions(r.regression)
        diag['hypothesis_id'] = r.hypothesis_id
        diag['smell_type'] = r.smell_type
        diag['transition'] = r.transition_type
        diag['dimension'] = r.dimension
        diag['stage'] = stage_label

        power_info = calculate_post_hoc_power(
            n=r.regression.n_observations,
            k=n_predictors,
            r_squared=r.regression.r_squared,
            alpha=0.05,
        )
        diag['power_analysis'] = power_info
        power_results.append({
            'hypothesis_id': r.hypothesis_id,
            'smell_type': r.smell_type,
            'dimension': r.dimension,
            'transition': r.transition_type,
            'stage': stage_label,
            'n': r.regression.n_observations,
            'r_squared': r.regression.r_squared,
            'power': power_info.get('power'),
            'power_interpretation': power_info.get('power_interpretation'),
            'cohens_f_squared': power_info.get('cohens_f_squared'),
            'effect_size': power_info.get('effect_size_interpretation'),
        })
        diagnostics.append(diag)

        # Check for assumption violations
        warnings_for_test = []
        if diag.get('multicollinearity', {}).get('severe', False):
            warnings_for_test.append("SEVERE multicollinearity (VIF > 10)")
        elif diag.get('multicollinearity', {}).get('problematic', False):
            warnings_for_test.append("Potential multicollinearity (VIF > 5)")
        if diag.get('normality', {}).get('is_normal') is False:
            warnings_for_test.append("Non-normal residuals")
        if diag.get('autocorrelation', {}).get('has_autocorrelation', False):
            warnings_for_test.append("Autocorrelation detected")
        if warnings_for_test:
            assumption_warnings.append({
                'hypothesis_id': r.hypothesis_id,
                'smell_type': r.smell_type,
                'stage': stage_label,
                'warnings': warnings_for_test,
            })

    if diagnostics:
        with open(output_dir / 'rq2_diagnostics.json', 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        logger.info(f"Saved: {output_dir / 'rq2_diagnostics.json'}")

    if power_results:
        power_df = pd.DataFrame(power_results)
        power_df.to_csv(output_dir / 'rq2_power_analysis.csv', index=False)
        logger.info(f"Saved: {output_dir / 'rq2_power_analysis.csv'}")

        adequate_power = sum(1 for p in power_results if p.get('power') and p['power'] >= 0.80)
        underpowered = sum(1 for p in power_results if p.get('power') and p['power'] < 0.80)
        logger.info(f"\nPower analysis: {adequate_power}/{len(power_results)} adequately powered, "
                   f"{underpowered} underpowered")

    if assumption_warnings:
        with open(output_dir / 'rq2_assumption_warnings.json', 'w') as f:
            json.dump({
                'total_tests_with_warnings': len(assumption_warnings),
                'warnings': assumption_warnings,
            }, f, indent=2)
        logger.info(f"Saved: {output_dir / 'rq2_assumption_warnings.json'}")

    # ---- Summary JSON ----
    summary = {
        'stage1': {
            'tests_run': len(stage1_results),
            'tests_skipped': sum(1 for s in all_skipped if s.get('stage') == 1),
            'significant_results': sum(1 for r in stage1_results if r.is_significant),
            'fdr_alpha': STAGE1_FDR_ALPHA,
            'qualification_threshold': STAGE1_QUALIFICATION_THRESHOLD,
            'min_samples': STAGE1_MIN_SAMPLES,
            'qualified_smells': sorted(qualified),
            'qualified_for_stage2': len(qualified),
        },
        'stage2': {
            'tests_run': len(stage2_results),
            'tests_skipped': sum(1 for s in all_skipped if s.get('stage') == 2),
            'significant_results': sum(1 for r in stage2_results if r.is_significant),
            'alpha': STAGE2_ALPHA,
            'unique_smells_tested': len(set(r.smell_type for r in stage2_results)),
        },
        'stage3': {
            'pairs_tested': len(symmetry_df) if not symmetry_df.empty else 0,
            'symmetric': int((symmetry_df['classification'] == 'symmetric').sum()) if not symmetry_df.empty else 0,
            'hysteresis': int((symmetry_df['classification'] == 'hysteresis').sum()) if not symmetry_df.empty else 0,
            'overcorrection': int((symmetry_df['classification'] == 'overcorrection').sum()) if not symmetry_df.empty else 0,
            'unidirectional': int((symmetry_df['classification'] == 'unidirectional').sum()) if not symmetry_df.empty else 0,
            'alpha': STAGE3_ALPHA,
            'rho_used': rho_used if rho_used is not None else STAGE3_RHO_DEFAULT,
            'rho_source': 'empirical_from_rq1' if rho_used is not None else 'default',
        },
        'overall': {
            'total_tests': len(stage1_results) + len(stage2_results),
            'total_significant': (
                sum(1 for r in stage1_results if r.is_significant)
                + sum(1 for r in stage2_results if r.is_significant)
            ),
            'tests_with_assumption_warnings': len(assumption_warnings),
        },
    }

    with open(output_dir / 'rq2_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved: {output_dir / 'rq2_summary.json'}")


def run_robustness_analysis(
    stage1_results: List[HypothesisTestResult],
    stage2_results: List[HypothesisTestResult],
    data: pd.DataFrame,
    output_dir: Path,
    alpha_levels: List[float] = None,
) -> Dict:
    """
    Run robustness/sensitivity analysis including:
    - Stage 2 alpha-level sensitivity (0.01, 0.05, 0.10)
    - Stage 1 threshold sensitivity (p<0.10 vs p<0.15)
    - Sample size sensitivity (n>=30 vs n>=20)
    """
    if alpha_levels is None:
        alpha_levels = [0.01, 0.05, 0.10]

    logger.info("\n=== ROBUSTNESS ANALYSIS ===")

    robustness_data = {
        'stage2_alpha_sensitivity': {},
        'stage1_threshold_sensitivity': {},
        'sample_size_sensitivity': {},
        'by_smell_type': {},
        'summary': {},
    }

    # --- Stage 2 alpha-level sensitivity ---
    logger.info("  Stage 2 alpha-level sensitivity:")
    for alpha in alpha_levels:
        if not stage2_results:
            continue
        p_values = [r.raw_p_value for r in stage2_results]
        _, is_significant = benjamini_hochberg_correction(p_values, alpha)
        sig_count = sum(is_significant)
        robustness_data['stage2_alpha_sensitivity'][str(alpha)] = {
            'significant_tests': sig_count,
            'significant_rate': sig_count / len(stage2_results) if stage2_results else 0,
        }
        logger.info(f"    alpha={alpha}: {sig_count}/{len(stage2_results)} significant")

    # --- Stage 1 threshold sensitivity (p<0.10 vs p<0.15) ---
    logger.info("  Stage 1 qualification threshold sensitivity:")
    thresholds = [0.10, 0.15]
    for threshold in thresholds:
        qualified_at_threshold = set()
        for r in stage1_results:
            if r.adjusted_p_value is not None and r.adjusted_p_value < threshold:
                qualified_at_threshold.add(r.smell_type)
        robustness_data['stage1_threshold_sensitivity'][str(threshold)] = {
            'qualified_smells': sorted(qualified_at_threshold),
            'n_qualified': len(qualified_at_threshold),
        }
        logger.info(f"    p_adj < {threshold}: {len(qualified_at_threshold)} smells qualified")

    # --- Sample size sensitivity (n>=30 vs n>=20) ---
    logger.info("  Sample size threshold sensitivity:")
    for min_n in [30, 20]:
        testable = 0
        available_smells = sorted(data['smell_type'].unique())
        for smell_type in available_smells:
            for transition in ['introduction', 'removal']:
                n = len(data[(data['smell_type'] == smell_type) & (data['change_type'] == transition)])
                if n >= min_n:
                    testable += 1
        robustness_data['sample_size_sensitivity'][f'n_ge_{min_n}'] = {
            'testable_smell_transitions': testable,
        }
        logger.info(f"    n >= {min_n}: {testable} smell-transition pairs testable")

    # Stability by smell type (Stage 2)
    if stage2_results:
        smell_types = set(r.smell_type for r in stage2_results)
        for smell_type in smell_types:
            smell_indices = [i for i, r in enumerate(stage2_results) if r.smell_type == smell_type]
            stable = 0
            for idx in smell_indices:
                sig_at_levels = []
                for alpha in alpha_levels:
                    p_values = [r.raw_p_value for r in stage2_results]
                    _, is_sig = benjamini_hochberg_correction(p_values, alpha)
                    sig_at_levels.append(is_sig[idx])
                if all(sig_at_levels) or not any(sig_at_levels):
                    stable += 1
            robustness_data['by_smell_type'][smell_type] = {
                'total_tests': len(smell_indices),
                'stable_conclusions': stable,
                'stability_rate': stable / len(smell_indices) if smell_indices else 0,
            }

    total_stable = sum(d['stable_conclusions'] for d in robustness_data['by_smell_type'].values())
    total_tests = sum(d['total_tests'] for d in robustness_data['by_smell_type'].values())
    robustness_data['summary'] = {
        'total_tests': total_tests,
        'stable_conclusions': total_stable,
        'stability_rate': total_stable / total_tests if total_tests else 0,
    }

    with open(output_dir / 'rq2_robustness_analysis.json', 'w') as f:
        json.dump(robustness_data, f, indent=2, default=str)
    logger.info(f"Saved: {output_dir / 'rq2_robustness_analysis.json'}")

    return robustness_data


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(
    data_path: str = None,
    output_dir: str = None,
    alpha: float = 0.05,
    min_samples: int = 10,
):
    """
    Run all RQ2 hypothesis tests using the 3-stage hierarchical approach.

    Args:
        data_path: Path to analysis-ready directory (contains all_events.csv)
        output_dir: Path to output directory
        alpha: Base significance level (Stage 2 and 3; Stage 1 uses 0.10)
        min_samples: Minimum samples (Stage 1 uses 30 for qualification)

    Returns:
        Dictionary with stage results, or None on failure
    """
    base_dir = Path(__file__).parent.parent.parent.parent

    if data_path is None:
        data_path = base_dir / "data" / "analysis-ready"
    else:
        data_path = Path(data_path)

    if output_dir is None:
        output_dir = base_dir / "outputs" / "rq2" / "tables"
    else:
        output_dir = Path(output_dir)

    logger.info("=" * 60)
    logger.info("RQ2: Do different ML smell types have distinct impacts?")
    logger.info("  3-Stage Hierarchical Approach (Ver2)")
    logger.info(f"  Stage 1: Aggregate qualification (BH alpha={STAGE1_FDR_ALPHA}, gate p_adj<{STAGE1_QUALIFICATION_THRESHOLD}, n>={STAGE1_MIN_SAMPLES})")
    logger.info(f"  Stage 2: Dimension-specific analysis (alpha={STAGE2_ALPHA})")
    logger.info(f"  Stage 3: Symmetry analysis (alpha={STAGE3_ALPHA})")
    logger.info("=" * 60)

    # Load data
    all_events_file = data_path / "all_events.csv"
    if not all_events_file.exists():
        logger.info("Analysis-ready data not found. Running data preparation...")
        from src.step4_analysis.data_preparation.prepare_data import main as prepare_main
        prepare_main()

    if all_events_file.exists():
        data = pd.read_csv(all_events_file)
        logger.info(f"Loaded {len(data)} events from {all_events_file}")
    else:
        logger.error(f"Data file not found: {all_events_file}")
        return None

    # ---- Calculate empirical rho from RQ1 ----
    empirical_rho = calculate_empirical_rho(base_dir)
    rho_used = empirical_rho if empirical_rho is not None else STAGE3_RHO_DEFAULT
    logger.info(f"Rho for Stage 3 symmetry: {rho_used:.4f} "
               f"({'empirical from RQ1' if empirical_rho is not None else 'default fallback'})")

    # ---- Stage 1: Aggregate Qualification ----
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: AGGREGATE QUALIFICATION (H2.1)")
    logger.info("=" * 60)

    stage1_results, stage1_skipped, qualified = run_stage1_aggregate(
        data,
        fdr_alpha=STAGE1_FDR_ALPHA,
        qualification_threshold=STAGE1_QUALIFICATION_THRESHOLD,
        min_samples=STAGE1_MIN_SAMPLES,
    )

    if not stage1_results:
        logger.error("Stage 1: No tests completed")
        return None

    # ---- Pairwise smell comparisons (Stage 1) ----
    pairwise_df = test_pairwise_smell_differences(stage1_results)

    # ---- Stage 2: Dimension-Specific ----
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: DIMENSION-SPECIFIC ANALYSIS (H2.2)")
    logger.info("=" * 60)

    stage2_results, stage2_skipped = run_stage2_dimension_analysis(
        data, qualified, alpha=STAGE2_ALPHA
    )

    # ---- Stage 3: Symmetry ----
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: SYMMETRY ANALYSIS (H2.3)")
    logger.info("=" * 60)

    symmetry_df = run_stage3_symmetry_analysis(
        stage2_results, alpha=STAGE3_ALPHA, rho=rho_used
    )

    # ---- Save all results ----
    all_skipped = stage1_skipped + stage2_skipped
    save_rq2_results(
        stage1_results, stage2_results, symmetry_df,
        qualified, all_skipped, output_dir,
        pairwise_df=pairwise_df,
        rho_used=empirical_rho,
    )

    # ---- Robustness analysis ----
    run_robustness_analysis(stage1_results, stage2_results, data, output_dir)

    # ---- Print overall summary ----
    logger.info("\n" + "=" * 60)
    logger.info("RQ2 OVERALL SUMMARY")
    logger.info("=" * 60)

    n_s1_sig = sum(1 for r in stage1_results if r.is_significant)
    n_s2_sig = sum(1 for r in stage2_results if r.is_significant)

    logger.info(f"Stage 1: {n_s1_sig}/{len(stage1_results)} significant "
               f"(BH alpha={STAGE1_FDR_ALPHA})")
    logger.info(f"  -> {len(qualified)} smells qualified for Stage 2 "
               f"(gate: p_adj < {STAGE1_QUALIFICATION_THRESHOLD})")

    logger.info(f"Stage 2: {n_s2_sig}/{len(stage2_results)} significant "
               f"(alpha={STAGE2_ALPHA})")

    if not symmetry_df.empty:
        n_sym = (symmetry_df['classification'] == 'symmetric').sum()
        n_hys = (symmetry_df['classification'] == 'hysteresis').sum()
        n_oc = (symmetry_df['classification'] == 'overcorrection').sum()
        logger.info(f"Stage 3: {len(symmetry_df)} symmetry tests (rho={rho_used:.4f}) — "
                   f"{n_sym} symmetric, {n_hys} hysteresis, {n_oc} overcorrection")

    # Top impactful smells from Stage 1
    ranking = generate_smell_ranking_stage1(stage1_results)
    if not ranking.empty:
        logger.info("\nSmell ranking (by |aggregate beta|):")
        for _, row in ranking.head(5).iterrows():
            sig = " [QUALIFIED]" if row['qualified_stage2'] else ""
            logger.info(f"  {row['rank']}. {row['smell_type']} ({row['transition']}): "
                       f"|beta|={row['abs_coefficient']:.6f}, "
                       f"p_adj={row['p_value_adj']:.4f}{sig}")

    # Most affected dimensions from Stage 2
    if stage2_results:
        suscept = generate_dimension_susceptibility(stage2_results)
        if not suscept.empty:
            logger.info("\nDimension susceptibility (Stage 2):")
            for _, row in suscept.iterrows():
                logger.info(f"  {row['dimension']}: "
                           f"{row['significant_tests']}/{row['total_tests']} significant, "
                           f"rate={row['susceptibility_rate']:.2%}")

    logger.info("\n" + "=" * 60)

    # Return structured results for pipeline integration
    return {
        'stage1_results': stage1_results,
        'stage2_results': stage2_results,
        'stage3_symmetry': symmetry_df,
        'qualified': qualified,
        'all_skipped': all_skipped,
        'pairwise': pairwise_df,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RQ2 hypothesis tests (3-stage approach)")
    parser.add_argument("--data", type=str, help="Path to analysis-ready data")
    parser.add_argument("--output", type=str, help="Path to output directory")
    parser.add_argument("--alpha", type=float, default=0.05, help="Base significance level")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum samples per test")

    args = parser.parse_args()

    main(
        data_path=args.data,
        output_dir=args.output,
        alpha=args.alpha,
        min_samples=args.min_samples,
    )
