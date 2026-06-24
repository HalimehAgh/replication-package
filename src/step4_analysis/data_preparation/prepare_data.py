"""
Data Preparation Module for Hypothesis Testing

Loads and prepares normalized metrics data for RQ1 and RQ2 hypothesis tests.
Reads from the merged CSV in data/processed/merged/ and generates
analysis-ready datasets.

Output structure:
- analysis-ready/all_events.csv           <- Shared by RQ1 and RQ2
- analysis-ready/rq1/introduction_data.csv <- RQ1-specific
- analysis-ready/rq1/removal_data.csv      <- RQ1-specific
- analysis-ready/rq2/{smell_type}_data.csv <- RQ2-specific
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Import comprehensive quality checks
from src.step4_analysis.data_preparation.quality_checks import (
    run_all_quality_checks,
    check_missing_values,
    check_sample_sizes,
    check_outliers,
    check_distributions,
    check_data_types,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Constants for maintainability dimensions (6 tested dimensions)
# Historical context variables are used as control variables, not a tested dimension
DIMENSIONS = [
    'structural_complexity',
    'code_size',
    'defect_proneness',
    'understandability',
    'testing_quality',
    'ml_specific',
]

# Control variables for regression (aligned with RQ revision specification)
# Updated Formula (RQ1 Revised):
#   ΔMaintainability_d = β₀ + Σᵢβᵢ·SmellTypeᵢ + β₁₃·Baseline_Metric_d_At_A-1 +
#                        β₁₄·lines_added + β₁₅·lines_deleted + β₁₆·is_feature_commit +
#                        β₁₇·is_bug_fix + β₁₈·log_loc + ε
CONTROL_VARIABLES = [
    'total_smells_at_commit',  # Total number of smells at the transition commit
    'baseline_score',           # Baseline metric at A-1 (or B-1 for removal)
    'lines_added',              # Lines added in the commit (separate from deleted)
    'lines_deleted',            # Lines deleted in the commit (separate from added)
    'is_feature_commit',        # Binary flag for feature commits
    'is_bug_fix',               # Binary flag for bug fix commits
    'log_loc'                   # Log of lines of code
]

# Original 12 smell types from CodeSmile paper (Recupito et al., EMSE 2025)
# "When code smells meet ML: on the lifecycle of ML-specific code smells"
# We exclude smell types added to CodeSmile after the original publication.
ORIGINAL_CODESMILE_SMELLS = {
    'Chain_Indexing',                                      # CIDX
    'columns_and_datatype_not_explicitly_set',             # CDE
    'dataframe_conversion_api_misused',                    # DCA
    'in_place_apis_misused',                               # IPA
    'gradients_not_cleared_before_backward_propagation',   # GNC
    'matrix_multiplication_api_misused',                   # MMA
    'memory_not_freed',                                    # MNF
    'merge_api_parameter_not_explicitly_set',              # MAP
    'pytorch_call_method_misused',                         # PC
    'unnecessary_iteration',                               # UI
    # Two original smells not found in our dataset:
    # 'nan_equivalence_comparison_misused',                # NAN
    # 'tensor_array_not_used',                             # TA
}


def validate_no_duplicates(
    df: pd.DataFrame,
    group_cols: List[str],
    dataset_name: str
) -> bool:
    """
    Validate that there are no duplicate rows for the grouping key.

    This ensures proper aggregation was performed and no pseudo-replication exists.

    Args:
        df: DataFrame to validate
        group_cols: Columns that should be unique together
        dataset_name: Name for logging

    Returns:
        True if valid (no duplicates)

    Raises:
        ValueError: If duplicates found (with details)
    """
    duplicates = df.groupby(group_cols).size()
    duplicates = duplicates[duplicates > 1]

    if len(duplicates) > 0:
        logger.error(f"Found {len(duplicates)} duplicate groups in {dataset_name}")
        logger.error(f"Sample duplicates:\n{duplicates.head(5)}")
        raise ValueError(f"Pseudo-replication detected in {dataset_name}: {len(duplicates)} duplicate file+commit pairs")

    logger.info(f"Validated {dataset_name}: no duplicates found ({len(df)} unique observations)")
    return True


def aggregate_to_file_commit_level(
    df: pd.DataFrame,
    group_by_smell_type: bool = False
) -> pd.DataFrame:
    """
    Aggregate data from smell-instance level to file+commit level.

    This fixes pseudo-replication where multiple instances of the same
    smell in the same file+commit have identical delta values.

    Rules applied:
    1. Different smell TYPES in same commit -> Separate rows in RQ2 only
    2. Multiple INSTANCES of same type -> ONE row with instance count
    3. Different COMMITS -> Separate rows

    Args:
        df: DataFrame with one row per smell instance
        group_by_smell_type: If True (RQ2), group by smell_type too.
                            If False (RQ1), aggregate across all smell types.

    Returns:
        DataFrame with one row per unique file+commit+change_type [+smell_type]
    """
    # Base columns to group by
    base_group_cols = ['project', 'file_path', 'before_commit', 'after_commit', 'change_type']

    if group_by_smell_type:
        group_cols = base_group_cols + ['smell_type']
    else:
        group_cols = base_group_cols

    # Columns to take first value (identical within file+commit)
    metric_cols = []
    for dim in DIMENSIONS:
        metric_cols.extend([
            f'{dim}_before', f'{dim}_after', f'{dim}_delta', f'{dim}_baseline'
        ])

    control_cols = [
        'lines_added', 'lines_deleted', 'total_churn', 'commit_count', 'bug_fix_ratio',
        'is_bug_fix', 'is_feature_commit', 'loc', 'log_loc',
        'smell_count_before', 'smell_count_after', 'total_smells_at_commit'
    ]

    first_cols = metric_cols + control_cols + ['file_id']

    # Build aggregation dictionary
    agg_dict = {col: 'first' for col in first_cols if col in df.columns}

    # Count aggregations
    agg_dict['smell_instance_id'] = 'count'  # Total instances
    if not group_by_smell_type:
        agg_dict['smell_type'] = 'nunique'  # Count distinct smell types (RQ1 only)

    # Perform aggregation
    aggregated = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Rename count columns
    if group_by_smell_type:
        aggregated = aggregated.rename(columns={
            'smell_instance_id': 'instance_count'
        })
    else:
        aggregated = aggregated.rename(columns={
            'smell_instance_id': 'total_smell_instances',
            'smell_type': 'total_smell_types'
        })

    logger.info(f"Aggregated {len(df)} instances to {len(aggregated)} file+commit observations")

    return aggregated


def calculate_other_smell_types_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total instances of other smell types in the same file+commit.

    For each (file, commit, smell_type) row in RQ2, sum the instance_count
    of all OTHER smell types introduced/removed in the same commit.

    This controls for co-occurring smells in smell-specific regression.

    Args:
        df: Aggregated DataFrame with one row per file+commit+smell_type

    Returns:
        DataFrame with 'other_smell_types' column added
    """
    # Group by (file, commit, change_type) to get total instances per commit
    commit_group_cols = ['project', 'file_path', 'before_commit', 'after_commit', 'change_type']

    # Total instances across ALL smell types per commit
    commit_totals = df.groupby(commit_group_cols)['instance_count'].sum().reset_index(
        name='_total_instances_in_commit'
    )

    # Merge back
    df = df.merge(commit_totals, on=commit_group_cols, how='left')

    # Subtract this row's own instance_count to get other smells' instances
    df['other_smell_types'] = (df['_total_instances_in_commit'] - df['instance_count']).clip(lower=0)

    # Clean up
    df = df.drop(columns=['_total_instances_in_commit'])

    return df


def create_rq1_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform data to wide format with 12 smell type indicator columns for RQ1.

    This transforms data from long format (one row per smell instance) to wide format
    (one row per file+commit with 12 smell count columns).

    Input: One row per (file, commit, smell_type)
    Output: One row per (file, commit) with 12 smell type count columns

    Args:
        df: Aggregated DataFrame with one row per file+commit+smell_type

    Returns:
        Wide-format DataFrame with smell type indicator columns
    """
    # Define the 12 smell types (short names for column names)
    smell_type_mapping = {
        'Chain_Indexing': 'smell_CIDX',
        'columns_and_datatype_not_explicitly_set': 'smell_CDTNE',
        'dataframe_conversion_api_misused': 'smell_DCA',
        'in_place_apis_misused': 'smell_IPA',
        'matrix_multiplication_api_misused': 'smell_MMA',
        'merge_api_parameter_not_explicitly_set': 'smell_MAP',
        'nan_equivalence_comparison_misused': 'smell_NAN',
        'unnecessary_iteration': 'smell_UI',
        'gradients_not_cleared_before_backward_propagation': 'smell_GNC',
        'memory_not_freed': 'smell_MNF',
        'pytorch_call_method_misused': 'smell_PC',
        'tensor_array_not_used': 'smell_TA',
    }

    # Base columns for grouping
    group_cols = ['project', 'file_path', 'before_commit', 'after_commit', 'change_type']

    # Create copy and map smell types to short names
    df_copy = df.copy()
    df_copy['smell_short'] = df_copy['smell_type'].map(smell_type_mapping).fillna(
        df_copy['smell_type'].apply(lambda x: f'smell_{x[:8]}')
    )

    # Pivot smell counts - instance_count per smell type
    smell_pivot = df_copy.pivot_table(
        index=group_cols,
        columns='smell_short',
        values='instance_count',
        fill_value=0,
        aggfunc='sum'
    ).reset_index()

    # Get non-smell columns (metrics, controls) - take first since identical per file+commit
    metric_cols = []
    for dim in DIMENSIONS:
        metric_cols.extend([
            f'{dim}_before', f'{dim}_after', f'{dim}_delta', f'{dim}_baseline'
        ])

    control_cols = [
        'lines_added', 'lines_deleted', 'total_churn', 'commit_count', 'bug_fix_ratio',
        'is_bug_fix', 'is_feature_commit', 'loc', 'log_loc',
        'smell_count_before', 'smell_count_after', 'total_smells_at_commit', 'file_id'
    ]

    # Get first values for metrics and controls
    first_cols = [c for c in (metric_cols + control_cols) if c in df.columns]
    metrics_df = df.groupby(group_cols)[first_cols].first().reset_index()

    # Count total smell types and instances
    totals = df.groupby(group_cols).agg({
        'smell_type': 'nunique',
        'instance_count': 'sum'
    }).reset_index()
    totals = totals.rename(columns={
        'smell_type': 'total_smell_types',
        'instance_count': 'total_smell_instances'
    })

    # Merge all together
    result = smell_pivot.merge(metrics_df, on=group_cols, how='left')
    result = result.merge(totals, on=group_cols, how='left')

    # Fill missing smell columns with 0 (smell types not in data)
    for short_name in smell_type_mapping.values():
        if short_name not in result.columns:
            result[short_name] = 0

    logger.info(f"Created wide format: {len(df)} rows -> {len(result)} rows (file+commit level)")
    logger.info(f"  Smell columns: {[c for c in result.columns if c.startswith('smell_')]}")

    return result



def load_merged_data(merged_file: Path) -> pd.DataFrame:
    """
    Load merged data from the intermediate CSV file.

    This is the primary data loading function. The merged file is created by
    src/step3_metrics_calculation/merge_metrics.py.

    Args:
        merged_file: Path to data/processed/merged/all_projects_merged.csv

    Returns:
        DataFrame with all smell events
    """
    if not merged_file.exists():
        raise FileNotFoundError(
            f"Merged file not found: {merged_file}\n"
            "Run 'python -m src.step3_metrics_calculation.merge_metrics' first."
        )

    logger.info(f"Loading merged data from: {merged_file}")
    df = pd.read_csv(merged_file)
    logger.info(f"Loaded {len(df)} events from merged file")

    return df


def extract_regression_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and prepare variables for regression analysis.

    Creates:
    - file_id: Unique identifier for clustering (project + file_path)
    - before_commit, after_commit: Commit hashes for aggregation
    - Dimension scores and deltas
    - Control variables

    Args:
        df: DataFrame from load_merged_data (flattened CSV)

    Returns:
        DataFrame with regression-ready variables
    """
    result = pd.DataFrame()

    # Basic identifiers
    result['smell_instance_id'] = df['smell_instance_id']

    # Handle project column naming
    if 'project_dir' in df.columns:
        result['project'] = df['project_dir']
    elif 'project_name' in df.columns:
        result['project'] = df['project_name'].apply(lambda x: x.replace('/', '_') if isinstance(x, str) else x)
    else:
        result['project'] = df['project']

    result['smell_type'] = df['smell_type']
    result['change_type'] = df['change_type']
    result['file_path'] = df['file_path']

    # Preserve commit columns for aggregation (critical for fixing pseudo-replication)
    result['before_commit'] = df['before_commit']
    result['after_commit'] = df['after_commit']

    # Create file_id for clustering
    result['file_id'] = result['project'] + '_' + result['file_path']

    # Dimension scores (already flattened in merged CSV)
    for dim in DIMENSIONS:
        result[f'{dim}_before'] = df[f'{dim}_before']
        result[f'{dim}_after'] = df[f'{dim}_after']
        result[f'{dim}_delta'] = df[f'{dim}_delta']

    # Control variables
    result['smell_count_before'] = df['smell_count_at_commit_before'].fillna(0)
    result['smell_count_after'] = df['smell_count_at_commit_after'].fillna(0)
    result['smell_count_change'] = result['smell_count_after'] - result['smell_count_before']

    # Total smells at transition commit
    result['total_smells_at_commit'] = np.where(
        df['change_type'] == 'introduction',
        result['smell_count_after'],
        result['smell_count_before']
    )

    # Calculate total_other_smells (count of OTHER smell type events at same file)
    # For RQ2: controls for co-occurring smells EXCLUDING target smell type
    # Formula: total_other_smells = (total events at file) - (events of same smell type)
    #
    # Example: File with 3 events of smell A and 2 events of smell B
    #   - For smell A events: total_other_smells = 5 - 3 = 2
    #   - For smell B events: total_other_smells = 5 - 2 = 3

    # Create file key for grouping (project + file_path)
    file_key = result['project'].astype(str) + '_' + result['file_path'].astype(str)

    # Count total events per file
    total_events_per_file = file_key.map(file_key.value_counts())

    # Count events of same smell type at same file
    result['_file_key'] = file_key
    same_type_count = result.groupby(['_file_key', 'smell_type']).transform('size')

    # total_other_smells = total events at file - events of same smell type
    result['total_other_smells'] = (total_events_per_file - same_type_count).clip(lower=0)

    # Clean up temporary column
    result = result.drop(columns=['_file_key'])

    # Churn control variables - separate lines added/deleted (RQ revision)
    # Use columns from merge_metrics if available, otherwise fall back to window_total_churn
    if 'lines_added' in df.columns:
        result['lines_added'] = df['lines_added'].fillna(0)
    else:
        result['lines_added'] = 0

    if 'lines_deleted' in df.columns:
        result['lines_deleted'] = df['lines_deleted'].fillna(0)
    else:
        result['lines_deleted'] = 0

    # Keep total_churn for backward compatibility
    result['total_churn'] = df['window_total_churn_before'].fillna(0)

    # Commit type classification flags (RQ revision)
    if 'is_bug_fix' in df.columns:
        result['is_bug_fix'] = df['is_bug_fix'].fillna(False).astype(bool)
    else:
        result['is_bug_fix'] = False

    if 'is_feature_commit' in df.columns:
        result['is_feature_commit'] = df['is_feature_commit'].fillna(False).astype(bool)
    else:
        result['is_feature_commit'] = False

    # Window-based bug fix ratio (keep for backward compat)
    result['bug_fix_ratio'] = df['window_bug_fix_ratio_before'].fillna(0)

    # Window commit count (spec γ₃ - Commit_Count_before)
    if 'window_commit_count_before' in df.columns:
        result['commit_count'] = df['window_commit_count_before'].fillna(0)
    else:
        result['commit_count'] = 0

    # LOC from raw metrics
    result['loc'] = df['raw_loc_before'].fillna(100)
    result['log_loc'] = np.log1p(result['loc'])

    # For each dimension, set baseline score (same as before value)
    for dim in DIMENSIONS:
        result[f'{dim}_baseline'] = result[f'{dim}_before']

    return result


def prepare_rq1_data(
    df: pd.DataFrame,
    base_dir: Path,
    rq1_dir: Path
) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for RQ1 hypothesis tests with proper aggregation.

    RQ1 tests the overall impact of ML code smells on maintainability,
    regardless of smell type. Aggregates ALL smell types together.

    One row per unique (file_id, before_commit, after_commit, change_type).

    Output columns include:
    - total_smell_types: Count of distinct smell types introduced/removed
    - total_smell_instances: Total instances across all types

    Output files:
    - rq1_dir/introduction_data.csv (RQ1-specific)
    - rq1_dir/removal_data.csv (RQ1-specific)

    Args:
        df: Prepared DataFrame from extract_regression_variables
        base_dir: Base analysis-ready directory for shared files
        rq1_dir: RQ1-specific output directory

    Returns:
        Dictionary with 'introduction' and 'removal' DataFrames
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    rq1_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate to file+commit level (across ALL smell types)
    logger.info("Aggregating RQ1 data to file+commit level...")
    aggregated_df = aggregate_to_file_commit_level(df, group_by_smell_type=False)

    datasets = {}

    for transition in ['introduction', 'removal']:
        original_count = len(df[df['change_type'] == transition])
        subset = aggregated_df[aggregated_df['change_type'] == transition].copy()

        # Validate no duplicates (critical check)
        validate_no_duplicates(
            subset,
            ['file_id', 'before_commit', 'after_commit'],
            f'RQ1 {transition}'
        )

        logger.info(f"RQ1 {transition}: {len(subset)} observations "
                   f"(aggregated from {original_count} instances)")

        # Save RQ1-specific files to rq1/
        output_file = rq1_dir / f'{transition}_data.csv'
        subset.to_csv(output_file, index=False)
        logger.info(f"Saved to {output_file}")

        datasets[transition] = subset

    return datasets


def prepare_rq2_data(df: pd.DataFrame, output_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for RQ2 hypothesis tests with proper aggregation.

    RQ2 tests smell-specific impacts, analyzing each smell type
    separately. Aggregates to file+commit+smell_type level.

    One row per unique (file_id, before_commit, after_commit, change_type, smell_type).

    Output columns include:
    - instance_count: Number of instances of THIS smell type
    - other_smell_types: Count of OTHER smell types in same commit

    Args:
        df: Prepared DataFrame from extract_regression_variables
        output_dir: Directory to save analysis-ready data

    Returns:
        Dictionary mapping smell_type to DataFrame
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate to file+commit+smell_type level
    logger.info("Aggregating RQ2 data to file+commit+smell_type level...")
    aggregated_df = aggregate_to_file_commit_level(df, group_by_smell_type=True)

    # Calculate other_smell_types for co-occurrence control
    aggregated_df = calculate_other_smell_types_count(aggregated_df)

    datasets = {}

    # Get unique smell types
    smell_types = aggregated_df['smell_type'].unique()
    logger.info(f"Found {len(smell_types)} unique smell types")

    for smell_type in smell_types:
        original_count = len(df[df['smell_type'] == smell_type])
        subset = aggregated_df[aggregated_df['smell_type'] == smell_type].copy()

        # Validate no duplicates within this smell type's dataset
        validate_no_duplicates(
            subset,
            ['file_id', 'before_commit', 'after_commit', 'change_type'],
            f'RQ2 {smell_type}'
        )

        # Count by transition
        intro_count = len(subset[subset['change_type'] == 'introduction'])
        removal_count = len(subset[subset['change_type'] == 'removal'])

        logger.info(f"  {smell_type}: {len(subset)} observations "
                   f"(intro={intro_count}, removal={removal_count}, "
                   f"from {original_count} instances)")

        # Save to CSV
        safe_name = smell_type.replace('/', '_').replace(' ', '_')
        output_file = output_dir / f'{safe_name}_data.csv'
        subset.to_csv(output_file, index=False)

        datasets[smell_type] = subset

    # Save summary statistics
    summary = []
    for smell_type, data in datasets.items():
        summary.append({
            'smell_type': smell_type,
            'total_events': len(data),
            'introductions': len(data[data['change_type'] == 'introduction']),
            'removals': len(data[data['change_type'] == 'removal']),
            'unique_files': data['file_id'].nunique(),
            'unique_projects': data['project'].nunique()
        })

    summary_df = pd.DataFrame(summary)
    summary_file = output_dir / 'smell_type_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved smell type summary to {summary_file}")

    return datasets


def run_quality_checks(df: pd.DataFrame, output_file: Path = None) -> Dict[str, any]:
    """
    Run comprehensive quality checks on prepared data.

    Uses the quality_checks module for thorough validation:
    1. Missing values with severity assessment
    2. Sample sizes with cluster analysis
    3. Outlier detection (IQR and z-score methods)
    4. Distribution checks with normality tests
    5. Data type validation

    Args:
        df: Prepared DataFrame
        output_file: Optional path to save the quality report

    Returns:
        Dictionary with comprehensive quality check results
    """
    # Use the comprehensive quality checks from quality_checks.py
    return run_all_quality_checks(df, output_file)


def main(
    merged_file: str = None,
    output_dir: str = None,
    run_checks: bool = True
):
    """
    Main function to prepare all data for hypothesis testing.

    Reads from the merged CSV file (created by merge_metrics.py) and generates
    analysis-ready datasets for RQ1 and RQ2.

    Output structure:
    - analysis-ready/all_events.csv (shared)
    - analysis-ready/rq1/introduction_data.csv
    - analysis-ready/rq1/removal_data.csv
    - analysis-ready/rq2/{smell_type}_data.csv

    Args:
        merged_file: Path to data/processed/merged/all_projects_merged.csv
        output_dir: Base path for output directories
        run_checks: Whether to run quality checks
    """
    # Default paths
    code_root = Path(__file__).parent.parent.parent.parent

    if merged_file is None:
        merged_file = code_root / "data" / "processed" / "merged" / "all_projects_merged.csv"
    else:
        merged_file = Path(merged_file)

    if output_dir is None:
        output_dir = code_root / "data" / "analysis-ready"
    else:
        output_dir = Path(output_dir)

    logger.info(f"Loading merged data from: {merged_file}")
    logger.info(f"Output directory: {output_dir}")

    # Load merged data
    raw_df = load_merged_data(merged_file)

    if len(raw_df) == 0:
        logger.error("No data loaded!")
        return

    # Filter to original 12 CodeSmile smell types
    original_count = len(raw_df)
    original_types = raw_df['smell_type'].nunique()
    excluded_types = set(raw_df['smell_type'].unique()) - ORIGINAL_CODESMILE_SMELLS

    raw_df = raw_df[raw_df['smell_type'].isin(ORIGINAL_CODESMILE_SMELLS)]

    filtered_types = raw_df['smell_type'].nunique()
    logger.info(f"Filtered to original CodeSmile smell types: {original_count} -> {len(raw_df)} events")
    logger.info(f"  Smell types: {original_types} -> {filtered_types}")
    if excluded_types:
        logger.info(f"  Excluded (added after original paper): {sorted(excluded_types)}")

    # Extract regression variables
    prepared_df = extract_regression_variables(raw_df)

    logger.info(f"Prepared {len(prepared_df)} events with {len(prepared_df.columns)} variables")

    # Quality checks
    if run_checks:
        logger.info("Running comprehensive quality checks...")
        checks_file = output_dir / 'quality_checks.json'
        checks_file.parent.mkdir(parents=True, exist_ok=True)
        checks = run_quality_checks(prepared_df, checks_file)

        # Log summary from comprehensive checks
        if 'summary' in checks:
            logger.info(f"Quality assessment: {checks['summary']['data_quality']}")
            if checks['summary']['issues_found']:
                logger.warning(f"Issues found in: {', '.join(checks['summary']['issues_found'])}")

        # Warn about low sample combinations
        if 'sample_sizes' in checks and checks['sample_sizes'].get('low_sample_combinations'):
            low_combos = checks['sample_sizes']['low_sample_combinations']
            logger.warning(f"Found {len(low_combos)} combinations with low sample sizes (<10):")
            for combo in low_combos[:5]:  # Show first 5
                logger.warning(f"  {combo['smell']} + {combo['transition']}: n={combo['count']}")

    # Prepare RQ1 data (aggregated across ALL smell types)
    logger.info("\n=== Preparing RQ1 data ===")
    rq1_output = output_dir / 'rq1'
    rq1_datasets = prepare_rq1_data(prepared_df, output_dir, rq1_output)

    # Prepare RQ2 data (aggregated per smell type)
    logger.info("\n=== Preparing RQ2 data ===")
    rq2_output = output_dir / 'rq2'
    rq2_datasets = prepare_rq2_data(prepared_df, rq2_output)

    # Save all_events.csv at RQ2 level (most granular aggregated level)
    # This is the aggregated data with one row per file+commit+smell_type
    logger.info("\n=== Saving shared all_events.csv ===")
    aggregated_all = aggregate_to_file_commit_level(prepared_df, group_by_smell_type=True)
    aggregated_all = calculate_other_smell_types_count(aggregated_all)
    combined_file = output_dir / 'all_events.csv'
    aggregated_all.to_csv(combined_file, index=False)
    logger.info(f"Saved aggregated all_events.csv to {combined_file}")

    # Create and save wide-format RQ1 data (12 smell indicator columns)
    logger.info("\n=== Creating RQ1 wide format (12 smell columns) ===")
    wide_format_df = create_rq1_wide_format(aggregated_all)


    # Save wide format for each transition
    for transition in ['introduction', 'removal']:
        wide_subset = wide_format_df[wide_format_df['change_type'] == transition].copy()
        wide_file = rq1_output / f'{transition}_wide.csv'
        wide_subset.to_csv(wide_file, index=False)
        logger.info(f"Saved wide-format RQ1 data: {wide_file} ({len(wide_subset)} rows)")

    # Calculate RQ1 totals
    rq1_intro_count = len(rq1_datasets.get('introduction', []))
    rq1_removal_count = len(rq1_datasets.get('removal', []))

    # Summary
    logger.info("\n=== Data Preparation Complete ===")
    logger.info(f"Original instances: {len(prepared_df)}")
    logger.info(f"RQ1 observations: {rq1_intro_count + rq1_removal_count} "
               f"(intro={rq1_intro_count}, removal={rq1_removal_count})")
    logger.info(f"RQ2 observations: {len(aggregated_all)} "
               f"(across {len(rq2_datasets)} smell types)")
    logger.info(f"Projects: {prepared_df['project'].nunique()}")
    logger.info(f"Unique files: {prepared_df['file_id'].nunique()}")

    return prepared_df, rq1_datasets, rq2_datasets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data for hypothesis testing")
    parser.add_argument(
        "--merged-file",
        type=str,
        help="Path to merged CSV file (default: data/processed/merged/all_projects_merged.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory (default: data/analysis-ready)"
    )
    parser.add_argument("--no-checks", action="store_true", help="Skip quality checks")

    args = parser.parse_args()

    main(
        merged_file=args.merged_file,
        output_dir=args.output_dir,
        run_checks=not args.no_checks
    )
