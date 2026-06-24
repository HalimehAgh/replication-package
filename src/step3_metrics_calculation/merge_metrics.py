"""
Merge Metrics Module

Merges all per-project normalized_metrics.jsonl files into a single
intermediate CSV file at data/processed/merged/all_projects_merged.csv.

This creates an intermediate step between metrics calculation (step 3)
and analysis preparation (step 4), making the data pipeline more transparent.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maintainability dimensions
DIMENSIONS = [
    'structural_complexity',
    'code_size',
    'defect_proneness',
    'understandability',
    'testing_quality',
    'ml_specific'
]

# Control variables stored in control_variables dicts
CONTROL_VARS = [
    'window_total_churn',
    'window_modified_files',
    'window_commit_count',
    'window_bug_fix_ratio',
    'smell_count_at_commit'
]

# Raw metrics to preserve (subset for debugging)
RAW_METRICS = [
    'cyclomatic_complexity',
    'loc',
    'sloc',
    'cognitive_complexity',
    'docstring_coverage',
    'comment_ratio'
]


def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a nested JSONL record into a flat dictionary for CSV.

    Args:
        record: Single record from normalized_metrics.jsonl

    Returns:
        Flattened dictionary with all columns
    """
    flat = {}

    # Core identifiers
    flat['smell_instance_id'] = record.get('smell_instance_id')
    flat['project_name'] = record.get('project_name')
    flat['smell_type'] = record.get('smell_type')
    flat['change_type'] = record.get('change_type')
    flat['file_path'] = record.get('file_path')
    flat['function_id'] = record.get('function_id')

    # Commit hashes
    flat['before_commit'] = record.get('before_commit')
    flat['after_commit'] = record.get('after_commit')
    flat['unique_event_key'] = record.get('unique_event_key')

    # Before dimension scores
    before_dims = record.get('before_dimension_scores', {})
    for dim in DIMENSIONS:
        flat[f'{dim}_before'] = before_dims.get(dim)

    # After dimension scores
    after_dims = record.get('after_dimension_scores', {})
    for dim in DIMENSIONS:
        flat[f'{dim}_after'] = after_dims.get(dim)

    # Delta dimension scores
    delta_dims = record.get('delta_dimension_scores', {})
    for dim in DIMENSIONS:
        flat[f'{dim}_delta'] = delta_dims.get(dim)

    # Overall scores
    flat['before_overall_score'] = record.get('before_overall_score')
    flat['after_overall_score'] = record.get('after_overall_score')
    flat['delta_overall_score'] = record.get('delta_overall_score')

    # Before control variables
    before_ctrl = record.get('before_control_variables', {})
    for var in CONTROL_VARS:
        flat[f'{var}_before'] = before_ctrl.get(var)

    # After control variables
    after_ctrl = record.get('after_control_variables', {})
    for var in CONTROL_VARS:
        flat[f'{var}_after'] = after_ctrl.get(var)

    # Delta control variables
    delta_ctrl = record.get('delta_control_variables', {})
    for var in CONTROL_VARS:
        flat[f'{var}_delta'] = delta_ctrl.get(var)

    # Raw metrics (before) for debugging
    before_metrics = record.get('before_metrics', {})
    for metric in RAW_METRICS:
        flat[f'raw_{metric}_before'] = before_metrics.get(metric)

    # Raw metrics (after) for debugging
    after_metrics = record.get('after_metrics', {})
    for metric in RAW_METRICS:
        flat[f'raw_{metric}_after'] = after_metrics.get(metric)

    # Additional context from raw metrics - separate line counts for RQ revisions
    flat['lines_added'] = after_metrics.get('lines_added', 0)
    flat['lines_deleted'] = after_metrics.get('lines_deleted', 0)
    flat['raw_total_churn'] = after_metrics.get('total_churn', 0)  # Keep for backward compat

    # Commit type classification flags
    flat['is_bug_fix'] = after_metrics.get('is_bug_fix', False)
    flat['is_feature_commit'] = after_metrics.get('is_feature_commit', False)

    return flat


def load_all_normalized_metrics(
    metrics_dir: Path,
    project_filter: Optional[set] = None
) -> List[Dict[str, Any]]:
    """
    Load all normalized_metrics.jsonl files from project subdirectories.

    Args:
        metrics_dir: Path to data/processed/metrics
        project_filter: Optional set of project directory names (owner_repo)
                       to include. If None, all projects are loaded.

    Returns:
        List of flattened records from all projects
    """
    all_records = []

    project_dirs = sorted([d for d in metrics_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(project_dirs)} project directories")

    if project_filter is not None:
        project_dirs = [d for d in project_dirs if d.name in project_filter]
        logger.info(f"Filtered to {len(project_dirs)} selected projects")

    for project_dir in project_dirs:
        metrics_file = project_dir / "normalized_metrics.jsonl"

        if not metrics_file.exists():
            logger.warning(f"No normalized_metrics.jsonl in {project_dir.name}")
            continue

        # Count records for this project
        project_count = 0

        with open(metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        flat = flatten_record(record)
                        # Add project directory name (owner_repo format)
                        flat['project_dir'] = project_dir.name
                        all_records.append(flat)
                        project_count += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line in {metrics_file}: {e}")

        logger.info(f"  {project_dir.name}: {project_count} events")

    return all_records


def merge_metrics(
    metrics_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    project_filter: Optional[set] = None
) -> pd.DataFrame:
    """
    Merge all per-project normalized metrics into a single CSV.

    Args:
        metrics_dir: Path to data/processed/metrics
        output_dir: Path to data/processed/merged
        project_filter: Optional set of project directory names to include

    Returns:
        DataFrame with all merged metrics
    """
    # Default paths
    code_root = Path(__file__).parent.parent.parent

    if metrics_dir is None:
        metrics_dir = code_root / "data" / "processed" / "metrics"

    if output_dir is None:
        output_dir = code_root / "data" / "processed" / "merged"

    logger.info(f"Reading from: {metrics_dir}")
    logger.info(f"Writing to: {output_dir}")

    # Load all records
    records = load_all_normalized_metrics(metrics_dir, project_filter=project_filter)

    if not records:
        logger.error("No records loaded!")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_file = output_dir / "all_projects_merged.csv"
    df.to_csv(output_file, index=False)

    logger.info(f"Saved {len(df)} events to {output_file}")
    logger.info(f"Columns: {len(df.columns)}")

    # Summary statistics
    logger.info("\n=== Merge Summary ===")
    logger.info(f"Total events: {len(df)}")
    logger.info(f"Projects: {df['project_dir'].nunique()}")
    logger.info(f"Smell types: {df['smell_type'].nunique()}")
    logger.info(f"By change type:")
    for ct, count in df['change_type'].value_counts().items():
        logger.info(f"  {ct}: {count}")

    return df


def main():
    """Main entry point for merge_metrics script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge per-project normalized metrics into single CSV"
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        help="Path to data/processed/metrics directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory (default: data/processed/merged)"
    )
    parser.add_argument(
        "--selected-projects",
        type=str,
        help="Path to selected_projects.csv to filter by (project_id column, owner/repo format)"
    )

    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    project_filter = None
    if args.selected_projects:
        import csv
        with open(args.selected_projects, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            project_filter = {row["project_id"].replace("/", "_") for row in reader}
        logger.info(f"Loaded {len(project_filter)} projects from {args.selected_projects}")

    merge_metrics(metrics_dir, output_dir, project_filter=project_filter)


if __name__ == "__main__":
    main()
