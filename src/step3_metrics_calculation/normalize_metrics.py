"""
Metrics Normalization and Dimension Score Calculation

Post-processing script that:
1. Loads raw metrics from validated_metrics.jsonl files
2. Calculates global min/max across all data points
3. Applies Min-Max normalization to [0,1] range
4. Inverts "higher is better" metrics (so 0=best, 1=worst for all)
5. Calculates dimension scores as averages of normalized metrics
6. Outputs normalized_metrics.jsonl with dimension scores

Usage:
    python -m src.step3_metrics_calculation.normalize_metrics --metrics-dir data/processed/metrics

Reference:
- ISO/IEC 25010: Maintainability sub-characteristics
- Min-Max normalization: (x - min) / (max - min)
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from src import config
from src.step3_metrics_calculation.constants import (
    METRIC_DIMENSIONS,
    INVERT_METRICS,
    SNAPSHOT_METRICS,
    TIME_WINDOW_METRICS,
    CONTROL_VARIABLES,
    ALL_CONTROL_VARIABLES,
    ALL_METRICS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Quality metrics for dimension scores (excludes control variables)
QUALITY_METRICS = SNAPSHOT_METRICS

# All metrics to track statistics for (quality + control)
NORMALIZABLE_METRICS = ALL_METRICS + ['smell_count_at_commit']


class MetricsNormalizer:
    """
    Normalizes metrics across repositories and calculates dimension scores.

    Normalization approach:
    - Min-Max scaling to [0, 1] range using global min/max
    - Inversion for "higher is better" metrics
    - Dimension scores as arithmetic mean of constituent metrics
    """

    def __init__(self):
        """Initialize with empty global statistics."""
        # Global min/max for each metric across all data points
        self.global_min: Dict[str, float] = {}
        self.global_max: Dict[str, float] = {}
        # Count of valid values per metric
        self.value_counts: Dict[str, int] = defaultdict(int)

    def collect_statistics(
        self, metrics_dir: Path, project_filter: Optional[set] = None
    ) -> None:
        """
        Collect global min/max statistics from all metric files.

        Args:
            metrics_dir: Directory containing per-repository metric results
            project_filter: Optional set of project directory names to include.
                           If None, all projects are used.
        """
        logger.info("Collecting global statistics from all repositories...")

        # Initialize with extreme values for all normalizable metrics
        for metric in NORMALIZABLE_METRICS:
            self.global_min[metric] = float('inf')
            self.global_max[metric] = float('-inf')

        # Scan all repository directories
        repo_dirs = [d for d in metrics_dir.iterdir() if d.is_dir() and d.name != 'reports']

        if project_filter is not None:
            repo_dirs = [d for d in repo_dirs if d.name in project_filter]
            logger.info(f"Filtered to {len(repo_dirs)} selected projects")

        for repo_dir in repo_dirs:
            metrics_file = repo_dir / config.VALIDATED_METRICS_FILE
            if not metrics_file.exists():
                continue

            self._collect_from_file(metrics_file)

        # Report statistics
        logger.info(f"Collected statistics from {len(repo_dirs)} repositories")
        for metric in NORMALIZABLE_METRICS:
            if self.value_counts[metric] > 0:
                logger.debug(
                    f"  {metric}: min={self.global_min[metric]:.4f}, "
                    f"max={self.global_max[metric]:.4f}, "
                    f"n={self.value_counts[metric]}"
                )

    def _collect_from_file(self, metrics_file: Path) -> None:
        """Collect statistics from a single metrics file."""
        with open(metrics_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    self._update_stats_from_record(record)
                except json.JSONDecodeError:
                    continue

    def _update_stats_from_record(self, record: Dict[str, Any]) -> None:
        """Update global statistics from a single record."""
        # Check both before and after metrics
        for metrics_key in ['before_metrics', 'after_metrics']:
            metrics = record.get(metrics_key, {})
            if not metrics:
                continue

            for metric in NORMALIZABLE_METRICS:
                value = metrics.get(metric)
                if value is not None and self._is_valid_number(value):
                    self.global_min[metric] = min(self.global_min[metric], value)
                    self.global_max[metric] = max(self.global_max[metric], value)
                    self.value_counts[metric] += 1

    def _is_valid_number(self, value: Any) -> bool:
        """Check if value is a valid finite number."""
        if value is None:
            return False
        try:
            f = float(value)
            return not (f != f or f == float('inf') or f == float('-inf'))  # Check for NaN/inf
        except (TypeError, ValueError):
            return False

    def normalize_value(self, metric: str, value: Optional[float]) -> Optional[float]:
        """
        Normalize a single metric value using Min-Max scaling.

        Args:
            metric: Name of the metric
            value: Raw metric value

        Returns:
            Normalized value in [0, 1], or None if invalid
        """
        if value is None or not self._is_valid_number(value):
            return None

        min_val = self.global_min.get(metric, 0)
        max_val = self.global_max.get(metric, 1)

        # Handle case where all values are the same
        if max_val == min_val:
            normalized = 0.0  # All values equal, treat as minimum
        else:
            normalized = (value - min_val) / (max_val - min_val)

        # Clamp to [0, 1] for safety
        normalized = max(0.0, min(1.0, normalized))

        # Invert if this metric has "higher is better" semantics
        # After inversion: 0 = best, 1 = worst for ALL metrics
        if metric in INVERT_METRICS:
            normalized = 1.0 - normalized

        return normalized

    def normalize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """
        Normalize QUALITY metrics in a dictionary (excludes control variables).

        Args:
            metrics: Dictionary of raw metric values

        Returns:
            Dictionary of normalized QUALITY metric values (for dimension scoring)
        """
        normalized = {}
        for metric in QUALITY_METRICS:
            value = metrics.get(metric)
            normalized[metric] = self.normalize_value(metric, value)
        return normalized

    def normalize_control_variables(self, metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """
        Normalize CONTROL variable metrics separately (not for dimension scoring).

        Per Khomh et al. (2012) and Palomba et al. (2018):
        - Time-window metrics measure file CONTEXT (stable/active)
        - Smell co-occurrence tracks confounding effects
        These are used for stratification analysis, not quality measurement.

        Args:
            metrics: Dictionary of raw metric values

        Returns:
            Dictionary of normalized control variable values
        """
        normalized = {}
        for metric in TIME_WINDOW_METRICS:
            value = metrics.get(metric)
            normalized[metric] = self.normalize_value(metric, value)
        # Include smell count
        smell_count = metrics.get('smell_count_at_commit')
        normalized['smell_count_at_commit'] = self.normalize_value('smell_count_at_commit', smell_count)
        return normalized

    def calculate_dimension_scores(
        self,
        normalized_metrics: Dict[str, Optional[float]]
    ) -> Dict[str, Optional[float]]:
        """
        Calculate dimension scores as averages of constituent normalized metrics.

        Args:
            normalized_metrics: Dictionary of normalized metric values

        Returns:
            Dictionary of dimension scores
        """
        dimension_scores = {}

        for dimension, metrics in METRIC_DIMENSIONS.items():
            valid_values = []
            for metric in metrics:
                value = normalized_metrics.get(metric)
                if value is not None:
                    valid_values.append(value)

            if valid_values:
                dimension_scores[dimension] = sum(valid_values) / len(valid_values)
            else:
                dimension_scores[dimension] = None

        return dimension_scores

    def calculate_overall_score(
        self,
        dimension_scores: Dict[str, Optional[float]]
    ) -> Optional[float]:
        """
        Calculate overall maintainability score as average of dimension scores.

        Args:
            dimension_scores: Dictionary of dimension scores

        Returns:
            Overall score in [0, 1], or None if no valid dimensions
        """
        valid_scores = [
            score for score in dimension_scores.values()
            if score is not None
        ]

        if not valid_scores:
            return None

        return sum(valid_scores) / len(valid_scores)

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single metric record: normalize and add dimension scores.

        Output structure:
        - *_normalized: Normalized QUALITY metrics (for dimension scoring)
        - *_control_variables: Normalized CONTROL variables (for stratification)
        - *_dimension_scores: Average scores per quality dimension
        - *_overall_score: Average of dimension scores

        Args:
            record: Raw metric change record

        Returns:
            Enhanced record with normalized metrics, dimension scores, and control variables
        """
        result = record.copy()

        # Process before_metrics
        before_metrics = record.get('before_metrics', {})
        if before_metrics:
            # Quality metrics for dimension scores
            before_normalized = self.normalize_metrics(before_metrics)
            before_dimensions = self.calculate_dimension_scores(before_normalized)
            before_overall = self.calculate_overall_score(before_dimensions)

            result['before_normalized'] = before_normalized
            result['before_dimension_scores'] = before_dimensions
            result['before_overall_score'] = before_overall

            # Control variables (separate from quality dimensions)
            before_control = self.normalize_control_variables(before_metrics)
            result['before_control_variables'] = before_control

        # Process after_metrics
        after_metrics = record.get('after_metrics', {})
        if after_metrics:
            # Quality metrics for dimension scores
            after_normalized = self.normalize_metrics(after_metrics)
            after_dimensions = self.calculate_dimension_scores(after_normalized)
            after_overall = self.calculate_overall_score(after_dimensions)

            result['after_normalized'] = after_normalized
            result['after_dimension_scores'] = after_dimensions
            result['after_overall_score'] = after_overall

            # Control variables (separate from quality dimensions)
            after_control = self.normalize_control_variables(after_metrics)
            result['after_control_variables'] = after_control

        # Calculate delta dimension scores if both before and after exist
        if 'before_dimension_scores' in result and 'after_dimension_scores' in result:
            delta_dimensions = {}
            for dimension in METRIC_DIMENSIONS.keys():
                before_val = result['before_dimension_scores'].get(dimension)
                after_val = result['after_dimension_scores'].get(dimension)
                if before_val is not None and after_val is not None:
                    delta_dimensions[dimension] = after_val - before_val
                else:
                    delta_dimensions[dimension] = None
            result['delta_dimension_scores'] = delta_dimensions

            # Overall delta
            if result.get('before_overall_score') is not None and result.get('after_overall_score') is not None:
                result['delta_overall_score'] = result['after_overall_score'] - result['before_overall_score']

        # Calculate delta control variables if both before and after exist
        if 'before_control_variables' in result and 'after_control_variables' in result:
            delta_control = {}
            for var in ALL_CONTROL_VARIABLES:
                before_val = result['before_control_variables'].get(var)
                after_val = result['after_control_variables'].get(var)
                if before_val is not None and after_val is not None:
                    delta_control[var] = after_val - before_val
                else:
                    delta_control[var] = None
            result['delta_control_variables'] = delta_control

        return result

    def process_repository(self, repo_dir: Path) -> int:
        """
        Process all metrics for a single repository.

        Args:
            repo_dir: Repository results directory

        Returns:
            Number of records processed
        """
        metrics_file = repo_dir / config.VALIDATED_METRICS_FILE
        output_file = repo_dir / config.NORMALIZED_METRICS_FILE

        if not metrics_file.exists():
            logger.warning(f"No metrics file found in {repo_dir}")
            return 0

        processed = 0
        with open(metrics_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    normalized_record = self.process_record(record)
                    outfile.write(json.dumps(normalized_record) + '\n')
                    processed += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {metrics_file}: {e}")
                    continue

        return processed

    def process_all(
        self, metrics_dir: Path, project_filter: Optional[set] = None
    ) -> Dict[str, int]:
        """
        Process all repositories in metrics directory.

        Args:
            metrics_dir: Root metrics directory
            project_filter: Optional set of project directory names to include.
                           If None, all projects are processed.

        Returns:
            Dictionary mapping repo names to processed record counts
        """
        # First collect global statistics
        self.collect_statistics(metrics_dir, project_filter=project_filter)

        # Then process each repository
        results = {}
        repo_dirs = [d for d in metrics_dir.iterdir() if d.is_dir() and d.name != 'reports']

        if project_filter is not None:
            repo_dirs = [d for d in repo_dirs if d.name in project_filter]

        for repo_dir in repo_dirs:
            count = self.process_repository(repo_dir)
            if count > 0:
                results[repo_dir.name] = count
                logger.info(f"Processed {count} records for {repo_dir.name}")

        return results

    def generate_statistics_report(self, metrics_dir: Path) -> None:
        """
        Generate a statistics report showing global min/max and normalization info.

        Args:
            metrics_dir: Root metrics directory
        """
        reports_dir = metrics_dir / 'reports'
        reports_dir.mkdir(exist_ok=True)

        report = {
            'global_statistics': {
                metric: {
                    'min': self.global_min.get(metric),
                    'max': self.global_max.get(metric),
                    'count': self.value_counts.get(metric, 0),
                    'inverted': metric in INVERT_METRICS,
                    'type': 'control' if metric in ALL_CONTROL_VARIABLES else 'quality'
                }
                for metric in NORMALIZABLE_METRICS
                if self.value_counts.get(metric, 0) > 0
            },
            'quality_dimensions': METRIC_DIMENSIONS,
            'control_variables': CONTROL_VARIABLES,
            'inverted_metrics': list(INVERT_METRICS)
        }

        output_file = reports_dir / 'normalization_statistics.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Statistics report saved to {output_file}")


def main():
    """Main entry point for metrics normalization."""
    parser = argparse.ArgumentParser(
        description="Normalize metrics and calculate dimension scores"
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=config.PROCESSED_METRICS_DIR,
        help="Directory containing per-repository metric results"
    )
    parser.add_argument(
        "--selected-projects",
        type=str,
        help="Path to selected_projects.csv to filter by (project_id column, owner/repo format)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.metrics_dir.exists():
        logger.error(f"Metrics directory not found: {args.metrics_dir}")
        sys.exit(1)

    project_filter = None
    if args.selected_projects:
        import csv
        with open(args.selected_projects, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            project_filter = {row["project_id"].replace("/", "_") for row in reader}
        logger.info(f"Loaded {len(project_filter)} projects from {args.selected_projects}")

    # Run normalization
    normalizer = MetricsNormalizer()
    results = normalizer.process_all(args.metrics_dir, project_filter=project_filter)

    # Generate statistics report
    normalizer.generate_statistics_report(args.metrics_dir)

    # Summary
    total_records = sum(results.values())
    logger.info(f"\nNormalization complete:")
    logger.info(f"  Repositories processed: {len(results)}")
    logger.info(f"  Total records normalized: {total_records}")
    logger.info(f"  Output files: {config.NORMALIZED_METRICS_FILE} in each repo directory")


if __name__ == "__main__":
    main()
