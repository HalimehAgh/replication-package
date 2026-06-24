"""
Parallel Runner for Historical Smell Detection

Runs historical smell detection across all projects in parallel with:
- Progress tracking with ETA
- Parallel processing with configurable workers
- Resume capability (skips completed projects)
- Combined output generation
- Error logging and statistics

Usage:
    python -m src.step2_smell_detection.parallel_runner [--parallel N] [--status]
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src import config
from src.utils import setup_logger
from src.utils.file_utils import repo_to_dir_name

logger = setup_logger(__name__)


class ParallelRunner:
    """Orchestrates parallel smell detection across all projects."""

    def __init__(self, max_parallel: int = 4):
        """
        Initialize runner.

        Args:
            max_parallel: Maximum number of parallel workers
        """
        self.max_parallel = max_parallel
        self.project_root = config.PROJECT_ROOT
        self.logs_dir = self.project_root / "logs" / "smell_detection"
        self.output_dir = config.RAW_SMELL_DETECTION_DIR

        # Create directories
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_repos(self) -> pd.DataFrame:
        """
        Load repository list with domain and size information.

        Returns:
            DataFrame with repo_name, domain, size_category
        """
        # Try unified dataset first
        if (config.DATA_EXTERNAL_DIR / "repos_279_unified.csv").exists():
            filepath = config.DATA_EXTERNAL_DIR / "repos_279_unified.csv"
        elif config.REPOS_CSV.exists():
            filepath = config.REPOS_CSV
        else:
            raise FileNotFoundError(
                "Repository dataset not found. Run prepare_dataset.py first."
            )

        logger.info(f"Loading repositories from: {filepath}")
        df = pd.read_csv(filepath)

        # Normalize columns
        column_mapping = {
            "GitHub Repo": "repo_name",
            "github_repo": "repo_name",
            "Domain": "domain",
            "Size Category": "size_category",
            "size": "size_category",
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        logger.info(f"Loaded {len(df)} repositories")
        return df

    def is_already_analyzed(self, project_id: str) -> bool:
        """Check if a project has already been analyzed."""
        repo_dir = self.output_dir / repo_to_dir_name(project_id)
        metrics_file = repo_dir / config.METRICS_FILE
        return metrics_file.exists()

    def get_repo_path(self, project_id: str, domain: str) -> Path:
        """Get the path to a cloned repository."""
        domain_abbrev = config.DOMAIN_ABBREVIATIONS.get(domain, "other")
        domain_path = config.REPOS_DIR / domain_abbrev / repo_to_dir_name(project_id)

        if domain_path.exists():
            return domain_path

        # Fallback to flat structure
        flat_path = config.REPOS_DIR / repo_to_dir_name(project_id)
        if flat_path.exists():
            return flat_path

        return None

    def run_analysis(
        self,
        project_id: str,
        domain: str,
        size_category: str
    ) -> Tuple[str, bool, str, float]:
        """
        Run analysis for a single project.

        Returns:
            (project_id, success, message, duration)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"{repo_to_dir_name(project_id)}_{timestamp}.log"
        start_time = time.time()

        try:
            # Find repo path
            repo_path = self.get_repo_path(project_id, domain)
            if repo_path is None:
                return (project_id, False, "Repository not cloned", 0)

            logger.info(f"[START] {project_id}")

            # Run analysis as subprocess
            cmd = [
                sys.executable,
                "-m",
                "src.step2_smell_detection.historical_smell_detection",
                "--repo", project_id,
                "--repo-path", str(repo_path),
                "--domain", domain,
                "--size", size_category,
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=None,  # No global timeout - per-commit timeout handled internally
            )

            # Save log
            with open(log_file, "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write("=" * 80 + "\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n" + "=" * 80 + "\n")
                f.write("STDERR:\n")
                f.write(result.stderr)

            duration = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"[DONE] {project_id} ({duration:.1f}s)")
                return (project_id, True, "Success", duration)
            else:
                logger.error(f"[FAIL] {project_id}")
                return (project_id, False, f"Exit code {result.returncode}", duration)

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.warning(f"[TIMEOUT] {project_id}")
            return (project_id, False, "Global timeout", duration)

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[ERROR] {project_id}: {e}")
            return (project_id, False, str(e)[:200], duration)

    def run_parallel(
        self,
        repos_df: pd.DataFrame,
        skip_analyzed: bool = True,
        auto_confirm: bool = False
    ) -> List[Dict]:
        """
        Run analysis for all projects in parallel.

        Args:
            repos_df: DataFrame with repo_name, domain, size_category
            skip_analyzed: Skip already analyzed projects
            auto_confirm: Skip confirmation prompt

        Returns:
            List of result dictionaries
        """
        # Determine which repos need analysis
        repos_to_analyze = []
        already_analyzed = []
        not_cloned = []

        for _, row in repos_df.iterrows():
            project_id = row["repo_name"]
            domain = row.get("domain", "")

            if skip_analyzed and self.is_already_analyzed(project_id):
                already_analyzed.append(project_id)
            elif self.get_repo_path(project_id, domain) is None:
                not_cloned.append(project_id)
            else:
                repos_to_analyze.append({
                    "project_id": project_id,
                    "domain": domain,
                    "size_category": row.get("size_category", ""),
                })

        # Log status
        logger.info("=" * 80)
        logger.info("PARALLEL ANALYSIS STATUS")
        logger.info("=" * 80)
        logger.info(f"Total projects: {len(repos_df)}")
        logger.info(f"Already analyzed: {len(already_analyzed)}")
        logger.info(f"Not cloned: {len(not_cloned)}")
        logger.info(f"To analyze: {len(repos_to_analyze)}")
        logger.info(f"Parallel workers: {self.max_parallel}")

        if not repos_to_analyze:
            logger.info("No projects to analyze!")
            return []

        # Confirmation
        if not auto_confirm:
            response = input(f"\nAnalyze {len(repos_to_analyze)} projects? (y/n): ").strip().lower()
            if response != "y":
                logger.info("Analysis cancelled.")
                return []

        # Run in parallel
        logger.info("")
        logger.info("Starting parallel analysis...")
        start_time = time.time()

        results = []
        completed = 0
        failed = 0
        total_events = 0

        with ProcessPoolExecutor(max_workers=self.max_parallel) as executor:
            # Submit all jobs
            future_to_repo = {
                executor.submit(
                    self.run_analysis,
                    repo["project_id"],
                    repo["domain"],
                    repo["size_category"]
                ): repo
                for repo in repos_to_analyze
            }

            # Process as completed
            for future in as_completed(future_to_repo):
                repo = future_to_repo[future]

                try:
                    project_id, success, message, duration = future.result()

                    result = {
                        "project_id": project_id,
                        "domain": repo["domain"],
                        "size_category": repo["size_category"],
                        "success": success,
                        "message": message,
                        "duration_seconds": round(duration, 2),
                    }

                    # Load event counts if successful
                    if success:
                        metrics_file = self.output_dir / repo_to_dir_name(project_id) / config.METRICS_FILE
                        if metrics_file.exists():
                            metrics = json.load(open(metrics_file))
                            result["introductions"] = metrics.get("total_introductions", 0)
                            result["removals"] = metrics.get("total_removals", 0)
                            total_events += result["introductions"] + result["removals"]

                    results.append(result)
                    completed += 1

                    if not success:
                        failed += 1

                    # Progress update
                    if completed % 5 == 0 or completed == len(repos_to_analyze):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(repos_to_analyze) - completed) / rate / 60 if rate > 0 else 0

                        logger.info(
                            f"Progress: {completed}/{len(repos_to_analyze)} "
                            f"({failed} failed) | "
                            f"Events: {total_events} | "
                            f"ETA: {eta:.1f} min"
                        )

                except Exception as e:
                    logger.error(f"Error with {repo['project_id']}: {e}")
                    results.append({
                        "project_id": repo["project_id"],
                        "domain": repo["domain"],
                        "size_category": repo["size_category"],
                        "success": False,
                        "message": str(e)[:200],
                        "duration_seconds": 0,
                    })
                    completed += 1
                    failed += 1

        # Final summary
        total_time = time.time() - start_time
        logger.info("")
        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Projects analyzed: {completed - failed}")
        logger.info(f"Projects failed: {failed}")
        logger.info(f"Total events detected: {total_events}")

        return results

    def generate_combined_outputs(self, repos_df: pd.DataFrame) -> None:
        """
        Generate combined CSV files from all analyzed projects.

        Creates:
        - Combined events CSV: All events from all projects
        - Project summary CSV: Per-project summary
        - error_log.txt: Consolidated error log from all projects
        """
        logger.info("")
        logger.info("Generating combined outputs...")

        all_events = []
        project_summaries = []
        all_errors = []

        for _, row in repos_df.iterrows():
            project_id = row["repo_name"]
            repo_dir = self.output_dir / repo_to_dir_name(project_id)
            events_file = repo_dir / config.EVENTS_FILE
            metrics_file = repo_dir / config.METRICS_FILE

            # Load events
            if events_file.exists():
                with open(events_file, "r") as f:
                    for line in f:
                        if line.strip():
                            event = json.loads(line)
                            all_events.append(event)

            # Load metrics for summary
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                summary = {
                    "project_id": project_id,
                    "domain": row.get("domain", ""),
                    "size_category": row.get("size_category", ""),
                    "total_introductions": metrics.get("total_introductions", 0),
                    "total_removals": metrics.get("total_removals", 0),
                    "total_events": metrics.get("total_introductions", 0) + metrics.get("total_removals", 0),
                    "commits_analyzed": metrics.get("commits_analyzed", 0),
                    "files_analyzed": metrics.get("files_analyzed", 0),
                    "date_range_start": metrics.get("date_range_start", ""),
                    "date_range_end": metrics.get("date_range_end", ""),
                }

                # Add per-smell counts
                for smell_type, count in metrics.get("smell_introductions", {}).items():
                    summary[f"intro_{smell_type}"] = count
                for smell_type, count in metrics.get("smell_removals", {}).items():
                    summary[f"removal_{smell_type}"] = count

                project_summaries.append(summary)

                # Collect errors
                for error in metrics.get("errors", []):
                    all_errors.append({
                        "project_id": project_id,
                        **error
                    })

        # Save combined events
        if all_events:
            events_df = pd.DataFrame(all_events)
            events_df.to_csv(config.EVENTS_ALL_CSV, index=False)
            logger.info(f"Saved {len(events_df)} events to: {config.EVENTS_ALL_CSV}")

        # Save project summary
        if project_summaries:
            summary_df = pd.DataFrame(project_summaries)
            summary_df.to_csv(config.PROJECT_SUMMARY_CSV, index=False)
            logger.info(f"Saved {len(summary_df)} project summaries to: {config.PROJECT_SUMMARY_CSV}")

        # Generate consolidated error log
        self._generate_error_log(all_errors)

        # Generate statistics
        self._generate_statistics(all_events, project_summaries)

    def _generate_statistics(
        self,
        all_events: List[Dict],
        project_summaries: List[Dict]
    ) -> None:
        """Generate detection statistics report."""
        lines = [
            "=" * 80,
            "PROJECT SMELL DETECTION STATISTICS",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 80,
            "",
            f"Total projects analyzed: {len(project_summaries)}",
            f"Total events detected: {len(all_events)}",
            "",
        ]

        if all_events:
            events_df = pd.DataFrame(all_events)

            # Event type breakdown
            intro_count = len(events_df[events_df["event_type"] == "introduction"])
            removal_count = len(events_df[events_df["event_type"] == "removal"])
            lines.append(f"Introduction events: {intro_count}")
            lines.append(f"Removal events: {removal_count}")
            lines.append("")

            # Events by smell type
            lines.append("EVENTS BY SMELL TYPE:")
            smell_counts = events_df["smell_type"].value_counts()
            for smell, count in smell_counts.items():
                meets_threshold = "YES" if count >= config.MIN_EVENTS_PER_SMELL else "NO"
                lines.append(f"  {smell}: {count} (>=30: {meets_threshold})")

            lines.append("")

            # Events by domain
            lines.append("EVENTS BY DOMAIN:")
            domain_counts = events_df["domain"].value_counts()
            for domain, count in domain_counts.items():
                lines.append(f"  {domain}: {count}")

            lines.append("")

            # Events by size
            lines.append("EVENTS BY SIZE CATEGORY:")
            size_counts = events_df["size_category"].value_counts()
            for size, count in size_counts.items():
                lines.append(f"  {size}: {count}")

        # Save statistics
        with open(config.DETECTION_STATS, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Saved statistics to: {config.DETECTION_STATS}")

    def _generate_error_log(self, all_errors: List[Dict]) -> None:
        """
        Generate consolidated error log from all projects.

        As required by PDF specification, this consolidates:
        - Projects that failed completely
        - Commits that failed (timeout, parsing errors)
        - Files that failed (syntax errors, CodeSmile crashes)
        """
        lines = [
            "=" * 80,
            "PROJECT SMELL DETECTION ERROR LOG",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 80,
            "",
            f"Total errors recorded: {len(all_errors)}",
            "",
        ]

        if not all_errors:
            lines.append("No errors recorded.")
        else:
            # Group errors by type
            error_types = {}
            for error in all_errors:
                error_type = error.get("type", "unknown")
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(error)

            lines.append("ERRORS BY TYPE:")
            for error_type, count in sorted(
                [(t, len(e)) for t, e in error_types.items()],
                key=lambda x: -x[1]
            ):
                lines.append(f"  {error_type}: {count}")

            lines.append("")

            # Detail each error type
            for error_type, errors in error_types.items():
                lines.append("-" * 80)
                lines.append(f"{error_type.upper()} ERRORS ({len(errors)})")
                lines.append("-" * 80)

                for error in errors[:50]:  # Limit to 50 per type
                    project = error.get("project_id", "unknown")
                    commit = error.get("commit", "N/A")[:8] if error.get("commit") else "N/A"
                    file_path = error.get("file", "N/A")
                    message = error.get("message", "No message")[:200]

                    lines.append(f"  Project: {project}")
                    if commit != "N/A":
                        lines.append(f"    Commit: {commit}")
                    if file_path != "N/A":
                        lines.append(f"    File: {file_path}")
                    lines.append(f"    Message: {message}")
                    lines.append("")

                if len(errors) > 50:
                    lines.append(f"  ... and {len(errors) - 50} more {error_type} errors")
                    lines.append("")

        # Save error log
        with open(config.DETECTION_ERROR_LOG, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Saved error log ({len(all_errors)} errors) to: {config.DETECTION_ERROR_LOG}")

    def print_status(self, repos_df: pd.DataFrame) -> None:
        """Print current analysis status."""
        analyzed = []
        not_analyzed = []
        not_cloned = []

        for _, row in repos_df.iterrows():
            project_id = row["repo_name"]
            domain = row.get("domain", "")

            if self.is_already_analyzed(project_id):
                analyzed.append(project_id)
            elif self.get_repo_path(project_id, domain) is None:
                not_cloned.append(project_id)
            else:
                not_analyzed.append(project_id)

        logger.info("=" * 80)
        logger.info("ANALYSIS STATUS")
        logger.info("=" * 80)
        logger.info(f"Total projects: {len(repos_df)}")
        logger.info(f"Analyzed: {len(analyzed)}")
        logger.info(f"Not analyzed (cloned): {len(not_analyzed)}")
        logger.info(f"Not cloned: {len(not_cloned)}")

        if not_cloned:
            logger.info("")
            logger.info("Projects not cloned:")
            for repo in not_cloned[:10]:
                logger.info(f"  - {repo}")
            if len(not_cloned) > 10:
                logger.info(f"  ... and {len(not_cloned) - 10} more")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run historical smell detection on all projects"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=config.MAX_PARALLEL_WORKERS,
        help=f"Number of parallel workers (default: {config.MAX_PARALLEL_WORKERS})"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show analysis status without running"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-analyze already analyzed projects"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Only generate combined outputs from existing results"
    )

    args = parser.parse_args()

    runner = ParallelRunner(max_parallel=args.parallel)

    try:
        repos_df = runner.load_repos()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Status only
    if args.status:
        runner.print_status(repos_df)
        return

    # Combine only
    if args.combine_only:
        runner.generate_combined_outputs(repos_df)
        return

    # Run analysis
    results = runner.run_parallel(
        repos_df,
        skip_analyzed=not args.force,
        auto_confirm=args.yes
    )

    # Save results log
    if results:
        results_df = pd.DataFrame(results)
        results_file = config.RAW_SMELL_DETECTION_DIR / "analysis_results.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Saved results log: {results_file}")

    # Generate combined outputs
    runner.generate_combined_outputs(repos_df)


if __name__ == "__main__":
    main()
