"""
Repository Commit Analysis with CodeSmile

Analyzes ML repositories for code smells across commits, tracking when smells
are introduced and removed. Uses CodeSmile for static smell detection and
PyDriller for efficient git operations.

Usage:
    python -m src.step2_smell_detection.run_codesmile [--repo REPO_NAME] [--resume] [--output-dir DIR]
"""

import argparse
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
from pydriller import Repository

# Import project modules
from src import config
from src.utils import (
    setup_logger,
    read_json,
    write_json,
    write_jsonl,
    parse_repo_name,
    get_repo_url,
    get_repo_path,
    get_repo_output_dir,
)
from src.utils.codesmile_utils import create_inspector
from src.utils.reporters import AnalysisReporter
from src.step2_smell_detection.smell_parser import ML_SMELL_TYPES

# Setup logging
logger = setup_logger(__name__)


class SmellEvent:
    """Represents a smell introduction or removal event."""

    def __init__(
        self,
        smell_instance_id: str,
        project_name: str,
        smell_type: str,
        commit_hash: str,
        commit_type: str,  # A-1, A, B-1, B
        file_path: str,
        function_id: str,
        timestamp: str,
        author: str,
        is_smelly: bool,
        lines_added: int,
        lines_deleted: int,
        commit_message: str,
    ):
        self.smell_instance_id = smell_instance_id
        self.project_name = project_name
        self.smell_type = smell_type
        self.commit_hash = commit_hash
        self.commit_type = commit_type
        self.file_path = file_path
        self.function_id = function_id
        self.timestamp = timestamp
        self.author = author
        self.is_smelly = is_smelly
        self.lines_added = lines_added
        self.lines_deleted = lines_deleted
        self.commit_message = commit_message

        # Enrich with metadata from ML_SMELL_TYPES
        smell_info = ML_SMELL_TYPES.get(smell_type, {})
        self.category = smell_info.get('category', 'ml_specific')
        self.severity = smell_info.get('severity', 'medium')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "smell_instance_id": self.smell_instance_id,
            "project_name": self.project_name,
            "smell_type": self.smell_type,
            "category": self.category,
            "severity": self.severity,
            "commit_hash": self.commit_hash,
            "commit_type": self.commit_type,
            "file_path": self.file_path,
            "function_id": self.function_id,
            "timestamp": self.timestamp,
            "author": self.author,
            "is_smelly": self.is_smelly,
            "lines_added": self.lines_added,
            "lines_deleted": self.lines_deleted,
            "commit_message": self.commit_message,
        }


class CommitAnalyzer:
    """Analyzes repository commits for code smells."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize CommitAnalyzer.

        Args:
            output_dir: Directory for analysis outputs (defaults to config.RAW_SMELL_DETECTION_DIR)
        """
        self.output_dir = output_dir or config.RAW_SMELL_DETECTION_DIR
        self.inspector = None
        self.temp_dir = None

    def _init_inspector(self) -> None:
        """Initialize CodeSmile inspector with temp directory."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="codesmile_")
        if self.inspector is None:
            self.inspector = create_inspector(self.temp_dir)

    def _cleanup(self) -> None:
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def load_projects(
        self, csv_path: str, repo_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load projects from selection CSV."""
        df = pd.read_csv(csv_path)

        if repo_filter:
            # Filter by repository names
            df = df[
                df["GitHub Repo"].apply(
                    lambda x: any(
                        filter_name.lower() in x.lower() for filter_name in repo_filter
                    )
                )
            ]

        if df.empty:
            raise ValueError(f"No repositories found matching filter: {repo_filter}")

        return df

    def clone_or_update_repo(self, repo_url: str, target_dir: str) -> str:
        """Clone repository if not exists, otherwise return path."""
        if os.path.exists(target_dir):
            logger.info(f"Repository already exists: {target_dir}")
            return target_dir

        logger.info(f"Cloning repository: {repo_url}")
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)

        import git

        git.Repo.clone_from(repo_url, target_dir)
        return target_dir

    def get_repo_output_dir(self, repo_name: str) -> Path:
        """Get output directory for a specific repository."""
        return get_repo_output_dir(repo_name, self.output_dir)

    def analyze_file_for_smells(self, file_path: str) -> List[Dict[str, Any]]:
        """Analyze a single file for code smells using CodeSmile."""
        self._init_inspector()

        try:
            result_df = self.inspector.inspect(file_path)

            if result_df.empty:
                return []

            # Convert DataFrame to list of dicts
            smells = []
            for _, row in result_df.iterrows():
                smells.append(
                    {
                        "filename": row.get("filename", file_path),
                        "function_name": row.get("function_name", ""),
                        "smell_name": row.get("smell_name", ""),
                        "line": row.get("line", -1),
                        "description": row.get("description", ""),
                        "additional_info": row.get("additional_info", ""),
                    }
                )

            return smells

        except Exception as e:
            logger.debug(f"Error analyzing file {file_path}: {e}")
            return []

    def create_smell_key(self, smell: Dict[str, Any]) -> str:
        """Create unique key for a smell instance.

        Note: We don't include line numbers because smells can move when code
        is refactored. A smell in a specific function should be tracked as the
        same instance even if its line number changes.
        """
        return f"{smell['filename']}:{smell['function_name']}:{smell['smell_name']}"

    def compare_smells(
        self, prev_smells: List[Dict[str, Any]], curr_smells: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Compare smells between two commits.

        Returns:
            (introduced_smells, removed_smells)
        """
        prev_keys = {self.create_smell_key(s): s for s in prev_smells}
        curr_keys = {self.create_smell_key(s): s for s in curr_smells}

        introduced = [curr_keys[k] for k in curr_keys if k not in prev_keys]
        removed = [prev_keys[k] for k in prev_keys if k not in curr_keys]

        return introduced, removed

    def save_jsonl(self, data: List[Dict[str, Any]], filepath: Path) -> None:
        """Save data as JSONL (one JSON object per line)."""
        write_jsonl(data, Path(filepath), append=True)

    def _record_smell_introduction(
        self,
        smell: Dict[str, Any],
        repo_name: str,
        file_path: str,
        commit,
        prev_commit_hash: Optional[str],
        events_file: Path,
        metrics: Dict,
    ) -> None:
        """
        Record smell introduction events (A-1 and A).

        Args:
            smell: Smell dictionary with metadata
            repo_name: Repository name
            file_path: File where smell was introduced
            commit: Current commit object
            prev_commit_hash: Previous commit hash (for A-1 event)
            events_file: Path to events output file
            metrics: Metrics dictionary to update
        """
        smell_id = f"{repo_name}-{smell['smell_name']}-{file_path}-{smell['function_name']}".replace(
            "/", "-"
        ).replace(" ", "_")

        # A-1: commit before introduction (no smell)
        if prev_commit_hash:
            event_a_minus_1 = SmellEvent(
                smell_instance_id=smell_id,
                project_name=repo_name,
                smell_type=smell["smell_name"],
                commit_hash=prev_commit_hash,
                commit_type="A-1",
                file_path=file_path,
                function_id=smell["function_name"],
                timestamp=commit.author_date.isoformat(),
                author=commit.author.name,
                is_smelly=False,
                lines_added=commit.insertions,
                lines_deleted=commit.deletions,
                commit_message=commit.msg,
            )
            self.save_jsonl([event_a_minus_1.to_dict()], events_file)

        # A: commit with introduction (has smell)
        event_a = SmellEvent(
            smell_instance_id=smell_id,
            project_name=repo_name,
            smell_type=smell["smell_name"],
            commit_hash=commit.hash,
            commit_type="A",
            file_path=file_path,
            function_id=smell["function_name"],
            timestamp=commit.author_date.isoformat(),
            author=commit.author.name,
            is_smelly=True,
            lines_added=commit.insertions,
            lines_deleted=commit.deletions,
            commit_message=commit.msg,
        )
        self.save_jsonl([event_a.to_dict()], events_file)
        metrics["smell_introductions"][smell["smell_name"]] += 1

    def _record_smell_removal(
        self,
        smell: Dict[str, Any],
        repo_name: str,
        file_path: str,
        commit,
        prev_commit_hash: Optional[str],
        events_file: Path,
        metrics: Dict,
    ) -> None:
        """
        Record smell removal events (B-1 and B).

        Args:
            smell: Smell dictionary with metadata
            repo_name: Repository name
            file_path: File where smell was removed
            commit: Current commit object
            prev_commit_hash: Previous commit hash (for B-1 event)
            events_file: Path to events output file
            metrics: Metrics dictionary to update
        """
        smell_id = f"{repo_name}-{smell['smell_name']}-{file_path}-{smell['function_name']}".replace(
            "/", "-"
        ).replace(" ", "_")

        # B-1: commit before removal (has smell)
        if prev_commit_hash:
            event_b_minus_1 = SmellEvent(
                smell_instance_id=smell_id,
                project_name=repo_name,
                smell_type=smell["smell_name"],
                commit_hash=prev_commit_hash,
                commit_type="B-1",
                file_path=file_path,
                function_id=smell["function_name"],
                timestamp=commit.author_date.isoformat(),
                author=commit.author.name,
                is_smelly=True,
                lines_added=commit.insertions,
                lines_deleted=commit.deletions,
                commit_message=commit.msg,
            )
            self.save_jsonl([event_b_minus_1.to_dict()], events_file)

        # B: commit with removal (no smell)
        event_b = SmellEvent(
            smell_instance_id=smell_id,
            project_name=repo_name,
            smell_type=smell["smell_name"],
            commit_hash=commit.hash,
            commit_type="B",
            file_path=file_path,
            function_id=smell["function_name"],
            timestamp=commit.author_date.isoformat(),
            author=commit.author.name,
            is_smelly=False,
            lines_added=commit.insertions,
            lines_deleted=commit.deletions,
            commit_message=commit.msg,
        )
        self.save_jsonl([event_b.to_dict()], events_file)
        metrics["smell_removals"][smell["smell_name"]] += 1

    def _process_smell_changes(
        self,
        curr_commit_smells: Dict[str, List],
        prev_commit_smells: Dict[str, List],
        repo_name: str,
        commit,
        prev_commit_hash: Optional[str],
        events_file: Path,
        metrics: Dict,
    ) -> None:
        """
        Process smell changes between commits and record events.

        Args:
            curr_commit_smells: Smells in current commit
            prev_commit_smells: Smells in previous commit
            repo_name: Repository name
            commit: Current commit object
            prev_commit_hash: Previous commit hash
            events_file: Path to events output file
            metrics: Metrics dictionary to update
        """
        # Only compare files that were modified in this commit
        for file_path in curr_commit_smells.keys():
            prev_smells = prev_commit_smells.get(file_path, [])
            curr_smells = curr_commit_smells.get(file_path, [])

            introduced, removed = self.compare_smells(prev_smells, curr_smells)

            # Record smell introductions
            for smell in introduced:
                self._record_smell_introduction(
                    smell,
                    repo_name,
                    file_path,
                    commit,
                    prev_commit_hash,
                    events_file,
                    metrics,
                )

            # Record smell removals
            for smell in removed:
                self._record_smell_removal(
                    smell,
                    repo_name,
                    file_path,
                    commit,
                    prev_commit_hash,
                    events_file,
                    metrics,
                )

    # ============================================================
    # REPOSITORY ANALYSIS - INITIALIZATION
    # ============================================================

    def _initialize_analysis_metrics(self, repo_name: str) -> Dict[str, Any]:
        """
        Create empty metrics dictionary for tracking analysis.

        Returns:
            Dictionary with initialized metric counters
        """
        return {
            "repo_name": repo_name,
            "total_commits": 0,
            "commits_with_python": 0,
            "commits_skipped": 0,
            "commits_analyzed": 0,
            "files_analyzed": 0,
            "total_time": 0.0,
            "smell_introductions": defaultdict(int),
            "smell_removals": defaultdict(int),
        }

    def _setup_output_files(self, repo_name: str) -> Dict[str, Path]:
        """
        Create output directory and return paths to all output files.

        Args:
            repo_name: Repository name for creating output directory

        Returns:
            Dictionary mapping file types to their paths
        """
        repo_output_dir = self.get_repo_output_dir(repo_name)

        # Note: commits are now saved separately in data/raw/commit_history/
        # by the extract_commits module (Step 1)
        return {
            "smells": repo_output_dir / config.SMELLS_FILE,
            "events": repo_output_dir / config.EVENTS_FILE,
            "checkpoint": repo_output_dir / config.CHECKPOINT_FILE,
            "metrics": repo_output_dir / config.METRICS_FILE,
        }

    # ============================================================
    # REPOSITORY ANALYSIS - RESUME LOGIC
    # ============================================================

    def _check_resume_point(self, resume: bool, checkpoint_file: Path) -> Optional[str]:
        """
        Check if we should resume from a previous run.

        Args:
            resume: Whether resume mode is enabled
            checkpoint_file: Path to checkpoint file

        Returns:
            Last analyzed commit hash if resuming, None otherwise
        """
        if resume and checkpoint_file.exists():
            checkpoint = read_json(checkpoint_file)
            last_analyzed_hash = checkpoint.get("last_commit_hash")
            logger.info(f"Resuming from commit: {last_analyzed_hash}")
            return last_analyzed_hash
        return None

    def _clear_previous_results(self, output_files: Dict[str, Path]) -> None:
        """
        Delete previous run's output files (when not resuming).

        Args:
            output_files: Dictionary of output file paths
        """
        for file_type in ["smells", "events", "checkpoint"]:
            file_path = output_files.get(file_type)
            if file_path and file_path.exists():
                file_path.unlink()

    def _get_total_commit_count(self, repo_path: str) -> Any:
        """
        Get total commit count for progress tracking.

        Args:
            repo_path: Path to the git repository

        Returns:
            Total commit count (int) or "unknown" if unavailable
        """
        try:
            import git

            repo_git = git.Repo(repo_path)
            total_commits = len(list(repo_git.iter_commits()))
            logger.info(f"Total commits in repository: {total_commits}")
            return total_commits
        except Exception as e:
            logger.warning(f"Could not get total commit count: {e}")
            return "unknown"

    # ============================================================
    # REPOSITORY ANALYSIS - PROGRESS & STATISTICS
    # ============================================================

    def _log_analysis_progress(
        self, metrics: Dict[str, Any], start_time: float, total_commits_in_repo: Any
    ) -> None:
        """
        Log progress every N commits.

        Args:
            metrics: Current metrics dictionary
            start_time: Analysis start timestamp
            total_commits_in_repo: Total commits (int or "unknown")
        """
        elapsed = time.time() - start_time
        rate = metrics["commits_analyzed"] / elapsed if elapsed > 0 else 0

        # Calculate progress percentage if total is known
        if isinstance(total_commits_in_repo, int):
            progress_pct = (metrics["total_commits"] / total_commits_in_repo) * 100
            total_str = f"{metrics['total_commits']}/{total_commits_in_repo} ({progress_pct:.1f}%)"
        else:
            total_str = f"{metrics['total_commits']}"

        logger.info(
            f"Progress: {metrics['commits_analyzed']} commits analyzed | "
            f"{total_str} total processed | "
            f"{metrics['commits_skipped']} skipped (no Python) | "
            f"{rate:.2f} commits/sec | "
            f"Elapsed: {elapsed:.1f}s"
        )

    def _calculate_final_statistics(
        self, metrics: Dict[str, Any], start_time: float
    ) -> Dict[str, Any]:
        """
        Calculate summary statistics after analysis completes.

        Args:
            metrics: Metrics dictionary with counters
            start_time: Analysis start timestamp

        Returns:
            Updated metrics with calculated statistics
        """
        metrics["total_time"] = time.time() - start_time
        metrics["avg_files_per_commit"] = (
            metrics["files_analyzed"] / metrics["commits_analyzed"]
            if metrics["commits_analyzed"] > 0
            else 0
        )
        metrics["avg_time_per_commit"] = (
            metrics["total_time"] / metrics["commits_analyzed"]
            if metrics["commits_analyzed"] > 0
            else 0
        )

        # Convert defaultdicts to regular dicts for JSON serialization
        metrics["smell_introductions"] = dict(metrics["smell_introductions"])
        metrics["smell_removals"] = dict(metrics["smell_removals"])

        # Log final statistics
        logger.info(f"Completed analysis for {metrics['repo_name']}")
        logger.info(f"Total commits: {metrics['total_commits']}")
        logger.info(f"Commits with Python changes: {metrics['commits_with_python']}")
        logger.info(f"Commits analyzed: {metrics['commits_analyzed']}")
        logger.info(f"Files analyzed: {metrics['files_analyzed']}")
        logger.info(f"Total time: {metrics['total_time']:.2f}s")

        return metrics

    def _save_analysis_results(
        self, metrics: Dict[str, Any], output_files: Dict[str, Path]
    ) -> None:
        """
        Save metrics JSON to output directory.

        Args:
            metrics: Final metrics dictionary
            output_files: Dictionary of output file paths
        """
        write_json(metrics, output_files["metrics"])

    def _process_all_commits(
        self,
        repo_path: str,
        repo_name: str,
        resume_from_hash: Optional[str],
        no_skip: bool,
        output_files: Dict[str, Path],
        metrics: Dict[str, Any],
        start_time: float,
    ) -> Dict[str, Any]:
        """
        Main commit traversal loop.

        Processes each commit in the repository, tracking smell introductions/removals.

        Args:
            repo_path: Path to git repository
            repo_name: Repository name
            resume_from_hash: Hash to resume from (or None)
            no_skip: Whether to analyze all commits (even without Python changes)
            output_files: Dictionary of output file paths
            metrics: Metrics dictionary to update
            start_time: Analysis start timestamp

        Returns:
            Updated metrics dictionary
        """
        # Get total commit count for progress tracking
        total_commits_in_repo = self._get_total_commit_count(repo_path)

        smells_before_commit = {}  # file_path -> List[smell]
        previous_commit_hash = None
        skip_until_resume = resume_from_hash is not None

        # Traverse commits using PyDriller
        for commit in Repository(str(repo_path)).traverse_commits():
            metrics["total_commits"] += 1

            # Skip until we reach the resume point
            if skip_until_resume:
                if commit.hash == resume_from_hash:
                    skip_until_resume = False
                continue

            # Check if commit modifies Python files
            python_files = [
                modified_file.new_path
                for modified_file in commit.modified_files
                if modified_file.new_path and modified_file.new_path.endswith(".py")
            ]

            # Skip commits without Python files unless --no-skip is enabled
            if not python_files and not no_skip:
                metrics["commits_skipped"] += 1
                continue

            if python_files:
                metrics["commits_with_python"] += 1
            else:
                # no_skip is True but no Python files - still track but separately
                metrics["commits_skipped"] += 1

            # Analyze Python files in this commit
            # Note: commit metadata is saved by extract_commits (Step 1)
            smells_after_commit = {}

            if python_files:
                # Analyze modified Python files
                for modified_file in commit.modified_files:
                    if (
                        not modified_file.new_path
                        or not modified_file.new_path.endswith(".py")
                    ):
                        continue

                    # Get file content at this commit
                    if modified_file.source_code is None:
                        # File was deleted
                        smells_after_commit[modified_file.new_path] = []
                        continue

                    # Write file to temp location for analysis
                    temp_file_path = os.path.join(self.temp_dir, "temp_analysis.py")
                    with open(temp_file_path, "w", encoding="utf-8") as f:
                        f.write(modified_file.source_code)

                    # Analyze for smells
                    smells = self.analyze_file_for_smells(temp_file_path)

                    # Update file path in smell results
                    for smell in smells:
                        smell["filename"] = modified_file.new_path

                    smells_after_commit[modified_file.new_path] = smells
                    metrics["files_analyzed"] += 1
            else:
                # No Python modifications - with --no-skip, inherit previous state
                # This validates that smell state doesn't change without Python modifications
                smells_after_commit = {
                    k: list(v) for k, v in smells_before_commit.items()
                }

            # Save smells for this commit
            smell_commit_record = {
                "commit_hash": commit.hash,
                "timestamp": commit.author_date.isoformat(),
                "smells": smells_after_commit,
            }
            self.save_jsonl([smell_commit_record], output_files["smells"])

            # Process smell changes and record events
            self._process_smell_changes(
                smells_after_commit,
                smells_before_commit,
                repo_name,
                commit,
                previous_commit_hash,
                output_files["events"],
                metrics,
            )

            # Update previous commit data
            # Merge smells_after_commit into smells_before_commit to maintain state
            # of all files, not just modified ones
            smells_before_commit.update(smells_after_commit)
            previous_commit_hash = commit.hash
            metrics["commits_analyzed"] += 1

            # Save checkpoint
            write_json(
                {"last_commit_hash": commit.hash}, output_files["checkpoint"]
            )

            # Progress update
            if metrics["commits_analyzed"] % config.PROGRESS_UPDATE_INTERVAL == 0:
                self._log_analysis_progress(metrics, start_time, total_commits_in_repo)

        return metrics

    # ============================================================
    # REPOSITORY ANALYSIS - MAIN WORKFLOW
    # ============================================================

    def analyze_repository(
        self,
        repo_path: str,
        repo_name: str,
        resume: bool = False,
        no_skip: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze commits in a repository for code smells.

        Workflow:
        1. Initialize metrics and output files
        2. Check resume logic
        3. Process all commits (track smell changes)
        4. Calculate summary statistics
        5. Save results

        Parameters:
        - repo_path: Path to the git repository
        - repo_name: Name of the repository
        - resume: Resume from last checkpoint
        - no_skip: If True, analyze ALL commits (even without Python changes).
                   Use this to validate that skipping doesn't miss smell changes.

        Returns:
            Dictionary containing analysis metrics
        """
        logger.info("=" * 80)
        logger.info(f"Analyzing repository: {repo_name}")
        if no_skip:
            logger.info("Mode: NO-SKIP (analyzing ALL commits for validation)")
        logger.info("=" * 80)

        # ========== STEP 1: Initialize ==========
        metrics = self._initialize_analysis_metrics(repo_name)
        output_files = self._setup_output_files(repo_name)

        # ========== STEP 2: Resume Logic ==========
        resume_from_hash = self._check_resume_point(resume, output_files["checkpoint"])
        if not resume:
            self._clear_previous_results(output_files)

        # ========== STEP 3: Process Commits (MAIN WORK) ==========
        self._init_inspector()  # Prepare CodeSmile
        start_time = time.time()

        metrics = self._process_all_commits(
            repo_path,
            repo_name,
            resume_from_hash,
            no_skip,
            output_files,
            metrics,
            start_time,
        )

        # ========== STEP 4: Calculate Statistics ==========
        metrics = self._calculate_final_statistics(metrics, start_time)

        # ========== STEP 5: Save Results ==========
        self._save_analysis_results(metrics, output_files)

        return metrics

    def generate_reports(self) -> None:
        """Generate JSON, CSV, and Markdown reports from ALL analyzed repositories."""
        reporter = AnalysisReporter(self.output_dir)
        reporter.generate_all_reports()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ML repositories for code smells across commits using CodeSmile"
    )
    parser.add_argument(
        "--repo",
        type=str,
        nargs="+",
        help="Specific repository name(s) to analyze (optional, analyzes all if not specified)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last analyzed commit"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.RAW_SMELL_DETECTION_DIR),
        help=f"Output directory for analysis results (default: {config.RAW_SMELL_DETECTION_DIR})",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Analyze ALL commits (don't skip commits without Python changes). "
        "Use this to validate that skipping logic doesn't miss any smell changes.",
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = CommitAnalyzer(output_dir=Path(args.output_dir))

    try:
        # Load projects
        logger.info(f"Loading projects from {config.PROJECT_SELECTION_CSV}")
        projects_df = analyzer.load_projects(
            str(config.PROJECT_SELECTION_CSV), repo_filter=args.repo
        )
        logger.info(f"Found {len(projects_df)} repositories to analyze")

        all_metrics = []

        # Analyze each repository
        for _, row in projects_df.iterrows():
            repo_url_or_name = row["GitHub Repo"]

            # Parse repository name using utils
            owner, name = parse_repo_name(repo_url_or_name)
            if not owner or not name:
                logger.error(f"Could not parse repository: {repo_url_or_name}")
                continue

            repo_name = f"{owner}/{name}"
            repo_url = get_repo_url(repo_name)

            # Clone or get repo path
            repo_path = get_repo_path(repo_name)
            analyzer.clone_or_update_repo(repo_url, str(repo_path))

            # Analyze repository
            metrics = analyzer.analyze_repository(
                repo_path, repo_name, resume=args.resume, no_skip=args.no_skip
            )
            all_metrics.append(metrics)

        # Generate reports from all analyzed repositories
        logger.info("Generating aggregate reports from all analyzed repositories...")
        analyzer.generate_reports()

        logger.info("=" * 80)
        logger.info("Analysis complete!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Cleanup
        analyzer._cleanup()


if __name__ == "__main__":
    main()
