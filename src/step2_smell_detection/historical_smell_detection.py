"""
Historical Smell Detection for Project Analysis

Commit-by-commit analysis of all projects to detect smell introduction
and removal events throughout each repository's history.

Features:
- Commit-by-commit analysis with change tracking
- First commit strategy: All smells in first commit treated as introductions
- Timeout handling: 5 minutes max per commit
- Error handling: Continue processing if single file fails
- Resume capability with checkpoints per project
- Progress tracking with ETA

Output:
- Per-project JSONL files with smell events
- Combined events CSV for all projects
- Project summary with event counts

Usage:
    python -m src.step2_smell_detection.historical_smell_detection --repo owner/repo
"""

import argparse
import json
import os
import signal
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydriller import Repository

from src import config
from src.utils import setup_logger, read_json, write_json, write_jsonl
from src.utils.codesmile_utils import create_inspector
from src.step2_smell_detection.smell_parser import ML_SMELL_TYPES

logger = setup_logger(__name__)


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


@contextmanager
def timeout(seconds: int):
    """
    Context manager for timeout handling.

    Args:
        seconds: Maximum seconds to allow

    Raises:
        TimeoutError: If operation exceeds timeout
    """
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Only works on Unix systems
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        # Windows fallback - no timeout enforcement
        yield


class SmellEvent:
    """Represents a smell introduction or removal event."""

    def __init__(
        self,
        project_id: str,
        commit_hash: str,
        commit_date: str,
        file_path: str,
        smell_type: str,
        event_type: str,  # "introduction" or "removal"
        function_name: str = "",
        domain: str = "",
        size_category: str = "",
        author: str = "",
        commit_message: str = "",
        is_first_commit: bool = False,
    ):
        self.project_id = project_id
        self.commit_hash = commit_hash
        self.commit_date = commit_date
        self.file_path = file_path
        self.smell_type = smell_type
        self.event_type = event_type
        self.function_name = function_name
        self.domain = domain
        self.size_category = size_category
        self.author = author
        self.commit_message = commit_message
        self.is_first_commit = is_first_commit

        # Enrich with metadata from ML_SMELL_TYPES
        smell_info = ML_SMELL_TYPES.get(smell_type, {})
        self.category = smell_info.get("category", "ml_specific")
        self.severity = smell_info.get("severity", "medium")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "project_id": self.project_id,
            "commit_hash": self.commit_hash,
            "commit_date": self.commit_date,
            "file_path": self.file_path,
            "smell_type": self.smell_type,
            "event_type": self.event_type,
            "function_name": self.function_name,
            "domain": self.domain,
            "size_category": self.size_category,
            "category": self.category,
            "severity": self.severity,
            "author": self.author,
            "is_first_commit": self.is_first_commit,
        }


class HistoricalSmellDetector:
    """
    Analyzes repository commit history for smell introduction/removal events.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        commit_timeout: int = None,
    ):
        """
        Initialize detector.

        Args:
            output_dir: Directory for outputs (defaults to RAW_SMELL_DETECTION_DIR)
            commit_timeout: Timeout per commit in seconds (defaults to config value)
        """
        self.output_dir = output_dir or config.RAW_SMELL_DETECTION_DIR
        self.commit_timeout = commit_timeout or config.COMMIT_TIMEOUT_SECONDS
        self.inspector = None
        self.temp_dir = None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            self.temp_dir = None
            self.inspector = None

    def get_repo_output_dir(self, project_id: str) -> Path:
        """Get output directory for a specific project."""
        from src.utils.file_utils import repo_to_dir_name
        repo_dir = self.output_dir / repo_to_dir_name(project_id)
        repo_dir.mkdir(parents=True, exist_ok=True)
        return repo_dir

    def analyze_file_for_smells(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyze a single file for code smells using CodeSmile.

        Args:
            file_path: Path to Python file

        Returns:
            List of smell dictionaries
        """
        self._init_inspector()

        try:
            result_df = self.inspector.inspect(file_path)

            if result_df.empty:
                return []

            smells = []
            for _, row in result_df.iterrows():
                smells.append({
                    "filename": row.get("filename", file_path),
                    "function_name": row.get("function_name", ""),
                    "smell_name": row.get("smell_name", ""),
                    "line": row.get("line", -1),
                    "description": row.get("description", ""),
                })

            return smells

        except Exception as e:
            logger.debug(f"Error analyzing file {file_path}: {e}")
            return []

    def create_smell_key(self, smell: Dict[str, Any]) -> str:
        """
        Create unique key for a smell instance.

        Note: Line numbers excluded as smells can move during refactoring.
        """
        return f"{smell['filename']}:{smell['function_name']}:{smell['smell_name']}"

    def compare_smells(
        self,
        prev_smells: List[Dict[str, Any]],
        curr_smells: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Compare smells between two commits.

        Args:
            prev_smells: Smells from previous commit
            curr_smells: Smells from current commit

        Returns:
            (introduced_smells, removed_smells)
        """
        prev_keys = {self.create_smell_key(s): s for s in prev_smells}
        curr_keys = {self.create_smell_key(s): s for s in curr_smells}

        introduced = [curr_keys[k] for k in curr_keys if k not in prev_keys]
        removed = [prev_keys[k] for k in prev_keys if k not in curr_keys]

        return introduced, removed

    def load_checkpoint(self, repo_output_dir: Path) -> Optional[Dict]:
        """Load checkpoint if exists."""
        checkpoint_file = repo_output_dir / config.CHECKPOINT_FILE
        if checkpoint_file.exists():
            return read_json(checkpoint_file)
        return None

    def save_checkpoint(self, repo_output_dir: Path, checkpoint_data: Dict) -> None:
        """Save checkpoint data."""
        write_json(checkpoint_data, repo_output_dir / config.CHECKPOINT_FILE)

    def analyze_repository(
        self,
        repo_path: str,
        project_id: str,
        domain: str = "",
        size_category: str = "",
        resume: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze all commits in a repository for smell events.

        Args:
            repo_path: Path to git repository
            project_id: Project identifier (owner/repo format)
            domain: ML domain classification
            size_category: Size category (small/medium/large)
            resume: Whether to resume from checkpoint

        Returns:
            Metrics dictionary with analysis results
        """
        logger.info("=" * 80)
        logger.info(f"Analyzing: {project_id}")
        logger.info(f"Domain: {domain} | Size: {size_category}")
        logger.info("=" * 80)

        # Initialize
        repo_output_dir = self.get_repo_output_dir(project_id)
        self._init_inspector()

        # Output files
        events_file = repo_output_dir / config.EVENTS_FILE
        smells_file = repo_output_dir / config.SMELLS_FILE

        # Metrics tracking
        metrics = {
            "project_id": project_id,
            "domain": domain,
            "size_category": size_category,
            "total_commits": 0,
            "commits_analyzed": 0,
            "commits_skipped": 0,
            "commits_timed_out": 0,
            "files_analyzed": 0,
            "files_failed": 0,
            "smell_introductions": defaultdict(int),
            "smell_removals": defaultdict(int),
            "total_introductions": 0,
            "total_removals": 0,
            "date_range_start": None,
            "date_range_end": None,
            "errors": [],
            "analysis_start": datetime.now(timezone.utc).isoformat(),
        }

        # Resume handling
        resume_from_hash = None
        smells_state = {}  # file_path -> List[smell]

        if resume:
            checkpoint = self.load_checkpoint(repo_output_dir)
            if checkpoint:
                resume_from_hash = checkpoint.get("last_commit_hash")
                smells_state = checkpoint.get("smells_state", {})
                logger.info(f"Resuming from commit: {resume_from_hash}")
            else:
                # Clear previous results if not resuming
                for f in [events_file, smells_file]:
                    if f.exists():
                        f.unlink()

        start_time = time.time()
        skip_until_resume = resume_from_hash is not None
        previous_commit_hash = None
        is_first_commit = True

        # Process commits
        try:
            for commit in Repository(str(repo_path)).traverse_commits():
                metrics["total_commits"] += 1

                # Update date range
                commit_date = commit.author_date.isoformat()
                if metrics["date_range_start"] is None:
                    metrics["date_range_start"] = commit_date
                metrics["date_range_end"] = commit_date

                # Skip until resume point
                if skip_until_resume:
                    if commit.hash == resume_from_hash:
                        skip_until_resume = False
                        is_first_commit = False
                    continue

                # Check for Python files
                python_files = [
                    mf for mf in commit.modified_files
                    if mf.new_path and mf.new_path.endswith(".py")
                ]

                if not python_files:
                    metrics["commits_skipped"] += 1
                    previous_commit_hash = commit.hash
                    is_first_commit = False
                    continue

                # Analyze commit with timeout
                try:
                    with timeout(self.commit_timeout):
                        smells_after = self._analyze_commit(
                            commit,
                            python_files,
                            smells_state,
                            metrics,
                        )
                except TimeoutError:
                    metrics["commits_timed_out"] += 1
                    metrics["errors"].append({
                        "type": "timeout",
                        "commit": commit.hash,
                        "message": f"Commit timed out after {self.commit_timeout}s",
                    })
                    logger.warning(f"[TIMEOUT] Commit {commit.hash[:8]} ({len(python_files)} files)")
                    smells_after = smells_state.copy()

                # Detect smell changes
                events = self._detect_events(
                    smells_state,
                    smells_after,
                    project_id,
                    commit,
                    domain,
                    size_category,
                    is_first_commit,
                    metrics,
                )

                # Save events
                if events:
                    write_jsonl([e.to_dict() for e in events], events_file)

                # Save smells state for this commit
                smell_record = {
                    "commit_hash": commit.hash,
                    "timestamp": commit_date,
                    "smells": smells_after,
                }
                write_jsonl([smell_record], smells_file)

                # Update state
                smells_state = smells_after
                previous_commit_hash = commit.hash
                metrics["commits_analyzed"] += 1
                is_first_commit = False

                # Save checkpoint periodically
                if metrics["commits_analyzed"] % 50 == 0:
                    self.save_checkpoint(repo_output_dir, {
                        "last_commit_hash": commit.hash,
                        "smells_state": smells_state,
                        "commits_analyzed": metrics["commits_analyzed"],
                    })

                    # Log progress
                    elapsed = time.time() - start_time
                    rate = metrics["commits_analyzed"] / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {metrics['commits_analyzed']} commits | "
                        f"{metrics['total_introductions']} intro | "
                        f"{metrics['total_removals']} removal | "
                        f"{rate:.1f} commits/sec"
                    )

        except Exception as e:
            logger.error(f"Error processing repository: {e}")
            metrics["errors"].append({
                "type": "repository_error",
                "message": str(e),
            })

        # Finalize metrics
        metrics["total_time_seconds"] = round(time.time() - start_time, 2)
        metrics["smell_introductions"] = dict(metrics["smell_introductions"])
        metrics["smell_removals"] = dict(metrics["smell_removals"])
        metrics["analysis_end"] = datetime.now(timezone.utc).isoformat()

        # Save final metrics
        write_json(metrics, repo_output_dir / config.METRICS_FILE)

        # Log summary
        logger.info("")
        logger.info(f"Completed: {project_id}")
        logger.info(f"  Commits: {metrics['commits_analyzed']} analyzed, {metrics['commits_skipped']} skipped")
        logger.info(f"  Introductions: {metrics['total_introductions']}")
        logger.info(f"  Removals: {metrics['total_removals']}")
        logger.info(f"  Time: {metrics['total_time_seconds']:.1f}s")

        return metrics

    def _analyze_commit(
        self,
        commit,
        python_files: List,
        current_smells: Dict[str, List],
        metrics: Dict,
    ) -> Dict[str, List]:
        """
        Analyze Python files in a commit for smells.

        Args:
            commit: PyDriller commit object
            python_files: List of modified Python files
            current_smells: Current smell state
            metrics: Metrics dict to update

        Returns:
            Updated smells state
        """
        smells_after = current_smells.copy()

        # Warn about large commits
        if len(python_files) > config.LARGE_COMMIT_FILE_THRESHOLD:
            logger.warning(
                f"Large commit {commit.hash[:8]}: {len(python_files)} Python files"
            )

        for modified_file in python_files:
            file_path = modified_file.new_path

            try:
                # Handle deleted files
                if modified_file.source_code is None:
                    smells_after[file_path] = []
                    continue

                # Write to temp file for analysis
                temp_file = os.path.join(self.temp_dir, "temp_analysis.py")
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(modified_file.source_code)

                # Analyze for smells
                smells = self.analyze_file_for_smells(temp_file)

                # Update filename in results
                for smell in smells:
                    smell["filename"] = file_path

                smells_after[file_path] = smells
                metrics["files_analyzed"] += 1

            except Exception as e:
                metrics["files_failed"] += 1
                metrics["errors"].append({
                    "type": "file_error",
                    "commit": commit.hash,
                    "file": file_path,
                    "message": str(e)[:200],
                })
                # Keep previous state for this file
                if file_path not in smells_after:
                    smells_after[file_path] = []

        return smells_after

    def _detect_events(
        self,
        prev_smells: Dict[str, List],
        curr_smells: Dict[str, List],
        project_id: str,
        commit,
        domain: str,
        size_category: str,
        is_first_commit: bool,
        metrics: Dict,
    ) -> List[SmellEvent]:
        """
        Detect smell introduction and removal events.

        Args:
            prev_smells: Smell state before commit
            curr_smells: Smell state after commit
            project_id: Project identifier
            commit: PyDriller commit object
            domain: Domain classification
            size_category: Size category
            is_first_commit: Whether this is the first commit
            metrics: Metrics dict to update

        Returns:
            List of SmellEvent objects
        """
        events = []
        commit_date = commit.author_date.isoformat()

        # For first commit, all smells are introductions (per methodology)
        if is_first_commit:
            for file_path, smells in curr_smells.items():
                for smell in smells:
                    event = SmellEvent(
                        project_id=project_id,
                        commit_hash=commit.hash,
                        commit_date=commit_date,
                        file_path=file_path,
                        smell_type=smell["smell_name"],
                        event_type="introduction",
                        function_name=smell.get("function_name", ""),
                        domain=domain,
                        size_category=size_category,
                        author=commit.author.name,
                        commit_message=commit.msg[:200] if commit.msg else "",
                        is_first_commit=True,
                    )
                    events.append(event)
                    metrics["smell_introductions"][smell["smell_name"]] += 1
                    metrics["total_introductions"] += 1
            return events

        # Compare modified files
        all_files = set(prev_smells.keys()) | set(curr_smells.keys())

        for file_path in all_files:
            prev = prev_smells.get(file_path, [])
            curr = curr_smells.get(file_path, [])

            introduced, removed = self.compare_smells(prev, curr)

            # Record introductions
            for smell in introduced:
                event = SmellEvent(
                    project_id=project_id,
                    commit_hash=commit.hash,
                    commit_date=commit_date,
                    file_path=file_path,
                    smell_type=smell["smell_name"],
                    event_type="introduction",
                    function_name=smell.get("function_name", ""),
                    domain=domain,
                    size_category=size_category,
                    author=commit.author.name,
                    commit_message=commit.msg[:200] if commit.msg else "",
                )
                events.append(event)
                metrics["smell_introductions"][smell["smell_name"]] += 1
                metrics["total_introductions"] += 1

            # Record removals
            for smell in removed:
                event = SmellEvent(
                    project_id=project_id,
                    commit_hash=commit.hash,
                    commit_date=commit_date,
                    file_path=file_path,
                    smell_type=smell["smell_name"],
                    event_type="removal",
                    function_name=smell.get("function_name", ""),
                    domain=domain,
                    size_category=size_category,
                    author=commit.author.name,
                    commit_message=commit.msg[:200] if commit.msg else "",
                )
                events.append(event)
                metrics["smell_removals"][smell["smell_name"]] += 1
                metrics["total_removals"] += 1

        return events


def main():
    """Main entry point for single-repository analysis."""
    parser = argparse.ArgumentParser(
        description="Historical smell detection for a single repository"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Repository in 'owner/repo' format"
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        help="Path to cloned repository (auto-detected if not provided)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="",
        help="Domain classification"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="",
        help="Size category (small/medium/large)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=config.COMMIT_TIMEOUT_SECONDS,
        help=f"Timeout per commit in seconds (default: {config.COMMIT_TIMEOUT_SECONDS})"
    )

    args = parser.parse_args()

    # Determine repo path
    if args.repo_path:
        repo_path = args.repo_path
    else:
        # Try to find in domain-based structure or flat structure
        from src.utils.file_utils import repo_to_dir_name
        repo_dir_name = repo_to_dir_name(args.repo)

        # Check domain-based paths
        for domain_abbrev in config.DOMAIN_ABBREVIATIONS.values():
            potential_path = config.REPOS_DIR / domain_abbrev / repo_dir_name
            if potential_path.exists():
                repo_path = str(potential_path)
                break
        else:
            # Check flat structure
            flat_path = config.REPOS_DIR / repo_dir_name
            if flat_path.exists():
                repo_path = str(flat_path)
            else:
                logger.error(f"Repository not found. Clone it first or provide --repo-path")
                sys.exit(1)

    # Initialize detector and run
    detector = HistoricalSmellDetector(commit_timeout=args.timeout)

    try:
        metrics = detector.analyze_repository(
            repo_path=repo_path,
            project_id=args.repo,
            domain=args.domain,
            size_category=args.size,
            resume=args.resume,
        )

        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)

    finally:
        detector._cleanup()


if __name__ == "__main__":
    main()
