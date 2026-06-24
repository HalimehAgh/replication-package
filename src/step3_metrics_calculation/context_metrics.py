"""
Time-Window Metrics Calculation for Code Smells Analysis

Calculates historical metrics over a 90-day window before each commit
using PyDriller for git history traversal.

These metrics are ONLY calculated at commits A and B (not A-1 and B-1)
because A-1/A and B-1/B are consecutive commits with nearly identical
time windows.

Time-window metrics measure code context, not code quality:
- Is it a stable file that rarely changes (low churn)?
- Is it an active file under constant development (high churn)?
- Is it a clean file with few bugs (low bug-fix ratio)?
- Is it a problematic file that's always being fixed (high bug-fix ratio)?

Reference:
- Nagappan & Ball (2007): Use of relative code churn measures
- Palomba et al. (2018): On the diffuseness and impact of code smells
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import logging

from pydriller import Repository
import git

from src import config
from src.step1_commit_extraction.commit_analyzer import is_bug_fix_commit

logger = logging.getLogger(__name__)


class TimeWindowMetricsCalculator:
    """
    Calculates time-window based metrics for a file at a specific commit.

    Metrics calculated over a retrospective window (default 90 days):
    - window_total_churn: sum(lines_added + lines_deleted) for all commits
    - window_modified_files: count of unique files touched by commits that touched this file
    - window_commit_count: number of commits touching this file
    - window_bug_fix_ratio: bug_fix_commits / all_commits touching this file
    """

    def __init__(
        self,
        repo_path: Path,
        window_days: int = None
    ):
        """
        Initialize calculator with repository path.

        Args:
            repo_path: Path to git repository
            window_days: Number of days for retrospective window (default from config)
        """
        self.repo_path = Path(repo_path)
        self.window_days = window_days or config.TIME_WINDOW_DAYS
        # Cache for performance - key: (file_path, commit_hash)
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def calculate_metrics(
        self,
        file_path: str,
        commit_hash: str
    ) -> Dict[str, Any]:
        """
        Calculate time-window metrics for a file at a specific commit.

        Looks back window_days from the commit date and calculates:
        - window_total_churn: sum(lines_added + lines_deleted)
        - window_modified_files: unique files touched in window
        - window_commit_count: commits touching this file in window
        - window_bug_fix_ratio: bug_commits / all_commits in window

        Args:
            file_path: File path within repository
            commit_hash: The commit to analyze from

        Returns:
            Dictionary with time-window metrics
        """
        # Check cache
        cache_key = (file_path, commit_hash)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get commit date
        commit_date = self._get_commit_date(commit_hash)
        if commit_date is None:
            logger.warning(f"Could not get date for commit {commit_hash}")
            return self._empty_metrics()

        # Calculate window boundaries
        window_start = commit_date - timedelta(days=self.window_days)

        # Get commits in window
        try:
            commits_data = self._get_commits_in_window(
                file_path=file_path,
                window_start=window_start,
                window_end=commit_date,
                end_commit_hash=commit_hash
            )
        except Exception as e:
            logger.warning(f"Error getting commits in window: {e}")
            return self._empty_metrics()

        # Calculate metrics
        metrics = self._compute_metrics_from_commits(commits_data)

        # Cache result
        self._cache[cache_key] = metrics

        return metrics

    def _get_commit_date(self, commit_hash: str) -> Optional[datetime]:
        """Get the date of a commit."""
        try:
            repo = git.Repo(str(self.repo_path))
            commit = repo.commit(commit_hash)
            return commit.committed_datetime.replace(tzinfo=None)
        except Exception as e:
            logger.debug(f"Error getting commit date: {e}")
            return None

    def _get_commits_in_window(
        self,
        file_path: str,
        window_start: datetime,
        window_end: datetime,
        end_commit_hash: str
    ) -> Tuple[List[Dict[str, Any]], set]:
        """
        Get all commits touching a file within the time window.

        Returns:
            Tuple of (list of commit data, set of all modified files)
        """
        commits_data = []
        all_modified_files = set()

        try:
            # Use PyDriller to traverse commits
            for commit in Repository(
                str(self.repo_path),
                since=window_start,
                to=window_end,
            ).traverse_commits():
                # Skip the end commit itself (we want history before it)
                if commit.hash == end_commit_hash:
                    continue

                # Check if this commit touches our file
                file_touched = False
                for mod in commit.modified_files:
                    mod_path = mod.new_path or mod.old_path
                    if mod_path:
                        all_modified_files.add(mod_path)

                        # Check if this is our target file
                        if mod_path == file_path or mod_path.endswith('/' + file_path):
                            file_touched = True
                            commits_data.append({
                                'hash': commit.hash,
                                'message': commit.msg or '',
                                'lines_added': mod.added_lines or 0,
                                'lines_deleted': mod.deleted_lines or 0,
                            })

        except Exception as e:
            logger.debug(f"PyDriller traversal error: {e}")

        return commits_data, all_modified_files

    def _compute_metrics_from_commits(
        self,
        commits_result: Tuple[List[Dict[str, Any]], set]
    ) -> Dict[str, Any]:
        """Compute metrics from commit data."""
        commits_data, all_modified_files = commits_result

        if not commits_data:
            return self._empty_metrics()

        # Calculate total churn
        total_churn = sum(
            c['lines_added'] + c['lines_deleted']
            for c in commits_data
        )

        # Count commits
        commit_count = len(commits_data)

        # Count bug-fix commits
        bug_fix_count = sum(
            1 for c in commits_data
            if is_bug_fix_commit(c['message'])
        )

        # Bug-fix ratio
        bug_fix_ratio = bug_fix_count / commit_count if commit_count > 0 else 0.0

        # Count unique modified files
        modified_files_count = len(all_modified_files)

        return {
            'window_total_churn': total_churn,
            'window_modified_files': modified_files_count,
            'window_commit_count': commit_count,
            'window_bug_fix_ratio': bug_fix_ratio,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty/zero metrics."""
        return {
            'window_total_churn': 0,
            'window_modified_files': 0,
            'window_commit_count': 0,
            'window_bug_fix_ratio': 0.0,
        }

    def clear_cache(self) -> None:
        """Clear the metrics cache."""
        self._cache.clear()


# =============================================================================
# Convenience functions for integration
# =============================================================================

# Repository calculator cache
_calculators: Dict[str, TimeWindowMetricsCalculator] = {}


def get_calculator(repo_path: Path) -> TimeWindowMetricsCalculator:
    """Get or create calculator for a repository."""
    repo_key = str(repo_path)
    if repo_key not in _calculators:
        _calculators[repo_key] = TimeWindowMetricsCalculator(repo_path)
    return _calculators[repo_key]


def calculate_time_window_metrics(
    repo_path: Path,
    file_path: str,
    commit_hash: str
) -> Dict[str, Any]:
    """
    Calculate time-window metrics for a file at a specific commit.

    Convenience function for use in analyze_metrics.py.

    Args:
        repo_path: Path to git repository
        file_path: File path within repository
        commit_hash: The commit to analyze from

    Returns:
        Dictionary with time-window metrics
    """
    calculator = get_calculator(repo_path)
    return calculator.calculate_metrics(file_path, commit_hash)
