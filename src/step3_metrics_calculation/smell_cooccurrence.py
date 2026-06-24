"""
Smell Co-occurrence Tracking for Metrics Analysis

Loads smell counts from smells_per_commit.jsonl to track how many
smells exist in a file at each commit. Used as a control variable
to account for confounding effects of multiple smells.

Reference:
- Khomh et al. (2012): "To control for the confounding effect of multiple
  antipatterns in the same class, we include the variable 'number of
  antipatterns' as a covariate in our regression models."
- Palomba et al. (2018): "Code smells rarely occur in isolation. In our
  dataset, 63% of smelly commits contain multiple smell types."

Usage:
    from src.step3_metrics_calculation.smell_cooccurrence import get_smell_count_at_commit

    count = get_smell_count_at_commit(
        repo_name="ultralytics/yolov5",
        commit_hash="abc123...",
        file_path="models/yolo.py"
    )
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from collections import defaultdict

from src import config
from src.utils import repo_to_dir_name, read_jsonl

logger = logging.getLogger(__name__)


class SmellCooccurrenceTracker:
    """
    Tracks smell counts at commits for use as control variables.

    Loads smells_per_commit.jsonl and provides lookup by commit hash.
    Uses caching for performance when querying multiple times.
    """

    def __init__(self, repo_name: str):
        """
        Initialize tracker for a repository.

        Args:
            repo_name: Repository name (e.g., "ultralytics/yolov5")
        """
        self.repo_name = repo_name
        # Key: commit_hash, Value: Dict[file_path, List[smell_dicts]]
        self._smells_by_commit: Dict[str, Dict[str, List[Dict]]] = {}
        self._loaded = False

    def load(self) -> None:
        """Load smells_per_commit.jsonl for the repository."""
        if self._loaded:
            return

        repo_safe = repo_to_dir_name(self.repo_name)
        smells_file = config.RAW_SMELL_DETECTION_DIR / repo_safe / config.SMELLS_FILE

        if not smells_file.exists():
            logger.warning(f"Smells file not found: {smells_file}")
            self._loaded = True  # Mark as loaded to avoid repeated attempts
            return

        try:
            for record in read_jsonl(smells_file):
                commit_hash = record.get("commit_hash")
                if commit_hash:
                    # smells is a dict: file_path -> list of smell objects
                    self._smells_by_commit[commit_hash] = record.get("smells", {})

            logger.debug(
                f"Loaded smell data for {len(self._smells_by_commit)} commits "
                f"from {self.repo_name}"
            )
        except Exception as e:
            logger.warning(f"Error loading smells file {smells_file}: {e}")

        self._loaded = True

    def get_smell_count_at_commit(
        self,
        commit_hash: str,
        file_path: Optional[str] = None
    ) -> int:
        """
        Get total smell count at a commit.

        Args:
            commit_hash: Git commit hash
            file_path: Optional file path to filter by (if None, counts all files)

        Returns:
            Number of smells (0 if commit not found or no smells)
        """
        if not self._loaded:
            self.load()

        smells_data = self._smells_by_commit.get(commit_hash, {})

        if file_path:
            # Get smells for specific file
            # Try exact match first
            file_smells = smells_data.get(file_path, [])
            if not file_smells:
                # Try matching by filename (in case of path differences)
                for stored_path, smells in smells_data.items():
                    if stored_path.endswith(file_path) or file_path.endswith(stored_path):
                        file_smells = smells
                        break
            return len(file_smells)
        else:
            # Get total smells across all files
            total = 0
            for file_smells in smells_data.values():
                total += len(file_smells)
            return total

    def get_smell_types_at_commit(
        self,
        commit_hash: str,
        file_path: Optional[str] = None
    ) -> List[str]:
        """
        Get list of smell types at a commit.

        Args:
            commit_hash: Git commit hash
            file_path: Optional file path to filter by

        Returns:
            List of smell type names
        """
        if not self._loaded:
            self.load()

        smells_data = self._smells_by_commit.get(commit_hash, {})
        smell_types = []

        if file_path:
            file_smells = smells_data.get(file_path, [])
            if not file_smells:
                for stored_path, smells in smells_data.items():
                    if stored_path.endswith(file_path) or file_path.endswith(stored_path):
                        file_smells = smells
                        break
            for smell in file_smells:
                smell_type = smell.get("smell_name") or smell.get("smell_type")
                if smell_type:
                    smell_types.append(smell_type)
        else:
            for file_smells in smells_data.values():
                for smell in file_smells:
                    smell_type = smell.get("smell_name") or smell.get("smell_type")
                    if smell_type:
                        smell_types.append(smell_type)

        return smell_types

    def clear_cache(self) -> None:
        """Clear loaded data to free memory."""
        self._smells_by_commit.clear()
        self._loaded = False


# =============================================================================
# Module-level cache and convenience functions
# =============================================================================

# Cache of trackers by repository
_trackers: Dict[str, SmellCooccurrenceTracker] = {}


def get_tracker(repo_name: str) -> SmellCooccurrenceTracker:
    """Get or create tracker for a repository."""
    if repo_name not in _trackers:
        _trackers[repo_name] = SmellCooccurrenceTracker(repo_name)
        _trackers[repo_name].load()
    return _trackers[repo_name]


def get_smell_count_at_commit(
    repo_name: str,
    commit_hash: str,
    file_path: Optional[str] = None
) -> int:
    """
    Get smell count at a specific commit.

    Convenience function for use in analyze_metrics.py.

    Args:
        repo_name: Repository name (e.g., "ultralytics/yolov5")
        commit_hash: Git commit hash
        file_path: Optional file path to filter by

    Returns:
        Number of smells at this commit (for the file if specified)
    """
    tracker = get_tracker(repo_name)
    return tracker.get_smell_count_at_commit(commit_hash, file_path)


def get_smell_types_at_commit(
    repo_name: str,
    commit_hash: str,
    file_path: Optional[str] = None
) -> List[str]:
    """
    Get list of smell types at a specific commit.

    Args:
        repo_name: Repository name
        commit_hash: Git commit hash
        file_path: Optional file path to filter by

    Returns:
        List of smell type names
    """
    tracker = get_tracker(repo_name)
    return tracker.get_smell_types_at_commit(commit_hash, file_path)


def clear_all_trackers() -> None:
    """Clear all cached trackers to free memory."""
    for tracker in _trackers.values():
        tracker.clear_cache()
    _trackers.clear()
