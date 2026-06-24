"""
Commit Extraction Module

Extracts commit history from selected repositories for smell detection analysis.
Outputs JSONL format compatible with smell detection step.

Usage
-----
python -m src.step1_commit_extraction.extract_commits
python -m src.step1_commit_extraction.extract_commits --repo owner/repo
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from pydriller import Repository

from src import config
from src.utils import (
    parse_repo_name,
    get_repo_url,
    get_repo_path,
    write_jsonl,
    write_json,
)
from src.utils.file_utils import repo_to_dir_name

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

# File extensions to track for Python ML projects
TRACKED_EXTENSIONS = {'.py', '.pyx', '.pxd'}

# Progress update interval
PROGRESS_INTERVAL = 100


# ============================================================
# COMMIT EXTRACTION
# ============================================================

def extract_commit_info(commit) -> Dict[str, Any]:
    """
    Extract commit metadata in the format expected by smell detection.

    This format matches what run_codesmile.py saves, ensuring compatibility
    between commit extraction and smell detection steps.

    Args:
        commit: PyDriller Commit object

    Returns:
        Dictionary with commit metadata
    """
    # Get modified Python files
    python_files = [
        modified_file.new_path
        for modified_file in commit.modified_files
        if modified_file.new_path and modified_file.new_path.endswith('.py')
    ]

    return {
        'commit_hash': commit.hash,
        'author': commit.author.name,
        'timestamp': commit.author_date.isoformat(),
        'message': commit.msg,
        'modified_files': python_files,
        'lines_added': commit.insertions,
        'lines_deleted': commit.deletions,
    }


def extract_commits_from_repo(
    repo_path: str,
    max_commits: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Extract all commits from a repository using PyDriller.

    Args:
        repo_path: Path to the git repository
        max_commits: Maximum number of commits to extract (None for all)

    Returns:
        List of commit dictionaries in chronological order
    """
    commits_data = []

    logger.info(f"Traversing commits in {repo_path}")

    for i, commit in enumerate(Repository(str(repo_path)).traverse_commits()):
        if max_commits and i >= max_commits:
            break

        try:
            commit_info = extract_commit_info(commit)
            commits_data.append(commit_info)

            if (i + 1) % PROGRESS_INTERVAL == 0:
                logger.info(f"  Extracted {i + 1} commits...")

        except Exception as e:
            logger.warning(f"Error processing commit {commit.hash[:8]}: {e}")
            continue

    logger.info(f"Extracted {len(commits_data)} commits total")
    return commits_data


# ============================================================
# MAIN WORKFLOW
# ============================================================

def get_commit_output_dir(repo_name: str) -> Path:
    """
    Get output directory for a repository's commits.

    Args:
        repo_name: Repository name (e.g., "owner/repo")

    Returns:
        Path to output directory (e.g., data/raw/commit_history/owner_repo/)
    """
    dir_name = repo_to_dir_name(repo_name)
    output_dir = config.RAW_COMMIT_HISTORY_DIR / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def extract_commits_for_project(
    repo_identifier: str,
    max_commits: Optional[int] = None
) -> Optional[Path]:
    """
    Extract commits for a single project and save as JSONL.

    Args:
        repo_identifier: Repository string (e.g., "owner/repo")
        max_commits: Maximum commits to extract

    Returns:
        Path to output JSONL file, or None if failed
    """
    owner, name = parse_repo_name(repo_identifier)
    if not owner or not name:
        logger.error(f"Could not parse repository: {repo_identifier}")
        return None

    repo_name = f"{owner}/{name}"

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing: {repo_name}")
    logger.info(f"{'=' * 60}")

    # Get repo path (use existing clone from repos/)
    repo_path = get_repo_path(repo_name)

    if not repo_path.exists():
        logger.error(f"Repository not cloned: {repo_path}")
        logger.info("Run smell detection first to clone repositories, or clone manually.")
        return None

    logger.info(f"Using existing clone: {repo_path}")

    try:
        # Extract commits
        commits = extract_commits_from_repo(str(repo_path), max_commits)

        if not commits:
            logger.warning(f"No commits extracted for {repo_name}")
            return None

        # Setup output directory
        output_dir = get_commit_output_dir(repo_name)
        output_file = output_dir / config.COMMITS_FILE  # commits.jsonl

        # Clear existing file and write new data
        if output_file.exists():
            output_file.unlink()

        write_jsonl(commits, output_file)
        logger.info(f"Saved: {output_file} ({len(commits)} commits)")

        # Save extraction metadata
        metadata = {
            'repository': repo_name,
            'total_commits': len(commits),
            'commits_with_python': sum(1 for c in commits if c['modified_files']),
        }
        write_json(metadata, output_dir / 'extraction_metadata.json')

        return output_file

    except Exception as e:
        logger.error(f"Failed to extract commits for {repo_identifier}: {e}")
        return None


def main(
    projects_csv: Optional[str] = None,
    repo_filter: Optional[List[str]] = None,
    max_commits: Optional[int] = None
) -> None:
    """
    Main workflow to extract commits from selected projects.

    Args:
        projects_csv: Path to project selection CSV
        repo_filter: List of specific repositories to process
        max_commits: Maximum commits per repository
    """
    # Default paths
    if projects_csv is None:
        projects_csv = config.PROJECT_SELECTION_CSV
    else:
        projects_csv = Path(projects_csv)

    # Load selected projects
    if not Path(projects_csv).exists():
        logger.error(f"Project selection file not found: {projects_csv}")
        logger.info("Run project selection first: python -m src.step1_commit_extraction.project_selection")
        return

    projects_df = pd.read_csv(projects_csv)
    logger.info(f"Loaded {len(projects_df)} projects from {projects_csv}")

    # Apply filter if specified
    if repo_filter:
        projects_df = projects_df[
            projects_df['GitHub Repo'].apply(
                lambda x: any(f.lower() in x.lower() for f in repo_filter)
            )
        ]
        logger.info(f"Filtered to {len(projects_df)} projects matching: {repo_filter}")

    # Extract commits for each project
    results = []
    for _, row in projects_df.iterrows():
        repo_id = str(row['GitHub Repo']).strip()

        result_path = extract_commits_for_project(
            repo_identifier=repo_id,
            max_commits=max_commits
        )

        results.append({
            'repository': repo_id,
            'output_file': str(result_path) if result_path else None,
            'success': result_path is not None
        })

    # Summary
    success_count = sum(1 for r in results if r['success'])
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EXTRACTION COMPLETE: {success_count}/{len(results)} successful")
    logger.info(f"{'=' * 60}")

    # Save extraction summary
    summary_df = pd.DataFrame(results)
    summary_file = config.RAW_COMMIT_HISTORY_DIR / 'extraction_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract commits from selected repositories"
    )
    parser.add_argument(
        "--projects",
        type=str,
        help="Path to project selection CSV"
    )
    parser.add_argument(
        "--repo",
        type=str,
        nargs="+",
        help="Specific repository name(s) to process"
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        help="Maximum commits per repository"
    )

    args = parser.parse_args()

    main(
        projects_csv=args.projects,
        repo_filter=args.repo,
        max_commits=args.max_commits
    )
