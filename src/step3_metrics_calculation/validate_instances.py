"""
Validation script for smell instances (new schema).

This script handles the NEW smell detection schema from historical_smell_detection.py:
- Input: simple intro/removal events with project_id, commit_hash, event_type
- Output: validated_smell_instances.jsonl compatible with calculate_metrics.py

Key differences from validate_instances.py (old schema):
- OLD schema: Pre-computed A-1/A and B-1/B pairs with smell_instance_id
- NEW schema: Simple events requiring parent commit computation

The script:
1. Loads smell events from smell_events.jsonl (new schema)
2. Generates smell_instance_id from event fields
3. Computes parent commit (before_commit) using local git
4. Fetches commit metadata (author, message, lines changed)
5. Validates file existence at both commits
6. Outputs validated_smell_instances.jsonl in format expected by calculate_metrics.py

Usage:
    python -m src.step3_metrics_calculation.validate_instances --repo owner/repo
"""

import argparse
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from src import config
from src.utils import read_jsonl, write_jsonl, write_json, repo_to_dir_name
from src.step3_metrics_calculation.constants import VALIDATION_PREVIEW_LIMIT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


# =============================================================================
# Git Operations
# =============================================================================

def run_git_command(repo_path: Path, args: List[str], timeout: int = 30) -> Optional[str]:
    """
    Run a git command in the repository directory.

    Args:
        repo_path: Path to the repository
        args: Git command arguments (without 'git' prefix)
        timeout: Command timeout in seconds

    Returns:
        Command stdout on success, None on failure
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"Git command failed: {e}")
        return None


def get_parent_commit_local(repo_path: Path, commit_hash: str) -> Optional[str]:
    """
    Get parent commit hash using local git.

    Args:
        repo_path: Path to the repository
        commit_hash: The commit to find parent of

    Returns:
        Parent commit hash, or None if not found
    """
    result = run_git_command(repo_path, ["rev-parse", f"{commit_hash}^"])
    return result


def get_commit_info_local(repo_path: Path, commit_hash: str) -> Optional[Dict]:
    """
    Get commit metadata using local git.

    Args:
        repo_path: Path to the repository
        commit_hash: Commit to get info for

    Returns:
        Dict with author, message, date, or None if failed
    """
    # Get basic commit info
    format_str = "%an|%ae|%ci|%s"  # author name|email|date|subject
    result = run_git_command(repo_path, ["log", "-1", f"--format={format_str}", commit_hash])
    if not result:
        return None

    parts = result.split("|", 3)
    if len(parts) < 4:
        return None

    # Get lines added/deleted using diff --shortstat
    stat_result = run_git_command(repo_path, ["diff", "--shortstat", f"{commit_hash}^", commit_hash])
    lines_added = 0
    lines_deleted = 0

    if stat_result:
        # Parse "N files changed, M insertions(+), P deletions(-)"
        import re
        insertions = re.search(r"(\d+) insertion", stat_result)
        deletions = re.search(r"(\d+) deletion", stat_result)
        if insertions:
            lines_added = int(insertions.group(1))
        if deletions:
            lines_deleted = int(deletions.group(1))

    return {
        "author": parts[0],
        "author_email": parts[1],
        "date": parts[2],
        "message": parts[3],
        "lines_added": lines_added,
        "lines_deleted": lines_deleted
    }


def check_file_exists_local(repo_path: Path, file_path: str, commit_hash: str) -> bool:
    """
    Check if file exists at a specific commit using local git.

    Args:
        repo_path: Path to the repository
        file_path: Path to file within repository
        commit_hash: Commit to check

    Returns:
        True if file exists at commit, False otherwise
    """
    result = run_git_command(repo_path, ["ls-tree", commit_hash, "--", file_path])
    return result is not None and len(result) > 0


# =============================================================================
# GitHub API Operations (Fallback)
# =============================================================================

def check_file_exists_github(repo_name: str, file_path: str, commit: str) -> bool:
    """Check if file exists at commit via GitHub API (fallback)."""
    url = f"https://api.github.com/repos/{repo_name}/contents/{file_path}?ref={commit}"

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.debug(f"GitHub API error checking {file_path} at {commit[:8]}: {e}")
        return False


def get_parent_commit_github(repo_name: str, commit_hash: str) -> Optional[str]:
    """Get parent commit via GitHub API (fallback)."""
    url = f"https://api.github.com/repos/{repo_name}/commits/{commit_hash}"

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            parents = data.get("parents", [])
            if parents:
                return parents[0].get("sha")
        return None
    except Exception as e:
        logger.debug(f"GitHub API error getting parent of {commit_hash[:8]}: {e}")
        return None


def get_commit_info_github(repo_name: str, commit_hash: str) -> Optional[Dict]:
    """Get commit metadata via GitHub API (fallback)."""
    url = f"https://api.github.com/repos/{repo_name}/commits/{commit_hash}"

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            commit_data = data.get("commit", {})
            stats = data.get("stats", {})
            return {
                "author": commit_data.get("author", {}).get("name", "Unknown"),
                "author_email": commit_data.get("author", {}).get("email", ""),
                "date": commit_data.get("author", {}).get("date", ""),
                "message": commit_data.get("message", ""),
                "lines_added": stats.get("additions", 0),
                "lines_deleted": stats.get("deletions", 0)
            }
        return None
    except Exception as e:
        logger.debug(f"GitHub API error getting commit info for {commit_hash[:8]}: {e}")
        return None


# =============================================================================
# Transformation Utilities
# =============================================================================

def generate_smell_instance_id(event: Dict) -> str:
    """
    Generate a unique smell_instance_id from event fields.

    Format: {project_id}-{smell_type}-{file_path}-{function_name}
    with slashes replaced by dashes.
    """
    project_id = event.get("project_id", "unknown")
    smell_type = event.get("smell_type", "unknown")
    file_path = event.get("file_path", "unknown")
    function_name = event.get("function_name", "unknown")

    instance_id = f"{project_id}-{smell_type}-{file_path}-{function_name}"
    return instance_id.replace("/", "-").replace(" ", "_")


def find_repo_path(repo_name: str, domain: Optional[str] = None) -> Optional[Path]:
    """
    Find the repository path, checking domain-based structure first.

    Args:
        repo_name: Repository name (owner/repo)
        domain: Optional domain hint for faster lookup

    Returns:
        Path to repository if found, None otherwise
    """
    repo_dir_name = repo_to_dir_name(repo_name)

    # If domain provided, check there first
    if domain:
        domain_abbrev = config.DOMAIN_ABBREVIATIONS.get(domain)
        if domain_abbrev:
            domain_path = config.REPOS_DIR / domain_abbrev / repo_dir_name
            if domain_path.exists() and (domain_path / ".git").exists():
                return domain_path

    # Check all domain directories
    for domain_abbrev in config.DOMAIN_ABBREVIATIONS.values():
        domain_path = config.REPOS_DIR / domain_abbrev / repo_dir_name
        if domain_path.exists() and (domain_path / ".git").exists():
            return domain_path

    # Check flat structure (legacy repos)
    flat_path = config.REPOS_DIR / repo_dir_name
    if flat_path.exists() and (flat_path / ".git").exists():
        return flat_path

    return None


# =============================================================================
# Core Validation Logic
# =============================================================================

def validate_single_event(
    event: Dict,
    repo_path: Optional[Path],
    repo_name: str,
    use_github_api: bool = False
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Validate a single smell event and transform to output schema.

    Args:
        event: Raw smell event from new schema
        repo_path: Path to local repository (None to use GitHub API)
        repo_name: Repository name (owner/repo)
        use_github_api: Force GitHub API even if local repo exists

    Returns:
        Tuple of (valid_instance, invalid_reason) - one will be None
    """
    # Skip first commits - no parent available
    if event.get("is_first_commit", False):
        return None, {"reason": "is_first_commit", "event": event}

    commit_hash = event.get("commit_hash")
    file_path = event.get("file_path")

    if not commit_hash or not file_path:
        return None, {"reason": "missing_commit_hash_or_file_path", "event": event}

    # Generate smell_instance_id
    smell_instance_id = generate_smell_instance_id(event)

    # Get parent commit (before_commit)
    before_commit = None
    if repo_path and not use_github_api:
        before_commit = get_parent_commit_local(repo_path, commit_hash)
    if not before_commit:
        before_commit = get_parent_commit_github(repo_name, commit_hash)

    if not before_commit:
        return None, {
            "reason": "cannot_find_parent_commit",
            "smell_instance_id": smell_instance_id,
            "commit_hash": commit_hash
        }

    # Check file existence at both commits
    exists_before = False
    exists_after = False

    if repo_path and not use_github_api:
        exists_before = check_file_exists_local(repo_path, file_path, before_commit)
        exists_after = check_file_exists_local(repo_path, file_path, commit_hash)
    else:
        exists_before = check_file_exists_github(repo_name, file_path, before_commit)
        exists_after = check_file_exists_github(repo_name, file_path, commit_hash)

    if not exists_before or not exists_after:
        reasons = []
        if not exists_before:
            reasons.append("file missing at before commit")
        if not exists_after:
            reasons.append("file missing at after commit")

        return None, {
            "smell_instance_id": smell_instance_id,
            "event_type": event.get("event_type"),
            "file_path": file_path,
            "before_commit": before_commit,
            "after_commit": commit_hash,
            "reason": " AND ".join(reasons),
            "smell_type": event.get("smell_type")
        }

    # Get commit metadata
    commit_info = None
    if repo_path and not use_github_api:
        commit_info = get_commit_info_local(repo_path, commit_hash)
    if not commit_info:
        commit_info = get_commit_info_github(repo_name, commit_hash)

    # Build valid instance record
    valid_instance = {
        "smell_instance_id": smell_instance_id,
        "project_name": event.get("project_id", repo_name),
        "smell_type": event.get("smell_type"),
        "event_type": event.get("event_type"),
        "file_path": file_path,
        "function_id": event.get("function_name", ""),
        "before_commit": before_commit,
        "after_commit": commit_hash,
        "before_timestamp": None,  # Not available from new schema
        "after_timestamp": event.get("commit_date"),
        "author": commit_info.get("author", event.get("author", "Unknown")) if commit_info else event.get("author", "Unknown"),
        "commit_message": commit_info.get("message", "") if commit_info else "",
        "lines_added": commit_info.get("lines_added", 0) if commit_info else 0,
        "lines_deleted": commit_info.get("lines_deleted", 0) if commit_info else 0,
    }

    return valid_instance, None


def load_smell_events(smell_events_file: Path) -> List[Dict]:
    """Load smell events from JSONL file."""
    return read_jsonl(smell_events_file)


def validate_smell_instances(repo_name: str, use_github_api: bool = False):
    """
    Main validation workflow for smell instances.

    Args:
        repo_name: Repository name (e.g., "owner/repo")
        use_github_api: Force GitHub API instead of local git

    Returns:
        Tuple of (valid_instances, invalid_instances)
    """
    repo_safe = repo_to_dir_name(repo_name)
    smell_events_file = config.RAW_SMELL_DETECTION_DIR / repo_safe / config.EVENTS_FILE
    validated_output = config.PROCESSED_METRICS_DIR / repo_safe / "validated_smell_instances.jsonl"
    validation_report = config.PROCESSED_METRICS_DIR / repo_safe / "validation_report.json"

    # Check GitHub authentication
    if GITHUB_TOKEN:
        logger.info("Using GitHub authentication (token found)")
    else:
        logger.warning("No GitHub token found! Set via: export GITHUB_TOKEN='your_token'")
        logger.warning("Will hit rate limits quickly without authentication.")

    # Load events
    logger.info(f"Loading smell events from: {smell_events_file}")
    if not smell_events_file.exists():
        raise FileNotFoundError(f"Smell events file not found: {smell_events_file}")

    events = load_smell_events(smell_events_file)
    logger.info(f"Loaded {len(events)} total smell events")

    # Count by event type
    introductions = [e for e in events if e.get("event_type") == "introduction"]
    removals = [e for e in events if e.get("event_type") == "removal"]
    first_commits = [e for e in events if e.get("is_first_commit", False)]

    logger.info(f"Introductions: {len(introductions)}")
    logger.info(f"Removals: {len(removals)}")
    logger.info(f"First-commit events (will be skipped): {len(first_commits)}")

    # Find repository path
    domain = events[0].get("domain") if events else None
    repo_path = find_repo_path(repo_name, domain)

    if repo_path:
        logger.info(f"Found local repository at: {repo_path}")
        if use_github_api:
            logger.info("Forcing GitHub API as requested")
    else:
        logger.warning(f"Local repository not found. Using GitHub API (slower).")
        use_github_api = True

    valid_instances = []
    invalid_instances = []
    skipped_first_commits = 0

    # Validate events
    logger.info("=" * 60)
    logger.info("VALIDATING SMELL EVENTS")
    logger.info("=" * 60)

    for i, event in enumerate(events, 1):
        if event.get("is_first_commit", False):
            skipped_first_commits += 1
            continue

        valid_instance, invalid_reason = validate_single_event(
            event, repo_path, repo_name, use_github_api
        )

        # Progress logging
        if i <= VALIDATION_PREVIEW_LIMIT or invalid_reason:
            event_type = event.get("event_type", "unknown")
            smell_type = event.get("smell_type", "unknown")
            status = "VALID" if valid_instance else "INVALID"
            logger.info(f"[{i}/{len(events)}] {status} - {smell_type} ({event_type})")

            if invalid_reason:
                logger.info(f"   Reason: {invalid_reason.get('reason', 'unknown')}")

        if valid_instance:
            valid_instances.append(valid_instance)
        elif invalid_reason:
            invalid_instances.append(invalid_reason)

    # Summary
    total_processable = len(events) - skipped_first_commits
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total events: {len(events)}")
    logger.info(f"Skipped (first commit): {skipped_first_commits}")
    logger.info(f"Processable events: {total_processable}")

    if total_processable > 0:
        validation_rate = len(valid_instances) / total_processable * 100
        logger.info(f"Valid instances: {len(valid_instances)} ({validation_rate:.1f}%)")
    else:
        logger.info(f"Valid instances: {len(valid_instances)}")
        validation_rate = 0

    logger.info(f"Invalid instances: {len(invalid_instances)}")

    # Count valid by event type
    valid_intros = sum(1 for inst in valid_instances if inst["event_type"] == "introduction")
    valid_removals = sum(1 for inst in valid_instances if inst["event_type"] == "removal")
    logger.info(f"  - Valid introductions: {valid_intros}")
    logger.info(f"  - Valid removals: {valid_removals}")

    # Save results
    validated_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(valid_instances, validated_output, append=False)
    logger.info(f"Validated instances written to: {validated_output}")

    # Save validation report
    report_data = {
        "source_file": str(smell_events_file),
        "schema_version": "full-dataset",
        "total_smell_events": len(events),
        "skipped_first_commits": skipped_first_commits,
        "processable_events": total_processable,
        "valid_instances": len(valid_instances),
        "valid_introductions": valid_intros,
        "valid_removals": valid_removals,
        "invalid_instances": len(invalid_instances),
        "validation_rate_percent": round(validation_rate, 1) if total_processable > 0 else 0,
        "used_local_git": repo_path is not None and not use_github_api,
        "repo_path": str(repo_path) if repo_path else None,
        "invalid_details": invalid_instances[:50],  # Limit to first 50
    }

    write_json(report_data, validation_report)
    logger.info(f"Validation report written to: {validation_report}")

    logger.info("=" * 60)
    logger.info("NEXT STEPS")
    logger.info("=" * 60)
    logger.info(f"1. Run metrics: python -m src.step3_metrics_calculation.calculate_metrics --repo {repo_name}")
    logger.info("2. Normalize: python -m src.step3_metrics_calculation.normalize_metrics")

    return valid_instances, invalid_instances


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Validate smell instances for project dataset (new schema)"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Repository name (e.g., onnx/onnxmltools)",
    )
    parser.add_argument(
        "--use-github-api",
        action="store_true",
        help="Force GitHub API instead of local git operations",
    )

    args = parser.parse_args()

    try:
        validate_smell_instances(args.repo, use_github_api=args.use_github_api)
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
