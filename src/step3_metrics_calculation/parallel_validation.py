"""
Parallel Validation Script

Runs smell instance validation across multiple repositories in parallel to speed up processing.
Each repository requires GitHub API calls (rate-limited to 5,000/hour with token), so use
a low worker count (2-4 recommended) to avoid rate limiting issues.

Usage:
    # Check current validation status
    python -m src.step3_metrics_calculation.parallel_validation --status

    # Run validation with resume support
    python -m src.step3_metrics_calculation.parallel_validation --workers 2 --resume

    # Run fresh validation (re-process all repos)
    python -m src.step3_metrics_calculation.parallel_validation --workers 2
"""

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from src import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_single_repo(args: Tuple[str, Path, Path]) -> Dict:
    """
    Validate smell instances for a single repository (runs in worker process).

    Args:
        args: Tuple of (repo_name, smell_dir, output_dir)

    Returns:
        Dict with status and stats
    """
    repo_name, smell_dir, output_dir = args

    # Import inside worker to avoid multiprocessing pickling issues
    # Use 279-project validation script for new schema
    from src.step3_metrics_calculation.validate_instances_279 import validate_smell_instances

    try:
        valid_instances, invalid_instances = validate_smell_instances(repo_name)
        return {
            'repo': repo_name,
            'status': 'success',
            'valid_count': len(valid_instances),
            'invalid_count': len(invalid_instances),
        }
    except FileNotFoundError as e:
        return {
            'repo': repo_name,
            'status': 'error',
            'error': f"File not found: {e}"
        }
    except Exception as e:
        return {
            'repo': repo_name,
            'status': 'error',
            'error': str(e)
        }


def discover_repos(smell_dir: Path) -> List[str]:
    """
    Scan for repositories with smell_events.jsonl files.

    Args:
        smell_dir: Directory containing smell detection results

    Returns:
        List of repository names (owner/repo format)
    """
    repos = []
    for repo_dir in smell_dir.iterdir():
        if repo_dir.is_dir() and repo_dir.name != "reports":
            events_file = repo_dir / config.EVENTS_FILE
            if events_file.exists():
                # Convert directory name back to repo name
                repo_name = repo_dir.name.replace("_", "/", 1)
                repos.append(repo_name)
    return sorted(repos)


def get_pending_repos(smell_dir: Path, output_dir: Path, resume: bool) -> List[str]:
    """
    Get list of repositories that need validation.

    Args:
        smell_dir: Directory with smell detection results
        output_dir: Directory for validation output
        resume: If True, skip repos that already have validated output

    Returns:
        List of repository names to process
    """
    all_repos = discover_repos(smell_dir)
    logger.info(f"Found {len(all_repos)} total repositories with smell events")

    if not resume:
        return all_repos

    # Filter out repos that already have validated_smell_instances.jsonl
    pending = []
    skipped = 0
    for repo_name in all_repos:
        repo_dir_name = repo_name.replace("/", "_")
        validated_file = output_dir / repo_dir_name / "validated_smell_instances.jsonl"

        if validated_file.exists():
            skipped += 1
        else:
            pending.append(repo_name)

    logger.info(f"Skipping {skipped} already-validated repositories")
    return pending


def show_status(smell_dir: Path, output_dir: Path) -> None:
    """
    Show current validation status without running validation.

    Args:
        smell_dir: Directory with smell detection results
        output_dir: Directory for validation output
    """
    all_repos = discover_repos(smell_dir)

    validated = 0
    pending = 0
    validated_repos = []
    pending_repos = []

    for repo_name in all_repos:
        repo_dir_name = repo_name.replace("/", "_")
        validated_file = output_dir / repo_dir_name / "validated_smell_instances.jsonl"

        if validated_file.exists():
            validated += 1
            validated_repos.append(repo_name)
        else:
            pending += 1
            pending_repos.append(repo_name)

    logger.info("=" * 60)
    logger.info("VALIDATION STATUS")
    logger.info("=" * 60)
    logger.info(f"Total repositories: {len(all_repos)}")
    logger.info(f"Validated: {validated}")
    logger.info(f"Pending: {pending}")

    if validated > 0:
        logger.info("")
        logger.info(f"Validated repositories ({validated}):")
        for repo in validated_repos[:10]:
            logger.info(f"  - {repo}")
        if validated > 10:
            logger.info(f"  ... and {validated - 10} more")

    if pending > 0:
        logger.info("")
        logger.info(f"Pending repositories ({pending}):")
        for repo in pending_repos[:10]:
            logger.info(f"  - {repo}")
        if pending > 10:
            logger.info(f"  ... and {pending - 10} more")

    logger.info("=" * 60)


def run_parallel(
    workers: int = 2,
    resume: bool = True,
    smell_dir: Path = None,
    output_dir: Path = None
) -> Dict:
    """
    Run validation in parallel across repositories.

    Args:
        workers: Number of parallel workers (default 2, max 4 recommended)
        resume: Skip already-validated repositories
        smell_dir: Directory with smell detection results
        output_dir: Directory for validation output

    Returns:
        Summary dict with completion statistics
    """
    # Use 279-project smell detection directory for new schema
    smell_dir = smell_dir or config.SMELL_DETECTION_279_DIR
    output_dir = output_dir or config.PROCESSED_METRICS_DIR

    # Check for GitHub token
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        logger.info("GitHub token found - using authenticated API calls")
    else:
        logger.warning("=" * 60)
        logger.warning("WARNING: No GITHUB_TOKEN environment variable set!")
        logger.warning("API rate limit will be 60 requests/hour (vs 5,000/hour with token)")
        logger.warning("Set via: export GITHUB_TOKEN='your_token'")
        logger.warning("=" * 60)

    # Get repositories to process
    repos = get_pending_repos(smell_dir, output_dir, resume)
    logger.info(f"Found {len(repos)} repositories to validate")

    if not repos:
        logger.info("No repositories to validate!")
        return {
            'total': 0,
            'completed': 0,
            'failed': 0,
            'elapsed_seconds': 0
        }

    # Prepare arguments for workers
    work_items = [
        (repo, smell_dir, output_dir)
        for repo in repos
    ]

    # Track progress
    completed = 0
    failed = 0
    total_valid = 0
    total_invalid = 0
    start_time = time.time()

    logger.info(f"Starting parallel validation with {workers} workers...")
    logger.info("Note: Keep workers low (2-4) to respect GitHub API rate limits")

    # Run in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(validate_single_repo, item): item[0] for item in work_items}

        for future in as_completed(futures):
            repo_name = futures[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    completed += 1
                    total_valid += result.get('valid_count', 0)
                    total_invalid += result.get('invalid_count', 0)
                    logger.info(
                        f"[{completed + failed}/{len(repos)}] Completed: {repo_name} "
                        f"(valid={result.get('valid_count', 0)}, invalid={result.get('invalid_count', 0)})"
                    )
                else:
                    failed += 1
                    logger.error(
                        f"[{completed + failed}/{len(repos)}] Failed: {repo_name} - "
                        f"{result.get('error', 'unknown error')}"
                    )
            except Exception as e:
                failed += 1
                logger.error(f"[{completed + failed}/{len(repos)}] Exception: {repo_name} - {e}")

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PARALLEL VALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total repos processed: {len(repos)}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total valid instances: {total_valid}")
    logger.info(f"Total invalid instances: {total_invalid}")
    logger.info(f"Time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    if repos:
        logger.info(f"Avg per repo: {elapsed/len(repos):.1f} seconds")
    logger.info("=" * 60)
    logger.info("")
    logger.info("NEXT STEPS")
    logger.info("Run metrics calculation:")
    logger.info("  python -m src.step3_metrics_calculation.parallel_metrics --workers 8 --resume")
    logger.info("=" * 60)

    return {
        'total': len(repos),
        'completed': completed,
        'failed': failed,
        'total_valid': total_valid,
        'total_invalid': total_invalid,
        'elapsed_seconds': elapsed
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run smell instance validation in parallel across repositories"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=2,
        help="Number of parallel workers (default: 2, max 4 recommended due to API limits)"
    )
    parser.add_argument(
        "--resume", "-r", action="store_true",
        help="Skip already-validated repositories"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show validation status without running validation"
    )
    parser.add_argument(
        "--smell-dir", type=str, default=str(config.SMELL_DETECTION_279_DIR),
        help="Directory with smell detection results (279-project schema)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(config.PROCESSED_METRICS_DIR),
        help="Directory for validation output"
    )

    args = parser.parse_args()

    smell_dir = Path(args.smell_dir)
    output_dir = Path(args.output_dir)

    if args.status:
        show_status(smell_dir, output_dir)
        return

    if args.workers > 4:
        logger.warning(
            f"Using {args.workers} workers. Consider using <= 4 workers to avoid "
            "GitHub API rate limit issues."
        )

    run_parallel(
        workers=args.workers,
        resume=args.resume,
        smell_dir=smell_dir,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
