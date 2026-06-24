"""
Parallel Metrics Calculation

Runs metrics calculation across multiple repositories in parallel to speed up processing.

Usage:
    python -m src.step3_metrics_calculation.parallel_metrics --workers 4 --resume
    python -m src.step3_metrics_calculation.parallel_metrics --workers 8 --resume

With 8 workers on a 10-core machine, expect ~6-8x speedup (50 hours -> 6-8 hours).
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json

from src import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_single_repo(args: Tuple[str, Path, Path, bool]) -> dict:
    """
    Process a single repository (runs in worker process).

    Args:
        args: Tuple of (repo_name, analysis_dir, output_dir, resume)

    Returns:
        Dict with status and stats
    """
    repo_name, analysis_dir, output_dir, resume = args

    # Import here to avoid pickling issues
    from src.step3_metrics_calculation.calculate_metrics import MetricsAnalyzer

    try:
        analyzer = MetricsAnalyzer(output_dir=output_dir)
        stats = analyzer.analyze_validated_instances(repo_name, resume=resume)
        return {
            'repo': repo_name,
            'status': 'success',
            'stats': stats
        }
    except Exception as e:
        return {
            'repo': repo_name,
            'status': 'error',
            'error': str(e)
        }


def get_pending_repos(analysis_dir: Path, output_dir: Path, resume: bool) -> List[str]:
    """
    Get list of repositories that need processing.

    If resume=True, skip repos that already have completed metrics.
    """
    all_repos = []
    for repo_dir in analysis_dir.iterdir():
        if repo_dir.is_dir() and repo_dir.name != "reports":
            events_file = repo_dir / config.EVENTS_FILE
            if events_file.exists():
                repo_name = repo_dir.name.replace("_", "/", 1)
                all_repos.append(repo_name)

    logger.info(f"Found {len(all_repos)} total repositories with smell events")

    if not resume:
        return all_repos

    # Filter out completed repos
    pending = []
    skipped = 0
    for repo_name in all_repos:
        repo_dir_name = repo_name.replace("/", "_")
        metrics_file = output_dir / repo_dir_name / "validated_metrics.jsonl"
        instances_file = output_dir / repo_dir_name / "validated_smell_instances.jsonl"
        checkpoint_file = output_dir / repo_dir_name / "metrics_checkpoint.json"

        # Check if already complete by comparing checkpoint to instances
        if checkpoint_file.exists() and instances_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                n_processed = len(checkpoint.get('processed_ids', []))
                n_instances = sum(1 for _ in open(instances_file))
                if n_processed >= n_instances and n_instances > 0:
                    skipped += 1
                    continue
            except Exception:
                pass  # If we can't read checkpoint, process the repo

        pending.append(repo_name)

    logger.info(f"Skipping {skipped} already-complete repositories")
    return pending


def run_parallel(
    workers: int = 4,
    resume: bool = True,
    analysis_dir: Path = None,
    output_dir: Path = None
):
    """
    Run metrics calculation in parallel across repositories.

    Args:
        workers: Number of parallel workers (default 4)
        resume: Skip already-completed repositories
        analysis_dir: Directory with smell detection results
        output_dir: Directory for metrics output
    """
    analysis_dir = analysis_dir or config.RAW_SMELL_DETECTION_DIR
    output_dir = output_dir or config.PROCESSED_METRICS_DIR

    # Get repositories to process
    repos = get_pending_repos(analysis_dir, output_dir, resume)
    logger.info(f"Found {len(repos)} repositories to process")

    if not repos:
        logger.info("No repositories to process!")
        return

    # Prepare arguments for workers
    work_items = [
        (repo, analysis_dir, output_dir, resume)
        for repo in repos
    ]

    # Track progress
    completed = 0
    failed = 0
    start_time = time.time()

    logger.info(f"Starting parallel processing with {workers} workers...")

    # Run in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_repo, item): item[0] for item in work_items}

        for future in as_completed(futures):
            repo_name = futures[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    completed += 1
                    logger.info(f"[{completed}/{len(repos)}] Completed: {repo_name}")
                else:
                    failed += 1
                    logger.error(f"[{completed + failed}/{len(repos)}] Failed: {repo_name} - {result.get('error', 'unknown')}")
            except Exception as e:
                failed += 1
                logger.error(f"[{completed + failed}/{len(repos)}] Exception: {repo_name} - {e}")

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PARALLEL PROCESSING COMPLETE")
    logger.info(f"  Total repos: {len(repos)}")
    logger.info(f"  Completed: {completed}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    logger.info(f"  Avg per repo: {elapsed/len(repos):.1f} seconds")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run metrics calculation in parallel across repositories"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Number of parallel workers (default: 4, recommended: CPU cores - 1)"
    )
    parser.add_argument(
        "--resume", "-r", action="store_true",
        help="Skip already-completed repositories"
    )
    parser.add_argument(
        "--analysis-dir", type=str, default=str(config.RAW_SMELL_DETECTION_DIR),
        help="Directory with smell detection results"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(config.PROCESSED_METRICS_DIR),
        help="Directory for metrics output"
    )

    args = parser.parse_args()

    # Show CPU info
    cpu_count = mp.cpu_count()
    logger.info(f"System has {cpu_count} CPU cores")
    logger.info(f"Using {args.workers} workers")

    run_parallel(
        workers=args.workers,
        resume=args.resume,
        analysis_dir=Path(args.analysis_dir),
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
