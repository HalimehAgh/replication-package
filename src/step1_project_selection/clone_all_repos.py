"""
Clone All Repositories

Full clone (NOT shallow) of all repositories for complete commit history analysis.
Uses domain-based directory structure: repos/{domain_abbrev}/{owner}_{repo}/

Features:
- Full clone with complete history (required for commit-by-commit analysis)
- Parallel cloning with configurable workers
- Clone date documentation (as required by methodology)
- Verification of .git directory
- Logging of failures for investigation

Usage:
    python -m src.step1_commit_extraction.clone_all_repos [--parallel N] [--verify-only]
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

from src import config
from src.utils.logger import setup_logger
from src.utils.file_utils import repo_to_dir_name, get_repo_url

logger = setup_logger(__name__)


class RepoCloner:
    """Handles cloning of repositories with domain-based organization."""

    def __init__(self, max_parallel: int = 4):
        """
        Initialize RepoCloner.

        Args:
            max_parallel: Maximum number of parallel clone operations
        """
        self.max_parallel = max_parallel
        self.clone_date = datetime.now(timezone.utc)
        self.results: List[Dict] = []

    def load_repos(self) -> pd.DataFrame:
        """
        Load repository list from prepared dataset.

        Returns:
            DataFrame with repo_name, domain, size_category columns

        Raises:
            FileNotFoundError: If dataset file doesn't exist
        """
        # Try unified dataset first, then fall back to enriched dataset
        if (config.DATA_EXTERNAL_DIR / "repos_279_unified.csv").exists():
            filepath = config.DATA_EXTERNAL_DIR / "repos_279_unified.csv"
        elif config.REPOS_CSV.exists():
            filepath = config.REPOS_CSV
        else:
            raise FileNotFoundError(
                f"No repository dataset found. Run prepare_dataset.py first.\n"
                f"Expected: {config.REPOS_CSV}"
            )

        logger.info(f"Loading repositories from: {filepath}")
        df = pd.read_csv(filepath)

        # Normalize column names
        column_mapping = {
            "GitHub Repo": "repo_name",
            "github_repo": "repo_name",
            "repository": "repo_name",
            "Domain": "domain",
            "Size Category": "size_category",
            "size": "size_category",
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        required = ["repo_name", "domain"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Loaded {len(df)} repositories")
        return df

    def get_repo_path(self, repo_name: str, domain: str) -> Path:
        """
        Get the target path for a repository.

        Args:
            repo_name: Repository in 'owner/repo' format
            domain: Full domain name

        Returns:
            Path: repos/{domain_abbrev}/{owner}_{repo}/
        """
        domain_abbrev = config.DOMAIN_ABBREVIATIONS.get(domain, "other")
        return config.REPOS_DIR / domain_abbrev / repo_to_dir_name(repo_name)

    def is_valid_clone(self, repo_path: Path) -> bool:
        """
        Verify that a directory contains a valid git clone.

        Args:
            repo_path: Path to repository directory

        Returns:
            True if .git directory exists and has objects
        """
        git_dir = repo_path / ".git"
        if not git_dir.is_dir():
            return False

        # Check for essential git components
        objects_dir = git_dir / "objects"
        refs_dir = git_dir / "refs"
        head_file = git_dir / "HEAD"

        return objects_dir.is_dir() and refs_dir.is_dir() and head_file.is_file()

    def clone_repo(self, repo_name: str, domain: str) -> Dict:
        """
        Clone a single repository with full history.

        Args:
            repo_name: Repository in 'owner/repo' format
            domain: Full domain name

        Returns:
            Dictionary with clone result
        """
        repo_path = self.get_repo_path(repo_name, domain)
        repo_url = get_repo_url(repo_name)
        start_time = time.time()

        result = {
            "repo_name": repo_name,
            "domain": domain,
            "domain_abbrev": config.DOMAIN_ABBREVIATIONS.get(domain, "other"),
            "target_path": str(repo_path),
            "clone_date": self.clone_date.isoformat(),
            "status": "unknown",
            "duration_seconds": 0,
            "error": None,
        }

        try:
            # Check if already cloned and valid
            if repo_path.exists() and self.is_valid_clone(repo_path):
                result["status"] = "already_exists"
                result["duration_seconds"] = time.time() - start_time
                logger.info(f"[EXISTS] {repo_name}")
                return result

            # Create parent directory
            repo_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove partial clone if exists
            if repo_path.exists():
                import shutil
                shutil.rmtree(repo_path)

            # Full clone (not shallow - required for complete history)
            logger.info(f"[CLONE] {repo_name} -> {repo_path}")

            cmd = [
                "git", "clone",
                "--progress",
                repo_url,
                str(repo_path)
            ]

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per repo
            )

            if process.returncode != 0:
                result["status"] = "failed"
                result["error"] = process.stderr[:500] if process.stderr else "Unknown error"
                logger.error(f"[FAILED] {repo_name}: {result['error'][:100]}")
            else:
                # Verify clone
                if self.is_valid_clone(repo_path):
                    result["status"] = "success"
                    logger.info(f"[SUCCESS] {repo_name}")
                else:
                    result["status"] = "invalid"
                    result["error"] = "Clone completed but .git directory is invalid"
                    logger.error(f"[INVALID] {repo_name}")

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["error"] = "Clone timed out after 1 hour"
            logger.error(f"[TIMEOUT] {repo_name}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)[:500]
            logger.error(f"[ERROR] {repo_name}: {e}")

        result["duration_seconds"] = round(time.time() - start_time, 2)
        return result

    def clone_all(
        self,
        repos_df: pd.DataFrame,
        skip_existing: bool = True,
        auto_confirm: bool = False
    ) -> List[Dict]:
        """
        Clone all repositories in parallel.

        Args:
            repos_df: DataFrame with repo_name and domain columns
            skip_existing: If True, skip already cloned repos
            auto_confirm: If True, skip confirmation prompt

        Returns:
            List of clone results
        """
        # Filter to repos that need cloning if skip_existing
        repos_to_clone = []
        already_cloned = []

        for _, row in repos_df.iterrows():
            repo_name = row["repo_name"]
            domain = row["domain"]
            repo_path = self.get_repo_path(repo_name, domain)

            if skip_existing and repo_path.exists() and self.is_valid_clone(repo_path):
                already_cloned.append(repo_name)
            else:
                repos_to_clone.append((repo_name, domain))

        logger.info("=" * 80)
        logger.info("CLONE STATUS")
        logger.info("=" * 80)
        logger.info(f"Total repositories: {len(repos_df)}")
        logger.info(f"Already cloned: {len(already_cloned)}")
        logger.info(f"To clone: {len(repos_to_clone)}")
        logger.info(f"Parallel workers: {self.max_parallel}")

        if not repos_to_clone:
            logger.info("All repositories already cloned!")
            return []

        # Confirmation
        if not auto_confirm:
            response = input(f"\nClone {len(repos_to_clone)} repositories? (y/n): ").strip().lower()
            if response != "y":
                logger.info("Clone cancelled.")
                return []

        # Clone in parallel
        logger.info("")
        logger.info("Starting parallel cloning...")
        start_time = time.time()

        results = []
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # Submit all clone jobs
            future_to_repo = {
                executor.submit(self.clone_repo, repo_name, domain): repo_name
                for repo_name, domain in repos_to_clone
            }

            # Process as they complete
            for future in as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                try:
                    result = future.result()
                    results.append(result)

                    completed += 1
                    if result["status"] not in ["success", "already_exists"]:
                        failed += 1

                    if completed % 10 == 0 or completed == len(repos_to_clone):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(repos_to_clone) - completed) / rate if rate > 0 else 0
                        logger.info(
                            f"Progress: {completed}/{len(repos_to_clone)} "
                            f"({failed} failed) | "
                            f"Rate: {rate:.2f}/min | "
                            f"ETA: {eta/60:.1f} min"
                        )

                except Exception as e:
                    logger.error(f"Error processing {repo_name}: {e}")
                    results.append({
                        "repo_name": repo_name,
                        "status": "error",
                        "error": str(e)
                    })
                    failed += 1

        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 80)
        logger.info("CLONE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Successful: {completed - failed}")
        logger.info(f"Failed: {failed}")

        return results

    def save_results(self, results: List[Dict]) -> None:
        """
        Save clone results to CSV and JSON.

        Args:in the
            results: List of clone result dictionaries
        """
        config.RAW_PROJECT_SELECTION_DIR.mkdir(parents=True, exist_ok=True)

        # Save clone log CSV
        if results:
            df = pd.DataFrame(results)
            df.to_csv(config.CLONE_LOG_CSV, index=False)
            logger.info(f"Saved clone log: {config.CLONE_LOG_CSV}")

        # Save metadata JSON
        metadata = {
            "clone_date": self.clone_date.isoformat(),
            "total_repos": len(results),
            "success_count": len([r for r in results if r.get("status") == "success"]),
            "already_exists_count": len([r for r in results if r.get("status") == "already_exists"]),
            "failed_count": len([r for r in results if r.get("status") not in ["success", "already_exists"]]),
            "parallel_workers": self.max_parallel,
            "domain_counts": {},
        }

        # Count by domain
        for result in results:
            domain = result.get("domain_abbrev", "unknown")
            if domain not in metadata["domain_counts"]:
                metadata["domain_counts"][domain] = {"total": 0, "success": 0, "failed": 0}
            metadata["domain_counts"][domain]["total"] += 1
            if result.get("status") in ["success", "already_exists"]:
                metadata["domain_counts"][domain]["success"] += 1
            else:
                metadata["domain_counts"][domain]["failed"] += 1

        with open(config.CLONE_METADATA_JSON, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved clone metadata: {config.CLONE_METADATA_JSON}")

    def verify_all(self, repos_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Verify all repositories are properly cloned.

        Args:
            repos_df: DataFrame with repo_name and domain columns

        Returns:
            Tuple of (valid_repos, invalid_repos)
        """
        valid = []
        invalid = []

        for _, row in repos_df.iterrows():
            repo_name = row["repo_name"]
            domain = row["domain"]
            repo_path = self.get_repo_path(repo_name, domain)

            if repo_path.exists() and self.is_valid_clone(repo_path):
                valid.append(repo_name)
            else:
                invalid.append(repo_name)

        return valid, invalid


def main():
    """Main cloning workflow."""
    parser = argparse.ArgumentParser(
        description="Clone all repositories with full history"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=config.MAX_PARALLEL_WORKERS,
        help=f"Maximum parallel clone operations (default: {config.MAX_PARALLEL_WORKERS})"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing clones, don't clone new repos"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-clone even if repository already exists"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("CLONE ALL REPOSITORIES")
    logger.info("=" * 80)

    # Initialize cloner
    cloner = RepoCloner(max_parallel=args.parallel)

    # Load repository list
    try:
        repos_df = cloner.load_repos()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Verify only mode
    if args.verify_only:
        logger.info("")
        logger.info("Verifying existing clones...")
        valid, invalid = cloner.verify_all(repos_df)

        logger.info("")
        logger.info("=" * 80)
        logger.info("VERIFICATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Valid clones: {len(valid)}")
        logger.info(f"Missing/invalid: {len(invalid)}")

        if invalid:
            logger.info("")
            logger.info("Missing or invalid repositories:")
            for repo in invalid[:20]:
                logger.info(f"  - {repo}")
            if len(invalid) > 20:
                logger.info(f"  ... and {len(invalid) - 20} more")

        return

    # Clone repositories
    results = cloner.clone_all(
        repos_df,
        skip_existing=not args.force,
        auto_confirm=args.yes
    )

    # Save results
    if results:
        cloner.save_results(results)

    # Final verification
    logger.info("")
    logger.info("Final verification...")
    valid, invalid = cloner.verify_all(repos_df)
    logger.info(f"Verified: {len(valid)}/{len(repos_df)} repositories ready for analysis")

    if invalid:
        logger.warning(f"{len(invalid)} repositories still missing or invalid")


if __name__ == "__main__":
    main()
