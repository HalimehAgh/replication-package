"""
Prepare Project Dataset for Analysis

Validates and merges the enriched project dataset for analysis.

Required input files in data/external/:
- repos_279_enriched.csv (projects with domain classifications)
- smell_analysis_279.csv (projects with smell counts) - optional

Usage:
    python -m src.step1_commit_extraction.prepare_dataset
"""

import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.utils.logger import setup_logger
from src.utils.file_utils import extract_organization

logger = setup_logger(__name__)


def load_and_validate_repos(filepath: Path) -> pd.DataFrame:
    """
    Load and validate the repos with domains CSV.

    Expected columns: repo_name (or GitHub Repo), domain, size_category, LOC, stars

    Args:
        filepath: Path to repos CSV

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If validation fails
    """
    logger.info(f"Loading repos from: {filepath}")
    df = pd.read_csv(filepath)

    # Normalize column names (handle different naming conventions)
    column_mapping = {
        "GitHub Repo": "repo_name",
        "github_repo": "repo_name",
        "repository": "repo_name",
        "Domain": "domain",
        "Size Category": "size_category",
        "size": "size_category",
        "Project Size": "size_category",
        "project_size": "size_category",
        "Lines of Code": "loc",
        "LOC": "loc",
        "Stars": "stars",
    }

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Normalize size_category values to lowercase
    if "size_category" in df.columns:
        df["size_category"] = df["size_category"].str.lower().str.strip()

    # Validate required columns
    required = ["repo_name", "domain", "size_category"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    # Validate project count
    # Validate project count (informational only)
        logger.info(f"Total projects: {len(df)}")

    # Validate domains
    expected_domains = set(config.DOMAINS)
    actual_domains = set(df["domain"].unique())
    missing_domains = expected_domains - actual_domains
    extra_domains = actual_domains - expected_domains

    if missing_domains:
        logger.warning(f"Missing expected domains: {missing_domains}")
    if extra_domains:
        logger.warning(f"Unexpected domains found: {extra_domains}")

    # Validate size categories
    expected_sizes = {"small", "medium", "large"}
    actual_sizes = set(df["size_category"].str.lower().unique())
    if actual_sizes != expected_sizes:
        logger.warning(f"Size categories mismatch. Expected: {expected_sizes}, Found: {actual_sizes}")

    # Log distribution
    logger.info("Domain distribution:")
    for domain, count in df["domain"].value_counts().items():
        logger.info(f"  {domain}: {count}")

    logger.info("Size distribution:")
    for size, count in df["size_category"].value_counts().items():
        logger.info(f"  {size}: {count}")

    return df


def load_and_validate_smells(filepath: Path) -> pd.DataFrame:
    """
    Load and validate the smell analysis CSV.

    Expected columns: repo_name (or similar), smell counts per type

    Args:
        filepath: Path to smell analysis CSV

    Returns:
        Validated DataFrame
    """
    logger.info(f"Loading smell analysis from: {filepath}")
    df = pd.read_csv(filepath)

    # Normalize column names
    column_mapping = {
        "GitHub Repo": "repo_name",
        "github_repo": "repo_name",
        "repository": "repo_name",
    }

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Validate repo_name column exists
    if "repo_name" not in df.columns:
        # Try to find any column that might be the repo name
        potential_cols = [c for c in df.columns if "repo" in c.lower() or "name" in c.lower()]
        if potential_cols:
            df = df.rename(columns={potential_cols[0]: "repo_name"})
            logger.info(f"Using '{potential_cols[0]}' as repo_name column")
        else:
            raise ValueError(f"Cannot find repo_name column. Available: {list(df.columns)}")

    logger.info(f"Loaded {len(df)} projects with smell analysis")

    # Log smell columns found
    smell_columns = [c for c in df.columns if c != "repo_name" and df[c].dtype in ["int64", "float64"]]
    logger.info(f"Smell columns found: {len(smell_columns)}")

    return df


def create_unified_dataset(repos_df: pd.DataFrame, smells_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unified dataset by merging repos with smell analysis data.

    Args:
        repos_df: DataFrame with repo info and domains
        smells_df: DataFrame with smell counts per project

    Returns:
        Unified DataFrame with all columns
    """
    logger.info("Creating unified dataset...")

    # Ensure both DataFrames use the same repo_name format
    # Normalize repo names (remove whitespace, lowercase for matching)
    repos_df["repo_name_normalized"] = repos_df["repo_name"].str.strip().str.lower()
    smells_df["repo_name_normalized"] = smells_df["repo_name"].str.strip().str.lower()

    # Merge on normalized repo name
    unified = repos_df.merge(
        smells_df,
        on="repo_name_normalized",
        how="left",
        suffixes=("", "_smell")
    )

    # Use original repo_name from repos_df
    if "repo_name_smell" in unified.columns:
        unified = unified.drop(columns=["repo_name_smell"])

    # Drop normalization column
    unified = unified.drop(columns=["repo_name_normalized"])

    # Add organization column
    unified["organization"] = unified["repo_name"].apply(extract_organization)

    # Log merge results
    matched = unified.dropna(subset=[c for c in smells_df.columns if c not in ["repo_name", "repo_name_normalized"]][:1])
    logger.info(f"Merged: {len(matched)} projects with smell data out of {len(repos_df)}")

    unmatched = len(repos_df) - len(matched)
    if unmatched > 0:
        logger.warning(f"{unmatched} projects without smell data")

    return unified


def generate_statistics_report(unified_df: pd.DataFrame) -> str:
    """
    Generate a statistics report for the unified dataset.

    Args:
        unified_df: Unified DataFrame

    Returns:
        Report as string
    """
    lines = [
        "=" * 80,
        "PROJECT DATASET STATISTICS",
        "=" * 80,
        "",
        f"Total projects: {len(unified_df)}",
        "",
        "DOMAIN DISTRIBUTION:",
    ]

    for domain in config.DOMAINS:
        count = len(unified_df[unified_df["domain"] == domain])
        lines.append(f"  {domain}: {count}")

    lines.append("")
    lines.append("SIZE DISTRIBUTION:")
    for size in ["small", "medium", "large"]:
        count = len(unified_df[unified_df["size_category"].str.lower() == size])
        lines.append(f"  {size}: {count}")

    lines.append("")
    lines.append("STRATUM DISTRIBUTION (Domain x Size):")
    for domain in config.DOMAINS:
        domain_abbrev = config.DOMAIN_ABBREVIATIONS.get(domain, "other")
        for size in ["small", "medium", "large"]:
            count = len(unified_df[
                (unified_df["domain"] == domain) &
                (unified_df["size_category"].str.lower() == size)
            ])
            lines.append(f"  {domain_abbrev}_{size}: {count}")

    lines.append("")
    lines.append("ORGANIZATION DIVERSITY:")
    org_counts = unified_df["organization"].value_counts()
    lines.append(f"  Unique organizations: {len(org_counts)}")
    lines.append(f"  Organizations with 2+ projects: {len(org_counts[org_counts >= 2])}")
    lines.append(f"  Top organizations:")
    for org, count in org_counts.head(10).items():
        lines.append(f"    {org}: {count}")

    return "\n".join(lines)


def main():
    """Main data preparation workflow."""
    parser = argparse.ArgumentParser(
        description="Prepare project dataset for analysis"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-creation of unified dataset even if it exists"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("STEP 1: PREPARE PROJECT DATASET")
    logger.info("=" * 80)

    # Ensure directories exist
    config.ensure_directories()

    # Load and validate repos
    if not config.REPOS_CSV.exists():
        logger.error(f"Repos file not found: {config.REPOS_CSV}")
        logger.error("Please copy repos_279_enriched.csv to data/external/")
        return

    repos_df = load_and_validate_repos(config.REPOS_CSV)

    # Step 3: Load and validate smells (optional - may not exist yet)
    smells_df = None
    if config.SMELL_ANALYSIS_CSV.exists():
        smells_df = load_and_validate_smells(config.SMELL_ANALYSIS_CSV)
    else:
        logger.info("Smell analysis file not found - will proceed without it")

    # Step 4: Create unified dataset
    logger.info("")
    unified_output = config.DATA_EXTERNAL_DIR / "repos_279_unified.csv"

    if unified_output.exists() and not args.force:
        logger.info(f"Unified dataset already exists: {unified_output}")
        unified_df = pd.read_csv(unified_output)
    else:
        if smells_df is not None:
            unified_df = create_unified_dataset(repos_df, smells_df)
        else:
            unified_df = repos_df.copy()
            unified_df["organization"] = unified_df["repo_name"].apply(extract_organization)

        unified_df.to_csv(unified_output, index=False)
        logger.info(f"Saved unified dataset: {unified_output}")

    # Step 5: Generate and print statistics
    logger.info("")
    report = generate_statistics_report(unified_df)
    logger.info(report)

    # Save statistics report
    stats_file = config.RAW_PROJECT_SELECTION_DIR / "dataset_statistics.txt"
    config.RAW_PROJECT_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w") as f:
        f.write(report)
    logger.info(f"Saved statistics: {stats_file}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
