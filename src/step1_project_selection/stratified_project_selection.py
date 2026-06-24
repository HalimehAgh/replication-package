"""
Stratified Multi-Criteria Project Selection

After analyzing all projects, selects a subset using multi-criteria optimization:

Selection Criteria:
- Criterion A: Statistical Power - Maximize smell types reaching >=30 events
- Criterion B: Balanced Stratification - Proportional allocation across 21 strata
- Criterion C: Organizational Diversity - Max 2 projects per organization
- Criterion D: Anti-Dominance - No project >10% of total events

Conflict Resolution Priority:
1. Anti-dominance (hard constraint)
2. Organizational diversity (hard constraint)
3. Balanced stratification (soft target)
4. Statistical power (optimize)

Usage:
    python -m src.step1_commit_extraction.stratified_project_selection [--target-size N]
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

from src import config
from src.utils.logger import setup_logger
from src.utils.file_utils import extract_organization

logger = setup_logger(__name__)


class StratifiedSelector:
    """
    Multi-criteria stratified project selection from analyzed projects.
    """

    def __init__(self, target_sample_size: int = 200):
        """
        Initialize selector.

        Args:
            target_sample_size: Target number of projects to select
        """
        self.target_sample_size = target_sample_size
        self.max_per_org = config.MAX_PROJECTS_PER_ORG
        self.max_event_pct = config.MAX_EVENT_CONTRIBUTION_PCT
        self.min_per_stratum = config.MIN_PROJECTS_PER_STRATUM
        self.tolerance = config.STRATIFICATION_TOLERANCE

    def load_project_data(self) -> pd.DataFrame:
        """
        Load project summary with event counts.

        Returns:
            DataFrame with project_id, domain, size_category, event counts
        """
        # Load project summary from smell detection
        if not config.PROJECT_SUMMARY_CSV.exists():
            raise FileNotFoundError(
                f"Project summary not found: {config.PROJECT_SUMMARY_CSV}\n"
                "Run parallel_runner.py first to generate smell detection results."
            )

        logger.info(f"Loading project summary from: {config.PROJECT_SUMMARY_CSV}")
        df = pd.read_csv(config.PROJECT_SUMMARY_CSV)

        # Ensure required columns
        required = ["project_id", "total_introductions", "total_removals"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Calculate total events
        df["total_events"] = df["total_introductions"] + df["total_removals"]

        # Extract organization
        df["organization"] = df["project_id"].apply(extract_organization)

        # Count smell types with events (smell breadth)
        smell_intro_cols = [c for c in df.columns if c.startswith("intro_")]
        smell_removal_cols = [c for c in df.columns if c.startswith("removal_")]

        def count_smell_types(row):
            count = 0
            for col in smell_intro_cols + smell_removal_cols:
                if col in row and pd.notna(row[col]) and row[col] > 0:
                    # Extract smell type from column name
                    if col.startswith("intro_"):
                        smell_type = col[6:]
                    else:
                        smell_type = col[8:]
                    count += 1
            return count

        # Deduplicate smell types
        def count_unique_smell_types(row):
            smell_types = set()
            for col in smell_intro_cols:
                if col in row and pd.notna(row[col]) and row[col] > 0:
                    smell_types.add(col[6:])
            for col in smell_removal_cols:
                if col in row and pd.notna(row[col]) and row[col] > 0:
                    smell_types.add(col[8:])
            return len(smell_types)

        df["smell_breadth"] = df.apply(count_unique_smell_types, axis=1)

        # Load stars from enriched dataset if available
        if config.REPOS_CSV.exists():
            repos_df = pd.read_csv(config.REPOS_CSV)
            # Normalize column names
            if "GitHub Repo" in repos_df.columns:
                repos_df = repos_df.rename(columns={"GitHub Repo": "repo_name"})
            if "Stars" in repos_df.columns:
                repos_df = repos_df.rename(columns={"Stars": "stars"})

            if "repo_name" in repos_df.columns and "stars" in repos_df.columns:
                # Merge stars
                repos_df["repo_name"] = repos_df["repo_name"].str.strip()
                df["project_id"] = df["project_id"].str.strip()
                df = df.merge(
                    repos_df[["repo_name", "stars"]],
                    left_on="project_id",
                    right_on="repo_name",
                    how="left"
                ).drop(columns=["repo_name"], errors="ignore")

        if "stars" not in df.columns:
            df["stars"] = 0

        logger.info(f"Loaded {len(df)} projects with event data")
        return df

    def calculate_stratum_targets(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Calculate proportional allocation targets for each stratum.

        Args:
            df: Project DataFrame

        Returns:
            Dictionary mapping stratum to target count
        """
        targets = {}

        for domain in config.DOMAINS:
            for size in ["small", "medium", "large"]:
                stratum_key = f"{domain}|{size}"

                # Count projects in this stratum
                stratum_count = len(df[
                    (df["domain"] == domain) &
                    (df["size_category"].str.lower() == size)
                ])

                # Proportional allocation
                target = round((stratum_count / len(df)) * self.target_sample_size)

                # Ensure minimum if stratum has projects
                if stratum_count > 0:
                    target = max(target, self.min_per_stratum)

                targets[stratum_key] = min(target, stratum_count)  # Can't exceed available

        return targets

    def rank_projects_in_stratum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank projects within each stratum by selection criteria.

        Ranking order:
        1. Total events (descending)
        2. Smell breadth (descending)
        3. Stars (descending)

        Args:
            df: Project DataFrame

        Returns:
            DataFrame with rank column
        """
        # Create stratum column
        df["stratum"] = df["domain"] + "|" + df["size_category"].str.lower()

        # Sort by ranking criteria
        df = df.sort_values(
            by=["stratum", "total_events", "smell_breadth", "stars"],
            ascending=[True, False, False, False]
        ).reset_index(drop=True)

        # Assign ranks within each stratum
        df["stratum_rank"] = df.groupby("stratum").cumcount() + 1

        return df

    def apply_organizational_diversity(
        self,
        ranked_df: pd.DataFrame,
        targets: Dict[str, int]
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Select projects while enforcing organizational diversity constraint.

        Args:
            ranked_df: Ranked project DataFrame
            targets: Target counts per stratum

        Returns:
            (selected_df, exclusions_dict)
        """
        selected = []
        org_counts = defaultdict(int)
        exclusions = defaultdict(list)  # Tracks why projects were excluded

        for stratum, target in targets.items():
            if target == 0:
                continue

            # Get projects in this stratum
            stratum_projects = ranked_df[ranked_df["stratum"] == stratum].copy()
            stratum_selected = 0

            for _, project in stratum_projects.iterrows():
                org = project["organization"]

                # Check organizational diversity constraint
                if org_counts[org] >= self.max_per_org:
                    exclusions["org_limit"].append(
                        f"{project['project_id']} ({org} already has {org_counts[org]} projects)"
                    )
                    continue

                # Select project
                selected.append(project)
                org_counts[org] += 1
                stratum_selected += 1

                if stratum_selected >= target:
                    break

        selected_df = pd.DataFrame(selected)
        return selected_df, dict(exclusions)

    def check_anti_dominance(
        self,
        selected_df: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """
        Check if any project violates the anti-dominance constraint.

        Args:
            selected_df: Selected projects DataFrame

        Returns:
            (passes, list of violating project_ids)
        """
        total_intro = selected_df["total_introductions"].sum()
        total_removal = selected_df["total_removals"].sum()

        violators = []

        for _, project in selected_df.iterrows():
            intro_pct = project["total_introductions"] / total_intro if total_intro > 0 else 0
            removal_pct = project["total_removals"] / total_removal if total_removal > 0 else 0

            if intro_pct > self.max_event_pct or removal_pct > self.max_event_pct:
                violators.append({
                    "project_id": project["project_id"],
                    "intro_pct": intro_pct,
                    "removal_pct": removal_pct,
                })

        return len(violators) == 0, violators

    def replace_dominant_projects(
        self,
        selected_df: pd.DataFrame,
        all_ranked_df: pd.DataFrame,
        org_counts: Dict[str, int],
        violators: List[Dict]
    ) -> pd.DataFrame:
        """
        Replace dominant projects with next-ranked alternatives.

        Args:
            selected_df: Current selection
            all_ranked_df: All ranked projects
            org_counts: Current organization counts
            violators: List of violating projects

        Returns:
            Updated selection DataFrame
        """
        for violator in violators:
            project_id = violator["project_id"]
            logger.warning(
                f"Replacing dominant project: {project_id} "
                f"(intro: {violator['intro_pct']:.1%}, removal: {violator['removal_pct']:.1%})"
            )

            # Get stratum
            project_row = selected_df[selected_df["project_id"] == project_id].iloc[0]
            stratum = project_row["stratum"]
            org = project_row["organization"]

            # Remove from selection
            selected_df = selected_df[selected_df["project_id"] != project_id]
            org_counts[org] -= 1

            # Find replacement in same stratum
            stratum_candidates = all_ranked_df[
                (all_ranked_df["stratum"] == stratum) &
                (~all_ranked_df["project_id"].isin(selected_df["project_id"]))
            ]

            for _, candidate in stratum_candidates.iterrows():
                cand_org = candidate["organization"]
                if org_counts.get(cand_org, 0) < self.max_per_org:
                    # Add replacement
                    selected_df = pd.concat([selected_df, candidate.to_frame().T], ignore_index=True)
                    org_counts[cand_org] = org_counts.get(cand_org, 0) + 1
                    logger.info(f"  Replaced with: {candidate['project_id']}")
                    break
            else:
                logger.warning(f"  No valid replacement found for stratum {stratum}")

        return selected_df

    def calculate_stratification_deviations(
        self,
        selected_df: pd.DataFrame,
        targets: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Calculate deviation from target for each stratum.

        Args:
            selected_df: Selected projects
            targets: Target counts per stratum

        Returns:
            DataFrame with stratum distribution
        """
        results = []

        for stratum, target in targets.items():
            domain, size = stratum.split("|")
            actual = len(selected_df[selected_df["stratum"] == stratum]) if "stratum" in selected_df.columns else 0

            deviation_pct = ((actual - target) / target * 100) if target > 0 else 0

            results.append({
                "domain": domain,
                "size": size,
                "target_count": target,
                "actual_count": actual,
                "deviation_pct": round(deviation_pct, 1),
                "within_tolerance": abs(deviation_pct) <= self.tolerance * 100,
            })

        return pd.DataFrame(results)

    def calculate_smell_coverage(
        self,
        selected_df: pd.DataFrame,
        all_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate event coverage by smell type for selected projects.

        Args:
            selected_df: Selected projects
            all_df: All projects (for reference)

        Returns:
            DataFrame with smell coverage statistics
        """
        # Get smell columns
        intro_cols = [c for c in selected_df.columns if c.startswith("intro_")]
        removal_cols = [c for c in selected_df.columns if c.startswith("removal_")]

        results = []

        # Process each smell type
        smell_types = set()
        for col in intro_cols:
            smell_types.add(col[6:])
        for col in removal_cols:
            smell_types.add(col[8:])

        for smell_type in sorted(smell_types):
            intro_col = f"intro_{smell_type}"
            removal_col = f"removal_{smell_type}"

            intro_count = selected_df[intro_col].sum() if intro_col in selected_df.columns else 0
            removal_count = selected_df[removal_col].sum() if removal_col in selected_df.columns else 0
            total = intro_count + removal_count

            results.append({
                "smell_type": smell_type,
                "introduction_events": int(intro_count),
                "removal_events": int(removal_count),
                "total_events": int(total),
                "meets_threshold_30": total >= config.MIN_EVENTS_PER_SMELL,
            })

        return pd.DataFrame(results).sort_values("total_events", ascending=False)

    def select_projects(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Main selection algorithm implementing multi-criteria optimization.

        Args:
            df: Project DataFrame with event data

        Returns:
            (selected_df, selection_metadata)
        """
        logger.info("=" * 80)
        logger.info("STRATIFIED PROJECT SELECTION")
        logger.info("=" * 80)
        logger.info(f"Total projects: {len(df)}")
        logger.info(f"Target sample size: {self.target_sample_size}")
        logger.info(f"Max per organization: {self.max_per_org}")
        logger.info(f"Max event contribution: {self.max_event_pct:.0%}")

        # Step 1: Calculate proportional allocation
        logger.info("")
        logger.info("Step 1: Calculating proportional allocation...")
        targets = self.calculate_stratum_targets(df)
        logger.info(f"  Allocated {sum(targets.values())} slots across {len([t for t in targets.values() if t > 0])} strata")

        # Step 2: Rank projects within each stratum
        logger.info("")
        logger.info("Step 2: Ranking projects by criteria...")
        ranked_df = self.rank_projects_in_stratum(df)

        # Step 3: Apply organizational diversity constraint
        logger.info("")
        logger.info("Step 3: Applying organizational diversity constraint...")
        selected_df, exclusions = self.apply_organizational_diversity(ranked_df, targets)
        logger.info(f"  Selected {len(selected_df)} projects")
        if "org_limit" in exclusions:
            logger.info(f"  Excluded {len(exclusions['org_limit'])} due to org limit")

        # Step 4: Verify anti-dominance constraint
        logger.info("")
        logger.info("Step 4: Verifying anti-dominance constraint...")

        # Calculate org counts for potential replacements
        org_counts = defaultdict(int)
        for _, p in selected_df.iterrows():
            org_counts[p["organization"]] += 1

        passes, violators = self.check_anti_dominance(selected_df)
        if not passes:
            logger.warning(f"  Found {len(violators)} dominant projects, replacing...")
            selected_df = self.replace_dominant_projects(
                selected_df, ranked_df, org_counts, violators
            )
            # Re-check
            passes, violators = self.check_anti_dominance(selected_df)
            if not passes:
                logger.warning(f"  Still have {len(violators)} dominant projects after replacement")
        else:
            logger.info("  Anti-dominance constraint satisfied")

        # Step 5: Verify stratification balance
        logger.info("")
        logger.info("Step 5: Verifying stratification balance...")
        deviation_df = self.calculate_stratification_deviations(selected_df, targets)
        within_tolerance = deviation_df["within_tolerance"].all()

        if within_tolerance:
            logger.info(f"  All strata within ±{self.tolerance:.0%} tolerance")
        else:
            outside = deviation_df[~deviation_df["within_tolerance"]]
            logger.warning(f"  {len(outside)} strata outside tolerance")

        # Build metadata
        metadata = {
            "total_projects": len(df),
            "target_sample_size": self.target_sample_size,
            "actual_selected": len(selected_df),
            "stratum_targets": targets,
            "exclusions": exclusions,
            "anti_dominance_passed": passes,
            "anti_dominance_violators": violators,
            "stratification_within_tolerance": within_tolerance,
            "selection_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return selected_df, metadata

    def generate_selection_report(
        self,
        selected_df: pd.DataFrame,
        all_df: pd.DataFrame,
        metadata: Dict,
        deviation_df: pd.DataFrame,
        coverage_df: pd.DataFrame
    ) -> str:
        """Generate comprehensive selection report."""
        lines = [
            "=" * 80,
            "STRATIFIED PROJECT SELECTION REPORT",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 80,
            "",
            "SUMMARY",
            "-" * 40,
            f"Total projects analyzed: {metadata['total_projects']}",
            f"Target sample size: {metadata['target_sample_size']}",
            f"Actual selected: {metadata['actual_selected']}",
            "",
            f"Anti-dominance constraint: {'PASSED' if metadata['anti_dominance_passed'] else 'FAILED'}",
            f"Stratification tolerance: {'PASSED' if metadata['stratification_within_tolerance'] else 'SOME STRATA OUTSIDE'}",
            "",
            "STRATUM DISTRIBUTION",
            "-" * 40,
        ]

        for _, row in deviation_df.iterrows():
            status = "OK" if row["within_tolerance"] else "OUTSIDE"
            lines.append(
                f"  {row['domain'][:30]:30} | {row['size']:6} | "
                f"Target: {row['target_count']:3} | Actual: {row['actual_count']:3} | "
                f"Dev: {row['deviation_pct']:+6.1f}% | {status}"
            )

        lines.extend([
            "",
            "SMELL COVERAGE",
            "-" * 40,
        ])

        for _, row in coverage_df.iterrows():
            status = ">=30" if row["meets_threshold_30"] else "<30"
            lines.append(
                f"  {row['smell_type']:35} | "
                f"Intro: {row['introduction_events']:5} | "
                f"Removal: {row['removal_events']:5} | "
                f"Total: {row['total_events']:5} | {status}"
            )

        # Count smell types meeting threshold
        meets_threshold = coverage_df["meets_threshold_30"].sum()
        total_smells = len(coverage_df)
        lines.extend([
            "",
            f"Smell types meeting >=30 threshold: {meets_threshold}/{total_smells}",
        ])

        if metadata.get("exclusions", {}).get("org_limit"):
            lines.extend([
                "",
                "ORGANIZATIONAL DIVERSITY EXCLUSIONS",
                "-" * 40,
            ])
            for exclusion in metadata["exclusions"]["org_limit"][:20]:
                lines.append(f"  {exclusion}")
            if len(metadata["exclusions"]["org_limit"]) > 20:
                lines.append(f"  ... and {len(metadata['exclusions']['org_limit']) - 20} more")

        if metadata.get("anti_dominance_violators"):
            lines.extend([
                "",
                "ANTI-DOMINANCE VIOLATIONS",
                "-" * 40,
            ])
            for v in metadata["anti_dominance_violators"]:
                lines.append(
                    f"  {v['project_id']}: "
                    f"intro={v['intro_pct']:.1%}, removal={v['removal_pct']:.1%}"
                )

        lines.extend([
            "",
            "SELECTED PROJECTS (by domain)",
            "-" * 40,
        ])

        for domain in config.DOMAINS:
            domain_projects = selected_df[selected_df["domain"] == domain]
            lines.append(f"\n{domain} ({len(domain_projects)} projects):")
            for _, p in domain_projects.head(10).iterrows():
                lines.append(
                    f"  {p['project_id']:40} | "
                    f"Events: {p['total_events']:5} | "
                    f"Size: {p.get('size_category', 'N/A'):6}"
                )
            if len(domain_projects) > 10:
                lines.append(f"  ... and {len(domain_projects) - 10} more")

        return "\n".join(lines)


def main():
    """Main selection workflow."""
    parser = argparse.ArgumentParser(
        description="Stratified multi-criteria project selection from analyzed projects"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=200,
        help="Target number of projects to select (default: 200)"
    )
    parser.add_argument(
        "--max-per-org",
        type=int,
        default=config.MAX_PROJECTS_PER_ORG,
        help=f"Max projects per organization (default: {config.MAX_PROJECTS_PER_ORG})"
    )

    args = parser.parse_args()

    # Initialize selector
    selector = StratifiedSelector(target_sample_size=args.target_size)
    selector.max_per_org = args.max_per_org

    # Ensure output directory exists
    config.RAW_PROJECT_SELECTION_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        df = selector.load_project_data()

        # Run selection
        selected_df, metadata = selector.select_projects(df)

        # Calculate additional metrics
        targets = metadata["stratum_targets"]
        deviation_df = selector.calculate_stratification_deviations(selected_df, targets)
        coverage_df = selector.calculate_smell_coverage(selected_df, df)

        # Save outputs
        logger.info("")
        logger.info("=" * 80)
        logger.info("SAVING OUTPUTS")
        logger.info("=" * 80)

        # Selected projects CSV
        output_cols = [
            "project_id", "domain", "size_category", "organization",
            "total_events", "total_introductions", "total_removals",
            "smell_breadth", "stars"
        ]
        available_cols = [c for c in output_cols if c in selected_df.columns]
        selected_df[available_cols].to_csv(config.SELECTED_PROJECTS_CSV, index=False)
        logger.info(f"Saved: {config.SELECTED_PROJECTS_CSV}")

        # Stratum distribution CSV
        deviation_df.to_csv(config.STRATUM_DISTRIBUTION_CSV, index=False)
        logger.info(f"Saved: {config.STRATUM_DISTRIBUTION_CSV}")

        # Smell coverage CSV
        coverage_df.to_csv(config.SMELL_COVERAGE_CSV, index=False)
        logger.info(f"Saved: {config.SMELL_COVERAGE_CSV}")

        # Selection report
        report = selector.generate_selection_report(
            selected_df, df, metadata, deviation_df, coverage_df
        )
        with open(config.SELECTION_REPORT, "w") as f:
            f.write(report)
        logger.info(f"Saved: {config.SELECTION_REPORT}")

        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("SELECTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Selected {len(selected_df)} projects")
        logger.info(f"Total events: {selected_df['total_events'].sum()}")
        logger.info(f"Smell types >= 30 events: {coverage_df['meets_threshold_30'].sum()}/{len(coverage_df)}")

    except FileNotFoundError as e:
        logger.error(str(e))
        return
    except Exception as e:
        logger.error(f"Selection failed: {e}")
        raise


if __name__ == "__main__":
    main()
