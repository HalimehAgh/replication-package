"""
Smell Transition Identification Module

Identifies smell introduction (A) and removal (B) commits from smell event data.
Creates matched pairs for before/after analysis.

Key Concepts:
- Introduction: Commit A where smell first appears (A-1 → A transition)
- Removal: Commit B where smell is removed (B-1 → B transition)
- Matched Pair: Complete lifecycle from introduction to removal

Usage
-----
python -m src.step2_smell_detection.identify_transitions
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

import pandas as pd

from src import config
from src.utils import read_jsonl, write_jsonl, write_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class SmellTransition:
    """Represents a smell introduction or removal transition."""
    smell_instance_id: str
    project_name: str
    smell_type: str
    file_path: str
    function_id: str
    transition_type: str  # 'introduction' or 'removal'

    # Commit before transition
    before_commit_hash: str
    before_commit_type: str  # 'A-1' or 'B-1'
    before_timestamp: str
    before_is_smelly: bool

    # Commit after transition
    after_commit_hash: str
    after_commit_type: str  # 'A' or 'B'
    after_timestamp: str
    after_is_smelly: bool

    # Additional metadata
    author: str
    commit_message: str
    lines_added: int
    lines_deleted: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate time between before and after commits."""
        try:
            before_dt = datetime.fromisoformat(self.before_timestamp.replace('Z', '+00:00'))
            after_dt = datetime.fromisoformat(self.after_timestamp.replace('Z', '+00:00'))
            return (after_dt - before_dt).total_seconds()
        except (ValueError, AttributeError):
            return None


@dataclass
class SmellLifecycle:
    """Represents complete lifecycle of a smell instance."""
    smell_instance_id: str
    project_name: str
    smell_type: str
    file_path: str
    function_id: str

    # Introduction
    introduction_commit: str
    introduction_timestamp: str
    introduction_before_commit: Optional[str]

    # Removal (None if smell still exists)
    removal_commit: Optional[str]
    removal_timestamp: Optional[str]
    removal_before_commit: Optional[str]

    # Lifecycle status
    is_complete: bool  # True if both introduced and removed
    lifespan_commits: int  # Number of commits smell existed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# ============================================================
# TRANSITION IDENTIFICATION
# ============================================================

def identify_transitions_from_events(events: List[Dict]) -> Tuple[List[SmellTransition], List[SmellTransition]]:
    """
    Identify introduction and removal transitions from smell events.

    Events are grouped by smell_instance_id, then paired:
    - A-1 + A = introduction transition
    - B-1 + B = removal transition

    Args:
        events: List of smell event dictionaries

    Returns:
        Tuple of (introduction_transitions, removal_transitions)
    """
    # Group events by smell instance
    events_by_instance: Dict[str, List[Dict]] = {}

    for event in events:
        instance_id = event.get('smell_instance_id', '')
        if instance_id:
            if instance_id not in events_by_instance:
                events_by_instance[instance_id] = []
            events_by_instance[instance_id].append(event)

    introductions = []
    removals = []

    for instance_id, instance_events in events_by_instance.items():
        # Sort events by timestamp
        instance_events.sort(key=lambda x: x.get('timestamp', ''))

        # Find A-1/A pairs (introduction)
        a_minus_1 = None
        a_event = None

        for event in instance_events:
            commit_type = event.get('commit_type', '')

            if commit_type == 'A-1':
                a_minus_1 = event
            elif commit_type == 'A':
                a_event = event

                # Create introduction transition
                if a_minus_1 or a_event:
                    transition = SmellTransition(
                        smell_instance_id=instance_id,
                        project_name=event.get('project_name', ''),
                        smell_type=event.get('smell_type', ''),
                        file_path=event.get('file_path', ''),
                        function_id=event.get('function_id', ''),
                        transition_type='introduction',
                        before_commit_hash=a_minus_1.get('commit_hash', '') if a_minus_1 else '',
                        before_commit_type='A-1',
                        before_timestamp=a_minus_1.get('timestamp', '') if a_minus_1 else '',
                        before_is_smelly=False,
                        after_commit_hash=a_event.get('commit_hash', ''),
                        after_commit_type='A',
                        after_timestamp=a_event.get('timestamp', ''),
                        after_is_smelly=True,
                        author=a_event.get('author', ''),
                        commit_message=a_event.get('commit_message', ''),
                        lines_added=a_event.get('lines_added', 0),
                        lines_deleted=a_event.get('lines_deleted', 0),
                    )
                    introductions.append(transition)

        # Find B-1/B pairs (removal)
        b_minus_1 = None
        b_event = None

        for event in instance_events:
            commit_type = event.get('commit_type', '')

            if commit_type == 'B-1':
                b_minus_1 = event
            elif commit_type == 'B':
                b_event = event

                # Create removal transition
                if b_minus_1 or b_event:
                    transition = SmellTransition(
                        smell_instance_id=instance_id,
                        project_name=event.get('project_name', ''),
                        smell_type=event.get('smell_type', ''),
                        file_path=event.get('file_path', ''),
                        function_id=event.get('function_id', ''),
                        transition_type='removal',
                        before_commit_hash=b_minus_1.get('commit_hash', '') if b_minus_1 else '',
                        before_commit_type='B-1',
                        before_timestamp=b_minus_1.get('timestamp', '') if b_minus_1 else '',
                        before_is_smelly=True,
                        after_commit_hash=b_event.get('commit_hash', ''),
                        after_commit_type='B',
                        after_timestamp=b_event.get('timestamp', ''),
                        after_is_smelly=False,
                        author=b_event.get('author', ''),
                        commit_message=b_event.get('commit_message', ''),
                        lines_added=b_event.get('lines_added', 0),
                        lines_deleted=b_event.get('lines_deleted', 0),
                    )
                    removals.append(transition)

    logger.info(f"Identified {len(introductions)} introductions and {len(removals)} removals")
    return introductions, removals


def build_smell_lifecycles(
    introductions: List[SmellTransition],
    removals: List[SmellTransition]
) -> List[SmellLifecycle]:
    """
    Build complete smell lifecycles by matching introductions with removals.

    Args:
        introductions: List of introduction transitions
        removals: List of removal transitions

    Returns:
        List of SmellLifecycle objects
    """
    # Create lookup for removals
    removal_by_instance: Dict[str, SmellTransition] = {}
    for removal in removals:
        removal_by_instance[removal.smell_instance_id] = removal

    lifecycles = []

    for intro in introductions:
        removal = removal_by_instance.get(intro.smell_instance_id)

        lifecycle = SmellLifecycle(
            smell_instance_id=intro.smell_instance_id,
            project_name=intro.project_name,
            smell_type=intro.smell_type,
            file_path=intro.file_path,
            function_id=intro.function_id,
            introduction_commit=intro.after_commit_hash,
            introduction_timestamp=intro.after_timestamp,
            introduction_before_commit=intro.before_commit_hash or None,
            removal_commit=removal.after_commit_hash if removal else None,
            removal_timestamp=removal.after_timestamp if removal else None,
            removal_before_commit=removal.before_commit_hash if removal else None,
            is_complete=removal is not None,
            lifespan_commits=0,  # Would need commit traversal to calculate
        )

        lifecycles.append(lifecycle)

    logger.info(f"Built {len(lifecycles)} lifecycles, {sum(1 for l in lifecycles if l.is_complete)} complete")
    return lifecycles


# ============================================================
# STATISTICS & ANALYSIS
# ============================================================

def compute_transition_statistics(
    introductions: List[SmellTransition],
    removals: List[SmellTransition],
    lifecycles: List[SmellLifecycle]
) -> Dict[str, Any]:
    """
    Compute statistics about smell transitions.

    Args:
        introductions: List of introduction transitions
        removals: List of removal transitions
        lifecycles: List of smell lifecycles

    Returns:
        Dictionary with transition statistics
    """
    stats = {
        'total_introductions': len(introductions),
        'total_removals': len(removals),
        'total_lifecycles': len(lifecycles),
        'complete_lifecycles': sum(1 for l in lifecycles if l.is_complete),
        'incomplete_lifecycles': sum(1 for l in lifecycles if not l.is_complete),
    }

    # Statistics by smell type
    intro_by_type: Dict[str, int] = {}
    removal_by_type: Dict[str, int] = {}

    for intro in introductions:
        intro_by_type[intro.smell_type] = intro_by_type.get(intro.smell_type, 0) + 1

    for removal in removals:
        removal_by_type[removal.smell_type] = removal_by_type.get(removal.smell_type, 0) + 1

    stats['introductions_by_type'] = intro_by_type
    stats['removals_by_type'] = removal_by_type

    # Statistics by project
    intro_by_project: Dict[str, int] = {}
    removal_by_project: Dict[str, int] = {}

    for intro in introductions:
        intro_by_project[intro.project_name] = intro_by_project.get(intro.project_name, 0) + 1

    for removal in removals:
        removal_by_project[removal.project_name] = removal_by_project.get(removal.project_name, 0) + 1

    stats['introductions_by_project'] = intro_by_project
    stats['removals_by_project'] = removal_by_project

    # Net smell change per project
    net_change_by_project = {}
    all_projects = set(intro_by_project.keys()) | set(removal_by_project.keys())

    for project in all_projects:
        intros = intro_by_project.get(project, 0)
        rems = removal_by_project.get(project, 0)
        net_change_by_project[project] = intros - rems

    stats['net_change_by_project'] = net_change_by_project

    return stats


# ============================================================
# MAIN WORKFLOW
# ============================================================

def identify_transitions_for_repository(repo_dir: Path) -> Tuple[List[SmellTransition], List[SmellTransition], Dict]:
    """
    Identify transitions for a single repository.

    Args:
        repo_dir: Path to repository output directory

    Returns:
        Tuple of (introductions, removals, statistics)
    """
    events_file = repo_dir / config.EVENTS_FILE

    if not events_file.exists():
        logger.warning(f"Events file not found: {events_file}")
        return [], [], {}

    # Load events
    events = read_jsonl(events_file)

    if not events:
        logger.warning(f"No events found in {events_file}")
        return [], [], {}

    # Identify transitions
    introductions, removals = identify_transitions_from_events(events)

    # Build lifecycles
    lifecycles = build_smell_lifecycles(introductions, removals)

    # Compute statistics
    stats = compute_transition_statistics(introductions, removals, lifecycles)
    stats['repository'] = repo_dir.name

    return introductions, removals, stats


def save_transitions(
    introductions: List[SmellTransition],
    removals: List[SmellTransition],
    output_dir: Path,
    repo_name: str
) -> None:
    """
    Save transitions to output files.

    Args:
        introductions: List of introduction transitions
        removals: List of removal transitions
        output_dir: Output directory
        repo_name: Repository name for file naming
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save introductions
    intro_file = output_dir / f"{repo_name}_introductions.jsonl"
    intro_data = [t.to_dict() for t in introductions]
    write_jsonl(intro_data, intro_file)
    logger.info(f"Saved {len(introductions)} introductions to {intro_file}")

    # Save removals
    removal_file = output_dir / f"{repo_name}_removals.jsonl"
    removal_data = [t.to_dict() for t in removals]
    write_jsonl(removal_data, removal_file)
    logger.info(f"Saved {len(removals)} removals to {removal_file}")


def main(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None
) -> None:
    """
    Main workflow to identify transitions from all repositories.

    Args:
        input_dir: Directory containing repository analysis results
        output_dir: Directory to save transition data
    """
    # Default paths
    if input_dir is None:
        input_dir = config.RAW_SMELL_DETECTION_DIR
    else:
        input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir / 'transitions'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all repository directories
    repo_dirs = [d for d in input_dir.iterdir()
                 if d.is_dir() and d.name not in ('parsed', 'transitions', 'reports')]

    if not repo_dirs:
        logger.warning(f"No repository directories found in {input_dir}")
        return

    logger.info(f"Found {len(repo_dirs)} repositories to process")

    # Process each repository
    all_stats = []

    for repo_dir in repo_dirs:
        repo_name = repo_dir.name
        logger.info(f"Processing: {repo_name}")

        try:
            introductions, removals, stats = identify_transitions_for_repository(repo_dir)

            if introductions or removals:
                # Save transitions
                save_transitions(introductions, removals, output_dir, repo_name)
                all_stats.append(stats)

        except Exception as e:
            logger.error(f"Error processing {repo_name}: {e}")
            continue

    # Save aggregate statistics
    if all_stats:
        # Summary CSV
        summary_df = pd.DataFrame([{
            'repository': s['repository'],
            'introductions': s['total_introductions'],
            'removals': s['total_removals'],
            'complete_lifecycles': s['complete_lifecycles'],
            'incomplete_lifecycles': s['incomplete_lifecycles'],
        } for s in all_stats])

        summary_file = output_dir / 'transition_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary: {summary_file}")

        # Detailed JSON
        stats_file = output_dir / 'transition_statistics.json'
        write_json(all_stats, stats_file)
        logger.info(f"Saved statistics: {stats_file}")

    logger.info(f"Transition identification complete: {len(all_stats)}/{len(repo_dirs)} repositories")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Identify smell transitions")
    parser.add_argument("--input", type=str, help="Input directory with analysis results")
    parser.add_argument("--output", type=str, help="Output directory for transition data")

    args = parser.parse_args()

    main(
        input_dir=args.input,
        output_dir=args.output
    )
