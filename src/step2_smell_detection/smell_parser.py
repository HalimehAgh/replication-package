"""
Smell Parser Module

Parses CodeSmile detection output into structured format for analysis.
Handles conversion of raw smell detection results to standardized records.

Usage
-----
python -m src.step2_smell_detection.smell_parser
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import logging

import pandas as pd

from src import config
from src.utils import read_jsonl, write_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# SMELL TYPE DEFINITIONS
# ============================================================

# ML-specific code smell types from CodeSmile
ML_SMELL_TYPES = {
    'matrix_multiplication_api_misused': {
        'category': 'ml_specific',
        'description': 'Incorrect usage of matrix multiplication APIs',
        'severity': 'high',
    },
    'chain_indexing': {
        'category': 'ml_specific',
        'description': 'Chained indexing in pandas operations',
        'severity': 'medium',
    },
    'columns_and_datatype_not_explicitly_set': {
        'category': 'ml_specific',
        'description': 'DataFrame columns or dtypes not explicitly defined',
        'severity': 'low',
    },
    'empty_column_misinitialization': {
        'category': 'ml_specific',
        'description': 'Empty column created with wrong initialization',
        'severity': 'medium',
    },
    'merge_api_parameter_not_explicitly_set': {
        'category': 'ml_specific',
        'description': 'DataFrame merge parameters not explicitly set',
        'severity': 'medium',
    },
    'inplace_api_misused': {
        'category': 'ml_specific',
        'description': 'Misuse of inplace operations in pandas',
        'severity': 'medium',
    },
    'dataframe_conversion_api_misused': {
        'category': 'ml_specific',
        'description': 'Incorrect DataFrame type conversion',
        'severity': 'medium',
    },
    'gradients_not_cleared_before_backward_propagation': {
        'category': 'ml_specific',
        'description': 'Gradients not zeroed before backward pass',
        'severity': 'high',
    },
    'pytorch_call_method_misused': {
        'category': 'ml_specific',
        'description': 'PyTorch model called incorrectly',
        'severity': 'high',
    },
    'tensor_array_not_used': {
        'category': 'ml_specific',
        'description': 'TensorArray not properly used',
        'severity': 'medium',
    },
    'deterministic_algorithm_option_not_used': {
        'category': 'ml_specific',
        'description': 'Determinism not ensured for reproducibility',
        'severity': 'low',
    },
    'randomness_control_missing': {
        'category': 'ml_specific',
        'description': 'Random seed not set for reproducibility',
        'severity': 'low',
    },
    'missing_mask_argument': {
        'category': 'ml_specific',
        'description': 'Mask argument missing in masked operations',
        'severity': 'high',
    },
    'hyperparameters_not_explicitly_set': {
        'category': 'ml_specific',
        'description': 'Hyperparameters using default values',
        'severity': 'low',
    },
    'nan_equivalence_comparison_misused': {
        'category': 'ml_specific',
        'description': 'NaN comparison using equality operator',
        'severity': 'high',
    },
    'broadcasting_feature_not_used': {
        'category': 'ml_specific',
        'description': 'Broadcasting not utilized for efficiency',
        'severity': 'low',
    },
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class SmellInstance:
    """Represents a single smell detection instance."""
    smell_id: str
    smell_type: str
    file_path: str
    function_name: str
    line_number: int
    description: str
    category: str = 'ml_specific'
    severity: str = 'medium'
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_codesmile_row(cls, row: Dict[str, Any], file_path: str) -> 'SmellInstance':
        """Create instance from CodeSmile output row."""
        smell_type = row.get('smell_name', 'unknown')
        smell_info = ML_SMELL_TYPES.get(smell_type, {})

        # Generate unique ID
        smell_id = f"{file_path}:{row.get('function_name', 'global')}:{smell_type}"
        smell_id = smell_id.replace('/', '_').replace(' ', '_')

        return cls(
            smell_id=smell_id,
            smell_type=smell_type,
            file_path=file_path,
            function_name=row.get('function_name', ''),
            line_number=int(row.get('line', -1)),
            description=row.get('description', smell_info.get('description', '')),
            category=smell_info.get('category', 'ml_specific'),
            severity=smell_info.get('severity', 'medium'),
            additional_info={
                'original_info': row.get('additional_info', ''),
            }
        )


@dataclass
class SmellCommitRecord:
    """Represents smell state at a specific commit."""
    commit_hash: str
    timestamp: str
    file_path: str
    smells: List[SmellInstance]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'commit_hash': self.commit_hash,
            'timestamp': self.timestamp,
            'file_path': self.file_path,
            'smell_count': len(self.smells),
            'smells': [s.to_dict() for s in self.smells],
        }


# ============================================================
# PARSING FUNCTIONS
# ============================================================

def parse_codesmile_dataframe(df: pd.DataFrame, file_path: str) -> List[SmellInstance]:
    """
    Parse CodeSmile DataFrame output to list of SmellInstance objects.

    Args:
        df: CodeSmile result DataFrame
        file_path: Path of the analyzed file

    Returns:
        List of SmellInstance objects
    """
    smells = []

    for _, row in df.iterrows():
        try:
            smell = SmellInstance.from_codesmile_row(row.to_dict(), file_path)
            smells.append(smell)
        except Exception as e:
            logger.warning(f"Error parsing smell row: {e}")
            continue

    return smells


def parse_smell_events_jsonl(events_file: Path) -> List[Dict[str, Any]]:
    """
    Parse smell events from JSONL file.

    Args:
        events_file: Path to smell_events.jsonl

    Returns:
        List of smell event dictionaries
    """
    if not events_file.exists():
        logger.warning(f"Events file not found: {events_file}")
        return []

    events = read_jsonl(events_file)
    logger.info(f"Loaded {len(events)} smell events from {events_file}")

    return events


def parse_smells_per_commit_jsonl(smells_file: Path) -> Dict[str, Dict[str, List]]:
    """
    Parse smells per commit from JSONL file.

    Args:
        smells_file: Path to smells_per_commit.jsonl

    Returns:
        Dictionary mapping commit_hash to file_path to smell list
    """
    if not smells_file.exists():
        logger.warning(f"Smells file not found: {smells_file}")
        return {}

    records = read_jsonl(smells_file)

    commit_smells = {}
    for record in records:
        commit_hash = record.get('commit_hash')
        smells = record.get('smells', {})

        if commit_hash:
            commit_smells[commit_hash] = smells

    logger.info(f"Loaded smell data for {len(commit_smells)} commits")
    return commit_smells


# ============================================================
# AGGREGATION FUNCTIONS
# ============================================================

def aggregate_smells_by_type(events: List[Dict]) -> Dict[str, Dict[str, int]]:
    """
    Aggregate smell events by smell type.

    Args:
        events: List of smell event dictionaries

    Returns:
        Dictionary with smell type statistics
    """
    type_stats = {}

    for event in events:
        smell_type = event.get('smell_type', 'unknown')
        commit_type = event.get('commit_type', '')

        if smell_type not in type_stats:
            type_stats[smell_type] = {
                'total_events': 0,
                'introductions': 0,  # A events
                'removals': 0,       # B events
                'unique_files': set(),
                'unique_functions': set(),
            }

        type_stats[smell_type]['total_events'] += 1

        if commit_type == 'A':
            type_stats[smell_type]['introductions'] += 1
        elif commit_type == 'B':
            type_stats[smell_type]['removals'] += 1

        type_stats[smell_type]['unique_files'].add(event.get('file_path', ''))
        type_stats[smell_type]['unique_functions'].add(event.get('function_id', ''))

    # Convert sets to counts for serialization
    for smell_type in type_stats:
        type_stats[smell_type]['unique_files'] = len(type_stats[smell_type]['unique_files'])
        type_stats[smell_type]['unique_functions'] = len(type_stats[smell_type]['unique_functions'])

    return type_stats


def aggregate_smells_by_file(events: List[Dict]) -> Dict[str, Dict[str, int]]:
    """
    Aggregate smell events by file path.

    Args:
        events: List of smell event dictionaries

    Returns:
        Dictionary with file-level statistics
    """
    file_stats = {}

    for event in events:
        file_path = event.get('file_path', 'unknown')
        commit_type = event.get('commit_type', '')
        smell_type = event.get('smell_type', '')

        if file_path not in file_stats:
            file_stats[file_path] = {
                'total_events': 0,
                'introductions': 0,
                'removals': 0,
                'smell_types': set(),
            }

        file_stats[file_path]['total_events'] += 1

        if commit_type == 'A':
            file_stats[file_path]['introductions'] += 1
        elif commit_type == 'B':
            file_stats[file_path]['removals'] += 1

        file_stats[file_path]['smell_types'].add(smell_type)

    # Convert sets to lists for serialization
    for file_path in file_stats:
        file_stats[file_path]['smell_types'] = list(file_stats[file_path]['smell_types'])

    return file_stats


# ============================================================
# MAIN WORKFLOW
# ============================================================

def parse_repository_smells(repo_dir: Path) -> Dict[str, Any]:
    """
    Parse all smell data for a single repository.

    Args:
        repo_dir: Path to repository output directory

    Returns:
        Dictionary with parsed smell data and statistics
    """
    events_file = repo_dir / config.EVENTS_FILE
    smells_file = repo_dir / config.SMELLS_FILE

    # Parse events
    events = parse_smell_events_jsonl(events_file)

    # Parse per-commit smells
    commit_smells = parse_smells_per_commit_jsonl(smells_file)

    # Aggregate statistics
    type_stats = aggregate_smells_by_type(events)
    file_stats = aggregate_smells_by_file(events)

    return {
        'events': events,
        'commit_smells': commit_smells,
        'type_statistics': type_stats,
        'file_statistics': file_stats,
        'summary': {
            'total_events': len(events),
            'unique_smell_types': len(type_stats),
            'unique_files': len(file_stats),
            'commits_with_smells': len(commit_smells),
        }
    }


def main(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None
) -> None:
    """
    Main workflow to parse smell data from all repositories.

    Args:
        input_dir: Directory containing repository analysis results
        output_dir: Directory to save parsed results
    """
    # Default paths
    if input_dir is None:
        input_dir = config.RAW_SMELL_DETECTION_DIR
    else:
        input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir / 'parsed'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all repository directories
    repo_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name != 'parsed']

    if not repo_dirs:
        logger.warning(f"No repository directories found in {input_dir}")
        return

    logger.info(f"Found {len(repo_dirs)} repositories to parse")

    # Parse each repository
    all_summaries = []

    for repo_dir in repo_dirs:
        repo_name = repo_dir.name
        logger.info(f"Parsing: {repo_name}")

        try:
            parsed_data = parse_repository_smells(repo_dir)

            # Save parsed data
            parsed_file = output_dir / f"{repo_name}_parsed.json"
            with open(parsed_file, 'w') as f:
                # Don't save full events to avoid large files
                save_data = {
                    'type_statistics': parsed_data['type_statistics'],
                    'file_statistics': parsed_data['file_statistics'],
                    'summary': parsed_data['summary'],
                }
                json.dump(save_data, f, indent=2)

            parsed_data['summary']['repository'] = repo_name
            all_summaries.append(parsed_data['summary'])

        except Exception as e:
            logger.error(f"Error parsing {repo_name}: {e}")
            continue

    # Save aggregate summary
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_file = output_dir / 'parsing_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary: {summary_file}")

    logger.info(f"Parsing complete: {len(all_summaries)}/{len(repo_dirs)} repositories")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse smell detection results")
    parser.add_argument("--input", type=str, help="Input directory with analysis results")
    parser.add_argument("--output", type=str, help="Output directory for parsed data")

    args = parser.parse_args()

    main(
        input_dir=args.input,
        output_dir=args.output
    )
