"""
Commit Analyzer Module

Analyzes commit content and metadata to identify:
- Bug fix commits
- Refactoring commits
- Feature additions
- Commit patterns

Used to support control variable extraction for hypothesis testing.

Usage
-----
python -m src.step1_commit_extraction.commit_analyzer
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# COMMIT CLASSIFICATION PATTERNS
# ============================================================

# Bug fix patterns in commit messages
BUG_FIX_PATTERNS = [
    r'\bfix(?:es|ed|ing)?\b',
    r'\bbug(?:s|fix)?\b',
    r'\bpatch(?:es|ed|ing)?\b',
    r'\bresolve[ds]?\b',
    r'\brepair(?:s|ed|ing)?\b',
    r'\bcorrect(?:s|ed|ing|ion)?\b',
    r'\bissue\s*#?\d+',
    r'\bhotfix\b',
    r'\bdefect\b',
    r'\berror(?:s)?\b',
    r'\bcrash(?:es|ed|ing)?\b',
]

# Refactoring patterns
REFACTORING_PATTERNS = [
    r'\brefactor(?:s|ed|ing)?\b',
    r'\bclean\s*up\b',
    r'\brestructure[ds]?\b',
    r'\breorganize[ds]?\b',
    r'\bsimplif(?:y|ies|ied|ying)\b',
    r'\boptimize[ds]?\b',
    r'\bimprove[ds]?\s+code\b',
    r'\bcode\s+quality\b',
    r'\bremove\s+(?:unused|dead)\b',
    r'\brename[ds]?\b',
    r'\bmove[ds]?\b.*(?:to|from)',
    r'\bextract(?:s|ed|ing)?\s+(?:method|function|class)\b',
]

# Feature addition patterns
FEATURE_PATTERNS = [
    r'\badd(?:s|ed|ing)?\b',
    r'\bnew\s+feature\b',
    r'\bimplement(?:s|ed|ing)?\b',
    r'\bintroduce[ds]?\b',
    r'\bfeature\b',
    r'\benhance(?:ment)?s?\b',
    r'\bextend(?:s|ed|ing)?\b',
    r'\bsupport(?:s|ed|ing)?\s+(?:for)?\b',
    r'\benable[ds]?\b',
]

# Test-related patterns
TEST_PATTERNS = [
    r'\btest(?:s|ing|ed)?\b',
    r'\bunit\s*test\b',
    r'\bintegration\s*test\b',
    r'\bcoverage\b',
    r'\bspec(?:s)?\b',
    r'\bassert(?:ions?)?\b',
    r'\bmock(?:s|ed|ing)?\b',
]

# Documentation patterns
DOC_PATTERNS = [
    r'\bdoc(?:s|umentation)?\b',
    r'\breadme\b',
    r'\bcomment(?:s|ed|ing)?\b',
    r'\bdocstring(?:s)?\b',
    r'\btypehint(?:s)?\b',
    r'\btype\s*annotation(?:s)?\b',
]


def compile_patterns(patterns: List[str]) -> re.Pattern:
    """Compile list of patterns into a single regex."""
    combined = '|'.join(f'({p})' for p in patterns)
    return re.compile(combined, re.IGNORECASE)


# Compiled patterns
BUG_FIX_REGEX = compile_patterns(BUG_FIX_PATTERNS)
REFACTORING_REGEX = compile_patterns(REFACTORING_PATTERNS)
FEATURE_REGEX = compile_patterns(FEATURE_PATTERNS)
TEST_REGEX = compile_patterns(TEST_PATTERNS)
DOC_REGEX = compile_patterns(DOC_PATTERNS)


# ============================================================
# COMMIT CLASSIFICATION
# ============================================================

def classify_commit_type(message: str) -> Dict[str, bool]:
    """
    Classify a commit based on its message.

    Args:
        message: Commit message text

    Returns:
        Dictionary with boolean flags for each type
    """
    if not message:
        return {
            'is_bug_fix': False,
            'is_refactoring': False,
            'is_feature': False,
            'is_test': False,
            'is_documentation': False,
        }

    return {
        'is_bug_fix': bool(BUG_FIX_REGEX.search(message)),
        'is_refactoring': bool(REFACTORING_REGEX.search(message)),
        'is_feature': bool(FEATURE_REGEX.search(message)),
        'is_test': bool(TEST_REGEX.search(message)),
        'is_documentation': bool(DOC_REGEX.search(message)),
    }


def is_bug_fix_commit(message: str) -> bool:
    """
    Check if commit message indicates a bug fix.

    Simple wrapper around classify_commit_type for backward compatibility.

    Args:
        message: Commit message text

    Returns:
        True if the message matches bug-fix patterns
    """
    return classify_commit_type(message)['is_bug_fix']


def analyze_commit_message(message: str) -> Dict[str, any]:
    """
    Perform detailed analysis of a commit message.

    Args:
        message: Commit message text

    Returns:
        Dictionary with analysis results
    """
    if not message:
        return {
            'message_length': 0,
            'line_count': 0,
            'has_body': False,
            'has_issue_reference': False,
            'issue_numbers': [],
        }

    lines = message.strip().split('\n')
    first_line = lines[0] if lines else ''

    # Extract issue references
    issue_pattern = r'#(\d+)'
    issue_matches = re.findall(issue_pattern, message)

    return {
        'message_length': len(message),
        'first_line_length': len(first_line),
        'line_count': len(lines),
        'has_body': len(lines) > 1 and any(line.strip() for line in lines[1:]),
        'has_issue_reference': bool(issue_matches),
        'issue_numbers': [int(n) for n in issue_matches],
    }


def analyze_file_changes(changed_files: List[Dict]) -> Dict[str, any]:
    """
    Analyze the files changed in a commit.

    Args:
        changed_files: List of file change dictionaries

    Returns:
        Dictionary with file analysis results
    """
    if not changed_files:
        return {
            'total_files': 0,
            'python_files': 0,
            'test_files': 0,
            'config_files': 0,
            'doc_files': 0,
            'has_python_changes': False,
            'has_test_changes': False,
        }

    python_files = []
    test_files = []
    config_files = []
    doc_files = []

    config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
    doc_extensions = {'.md', '.rst', '.txt'}

    for file_info in changed_files:
        path = file_info.get('b_path') or file_info.get('a_path') or ''
        path_lower = path.lower()
        ext = Path(path).suffix.lower()

        # Python files
        if ext in {'.py', '.pyx', '.pxd'}:
            python_files.append(path)

            # Test files
            if 'test' in path_lower or path_lower.startswith('test_'):
                test_files.append(path)

        # Config files
        elif ext in config_extensions:
            config_files.append(path)

        # Documentation files
        elif ext in doc_extensions:
            doc_files.append(path)

    return {
        'total_files': len(changed_files),
        'python_files': len(python_files),
        'test_files': len(test_files),
        'config_files': len(config_files),
        'doc_files': len(doc_files),
        'has_python_changes': len(python_files) > 0,
        'has_test_changes': len(test_files) > 0,
        'python_file_paths': python_files,
        'test_file_paths': test_files,
    }


# ============================================================
# COMMIT ANALYSIS WORKFLOW
# ============================================================

def analyze_commit(commit_data: Dict) -> Dict:
    """
    Perform complete analysis of a single commit.

    Args:
        commit_data: Dictionary with commit metadata

    Returns:
        Dictionary with original data plus analysis results
    """
    result = commit_data.copy()

    # Get message
    message = commit_data.get('message', '')

    # Classify commit type
    type_classification = classify_commit_type(message)
    result.update(type_classification)

    # Analyze message
    message_analysis = analyze_commit_message(message)
    result.update(message_analysis)

    # Analyze file changes
    changed_files = commit_data.get('changed_files', [])
    if isinstance(changed_files, str):
        try:
            changed_files = json.loads(changed_files)
        except json.JSONDecodeError:
            changed_files = []

    file_analysis = analyze_file_changes(changed_files)
    result.update(file_analysis)

    return result


def analyze_commits_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze all commits in a DataFrame.

    Args:
        df: DataFrame with commit data

    Returns:
        DataFrame with additional analysis columns
    """
    analyzed_rows = []

    for _, row in df.iterrows():
        commit_dict = row.to_dict()
        analyzed = analyze_commit(commit_dict)
        analyzed_rows.append(analyzed)

    return pd.DataFrame(analyzed_rows)


# ============================================================
# AGGREGATE STATISTICS
# ============================================================

def compute_commit_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute aggregate statistics for a set of commits.

    Args:
        df: DataFrame with analyzed commits

    Returns:
        Dictionary with aggregate statistics
    """
    total_commits = len(df)

    if total_commits == 0:
        return {'total_commits': 0}

    stats = {
        'total_commits': total_commits,
        'bug_fix_commits': int(df['is_bug_fix'].sum()) if 'is_bug_fix' in df else 0,
        'refactoring_commits': int(df['is_refactoring'].sum()) if 'is_refactoring' in df else 0,
        'feature_commits': int(df['is_feature'].sum()) if 'is_feature' in df else 0,
        'test_commits': int(df['is_test'].sum()) if 'is_test' in df else 0,
        'doc_commits': int(df['is_documentation'].sum()) if 'is_documentation' in df else 0,
        'merge_commits': int(df['is_merge'].sum()) if 'is_merge' in df else 0,
    }

    # Compute ratios
    stats['bug_fix_ratio'] = stats['bug_fix_commits'] / total_commits
    stats['refactoring_ratio'] = stats['refactoring_commits'] / total_commits
    stats['feature_ratio'] = stats['feature_commits'] / total_commits

    # Compute averages
    if 'python_files' in df:
        stats['avg_python_files_per_commit'] = float(df['python_files'].mean())

    if 'total_files' in df:
        stats['avg_files_per_commit'] = float(df['total_files'].mean())

    return stats


# ============================================================
# MAIN WORKFLOW
# ============================================================

def analyze_project_commits(
    input_csv: Path,
    output_csv: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze commits for a single project.

    Args:
        input_csv: Path to raw commits CSV
        output_csv: Path to save analyzed commits (optional)

    Returns:
        Tuple of (analyzed DataFrame, statistics dict)
    """
    logger.info(f"Analyzing commits from: {input_csv}")

    # Load commits
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} commits")

    # Analyze each commit
    analyzed_df = analyze_commits_dataframe(df)

    # Compute statistics
    stats = compute_commit_statistics(analyzed_df)

    # Save if output path provided
    if output_csv:
        # Don't save list columns to CSV
        save_df = analyzed_df.drop(columns=['python_file_paths', 'test_file_paths', 'issue_numbers'],
                                    errors='ignore')
        save_df.to_csv(output_csv, index=False)
        logger.info(f"Saved analyzed commits: {output_csv}")

    return analyzed_df, stats


def main(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None
) -> None:
    """
    Main workflow to analyze commits from all projects.

    Args:
        input_dir: Directory containing raw commit CSVs
        output_dir: Directory to save analyzed commits
    """
    # Default paths
    if input_dir is None:
        input_dir = config.RAW_COMMIT_HISTORY_DIR
    else:
        input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = config.RAW_COMMIT_HISTORY_DIR / 'analyzed'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all commit CSV files
    commit_files = list(input_dir.glob('*_commits.csv'))
    if not commit_files:
        logger.warning(f"No commit files found in {input_dir}")
        return

    logger.info(f"Found {len(commit_files)} commit files to analyze")

    # Analyze each project
    all_stats = []

    for commit_file in commit_files:
        project_name = commit_file.stem.replace('_commits', '')
        output_file = output_dir / f'{project_name}_analyzed.csv'

        try:
            _, stats = analyze_project_commits(commit_file, output_file)
            stats['project'] = project_name
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Error analyzing {project_name}: {e}")
            continue

    # Save aggregate statistics
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_file = output_dir / 'commit_statistics.csv'
        stats_df.to_csv(stats_file, index=False)
        logger.info(f"Saved aggregate statistics: {stats_file}")

    logger.info(f"\nAnalysis complete: {len(all_stats)}/{len(commit_files)} projects")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze commit history")
    parser.add_argument("--input", type=str, help="Input directory with commit CSVs")
    parser.add_argument("--output", type=str, help="Output directory for analyzed commits")

    args = parser.parse_args()

    main(
        input_dir=args.input,
        output_dir=args.output
    )
