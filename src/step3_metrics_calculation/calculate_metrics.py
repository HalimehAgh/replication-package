"""
Maintainability Metrics Analysis for Code Smells

Analyzes how maintainability metrics change when code smells are introduced
or removed. Extracts metrics at commit pairs (A-1 vs A, B-1 vs B) to measure
the impact of smells on code quality.

Usage:
    python -m src.step3_metrics_calculation.calculate_metrics --repo <repo_name>
    python -m src.step3_metrics_calculation.calculate_metrics --all
"""

import argparse
import ast
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from radon.complexity import cc_visit
from radon.raw import analyze

from src import config
from src.utils import (
    setup_logger,
    read_json,
    write_json,
    read_jsonl,
    write_jsonl,
    repo_to_dir_name,
    get_repo_path,
)
from src.step3_metrics_calculation import constants
from src.step3_metrics_calculation.external_tools import calculate_code_clone_ratio
from src.step3_metrics_calculation.context_metrics import calculate_time_window_metrics
from src.step1_commit_extraction.commit_analyzer import is_bug_fix_commit, classify_commit_type
from src.step3_metrics_calculation.smell_cooccurrence import get_smell_count_at_commit

# Import dimension-specific metrics from refactored modules
from src.step3_metrics_calculation.metrics import (
    # Structural complexity
    count_loops_and_nesting,
    get_cyclomatic_complexity,
    # Code size
    get_raw_metrics,
    calculate_avg_function_length,
    get_function_metrics,
    count_classes,
    # Defect proneness
    calculate_exception_coverage,
    calculate_type_annotation_coverage,
    # Understandability
    calculate_docstring_coverage,
    calculate_comment_ratio,
    calculate_cognitive_complexity,
    # Testing quality
    analyze_test_functions,
    # ML complexity
    get_ml_metrics_from_tree,
)

logger = setup_logger(__name__)


class FileMetrics:
    """Container for file-level code metrics."""

    def __init__(
        self,
        file_path: str,
        commit_hash: str,
        cyclomatic_complexity: float,
        loc: int,
        sloc: int,
        comments: int,
        blank: int,
        num_functions: int,
        num_classes: int,
        avg_complexity_per_function: float,
        max_complexity: int,
        function_metrics: List[Dict[str, Any]],
        # Structural complexity metrics
        loop_count: int = 0,
        nested_loop_depth: int = 0,
        # Code size metrics
        avg_function_length: float = 0.0,
        # Defect proneness metrics
        exception_handling_coverage: float = 0.0,
        code_clone_ratio: float = 0.0,
        type_annotation_coverage: float = 0.0,
        # Understandability metrics
        docstring_coverage: float = 0.0,
        comment_ratio: float = 0.0,
        cognitive_complexity: float = 0.0,
        # Testing quality metrics
        test_function_ratio: float = 0.0,
        assertion_density: float = 0.0,
        # ML-specific metrics
        data_operation_density: float = 0.0,
        config_explicitness_ratio: float = 0.0,
        # Per-commit churn metrics (from smell event data)
        lines_added: int = 0,
        lines_deleted: int = 0,
        total_churn: int = 0,
        is_bug_fix: bool = False,
        is_feature_commit: bool = False,
        # Time-window metrics (only at A and B commits, null otherwise)
        window_total_churn: Optional[int] = None,
        window_modified_files: Optional[int] = None,
        window_commit_count: Optional[int] = None,
        window_bug_fix_ratio: Optional[float] = None,
        # Smell co-occurrence control variable (Khomh et al. 2012)
        smell_count_at_commit: int = 0,
    ):
        self.file_path = file_path
        self.commit_hash = commit_hash
        self.cyclomatic_complexity = cyclomatic_complexity
        self.loc = loc
        self.sloc = sloc
        self.comments = comments
        self.blank = blank
        self.num_functions = num_functions
        self.num_classes = num_classes
        self.avg_complexity_per_function = avg_complexity_per_function
        self.max_complexity = max_complexity
        self.function_metrics = function_metrics
        self.loop_count = loop_count
        self.nested_loop_depth = nested_loop_depth
        self.avg_function_length = avg_function_length
        self.exception_handling_coverage = exception_handling_coverage
        self.code_clone_ratio = code_clone_ratio
        self.type_annotation_coverage = type_annotation_coverage
        self.docstring_coverage = docstring_coverage
        self.comment_ratio = comment_ratio
        self.cognitive_complexity = cognitive_complexity
        self.test_function_ratio = test_function_ratio
        self.assertion_density = assertion_density
        self.data_operation_density = data_operation_density
        self.config_explicitness_ratio = config_explicitness_ratio
        self.lines_added = lines_added
        self.lines_deleted = lines_deleted
        self.total_churn = total_churn
        self.is_bug_fix = is_bug_fix
        self.is_feature_commit = is_feature_commit
        self.window_total_churn = window_total_churn
        self.window_modified_files = window_modified_files
        self.window_commit_count = window_commit_count
        self.window_bug_fix_ratio = window_bug_fix_ratio
        self.smell_count_at_commit = smell_count_at_commit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "commit_hash": self.commit_hash,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "loc": self.loc,
            "sloc": self.sloc,
            "comments": self.comments,
            "blank": self.blank,
            "num_functions": self.num_functions,
            "num_classes": self.num_classes,
            "avg_complexity_per_function": self.avg_complexity_per_function,
            "max_complexity": self.max_complexity,
            "function_metrics": self.function_metrics,
            "loop_count": self.loop_count,
            "nested_loop_depth": self.nested_loop_depth,
            "avg_function_length": self.avg_function_length,
            "exception_handling_coverage": self.exception_handling_coverage,
            "code_clone_ratio": self.code_clone_ratio,
            "type_annotation_coverage": self.type_annotation_coverage,
            "docstring_coverage": self.docstring_coverage,
            "comment_ratio": self.comment_ratio,
            "cognitive_complexity": self.cognitive_complexity,
            "test_function_ratio": self.test_function_ratio,
            "assertion_density": self.assertion_density,
            "data_operation_density": self.data_operation_density,
            "config_explicitness_ratio": self.config_explicitness_ratio,
            "lines_added": self.lines_added,
            "lines_deleted": self.lines_deleted,
            "total_churn": self.total_churn,
            "is_bug_fix": self.is_bug_fix,
            "is_feature_commit": self.is_feature_commit,
            "window_total_churn": self.window_total_churn,
            "window_modified_files": self.window_modified_files,
            "window_commit_count": self.window_commit_count,
            "window_bug_fix_ratio": self.window_bug_fix_ratio,
            "smell_count_at_commit": self.smell_count_at_commit,
        }


class MetricChange:
    """Represents a change in metrics between two commits."""

    def __init__(
        self,
        smell_instance_id: str,
        project_name: str,
        smell_type: str,
        change_type: str,
        before_commit: str,
        after_commit: str,
        file_path: str,
        function_id: str,
        before_metrics: FileMetrics,
        after_metrics: FileMetrics,
        delta_metrics: Dict[str, float]
    ):
        self.smell_instance_id = smell_instance_id
        self.project_name = project_name
        self.smell_type = smell_type
        self.change_type = change_type
        self.before_commit = before_commit
        self.after_commit = after_commit
        self.file_path = file_path
        self.function_id = function_id
        self.before_metrics = before_metrics
        self.after_metrics = after_metrics
        self.delta_metrics = delta_metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "smell_instance_id": self.smell_instance_id,
            "project_name": self.project_name,
            "smell_type": self.smell_type,
            "change_type": self.change_type,
            "before_commit": self.before_commit,
            "after_commit": self.after_commit,
            "file_path": self.file_path,
            "function_id": self.function_id,
            "before_metrics": self.before_metrics.to_dict(),
            "after_metrics": self.after_metrics.to_dict(),
            "delta_metrics": self.delta_metrics
        }


class MetricsAnalyzer:
    """Analyzes maintainability metrics at smell introduction/removal points."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.PROCESSED_METRICS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_repo_output_dir(self, repo_name: str) -> Path:
        """Get output directory for a specific repository."""
        repo_dir = self.output_dir / repo_to_dir_name(repo_name)
        repo_dir.mkdir(parents=True, exist_ok=True)
        return repo_dir

    def extract_file_metrics(self, file_path: str, commit_hash: str) -> Optional[FileMetrics]:
        """Extract comprehensive metrics from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            if not code.strip():
                logger.debug(f"Skipping empty file: {file_path}")
                return None

            try:
                raw = analyze(code)
            except SyntaxError as e:
                logger.debug(f"Syntax error in {file_path}: {e}")
                return None

            try:
                complexity_results = cc_visit(code)
                total_complexity = sum(item.complexity for item in complexity_results)

                from radon.visitors import Function, Class
                functions_only = [item for item in complexity_results if isinstance(item, Function)]
                classes_only = [item for item in complexity_results if isinstance(item, Class)]

                num_functions = len(functions_only)
                num_classes = len(classes_only)
                max_complexity = max((item.complexity for item in functions_only), default=0)
                avg_complexity = total_complexity / num_functions if num_functions > 0 else 0.0

                function_metrics = []
                for item in functions_only:
                    function_metrics.append({
                        "name": item.name,
                        "classname": item.classname or None,
                        "complexity": item.complexity,
                        "line_start": item.lineno,
                        "line_end": item.endline
                    })
            except SyntaxError as e:
                logger.debug(f"Syntax error calculating complexity in {file_path}: {e}")
                return None

            try:
                tree = ast.parse(code)
                loop_count, nested_loop_depth = count_loops_and_nesting(tree)
                avg_function_length = calculate_avg_function_length(raw.loc, num_functions)
                exception_handling_coverage = calculate_exception_coverage(tree)
                type_annotation_coverage = calculate_type_annotation_coverage(tree)
                docstring_coverage = calculate_docstring_coverage(tree)
                comment_ratio = calculate_comment_ratio(raw.comments, raw.loc)
                cognitive_complexity = calculate_cognitive_complexity(code)
                test_function_ratio, assertion_density = analyze_test_functions(tree, raw.sloc)
                data_operation_density, config_explicitness_ratio = get_ml_metrics_from_tree(tree)
                code_clone_ratio = calculate_code_clone_ratio(code)
            except SyntaxError:
                loop_count = nested_loop_depth = 0
                avg_function_length = docstring_coverage = comment_ratio = 0.0
                cognitive_complexity = exception_handling_coverage = type_annotation_coverage = 0.0
                test_function_ratio = assertion_density = 0.0
                data_operation_density = config_explicitness_ratio = code_clone_ratio = 0.0

            return FileMetrics(
                file_path=file_path,
                commit_hash=commit_hash,
                cyclomatic_complexity=total_complexity,
                loc=raw.loc,
                sloc=raw.sloc,
                comments=raw.comments,
                blank=raw.blank,
                num_functions=num_functions,
                num_classes=num_classes,
                avg_complexity_per_function=avg_complexity,
                max_complexity=max_complexity,
                function_metrics=function_metrics,
                loop_count=loop_count,
                nested_loop_depth=nested_loop_depth,
                avg_function_length=avg_function_length,
                exception_handling_coverage=exception_handling_coverage,
                code_clone_ratio=code_clone_ratio,
                type_annotation_coverage=type_annotation_coverage,
                docstring_coverage=docstring_coverage,
                comment_ratio=comment_ratio,
                cognitive_complexity=cognitive_complexity,
                test_function_ratio=test_function_ratio,
                assertion_density=assertion_density,
                data_operation_density=data_operation_density,
                config_explicitness_ratio=config_explicitness_ratio,
            )

        except Exception as e:
            logger.warning(f"Unexpected error extracting metrics from {file_path}: {e}")
            return None

    def calculate_metric_deltas(
        self,
        before_metrics: FileMetrics,
        after_metrics: FileMetrics
    ) -> Dict[str, Any]:
        """Calculate the difference between before and after metrics."""
        return {
            "delta_cyclomatic_complexity": after_metrics.cyclomatic_complexity - before_metrics.cyclomatic_complexity,
            "delta_loc": after_metrics.loc - before_metrics.loc,
            "delta_sloc": after_metrics.sloc - before_metrics.sloc,
            "delta_comments": after_metrics.comments - before_metrics.comments,
            "delta_blank": after_metrics.blank - before_metrics.blank,
            "delta_num_functions": after_metrics.num_functions - before_metrics.num_functions,
            "delta_num_classes": after_metrics.num_classes - before_metrics.num_classes,
            "delta_avg_complexity_per_function": after_metrics.avg_complexity_per_function - before_metrics.avg_complexity_per_function,
            "delta_max_complexity": after_metrics.max_complexity - before_metrics.max_complexity,
            "delta_loop_count": after_metrics.loop_count - before_metrics.loop_count,
            "delta_nested_loop_depth": after_metrics.nested_loop_depth - before_metrics.nested_loop_depth,
            "delta_avg_function_length": after_metrics.avg_function_length - before_metrics.avg_function_length,
            "delta_exception_handling_coverage": after_metrics.exception_handling_coverage - before_metrics.exception_handling_coverage,
            "delta_code_clone_ratio": after_metrics.code_clone_ratio - before_metrics.code_clone_ratio,
            "delta_type_annotation_coverage": after_metrics.type_annotation_coverage - before_metrics.type_annotation_coverage,
            "delta_docstring_coverage": after_metrics.docstring_coverage - before_metrics.docstring_coverage,
            "delta_comment_ratio": after_metrics.comment_ratio - before_metrics.comment_ratio,
            "delta_cognitive_complexity": after_metrics.cognitive_complexity - before_metrics.cognitive_complexity,
            "delta_test_function_ratio": after_metrics.test_function_ratio - before_metrics.test_function_ratio,
            "delta_assertion_density": after_metrics.assertion_density - before_metrics.assertion_density,
            "delta_data_operation_density": after_metrics.data_operation_density - before_metrics.data_operation_density,
            "delta_config_explicitness_ratio": after_metrics.config_explicitness_ratio - before_metrics.config_explicitness_ratio,
            "lines_added": after_metrics.lines_added,
            "lines_deleted": after_metrics.lines_deleted,
            "total_churn": after_metrics.total_churn,
            "is_bug_fix": after_metrics.is_bug_fix,
            "is_feature_commit": after_metrics.is_feature_commit,
            "before_window_total_churn": before_metrics.window_total_churn,
            "before_window_modified_files": before_metrics.window_modified_files,
            "before_window_commit_count": before_metrics.window_commit_count,
            "before_window_bug_fix_ratio": before_metrics.window_bug_fix_ratio,
            "after_window_total_churn": after_metrics.window_total_churn,
            "after_window_modified_files": after_metrics.window_modified_files,
            "after_window_commit_count": after_metrics.window_commit_count,
            "after_window_bug_fix_ratio": after_metrics.window_bug_fix_ratio,
            "before_smell_count_at_commit": before_metrics.smell_count_at_commit,
            "after_smell_count_at_commit": after_metrics.smell_count_at_commit,
        }

    def get_file_content_at_commit(
        self,
        repo_path: str,
        commit_hash: str,
        file_path: str
    ) -> Optional[str]:
        """Get file content at a specific commit using GitPython."""
        try:
            import git
            repo = git.Repo(repo_path)
            commit = repo.commit(commit_hash)
            try:
                blob = commit.tree / file_path
                return blob.data_stream.read().decode('utf-8')
            except KeyError:
                logger.debug(f"File {file_path} not found in commit {commit_hash}")
                return None
        except Exception as e:
            logger.debug(f"Error getting file content at {commit_hash}:{file_path}: {e}")
            return None

    def analyze_smell_event_pair(
        self,
        repo_path: str,
        repo_name: str,
        smell_instance_id: str,
        smell_type: str,
        change_type: str,
        before_commit: str,
        after_commit: str,
        file_path: str,
        function_id: str,
        lines_added: int = 0,
        lines_deleted: int = 0,
        commit_message: str = ""
    ) -> Optional[MetricChange]:
        """Analyze metrics for a smell event pair (A-1 vs A or B-1 vs B)."""
        with tempfile.TemporaryDirectory(prefix="metrics_") as temp_dir:
            before_content = self.get_file_content_at_commit(repo_path, before_commit, file_path)
            if before_content is None:
                logger.warning(f"Could not get file content at {before_commit}:{file_path}")
                return None

            after_content = self.get_file_content_at_commit(repo_path, after_commit, file_path)
            if after_content is None:
                logger.warning(f"Could not get file content at {after_commit}:{file_path}")
                return None

            before_file = Path(temp_dir) / "before.py"
            after_file = Path(temp_dir) / "after.py"

            with open(before_file, 'w', encoding='utf-8') as f:
                f.write(before_content)
            with open(after_file, 'w', encoding='utf-8') as f:
                f.write(after_content)

            before_metrics = self.extract_file_metrics(str(before_file), before_commit)
            after_metrics = self.extract_file_metrics(str(after_file), after_commit)

            if before_metrics is None or after_metrics is None:
                logger.warning(f"Could not extract metrics for {smell_instance_id}")
                return None

            # Store separate line counts and total churn
            after_metrics.lines_added = lines_added
            after_metrics.lines_deleted = lines_deleted
            after_metrics.total_churn = lines_added + lines_deleted

            # Classify commit type using commit message
            commit_classification = classify_commit_type(commit_message)
            after_metrics.is_bug_fix = commit_classification.get('is_bug_fix', False)
            after_metrics.is_feature_commit = commit_classification.get('is_feature', False)

            # Before metrics have no churn (baseline)
            before_metrics.lines_added = 0
            before_metrics.lines_deleted = 0
            before_metrics.total_churn = 0
            before_metrics.is_bug_fix = False
            before_metrics.is_feature_commit = False

            try:
                before_time_window = calculate_time_window_metrics(
                    repo_path=Path(repo_path),
                    file_path=file_path,
                    commit_hash=before_commit
                )
                before_metrics.window_total_churn = before_time_window.get('window_total_churn')
                before_metrics.window_modified_files = before_time_window.get('window_modified_files')
                before_metrics.window_commit_count = before_time_window.get('window_commit_count')
                before_metrics.window_bug_fix_ratio = before_time_window.get('window_bug_fix_ratio')
            except Exception as e:
                logger.debug(f"Error calculating time-window metrics at before_commit: {e}")

            try:
                after_time_window = calculate_time_window_metrics(
                    repo_path=Path(repo_path),
                    file_path=file_path,
                    commit_hash=after_commit
                )
                after_metrics.window_total_churn = after_time_window.get('window_total_churn')
                after_metrics.window_modified_files = after_time_window.get('window_modified_files')
                after_metrics.window_commit_count = after_time_window.get('window_commit_count')
                after_metrics.window_bug_fix_ratio = after_time_window.get('window_bug_fix_ratio')
            except Exception as e:
                logger.debug(f"Error calculating time-window metrics at after_commit: {e}")

            before_metrics.smell_count_at_commit = get_smell_count_at_commit(
                repo_name=repo_name, commit_hash=before_commit, file_path=file_path
            )
            after_metrics.smell_count_at_commit = get_smell_count_at_commit(
                repo_name=repo_name, commit_hash=after_commit, file_path=file_path
            )

            deltas = self.calculate_metric_deltas(before_metrics, after_metrics)

            return MetricChange(
                smell_instance_id=smell_instance_id,
                project_name=repo_name,
                smell_type=smell_type,
                change_type=change_type,
                before_commit=before_commit,
                after_commit=after_commit,
                file_path=file_path,
                function_id=function_id,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                delta_metrics=deltas
            )

    def load_smell_events(self, repo_name: str, analysis_dir: Path) -> List[Dict[str, Any]]:
        """Load smell events from JSONL file."""
        events_file = analysis_dir / repo_to_dir_name(repo_name) / config.EVENTS_FILE
        if not events_file.exists():
            raise FileNotFoundError(f"Smell events file not found: {events_file}")
        return read_jsonl(events_file)

    def load_validated_instances(self, repo_name: str) -> List[Dict[str, Any]]:
        """Load validated smell instances from unified validation."""
        validated_file = (
            config.PROCESSED_METRICS_DIR / repo_to_dir_name(repo_name) / "validated_smell_instances.jsonl"
        )
        if not validated_file.exists():
            raise FileNotFoundError(
                f"Validated instances file not found: {validated_file}\n"
                f"Run: python -m src.step3_metrics_calculation.validate_instances --repo {repo_name}"
            )
        return read_jsonl(validated_file)

    def group_smell_events_into_pairs(
        self, events: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple[Dict, Dict]], List[Tuple[Dict, Dict]]]:
        """Group smell events into (A-1, A) and (B-1, B) pairs."""
        events_by_id = defaultdict(list)
        for event in events:
            events_by_id[event["smell_instance_id"]].append(event)

        introduction_pairs = []
        removal_pairs = []

        for smell_id, smell_events in events_by_id.items():
            a_minus_1 = a = b_minus_1 = b = None
            for event in smell_events:
                if event["commit_type"] == "A-1":
                    a_minus_1 = event
                elif event["commit_type"] == "A":
                    a = event
                elif event["commit_type"] == "B-1":
                    b_minus_1 = event
                elif event["commit_type"] == "B":
                    b = event

            if a_minus_1 and a:
                introduction_pairs.append((a_minus_1, a))
            if b_minus_1 and b:
                removal_pairs.append((b_minus_1, b))

        return introduction_pairs, removal_pairs

    def analyze_validated_instances(
        self, repo_name: str, resume: bool = False
    ) -> Dict[str, Any]:
        """Analyze maintainability metrics for VALIDATED smell instances only."""
        logger.info("=" * 80)
        logger.info(f"Analyzing VALIDATED metrics for: {repo_name}")
        logger.info("=" * 80)

        repo_output_dir = self.get_repo_output_dir(repo_name)
        metric_changes_file = repo_output_dir / "validated_metrics.jsonl"
        checkpoint_file = repo_output_dir / "metrics_checkpoint.json"

        validated_instances = self.load_validated_instances(repo_name)
        logger.info(f"Loaded {len(validated_instances)} validated smell instances")

        introductions = [inst for inst in validated_instances if inst["event_type"] == "introduction"]
        removals = [inst for inst in validated_instances if inst["event_type"] == "removal"]
        logger.info(f"  - {len(introductions)} introductions (A-1 → A)")
        logger.info(f"  - {len(removals)} removals (B-1 → B)")

        repo_path = get_repo_path(repo_name)
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        processed_ids = set()
        if resume and checkpoint_file.exists():
            checkpoint = read_json(checkpoint_file)
            processed_ids = set(checkpoint.get("processed_ids", []))
            logger.info(f"Resuming: {len(processed_ids)} instances already processed")
        elif not resume:
            if metric_changes_file.exists():
                metric_changes_file.unlink()

        stats = {
            "repo_name": repo_name,
            "total_validated_instances": len(validated_instances),
            "introduction_instances": len(introductions),
            "removal_instances": len(removals),
            "successful_analyses": 0,
            "failed_analyses": 0,
            "failed_instances": [],
        }

        logger.info("Processing validated instances...")
        for idx, instance in enumerate(validated_instances, 1):
            instance_key = (
                f"{instance['smell_instance_id']}_"
                f"{instance['before_commit'][:8]}_"
                f"{instance['after_commit'][:8]}_"
                f"{instance['event_type']}"
            )

            if instance_key in processed_ids:
                continue

            logger.info(f"[{idx}/{len(validated_instances)}] Analyzing {instance['smell_type']} ({instance['event_type']})")

            result = self.analyze_smell_event_pair(
                repo_path=str(repo_path),
                repo_name=instance.get("project_name", repo_name),
                smell_instance_id=instance["smell_instance_id"],
                smell_type=instance["smell_type"],
                change_type=instance["event_type"],
                before_commit=instance["before_commit"],
                after_commit=instance["after_commit"],
                file_path=instance["file_path"],
                function_id=instance["function_id"],
                lines_added=instance.get("lines_added", 0),
                lines_deleted=instance.get("lines_deleted", 0),
                commit_message=instance.get("commit_message", ""),
            )

            if result:
                result_dict = result.to_dict()
                result_dict["unique_event_key"] = instance_key
                write_jsonl([result_dict], metric_changes_file, append=True)
                stats["successful_analyses"] += 1
                logger.info(f"  Success")
            else:
                stats["failed_analyses"] += 1
                stats["failed_instances"].append({
                    "smell_instance_id": instance["smell_instance_id"],
                    "event_type": instance["event_type"],
                    "file_path": instance["file_path"],
                })
                logger.warning(f"  Failed to extract metrics")

            processed_ids.add(instance_key)
            write_json({"processed_ids": list(processed_ids)}, checkpoint_file)

        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"Successful: {stats['successful_analyses']}")
        logger.info(f"Failed: {stats['failed_analyses']}")
        logger.info(f"Output file: {metric_changes_file}")

        return stats

    def analyze_repository(
        self, repo_name: str, analysis_dir: Path, resume: bool = False
    ) -> Dict[str, Any]:
        """Analyze maintainability metrics for all smell events in a repository."""
        logger.info("=" * 80)
        logger.info(f"Analyzing maintainability metrics for: {repo_name}")
        logger.info("=" * 80)

        repo_output_dir = self.get_repo_output_dir(repo_name)
        metric_changes_file = repo_output_dir / config.VALIDATED_METRICS_FILE
        checkpoint_file = repo_output_dir / config.METRIC_CHECKPOINT_FILE

        events = self.load_smell_events(repo_name, analysis_dir)
        logger.info(f"Loaded {len(events)} smell events")

        introduction_pairs, removal_pairs = self.group_smell_events_into_pairs(events)
        logger.info(f"Found {len(introduction_pairs)} introduction pairs")
        logger.info(f"Found {len(removal_pairs)} removal pairs")

        repo_path = get_repo_path(repo_name)
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        processed_ids = set()
        if resume and checkpoint_file.exists():
            checkpoint = read_json(checkpoint_file)
            processed_ids = set(checkpoint.get("processed_smell_ids", []))
            logger.info(f"Resuming: {len(processed_ids)} already processed")
        elif not resume and metric_changes_file.exists():
            metric_changes_file.unlink()

        stats = {
            "repo_name": repo_name,
            "total_smell_events": len(events),
            "introduction_pairs": len(introduction_pairs),
            "removal_pairs": len(removal_pairs),
            "successful_analyses": 0,
            "failed_analyses": 0
        }

        def process_pair(before_event, after_event, change_type):
            smell_id = before_event["smell_instance_id"]
            if smell_id in processed_ids:
                return

            metric_change = self.analyze_smell_event_pair(
                repo_path=str(repo_path),
                repo_name=repo_name,
                smell_instance_id=smell_id,
                smell_type=before_event["smell_type"],
                change_type=change_type,
                before_commit=before_event["commit_hash"],
                after_commit=after_event["commit_hash"],
                file_path=before_event["file_path"],
                function_id=before_event["function_id"],
                lines_added=after_event.get("lines_added", 0),
                lines_deleted=after_event.get("lines_deleted", 0),
                commit_message=after_event.get("commit_message", "")
            )

            if metric_change:
                write_jsonl([metric_change.to_dict()], metric_changes_file, append=True)
                stats["successful_analyses"] += 1
            else:
                stats["failed_analyses"] += 1

            processed_ids.add(smell_id)
            write_json({"processed_smell_ids": list(processed_ids)}, checkpoint_file)

        logger.info("Processing introduction pairs...")
        for before_event, after_event in introduction_pairs:
            process_pair(before_event, after_event, "introduction")

        logger.info("Processing removal pairs...")
        for before_event, after_event in removal_pairs:
            process_pair(before_event, after_event, "removal")

        logger.info(f"Completed: {stats['successful_analyses']} successful, {stats['failed_analyses']} failed")
        return stats

    def generate_reports(self) -> None:
        """Generate aggregate reports from all analyzed repositories."""
        logger.info("Generating aggregate metric impact reports...")

        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        all_changes = []
        for repo_dir in self.output_dir.iterdir():
            if repo_dir.is_dir() and repo_dir.name != "reports":
                changes_file = repo_dir / config.VALIDATED_METRICS_FILE
                if changes_file.exists():
                    changes = read_jsonl(changes_file)
                    all_changes.extend(changes)

        if not all_changes:
            logger.warning("No metric changes found to report")
            return

        logger.info(f"Collected {len(all_changes)} metric changes")

        aggregated = defaultdict(lambda: defaultdict(list))
        for change in all_changes:
            smell_type = change["smell_type"]
            deltas = change["delta_metrics"]
            for metric_name, delta_value in deltas.items():
                aggregated[smell_type][metric_name].append({
                    "change_type": change["change_type"],
                    "delta": delta_value,
                })

        summary = {}
        for smell_type, metrics in aggregated.items():
            summary[smell_type] = {}
            for metric_name, values in metrics.items():
                introductions = [v["delta"] for v in values if v["change_type"] == "introduction"]
                removals = [v["delta"] for v in values if v["change_type"] == "removal"]
                summary[smell_type][metric_name] = {
                    "introduction": {
                        "count": len(introductions),
                        "mean": sum(introductions) / len(introductions) if introductions else 0.0,
                    },
                    "removal": {
                        "count": len(removals),
                        "mean": sum(removals) / len(removals) if removals else 0.0,
                    }
                }

        json_report = reports_dir / config.AGGREGATE_METRIC_IMPACT_JSON
        write_json(summary, json_report)
        logger.info(f"Saved aggregate report: {json_report}")

    def generate_validation_summary(self, raw_smell_dir: Path) -> Dict[str, Any]:
        """
        Generate validation summary comparing raw smell events to validated instances.

        This report documents:
        - How many smell events were detected (raw)
        - How many passed validation (file exists at both before/after commits)
        - Validation rate per repository and overall
        - Reasons for invalid instances (for thesis methodology section)

        Args:
            raw_smell_dir: Directory containing raw smell detection results

        Returns:
            Validation summary dictionary
        """
        logger.info("Generating validation summary report...")

        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        per_repo_stats = []
        total_raw_events = 0
        total_validated = 0

        # Process each repository
        for repo_dir in self.output_dir.iterdir():
            if not repo_dir.is_dir() or repo_dir.name == "reports":
                continue

            repo_name = repo_dir.name.replace("_", "/", 1)

            # Load validated instances
            validated_file = repo_dir / "validated_smell_instances.jsonl"
            validated_instances = read_jsonl(validated_file) if validated_file.exists() else []

            # Load raw smell events
            raw_repo_dir = raw_smell_dir / repo_dir.name
            raw_events_file = raw_repo_dir / config.EVENTS_FILE
            raw_events = read_jsonl(raw_events_file) if raw_events_file.exists() else []

            # Count raw events by type (A = introduction, B = removal)
            # Note: Same smell can be introduced/removed multiple times (re-introductions)
            raw_introductions = 0
            raw_removals = 0
            for event in raw_events:
                commit_type = event.get("commit_type", "")
                if commit_type == "A":
                    raw_introductions += 1
                elif commit_type == "B":
                    raw_removals += 1

            # Count validated instances by type
            validated_introductions = len([v for v in validated_instances if v.get("event_type") == "introduction"])
            validated_removals = len([v for v in validated_instances if v.get("event_type") == "removal"])

            # Calculate invalid instances by type
            invalid_introductions = raw_introductions - validated_introductions
            invalid_removals = raw_removals - validated_removals

            raw_total = raw_introductions + raw_removals
            validated_total = validated_introductions + validated_removals
            invalid_total = invalid_introductions + invalid_removals

            repo_stats = {
                "repository": repo_name,
                "raw_transitions": raw_total,
                "raw_introductions": raw_introductions,
                "raw_removals": raw_removals,
                "validated_instances": validated_total,
                "validated_introductions": validated_introductions,
                "validated_removals": validated_removals,
                "invalid_instances": invalid_total,
                "invalid_introductions": invalid_introductions,
                "invalid_removals": invalid_removals,
                "validation_rate_percent": round(
                    validated_total / raw_total * 100, 1
                ) if raw_total else 0.0,
            }

            per_repo_stats.append(repo_stats)
            total_raw_events += raw_total
            total_validated += validated_total

        # Sort by repository name
        per_repo_stats.sort(key=lambda x: x["repository"])

        # Calculate totals from per-repo stats
        total_raw_introductions = sum(r["raw_introductions"] for r in per_repo_stats)
        total_raw_removals = sum(r["raw_removals"] for r in per_repo_stats)
        total_validated_introductions = sum(r["validated_introductions"] for r in per_repo_stats)
        total_validated_removals = sum(r["validated_removals"] for r in per_repo_stats)
        total_invalid_introductions = sum(r["invalid_introductions"] for r in per_repo_stats)
        total_invalid_removals = sum(r["invalid_removals"] for r in per_repo_stats)

        summary = {
            "description": "Validation summary comparing raw smell events to validated instances",
            "validation_criteria": {
                "file_must_exist_at_before_commit": True,
                "file_must_exist_at_after_commit": True,
                "reason": "Enables before/after metric comparison for statistical analysis"
            },
            "invalid_reasons": {
                "introduction_failures": {
                    "count": total_invalid_introductions,
                    "typical_reason": "File created with smell already present (file does not exist at commit A-1)",
                    "explanation": "When a new file is added that already contains a smell, there is no 'before' state to compare metrics against"
                },
                "removal_failures": {
                    "count": total_invalid_removals,
                    "typical_reason": "File deleted along with smell removal (file does not exist at commit B)",
                    "explanation": "When a file is deleted, there is no 'after' state to compare metrics against"
                }
            },
            "totals": {
                "raw_transitions": total_raw_events,
                "raw_introductions": total_raw_introductions,
                "raw_removals": total_raw_removals,
                "validated_instances": total_validated,
                "validated_introductions": total_validated_introductions,
                "validated_removals": total_validated_removals,
                "invalid_instances": total_raw_events - total_validated,
                "invalid_introductions": total_invalid_introductions,
                "invalid_removals": total_invalid_removals,
                "validation_rate_percent": round(
                    total_validated / total_raw_events * 100, 1
                ) if total_raw_events else 0.0,
            },
            "per_repository": per_repo_stats,
            "usage_notes": {
                "thesis_methodology": "Use totals.validation_rate_percent to report data filtering in methodology section",
                "thesis_threats": "Invalid instances represent inherent limitations in commit-level before/after analysis"
            }
        }

        # Save report
        report_file = reports_dir / "validation_summary.json"
        write_json(summary, report_file)
        logger.info(f"Saved validation summary: {report_file}")

        # Log summary
        logger.info(f"Validation Summary:")
        logger.info(f"  Raw transitions: {total_raw_events} ({total_raw_introductions} intro, {total_raw_removals} removal)")
        logger.info(f"  Validated: {total_validated} ({summary['totals']['validation_rate_percent']}%)")
        logger.info(f"  Invalid: {total_raw_events - total_validated} ({total_invalid_introductions} intro, {total_invalid_removals} removal)")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze maintainability metrics impact of code smells"
    )
    parser.add_argument("--repo", type=str, nargs="+", help="Repository name(s) to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all repositories")
    parser.add_argument("--analysis-dir", type=str, default=str(config.RAW_SMELL_DETECTION_DIR))
    parser.add_argument("--output-dir", type=str, default=str(config.PROCESSED_METRICS_DIR))
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--validated", action="store_true", help="Analyze only validated instances")
    parser.add_argument("--validation-summary", action="store_true",
                        help="Generate validation summary report (no analysis, just report)")

    args = parser.parse_args()

    analyzer = MetricsAnalyzer(output_dir=Path(args.output_dir))
    analysis_dir = Path(args.analysis_dir)

    try:
        # Handle validation summary report (standalone operation)
        if args.validation_summary:
            analyzer.generate_validation_summary(analysis_dir)
            logger.info("Validation summary complete!")
            return

        if args.all:
            repo_names = []
            for repo_dir in analysis_dir.iterdir():
                if repo_dir.is_dir() and repo_dir.name != "reports":
                    events_file = repo_dir / config.EVENTS_FILE
                    if events_file.exists():
                        repo_name = repo_dir.name.replace("_", "/", 1)
                        repo_names.append(repo_name)
            logger.info(f"Found {len(repo_names)} repositories to analyze")
        elif args.repo:
            repo_names = args.repo
        else:
            logger.error("Must specify --repo or --all")
            sys.exit(1)

        for repo_name in repo_names:
            try:
                if args.validated:
                    stats = analyzer.analyze_validated_instances(repo_name, resume=args.resume)
                else:
                    stats = analyzer.analyze_repository(repo_name, analysis_dir, resume=args.resume)
                logger.info(f"Completed {repo_name}: {stats}")
            except Exception as e:
                logger.error(f"Error analyzing {repo_name}: {e}", exc_info=True)

        analyzer.generate_reports()
        logger.info("Metrics analysis complete!")

    except Exception as e:
        logger.error(f"Error during metrics analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
