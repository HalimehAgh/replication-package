"""
Shared Constants for Metrics Analysis

Centralizes magic numbers and configuration values used across multiple
metrics analysis modules to improve code maintainability and clarity.

Note: Bug-fix detection patterns are in step1_commit_extraction/commit_analyzer.py
"""

# ============================================================
# API CONFIGURATION
# ============================================================

# API pagination
DEFAULT_PAGE_SIZE = 500

# HTTP request timeouts (seconds)
DEFAULT_TIMEOUT_SECONDS = 120

# Retry configuration
RETRY_INTERVAL_SECONDS = 5
MAX_RETRY_ATTEMPTS = 3


# ============================================================
# ANALYSIS CONFIGURATION
# ============================================================

# Validation settings
VALIDATION_PREVIEW_LIMIT = 5  # Number of instances to preview in validation reports

# Statistical analysis
MIN_SAMPLE_SIZE = 3  # Minimum samples needed for statistical comparison

# Parallel processing
DEFAULT_PARALLEL_WORKERS = 4  # Default number of parallel analysis workers


# ============================================================
# PROGRESS TRACKING
# ============================================================

# Checkpoint intervals
PROGRESS_UPDATE_INTERVAL = 10  # Log progress every N commits/instances


# ============================================================
# MAINTAINABILITY METRIC DIMENSIONS (Dependent Variables)
# ============================================================
# Quality metrics grouped by maintainability dimension (ISO/IEC 25010)
# Reference: Tufano et al. (2017), Palomba et al. (2018)
# Note: Time-window metrics moved to CONTROL_VARIABLES

METRIC_DIMENSIONS = {
    "structural_complexity": [
        "cyclomatic_complexity",
        "nested_loop_depth",
        "loop_count"
    ],
    "code_size": [
        "loc",
        "sloc",
        "num_functions",
        "avg_function_length"
    ],
    "defect_proneness": [
        "exception_handling_coverage",
        "code_clone_ratio",
        "type_annotation_coverage"
    ],
    "understandability": [
        "docstring_coverage",
        "comment_ratio",
        "cognitive_complexity"
    ],
    "testing_quality": [
        "test_function_ratio",
        "assertion_density"
    ],
    "ml_specific": [
        "data_operation_density",
        "config_explicitness_ratio"
    ],
    "historical_context": [
        "window_total_churn",
        "window_modified_files",
        "window_commit_count",
        "window_bug_fix_ratio"
    ]
}

# ============================================================
# CONTROL VARIABLES (For Stratification, Not Quality Metrics)
# ============================================================
# These variables measure code CONTEXT, not code QUALITY.
# Used to stratify analysis by file characteristics:
# - Stable vs active files (by churn)
# - Clean vs problematic files (by bug-fix ratio)
# Reference: Khomh et al. (2012), Palomba et al. (2018)

CONTROL_VARIABLES = {
    "time_window": [
        "window_total_churn",
        "window_modified_files",
        "window_commit_count",
        "window_bug_fix_ratio"
    ],
    "smell_context": [
        "smell_count_at_commit"  # Co-occurrence tracking per Khomh et al. (2012)
    ]
}

# Metrics where HIGHER values indicate BETTER maintainability
# These need to be INVERTED during normalization so that:
# 0 = best maintainability, 1 = worst maintainability
INVERT_METRICS = {
    "docstring_coverage",
    "test_function_ratio",
    "exception_handling_coverage",
    "type_annotation_coverage",
    "config_explicitness_ratio"
}

# All metrics that are snapshot-based (measured at commit point)
SNAPSHOT_METRICS = [
    "cyclomatic_complexity", "nested_loop_depth", "loop_count",
    "loc", "sloc", "num_functions", "avg_function_length",
    "exception_handling_coverage", "code_clone_ratio", "type_annotation_coverage",
    "docstring_coverage", "comment_ratio", "cognitive_complexity",
    "test_function_ratio", "assertion_density",
    "data_operation_density", "config_explicitness_ratio"
]

# Metrics that use time-window (3-month retrospective)
# Measured ONLY at A and B commits (not A-1 and B-1)
TIME_WINDOW_METRICS = [
    "window_total_churn",
    "window_modified_files",
    "window_commit_count",
    "window_bug_fix_ratio"
]

# All metrics combined (for convenience)
ALL_METRICS = SNAPSHOT_METRICS + TIME_WINDOW_METRICS

# Flatten control variables for easy lookup
ALL_CONTROL_VARIABLES = [
    var for vars_list in CONTROL_VARIABLES.values()
    for var in vars_list
]
