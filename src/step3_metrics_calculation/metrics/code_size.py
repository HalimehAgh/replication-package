"""
Code Size Metrics

Metrics: loc, sloc, num_functions, avg_function_length
"""

from typing import Dict, Any, List


def get_raw_metrics(code: str) -> Dict[str, int]:
    """
    Get raw code metrics using radon.

    Args:
        code: Python source code

    Returns:
        Dictionary with loc, sloc, comments, blank counts
    """
    from radon.raw import analyze

    try:
        raw = analyze(code)
        return {
            "loc": raw.loc,
            "sloc": raw.sloc,
            "comments": raw.comments,
            "blank": raw.blank,
        }
    except SyntaxError:
        return {"loc": 0, "sloc": 0, "comments": 0, "blank": 0}


def calculate_avg_function_length(loc: int, num_functions: int) -> float:
    """
    Calculate average function length.

    Args:
        loc: Total lines of code
        num_functions: Number of functions in the file

    Returns:
        Average lines per function
    """
    if num_functions <= 0:
        return 0.0
    return loc / num_functions


def get_function_metrics(code: str) -> List[Dict[str, Any]]:
    """
    Extract per-function metrics using radon.

    Args:
        code: Python source code

    Returns:
        List of dictionaries with function metrics
    """
    from radon.complexity import cc_visit
    from radon.visitors import Function

    try:
        complexity_results = cc_visit(code)
        functions_only = [item for item in complexity_results if isinstance(item, Function)]

        function_metrics = []
        for item in functions_only:
            function_metrics.append({
                "name": item.name,
                "classname": item.classname or None,
                "complexity": item.complexity,
                "line_start": item.lineno,
                "line_end": item.endline
            })
        return function_metrics
    except SyntaxError:
        return []


def count_classes(code: str) -> int:
    """
    Count number of classes in the code.

    Args:
        code: Python source code

    Returns:
        Number of classes
    """
    from radon.complexity import cc_visit
    from radon.visitors import Class

    try:
        complexity_results = cc_visit(code)
        return len([item for item in complexity_results if isinstance(item, Class)])
    except SyntaxError:
        return 0
