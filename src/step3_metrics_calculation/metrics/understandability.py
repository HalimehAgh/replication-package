"""
Understandability Metrics

Metrics: docstring_coverage, comment_ratio, cognitive_complexity
"""

import ast


def calculate_docstring_coverage(tree: ast.AST) -> float:
    """
    Calculate percentage of functions with docstrings.

    Args:
        tree: AST of the Python file

    Returns:
        Ratio of functions with docstrings (0.0 to 1.0)
    """
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        return 0.0

    with_docstring = sum(1 for func in functions if ast.get_docstring(func))
    return with_docstring / len(functions)


def calculate_comment_ratio(comments: int, loc: int) -> float:
    """
    Calculate ratio of comment lines to total lines.

    Args:
        comments: Number of comment lines
        loc: Total lines of code

    Returns:
        Ratio of comments to total lines (0.0 to 1.0)
    """
    if loc <= 0:
        return 0.0
    return comments / loc


def calculate_cognitive_complexity(code: str) -> float:
    """
    Calculate total cognitive complexity for a file.

    Cognitive complexity measures how difficult code is to understand,
    based on nesting, control flow breaks, and recursion.

    Args:
        code: Python source code

    Returns:
        Total cognitive complexity score
    """
    try:
        from cognitive_complexity.api import get_cognitive_complexity
        tree = ast.parse(code)
    except (ImportError, SyntaxError):
        return 0.0

    total_cc = 0.0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                cc = get_cognitive_complexity(node)
                total_cc += cc
            except Exception:
                pass

    return total_cc
