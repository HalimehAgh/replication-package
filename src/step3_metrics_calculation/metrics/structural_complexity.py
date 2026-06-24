"""
Structural Complexity Metrics

Metrics: cyclomatic_complexity, nested_loop_depth, loop_count
"""

import ast
from typing import Tuple


def count_loops_and_nesting(tree: ast.AST) -> Tuple[int, int]:
    """
    Count total loops and find maximum nesting depth.

    Args:
        tree: AST of the Python file

    Returns:
        Tuple of (loop_count, max_nesting_depth)
    """
    loop_count = 0
    max_depth = 0

    def visit(node: ast.AST, current_depth: int = 0) -> None:
        nonlocal loop_count, max_depth
        if isinstance(node, (ast.For, ast.While)):
            loop_count += 1
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        for child in ast.iter_child_nodes(node):
            visit(child, current_depth)

    visit(tree)
    return loop_count, max_depth


def get_cyclomatic_complexity(code: str) -> Tuple[float, int, int]:
    """
    Calculate cyclomatic complexity metrics using radon.

    Args:
        code: Python source code

    Returns:
        Tuple of (total_complexity, max_complexity, num_functions)
    """
    from radon.complexity import cc_visit
    from radon.visitors import Function

    try:
        complexity_results = cc_visit(code)
        total_complexity = sum(item.complexity for item in complexity_results)

        functions_only = [item for item in complexity_results if isinstance(item, Function)]
        num_functions = len(functions_only)
        max_complexity = max((item.complexity for item in functions_only), default=0)

        return total_complexity, max_complexity, num_functions
    except SyntaxError:
        return 0.0, 0, 0
