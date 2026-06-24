"""
Testing Quality Metrics

Metrics: test_function_ratio, assertion_density
"""

import ast
from typing import Tuple


def analyze_test_functions(tree: ast.AST, sloc: int = 1) -> Tuple[float, float]:
    """
    Analyze test functions and assertion density.

    Identifies test functions by:
    - Function names starting with 'test_' or 'Test'
    - Methods in classes starting with 'Test'

    Counts assertions from:
    - assert statements
    - Method calls starting with 'assert' (e.g., assertEqual, assertTrue)

    Args:
        tree: AST of the Python file
        sloc: Source lines of code (unused, kept for API compatibility)

    Returns:
        Tuple of (test_function_ratio, assertion_density)
        - test_function_ratio: Ratio of test functions to all functions
        - assertion_density: Average assertions per test function
    """
    all_functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    test_functions = []
    for func in all_functions:
        if func.name.startswith('test_') or func.name.startswith('Test'):
            test_functions.append(func)

    # Also check for methods in Test* classes
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item not in test_functions and item.name.startswith('test'):
                        test_functions.append(item)

    test_ratio = len(test_functions) / len(all_functions) if all_functions else 0.0

    # Count assertions
    total_assertions = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            total_assertions += 1
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr.startswith('assert'):
                    total_assertions += 1

    assertion_density = total_assertions / len(test_functions) if test_functions else 0.0

    return test_ratio, assertion_density


def is_test_file(file_path: str) -> bool:
    """
    Check if a file is likely a test file based on its path.

    Args:
        file_path: Path to the file

    Returns:
        True if the file appears to be a test file
    """
    path_lower = file_path.lower()
    return (
        '/test/' in path_lower or
        '/tests/' in path_lower or
        path_lower.endswith('_test.py') or
        path_lower.endswith('test_.py') or
        '/test_' in path_lower
    )
