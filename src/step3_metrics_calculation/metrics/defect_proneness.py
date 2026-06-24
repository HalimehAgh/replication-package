"""
Defect Proneness Metrics

Metrics: exception_handling_coverage, code_clone_ratio, type_annotation_coverage
"""

import ast


def calculate_exception_coverage(tree: ast.AST) -> float:
    """
    Calculate percentage of functions containing try/except blocks.

    Args:
        tree: AST of the Python file

    Returns:
        Ratio of functions with exception handling (0.0 to 1.0)
    """
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        return 0.0

    functions_with_try = 0
    for func in functions:
        for node in ast.walk(func):
            if isinstance(node, ast.Try):
                functions_with_try += 1
                break

    return functions_with_try / len(functions)


def calculate_type_annotation_coverage(tree: ast.AST) -> float:
    """
    Calculate percentage of functions with complete type hints.

    A function is considered fully annotated if:
    - All parameters (except self/cls) have type annotations
    - The function has a return type annotation (or is __init__)

    Args:
        tree: AST of the Python file

    Returns:
        Ratio of fully annotated functions (0.0 to 1.0)
    """
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        return 0.0

    fully_annotated = 0
    for func in functions:
        params_with_types = 0
        total_params = 0

        for arg in func.args.args:
            if arg.arg in ('self', 'cls'):
                continue
            total_params += 1
            if arg.annotation is not None:
                params_with_types += 1

        if func.args.vararg:
            total_params += 1
            if func.args.vararg.annotation is not None:
                params_with_types += 1
        if func.args.kwarg:
            total_params += 1
            if func.args.kwarg.annotation is not None:
                params_with_types += 1

        has_return = func.returns is not None or func.name == '__init__'
        params_ok = total_params == 0 or params_with_types == total_params

        if params_ok and has_return:
            fully_annotated += 1

    return fully_annotated / len(functions)


# Note: code_clone_ratio is in external_tools.py (uses PMD CPD)
