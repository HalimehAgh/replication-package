"""
ML-Specific Metrics for Code Quality Analysis

Calculates metrics specific to machine learning code patterns:
- Data Operation Density: pandas/numpy/tensor operations per total operations
- Configuration Explicitness Ratio: explicit params / total params in ML function calls

These metrics operationalize ML-specific maintainability concerns identified in:
- Sculley et al. (2015): Hidden technical debt in ML systems
- Zhang et al. (2022): Code smells for ML applications
- Shivashankar et al. (2025): ML maintainability challenges

Reference: ISO/IEC 25010 - Analyzability, Modifiability, Testability
"""

import ast
from typing import Tuple, Set, Dict, Any

# =============================================================================
# ML Library Operation Patterns
# =============================================================================

# Common ML/data library method calls to detect
ML_DATA_OPERATIONS = {
    # Pandas operations
    'pandas': {
        'read_csv', 'read_excel', 'read_json', 'read_parquet', 'read_sql',
        'DataFrame', 'Series', 'merge', 'concat', 'groupby', 'pivot',
        'apply', 'transform', 'agg', 'rolling', 'resample',
        'fillna', 'dropna', 'drop_duplicates', 'sort_values',
        'to_csv', 'to_excel', 'to_json', 'to_parquet',
    },
    # NumPy operations
    'numpy': {
        'array', 'ndarray', 'zeros', 'ones', 'empty', 'full',
        'reshape', 'transpose', 'flatten', 'ravel',
        'concatenate', 'stack', 'vstack', 'hstack', 'split',
        'dot', 'matmul', 'einsum', 'tensordot',
        'sum', 'mean', 'std', 'var', 'min', 'max',
        'argmax', 'argmin', 'argsort', 'sort',
        'random', 'seed', 'shuffle',
    },
    # PyTorch operations
    'torch': {
        'tensor', 'Tensor', 'zeros', 'ones', 'empty', 'rand', 'randn',
        'cat', 'stack', 'squeeze', 'unsqueeze', 'view', 'reshape',
        'mm', 'bmm', 'matmul', 'einsum',
        'nn', 'Module', 'Linear', 'Conv2d', 'LSTM', 'Transformer',
        'optim', 'Adam', 'SGD', 'lr_scheduler',
        'backward', 'step', 'zero_grad',
        'cuda', 'to', 'device',
        'save', 'load', 'state_dict',
    },
    # TensorFlow/Keras operations
    'tensorflow': {
        'Variable', 'constant', 'placeholder', 'Session',
        'keras', 'Sequential', 'Model', 'Dense', 'Conv2D', 'LSTM',
        'compile', 'fit', 'predict', 'evaluate',
        'GradientTape', 'function',
    },
    # Scikit-learn operations
    'sklearn': {
        'fit', 'predict', 'transform', 'fit_transform',
        'train_test_split', 'cross_val_score',
        'GridSearchCV', 'RandomizedSearchCV',
        'Pipeline', 'ColumnTransformer',
    },
}

# Common ML function calls that typically have many parameters
ML_CONFIGURABLE_FUNCTIONS = {
    # PyTorch
    'Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'LSTM', 'GRU', 'Transformer',
    'BatchNorm1d', 'BatchNorm2d', 'Dropout', 'MaxPool2d', 'AvgPool2d',
    'Adam', 'SGD', 'AdamW', 'RMSprop',
    'DataLoader', 'Dataset',
    # TensorFlow/Keras
    'Dense', 'Conv2D', 'LSTM', 'GRU', 'Dropout', 'BatchNormalization',
    'compile', 'fit', 'Sequential', 'Model',
    # Scikit-learn
    'RandomForestClassifier', 'RandomForestRegressor',
    'GradientBoostingClassifier', 'GradientBoostingRegressor',
    'LogisticRegression', 'SVC', 'SVR',
    'KMeans', 'DBSCAN', 'PCA',
    'GridSearchCV', 'RandomizedSearchCV',
    # Pandas
    'read_csv', 'read_excel', 'to_csv', 'merge', 'groupby',
    'fillna', 'dropna', 'pivot_table',
}

# Aliases for common ML libraries
ML_LIBRARY_ALIASES = {
    'pd': 'pandas',
    'np': 'numpy',
    'tf': 'tensorflow',
    'torch': 'torch',
    'nn': 'torch',
    'F': 'torch',
    'sk': 'sklearn',
}


class MLOperationVisitor(ast.NodeVisitor):
    """AST visitor to count ML-related operations."""

    def __init__(self):
        self.ml_operations = 0
        self.total_operations = 0
        self.explicit_params = 0
        self.total_possible_params = 0

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls to detect ML operations."""
        self.total_operations += 1

        func_name = self._get_func_name(node)

        # Check if this is an ML operation
        if self._is_ml_operation(node, func_name):
            self.ml_operations += 1

        # Check configuration explicitness for ML functions
        if func_name in ML_CONFIGURABLE_FUNCTIONS:
            # Count explicit keyword arguments
            self.explicit_params += len(node.keywords)
            # Estimate total possible params (explicit + positional + some typical defaults)
            # Most ML functions have 3-10 important parameters
            estimated_params = max(len(node.args) + len(node.keywords), 3)
            self.total_possible_params += estimated_params

        self.generic_visit(node)

    def _get_func_name(self, node: ast.Call) -> str:
        """Extract function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _is_ml_operation(self, node: ast.Call, func_name: str) -> bool:
        """Check if a function call is an ML-related operation."""
        # Check direct function name
        for lib_ops in ML_DATA_OPERATIONS.values():
            if func_name in lib_ops:
                return True

        # Check attribute access (e.g., pd.read_csv, torch.tensor)
        if isinstance(node.func, ast.Attribute):
            # Get the object being accessed
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                # Check if it's an ML library alias
                if obj_name in ML_LIBRARY_ALIASES:
                    lib_name = ML_LIBRARY_ALIASES[obj_name]
                    if func_name in ML_DATA_OPERATIONS.get(lib_name, set()):
                        return True
                # Check direct library names
                if obj_name in ML_DATA_OPERATIONS:
                    if func_name in ML_DATA_OPERATIONS[obj_name]:
                        return True

        return False


def calculate_data_operation_density(tree: ast.AST) -> float:
    """
    Calculate density of ML/data operations in the code.

    Data Operation Density = ML operations / Total function calls

    This metric captures the intensity of data processing operations
    (pandas, numpy, tensor manipulations) relative to total operations.
    Higher values indicate more data-intensive code.

    Args:
        tree: Parsed AST

    Returns:
        Ratio of ML ops to total operations (0.0-1.0)
    """
    visitor = MLOperationVisitor()
    visitor.visit(tree)

    if visitor.total_operations == 0:
        return 0.0

    return visitor.ml_operations / visitor.total_operations


def calculate_config_explicitness_ratio(tree: ast.AST) -> float:
    """
    Calculate ratio of explicitly set parameters in ML function calls.

    Config Explicitness Ratio = Explicit kwargs / Estimated total params

    This metric measures how much ML configurations (hyperparameters,
    data types, merge parameters) are explicitly defined versus left
    to implicit defaults.

    Higher values indicate more explicit configuration (better for
    reproducibility and maintainability).

    Args:
        tree: Parsed AST

    Returns:
        Ratio of explicit params to total possible params (0.0-1.0)
    """
    visitor = MLOperationVisitor()
    visitor.visit(tree)

    if visitor.total_possible_params == 0:
        return 0.0

    # Cap at 1.0 since explicit params might exceed our estimate
    return min(1.0, visitor.explicit_params / visitor.total_possible_params)


def calculate_ml_metrics(code: str) -> Dict[str, float]:
    """
    Calculate all ML-specific metrics for a code file.

    Args:
        code: Source code string

    Returns:
        Dictionary with data_operation_density and config_explicitness_ratio
    """
    try:
        tree = ast.parse(code)
        return {
            "data_operation_density": calculate_data_operation_density(tree),
            "config_explicitness_ratio": calculate_config_explicitness_ratio(tree),
        }
    except SyntaxError:
        return {
            "data_operation_density": 0.0,
            "config_explicitness_ratio": 0.0,
        }


# =============================================================================
# Convenience function for integration with analyze_metrics.py
# =============================================================================

def get_ml_metrics_from_tree(tree: ast.AST) -> Tuple[float, float]:
    """
    Calculate ML metrics from an already-parsed AST.

    Args:
        tree: Parsed AST

    Returns:
        Tuple of (data_operation_density, config_explicitness_ratio)
    """
    return (
        calculate_data_operation_density(tree),
        calculate_config_explicitness_ratio(tree),
    )
