"""
Clustering Module for Hypothesis Testing

Provides utilities for clustering analysis in regression models:
- Cluster statistics and summaries
- ICC (Intraclass Correlation Coefficient) estimation
- Cluster diagnostics and recommendations

Used to account for within-cluster correlation in smell events.

Usage
-----
from src.step4_analysis.hypothesis_testing.clustering import (
    cluster_summary,
    diagnose_cluster_effects,
    run_cluster_analysis
)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CLUSTER STATISTICS
# ============================================================

def cluster_summary(
    df: pd.DataFrame,
    cluster_col: str = 'file_id'
) -> Dict[str, Any]:
    """
    Generate summary statistics about clusters.

    Args:
        df: DataFrame with observations
        cluster_col: Column name for cluster identifier

    Returns:
        Dictionary with cluster statistics
    """
    clusters = df[cluster_col]
    cluster_sizes = clusters.value_counts()

    return {
        'n_observations': len(df),
        'n_clusters': clusters.nunique(),
        'avg_cluster_size': float(cluster_sizes.mean()),
        'median_cluster_size': float(cluster_sizes.median()),
        'min_cluster_size': int(cluster_sizes.min()),
        'max_cluster_size': int(cluster_sizes.max()),
        'std_cluster_size': float(cluster_sizes.std()),
        'singletons': int((cluster_sizes == 1).sum()),
        'singleton_ratio': float((cluster_sizes == 1).sum() / len(cluster_sizes)),
    }


def _estimate_icc(
    df: pd.DataFrame,
    cluster_col: str,
    value_col: str
) -> float:
    """
    Estimate Intraclass Correlation Coefficient (ICC).

    ICC measures the proportion of total variance that is between-cluster.

    Args:
        df: DataFrame with observations
        cluster_col: Column name for cluster identifier
        value_col: Column name for value to analyze

    Returns:
        ICC estimate (0 to 1)
    """
    if value_col not in df.columns:
        return 0.0

    # Calculate between-cluster and within-cluster variance
    group_means = df.groupby(cluster_col)[value_col].mean()
    grand_mean = df[value_col].mean()

    # Between-cluster sum of squares
    group_sizes = df.groupby(cluster_col).size()
    ss_between = ((group_means - grand_mean) ** 2 * group_sizes).sum()

    # Total sum of squares
    ss_total = ((df[value_col] - grand_mean) ** 2).sum()

    if ss_total == 0:
        return 0.0

    # ICC is proportion of variance that is between-cluster
    return float(ss_between / ss_total)


# ============================================================
# CLUSTER DIAGNOSTICS
# ============================================================

def diagnose_cluster_effects(
    df: pd.DataFrame,
    y_col: str,
    cluster_col: str = 'file_id'
) -> Dict[str, Any]:
    """
    Diagnose cluster effects on the dependent variable.

    Args:
        df: DataFrame with observations
        y_col: Dependent variable column
        cluster_col: Cluster identifier column

    Returns:
        Dictionary with diagnostic statistics
    """
    if y_col not in df.columns:
        return {'error': f'Column {y_col} not found'}

    clusters = df[cluster_col]
    y_values = df[y_col]

    # Calculate cluster-level statistics
    cluster_stats = df.groupby(cluster_col)[y_col].agg(['mean', 'std', 'count'])

    # ICC for dependent variable
    icc = _estimate_icc(df, cluster_col, y_col)

    # Design effect (rough approximation)
    avg_cluster_size = cluster_stats['count'].mean()
    design_effect = 1 + (avg_cluster_size - 1) * icc

    return {
        'icc': icc,
        'design_effect': design_effect,
        'effective_sample_size': len(df) / design_effect if design_effect > 0 else len(df),
        'n_clusters': clusters.nunique(),
        'cluster_y_means_var': float(cluster_stats['mean'].var()),
        'cluster_y_means_range': float(cluster_stats['mean'].max() - cluster_stats['mean'].min()),
        'recommendation': _recommend_clustering_approach(icc, design_effect),
    }


def _recommend_clustering_approach(icc: float, design_effect: float) -> str:
    """
    Provide recommendation based on ICC and design effect.

    Args:
        icc: Intraclass correlation coefficient
        design_effect: Estimated design effect

    Returns:
        Recommendation string
    """
    if icc < 0.05:
        return "Low ICC (<0.05): Clustering may not be necessary, but using clustered SE is still recommended"
    elif icc < 0.15:
        return "Moderate ICC (0.05-0.15): Clustered standard errors are appropriate"
    elif icc < 0.30:
        return "High ICC (0.15-0.30): Clustered SE essential; consider multilevel models"
    else:
        return "Very high ICC (>0.30): Strong cluster effects; multilevel modeling recommended"


# ============================================================
# CLUSTER COMPARISON
# ============================================================

def compare_cluster_levels(
    df: pd.DataFrame,
    y_col: str,
    file_cluster_col: str = 'file_id',
    project_cluster_col: str = 'project'
) -> Dict[str, Any]:
    """
    Compare clustering at file vs project level.

    Args:
        df: DataFrame with observations
        y_col: Dependent variable column
        file_cluster_col: File-level cluster column
        project_cluster_col: Project-level cluster column

    Returns:
        Dictionary comparing cluster levels
    """
    file_diagnostics = diagnose_cluster_effects(df, y_col, file_cluster_col)
    project_diagnostics = diagnose_cluster_effects(df, y_col, project_cluster_col)

    return {
        'file_level': {
            'n_clusters': file_diagnostics.get('n_clusters'),
            'icc': file_diagnostics.get('icc'),
            'design_effect': file_diagnostics.get('design_effect'),
        },
        'project_level': {
            'n_clusters': project_diagnostics.get('n_clusters'),
            'icc': project_diagnostics.get('icc'),
            'design_effect': project_diagnostics.get('design_effect'),
        },
        'recommendation': _recommend_cluster_level(
            file_diagnostics.get('icc', 0),
            project_diagnostics.get('icc', 0),
            file_diagnostics.get('n_clusters', 0),
            project_diagnostics.get('n_clusters', 0),
        ),
    }


def _recommend_cluster_level(
    file_icc: float,
    project_icc: float,
    n_file_clusters: int,
    n_project_clusters: int
) -> str:
    """
    Recommend appropriate clustering level.

    Args:
        file_icc: ICC at file level
        project_icc: ICC at project level
        n_file_clusters: Number of file clusters
        n_project_clusters: Number of project clusters

    Returns:
        Recommendation string
    """
    # Need at least some clusters for valid inference
    if n_project_clusters < 10:
        return f"Use file-level clustering: Too few projects ({n_project_clusters}) for project-level clustering"

    if file_icc > project_icc * 1.5:
        return "Use file-level clustering: Higher ICC at file level suggests stronger file effects"
    elif project_icc > file_icc * 1.5:
        return "Use project-level clustering: Higher ICC at project level suggests stronger project effects"
    else:
        return "Consider two-way clustering: Similar ICC at both levels"


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_cluster_analysis(
    df: pd.DataFrame,
    y_col: str,
    cluster_cols: List[str] = None
) -> Dict[str, Any]:
    """
    Run comprehensive cluster analysis.

    Args:
        df: DataFrame with observations
        y_col: Dependent variable column
        cluster_cols: List of cluster columns to analyze

    Returns:
        Dictionary with full cluster analysis
    """
    if cluster_cols is None:
        cluster_cols = ['file_id', 'project']

    results = {}

    for cluster_col in cluster_cols:
        if cluster_col in df.columns:
            logger.info(f"Analyzing cluster level: {cluster_col}")

            results[cluster_col] = {
                'summary': cluster_summary(df, cluster_col),
                'diagnostics': diagnose_cluster_effects(df, y_col, cluster_col),
            }

    # Add comparison if both levels available
    if 'file_id' in results and 'project' in results:
        results['comparison'] = compare_cluster_levels(df, y_col)

    return results
