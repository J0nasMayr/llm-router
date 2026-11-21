import logging

import numpy as np
import pandas as pd

from config.manager import load_config
from src.metrics.normalizer import MetricNormalizer

logger = logging.getLogger(__name__)


def calculate_reward(
    accuracy_norm: float, energy_norm: float, lambda_weight: float
) -> float:
    """Calculate reward based on normalized accuracy and energy.

    Args:
        accuracy_norm (float): Normalized accuracy (0 to 1).
        energy_norm (float): Normalized energy (0 to 1, where 0 is best).
        lambda_weight (float): Weight for energy efficiency (0 to 1).

    Returns:
        float: Calculated reward (0 to 1), or NaN if inputs are NaN.
    """
    if pd.isna(accuracy_norm) or pd.isna(energy_norm):
        return np.nan
    energy_efficiency_norm = 1.0 - energy_norm
    reward = (
        1.0 - lambda_weight
    ) * accuracy_norm + lambda_weight * energy_efficiency_norm
    return max(0.0, min(1.0, reward))


def normalize_metrics(inference_results_df: pd.DataFrame):
    """
    Normalize performance metrics using dataset-specific bounds.
    Assumes input DataFrame has 'dataset', 'accuracy', 'latency',
    'energy_consumption', 'input_tokens', 'output_tokens'.
    Adds '_norm' columns.

    Args:
        inference_results_df (pd.DataFrame): DataFrame with raw inference results.

    Returns:
        pd.DataFrame: DataFrame with added normalized columns, or original if error.
    """
    df_copy = inference_results_df.copy()
    if "energy_per_token" not in df_copy.columns:
        logger.info("Calculating energy per token...")
        required_cols = ["energy_consumption", "input_tokens", "output_tokens"]
        if not all(col in df_copy.columns for col in required_cols):
            logger.error(
                f"Missing columns required for energy_per_token calculation: {required_cols}"
            )
            return inference_results_df

        for col in required_cols:
            before = df_copy[col].copy()
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
            corrupted = df_copy[col].isna() & before.notna()
            if corrupted.any():
                logger.warning(
                    f"Corrupted {col} values converted to NaN: {before[corrupted].tolist()}"
                )

        input_tokens = df_copy["input_tokens"].fillna(0)
        output_tokens = df_copy["output_tokens"].fillna(0)
        total_tokens = input_tokens + output_tokens

        df_copy["energy_per_token"] = np.where(
            (total_tokens > 0) & (df_copy["energy_consumption"].notna()),
            df_copy["energy_consumption"] / total_tokens,
            np.nan,
        )
        ept_nan_count = df_copy["energy_per_token"].isnull().sum()
        if ept_nan_count > 0:
            logger.info(f"  Generated {ept_nan_count} NaN values for energy_per_token.")
    if "dataset" not in df_copy.columns:
        logger.error(
            "Input DataFrame must contain a 'dataset' column for normalization."
        )
        return inference_results_df

    try:
        normalizer = MetricNormalizer(load_config("metrics"))
        if not normalizer.bounds:
            logger.error(
                "Failed to load metric normalization bounds. Check config/metrics.yaml."
            )
            return inference_results_df
    except Exception as e:
        logger.error(f"Error initializing MetricNormalizer: {e}")
        return inference_results_df

    metrics_to_normalize = [
        "accuracy",
        "latency",
        "energy_consumption",
        "energy_per_token",
    ]

    logger.info("Normalizing metrics based on dataset bounds...")
    for metric in metrics_to_normalize:
        norm_col_name = f"{metric}_norm"
        if metric not in df_copy.columns:
            logger.warning(
                f"Metric column '{metric}' not found in DataFrame. Skipping normalization."
            )
            df_copy[norm_col_name] = np.nan
            continue

        logger.info(f"  Normalizing {metric}...")
        df_copy[metric] = pd.to_numeric(df_copy[metric], errors="coerce")
        df_copy[norm_col_name] = df_copy.apply(
            lambda row: normalizer.normalize(metric, row[metric], row["dataset"]),
            axis=1,
        )

    return df_copy


def calculate_summary_stats(
    results_df: pd.DataFrame, groupby_col: str = "algorithm", metrics_cols: list = None
):
    """
    Calculate summary statistics (mean, std, median, sum) for specified metrics,
    grouped by a given column.

    Args:
        results_df (pd.DataFrame): DataFrame with detailed results.
        groupby_col (str): Column to group by (e.g., 'algorithm').
        metrics_cols (list, optional): List of numeric columns to summarize.
                                     Defaults to likely metrics if None.

    Returns:
        pd.DataFrame: DataFrame with summary statistics, indexed by the groupby_col.
                      Returns empty DataFrame on error.
    """
    if metrics_cols is None:
        metrics_cols = ["accuracy_norm", "energy_per_token_norm", "latency", "reward"]
        metrics_cols = [col for col in metrics_cols if col in results_df.columns]

    if not metrics_cols:
        logger.error(
            "No valid metric columns found or provided for summary statistics."
        )
        return pd.DataFrame()

    if groupby_col not in results_df.columns:
        logger.error(f"Groupby column '{groupby_col}' not found in DataFrame.")
        return pd.DataFrame()

    logger.info(
        f"Calculating summary statistics grouped by '{groupby_col}' for metrics: {metrics_cols}"
    )
    for col in metrics_cols:
        if col not in results_df.columns:
            logger.warning(
                f"Metric column '{col}' not found for summary stats. Skipping."
            )
            continue
        if not pd.api.types.is_numeric_dtype(results_df[col]):
            logger.warning(
                f"Column '{col}' is not numeric. Attempting conversion for summary stats."
            )
            results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
    aggregations = {metric: ["mean", "std", "median", "sum"] for metric in metrics_cols}

    try:
        grouped = results_df.groupby(groupby_col)
        summary = grouped.agg(aggregations)
        if "model_id" in results_df.columns:
            grouped["model_id"].value_counts().unstack(fill_value=0)
            summary[("model_distribution", "counts")] = grouped["model_id"].apply(
                lambda x: x.value_counts().to_dict()
            )

        return summary
    except Exception as e:
        logger.error(f"Error calculating summary statistics: {e}")
        return pd.DataFrame()


def find_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    lower_x_is_better: bool = True,
    higher_y_is_better: bool = True,
):
    """Identifies Pareto optimal points in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data points.
        x_col (str): Name of the column representing the X-axis.
        y_col (str): Name of the column representing the Y-axis.
        lower_x_is_better (bool): True if lower values are better for X-axis.
        higher_y_is_better (bool): True if higher values are better for Y-axis.

    Returns:
        pd.DataFrame: A subset of the input DataFrame containing only the Pareto optimal points.
    """
    df_copy = df.dropna(subset=[x_col, y_col]).copy()
    if df_copy.empty:
        return pd.DataFrame()

    is_pareto = np.ones(df_copy.shape[0], dtype=bool)

    for i, point in df_copy.iterrows():
        if not is_pareto[i]:
            continue
        for j, other_point in df_copy.iterrows():
            if i == j:
                continue
            x_dominates = (
                other_point[x_col] < point[x_col]
                if lower_x_is_better
                else other_point[x_col] > point[x_col]
            )
            y_dominates = (
                other_point[y_col] > point[y_col]
                if higher_y_is_better
                else other_point[y_col] < point[y_col]
            )

            x_equal = other_point[x_col] == point[x_col]
            y_equal = other_point[y_col] == point[y_col]
            dominates_cond1 = x_dominates and (y_dominates or y_equal)
            dominates_cond2 = y_dominates and (x_dominates or x_equal)

            if dominates_cond1 or dominates_cond2:
                is_pareto[i] = False
                break

    return df_copy[is_pareto]


def prepare_reward_structures(
    results_norm_df: pd.DataFrame, lambda_weight: float
) -> tuple[dict, dict]:
    """Calculates per-query optimal rewards and a reward lookup dictionary.

    Adds a 'reward' column to the input DataFrame based on the lambda_weight.

    Args:
        results_norm_df (pd.DataFrame): DataFrame with normalized metrics
                                       (must include 'query_id', 'model_id',
                                       'accuracy_norm', 'energy_per_token_norm').
        lambda_weight (float): Weight for energy efficiency (0 to 1).

    Returns:
        tuple[dict, dict]:
            - optimal_rewards_dict: {query_id: max_reward}
            - reward_lookup: {(query_id, model_id): actual_reward}
        Returns ({}, {}) if input is invalid or reward calculation fails.
    """
    logger.info(f"Preparing reward structures with lambda = {lambda_weight}...")

    required_cols = ["query_id", "model_id", "accuracy_norm", "energy_per_token_norm"]
    if not all(col in results_norm_df.columns for col in required_cols):
        logger.error(
            f"Input DataFrame missing required columns for reward calculation: {required_cols}"
        )
        return {}, {}

    if results_norm_df.empty:
        logger.error("Input DataFrame is empty, cannot prepare reward structures.")
        return {}, {}

    df_copy = results_norm_df.copy()

    try:
        df_copy["reward"] = df_copy.apply(
            lambda row: calculate_reward(
                row["accuracy_norm"], row["energy_per_token_norm"], lambda_weight
            ),
            axis=1,
        )
    except Exception as e:
        logger.error(f"Error calculating reward column: {e}", exc_info=True)
        return {}, {}
    if df_copy["reward"].isnull().all():
        logger.error("Reward calculation resulted in all NaN values.")
        return {}, {}
    try:
        optimal_rewards = df_copy.loc[df_copy.groupby("query_id")["reward"].idxmax()]
        optimal_rewards_dict = optimal_rewards.set_index("query_id")["reward"].to_dict()
    except Exception as e:
        logger.error(f"Error calculating optimal rewards per query: {e}", exc_info=True)
        return {}, {}
    try:
        reward_lookup = df_copy.set_index(["query_id", "model_id"])["reward"].to_dict()
    except Exception as e:
        logger.error(f"Error creating reward lookup dictionary: {e}", exc_info=True)
        return optimal_rewards_dict, {}

    logger.info(f"Calculated optimal rewards for {len(optimal_rewards_dict)} queries.")
    logger.info(f"Created reward lookup for {len(reward_lookup)} (query, model) pairs.")

    return optimal_rewards_dict, reward_lookup
