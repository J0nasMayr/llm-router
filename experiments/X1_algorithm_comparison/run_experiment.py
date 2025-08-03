"""
Algorithm Bakeoff Experiment (A5)

This experiment compares the performance of different multi-armed bandit algorithms
for contextual LLM routing. It evaluates epsilon-greedy variants, LinUCB, and
Thompson Sampling against several baselines including random selection and fixed
model choices.

The experiment measures:
- Cumulative regret over time
- Final performance metrics (accuracy vs energy trade-offs)
- Statistical significance of differences between algorithms
"""

# ============================================================================
# IMPORTS
# ============================================================================

import itertools
import logging
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from .plotting import generate_pareto_plot_wh

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from experiments.X2_feature_ablation.utils import create_bandit_instance

# Local imports
from experiments.shared import (
    analysis_utils,
    baseline_selectors,
    config_loader,
    data_loader,
    results_handler,
)
from src.services.feature_service import FeatureService

from .plotting import (
    ALGO_DISPLAY_NAMES,
    generate_pareto_plot,
    generate_combined_regret_plot,
    generate_cumulative_regret_plot,
    generate_grouped_norm_metric_bar_plot,
    generate_moving_average_regret_plot,
)

EXPERIMENT_NAME = "X1_algorithm_comparison"
MAX_QUERIES_DEBUG = None


def main():
    """
    Main entry point for the algorithm bakeoff experiment.

    Runs multiple bandit algorithms with the same configuration and compares
    their performance on the LLM routing task.
    """

    # ========================================================================
    # EXPERIMENT SETUP
    # ========================================================================

    exp_dirs = config_loader.setup_experiment_dirs(EXPERIMENT_NAME)
    configs = config_loader.load_config(
        "datasets",
        "models",
        "experiments",
        "feature_extraction",
        "baselines",
        config_dir="experiments/config",
    )
    defaults = configs["experiments"]["defaults"]
    try:
        experiment_config = configs["experiments"][EXPERIMENT_NAME]
        plot_config = experiment_config.get("plotting", {})
    except KeyError as e:
        logger.error(f"Missing required configuration section in experiments.yaml: {e}")
        return
    try:
        algorithms_to_run = experiment_config["algorithms"]
        feature_names = experiment_config["features"]
        n_runs = experiment_config["n_runs"]
        base_seed = experiment_config["random_seed"]
        lambda_weight = experiment_config["lambda_weight"]
        dataset_names = defaults.get("datasets", ["all"])
        samples_per_dataset = defaults.get("samples_per_dataset", 500)
    except KeyError as e:
        logger.error(
            f"Missing required configuration key directly in the '{EXPERIMENT_NAME}' section of experiments.yaml: {e}"
        )
        return

    logger.info(
        f"Running {EXPERIMENT_NAME} with Features: {feature_names}, Runs per config: {n_runs}, Lambda: {lambda_weight}, Seed: {base_seed}"
    )
    try:
        feature_config = configs["feature_extraction"]
    except KeyError as e:
        logger.error(
            f"Missing required configuration section 'feature_extraction' in config files: {e}"
        )
        return

    conn = data_loader.connect_db()
    model_specs_df = data_loader.get_model_specs(conn)
    all_model_ids = model_specs_df["model_id"].tolist()
    actual_samples = samples_per_dataset

    if MAX_QUERIES_DEBUG:
        num_datasets = (
            len(dataset_names)
            if isinstance(dataset_names, list) and "all" not in dataset_names
            else 5
        )
        actual_samples = max(1, MAX_QUERIES_DEBUG // num_datasets)
    eval_df = data_loader.load_evaluation_dataset(
        conn, dataset_names, actual_samples, base_seed
    )

    if MAX_QUERIES_DEBUG and len(eval_df) > MAX_QUERIES_DEBUG:
        eval_df = eval_df.sample(n=MAX_QUERIES_DEBUG, random_state=base_seed)

    if eval_df.empty or "query_id" not in eval_df.columns:
        logger.error(
            f"Failed to load evaluation data or 'query_id' column missing for datasets: {dataset_names}. Check dataset names and database content."
        )
        if conn:
            conn.close()
        return
    query_ids = eval_df["query_id"].unique().tolist()

    if not query_ids:
        logger.error(
            f"No query IDs found in the loaded data for datasets: {dataset_names}."
        )
        if conn:
            conn.close()
        return
    _, _, raw_results_data_df = data_loader.check_data_completeness(
        conn, query_ids, all_model_ids
    )

    if raw_results_data_df is None or raw_results_data_df.empty:
        logger.error("Failed to load raw results data. Exiting.")
        if conn:
            conn.close()
        return
    raw_results_data_df = raw_results_data_df[
        raw_results_data_df["query_id"].isin(query_ids)
    ]
    logger.debug("Calculating Raw Energy Per Token...")
    raw_results_data_df["total_tokens"] = (
        raw_results_data_df["input_tokens"] + raw_results_data_df["output_tokens"]
    )
    raw_results_data_df["raw_energy_per_token"] = raw_results_data_df.apply(
        lambda row: (
            row["energy_consumption"] / row["total_tokens"]
            if row["total_tokens"] > 0
            else 0
        ),
        axis=1,
    )
    logger.debug("Finished calculating Raw Energy Per Token.")
    norm_df = analysis_utils.normalize_metrics(raw_results_data_df.copy())
    if norm_df.empty:
        logger.error("Normalization failed. Exiting.")
        if conn:
            conn.close()
        return
    optimal_rewards_dict, reward_lkp = analysis_utils.prepare_reward_structures(
        norm_df, lambda_weight
    )
    query_meta = eval_df.set_index("query_id")[["text", "dataset"]].to_dict("index")
    if conn:
        conn.close()
    logger.debug("Data loading and normalization complete.")
    logger.debug("Computing features...")
    logger.debug(f"Preparing features: {feature_names}...")
    try:
        feature_service = FeatureService(feature_config)
    except Exception as e:
        logger.error(
            f"Failed to initialize FeatureService: {e}. Cannot compute contexts.",
            exc_info=True,
        )
        return

    indices, idx = [], 0
    n_task = (
        len(feature_service.task_types)
        if "task" in feature_names or "dataset_index" in feature_names
        else 0
    )
    n_cluster = (
        feature_service.num_clusters if "semantic_cluster" in feature_names else 0
    )
    n_complex = (
        feature_service.num_complexity_bins if "complexity" in feature_names else 0
    )

    if "task" in feature_names or "dataset_index" in feature_names:
        indices.extend(range(idx, idx + n_task))
    idx += len(feature_service.task_types)
    if "semantic_cluster" in feature_names:
        indices.extend(range(idx, idx + n_cluster))
    idx += feature_service.num_clusters
    if "complexity" in feature_names:
        indices.extend(range(idx, idx + n_complex))
    if not indices:
        eff_dim = 1
    else:
        eff_dim = len(indices) + 1

    logger.debug(f"Effective context dimension for features {feature_names}: {eff_dim}")
    features_list = [
        {
            "query_id": qid,
            **feature_service.extract_features(
                query_meta[qid]["text"], {"dataset": query_meta[qid]["dataset"]}
            ),
        }
        for qid in tqdm(query_ids, desc="Extracting Features")
    ]
    features_df = pd.DataFrame(features_list).set_index("query_id")
    base_contexts = features_df["context_vector"].to_dict()
    total_dim_base = len(next(iter(base_contexts.values())))
    fs_contexts = {}
    for qid in query_ids:
        if eff_dim == 1:
            fs_contexts[qid] = np.array([1.0], dtype=np.float32)
        else:
            base_vec = base_contexts.get(qid)
            fallback = np.zeros(eff_dim, dtype=np.float32)
            fallback[0] = 1.0
            if base_vec is None or len(base_vec) != total_dim_base:
                fs_contexts[qid] = fallback
            else:
                fs_contexts[qid] = np.concatenate(
                    ([1.0], np.array(base_vec)[indices])
                ).astype(np.float32)
    logger.debug("Context vectors computed.")
    if fs_contexts is None or eff_dim is None:
        logger.error(
            "Context vectors (fs_contexts) or eff_dim could not be prepared. Exiting."
        )
        return
    all_detailed_run_results = []
    logger.debug(f"[A5 Setup] Initial len(query_ids): {len(query_ids)}")
    logger.debug(f"[A5 Setup] Initial query_ids sample (first 10): {query_ids[:10]}")
    logger.debug(f"[A5 Setup] Initial query_ids sample (last 10): {query_ids[-10:]}")
    query_sequence = query_ids.copy()
    baselines_to_run = {}
    try:
        baseline_config = configs["baselines"]
        logger.info(f"Loaded baselines config: {baseline_config}")

        required_keys = ["largest_model_id", "smallest_model_id", "accuracy_model_id"]
        if not all(key in baseline_config for key in required_keys):
            missing_keys = [key for key in required_keys if key not in baseline_config]
            raise KeyError(f"Missing required keys in baselines config: {missing_keys}")
        baselines_to_run = {
            "random": baseline_selectors.RandomModelSelector(
                all_model_ids, seed=base_seed
            ),
            "largest": baseline_selectors.LargestModelSelector(
                baseline_config["largest_model_id"]
            ),
            "smallest": baseline_selectors.SmallestModelSelector(
                baseline_config["smallest_model_id"]
            ),
            "accuracy": baseline_selectors.AccuracyOptimizedSelector(
                baseline_config["accuracy_model_id"]
            ),
        }
        logger.debug(
            f"Initialized baselines for simulation: {list(baselines_to_run.keys())}"
        )
    except KeyError as e:
        logger.error(
            f"Could not initialize baselines: Missing key {e} in baselines config. Skipping baseline simulation."
        )
        baselines_to_run = {}
    except Exception as e:
        logger.error(f"Error initializing baselines: {e}", exc_info=True)
        baselines_to_run = {}

    all_strategies = {}
    for algo_name, fixed_params in algorithms_to_run.items():
        all_strategies[algo_name] = {"type": "bandit", "params": fixed_params}
    for baseline_name, selector_instance in baselines_to_run.items():
        if baseline_name in all_strategies:
            logger.warning(
                f"Baseline name '{baseline_name}' clashes with A5 algorithm name. A5 algorithm config will be used."
            )
        else:
            all_strategies[baseline_name] = {
                "type": "baseline",
                "instance": selector_instance,
            }
    logger.debug(f"Total strategies to simulate: {list(all_strategies.keys())}")
    for strategy_name, strategy_config in all_strategies.items():
        logger.debug(
            f"===== Running Strategy: {strategy_name} ({strategy_config['type']}) ===="
        )
        current_reg_lambda = None
        if strategy_config["type"] == "bandit":
            fixed_params = strategy_config["params"]
            current_reg_lambda = fixed_params.get(
                "lambda_", fixed_params.get("regularization")
            )
            param_str = ", ".join([f"{k}={v}" for k, v in fixed_params.items()])
            logger.info(
                f"--- Bandit Params: {{{param_str}}} --- Lambda: {current_reg_lambda}"
            )

        for r in range(n_runs):
            seed = base_seed + r
            random.seed(seed)
            np.random.seed(seed)
            algo_instance = None
            if strategy_config["type"] == "bandit":
                fixed_params = strategy_config["params"]
                logger.info(
                    f"[A5 Run {r+1}] Instantiating {strategy_name} with params: {fixed_params} | Reg Lambda: {current_reg_lambda}"
                )
                algo_instance = create_bandit_instance(
                    algo_name=strategy_name,
                    algo_params=fixed_params,
                    model_ids=all_model_ids,
                    context_dimension=eff_dim,
                    reg_lambda=current_reg_lambda,
                    seed=seed,
                )
                if algo_instance is None:
                    logger.warning(
                        f"Failed to instantiate bandit '{strategy_name}' for run {r+1}. Skipping run."
                    )
                    continue
                if hasattr(algo_instance, "reset"):
                    algo_instance.reset()
            elif strategy_config["type"] == "baseline":
                algo_instance = strategy_config["instance"]
                if strategy_name == "random" and hasattr(algo_instance, "seed"):
                    algo_instance.seed(seed)
            else:
                logger.error(
                    f"Unknown strategy type '{strategy_config['type']}' for {strategy_name}. Skipping run."
                )
                continue

            random.shuffle(query_sequence)
            if r == 0:
                logger.info(
                    f"[A5 Run {r+1}] Initial query sequence sample (first 20): {query_sequence[:20]}"
                )

            for qid_idx, qid in enumerate(
                tqdm(
                    query_sequence,
                    desc=f"Strategy '{strategy_name}' Run {r+1}/{n_runs}",
                    leave=False,
                )
            ):
                context = fs_contexts.get(qid, np.zeros(eff_dim, dtype=np.float32))
                select_start_time = time.time()
                chosen_model = algo_instance.select_model(context=context)
                select_end_time = time.time()
                selection_duration = select_end_time - select_start_time

                reward = reward_lkp.get((qid, chosen_model), 0.0)
                if strategy_config["type"] == "bandit":
                    algo_instance.update(chosen_model, reward, context=context)

                optimal_reward = optimal_rewards_dict.get(qid, 0.0)
                step_regret = optimal_reward - reward

                result_row = {
                    "run_id": seed,
                    "query_index": qid_idx,
                    "query_id": qid,
                    "chosen_model": chosen_model,
                    "reward": reward,
                    "step_regret": step_regret,
                    "algorithm": strategy_name,
                }
                all_detailed_run_results.append(result_row)

    if not all_detailed_run_results:
        logger.error(
            "No detailed results were collected. Exiting before analysis and plotting."
        )
        return
    detailed_results_df = pd.DataFrame(all_detailed_run_results)
    logger.info(
        f"Simulation complete. Collected {len(detailed_results_df)} total steps across all runs."
    )
    detailed_results_filename = (
        exp_dirs["results"] / f"{EXPERIMENT_NAME}_detailed_results.csv"
    )
    try:
        detailed_results_df.to_csv(detailed_results_filename, index=False)
        logger.info(f"Saved detailed run results to {detailed_results_filename}")
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}")

    logger.info(
        "Performing statistical significance testing on final cumulative regret..."
    )
    if "cumulative_regret" not in detailed_results_df.columns:
        if "step_regret" not in detailed_results_df.columns:
            logger.error(
                "Cannot calculate cumulative regret: 'step_regret' column missing."
            )
        else:
            logger.info("Calculating cumulative regret for significance testing...")
            detailed_results_df = detailed_results_df.sort_values(
                by=["run_id", "algorithm", "query_index"]
            )
            detailed_results_df["cumulative_regret"] = detailed_results_df.groupby(
                ["run_id", "algorithm"]
            )["step_regret"].cumsum()

    if "cumulative_regret" in detailed_results_df.columns:
        final_step_df = detailed_results_df.loc[
            detailed_results_df.groupby(["run_id", "algorithm"])["query_index"].idxmax()
        ]
        final_regret_data = final_step_df[
            ["algorithm", "run_id", "cumulative_regret"]
        ].copy()

        logger.info("--- Inspecting final_regret_data before Mann-Whitney U Test ---")
        logger.info(f"Shape: {final_regret_data.shape}")
        logger.info(f"Columns: {final_regret_data.columns.tolist()}")
        logger.info(f"NaN Check:\n{final_regret_data.isnull().sum().to_string()}")
        logger.info(
            f"Value Counts per Algorithm:\n{final_regret_data['algorithm'].value_counts().to_string()}"
        )
        try:
            logger.info(
                f"Descriptive Stats for final cumulative_regret per algorithm:\n{final_regret_data.groupby('algorithm')['cumulative_regret'].describe().to_string()}"
            )
        except Exception as e_desc:
            logger.error(
                f"Could not compute descriptive stats for final regret: {e_desc}"
            )
            logger.info(
                f"Sample final_regret_data head:\n{final_regret_data.head().to_string()}"
            )
        logger.info("-----------------------------------------------------------------")

        algorithms = sorted(final_regret_data["algorithm"].unique())
        results_stats = []
        p_values_raw = []
        pairs = list(itertools.combinations(algorithms, 2))

        logger.info(f"Comparing algorithms: {algorithms}")

        for algo1, algo2 in pairs:
            data1 = final_regret_data.loc[
                final_regret_data["algorithm"] == algo1, "cumulative_regret"
            ]
            data2 = final_regret_data.loc[
                final_regret_data["algorithm"] == algo2, "cumulative_regret"
            ]
            if data1.empty or data2.empty or len(data1) < 1 or len(data2) < 1:
                logger.warning(
                    f"Skipping comparison between {algo1} and {algo2} due to insufficient data (n1={len(data1)}, n2={len(data2)}). Assigning p-value=NaN."
                )
                p_value = np.nan
            else:
                if (
                    data1.nunique() <= 1
                    and data2.nunique() <= 1
                    and data1.iloc[0] == data2.iloc[0]
                ):
                    logger.warning(
                        f"Data for {algo1} and {algo2} are identical constants. Assigning p-value=1.0"
                    )
                    p_value = 1.0
                    stat = np.nan
                else:
                    try:
                        stat, p_value = mannwhitneyu(
                            data1, data2, alternative="two-sided", nan_policy="omit"
                        )
                    except ValueError as e:
                        logger.warning(
                            f"Mann-Whitney U test failed for {algo1} vs {algo2}: {e}. Assigning p-value=1.0"
                        )
                        p_value = 1.0

            p_values_raw.append(p_value)
            display_algo1 = ALGO_DISPLAY_NAMES.get(algo1, algo1)
            display_algo2 = ALGO_DISPLAY_NAMES.get(algo2, algo2)
            results_stats.append(
                {"pair": f"{display_algo1} vs {display_algo2}", "p_value_raw": p_value}
            )
        valid_indices = [i for i, p in enumerate(p_values_raw) if not np.isnan(p)]

        if not valid_indices:
            logger.warning(
                "No valid p-values obtained for multiple comparison correction."
            )
            corrected_p_values = []
            reject = []
            valid_pairs_tuples = []
        else:
            valid_p_values = [p_values_raw[i] for i in valid_indices]
            valid_pairs_tuples = [pairs[i] for i in valid_indices]
            reject, corrected_p_values, _, _ = multipletests(
                valid_p_values, alpha=0.05, method="fdr_bh"
            )

        corrected_map = {
            pair_tuple: (corr_p, rej)
            for pair_tuple, corr_p, rej in zip(
                valid_pairs_tuples, corrected_p_values, reject
            )
        }
        final_stats_output = []
        for i, result in enumerate(results_stats):
            pair_tuple = pairs[i]
            if np.isnan(result["p_value_raw"]):
                result["p_value_corrected_bh"] = np.nan
                result["significant_bh"] = False
            elif pair_tuple in corrected_map:
                corr_p, rej = corrected_map[pair_tuple]
                result["p_value_corrected_bh"] = corr_p
                result["significant_bh"] = rej
            else:
                result["p_value_corrected_bh"] = np.nan
                result["significant_bh"] = False
            final_stats_output.append(result)
        logger.info("--- Statistical Significance Test Results ---")
        logger.info(
            "Pairwise comparisons of final cumulative regret using Mann-Whitney U test."
        )
        logger.info(
            "P-values adjusted using Benjamini-Hochberg method (controlling FDR at alpha=0.05)."
        )

        if not final_stats_output:
            logger.warning("No significance test results to report.")
        else:
            stats_df = pd.DataFrame(final_stats_output)
            stats_df = stats_df.sort_values(
                by=["p_value_corrected_bh", "p_value_raw"]
            ).reset_index(drop=True)
            stats_df["p_value_raw"] = stats_df["p_value_raw"].map("{:.3g}".format)
            stats_df["p_value_corrected_bh"] = stats_df["p_value_corrected_bh"].map(
                "{:.3g}".format
            )

            logger.info(f"\n{stats_df.to_string()}")
            stats_filename = (
                exp_dirs["results"] / f"{EXPERIMENT_NAME}_significance_tests.csv"
            )
            try:
                stats_df.to_csv(stats_filename, index=False)
                logger.info(f"Significance test results saved to {stats_filename}")
            except Exception as e:
                logger.error(
                    f"Failed to save significance test results to {stats_filename}: {e}"
                )

    else:
        logger.warning(
            "Skipping statistical significance testing because 'cumulative_regret' could not be determined."
        )

    logger.info("Starting analysis and plotting...")

    a5_detailed_sim_df = pd.DataFrame(detailed_results_df)
    a5_detailed_sim_df = a5_detailed_sim_df.sort_values(
        by=["run_id", "algorithm", "query_index"]
    )
    a5_detailed_sim_df["cumulative_regret"] = a5_detailed_sim_df.groupby(
        ["run_id", "algorithm"]
    )["step_regret"].cumsum()
    a5_detailed_sim_df["cumulative_reward"] = a5_detailed_sim_df.groupby(
        ["run_id", "algorithm"]
    )["reward"].cumsum()
    final_step_indices = a5_detailed_sim_df.groupby(["run_id", "algorithm"])[
        "query_index"
    ].idxmax()
    final_step_df = a5_detailed_sim_df.loc[final_step_indices].copy()
    agg_grouping_cols_regret = ["algorithm"]

    regret_summary = (
        final_step_df.groupby(agg_grouping_cols_regret)["cumulative_regret"]
        .agg(mean_final_regret="mean", std_final_regret="std")
        .reset_index()
    )
    logger.info("Calculating Mean Normalized Accuracy, Energy, and Reward metrics...")
    metrics_to_average = ["accuracy_norm", "energy_per_token_norm"]
    if "latency_norm" in norm_df.columns:
        metrics_to_average.append("latency_norm")
    reward_col_in_detailed = ["reward"]

    merged_norm_df = pd.merge(
        detailed_results_df[
            ["run_id", "algorithm", "query_id", "chosen_model"] + reward_col_in_detailed
        ],
        norm_df[["query_id", "model_id"] + metrics_to_average],
        left_on=["query_id", "chosen_model"],
        right_on=["query_id", "model_id"],
        how="left",
    )
    logger.info(
        "--- Inspecting merged_metrics_df BEFORE fillna/run means calculation ---"
    )
    logger.info(f"Shape: {merged_norm_df.shape}")
    logger.info(f"Columns: {merged_norm_df.columns.tolist()}")
    if "raw_energy_per_token" in merged_norm_df.columns:
        logger.info(
            f"'raw_energy_per_token' dtype: {merged_norm_df['raw_energy_per_token'].dtype}"
        )
        logger.info(
            f"'raw_energy_per_token' NaNs: {merged_norm_df['raw_energy_per_token'].isnull().sum()}"
        )
        logger.info(
            f"'raw_energy_per_token' non-numeric check: {pd.to_numeric(merged_norm_df['raw_energy_per_token'], errors='coerce').isnull().sum()} potential non-numeric values (incl NaNs)"
        )
        logger.info(
            f"Head with 'raw_energy_per_token':\n{merged_norm_df[['run_id', 'algorithm', 'query_id', 'raw_energy_per_token']].head().to_string()}"
        )
    else:
        logger.warning("'raw_energy_per_token' column NOT FOUND in merged_metrics_df!")
    logger.info(
        "-------------------------------------------------------------------------"
    )

    all_metrics_to_average_per_run = metrics_to_average + reward_col_in_detailed
    run_metrics_means = (
        merged_norm_df.groupby(["run_id", "algorithm"])[all_metrics_to_average_per_run]
        .mean()
        .reset_index()
    )

    logger.info("--- Inspecting run_metrics_means BEFORE final aggregation ---")
    logger.info(f"Shape: {run_metrics_means.shape}")
    logger.info(f"Columns: {run_metrics_means.columns.tolist()}")

    if "raw_energy_per_token" in run_metrics_means.columns:
        logger.info(
            f"'raw_energy_per_token' dtype: {run_metrics_means['raw_energy_per_token'].dtype}"
        )
        logger.info(
            f"'raw_energy_per_token' NaNs: {run_metrics_means['raw_energy_per_token'].isnull().sum()}"
        )
        logger.info(
            f"Head with 'raw_energy_per_token':\n{run_metrics_means[['run_id', 'algorithm', 'raw_energy_per_token']].head().to_string()}"
        )
    else:
        logger.warning("'raw_energy_per_token' column NOT FOUND in run_metrics_means!")
    logger.info("---------------------------------------------------------------")

    final_agg_dict = {}
    if "run_id" in run_metrics_means.columns:
        final_agg_dict["count"] = ("run_id", "count")
    else:
        logger.warning(
            "'run_id' not found in run_metrics_means, cannot calculate count for error bars."
        )
    for col in all_metrics_to_average_per_run:
        final_agg_dict[f"mean_{col}"] = (col, "mean")
        final_agg_dict[f"std_{col}"] = (col, "std")
    logger.info("=== DIAGNOSTIC: final_agg_dict ===")
    logger.info(f"Keys: {list(final_agg_dict.keys())}")
    logger.info("===========================")

    aggregated_means_stddevs = (
        run_metrics_means.groupby("algorithm").agg(**final_agg_dict).reset_index()
    )
    logger.info("--- Inspecting aggregated_means_stddevs columns ---")
    logger.info(f"{aggregated_means_stddevs.columns.tolist()}")
    if "mean_raw_energy_per_token" in aggregated_means_stddevs.columns:
        logger.info(
            f"mean_raw_energy_per_token sample data: {aggregated_means_stddevs['mean_raw_energy_per_token'].head(3).tolist()}"
        )
    else:
        logger.error(
            "CRITICAL ERROR: mean_raw_energy_per_token missing from aggregated_means_stddevs!"
        )
    logger.info("--------------------------------------------------")

    summary_stats_df = pd.merge(
        regret_summary, aggregated_means_stddevs, on="algorithm", how="outer"
    )
    logger.info("--- Inspecting summary_stats_df columns (after merge) ---")
    logger.info(f"{summary_stats_df.columns.tolist()}")
    logger.info("-------------------------------------------------------")

    logger.info("Calculating final reward summary...")
    reward_summary = (
        final_step_df.groupby(agg_grouping_cols_regret)["cumulative_reward"]
        .agg(mean_final_reward="mean", std_final_reward="std")
        .reset_index()
    )

    summary_stats_df = pd.merge(
        summary_stats_df, reward_summary, on="algorithm", how="outer"
    )
    logger.info(
        f"Columns in summary_stats_df BEFORE rename: {summary_stats_df.columns.tolist()}"
    )
    rename_mapping = {
        "algorithm": "Algorithm",
        "mean_accuracy_norm": "Mean Normalized Accuracy",
        "mean_energy_per_token_norm": "Mean Normalized Energy",
        "mean_reward": "Mean Reward",
        "std_accuracy_norm": "Std Norm Accuracy",
        "std_reward": "Std Reward",
        "count": "Count",
        "mean_final_reward": "Mean Final Reward",
        "std_final_reward": "Std Final Reward",
        "mean_final_regret": "Mean Final Regret",
        "std_final_regret": "Std Final Regret",
    }
    a5_summary_df = summary_stats_df.copy()
    a5_summary_df.rename(
        columns={
            k: v for k, v in rename_mapping.items() if k in summary_stats_df.columns
        },
        inplace=True,
    )
    logger.info(
        f"Columns in a5_summary_df AFTER rename: {a5_summary_df.columns.tolist()}"
    )
    logger.info("=== FINAL DATAFRAME DIAGNOSTIC ===")
    logger.info(f"a5_summary_df shape: {a5_summary_df.shape}")
    logger.info(f"All columns: {a5_summary_df.columns.tolist()}")
    logger.info(
        f"Required columns for bar plot: {['Algorithm', 'Mean Normalized Accuracy', 'Mean Raw Energy per Token']}"
    )
    has_req_cols = all(
        col in a5_summary_df.columns
        for col in [
            "Algorithm",
            "Mean Normalized Accuracy",
            "Mean Raw Energy per Token",
        ]
    )
    logger.info(f"Has all required columns? {has_req_cols}")

    if not has_req_cols and "mean_raw_energy_per_token" in summary_stats_df.columns:
        logger.info(
            "EMERGENCY FIX: Adding missing 'Mean Raw Energy per Token' column directly"
        )
        a5_summary_df["Mean Raw Energy per Token"] = summary_stats_df[
            "mean_raw_energy_per_token"
        ]
        logger.info(f"Columns after fix: {a5_summary_df.columns.tolist()}")
    logger.info("===================================")

    logger.info("--- Final Mean Cumulative Regret & Reward per Strategy --- ")
    summary_cols_to_print = ["Algorithm"]
    if "Mean Final Regret" in a5_summary_df.columns:
        summary_cols_to_print.append("Mean Final Regret")
    if "Mean Final Reward" in a5_summary_df.columns:
        summary_cols_to_print.append("Mean Final Reward")

    if len(summary_cols_to_print) > 1:
        sort_col = (
            "Mean Final Regret"
            if "Mean Final Regret" in summary_cols_to_print
            else "Mean Final Reward"
        )
        ascending_sort = True if sort_col == "Mean Final Regret" else False
        summary_print_df = a5_summary_df[summary_cols_to_print].sort_values(
            by=sort_col, ascending=ascending_sort
        )
        logger.info("\n" + summary_print_df.to_string(index=False))
    else:
        logger.warning(
            "Could not print final regret/reward summary. Relevant columns missing."
        )
    logger.info("--------------------------------------------------")

    logger.info("Calculating model choices for LAST run...")
    try:
        last_run_id = detailed_results_df["run_id"].max()
        last_run_df = detailed_results_df[detailed_results_df["run_id"] == last_run_id]
        model_counts_last_run = (
            last_run_df.groupby(["algorithm", "chosen_model"])
            .size()
            .unstack(fill_value=0)
        )

        logger.info(
            f"--- Model Choices per Strategy (Actual Counts in LAST Run: {last_run_id}) ---"
        )
        for algo in sorted(model_counts_last_run.index):
            logger.info(f"Algorithm: {algo}")
            choices = model_counts_last_run.loc[algo].sort_values(ascending=False)
            choices = choices[choices > 0]
            if not choices.empty:
                for model, count in choices.items():
                    logger.info(f"  - {model}: {count}")
            else:
                logger.info("  (No models chosen or counts were zero in last run)")
        logger.info("-------------------------------------------------------------")

    except Exception as e:
        logger.error(
            f"Failed to calculate or log last run model choices: {e}", exc_info=True
        )

    results_handler.save_results(
        df=a5_summary_df,
        results_dir=exp_dirs["results"],
        filename_prefix=f"{EXPERIMENT_NAME}_summary_stats",
        format="csv",
    )
    logger.info("Generating plots...")

    models_df_plot = None
    try:
        models_file = project_root / "experiments" / "config" / "models.csv"
        models_df_plot = pd.read_csv(models_file)
        if (
            "model_id" not in models_df_plot.columns
            or "parameter_count" not in models_df_plot.columns
        ):
            logger.warning(
                f"Loaded models file {models_file}, but missing 'model_id' or 'parameter_count'. Size sorting/annotation may fail."
            )
            models_df_plot = None
    except FileNotFoundError:
        logger.warning(
            f"Models file not found at {models_file}. Model sizes unavailable for sorting/annotation."
        )
    except Exception as e:
        logger.error(f"Error loading models file {models_file}: {e}")

    

    raw_df_for_plotting = locals().get("raw_results_data_df")
    if raw_df_for_plotting is None or raw_df_for_plotting.empty:
        logger.error(
            "Raw results data (raw_results_data_df) not available for plotting total energy. Skipping relevant plots."
        )
    logger.info("Calling A5 Pareto plot function (Normalized Energy)...")
    try:
        generate_pareto_plot(
            a5_summary_df=a5_summary_df,
            all_norm_results_df=norm_df,
            exp_dirs=exp_dirs,
            models_df=models_df_plot,
        )
    except Exception as e:
        logger.error(f"Error generating A5 Pareto plot: {e}", exc_info=True)
    logger.info("Calling A5 Pareto plot function (Total Energy Wh)...")
    if (
        raw_df_for_plotting is not None
        and not raw_df_for_plotting.empty
        and "a5_detailed_sim_df" in locals()
        and not a5_detailed_sim_df.empty
    ):
        try:
            generate_pareto_plot_wh(
            a5_summary_df=a5_summary_df,
            a5_detailed_results_df=a5_detailed_sim_df,
            raw_results_data_df=raw_df_for_plotting,
            norm_df=norm_df,
            exp_dirs=exp_dirs,
            models_df=models_df_plot,
        )
        except ImportError:
            logger.error(
                "generate_a5_pareto_plot_wh function not found or import failed."
            )
        except Exception as e:
            logger.error(
                f"Error generating A5 Pareto plot (Total Energy Wh): {e}", exc_info=True
            )
    else:
        logger.warning(
            "Skipping A5 Pareto plot (Total Energy Wh) due to missing raw or detailed results data."
        )
    logger.info("Preparing data for bar plot with conditional CIs...")
    std_cols_needed = ["Std Norm Accuracy", "std_energy_per_token_norm"]
    count_col_needed = "Count"  # Use renamed Count
    has_error_bar_data = False
    if count_col_needed in a5_summary_df.columns and all(
        c in a5_summary_df.columns for c in std_cols_needed
    ):
        has_error_bar_data = True
        logger.info(
            f"Found required columns for error bars ({std_cols_needed + [count_col_needed]}). Enabling CI plotting."
        )
    else:
        logger.warning(
            f"Missing columns required for error bars in a5_summary_df. Needed: {std_cols_needed + [count_col_needed]}. Available: {a5_summary_df.columns.tolist()}"
        )
        logger.warning("Disabling CI plotting for original bar plot.")

    required_bar_cols = [
        "Algorithm",
        "Mean Normalized Accuracy",
        "Mean Normalized Energy",
    ]
    if (
        isinstance(a5_summary_df, pd.DataFrame)
        and not a5_summary_df.empty
        and all(c in a5_summary_df.columns for c in required_bar_cols)
    ):
        logger.info(
            "Calling original grouped bar plot function (Norm Acc vs Norm Energy/Token)..."
        )
        try:
            generate_grouped_norm_metric_bar_plot(
                a5_summary_df=a5_summary_df,
                exp_dirs=exp_dirs,
                has_error_bar_data=has_error_bar_data,
            )
        except ImportError:
            logger.error(
                "generate_grouped_norm_metric_bar_plot function not found or import failed."
            )
        except Exception as e:
            logger.error(
                f"Error generating original grouped bar plot: {e}", exc_info=True
            )
    else:
        logger.warning(
            "Skipping original grouped bar plot as summary data is not available or missing required columns."
        )
    logger.info(
        "Calling new grouped bar plot function (Norm Acc vs Total Energy Wh)..."
    )
    if (
        raw_df_for_plotting is not None
        and not raw_df_for_plotting.empty
        and "a5_detailed_sim_df" in locals()
        and not a5_detailed_sim_df.empty
        and isinstance(a5_summary_df, pd.DataFrame)
        and not a5_summary_df.empty
    ):
        try:
            from .plotting import generate_grouped_bar_plot_total_energy

            generate_grouped_bar_plot_total_energy(
                a5_summary_df=a5_summary_df,
                a5_detailed_results_df=a5_detailed_sim_df,
                raw_results_data_df=raw_df_for_plotting,
                exp_dirs=exp_dirs,
            )
        except ImportError:
            logger.error(
                "generate_grouped_bar_plot_total_energy function not found or import failed."
            )
        except Exception as e:
            logger.error(
                f"Error generating new grouped bar plot (Total Energy Wh): {e}",
                exc_info=True,
            )
    else:
        logger.warning(
            "Skipping new grouped bar plot (Total Energy Wh) due to missing raw, detailed, or summary results data."
        )
    logger.info("Calling cumulative regret plot function...")
    try:
        generate_cumulative_regret_plot(
            detailed_results_df=a5_detailed_sim_df, exp_dirs=exp_dirs
        )
    except ImportError:
        logger.error(
            "generate_cumulative_regret_plot function not found or import failed."
        )
    except Exception as e:
        logger.error(f"Error generating cumulative regret plot: {e}", exc_info=True)

    ma_window_size = 50
    logger.info(
        f"Calling moving average regret plot function (window={ma_window_size})..."
    )
    try:
        generate_moving_average_regret_plot(
            detailed_results_df=a5_detailed_sim_df,
            exp_dirs=exp_dirs,
            window_size=ma_window_size,
        )
    except ImportError:
        logger.error(
            "generate_moving_average_regret_plot function not found or import failed."
        )
    except Exception as e:
        logger.error(f"Error generating moving average regret plot: {e}", exc_info=True)
    logger.info("Calling combined regret plot function...")
    try:
        generate_combined_regret_plot(
            detailed_results_df=a5_detailed_sim_df,
            exp_dirs=exp_dirs,
            window_size=ma_window_size,
        )
    except ImportError:
        logger.error(
            "generate_combined_regret_plot function not found or import failed."
        )
    except Exception as e:
        logger.error(f"Error generating combined regret plot: {e}", exc_info=True)

    logger.info("Generating model choice timeline subplots (last run)...")
    bandit_algorithms_run = sorted(algorithms_to_run.keys())

    if bandit_algorithms_run:
        n_algos = len(bandit_algorithms_run)
        n_cols = 2
        n_rows = (n_algos + n_cols - 1) // n_cols

        try:
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(8 * n_cols, 5 * n_rows),
                sharex=True,
                sharey=True,
            )
            axes = axes.flatten()

            all_models_across_subplots = set()

            from .plotting import generate_model_choice_timeline

            for i, bandit_name in enumerate(bandit_algorithms_run):
                ax = axes[i]
                try:
                    logger.info(f"  Generating timeline subplot for: {bandit_name}")
                    models_in_plot = generate_model_choice_timeline(
                        detailed_results_df=a5_detailed_sim_df,
                        algo_name=bandit_name,
                        ax=ax,
                        all_model_ids=all_model_ids,
                    )
                    all_models_across_subplots.update(models_in_plot)

                    display_name = ALGO_DISPLAY_NAMES.get(bandit_name, bandit_name)
                    ax.set_title(display_name)

                    if i % n_cols != 0:
                        ax.set_ylabel("")
                    if i // n_cols != n_rows - 1:
                        ax.set_xlabel("")

                except Exception as e:
                    logger.error(
                        f"Error generating timeline subplot for {bandit_name}: {e}",
                        exc_info=True,
                    )
                    ax.text(
                        0.5,
                        0.5,
                        "Plot Error",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{bandit_name} (Error)")
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            fig.tight_layout()

            plot_filename_base = f"{EXPERIMENT_NAME}_model_choice_timeline_last_run"
            try:
                dpi = plot_config.get("dpi", 300)
                formats = plot_config.get("formats", ["png"])
                for fmt in formats:
                    plot_filename = f"{plot_filename_base}.{fmt}"
                    fig.savefig(
                        exp_dirs["plots"] / plot_filename, dpi=dpi, bbox_inches="tight"
                    )
                logger.info(
                    f"Saved combined model choice timeline plot to {exp_dirs['plots'] / plot_filename_base} (formats: {formats})"
                )
            except Exception as e:
                logger.error(
                    f"Failed to save combined timeline plot {plot_filename_base}: {e}"
                )
            plt.close(fig)

        except ImportError:
            logger.error(
                "generate_a5_model_choice_timeline not found in plotting module. Skipping timeline plots."
            )
        except Exception as e:
            logger.error(
                f"Error during timeline subplot generation: {e}", exc_info=True
            )
            if "fig" in locals():
                plt.close(fig)
    else:
        logger.info(
            "No bandit algorithms found in the run, skipping timeline subplots."
        )

    logger.info("--- Fixed Parameters Used per Algorithm in A5 Run ---")
    if (
        "a5_summary_df" in locals()
        and isinstance(a5_summary_df, pd.DataFrame)
        and "Algorithm" in a5_summary_df.columns
    ):
        summary_metrics_for_log = a5_summary_df.set_index("Algorithm")
        for algo_name, fixed_params in algorithms_to_run.items():

            if algo_name in summary_metrics_for_log.index:
                metrics = summary_metrics_for_log.loc[algo_name]
                regret_str = f"Mean Final Regret: {metrics.get('Mean Final Regret', 'N/A'):.2f}"  # Use .get for safety
                reward_str = (
                    f"Mean Final Reward: {metrics.get('Mean Final Reward', 'N/A'):.2f}"
                )
            else:
                regret_str = "(Metrics N/A)"
                reward_str = ""

            param_str = ", ".join([f"{k}={v}" for k, v in fixed_params.items()])
            logger.info(f"  Algorithm: {algo_name}")
            logger.info(f"    Params: {{{param_str}}}")
            logger.info(f"    Result: {regret_str} | {reward_str}")
    else:
        logger.warning(
            "Could not log hyperparameters used: a5_summary_df not available or missing 'Algorithm' column."
        )
        for algo_name, fixed_params in algorithms_to_run.items():
            param_str = ", ".join([f"{k}={v}" for k, v in fixed_params.items()])
            logger.info(f"  Algorithm: {algo_name}")
            logger.info(f"    Params (from config): {{{param_str}}}")
    logger.info("--------------------------------------------------")

    logger.info(f"Experiment {EXPERIMENT_NAME} finished.")
    logger.info("=" * 50)
    logger.info("Final Statistical Significance Test Results (from CSV)")
    logger.info("=" * 50)
    stats_filename = exp_dirs["results"] / f"{EXPERIMENT_NAME}_significance_tests.csv"
    try:
        if stats_filename.is_file():
            stats_df_final = pd.read_csv(stats_filename)

            if "significant_bh" in stats_df_final.columns:
                stats_df_final["significant_bh"] = stats_df_final["significant_bh"].map(
                    {True: "Yes", False: "No"}
                )
            logger.info(f"\n{stats_df_final.to_string(index=False)}\n")
        else:
            logger.warning(
                f"Significance test results file not found: {stats_filename}"
            )
    except Exception as e:
        logger.error(
            f"Error reading or printing significance test results file {stats_filename}: {e}"
        )


if __name__ == "__main__":
    main()
