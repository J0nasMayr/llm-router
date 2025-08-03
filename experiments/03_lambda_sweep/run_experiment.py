"""
Lambda Parameter Sweep Experiment (A6)

This experiment systematically varies the lambda parameter (accuracy vs energy
trade-off weight) to understand its impact on routing performance. It helps
identify optimal lambda values for different deployment scenarios.

Key measurements:
- Performance across lambda values [0.0, 1.0]
- Pareto frontier analysis
- Optimal lambda selection for different objectives
"""

# ============================================================================
# IMPORTS
# ============================================================================

import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from experiments.02_feature_ablation.utils import create_bandit_instance
from experiments.shared import (
    analysis_utils,
    baseline_selectors,
    config_loader,
    data_loader,
    results_handler,
)
from src.services.feature_service import FeatureService

from .plotting import (
    generate_lambda_sweep_accuracy_total_energy_boxplots_wh,
    generate_lambda_sweep_pareto_subplots_wh,
)

EXPERIMENT_NAME = "03_lambda_sweep"  # CHANGED for A6
MAX_QUERIES_DEBUG = None


def main():

    exp_dirs = config_loader.setup_experiment_dirs(EXPERIMENT_NAME)
    logger.info("Loading configurations...")
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
    except KeyError as e:
        logger.error(f"Missing required configuration section in experiments.yaml: {e}")
        return

    # Load plot configuration
    experiment_config.get("plotting", {})
    try:
        algorithms_to_run = experiment_config["algorithms"]
        feature_names = experiment_config["features"]
        n_runs = experiment_config["n_runs"]
        base_seed = experiment_config["random_seed"]
        lambda_values = experiment_config["lambda_values"]
        dataset_names = defaults.get("datasets", ["all"])
        samples_per_dataset = defaults.get("samples_per_dataset", 500)
    except KeyError as e:
        logger.error(
            f"Missing required configuration key directly in the '{EXPERIMENT_NAME}' section of experiments.yaml: {e}"
        )
        return
    logger.info(
        f"Running {EXPERIMENT_NAME} with Features: {feature_names}, Runs per config: {n_runs}, Lambdas: {lambda_values}, Seed: {base_seed}"
    )
    try:
        feature_config = configs["feature_extraction"]
    except KeyError as e:
        logger.error(
            f"Missing required configuration section 'feature_extraction' in config files: {e}"
        )
        return
    logger.info("Loading and preparing data...")
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
    logger.info(f"Loaded {len(raw_results_data_df)} raw result rows.")

    logger.info("Calculating Raw Energy Per Token...")
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
    logger.info("Finished calculating Raw Energy Per Token.")
    norm_df = analysis_utils.normalize_metrics(raw_results_data_df.copy())
    if norm_df.empty:
        logger.error("Normalization failed. Exiting.")
        if conn:
            conn.close()
        return
    raw_energy_lkp = raw_results_data_df.set_index(["query_id", "model_id"])[
        "raw_energy_per_token"
    ].to_dict()
    query_meta = eval_df.set_index("query_id")[["text", "dataset"]].to_dict("index")
    if conn:
        conn.close()
    logger.info("Data loading and normalization complete.")
    logger.info("Computing features...")
    logger.info(f"Preparing features: {feature_names}...")
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
    logger.info(f"Effective context dimension for features {feature_names}: {eff_dim}")

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

    logger.info("Context vectors computed.")
    if fs_contexts is None or eff_dim is None:
        logger.error(
            "Context vectors (fs_contexts) or eff_dim could not be prepared. Exiting."
        )
        return

    all_detailed_run_results = []
    query_sequence = query_ids.copy()
    baselines_to_run = {}

    try:
        baseline_config = configs["baselines"]
        baselines_to_run = {
            "largest": baseline_selectors.LargestModelSelector(
                baseline_config["largest_model_id"]
            ),
            "smallest": baseline_selectors.SmallestModelSelector(
                baseline_config["smallest_model_id"]
            ),
            "accuracy": baseline_selectors.AccuracyOptimizedSelector(
                baseline_config["accuracy_model_id"]
            ),
            "random": baseline_selectors.RandomModelSelector(
                all_model_ids, seed=base_seed
            ),
        }
        logger.info(
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
                f"Baseline name '{baseline_name}' clashes with bandit algorithm name. Bandit config will be used."
            )
        else:
            all_strategies[baseline_name] = {
                "type": "baseline",
                "instance": selector_instance,
            }
    logger.info(f"Total strategies to simulate: {list(all_strategies.keys())}")

    for lambda_weight in lambda_values:
        logger.info(f"===== Processing Lambda = {lambda_weight:.2f} =====")
        optimal_rewards_dict, reward_lkp = analysis_utils.prepare_reward_structures(
            norm_df, lambda_weight
        )
        for strategy_name, strategy_config in all_strategies.items():
            logger.info(
                f"--- Running Strategy: {strategy_name} (Lambda={lambda_weight:.2f}) ---"
            )
            current_reg_lambda = None
            if strategy_config["type"] == "bandit":
                fixed_params = strategy_config["params"]
                current_reg_lambda = fixed_params.get(
                    "lambda_", fixed_params.get("regularization")
                )

            for r in range(n_runs):
                seed = base_seed + r + int(lambda_weight * 1000)
                random.seed(seed)
                np.random.seed(seed)

                algo_instance = None
                if strategy_config["type"] == "bandit":
                    fixed_params = strategy_config["params"]
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
                            f"Failed to instantiate bandit '{strategy_name}' for run {r+1}, lambda {lambda_weight:.2f}. Skipping run."
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

                for qid_idx, qid in enumerate(
                    tqdm(
                        query_sequence,
                        desc=f"Lambda {lambda_weight:.2f}, Strat '{strategy_name}', Run {r+1}/{n_runs}",
                        leave=False,
                    )
                ):
                    context = fs_contexts.get(qid, np.zeros(eff_dim, dtype=np.float32))
                    chosen_model = algo_instance.select_model(context=context)
                    reward = reward_lkp.get((qid, chosen_model), 0.0)
                    raw_energy = raw_energy_lkp.get((qid, chosen_model), 0.0)

                    if strategy_config["type"] == "bandit":
                        algo_instance.update(chosen_model, reward, context=context)

                    optimal_reward = optimal_rewards_dict.get(qid, 0.0)
                    step_regret = optimal_reward - reward

                    result_row = {
                        "lambda_weight": lambda_weight,
                        "run_id": seed,
                        "query_index": qid_idx,
                        "query_id": qid,
                        "chosen_model": chosen_model,
                        "reward": reward,
                        "raw_energy_per_token": raw_energy,
                        "step_regret": step_regret,
                        "algorithm": strategy_name,
                        "lambda_value": lambda_weight,
                    }
                    all_detailed_run_results.append(result_row)
    if not all_detailed_run_results:
        logger.error("No simulation results generated.")
        return
    detailed_results_df = pd.DataFrame(all_detailed_run_results)
    logger.info("Calculating Total Energy (Wh) per run...")
    run_total_energy_wh = None
    if (
        raw_results_data_df is None
        or "energy_consumption" not in raw_results_data_df.columns
    ):
        logger.error(
            "Raw results data missing 'energy_consumption'. Cannot calculate total energy per run."
        )
    else:
        logger.debug("Merging detailed results with raw energy...")
        detailed_with_raw_energy = pd.merge(
            detailed_results_df[
                ["lambda_weight", "run_id", "algorithm", "query_id", "chosen_model"]
            ],
            raw_results_data_df[["query_id", "model_id", "energy_consumption"]],
            left_on=["query_id", "chosen_model"],
            right_on=["query_id", "model_id"],
            how="left",
        )
        logger.debug(
            f"Merged detailed_with_raw_energy shape: {detailed_with_raw_energy.shape}"
        )
        logger.debug("Grouping by run to sum energy...")
        total_energy_ws_per_run = (
            detailed_with_raw_energy.groupby(
                ["lambda_weight", "run_id", "algorithm"], dropna=False
            )["energy_consumption"]
            .sum(min_count=1)
            .reset_index()
        )

        total_energy_ws_per_run["total_energy_wh"] = (
            total_energy_ws_per_run["energy_consumption"] / 3600.0
        )
        run_total_energy_wh = total_energy_ws_per_run[
            ["lambda_weight", "run_id", "algorithm", "total_energy_wh"]
        ]
        logger.info(
            f"Calculated total energy (Wh) for {len(run_total_energy_wh)} run combinations."
        )
        if not run_total_energy_wh.empty:
            logger.debug(
                f"Sample of calculated per-run total energy (Wh):\n{run_total_energy_wh.head().to_string()}"
            )
            nan_check = run_total_energy_wh["total_energy_wh"].isnull().sum()
            if nan_check > 0:
                logger.warning(
                    f"Found {nan_check} NaN values in calculated 'total_energy_wh'."
                )

    logger.info("Calculating mean normalized metrics and reward per run...")
    metrics_to_average_norm = ["accuracy_norm", "energy_per_token_norm"]
    if "latency_norm" in norm_df.columns:
        metrics_to_average_norm.append("latency_norm")
    cols_from_detailed_for_mean = ["reward"]

    logger.debug("Merging detailed results with normalized metrics...")
    merged_norm_df = pd.merge(
        detailed_results_df[
            ["lambda_weight", "run_id", "algorithm", "query_id", "chosen_model"]
            + cols_from_detailed_for_mean
        ],
        norm_df[["query_id", "model_id"] + metrics_to_average_norm],
        left_on=["query_id", "chosen_model"],
        right_on=["query_id", "model_id"],
        how="left",
    )
    logger.debug(f"Merged detailed_with_normalized shape: {merged_norm_df.shape}")
    all_metrics_to_average_per_run = (
        metrics_to_average_norm + cols_from_detailed_for_mean
    )
    logger.debug(
        f"Grouping by run to calculate means for: {all_metrics_to_average_per_run}"
    )
    run_metrics_means = (
        merged_norm_df.groupby(["lambda_weight", "run_id", "algorithm"], dropna=False)[
            all_metrics_to_average_per_run
        ]
        .mean()
        .reset_index()
    )
    logger.info(
        f"Calculated per-run means for normalized metrics and reward for {len(run_metrics_means)} runs."
    )
    if not run_metrics_means.empty:
        logger.debug(
            f"Sample of calculated per-run means:\n{run_metrics_means.head().to_string()}"
        )
    if run_total_energy_wh is not None:
        logger.info(
            "Merging per-run total energy (Wh) with other per-run mean metrics..."
        )
        run_metrics_means = pd.merge(
            run_metrics_means,
            run_total_energy_wh,
            on=["lambda_weight", "run_id", "algorithm"],
            how="left",  # Use left merge to keep all runs
        )
        logger.info("Merge complete.")
        if "total_energy_wh" in run_metrics_means.columns:
            nan_check_after_merge = run_metrics_means["total_energy_wh"].isnull().sum()
            if nan_check_after_merge > 0:
                logger.warning(
                    f"Found {nan_check_after_merge} NaN values in 'total_energy_wh' after merging into run_metrics_means."
                )
        else:
            logger.error("'total_energy_wh' column missing after merge!")
    else:
        if "total_energy_wh" not in run_metrics_means.columns:
            run_metrics_means["total_energy_wh"] = np.nan
            logger.warning(
                "Total energy (Wh) calculation failed or was skipped, 'total_energy_wh' column added and filled with NaN."
            )
    logger.info("Aggregating metrics across runs (mean of means)...")
    final_agg_dict = {}
    grouping_cols = ["lambda_weight", "algorithm"]

    if "run_id" in run_metrics_means.columns:
        final_agg_dict["count"] = ("run_id", "nunique")
    else:
        logger.warning(
            "'run_id' not found in run_metrics_means, cannot calculate count for final summary."
        )
    metrics_for_final_agg = all_metrics_to_average_per_run
    if "total_energy_wh" in run_metrics_means.columns:
        if run_metrics_means["total_energy_wh"].notna().any():
            metrics_for_final_agg = metrics_for_final_agg + ["total_energy_wh"]
            logger.debug("Including 'total_energy_wh' in final aggregation.")
        else:
            logger.warning(
                "Skipping 'total_energy_wh' in final aggregation as it contains only NaN values."
            )

    for col in metrics_for_final_agg:
        if col in run_metrics_means.columns:
            final_agg_dict[f"mean_{col}"] = (col, "mean")
            final_agg_dict[f"std_{col}"] = (col, "std")
        else:
            logger.warning(
                f"Column '{col}' not found in run_metrics_means during final aggregation setup."
            )
    if not final_agg_dict or len(grouping_cols) == 0:
        logger.error(
            "No metrics or grouping columns available for final aggregation. Skipping summary stats creation."
        )
        a6_summary_df = pd.DataFrame()
    else:
        logger.debug(f"Performing final aggregation with rules: {final_agg_dict}")
        try:
            aggregated_means_stddevs = run_metrics_means.groupby(
                grouping_cols, dropna=False
            ).agg(**final_agg_dict)
            if isinstance(aggregated_means_stddevs.index, pd.MultiIndex):
                aggregated_means_stddevs = aggregated_means_stddevs.reset_index()
            a6_summary_df = aggregated_means_stddevs

            rename_mapping = {
                "algorithm": "Algorithm",
                "lambda_weight": "Lambda",
                "mean_accuracy_norm": "Mean Normalized Accuracy",
                "mean_energy_per_token_norm": "Mean Normalized Energy",
                "mean_reward": "Mean Reward",
                "mean_total_energy_wh": "Mean Total Energy (Wh)",
                "std_accuracy_norm": "Std Norm Accuracy",
                "std_energy_per_token_norm": "Std Norm Energy",
                "std_reward": "Std Reward",
                "std_total_energy_wh": "Std Total Energy (Wh)",
                "count": "Count",
            }
            actual_rename = {
                k: v for k, v in rename_mapping.items() if k in a6_summary_df.columns
            }
            if actual_rename:
                a6_summary_df.rename(columns=actual_rename, inplace=True)
            logger.info(
                f"Final aggregated summary created with shape {a6_summary_df.shape}."
            )
            logger.debug(f"Final summary columns: {a6_summary_df.columns.tolist()}")
            if not a6_summary_df.empty:
                logger.debug(
                    f"Sample of final summary:\n{a6_summary_df.head().to_string()}"
                )

        except Exception as agg_error:
            logger.error(f"Error during final aggregation: {agg_error}", exc_info=True)
            a6_summary_df = pd.DataFrame()
    logger.info("Saving results files...")
    results_handler.save_results(
        df=detailed_results_df,
        results_dir=exp_dirs["results"],
        filename_prefix=f"{EXPERIMENT_NAME}_detailed_results",
        format="csv",
    )
    results_handler.save_results(
        df=run_metrics_means,
        results_dir=exp_dirs["results"],
        filename_prefix=f"{EXPERIMENT_NAME}_run_means",
        format="csv",
    )
    if not a6_summary_df.empty:
        results_handler.save_results(
            df=a6_summary_df,
            results_dir=exp_dirs["results"],
            filename_prefix=f"{EXPERIMENT_NAME}_summary_stats",
            format="csv",
        )

    logger.info("Finished saving results.")
    logger.info("Generating plots for A6...")
    models_file = project_root / "experiments" / "config" / "models.csv"
    models_df_plot = None

    try:
        models_df_plot = pd.read_csv(models_file)
        if "model_id" not in models_df_plot.columns:
            logger.warning(
                f"Loaded models file {models_file}, but 'model_id' column is missing."
            )
            models_df_plot = None
    except FileNotFoundError:
        logger.warning(
            f"Models file not found at {models_file}. Model sizes will not be annotated."
        )
    except Exception as e:
        logger.error(f"Error loading models file {models_file}: {e}")
    if not a6_summary_df.empty:
        if (
            "Mean Total Energy (Wh)" in a6_summary_df.columns
            and a6_summary_df["Mean Total Energy (Wh)"].notna().any()
        ):
            if raw_results_data_df is not None and not raw_results_data_df.empty:
                logger.info("Calling Pareto subplot function (Total Energy Wh)...")
                try:
                    generate_lambda_sweep_pareto_subplots_wh(
                        a6_summary_df=a6_summary_df,
                        raw_results_data_df=raw_results_data_df,
                        exp_dirs=exp_dirs,
                        models_df=models_df_plot,
                        lambdas_to_plot=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    )
                except ImportError:
                    logger.error(
                        "generate_lambda_sweep_pareto_subplots_wh not found in plotting module."
                    )
                except Exception as e:
                    logger.error(
                        f"Error generating Total Energy Pareto subplot grid: {e}",
                        exc_info=True,
                    )
            else:
                logger.warning(
                    "Skipping Total Energy Pareto plot: Raw results data missing for A1 background."
                )
        else:
            logger.warning(
                "Skipping Total Energy Pareto plot: 'Mean Total Energy (Wh)' column missing or empty in summary."
            )
    else:
        logger.warning("Skipping Pareto plots: Aggregated summary DataFrame is empty.")
    logger.info("Calling combined Accuracy / Total Energy Wh boxplot function...")
    try:
        generate_lambda_sweep_accuracy_total_energy_boxplots_wh(
            run_metrics_combined_df=run_metrics_means, exp_dirs=exp_dirs
        )

    except ImportError:
        logger.error(
            "generate_lambda_sweep_accuracy_total_energy_boxplots_wh function not found or import failed."
        )
    except Exception as e:
        logger.error(
            f"Error generating combined Accuracy / Total Energy Wh boxplot: {e}",
            exc_info=True,
        )
    logger.info(f"Experiment {EXPERIMENT_NAME} finished.")


if __name__ == "__main__":
    main()
