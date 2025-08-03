# experiments/04_hyperparameter_tuning/run_experiment.py


import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from experiments.02_feature_ablation.utils import create_bandit_instance
from experiments.shared import (
    analysis_utils,
    config_loader,
    data_loader,
    plot_utils,
    results_handler,
)
from src.services.feature_service import FeatureService

from .plotting import generate_hyperparameter_performance_plot
from .utils import generate_hyperparameter_configs

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "04_hyperparameter_tuning"
MAX_QUERIES_DEBUG = None

# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    # ========================================================================
    # CONFIGURATION LOADING
    # ========================================================================

    exp_dirs = config_loader.setup_experiment_dirs(EXPERIMENT_NAME)
    logger.info("Loading configurations...")
    configs = config_loader.load_config(
        "datasets",
        "models",
        "experiments",
        "feature_extraction",
        config_dir="experiments/config",
    )
    defaults = configs["experiments"]["defaults"]
    try:
        experiment_config = configs["experiments"][EXPERIMENT_NAME]
        defaults = configs["experiments"].get("defaults", {})
    except KeyError as e:
        logger.error(f"Missing required configuration section in experiments.yaml: {e}")
        return
    try:
        algorithms_to_tune = experiment_config["algorithms"]
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

    # ========================================================================
    # DATA LOADING AND PREPARATION
    # ========================================================================

    logger.info("Loading and preparing data...")
    conn = data_loader.connect_db()
    all_model_ids = data_loader.get_model_specs(conn)["model_id"].tolist()
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

    _, _, results_df = data_loader.check_data_completeness(
        conn, query_ids, all_model_ids
    )
    results_df = results_df[results_df["query_id"].isin(query_ids)]
    norm_df = analysis_utils.normalize_metrics(results_df)
    optimal_rewards_dict, reward_lkp = analysis_utils.prepare_reward_structures(
        norm_df, lambda_weight
    )
    query_meta = eval_df.set_index("query_id")[["text", "dataset"]].to_dict("index")

    if conn:
        conn.close()
    logger.info("Data loading and normalization complete.")

    # ========================================================================
    # FEATURE EXTRACTION AND CONTEXT PREPARATION
    # ========================================================================

    logger.info(f"Preparing features: {feature_names}...")
    feature_service = FeatureService(feature_config)

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
        logger.warning(
            f"Specified features {feature_names} resulted in no indices. Context dim will be 1 (intercept only)."
        )
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
        for qid in query_ids
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
                sliced_vec = np.array(base_vec)[indices]
                fs_contexts[qid] = np.concatenate(([1.0], sliced_vec)).astype(
                    np.float32
                )

    logger.info("Context vectors prepared.")

    # ========================================================================
    # HYPERPARAMETER TUNING SIMULATION
    # ========================================================================

    all_detailed_run_results = []
    logger.info(f"[Hyperparameter Tuning Setup] Initial len(query_ids): {len(query_ids)}")
    logger.info(f"[Hyperparameter Tuning Setup] Initial query_ids sample (first 10): {query_ids[:10]}")
    logger.info(f"[Hyperparameter Tuning Setup] Initial query_ids sample (last 10): {query_ids[-10:]}")
    query_sequence = query_ids.copy()

    for algo_name, base_algo_params in algorithms_to_tune.items():
        logger.info(f"===== Tuning Algorithm: {algo_name} ====")
        hyperparam_configs = generate_hyperparameter_configs(base_algo_params)
        if not hyperparam_configs:
            logger.warning(
                f"No hyperparameter combinations generated for {algo_name}. Skipping."
            )
            continue

        for specific_params in hyperparam_configs:
            param_str = ", ".join([f"{k}={v}" for k, v in specific_params.items()])
            logger.info(
                f"--- Running Config: {algo_name} | Params: {{{param_str}}} ---"
            )
            current_reg_lambda = specific_params.get(
                "lambda_", specific_params.get("regularization")
            )
            for r in range(n_runs):
                seed = base_seed + r
                random.seed(seed)
                np.random.seed(seed)
                logger.info(
                    f"[Run {r+1}] Instantiating {algo_name} with params: {specific_params} | Reg Lambda: {current_reg_lambda}"
                )
                algo = create_bandit_instance(
                    algo_name=algo_name,
                    algo_params=specific_params,
                    model_ids=all_model_ids,
                    context_dimension=eff_dim,
                    reg_lambda=current_reg_lambda,
                    seed=seed,
                )
                if algo is None:
                    continue
                if hasattr(algo, "reset"):
                    algo.reset()

                random.shuffle(query_sequence)
                if r == 0:
                    logger.info(
                        f"[Run {r+1}] Initial query sequence sample (first 20): {query_sequence[:20]}"
                    )
                for qid_idx, qid in enumerate(
                    tqdm(
                        query_sequence,
                        desc=f"Algo '{algo_name}' Run {r+1}/{n_runs} Params {param_str[:30]}...",
                        leave=False,
                    )
                ):
                    context = fs_contexts.get(qid, np.zeros(eff_dim, dtype=np.float32))
                    chosen_model = algo.select_model(context=context)
                    reward = reward_lkp.get((qid, chosen_model), 0.0)
                    algo.update(chosen_model, reward, context=context)
                    optimal_reward = optimal_rewards_dict.get(qid, 0.0)
                    step_regret = optimal_reward - reward
                    result_row = {
                        "run_id": seed,
                        "query_index": qid_idx,
                        "chosen_model": chosen_model,
                        "step_regret": step_regret,
                        "algorithm": algo_name,
                    }
                    result_row.update(specific_params)
                    all_detailed_run_results.append(result_row)
    if not all_detailed_run_results:
        logger.error("No simulation results generated across all configurations.")
        return

    results_df = pd.DataFrame(all_detailed_run_results)
    known_cols_for_grouping = [
        "run_id",
        "algorithm",
        "query_index",
        "chosen_model",
        "step_regret",
        "cumulative_regret",
    ]
    hyperparam_cols = sorted(
        [col for col in results_df.columns if col not in known_cols_for_grouping]
    )

    grouping_cols = ["run_id", "algorithm"] + hyperparam_cols
    grouping_cols = [col for col in grouping_cols if col in results_df.columns]

    results_df = results_df.sort_values(by=grouping_cols + ["query_index"])
    results_df["cumulative_regret"] = results_df.groupby(grouping_cols, dropna=False)[
        "step_regret"
    ].cumsum()
    final_step_df = results_df.loc[
        results_df.groupby(grouping_cols, dropna=False)["query_index"].idxmax()
    ].copy()
    agg_grouping_cols = ["algorithm"] + hyperparam_cols
    agg_grouping_cols = [
        col for col in agg_grouping_cols if col in final_step_df.columns
    ]
    summary_stats_df = (
        final_step_df.groupby(agg_grouping_cols, dropna=False)["cumulative_regret"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary_stats_df.rename(
        columns={"mean": "mean_final_regret", "std": "std_final_regret"}, inplace=True
    )
    summary_stats_df = summary_stats_df.sort_values(
        by=["algorithm", "mean_final_regret"]
    )
    logger.info(
        "--- Hyperparameter Tuning Results (Mean Final Regret per Algorithm) ---"
    )
    all_algorithms = summary_stats_df["algorithm"].unique()
    for algo_name in all_algorithms:
        algo_summary = summary_stats_df[
            summary_stats_df["algorithm"] == algo_name
        ].copy()
        algo_hyperparams = [
            col
            for col in hyperparam_cols
            if col in algo_summary.columns and not algo_summary[col].isna().all()
        ]
        display_cols = (
            ["algorithm"] + algo_hyperparams + ["mean_final_regret", "std_final_regret"]
        )
        display_cols = [col for col in display_cols if col in algo_summary.columns]
        algo_summary_display = algo_summary[display_cols].fillna("N/A")
        algo_summary_display = algo_summary_display.sort_values(by="mean_final_regret")

        print(f"\n===== Results for: {algo_name} ====")
        print(algo_summary_display.to_string(index=False))
    logger.info("--- End Algorithm-Specific Results ---")
    best_configs_list = []

    for algo_name in all_algorithms:
        algo_summary = summary_stats_df[summary_stats_df["algorithm"] == algo_name]
        if algo_summary.empty:
            continue
        best_config_row = algo_summary.loc[algo_summary["mean_final_regret"].idxmin()]
        best_configs_list.append(best_config_row)

    if best_configs_list:
        best_results_df = pd.DataFrame(best_configs_list)
        best_display_cols = [
            "algorithm",
            "mean_final_regret",
            "std_final_regret",
        ] + hyperparam_cols
        best_display_cols = [
            col for col in best_display_cols if col in best_results_df.columns
        ]
        best_results_df = best_results_df[best_display_cols].sort_values(
            by="mean_final_regret"
        )

        logger.info("--- Best Configuration per Algorithm --- ")
        print(best_results_df.fillna("-").to_string(index=False))
        logger.info("--- End Best Configuration Table --- ")
    else:
        logger.info("No best configurations found to display.")
    best_params_log = []

    for algo_name in all_algorithms:
        algo_summary = summary_stats_df[summary_stats_df["algorithm"] == algo_name]
        if algo_summary.empty:
            continue
        best_config = algo_summary.loc[algo_summary["mean_final_regret"].idxmin()]
        best_param_values = {
            k: best_config[k]
            for k in hyperparam_cols
            if k in best_config and pd.notna(best_config[k])
        }
        best_params_log.append(
            f"  Best for {algo_name}: Regret={best_config['mean_final_regret']:.2f} with Params: { best_param_values }"
        )

    logger.info("Best Hyperparameters Found (Full Log):\n" + "\n".join(best_params_log))
    logger.info("Saving results...")
    results_handler.save_results(
        results_df, exp_dirs["results"], f"{EXPERIMENT_NAME}_detailed_results"
    )
    results_handler.save_results(
        summary_stats_df, exp_dirs["results"], f"{EXPERIMENT_NAME}_summary_stats"
    )

    if best_configs_list:
        results_handler.save_results(
            best_results_df, exp_dirs["results"], f"{EXPERIMENT_NAME}_best_configs"
        )
    logger.info("Generating plots...")
    plot_utils.setup_plotting()
    generate_hyperparameter_performance_plot(summary_stats_df, exp_dirs)

    logger.info(f"Experiment {EXPERIMENT_NAME} finished.")


if __name__ == "__main__":
    main()
