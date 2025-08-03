"""
Algorithm Warm-up Experiment (00)

This experiment analyzes the exploration-exploitation behavior of different
bandit algorithms during their initial learning phase. It focuses on how
quickly algorithms converge to good routing policies and their exploration
strategies.

Key measurements:
- Convergence speed
- Model selection patterns over time
- Cumulative regret during warm-up period
"""

import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Local imports
from experiments.shared import (
    analysis_utils,
    baseline_selectors,
    config_loader,
    data_loader,
    plot_utils,
    results_handler,
)
from src.bandit.epsilon_greedy import EpsilonGreedy

from .plotting import generate_cumulative_regret_plot, generate_model_selection_heatmap

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# EXPERIMENT CONSTANTS
# ============================================================================

EXPERIMENT_NAME = "X0_warmup"
N_RUNS = 5


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """
    Main function to execute the 00 warm-up experiment.

    Runs epsilon-greedy algorithm multiple times to analyze exploration
    patterns and convergence behavior during the initial learning phase.
    """

    # ========================================================================
    # EXPERIMENT INITIALIZATION
    # ========================================================================

    logger.info(f"Starting Experiment: {EXPERIMENT_NAME} with {N_RUNS} runs")
    run_timestamp = datetime.now()
    run_id = f"{EXPERIMENT_NAME}_{run_timestamp.strftime('%Y%m%d_%H%M%S')}"

    exp_dirs = config_loader.setup_experiment_dirs(EXPERIMENT_NAME)
    if not exp_dirs:
        return

    # ========================================================================
    # CLEANUP PREVIOUS RESULTS
    # ========================================================================
    plots_dir = exp_dirs.get("plots")
    if plots_dir and plots_dir.exists():
        logger.info(f"Cleaning existing plots from {plots_dir}...")
        deleted_count = 0
        for plot_file in plots_dir.glob("*.png"):
            try:
                plot_file.unlink()
                deleted_count += 1
            except OSError as e:
                logger.warning(f"Could not delete plot file {plot_file}: {e}")
        logger.info(f"Deleted {deleted_count} previous plot files.")
    else:
        logger.info(
            f"Plots directory {plots_dir} not found or not configured, skipping cleaning."
        )

    plot_utils.setup_plotting()

    # ========================================================================
    # CONFIGURATION LOADING
    # ========================================================================
    configs = config_loader.load_config(
        "datasets", "models", "mab", "experiments", config_dir="experiments/config"
    )
    print(f"Configs: {configs}")
    if (
        not configs.get("datasets")
        or not configs.get("models")
        or not configs.get("mab")
    ):
        logger.error(
            "Failed to load required configurations (datasets, models, mab) using config_loader."
        )
        return

    configs.get("experiments", {}).get(EXPERIMENT_NAME, {})
    exp_yaml_path = project_root / "experiments" / "config" / "experiments.yaml"
    try:
        with open(exp_yaml_path, "r") as f:
            exp_configs = yaml.safe_load(f)
        if not exp_configs:
            raise FileNotFoundError("experiments.yaml loaded empty or was not found.")
    except FileNotFoundError:
        logger.error(f"Could not find or load {exp_yaml_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing {exp_yaml_path}: {e}")
        return
    try:
        config = exp_configs.get(EXPERIMENT_NAME, {})
        if not config:
            raise KeyError(
                f"Configuration section '{EXPERIMENT_NAME}' missing in experiments.yaml"
            )

        defaults_config = exp_configs.get("defaults", {})
        dataset_names = defaults_config.get("datasets", ["all"])
        samples_per_dataset = defaults_config.get("samples_per_dataset", 500)
        lambda_weight = config["lambda_weight"]
        base_random_seed = config["random_seed"]

        algo_configs = config.get("algorithms", {})
        if not algo_configs:
            raise KeyError("Section 'algorithms' missing in 00 config")

        eg_config = algo_configs.get("epsilon_greedy", {})
        if not eg_config:
            raise KeyError(
                "Configuration for 'epsilon_greedy' missing in 00 algorithms config"
            )
        initial_epsilon = eg_config["initial_epsilon"]
        decay_factor = eg_config["decay_factor"]
        min_epsilon = eg_config["min_epsilon"]

        logger.info(
            f"Using Lambda: {lambda_weight}, Base Seed: {base_random_seed} (from {EXPERIMENT_NAME} config)"
        )
        logger.info(
            f"EpsilonGreedy Params (from {EXPERIMENT_NAME} config): initial={initial_epsilon}, decay={decay_factor}, min={min_epsilon}"
        )

    except KeyError as e:
        logger.error(
            f"Configuration error: Missing key {e} in experiments.yaml structure (section: {EXPERIMENT_NAME})."
        )
        return

    # ========================================================================
    # DATA LOADING AND PREPARATION
    # ========================================================================
    conn = data_loader.connect_db()
    if not conn:
        return

    try:
        models_df = data_loader.get_model_specs(conn)
        if models_df.empty:
            logger.error("Failed to load model specifications.")
            return
        all_model_ids = models_df["model_id"].tolist()
        logger.info(f"Loaded {len(all_model_ids)} model IDs: {all_model_ids}")

        evaluation_queries_df = data_loader.load_evaluation_dataset(
            conn, dataset_names, samples_per_dataset, base_random_seed
        )
        if evaluation_queries_df.empty:
            logger.error("Failed to load evaluation query IDs.")
            return
        required_query_ids = evaluation_queries_df["query_id"].unique().tolist()
        logger.info(f"Loaded {len(required_query_ids)} evaluation query IDs.")

        logger.info("Checking for pre-generated results...")
        is_complete, missing, all_inference_results_df = (
            data_loader.check_data_completeness(conn, required_query_ids, all_model_ids)
        )
        if not is_complete:
            logger.error("Pre-generated results are MISSING.")
            results_handler.save_results(
                missing, exp_dirs["results"], f"{EXPERIMENT_NAME}_missing_data"
            )
            return
        if all_inference_results_df.empty:
            logger.error("Loaded pre-generated results DataFrame is empty.")
            return
        logger.info(
            f"Loaded {len(all_inference_results_df)} pre-generated results. Data is complete."
        )

        logger.info("Normalizing loaded metrics...")
        results_norm_df = analysis_utils.normalize_metrics(all_inference_results_df)
        if results_norm_df.empty or "accuracy_norm" not in results_norm_df.columns:
            logger.error("Metric normalization failed.")
            return

    except Exception as e:
        logger.error(f"Error during data loading: {e}", exc_info=True)
        return
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

    # ========================================================================
    # REWARD STRUCTURE PREPARATION
    # ========================================================================
    optimal_rewards_dict, reward_lookup = analysis_utils.prepare_reward_structures(
        results_norm_df, lambda_weight
    )
    if not optimal_rewards_dict or not reward_lookup:
        logger.error("Failed to prepare reward structures. Exiting.")
        return
    base_query_sequence = required_query_ids.copy()
    query_count = len(base_query_sequence)
    if query_count == 0:
        logger.error("Query sequence is empty after loading.")
        return
    logger.info(f"Base query sequence length: {query_count}")

    # ========================================================================
    # ALGORITHM SETUP
    # ========================================================================
    def algorithm_factory(run_seed: int) -> dict:
        return {
            "random": baseline_selectors.RandomModelSelector(
                all_model_ids, seed=run_seed
            ),
            "epsilon_greedy": EpsilonGreedy(
                model_ids=all_model_ids,
                initial_epsilon=initial_epsilon,
                decay_factor=decay_factor,
                min_epsilon=min_epsilon,
                seed=run_seed,
            ),
        }

    # --- Simulation Functions ---
    def _run_simulation_trials(
        n_runs: int,
        base_seed: int,
        query_seq: list,
        algo_factory: callable,
        opt_rewards: dict,
        reward_lkp: dict,
        context_features_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Internal helper to run simulation trials for 00 (now with basic context handling)."""
        all_runs_results_list = []
        logger.info(f"Starting {n_runs} simulation runs...")

        if query_seq is None or len(query_seq) == 0:
            logger.error("Query sequence is empty. Aborting simulation.")
            return pd.DataFrame()

        for run_index in range(n_runs):
            run_seed = base_seed + run_index
            logger.info(
                f"--- Starting Run {run_index + 1}/{n_runs} (Seed: {run_seed}) ---"
            )
            random.seed(run_seed)
            np.random.seed(run_seed)
            current_query_sequence = query_seq.copy()
            random.shuffle(current_query_sequence)

            try:
                algorithms = algo_factory(run_seed=run_seed)
            except Exception as e:
                logger.error(
                    f"Error initializing algorithms for run {run_index + 1}: {e}",
                    exc_info=True,
                )
                continue
            run_detailed_results_list = []

            for t, query_id in enumerate(
                tqdm(current_query_sequence, desc=f"Simulating Run {run_index + 1}")
            ):
                optimal_reward = opt_rewards.get(query_id, np.nan)

                if pd.isna(optimal_reward):
                    logger.warning(
                        f"[Run {run_index+1}] Optimal reward not found for query_id {query_id}. Skipping query."
                    )
                    continue
                context_vector = None
                context_available = False

                if context_features_df is not None and not context_features_df.empty:
                    if query_id in context_features_df.index:
                        context_vector = context_features_df.loc[query_id, "context"]
                        if context_vector is not None:
                            context_available = True
                        else:
                            logger.warning(
                                f"[Run {run_index+1}] Context vector is null/NaN for query_id {query_id}."
                            )
                    else:
                        logger.warning(
                            f"[Run {run_index+1}] Context features expected but index not found for query_id {query_id}."
                        )

                for algo_name, algo_instance in algorithms.items():
                    chosen_model_id = None
                    is_contextual = (
                        hasattr(algo_instance, "requires_context")
                        and algo_instance.requires_context
                    )

                    try:
                        if is_contextual:
                            if not context_available:
                                logger.warning(
                                    f"[Run {run_index+1}, Algo {algo_name}] Context required but not available for query {query_id}. Skipping select."
                                )
                                continue

                            chosen_model_id = algo_instance.select_model(
                                context=context_vector
                            )
                        else:
                            chosen_model_id = algo_instance.select_model()
                    except Exception as e:
                        logger.error(
                            f"[Run {run_index+1}, Algo {algo_name}] Error during select_model for query {query_id}: {e}",
                            exc_info=True,
                        )

                    if chosen_model_id is None:
                        actual_reward = 0.0
                        step_regret = (
                            (optimal_reward - actual_reward)
                            if not pd.isna(optimal_reward)
                            else np.nan
                        )
                        if not (is_contextual and not context_available):
                            logger.warning(
                                f"[Run {run_index+1}, Algo {algo_name}] Model selection failed/skipped for query {query_id}. Assigning 0 reward."
                            )
                    else:
                        actual_reward = reward_lkp.get(
                            (query_id, chosen_model_id), np.nan
                        )
                        if pd.isna(actual_reward):
                            actual_reward = 0.0
                        step_regret = (
                            (optimal_reward - actual_reward)
                            if not pd.isna(optimal_reward)
                            else np.nan
                        )
                    if chosen_model_id is not None:
                        try:
                            if is_contextual:
                                algo_instance.update(
                                    chosen_model_id,
                                    actual_reward,
                                    context=context_vector,
                                )
                            elif hasattr(algo_instance, "update"):
                                algo_instance.update(chosen_model_id, actual_reward)
                        except Exception as e:
                            logger.error(
                                f"[Run {run_index+1}, Algo {algo_name}] Error during update for model {chosen_model_id} on query {query_id}: {e}",
                                exc_info=True,
                            )
                    run_detailed_results_list.append(
                        {
                            "run_id": run_index,
                            "algorithm": algo_name,
                            "query_index": t,
                            "query_id": query_id,
                            "chosen_model": chosen_model_id,
                            "actual_reward": actual_reward,
                            "optimal_reward": optimal_reward,
                            "step_regret": step_regret,
                        }
                    )

            all_runs_results_list.extend(run_detailed_results_list)
            processed_queries = (
                len(run_detailed_results_list) // len(algorithms) if algorithms else 0
            )
            logger.info(
                f"--- Finished Run {run_index + 1}/{n_runs} --- Processed {processed_queries} queries for this run."
            )
        if not all_runs_results_list:
            logger.error("Simulation produced no results across all runs.")
            return pd.DataFrame()

        logger.info("Aggregating results from all runs...")
        detailed_results_df = pd.DataFrame(all_runs_results_list)
        if detailed_results_df.empty:
            logger.error("Failed to create DataFrame from aggregated results.")
            return pd.DataFrame()

        detailed_results_df["step_regret"] = pd.to_numeric(
            detailed_results_df["step_regret"], errors="coerce"
        ).fillna(0)
        return detailed_results_df

    # --- Main Simulation Execution ---
    detailed_results_df = _run_simulation_trials(
        n_runs=N_RUNS,
        base_seed=base_random_seed,
        query_seq=base_query_sequence,
        algo_factory=algorithm_factory,
        opt_rewards=optimal_rewards_dict,
        reward_lkp=reward_lookup,
        context_features_df=None,
    )
    if detailed_results_df.empty:
        logger.error("Simulation failed to produce results.")
        return

    # --- Analysis and Statistics ---
    logger.info("Calculating cumulative regret per run...")
    detailed_results_df["cumulative_regret"] = detailed_results_df.groupby(
        ["run_id", "algorithm"]
    )["step_regret"].cumsum()

    logger.info("Analyzing Epsilon-Greedy model selection variability across runs...")
    eg_results = detailed_results_df[
        detailed_results_df["algorithm"] == "epsilon_greedy"
    ].copy()

    if not eg_results.empty:
        max_query_index = eg_results["query_index"].max()
        interval_size = 500
        intervals = range(0, max_query_index + 1, interval_size)
        model_ids = sorted(eg_results["chosen_model"].unique())
        eg_results["run_id"].nunique()

        all_interval_stats = []

        for i in range(len(intervals) - 1):
            start_idx = intervals[i]
            end_idx = intervals[i + 1] - 1
            interval_label = f"Queries {start_idx}-{end_idx}"
            logger.info(f"  Analyzing interval: {interval_label}")
            interval_data = eg_results[
                (eg_results["query_index"] >= start_idx)
                & (eg_results["query_index"] <= end_idx)
            ]

            if interval_data.empty:
                logger.info("    No data in this interval.")
                continue

            run_interval_freq = (
                interval_data.groupby("run_id")["chosen_model"]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
            )

            run_interval_freq = run_interval_freq.reindex(
                columns=model_ids, fill_value=0
            )
            mean_freq = run_interval_freq.mean()
            std_freq = run_interval_freq.std().fillna(0)

            all_interval_stats.append(
                {
                    "interval": interval_label,
                    "mean_freq": mean_freq.to_dict(),
                    "std_freq": std_freq.to_dict(),
                }
            )
    else:
        logger.warning("No Epsilon-Greedy results found to analyze variability.")

    logger.info("Calculating summary statistics across runs...")
    final_regret_per_run = detailed_results_df.loc[
        detailed_results_df.groupby(["run_id", "algorithm"])["query_index"].idxmax()
    ]

    summary_stats_df = final_regret_per_run.groupby("algorithm")[
        "cumulative_regret"
    ].agg(["mean", "std", "min", "max"])
    summary_stats_df.rename(
        columns={
            "mean": "mean_final_cumulative_regret",
            "std": "std_final_cumulative_regret",
            "min": "min_final_cumulative_regret",
            "max": "max_final_cumulative_regret",
        },
        inplace=True,
    )

    logger.info(
        "Final Cumulative Regret Statistics (across runs):\n"
        + summary_stats_df.to_string()
    )

    # --- Results Saving ---
    logger.info("Saving aggregated results...")

    results_handler.save_results(
        detailed_results_df,
        exp_dirs["results"],
        f"{EXPERIMENT_NAME}_detailed_results_all_runs",
    )
    results_handler.save_results(
        summary_stats_df,
        exp_dirs["results"],
        f"{EXPERIMENT_NAME}_summary_stats_across_runs",
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "run_id": run_id,
        "run_timestamp": run_timestamp.isoformat(),
        "configs_used": {k: v for k, v in configs.items() if v is not None},
        "algorithms_compared": list(detailed_results_df["algorithm"].unique()),
        "n_runs": N_RUNS,
        "base_random_seed": base_random_seed,
    }
    results_handler.save_experiment_metadata(
        metadata, exp_dirs["results"], EXPERIMENT_NAME
    )

    # --- Plotting and Visualization ---
    logger.info("Generating plots (using aggregated data)...")
    try:
        generate_cumulative_regret_plot(detailed_results_df, exp_dirs)
        generate_model_selection_heatmap(detailed_results_df, exp_dirs)
    except NameError as e:
        logger.error(f"Plotting function not found: {e}")
    except Exception as e:
        logger.error(f"Error during plotting: {e}", exc_info=True)

    logger.info(f"Experiment {EXPERIMENT_NAME} finished.")


if __name__ == "__main__":
    main()
