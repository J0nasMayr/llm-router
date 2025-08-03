"""
Model Pool Adaptability Experiment

This experiment evaluates how well bandit algorithms adapt to changes in the
available model pool. It simulates adding/removing models mid-experiment to
test algorithm robustness and adaptation speed.

Key measurements:
- Adaptation speed after model pool changes
- Performance impact of model additions/removals
- Algorithm stability during transitions
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

# Local imports
from experiments.02_feature_ablation.utils import create_bandit_instance
from experiments.shared import (
    analysis_utils,
    config_loader,
    data_loader,
    plot_utils,
    results_handler,
)
from src.services.feature_service import FeatureService

EXPERIMENT_NAME = "05_adaptability"
MAX_QUERIES_DEBUG = None


def main():
    exp_dirs = config_loader.setup_experiment_dirs(EXPERIMENT_NAME)
    plot_utils.setup_plotting()
    configs = config_loader.load_config(
        "datasets",
        "models",
        "experiments",
        "feature_extraction",
        config_dir="experiments/config",
    )
    exp_config = configs["experiments"][EXPERIMENT_NAME]
    exp_config.get("plotting", {})
    algo_name = exp_config.get("active_algorithm")
    if not algo_name:
        logger.error("'active_algorithm' key not found in config. Exiting.")
        return
    all_algo_params_dict = exp_config.get("algorithms", {})
    algo_params = all_algo_params_dict.get(algo_name)
    if algo_params is None:
        logger.error(
            f"Parameters for active algorithm '{algo_name}' not found. Exiting."
        )
        return
    reg_lambda = algo_params.get("lambda_") or algo_params.get("regularization")
    change_point_idx = exp_config.get("change_point_query_index")
    model_to_add = exp_config.get("model_to_add")
    model_to_remove = exp_config.get("model_to_remove")
    if (
        change_point_idx is None
        or not isinstance(change_point_idx, int)
        or change_point_idx < 0
    ):
        logger.error(
            "'change_point_query_index' must be a non-negative integer. Exiting."
        )
        return
    if model_to_add is None and model_to_remove is None:
        logger.warning(
            "Neither 'model_to_add' nor 'model_to_remove' specified. Experiment will run without adaptation."
        )
    lambda_weight = exp_config["lambda_weight"]
    feature_config = exp_config.get("features", ["all"])
    n_runs = exp_config["n_runs"]
    base_seed = exp_config["random_seed"]
    dataset_names = exp_config.get("datasets", ["all"])
    samples_per_dataset = exp_config.get("samples_per_dataset", 500)

    logger.info(
        f"Starting {EXPERIMENT_NAME} ({algo_name}) | Runs: {n_runs} | Lambda: {lambda_weight}"
    )
    logger.info(
        f"Adaptation: Change Point Index = {change_point_idx}, Add = {model_to_add}, Remove = {model_to_remove}"
    )
    logger.info(f"Using Features: {feature_config}")
    conn = data_loader.connect_db()
    all_db_models = data_loader.get_model_specs(conn)["model_id"].tolist()
    logger.info(f"Loaded {len(all_db_models)} total models from database.")
    initial_bandit_models = all_db_models.copy()

    if model_to_add and model_to_add in initial_bandit_models:
        logger.info(
            f"'{model_to_add}' specified to be added later; removing from initial pool."
        )
        initial_bandit_models.remove(model_to_add)

    if model_to_remove and model_to_remove not in initial_bandit_models:
        logger.warning(
            f"'{model_to_remove}' specified for removal but not in initial pool derived from DB (after potentially removing model_to_add). It might be added back if present in DB."
        )
        if (
            model_to_remove in all_db_models
            and model_to_remove not in initial_bandit_models
        ):
            initial_bandit_models.append(model_to_remove)
            logger.info(
                f"Re-added '{model_to_remove}' to initial pool to ensure it can be removed later."
            )

    if not initial_bandit_models:
        logger.error("Initial bandit model pool is empty! Check configuration.")
        return

    logger.info(
        f"Initial bandit model pool ({len(initial_bandit_models)} models): {initial_bandit_models}"
    )
    actual_samples = samples_per_dataset
    if MAX_QUERIES_DEBUG:
        actual_samples = max(1, MAX_QUERIES_DEBUG)

    eval_df = data_loader.load_evaluation_dataset(
        conn, dataset_names, actual_samples, base_seed
    )
    if MAX_QUERIES_DEBUG and len(eval_df) > MAX_QUERIES_DEBUG:
        eval_df = eval_df.sample(n=MAX_QUERIES_DEBUG, random_state=base_seed)

    query_ids = eval_df["query_id"].unique().tolist()
    if not query_ids:
        logger.error("No query IDs loaded.")
        return

    logger.info(f"Loaded {len(query_ids)} unique queries for simulation.")
    _, _, results_df = data_loader.check_data_completeness(
        conn, query_ids, all_db_models
    )
    results_df = results_df[results_df["query_id"].isin(query_ids)]
    norm_df = analysis_utils.normalize_metrics(results_df)
    optimal_rewards_dict, reward_lkp = analysis_utils.prepare_reward_structures(
        norm_df, lambda_weight
    )
    query_meta = eval_df.set_index("query_id")[["text", "dataset"]].to_dict("index")

    if conn:
        conn.close()
    logger.info(f"Preparing features based on config: {feature_config}...")

    try:
        feature_service = FeatureService(configs.get("feature_extraction"))
    except Exception as e:
        logger.error(
            f"Failed to initialize FeatureService: {e}. Cannot compute contexts.",
            exc_info=True,
        )
        return

    indices, idx = [], 0

    try:
        n_task_features = len(feature_service.task_types)
        n_cluster_features = feature_service.num_clusters
        n_complexity_features = feature_service.num_complexity_bins
        logger.info(
            f"FeatureService reports dimensions: Task={n_task_features}, Cluster={n_cluster_features}, Complexity={n_complexity_features}"
        )
    except AttributeError as e:
        logger.error(
            f"FeatureService instance is missing expected attributes (task_types, num_clusters, num_complexity_bins): {e}"
        )
        return

    if "task" in feature_config or "dataset_index" in feature_config:
        indices.extend(range(idx, idx + n_task_features))
        logger.info(
            f"Including Task features (indices {idx} to {idx + n_task_features - 1})"
        )
    idx += n_task_features

    if "semantic_cluster" in feature_config:
        indices.extend(range(idx, idx + n_cluster_features))
        logger.info(
            f"Including Cluster features (indices {idx} to {idx + n_cluster_features - 1})"
        )
    idx += n_cluster_features

    if "complexity" in feature_config:
        indices.extend(range(idx, idx + n_complexity_features))
        logger.info(
            f"Including Complexity features (indices {idx} to {idx + n_complexity_features - 1})"
        )
    logger.info(f"Final selected feature indices (relative to base vector): {indices}")

    if not indices:
        context_dim = 1
        logger.info(
            "No specific features selected based on config. Using intercept only (dim=1)."
        )
    else:
        context_dim = len(indices) + 1

    logger.info(
        f"Effective context dimension for features {feature_config}: {context_dim}"
    )
    logger.info("Extracting full base context vectors...")

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

    if "context_vector" not in features_df.columns:
        logger.error(
            "'context_vector' not found in DataFrame after feature extraction."
        )
        return

    base_contexts = features_df["context_vector"].to_dict()
    try:
        total_dim_base = len(next(iter(base_contexts.values())))
    except StopIteration:
        logger.error(
            "Could not determine base feature dimension - no contexts extracted?"
        )
        return

    logger.info("Creating final context dictionary with slicing and intercept...")
    contexts_dict = {}
    
    # Define fallback context vector
    fallback = np.zeros(context_dim, dtype=np.float32)
    fallback[0] = 1.0  # Set intercept

    for qid in query_ids:
        if context_dim == 1:
            contexts_dict[qid] = np.array([1.0], dtype=np.float32)
        else:
            base_vec = base_contexts.get(qid)
            if base_vec is None or len(base_vec) != total_dim_base:
                logger.warning(
                    f"Invalid base context vector for query {qid}. Using fallback."
                )
                contexts_dict[qid] = fallback
            elif not indices:
                logger.warning(
                    f"No indices defined but context_dim > 1 for query {qid}. Using fallback."
                )
                contexts_dict[qid] = fallback
            else:
                try:
                    sliced_vec = np.array(base_vec)[indices]
                    contexts_dict[qid] = np.concatenate(([1.0], sliced_vec)).astype(
                        np.float32
                    )
                except IndexError:
                    logger.error(
                        f"IndexError slicing base vector for query {qid}. Base len: {len(base_vec)}, Indices: {indices}. Using fallback."
                    )
                    contexts_dict[qid] = fallback
                except Exception as e:
                    logger.error(
                        f"Error processing context for query {qid}: {e}. Using fallback."
                    )
                    contexts_dict[qid] = fallback

    logger.info(f"Context preparation finished. Context dimension: {context_dim}")
    detailed_run_results = []
    query_sequence = query_ids.copy()
    total_queries_in_sequence = len(query_sequence)
    logger.info(
        f"Total queries in sequence: {total_queries_in_sequence}. Change point index: {change_point_idx}"
    )

    if change_point_idx >= total_queries_in_sequence:
        logger.warning(
            f"Change point index {change_point_idx} is >= total queries {total_queries_in_sequence}. No adaptation will occur."
        )

    for r in range(n_runs):
        seed = base_seed + r
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"--- Starting Run {r+1}/{n_runs} (Seed: {seed}) ---")
        algo = create_bandit_instance(
            algo_name=algo_name,
            algo_params=algo_params,
            model_ids=initial_bandit_models.copy(),
            context_dimension=context_dim,
            reg_lambda=reg_lambda,
            seed=seed,
        )
        if algo is None:
            logger.error(f"Failed to create bandit instance for run {r+1}, skipping.")
            continue
        if not hasattr(algo, "add_model") or not hasattr(algo, "remove_model"):
            logger.error(
                f"Algorithm '{algo_name}' does not support add_model/remove_model. Skipping run."
            )
            continue

        if hasattr(algo, "reset"):
            algo.reset()
        random.shuffle(query_sequence)

        adaptation_performed = False

        for qid_idx, qid in enumerate(
            tqdm(query_sequence, desc=f"Run {r+1}/{n_runs}", leave=False)
        ):
            if not adaptation_performed and qid_idx == change_point_idx:
                logger.info(
                    f"[Run {r+1}, Index {qid_idx}] Reached change point. Attempting model pool adaptation..."
                )
                removed_ok, added_ok = True, True
                if model_to_remove:
                    logger.info(f"  Attempting to remove model: {model_to_remove}")
                    removed_ok = algo.remove_model(model_to_remove)
                    if not removed_ok:
                        logger.warning(f"  Failed to remove model '{model_to_remove}'.")
                if model_to_add:
                    logger.info(f"  Attempting to add model: {model_to_add}")
                    added_ok = algo.add_model(model_to_add)
                    if not added_ok:
                        logger.warning(f"  Failed to add model '{model_to_add}'.")
                adaptation_performed = True
                logger.info(
                    f"  Adaptation attempt finished. Current models: {algo.model_ids}"
                )

            context = contexts_dict.get(qid, np.zeros(context_dim, dtype=np.float32))
            chosen_model = algo.select_model(context=context)
            reward = reward_lkp.get((qid, chosen_model), 0.0)
            algo.update(chosen_model, reward, context=context)

            optimal_reward = optimal_rewards_dict.get(qid, 0.0)
            step_regret = optimal_reward - reward

            detailed_run_results.append(
                {
                    "run_id": seed,
                    "query_index": qid_idx,
                    "chosen_model": chosen_model,
                    "step_regret": step_regret,
                    "reward": reward,
                    "algorithm": algo_name,
                    "available_models": tuple(sorted(algo.model_ids)),
                }
            )

        logger.info(f"--- Finished Run {r+1}/{n_runs} ---")
    if not detailed_run_results:
        logger.error("No simulation results.")
        return
    results_df = pd.DataFrame(detailed_run_results)
    results_handler.save_results(
        results_df, exp_dirs["results"], f"{EXPERIMENT_NAME}_detailed_results"
    )
    logger.info("--- Debugging Model Choices Before Plotting ---")

    if not results_df.empty:
        logger.info(
            f"Unique models CHOSEN during simulation: {results_df['chosen_model'].unique().tolist()}"
        )
        try:
            unique_pools = {
                tuple(sorted(pool)) for pool in results_df["available_models"]
            }
            logger.info(
                f"Unique AVAILABLE model pools during simulation: {list(unique_pools)}"
            )
        except Exception as e:
            logger.error(f"Could not process available_models column: {e}")
        logger.info("Value Counts of Chosen Models:")
        if change_point_idx < total_queries_in_sequence:
            before_change = results_df[results_df["query_index"] < change_point_idx][
                "chosen_model"
            ].value_counts()
            after_change = results_df[results_df["query_index"] >= change_point_idx][
                "chosen_model"
            ].value_counts()
            logger.info(
                f"  Before Change Point (Index < {change_point_idx}):\n{before_change.to_string()}"
            )
            logger.info(
                f"  After Change Point (Index >= {change_point_idx}):\n{after_change.to_string()}"
            )
        else:
            full_run_counts = results_df["chosen_model"].value_counts()
            logger.info(
                f"  Full Run (No effective change point):\n{full_run_counts.to_string()}"
            )
    else:
        logger.warning("Results DataFrame is empty, cannot perform debug checks.")
    logger.info("--- End Debugging --- ")

    logger.info("Generating adaptability plots...")
    try:
        from .plotting import (
            generate_cumulative_regret_plot,
            generate_model_choice_timeline,
            generate_model_selection_plot,
        )

        cum_regret_title = f"{EXPERIMENT_NAME} ({algo_name}, $\\lambda={lambda_weight}$)\nMean Cumulative Regret"
        generate_cumulative_regret_plot(
            results_df, exp_dirs, title=cum_regret_title, change_point=change_point_idx
        )
        logger.info(f"Cumulative regret plot saved to {exp_dirs['plots']}")
        selection_plot_title = f"{EXPERIMENT_NAME} ({algo_name}, $\\lambda={lambda_weight}$)\nModel Selection Frequency Over Time"
        generate_model_selection_plot(
            results_df,
            exp_dirs,
            title=selection_plot_title,
            change_point=change_point_idx,
            model_added=model_to_add,
            model_removed=model_to_remove,
            all_models_ever=tuple(
                sorted(
                    list(
                        set(initial_bandit_models)
                        | ({model_to_add} if model_to_add else set())
                    )
                )
            ),
        )
        logger.info(f"Model selection plot saved to {exp_dirs['plots']}")
        timeline_plot_title = f"{EXPERIMENT_NAME} ({algo_name}, $\\lambda={lambda_weight}$)\nModel Choice Timeline"
        generate_model_choice_timeline(
            results_df,
            exp_dirs,
            title=timeline_plot_title,
            change_point=change_point_idx,
            model_added=model_to_add,
            model_removed=model_to_remove,
            all_models_ever=tuple(
                sorted(
                    list(
                        set(initial_bandit_models)
                        | ({model_to_add} if model_to_add else set())
                    )
                )
            ),
        )
        logger.info(f"Model choice timeline plot saved to {exp_dirs['plots']}")

    except ImportError:
        logger.error(
            "Could not import plotting functions from experiments.05_adaptability.plotting. Skipping plots."
        )
    except Exception as e:
        logger.error(f"Error during plotting: {e}", exc_info=True)
    logger.info(f"Experiment {EXPERIMENT_NAME} finished.")


if __name__ == "__main__":
    main()
