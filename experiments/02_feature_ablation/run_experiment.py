# --- Imports ---
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from experiments.shared import (
    analysis_utils,
    config_loader,
    data_loader,
    plot_utils,
    results_handler,
)
from src.bandit.epsilon_greedy import EpsilonGreedy
from src.services.feature_service import FeatureService

from .utils import (
    calculate_performance_gaps,
    create_bandit_instance,
    get_bandit_parameters,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "02_feature_ablation"
MAX_QUERIES_DEBUG = None

# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================


def perform_statistical_analysis(
    regret_df, metric_col="cumulative_regret", group_col="feature_set", alpha=0.05
):
    """
    Performs ANOVA and Tukey's HSD test to compare performance across groups.

    Args:
        regret_df (pd.DataFrame): DataFrame containing per-run results.
                                  Must have columns specified by metric_col and group_col.
        metric_col (str): The name of the column containing the performance metric (e.g., 'cumulative_regret').
        group_col (str): The name of the column containing the group labels (e.g., 'feature_set').
        alpha (float): The significance level for the tests.
    """
    logger.info(f"--- Starting Statistical Analysis (Alpha={alpha}) ---")
    if metric_col not in regret_df.columns or group_col not in regret_df.columns:
        logger.error(
            f"Required columns ('{metric_col}', '{group_col}') not found in DataFrame. Skipping statistical analysis."
        )
        return
    groups = regret_df[group_col].unique()
    if len(groups) < 2:
        logger.warning(
            f"Only {len(groups)} group(s) found. ANOVA requires at least 2 groups. Skipping statistical analysis."
        )
        return

    data_groups = [
        regret_df[regret_df[group_col] == group][metric_col].values for group in groups
    ]
    valid_groups = []
    for i, group_data in enumerate(data_groups):
        if len(group_data) > 1:
            valid_groups.append(group_data)
        else:
            logger.warning(
                f"Group '{groups[i]}' has insufficient data ({len(group_data)} points). Excluding from ANOVA."
            )

    if len(valid_groups) < 2:
        logger.warning(
            "Less than 2 groups have sufficient data for ANOVA. Skipping statistical analysis."
        )
        return
    try:
        f_statistic, p_value = stats.f_oneway(*valid_groups)
        logger.info(
            f"One-Way ANOVA Result: F-statistic = {f_statistic:.4f}, p-value = {p_value:.4g}"
        )
        if p_value < alpha:
            logger.info(
                f"ANOVA indicates a significant difference (p < {alpha}). Performing Tukey's HSD test..."
            )
            valid_group_names = [
                groups[i]
                for i, group_data in enumerate(data_groups)
                if len(group_data) > 1
            ]
            tukey_df = regret_df[regret_df[group_col].isin(valid_group_names)]

            try:
                tukey_result = pairwise_tukeyhsd(
                    tukey_df[metric_col], tukey_df[group_col], alpha=alpha
                )
                logger.info(
                    "Tukey's HSD Post-Hoc Test Results:\n" + str(tukey_result.summary())
                )
            except Exception as e:
                logger.error(f"Error during Tukey's HSD test: {e}")
        else:
            logger.info(
                f"ANOVA indicates no significant difference among groups (p >= {alpha})."
            )

    except ValueError as ve:
        logger.error(
            f"Error during ANOVA calculation (likely due to input data issues): {ve}"
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during statistical analysis: {e}")

    logger.info("--- Finished Statistical Analysis ---")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    # --- Experiment Configuration Loading ---
    exp_dirs = config_loader.setup_experiment_dirs(EXPERIMENT_NAME)
    configs = config_loader.load_config(
        "datasets",
        "models",
        "experiments",
        "feature_extraction",
        config_dir="experiments/config",
    )
    a3_config = configs["experiments"][EXPERIMENT_NAME]
    algo_name = a3_config.get("active_algorithm")

    if not algo_name:
        logger.error(
            f"'active_algorithm' key not found in configuration for {EXPERIMENT_NAME}. Exiting."
        )
        return
    all_algo_params_dict = a3_config.get("algorithms", {})
    algo_params = all_algo_params_dict.get(algo_name)
    if algo_params is None:
        logger.error(
            f"Parameters for active algorithm '{algo_name}' not found under 'algorithms' key. Exiting."
        )
        return
    reg_lambda = algo_params.get("lambda_")
    if reg_lambda is None:
        reg_lambda = algo_params.get("regularization")

    lambda_weight = a3_config["lambda_weight"]
    defaults = configs["experiments"]["defaults"]
    feature_sets = a3_config["feature_sets"]
    n_runs = a3_config["n_runs"]
    base_seed = a3_config["random_seed"]
    dataset_names = defaults.get("datasets", ["all"])
    samples_per_dataset = defaults.get("samples_per_dataset", 500)
    logger.info(
        f"Starting {EXPERIMENT_NAME} ({algo_name}) | Feature Sets: {feature_sets} | Runs: {n_runs} | Lambda: {reg_lambda} (Regularization)"
    )

    # --- Data Loading and Preparation ---
    conn = data_loader.connect_db()
    all_model_ids = data_loader.get_model_specs(conn)["model_id"].tolist()

    actual_samples = samples_per_dataset
    if MAX_QUERIES_DEBUG:
        num_datasets = len(dataset_names) if "all" not in dataset_names else 5
        actual_samples = max(1, MAX_QUERIES_DEBUG // num_datasets)

    eval_df = data_loader.load_evaluation_dataset(
        conn, dataset_names, actual_samples, base_seed
    )
    if MAX_QUERIES_DEBUG and len(eval_df) > MAX_QUERIES_DEBUG:
        eval_df = eval_df.sample(n=MAX_QUERIES_DEBUG, random_state=base_seed)

    query_ids = eval_df["query_id"].unique().tolist()
    if not query_ids:
        raise ValueError("No query IDs loaded.")

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
    logger.info(f"Feature Config: {configs.get('feature_extraction')}")

    feature_service = FeatureService(configs.get("feature_extraction"))
    n_task = len(feature_service.task_types)
    n_cluster = feature_service.num_clusters
    n_complex = feature_service.num_complexity_bins
    total_dim = n_task + n_cluster + n_complex

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
    feature_dims = {
        "n_task": n_task,
        "n_cluster": n_cluster,
        "n_complex": n_complex,
        "total": total_dim,
    }

    logger.info("--- Base Feature Distribution --- ")
    if "task_type" in features_df.columns:
        logger.info(
            f"Task Type Counts:\n{features_df['task_type'].value_counts().to_string()}"
        )
    else:
        logger.warning("'task_type' column not found in features_df.")

    if "semantic_cluster" in features_df.columns:
        logger.info(
            f"Semantic Cluster Counts:\n{features_df['semantic_cluster'].value_counts().sort_index().to_string()}"
        )
    else:
        logger.warning("'semantic_cluster' column not found in features_df.")

    if "complexity_bin" in features_df.columns:
        logger.info(
            f"Complexity Bin Counts:\n{features_df['complexity_bin'].value_counts().sort_index().to_string()}"
        )
    else:
        logger.warning("'complexity_bin' column not found in features_df.")
    logger.info("--- End Base Feature Distribution --- ")
    detailed_run_results = []
    query_sequence = query_ids.copy()

    for fs_name in feature_sets:
        logger.info(f"Processing Feature Set: {fs_name}")
        indices, idx = [], 0
        n_task, n_cluster, n_complex = (
            feature_dims["n_task"],
            feature_dims["n_cluster"],
            feature_dims["n_complex"],
        )

        if fs_name == "none":
            eff_dim = 1
        else:
            if "task" in fs_name or "full" in fs_name:
                indices.extend(range(idx, idx + n_task))
            idx += n_task
            if "cluster" in fs_name or "full" in fs_name:
                indices.extend(range(idx, idx + n_cluster))
            idx += n_cluster
            if "complex" in fs_name or "full" in fs_name:
                indices.extend(range(idx, idx + n_complex))

            if not indices:
                logger.warning(
                    f"Feature set '{fs_name}' resulted in no indices, treating as 'none'."
                )
                eff_dim = 1
            else:
                eff_dim = len(indices) + 1

        fs_contexts = {}
        total_dim_base = feature_dims["total"]
        for qid in query_ids:
            if eff_dim == 1 and fs_name == "none":
                fs_contexts[qid] = np.array([1.0], dtype=np.float32)
            else:
                base_vec = base_contexts.get(qid)
                fallback = np.zeros(eff_dim, dtype=np.float32)
                fallback[0] = 1.0

                if base_vec is None or len(base_vec) != total_dim_base:
                    fs_contexts[qid] = fallback
                elif not indices:
                    fs_contexts[qid] = np.array([1.0], dtype=np.float32)
                else:
                    sliced_vec = np.array(base_vec)[indices]
                    fs_contexts[qid] = np.concatenate(([1.0], sliced_vec)).astype(
                        np.float32
                    )
        unique_context_vectors = set(tuple(v) for v in fs_contexts.values())
        logger.info(
            f"Generated {len(unique_context_vectors)} unique context vectors for feature set '{fs_name}'. Dimension: {eff_dim}"
        )
        logger.debug(f"  Unique Context Vectors for '{fs_name}':")
        for i, vec_tuple in enumerate(unique_context_vectors):
            logger.debug(f"    {i+1}: {vec_tuple}")
        for r in range(n_runs):
            seed = base_seed + r
            random.seed(seed)
            np.random.seed(seed)
            algo = None
            current_algo_name = None

            if fs_name == "none":
                current_algo_name = (
                    "epsilon_greedy"  # Using the standard non-contextual EpsilonGreedy
                )
                epsilon_greedy_params = all_algo_params_dict.get(current_algo_name)
                if epsilon_greedy_params is None:
                    logger.error(
                        f"Parameters for '{current_algo_name}' not found under 'algorithms' key. Skipping run."
                    )
                    continue
                try:
                    algo = EpsilonGreedy(
                        model_ids=all_model_ids,
                        initial_epsilon=epsilon_greedy_params["initial_epsilon"],
                        decay_factor=epsilon_greedy_params["decay_factor"],
                        min_epsilon=epsilon_greedy_params["min_epsilon"],
                        seed=seed,
                        context_dimension=1,
                    )
                    logger.debug(f"Instantiated EpsilonGreedy for fs='{fs_name}'")
                except Exception as e:
                    logger.error(
                        f"Error instantiating EpsilonGreedy for fs='{fs_name}': {e}",
                        exc_info=True,
                    )
                    continue

            else:
                current_algo_name = algo_name
                try:
                    algo = create_bandit_instance(
                        algo_name=current_algo_name,
                        algo_params=algo_params,
                        model_ids=all_model_ids,
                        context_dimension=eff_dim,
                        reg_lambda=reg_lambda,
                        seed=seed,
                    )
                    logger.debug(
                        f"Instantiated {current_algo_name} for fs='{fs_name}' with dim={eff_dim}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error instantiating {current_algo_name} for fs='{fs_name}': {e}",
                        exc_info=True,
                    )
                    continue

            if algo is None:
                logger.error(
                    f"Failed to create bandit instance for run {r+1}, feature set '{fs_name}'. Skipping run."
                )
                continue

            if hasattr(algo, "reset"):
                algo.reset()

            random.shuffle(query_sequence)

            for qid_idx, qid in enumerate(
                tqdm(
                    query_sequence,
                    desc=f"FS '{fs_name}' Run {r+1}/{n_runs}",
                    leave=False,
                )
            ):
                context = None
                if fs_name != "none":
                    context = fs_contexts.get(qid, np.zeros(eff_dim, dtype=np.float32))

                chosen_model = algo.select_model(context=context)
                reward = reward_lkp.get((qid, chosen_model), 0.0)
                algo.update(chosen_model, reward, context=context)
                optimal_reward = optimal_rewards_dict.get(qid, 0.0)
                step_regret = optimal_reward - reward

                detailed_run_results.append(
                    {
                        "run_id": seed,
                        "feature_set": fs_name,
                        "query_index": qid_idx,
                        "chosen_model": chosen_model,
                        "step_regret": step_regret,
                        "algorithm": current_algo_name,
                    }
                )
            if r == n_runs - 1:
                logger.info(
                    f"Attempting to log parameters for Algo: {current_algo_name}, Feature Set: {fs_name}..."
                )
                learned_params = get_bandit_parameters(algo)

                if learned_params is not None and isinstance(learned_params, dict):
                    logger.info(
                        f"--- Learned Parameters (Algo: {current_algo_name}, FS: {fs_name}, Dim: {eff_dim}) ---"
                    )
                    if not learned_params:
                        logger.info("  Parameter dictionary is empty.")
                    else:
                        for model_id, param_vector in learned_params.items():
                            if not isinstance(param_vector, np.ndarray):
                                param_vector = np.array(param_vector)

                            param_str = np.array2string(
                                param_vector,
                                precision=3,
                                suppress_small=True,
                                max_line_width=120,
                            )
                            logger.info(f"  Params for {model_id:<25}: {param_str}")
                    logger.info("--- End Learned Parameters --- ")
                else:
                    logger.info(
                        f"Could not retrieve or parameters not applicable for {current_algo_name} with feature set {fs_name}."
                    )

            if r == n_runs - 1 and learned_params:
                logger.info(
                    f"--- Final Best Model per Unique Context (Algo: {current_algo_name}, FS: {fs_name}) ---"
                )
                num_logged = 0
                max_to_log = 50

                for i, context_tuple in enumerate(unique_context_vectors):
                    if num_logged >= max_to_log:
                        logger.info(
                            f"(Skipping remaining {len(unique_context_vectors) - num_logged} unique contexts in log...)"
                        )
                        break

                    context_vector = np.array(context_tuple)
                    best_model_for_context = None
                    max_pred = -np.inf
                    predictions = {}

                    for model_id in algo.model_ids:
                        if model_id in learned_params:
                            theta_vector = learned_params[model_id]
                            pred = context_vector @ theta_vector
                            predictions[model_id] = pred
                            if pred > max_pred:
                                max_pred = pred
                                best_model_for_context = model_id
                        else:
                            logger.warning(
                                f"Parameters missing for model {model_id} when predicting best model."
                            )
                            predictions[model_id] = -np.inf

                    context_str = np.array2string(
                        context_vector,
                        precision=2,
                        suppress_small=True,
                        max_line_width=80,
                    )
                    logger.info(
                        f"  Context {i+1}: {context_str} -> Best Model: {best_model_for_context} (Max Pred: {max_pred:.3f})"
                    )

                    num_logged += 1
                logger.info("--- End Best Model per Context --- ")

            if r == n_runs - 1 and learned_params:
                logger.info(
                    f"Calculating performance gaps for Algo: {current_algo_name}, FS: {fs_name}..."
                )
                performance_gaps = calculate_performance_gaps(
                    learned_params, unique_context_vectors, algo.model_ids
                )

                if performance_gaps:
                    gaps_df = pd.DataFrame(performance_gaps)
                    logger.info(
                        f"--- Performance Gap Summary (Algo: {current_algo_name}, FS: {fs_name}) ---"
                    )
                    logger.info(f"Mean Gap: {gaps_df['gap'].mean():.4f}")
                    logger.info(f"Max Gap:  {gaps_df['gap'].max():.4f}")
                    logger.info(f"Min Gap:  {gaps_df['gap'].min():.4f}")
                    logger.info(f"Std Dev Gap: {gaps_df['gap'].std():.4f}")
                    top_gaps = gaps_df.nlargest(5, "gap")
                    logger.info("Contexts with Largest Gaps:")
                    for _, row in top_gaps.iterrows():
                        context_str = np.array2string(
                            np.array(row["context"]),
                            precision=2,
                            suppress_small=True,
                            max_line_width=80,
                        )
                        logger.info(
                            f"  Gap: {row['gap']:.3f} (Best: {row['best_model']}={row['best_pred']:.3f}, Avg: {row['avg_pred']:.3f}) for Context: {context_str}"
                        )
                    logger.info("--- End Performance Gap Summary --- ")
                else:
                    logger.info("Could not calculate performance gaps.")
    if not detailed_run_results:
        logger.error("No simulation results generated.")
        return

    results_df = pd.DataFrame(detailed_run_results)

    results_df = results_df.sort_values(by=["run_id", "feature_set", "query_index"])
    results_df["cumulative_regret"] = results_df.groupby(["run_id", "feature_set"])[
        "step_regret"
    ].cumsum()

    final_step_indices = results_df.groupby(["run_id", "feature_set"])[
        "query_index"
    ].idxmax()
    final_step_df = results_df.loc[final_step_indices].copy()

    summary_stats_df = (
        final_step_df.groupby("feature_set")["cumulative_regret"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary_stats_df.rename(
        columns={
            "mean": "mean_final_regret",
            "std": "std_final_regret",
            "min": "min_final_regret",
            "max": "max_final_regret",
        },
        inplace=True,
    )

    logger.info("Renaming feature sets for plots...")
    label_map = {
        "none": "None",
        "task": "Task",
        "cluster": "Cluster",
        "complex": "Complexity",
        "task_cluster": "Task + Cluster",
        "task_complex": "Task + Complexity",
        "cluster_complex": "Cluster + Complexity",
        "full": "Full Features",
    }

    for df in [results_df, final_step_df, summary_stats_df]:
        if "feature_set" in df.columns:
            if not all(
                item in label_map.values() for item in df["feature_set"].unique()
            ):
                df["feature_set"] = (
                    df["feature_set"].map(label_map).fillna(df["feature_set"])
                )
            else:
                logger.warning(
                    "Feature set appears to be already renamed. Skipping map."
                )
        else:
            logger.warning("DataFrame missing 'feature_set' column during rename step.")

    all_sets_renamed = sorted(results_df["feature_set"].unique())
    none_set_new = "None" if "None" in all_sets_renamed else None
    full_set_new = "Full Features" if "Full Features" in all_sets_renamed else None
    single_features_new = sorted(
        [
            s
            for s in all_sets_renamed
            if "+" not in s and s not in ["None", "Full Features"]
        ]
    )
    pair_features_new = sorted(
        [s for s in all_sets_renamed if "+" in s and s != "Full Features"]
    )

    plot_order_new = []
    if none_set_new:
        plot_order_new.append(none_set_new)
    plot_order_new.extend(single_features_new)
    plot_order_new.extend(pair_features_new)

    if full_set_new:
        plot_order_new.append(full_set_new)
    logger.info(f"Plotting order using new labels: {plot_order_new}")

    summary_stats_df["feature_set"] = pd.Categorical(
        summary_stats_df["feature_set"], categories=plot_order_new, ordered=True
    )
    summary_stats_df = summary_stats_df.sort_values("feature_set").dropna(
        subset=["feature_set"]
    )

    final_step_df["feature_set"] = pd.Categorical(
        final_step_df["feature_set"], categories=plot_order_new, ordered=True
    )
    final_step_df = final_step_df.sort_values("feature_set").dropna(
        subset=["feature_set"]
    )

    results_df["feature_set"] = pd.Categorical(
        results_df["feature_set"], categories=plot_order_new, ordered=True
    )
    results_df = results_df.sort_values("feature_set").dropna(subset=["feature_set"])

    logger.info(
        f"Final Cumulative Regret Stats (Plot Order):\n{summary_stats_df.to_string(index=False)}"
    )
    results_handler.save_results(
        results_df, exp_dirs["results"], f"{EXPERIMENT_NAME}_detailed_results"
    )
    results_handler.save_results(
        summary_stats_df, exp_dirs["results"], f"{EXPERIMENT_NAME}_summary_stats"
    )
    plot_utils.setup_plotting()
    plot_title_base = f"{EXPERIMENT_NAME} ({algo_name}, $\\lambda={lambda_weight}$)"  # Base for titles
    logger.info("Generating final cumulative regret box plot...")
    try:
        from .plotting import generate_regret_boxplot

        boxplot_title = f"{plot_title_base}\nFinal Cumulative Regret Distribution"

        generate_regret_boxplot(
            final_step_df, exp_dirs, title=boxplot_title, category_order=plot_order_new
        )
        logger.info(f"Regret box plot saved to {exp_dirs['plots']}")
    except ImportError:
        logger.warning("generate_regret_boxplot not found in plotting.py. Skipping.")
    except Exception as e:
        logger.error(f"Could not generate regret box plot: {e}", exc_info=True)

    logger.info(f"Experiment {EXPERIMENT_NAME} finished.")
    detailed_results_df = pd.DataFrame(detailed_run_results)
    if detailed_results_df.empty:
        logger.error(
            "No simulation results generated. Cannot perform analysis or plotting."
        )
        return

    cumulative_regret_per_run = (
        detailed_results_df.groupby(["run_id", "feature_set"])["step_regret"]
        .sum()
        .reset_index()
    )
    cumulative_regret_per_run = cumulative_regret_per_run.rename(
        columns={"step_regret": "cumulative_regret"}
    )
    if not cumulative_regret_per_run.empty:
        logger.info("--- Performing Statistical Analysis on FULL RUN Regret ---")
        perform_statistical_analysis(
            cumulative_regret_per_run, metric_col="cumulative_regret"
        )
    else:
        logger.warning(
            "Cumulative regret per run data is empty. Skipping full run statistical analysis."
        )

    logger.info("--- Calculating Regret for Second Half of Queries ---")
    if (
        not detailed_results_df.empty
        and "query_index" in detailed_results_df.columns
        and "step_regret" in detailed_results_df.columns
    ):
        max_query_index = detailed_results_df["query_index"].max()
        midpoint_index = (max_query_index + 1) // 2
        logger.info(
            f"Total query steps: {max_query_index + 1}. Analyzing from index {midpoint_index} onwards."
        )

        second_half_df = detailed_results_df[
            detailed_results_df["query_index"] >= midpoint_index
        ].copy()

        if not second_half_df.empty:
            regret_second_half_per_run = (
                second_half_df.groupby(["run_id", "feature_set"])["step_regret"]
                .sum()
                .reset_index()
            )
            regret_second_half_per_run = regret_second_half_per_run.rename(
                columns={"step_regret": "second_half_cumulative_regret"}
            )
            logger.info("--- Performing Statistical Analysis on SECOND HALF Regret ---")
            perform_statistical_analysis(
                regret_second_half_per_run, metric_col="second_half_cumulative_regret"
            )

        else:
            logger.warning(
                "No data found for the second half of queries. Skipping second half statistical analysis."
            )
    else:
        logger.warning(
            "Detailed results DataFrame is empty or missing required columns ('query_index', 'step_regret'). Skipping second half analysis."
        )

    (
        cumulative_regret_per_run.groupby("feature_set")["cumulative_regret"]
        .mean()
        .reset_index()
    )

    final_step_indices = results_df.groupby(["run_id", "feature_set"])[
        "query_index"
    ].idxmax()
    final_step_df = results_df.loc[final_step_indices].copy()

    summary_stats_df = (
        final_step_df.groupby("feature_set")["cumulative_regret"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary_stats_df.rename(
        columns={
            "mean": "mean_final_regret",
            "std": "std_final_regret",
            "min": "min_final_regret",
            "max": "max_final_regret",
        },
        inplace=True,
    )


if __name__ == "__main__":
    main()
