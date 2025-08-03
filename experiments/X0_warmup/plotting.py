import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.shared import plot_utils

logger = logging.getLogger(__name__)

# ============================================================================
# CUMULATIVE REGRET PLOTTING
# ============================================================================


def generate_cumulative_regret_plot(detailed_results_df: pd.DataFrame, exp_dirs: dict):
    """Generates the cumulative regret plot for 00 with 95% CI shading."""
    logger.info("Generating 00 cumulative regret plot with 95% CI...")

    # --- Data Validation ---
    if detailed_results_df.empty:
        logger.warning(
            "Detailed results DataFrame is empty. Cannot generate regret plot."
        )
        return
    required_cols = ["run_id", "algorithm", "query_index", "cumulative_regret"]
    if not all(col in detailed_results_df.columns for col in required_cols):

        if all(
            col in detailed_results_df.columns
            for col in ["algorithm", "query_index", "cumulative_regret"]
        ):

            logger.warning(
                "Input data seems to be from a single run (missing 'run_id'). Plotting without CI."
            )
            fig, ax = plt.subplots()
            algorithms = detailed_results_df["algorithm"].unique()
            for algo_name in algorithms:
                algo_data = detailed_results_df[
                    detailed_results_df["algorithm"] == algo_name
                ].sort_values("query_index")
                ax.plot(
                    algo_data["query_index"],
                    algo_data["cumulative_regret"],
                    label=algo_name,
                    color=plot_utils.get_color(algo_name),
                    linewidth=2,
                )
            ax.axhline(
                0, color="grey", linestyle="--", linewidth=1, label="Zero Regret"
            )
            ax.set_xlabel("Query Index (Time Step)")
            ax.set_ylabel("Cumulative Regret")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            plot_filename = f"{exp_dirs['base'].name}_cumulative_regret_single_run.png"
            plot_utils.save_plot(fig, exp_dirs["plots"], plot_filename)
            return
        else:
            logger.error(
                f"Missing required columns for regret plot (multi-run): {required_cols}"
            )
            return

    # --- Plot Setup and Data Processing ---
    fig, ax = plt.subplots()

    algorithms = detailed_results_df["algorithm"].unique()
    query_indices = detailed_results_df["query_index"].unique()
    query_indices.sort()

    for algo_name in algorithms:
        logger.info(f"  Processing algorithm: {algo_name}")
        algo_data = detailed_results_df[detailed_results_df["algorithm"] == algo_name]
        grouped = algo_data.groupby("query_index")["cumulative_regret"]

        mean_regret = grouped.mean().reindex(query_indices).ffill()
        std_regret = grouped.std().reindex(query_indices).ffill().fillna(0)
        run_counts = grouped.count().reindex(query_indices).ffill().fillna(0)

        sem = np.divide(
            std_regret,
            np.sqrt(run_counts),
            out=np.zeros_like(std_regret),
            where=run_counts > 0,
        )
        confidence_interval = 1.96 * sem

        lower_bound = mean_regret - confidence_interval
        upper_bound = mean_regret + confidence_interval
        color = plot_utils.get_color(algo_name)
        ax.plot(
            query_indices,
            mean_regret,
            label=f"{algo_name} (Mean)",
            color=color,
            linewidth=2,
        )
        ax.fill_between(
            query_indices,
            lower_bound,
            upper_bound,
            color=color,
            alpha=0.2,
            label=f"{algo_name} (95% CI)",
        )

    # --- Plot Styling and Formatting ---
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.titlesize": 12,
        }
    )
    ax.set_xlabel("Query Index (Time Step)")
    ax.set_ylabel("Cumulative Regret")
    ax.grid(True, linestyle="--", alpha=0.6)

    # --- Legend Filtering and Customization ---
    handles, labels = ax.get_legend_handles_labels()
    filtered_handles = []
    filtered_labels = []
    desired_algorithms = ["random", "epsilon_greedy"]
    display_name_map = {"random": "Random", "epsilon_greedy": "Epsilon Greedy (NC)"}

    for handle, label in zip(handles, labels):

        original_algo_name = None
        if label.endswith(" (Mean)"):
            original_algo_name = label.replace(" (Mean)", "")

        if original_algo_name in desired_algorithms:
            filtered_handles.append(handle)
            display_name = display_name_map.get(original_algo_name, original_algo_name)
            filtered_labels.append(display_name)
    ax.legend(filtered_handles, filtered_labels, title=None, frameon=True)

    # --- Save Plot ---
    plot_filename = f"{exp_dirs['base'].name}_cumulative_regret_with_ci.png"
    plot_utils.save_plot(fig, exp_dirs["plots"], plot_filename)


# ============================================================================
# MODEL SELECTION HEATMAP
# ============================================================================


def generate_model_selection_heatmap(detailed_results_df: pd.DataFrame, exp_dirs: dict):
    """Generates a heatmap showing Epsilon-Greedy model selection frequency per run."""
    logger.info("Generating Epsilon-Greedy model selection heatmap...")

    # --- Data Validation ---
    if detailed_results_df.empty:
        logger.warning("Detailed results DataFrame is empty. Cannot generate heatmap.")
        return
    required_cols = ["run_id", "algorithm", "chosen_model"]
    if not all(col in detailed_results_df.columns for col in required_cols):
        logger.error(f"Missing required columns for heatmap: {required_cols}")
        return

    # --- Data Preparation ---
    eg_results = detailed_results_df[
        detailed_results_df["algorithm"] == "epsilon_greedy"
    ].copy()
    if eg_results.empty:
        logger.warning("No Epsilon-Greedy results found to generate heatmap.")
        return
    selection_freq = (
        eg_results.groupby("run_id")["chosen_model"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    selection_freq.index = selection_freq.index + 1
    selection_freq = selection_freq.reindex(sorted(selection_freq.columns), axis=1)
    # --- Import Dependencies ---
    try:
        import seaborn as sns
    except ImportError:
        logger.error(
            "Seaborn library is required for heatmap generation. Please install it."
        )
        return

    # --- Plot Configuration ---
    fig_height = max(4, len(selection_freq) * 0.5)
    fig_width = max(6, len(selection_freq.columns) * 1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # --- Heatmap Generation ---
    sns.heatmap(
        selection_freq,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"label": "Selection Frequency"},
        ax=ax,
    )

    # --- Plot Styling ---
    ax.set_xlabel("Model ID")
    ax.set_ylabel("Run ID")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # --- Save Plot ---
    plot_filename = f"{exp_dirs['base'].name}_egreedy_selection_heatmap.png"
    plot_utils.save_plot(fig, exp_dirs["plots"], plot_filename)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Regenerate plots from existing experiment data."""
    # Set up paths and get timestamp
    base_exp_dir = Path("experiments/X0_warmup")
    exp_dirs = plot_utils.setup_experiment_directories(base_exp_dir)
    
    timestamp, success = plot_utils.get_experiment_timestamp(
        "X0_warmup", 
        exp_dirs["results"], 
        "*_detailed_results_all_runs_*.csv"
    )
    if not success:
        return
    
    # Load detailed results
    detailed_results_file = exp_dirs["results"] / f"X0_warmup_detailed_results_all_runs_{timestamp}.csv"
    if not detailed_results_file.exists():
        logger.error(f"Detailed results file not found: {detailed_results_file}")
        return
    
    detailed_results_df = pd.read_csv(detailed_results_file)
    
    # Generate plots
    generate_cumulative_regret_plot(detailed_results_df, exp_dirs)
    generate_model_selection_heatmap(detailed_results_df, exp_dirs)
    
    logger.info("Plotting complete!")


if __name__ == "__main__":
    main()
