import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.shared import plot_utils

# Logger configuration
logger = logging.getLogger(__name__)

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def generate_reward_barplot(
    summary_stats: pd.DataFrame,
    exp_dirs: dict,
    title: str = "Average Performance by Feature Set",
):
    """Generates a bar plot comparing average reward across feature sets."""
    logger.info("Generating Feature Ablation reward bar plot...")

    if summary_stats.empty:
        logger.warning(
            "Summary statistics DataFrame is empty. Cannot generate bar plot."
        )
        return

    # Check for either mean_reward or mean_final_regret
    if "mean_reward" in summary_stats.columns:
        metric_col = "mean_reward"
        ylabel = "Mean Reward"
    elif "mean_final_regret" in summary_stats.columns:
        metric_col = "mean_final_regret"
        ylabel = "Mean Final Regret"
    else:
        logger.error("Neither mean_reward nor mean_final_regret found in summary stats")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [plot_utils.get_color(fs) for fs in summary_stats["feature_set"]]
    bars = ax.bar(
        summary_stats["feature_set"],
        summary_stats[metric_col],
        color=colors,
        alpha=0.8,
    )

    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("Feature Set")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    if len(summary_stats["feature_set"]) > 5:
        plt.xticks(rotation=30, ha="right")
    else:
        plt.xticks(rotation=0)
    min_val = 0
    max_val = summary_stats[metric_col].max()
    padding = max_val * 0.1
    ax.set_ylim(min_val, max_val + padding)

    plt.tight_layout()

    plot_filename = f"{exp_dirs['base'].name}_performance_barplot.png"
    plot_utils.save_plot(fig, exp_dirs["plots"], plot_filename)


def generate_regret_boxplot(
    final_regret_df: pd.DataFrame,
    exp_dirs: dict,
    title: str = "Final Cumulative Regret Distribution",
    category_order: list = None,
):
    """Generates a box plot showing the distribution of final cumulative regret per feature set."""

    logger.info("Generating Feature Ablation final cumulative regret box plot...")

    if final_regret_df.empty:
        logger.warning("Final regret DataFrame is empty. Cannot generate box plot.")
        return

    required_cols = ["feature_set", "cumulative_regret"]
    if not all(col in final_regret_df.columns for col in required_cols):
        logger.error(f"Missing required columns for regret box plot: {required_cols}")
        return
    if category_order:
        plot_order = category_order
        if not pd.api.types.is_categorical_dtype(final_regret_df["feature_set"]):
            final_regret_df["feature_set"] = pd.Categorical(
                final_regret_df["feature_set"], categories=plot_order, ordered=True
            )
        final_regret_df = final_regret_df.sort_values("feature_set")
    else:
        logger.warning(
            "No category_order provided to generate_regret_boxplot. Using alphabetical order."
        )
        plot_order = sorted(final_regret_df["feature_set"].unique())
        if not pd.api.types.is_categorical_dtype(final_regret_df["feature_set"]):
            final_regret_df["feature_set"] = pd.Categorical(
                final_regret_df["feature_set"], categories=plot_order, ordered=True
            )
            final_regret_df = final_regret_df.sort_values("feature_set")
    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = {
        "none": "#d3d3d3",
        "single": "#1f77b4",
        "double": "#ff7f0e",
        "full": "#2ca02c",
    }

    palette = []
    for fs in plot_order:
        if fs == "none":
            palette.append(color_map["none"])
        elif fs == "full":
            palette.append(color_map["full"])
        elif "_" in fs:
            palette.append(color_map["double"])
        else:
            palette.append(color_map["single"])

    sns.boxplot(
        data=final_regret_df,
        x="feature_set",
        y="cumulative_regret",
        hue="feature_set",
        order=plot_order,
        palette=palette,
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("Feature Set Used")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plot_filename = f"{exp_dirs['base'].name}_final_regret_boxplot.png"
    plot_utils.save_plot(fig, exp_dirs["plots"], plot_filename)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Regenerate plots from existing experiment data."""
    # Get timestamp from command line or use latest
    base_exp_dir = Path("experiments/02_feature_ablation")
    results_dir = base_exp_dir / "results"
    
    if len(sys.argv) > 1:
        timestamp = sys.argv[1]
    else:
        files = sorted(results_dir.glob("*_detailed_results_*.csv"))
        if not files:
            logger.error("No results found. Run the experiment first.")
            return
        timestamp = "_".join(files[-1].stem.split("_")[-2:])
    
    logger.info(f"Loading results from: {timestamp}")
    
    # Set up directories
    exp_dirs = {
        "base": base_exp_dir,
        "results": results_dir,
        "plots": base_exp_dir / "plots"
    }
    exp_dirs["plots"].mkdir(exist_ok=True)
    
    # Load detailed results
    detailed_results_file = results_dir / f"02_feature_ablation_detailed_results_{timestamp}.csv"
    if not detailed_results_file.exists():
        logger.error(f"Detailed results file not found: {detailed_results_file}")
        return
    
    # Load and process data the same way as run_experiment.py
    detailed_df = pd.read_csv(detailed_results_file, keep_default_na=False)
    
    # Calculate cumulative regret if not present
    if 'cumulative_regret' not in detailed_df.columns:
        detailed_df['cumulative_regret'] = detailed_df.groupby(['run_id', 'feature_set'])['step_regret'].cumsum()
    
    # Get final step for each run/feature_set combination
    final_step_indices = detailed_df.groupby(['run_id', 'feature_set'])['query_index'].idxmax()
    final_step_df = detailed_df.loc[final_step_indices].copy()
    
    # Replace feature set names to match original
    fs_mapping = {
        "none": "None",
        "task": "Task",
        "cluster": "Cluster",
        "complexity": "Complexity",
        "task_cluster": "Task + Cluster",
        "task_complex": "Task + Complexity",
        "cluster_complex": "Cluster + Complexity",
        "full": "Full Features",
        "None": "No Features"  # Handle the special case
    }
    
    # Apply mapping
    final_step_df['feature_set'] = final_step_df['feature_set'].replace(fs_mapping)
    
    # Define plot order
    plot_order = [
        "No Features",
        "Cluster",
        "Complexity", 
        "Task",
        "Cluster + Complexity",
        "Task + Cluster",
        "Task + Complexity",
        "Full Features"
    ]
    
    # Setup plotting style
    plot_utils.setup_plotting()
    
    # Construct plot order dynamically like the original
    all_sets = sorted(final_step_df["feature_set"].unique())
    none_set = "No Features" if "No Features" in all_sets else None
    full_set = "Full Features" if "Full Features" in all_sets else None
    
    single_features = sorted([
        s for s in all_sets 
        if "+" not in s and s not in ["No Features", "Full Features"]
    ])
    pair_features = sorted([
        s for s in all_sets 
        if "+" in s and s != "Full Features"
    ])
    
    plot_order_new = []
    if none_set:
        plot_order_new.append(none_set)
    plot_order_new.extend(single_features)
    plot_order_new.extend(pair_features)
    if full_set:
        plot_order_new.append(full_set)
    
    # Set categorical ordering
    final_step_df["feature_set"] = pd.Categorical(
        final_step_df["feature_set"], categories=plot_order_new, ordered=True
    )
    final_step_df = final_step_df.sort_values("feature_set").dropna(subset=["feature_set"])
    
    # Generate boxplot with title
    plot_title = "02_feature_ablation (epsilon_greedy, λ=0.5)\nFinal Cumulative Regret Distribution"
    generate_regret_boxplot(final_step_df, exp_dirs, title=plot_title, category_order=plot_order_new)
    
    logger.info("Plotting complete!")


if __name__ == "__main__":
    main()
