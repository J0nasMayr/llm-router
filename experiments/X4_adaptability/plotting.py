# experiments/X4_adaptability/plotting.py

# --- Imports ---
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from experiments.shared import config_loader

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


# --- Cumulative Regret Plotting ---
def generate_cumulative_regret_plot(
    results_df, exp_dirs, title=None, change_point=None
):
    """Generates a plot of cumulative regret over time for the adaptability experiment."""
    # --- Input Validation ---
    if results_df.empty:
        logger.warning("Results DataFrame is empty. Skipping cumulative regret plot.")
        return

    # --- Output Directory Setup ---
    output_dir = exp_dirs["plots"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data Preparation ---
    # Calculate cumulative regret per run if not already present
    if "cumulative_regret" not in results_df.columns:
        results_df = results_df.sort_values(by=["run_id", "query_index"])
        results_df["cumulative_regret"] = results_df.groupby("run_id")[
            "step_regret"
        ].cumsum()

    # --- Plot Creation ---
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=results_df,
        x="query_index",
        y="cumulative_regret",
        errorbar=("ci", 95),  # Show 95% confidence interval
        legend=False,
    )  # Assuming only one algorithm runs per execution

    # --- Change Point Annotation ---
    # Add vertical line for change point
    if change_point is not None:
        plt.axvline(
            x=change_point,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Adaptation Point ({change_point})",
        )
        plt.legend()

    # --- Plot Styling ---
    plt.xlabel("Query Index")
    plt.ylabel("Mean Cumulative Regret")
    if title:
        # plt.title(title, fontsize=14)
        pass
    else:
        # plt.title("Mean Cumulative Regret Over Time", fontsize=14)
        pass
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # --- Plot Saving ---
    filename = output_dir / "X4_adaptability_cumulative_regret.png"
    plt.savefig(filename, dpi=300)
    logger.info(f"Saved cumulative regret plot to {filename}")
    plt.close()


# --- Model Selection Frequency Plotting ---
def generate_model_selection_heatmap(
    results_df,
    exp_dirs,
    title=None,
    change_point=None,
    model_added=None,
    model_removed=None,
    all_models_ever=None,
    bin_size=25,  # Number of queries per bin
):
    """
    Generates a heatmap showing model selection frequency over time.
    This provides a cleaner alternative to the stacked area chart for better readability.
    """
    # --- Input Validation ---
    if results_df.empty:
        logger.warning("Results DataFrame is empty. Skipping model selection heatmap.")
        return

    # --- Output Directory Setup ---
    output_dir = exp_dirs["plots"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Column Validation ---
    if (
        "chosen_model" not in results_df.columns
        or "query_index" not in results_df.columns
    ):
        logger.error(
            "Required columns ('chosen_model', 'query_index') not found. Skipping model selection heatmap."
        )
        return

    # --- Data Binning ---
    max_query_index = results_df["query_index"].max()
    bins = np.arange(0, max_query_index + bin_size, bin_size)
    labels = bins[:-1]
    results_df["query_bin"] = pd.cut(
        results_df["query_index"], bins=bins, labels=labels, right=False
    )

    # --- Frequency Calculation ---
    selection_counts = (
        results_df.groupby(["run_id", "query_bin", "chosen_model"])
        .size()
        .unstack(fill_value=0)
    )
    mean_selection_counts = selection_counts.groupby(level="query_bin").mean()
    mean_selection_freq = mean_selection_counts.apply(lambda x: x / x.sum(), axis=1)

    # --- Data Preparation for Plotting ---
    if all_models_ever:
        for model in all_models_ever:
            if model not in mean_selection_freq.columns:
                mean_selection_freq[model] = 0.0
        # Sort models by average selection frequency (most used first)
        model_avg_freq = mean_selection_freq.mean().sort_values(ascending=False)
        mean_selection_freq = mean_selection_freq[model_avg_freq.index]

    # --- Plot Creation ---
    fig, ax = plt.subplots(figsize=(16, max(8, len(mean_selection_freq.columns) * 0.4)))

    # --- Heatmap Generation ---
    # Transpose for better visualization (models on y-axis, time on x-axis)
    heatmap_data = mean_selection_freq.T
    
    # Create heatmap with custom colormap for better contrast
    # Using 'plasma' colormap which provides better differentiation for low values
    # Alternative options: 'viridis', 'plasma', 'inferno', 'RdYlBu_r', 'coolwarm'
    im = ax.imshow(
        heatmap_data.values,
        cmap='RdYlBu_r',
        aspect='auto',
        interpolation='nearest',
        vmin=0,
        vmax=1
    )

    # --- Colorbar ---
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Selection Frequency', fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    # --- Axis Configuration ---
    ax.set_xticks(range(0, len(heatmap_data.columns), max(1, len(heatmap_data.columns) // 10)))
    ax.set_xticklabels([str(int(heatmap_data.columns[i])) for i in ax.get_xticks() if i < len(heatmap_data.columns)], 
                       rotation=45, ha='right', fontsize=14)
    
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=14)
    
    # --- Change Point Annotation ---
    if change_point is not None:
        change_bin = change_point // bin_size
        if change_bin < len(heatmap_data.columns):
            ax.axvline(x=change_bin, color='black', linestyle='--', linewidth=2, alpha=0.9)
            ax.text(change_bin, -0.8, f'Adaptation Point', 
                   ha='center', va='top', fontsize=14, color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

    # --- Labels and Styling ---
    ax.set_xlabel(f'Query Epoch (Window Size = {bin_size})', fontsize=14)
    ax.set_ylabel('Models', fontsize=14)
    ax.grid(False)  # Turn off grid for cleaner heatmap
    
    if title:
        ax.set_title(title, fontsize=14, pad=20)

    # --- Layout and Saving ---
    plt.tight_layout()
    filename = output_dir / "a8_model_selection_heatmap.pdf"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    logger.info(f"Saved model selection heatmap to {filename}")
    plt.close()


def generate_model_selection_plot(
    results_df,
    exp_dirs,
    title=None,
    change_point=None,
    model_added=None,
    model_removed=None,
    all_models_ever=None,
    bin_size=25,  # Number of queries per bin
):
    """Generates a plot showing model selection frequency over time."""
    # --- Input Validation ---
    if results_df.empty:
        logger.warning("Results DataFrame is empty. Skipping model selection plot.")
        return

    # --- Output Directory Setup ---
    output_dir = exp_dirs["plots"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Column Validation ---
    if (
        "chosen_model" not in results_df.columns
        or "query_index" not in results_df.columns
    ):
        logger.error(
            "Required columns ('chosen_model', 'query_index') not found. Skipping model selection plot."
        )
        return

    # --- Data Binning ---
    # Determine the bins
    max_query_index = results_df["query_index"].max()
    bins = np.arange(0, max_query_index + bin_size, bin_size)
    labels = bins[:-1]  # Label bins by their starting index
    results_df["query_bin"] = pd.cut(
        results_df["query_index"], bins=bins, labels=labels, right=False
    )

    # --- Frequency Calculation ---
    # Calculate selection counts per bin per run
    selection_counts = (
        results_df.groupby(["run_id", "query_bin", "chosen_model"])
        .size()
        .unstack(fill_value=0)
    )

    # Average counts across runs
    mean_selection_counts = selection_counts.groupby(level="query_bin").mean()

    # Calculate frequencies (proportions)
    mean_selection_freq = mean_selection_counts.apply(lambda x: x / x.sum(), axis=1)

    # --- Data Preparation for Plotting ---
    # Prepare for plotting - ensure all models ever available are columns
    if all_models_ever:
        for model in all_models_ever:
            if model not in mean_selection_freq.columns:
                mean_selection_freq[model] = 0.0
        # Sort columns for consistent legend order (optional but good practice)
        mean_selection_freq = mean_selection_freq[list(all_models_ever)]

    # --- Plot Creation ---
    plt.figure(figsize=(15, 8))

    # --- Color Mapping ---
    # Use a colormap suitable for many categories
    # Example: Tab20 or generate custom colors
    num_models = len(mean_selection_freq.columns)
    colors = (
        plt.cm.get_cmap("tab20", num_models)
        if num_models <= 20
        else plt.cm.get_cmap("viridis", num_models)
    )

    # --- Area Chart Plotting ---
    # Plot as stacked area chart
    mean_selection_freq.plot(kind="area", stacked=True, colormap=colors, ax=plt.gca())

    # --- Change Point Annotation ---
    # Add vertical line for change point
    if change_point is not None:
        plt.axvline(
            x=change_point / bin_size,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Adaptation Point ({change_point})",
        )

    # --- Plot Styling ---
    plt.xlabel(f"Query Epoch (Bin Size = {bin_size})")
    plt.ylabel("Mean Selection Frequency")
    if title:
        # plt.title(title, fontsize=14)
        pass
    else:
        # plt.title("Model Selection Frequency Over Time", fontsize=14)
        pass

    # --- Legend and Grid Styling ---
    # Adjust legend position
    plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", axis="y", alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
    plt.ylim(0, 1)  # Ensure y-axis is fixed from 0 to 1

    # --- Axis Tick Adjustment ---
    # Adjust x-axis ticks/labels if needed (bins represent epochs)
    tick_positions = np.arange(len(mean_selection_freq.index))
    tick_labels = mean_selection_freq.index.astype(str)
    plt.xticks(
        ticks=tick_positions[:: max(1, len(tick_positions) // 10)],
        labels=tick_labels[:: max(1, len(tick_positions) // 10)],
        rotation=45,
        ha="right",
    )

    # --- Plot Saving ---
    filename = output_dir / "X4_adaptability_model_selection_frequency.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    logger.info(f"Saved model selection frequency plot to {filename}")
    plt.close()


# --- Model Choice Timeline Plotting ---
def generate_model_choice_timeline(
    results_df,
    exp_dirs,
    title=None,
    change_point=None,
    model_added=None,
    model_removed=None,
    all_models_ever=None,
):
    """Generates a timeline plot showing individual model choices over time."""
    # --- Input Validation ---
    if results_df.empty:
        logger.warning(
            "Results DataFrame is empty. Skipping model choice timeline plot."
        )
        return

    # --- Output Directory Setup ---
    output_dir = exp_dirs["plots"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Column Validation ---
    if (
        "chosen_model" not in results_df.columns
        or "query_index" not in results_df.columns
    ):
        logger.error(
            "Required columns ('chosen_model', 'query_index') not found. Skipping timeline plot."
        )
        return

    # --- Model Order Determination ---
    # Use all models that were ever chosen or specified for consistent Y-axis
    if all_models_ever:
        model_order = sorted(list(all_models_ever))
    else:
        model_order = sorted(results_df["chosen_model"].unique())

    # --- Color Assignment ---
    # Assign colors
    num_models = len(model_order)
    colors = (
        plt.cm.get_cmap("tab20", num_models)
        if num_models <= 20
        else plt.cm.get_cmap("viridis", num_models)
    )
    model_colors = {model: colors(i) for i, model in enumerate(model_order)}

    # --- Plot Creation ---
    plt.figure(
        figsize=(18, max(6, num_models * 0.4))
    )  # Adjust height based on num models

    # --- Scatter Plot Generation ---
    # Plot choices for each model
    # Using scatter can be slow for many points, consider alternatives if performance is an issue
    # For loop is clearer for assigning colors and labels initially
    plotted_models = set()
    for model in model_order:
        model_choices = results_df[results_df["chosen_model"] == model]
        if not model_choices.empty:
            plt.scatter(
                model_choices["query_index"],
                model_choices["chosen_model"],
                color=model_colors[model],
                label=model,
                s=5,  # Small marker size
                marker="|",
            )  # Use vertical line marker
            plotted_models.add(model)

    # --- Legend Construction ---
    # Ensure all models appear in the legend even if not chosen
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles, new_labels = [], []
    for model in model_order:
        if model in plotted_models:
            idx = labels.index(model)
            new_handles.append(handles[idx])
            new_labels.append(labels[idx])
        else:  # Add a dummy handle for models never chosen
            new_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="|",
                    color=model_colors[model],
                    linestyle="None",
                    markersize=5,
                )
            )
            new_labels.append(model)
    plt.legend(
        new_handles,
        new_labels,
        title="Models",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        markerscale=2,
    )

    # --- Change Point Annotation ---
    # Add vertical line for change point
    if change_point is not None:
        plt.axvline(
            x=change_point,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Adaptation Point ({change_point})",
        )
        # Add the legend entry for the line separately if needed, or rely on plot legend

    # --- Plot Styling ---
    plt.xlabel("Query Index")
    plt.ylabel("Chosen Model")
    if title:
        # plt.title(title, fontsize=14)
        pass
    else:
        # plt.title("Model Choice Timeline", fontsize=14)
        pass

    # --- Axis and Grid Styling ---
    plt.yticks(model_order)  # Ensure y-axis labels are the model names
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout for legend

    # Optional: Improve x-axis tick readability if needed
    # plt.xticks(rotation=45)

    # --- Plot Saving ---
    filename = output_dir / "X4_adaptability_model_choice_timeline.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    logger.info(f"Saved model choice timeline plot to {filename}")
    plt.close()


def main():
    """Regenerate plots from existing experiment data."""
    # Get results directory and timestamp
    results_dir = Path(__file__).parent / "results"
    if len(sys.argv) > 1:
        timestamp = sys.argv[1]
    else:
        files = sorted(results_dir.glob("*_detailed_results_*.csv"))
        if not files:
            logger.error("No results found. Run the experiment first.")
            return
        timestamp = "_".join(files[-1].stem.split("_")[-2:])

    logger.info(f"Loading results from: {timestamp}")
    exp_dirs = config_loader.setup_experiment_dirs("X4_adaptability")

    # Load detailed results
    results_df = pd.read_csv(
        results_dir / f"X4_adaptability_detailed_results_{timestamp}.csv"
    )

    # Load experiment config to get parameters
    config_path = Path(__file__).parent.parent / "config" / "experiments.yaml"
    with open(config_path, "r") as f:
        experiments_config = yaml.safe_load(f)
    exp_config = experiments_config["X4_adaptability"]

    # Extract parameters for plot titles
    algo_name = exp_config.get("active_algorithm", "Unknown")
    lambda_weight = exp_config.get("lambda_weight", 0)
    change_point_idx = exp_config.get("change_point_query_index", None)
    model_to_add = exp_config.get("model_to_add")
    model_to_remove = exp_config.get("model_to_remove")

    # Get all models that were ever available
    (
        set(
            results_df[results_df["query_index"] < change_point_idx][
                "chosen_model"
            ].unique()
        )
        if change_point_idx
        else set()
    )
    all_models_ever = tuple(sorted(set(results_df["chosen_model"].unique())))

    # Generate plots
    logger.info("Generating plots...")

    # Generate model selection frequency plot only
    selection_plot_title = f"X4_adaptability ({algo_name}, $\\lambda={lambda_weight}$)\nModel Selection Frequency Over Time"
    generate_model_selection_plot(
        results_df,
        exp_dirs,
        title=selection_plot_title,
        change_point=change_point_idx,
        model_added=model_to_add,
        model_removed=model_to_remove,
        all_models_ever=all_models_ever,
    )

    # Generate model selection heatmap (new cleaner representation)
    heatmap_plot_title = f"Model Selection Heatmap"
    generate_model_selection_heatmap(
        results_df,
        exp_dirs,
        title=heatmap_plot_title,
        change_point=change_point_idx,
        model_added=model_to_add,
        model_removed=model_to_remove,
        all_models_ever=all_models_ever,
    )

    logger.info(f"All plots saved to: {exp_dirs['plots']}")


if __name__ == "__main__":
    main()
