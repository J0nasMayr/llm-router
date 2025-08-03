# ============================================================================
# PLOTTING FUNCTIONS FOR ALGORITHM COMPARISON
# ============================================================================

import yaml
import logging
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import EngFormatter

from experiments.shared import (
    analysis_utils,
    config_loader,
    data_loader,
    plot_utils,
    results_handler,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

KNOWN_BASELINE_NAMES = ["random", "largest", "smallest", "accuracy"]
ALGO_DISPLAY_NAMES = {
    "epsilon_greedy": "ε‑Greedy (NC)",
    "linear_epsilon_greedy": "ε‑Greedy (C)",
    "linucb": "LinUCB (C)",
    "thompson_sampling": "Thompson Sampling (C)",
    "random": "Random",
    "largest": "Largest",
    "smallest": "Smallest",
    "accuracy": "Accuracy",
}
DEFAULT_PALETTE = {name: plot_utils.get_color(name) for name in ALGO_DISPLAY_NAMES}

DEFAULT_DASHES = {
    "epsilon_greedy": "",
    "linear_epsilon_greedy": (4, 1.5),
    "linucb": (1, 1),
    "thompson_sampling": (5, 1, 1, 1),
    "random": (2, 2),
    "largest": (5, 5),
    "smallest": (8, 3),
    "accuracy": (5, 1, 1, 1, 1, 1),
}
CONTEXTUAL_BANDITS = ["linear_epsilon_greedy", "linucb", "thompson_sampling"]
NON_CONTEXTUAL_BANDITS = ["epsilon_greedy"]

# ============================================================================
# STYLING UTILITIES
# ============================================================================


def get_regret_plot_styles(algorithms_to_plot: list) -> (dict, dict):
    """Generates color and dash styles specifically for regret plots."""
    palette = {}
    dashes = {}
    baseline_colors = {
        "random": plot_utils.get_color("random"),
        "largest": plot_utils.get_color("largest"),
        "smallest": plot_utils.get_color("smallest"),
        "accuracy": plot_utils.get_color("accuracy"),
    }
    for name in KNOWN_BASELINE_NAMES:
        if name in algorithms_to_plot:
            display_name = ALGO_DISPLAY_NAMES.get(name, name)
            palette[display_name] = baseline_colors.get(name, "black")
            dashes[display_name] = ""  # Solid line

    nc_color = plot_utils.get_color("epsilon_greedy")
    nc_dash = (2, 2)
    for name in NON_CONTEXTUAL_BANDITS:
        if name in algorithms_to_plot:
            display_name = ALGO_DISPLAY_NAMES.get(name, name)
            palette[display_name] = nc_color
            dashes[display_name] = nc_dash

    contextual_base_colors = {
        "linear_epsilon_greedy": plot_utils.get_color("linear_epsilon_greedy"),
        "linucb": plot_utils.get_color("linucb"),
        "thompson_sampling": "#808000",  # <<< Changed to Olive Green for visibility >>>
    }
    contextual_dashes = {
        "linear_epsilon_greedy": (4, 1.5),
        "linucb": (1, 1),
        "thompson_sampling": (5, 1, 1, 1),
    }
    for name in CONTEXTUAL_BANDITS:
        if name in algorithms_to_plot:
            display_name = ALGO_DISPLAY_NAMES.get(name, name)
            palette[display_name] = contextual_base_colors.get(name, "#029e73")
            dashes[display_name] = contextual_dashes.get(name, (3, 1, 1, 1))

    return palette, dashes


# ============================================================================
# PARETO PLOTS
# ============================================================================


def generate_pareto_plot(
    a5_summary_df: pd.DataFrame,
    all_norm_results_df: pd.DataFrame,
    exp_dirs: dict,
    models_df: pd.DataFrame = None,
):
    """
    Generates the Pareto plot comparing baselines/models with the tuned algorithm results.

    Args:
        a5_summary_df (pd.DataFrame): Aggregated results from algorithm comparison (Algorithm, Mean Accuracy, Mean Energy).
                                      This MUST contain results for BOTH A5 algorithms AND A1 baselines simulated within A5.
        all_norm_results_df (pd.DataFrame): DataFrame containing normalized results for **all individual A1 models** (for background/frontier).
        exp_dirs (dict): Experiment directories.
        models_df (pd.DataFrame, optional): DataFrame containing model metadata (like parameter count). Defaults to None.
    """
    logger.info("Generating Algorithm Comparison Pareto plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    all_processed_points = []
    if all_norm_results_df is not None and not all_norm_results_df.empty:
        required_cols = ["model_id", "accuracy_norm", "energy_per_token_norm"]
        if all(col in all_norm_results_df.columns for col in required_cols):
            model_avg_perf = (
                all_norm_results_df.groupby("model_id")[
                    ["accuracy_norm", "energy_per_token_norm"]
                ]
                .mean()
                .reset_index()
            )
            logger.info(
                f"Calculated average performance for {len(model_avg_perf)} individual A1 models."
            )
            for _, row in model_avg_perf.iterrows():
                if row["model_id"] not in KNOWN_BASELINE_NAMES:
                    all_processed_points.append(
                        {
                            "Name": row["model_id"],
                            "Mean Normalized Accuracy": row["accuracy_norm"],
                            "Mean Normalized Energy": row["energy_per_token_norm"],
                            "Type": "A1 Model",
                        }
                    )
        else:
            logger.warning(
                f"'all_norm_results_df' missing required columns {required_cols}. Cannot plot individual A1 models."
            )
    else:
        logger.warning(
            "'all_norm_results_df' not provided or empty. Cannot plot individual A1 models."
        )
    if a5_summary_df is not None and not a5_summary_df.empty:
        required_a5_cols = [
            "Algorithm",
            "Mean Normalized Accuracy",
            "Mean Normalized Energy",
        ]
        if all(col in a5_summary_df.columns for col in required_a5_cols):
            for _, row in a5_summary_df.iterrows():
                algo_name = row["Algorithm"]
                point_type = (
                    "A1 Baseline"
                    if algo_name in KNOWN_BASELINE_NAMES
                    else "A5 Algorithm"
                )
                all_processed_points.append(
                    {
                        "Name": algo_name,
                        "Mean Normalized Accuracy": row["Mean Normalized Accuracy"],
                        "Mean Normalized Energy": row["Mean Normalized Energy"],
                        "Type": point_type,
                    }
                )
            logger.info(
                f"Processed {len(a5_summary_df)} strategies from a5_summary_df."
            )
        else:
            logger.warning(
                f"A5 summary DataFrame missing required columns {required_a5_cols}. Cannot process A5 results."
            )
    else:
        logger.warning(
            "A5 summary DataFrame not provided or empty. Skipping A5 algorithms and A1 baselines from A5 sim."
        )
    if not all_processed_points:
        logger.error("No data points processed from any source. Cannot generate plot.")
        plt.close()
        return

    combined_df = pd.DataFrame(all_processed_points)
    logger.info(
        f"Value counts of 'Type' in combined_df BEFORE dropna:\n{combined_df['Type'].value_counts()}"
    )
    logger.info(f"Head of combined_df BEFORE dropna:\n{combined_df.head().to_string()}")
    baseline_rows_before_drop = combined_df[combined_df["Type"] == "A1 Baseline"]
    if not baseline_rows_before_drop.empty:
        nan_check_baselines = (
            baseline_rows_before_drop[
                ["Mean Normalized Accuracy", "Mean Normalized Energy"]
            ]
            .isnull()
            .sum()
        )
        if nan_check_baselines.sum() > 0:
            logger.warning(
                f"NaN values found in baseline rows BEFORE dropna:\n{nan_check_baselines}"
            )
        else:
            logger.info("No NaNs found in baseline rows BEFORE dropna.")
    else:
        logger.info("No baseline rows found BEFORE dropna.")

    combined_df.dropna(
        subset=["Mean Normalized Accuracy", "Mean Normalized Energy"], inplace=True
    )

    logger.info(
        f"Value counts of 'Type' in combined_df AFTER dropna:\n{combined_df['Type'].value_counts()}"
    )

    if combined_df.empty:
        logger.error(
            "DataFrame is empty after processing and dropping NaNs. Cannot generate plot."
        )
        plt.close()
        return

    a1_models_df_for_pareto = combined_df[combined_df["Type"] == "A1 Model"].copy()

    if (
        models_df is not None
        and not models_df.empty
        and "model_id" in models_df.columns
        and "parameter_count" in models_df.columns
    ):
        a1_models_df_for_pareto = pd.merge(
            a1_models_df_for_pareto,
            models_df[["model_id", "parameter_count"]],
            left_on="Name",
            right_on="model_id",
            how="left",
        )
        a1_models_df_for_pareto["parameter_count"] = a1_models_df_for_pareto[
            "parameter_count"
        ].fillna(0)
    else:
        a1_models_df_for_pareto["parameter_count"] = 0

    logger.info("Calculating Pareto frontier for individual A1 models...")
    pareto_points = pd.DataFrame()

    if not a1_models_df_for_pareto.empty:
        pareto_points = analysis_utils.find_pareto_frontier(
            df=a1_models_df_for_pareto,
            x_col="Mean Normalized Energy",
            y_col="Mean Normalized Accuracy",
            lower_x_is_better=True,
            higher_y_is_better=True,
        )
    pareto_model_names = (
        pareto_points["Name"].tolist() if not pareto_points.empty else []
    )
    logger.info(f"A1 Models on Pareto Frontier: {pareto_model_names}")


    a5_styles = {
        "epsilon_greedy": {"color": "#0173b2", "marker": "o"},
        "linear_epsilon_greedy": {"color": "#de8f05", "marker": "s"},
        "linucb": {"color": "#029e73", "marker": "h"},
        "thompson_sampling": {"color": "#d55e00", "marker": "*"},
    }

    baseline_marker_map = {
        "smallest": plot_utils.get_marker("smallest"),
        "largest": plot_utils.get_marker("largest"),
        "random": plot_utils.get_marker("random"),
        "accuracy": plot_utils.get_marker("accuracy"),
        "default": "x",
    }
    legend_handles = {}
    for _, row in combined_df.iterrows():
        point_type = row["Type"]
        name = row["Name"]
        energy_val = row["Mean Normalized Energy"]
        accuracy_val = row["Mean Normalized Accuracy"]

        handle = None
        current_label = None
        color = None
        marker = None
        size = 60
        alpha = 0.7
        edgecolor = "none"  # Default edge
        zorder = 2
        facecolor = None

        if point_type == "A1 Model":
            marker = "o"
            color = "#1f77b4"  # Keep A1 models a consistent color
            facecolor = color
            current_label = None

        elif point_type == "A1 Baseline":
            marker = baseline_marker_map.get(name, baseline_marker_map["default"])
            color = plot_utils.get_color(name)
            facecolor = color
            size = 120
            alpha = 0.9
            edgecolor = "black"
            zorder = 4
            current_label = None

        elif point_type == "A5 Algorithm":
            style = a5_styles.get(name)
            if style:
                color = style["color"]
                marker = style["marker"]
            else:
                logger.warning(f"Style not defined for A5 algorithm: {name}")
                color = plot_utils.get_color(name)
                marker = "X"  # Fallback marker
            facecolor = color
            size = 100
            alpha = 0.9
            edgecolor = "black"
            zorder = 3
            display_name = ALGO_DISPLAY_NAMES.get(name, name)
            current_label = display_name

        else:
            logger.warning(f"Unknown point type '{point_type}' for point '{name}'")
            marker = baseline_marker_map["default"]
            color = "grey"
            facecolor = color
            size = 50
            alpha = 0.5
            zorder = 1
            current_label = point_type

        logger.info(
            f"Plotting Check: Name={name}, Type={point_type}, Color='{color}', Marker='{marker}', Condition={bool(color and marker)}"
        )
        if color and marker:
            handle = ax.scatter(
                energy_val,
                accuracy_val,
                marker=marker,
                color=color,
                facecolors=facecolor,
                s=size,
                alpha=alpha,
                edgecolors=edgecolor,
                zorder=zorder,
                label=current_label,
            )
            if point_type == "A1 Baseline":
                legend_handles[name] = handle
            elif (
                point_type == "A5 Algorithm"
                and current_label
                and current_label not in legend_handles
            ):
                legend_handles[current_label] = handle

        else:
            logger.error(
                f"Could not determine style for point: Name={name}, Type={point_type}"
            )
    if not pareto_points.empty:
        pareto_points_sorted = pareto_points.sort_values(by="Mean Normalized Energy")
        handle_pareto = ax.plot(
            pareto_points_sorted["Mean Normalized Energy"],
            pareto_points_sorted["Mean Normalized Accuracy"],
            linestyle="--",
            color="red",
            marker=".",
            linewidth=1.5,
            zorder=3,
        )[0]
        legend_handles["Pareto Front"] = handle_pareto

    ax.set_xlabel("Mean Normalized Energy per Token", fontsize=12)
    ax.set_ylabel("Mean Normalized Accuracy", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(left=-0.05, right=1.05)
    ax.set_ylim(bottom=-0.05, top=1.05)

    final_legend_elements = []
    final_legend_labels = []

    if "Pareto Front" in legend_handles:
        final_legend_elements.append(legend_handles["Pareto Front"])
        final_legend_labels.append("Pareto Front")

    baseline_order = [
        name
        for name in ["random", "largest", "smallest", "accuracy"]
        if name in legend_handles
    ]

    if baseline_order:

        for name in baseline_order:
            final_legend_elements.append(legend_handles[name])
            final_legend_labels.append(ALGO_DISPLAY_NAMES.get(name, name))

    a5_algo_display_names = sorted(
        [name for name in legend_handles.keys() if name in ALGO_DISPLAY_NAMES.values()]
    )

    if a5_algo_display_names:

        for display_name in a5_algo_display_names:
            final_legend_elements.append(legend_handles[display_name])
            final_legend_labels.append(display_name)

    if final_legend_elements:
        if len(final_legend_elements) != len(final_legend_labels):
            logger.error(
                f"Legend handle/label mismatch: {len(final_legend_elements)} handles, {len(final_legend_labels)} labels. Legend may be incorrect."
            )
            ax.legend(loc="best")
        else:
            ax.legend(final_legend_elements, final_legend_labels, loc="best")

    else:
        logger.warning("No handles collected for custom legend.")
        ax.legend(loc="best")

    plt.tight_layout()
    plot_filename = f"{exp_dirs['base'].name}_pareto_plot.png"
    plot_utils.save_plot(fig, exp_dirs["plots"], plot_filename)
    logger.info(f"A5 Pareto plot saved to {exp_dirs['plots'] / plot_filename}")


def generate_pareto_plot_wh(
    a5_summary_df: pd.DataFrame,
    a5_detailed_results_df: pd.DataFrame,
    raw_results_data_df: pd.DataFrame,
    norm_df: pd.DataFrame,
    exp_dirs: dict,
    models_df: pd.DataFrame = None,
):
    """
    Generates the Pareto plot using TOTAL ENERGY (Wh) vs. Mean Normalized Accuracy.

    Args:
        a5_summary_df (pd.DataFrame): Aggregated results from A5 sim (Mean Norm Acc/Energy).
                                      Used mainly for Mean Normalized Accuracy values.
        a5_detailed_results_df (pd.DataFrame): Detailed simulation results for ALL runs/algos.
                                             Used to calculate avg total energy per algorithm.
        raw_results_data_df (pd.DataFrame): Raw results with energy_consumption (assumed Ws).
                                          Used for individual model total energy and lookup.
        norm_df (pd.DataFrame): DataFrame with globally normalized metrics (accuracy_norm, etc.).
        exp_dirs (dict): Experiment directories.
        models_df (pd.DataFrame, optional): Model metadata.
    """
    logger.info("Generating A5 Algorithm Bakeoff Pareto plot (Total Energy Wh)...")
    fig, ax = plt.subplots(figsize=(10, 7))
    if "energy_consumption" not in raw_results_data_df.columns:
        logger.error(
            "Raw data missing 'energy_consumption'. Cannot generate total energy plot."
        )
        plt.close()
        return
    total_energy_ws_df = (
        raw_results_data_df.groupby("model_id")["energy_consumption"]
        .sum()
        .reset_index()
    )
    total_energy_ws_df.rename(
        columns={"energy_consumption": "total_energy_ws"}, inplace=True
    )
    total_energy_ws_df["total_energy_wh"] = (
        total_energy_ws_df["total_energy_ws"] / 3600.0
    )
    logger.info(f"Calculated total energy (Wh) for {len(total_energy_ws_df)} models.")
    all_processed_points = []
    if norm_df is not None and not norm_df.empty and total_energy_ws_df is not None:
        if "accuracy_norm" not in norm_df.columns:
            logger.error(
                "Globally normalized data missing 'accuracy_norm'. Cannot plot A1 models correctly."
            )
        else:
            model_avg_norm_acc = (
                norm_df.groupby("model_id")["accuracy_norm"].mean().reset_index()
            )
            model_perf_total_energy = pd.merge(
                model_avg_norm_acc,
                total_energy_ws_df[["model_id", "total_energy_wh"]],
                on="model_id",
                how="inner",
            )
            logger.info(
                f"Processed {len(model_perf_total_energy)} individual models for total energy plot using norm_df."
            )
            for _, row in model_perf_total_energy.iterrows():
                if row["model_id"] not in KNOWN_BASELINE_NAMES:
                    all_processed_points.append(
                        {
                            "Name": row["model_id"],
                            "Mean Normalized Accuracy": row["accuracy_norm"],
                            "Total Energy (Wh)": row["total_energy_wh"],
                            "Type": "A1 Model",
                        }
                    )
    else:
        logger.warning(
            "Normalized data (norm_df) or total energy data not available. Cannot plot individual A1 models."
        )
    if a5_detailed_results_df is not None and not a5_detailed_results_df.empty:
        logger.info(
            "Calculating average total energy per algorithm run from detailed A5 results..."
        )
        if "energy_consumption" not in raw_results_data_df.columns:
            logger.error(
                "Raw results data missing 'energy_consumption'. Cannot calculate A5 total energy."
            )
        else:
            detailed_with_raw_energy = pd.merge(
                a5_detailed_results_df[
                    ["run_id", "algorithm", "query_id", "chosen_model"]
                ],
                raw_results_data_df[["query_id", "model_id", "energy_consumption"]],
                left_on=["query_id", "chosen_model"],
                right_on=["query_id", "model_id"],
                how="left",
            )
            total_energy_ws_per_run = (
                detailed_with_raw_energy.groupby(["run_id", "algorithm"])[
                    "energy_consumption"
                ]
                .sum()
                .reset_index()
            )
            total_energy_ws_per_run["total_energy_wh"] = (
                total_energy_ws_per_run["energy_consumption"] / 3600.0
            )

            avg_total_energy_per_algo = (
                total_energy_ws_per_run.groupby("algorithm")["total_energy_wh"]
                .mean()
                .reset_index()
            )
            avg_total_energy_per_algo.rename(
                columns={"total_energy_wh": "Mean Total Energy (Wh)"}, inplace=True
            )

            std_total_energy_per_algo = (
                total_energy_ws_per_run.groupby("algorithm")["total_energy_wh"]
                .std()
                .reset_index()
            )
            std_total_energy_per_algo.rename(
                columns={"total_energy_wh": "Std Total Energy (Wh)"}, inplace=True
            )
            std_total_energy_per_algo["Std Total Energy (Wh)"] = (
                std_total_energy_per_algo["Std Total Energy (Wh)"].fillna(0)
            )

            logger.info(
                f"Calculated average total energy (Wh) for {len(avg_total_energy_per_algo)} algorithms."
            )

            if "Algorithm" not in a5_summary_df.columns:
                logger.error("a5_summary_df missing 'Algorithm' column for merging.")
            elif "Mean Normalized Accuracy" not in a5_summary_df.columns:
                logger.error("a5_summary_df missing 'Mean Normalized Accuracy' column.")
            else:
                algo_perf_total_energy = pd.merge(
                    avg_total_energy_per_algo,
                    a5_summary_df[["Algorithm", "Mean Normalized Accuracy"]],
                    left_on="algorithm",
                    right_on="Algorithm",
                    how="inner",
                )
                for _, row in algo_perf_total_energy.iterrows():
                    algo_name = row["Algorithm"]
                    point_type = (
                        "A1 Baseline"
                        if algo_name in KNOWN_BASELINE_NAMES
                        else "A5 Algorithm"
                    )
                    all_processed_points.append(
                        {
                            "Name": algo_name,
                            "Mean Normalized Accuracy": row["Mean Normalized Accuracy"],
                            "Total Energy (Wh)": row["Mean Total Energy (Wh)"],
                            "Type": point_type,
                        }
                    )
                logger.info(
                    f"Processed {len(algo_perf_total_energy)} strategies from A5 detailed results."
                )

    else:
        logger.warning(
            "A5 detailed results DataFrame not provided or empty. Skipping A5 algorithms and baselines."
        )
    if not all_processed_points:
        logger.error(
            "No data points processed for total energy plot. Cannot generate plot."
        )
        plt.close()
        return

    combined_df = pd.DataFrame(all_processed_points)
    combined_df.dropna(
        subset=["Mean Normalized Accuracy", "Total Energy (Wh)"], inplace=True
    )

    if combined_df.empty:
        logger.error(
            "DataFrame is empty after processing for total energy plot. Cannot generate plot."
        )
        plt.close()
        return
    a1_models_df_for_pareto = combined_df[combined_df["Type"] == "A1 Model"].copy()
    pareto_points = pd.DataFrame()

    if not a1_models_df_for_pareto.empty:
        pareto_points = analysis_utils.find_pareto_frontier(
            df=a1_models_df_for_pareto,
            x_col="Total Energy (Wh)",
            y_col="Mean Normalized Accuracy",
            lower_x_is_better=True,
            higher_y_is_better=True,
        )
    logger.info(
        f"Models on Total Energy Pareto Frontier: {pareto_points['Name'].tolist() if not pareto_points.empty else 'None'}"
    )

    a5_styles = {
        "epsilon_greedy": {"color": "#0173b2", "marker": "o"},
        "linear_epsilon_greedy": {"color": "#de8f05", "marker": "s"},
        "linucb": {"color": "#029e73", "marker": "h"},
        "thompson_sampling": {"color": "#d55e00", "marker": "*"},
    }
    baseline_marker_map = {
        "smallest": plot_utils.get_marker("smallest"),
        "largest": plot_utils.get_marker("largest"),
        "random": plot_utils.get_marker("random"),
        "accuracy": plot_utils.get_marker("accuracy"),
        "default": "x",
    }
    legend_handles = {}

    for _, row in combined_df.iterrows():
        point_type = row["Type"]
        name = row["Name"]
        energy_val = row["Total Energy (Wh)"]
        accuracy_val = row["Mean Normalized Accuracy"]
        handle = None
        current_label = None
        color = None
        marker = None
        size = 60
        alpha = 0.7
        edgecolor = "none"
        zorder = 2
        facecolor = None

        if point_type == "A1 Model":
            marker = "o"
            color = "#1f77b4"
            facecolor = color
        elif point_type == "A1 Baseline":
            marker = baseline_marker_map.get(name, baseline_marker_map["default"])
            color = plot_utils.get_color(name)
            facecolor = color
            size = 120
            alpha = 0.9
            edgecolor = "black"
            zorder = 4
        elif point_type == "A5 Algorithm":
            style = a5_styles.get(name)
            color = style["color"]
            marker = style["marker"]
            facecolor = color
            size = 100
            alpha = 0.9
            edgecolor = "black"
            zorder = 3
            display_name = ALGO_DISPLAY_NAMES.get(name, name)
            current_label = display_name
        else:
            marker = baseline_marker_map["default"]
            color = "grey"
            facecolor = color
            size = 50
            alpha = 0.5
            zorder = 1
            current_label = point_type

        if color and marker:
            handle = ax.scatter(
                energy_val,
                accuracy_val,
                marker=marker,
                color=color,
                facecolors=facecolor,
                s=size,
                alpha=alpha,
                edgecolors=edgecolor,
                zorder=zorder,
                label=current_label,
            )
            if point_type == "A1 Baseline":
                legend_handles[name] = handle
            elif (
                point_type == "A5 Algorithm"
                and current_label
                and current_label not in legend_handles
            ):
                legend_handles[current_label] = handle
        else:
            logger.error(
                f"Could not determine style for point: Name={name}, Type={point_type}"
            )

    if not pareto_points.empty:
        pareto_points_sorted = pareto_points.sort_values(by="Total Energy (Wh)")
        handle_pareto = ax.plot(
            pareto_points_sorted["Total Energy (Wh)"],
            pareto_points_sorted["Mean Normalized Accuracy"],
            linestyle="--",
            color="red",
            marker=".",
            linewidth=1.5,
            zorder=3,
        )[0]
        legend_handles["Pareto Front"] = handle_pareto

    ax.set_xlabel("Total Energy Consumption (Wh)", fontsize=12)
    ax.set_ylabel("Mean Normalized Accuracy", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.set_ylim(bottom=-0.05, top=1.05)
    try:
        formatter_x = EngFormatter(unit="Wh")
        ax.xaxis.set_major_formatter(formatter_x)
    except ImportError:
        logger.warning(
            "EngFormatter not available. Using default numeric format for Energy axis."
        )

    final_legend_elements = []
    final_legend_labels = []
    if "Pareto Front" in legend_handles:
        final_legend_elements.append(legend_handles["Pareto Front"])
        final_legend_labels.append("Pareto Front")
    baseline_order = [
        name
        for name in ["random", "largest", "smallest", "accuracy"]
        if name in legend_handles
    ]

    if baseline_order:
        for name in baseline_order:
            final_legend_elements.append(legend_handles[name])
            final_legend_labels.append(ALGO_DISPLAY_NAMES.get(name, name))
    a5_algo_display_names = sorted(
        [name for name in legend_handles.keys() if name in ALGO_DISPLAY_NAMES.values()]
    )

    if a5_algo_display_names:
        for display_name in a5_algo_display_names:
            final_legend_elements.append(legend_handles[display_name])
            final_legend_labels.append(display_name)

    if final_legend_elements:
        if len(final_legend_elements) != len(final_legend_labels):
            logger.error("Legend mismatch")
            ax.legend(loc="best")
        else:
            ax.legend(final_legend_elements, final_legend_labels, loc="best")
    else:
        logger.warning("No handles collected for custom legend.")
        ax.legend(loc="best")

    plt.tight_layout()
    plot_filename = f"{exp_dirs['base'].name}_pareto_plot_wh.png"  # Updated filename
    plot_utils.save_plot(fig, exp_dirs["plots"], plot_filename)
    logger.info(
        f"A5 Pareto plot (Total Energy Wh) saved to {exp_dirs['plots'] / plot_filename}"
    )


# ============================================================================
# BAR PLOTS
# ============================================================================


def generate_grouped_norm_metric_bar_plot(
    a5_summary_df: pd.DataFrame, exp_dirs: dict, has_error_bar_data: bool = False
):
    """
    Generates bar plot: 2-color (grey/blue), hatched energy, conditional CIs,
    specific sort order, mapped names, full legend, no grid/title/x-label.

    Args:
        a5_summary_df (pd.DataFrame): Aggregated results (already renamed columns).
        exp_dirs (dict): Experiment directories.
        has_error_bar_data (bool): Flag indicating if required std dev/count data is present.
    """
    logger.info("Generating final A5 grouped bar plot (Corrected Order/Legend)...")

    required_cols = ["Algorithm", "Mean Normalized Accuracy", "Mean Normalized Energy"]

    if not all(col in a5_summary_df.columns for col in required_cols):
        logger.error(
            f"Required columns for bar plot missing: {required_cols}. Available: {a5_summary_df.columns.tolist()}"
        )
        return
    plot_df = a5_summary_df.copy()

    if "Algorithm" not in plot_df.columns:
        logger.error(
            "Input DataFrame for bar plot missing 'Algorithm' column (expected original names)."
        )
        return

    plot_df["Algorithm"] = pd.Categorical(
        plot_df["Algorithm"], categories=ALGO_DISPLAY_NAMES.keys(), ordered=True
    )

    plot_df.dropna(subset=["Algorithm"], inplace=True)
    plot_df = plot_df.sort_values(by="Algorithm")
    plot_order_display_sorted = plot_df["Algorithm"].map(ALGO_DISPLAY_NAMES).tolist()

    logger.info(f"Algorithm display order for plot: {plot_order_display_sorted}")
    plot_df["Algorithm Display"] = plot_df["Algorithm"].map(ALGO_DISPLAY_NAMES)

    df_melted = plot_df.melt(
        id_vars=["Algorithm", "Algorithm Display"],
        value_vars=["Mean Normalized Accuracy", "Mean Normalized Energy"],
        var_name="Metric",
        value_name="Mean Value",
    )

    metric_display_map = {
        "Mean Normalized Accuracy": "Accuracy",
        "Mean Normalized Energy": "Energy per Token",
    }

    df_melted["Metric Display"] = df_melted["Metric"].map(metric_display_map)
    error_data = {}
    calculated_error_keys = []

    if has_error_bar_data:
        std_acc_col = "Std Norm Accuracy"
        std_energy_col = "std_energy_per_token_norm"  # Assumed not renamed
        count_col = "Count"
        required_ci_cols = [std_acc_col, std_energy_col, count_col]
        if all(c in plot_df.columns for c in required_ci_cols):
            z_value = 1.96
            for _, row in plot_df.iterrows():
                algo_display = row["Algorithm Display"]
                count = row[count_col]
                if count > 0:
                    se_acc = row.get(std_acc_col, 0) / np.sqrt(count)
                    se_energy = row.get(std_energy_col, 0) / np.sqrt(count)
                    error_data[algo_display] = {
                        "Accuracy": z_value * se_acc,
                        "Energy per Token": z_value * se_energy,
                    }
                    calculated_error_keys.append(algo_display)
            logger.info(f"Successfully calculated CIs for: {calculated_error_keys}")
        else:
            logger.warning(
                f"Required CI columns ({required_ci_cols}) not found. CIs disabled."
            )
    else:
        logger.info("has_error_bar_data flag is False. CIs disabled.")

    fig, ax1 = plt.subplots(figsize=(12, 7))
    color_a5_algo = "#1f77b4"
    color_energy = "#d62728"
    n_algorithms = len(plot_order_display_sorted)
    bar_width = 0.35
    index = np.arange(n_algorithms)

    for i, algorithm_display_name in enumerate(plot_order_display_sorted):

        original_row = plot_df[
            plot_df["Algorithm Display"] == algorithm_display_name
        ].iloc[0]
        algorithm_original_name = original_row["Algorithm"]

        is_baseline = algorithm_original_name in KNOWN_BASELINE_NAMES
        current_acc_color = color_a5_algo
        acc_row = df_melted[
            (df_melted["Algorithm Display"] == algorithm_display_name)
            & (df_melted["Metric Display"] == "Accuracy")
        ]
        energy_row = df_melted[
            (df_melted["Algorithm Display"] == algorithm_display_name)
            & (df_melted["Metric Display"] == "Energy per Token")
        ]

        apply_ci = has_error_bar_data and not is_baseline
        if not acc_row.empty:
            acc_val = acc_row["Mean Value"].iloc[0]
            acc_err = (
                error_data.get(algorithm_display_name, {}).get("Accuracy", 0)
                if apply_ci
                else None
            )
            ax1.bar(
                index[i] - bar_width / 2,
                acc_val,
                bar_width,
                color=current_acc_color,
                edgecolor="black",
                linewidth=0.5,
                yerr=acc_err,
                capsize=5 if acc_err is not None else 0,
            )
        if not energy_row.empty:
            energy_val = energy_row["Mean Value"].iloc[0]
            energy_err = (
                error_data.get(algorithm_display_name, {}).get("Energy per Token", 0)
                if apply_ci
                else None
            )
            ax1.bar(
                index[i] + bar_width / 2,
                energy_val,
                bar_width,
                color=color_energy,
                hatch="//",
                edgecolor="black",
                linewidth=0.5,
                yerr=energy_err,
                capsize=5 if energy_err is not None else 0,
            )

    ax1.set_ylabel("Mean Normalized Value", fontsize=14)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.set_xticks(index)

    logger.info(f"Setting xticklabels with: {plot_order_display_sorted}")
    ax1.set_xticklabels(plot_order_display_sorted, rotation=45, ha="right", fontsize=12)
    max_mean_val = df_melted["Mean Value"].max()

    ax1.set_ylim(0, max(1.0, max_mean_val * 1.1) if not pd.isna(max_mean_val) else 1.0)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax1.set_xlabel(None)
    patch_acc_simple = mpatches.Patch(
        facecolor=color_a5_algo, label="Accuracy", edgecolor="black"
    )
    patch_energy = mpatches.Patch(
        facecolor=color_energy, label="Energy per Token", hatch="//", edgecolor="black"
    )

    ax1.legend(handles=[patch_acc_simple, patch_energy], loc="best", fontsize=12)
    fig.tight_layout()
    plot_filename = exp_dirs["plots"] / f"{exp_dirs['base'].name}_bar_plot_final.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    logger.info(f"Final bar plot saved to {plot_filename}")
    plt.close(fig)


def generate_grouped_bar_plot_total_energy(
    a5_summary_df: pd.DataFrame,
    a5_detailed_results_df: pd.DataFrame,
    raw_results_data_df: pd.DataFrame,
    exp_dirs: dict,
):
    """
    Generates grouped bar plot: Mean Normalized Accuracy vs. Mean Total Energy (Wh)
    using dual Y-axes.

    Args:
        a5_summary_df (pd.DataFrame): Aggregated results (needs 'Algorithm', 'Mean Normalized Accuracy').
        a5_detailed_results_df (pd.DataFrame): Detailed simulation results.
        raw_results_data_df (pd.DataFrame): Raw results with energy_consumption (assumed Ws).
        exp_dirs (dict): Experiment directories.
    """
    logger.info("Generating A5 grouped bar plot (Accuracy vs Total Energy Wh)...")
    if not all(
        col in a5_summary_df.columns
        for col in ["Algorithm", "Mean Normalized Accuracy"]
    ):
        logger.error("a5_summary_df missing required columns for Accuracy bar plot.")
        return

    if a5_detailed_results_df is None or a5_detailed_results_df.empty:
        logger.error(
            "a5_detailed_results_df is missing. Cannot calculate total energy."
        )
        return

    if (
        raw_results_data_df is None
        or raw_results_data_df.empty
        or "energy_consumption" not in raw_results_data_df.columns
    ):
        logger.error(
            "raw_results_data_df is missing or lacks 'energy_consumption'. Cannot calculate total energy."
        )
        return

    logger.info("Calculating average total energy (Wh) per algorithm run...")

    detailed_with_raw_energy = pd.merge(
        a5_detailed_results_df[["run_id", "algorithm", "query_id", "chosen_model"]],
        raw_results_data_df[["query_id", "model_id", "energy_consumption"]],
        left_on=["query_id", "chosen_model"],
        right_on=["query_id", "model_id"],
        how="left",
    )
    total_energy_ws_per_run = (
        detailed_with_raw_energy.groupby(["run_id", "algorithm"])["energy_consumption"]
        .sum()
        .reset_index()
    )
    total_energy_ws_per_run["total_energy_wh"] = (
        total_energy_ws_per_run["energy_consumption"] / 3600.0
    )
    avg_total_energy_per_algo = (
        total_energy_ws_per_run.groupby("algorithm")["total_energy_wh"]
        .mean()
        .reset_index()
    )
    avg_total_energy_per_algo.rename(
        columns={"total_energy_wh": "Mean Total Energy (Wh)"}, inplace=True
    )

    std_total_energy_per_algo = (
        total_energy_ws_per_run.groupby("algorithm")["total_energy_wh"]
        .std()
        .reset_index()
    )
    std_total_energy_per_algo.rename(
        columns={"total_energy_wh": "Std Total Energy (Wh)"}, inplace=True
    )
    std_total_energy_per_algo["Std Total Energy (Wh)"] = std_total_energy_per_algo[
        "Std Total Energy (Wh)"
    ].fillna(0)
    logger.info(
        f"Calculated average total energy (Wh) for {len(avg_total_energy_per_algo)} algorithms."
    )

    plot_data = pd.merge(
        a5_summary_df[["Algorithm", "Mean Normalized Accuracy"]],
        avg_total_energy_per_algo,
        left_on="Algorithm",
        right_on="algorithm",
        how="inner",  # Only plot algos present in both
    )
    plot_data = pd.merge(
        plot_data,
        std_total_energy_per_algo,
        left_on="Algorithm",
        right_on="algorithm",
        how="left",
    )
    if "Std Norm Accuracy" in a5_summary_df.columns:
        plot_data = pd.merge(
            plot_data,
            a5_summary_df[["Algorithm", "Std Norm Accuracy"]],
            on="Algorithm",
            how="left",
        )
        plot_data["Std Norm Accuracy"] = plot_data["Std Norm Accuracy"].fillna(0)
    else:
        logger.warning(
            "'Std Norm Accuracy' not found in a5_summary_df. Accuracy error bars disabled."
        )
        plot_data["Std Norm Accuracy"] = 0

    plot_data = plot_data.loc[:, ~plot_data.columns.duplicated()]
    plot_data["Algorithm Cat"] = pd.Categorical(
        plot_data["Algorithm"], categories=ALGO_DISPLAY_NAMES.keys(), ordered=True
    )
    plot_data.dropna(subset=["Algorithm Cat"], inplace=True)
    plot_data = plot_data.sort_values(by="Algorithm Cat")
    plot_order_display_sorted = (
        plot_data["Algorithm Cat"].map(ALGO_DISPLAY_NAMES).tolist()
    )
    logger.info(f"Algorithm display order for plot: {plot_order_display_sorted}")

    if plot_data.empty:
        logger.error(
            "No data available for plotting after merging accuracy and total energy."
        )
        return
    fig, ax1 = plt.subplots(figsize=(12, 7))
    color_acc = "#1f77b4"  # Standard Blue
    color_energy = "#d62728"

    n_algorithms = len(plot_data)
    bar_width = 0.35
    index = np.arange(n_algorithms)
    logger.info("--- Inspecting plot_data before generating bars ---")
    logger.info(f"\n{plot_data.to_string()}")
    logger.info("--------------------------------------------------")

    bars_acc = ax1.bar(
        index - bar_width / 2,
        plot_data["Mean Normalized Accuracy"],
        bar_width,
        color=color_acc,
        edgecolor="black",
        linewidth=0.5,
        label="Accuracy",
        yerr=plot_data["Std Norm Accuracy"],
        capsize=5,
    )

    ax2 = ax1.twinx()
    bars_energy = ax2.bar(
        index + bar_width / 2,
        plot_data["Mean Total Energy (Wh)"],
        bar_width,
        color=color_energy,
        hatch="//",
        edgecolor="black",
        linewidth=0.5,
        label="Total Energy",
        yerr=plot_data["Std Total Energy (Wh)"],
        capsize=5,
    )

    ax1.set_ylabel("Mean Normalized Accuracy", fontsize=14)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(index)
    ax1.set_xticklabels(plot_order_display_sorted, rotation=45, ha="right", fontsize=12)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)

    ax2.set_ylabel("Total Energy Consumption (Wh)", fontsize=14)
    ax2.tick_params(axis="y", labelsize=12)
    max_energy = plot_data["Mean Total Energy (Wh)"].max()
    ax2.set_ylim(0, max(1, max_energy * 1.1) if pd.notna(max_energy) else 1)

    try:
        formatter_y = EngFormatter(unit="Wh")
        ax2.yaxis.set_major_formatter(formatter_y)
    except ImportError:
        logger.warning(
            "EngFormatter not available. Using default numeric format for Energy axis."
        )
    ax2.grid(False)

    handles = [bars_acc[0], bars_energy[0]]
    labels = [h.get_label() for h in handles]

    ax1.legend(handles, labels, loc="best", fontsize=12)
    fig.tight_layout()
    plot_filename = (
        exp_dirs["plots"] / f"{exp_dirs['base'].name}_bar_plot_total_energy.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    logger.info(f"Grouped bar plot (Acc vs Total Energy Wh) saved to {plot_filename}")
    plt.close(fig)


# ============================================================================
# REGRET PLOTS
# ============================================================================


def generate_cumulative_regret_plot(detailed_results_df: pd.DataFrame, exp_dirs: dict):
    """Generates a plot of cumulative regret over query steps.

    Args:
        detailed_results_df (pd.DataFrame): DataFrame with detailed simulation results,
                                            including 'run_id', 'algorithm', 'query_index',
                                            and 'step_regret'.
        exp_dirs (dict): Experiment directories.
    """
    logger.info("Generating cumulative regret plot...")

    required_cols = ["run_id", "algorithm", "query_index", "step_regret"]
    if detailed_results_df is None or detailed_results_df.empty:
        logger.warning("Detailed results DataFrame is empty. Skipping regret plot.")
        return
    if not all(col in detailed_results_df.columns for col in required_cols):
        logger.error(
            f"Detailed results DataFrame missing required columns {required_cols}. Cannot generate regret plot."
        )
        logger.warning(f"Available columns: {detailed_results_df.columns.tolist()}")
        return
    algorithms_in_input = detailed_results_df["algorithm"].unique()
    logger.info(
        f"[Cumulative Regret Plot] Algorithms found in input data: {list(algorithms_in_input)}"
    )

    df = detailed_results_df.copy()
    df = df.sort_values(by=["run_id", "algorithm", "query_index"])
    df["cumulative_regret"] = df.groupby(["run_id", "algorithm"])[
        "step_regret"
    ].cumsum()

    df["Algorithm Name"] = (
        df["algorithm"].map(ALGO_DISPLAY_NAMES).fillna(df["algorithm"])
    )
    plot_order_base = KNOWN_BASELINE_NAMES + list(
        ALGO_DISPLAY_NAMES.keys() - set(KNOWN_BASELINE_NAMES)
    )
    plot_order_actual = [
        algo for algo in plot_order_base if algo in algorithms_in_input
    ]
    plot_order_display = [
        ALGO_DISPLAY_NAMES.get(algo, algo) for algo in plot_order_actual
    ]
    palette, dashes = get_regret_plot_styles(algorithms_in_input)
    logger.info("--- Cumulative Regret Plot: Building Plot Order & Styles ---")

    found_baselines = [
        algo for algo in plot_order_actual if algo in KNOWN_BASELINE_NAMES
    ]
    found_bandits = [
        algo for algo in plot_order_actual if algo not in KNOWN_BASELINE_NAMES
    ]
    logger.info(f"  Baselines found and plotting: {found_baselines}")
    logger.info(f"  Bandits found and plotting: {found_bandits}")
    logger.info(f"Final plot_order_display: {plot_order_display}")
    logger.info(f"Cumulative Regret Plot Palette: {palette}")
    logger.info(f"Cumulative Regret Plot Dashes: {dashes}")
    logger.info("--------------------------------------------------------")

    if df.empty or df["cumulative_regret"].isnull().all():
        logger.error(
            "DataFrame is empty or cumulative_regret is all NaN after processing. Skipping lineplot."
        )
        plt.close()
        return
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="query_index",
        y="cumulative_regret",
        hue="Algorithm Name",
        style="Algorithm Name",
        hue_order=plot_order_display,
        style_order=plot_order_display,
        palette=palette,
        dashes=dashes,
        errorbar="sd",
        ax=ax,
        linewidth=1.8,
    )
    ax.set_xlabel("Query Index (Time Step)")
    ax.set_ylabel("Cumulative Regret")
    ax.grid(True, axis="both", linestyle="--", alpha=0.6)
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = []
    ordered_labels = []
    current_legend_map = dict(zip(labels, handles))

    for label in plot_order_display:
        if label in current_legend_map:
            ordered_handles.append(current_legend_map[label])
            ordered_labels.append(label)
    if ordered_handles:
        ax.legend(ordered_handles, ordered_labels, loc="best", title=None)
    else:
        ax.legend(loc="best", title=None)
    sns.despine(fig=fig)
    plt.tight_layout()
    plot_filename = f'{exp_dirs["base"].name}_cumulative_regret.png'
    plot_utils.save_plot(fig, exp_dirs["plots"], plot_filename)
    logger.info(f'Cumulative regret plot saved to {exp_dirs["plots"] / plot_filename}')


def generate_moving_average_regret_plot(
    detailed_results_df: pd.DataFrame, exp_dirs: dict, window_size: int = 50
):
    """Generates a plot of moving average regret over query steps.

    Args:
        detailed_results_df (pd.DataFrame): DataFrame with detailed simulation results,
                                            including 'run_id', 'algorithm', 'query_index',
                                            and 'step_regret'.
        exp_dirs (dict): Experiment directories.
        window_size (int): The window size for the moving average calculation.
    """
    logger.info(f"Generating moving average regret plot (Window={window_size})...")

    required_cols = ["run_id", "algorithm", "query_index", "step_regret"]
    if detailed_results_df is None or detailed_results_df.empty:
        logger.warning(
            "Detailed results DataFrame is empty. Skipping moving average regret plot."
        )
        return

    if not all(col in detailed_results_df.columns for col in required_cols):
        logger.error(
            f"Detailed results DataFrame missing required columns {required_cols}. Cannot generate moving average regret plot."
        )
        logger.warning(f"Available columns: {detailed_results_df.columns.tolist()}")
        return

    algorithms_in_input = detailed_results_df["algorithm"].unique()
    logger.info(
        f"[Moving Avg Regret Plot] Algorithms found in input data: {list(algorithms_in_input)}"
    )

    plot_df = detailed_results_df.copy()
    plot_df = plot_df.sort_values(by=["run_id", "algorithm", "query_index"])
    plot_df["moving_avg_regret"] = plot_df.groupby(["run_id", "algorithm"])[
        "step_regret"
    ].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())

    plot_df["Algorithm Name"] = (
        plot_df["algorithm"].map(ALGO_DISPLAY_NAMES).fillna(plot_df["algorithm"])
    )
    plot_order_base = KNOWN_BASELINE_NAMES + list(
        ALGO_DISPLAY_NAMES.keys() - set(KNOWN_BASELINE_NAMES)
    )
    plot_order_actual = [
        algo for algo in plot_order_base if algo in algorithms_in_input
    ]
    plot_order_display = [
        ALGO_DISPLAY_NAMES.get(algo, algo) for algo in plot_order_actual
    ]
    palette, dashes = get_regret_plot_styles(algorithms_in_input)
    logger.info("--- Moving Avg Regret Plot: Building Plot Order & Styles ---")

    found_baselines = [
        algo for algo in plot_order_actual if algo in KNOWN_BASELINE_NAMES
    ]
    found_bandits = [
        algo for algo in plot_order_actual if algo not in KNOWN_BASELINE_NAMES
    ]
    logger.info(f"  Baselines found and plotting: {found_baselines}")
    logger.info(f"  Bandits found and plotting: {found_bandits}")
    logger.info(f"Final plot_order_display: {plot_order_display}")
    logger.info(f"Moving Avg Regret Plot Palette: {palette}")
    logger.info(f"Moving Avg Regret Plot Dashes: {dashes}")
    logger.info("--------------------------------------------------------------")

    if plot_df.empty or plot_df["moving_avg_regret"].isnull().all():
        logger.error(
            "DataFrame is empty or moving_avg_regret is all NaN after processing. Skipping lineplot."
        )
        plt.close()
        return
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=plot_df,
        x="query_index",
        y="moving_avg_regret",
        hue="Algorithm Name",
        style="Algorithm Name",
        hue_order=plot_order_display,
        style_order=plot_order_display,
        palette=palette,
        dashes=dashes,
        errorbar="sd",
        ax=ax,
        linewidth=1.8,
    )
    ax.set_xlabel("Query Index (Time Step)")
    ax.set_ylabel(f"Moving Average Regret (Window={window_size})")
    ax.grid(True, axis="both", linestyle="--", alpha=0.6)
    ax.set_ylim(bottom=0)

    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = []
    ordered_labels = []
    current_legend_map = dict(zip(labels, handles))
    for label in plot_order_display:
        if label in current_legend_map:
            ordered_handles.append(current_legend_map[label])
            ordered_labels.append(label)
    if ordered_handles:
        ax.legend(ordered_handles, ordered_labels, loc="best", title=None)
    else:
        ax.legend(loc="best", title=None)

    sns.despine(fig=fig)
    plt.tight_layout()
    plot_filename = f'{exp_dirs["base"].name}_moving_avg_regret_w{window_size}.png'
    plot_utils.save_plot(fig, exp_dirs["plots"], plot_filename)
    logger.info(
        f'Moving average regret plot saved to {exp_dirs["plots"] / plot_filename}'
    )


def generate_combined_regret_plot(
    detailed_results_df: pd.DataFrame, exp_dirs: dict, window_size: int = 50
):
    """Generates a side-by-side plot of cumulative and moving average regret.

    Args:
        detailed_results_df (pd.DataFrame): DataFrame with detailed simulation results.
        exp_dirs (dict): Experiment directories.
        window_size (int): Window size for moving average.
    """
    logger.info(f"Generating combined regret plot (Window={window_size})...")

    required_cols = ["run_id", "algorithm", "query_index", "step_regret"]
    if detailed_results_df is None or detailed_results_df.empty:
        logger.warning(
            "Detailed results DataFrame is empty. Skipping combined regret plot."
        )
        return
    if not all(col in detailed_results_df.columns for col in required_cols):
        logger.error(
            f"Detailed results DataFrame missing required columns {required_cols}. Cannot generate combined regret plot."
        )
        return
    plot_df = detailed_results_df.copy()
    plot_df = plot_df.sort_values(by=["run_id", "algorithm", "query_index"])
    plot_df["cumulative_regret"] = plot_df.groupby(["run_id", "algorithm"])[
        "step_regret"
    ].cumsum()
    plot_df["moving_avg_regret"] = plot_df.groupby(["run_id", "algorithm"])[
        "step_regret"
    ].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    algorithms_in_input = plot_df["algorithm"].unique()
    logger.info(
        f"[Combined Regret Plot] Algorithms found in input data: {list(algorithms_in_input)}"
    )

    plot_df["Algorithm Name"] = (
        plot_df["algorithm"].map(ALGO_DISPLAY_NAMES).fillna(plot_df["algorithm"])
    )

    plot_order_display = [
        ALGO_DISPLAY_NAMES.get(algo, algo) for algo in algorithms_in_input
    ]
    try:
        algo_order_map = {name: i for i, name in enumerate(ALGO_DISPLAY_NAMES.keys())}
        plot_order_display.sort(
            key=lambda x: algo_order_map.get(
                next((k for k, v in ALGO_DISPLAY_NAMES.items() if v == x), None),
                float("inf"),
            )
        )
    except Exception as e:
        logger.warning(
            f"Could not sort plot order based on ALGO_DISPLAY_NAMES, using default: {e}"
        )
        plot_order_display = sorted(plot_order_display)

    palette, dashes = get_regret_plot_styles(algorithms_in_input)

    logger.info(f"Combined Regret Plot Order: {plot_order_display}")
    logger.info(f"Combined Regret Plot Palette: {palette}")
    logger.info(f"Combined Regret Plot Dashes: {dashes}")
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    ax1, ax2 = axes

    if (
        "cumulative_regret" not in plot_df.columns
        or plot_df["cumulative_regret"].isnull().all()
    ):
        logger.warning("Cumulative regret data missing or all NaN. Skipping subplot.")
        ax1.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax1.transAxes)
    else:
        sns.lineplot(
            data=plot_df,
            x="query_index",
            y="cumulative_regret",
            hue="Algorithm Name",
            style="Algorithm Name",
            hue_order=plot_order_display,
            style_order=plot_order_display,
            palette=palette,
            dashes=dashes,
            errorbar="sd",
            ax=ax1,
            linewidth=1.5,
        )
        ax1.set_ylabel("Cumulative Regret")
        ax1.grid(True, axis="both", linestyle="--", alpha=0.6)
        ax1.set_ylim(bottom=0)
        handles1, labels1 = ax1.get_legend_handles_labels()
        ordered_handles1, ordered_labels1 = [], []
        current_legend_map1 = dict(zip(labels1, handles1))
        for label in plot_order_display:
            if label in current_legend_map1:
                ordered_handles1.append(current_legend_map1[label])
                ordered_labels1.append(label)
        if ordered_handles1:
            ax1.legend(ordered_handles1, ordered_labels1, loc="best", title=None)
        else:
            ax1.legend(loc="best", title=None)

    if (
        "moving_avg_regret" not in plot_df.columns
        or plot_df["moving_avg_regret"].isnull().all()
    ):
        logger.warning(
            "Moving average regret data missing or all NaN. Skipping subplot."
        )
        ax2.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax2.transAxes)
    else:
        sns.lineplot(
            data=plot_df,
            x="query_index",
            y="moving_avg_regret",
            hue="Algorithm Name",
            style="Algorithm Name",
            hue_order=plot_order_display,
            style_order=plot_order_display,
            palette=palette,
            dashes=dashes,
            errorbar="sd",
            ax=ax2,
            linewidth=1.5,
        )
        ax2.set_ylabel(f"Moving Average Regret (W={window_size})")
        ax2.grid(True, axis="both", linestyle="--", alpha=0.6)
        ax2.set_ylim(bottom=0)

        handles2, labels2 = ax2.get_legend_handles_labels()
        ordered_handles2, ordered_labels2 = [], []
        current_legend_map2 = dict(zip(labels2, handles2))
        for label in plot_order_display:
            if label in current_legend_map2:
                ordered_handles2.append(current_legend_map2[label])
                ordered_labels2.append(label)
        if ordered_handles2:
            ax2.legend(ordered_handles2, ordered_labels2, loc="best", title=None)
        else:
            ax2.legend(loc="best", title=None)

    ax1.set_xlabel("Query Index (Time Step)")
    ax2.set_xlabel("")

    sns.despine(fig=fig)
    plt.tight_layout()
    plot_filename = f'{exp_dirs["base"].name}_combined_regret_w{window_size}.png'
    plot_utils.save_plot(fig, exp_dirs["plots"], plot_filename)
    logger.info(f'Combined regret plot saved to {exp_dirs["plots"] / plot_filename}')


# ============================================================================
# MODEL CHOICE TIMELINE PLOTS
# ============================================================================


def _capitalize_model_name(model_id: str) -> str:
    """Helper function to capitalize model names for display."""
    if not isinstance(model_id, str):
        return str(model_id)
    return "-".join(part.capitalize() for part in model_id.split("-"))


def generate_model_choice_timeline(
    detailed_results_df, algo_name, ax, all_model_ids
):
    """Generates a timeline plot for a single algorithm's LAST run onto a given axis,
       with Y-axis sorted alphabetically.

    Args:
        detailed_results_df (pd.DataFrame): DataFrame with detailed simulation results for ALL runs.
        algo_name (str): The specific algorithm to plot.
        ax (matplotlib.axes.Axes): The axes object to plot onto.
        all_model_ids (list): List of ALL model IDs available in the experiment (for consistent Y-axis).

    Returns:
        set: Set of unique models chosen by this algorithm in the last run,
             or an empty set if plotting fails.
    """

    if detailed_results_df is None or detailed_results_df.empty:
        return set()
    required_cols = ["run_id", "algorithm", "query_index", "chosen_model"]

    if not all(col in detailed_results_df.columns for col in required_cols):
        return set()
    last_run_id = detailed_results_df["run_id"].max()
    algo_last_run_df = detailed_results_df[
        (detailed_results_df["algorithm"] == algo_name)
        & (detailed_results_df["run_id"] == last_run_id)
    ].copy()

    if not algo_last_run_df.empty:
        pass
    else:
        logger.warning(
            f"    Timeline Plot Debug ({algo_name}): Filtered data is EMPTY for last run {last_run_id}."
        )
        ax.text(
            0.5,
            0.5,
            "No Data (Last Run)",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"{algo_name} (No Data)")
        return set()

    if not all_model_ids:
        logger.error("No model IDs provided for Y-axis.")
        return set()
    y_axis_model_order = sorted(list(set(all_model_ids)))
    logger.debug(f"Using alphabetical Y-axis order: {y_axis_model_order}")
    chosen_models_in_run = set(algo_last_run_df["chosen_model"].unique())

    if not chosen_models_in_run:
        logger.warning(f"No models were chosen by {algo_name} in the last run.")
        ax.set_yticks(range(len(y_axis_model_order)))
        ax.set_yticklabels(y_axis_model_order)
        ax.set_ylim(-0.5, len(y_axis_model_order) - 0.5)
        ax.set_title(f"{algo_name} (No Choices)")
        return set()

    try:
        model_colors = {
            model: plot_utils.get_color(model) for model in chosen_models_in_run
        }
    except Exception as e:
        logger.warning(
            f"Color mapping failed for {algo_name}: {e}. Using fallback cmap."
        )
        cmap = (
            plt.cm.get_cmap("tab20", len(chosen_models_in_run))
            if len(chosen_models_in_run) <= 20
            else plt.cm.get_cmap("viridis", len(chosen_models_in_run))
        )
        model_colors = {
            model: cmap(i) for i, model in enumerate(sorted(list(chosen_models_in_run)))
        }
    algo_last_run_df["chosen_model_cat"] = pd.Categorical(
        algo_last_run_df["chosen_model"], categories=y_axis_model_order, ordered=True
    )

    plot_df = algo_last_run_df.dropna(subset=["chosen_model_cat"])
    if plot_df.empty:
        logger.warning(
            f"No valid categorical data points to plot for {algo_name} after mapping."
        )
        ax.set_yticks(range(len(y_axis_model_order)))
        ax.set_yticklabels(y_axis_model_order)
        ax.set_ylim(-0.5, len(y_axis_model_order) - 0.5)
        ax.set_title(f"{algo_name} (Plotting Error)")
        return set()

    point_colors = plot_df["chosen_model"].map(model_colors)
    ax.scatter(
        plot_df["query_index"],
        plot_df["chosen_model_cat"].cat.codes,
        color=point_colors,
        s=5,
        marker="|",
        alpha=1.0,
    )
    ax.set_xlabel("Query Index (Time Step)")
    ax.set_ylabel("Chosen Model")
    ax.set_title(f"{ALGO_DISPLAY_NAMES.get(algo_name, algo_name)}")
    ax.set_yticks(range(len(y_axis_model_order)))

    y_axis_display_labels = [
        _capitalize_model_name(name) for name in y_axis_model_order
    ]
    ax.set_yticklabels(y_axis_display_labels)
    ax.set_ylim(-0.5, len(y_axis_model_order) - 0.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    return chosen_models_in_run


# ============================================================================
# PLOT-ONLY ENTRY POINT
# ============================================================================


def main():
    """Regenerate plots from existing experiment data."""
    # Get results directory and timestamp
    results_dir = Path(__file__).parent / "results"
    if len(sys.argv) > 1:
        timestamp = sys.argv[1]
    else:
        files = sorted(results_dir.glob("*_summary_stats_*.csv"))
        if not files:
            logger.error("No results found. Run the experiment first.")
            return
        timestamp = "_".join(files[-1].stem.split("_")[-2:])

    logger.info(f"Loading results from: {timestamp}")
    exp_dirs = config_loader.setup_experiment_dirs("X1_algorithm_comparison")

    # Load required data files
    summary_df = pd.read_csv(
        results_dir / f"X1_algorithm_comparison_summary_stats_{timestamp}.csv"
    )

    # For a5, detailed results is a single file without timestamp
    detailed_df = pd.read_csv(results_dir / "X1_algorithm_comparison_detailed_results.csv")

    # Load models config if available
    try:
        models_file = project_root / "experiments" / "config" / "models.yaml"
        with open(models_file, 'r') as f:
            models_data = yaml.safe_load(f)
        models_df_plot = pd.DataFrame.from_dict(models_data, orient='index')
        models_df_plot.reset_index(inplace=True)
        models_df_plot.rename(columns={'index': 'model_id', 'name': 'hf_name'}, inplace=True)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.warning(f"Could not load or parse models.yaml: {e}. Model sizes will not be available for plotting.")
        models_df_plot = None

    # Note: Individual model performance data is loaded from DB below, not from a1

    # Load data from DB for comprehensive plots
    conn = data_loader.connect_db()
    model_specs_df = data_loader.get_model_specs(conn)
    all_model_ids = model_specs_df["model_id"].tolist()

    # Get query IDs from detailed results
    query_ids = detailed_df["query_id"].unique().tolist()
    _, _, raw_results_df = data_loader.check_data_completeness(
        conn, query_ids, all_model_ids
    )
    conn.close()

    # Normalize the raw results
    norm_df = (
        analysis_utils.normalize_metrics(raw_results_df.copy())
        if raw_results_df is not None
        else None
    )

    # Process detailed results for cumulative metrics
    detailed_df = detailed_df.sort_values(by=["run_id", "algorithm", "query_index"])
    detailed_df["cumulative_regret"] = detailed_df.groupby(["run_id", "algorithm"])[
        "step_regret"
    ].cumsum()

    # Generate plots
    logger.info("Generating plots...")

    # 1. Pareto plot (normalized energy)
    if norm_df is not None:
        generate_pareto_plot(summary_df, norm_df, exp_dirs, models_df)
        logger.info("Generated Pareto plot (normalized energy)")

    # 2. Pareto plot (total energy Wh)
    if raw_results_df is not None and norm_df is not None:
        generate_pareto_plot_wh(
            summary_df, detailed_df, raw_results_df, norm_df, exp_dirs, models_df
        )
        logger.info("Generated Pareto plot (total energy)")

    # 3. Bar plot (normalized metrics)
    has_error_bar_data = all(
        col in summary_df.columns
        for col in ["Std Norm Accuracy", "std_energy_per_token_norm", "Count"]
    )
    generate_grouped_norm_metric_bar_plot(summary_df, exp_dirs, has_error_bar_data)
    logger.info("Generated bar plot (normalized metrics)")

    # 4. Bar plot (total energy)
    if raw_results_df is not None:
        generate_grouped_bar_plot_total_energy(
            summary_df, detailed_df, raw_results_df, exp_dirs
        )
        logger.info("Generated bar plot (total energy)")

    # 5. Regret plots
    generate_cumulative_regret_plot(detailed_df, exp_dirs)
    generate_moving_average_regret_plot(detailed_df, exp_dirs)
    generate_combined_regret_plot(detailed_df, exp_dirs)
    logger.info("Generated regret plots")

    # 6. Model choice timeline
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    all_chosen_models = set()
    algorithms = detailed_df["algorithm"].unique()[:8]  # Max 8 subplots
    for i, algo in enumerate(algorithms):
        chosen = generate_a5_model_choice_timeline(
            detailed_df, algo, axes[i], all_model_ids
        )
        all_chosen_models.update(chosen)

    for j in range(len(algorithms), 8):
        axes[j].axis("off")

    plt.tight_layout()
    plot_utils.save_plot(
        fig,
        exp_dirs["plots"],
        f"{exp_dirs['base'].name}_model_choice_timeline_last_run.png",
    )
    logger.info("Generated model choice timeline")

    logger.info(f"All plots saved to: {exp_dirs['plots']}")


if __name__ == "__main__":
    main()
