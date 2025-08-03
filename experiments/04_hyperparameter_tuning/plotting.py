# experiments/04_hyperparameter_tuning/plotting.py

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.shared import config_loader, plot_utils, results_handler

logger = logging.getLogger(__name__)


# ============================================================================
# HYPERPARAMETER PERFORMANCE PLOTTING
# ============================================================================
def generate_hyperparameter_performance_plot(summary_df: pd.DataFrame, exp_dirs: dict):
    """
    Generates plots showing the performance (mean final regret) against
    varying hyperparameters for each algorithm.

    Args:
        summary_df (pd.DataFrame): DataFrame containing summary statistics,
                                   including 'algorithm', 'mean_final_regret',
                                   and columns for each hyperparameter tested.
        exp_dirs (dict): Dictionary containing experiment directories,
                         including 'plots'.
    """

    plots_dir = Path(exp_dirs["plots"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    if summary_df.empty:
        logger.warning("Summary DataFrame is empty. Skipping hyperparameter plots.")
        return
    known_cols = ["algorithm", "mean_final_regret", "std_final_regret", "min", "max"]
    potential_hyperparam_cols = [
        col for col in summary_df.columns if col not in known_cols
    ]

    algorithms = summary_df["algorithm"].unique()
    for algo in algorithms:
        algo_df = summary_df[summary_df["algorithm"] == algo].copy()
        hyperparam_cols_for_algo = []
        for col in potential_hyperparam_cols:
            if col in algo_df.columns and algo_df[col].nunique() > 1:
                hyperparam_cols_for_algo.append(col)
            elif col in algo_df.columns and algo_df[col].nunique() == 1:
                pass

        if not hyperparam_cols_for_algo:
            logger.info(
                f"No hyperparameters with multiple values found for {algo}. Skipping plot."
            )
            continue
        try:
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            if len(hyperparam_cols_for_algo) == 1:
                param = hyperparam_cols_for_algo[0]
                if pd.api.types.is_numeric_dtype(algo_df[param]):
                    algo_df[param] = pd.to_numeric(algo_df[param])
                    sns.lineplot(
                        data=algo_df, x=param, y="mean_final_regret", marker="o", ax=ax
                    )

                    std_dev_data = algo_df.get("std_final_regret")
                    if std_dev_data is not None and std_dev_data.notna().any():
                        ax.errorbar(
                            algo_df[param],
                            algo_df["mean_final_regret"],
                            yerr=std_dev_data,
                            fmt="none",
                            color="grey",
                            alpha=0.5,
                            capsize=5,
                        )
                    else:
                        logger.warning(
                            f"Standard deviation data invalid or missing for {algo} with param {param}. Skipping error bars."
                        )

                    ax.set_xlabel(f"Hyperparameter: {param}")
                    ax.set_ylabel("Mean Final Regret")

                else:
                    sns.barplot(data=algo_df, x=param, y="mean_final_regret", ax=ax)

                    std_dev_data = algo_df.get("std_final_regret")
                    if std_dev_data is not None and std_dev_data.notna().any():
                        positions = range(len(algo_df))
                        ax.errorbar(
                            x=positions,
                            y=algo_df["mean_final_regret"],
                            yerr=std_dev_data,
                            fmt="none",
                            color="grey",
                            alpha=0.5,
                            capsize=5,
                        )
                    else:
                        logger.warning(
                            f"Standard deviation data invalid or missing for {algo} with param {param}. Skipping error bars."
                        )

                    plt.xticks(rotation=45, ha="right")
                    ax.set_xlabel(f"Hyperparameter: {param}")
                    ax.set_ylabel("Mean Final Regret")

                plt.tight_layout()
                plot_filename = plots_dir / f"04_hyperparameter_tuning_hyperparam_perf_{algo}.png"
                fig.savefig(plot_filename, bbox_inches="tight")
                logger.info(f"Saved hyperparameter performance plot: {plot_filename}")
                plt.close(fig)
            elif len(hyperparam_cols_for_algo) >= 2:
                plt.close(fig)
                param1, param2 = (
                    hyperparam_cols_for_algo[0],
                    hyperparam_cols_for_algo[1],
                )
                if algo_df[param1].nunique() >= algo_df[param2].nunique():
                    x_param, style_param = param1, param2
                else:
                    x_param, style_param = param2, param1

                if pd.api.types.is_numeric_dtype(algo_df[x_param]):
                    algo_df[x_param] = pd.to_numeric(algo_df[x_param])
                if pd.api.types.is_numeric_dtype(algo_df[style_param]):
                    algo_df[style_param] = pd.to_numeric(algo_df[style_param])

                col_param = (
                    hyperparam_cols_for_algo[2]
                    if len(hyperparam_cols_for_algo) > 2
                    else None
                )
                col_wrap_val = 3 if col_param else None

                if len(hyperparam_cols_for_algo) > 2:
                    logger.warning(
                        f"Plotting first two hyperparameters ({x_param}, {style_param}) for {algo}, using {col_param} for columns."
                    )
                else:
                    pass

                g = sns.relplot(
                    data=algo_df,
                    x=x_param,
                    y="mean_final_regret",
                    hue=style_param,
                    style=style_param,
                    kind="line",
                    marker="o",
                    col=col_param,
                    col_wrap=col_wrap_val,
                    facet_kws={"sharey": True, "sharex": True},
                    height=4 if col_param else 5,
                    aspect=1.2 if col_param else 1.5,
                )

                g.set_axis_labels(f"Hyperparameter: {x_param}", "Mean Final Regret")
                g.legend.set_title(f"{style_param}")
                plt.tight_layout(rect=[0, 0, 1, 0.97])

                plot_filename = plots_dir / f"04_hyperparameter_tuning_hyperparam_perf_{algo}.png"
                g.savefig(plot_filename, bbox_inches="tight")
                logger.info(f"Saved hyperparameter performance plot: {plot_filename}")
                plt.close(g.fig)
        except Exception as e:
            logger.error(
                f"Failed to generate hyperparameter plot for {algo}: {e}", exc_info=True
            )
            plt.close("all")


# ============================================================================
# MODEL CHOICE TIMELINE PLOTTING
# ============================================================================
def generate_model_choice_timeline(results_df, exp_dirs, strategy_name):
    """Generates a timeline plot showing individual model choices over time for a specific strategy."""

    if results_df is None or results_df.empty:
        logger.warning(
            f"Results DataFrame for {strategy_name} is empty. Skipping model choice timeline plot."
        )
        return

    output_dir = Path(exp_dirs["plots"])
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = ["chosen_model", "query_index", "run_id"]
    if not all(col in results_df.columns for col in required_cols):
        logger.error(
            f"Required columns ('chosen_model', 'query_index', 'run_id') not found in detailed results for {strategy_name}. Skipping timeline plot."
        )
        return
    model_order = sorted(results_df["chosen_model"].unique())
    num_models = len(model_order)
    try:
        model_colors = {model: plot_utils.get_color(model) for model in model_order}
        logger.info(f"Using plot_utils colors for timeline: {model_colors}")
    except Exception as e:
        logger.warning(
            f"Could not use plot_utils for all colors ({e}). Falling back to colormap."
        )
        colors = (
            plt.cm.get_cmap("tab20", num_models)
            if num_models <= 20
            else plt.cm.get_cmap("viridis", num_models)
        )
        model_colors = {model: colors(i) for i, model in enumerate(model_order)}
    plt.figure(figsize=(18, max(6, num_models * 0.4)))

    plotted_models = set()

    for model in model_order:
        model_choices = results_df[results_df["chosen_model"] == model]
        if not model_choices.empty:
            plt.scatter(
                model_choices["query_index"],
                model_choices["chosen_model"],
                color=model_colors[model],
                label=model,
                s=5,
                marker="|",
                alpha=0.1,
            )
            plotted_models.add(model)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="|",
            color=model_colors[model],
            linestyle="None",
            markersize=10,
        )
        for model in model_order
    ]
    labels = model_order
    plt.legend(
        handles,
        labels,
        title="Models",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        markerscale=2,
    )

    plt.xlabel("Query Index")
    plt.ylabel("Chosen Model")

    plt.yticks(model_order)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    filename = output_dir / f"04_hyperparameter_tuning_model_choice_timeline_{strategy_name}.png"
    try:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved model choice timeline plot to {filename}")
    except Exception as e:
        logger.error(f"Failed to save timeline plot {filename}: {e}")
    plt.close()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def _format_params_for_filename(params_dict):
    """Formats a dictionary of parameters into a string suitable for filenames."""
    if not params_dict:
        return ""
    parts = []
    for key, value in sorted(params_dict.items()):
        if value is None:
            continue

        value_str = str(value).replace(".", "_")
        parts.append(f"{key}_{value_str}")
    return "_" + "_".join(parts) if parts else ""


# ============================================================================
# BEST STRATEGY ANALYSIS
# ============================================================================
def generate_best_strategy_timeline_plot(summary_df: pd.DataFrame, exp_dirs: dict):
    """Finds the best strategy in the summary, loads its detailed data, and plots the model choice timeline."""

    logger.info("Generating model choice timeline for the best strategy...")

    if (
        summary_df.empty
        or "mean_final_regret" not in summary_df.columns
        or "algorithm" not in summary_df.columns
    ):
        logger.error(
            "Summary DataFrame is empty or missing required columns ('mean_final_regret', 'algorithm'). Cannot find best strategy."
        )
        return

    try:
        best_row_idx = summary_df["mean_final_regret"].idxmin()
        best_row = summary_df.loc[best_row_idx]
        best_algo = best_row["algorithm"]
        known_cols = [
            "algorithm",
            "mean_final_regret",
            "std_final_regret",
            "min",
            "max",
        ]
        potential_hyperparam_cols = [
            col for col in summary_df.columns if col not in known_cols
        ]
        hyperparam_cols_for_algo = []
        for col in potential_hyperparam_cols:
            if col in best_row.index and pd.notna(best_row[col]):
                algo_df_slice = summary_df[summary_df["algorithm"] == best_algo]
                if col in algo_df_slice.columns and algo_df_slice[col].nunique() > 1:
                    hyperparam_cols_for_algo.append(col)
                elif col in algo_df_slice.columns and algo_df_slice[col].nunique() == 1:
                    hyperparam_cols_for_algo.append(col)
        best_params = {col: best_row[col] for col in hyperparam_cols_for_algo}

        logger.info(
            f"Best strategy identified: Algorithm='{best_algo}', Parameters={best_params}, Regret={best_row['mean_final_regret']:.4f}"
        )
        param_str = _format_params_for_filename(best_params)
        strategy_name_for_file = f"{best_algo}{param_str}"  # For filename
        f"{best_algo} ({', '.join(f'{k}={v}' for k, v in best_params.items())})"  # For plot title/logging
        file_prefix = f"04_hyperparameter_tuning_detailed_results_{strategy_name_for_file}"
        results_dir = Path(exp_dirs.get("results", None))

        if not results_dir or not results_dir.exists():
            logger.error(
                f"Results directory not found or specified in exp_dirs: {results_dir}. Cannot load detailed results."
            )
            return

        logger.info(
            f"Attempting to load detailed results from '{results_dir}' with prefix '{file_prefix}'..."
        )
        detailed_df = results_handler.load_results(
            results_dir, prefix=file_prefix, file_type="detailed_results"
        )
        if detailed_df is not None and not detailed_df.empty:
            logger.info(
                f"Loaded detailed results for best strategy ({len(detailed_df)} rows). Generating timeline plot..."
            )
            generate_model_choice_timeline(
                detailed_df, exp_dirs, strategy_name_for_file
            )
        else:
            logger.warning(
                f"Could not load detailed results for the best strategy (Prefix: {file_prefix}). Timeline plot will not be generated."
            )
    except KeyError as e:
        logger.error(
            f"Missing expected column in summary_df for finding best strategy: {e}"
        )
    except ValueError as e:
        logger.error(
            f"Error finding best strategy (potentially empty summary or no valid regret values): {e}"
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while generating the best strategy timeline: {e}",
            exc_info=True,
        )


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
    exp_dirs = config_loader.setup_experiment_dirs("04_hyperparameter_tuning")

    # Load summary stats
    summary_df = pd.read_csv(
        results_dir / f"04_hyperparameter_tuning_summary_stats_{timestamp}.csv"
    )

    # Generate hyperparameter performance plots
    logger.info("Generating hyperparameter performance plots...")
    generate_hyperparameter_performance_plot(summary_df, exp_dirs)

    logger.info(f"All plots saved to: {exp_dirs['plots']}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    main()
