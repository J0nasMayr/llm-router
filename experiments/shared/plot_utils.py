import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
COLORS = {
    "epsilon_greedy": "#4477AA",
    "contextual_epsilon_greedy": "#66CCEE",
    "linear_epsilon_greedy": "#228833",
    "linucb": "#228833",
    "thompson_sampling": "#66CC99",
    "largest": "#CC3311",
    "smallest": "#EE7733",
    "random": "#BBBBBB",
    "accuracy": "#AA3377",
    "efficiency": "#EE3377",
    "oracle": "#000000",
    "none": "#BBBBBB",
    "task": "#EE7733",
    "cluster": "#66CCEE",
    "complex": "#AA3377",
    "task_cluster": "#4477AA",
    "task_complex": "#EE3377",
    "cluster_complex": "#66CC99",
    "full": "#228833",
    "default_bar_color": "#337ab7",
    "default": [
        "#4477AA",
        "#66CCEE",
        "#228833",
        "#66CC99",
        "#CC3311",
        "#EE7733",
        "#BBBBBB",
        "#AA3377",
    ],
}
MARKERS = {
    "largest": "^",
    "smallest": "v",
    "random": "D",
    "accuracy": "P",
    "efficiency": "*",
    "oracle": ".",
    "linucb": "o",
    "thompson_sampling": "s",
    "epsilon_greedy": "H",
    "contextual_epsilon_greedy": ">",
    "default": "o",
}


def setup_plotting(
    style="seaborn-v0_8-whitegrid", context="notebook", font_scale=1.2, figsize=(10, 6)
):
    """Set up default plotting style using seaborn."""
    try:
        plt.style.use(style)
        plt.rcParams["figure.figsize"] = figsize
        sns.set_context(context, font_scale=font_scale)
        logger.info(f"Plotting style set to '{style}' with context '{context}'")
    except Exception as e:
        logger.error(f"Failed to set plotting style: {e}")


def get_color(name: str):
    """Get the predefined color for an algorithm or baseline name."""
    return COLORS.get(name, COLORS["default"][0])


def get_marker(name: str):
    """Get the predefined marker for an algorithm or baseline name."""
    return MARKERS.get(name, MARKERS["default"])


def save_plot(fig, plot_dir: Path, plot_name: str, dpi=300, bbox_inches="tight"):
    """
    Save the given matplotlib figure to the specified directory and filename.

    Args:
        fig: Matplotlib figure object.
        plot_dir (Path): Directory to save the plot in.
        plot_name (str): Filename for the plot (e.g., 'figure1.png').
        dpi (int): Dots per inch for saving.
        bbox_inches (str): Bounding box setting for saving.
    """
    if not plot_dir.is_dir():
        logger.error(
            f"Plot directory does not exist: {plot_dir}. Cannot save plot '{plot_name}'."
        )
        return

    save_path = plot_dir / plot_name
    try:
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {save_path}: {e}")
    finally:
        plt.close(fig)


def get_experiment_timestamp(experiment_name: str, results_dir: Path, file_pattern: str):
    """
    Get timestamp for experiment data, either from command line or latest file.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'X0_warmup')
        results_dir: Path to results directory
        file_pattern: Glob pattern to find result files (e.g., '*_detailed_results_*.csv')
    
    Returns:
        tuple: (timestamp: str, success: bool)
    """
    # Check command line argument
    if len(sys.argv) > 1:
        timestamp = sys.argv[1]
        logger.info(f"Using provided timestamp: {timestamp}")
        return timestamp, True
    
    # Find latest file
    files = sorted(results_dir.glob(file_pattern))
    if not files:
        logger.error(f"No results found matching {file_pattern}. Run the {experiment_name} experiment first.")
        return None, False
    
    # Extract timestamp from filename
    timestamp = "_".join(files[-1].stem.split("_")[-2:])
    logger.info(f"Using latest results: {timestamp}")
    return timestamp, True


def setup_experiment_directories(base_exp_dir: Path):
    """
    Set up standard experiment directories structure.
    
    Args:
        base_exp_dir: Base experiment directory path
        
    Returns:
        dict: Dictionary with 'base', 'results', and 'plots' paths
    """
    exp_dirs = {
        "base": base_exp_dir,
        "results": base_exp_dir / "results",
        "plots": base_exp_dir / "plots"
    }
    exp_dirs["plots"].mkdir(exist_ok=True)
    return exp_dirs
