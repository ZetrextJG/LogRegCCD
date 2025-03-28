import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

from metrics import MetricsCollated


def plot_betas(betas: np.ndarray, lmbdas: np.ndarray):
    # betas: (num_lmbdas, D)
    # lmbdas: (num_lmbdas,)
    D = betas.shape[1]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    for d in range(D):
        sns.lineplot(x=lmbdas, y=betas[:, d], label=f"Beta {d+1}", ax=ax)
        sns.scatterplot(x=lmbdas, y=betas[:, d], ax=ax, s=20)

    ax.set_xscale("log")
    ax.set_xlabel("Lambda (log scale)")
    ax.set_ylabel("Beta Coefficients")
    ax.set_title("Regularization Path")
    if ax.legend_ is not None:
        ax.legend_.remove()

    return fig


def plot_metrics(metrics: MetricsCollated, lmbdas: np.ndarray, metric_name: str):
    """
    Plots the selected metric against lambda values.

    Parameters:
    - metrics: MetricsCollated - Dictionary of metric lists.
    - lmbdas: np.ndarray - Array of lambda values.
    - metric_name: str - The metric to plot (e.g., "accuracy", "precision").

    Returns:
    - fig: Matplotlib figure object.
    """

    if metric_name not in metrics:
        raise ValueError(
            f"Invalid metric name '{metric_name}'. Choose from {list(metrics.keys())}"
        )

    sns.set_style("whitegrid")  # Set Seaborn style
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.lineplot(x=lmbdas, y=metrics[metric_name], marker="o", ax=ax)

    ax.set_xscale("log")  # Log scale for lambda
    ax.set_ylim(0, 1)  # Set y-axis limits
    ax.set_xlabel("Lambda")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"{metric_name.capitalize()} vs Lambda")

    return fig  # Return figure object instead of showing it
