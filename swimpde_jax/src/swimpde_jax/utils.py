import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import functools
import time


def compute_metrics(
    u_pred: npt.ArrayLike, u_exact: npt.ArrayLike
) -> tuple[npt.ArrayLike, float]:
    if u_pred.shape != u_exact.shape:
        raise ValueError(f"Cannot compute metrics: {u_pred.shape=}, {u_exact.shape=}.")

    abs = np.abs(u_pred - u_exact)
    rel_l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    return abs, rel_l2

def normalized_l1_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes the normalized L1 loss:
    
    L(θ) = (1/M) * Σ |u(x_i, θ) - u(x_i)| / σ
    
    where:
    - predictions: np.ndarray of predicted values u(x_i, θ)
    - ground_truth: np.ndarray of true values u(x_i)
    - σ: standard deviation of the true values
    
    Returns:
    - Normalized L1 loss (float)
    """
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    M = len(targets)  # Number of test points
    sigma = np.std(targets)  # Standard deviation of true values
    
    if sigma == 0:
        raise ValueError("Standard deviation of ground truth values is zero, normalization is undefined.")

    loss = np.mean(np.abs(predictions - targets) / sigma)
    return loss


def plot_results(
    x_eval: npt.ArrayLike,
    t_eval: npt.ArrayLike,
    u_exact: npt.ArrayLike,
    u_pred: npt.ArrayLike,
    abs_error: npt.ArrayLike,
) -> plt.Figure:
    fig, (axes_full, axes_samples) = plt.subplots(2, 3, figsize=(18, 8))

    # Plot solutions for all t.
    data_dict = {
        "Ground truth": u_exact,
        "Predictions": u_pred,
        "Absolute error": abs_error,
    }
    extent = [x_eval.min(), x_eval.max(), t_eval.min(), t_eval.max()]
    for ax, (label, data) in zip(axes_full, data_dict.items()):
        im = ax.imshow(data.T, extent=extent, origin="lower")
        fig.colorbar(im, ax=ax, location="bottom")
        ax.set_title(label)

    # Plot solutions and for certain values of t.
    for i, ax in enumerate(axes_samples):
        idx = i * t_eval.shape[0] // 3
        ax.plot(x_eval, u_exact[:, idx], "b-", label="Ground truth")
        ax.plot(x_eval, u_pred[:, idx], "r--", label="Predictions")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u(t,x)$")
        ax.set_aspect("equal")
        ax.set_title(f"$t = {t_eval[idx]:.3f}$")

    return fig


def plot_static_results_2D(
    inputs: npt.ArrayLike,
    targets: npt.ArrayLike,
    predictions: npt.ArrayLike,
    abs_error: npt.ArrayLike,
    show=True
) -> plt.Figure:
    
    fig, (axes_full) = plt.subplots(1, 3, figsize=(18, 8))

    # Plotting data
    data_dict = {
        "Ground truth": targets,
        "Predictions": predictions,
        "Absolute error": abs_error,
    }

    extent = [inputs[:,0].min(), inputs[:,0].max(), inputs[:,1].min(), inputs[:,1].max()]
    for ax, (label, data) in zip(axes_full, data_dict.items()):
        im = ax.imshow(data.T, extent=extent, origin="lower")
        fig.colorbar(im, ax=ax, location="bottom", format='%.0e')
        ax.set_title(label)
    if show:
        plt.show()

    return fig

def visualize_domain_points_in_2D(domain) -> None:
    """
    Visualizes the boundary and interior points of a domain.
    Parameters:
        domain: A domain object with `boundary_points` and `interior_points` attributes.
                Both attributes are expected to be 2D arrays where each row is a point (x, y).
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Plot boundary points
    boundary_plot = ax.scatter(
        *domain.boundary_points.T,
        color='orange', 
        label='Boundary Points', 
        alpha=1.0
    )

    # Plot interior points
    interior_plot = ax.scatter(
        *domain.interior_points.T,
        color='gray', 
        label='Interior Points', 
        alpha=0.2
    )

    # Add legend and labels
    ax.legend()
    ax.set_title("Domain Points")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # Show plot
    plt.show()

def plot_swim_dd_1D(inputs, targets, prediction, local_predictions, window_fn_evaluations, legend=True, title=None):
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    if title is not None:
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
    ax[0].scatter(inputs.ravel(), prediction.ravel(), marker='x', label='Prediction', lw=2, c='orange')
    ax[0].plot(inputs, targets,label='True solution', lw=2, c='green', alpha=0.5)
    ax[0].set_title("Prediction - Target Comparison")
    ax[0].set_xlabel("inputs")
    ax[0].set_ylabel("outputs")
    ax[0].grid(True)
    if legend: ax[0].legend()

    for i, local_pred in enumerate(local_predictions):
        ax[1].plot(inputs, local_pred.flatten(), '-.', label=f"pred {i+1}", lw=2)

    ax[1].set_title("Local network predictions")
    ax[1].set_xlabel("inputs")
    ax[1].set_ylabel("outputs")
    ax[1].grid(True)
    if legend: ax[1].legend()

    partition_of_unity = np.zeros_like(window_fn_evaluations[0])
    for j, window_eval in enumerate(window_fn_evaluations):
        ax[2].plot(inputs, window_eval, '-.', label=f"func {j+1}", lw=2)
        partition_of_unity += window_eval
    mask = np.where(partition_of_unity != 0, True, False)   
    ax[2].plot(inputs[mask], partition_of_unity[mask], '--', label=f"Unity", color="black", lw=4, alpha=0.4)
    ax[2].set_title("Window_functions")
    ax[2].set_xlabel("inputs")
    ax[2].set_ylabel("outputs")
    ax[2].grid(True)
    if legend: ax[1].legend()

    plt.tight_layout()
    plt.show()

def plot_swim_mldd_1D(inputs, targets, prediction, level_predictions, level_base_local_predictions, level_base_window_fn_evaluations, legend=True, title=None):
    
    num_levels = len(level_predictions)

    # Adjusted figure size for better spacing
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    
    if title is not None:
        fig.suptitle(title, fontsize=20, fontweight='bold')

    # Convert axes array to flat indexing for robustness
    axes = axes.flat

    # 1. Prediction vs. Target
    axes[0].scatter(inputs.ravel(), prediction.ravel(), marker='x', label='Prediction', lw=2, c='orange')
    axes[0].plot(inputs, targets, label='True solution', lw=2, c='green', alpha=0.6)
    axes[0].set_title("Prediction - Target Comparison")
    axes[0].set_xlabel("Inputs")
    axes[0].set_ylabel("Outputs")
    axes[0].grid(True)
    if legend: axes[0].legend(loc="best")

    # 2. Level Predictions
    for i, level_prediction in enumerate(level_predictions):
        axes[1].plot(inputs, level_prediction.flatten(), '-.', label=f"Level {i+1} Prediction", lw=2)
    
    axes[1].set_title("Level Predictions")
    axes[1].set_xlabel("Inputs")
    axes[1].set_ylabel("Outputs")
    axes[1].grid(True)
    if legend: axes[1].legend(loc="best")

    # 3. Level-Based Local Predictions
    for i in range(num_levels):
        first_label = True  # Track the first entry for each level
        for j, local_prediction in enumerate(level_base_local_predictions[i]):
            label = f"Level {i+1} Local Preds" if first_label else None  # Label only for the first entry
            axes[2].plot(inputs, local_prediction.flatten(), '--', label=label, lw=1.5, alpha=0.8)
            first_label = False  # Ensure subsequent plots have no label

    axes[2].set_title("Level-Based Local Predictions")
    axes[2].set_xlabel("Inputs")
    axes[2].set_ylabel("Outputs")
    axes[2].grid(True)
    if legend: axes[2].legend(loc="best")

    # 4. Window Functions & Partition of Unity
    for i in range(num_levels):
        partition_of_unity = np.zeros_like(level_base_window_fn_evaluations[0])

        for window_eval in level_base_window_fn_evaluations[i]:
            axes[3].plot(inputs, window_eval, '-.', label=f"Level {i+1} Functions", lw=1.5)
            partition_of_unity += window_eval

        # Mask where partition of unity is nonzero
        mask = partition_of_unity != 0
        axes[3].plot(inputs[mask], partition_of_unity[mask], '--', label=f"Unity {i+1}", color="black", lw=3, alpha=0.5)

    axes[3].set_title("Window Functions")
    axes[3].set_xlabel("Inputs")
    axes[3].set_ylabel("Outputs")
    axes[3].grid(True)
    if legend: axes[3].legend(loc="best")

    plt.tight_layout()
    plt.show()

def plot_swim_1D(inputs: npt.ArrayLike, targets: npt.ArrayLike, prediction: npt.ArrayLike, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.scatter(inputs.ravel(), prediction.ravel(), marker='x', label='Prediction', lw=2, c='orange')
    ax.plot(inputs, targets,label='True solution', lw=2, c='green', alpha=0.5)
    ax.set_title("Prediction - Target Comparison")
    ax.set_xlabel("inputs")
    ax.set_ylabel("outputs")
    ax.grid(True)
    ax.legend()
    if title is not None:
        fig.suptitle(title, fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_swim_2D(  
    inputs: npt.ArrayLike,
    targets: npt.ArrayLike | None,
    prediction: npt.ArrayLike ,
    abs_error: npt.ArrayLike | None ,
    level_predictions: list[npt.ArrayLike] | None = None,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
    ) -> None:
    
    plot_column = 3
    data_dict = {
        "Ground truth": targets,
        "Prediction": prediction,
        "Absolute error": abs_error,
    }

    if level_predictions is not None:
        for l, level_prediction in enumerate(level_predictions):
            data_dict[f"Level {l+1} Prediction"] = level_prediction

    num_plots = len(data_dict)
    used_idxs = [idx for idx, data in enumerate(data_dict.values()) if data is not None]
    plot_row = -(-num_plots // plot_column)  # Equivalent to math.ceil(num_plots / plot_column)

    fig, axes = plt.subplots(
        plot_row, plot_column,
        figsize=(plot_column * 4, plot_row * 4),
        constrained_layout=True,
        gridspec_kw={'width_ratios': [1]*plot_column, 'height_ratios': [1]*plot_row}  # Equal proportions
    )

    # Flatten axes array for easier handling
    axes = (np.array(axes).reshape(-1))
    
    extent = [inputs[:, 0].min(), inputs[:, 0].max(), inputs[:, 1].min(), inputs[:, 1].max()]

    for ax, (label, data) in zip(axes, data_dict.items()):
        if data is not None:
            im = ax.imshow(data.T, extent=extent, origin="lower", aspect='equal', zorder=1)  # Preserve equal aspect
            cbar = fig.colorbar(im, ax=ax, location="bottom", shrink=0.8, pad=0.05)  # Keep colorbar proportional
            cbar.ax.tick_params(labelsize=8)
            ax.set_title(label)

    # Remove unused subplots
    for idx in range(0, num_plots):
        if idx not in used_idxs:
            fig.delaxes(axes[idx])
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    if title is not None:
        fig.suptitle(title, fontsize=20, fontweight='bold')

    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()

def timer(message="Execution time"): 
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"{message}: {elapsed_time:.4f}s")
            return result
        return wrapper
    return decorator

