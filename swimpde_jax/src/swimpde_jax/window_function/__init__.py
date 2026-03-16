from swimpde_jax.abstract import WindowKernel
from .cosine import CosineKernel
from .gaussian import GaussianKernel
from .wendland import WendlandKernel
from .trapezoidal import TrapezoidalKernel

import numpy.typing as npt
import numpy as np

import jax.numpy as jnp
from jax import Array

class WindowFunction:
    """
    WindowFunction = (geometry: mins/maxs) + (kernel)
    Works for one subdomain or many subdomains at once.
    """

    def __init__(self, ax_mins, ax_maxs, kernel: WindowKernel):
        self.ax_mins = jnp.asarray(ax_mins)
        self.ax_maxs = jnp.asarray(ax_maxs)
        self.mu = 0.5 * (self.ax_mins + self.ax_maxs)
        self.sd = 0.5 * (self.ax_maxs - self.ax_mins)
        self.kernel = kernel
        self.name = f"window[{kernel.name}]"

    def normalize(self, x: Array) -> Array:
        return (x - self.mu) / (self.sd)

    def f(self, x: Array) -> Array:
        """
        x: (D,) if single-window, or (D,) point with batched windows (M,D) stored in self.mu/sd,
           broadcasting will yield z: (M,D).
        returns: () or (M,)
        """
        z = self.normalize(jnp.asarray(x))
        return self.kernel()(z)
    
    
def get_window_function(window_function_name: str, ax_mins: npt.ArrayLike, ax_maxs: npt.ArrayLike):
    """Returns an object for the window function."""
    window_functions = {
        "trapezoidal": TrapezoidalKernel,
        "gaussian": GaussianKernel,
        "wendland": WendlandKernel,
        "cosine": CosineKernel,
        }
    if window_function_name not in window_functions:
        raise ValueError(f"Unknown window function {window_function_name}.")
    return WindowFunction(ax_mins=ax_mins, ax_maxs=ax_maxs, kernel=window_functions[window_function_name])


def calculate_normalization_array(x: npt.ArrayLike, window_functions: list) -> any:
    """
    Calculates normalization constant for partition of unity given evaluation points and list of window functions will be used along the domain
    
    Parameters
    ----------
    x : ndarray of shape (N, d)
            Points where we want to evaluate the window functions.

    window_functions : list of callable 
        List of window functions those will be applied to a decomposed domain
    
    Returns
    -------
    normalization_array : ndarray of shape
        An array needed to be applied each local window function evaluations to ensure partition of unity
    """    
    normalization_array = np.zeros(x.shape[0])[:, None]
    for window_fn in window_functions:
        normalization_array += window_fn(x)

    return normalization_array

__all__ = [
    "get_window_function",
    "calculate_normalization_array_for_window_functions"
    "WindowFunction"
]