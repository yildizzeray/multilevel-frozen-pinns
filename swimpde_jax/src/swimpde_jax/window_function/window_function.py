import jax.numpy as jnp
from jax import Array

from abstract import WindowKernel

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

        assert jnp.all(self.ax_mins <= self.ax_maxs), "Bound assignment is not valid."

    def normalize(self, x: Array) -> Array:
        return (x - self.mu) / (self.sd)

    def f(self, x: Array) -> Array:
        """
        x: (D,) if single-window, or (D,) point with batched windows (M,D) stored in self.mu/sd,
           broadcasting will yield z: (M,D).
        returns: () or (M,)
        """
        z = self.normalize(jnp.asarray(x))
        return self.kernel(z)
