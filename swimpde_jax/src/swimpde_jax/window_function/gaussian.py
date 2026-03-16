from swimpde_jax.abstract import WindowKernel

from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp

class GaussianKernel(WindowKernel):
    name = "gaussian"

    def __call__(self, z: Array) -> Array:
        """
        z: (..., D) normalized coords, z = (x - mu)/sd
        returns: (...) window values
        """
        return jnp.exp(-0.5 * jnp.sum((z / 0.25) ** 2, axis=-1))
