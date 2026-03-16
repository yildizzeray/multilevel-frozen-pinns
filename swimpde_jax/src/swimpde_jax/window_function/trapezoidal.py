from swimpde_jax.abstract import WindowKernel

from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp


class TrapezoidalKernel(WindowKernel):
    name = "trapezoidal"

    def __call__(self, z: Array) -> Array:
        """
        z: (..., D) normalized coords, z = (x - mu)/sd
        returns: (...) window values
        """
        w1d = jnp.clip(1.0 - jnp.abs(z), 0.0, 1.0) 
        return jnp.prod(w1d, axis=-1)
            