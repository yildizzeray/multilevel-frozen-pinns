from swimpde_jax.abstract import WindowKernel

from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp


class CosineKernel(WindowKernel):
    name = "cosine"

    def __call__(self, z: Array) -> Array:
        """
        z: (..., D) normalized coords, z = (x - mu)/sd
        returns: (...) window values
        """
        inside = jnp.all(jnp.abs(z) <= 1.0, axis=-1)

        ws = ((1.0 + jnp.cos(jnp.pi * jnp.clip(z, -1.0, 1.0))) / 2.0) ** 2
        w = jnp.prod(ws, axis=-1)

        return jnp.where(inside, w, 0.0)
