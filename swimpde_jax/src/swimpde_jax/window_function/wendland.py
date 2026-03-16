from swimpde_jax.abstract import WindowKernel

from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp


class WendlandKernel(WindowKernel):
    name = "wendland"

    def __call__(self, z: Array) -> Array:
        """
        z: (..., D) normalized coords, z = (x - mu)/sd
        returns: (...) window values
        """
        r = jnp.abs(z)                         
        inside = jnp.all(r <= 1.0, axis=-1)   

        rc = jnp.clip(r, 0.0, 1.0)
        ws = ((1.0 - rc) ** 4) * (4.0 * rc + 1.0)  

        w = jnp.prod(ws, axis=-1)            
        return jnp.where(inside, w, 0.0)

