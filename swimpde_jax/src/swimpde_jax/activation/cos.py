import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array, jit

from swimpde_jax.abstract import Activation


class Cos(Activation):
    name: str = "cos"
    
    def f(self, x: ArrayLike) -> Array:
        return jnp.cos(x)