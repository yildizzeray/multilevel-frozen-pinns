import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array

from swimpde_jax.abstract import Activation


class Sin(Activation):
    name: str = "sin"
        
    def f(self, x: ArrayLike) -> Array:
        return jnp.sin(x)