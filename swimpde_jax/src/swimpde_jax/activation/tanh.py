import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
from jax.nn import tanh

from swimpde_jax.abstract import Activation


class Tanh(Activation):
    name: str = "tanh"
        
    def f(self, x: ArrayLike) -> Array:
        return tanh(x)
