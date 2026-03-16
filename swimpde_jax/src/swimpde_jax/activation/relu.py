import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
from jax.nn import relu

from swimpde_jax.abstract import Activation


class ReLU(Activation):
    name: str = "relu"
        
    def f(self, x: ArrayLike) -> Array:
        return relu(x)
