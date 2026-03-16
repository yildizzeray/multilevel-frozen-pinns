from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax import Array
from typing import Any

class WindowKernel(ABC):
    name: str = "kernel"

    @abstractmethod
    def __call__(self, z: Array) -> Array:
        """
        z: (..., D) normalized coordinates
        returns: (...) window value
        """
        pass