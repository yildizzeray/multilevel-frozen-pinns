from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

from jax.typing import ArrayLike
from jax import Array, vmap, jit, grad, jacfwd, jacrev
import jax.numpy as jnp


class Activation(ABC):
    """A base class for auto-differentiable activation function.

    Attributes:
    -----------
    name: str
        Name of the activation function.


    Methods:
    --------
    get_f(self, order: int = 0) -> Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike]:
        Returns a function for the derivative of order 'order'.
    __call__(self, x: jnp.NDArray, order: int=0) -> jax.Array:
        Evaluates the activation's derivative of order 'order'.
    _f(self, x: jax.typing.ArrayLike) -> jax.Array:
        An abstract method for evaluating the activation function.
    _dx(self, x: jax.typing.ArrayLike, order: int) -> jax.Array:
        An abstract methods for evaluating activation's derivative.
    """

    name: str = "base"

    def __call__(self, x: ArrayLike, order: int = 0) -> Array:
        """Evaluates the activation's derivative of order 'order'.

        Evalutes the activation function itself when order is set to 0.
        """
        
        f = self.get_f(order)
        return f(x)

    def get_f(self, order: int = 0) -> Callable[[ArrayLike], ArrayLike]:
        if order < 0 or not isinstance(order, int):
            raise ValueError(
                "The order must be a non-negative integer. " f"Got {order} instead."
            )

        if order == 0:
            return self._f

        return partial(self._dx, order=order)
    
    def _f(self, x: ArrayLike) -> Array:
        """Evaluates the activation function."""
        return jit(vmap(self.f))(jnp.asarray(x))

    def _dx(self, x, order=1) -> Array:
        """Evaluates the derivative of order 'order'."""
        f = self.f
        x_arr = jnp.asarray(x).flatten()    
        for _ in range(order):
            f = grad(f)
        return vmap(f)(jnp.asarray(x_arr)).reshape(x.shape)
    
    @abstractmethod
    def f(self, x: ArrayLike) -> Array:
        """Evaluation function"""
        pass
