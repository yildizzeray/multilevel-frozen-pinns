from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp

from swimnetworks import Dense

from swimpde_jax.abstract import Activation, Ansatz
from swimpde_jax.activation import get_activation, get_parameter_sampler
from swimpde_jax.domain import Domain

from .utils import get_order


@dataclass(kw_only=True)
class BasicAnsatz(Ansatz):
    """Ansatz representing a simple neural network.

    Attributes:
    -----------
    n_basis: int
        Number of outer basis functions.
    n_inner_basis: int
        Number of inner basis functions.
    activation: Union[str, Activation]
        Activation function to use for the inner basis.
    parameter_sampler: str
        Parameter sampler to use in the SWIM algorithm.
        See the SWIM package for possible options)
    random_seed: int
        Random seed to be used in all processes that require randomness.

    """

    n_basis: int
    activation: str | Activation
    parameter_sampler: str | Callable | None = None
    random_seed: int = 1

    _layer: Dense | None = None

    def __post_init__(self):
        if isinstance(self.activation, str):
            self.activation = get_activation(self.activation)
        elif not isinstance(self.activation, Activation):
            raise TypeError(
                "Activation function must be of type str or "
                f"Activation: got {type(self.activation)}"
            )
        if self.parameter_sampler is None:
            self.parameter_sampler = get_parameter_sampler(self.activation.name)

        self._layer = Dense(
            layer_width=self.n_basis,
            activation=self.activation.get_f(),
            parameter_sampler=self.parameter_sampler,
            random_seed=self.random_seed,
            prune_duplicates=False,
        )

        self.n_outputs = self.n_basis

    def _transform(
        self,
        x: npt.ArrayLike,
        operator: str | None = None,
        coordinate_scaling: npt.ArrayLike | None = None,
    ) -> npt.ArrayLike:
        
        x = jnp.array(x)
        _, d = x.shape
        
        if coordinate_scaling is None:
            coordinate_scaling = jnp.ones(d)
        else:
            coordinate_scaling = jnp.array(coordinate_scaling)
            
        def layer_fn(x_):
            x_scaled = x_ * coordinate_scaling
            return self._layer.transform(x_scaled).squeeze()
        
        # Base activation (autodiff handles differentiation)
        self._layer.activation = self.activation.get_f(order=0)
                
        match operator:
            case None:
                out = jax.vmap(jax.jit(layer_fn))(x)  # (n_points, n_neurons)

            case "gradient":
                # Gradient wrt inputs -> (n_neurons, d)
                grad_fn = jax.jacfwd(layer_fn)
                out = jax.vmap(jax.jit(grad_fn))(x)  # (n_points, n_neurons, d)

            case "laplace":
                # Laplacian = trace of Hessian -> (n_neurons,)
                def laplace_fn(x_):
                    hess = jax.jacfwd(jax.jacrev(layer_fn))(x_)  # (n_neurons, d, d)
                    return jnp.trace(hess, axis1=-2, axis2=-1)

                out = jax.vmap(jax.jit(laplace_fn))(x)  # (n_points, n_neurons)

            case 'dxxxx':
                def bilaplace_fn(x_):
                    lap = jnp.trace(jax.jacfwd(jax.jacrev(layer_fn))(x_), axis1=-2, axis2=-1)
                    hess2 = jax.jacfwd(jax.jacrev(lambda x__: lap))(x_)
                    return jnp.trace(hess2, axis1=-2, axis2=-1)

                out = jax.vmap(jax.jit(bilaplace_fn))(x)  # (n_points, n_neurons)
            case _:
                raise ValueError(f"Unsupported operator: {operator}")

        return jax.device_get(out)
        

    def _fit(
        self,
        domain: Domain,
        target_fn: Callable[[npt.ArrayLike], npt.ArrayLike]
        | npt.ArrayLike
        | None = None,
        operator: str | None = None,
    ):
        """
        Fit the model to the data.

        Parameters:
        x: input values of shape (n_points, d)
        y: target values of shape (n_points,)
        order: order of the activation's derivative to apply before fitting.
        """
        if operator not in [None, "laplace"]:
            raise ValueError(
                "BasicAnsatz.fit() only supports opertors "
                f"in [None, 'laplace']. Got: {operator}."
            )

        if target_fn is None:
            target = np.zeros((domain.interior_points.shape[0], self.n_basis))
        elif callable(target_fn):
            target = target_fn(domain.interior_points)
        else:
            target = target_fn

        if target.shape[0] != domain.interior_points.shape[0]:
            raise ValueError(
                f"Target has shape {target.shape}, but "
                f"points have shape {domain.interior_points.shape}."
            )

        order = get_order(operator)
        self._layer.activation = self.activation.get_f(order=order)
        self._layer.fit(domain.interior_points, target)
