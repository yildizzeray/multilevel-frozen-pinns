from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import numpy.typing as npt
from swimnetworks import Linear

from swimpde_jax.abstract import Activation, Ansatz, BoundaryCondition
from swimpde_jax.boundary import get_boundary_condition
from swimpde_jax.domain import Domain

from .basic_ansatz import BasicAnsatz
from .utils import get_dense_layer_target


@dataclass(kw_only=True)
class BoundaryCompliantAnsatz(Ansatz):
    """Ansatz composed of basis functions that all comply to a boundary condition.

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
    boundary_condition: BoundaryCondition
        A boundary condition to satisfy by the outer functions.
    target_fn: str | Callable
        Target values on the domain interior for constructing the outer basis.
    domain_margin_percent: float
        If nonzero, use only interior points with a minimum distance from
        the boundary are used to determinine the outer coefficients. See
        BoundaryCondition for more details.
    regularization_scale: float
        Regularization scale used in least squares problems.
    """

    # Parameters of the inner ansatz
    n_basis: int
    n_inner_basis: int | None = None
    activation: str | Activation
    parameter_sampler: str | Callable | None = None
    random_seed: int = 1

    boundary_condition: BoundaryCondition
    target_fn: str | Callable[[npt.ArrayLike], npt.ArrayLike] = "tanh"
    domain_margin_percent: float = 0
    regularization_scale: float = 1e-13
    _inner_ansatz: BasicAnsatz = None
    _linear: Linear = None

    def __post_init__(self):
        if self.n_inner_basis is None:
            self.n_inner_basis = self.n_basis

        rng = np.random.default_rng(self.random_seed)
        ansatz_seed, target_seed = rng.integers(np.iinfo(np.int64).max, size=2)

        self._inner_ansatz = BasicAnsatz(
            activation=self.activation,
            n_basis=self.n_inner_basis,
            parameter_sampler=self.parameter_sampler,
            random_seed=ansatz_seed,
        )
        self._linear = Linear(regularization_scale=self.regularization_scale)

        # Set the target function for the inner ansatz.
        if isinstance(self.target_fn, str):
            self.target_fn = partial(
                get_dense_layer_target,
                n_targets=self.n_basis,
                activation=self.target_fn,
                random_seed=target_seed,
            )
        elif not callable(self.target_fn):
            assert ValueError(
                "'inner_target_fn' must be a callable or "
                f"a valid string. Got: {self.target_fn}."
            )

        # Set the boundary condition.
        if isinstance(self.boundary_condition, str):
            self.boundary_condition = get_boundary_condition(self.boundary_condition)

    def _transform(
        self,
        x,
        operator: str | None = None,
        coordinate_scaling: npt.ArrayLike | None = None,
    ) -> npt.ArrayLike:
        """
        Evaluate the model.

        input shape: (n_points, d)
        output shape: (n_points, n_neurons)
            n_neurons is either n_outer_basis if no constant basis was added (dirichlet) or n_outer_basis + 1 including the constant
        operator: operator to evaluate
        """
        ansatz_output = self._inner_ansatz.transform(
            x, operator=operator, coordinate_scaling=coordinate_scaling
        )  # shape (n_points, n_basis, d)
        linear_weights = self._linear.weights  # shape (n_basis, n_basis)

        match operator:
            case None:
                return self._linear.transform(ansatz_output)
            case "gradient":
                gradient = ansatz_output.swapaxes(1, 2) @ linear_weights
                return gradient.swapaxes(1, 2)
            case "laplace" | "dxxxx":
                return ansatz_output @ linear_weights
            case _:
                raise ValueError(f"Cannot evaluate the ansatz for {operator=}.")

    def _fit(
        self,
        domain: Domain,
        target_fn: Callable[[npt.ArrayLike], npt.ArrayLike] | None = None,
        operator: str | None = None,
    ):
        if operator is not None:
            raise ValueError(
                "BoundaryCompliantAnsatz.fit() only supports "
                f"operator=None. Got: {operator}."
            )

        if target_fn is None:
            target_fn = self.target_fn
            
        if not callable(target_fn):
            raise ValueError("BoundaryCompliantAnsatz requires a callable target.")


        # Calculate core interior points of the domain.
        core_mask = domain.get_core_interior_mask(self.domain_margin_percent)
        core_interior = domain.interior_points[core_mask]

        # Fit the inner ansatz to a function defined by 'target_fn'
        target = target_fn(domain.interior_points)
        self._inner_ansatz.fit(domain, target)

        # Evaluate the inner ansatz on core interior points
        # and the boundary condition on boundary points.
        core_inner_output = self._inner_ansatz.transform(core_interior)
        boundary_lhs = self.boundary_condition.get_lhs(self._inner_ansatz, domain)

        # Set up matrices for solving the output linear problem.
        X = np.row_stack([core_inner_output, boundary_lhs])
        bias = np.ones(X.shape[0])
        ### Some boundary conditions (e.g., Neumann) are not affected by bias.
        bias[-boundary_lhs.shape[0] :] *= self.boundary_condition.get_bias_value()
        X_bias = np.column_stack([X, bias])
        ### Create a target for all boundary points.
        boundary_target = self.boundary_condition.get_rhs(domain, self.n_basis)
        ### We want to map core points to something non-zero to avoid zero weights.
        ### If target has self.n_basis functions, we should use this value
        ### for fitting the linear problem. Otherwise, we use the output of
        ### the basic ansatz and try to match it.
        if target.shape[1] == self.n_basis:
            core_output_target = target[core_mask]
        elif target.shape[1] == 1 and self.n_basis == self.n_inner_basis:
            core_output_target = core_inner_output
        else:
            raise ValueError(
                f"Cannot use the target for {self.n_basis=}: " f"{target.shape[1]=}."
            )

        y = np.row_stack([core_output_target, boundary_target])

        # Find weights of linear layer that produces functions
        # compliant to boundary conditions.
        ### TODO: use self._linear.fit(X, y, bias=False) once swim can disable a bias.
        linear_weights = np.linalg.lstsq(X_bias, y, rcond=self.regularization_scale)[0]
        self._linear.weights = linear_weights[:-1, :]
        self._linear.biases = linear_weights[-1:, :]
        self._linear.layer_width = self.n_basis
