from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
from swimnetworks import Linear

from swimpde_jax.abstract import Ansatz, BoundaryCondition, StaticEquation
from swimpde_jax.boundary import get_boundary_condition
from swimpde_jax.domain import Domain


@dataclass(kw_only=True)
class StaticSolver:
    """A solver for a general static equation
        L(u)(x) = f(x) on 𝛀
        B(u)(x) = g(x) on ∂𝛀

    Attributes
    ----------
    equation: StaticEquation
        An equation to solve.
    domain: Domain
        A domain on which the equation should be solved.
    forcing: Callable[[npt.ArrayLike, float], npt.ArrayLike]
        The forcing term as a function of x.
    ansatz: Ansatz
        An ansatz to use for computing the basic functions.
    regularization_scale: float
        A regularization scale for solving least square problems.
    boundary_condition: str | BoundaryCondition
        A boundary condition for the equation.

    Methods
    -------
    fit(approx_sol: npt.ArrayLike)
        Solves the PDE by fitting the ansatz to the forcing and boundary.
    evaluate(self, x: npt.ArrayLike) -> npt.ArrayLike:
        Evaluates the solution at the given points.
    """

    equation: StaticEquation
    domain: Domain
    f: Callable[[npt.ArrayLike], npt.ArrayLike]
    ansatz: Ansatz
    regularization_scale: float
    boundary_condition: str | BoundaryCondition

    _linear: Linear = None
    _status: bool = None

    def __post_init__(self):
        if isinstance(self.boundary_condition, str):
            self.boundary_condition = get_boundary_condition(self.boundary_condition)

        self.ansatz = deepcopy(self.ansatz)
        self._linear = Linear(regularization_scale=self.regularization_scale)

    def _equation_operator(self, interior_points: npt.ArrayLike) -> npt.ArrayLike:
        return self.equation.equation_operator(self.ansatz, interior_points)

    def fit(self, approx_sol: npt.ArrayLike | None = None):
        """Solves the PDE by fitting the ansatz to the forcing and boundary.

        Parameters
        ----------
        approx_sol: npt.ArrayLike | None = None
            If provided, approx_sol is used to sample the ansatz.
        """
        self._status = True

        # Fit the ansatz accordingly to the equation.
        if approx_sol is None:
            target = np.zeros_like(self.f(self.domain.interior_points))
        else:
            target = approx_sol
        self.ansatz.fit(self.domain, target)

        # Define the input matrix for the linear problem.
        equation_output = self._equation_operator(self.domain.interior_points)
        boundary_output = self.boundary_condition.get_lhs(self.ansatz, self.domain)
        X = np.row_stack([equation_output, boundary_output])
        n_interior, n_boundary = len(equation_output), len(boundary_output)
        bias = np.row_stack(
            [
                np.ones((n_interior, 1)) * self.equation.equation_bias(),
                np.ones((n_boundary, 1)) * self.boundary_condition.get_bias_value(),
            ]
        )

        # Define the output matrix for the linear problem.
        equation_target = self.f(self.domain.interior_points)
        n_targets = equation_target.shape[1]
        boundary_target = self.boundary_condition.get_rhs(self.domain, n_targets)
        y = np.row_stack([equation_target, boundary_target])

        # Solve the linear problem.
        # TODO: use self._linear.fit() once swim can disable a bias.
        linear_weights = np.linalg.lstsq(
            np.column_stack([X, bias]), y, rcond=self.regularization_scale
        )[0]
        self._linear.weights = linear_weights[:-1]
        self._linear.biases = linear_weights[-1]
        self._linear.layer_width = linear_weights.shape[1]

        return self

    def evaluate(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Evaluates the solution at the given points.

        Parameters
        ----------
        x: npt.ArrayLike
            Points to evaluate the solution at.
        """
        ansatz_output = self.ansatz.transform(x)
        return self._linear.transform(ansatz_output)

    @property
    def status(self):
        return self._status
