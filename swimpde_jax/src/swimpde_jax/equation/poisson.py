from dataclasses import dataclass

import numpy.typing as npt

from swimpde_jax.abstract import Ansatz, StaticEquation


@dataclass
class Poisson(StaticEquation):
    """A representer of the Poisson equation

        𝚫u(x) = f(x)

    Attributes
    ----------
    parameter_scaling: npt.ArrayLike | None = None
        If given, scales the impact of the space coordinates.
    """

    parameter_scaling: npt.ArrayLike = None  # constant 1 if kept at None

    def equation_operator(self, ansatz: Ansatz, points: npt.ArrayLike) -> npt.ArrayLike:
        return ansatz.transform(
            points, operator="laplace", coordinate_scaling=self.parameter_scaling
        )

    def equation_bias(self) -> float:
        return 0
