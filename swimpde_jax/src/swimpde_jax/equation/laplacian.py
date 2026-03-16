from dataclasses import dataclass

import numpy.typing as npt

from swimpde_jax.abstract import Ansatz, StaticEquation


@dataclass
class Laplacian(StaticEquation):
    """A representer of the Laplacian equation

    -𝚫u(x) = f(x)
    """

    def equation_operator(self, ansatz: Ansatz, points: npt.ArrayLike) -> npt.ArrayLike:
        u_xx = ansatz.transform(points, operator="laplace")
        return - u_xx

    def equation_bias(self) -> float:
        return 0