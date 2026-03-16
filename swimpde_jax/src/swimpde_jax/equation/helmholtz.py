from dataclasses import dataclass

import numpy.typing as npt

from swimpde_jax.abstract import Ansatz, StaticEquation


@dataclass
class Helmholtz(StaticEquation):
    """A representer of the Helmholtz equation

    𝚫u(x) - k^2.u = f(x)
    """
    k: npt.ArrayLike = 1

    def equation_operator(self, ansatz: Ansatz, points: npt.ArrayLike) -> npt.ArrayLike:
        u = ansatz.transform(points)
        u_xx = ansatz.transform(points, operator="laplace")
        return u_xx + self.k**2 * u

    def equation_bias(self) -> float:
        return 1
