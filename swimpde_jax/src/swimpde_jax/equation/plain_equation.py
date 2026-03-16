from dataclasses import dataclass

import numpy.typing as npt

from swimpde_jax.abstract import Ansatz, StaticEquation


@dataclass
class PlainEquation(StaticEquation):
    """A representer of a function approximater

    u = f(x)
    """

    def equation_operator(self, ansatz: Ansatz, points: npt.ArrayLike) -> npt.ArrayLike:
        u = ansatz.transform(points)
        return u

    def equation_bias(self) -> float:
        return 1