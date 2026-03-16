from typing import Callable

import numpy as np
import numpy.typing as npt

from swimpde_jax.abstract import Ansatz, BoundaryCondition
from swimpde_jax.domain import Domain


class Dirichlet(BoundaryCondition):
    g: Callable[[npt.ArrayLike], npt.ArrayLike]

    def __init__(self, g: Callable[[npt.ArrayLike], npt.ArrayLike], scaling: float = 1):
        super().__init__(scaling)
        self.g = g

    def _get_lhs(
        self, ansatz: Ansatz, domain: Domain, mask: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Evaluate the basis functions at the boundary
        """
        output = ansatz.transform(domain.boundary_points[mask])
        return self.scaling * output

    def _get_rhs(
        self, domain: Domain, n_targets: int, mask: npt.ArrayLike
    ) -> npt.ArrayLike:
        g_output = self.g(domain.boundary_points[mask])
        if g_output.shape[1] != n_targets:
            if g_output.shape[1] == 1:
                g_output = np.repeat(g_output, n_targets, axis=1)
            else:
                raise ValueError("Function 'g' has a wrong number of outputs.")
        return self.scaling * g_output

    def get_bias_value(self) -> float:
        return 1


class ZeroDirichlet(Dirichlet):
    def __init__(self, scaling: float = 1):
        super().__init__(self.g, scaling)

    def g(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.zeros((x.shape[0], 1))
