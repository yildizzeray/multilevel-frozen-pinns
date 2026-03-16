import numpy as np
import numpy.typing as npt

from swimpde_jax.abstract import Ansatz, BoundaryCondition
from swimpde_jax.domain import Domain


class MixedBoundary(BoundaryCondition):
    boundary_conditions: tuple[tuple[npt.ArrayLike, BoundaryCondition]]

    def __init__(self, boundary_conditions):
        self.boundary_conditions = boundary_conditions

    def _get_lhs(
        self, ansatz: Ansatz, domain: Domain, mask: npt.ArrayLike
    ) -> npt.ArrayLike:
        lhs = []
        for boundary_mask, condition in self.boundary_conditions:
            # mask.shape -> (n_boundary_points,)
            # boundary_mask.shape -> (n_boundary_points,)
            combined_mask = mask & boundary_mask
            boundary_lhs = condition._get_lhs(ansatz, domain, combined_mask)
            lhs.append(boundary_lhs)
        return np.row_stack(lhs)

    def _get_rhs(
        self, domain: Domain, n_targets: int, mask: npt.ArrayLike
    ) -> npt.ArrayLike:
        rhs = []
        for boundary_mask, condition in self.boundary_conditions:
            # mask.shape -> (n_boundary_points,)
            # boundary_mask.shape -> (n_boundary_points,)
            combined_mask = mask & boundary_mask
            boundary_rhs = condition._get_rhs(domain, n_targets, combined_mask)
            rhs.append(boundary_rhs)

        return np.row_stack(rhs)

    def get_bias_value(self) -> float:
        """
        Return the value multiplied with the constant coefficient of the constructed (outer) basis functions.
        1 for Dirichlet to preserve the constant, and 0 for Neumann because it vanishes in the derivative.
        """
        return 1
