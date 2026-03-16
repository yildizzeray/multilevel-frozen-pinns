import numpy as np
import numpy.typing as npt

from swimpde_jax.abstract import Ansatz, BoundaryCondition
from swimpde_jax.domain import Domain


class ZeroNeumann(BoundaryCondition):
    def _get_lhs(
        self, ansatz: Ansatz, domain: Domain, mask: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Evaluate the derivative of basis functions evaluated in normal direction at the boundary
        """
        # gradients: shape (n_bc_points, n_outer_basis, d)
        # normal vectors: shape (n_bc_points, d)
        # normal gradients: shape (n_bc_points, n_outer_basis)
        gradients = ansatz.transform(domain.boundary_points[mask], operator="gradient")
        normal_gradients = np.einsum(
            "ij,ikj->ik", domain.normal_vectors[mask], gradients
        )
        return self.scaling * normal_gradients

    def _get_rhs(self, _: Domain, n_targets: int, mask: npt.ArrayLike) -> npt.ArrayLike:
        return np.zeros((mask.sum(), n_targets))

    def get_bias_value(self) -> float:
        return 0
