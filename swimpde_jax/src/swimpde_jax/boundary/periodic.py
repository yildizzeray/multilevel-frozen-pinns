import numpy as np
import numpy.typing as npt

from swimpde_jax.abstract import Ansatz, BoundaryCondition
from swimpde_jax.domain import Domain


class Periodic(BoundaryCondition):
    def _get_lhs(
        self, ansatz: Ansatz, domain: Domain, mask: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Evaluate the basis functions at the boundary and compute the difference of values at the boundary
        """
        self._validate_domain(domain)
        boundary_points = domain.boundary_points[mask]
        left_points, right_points = self._match_boundary_points(boundary_points)

        left_boundary = ansatz.transform(left_points)
        right_boundary = ansatz.transform(right_points)
        return self.scaling * (left_boundary - right_boundary)

    def _get_rhs(
        self, domain: Domain, n_targets: int, mask: npt.ArrayLike
    ) -> npt.ArrayLike:
        self._validate_domain(domain)
        boundary_points = domain.boundary_points[mask]
        left_points, _ = self._match_boundary_points(boundary_points)
        n_pairs = left_points.shape[0]
        return np.zeros((n_pairs, n_targets))

    def get_bias_value(self) -> float:
        return 0

    def _validate_domain(self, domain: Domain):
        if domain.n_dim > 2:
            raise ValueError(
                "Periodic boundary condition is defined only "
                f"for 1D or 2D domain with periodicity along "
                f"the x-axis. Got: {domain.n_dim} dimensions."
            )

    def _match_boundary_points(
        self, boundary_points: npt.ArrayLike
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        # We assume that the periodicity is along x-axis.
        x_min = np.min(boundary_points[:, 0])
        x_max = np.max(boundary_points[:, 0])
        left_points = boundary_points[boundary_points[:, 0] == x_min]
        right_points = boundary_points[boundary_points[:, 0] == x_max]

        if boundary_points.shape[1] == 1:
            # In case of 1D data, we don't need to 'match'
            # boundary points.
            return left_points, right_points

        # Sort points by the y-coordinate to match the boundary points.
        # This step is redundunt for 1D data.
        left_idx = np.argsort(left_points[:, -1])
        right_idx = np.argsort(right_points[:, -1])
        left_points = left_points[left_idx]
        right_points = right_points[right_idx]
        if not np.all(left_points[:, -1] == right_points[:, -1]):
            raise ValueError("The domain is not regular.")

        return left_points, right_points


class PeriodicStrict(Periodic):
    def _get_lhs(
        self, ansatz: Ansatz, domain: Domain, mask: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Evaluate the basis functions at the boundary and compute the difference of values at the boundary
        """
        self._validate_domain(domain)
        boundary_points = domain.boundary_points[mask]
        left_points, right_points = self._match_boundary_points(boundary_points)

        left_boundary = ansatz.transform(left_points)
        right_boundary = ansatz.transform(right_points)

        # Compute gradient in space
        left_gradient = ansatz.transform(left_points, operator="gradient")[..., 0]
        right_gradient = ansatz.transform(right_points, operator="gradient")[..., 0]

        lhs = np.concatenate(
            [left_boundary - right_boundary, left_gradient - right_gradient]
        )
        return self.scaling * lhs

    def _get_rhs(
        self, domain: Domain, n_targets: int, mask: npt.ArrayLike
    ) -> npt.ArrayLike:
        self._validate_domain(domain)
        boundary_points = domain.boundary_points[mask]
        left_points, _ = self._match_boundary_points(boundary_points)
        n_pairs = left_points.shape[0]
        return np.zeros((2 * n_pairs, n_targets))
