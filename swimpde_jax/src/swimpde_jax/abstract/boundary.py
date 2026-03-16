from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from swimpde_jax.domain import Domain

from .ansatz import Ansatz


class BoundaryCondition(ABC):
    """Base class for boundary conditions."""

    scaling: float

    def __init__(self, scaling: float = 1):
        self.scaling = scaling

    @abstractmethod
    def _get_lhs(
        self, ansatz: Ansatz, domain: Domain, mask: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Transforms the boundary points according to the boundary condition
        to generate the lhs of the linear system to satisfy the boundary condition.
        """
        pass

    @abstractmethod
    def _get_rhs(
        self, domain: Domain, n_targets: int, mask: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Transforms the boundary points according to the boundary condition
        to generate the rhs of the linear system to satisfy the boundary condition
        """
        pass

    def get_lhs(
        self, ansatz: Ansatz, domain: Domain, mask: npt.ArrayLike | None = None
    ) -> npt.ArrayLike:
        if mask is None:
            mask = np.full(domain.boundary_points.shape[0], True)
        return self._get_lhs(ansatz, domain, mask)

    def get_rhs(
        self, domain: Domain, n_targets: int, mask: npt.ArrayLike | None = None
    ) -> npt.ArrayLike:
        if mask is None:
            mask = np.full(domain.boundary_points.shape[0], True)
        return self._get_rhs(domain, n_targets, mask)

    @abstractmethod
    def get_bias_value(self) -> float:
        """Return a bias value for solving least square problems.

        Depending on the boundary condition, this can be 0 or 1.
        """
        pass
