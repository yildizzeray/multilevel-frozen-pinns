from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from swimpde_jax.domain import Domain


@dataclass(kw_only=True)
class Ansatz(ABC):
    """Base class for ansatz functions."""

    include_constant: bool = False
    n_outputs: int = None
    svd_cutoff: float | None = None
    _projection: npt.ArrayLike | None = None
    constraining_operator: Callable | None = None

    @abstractmethod
    def _transform(
        self,
        x: npt.ArrayLike,
        operator: str | None = None,
        coordinate_scaling: npt.ArrayLike = None,
    ) -> npt.ArrayLike:
        """Evaluate the model."""
        pass

    def transform(
        self,
        x: npt.ArrayLike,
        operator: str | None = None,
        coordinate_scaling: npt.ArrayLike | None = None,
    ) -> npt.ArrayLike:
        if coordinate_scaling is None:
            coordinate_scaling = np.ones(x.shape[1])
        output = self._transform(x, operator, coordinate_scaling)

        if self.svd_cutoff is None:
            return output

        if self._projection is None:
            raise RuntimeError(
                "The projection is not set. Call fit() "
                "to fit the model and the projection."
            )

        match operator:
            case None | "laplace" | "dxxxx":
                output = output @ self._projection
            case "gradient":
                output = np.swapaxes(output, 1, 2)
                output = output @ self._projection
                output = np.swapaxes(output, 1, 2)
            case _:
                raise ValueError(
                    "Ansatz output can not be projected for " f"{operator=}."
                )

        if self.include_constant:
            match operator:
                case None:
                    constant = np.ones_like(output[:, 0])
                case "laplace" | "gradient" | "dxxxx":
                    # These operator turn constant functions to 0.
                    constant = np.zeros_like(output[:, 0])
                case _:
                    raise ValueError(
                        "Constant function cannot be added " f"for {operator=}."
                    )
            output = np.append(output, constant, axis=1)

        return output

    @abstractmethod
    def _fit(
        self,
        domain: Domain,
        target_fn: Callable[[npt.ArrayLike], npt.ArrayLike]
        | npt.ArrayLike
        | None = None,
        operator: str | None = None,
    ):
        """Fit the model to the data."""
        pass

    def fit(
        self,
        domain: Domain,
        target_fn: Callable[[npt.ArrayLike], npt.ArrayLike]
        | npt.ArrayLike
        | None = None,
        operator: str | None = None,
    ):
        """Fit the model to the data."""
        self._fit(domain, target_fn, operator)
        if self.svd_cutoff is not None:
            self._set_projection(domain)

    def _set_projection(self, domain: Domain):
        if self.constraining_operator is None:
            points = domain.all_points
        else:
            points = domain.interior_points
        coordinate_scaling = np.ones(points.shape[1])
        output = self._transform(
            points, coordinate_scaling=coordinate_scaling
        )
        _, sigma, vh = np.linalg.svd(output, full_matrices=False)
        mask = sigma / np.max(sigma) > self.svd_cutoff
        self._projection = vh.T[:, mask]
        self.n_outputs = self._projection.shape[1]
