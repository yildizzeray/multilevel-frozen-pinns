from dataclasses import dataclass
from typing import Callable

import numpy.typing as npt

from swimpde_jax.abstract import Ansatz, TimeDependentEquation, PrecomputedODEParameters


@dataclass(kw_only=True)
class BurgersODEParameters(PrecomputedODEParameters):
    gradient: npt.ArrayLike
    laplace: npt.ArrayLike


@dataclass(kw_only=True)
class Burgers(TimeDependentEquation):
    """A representer for the Burgers' equation

        ∂u(x,t)/∂t + u(x,t) * ∂u(x,t)/∂x =  nu * ∂^2u(x,t)/∂^2x + f(x, t)

    Attributes:
    -----------
    u0: Callable[[npt.ArrayLike], npt.ArrayLike]
        Solution at time t=0.
    nu: float = 1
        Viscosity paramer.
    """

    u0: Callable[[npt.ArrayLike], npt.ArrayLike]
    nu: float = 1

    def precompute_ode_parameters(
        self, ansatz: Ansatz, points: npt.ArrayLike
    ) -> BurgersODEParameters:
        evaluation = ansatz.transform(points)
        gradient = ansatz.transform(points, operator="gradient")
        laplace = ansatz.transform(points, operator="laplace")
        return BurgersODEParameters(
            evaluation=evaluation, gradient=gradient, laplace=laplace
        )

    def get_initial_solution(self, points: npt.ArrayLike) -> npt.ArrayLike:
        return self.u0(points)

    def get_state_update(
        self, _: float, state: npt.ArrayLike, precomputed: BurgersODEParameters
    ) -> npt.ArrayLike:
        coeffs = state
        u = precomputed.evaluation @ coeffs
        u_x = precomputed.gradient.squeeze() @ coeffs
        u_xx = precomputed.laplace @ coeffs
        update = self.nu * u_xx - u * u_x
        return update
