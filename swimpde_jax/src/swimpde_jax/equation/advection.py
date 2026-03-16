from dataclasses import dataclass
from typing import Callable

import numpy.typing as npt

from swimpde_jax.abstract import (
    Ansatz,
    StaticEquation,
    TimeDependentEquation,
    PrecomputedODEParameters,
)


@dataclass(kw_only=True)
class AdvectionODEParameters(PrecomputedODEParameters):
    gradient: npt.ArrayLike


@dataclass(kw_only=True)
class Advection(StaticEquation, TimeDependentEquation):
    """A representer of the advection equation

        ∂u(x,t)/∂t = -c∂u(x,t)/∂x + f(t,x)

     Attributes
     ----------
    u0: Callable[[npt.ArrayLike], npt.ArrayLike]
        Solution at time t=0.
    beta: float = 1
        Travelling speed of the wave.
    """

    u0: Callable
    beta: float = 1

    def precompute_ode_parameters(
        self, ansatz: Ansatz, points: npt.ArrayLike
    ) -> AdvectionODEParameters:
        evaluation = ansatz.transform(points)
        gradient = ansatz.transform(points, operator="gradient")
        return AdvectionODEParameters(evaluation=evaluation, gradient=gradient)

    def get_initial_solution(self, points: npt.ArrayLike) -> npt.ArrayLike:
        return self.u0(points)

    def get_state_update(
        self, _: float, state: npt.ArrayLike, precomputed: AdvectionODEParameters
    ) -> npt.ArrayLike:
        coeffs = state
        u_x = precomputed.gradient.squeeze() @ coeffs
        update = -self.beta * u_x
        return update

    def equation_bias(self) -> float:
        return 1

    def equation_operator(self, ansatz: Ansatz, points: npt.ArrayLike) -> npt.ArrayLike:
        precomputed = self.precompute_ode_parameters(ansatz, points)
        u_x, u_t = precomputed.gradient[..., 0], precomputed.gradient[..., 1]
        return u_t + self.beta * u_x
