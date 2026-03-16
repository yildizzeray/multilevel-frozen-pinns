from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from swimpde_jax.abstract import Ansatz, TimeDependentEquation, PrecomputedODEParameters


@dataclass(kw_only=True)
class EulerBernoulliState(PrecomputedODEParameters):
    dxxx: npt.ArrayLike


@dataclass(kw_only=True)
class EulerBernoulli(TimeDependentEquation):
    """A representer for the Euler-Bernoulli beam equation

        ∂^2 u(x,t) / ∂t^2 + ∂^(4) u(x,t)/ ∂^(4) x = f(t,x)

    Attributes
    ----------
    u0: Callable[[npt.ArrayLike], npt.ArrayLike]
        Solution at time t=0.
    ut0: Callable[[npt.ArrayLike], npt.ArrayLike]
        Velocity at time t=0.
    """

    u0: Callable[[npt.ArrayLike], npt.ArrayLike]
    ut0: Callable[[npt.ArrayLike], npt.ArrayLike]

    def precompute_ode_parameters(
        self, ansatz: Ansatz, points: npt.ArrayLike
    ) -> EulerBernoulliState:
        evaluation = ansatz.transform(points)
        dxxx = ansatz.transform(points, operator="dxxxx")
        return EulerBernoulliState(evaluation=evaluation, dxxx=dxxx)

    def get_initial_solution(self, points: npt.ArrayLike) -> npt.ArrayLike:
        return np.row_stack([self.u0(points), self.ut0(points)])

    def get_state_update(
        self, _: float, state: npt.ArrayLike, precomputed: EulerBernoulliState
    ) -> npt.ArrayLike:
        n_coeffs = state.shape[0] // self.time_order
        coeffs = state[:n_coeffs]
        update = -precomputed.dxxx @ coeffs
        return update

    @property
    def time_order(self) -> int:
        return 2
