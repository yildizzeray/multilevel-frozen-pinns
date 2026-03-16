from abc import abstractmethod
from dataclasses import dataclass

import numpy.typing as npt

from .ansatz import Ansatz


@dataclass(kw_only=True)
class PrecomputedODEParameters:
    """A base dataclass specyfing precomputed parameters for the ODE.

    The idea of this class is to have a structured way to hold
    parameters for different equations. The attributes defined in this
    base class can be filled out and used by TimeSolver.

    Attributes
    ----------
    evaluation: npt.ArrayLike
        Output of the ansatz on all domain points.
    boundary_evaluation: npt.ArrayLike
        Output of the ansatz on boundary domain points.
    evaluation_inv; npt.ArrayLike
        Holds an inverse of 'evaluation' if no boundary
        condition is forced. Otherwise, hold an inverse
        of [evaluation, boundary_evaluation].
    """

    evaluation: npt.ArrayLike
    boundary_evaluation: npt.ArrayLike = None
    evaluation_inv: npt.ArrayLike = None


class TimeDependentEquation:
    """An abstract class to represent a time-dependent PDE."""

    @abstractmethod
    def precompute_ode_parameters(
        self, ansatz: Ansatz, points: npt.ArrayLike
    ) -> PrecomputedODEParameters:
        """Precomputes all required parameters for solving the ODE."""
        pass

    @abstractmethod
    def get_initial_solution(self, points: npt.ArrayLike) -> npt.ArrayLike:
        """Returns an initial condition for the equation."""
        pass

    @abstractmethod
    def get_state_update(
        self, t: float, state: npt.ArrayLike, precomputed: PrecomputedODEParameters
    ) -> npt.ArrayLike:
        """Returns a state derivative as RHS of the ODE.

        Parameters
        ----------
        t: float
            Time point.
        state: npt.ArrayLike
            State of the equation. If the equation contains time order higher than 1,
            state is assumed to be a concatenation of vectors [coeff, coeff_t].
        precomputed: PrecomputedODEParameters
            Parameters of the ODE precomputed by the equation.

        """
        pass

    @property
    def time_order(self) -> int:
        """Returns the highest order of derivative in time of the equation.

        This field is used by TimeSolver to split a state into coeff, coeff_t,
        and so on.
        """
        return 1
