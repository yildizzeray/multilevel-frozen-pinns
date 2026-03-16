from abc import abstractmethod

import numpy.typing as npt

from .ansatz import Ansatz


class StaticEquation:
    """An abstract class to represent a time-dependent PDE."""

    @abstractmethod
    def equation_operator(self, ansatz: Ansatz, points: npt.ArrayLike) -> npt.ArrayLike:
        """Evaluate the equation operator on the provided points.

        Parameters
        ----------
        ansatz: Ansatz
            An ansatz that computes basis functions.
        points: npt.ArrayLike
            Points to evalute the operator on.
        """
        pass

    @abstractmethod
    def equation_bias(self) -> float:
        """Return a bias value for solving least square problems.

        Depending on the equation operator, this can be 0 or 1.
        """
        pass
