import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from swimpde_jax.abstract import (
    Ansatz,
    BoundaryCondition,
    PrecomputedODEParameters,
    TimeDependentEquation,
)
from swimpde_jax.boundary import ZeroDirichlet, get_boundary_condition
from swimpde_jax.domain import Domain, ResamplingDomain


@dataclass(kw_only=True)
class TimeDependentSolver:
    """A solver for a general time-dependent equation
        ∂ᵏu(x, t)/∂tᵏ = L(u)(x, t) + N(u)(x, t) + f(x, t) on 𝛀
        B(u)(x, 0) = g(x) on ∂𝛀

    Attributes
    ----------
    equation: TimeDependentEquation
        An equation to solve.
    domain: Domain
        A domain on which the equation should be solved.
    forcing: Callable[[npt.ArrayLike, float], npt.ArrayLike]
        The forcing term as a function of x and t.
    ansatz: Ansatz
        An ansatz to use for computing the basic functions.
    regularization_scale: float
        A regularization scale for solving least square problems.
    forced_boundary_condition: str | BoundaryCondition | None = None
        If given, the boundary condition is forced directly by modifying
        the ODE for the coefficients. Currently, only supports zero
        dirichlet condition.
    ode_solver: str
        A solver for the ODE problem. See `scipy.solve_ivp' for possible
        options.
    rtol: float
        A relative tolerance to use in the ODE solver.
    atol: float
        An absolute tolerance to use in the ODE solver.
    diverged_threshold: float
        A threshold to use for detect divergence of the ODE solver.
    resample_domain: bool
        If True, resample domain accroding to the obtained solution in the
        beginning of each block. See fit(...) for more details.

    Methods
    -------
    fit(t_span: npt.ArrayLike, n_time_blocks: int)
        Solves the PDE by fitting the ansatz and solving an ODE for coefficients.
    evaluate(x: npt.ArrayLike, t: npt.ArrayLike, operator: str | None) -> npt.ArrayLike
        Evaluates the solution and given space and time points.
    """

    equation: TimeDependentEquation
    domain: Domain
    forcing: Callable[[npt.ArrayLike, float], npt.ArrayLike]
    ansatz: Ansatz
    regularization_scale: float = 1e-8
    forced_boundary_condition: str | BoundaryCondition | None = None
    ode_solver: str = "RK45"
    rtol: float = 1e-8
    atol: float = 1e-8
    diverged_threshold: float = 1e5
    resample_domain: bool = False

    _block_starts: npt.ArrayLike = None
    _block_solutions: list[Callable] = None
    _block_ansatzes: list[Ansatz] = None
    _status: bool = None

    def __post_init__(self):
        if self.resample_domain and not isinstance(self.domain, ResamplingDomain):
            raise ValueError(
                "TimeDependentSolver requires a ResamplingDomain when "
                f"resample_domain=True. Got {type(self.Domain)}."
            )

        if self.forced_boundary_condition is not None:
            if isinstance(self.forced_boundary_condition, str):
                self.forced_boundary_condition = get_boundary_condition(
                    self.forced_boundary_condition
                )

            if not isinstance(self.forced_boundary_condition, ZeroDirichlet):
                raise ValueError(
                    "Direct forcing of the boundary condition is "
                    "only implemented for ZeroDirichlet. "
                    f"Got: {type(self.forced_boundary_condition)}."
                )

        self.domain = deepcopy(self.domain)

    def _get_precomputed_parameters(self, ansatz: Ansatz) -> PrecomputedODEParameters:
        precomputed = self.equation.precompute_ode_parameters(
            ansatz, self.domain.all_points
        )

        if self.forced_boundary_condition is not None:
            # Force the condition directly.
            boundary_lhs = self.forced_boundary_condition.get_lhs(ansatz, self.domain)
            precomputed.boundary_evaluation = boundary_lhs
            evaluation_inv = np.linalg.pinv(
                np.row_stack([precomputed.evaluation, precomputed.boundary_evaluation]),
                self.regularization_scale,
            )
            precomputed.evaluation_inv = evaluation_inv
        else:
            evaluation_inv = np.linalg.pinv(
                precomputed.evaluation, self.regularization_scale
            )
            precomputed.evaluation_inv = evaluation_inv
        return precomputed

    def _get_initial_state(
        self,
        precomputed: PrecomputedODEParameters,
        initial_solution_fn: Callable[[npt.ArrayLike], npt.ArrayLike] | None = None,
    ) -> npt.ArrayLike:
        if self.forced_boundary_condition is not None:
            points = np.concatenate(
                [self.domain.all_points, self.domain.boundary_points]
            )
        else:
            points = self.domain.all_points

        if initial_solution_fn is None:
            initial_solution = self.equation.get_initial_solution(points)
        else:
            initial_solution = initial_solution_fn(points)

        # If the time order of the equation is large then one,
        # the state has a form [c, c_t, ...]. To compute the initial condition
        # for c, c_t, ..., we need to extract initial conditions for u, u_t, ...
        n_points = points.shape[0]
        initial_state = []
        for order in range(self.equation.time_order):
            order_slice = slice(n_points * order, n_points * (order + 1))
            initial_order_state = (
                precomputed.evaluation_inv @ initial_solution[order_slice]
            )
            initial_state.append(initial_order_state.ravel())

        initial_state = np.concatenate(initial_state)
        return initial_state

    def _state_derivative(
        self, t: float, state: npt.ArrayLike, precomputed: PrecomputedODEParameters
    ) -> npt.ArrayLike:
        update = self.equation.get_state_update(t, state, precomputed)
        update += self.forcing(self.domain.all_points, t).ravel()

        if self.forced_boundary_condition is not None:
            boundary_correction = -state @ precomputed.boundary_evaluation.T
            update = np.concatenate([update, boundary_correction], axis=0)

        # If time_order of the equation is larger than 1, we need to only compute
        # the highest order derivative:
        # state = (coeffs, coeffs_t, coeffs, tt)
        # state_t = (coeffs_t, coefs_tt, coeffs_ttt)
        highest_order = precomputed.evaluation_inv @ update
        n_coeffs = state.shape[0] // self.equation.time_order
        state_t = np.concatenate([state[n_coeffs:], highest_order])
        return state_t

    def fit(self, t_span: npt.ArrayLike, n_time_blocks: int = 1):
        """Solves the PDE by fitting the ansatz and solving an ODE for coefficients.

        Parameters
        ----------
        t_span: npt.ArrayLike
            The time value to which a solution should be integrated.
        n_time_blocks: int = 1
            Number of blocks to use when splitting the integration. Starting from the
            second block, the ansatz is fitted to approximate the solution generated by
            the previous block.
        """
        self._status = True

        # We add a ghost timestamp to mark the end of the blocks.
        self._block_starts = np.full(n_time_blocks + 1, np.inf)
        self._block_solutions = [None for _ in range(n_time_blocks)]
        self._block_ansatzes = [deepcopy(self.ansatz) for _ in range(n_time_blocks)]

        # Let the integration terminate when the solution becomes too large.
        is_diverged = lambda _, y, __: np.max(np.abs(y)) - self.diverged_threshold
        is_diverged.terminal = True

        initial_solution_fn = None
        block_size = (t_span[-1] - t_span[0]) / n_time_blocks
        for i in range(n_time_blocks):
            block_start = t_span[0] + i * block_size
            block_end = block_start + block_size

            if i > 0:
                if self.resample_domain:
                    gradient_fn = partial(
                        self._evaluate_block,
                        block_idx=i - 1,
                        t=np.array([block_start]),
                        operator="gradient",
                    )
                    self.domain.resample_domain(gradient_fn)

                # Get an approximation of the solution from the previous block.
                initial_solution_fn = partial(
                    self._evaluate_block, block_idx=i - 1, t=np.array([block_start])
                )

            self._block_starts[i] = block_start
            self._block_ansatzes[i].fit(self.domain, target_fn=initial_solution_fn)
            block_precomputed = self._get_precomputed_parameters(
                self._block_ansatzes[i]
            )
            initial_state = self._get_initial_state(
                precomputed=block_precomputed, initial_solution_fn=initial_solution_fn
            )

            # Solve the ODE system for the coefficients.
            solver = solve_ivp(
                fun=self._state_derivative,
                t_span=[block_start, block_end],
                y0=initial_state,
                dense_output=True,
                method=self.ode_solver,
                rtol=self.rtol,
                atol=self.atol,
                events=is_diverged,
                args=(block_precomputed,),
            )
            self._block_solutions[i] = solver.sol
            self._status = self._status and (solver.status == 0)
            if not self._status:
                warnings.warn(
                    f"Solution of the ODE system diverged in the block #{i+1}."
                )
                break
        return self

    def evaluate(
        self, x: npt.ArrayLike, t: npt.ArrayLike, operator: str | None = None
    ) -> npt.ArrayLike:
        """Evaluates the solution and given space and time points.

        Parameters
        ----------
        x: npt.ArrayLike
            The space points to evaluate the solution at.
        t: npt.ArrayLike
            The time points to evaluate the solution at.
        operator: str | None
            If "gradient" or "laplace", applies the corresponding operator
            to the solution.

        Returns
        -------
        npt.ArrayLike
            The evaluated solution.

        """
        t = np.array(t).ravel()
        solution = np.full((x.shape[0], t.shape[0]), np.nan)
        for i in range(len(self._block_solutions)):
            block_mask = (self._block_starts[i] <= t) & (t < self._block_starts[i + 1])
            if not block_mask.any():
                continue
            block_t = t[block_mask]
            solution[:, block_mask] = self._evaluate_block(x, i, block_t, operator)
        return solution

    def _evaluate_block(
        self,
        x: npt.ArrayLike,
        block_idx: int,
        t: npt.ArrayLike,
        operator: str | None = None,
    ) -> npt.ArrayLike:
        t = np.array(t).ravel()
        states = self._block_solutions[block_idx](t)
        n_coeffs = states.shape[0] // self.equation.time_order
        coeffs = states[:n_coeffs]
        ansatz_output = self._block_ansatzes[block_idx].transform(x, operator)

        match operator:
            case None | "laplace":
                return ansatz_output @ coeffs
            case "gradient":
                ansatz_output = np.swapaxes(ansatz_output, 1, 2)
                gradient = ansatz_output @ coeffs
                return np.swapaxes(gradient, 1, 2)
            case _:
                raise ValueError(f"Unknown {operator=}.")

    @property
    def status(self) -> bool:
        """Returns the status of the integration."""
        return self._status
