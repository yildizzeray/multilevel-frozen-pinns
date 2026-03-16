from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Any

import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy.typing as npt

from sklearn.pipeline import Pipeline

from swimnetworks import Linear

from swimpde_jax.abstract import Ansatz, BoundaryCondition, StaticEquation
from swimpde_jax.boundary import get_boundary_condition
from swimpde_jax.domain import Domain, DecomposedDomain, MultiLevelDecomposedDomain
from swimpde_jax.solver.lsrn import lsrn

@dataclass(kw_only=True)
class DecomposedDomainFunctionApproximator:
    """A solver for a general static equation
        L(u)(x) = f(x) on 𝛀
        B(u)(x) = g(x) on ∂𝛀

    Attributes
    ----------
    equation: StaticEquation
        An equation to solve.
    domain: Domain
        A domain on which the equation should be solved.
    forcing: Callable[[npt.ArrayLike, float], npt.ArrayLike]
        The forcing term as a function of x.
    ansatz: Ansatz
        An ansatz to use for computing the basic functions.
    regularization_scale: float
        A regularization scale for solving least square problems.
    boundary_condition: str | BoundaryCondition
        A boundary condition for the equation.

    Methods
    -------
    fit(approx_sol: npt.ArrayLike)
        Solves the PDE by fitting the ansatz to the forcing and boundary.
    evaluate(self, x: npt.ArrayLike) -> npt.ArrayLike:
        Evaluates the solution at the given points.
    """

    equation: StaticEquation
    domain: Domain | DecomposedDomain | MultiLevelDecomposedDomain
    f: Callable[[npt.ArrayLike], npt.ArrayLike]
    ansatz: Ansatz
    regularization_scale: float
    solver: str
    boundary_condition: str | BoundaryCondition
    use_disk: bool = False

    _models: List[List[Any]] = None
    _decomposed_domains: List[DecomposedDomain] = None
    _linear: Linear = None
    _status: bool = None

    def __post_init__(self):
        if not isinstance(self.solver, str):
            raise TypeError("Solver must be defined as str typed object.")
        else:
            available_solvers = ["lstsq", "lsrn", "lsqr", "lsmr"]
            if self.solver not in available_solvers:
                raise ValueError(f"Invalid solver definition. Available solvers are 'lstqs', 'lsrn', 'lsqr', 'lsmr'")

        if isinstance(self.boundary_condition, str):
            self.boundary_condition = get_boundary_condition(self.boundary_condition)

        if isinstance(self.domain, MultiLevelDecomposedDomain):
            self._decomposed_domains = list()
            for decomposed_domain in self.domain.decomposed_domains:
                self._decomposed_domains.append(decomposed_domain)
        elif isinstance(self.domain, DecomposedDomain):
            self._decomposed_domains = list()
            self._decomposed_domains.append(self.domain)
        elif isinstance(self.domain, Domain):
            pass
        else:
            TypeError(f"Invalid domain assignment with domain type {type(self.domain)}.")

    def _equation_operator(self, interior_points: npt.ArrayLike, ansatz: Ansatz = None) -> npt.ArrayLike:
        if ansatz is not None:
            return self.equation.equation_operator(ansatz, interior_points)
        return self.equation.equation_operator(self.ansatz, interior_points)

    def fit(self, approx_sol: npt.ArrayLike | None = None):
        """Solves the PDE by fitting the ansatz to the forcing and boundary.

        Parameters
        ----------
        approx_sol: npt.ArrayLike | None = None
            If provided, approx_sol is used to sample the ansatz.
        """
        self._status = True

        # Calculate target
        if approx_sol is None:
            target = np.zeros_like(self.f(self.domain.interior_points))
        else:
            target = approx_sol

        # Define the output matrix for the linear problem.
        equation_target = self.f(self.domain.interior_points)
        n_targets = equation_target.shape[1]
        boundary_target = self.boundary_condition.get_rhs(self.domain, n_targets)

        y = np.vstack([equation_target, boundary_target])

        # Initialize ansatz or ansatzes based on given domain type
        # If the domain is regular domain use StaticSolver solution
        if self._decomposed_domains is None:
            self.ansatz = deepcopy(self.ansatz)
            self._linear = Linear(regularization_scale=self.regularization_scale)
            self.ansatz.fit(self.domain, target)

            # Define the input matrix for the linear problem.
            equation_output = self._equation_operator(self.domain.interior_points)
            boundary_output = self.boundary_condition.get_lhs(self.ansatz, self.domain)
            X = np.vstack([equation_output, boundary_output])
            n_interior, n_boundary = len(equation_output), len(boundary_output)
            bias = np.vstack(
                [
                    np.ones((n_interior, 1)) * self.equation.equation_bias(),
                    np.ones((n_boundary, 1)) * self.boundary_condition.get_bias_value(),
                ]
            )

            # Solve the linear problem.
            # TODO: use self._linear.fit() once swim can disable a bias.
            linear_weights = np.linalg.lstsq(
                np.hstack([X, bias]), y, rcond=self.regularization_scale
            )[0]
            self._linear.weights = linear_weights[:-1]
            self._linear.biases = linear_weights[-1]
            self._linear.layer_width = linear_weights.shape[1]

        else:
            ### Decomposed domain evaluation procedure
            # Some statistics for global matrix creation
            num_points = self.domain.num_points
            num_neurons = self.ansatz.n_outputs * self.domain.num_subdomain

            # Construct global matrix
            windowed_hidden_layer_outputs = []

            # Initialize memory array if the disk will be used
            if self.use_disk:
                X = np.lib.format.open_memmap('X.npy', mode='w+', dtype='float64', shape=(num_points, num_neurons + self.domain.num_subdomain)) # + 1 if we add bias
            
            print("Initializing local NNs...")
            ansatzs = list()
            subdomain_counter = 0
            num_neurons_after_svd = 0
            for decomposed_domain in self._decomposed_domains:
                interior_window_fn_normalizer = decomposed_domain.window_fn_normalizer(point_type = "interior")
                boundary_output_window_fn_normalizer = decomposed_domain.window_fn_normalizer(point_type = "boundary")
                for subdomain in decomposed_domain.subdomains:
                    print(f"{subdomain_counter+1} / {self.domain.num_subdomain}")
                    # Initialize local NN with corresponded points
                    ansatz = deepcopy(self.ansatz)
                    sampled_subdomain, interior_point_evals = subdomain.get_sampled_subdomain_and_interior_point_evaluations(evaluation_function=self.f)
                    ansatz.fit(domain=sampled_subdomain, target_fn=interior_point_evals)
                    #ansatz.fit(subdomain, target[subdomain.interior_points_mask]) # Sample some fix number of points for each subdomain for initialization
                    num_neurons_after_svd += ansatz.n_outputs
                    ansatzs.append(ansatz)
                    # Obtain equation related outputs to create global matrix
                    equation_output = self._equation_operator(interior_points=decomposed_domain.interior_points, ansatz=ansatz)
                    # TODO: window function evaluations also needed to be handled by _equation_operator logic 
                    interior_window_fn_eval = subdomain.window_fn._f(decomposed_domain.interior_points)
                    interior_normalized_window_fn_eval = np.divide(interior_window_fn_eval, interior_window_fn_normalizer, 
                                 out=np.zeros_like(interior_window_fn_eval), where=interior_window_fn_normalizer != 0)
                    windowed_equation_output = np.hstack(
                        [equation_output, np.ones((decomposed_domain.num_interior_points, 1)) * self.equation.equation_bias()]
                     ) * interior_normalized_window_fn_eval
                    boundary_output = self.boundary_condition.get_lhs(ansatz, decomposed_domain)
                    # TODO: window function evaluations also needed to be handled by boundary condition logic 
                    boundary_window_fn_eval = subdomain.window_fn._f(decomposed_domain.boundary_points)
                    boundary_normalized_window_fn_eval = np.divide(boundary_window_fn_eval, boundary_output_window_fn_normalizer, 
                                 out=np.zeros_like(boundary_window_fn_eval), where=boundary_output_window_fn_normalizer != 0)
                    windowed_boundary_output = np.hstack(
                        [boundary_output, np.ones((decomposed_domain.num_boundary_points, 1)) * self.boundary_condition.get_bias_value()]
                     ) * boundary_normalized_window_fn_eval
                    windowed_hidden_output = np.vstack([windowed_equation_output, windowed_boundary_output])

                    # If the disk will be used than append this into the initialized numpy memory array
                    if self.use_disk:
                        index_begin = subdomain_counter*(ansatz.n_outputs+1)
                        index_end = min((subdomain_counter+1)*(ansatz.n_outputs+1), num_neurons + self.domain.num_subdomain)
                        X[:, index_begin:index_end] = windowed_hidden_output

                    else:
                        if self.solver == "lstsq":
                            windowed_hidden_layer_outputs.append(windowed_hidden_output)
                        else: # For the solvers those can be work for sparce matrix representations
                            windowed_hidden_layer_outputs.append(sp.csr_matrix(windowed_hidden_output))

                    subdomain_counter += 1

            # Finalize global matrix creation based on disk or ram usage preference
            if self.use_disk:
                print('Loading the global matrix from disk...')
                # Trim the redundant part of the previously created array if svd on local networks discards some info
                if num_neurons != num_neurons_after_svd:
                    trimmed_shape_after_svd = (num_points, num_neurons_after_svd + self.domain.num_subdomain)
                    X_orig = np.load('X.npy', mmap_mode='r')
                    X_trimmed = np.lib.format.open_memmap('X_trimmed.npy', mode='w+', dtype=X_orig.dtype, shape=trimmed_shape_after_svd)

                    chunk_size = 10000
                    for i in range(0, num_points, chunk_size):
                        end = min(i + chunk_size, num_points)
                        X_trimmed[i:end] = X_orig[i:end, :trimmed_shape_after_svd[1]]
                    os.remove("X.npy")
                    os.rename("X_trimmed.npy", "X.npy")

                X = np.load('X.npy', mmap_mode='r')
            else:
                if self.solver=="lstsq":
                    X = np.concatenate(windowed_hidden_layer_outputs, axis=1)
                else:
                    X = sp.hstack(windowed_hidden_layer_outputs, format='csr')

            # Solving the created global system 
            print("Solving global system...")
            match self.solver:
                case "lsmr":
                    outer_coeffs = spla.lsmr(X, y, damp=1e-3, maxiter=1000, atol=1e-14, btol=1e-14)[0][:, None]
                case "lsqr":
                    outer_coeffs = spla.lsqr(X, y, atol=1e-14, btol=1e-14)[0][:, None]
                case "lsrn":
                    outer_coeffs = lsrn(X, y, rcond = self.regularization_scale, solver = 'lsqr', show = False, block_size=1000)[0]
                case "lstsq":
                    outer_coeffs = np.linalg.lstsq(X, y, rcond=self.regularization_scale)[0]
                case _:
                    ValueError(f"Invalid solver definition. Defined solver '{self.solver}' does not exist.")
            print(outer_coeffs)
            # Constructs local SWIM-Network model
            print("Constracting local NNs using found coefficients...")
            self._models = list()
            subdomain_counter = 0
            start_idx = 0
            for l, decomposed_domain in enumerate(self._decomposed_domains):
                models_per_level = list()
                for s, subdomain in enumerate(decomposed_domain.subdomains):
                    steps = list()
                    ansatz = ansatzs[subdomain_counter]
                    swim_ansatz = ansatz
                    end_idx = start_idx + ansatz.n_outputs
                    steps.append((f"swim_level_{l+1}_subdomain_{s+1}_ansatz", swim_ansatz))
                    swim_linear = Linear(regularization_scale=self.regularization_scale)
                    swim_linear.weights = outer_coeffs[start_idx:end_idx]
                    swim_linear.biases = outer_coeffs[end_idx]    # biases set to zero for outer layers of local SWIM-NET
                    swim_linear.layer_width = swim_linear.weights.shape[1]
                    swim_linear.n_parameters = np.prod(swim_linear.weights.shape) + np.prod(swim_linear.biases.shape)
                    steps.append((f"swim_level_{l+1}_subdomain_{s+1}_linear", swim_linear))
                    start_idx += ansatz.n_outputs + 1
                    # model creation
                    model = Pipeline(steps, )
                    models_per_level.append(model)
                    subdomain_counter += 1
                self._models.append(models_per_level)

        return self, X

    def evaluate(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Evaluates the solution at the given points.

        Parameters
        ----------
        x: npt.ArrayLike
            Points to evaluate the solution at.
        """
        if self._decomposed_domains is None:
            ansatz_output = self.ansatz.transform(x)
            return self._linear.transform(ansatz_output)
        else:
            global_evaluation = np.zeros((x.shape[0], 1))
            level_evaluations = []
            for d, decomposed_domain in enumerate(self._decomposed_domains):
                level_evaluation = np.zeros((x.shape[0], 1))
                window_fn_normalizer = decomposed_domain.window_fn_normalizer(x)
                for s, subdomain in enumerate(decomposed_domain.subdomains):
                    unnormalized_window_evals = subdomain.window_fn._f(x)
                    normalized_window_evals = np.divide(unnormalized_window_evals, window_fn_normalizer, 
                                 out=np.zeros_like(unnormalized_window_evals), where=window_fn_normalizer != 0)
                    local_evaluation = self._models[d][s].predict(x) * normalized_window_evals
                    level_evaluation += local_evaluation

                global_evaluation += level_evaluation
                level_evaluations.append(level_evaluation)
            return global_evaluation, level_evaluations

    @property
    def status(self):
        return self._status
