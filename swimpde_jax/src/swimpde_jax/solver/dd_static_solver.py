from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Any, Tuple, Optional, Dict
from tqdm import tqdm

import os
import io
import time
import json
import zipfile

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy.typing as npt
import jax

from sklearn.pipeline import Pipeline

from swimnetworks import Linear

from swimpde_jax.abstract import Ansatz, BoundaryCondition, StaticEquation
from swimpde_jax.boundary import get_boundary_condition
from swimpde_jax.domain import Domain, DecomposedDomain, MultiLevelDecomposedDomain
from swimpde_jax.solver.lsrn import lsrn


import warnings
from sklearn.exceptions import NotFittedError
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.pipeline")

@dataclass(kw_only=True)
class DomainDecomposedStaticSolver:
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
    _info: dict | None = None

    def __post_init__(self):
        self._info = {"fit": {"status": False}}
        if not isinstance(self.solver, str):
            raise TypeError("Solver must be defined as str typed object.")
        else:
            available_solvers = ["lstsq", "lsrn", "lsqr", "lsmr"]
            if self.solver not in available_solvers:
                raise ValueError(f"Invalid solver definition. Available solvers are 'lstsq', 'lsrn', 'lsqr', 'lsmr'.")

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
        self._info["fit"]["status"] = True
        t_fit_start = time.perf_counter()

        # Calculate target
        if approx_sol is None:
            target = np.zeros_like(self.f(self.domain.interior_points))
        else:
            target = approx_sol

        # Define the output matrix for the linear problem.
        equation_target = self.f(self.domain.interior_points)
        
        # Initialize ansatz or ansatzes based on given domain type
        # If the domain is regular domain use StaticSolver solution
        if self._decomposed_domains is None:
            n_targets = equation_target.shape[1]
            boundary_target = self.boundary_condition.get_rhs(self.domain, n_targets)
            y = np.vstack([equation_target, boundary_target])

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
            if self.ansatz.constraining_operator is None:
                n_targets = equation_target.shape[1]
                boundary_target = self.boundary_condition.get_rhs(self.domain, n_targets)
                y = np.vstack([equation_target, boundary_target])
            else:
                y = equation_target
            
            # Some statistics for global matrix creation
            num_points = self.domain.num_interior_points if self.ansatz.constraining_operator is not None else self.domain.num_points
            num_neurons = self.ansatz.n_outputs * self.domain.num_subdomain

            # Construct global matrix
            windowed_hidden_layer_outputs = []

            # Initialize memory array if the disk will be used
            if self.use_disk:
                work_dir = os.path.join(os.getcwd(), "temp")
                os.makedirs(work_dir, exist_ok=True)
                X_path = os.path.join(work_dir,'X.npy')
                X = np.lib.format.open_memmap(
                    X_path, mode='w+', dtype='float64', 
                    shape=(num_points, num_neurons))
            
            print("Initializing local SWIM networks...")
            ansatzs = list()
            subdomain_counter = 0
            num_neurons_after_svd = 0
            for _, decomposed_domain in enumerate(tqdm(self._decomposed_domains, desc="Processing decomposed domains", unit="domain")):
                self.ansatz.attach_domain(decomposed_domain)
                for sd_idx, subdomain in enumerate(tqdm(decomposed_domain.subdomains, desc="  ↳ Subdomains", unit="sub", leave=True)):
                    # Initialize local ansatz
                    ansatz = deepcopy(self.ansatz)
                    ansatz.attach_domain(subdomain)
                    ansatz.attach_domain(decomposed_domain)
                    # Initialize ansatz weights using resampling if necessary
                    sampled_subdomain, interior_point_evals = decomposed_domain.get_sampled_subdomain_and_interior_point_evaluations(evaluation_function=self.f, subdomain_idx=sd_idx)
                    ansatz.fit(domain=sampled_subdomain, target_fn=interior_point_evals)
                    #ansatz.fit(subdomain, target[subdomain.interior_points_mask]) # Sample some fix number of points for each subdomain for initialization
                    num_neurons_after_svd += ansatz.n_outputs
                    ansatzs.append(ansatz)
                    # Obtain equation related outputs to create global matrix
                    windowed_equation_output = self._equation_operator(ansatz=ansatz, interior_points=decomposed_domain.interior_points)
                    #windowed_equation_output = np.hstack(
                    #    [equation_output, np.ones((decomposed_domain.num_interior_points, 1)) * self.equation.equation_bias()]
                    #)
                    if self.ansatz.constraining_operator is not None:
                        windowed_hidden_output = windowed_equation_output
                    else:
                        windowed_boundary_output = self.boundary_condition.get_lhs(ansatz, decomposed_domain)
                        windowed_hidden_output = np.vstack([windowed_equation_output, windowed_boundary_output])
                    #windowed_boundary_output = np.hstack(
                    #    [boundary_output, np.ones((decomposed_domain.num_boundary_points, 1)) * self.boundary_condition.get_bias_value()]
                    #)
                    ansatz.detach_domain(domain_type='subdomain')
                    
                    # If the disk will be used than append this into the initialized numpy memory array
                    if self.use_disk:
                        index_begin = subdomain_counter*ansatz.n_outputs
                        index_end = min((subdomain_counter+1)*(ansatz.n_outputs), num_neurons)
                        X[:, index_begin:index_end] = windowed_hidden_output

                    else:
                        if self.solver == "lstsq":
                            windowed_hidden_layer_outputs.append(windowed_hidden_output)
                        else: # For the solvers those can be work for sparce matrix representations
                            windowed_hidden_layer_outputs.append(sp.csr_matrix(windowed_hidden_output))

                    subdomain_counter += 1
                    
            print("All local networks are initialized.")
            
            print("Solving global system...")
            # Finalize global matrix creation based on disk or ram usage preference
            if self.use_disk:
                print('Loading the global matrix from disk...')
                # Trim the redundant part of the previously created array if svd on local networks discards some info
                if num_neurons != num_neurons_after_svd:
                    trimmed_shape_after_svd = (num_points, num_neurons_after_svd)
                    X_orig = np.load(X_path, mmap_mode='r')
                    X_trimmed_path = os.path.join(work_dir, 'X_trimmed.npy')
                    X_trimmed = np.lib.format.open_memmap(X_trimmed_path, mode='w+', dtype=X_orig.dtype, shape=trimmed_shape_after_svd)

                    chunk_size = 10000
                    for i in range(0, num_points, chunk_size):
                        end = min(i + chunk_size, num_points)
                        X_trimmed[i:end] = X_orig[i:end, :trimmed_shape_after_svd[1]]
                    
                    X = X_trimmed
                else:
                    X = np.load(X_path, mmap_mode='r')
            else:
                if self.solver=="lstsq":
                    X = np.concatenate(windowed_hidden_layer_outputs, axis=1)
                else:
                    X = sp.hstack(windowed_hidden_layer_outputs, format='csr')

            # Solving the created global system 
            match self.solver:
                case "lsmr":
                    outer_coeffs = spla.lsmr(X, y, damp=1e-3, maxiter=500, atol=1e-8, btol=1e-8)[0][:, None]
                case "lsqr":
                    outer_coeffs = spla.lsqr(X, y, atol=1e-8, btol=1e-8, iter_lim=500)[0][:, None]
                case "lsrn":
                    outer_coeffs = lsrn(X, y, rcond = self.regularization_scale, solver = 'lsqr', show = False)[0]
                case "lstsq":
                    outer_coeffs = np.linalg.lstsq(X, y, rcond=self.regularization_scale)[0]
                case _:
                    ValueError(f"Invalid solver definition. Defined solver '{self.solver}' does not exist.")
            
            # Constructs local SWIM-Network model
            print("Assigning found coefficients to corresponding local networks...")
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
                    swim_linear.biases = np.zeros_like(outer_coeffs[-1])    # biases set to zero for outer layers of local SWIM-NET
                    swim_linear.layer_width = swim_linear.weights.shape[1]
                    swim_linear.n_parameters = np.prod(swim_linear.weights.shape) + np.prod(swim_linear.biases.shape)
                    steps.append((f"swim_level_{l+1}_subdomain_{s+1}_linear", swim_linear))
                    start_idx += ansatz.n_outputs #+ 1
                    # model creation
                    model = Pipeline(steps, )
                    models_per_level.append(model)
                    subdomain_counter += 1
                self._models.append(models_per_level)
            
            print("SWIM network creations are completed.")

        out = (self, X)
        t_fit_completion = time.perf_counter()
        self._info["fit"]["seconds"] = float(t_fit_completion - t_fit_start)
        return out

    def evaluate(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Evaluates the solution at the given points.

        Parameters
        ----------
        x: npt.ArrayLike
            Points to evaluate the solution at.
        """
        x = jax.numpy.array(x)
        if self._decomposed_domains is None:
            ansatz_output = self.ansatz.transform(x)
            return self._linear.transform(ansatz_output)
        else:
            global_evaluation = jax.numpy.zeros((x.shape[0], 1))
            level_evaluations = []
            level_based_local_evaluations = []
            for d, decomposed_domain in enumerate(self._decomposed_domains):
                level_evaluation = jax.numpy.zeros((x.shape[0], 1))
                local_evaluations = []
                for s, subdomain in enumerate(decomposed_domain.subdomains):
                    normalized_window_evals = jax.vmap(decomposed_domain.calculate_window_evaluation, in_axes=(0, None, None))(x, s, True)[:, None]
                    normed_x = subdomain.window_fn.normalize(x)
                    swim_nn_eval = self._models[d][s].predict(normed_x)
                    local_evaluation = swim_nn_eval * normalized_window_evals
                    level_evaluation += local_evaluation
                    local_evaluations.append(np.array(local_evaluation))

                if self.ansatz.constraining_operator is not None:
                    level_evaluation = jax.vmap(self.ansatz.constraining_operator)(x)[:, None] * level_evaluation
                    
                global_evaluation += level_evaluation
                level_based_local_evaluations.append(local_evaluations)
                level_evaluations.append(level_evaluation)
            return np.array(global_evaluation), [np.array(le) for le in level_evaluations], level_based_local_evaluations

    @property
    def is_fitted(self):
        return self._info["fit"]["status"]

    def save(self, path: str) -> None:
        """
        Saves the model weights and training time.
        
        Format: a single zip file containing:
            - fit_seconds.txt
            - weights.npz
        """

        if not self.is_fitted:
            raise NotFittedError("Solver is not fitted. Call fit() before save().")
        
        arrays : Dict[str, np.ndarray] = {}

        print(f"Saving model parameters to {path}...")
        # for regular domain configuration
        if self._decomposed_domains is None:
            if self._linear is None:
                raise NotFittedError("No linear model found; solver not properly fitted.")
            
            arrays["linear_weights"] = self._linear.weights
            arrays["linear_biases"] = self._linear.biases

            arrays["dense_weights"] = self.ansatz._layer.weights
            arrays["dense_biases"] = self.ansatz._layer.biases
            if self.ansatz._layer.idx_from is not None:
                arrays["dense_idx_from"] = self.ansatz._layer.idx_from
            if self.ansatz._layer.idx_to is not None:
                arrays["dense_idx_to"] = self.ansatz._layer.idx_to
            arrays["ansatz_projection"] = self.ansatz._projection
        
        # for domain decomposed domain configuration
        else:
            if self._models is None:
                raise NotFittedError("No decomposed models found; solver not properly fitted.")
            
            for l, level_models in enumerate(self._models):
                l += 1
                for s, pipe in enumerate(level_models):
                    s += 1
                    ansatz_step, linear_step = None, None
                    for _, step in pipe.steps:
                        if step.__class__.__name__ == "Linear":
                            linear_step = step
                        elif "Ansatz" in step.__class__.__name__:
                            ansatz_step = step
                        else:
                            raise RuntimeError(f"Faced with an invalid step inside pipelines at level={l}, subdomain={s}.")
                        
                    if ansatz_step is None or linear_step is None:
                        raise RuntimeError(f"Pipeline missing expected steps at level={l}, subdomain={s}.")
                    
                    arrays[f"lvl{l}_sd{s}_dense_weights"] = ansatz_step._layer.weights
                    arrays[f"lvl{l}_sd{s}_dense_biases"] = ansatz_step._layer.biases
                    if ansatz_step._layer.idx_from is not None:
                        arrays[f"lvl{l}_sd{s}_dense_idx_from"] = ansatz_step._layer.idx_from
                    if ansatz_step._layer.idx_to is not None:
                        arrays[f"lvl{l}_sd{s}_dense_idx_to"] = ansatz_step._layer.idx_to
                    arrays[f"lvl{l}_sd{s}_ansatz_projection"] = ansatz_step._projection
                    
                    arrays[f"lvl{l}_sd{s}_linear_weights"] = linear_step.weights
                    arrays[f"lvl{l}_sd{s}_linear_biases"] = linear_step.biases

        # build weigths.npz in memory
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        buf.seek(0)

        # build other metadata file
        fit_seconds = self._info["fit"]["seconds"]

        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("fit_seconds.txt", f"{float(fit_seconds):.17g}")
            zf.writestr("weights.npz", buf.read())
                        

    def load(self, path:str) -> "DomainDecomposedStaticSolver":
        """
        Loads model weights and metadata from given file and returns solver instance.
        """

        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        
        with zipfile.ZipFile(path, mode="r") as zf:
            if "fit_seconds.txt" not in zf.namelist() or "weights.npz" not in zf.namelist():
                raise ValueError("Invalid file: expected fit_seconds.txt and weights.npz inside archive.")

            print(f"Loading model parameters from {path}...")
            fit_seconds_str = zf.read("fit_seconds.txt").decode("utf-8").strip()
            weights_bytes = zf.read("weights.npz")

        try:
            fit_seconds = float(fit_seconds_str)
        except ValueError:
            fit_seconds = float("nan")

        npz = np.load(io.BytesIO(weights_bytes), allow_pickle=False)

        # for regular domain configuration
        if self._decomposed_domains is None:
            if "linear_weights" not in npz.files or "linear_biases" not in npz.files:
                raise ValueError("Checkpoint missing linear_weights/linear_biases.")
            
            self._decomposed_domains = None
            self._models = None
            
            self.ansatz._layer.weights = npz["dense_weights"]
            self.ansatz._layer.biases = npz["dense_weights"]
            self.ansatz._layer.idx_from = npz.get("dense_idx_from", None)
            self.ansatz._layer.idx_to = npz.get("dense_idx_to", None)
            self.ansatz._projection = npz["ansatz_projection"]
            self.ansatz.n_parameters = np.prod(self.ansatz._layer.weights.shape) + np.prod(self.ansatz._layer.biases.shape)

            self._linear = Linear(regularization_scale=self.regularization_scale)
            self._linear.weights = npz["linear_weights"]
            self._linear.biases = npz["linear_biases"]
            self._linear.layer_width = self._linear.weights.shape[1]

        # for domain decomposed domain configuration
        else:
            self._models = list()
            for l, decomposed_domain in enumerate(self._decomposed_domains):
                l += 1
                models_per_level = list()
                for s, _ in enumerate(decomposed_domain.subdomains):
                    s += 1
                    dense_w_key = f"lvl{l}_sd{s}_dense_weights"
                    dense_b_key = f"lvl{l}_sd{s}_dense_biases"
                    dense_idx_from_key = f"lvl{l}_sd{s}_dense_idx_from"
                    dense_idx_to_key = f"lvl{l}_sd{s}_dense_idx_to"
                    ansatz_projection = f"lvl{l}_sd{s}_ansatz_projection"
                    linear_w_key = f"lvl{l}_sd{s}_linear_weights"
                    linear_b_key = f"lvl{l}_sd{s}_linear_biases"

                    if dense_w_key not in npz.files or dense_b_key not in npz.files or ansatz_projection not in npz.files:
                        raise ValueError(f"Checkpoint missing ansatz weights for level={l}, subdomain={s}.")
                    if linear_w_key not in npz.files or linear_b_key not in npz.files:
                        raise ValueError(f"Checkpoint missing linear weights for level={l}, subdomain={s}.")

                    steps = list()
                    ansatz = deepcopy(self.ansatz)
                    ansatz._layer.weights = npz[dense_w_key]
                    ansatz._layer.biases = npz[dense_b_key]
                    ansatz._layer.idx_from = npz.get(dense_idx_from_key, None)
                    ansatz._layer.idx_to = npz.get(dense_idx_to_key, None)
                    ansatz._projection = npz[ansatz_projection]
                    ansatz._layer.n_parameters = np.prod(ansatz._layer.weights.shape) + np.prod(ansatz._layer.biases.shape)
                    steps.append((f"swim_level_{l}_subdomain_{s}_ansatz", ansatz))
                    
                    linear = Linear(regularization_scale=self.regularization_scale)
                    linear.weights = npz[linear_w_key]
                    linear.biases = npz[linear_b_key]
                    linear.layer_width = linear.weights.shape[1]
                    linear.n_parameters = int(
                        np.prod(linear.weights.shape) + np.prod(linear.biases.shape)
                    )
                    steps.append((f"swim_level_{l}_subdomain_{s}_linear", linear))

                    model = Pipeline(steps, )
                    models_per_level.append(model)
                self._models.append(models_per_level)
        
        self._info["fit"]["status"] = True
        self._info["fit"]["seconds"] = fit_seconds
        return self