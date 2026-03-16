from swimpde_jax.window_function import WindowFunction
from swimpde_jax.window_function import get_window_function

from .domain import Domain

from dataclasses import dataclass
from copy import deepcopy
from typing import Callable, Type, Any
from functools import partial

from typing import List
import numpy as np
import numpy.typing as npt

import jax.numpy as jnp
from jax.typing import ArrayLike
import jax

@dataclass(kw_only=True)
class SubDomain:
    """
    Contains subdomain related info for DecomposedDomain class
    """
    ax_mins: npt.ArrayLike
    ax_maxs: npt.ArrayLike
    interior_points: npt.ArrayLike
    boundary_points: npt.ArrayLike
    window_fn: WindowFunction | str = None
    id: int = None
    
    def __post_init__(self):
        if self.window_fn is not None:
            if isinstance(self.window_fn, str):
                self.window_fn = get_window_function(window_function_name=self.window_fn, ax_mins=self.ax_mins, ax_maxs=self.ax_maxs)
            elif not isinstance(self.window_fn, WindowFunction):
                raise ValueError(f"The window function type {type(self.window_fn)} is invalid.")
        else:
            self.window_fn = get_window_function(window_function_name="gaussian", ax_mins=self.ax_mins, ax_maxs=self.ax_maxs)
    
    @property
    def all_points(self) -> npt.ArrayLike:
        return np.vstack([self.boundary_points, self.interior_points])
    
    @property
    def n_dim(self):
        return len(self.ax_mins)

@dataclass(kw_only=True)
class DecomposedDomain(Domain):
    """
    Specifies a partitioned domain as a collection of points.

    Attributes
    ----------
    num_partition_per_dim: int
        Specifices number of partition along each dimension 
    overlap_ratio: float [0, 1)
        Specifices overlapping ratio along each dimension
    window_function: Callable
        Window function will be used for partition of unity
    Methods
    -------
    get_core_interior_mask(self, margin: float) -> npt.ArrayLike
        Returns a mask of the interior points that are not near the boundary.
    """
    
    num_partition_per_dim: int
    overlap_ratio: float = 1.9
    window_fn: str = "gaussian"
    num_min_sample_point_per_subdomain: int = None
    subdomain_bounds : ArrayLike = None

    def __post_init__(self):
        super().__post_init__()
        
        if self.num_partition_per_dim <= 0:
            raise ValueError('"num_partition_per_dim" must be positive integer.') 
        
        self.subdomain_bounds = jnp.asarray(self._subdivide(), dtype=jnp.float32)
        self.subdomains = []
        for sd_id, sd_bound in enumerate(self.subdomain_bounds):
            mins, maxs = np.array(sd_bound[0]), np.array(sd_bound[1])
            sd_interior_pts_mask = np.all((self.interior_points >= mins) & (self.interior_points <= maxs), axis=1)
            sd_interior_pts = self.interior_points[sd_interior_pts_mask]
            sd_boundary_pts_mask = np.all((self.boundary_points >= mins) & (self.boundary_points <= maxs), axis=1)
            sd_boundary_pts = self.boundary_points[sd_boundary_pts_mask]
            self.subdomains.append(
                SubDomain(
                    ax_mins=mins, 
                    ax_maxs=maxs,
                    interior_points=sd_interior_pts,
                    boundary_points=sd_boundary_pts,
                    window_fn=self.window_fn,
                    id=sd_id,
                    )
            )

    @property
    def num_subdomain(self):
        return len(self.subdomains)
    
    def calculate_window_evaluation(self, point, subdomain_idx, normalized=True, eps=1e-12):
        bounds = jnp.asarray(self.subdomain_bounds)  # (M,2,D)
        M = bounds.shape[0]
        if not (0 <= subdomain_idx < M):
            return jnp.nan
        
        ax_mins = bounds[:, 0, :]
        ax_maxs = bounds[:, 1, :]
        window_fn = get_window_function(window_function_name=self.window_fn, ax_mins=ax_mins, ax_maxs=ax_maxs) 
        vals = window_fn.f(point)
        
        unnorm = vals[subdomain_idx]
        if not normalized:
            return unnorm

        norm = jnp.maximum(jnp.sum(vals), eps)
        return unnorm / norm

    
    def get_sampled_subdomain_and_interior_point_evaluations(self, evaluation_function, subdomain_idx, random_seed=17):
        
        assert subdomain_idx <= len(self.subdomains), ValueError('Subdomain index belongs is not valid.')
        
        subdomain = self.subdomains[subdomain_idx]
        
        if self.num_min_sample_point_per_subdomain is not None:
            if self.num_min_sample_point_per_subdomain > len(subdomain.interior_points):
                rng = np.random.default_rng(random_seed)
                resampled_points = rng.uniform(low=subdomain.ax_mins, high=subdomain.ax_maxs, size=(self.num_min_sample_point_per_subdomain, subdomain.n_dim))
                resampled_domain = deepcopy(subdomain)
                resampled_domain.interior_points = resampled_points
                return resampled_domain, evaluation_function(resampled_points)

        return subdomain, evaluation_function(subdomain.interior_points)
        
        
    def _subdivide(self) -> list[tuple[list[float], list[float]]]:
        """
        Divide the domain into overlapping subdomains consistent with
        the multilevel FBPINN formulation (Dolean et al., 2024).

        - Outer subdomains extend beyond the domain boundaries.
        - overlap_ratio (δ) > 1 means each subdomain overlaps its neighbors
          by (δ - 1) times its nominal width.
        - δ = 1 gives just-touching subdomains, δ < 1 gives gaps.

        Returns
        -------
        List[tuple[list[float], list[float]]]
            List of subdomains as (mins, maxs) per subdomain.
        """

        coords_per_dim = []
        for d in range(self.n_dim):
            n = self.num_partition_per_dim
            a, b = self.ax_mins[d], self.ax_maxs[d]

            if n == 1:
                # Single subdomain: extend halfway outside by overlap_ratio
                width = b - a
                extra = 0.5 * (self.overlap_ratio - 1.0) * width
                dim_partitions = [(a - extra, b + extra)]
            else:
                # Centers and half-widths as in Eq. (9)
                mu = np.linspace(0.0, 1.0, n)  # normalized centers (0 to 1)
                sigma = (self.overlap_ratio / 2.0) / (n - 1)  # normalized half-width
                dim_partitions = []

                for j in range(n):
                    # Normalized start/end, may extend beyond [0,1]
                    start_norm = mu[j] - sigma
                    end_norm = mu[j] + sigma
                    # Map back to physical coordinates
                    start = a + start_norm * (b - a)
                    end = a + end_norm * (b - a)
                    dim_partitions.append((start, end))

            coords_per_dim.append(dim_partitions)

        # Cartesian product to form all subdomain boxes
        subdomains_boundaries = [
            (
                [coords_per_dim[d][idx[d]][0] for d in range(self.n_dim)],
                [coords_per_dim[d][idx[d]][1] for d in range(self.n_dim)]
            )
            for idx in np.ndindex(*([self.num_partition_per_dim] * self.n_dim))
        ]
        return subdomains_boundaries

@dataclass(kw_only=True)
class MultiLevelDecomposedDomain(Domain):
    """
    Specifies a multilevel partitioned domain composed of DecomposedDomain.

    Attributes
    ----------
    num_partition_per_level: List[int, ...]
        Specifices number of partion along each dimension per level
    overlap_ratio: float [0, 1)
        Specifices overlapping ratio along each dimension
        
    Methods
    -------
    get_core_interior_mask(self, margin: float) -> npt.ArrayLike
        Returns a mask of the interior points that are not near the boundary.
    """
    num_partition_per_level: List[int]
    overlap_ratio: float
    window_fn: str = "gaussian"
    num_min_sample_point_per_subdomain: int = None

    def __post_init__(self):
        super().__post_init__()
        
        # Partition per level list check
        if any(self.num_partition_per_level) <= 0 or not isinstance(any(self.num_partition_per_level), int):
            raise ValueError('Number of partition per level values must be positive integer.')
        
        # Creation of decomposed domains for each level
        self.decomposed_domains = []
        for l, num_partition in enumerate(self.num_partition_per_level):
            self.decomposed_domains.append(DecomposedDomain(
                interior_points=self.interior_points, 
                boundary_points=self.boundary_points,
                num_partition_per_dim=num_partition,
                overlap_ratio=self.overlap_ratio,
                window_fn=self.window_fn,
                num_min_sample_point_per_subdomain=self.num_min_sample_point_per_subdomain))

        
    @property
    def num_level(self):
        """
        Returns number of level.
        """
        return len(self.num_partition_per_level)
    
    @property
    def num_subdomain_per_level(self):
        "Returns a list contains number of subdomain info per level."
        return [num_level**self.n_dim for num_level in self.num_partition_per_level]
    
    @property
    def num_subdomain(self):
        """
        Returns total number of subdomain.
        """
        return sum(self.num_subdomain_per_level)
    
    @property
    def subdomains(self):
        """
        Returns all subdomains in a list
        """
        return [subdomain for decomposed_domain in self.decomposed_domains for subdomain in decomposed_domain.subdomains]


if __name__ == "__main__":

    X_SPAN = 0, 1   # span is same along x1 and x2
    N_DIM = 2       # dimensions x1, x2
    N_POINTS_ALONG_AXES_TRAIN = 500   # Number of points will be sampled along each direction with equal distance
    N_POINTS_ALONG_AXES_EVAL = 256

    # Sampling the points within the problem domains
    x1_space = np.linspace(*X_SPAN, N_POINTS_ALONG_AXES_TRAIN)
    x2_space = np.linspace(*X_SPAN, N_POINTS_ALONG_AXES_TRAIN)
    x1x2_space = np.stack(np.meshgrid(x1_space, x2_space), axis=-1).reshape(-1, N_DIM)

    # Domain definition using sampled points and predefined boundary values
        # Mask definition for seperating interior and boundary points
    boundary_mask = (
        (x1x2_space[:, 0] == X_SPAN[0]) |
        (x1x2_space[:, 0] == X_SPAN[1]) |
        (x1x2_space[:, 1] == X_SPAN[0]) |
        (x1x2_space[:, 1] == X_SPAN[1])  
    )

    x_interior = x1x2_space[~boundary_mask]
    x_boundary = x1x2_space[boundary_mask]

    dd = DecomposedDomain(interior_points=x_interior, boundary_points=x_boundary, num_partition_per_dim=2, overlap_ratio=0.3)
    for idx, subdomain in enumerate(dd.subdomains):
        print(f"Subdomain {idx+1} : {subdomain.window_fn_eval}")