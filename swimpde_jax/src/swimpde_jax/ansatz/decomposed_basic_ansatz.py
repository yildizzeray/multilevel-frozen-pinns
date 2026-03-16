from dataclasses import dataclass
from typing import Callable
from functools import partial
import inspect

import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp

from swimnetworks import Dense

from swimpde_jax.abstract import Activation, Ansatz
from swimpde_jax.activation import get_activation, get_parameter_sampler
from swimpde_jax.domain import Domain, SubDomain, DecomposedDomain # in addition to BasicAnsatz

from .utils import get_order


@dataclass(kw_only=True)
class DecomposedBasicAnsatz(Ansatz):
    """Ansatz representing a simple neural network

    Attributes:
    -----------
    n_basis: int
        Number of outer basis functions.
    n_inner_basis: int
        Number of inner basis functions.
    activation: Union[str, Activation]
        Activation function to use for the inner basis.
    parameter_sampler: str
        Parameter sampler to use in the SWIM algorithm.
        See the SWIM package for possible options)
    constraining_operator: Callable[[npt.ArrayLike, float], npt.ArrayLike]
        The constraining operator will be applied to the swim network predictions if provided.
    random_seed: int
        Random seed to be used in all processes that require randomness.

    """

    n_basis: int
    activation: str | Activation
    parameter_sampler: str | Callable | None = None
    constraining_operator: Callable | None = None
    random_seed: int = 1

    _layer: Dense | None = None

    def __post_init__(self):
        if isinstance(self.activation, str):
            self.activation = get_activation(self.activation)
        elif not isinstance(self.activation, Activation):
            raise TypeError(
                "Activation function must be of type str or "
                f"Activation: got {type(self.activation)}"
            )
        if self.parameter_sampler is None:
            self.parameter_sampler = get_parameter_sampler(self.activation.name)

        self._layer = Dense(
            layer_width=self.n_basis,
            activation=self.activation.get_f(),
            parameter_sampler=self.parameter_sampler,
            random_seed=self.random_seed,
            prune_duplicates=False,
        )

        self.n_outputs = self.n_basis
        self.decomposed_domain = None  # in addition to BasicAnsatz
        self.subdomain = None  # in addition to BasicAnsatz
        
    def attach_domain(self, domain: DecomposedDomain | SubDomain): # in addition to BasicAnsatz
        """
        Attaches a subdomain into the ansatz so the ansatz can work on the attached subdomain during fit and transform operations.
        Used for the cases decomposed domains are preferred.

        Parameters:
        subdomain: SubDomain
            Subdomain that will be attached to the ansatz.
        """
        if isinstance(domain, DecomposedDomain):
            self.decomposed_domain = domain
        elif isinstance(domain, SubDomain):
            self.subdomain = domain
        else:
            raise TypeError('The domain that you want to attach has invalid type.')
        
    def detach_domain(self, domain_type='subdomain'): # in addition to BasicAnsatz
        """
        Detaches the set subdomain if exist from ansatz.
        Used before inference time
        """
        assert domain_type in ['subdomain', 'decomposed_domain'], 'Invalid domain type.'
        
        match domain_type:
            case 'subdomain':
                self.subdomain = None
            case 'decomposed_domain':
                self.decomposed_domain = None
        
    def _fit(
        self,
        domain: Domain,
        target_fn: Callable[[npt.ArrayLike], npt.ArrayLike]
        | npt.ArrayLike
        | None = None,
        operator: str | None = None,
    ):
        """
        Fit the model to the data.

        Parameters:
        x: input values of shape (n_points, d)
        y: target values of shape (n_points,)
        order: order of the activation's derivative to apply before fitting.
        """
        if operator not in [None, "laplace"]:
            raise ValueError(
                "BasicAnsatz.fit() only supports opertors "
                f"in [None, 'laplace']. Got: {operator}."
            )

        if target_fn is None:
            target = np.zeros((domain.interior_points.shape[0], self.n_basis))
        elif callable(target_fn):
            target = target_fn(domain.interior_points)
        else:
            target = target_fn

        if target.shape[0] != domain.interior_points.shape[0]:
            raise ValueError(
                f"Target has shape {target.shape}, but "
                f"points have shape {domain.interior_points.shape}."
            )

        order = get_order(operator)
        self._layer.activation = self.activation.get_f(order=order)
        self._layer.fit(domain.interior_points, target)

    def _transform(
        self,
        x: npt.ArrayLike,
        operator: str | None = None,
        coordinate_scaling: npt.ArrayLike | None = None,
    ) -> npt.ArrayLike:
        
        x = jnp.array(x)
        coordinate_scaling = (
            jnp.ones(x.shape[1]) if coordinate_scaling is None else jnp.array(coordinate_scaling)
        )
        
        # Case 1: During inference while no subdomain is attached (pure NN, via AD)
        if self.subdomain is None:
            def layer_fn(x_):
                x_scaled = x_ * coordinate_scaling
                ansatz_output = self._layer.transform(x_scaled)
                return ansatz_output.squeeze()
            
            return self._evaluate(eval_fn=layer_fn, x=x, operator=operator)
        
        # Case 2: During inference while subdomain attached (seperate window_fn, NN, constraint optionally, via analytical combination)
        sub_id = self.subdomain.id
        window_fn = self.decomposed_domain.subdomains[sub_id].window_fn
        
        # NN evaluation -> u(x) = layer(normalize(x_scaled))
        def u_fn(x_):
            x_scaled = x_ * coordinate_scaling
            x_normed = window_fn.normalize(x_scaled)
            u = self._layer.transform(x_normed).squeeze()   # (n_neurons)
            return u
        
        # Window part -> (normalized, depends on ALL windows): w(x)
        def w_fn(x_):
            x_scaled = x_ * coordinate_scaling
            return self.decomposed_domain.calculate_window_evaluation(
                x_scaled, subdomain_idx=sub_id, normalized=True
            ) # scalar
        
        # Constraining operator (optional): c(x)
        if self.constraining_operator is not None:  
            def c_fn(x_):
                return jnp.asarray(self.constraining_operator(x_))
            
        # Operator based evaluations
        if operator is None:
            if self.constraining_operator is not None:
                def eval_fn(xx):
                    w = w_fn(xx)
                    u = u_fn(xx)
                    c = c_fn(xx)
                    return (w * c) * u
            else:
                def eval_fn(xx):
                    w = w_fn(xx)
                    u = u_fn(xx)
                    return w * u
            return self._evaluate(eval_fn=eval_fn, x=x, operator=None)
        
        elif operator == "gradient":
            grad_w = jax.grad(w_fn)     # (d,)
            jac_u = jax.jacfwd(u_fn)    # (n_neurons, d)
            
            if self.constraining_operator is not None:
                grad_c = jax.grad(c_fn) # (d,)
                
                def eval_fn(xx):
                    w = w_fn(xx)
                    u = u_fn(xx)
                    c = c_fn(xx)
                    
                    gw = grad_w(xx) # (d,)
                    gu = jac_u(xx)  # (n_neurons, d)
                    gc = grad_c(xx) # (d,)
                    
                    # y = w*u*c
                    # ∇y = (∇w)*u*c + w*(∇u)*c + w*u*∇c
                    term1 = (gw[None, :] * (u * c)[:, None])    # (n_neurons, d)
                    term2 = (w * c) * gu                        # (n_neurons, d)
                    term3 = (w * u)[:, None] * gc[None, :]      # (n_neurons, d)
                    return term1 + term2 + term3                
            
            else: 
                def eval_fn(xx):
                    w = w_fn(xx)
                    u = u_fn(xx)
                    gw = grad_w(xx) # (d,)
                    gu = jac_u(xx)  # (n_neurons, d)

                    # y = w * u
                    # ∇y = (∇w) u + w (∇u)
                    return gw[None, :] * u[:, None] + w * gu

            return self._evaluate(eval_fn=eval_fn, x=x, operator=None)
        
        elif operator == "laplace":
            # Helpers: laplacian of scalar and vector-valued functions
            def laplace_scalar(f_scalar):
                grad_f = jax.grad(f_scalar)

                def lap(xx):
                    d = xx.shape[0]
                    eye = jnp.eye(d, dtype=xx.dtype)

                    def diag_hess_i(i):
                        ei = eye[i]
                        _, hv = jax.jvp(grad_f, (xx,), (ei,))  # hv = H*e_i
                        return hv[i]

                    return jnp.sum(jax.vmap(diag_hess_i)(jnp.arange(d)))

                return lap

            def laplace_vector(f_vec):
                jac_f = jax.jacfwd(f_vec)

                def lap(xx):
                    d = xx.shape[0]
                    eye = jnp.eye(d, dtype=xx.dtype)

                    def second_diag(i):
                        ei = eye[i]
                        _, hv = jax.jvp(jac_f, (xx,), (ei,))   # hv: (n_neurons, d)
                        return hv[:, i]                       # (n_neurons,)

                    return jnp.sum(jax.vmap(second_diag)(jnp.arange(d)), axis=0)  # (n_neurons,)

                return lap
            
            grad_w = jax.grad(w_fn)                 # (d,)
            lap_w = laplace_scalar(w_fn)            # scalar

            jac_u = jax.jacfwd(u_fn)                # (n_neurons, d)
            lap_u = laplace_vector(u_fn)            # (n_neurons,)

            if self.constraining_operator is not None:
                grad_c = jax.grad(c_fn)             # (d,)
                lap_c = laplace_scalar(c_fn)        # scalar

                def eval_fn(xx):
                    w = w_fn(xx)
                    u = u_fn(xx)
                    c = c_fn(xx)

                    gw = grad_w(xx)                 # (d,)
                    lw = lap_w(xx)                  # ()
                    gu = jac_u(xx)                  # (n_neurons, d)
                    lu = lap_u(xx)                  # (n_neurons,)

                    gc = grad_c(xx)                 # (d,)
                    lc = lap_c(xx)                  # ()

                    # Let a = w*c (scalar). Then y = a*u (vector).
                    # grad a = grad(w*c) = gw*c + w*gc
                    ga = gw * c + w * gc                           # (d,)
                    # lap a = lap(w*c) = lw*c + 2 gw·gc + w*lc
                    la = lw * c + 2.0 * jnp.dot(gw, gc) + w * lc    # ()

                    # Δ(a u) = (Δa)u + 2 (∇a · ∇u) + a Δu
                    grad_dot = jnp.sum(gu * ga[None, :], axis=1)     # (n_neurons,)
                    return la * u + 2.0 * grad_dot + (w * c) * lu

            else:
                def eval_fn(xx):
                    w = w_fn(xx)
                    u = u_fn(xx)

                    gw = grad_w(xx)                 # (d,)
                    lw = lap_w(xx)                  # ()
                    gu = jac_u(xx)                  # (n_neurons, d)
                    lu = lap_u(xx)                  # (n_neurons,)

                    # Δ(w u) = (Δw)u + 2 (∇w · ∇u) + w Δu
                    grad_dot = jnp.sum(gu * gw[None, :], axis=1)     # (n_neurons,)
                    return lw * u + 2.0 * grad_dot + w * lu

            return self._evaluate(eval_fn=eval_fn, x=x, operator=None)
        
        
        elif operator == "dxxxx":
            # For now: keep a safe fallback until we implement normalized-window 4th-order pieces cleanly.
            # This still benefits from the clean u_fn/w_fn/c_fn structure (easy to swap later).
            def full_fn(xx):
                return (w_fn(xx) * c_fn(xx)) * u_fn(xx)

            return self._evaluate(eval_fn=full_fn, x=x, operator="dxxxx")

        else:
            raise ValueError(f"Unsupported operator: {operator}")

    @staticmethod
    def _evaluate(eval_fn: Callable, x: npt.ArrayLike, operator: str=None):
        """
        Evaluates the given evaluation function based on requested operator

        Args:
            eval_fn (Callable): _description_
            operator (str, optional): _description_. Defaults to None.
        """
        if operator is None:
            fn = eval_fn                               # R^d → (n_neurons,)

        elif operator == "gradient":
            fn = jax.jacfwd(eval_fn)                  # R^d → (n_neurons, d)

        elif operator == "laplace" or operator == "dxxxx":
            
            def laplace_fn(eval_fn):
                grad_fn = jax.jacfwd(eval_fn)
                def lap(x):
                    d = x.shape[0]
                    def second_deriv(i):
                        ei = jnp.eye(d, dtype=x.dtype)[i]
                        _, hv = jax.jvp(grad_fn, (x,), (ei,))
                        return hv[:, i]

                    return jnp.sum(jax.vmap(second_deriv)(jnp.arange(d)), axis=0)  # (n_neurons,)
                return lap
            
            match operator:
                case "laplace":
                    fn = laplace_fn(eval_fn)
                case "dxxxx":
                    lap = laplace_fn(eval_fn)
                    fn = laplace_fn(lap)

        else:
            raise ValueError(f"Unsupported operator: {operator}")

        return jax.jit(jax.vmap(fn))(x) 