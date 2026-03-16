from .advection import Advection
from .burgers import Burgers
from .euler_bernoulli import EulerBernoulli
from .helmholtz import Helmholtz
from .poisson import Poisson
from .laplacian import Laplacian
from .plain_equation import PlainEquation

__all__ = [
    "Advection",
    "Burgers",
    "Helmholtz",
    "EulerBernoulli",
    "Poisson",
    "Laplacian",
    "PlainEquation",
]
