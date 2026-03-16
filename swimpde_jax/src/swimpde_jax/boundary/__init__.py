from swimpde_jax.abstract import BoundaryCondition
from .dirichlet import Dirichlet, ZeroDirichlet
from .mixed import MixedBoundary
from .neumann import ZeroNeumann
from .periodic import Periodic, PeriodicStrict


def get_boundary_condition(
    boundary_condition: str, scaling: float = 1
) -> BoundaryCondition:
    match boundary_condition.lower():
        case "zero derivative" | "zero neumann":
            return ZeroNeumann(scaling)
        case "zero" | "zero dirichlet":
            return ZeroDirichlet(scaling)
        case "periodic":
            return Periodic(scaling)
        case "periodic strict":
            return PeriodicStrict(scaling)
        case _:
            raise ValueError(f"Unknown boundary condition {boundary_condition}.")


__all__ = [
    "Dirichlet",
    "ZeroDirichlet",
    "ZeroNeumann",
    "Periodic",
    "PeriodicStrict",
    "MixedBoundary",
]
