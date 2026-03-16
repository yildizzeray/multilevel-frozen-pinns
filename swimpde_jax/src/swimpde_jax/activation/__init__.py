from swimpde_jax.abstract import Activation
from .cos import Cos
from .relu import ReLU
from .sin import Sin
from .tanh import Tanh


def get_activation(activation_name: str) -> Activation:
    """Returns an object for the activation function."""
    activations = {"tanh": Tanh(), "relu": ReLU(), "cos": Cos(), "sin": Sin()}
    if activation_name not in activations:
        raise ValueError(f"Unknown activation {activation_name}.")
    return activations[activation_name]


def get_parameter_sampler(activation_name: str) -> str:
    """Returns a parameter sampler for the activation function."""
    parameter_samplers = {
        "tanh": "tanh",
        "relu": "relu",
        "cos": "tanh",
        "sin": "tanh",
    }
    if activation_name not in parameter_samplers:
        raise ValueError(f"Unknown activation {activation_name}.")

    return parameter_samplers[activation_name]


__all__ = [
    "Cos",
    "Sin",
    "Tanh",
    "ReLU",
    "get_activation",
    "get_parameter_sampler",
]
