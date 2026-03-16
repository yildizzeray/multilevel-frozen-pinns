import numpy as np
import numpy.typing as npt
from swimnetworks import Dense


def get_dense_layer_target(
    x: npt.ArrayLike, n_targets: int, activation: str, random_seed: int
) -> npt.ArrayLike:
    layer = Dense(
        layer_width=n_targets,
        activation=activation,
        sample_uniformly=True,
        random_seed=random_seed,
    )
    layer.fit(x, np.zeros((x.shape[0], 1)))
    return layer.transform(x)


def get_order(operator: str | None = None) -> int:
    match operator:
        case None:
            return 0
        case "gradient":
            return 1
        case "laplace":
            return 2
        case "dxxxx":
            return 4
