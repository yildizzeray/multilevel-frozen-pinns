"""
Defines and runs all the experiments for Laplacian2D weak scaling problem for the models below:

SWIMPDE-MLDD: The SWIMPDE based models with 'tanh' sampling procedure.
ELMPDE-MLDD: The SWIMPDE based models with 'random' sampling procedure.

Multilevel FBPINNs: Details can be found in paper https://arxiv.org/abs/2306.05486. The configuration is remained as it is.
"""

### General imports
import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial
import time
import jax
print(jax.default_backend())

### SWIMPDE related imports
from swimpde_jax.domain import MultiLevelDecomposedDomain
from swimpde_jax.solver import DomainDecomposedStaticSolver
from swimpde_jax.boundary import ZeroDirichlet
from swimpde_jax.ansatz import DecomposedBasicAnsatz
from swimpde_jax.equation import Laplacian

### Multilevel FBPINNs related imports
from fbpinns.constants import Constants, get_subdomain_ws
from fbpinns.domains import RectangularDomainND
from fbpinns.decompositions import MultilevelRectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.trainers import FBPINNTrainer
from fbpinns.multilevel_paper.problems import Laplace2D_multiscale


### General parameter definitions
problem_name = Laplace2D_multiscale.__name__
tag = "weak"# test type
max_config = 3# maximum configuration to be run starting from 1 up to defined
n_seeds = 3# number of run per configuration
overlapping_ratio = 1.9

### SWIM-PDE related functions and definitions
swim_models = ["Frozen-PINN-swim-MLDD", "Frozen-PINN-elm-MLDD"]
swim_num_basis= {"Frozen-PINN-swim-MLDD": 32, "Frozen-PINN-elm-MLDD": 32}
swim_activation="tanh"
swim_param_samplers = {"Frozen-PINN-swim-MLDD": "tanh", "Frozen-PINN-elm-MLDD": "random"}
swim_svd_cutoff= 1e-10
swim_regularization_scale = 1e-10
swim_window_fn = "gaussian"
swim_solver = "lstsq"
swim_use_disk = False

swim_problem_configs = {
    1 : {"num_sin_basis": 1, "num_train_point_per_dim":20, "num_test_point_per_dim": 350, "partition": [1, 2]},
    2 : {"num_sin_basis": 2, "num_train_point_per_dim":40, "num_test_point_per_dim": 350, "partition": [1, 2, 4]},
    3 : {"num_sin_basis": 3, "num_train_point_per_dim":80, "num_test_point_per_dim": 350, "partition": [1, 2, 4, 8]},
    4 : {"num_sin_basis": 4, "num_train_point_per_dim":160, "num_test_point_per_dim": 350, "partition": [1, 2, 4, 8, 16]},
    5 : {"num_sin_basis": 5, "num_train_point_per_dim":320, "num_test_point_per_dim": 350, "partition": [1, 2, 4, 8, 16, 32]},
    6 : {"num_sin_basis": 6, "num_train_point_per_dim":640, "num_test_point_per_dim": 350, "partition": [1, 2, 4, 8, 16, 32, 64]},
}

span = (
        [0, 0], # mins along each axes
        [1, 1] # maxs along each axes
    )

n_dim = len(span)

def laplacian_forcing(x: npt.ArrayLike, n_basis: int) -> npt.ArrayLike:
    """ Source function definition."""
    coeffs = 2 ** np.arange(1, n_basis + 1)  
    omega_pi_squared = (coeffs * np.pi) ** 2  
    sin_x1 = np.sin(np.outer(coeffs, np.pi * x[:, 0]))  
    sin_x2 = np.sin(np.outer(coeffs, np.pi * x[:, 1])) 

    forcing = np.sum(omega_pi_squared[:, None] * sin_x1 * sin_x2, axis=0)  
    return ((2 / n_basis) * forcing)[:, None]

def constraining_operator(points: ArrayLike, sigma: float) -> ArrayLike:
    """
    [Cu](x) = tanh(x1/σ) tanh((1−x1)/σ) tanh(x2/σ) tanh((1−x2)/σ)
    """
    x, y = points[0], points[1]
    return (
        jnp.tanh(x / sigma) *
        jnp.tanh((1 - x) / sigma) *
        jnp.tanh(y / sigma) *
        jnp.tanh((1 - y) / sigma)
    )

# Training loop
for swim_model in swim_models:
    run_count = 0
    print(f"\nTraining of {swim_model} models are started.")
    num_model_run = n_seeds * max_config
    print(f"{num_model_run} number of run will be performed for {swim_model} including max configuration {max_config}.")
    for config in range(1, max_config+1):
        num_sin_basis = swim_problem_configs[config]['num_sin_basis']
        partition = swim_problem_configs[config]['partition']
        num_points_per_dim_train = swim_problem_configs[config]['num_train_point_per_dim']
        num_points_per_dim_test = swim_problem_configs[config]['num_test_point_per_dim']

        # Dataset Creation
        x_train = np.linspace(span[0][0], span[1][0],  num_points_per_dim_train)
        y_train = np.linspace(span[0][1], span[1][1],  num_points_per_dim_train)
        train_inputs = np.stack(np.meshgrid(x_train, y_train), axis=-1).reshape(-1, n_dim)

        # Domain Definition
        boundary_mask = (
            (train_inputs[:, 0] == span[0][0]) |
            (train_inputs[:, 0] == span[1][0]) |
            (train_inputs[:, 1] == span[0][1]) |
            (train_inputs[:, 1] == span[1][1])  
        )

        domain = MultiLevelDecomposedDomain(
            interior_points=train_inputs[~boundary_mask],
            boundary_points=train_inputs[boundary_mask],
            num_partition_per_level=partition,
            overlap_ratio=overlapping_ratio, 
            window_fn=swim_window_fn,
        )

        # Equation Definition
        equation = Laplacian()
    
        # Forcing function definition
        forcing_function = partial(laplacian_forcing, n_basis=num_sin_basis)

        # Boundary condition definition
        boundary_condition = ZeroDirichlet()
    
        # Other steps based on randomness

        for i, seed in enumerate(n_seeds):
            run_name = f"{swim_model}_{tag}_{problem_name}_{str(partition).replace(' ', '')}-levels_{overlapping_ratio}-overlap_{swim_num_basis[swim_model]}-basis_{seed}"
            print(f"\nRun {run_count+1} / {num_model_run} : {run_name}")
    
            # Ansatz definition 
            ansatz = DecomposedBasicAnsatz(
                n_basis=swim_num_basis[swim_model],
                activation=swim_activation,
                random_seed=seed,
                svd_cutoff=swim_svd_cutoff,
                constraining_operator=partial(constraining_operator, sigma=1/(2**num_sin_basis)),
                parameter_sampler=swim_param_samplers[swim_model],
            )

            # Solver definition
            solver = DomainDecomposedStaticSolver(
                domain=domain,
                ansatz=ansatz,
                boundary_condition=boundary_condition,
                equation=equation,
                f=forcing_function, 
                regularization_scale=swim_regularization_scale,
                solver=swim_solver,
                use_disk=swim_use_disk,
            )

            # Training and saving model
            solver.fit()
            solver.save(path=f"./results/swimpde/{run_name}")
            run_count += 1


### Multilevel FBPINNs  related functions and definitions
def pscan(p0, *pss):
    "scan from fixed point"
    assert len(p0) == len(pss)
    return [list(p0[:ip] + [p] + p0[ip+1:]) for ip, ps in enumerate(pss) for p in ps]


def run_FBPINN():
    run = f"FBPINN_{tag}_{problem.__name__}_{network.__name__}_{l}-levels_{overlapping_ratio}-overlap_{h}-layers_{p}-hidden_{n[0]}-n_{lr}-lr-{seed}"
    c = Constants(
        run=run,
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=problem,
        problem_init_kwargs=problem_init_kwargs,
        decomposition=MultilevelRectangularDecompositionND,
        decomposition_init_kwargs = dict(
                    subdomain_xss=subdomain_xss,
                    subdomain_wss=subdomain_wss,
                    unnorm=unnorm,
                    ),
        network=network,
        network_init_kwargs=network_init_kwargs,
        n_steps=n_steps,
        ns=(n,),
        n_test=n_test,
        optimiser_kwargs=dict(learning_rate=lr),
        seed=seed,
        test_freq=test_freq,
        model_save_freq=model_save_freq,
        )
    return c, "FBPINN"
runs=[]
model_save_freq=10000
test_freq=500
tag = "weak"

domain=RectangularDomainND
domain_init_kwargs=dict(xmin=np.array([0.,0]),
                        xmax=np.array([1.,1.]),)

problem=Laplace2D_multiscale
omegas=[2, 4, 8, 16, 32, 64][:max_config]
unnorm=(0., 0.75)# unnorm

n_steps=50000# number training steps
ns=[(20,20),(40,40),(80,80),(160,160),(320,320),(640,640)][:max_config] # add omegas to problem, whilst increasing levels and collocation points
n_test=(350,350)# number test points
lr = 1e-3# learning rate

overlapping_ratio=1.9# overlaping ratio

# increase levels, but not collocation points
ls=[2, 3, 4, 5, 6, 7][:max_config]# number of levels
lr = 1e-3# learning rate

for il,(l_,n) in enumerate(zip(ls, ns)):
    problem_init_kwargs=dict(omegas=omegas[:il+1], sd=1/omegas[il])
    h,p=1,16

    # multilevel scaling
    l = [2**i for i in range(l_)]
    subdomain_xss = [[np.array([0.5]),np.array([0.5])]] + [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l[1:]]
    subdomain_wss = [[np.array([overlapping_ratio*1.]),np.array([overlapping_ratio*1.])]] + [get_subdomain_ws(subdomain_xs, overlapping_ratio) for subdomain_xs in subdomain_xss[1:]]
    layer_sizes = [2,] + [p,]*h + [1,]
    network, network_init_kwargs = FCN, dict(layer_sizes=layer_sizes)
    for seed in range(n_seeds): runs.append(run_FBPINN())


trainers = {"FBPINN":FBPINNTrainer}
    
num_run = len(runs)
for run_count, (c, key) in enumerate(runs):
    print(f"\nRUN {run_count+1}/{num_run}")
    c.show_figures = c.clear_output = False
    run = trainers[key](c)
    run.train()