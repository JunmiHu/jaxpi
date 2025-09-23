"""
MLP model definition and training utilities for sin(2πkx) interpolation.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from tqdm import tqdm
from typing import Dict, List, Any, Optional
import json
from problem_utils import DTYPE, generate_data, loss_fn, rel_l2_error, make_fg_oracle
from jaxpi.ssbroyden_optax import scale_by_ssbroyden_wolfe_oracle


class MLP(nnx.Module):
    """Multi-layer perceptron with configurable width and depth."""

    def __init__(self, layer_sizes: List[int], activation, *, rngs: nnx.Rngs = nnx.Rngs(0)):
        layers = []
        for i in range(len(layer_sizes) - 2):
            layer = nnx.Linear(layer_sizes[i], layer_sizes[i + 1], rngs=rngs, param_dtype=DTYPE)
            layers.append(layer)
            layers.append(activation)
        layers.append(nnx.Linear(layer_sizes[-2], layer_sizes[-1], rngs=rngs, param_dtype=DTYPE))
        self.layers = nnx.Sequential(*layers)

    def __call__(self, x: jnp.ndarray):
        x = self.layers(x)
        return x


def create_mlp(width: int, depth: int, rngs: nnx.Rngs = nnx.Rngs(0)) -> MLP:
    """Create an MLP with specified width and depth."""
    layer_sizes = [1] + [width] * depth + [1]
    return MLP(layer_sizes, nnx.tanh, rngs=rngs)


def count_parameters(model: nnx.Module) -> int:
    """Count the total number of parameters in the model."""
    params = nnx.state(model, nnx.Param)
    total_params = 0

    def count_leaves(pytree):
        nonlocal total_params
        if hasattr(pytree, 'shape'):  # It's an array
            total_params += pytree.size
        elif isinstance(pytree, dict):
            for value in pytree.values():
                count_leaves(value)
        elif hasattr(pytree, '__dict__'):  # It's an object with attributes
            for value in pytree.__dict__.values():
                count_leaves(value)

    count_leaves(params)
    return total_params


def train_mlp_ssbroyden(
    model: MLP,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_eval: jnp.ndarray,
    y_eval: jnp.ndarray,
    steps: int = 50000,
    lr: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_ls: int = 20,
    log_interval: int = 10,
) -> Dict[str, List[float]]:
    """
    Train MLP using SSBroyden optimizer.

    Returns:
        Dictionary containing training history (loss_history, rel_l2_history)
    """

    # Create the optimizer - use the raw optax version
    tx = scale_by_ssbroyden_wolfe_oracle(lr=lr, c1=c1, c2=c2, max_ls=max_ls)
    optim = nnx.Optimizer(model, tx, wrt=nnx.Param) 
    
    # Build the oracle
    graphdef, base_state = nnx.split(model)
    base_state = nnx.freeze(nnx.pure(base_state))           # make it a pure arrays pytree for JIT-friendliness

    fg_oracle = make_fg_oracle(graphdef, base_state)

    @nnx.jit
    def train_step_ssbroyden(model, opt, x, y):
        def cur_loss(m):
            return loss_fn(m, x, y)
        f_k, grads = nnx.value_and_grad(cur_loss, argnums=nnx.DiffState(0, nnx.Param))(model)
        opt.update(model, grads, f_k=f_k, fg_oracle=fg_oracle, loss_args=(x, y))
        return f_k

    # Training loop
    loss_history = []
    rel_l2_history = []

    pbar = tqdm(range(steps), desc="Training MLP with SSBroyden")
    initial_error = rel_l2_error(model, x_eval, y_eval)
    print(f"Initial eval rel L2: {initial_error:.6e}")

    for i in pbar:
        loss = train_step_ssbroyden(model, optim, x_train, y_train)

        if i % log_interval == 0:
            eval_error = rel_l2_error(model, x_eval, y_eval)
            loss_history.append(float(loss))
            rel_l2_history.append(float(eval_error))
            pbar.set_description(f"step {i}, train loss={loss:.3e}, eval rel L2={eval_error:.3e}")

    # Final evaluation
    final_loss = float(loss_fn(model, x_train, y_train))
    final_error = float(rel_l2_error(model, x_eval, y_eval))
    loss_history.append(final_loss)
    rel_l2_history.append(final_error)

    return {
        'loss_history': loss_history,
        'rel_l2_history': rel_l2_history,
        'final_loss': final_loss,
        'final_rel_l2': final_error,
        'initial_rel_l2': float(initial_error)
    }


def run_mlp_experiment(
    width: int,
    depth: int,
    k: int,
    steps: int = 50000,
    n_train: int = 4097,
    n_eval: int = 10000,
    seed: int = 0,
    **optimizer_kwargs
) -> Dict[str, Any]:
    """
    Run a complete MLP experiment for sin(2πkx) interpolation.

    Returns:
        Dictionary containing experiment results and metadata
    """
    # Set up data
    x_train, y_train, x_eval, y_eval = generate_data(k, n_train, n_eval)

    # Create model
    rngs = nnx.Rngs(seed)
    model = create_mlp(width, depth, rngs)

    # Count parameters
    param_count = count_parameters(model)

    # Train model
    training_results = train_mlp_ssbroyden(
        model, x_train, y_train, x_eval, y_eval,
        steps=steps, **optimizer_kwargs
    )

    # Package results
    results = {
        'config': {
            'width': width,
            'depth': depth,
            'k': k,
            'steps': steps,
            'n_train': n_train,
            'n_eval': n_eval,
            'seed': seed,
            'param_count': param_count,
            **optimizer_kwargs
        },
        'training': training_results
    }

    return results


def save_results(results: Dict[str, Any], filepath: str):
    """Save experiment results to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)