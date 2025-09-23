"""
Utilities for the sin(2πkx) interpolation problem.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from typing import Callable, Tuple, Any

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64


def f_sin(x: jnp.ndarray, k: int) -> jnp.ndarray:
    """Target function: sin(2πkx)"""
    return jnp.sin(2.0 * jnp.pi * DTYPE(k) * x)


def generate_data(k: int, n_train: int, n_eval: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate training and evaluation data for sin(2πkx)"""
    x_train = jnp.linspace(-1.0, 1.0, n_train, dtype=DTYPE).reshape(-1, 1)
    y_train = f_sin(x_train, k).reshape(-1, 1)

    x_eval = jnp.linspace(-1.0, 1.0, n_eval, dtype=DTYPE).reshape(-1, 1)
    y_eval = f_sin(x_eval, k).reshape(-1, 1)

    return x_train, y_train, x_eval, y_eval


def loss_fn(model: nnx.Module, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error loss"""
    y_hat = model(x)
    return jnp.mean((y_hat - y) ** 2)


def rel_l2_error(model: nnx.Module, x_eval: jnp.ndarray, y_eval: jnp.ndarray) -> jnp.ndarray:
    """Relative L2 error"""
    y_hat = model(x_eval)
    return jnp.linalg.norm(y_hat - y_eval) / jnp.linalg.norm(y_eval)


def make_fg_oracle(graphdef: Any, base_state: Any) -> Callable:
    """
    Create an oracle function for SSBroyden optimizer.

    Oracle f(params, x, y) that rebuilds the model and stamps in `params` arrays.
    """
    base_state_pure = nnx.freeze(nnx.pure(base_state))   # pure arrays dict

    def f(params, x, y):
        # 1) Create state from pure dict
        node = nnx.merge(graphdef, base_state_pure)
        # 3) stamp the param arrays into the live Node (structure must match nnx.Param tree)
        nnx.update(node, params)
        # 4) compute loss
        return loss_fn(node, x, y)

    return nnx.jit(nnx.value_and_grad(f, argnums=0))