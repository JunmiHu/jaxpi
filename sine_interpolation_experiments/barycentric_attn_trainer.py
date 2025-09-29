"""
Barycentric attention model definition and training utilities for sin(2πkx) interpolation.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from tqdm import tqdm
from typing import Dict, List, Any, Optional
import json
from problem_utils import DTYPE, generate_data, loss_fn, rel_l2_error, make_fg_oracle, f_sin
from jaxpi.ssbroyden_optax import scale_by_ssbroyden_wolfe_oracle


def cheb_lobatto(N: int):
    """Generate Chebyshev-Lobatto nodes and canonical barycentric weights."""
    j = jnp.arange(N+1, dtype=jnp.float64)
    x = jnp.cos(jnp.pi * j / N)                      # nodes
    # canonical weights for 2nd barycentric form on Lobatto nodes:
    # w_j = (-1)^j * (1/2 at ends, 1 otherwise) — global factor cancels
    delta = jnp.where((j==0) | (j==N), 0.5, 1.0)
    w = ((-1.0) ** j) * delta
    return x, w


def barycentric_lambdas_safe(x, nodes, w):
    """
    Return λ(x) with exact-hit short-circuit, no 1/0 ever computed.

    Handles both constant weights (from RationalBaryWeights) and
    input-dependent weights (from LinearRationalBaryWeights).
    """
    x = x.ravel()

    # Get weights - could be constant or input-dependent
    wv = w(x)  # This will be shape (n+1,) for constant weights or (B, n+1) for input-dependent

    # Handle both cases: constant weights and input-dependent weights
    if wv.ndim == 1:
        # Constant weights case - broadcast to match batch size
        wv = wv[None, :]  # (1, n+1)
        wv = jnp.broadcast_to(wv, (x.shape[0], nodes.shape[0]))  # (B, n+1)
    elif wv.ndim == 2:
        # Input-dependent weights case - already correct shape (B, n+1)
        pass
    else:
        raise ValueError(f"Expected weights to be 1D or 2D, got shape {wv.shape}")

    def row(i):
        xi = x[i]
        wi = wv[i]  # weights for this specific input
        diff = xi - nodes
        k = jnp.argmin(jnp.abs(diff))
        hit = (diff[k] == 0.0)  # exact equality in FP64 at endpoints

        def on_hit(_):
            # r(x_j) = f_j ⇒ λ is one-hot; ensure zero gradient w.r.t. w
            return jax.lax.stop_gradient(jax.nn.one_hot(k, nodes.shape[0], dtype=xi.dtype))

        def off_hit(_):
            z = wi / diff                      # safe: no zero entries in diff here
            S = jnp.sum(z)
            return z / S

        return jax.lax.cond(hit, on_hit, off_hit, operand=None)

    return jax.vmap(row)(jnp.arange(x.shape[0]))


def init_theta_for_bary_weights(n: int, dtype=DTYPE) -> jnp.ndarray:
    """Initialize theta for rational barycentric weights."""
    delta = jnp.ones(n + 1, dtype=dtype).at[0].set(0.5).at[n].set(0.5)
    theta = jnp.log(delta)  # shape [n+1]
    return theta


def realize_weights_from_theta(theta: jnp.ndarray, *, s: jnp.ndarray) -> jnp.ndarray:
    """Realize barycentric weights from theta parameters."""
    theta = theta.astype(DTYPE)
    s = s.astype(DTYPE)
    return s * jnp.exp(theta - jnp.mean(theta))


def alpha_to_match_exact(delta: jnp.ndarray) -> DTYPE:
    """Compute alpha to match exact weights."""
    return jnp.exp(jnp.mean(jnp.log(delta))).astype(DTYPE)


class RationalBaryWeights(nnx.Module):
    """
    Stores trainable theta and an optional log_alpha that collectively
    produce barycentric weights via:
       w = s * exp(theta - mean(theta))             (no alpha)
       w = exp(log_alpha) * s * exp(theta - mean(theta))   (with alpha)
    """
    theta: nnx.Param
    log_alpha: DTYPE | None
    s: jnp.ndarray        # fixed sign pattern (-1)^j
    use_alpha: bool

    def __init__(self, n: int, *, use_alpha: bool = False, rngs: nnx.Rngs = nnx.Rngs(0)):
        theta0 = init_theta_for_bary_weights(n)        # log(delta)
        self.theta = nnx.Param(theta0)
        self.use_alpha = use_alpha
        self.s = (-1.0) ** jnp.arange(n + 1, dtype=DTYPE)
        if use_alpha:
            delta = jnp.ones(n + 1, dtype=DTYPE).at[0].set(0.5).at[n].set(0.5)
            alpha0 = alpha_to_match_exact(delta)       # geometric mean of delta
            self.log_alpha = jnp.log(alpha0)
        else:
            self.log_alpha = None

    def weights(self, xs) -> jnp.ndarray:
        w = realize_weights_from_theta(self.theta.value, s=self.s)
        if self.use_alpha:
            w = jnp.exp(self.log_alpha) * w
        return w

    def __call__(self, xs):
        return self.weights(xs)


class LinearRationalBaryWeights(nnx.Module):
    """
    Stores trainable theta, A matrix, and optional log_alpha that collectively
    produce input-dependent barycentric weights via:
       w(x) = A * x + s * exp(theta - mean(theta))             (no alpha)
       w(x) = A * x + exp(log_alpha) * s * exp(theta - mean(theta))   (with alpha)
    where A is initialized to zero matrix and w is parameterized like RationalBaryWeights
    """
    theta: nnx.Param
    A: nnx.Param          # input-dependent weight matrix
    log_alpha: DTYPE | None
    s: jnp.ndarray        # fixed sign pattern (-1)^j
    use_alpha: bool

    def __init__(self, n: int, *, use_alpha: bool = False, rngs: nnx.Rngs = nnx.Rngs(0)):
        theta0 = init_theta_for_bary_weights(n)        # log(delta)
        self.theta = nnx.Param(theta0)
        # Initialize A to zero matrix (n+1 weights, 1 input dimension)
        self.A = nnx.Param(jnp.zeros((n + 1, 1), dtype=DTYPE))
        self.use_alpha = use_alpha
        self.s = (-1.0) ** jnp.arange(n + 1, dtype=DTYPE)
        if use_alpha:
            delta = jnp.ones(n + 1, dtype=DTYPE).at[0].set(0.5).at[n].set(0.5)
            alpha0 = alpha_to_match_exact(delta)       # geometric mean of delta
            self.log_alpha = jnp.log(alpha0)
        else:
            self.log_alpha = None

    def weights(self, xs) -> jnp.ndarray:
        """
        Compute input-dependent weights w(x) = A * x + w_base
        xs: input of shape (B,) or (B, 1)
        returns: weights of shape (B, n+1) where each row is weights for one input
        """
        xs = jnp.asarray(xs, dtype=DTYPE)
        if xs.ndim == 1:
            xs = xs[:, None]  # (B, 1)

        # Base weights (constant part)
        w_base = realize_weights_from_theta(self.theta.value, s=self.s)
        if self.use_alpha:
            w_base = jnp.exp(self.log_alpha) * w_base

        # Input-dependent part: A @ x for each input in batch
        linear_part = xs @ self.A.value.T  # (B, 1) @ (1, n+1) = (B, n+1)

        # Combine: w(x) = A*x + w_base
        weights = linear_part + w_base[None, :]  # broadcast w_base to (B, n+1)

        return weights

    def __call__(self, xs):
        return self.weights(xs)


class KernelRationalBaryWeights(nnx.Module):
    """
    Produces purely input-dependent barycentric weights via:
       w(x) = A * x
    where A can be initialized with different strategies.
    No bias term is used - this is a pure kernel approach.
    """
    A: nnx.Param          # input-dependent weight matrix

    def __init__(self, n: int, *,
                 init_strategy: str = "zeros",
                 init_scale: float = 1e-3,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        """
        Initialize the A matrix with different strategies:
        - "zeros": Initialize to zero matrix (default)
        - "normal": Initialize with small random normal values
        - "uniform": Initialize with small random uniform values
        - "xavier": Initialize with Xavier/Glorot initialization
        - "he": Initialize with He initialization
        - "orthogonal": Initialize with orthogonal matrix (scaled down)
        - "identity": Initialize close to identity (for single input case)
        """
        shape = (n + 1, 1)  # (n+1 weights, 1 input dimension)

        if init_strategy == "zeros":
            A_init = jnp.zeros(shape, dtype=DTYPE)
        elif init_strategy == "normal":
            A_init = jax.random.normal(rngs(), shape, dtype=DTYPE) * init_scale
        elif init_strategy == "uniform":
            A_init = jax.random.uniform(rngs(), shape, dtype=DTYPE, minval=-init_scale, maxval=init_scale)
        elif init_strategy == "xavier":
            # Xavier/Glorot initialization: scale by sqrt(6/(fan_in + fan_out))
            fan_in, fan_out = shape[1], shape[0]
            limit = jnp.sqrt(6.0 / (fan_in + fan_out)) * init_scale
            A_init = jax.random.uniform(rngs(), shape, dtype=DTYPE, minval=-limit, maxval=limit)
        elif init_strategy == "he":
            # He initialization: scale by sqrt(2/fan_in)
            fan_in = shape[1]
            std = jnp.sqrt(2.0 / fan_in) * init_scale
            A_init = jax.random.normal(rngs(), shape, dtype=DTYPE) * std
        elif init_strategy == "orthogonal":
            # Orthogonal initialization (scaled down)
            if shape[0] >= shape[1]:
                # More rows than columns
                full_matrix = jax.random.normal(rngs(), shape, dtype=DTYPE)
                q, _ = jnp.linalg.qr(full_matrix)
                A_init = q * init_scale
            else:
                # More columns than rows - take transpose, QR, then transpose back
                full_matrix = jax.random.normal(rngs(), (shape[1], shape[0]), dtype=DTYPE)
                q, _ = jnp.linalg.qr(full_matrix)
                A_init = q.T * init_scale
        elif init_strategy == "identity":
            # Identity-like initialization (only works well for square matrices)
            A_init = jnp.eye(shape[0], shape[1], dtype=DTYPE) * init_scale
        else:
            raise ValueError(f"Unknown initialization strategy: {init_strategy}")

        self.A = nnx.Param(A_init)

    def weights(self, xs) -> jnp.ndarray:
        """
        Compute purely input-dependent weights w(x) = A * x
        xs: input of shape (B,) or (B, 1)
        returns: weights of shape (B, n+1) where each row is weights for one input
        """
        xs = jnp.asarray(xs, dtype=DTYPE)
        if xs.ndim == 1:
            xs = xs[:, None]  # (B, 1)

        # Pure kernel approach: w(x) = A * x
        weights = xs @ self.A.value.T  # (B, 1) @ (1, n+1) = (B, n+1)

        return weights

    def __call__(self, xs):
        return self.weights(xs)


class MLP(nnx.Module):
    """Multi-layer perceptron for residual path in mini_transformer."""

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


class mini_transformer(nnx.Module):
    """Mini transformer with barycentric attention and MLP residual path."""

    def __init__(self, mlp, bary_attention, *, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.mlp = mlp
        self.bary_attention = bary_attention

    def __call__(self, x: jnp.ndarray):
        x = self.bary_attention(x)
        res_x = x
        res_x = self.mlp(res_x)
        return x + res_x  # residual connection


class BarycentricAttention(nnx.Module):
    """Single-head barycentric attention over fixed keys (nodes) and weights."""
    nodes: jnp.ndarray | nnx.Param  # shape (M,)
    w: jnp.ndarray | RationalBaryWeights | LinearRationalBaryWeights | KernelRationalBaryWeights
    # Optional: make values trainable; or pass them at call-time.
    V: nnx.Param | None = None  # shape (M,D) if used (constant values)
    # Input-dependent values: V(x) = V_matrix * x + V_bias
    V_matrix: nnx.Param | None = None  # shape (M, D, 1) for input-dependent values
    V_bias: nnx.Param | None = None    # shape (M, D) for input-dependent values
    input_dependent_values: bool = False
    # Query learning parameters
    W: nnx.Param | None = None  # shape (query_hdim, 1) for query transformation
    a: nnx.Param | None = None  # shape (query_hdim, 1) for query transformation
    learn_query: bool = False

    def __init__(self, nodes, w, V_init=None, *,
                 learn_nodes: bool = False,
                 input_dependent_values: bool = False,
                 learn_query: bool = False,
                 query_hdim: int = 8,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        if learn_nodes:
            self.nodes = nnx.Param(jnp.asarray(nodes, dtype=jnp.float64))
        else:
            self.nodes = jnp.asarray(nodes, dtype=jnp.float64)

        if isinstance(w, (RationalBaryWeights, LinearRationalBaryWeights, KernelRationalBaryWeights)):
            self.w = w
        else:
            self.w = lambda x: jnp.asarray(w, dtype=jnp.float64)

        self.input_dependent_values = input_dependent_values
        self.learn_query = learn_query

        if V_init is not None:
            V_init = jnp.asarray(V_init, dtype=nodes.dtype)
            if input_dependent_values:
                # Input-dependent values: V(x) = V_matrix * x + V_bias
                # V_bias initialized to V_init, V_matrix initialized to zeros
                self.V_bias = nnx.Param(V_init)  # shape (M, D)
                M, D = V_init.shape
                self.V_matrix = nnx.Param(jnp.zeros((M, D, 1), dtype=nodes.dtype))  # shape (M, D, 1)
                self.V = None  # Not used in input-dependent mode
            else:
                # Constant values (original behavior)
                self.V = nnx.Param(V_init)
                self.V_bias = None
                self.V_matrix = None

        # Query learning parameters
        if learn_query:
            self.W = nnx.Param(jnp.identity(query_hdim, dtype=DTYPE)[:,0].reshape(-1,1))
            self.a = nnx.Param(jnp.identity(query_hdim, dtype=DTYPE)[:,0].reshape(-1,1))
        else:
            self.W = None
            self.a = None

    def __call__(self, x):
        """
        x: (B,) or (B,1)
        returns: (B,D)
        """
        x = jnp.asarray(x, dtype=self.nodes.dtype)

        if self.learn_query:
            # Transform input to create query
            query = (x @ self.W.T @ self.a).ravel()
            lam = barycentric_lambdas_safe(query, self.nodes, self.w)  # (B,M)
        else:
            # Use input directly as query
            x_ravel = x.ravel()
            lam = barycentric_lambdas_safe(x_ravel, self.nodes, self.w)  # (B,M)

        if self.input_dependent_values:
            # Compute input-dependent values: V(x) = V_matrix * x + V_bias
            # Use original x for computing values, not the transformed query
            x_for_values = x.ravel()  # (B,)
            x_batch = x_for_values[:, None, None, None]  # (B, 1, 1, 1)

            # V_matrix: (M, D, 1), x_batch: (B, 1, 1, 1)
            # We want: (B, M, D) where each (m,d) element is V_matrix[m,d,0] * x[b]
            V_linear = self.V_matrix.value[None, :, :, :] * x_batch  # (1, M, D, 1) * (B, 1, 1, 1) -> (B, M, D, 1)
            V_linear = V_linear.squeeze(-1)  # (B, M, D)

            V_values = V_linear + self.V_bias.value[None, :, :]  # (B, M, D) + (1, M, D) -> (B, M, D)

            # Apply barycentric weights: lam is (B, M), V_values is (B, M, D)
            # We want: sum over M of lam[b,m] * V_values[b,m,d] for each b,d
            result = jnp.einsum('bm,bmd->bd', lam, V_values)  # (B, D)
            return result
        else:
            # Original constant values behavior
            return lam @ self.V.value  # (B,M) @ (M,D) -> (B,D)


def create_barycentric_attention(
    N: int,
    k: int,
    use_exact_init: bool = True,
    rational_representation: str = "standard",
    kernel_init_strategy: str = "normal",
    kernel_init_scale: float = 1e-3,
    input_dependent_values: bool = False,
    learn_nodes: bool = False,
    learn_query: bool = False,
    query_hdim: int = 8,
    mlp_hdim: int = 0,  # 0 means no MLP
    rngs: nnx.Rngs = nnx.Rngs(0)
) -> nnx.Module:
    """Create a BarycentricAttention model with specified configuration."""

    # Create Chebyshev-Lobatto nodes and barycentric weights
    nodes, w_canonical = cheb_lobatto(N)

    # Initialize nodal values
    if use_exact_init:
        V_init = f_sin(nodes, k)[:, None]  # Exact interpolant initialization
    else:
        V_init = jax.random.normal(rngs(), (nodes.shape[0], 1), dtype=DTYPE) * 0.1

    # Create weight representation
    if rational_representation == "standard":
        w = RationalBaryWeights(N, use_alpha=True, rngs=rngs)
    elif rational_representation == "linear":
        w = LinearRationalBaryWeights(N, use_alpha=True, rngs=rngs)
    elif rational_representation == "kernel":
        w = KernelRationalBaryWeights(N,
                                    init_strategy=kernel_init_strategy,
                                    init_scale=kernel_init_scale,
                                    rngs=rngs)
    else:
        raise ValueError(f"Unknown rational representation: {rational_representation}. Use 'standard', 'linear', or 'kernel'")

    # Create barycentric attention with unified interface
    bary_attention = BarycentricAttention(nodes, w, V_init=V_init,
                                        learn_nodes=learn_nodes,
                                        input_dependent_values=input_dependent_values,
                                        learn_query=learn_query,
                                        query_hdim=query_hdim,
                                        rngs=rngs)

    # Add MLP if specified
    if mlp_hdim > 0:
        mlp = MLP([1, mlp_hdim, 1], activation=nnx.tanh, rngs=rngs)
        model = mini_transformer(mlp, bary_attention, rngs=rngs)
        return model
    else:
        return bary_attention


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


def train_barycentric_attention_ssbroyden(
    model: BarycentricAttention,
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
    Train BarycentricAttention using SSBroyden optimizer.

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
    iteration_history = []

    pbar = tqdm(range(steps), desc="Training BarycentricAttention with SSBroyden")
    initial_error = rel_l2_error(model, x_eval, y_eval)
    print(f"Initial eval rel L2: {initial_error:.6e}")

    for i in pbar:
        loss = train_step_ssbroyden(model, optim, x_train, y_train)

        if i % log_interval == 0:
            eval_error = rel_l2_error(model, x_eval, y_eval)
            loss_history.append(float(loss))
            rel_l2_history.append(float(eval_error))
            iteration_history.append(i)
            pbar.set_description(f"step {i}, train loss={loss:.3e}, eval rel L2={eval_error:.3e}")

    # Final evaluation
    final_loss = float(loss_fn(model, x_train, y_train))
    final_error = float(rel_l2_error(model, x_eval, y_eval))
    loss_history.append(final_loss)
    rel_l2_history.append(final_error)
    iteration_history.append(steps)

    return {
        'loss_history': loss_history,
        'rel_l2_history': rel_l2_history,
        'iteration_history': iteration_history,
        'final_loss': final_loss,
        'final_rel_l2': final_error,
        'initial_rel_l2': float(initial_error)
    }


def run_barycentric_attention_experiment(
    N: int,
    k: int,
    steps: int = 50000,
    n_train: int = 4097,
    n_eval: int = 10000,
    seed: int = 0,
    use_exact_init: bool = True,
    rational_representation: str = "standard",
    kernel_init_strategy: str = "normal",
    kernel_init_scale: float = 1e-3,
    input_dependent_values: bool = False,
    learn_nodes: bool = False,
    learn_query: bool = False,
    query_hdim: int = 8,
    mlp_hdim: int = 0,  # 0 means no MLP
    **optimizer_kwargs
) -> Dict[str, Any]:
    """
    Run a complete BarycentricAttention experiment for sin(2πkx) interpolation.

    Returns:
        Dictionary containing experiment results and metadata
    """
    # Set up data
    x_train, y_train, x_eval, y_eval = generate_data(k, n_train, n_eval)

    # Create model
    rngs = nnx.Rngs(seed)
    model = create_barycentric_attention(
        N=N,
        k=k,
        use_exact_init=use_exact_init,
        rational_representation=rational_representation,
        kernel_init_strategy=kernel_init_strategy,
        kernel_init_scale=kernel_init_scale,
        input_dependent_values=input_dependent_values,
        learn_nodes=learn_nodes,
        learn_query=learn_query,
        query_hdim=query_hdim,
        mlp_hdim=mlp_hdim,
        rngs=rngs
    )

    # Count parameters
    param_count = count_parameters(model)

    # Train model
    training_results = train_barycentric_attention_ssbroyden(
        model, x_train, y_train, x_eval, y_eval,
        steps=steps, **optimizer_kwargs
    )

    # Package results
    results = {
        'config': {
            'N': N,
            'k': k,
            'steps': steps,
            'n_train': n_train,
            'n_eval': n_eval,
            'seed': seed,
            'param_count': param_count,
            'use_exact_init': use_exact_init,
            'rational_representation': rational_representation,
            'kernel_init_strategy': kernel_init_strategy,
            'kernel_init_scale': kernel_init_scale,
            'input_dependent_values': input_dependent_values,
            'learn_nodes': learn_nodes,
            'learn_query': learn_query,
            'query_hdim': query_hdim,
            'mlp_hdim': mlp_hdim,
            **optimizer_kwargs
        },
        'training': training_results
    }

    return results


def save_results(results: Dict[str, Any], filepath: str):
    """Save experiment results to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)