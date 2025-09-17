# cheb_vs_plain_mlp.py
# Compare a plain tanh MLP vs. a Chebyshev-embedded tanh MLP on f(x)=sin(kx)
# Uses JAX + flax.nnx + optax
#
# Embeddings:
#   - "none":    input is x ∈ [-1, 1]
#   - "theta":   input is θ = arccos(x) ∈ [0, π]              (default)
#   - "chebM":   input is [cos(θ), cos(2θ), ..., cos(Mθ)]     (set --M > 0)
#
# Notes:
# * We clip x to (-1,1) before arccos to avoid boundary NaNs. Gradients are
#   only w.r.t. weights, so arccos' singular derivative at ±1 isn't an issue.
# * NNX API: Optimizer + value_and_grad + jit pattern taken from docs:
#   https://flax.readthedocs.io/en/latest/nnx_basics.html
#
# Example:
#   python cheb_vs_plain_mlp.py --k 8 --steps 5000 --widths 64,128 --depths 2,4 --embed theta
#   python cheb_vs_plain_mlp.py --k 8 --embed cheb --M 8

import argparse, functools, sys, os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
DTYPE = jnp.float64

import optax
from flax import nnx

# Add parent directory to path to import jaxpi
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ssbroyden_optax directly to avoid dependency issues
import importlib.util
spec = importlib.util.spec_from_file_location("ssbroyden_optax", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jaxpi", "ssbroyden_optax.py"))
ssbroyden_optax = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssbroyden_optax)
scale_by_ssbroyden_wolfe_oracle = ssbroyden_optax.scale_by_ssbroyden_wolfe_oracle

# ---------- Target function ----------
def target_fn(x, k: float):
    return jnp.sin(k * x)

# ---------- Embeddings ----------
def embed_none(x):
    # x: (B,)
    return x[:, None].astype(DTYPE)  # (B,1)

def embed_theta(x, eps=1e-7):
    # θ = arccos(x_clipped), shape (B,1)
    x_clip = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    theta = jnp.arccos(x_clip)
    return theta[:, None].astype(DTYPE)

def embed_chebM(x, M: int, eps=1e-7):
    # features = [cos(θ), cos(2θ), ..., cos(Mθ)], shape (B,M)
    x_clip = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    theta = jnp.arccos(x_clip)  # (B,)
    js = jnp.arange(1, M + 1, dtype=DTYPE)  # (M,)
    feats = jnp.cos(theta[:, None] * js[None, :])  # (B,M)
    return feats.astype(DTYPE)

# ---------- Barycentric 2nd-form helpers (Chebyshev–Lobatto) ----------
def lobatto_nodes_weights(N: int, dtype=DTYPE):
    """Chebyshev–Lobatto nodes x_j = cos(pi j / N) and canonical barycentric weights.
       w_j = (-1)^j * (1/2 at endpoints, 1 elsewhere). Global scale cancels."""
    j = jnp.arange(N + 1, dtype=dtype)
    x = jnp.cos(jnp.pi * j / N)
    delta = jnp.where((j == 0) | (j == N), 0.5, 1.0)
    w = ((-1.0) ** j) * delta
    return x, w  # shapes: (N+1,), (N+1,)

def barycentric_lambdas(x, nodes, w, tol=1e-15):
    """
    Compute λ(x) row-wise for a batch x:(B,) against fixed nodes/w:(M,).
    Returns (B,M). If x hits a node within tol, returns exact one-hot.
    """
    x = x.ravel()
    diff = x[:, None] - nodes[None, :]             # (B,M)
    # detect exact-node hits per row
    k_min = jnp.argmin(jnp.abs(diff), axis=1)      # (B,)
    hit = jnp.take_along_axis(jnp.abs(diff), k_min[:, None], axis=1)[:, 0] < tol

    # raw barycentric z_j = w_j / (x - x_j)
    z = w[None, :] / diff                          # (B,M)
    S = jnp.sum(z, axis=1, keepdims=True)          # (B,1)
    lam = z / S                                    # (B,M)

    # overwrite rows that hit a node: exact Kronecker property
    onehot = jax.nn.one_hot(k_min, nodes.shape[0], dtype=nodes.dtype)
    lam = jnp.where(hit[:, None], onehot, lam)
    return lam

def embed_bary_lambdas_factory(N: int, dtype=DTYPE):
    """
    Returns an embedding function x -> λ(x) ∈ R^{N+1} using Chebyshev–Lobatto nodes.
    Interprets N as the polynomial degree (so there are N+1 nodes/features).
    """
    nodes, w = lobatto_nodes_weights(N, dtype=dtype)
    def _embed_bary(x):
        # x: (B,) -> lambdas: (B, N+1)
        return barycentric_lambdas(x, nodes, w)
    return _embed_bary


def make_embedder(kind: str, M: int):
    if kind == "none":
        return embed_none
    elif kind == "theta":
        return embed_theta
    elif kind == "cheb":
        if M <= 0:
            raise ValueError("For --embed cheb you must set --M > 0.")
        return functools.partial(embed_chebM, M=M)
    elif kind == "bary":
        if M <= 0:
            raise ValueError("For --embed bary you must set --M > 0 (degree N).")
        # Here M is interpreted as degree N => N+1 lambdas
        return embed_bary_lambdas_factory(N=M, dtype=DTYPE)
    else:
        raise ValueError(f"Unknown embed kind: {kind}")

def infer_in_dim(embed_kind: str, M: int):
    if embed_kind == "none":  return 1
    if embed_kind == "theta": return 1
    if embed_kind == "cheb":  return M
    if embed_kind == "bary":  return M + 1  # N+1 lambdas
    raise ValueError(embed_kind)

# ---------- Model ----------
class TanhMLP(nnx.Module):
    def __init__(self, in_dim: int, width: int, depth: int, *, rngs: nnx.Rngs):
        layers = []
        din = in_dim
        for _ in range(depth):
            layers.append(nnx.Linear(din, width, rngs=rngs))
            din = width
        self.layers = tuple(layers)
        self.out = nnx.Linear(din, 1, rngs=rngs)

    def __call__(self, x):
        # x: (B, in_dim)
        h = x
        for lin in self.layers:
            h = jnp.tanh(lin(h))
        return self.out(h)  # (B,1)

# ---------- Data sampling ----------
def sample_uniform(key, n):
    # avoid the exact endpoints (±1) when using arccos
    return jax.random.uniform(key, (n,), minval=-0.999999, maxval=0.999999, dtype=DTYPE)

# ---------- Training step (NNX style) ----------
def make_train_step():
    @nnx.jit
    def train_step(model, optimizer, xs_embed, ys):
        def loss_fn(m: TanhMLP):
            preds = m(xs_embed)
            return jnp.mean((preds - ys) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)  # in-place update
        return loss
    return train_step

# ---------- Oracle for SSBroyden ----------
def make_fg_oracle(graphdef):
    """Create oracle function for SSBroyden optimizer"""
    def f(params, xs_embed, ys):
        model_tmp = nnx.merge(graphdef, params)
        preds = model_tmp(xs_embed)
        return jnp.mean((preds - ys) ** 2)
    return nnx.jit(nnx.value_and_grad(f, argnums=0))

# ---------- SSBroyden Training step ----------
def make_train_step_ssbroyden(graphdef):
    fg_oracle = make_fg_oracle(graphdef)
    
    @nnx.jit
    def train_step_ssbroyden(model, optimizer, xs_embed, ys):
        def cur_loss(m): 
            preds = m(xs_embed)
            return jnp.mean((preds - ys) ** 2)
        f_k, grads = nnx.value_and_grad(cur_loss, argnums=nnx.DiffState(0, nnx.Param))(model)
        optimizer.update(model, grads, f_k=f_k, fg_oracle=fg_oracle, loss_args=(xs_embed, ys))
        return f_k
    return train_step_ssbroyden

# ---------- Evaluate ----------
def evaluate(model, embed, xs, ys):
    preds = model(embed(xs))
    mse = jnp.mean((preds[:, 0] - ys) ** 2)
    l_inf = jnp.max(jnp.abs(preds[:, 0] - ys))
    l2re = jnp.linalg.norm(preds[:, 0] - ys)/jnp.linalg.norm(ys)
    return float(mse), float(l_inf), float(l2re)

# ---------- Runner for one config ----------
def run_one(width, depth, *, k, steps, batch_size, lr, train_n, test_n,
            embed_kind, M, seed, optimizer_type="adam"):
    # Embedder & dims
    embed = make_embedder(embed_kind, M)
    in_dim = infer_in_dim(embed_kind, M)

    # PRNGs
    key = jax.random.key(seed)
    key_model, key_train, key_test = jax.random.split(key, 3)

    # Build model
    model = TanhMLP(in_dim=in_dim, width=width, depth=depth, rngs=nnx.Rngs(key_model))
    
    # Setup optimizer and training step based on choice
    if optimizer_type == "adam":
        optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
        train_step = make_train_step()
    elif optimizer_type == "ssbroyden":
        tx = scale_by_ssbroyden_wolfe_oracle(lr=1.0, c1=1e-4, c2=0.9, max_ls=20)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        graphdef = nnx.graphdef(model)
        train_step = make_train_step_ssbroyden(graphdef)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Pre-make test set
    xs_test = jnp.linspace(-0.999999, 0.999999, test_n, dtype=DTYPE)
    ys_test = target_fn(xs_test, k)

    # Training loop
    k_loop = key_train
    for step in range(1, steps + 1):
        k_loop, k_batch = jax.random.split(k_loop)
        xs = sample_uniform(k_batch, batch_size)
        ys = target_fn(xs, k).reshape(-1, 1)
        xs_embed = embed(xs)

        loss = train_step(model, optimizer, xs_embed, ys)

        if step % max(steps // 10, 1) == 0:
            print(f"[w={width}, d={depth}, embed={embed_kind}, opt={optimizer_type}] step {step:5d}/{steps}  loss={float(loss):.6e}")

    # Final eval
    mse, linf, l2re = evaluate(model, embed, xs_test, ys_test)
    return mse, linf, l2re, model

# ---------- Grid comparison ----------
def compare_grid(args):
    # Parse embeddings list
    embeddings = [e.strip() for e in args.embed.split(",")]

    widths = [int(w) for w in args.widths.split(",")]
    depths = [int(d) for d in args.depths.split(",")]

    results = []

    print("\n=== Training & comparing models ===")
    for w in widths:
        for d in depths:
            print(f"\n--- Config: width={w}, depth={d} ---")

            config_results = []
            for embed_kind in embeddings:
                # Determine M value for this embedding
                M_val = args.M if embed_kind in ["cheb", "bary"] else 0

                mse, linf, l2re, _ = run_one(
                    w, d, k=args.k, steps=args.steps, batch_size=args.batch_size,
                    lr=args.lr, train_n=args.train_n, test_n=args.test_n,
                    embed_kind=embed_kind, M=M_val, seed=args.seed, optimizer_type=args.optimizer)

                config_results.append((embed_kind, mse, linf, l2re))

            results.append(((w, d), config_results))

    print("\n=== Summary (MSE / L_inf / L2_re on test grid) ===")
    for (w, d), embed_results in results:
        print(f"\nwidth={w:3d} depth={d:2d}:")
        for embed_kind, mse, linf, l2re in embed_results:
            print(f"  {embed_kind:>8s}: MSE={mse:.3e}  Linf={linf:.3e}  L2_re={l2re:.3e}")

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, default=2.0, help="Target is sin(k*x).")
    p.add_argument("--embed", type=str, default="theta",
               help=("Embedding(s) to use (comma-separated). "
                     "'none': plain x; "
                     "'theta': θ=arccos(x); "
                     "'cheb': [cos(θ),...,cos(Mθ)] (needs --M>0); "
                     "'bary': λ_j(x) over Chebyshev–Lobatto nodes of degree M (uses --M>0). "
                     "Example: --embed none,theta,bary"))

    p.add_argument("--M", type=int, default=0, help="Chebyshev feature count if --embed cheb.")
    p.add_argument("--widths", type=str, default="16, 32", help="Comma-separated widths.")
    p.add_argument("--depths", type=str, default="2,4,6", help="Comma-separated depths (hidden layers).")
    p.add_argument("--steps", type=int, default=7500)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train_n", type=int, default=512)
    p.add_argument("--test_n", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimizer", type=str, default="adam", 
                   choices=["adam", "ssbroyden"],
                   help="Optimizer to use: adam or ssbroyden.")
    args = p.parse_args()

    # Validate embeddings
    embeddings = [e.strip() for e in args.embed.split(",")]
    valid_embeddings = ["none", "theta", "cheb", "bary"]
    for embed in embeddings:
        if embed not in valid_embeddings:
            raise ValueError(f"Unknown embedding '{embed}'. Valid choices: {valid_embeddings}")

    # Check M requirement for cheb/bary embeddings
    if any(e in ["cheb", "bary"] for e in embeddings) and args.M <= 0:
        raise ValueError("For 'cheb' or 'bary' embeddings you must set --M > 0.")

    compare_grid(args)

if __name__ == "__main__":
    main()