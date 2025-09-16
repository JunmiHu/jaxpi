# SSBroyden-II as an Optax GradientTransformationExtraArgs
# --------------------------------------------------------
# Design follows Optax style (like scale_by_lbfgs): a factory function that returns
# a NamedTuple with init/update callables. Unlike LBFGS, SSB-II needs the stepsize
# (alpha) *inside* its inverse-H update, so we integrate a strong-Wolfe line search
# into the same transform. Branches are written with lax.cond / lax.while_loop so
# this is JIT-friendly.
#
# References (see Optax docs):
# - Optax separates direction builders (e.g., scale_by_lbfgs) from linesearch
#   (e.g., scale_by_zoom_linesearch). The LBFGS direction can be chained with a
#   linesearch that computes a stepsize \eta_k so w_{k+1} = w_k - \eta_k P_k u_k.
#   (Optax docs: transformations / optimizers / examples).  
# - Here, SSB-II update uses alpha explicitly, so we keep a monolithic transform.
#
# Usage (Linen):
#     tx = scale_by_ssbroyden_wolfe(loss_fn, lr=1.0)
#     opt_state = tx.init(params)
#     updates, opt_state = tx.update(grads, opt_state, params, extra_args=((x, y),))
#     params = optax.apply_updates(params, updates)
#
# Usage (NNX 0.11):
#     opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
#     opt.update(model, grads, extra_args=((x, y),))
#
from __future__ import annotations
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree


class GradientTransformationExtraArgs(NamedTuple):
    init: Callable[[Any], Any]
    update: Callable[..., Tuple[Any, Any]]


# ------------------------------
# Utilities: flatten / unflatten
# ------------------------------
class _Pack:
    def __init__(self, params):
        flat, unravel = ravel_pytree(params)
        self.unravel = unravel
        self.n = flat.size
    def ravel_params(self, params) -> jnp.ndarray:
        flat, _ = ravel_pytree(params)
        return flat
    def ravel_grads(self, grads) -> jnp.ndarray:
        flat, _ = ravel_pytree(grads)
        return flat
    def unravel_vec(self, vec):
        return self.unravel(vec)

# -------------------------------------------------------------------------
# Factory B: SSBroyden with strong-Wolfe using an external (f,g) ORACLE
# -------------------------------------------------------------------------

def scale_by_ssbroyden_wolfe_oracle(
    *,
    lr: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_ls: int = 20,
    init_scale: bool = True,
) -> GradientTransformationExtraArgs:
    """SSB-II with line search where **(f,g) evaluations are provided externally**.

    Update signature:
        updates, state = tx.update(grads, state, params,
                                   f_k=<scalar>,                # loss at params
                                   fg_oracle=<callable>,        # (params, *args)->(f, grads)
                                   loss_args=(...),             # tuple of runtime tensors
        )

    This satisfies the pattern you requested:
        v, g = jax.value_and_grad(loss_fn)(params, *loss_args)
        updates, state = tx.update(g, state, params, f_k=v, fg_oracle=fg, loss_args=loss_args)
    where fg = jax.value_and_grad(loss_fn) is created **outside** the optimizer.
    """

    def init_fn(params):
        leaves = jax.tree.leaves(params)
        p_dtype = leaves[0].dtype if len(leaves) else jnp.float64
        pack0 = _Pack(params)
        H0 = jnp.eye(pack0.n, dtype=p_dtype)
        return {"H": H0, "last_sqrt_arg": jnp.array(0.5, dtype=p_dtype), "did_scale": jnp.array(False)}

    def _oracle_flat(fg_oracle, pack: _Pack, x, loss_args, dtype):
        # Call user's oracle on a params-like PyTree; return (f, g_flat) in dtype
        params_x = pack.unravel_vec(x)
        f, gtree = fg_oracle(params_x, *loss_args)
        g_flat = pack.ravel_grads(gtree)
        return jnp.asarray(f, dtype), g_flat.astype(dtype)

    def _wolfe_oracle(xk, fk, gk, pk, gk_dot_pk, *, fg_oracle, pack, loss_args, lr, c1, c2, max_ls, alpha_min = 1e-32, alpha_max =1e6, shrink =0.5, grow = 1.1):
        dtype = xk.dtype
        alpha0   = jnp.asarray(lr, dtype)
        c1_      = jnp.asarray(c1, dtype)
        c2_      = jnp.asarray(c2, dtype)
        max_ls_  = jnp.asarray(max_ls, jnp.int32)
        amin_    = jnp.asarray(alpha_min, dtype)
        amax_    = jnp.asarray(alpha_max, dtype)
        shrink_  = jnp.asarray(shrink, dtype)
        grow_    = jnp.asarray(grow, dtype)

        # carry: (alpha, prev_f, have_prev, f_curr, g_curr, best_a, best_f, best_g, it, accepted, done)
        carry0 = (
            alpha0,
            jnp.array(jnp.inf, dtype),
            jnp.bool_(False),
            fk.astype(dtype), gk.astype(dtype),
            jnp.array(0.0, dtype), fk.astype(dtype), gk.astype(dtype),  # best is α=0 at start
            jnp.array(0, jnp.int32),
            jnp.bool_(False),
            jnp.bool_(False),
        )

        def body(carry):
            alpha, prev_f, have_prev, f_curr, g_curr, best_a, best_f, best_g, it, accepted, done = carry
            alpha = jnp.clip(alpha, amin_, amax_)
            x_new = xk + alpha * pk
            f_new, g_new = _oracle_flat(fg_oracle, pack, x_new, loss_args, dtype)

            # treat non-finite f as violation; don't update "best" with non-finite
            is_finite = jnp.isfinite(f_new)
            f_new = jnp.where(is_finite, f_new, jnp.array(jnp.inf, dtype))
            g_new = g_new  # if f is inf we’ll shrink; no need to sanitize g

            gpk = jnp.vdot(g_new, pk).astype(dtype)
            armijo_basic = f_new <= fk + c1_ * alpha * gk_dot_pk
            prev_violate = jnp.logical_and(have_prev, f_new >= prev_f)
            armijo_ok    = jnp.logical_and(armijo_basic, jnp.logical_not(prev_violate))
            curvature_ok = jnp.abs(gpk) <= c2_ * jnp.abs(gk_dot_pk)
            accept       = jnp.logical_and(armijo_ok, curvature_ok)

            # update best-so-far (finite only)
            better   = f_new < best_f
            best_a   = jnp.where(better, alpha, best_a)
            best_f   = jnp.where(better, f_new, best_f)
            best_g   = jnp.where(better, g_new, best_g)

            # step adjustment
            shrink_now = jnp.logical_or(jnp.logical_not(armijo_ok), prev_violate)
            alpha_half = alpha * shrink_
            alpha_up   = alpha * grow_
            alpha_next = jnp.where(
                accept, alpha,
                jnp.where(jnp.logical_or(shrink_now, gpk >= 0), alpha_half, alpha_up)
            )

            prev_f_next    = jnp.where(shrink_now, f_new, prev_f)
            have_prev_next = jnp.where(shrink_now, jnp.bool_(True), have_prev)

            it_next      = it + 1
            accepted_any = jnp.logical_or(accepted, accept)
            done_next    = jnp.logical_or(accept, (it_next >= max_ls_))

            # always carry last tried (f,g)
            return (alpha_next, prev_f_next, have_prev_next, f_new, g_new,
                    best_a, best_f, best_g,
                    it_next, accepted_any, done_next)

        def cond(carry):
            return jnp.logical_not(carry[-1])  # not done

        (alpha_last, _prev_f, _have_prev, f_last, g_last,
        best_a, best_f, best_g,
        _it, accepted_any, _done) = jax.lax.while_loop(cond, body, carry0)

        # If accepted, return the last tried (alpha_last, f_last, g_last).
        # Otherwise, fallback to the best finite f found (or α=0 with f=fk).
        alpha_final = jnp.where(accepted_any, alpha_last, best_a)
        f_final     = jnp.where(accepted_any, f_last,     best_f)
        g_final     = jnp.where(accepted_any, g_last,     best_g)
        return alpha_final, f_final, g_final

    def update_fn(grads, state, params, *, f_k, fg_oracle, loss_args=()):
        pack = _Pack(params)
        x_k = pack.ravel_params(params)
        dtype = x_k.dtype
        # g_k comes from caller (outside value_and_grad)
        g_k = pack.ravel_grads(grads).astype(dtype)
        f_k = jnp.asarray(f_k, dtype)

        # One-time H scaling
        def _maybe_scale_H(H, gk, did_scale_flag):
            def yes(_):
                gnorm = jnp.linalg.norm(gk)
                tau0 = jnp.maximum(gnorm, jnp.asarray(1e-12, dtype))
                return H / tau0, jnp.array(True)
            def no(_):
                return H, did_scale_flag
            return jax.lax.cond(jnp.logical_and(jnp.logical_not(did_scale_flag), jnp.array(init_scale)),
                                yes, no, operand=None)

        H = state["H"].astype(dtype)
        H, did_scale = _maybe_scale_H(H, g_k, state.get("did_scale", jnp.array(False)))

        # Direction and fallback
        pk_raw = -(H @ g_k)
        dot_pg = jnp.vdot(pk_raw, g_k).astype(dtype)
        pk = jax.lax.cond(dot_pg >= 0, lambda _: -g_k, lambda _: pk_raw, operand=None)
        gk_dot_pk = jnp.vdot(g_k, pk).astype(dtype)

        # Strong-Wolfe, using the caller-provided oracle for (f,g)
        alpha, f_new, g_new = _wolfe_oracle(
            x_k, f_k, g_k, pk, gk_dot_pk,
            fg_oracle=fg_oracle, pack=pack, loss_args=loss_args,
            lr=lr, c1=c1, c2=c2, max_ls=max_ls,
        )

        # Secant pair and SSB update
        x_new = x_k + alpha * pk
        s_k = x_new - x_k
        y_k = g_new - g_k
        rhok_inv = jnp.vdot(y_k, s_k).astype(dtype)
        eps = jnp.asarray(1e-32, dtype)
        small_curv = jnp.abs(rhok_inv) < eps

        n = x_k.shape[0]
        n_f = jnp.asarray(n, dtype)

        def do_update(_):
            rhok = 1.0 / rhok_inv
            Hkyk = H @ y_k
            ykHkyk = jnp.vdot(y_k, Hkyk).astype(dtype)
            h_k = ykHkyk * rhok
            b_k = -alpha * rhok * jnp.vdot(s_k, g_k).astype(dtype)
            a_k = b_k * h_k - dtype.type(1.0) 
            sqrt_arg = jnp.abs(a_k) / (dtype.type(1.0) + a_k)
            sqrt_arg_valid = jnp.logical_and(jnp.isfinite(sqrt_arg), sqrt_arg >= 0)
            use_arg = jnp.where(sqrt_arg_valid, sqrt_arg, state["last_sqrt_arg"].astype(dtype))
            rho_k_minus = jnp.minimum(dtype.type(1.0), h_k * (dtype.type(1.0) - jnp.sqrt(jnp.abs(use_arg))))
            new_last_ok = jnp.where(sqrt_arg_valid, use_arg, state["last_sqrt_arg"].astype(dtype))
            skip_small_rho = jnp.abs(rho_k_minus) < dtype.type(1e-16)
            theta_k_minus = (rho_k_minus - dtype.type(1.0)) / a_k
            theta_k_plus = dtype.type(1.0) / rho_k_minus
            theta_k = jnp.maximum(theta_k_minus, jnp.minimum(theta_k_plus, (dtype.type(1.0) - b_k) / b_k))
            rho_k_cap = jnp.minimum(dtype.type(1.0), dtype.type(1.0) / b_k)
            sigma_k = dtype.type(1.0) + theta_k * a_k
            power = dtype.type(1.0) / (dtype.type(1.0) - n_f)
            sigma_pow = jnp.abs(sigma_k) ** power
            tau_k = jnp.where(theta_k <= 0.0,
                              jnp.minimum(rho_k_cap * sigma_pow, sigma_k),
                              rho_k_cap * jnp.minimum(sigma_pow, dtype.type(1.0) / theta_k))
            v_k = rhok * s_k - Hkyk / ykHkyk
            phi_k = (dtype.type(1.0) - theta_k) / (dtype.type(1.0) + a_k * theta_k)
            H_cand = (H - jnp.outer(Hkyk, Hkyk) / ykHkyk + phi_k * ykHkyk * jnp.outer(v_k, v_k)) / tau_k \
                     + rhok * jnp.outer(s_k, s_k)
            H_new = jnp.where(skip_small_rho, H, H_cand)
            return H_new, new_last_ok

        H_new, last_ok = jax.lax.cond(small_curv,
                                      lambda _: (H, state["last_sqrt_arg"].astype(dtype)),
                                      do_update,
                                      operand=None)

        updates = pack.unravel_vec(x_new - x_k)
        new_state = {"H": H_new, "last_sqrt_arg": last_ok, "did_scale": did_scale}
        return updates, new_state

    return GradientTransformationExtraArgs(init_fn, update_fn)


# ------------------------------
# Minimal demo (if run directly)
# ------------------------------
if __name__ == "__main__":
    import jax.random as jr
    from flax import linen as nn

    jax.config.update("jax_enable_x64", True)

    class MLP(nn.Module):
        width: int
        depth: int
        @nn.compact
        def __call__(self, x):
            z = x.reshape(-1, 1)
            for _ in range(self.depth - 1):
                z = nn.tanh(nn.Dense(self.width)(z))
            return nn.Dense(1)(z).squeeze(-1)

    def make_data(k: int = 8, n: int = 257):
        xs = jnp.linspace(0.0, 1.0, n)
        ys = jnp.sin(2.0 * jnp.pi * k * xs)
        return xs, ys

    x, y = make_data()
    model = MLP(64, 3)
    params = model.init(jr.PRNGKey(0), x)

    def loss_fn(p, x_, y_):
        pred = model.apply(p, x_)
        return jnp.mean((pred - y_) ** 2)

    tx = scale_by_ssbroyden_wolfe(loss_fn, lr=1.0, init_scale=True)
    state = tx.init(params)

    @jax.jit
    def step(p, st):
        # grads are ignored by the transform; we pass zeros with the right structure
        dummy = jax.tree.map(jnp.zeros_like, p)
        updates, st = tx.update(dummy, st, p, extra_args=((x, y),))
        p = optax.apply_updates(p, updates)
        return p, st

    for _ in range(10):
        params, state = step(params, state)
    print("OK")

    print("testing oracle implementation")
    tx = scale_by_ssbroyden_wolfe_oracle(lr=1.0, c1=1e-4, c2=0.9, max_ls=20, init_scale=True)
    opt_state = tx.init(params)
    fg_oracle = jax.value_and_grad(loss_fn)

    @jax.jit
    def train_step(p, st):
        v, g = fg_oracle(p, x, y)
        updates, st= tx.update(
            g, st, params,
            f_k = v,
            fg_oracle = fg_oracle,
            loss_args = (x,y),
        )
        p = optax.apply_updates(p, updates)
        return v, p, st

    for i in range(1000):
        v, params, opt_state = train_step(params, opt_state)
        print(f"step {i} loss={float(v):.3e}")
    print("OK")
