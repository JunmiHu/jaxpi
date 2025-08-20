from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Dict

from flax.training import train_state
from flax import jax_utils

import jax.numpy as jnp
from jax import lax, jit, grad, pmap, random, jacfwd, jacrev
from jax.tree_util import tree_map, tree_reduce, tree_leaves

import optax

from jaxpi import archs
from jaxpi.utils import flatten_pytree

from soap_jax import soap  # Install from https://github.com/haydn-jones/SOAP_JAX
from psgd_jax.kron import kron

from jaxpi.ssbroyden_optax import scale_by_ssbroyden_wolfe_oracle


class TrainState(train_state.TrainState):
    weights: Dict
    momentum: float
    is_ssbroyden: bool = False
    loss_fn: Optional[Callable] = None

    def apply_weights(self, weights, **kwargs):
        """Updates `weights` using running average  in return value.

        Returns:
          An updated instance of `self` with new weights updated by applying `running_average`,
          and additional attributes replaced as specified by `kwargs`.
        """

        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        weights = tree_map(running_average, self.weights, weights)
        weights = lax.stop_gradient(weights)

        return self.replace(
            step=self.step,
            params=self.params,
            opt_state=self.opt_state,
            weights=weights,
            **kwargs,
        )
    
    def apply_gradients(self, *, grads, **kwargs):
        """Apply gradients with support for extra args needed by advanced optimizers."""
        return super().apply_gradients(grads=grads, **kwargs)
        
    def apply_gradients_ssbroyden(self, *, grads, **kwargs):
        batch = kwargs['batch']
        loss_fn = kwargs['loss_fn']
        loss_args = kwargs.get('loss_args', ())
        
        # Compute current loss value
        f_k = loss_fn(self.params, self.weights, batch, *loss_args)
        
        # Create fg_oracle that matches SSBroyden's expected signature
        def fg_oracle(params, weights, batch_data, *extra_args):
            f = loss_fn(params, weights, batch_data, *extra_args)
            g = grad(loss_fn)(params, weights, batch_data, *extra_args)
            return f, g
        
        # Use SSBroyden update
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params,
            f_k=f_k,
            fg_oracle=fg_oracle,
            loss_args=(self.weights, batch, *loss_args)
        )
        
        new_params = optax.apply_updates(self.params, updates)
        
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **{k: v for k, v in kwargs.items() if k not in ['batch', 'loss_fn', 'loss_args']},
        )


def _create_arch(config):
    if config.arch_name == "Mlp":
        arch = archs.Mlp(**config)

    elif config.arch_name == "ResNet":
        arch = archs.ResNet(**config)

    elif config.arch_name == "ModifiedMlp":
        arch = archs.ModifiedMlp(**config)

    elif config.arch_name == "PIResNet":
        arch = archs.PIResNet(**config)

    elif config.arch_name == "PirateNet":
        arch = archs.PirateNet(**config)

    elif config.arch_name == "DeepONet":
        arch = archs.DeepONet(**config)

    else:
        raise NotImplementedError(f"Arch {config.arch_name} not supported yet!")

    return arch


def _create_optimizer(config):

    lr = optax.exponential_decay(
        init_value=config.learning_rate,
        transition_steps=config.decay_steps,
        decay_rate=config.decay_rate,
        staircase=config.staircase
        )

    if config.warmup_steps > 0:
        warmup = optax.linear_schedule(init_value=0.0, end_value=config.learning_rate,
                                       transition_steps=config.warmup_steps)

        lr = optax.join_schedules([warmup, lr], [config.warmup_steps])

    if config.optimizer == "Adam":
        tx = optax.adam(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps
        )

    elif config.optimizer == "Soap":
        print("Using SOAP optimizer")

        tx = soap(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, weight_decay=0.0, precondition_frequency=2
            )

    elif config.optimizer == "Muon":
        tx = optax.contrib.muon(
            learning_rate=lr,
        )

    elif config.optimizer == "Lamb":
        tx = optax.lamb(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps
        )

    elif config.optimizer == "Adagrad":
        tx = optax.adagrad(
            learning_rate=lr, eps=config.eps
        )

    elif config.optimizer == "RMSProp":
        tx = optax.rmsprop(
            learning_rate=lr
        )
    
    elif config.optimizer == "SSBroyden":
        print("Using SSBroyden optimizer")
        tx = scale_by_ssbroyden_wolfe_oracle(
            lr=config.learning_rate,  # SSBroyden uses base lr, not schedule
            c1=getattr(config, 'c1', 1e-4),
            c2=getattr(config, 'c2', 0.9),
            max_ls=getattr(config, 'max_ls', 20),
            init_scale=getattr(config, 'init_scale', True)
        )

    # Gradient accumulation (not compatible with SSBroyden)
    if config.grad_accum_steps > 1 and config.optimizer != "SSBroyden":
        tx = optax.MultiSteps(tx, every_k_schedule=config.grad_accum_steps)

    return lr, tx


def _create_train_state(config, params=None, weights=None):
    # Initialize network
    arch = _create_arch(config.arch)
    x = jnp.ones(config.input_dim)

    # Initialize optax optimizer
    lr, tx = _create_optimizer(config.optim)

    if params is None:
        params = arch.init(random.PRNGKey(config.seed), x)

    if weights is None:
        weights = dict(config.weighting.init_weights)

    is_ssbroyden = config.optim.optimizer == "SSBroyden"
    
    state = TrainState.create(
        apply_fn=arch.apply,
        params=params,
        tx=tx,
        weights=weights,
        momentum=config.weighting.momentum,
        is_ssbroyden=is_ssbroyden,
        loss_fn=None,  # Will be set by the PINN instance
    )

    return jax_utils.replicate(state)


class PINN:
    def __init__(self, config):
        self.config = config
        self.state = _create_train_state(config)

    def u_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def r_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def losses(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def compute_diag_ntk(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, *args):
        # Compute losses
        losses = self.losses(params, batch, *args)
        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss

    @partial(jit, static_argnums=(0,))
    def compute_weights(self, params, batch, *args):
        if self.config.weighting.scheme == "grad_norm":
            # Compute the gradient of each loss w.r.t. the parameters
            grads = jacrev(self.losses)(params, batch, *args)

            # Compute the grad norm of each loss
            grad_norm_dict = {}
            for key, value in grads.items():
                flattened_grad = flatten_pytree(value)
                grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)

            # Compute the mean of grad norms over all losses
            mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))
            # Grad Norm Weighting
            w = tree_map(
                lambda x: (mean_grad_norm / (x + 1e-5 * mean_grad_norm)), grad_norm_dict
            )

        elif self.config.weighting.scheme == "ntk":
            # Compute the diagonal of the NTK of each loss
            ntk = self.compute_diag_ntk(params, batch, *args)

            # Compute the mean of the diagonal NTK corresponding to each loss
            mean_ntk_dict = tree_map(lambda x: jnp.mean(x), ntk)

            # Compute the average over all ntk means
            mean_ntk = jnp.mean(jnp.stack(tree_leaves(mean_ntk_dict)))
            # NTK Weighting
            w = tree_map(lambda x: (mean_ntk / (x + 1e-5 * mean_ntk)), mean_ntk_dict)

        return w

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def update_weights(self, state, batch, *args):
        weights = self.compute_weights(state.params, batch, *args)
        weights = lax.pmean(weights, "batch")
        state = state.apply_weights(weights=weights)
        return state

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,3))
    def step(self, state, batch, is_ssbroyden:bool, *args):
        grads = grad(self.loss)(state.params, state.weights, batch, *args)
        grads = lax.pmean(grads, "batch")
        
        # Pass additional kwargs for SSBroyden optimizer
        apply_kwargs = {"grads": grads}
        if is_ssbroyden:
            apply_kwargs.update({
                "batch": batch,
                "loss_fn": self.loss,
                "loss_args": args
            })
            state = state.apply_gradients_ssbroyden(**apply_kwargs)
        else:
            state = state.apply_gradients(**apply_kwargs)
        return state


class ForwardIVP(PINN):
    def __init__(self, config):
        super().__init__(config)

        if config.weighting.use_causal:
            self.tol = config.weighting.causal_tol
            self.num_chunks = config.weighting.num_chunks
            self.M = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1).T


class ForwardBVP(PINN):
    def __init__(self, config):
        super().__init__(config)
