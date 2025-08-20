import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Wave"
    wandb.name = "pirate_ssbroyden"
    wandb.tag = None

    # Physics-informed initialization
    config.use_pi_init = True

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 3
    arch.hidden_dim = 32
    arch.out_dim = 1
    arch.activation = "tanh"
    # arch.periodicity = ml_collections.ConfigDict(
    #     {"period": (2 * jnp.pi,), "axis": (1,), "trainable": (False,)})
    arch.nonlinearity = 0.0
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 10.0, "embed_dim": 32})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )
    arch.pi_init = None

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "SSBroyden"
    optim.learning_rate = 1.0  # SSBroyden base learning rate
    optim.c1 = 1e-4  # Armijo condition parameter
    optim.c2 = 0.9   # Curvature condition parameter
    optim.max_ls = 20  # Maximum line search iterations
    optim.init_scale = True  # Scale initial Hessian approximation
    optim.grad_accum_steps = 0  # Not compatible with SSBroyden
    # The following are required by _create_optimizer but unused by SSBroyden
    optim.decay_rate = 0.9      # Unused - SSBroyden uses line search for step size
    optim.decay_steps = 2000    # Unused - SSBroyden uses line search for step size  
    optim.staircase = False     # Unused - SSBroyden uses line search for step size
    optim.warmup_steps = 0      # Unused - SSBroyden uses line search for step size

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 10000  # Reduced for SSBroyden testing
    training.batch_size_per_device = 8192

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({"u0": 1.0, "u_t0": 1.0,  "res": 1.0, "bcs": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = True
    weighting.causal_tol = 1e-2
    weighting.num_chunks = 16

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_nonlinearities = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 10

    # # Input shape for initializing Flax models
    config.input_dim = 2

    # Integer for PRNG random seed.
    config.seed = 42

    return config