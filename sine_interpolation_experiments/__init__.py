"""
Sine interpolation experiments package for benchmarking different architectures
on the sin(2πkx) interpolation problem.
"""

from .problem_utils import f_sin, generate_data, loss_fn, rel_l2_error, make_fg_oracle
from .mlp_trainer import MLP, create_mlp, train_mlp_ssbroyden, run_mlp_experiment, save_results

__all__ = [
    'f_sin',
    'generate_data',
    'loss_fn',
    'rel_l2_error',
    'make_fg_oracle',
    'MLP',
    'create_mlp',
    'train_mlp_ssbroyden',
    'run_mlp_experiment',
    'save_results'
]