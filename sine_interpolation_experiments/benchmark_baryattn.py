#!/usr/bin/env python3
"""
Benchmark script for BarycentricAttention architectures on sin(2πkx) interpolation problem.

This script runs ablation studies across different BarycentricAttention configurations,
including different N values (number of nodes), rational representations, and k values,
saving the results for analysis.
"""

import os
import argparse
import itertools
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from barycentric_attn_trainer import run_barycentric_attention_experiment, save_results


def run_ablation_study(
    Ns: List[int],
    ks: List[int],
    rational_representations: List[str] = None,
    steps: int = 50000,
    n_train: int = 4097,
    n_eval: int = 10000,
    seeds: List[int] = None,
    use_exact_init: bool = True,
    kernel_init_strategies: List[str] = None,
    kernel_init_scales: List[float] = None,
    input_dependent_values_options: List[bool] = None,
    learn_nodes_options: List[bool] = None,
    learn_query_options: List[bool] = None,
    query_hdims: List[int] = None,
    mlp_hdims: List[int] = None,
    output_dir: str = "results",
    **optimizer_kwargs
):
    """
    Run a complete ablation study across BarycentricAttention configurations.

    Args:
        Ns: List of N values (number of Chebyshev-Lobatto nodes = N+1)
        ks: List of k values for sin(2πkx)
        rational_representations: List of weight representations ("standard", "linear", "kernel")
        steps: Number of training steps
        n_train: Number of training points
        n_eval: Number of evaluation points
        seeds: List of random seeds for multiple runs (default: [0])
        use_exact_init: Whether to use exact function values for initialization
        kernel_init_strategies: List of kernel initialization strategies (for "kernel" representation)
        kernel_init_scales: List of kernel initialization scales (for "kernel" representation)
        input_dependent_values_options: List of booleans for input-dependent values
        learn_nodes_options: List of booleans for learnable nodes
        learn_query_options: List of booleans for query learning
        query_hdims: List of query hidden dimensions (for query learning)
        mlp_hdims: List of MLP hidden dimensions (0 means no MLP)
        output_dir: Directory to save results
        **optimizer_kwargs: Additional optimizer parameters
    """
    # Set defaults
    if seeds is None:
        seeds = [0]
    if rational_representations is None:
        rational_representations = ["standard"]
    if kernel_init_strategies is None:
        kernel_init_strategies = ["normal"]
    if kernel_init_scales is None:
        kernel_init_scales = [1e-3]
    if input_dependent_values_options is None:
        input_dependent_values_options = [False]
    if learn_nodes_options is None:
        learn_nodes_options = [False]
    if learn_query_options is None:
        learn_query_options = [False]
    if query_hdims is None:
        query_hdims = [8]
    if mlp_hdims is None:
        mlp_hdims = [0]  # 0 means no MLP

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"baryattn_ablation_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    # Save configuration
    config = {
        'Ns': Ns,
        'ks': ks,
        'rational_representations': rational_representations,
        'steps': steps,
        'n_train': n_train,
        'n_eval': n_eval,
        'seeds': seeds,
        'use_exact_init': use_exact_init,
        'kernel_init_strategies': kernel_init_strategies,
        'kernel_init_scales': kernel_init_scales,
        'input_dependent_values_options': input_dependent_values_options,
        'learn_nodes_options': learn_nodes_options,
        'learn_query_options': learn_query_options,
        'query_hdims': query_hdims,
        'mlp_hdims': mlp_hdims,
        'optimizer_kwargs': optimizer_kwargs,
        'timestamp': timestamp
    }

    config_path = run_dir / "config.json"
    save_results(config, str(config_path))
    print(f"Saved configuration to {config_path}")

    # Generate all combinations based on rational representation
    combinations = []
    for N, k, seed, rational_rep, input_dep_vals, learn_nodes, learn_query, mlp_hdim in itertools.product(
        Ns, ks, seeds, rational_representations, input_dependent_values_options,
        learn_nodes_options, learn_query_options, mlp_hdims
    ):
        if rational_rep == "kernel":
            # For kernel representation, include kernel-specific parameters
            for kernel_strategy, kernel_scale in itertools.product(kernel_init_strategies, kernel_init_scales):
                if learn_query:
                    # For query learning, include query_hdim variations
                    for query_hdim in query_hdims:
                        combinations.append((N, k, seed, rational_rep, input_dep_vals, learn_nodes,
                                          learn_query, kernel_strategy, kernel_scale, query_hdim, mlp_hdim))
                else:
                    combinations.append((N, k, seed, rational_rep, input_dep_vals, learn_nodes,
                                      learn_query, kernel_strategy, kernel_scale, query_hdims[0], mlp_hdim))
        else:
            # For standard/linear representations, use default kernel parameters
            if learn_query:
                # For query learning, include query_hdim variations
                for query_hdim in query_hdims:
                    combinations.append((N, k, seed, rational_rep, input_dep_vals, learn_nodes,
                                      learn_query, kernel_init_strategies[0], kernel_init_scales[0], query_hdim, mlp_hdim))
            else:
                combinations.append((N, k, seed, rational_rep, input_dep_vals, learn_nodes,
                                  learn_query, kernel_init_strategies[0], kernel_init_scales[0], query_hdims[0], mlp_hdim))

    total_experiments = len(combinations)
    print(f"Running {total_experiments} experiments...")

    experiment_count = 0
    for combination in combinations:
        N, k, seed, rational_rep, input_dep_vals, learn_nodes, learn_query, kernel_strategy, kernel_scale, query_hdim, mlp_hdim = combination
        experiment_count += 1
        print(f"\n--- Experiment {experiment_count}/{total_experiments} ---")
        print(f"N: {N}, k: {k}, Seed: {seed}")
        print(f"Rational representation: {rational_rep}")
        print(f"Input dependent values: {input_dep_vals}, Learn nodes: {learn_nodes}, Learn query: {learn_query}")
        if rational_rep == "kernel":
            print(f"Kernel strategy: {kernel_strategy}, Kernel scale: {kernel_scale}")
        if learn_query:
            print(f"Query hdim: {query_hdim}")
        if mlp_hdim > 0:
            print(f"MLP hdim: {mlp_hdim}")

        try:
            # Run experiment
            results = run_barycentric_attention_experiment(
                N=N,
                k=k,
                steps=steps,
                n_train=n_train,
                n_eval=n_eval,
                seed=seed,
                use_exact_init=use_exact_init,
                rational_representation=rational_rep,
                kernel_init_strategy=kernel_strategy,
                kernel_init_scale=kernel_scale,
                input_dependent_values=input_dep_vals,
                learn_nodes=learn_nodes,
                learn_query=learn_query,
                query_hdim=query_hdim,
                mlp_hdim=mlp_hdim,
                **optimizer_kwargs
            )

            # Save results with descriptive filename
            filename_parts = [f"baryattn_N{N}_k{k}_s{seed}_{rational_rep}"]
            if input_dep_vals:
                filename_parts.append("inputdep")
            if learn_nodes:
                filename_parts.append("learnnodes")
            if learn_query:
                filename_parts.append(f"learnquery{query_hdim}")
            if mlp_hdim > 0:
                filename_parts.append(f"mlp{mlp_hdim}")
            if rational_rep == "kernel":
                filename_parts.append(f"{kernel_strategy}{kernel_scale}")

            filename = "_".join(filename_parts) + ".json"
            filepath = run_dir / filename
            save_results(results, str(filepath))

            final_error = results['training']['final_rel_l2']
            print(f"Final rel L2 error: {final_error:.6e}")
            print(f"Parameter count: {results['config']['param_count']}")
            print(f"Results saved to {filepath}")

        except Exception as e:
            print(f"Error in experiment: {e}")
            # Save error info
            error_info = {
                'config': {
                    'N': N, 'k': k, 'seed': seed, 'rational_representation': rational_rep,
                    'input_dependent_values': input_dep_vals, 'learn_nodes': learn_nodes,
                    'learn_query': learn_query, 'query_hdim': query_hdim, 'mlp_hdim': mlp_hdim,
                    'kernel_init_strategy': kernel_strategy, 'kernel_init_scale': kernel_scale
                },
                'error': str(e),
                'experiment_count': experiment_count
            }
            error_filename = f"error_N{N}_k{k}_s{seed}_{rational_rep}.json"
            error_filepath = run_dir / error_filename
            save_results(error_info, str(error_filepath))
            print(f"Error info saved to {error_filepath}")

    print(f"\nCompleted ablation study. Results saved in {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark BarycentricAttention architectures on sin(2πkx) interpolation")

    # Architecture parameters
    parser.add_argument("--Ns", nargs="+", type=int, default=[16, 32, 64, 128],
                        help="N values (number of Chebyshev-Lobatto nodes = N+1)")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32],
                        help="k values for sin(2πkx)")
    parser.add_argument("--rational-representations", nargs="+", type=str,
                        default=["standard", "linear"],
                        choices=["standard", "linear", "kernel"],
                        help="Rational weight representations to test")

    # Training parameters
    parser.add_argument("--steps", type=int, default=50000,
                        help="Number of training steps")
    parser.add_argument("--n-train", type=int, default=2048,
                        help="Number of training points")
    parser.add_argument("--n-eval", type=int, default=10000,
                        help="Number of evaluation points")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5],
                        help="Random seeds for multiple runs")

    # Model configuration
    parser.add_argument("--use-exact-init", action="store_true", default=True,
                        help="Use exact function values for initialization")
    parser.add_argument("--no-exact-init", dest="use_exact_init", action="store_false",
                        help="Use random initialization instead of exact values")
    parser.add_argument("--kernel-init-strategies", nargs="+", type=str,
                        default=["zeros"],
                        choices=["zeros", "normal", "uniform", "xavier", "he", "orthogonal", "identity"],
                        help="Kernel initialization strategies (for kernel representation)")
    parser.add_argument("--kernel-init-scales", nargs="+", type=float,
                        default=[1e-4],
                        help="Kernel initialization scales (for kernel representation)")
    parser.add_argument("--input-dependent-values", action="store_true", default=True,
                        help="Test input-dependent values option")
    parser.add_argument("--learn-nodes", action="store_true",
                        help="Test learnable nodes option")
    parser.add_argument("--learn-query", action="store_true",
                        help="Test query learning option")
    parser.add_argument("--query-hdims", nargs="+", type=int, default=[4, 8, 16],
                        help="Query hidden dimensions (for query learning)")
    parser.add_argument("--mlp-hdims", nargs="+", type=int, default=[0, 8, 16, 32],
                        help="MLP hidden dimensions (0 means no MLP)")

    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=1.0,
                        help="Learning rate")
    parser.add_argument("--c1", type=float, default=1e-4,
                        help="Wolfe condition c1")
    parser.add_argument("--c2", type=float, default=0.9,
                        help="Wolfe condition c2")
    parser.add_argument("--max-ls", type=int, default=20,
                        help="Maximum line search iterations")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Logging interval")

    # Output
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")

    # Quick test mode
    parser.add_argument("--quick-test", action="store_true",
                        help="Run a quick test with minimal configuration")

    args = parser.parse_args()

    # Convert boolean flags to lists for ablation
    input_dependent_values_options = [args.input_dependent_values] if hasattr(args, 'input_dependent_values') else [False]
    learn_nodes_options = [args.learn_nodes] if hasattr(args, 'learn_nodes') else [False]
    learn_query_options = [args.learn_query] if hasattr(args, 'learn_query') else [False]

    if args.quick_test:
        print("Running quick test...")
        run_ablation_study(
            Ns=[32],
            ks=[1],
            rational_representations=["linear"],
            steps=1000,
            n_train=100,
            n_eval=500,
            seeds=[0],
            use_exact_init=args.use_exact_init,
            kernel_init_strategies=["normal"],
            kernel_init_scales=[1e-3],
            input_dependent_values_options=[True],
            learn_nodes_options=[False],
            learn_query_options=[False],
            query_hdims=[0],
            mlp_hdims=[8],
            output_dir=args.output_dir,
            lr=args.lr,
            c1=args.c1,
            c2=args.c2,
            max_ls=args.max_ls,
            log_interval=args.log_interval
        )
    else:
        run_ablation_study(
            Ns=args.Ns,
            ks=args.ks,
            rational_representations=args.rational_representations,
            steps=args.steps,
            n_train=args.n_train,
            n_eval=args.n_eval,
            seeds=args.seeds,
            use_exact_init=args.use_exact_init,
            kernel_init_strategies=args.kernel_init_strategies,
            kernel_init_scales=args.kernel_init_scales,
            input_dependent_values_options=input_dependent_values_options,
            learn_nodes_options=learn_nodes_options,
            learn_query_options=learn_query_options,
            query_hdims=args.query_hdims,
            mlp_hdims=args.mlp_hdims,
            output_dir=args.output_dir,
            lr=args.lr,
            c1=args.c1,
            c2=args.c2,
            max_ls=args.max_ls,
            log_interval=args.log_interval
        )


if __name__ == "__main__":
    main()