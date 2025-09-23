#!/usr/bin/env python3
"""
Benchmark script for MLP architectures on sin(2πkx) interpolation problem.

This script runs ablation studies across different MLP widths, depths, and k values,
saving the results for analysis.
"""

import os
import argparse
import itertools
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from mlp_trainer import run_mlp_experiment, save_results


def run_ablation_study(
    widths: List[int],
    depths: List[int],
    ks: List[int],
    steps: int = 50000,
    n_train: int = 4097,
    n_eval: int = 10000,
    seeds: List[int] = None,
    output_dir: str = "results",
    **optimizer_kwargs
):
    """
    Run a complete ablation study across widths, depths, and k values.

    Args:
        widths: List of MLP widths to test
        depths: List of MLP depths to test
        ks: List of k values for sin(2πkx)
        steps: Number of training steps
        n_train: Number of training points
        n_eval: Number of evaluation points
        seeds: List of random seeds for multiple runs (default: [0])
        output_dir: Directory to save results
        **optimizer_kwargs: Additional optimizer parameters
    """
    if seeds is None:
        seeds = [0]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"mlp_ablation_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    # Save configuration
    config = {
        'widths': widths,
        'depths': depths,
        'ks': ks,
        'steps': steps,
        'n_train': n_train,
        'n_eval': n_eval,
        'seeds': seeds,
        'optimizer_kwargs': optimizer_kwargs,
        'timestamp': timestamp
    }

    config_path = run_dir / "config.json"
    save_results(config, str(config_path))
    print(f"Saved configuration to {config_path}")

    # Generate all combinations
    total_experiments = len(widths) * len(depths) * len(ks) * len(seeds)
    print(f"Running {total_experiments} experiments...")

    experiment_count = 0
    for width, depth, k, seed in itertools.product(widths, depths, ks, seeds):
        experiment_count += 1
        print(f"\n--- Experiment {experiment_count}/{total_experiments} ---")
        print(f"Width: {width}, Depth: {depth}, k: {k}, Seed: {seed}")

        try:
            # Run experiment
            results = run_mlp_experiment(
                width=width,
                depth=depth,
                k=k,
                steps=steps,
                n_train=n_train,
                n_eval=n_eval,
                seed=seed,
                **optimizer_kwargs
            )

            # Save results
            filename = f"mlp_w{width}_d{depth}_k{k}_s{seed}.json"
            filepath = run_dir / filename
            save_results(results, str(filepath))

            final_error = results['training']['final_rel_l2']
            print(f"Final rel L2 error: {final_error:.6e}")
            print(f"Results saved to {filepath}")

        except Exception as e:
            print(f"Error in experiment: {e}")
            # Save error info
            error_info = {
                'config': {'width': width, 'depth': depth, 'k': k, 'seed': seed},
                'error': str(e),
                'experiment_count': experiment_count
            }
            error_filename = f"error_w{width}_d{depth}_k{k}_s{seed}.json"
            error_filepath = run_dir / error_filename
            save_results(error_info, str(error_filepath))
            print(f"Error info saved to {error_filepath}")

    print(f"\nCompleted ablation study. Results saved in {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLP architectures on sin(2πkx) interpolation")

    # Architecture parameters
    parser.add_argument("--widths", nargs="+", type=int, default=[16, 32, 64, 128],
                        help="MLP widths to test")
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="MLP depths to test")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32],
                        help="k values for sin(2πkx)")

    # Training parameters
    parser.add_argument("--steps", type=int, default=50000,
                        help="Number of training steps")
    parser.add_argument("--n-train", type=int, default=2048,
                        help="Number of training points")
    parser.add_argument("--n-eval", type=int, default=10000,
                        help="Number of evaluation points")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5],
                        help="Random seeds for multiple runs")

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

    if args.quick_test:
        print("Running quick test...")
        run_ablation_study(
            widths=[32],
            depths=[1],
            ks=[1],
            steps=1000,
            n_train=100,
            n_eval=500,
            seeds=[0],
            output_dir=args.output_dir,
            lr=args.lr,
            c1=args.c1,
            c2=args.c2,
            max_ls=args.max_ls,
            log_interval=args.log_interval
        )
    else:
        run_ablation_study(
            widths=args.widths,
            depths=args.depths,
            ks=args.ks,
            steps=args.steps,
            n_train=args.n_train,
            n_eval=args.n_eval,
            seeds=args.seeds,
            output_dir=args.output_dir,
            lr=args.lr,
            c1=args.c1,
            c2=args.c2,
            max_ls=args.max_ls,
            log_interval=args.log_interval
        )


if __name__ == "__main__":
    main()