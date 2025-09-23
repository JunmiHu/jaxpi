#!/usr/bin/env python3
"""
Analysis script for MLP benchmark results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np
from typing import List, Dict, Any


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all experiment results from a directory into a DataFrame."""
    results_path = Path(results_dir)
    data = []

    for json_file in results_path.glob("mlp_w*_d*_k*_s*.json"):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)

            # Extract config and training results
            config = result['config']
            training = result['training']

            row = {
                'width': config['width'],
                'depth': config['depth'],
                'k': config['k'],
                'seed': config['seed'],
                'steps': config['steps'],
                'final_rel_l2': training['final_rel_l2'],
                'initial_rel_l2': training['initial_rel_l2'],
                'final_loss': training['final_loss'],
                'n_train': config['n_train'],
                'n_eval': config['n_eval'],
                'param_count': config.get('param_count', None)  # Handle old results without param_count
            }
            data.append(row)

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    if not data:
        raise ValueError(f"No valid results found in {results_dir}")

    return pd.DataFrame(data)


def plot_results(df: pd.DataFrame, output_dir: str = None):
    """Create various plots analyzing the results."""
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Heatmap of final rel L2 error by width and depth (averaged over seeds and k)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Average over seeds for each k
    df_avg = df.groupby(['width', 'depth', 'k'])['final_rel_l2'].mean().reset_index()

    # Plot for different k values
    k_values = sorted(df['k'].unique())
    for i, k in enumerate(k_values[:2]):  # Show first 2 k values
        subset = df_avg[df_avg['k'] == k]
        pivot = subset.pivot(index='depth', columns='width', values='final_rel_l2')

        sns.heatmap(pivot, annot=True, fmt='.2e', cmap='viridis_r',
                   ax=axes[i], cbar_kws={'label': 'Final Rel L2 Error'})
        axes[i].set_title(f'Final Rel L2 Error (k={k})')
        axes[i].set_xlabel('Width')
        axes[i].set_ylabel('Depth')

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_path / 'heatmap_rel_l2.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Line plot: Effect of width for different depths
    fig, ax = plt.subplots(figsize=(10, 6))
    depths = sorted(df['depth'].unique())

    for depth in depths:
        subset = df[df['depth'] == depth].groupby('width')['final_rel_l2'].mean()
        ax.plot(subset.index, subset.values, marker='o', label=f'Depth {depth}')

    ax.set_xlabel('Width')
    ax.set_ylabel('Final Rel L2 Error')
    ax.set_yscale('log')
    ax.set_title('Effect of Width on Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_dir:
        plt.savefig(output_path / 'width_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Line plot: Effect of k value
    fig, ax = plt.subplots(figsize=(10, 6))

    # Average over seeds and architectures
    k_effect = df.groupby('k')['final_rel_l2'].agg(['mean', 'std']).reset_index()

    ax.errorbar(k_effect['k'], k_effect['mean'], yerr=k_effect['std'],
               marker='o', capsize=5, capthick=2)
    ax.set_xlabel('k (frequency)')
    ax.set_ylabel('Final Rel L2 Error')
    ax.set_yscale('log')
    ax.set_title('Effect of Frequency k on Performance')
    ax.grid(True, alpha=0.3)

    if output_dir:
        plt.savefig(output_path / 'frequency_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Box plot: Distribution of errors by architecture size
    df['total_params'] = df['width'] * (df['depth'] + 1) + df['width'] + 1  # Approximate parameter count
    df['arch_size'] = pd.cut(df['total_params'], bins=5, labels=['XS', 'S', 'M', 'L', 'XL'])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='arch_size', y='final_rel_l2', ax=ax)
    ax.set_yscale('log')
    ax.set_xlabel('Architecture Size')
    ax.set_ylabel('Final Rel L2 Error')
    ax.set_title('Performance Distribution by Architecture Size')

    if output_dir:
        plt.savefig(output_path / 'architecture_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("=== Experiment Summary ===")
    print(f"Total experiments: {len(df)}")
    print(f"Unique architectures: {len(df.groupby(['width', 'depth']))}")
    print(f"k values tested: {sorted(df['k'].unique())}")
    print(f"Seeds used: {sorted(df['seed'].unique())}")

    print("\n=== Best Results ===")
    best_overall = df.loc[df['final_rel_l2'].idxmin()]
    print(f"Best overall: Width={best_overall['width']}, Depth={best_overall['depth']}, "
          f"k={best_overall['k']}, Rel L2={best_overall['final_rel_l2']:.6e}")

    print("\n=== Best by k value ===")
    for k in sorted(df['k'].unique()):
        subset = df[df['k'] == k]
        best_k = subset.loc[subset['final_rel_l2'].idxmin()]
        avg_error = subset['final_rel_l2'].mean()
        param_count = best_k.get('param_count', 'N/A')
        print(f"k={k}: Best Width={best_k['width']}, Depth={best_k['depth']}, "
              f"Params={param_count}, Rel L2={best_k['final_rel_l2']:.6e}, Avg={avg_error:.6e}")

    print("\n=== Performance by Architecture Size ===")
    if 'param_count' in df.columns and df['param_count'].notna().any():
        arch_stats = df.groupby(['width', 'depth']).agg({
            'final_rel_l2': ['mean', 'std', 'min'],
            'param_count': 'first'
        }).round(6)
        arch_stats.columns = ['rel_l2_mean', 'rel_l2_std', 'rel_l2_min', 'param_count']
        print(arch_stats.head(10))
    else:
        size_stats = df.groupby(['width', 'depth'])['final_rel_l2'].agg(['mean', 'std', 'min']).round(6)
        print(size_stats.head(10))


def main():
    parser = argparse.ArgumentParser(description="Analyze MLP benchmark results")
    parser.add_argument("results_dir", help="Directory containing result JSON files")
    parser.add_argument("--output-dir", help="Directory to save plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_dir}...")
    df = load_results(args.results_dir)

    # Print summary
    print_summary(df)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        plot_results(df, args.output_dir)

    print("Analysis complete!")


if __name__ == "__main__":
    main()