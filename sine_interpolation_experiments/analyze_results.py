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


def load_training_curves(results_dir: str) -> List[Dict[str, Any]]:
    """Load training curves from all experiment results."""
    results_path = Path(results_dir)
    curves_data = []

    for json_file in results_path.glob("mlp_w*_d*_k*_s*.json"):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)

            config = result['config']
            training = result['training']

            # Only include if iteration history is available
            if 'iteration_history' in training:
                curve_info = {
                    'filename': json_file.name,
                    'width': config['width'],
                    'depth': config['depth'],
                    'k': config['k'],
                    'seed': config['seed'],
                    'param_count': config.get('param_count', None),
                    'iterations': training['iteration_history'],
                    'loss_history': training['loss_history'],
                    'rel_l2_history': training['rel_l2_history']
                }
                curves_data.append(curve_info)

        except Exception as e:
            print(f"Error loading training curves from {json_file}: {e}")

    return curves_data


def plot_training_curves(df: pd.DataFrame, output_dir: str = None):
    """Plot training curves for loss and relative L2 error."""
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Get the results directory from the first row (assuming all results are from same dir)
    if len(df) == 0:
        print("No data available for plotting training curves")
        return

    # Try to load training curves from the same directory structure
    # We'll need to reconstruct the path from the analysis
    print("Loading training curve data...")

    # For now, create placeholder plots that will work with future data
    # 1. Loss and Rel L2 curves side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Check if we have access to the raw result files
    has_curve_data = False

    # Plot loss curves
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Training Loss')
    ax1.set_yscale('log')
    ax1.set_title('Training Loss Curves')
    ax1.grid(True, alpha=0.3)

    # Plot rel L2 error curves
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Relative L2 Error')
    ax2.set_yscale('log')
    ax2.set_title('Relative L2 Error Curves')
    ax2.grid(True, alpha=0.3)

    if not has_curve_data:
        ax1.text(0.5, 0.5, 'Training curves will be plotted here\nwhen results with iteration_history are analyzed.\n\nRun new experiments to generate training curves.',
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        ax2.text(0.5, 0.5, 'Rel L2 error curves will be plotted here\nwhen results with iteration_history are analyzed.\n\nRun new experiments to generate training curves.',
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Convergence comparison by k value
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative L2 Error')
    ax.set_yscale('log')
    ax.set_title('Convergence Rate by Frequency k')
    ax.grid(True, alpha=0.3)

    if not has_curve_data:
        ax.text(0.5, 0.5, 'Convergence comparison will be plotted here\nwhen results with iteration_history are analyzed.\n\nRun new experiments to generate convergence plots.',
               transform=ax.transAxes, ha='center', va='center', fontsize=12)

    if output_dir:
        plt.savefig(output_path / 'convergence_by_k.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves_from_dir(results_dir: str, output_dir: str = None):
    """Plot training curves directly from results directory."""
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Load training curves
    curves_data = load_training_curves(results_dir)

    if not curves_data:
        print("No training curve data found with iteration_history")
        return

    print(f"Found {len(curves_data)} experiments with training curves")

    # 1. Training curves for different architectures (fixed k=1, seed=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Filter for k=1, seed=0 to compare architectures
    k1_curves = [c for c in curves_data if c['k'] == 1 and c['seed'] == 0]

    for curve in k1_curves[:5]:  # Limit to first 5 for readability
        label = f"w={curve['width']}, d={curve['depth']}"
        ax1.plot(curve['iterations'], curve['loss_history'], label=label, alpha=0.8)
        ax2.plot(curve['iterations'], curve['rel_l2_history'], label=label, alpha=0.8)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Training Loss')
    ax1.set_yscale('log')
    ax1.set_title('Training Loss (k=1, seed=0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Relative L2 Error')
    ax2.set_yscale('log')
    ax2.set_title('Relative L2 Error (k=1, seed=0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_path / 'training_curves_architectures.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Convergence by k value (fixed architecture)
    if len(curves_data) > 0:
        # Use the most common architecture
        arch_counts = {}
        for curve in curves_data:
            arch = (curve['width'], curve['depth'])
            arch_counts[arch] = arch_counts.get(arch, 0) + 1

        most_common_arch = max(arch_counts, key=arch_counts.get)
        w_common, d_common = most_common_arch

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot convergence for different k values with this architecture (seed=0)
        k_curves = [c for c in curves_data if c['width'] == w_common and c['depth'] == d_common and c['seed'] == 0]

        for curve in sorted(k_curves, key=lambda x: x['k']):
            ax.plot(curve['iterations'], curve['rel_l2_history'],
                   label=f"k={curve['k']}", alpha=0.8, marker='o', markersize=3)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative L2 Error')
        ax.set_yscale('log')
        ax.set_title(f'Convergence by k (width={w_common}, depth={d_common}, seed=0)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_dir:
            plt.savefig(output_path / 'convergence_by_k.png', dpi=300, bbox_inches='tight')
        plt.show()


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all experiment results from a directory into a DataFrame."""
    results_path = Path(results_dir)
    data = []

    for json_file in results_path.glob("mlp_w*_d*_k*_s*.json"):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)

            # Skip error files (they won't have 'training' key)
            if 'training' not in result:
                print(f"Skipping error file: {json_file.name}")
                continue

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

    # Use median over seeds for each k (more robust to outliers)
    df_median = df.groupby(['width', 'depth', 'k'])['final_rel_l2'].median().reset_index()

    # Plot for different k values
    k_values = sorted(df['k'].unique())
    for i, k in enumerate(k_values[:2]):  # Show first 2 k values
        subset = df_median[df_median['k'] == k]
        pivot = subset.pivot(index='depth', columns='width', values='final_rel_l2')

        sns.heatmap(pivot, annot=True, fmt='.2e', cmap='viridis_r',
                   ax=axes[i], cbar_kws={'label': 'Median Rel L2 Error'})
        axes[i].set_title(f'Median Rel L2 Error (k={k})')
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
        subset = df[df['depth'] == depth].groupby('width')['final_rel_l2'].median()
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

    # Use median over seeds and architectures (more robust)
    k_effect = df.groupby('k')['final_rel_l2'].agg(['median', 'std']).reset_index()

    ax.errorbar(k_effect['k'], k_effect['median'], yerr=k_effect['std'],
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

    # 5. Parameter count vs performance (loglog with power law fit)
    if 'param_count' in df.columns and df['param_count'].notna().any():
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by architecture (width, depth) and get median performance and param count
        arch_performance = df.groupby(['width', 'depth']).agg({
            'final_rel_l2': 'median',
            'param_count': 'first'  # param count is the same for all runs of same architecture
        }).reset_index()

        # Filter out any zero or negative values for log scale
        valid_data = arch_performance[(arch_performance['param_count'] > 0) &
                                    (arch_performance['final_rel_l2'] > 0)].copy()

        if len(valid_data) >= 2:  # Need at least 2 points for a fit
            # Create scatter plot
            scatter = ax.scatter(valid_data['param_count'], valid_data['final_rel_l2'],
                               alpha=0.7, s=60, c=valid_data['width'], cmap='viridis', zorder=3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Width')

            # Fit power law: error = A * params^B
            # Taking log: log(error) = log(A) + B * log(params)
            log_params = np.log(valid_data['param_count'])
            log_error = np.log(valid_data['final_rel_l2'])

            # Linear fit in log space
            coeffs = np.polyfit(log_params, log_error, 1)
            slope, intercept = coeffs
            A = np.exp(intercept)  # Convert back from log space
            B = slope

            # Create fit line
            param_range = np.logspace(np.log10(valid_data['param_count'].min()),
                                    np.log10(valid_data['param_count'].max()), 100)
            fit_line = A * (param_range ** B)

            ax.plot(param_range, fit_line, 'r--', alpha=0.8, linewidth=2, zorder=2,
                   label=f'Power law fit: error ∝ params^{B:.2f}')

            # Add labels for each point
            for _, row in valid_data.iterrows():
                ax.annotate(f"w={int(row['width'])},d={int(row['depth'])}",
                           (row['param_count'], row['final_rel_l2']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

            # Calculate R-squared
            y_pred = A * (valid_data['param_count'] ** B)
            ss_res = np.sum((valid_data['final_rel_l2'] - y_pred) ** 2)
            ss_tot = np.sum((valid_data['final_rel_l2'] - valid_data['final_rel_l2'].mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            ax.text(0.05, 0.95, f'Power law: error = {A:.2e} × params^{B:.2f}\nR² = {r_squared:.3f}',
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=10)

            ax.legend()

        ax.set_xlabel('Parameter Count')
        ax.set_ylabel('Median Relative L2 Error')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Performance vs Model Size (Power Law Scaling)')
        ax.grid(True, alpha=0.3)

        if output_dir:
            plt.savefig(output_path / 'performance_vs_params.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 6. Training curves: Loss and Rel L2 Error over iterations
    plot_training_curves(df, output_dir)


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
        median_error = subset['final_rel_l2'].median()
        param_count = best_k.get('param_count', 'N/A')
        print(f"k={k}: Best Width={best_k['width']}, Depth={best_k['depth']}, "
              f"Params={param_count}, Rel L2={best_k['final_rel_l2']:.6e}, Median={median_error:.6e}")

    print("\n=== Performance by Architecture Size ===")
    if 'param_count' in df.columns and df['param_count'].notna().any():
        arch_stats = df.groupby(['width', 'depth']).agg({
            'final_rel_l2': ['median', 'std', 'min', 'count'],
            'param_count': 'first'
        }).round(6)
        arch_stats.columns = ['rel_l2_median', 'rel_l2_std', 'rel_l2_min', 'num_runs', 'param_count']
        print(arch_stats.head(10))
    else:
        size_stats = df.groupby(['width', 'depth'])['final_rel_l2'].agg(['median', 'std', 'min', 'count']).round(6)
        print(size_stats.head(10))


def main():
    parser = argparse.ArgumentParser(description="Analyze MLP benchmark results")
    parser.add_argument("results_dir", help="Directory containing result JSON files")
    parser.add_argument("--output-dir", help="Directory to save plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--training-curves-only", action="store_true",
                        help="Only plot training curves (requires iteration_history in results)")

    args = parser.parse_args()

    if args.training_curves_only:
        print(f"Plotting training curves from {args.results_dir}...")
        plot_training_curves_from_dir(args.results_dir, args.output_dir)
        return

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