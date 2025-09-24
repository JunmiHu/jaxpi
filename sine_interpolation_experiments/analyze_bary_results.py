#!/usr/bin/env python3
"""
Analysis script for barycentric attention benchmark results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np
from typing import List, Dict, Any
import itertools


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all barycentric attention experiment results from a directory into a DataFrame."""
    results_path = Path(results_dir)
    data = []

    for json_file in results_path.glob("baryattn_N*_k*_s*.json"):
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
                'N': config['N'],
                'k': config['k'],
                'seed': config['seed'],
                'steps': config['steps'],
                'final_rel_l2': training['final_rel_l2'],
                'initial_rel_l2': training['initial_rel_l2'],
                'final_loss': training['final_loss'],
                'n_train': config['n_train'],
                'n_eval': config['n_eval'],
                'param_count': config.get('param_count', None),
                'use_exact_init': config.get('use_exact_init', None),
                'rational_representation': config.get('rational_representation', 'unknown'),
                'kernel_init_strategy': config.get('kernel_init_strategy', 'unknown'),
                'kernel_init_scale': config.get('kernel_init_scale', None),
                'input_dependent_values': config.get('input_dependent_values', False),
                'learn_nodes': config.get('learn_nodes', False),
                'learn_query': config.get('learn_query', False),
                'query_hdim': config.get('query_hdim', None),
                'lr': config.get('lr', None),
                'c1': config.get('c1', None),
                'c2': config.get('c2', None),
                'filename': json_file.name
            }
            data.append(row)

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    if not data:
        raise ValueError(f"No valid barycentric attention results found in {results_dir}")

    return pd.DataFrame(data)


def load_training_curves(results_dir: str) -> List[Dict[str, Any]]:
    """Load training curves from all barycentric attention experiment results."""
    results_path = Path(results_dir)
    curves_data = []

    for json_file in results_path.glob("baryattn_N*_k*_s*.json"):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)

            config = result['config']
            training = result['training']

            # Only include if iteration history is available
            if 'iteration_history' in training:
                curve_info = {
                    'filename': json_file.name,
                    'N': config['N'],
                    'k': config['k'],
                    'seed': config['seed'],
                    'param_count': config.get('param_count', None),
                    'rational_representation': config.get('rational_representation', 'unknown'),
                    'kernel_init_strategy': config.get('kernel_init_strategy', 'unknown'),
                    'kernel_init_scale': config.get('kernel_init_scale', None),
                    'input_dependent_values': config.get('input_dependent_values', False),
                    'learn_nodes': config.get('learn_nodes', False),
                    'learn_query': config.get('learn_query', False),
                    'query_hdim': config.get('query_hdim', None),
                    'iterations': training['iteration_history'],
                    'loss_history': training['loss_history'],
                    'rel_l2_history': training['rel_l2_history']
                }
                curves_data.append(curve_info)

        except Exception as e:
            print(f"Error loading training curves from {json_file}: {e}")

    return curves_data


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

    # 1. Training curves for different N values (fixed k=1, seed=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Filter for k=1, seed=0 to compare architectures
    k1_curves = [c for c in curves_data if c['k'] == 1 and c['seed'] == 0]

    for curve in k1_curves[:10]:  # Limit to first 10 for readability
        label = f"N={curve['N']}, {curve['rational_representation']}"
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
        plt.savefig(output_path / 'bary_training_curves_architectures.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Convergence by k value (fixed N)
    if len(curves_data) > 0:
        # Use the most common N value
        n_counts = {}
        for curve in curves_data:
            n = curve['N']
            n_counts[n] = n_counts.get(n, 0) + 1

        most_common_n = max(n_counts, key=n_counts.get)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot convergence for different k values with this N (seed=0)
        n_curves = [c for c in curves_data if c['N'] == most_common_n and c['seed'] == 0]

        for curve in sorted(n_curves, key=lambda x: x['k']):
            ax.plot(curve['iterations'], curve['rel_l2_history'],
                   label=f"k={curve['k']}", alpha=0.8, marker='o', markersize=3)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative L2 Error')
        ax.set_yscale('log')
        ax.set_title(f'Convergence by k (N={most_common_n}, seed=0)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_dir:
            plt.savefig(output_path / 'bary_convergence_by_k.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 3. Training curves by configuration parameters
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot by rational representation
    representations = list(set(c['rational_representation'] for c in curves_data))
    for rep in representations:
        rep_curves = [c for c in curves_data if c['rational_representation'] == rep and c['seed'] == 0][:5]
        for curve in rep_curves:
            axes[0, 0].plot(curve['iterations'], curve['rel_l2_history'],
                           label=f"{rep}, N={curve['N']}, k={curve['k']}", alpha=0.7)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Relative L2 Error')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('By Rational Representation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot by kernel init strategy
    init_strategies = list(set(c['kernel_init_strategy'] for c in curves_data))
    for strategy in init_strategies:
        strategy_curves = [c for c in curves_data if c['kernel_init_strategy'] == strategy and c['seed'] == 0][:5]
        for curve in strategy_curves:
            axes[0, 1].plot(curve['iterations'], curve['rel_l2_history'],
                           label=f"{strategy}, N={curve['N']}, k={curve['k']}", alpha=0.7)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Relative L2 Error')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('By Kernel Init Strategy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot by input dependent values
    for idv in [True, False]:
        idv_curves = [c for c in curves_data if c['input_dependent_values'] == idv and c['seed'] == 0][:5]
        for curve in idv_curves:
            axes[1, 0].plot(curve['iterations'], curve['rel_l2_history'],
                           label=f"input_dep={idv}, N={curve['N']}, k={curve['k']}", alpha=0.7)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Relative L2 Error')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('By Input Dependent Values')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot by learn nodes/query
    for ln, lq in itertools.product([True, False], repeat=2):
        learn_curves = [c for c in curves_data if c['learn_nodes'] == ln and c['learn_query'] == lq and c['seed'] == 0][:3]
        for curve in learn_curves:
            axes[1, 1].plot(curve['iterations'], curve['rel_l2_history'],
                           label=f"nodes={ln}, query={lq}, N={curve['N']}, k={curve['k']}", alpha=0.7)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Relative L2 Error')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('By Learning Nodes/Query')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_path / 'bary_training_curves_by_config.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_results(df: pd.DataFrame, output_dir: str = None):
    """Create various plots analyzing the barycentric attention results."""
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Heatmap of final rel L2 error by N and rational representation for different k values
    fig, axes = plt.subplots(1, min(len(df['k'].unique()), 3), figsize=(15, 6))
    if len(df['k'].unique()) == 1:
        axes = [axes]

    # Use median over seeds for each configuration
    df_median = df.groupby(['N', 'rational_representation', 'k'])['final_rel_l2'].median().reset_index()

    k_values = sorted(df['k'].unique())
    for i, k in enumerate(k_values[:3]):  # Show first 3 k values
        subset = df_median[df_median['k'] == k]
        if len(subset) == 0:
            continue

        pivot = subset.pivot(index='rational_representation', columns='N', values='final_rel_l2')

        ax = axes[i] if len(axes) > 1 else axes[0]
        sns.heatmap(pivot, annot=True, fmt='.2e', cmap='viridis_r',
                   ax=ax, cbar_kws={'label': 'Median Rel L2 Error'})
        ax.set_title(f'Median Rel L2 Error (k={k})')
        ax.set_xlabel('N (Number of Nodes)')
        ax.set_ylabel('Rational Representation')

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_path / 'bary_heatmap_rel_l2.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Line plot: Effect of N for different configurations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Effect of N by rational representation
    representations = df['rational_representation'].unique()
    for rep in representations:
        subset = df[df['rational_representation'] == rep].groupby('N')['final_rel_l2'].median()
        if len(subset) > 0:
            axes[0, 0].plot(subset.index, subset.values, marker='o', label=f'{rep}')
    axes[0, 0].set_xlabel('N (Number of Nodes)')
    axes[0, 0].set_ylabel('Final Rel L2 Error')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('Effect of N by Rational Representation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Effect of N by kernel init strategy
    init_strategies = df['kernel_init_strategy'].unique()
    for strategy in init_strategies:
        subset = df[df['kernel_init_strategy'] == strategy].groupby('N')['final_rel_l2'].median()
        if len(subset) > 0:
            axes[0, 1].plot(subset.index, subset.values, marker='o', label=f'{strategy}')
    axes[0, 1].set_xlabel('N (Number of Nodes)')
    axes[0, 1].set_ylabel('Final Rel L2 Error')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Effect of N by Kernel Init Strategy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Effect of N by input dependent values
    for idv in df['input_dependent_values'].unique():
        subset = df[df['input_dependent_values'] == idv].groupby('N')['final_rel_l2'].median()
        if len(subset) > 0:
            axes[1, 0].plot(subset.index, subset.values, marker='o', label=f'Input Dep: {idv}')
    axes[1, 0].set_xlabel('N (Number of Nodes)')
    axes[1, 0].set_ylabel('Final Rel L2 Error')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Effect of N by Input Dependent Values')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Effect of N by learning parameters
    learn_combinations = df[['learn_nodes', 'learn_query']].drop_duplicates()
    for _, row in learn_combinations.iterrows():
        subset = df[(df['learn_nodes'] == row['learn_nodes']) &
                   (df['learn_query'] == row['learn_query'])].groupby('N')['final_rel_l2'].median()
        if len(subset) > 0:
            label = f"Nodes: {row['learn_nodes']}, Query: {row['learn_query']}"
            axes[1, 1].plot(subset.index, subset.values, marker='o', label=label)
    axes[1, 1].set_xlabel('N (Number of Nodes)')
    axes[1, 1].set_ylabel('Final Rel L2 Error')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Effect of N by Learning Parameters')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_path / 'bary_n_effect_by_config.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Line plot: Effect of k value
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use median over seeds and configurations (more robust)
    k_effect = df.groupby('k')['final_rel_l2'].agg(['median', 'std']).reset_index()

    ax.errorbar(k_effect['k'], k_effect['median'], yerr=k_effect['std'],
               marker='o', capsize=5, capthick=2)
    ax.set_xlabel('k (frequency)')
    ax.set_ylabel('Final Rel L2 Error')
    ax.set_yscale('log')
    ax.set_title('Effect of Frequency k on Performance')
    ax.grid(True, alpha=0.3)

    if output_dir:
        plt.savefig(output_path / 'bary_frequency_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Parameter count vs performance (loglog with power law fit)
    if 'param_count' in df.columns and df['param_count'].notna().any():
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by configuration and get median performance and param count
        config_performance = df.groupby(['N', 'rational_representation', 'input_dependent_values',
                                       'learn_nodes', 'learn_query']).agg({
            'final_rel_l2': 'median',
            'param_count': 'first'  # param count is the same for all runs of same configuration
        }).reset_index()

        # Filter out any zero or negative values for log scale
        valid_data = config_performance[(config_performance['param_count'] > 0) &
                                      (config_performance['final_rel_l2'] > 0)].copy()

        if len(valid_data) >= 2:  # Need at least 2 points for a fit
            # Create scatter plot colored by N
            scatter = ax.scatter(valid_data['param_count'], valid_data['final_rel_l2'],
                               alpha=0.7, s=60, c=valid_data['N'], cmap='viridis', zorder=3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('N (Number of Nodes)')

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
                label = f"N={int(row['N'])},{row['rational_representation'][:4]}"
                ax.annotate(label, (row['param_count'], row['final_rel_l2']),
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
            plt.savefig(output_path / 'bary_performance_vs_params.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 5. Configuration comparison heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a combined configuration string for better visualization
    df_config = df.copy()
    df_config['config_str'] = df_config.apply(lambda row:
        f"N{row['N']}_{row['rational_representation'][:4]}_{'idv' if row['input_dependent_values'] else 'const'}_"
        f"{'ln' if row['learn_nodes'] else 'fn'}_{'lq' if row['learn_query'] else 'fq'}", axis=1)

    config_summary = df_config.groupby(['config_str', 'k'])['final_rel_l2'].median().unstack(fill_value=np.nan)

    sns.heatmap(config_summary, annot=True, fmt='.2e', cmap='viridis_r',
               ax=ax, cbar_kws={'label': 'Median Rel L2 Error'})
    ax.set_title('Performance Heatmap by Configuration and k')
    ax.set_xlabel('k (frequency)')
    ax.set_ylabel('Configuration')

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_path / 'bary_config_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 6. Line plots for all scaled parameters
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Kernel init scale effect
    if 'kernel_init_scale' in df.columns and df['kernel_init_scale'].notna().any():
        scale_effect = df.groupby('kernel_init_scale')['final_rel_l2'].agg(['median', 'std']).reset_index()
        axes[0].errorbar(scale_effect['kernel_init_scale'], scale_effect['median'],
                        yerr=scale_effect['std'], marker='o', capsize=5)
        axes[0].set_xlabel('Kernel Init Scale')
        axes[0].set_ylabel('Final Rel L2 Error')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_title('Effect of Kernel Init Scale')
        axes[0].grid(True, alpha=0.3)

    # Learning rate effect
    if 'lr' in df.columns and df['lr'].notna().any():
        lr_effect = df.groupby('lr')['final_rel_l2'].agg(['median', 'std']).reset_index()
        axes[1].errorbar(lr_effect['lr'], lr_effect['median'],
                        yerr=lr_effect['std'], marker='o', capsize=5)
        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('Final Rel L2 Error')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_title('Effect of Learning Rate')
        axes[1].grid(True, alpha=0.3)

    # c1 parameter effect
    if 'c1' in df.columns and df['c1'].notna().any():
        c1_effect = df.groupby('c1')['final_rel_l2'].agg(['median', 'std']).reset_index()
        axes[2].errorbar(c1_effect['c1'], c1_effect['median'],
                        yerr=c1_effect['std'], marker='o', capsize=5)
        axes[2].set_xlabel('c1 Parameter')
        axes[2].set_ylabel('Final Rel L2 Error')
        axes[2].set_xscale('log')
        axes[2].set_yscale('log')
        axes[2].set_title('Effect of c1 Parameter')
        axes[2].grid(True, alpha=0.3)

    # c2 parameter effect
    if 'c2' in df.columns and df['c2'].notna().any():
        c2_effect = df.groupby('c2')['final_rel_l2'].agg(['median', 'std']).reset_index()
        axes[3].errorbar(c2_effect['c2'], c2_effect['median'],
                        yerr=c2_effect['std'], marker='o', capsize=5)
        axes[3].set_xlabel('c2 Parameter')
        axes[3].set_ylabel('Final Rel L2 Error')
        axes[3].set_yscale('log')
        axes[3].set_title('Effect of c2 Parameter')
        axes[3].grid(True, alpha=0.3)

    # Query hidden dimension effect
    if 'query_hdim' in df.columns and df['query_hdim'].notna().any():
        hdim_effect = df.groupby('query_hdim')['final_rel_l2'].agg(['median', 'std']).reset_index()
        axes[4].errorbar(hdim_effect['query_hdim'], hdim_effect['median'],
                        yerr=hdim_effect['std'], marker='o', capsize=5)
        axes[4].set_xlabel('Query Hidden Dimension')
        axes[4].set_ylabel('Final Rel L2 Error')
        axes[4].set_yscale('log')
        axes[4].set_title('Effect of Query Hidden Dimension')
        axes[4].grid(True, alpha=0.3)

    # Number of training steps effect
    steps_effect = df.groupby('steps')['final_rel_l2'].agg(['median', 'std']).reset_index()
    axes[5].errorbar(steps_effect['steps'], steps_effect['median'],
                    yerr=steps_effect['std'], marker='o', capsize=5)
    axes[5].set_xlabel('Training Steps')
    axes[5].set_ylabel('Final Rel L2 Error')
    axes[5].set_yscale('log')
    axes[5].set_title('Effect of Training Steps')
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_path / 'bary_scaled_parameters_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 7. Rational representation comparison plots
    plot_representation_comparisons(df, output_dir)


def plot_representation_comparisons(df: pd.DataFrame, output_dir: str = None):
    """Create comparison plots between different rational representations."""
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Check if we have multiple representations to compare
    representations = df['rational_representation'].unique()
    if len(representations) <= 1:
        print(f"Only found {len(representations)} representation(s): {representations}")
        print("Skipping representation comparison plots")
        return

    print(f"Comparing representations: {list(representations)}")

    # Set style for comparison plots
    plt.style.use('default')
    colors = plt.cm.Set1(np.linspace(0, 1, len(representations)))

    # 1. Performance comparison boxplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Overall performance comparison
    sns.boxplot(data=df, x='rational_representation', y='final_rel_l2', ax=axes[0, 0])
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('Performance Comparison by Representation')
    axes[0, 0].set_xlabel('Rational Representation')
    axes[0, 0].set_ylabel('Final Rel L2 Error')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Parameter count vs performance scatter
    if 'param_count' in df.columns and df['param_count'].notna().any():
        for i, repr_type in enumerate(representations):
            subset = df[df['rational_representation'] == repr_type]
            axes[0, 1].scatter(subset['param_count'], subset['final_rel_l2'],
                             label=repr_type, alpha=0.7, s=60, color=colors[i])
        axes[0, 1].set_xlabel('Parameter Count')
        axes[0, 1].set_ylabel('Final Rel L2 Error')
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Parameter Count vs Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Performance by N value
    for i, repr_type in enumerate(representations):
        subset = df[df['rational_representation'] == repr_type]
        if len(subset) > 0:
            n_perf = subset.groupby('N')['final_rel_l2'].median()
            axes[1, 0].plot(n_perf.index, n_perf.values, marker='o',
                           label=repr_type, color=colors[i], linewidth=2)
    axes[1, 0].set_xlabel('N (Number of Nodes)')
    axes[1, 0].set_ylabel('Median Final Rel L2 Error')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Performance by N Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Performance by k value
    for i, repr_type in enumerate(representations):
        subset = df[df['rational_representation'] == repr_type]
        if len(subset) > 0:
            k_perf = subset.groupby('k')['final_rel_l2'].median()
            axes[1, 1].plot(k_perf.index, k_perf.values, marker='o',
                           label=repr_type, color=colors[i], linewidth=2)
    axes[1, 1].set_xlabel('k (Frequency)')
    axes[1, 1].set_ylabel('Median Final Rel L2 Error')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Performance by Frequency k')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_path / 'representation_comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Statistical comparison heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create summary statistics for each representation
    stats_data = []
    for repr_type in representations:
        subset = df[df['rational_representation'] == repr_type]
        if len(subset) > 0:
            stats = {
                'representation': repr_type,
                'count': len(subset),
                'best_rel_l2': subset['final_rel_l2'].min(),
                'median_rel_l2': subset['final_rel_l2'].median(),
                'mean_rel_l2': subset['final_rel_l2'].mean(),
                'std_rel_l2': subset['final_rel_l2'].std(),
                'worst_rel_l2': subset['final_rel_l2'].max(),
                'param_count_typical': subset['param_count'].median() if 'param_count' in subset.columns else None
            }
            stats_data.append(stats)

    stats_df = pd.DataFrame(stats_data)

    # Create a heatmap of log10(rel_l2) values for better visualization
    heatmap_data = stats_df[['best_rel_l2', 'median_rel_l2', 'mean_rel_l2', 'worst_rel_l2']].copy()
    heatmap_data = heatmap_data.apply(lambda x: np.log10(x))
    heatmap_data.index = stats_df['representation']

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='viridis_r',
               ax=ax, cbar_kws={'label': 'log10(Rel L2 Error)'})
    ax.set_title('Performance Statistics Comparison (log10 scale)')
    ax.set_xlabel('Statistics')
    ax.set_ylabel('Representation')

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_path / 'representation_stats_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Head-to-head comparison for matched configurations
    print("\n=== HEAD-TO-HEAD COMPARISON ===")

    # Find configurations that exist for multiple representations
    config_columns = ['N', 'k', 'seed']
    common_configs = df.groupby(config_columns)['rational_representation'].nunique()
    common_configs = common_configs[common_configs > 1].index

    if len(common_configs) > 0:
        print(f"Found {len(common_configs)} configurations with multiple representations")

        # Create head-to-head comparison data
        h2h_data = []
        for config in common_configs:
            n, k, seed = config
            config_data = df[(df['N'] == n) & (df['k'] == k) & (df['seed'] == seed)]

            row = {'N': n, 'k': k, 'seed': seed}
            for _, result in config_data.iterrows():
                repr_type = result['rational_representation']
                row[f'{repr_type}_rel_l2'] = result['final_rel_l2']
                row[f'{repr_type}_params'] = result.get('param_count', None)
            h2h_data.append(row)

        h2h_df = pd.DataFrame(h2h_data)

        # Plot head-to-head comparison
        rel_l2_cols = [col for col in h2h_df.columns if col.endswith('_rel_l2')]
        if len(rel_l2_cols) >= 2:
            fig, axes = plt.subplots(1, min(2, len(rel_l2_cols)-1), figsize=(15, 6))
            if len(rel_l2_cols) == 2:
                axes = [axes]

            # Pairwise comparisons
            comparisons_made = 0
            for i in range(len(rel_l2_cols)):
                for j in range(i+1, len(rel_l2_cols)):
                    if comparisons_made >= 2:  # Limit to 2 comparison plots
                        break

                    repr1 = rel_l2_cols[i].replace('_rel_l2', '')
                    repr2 = rel_l2_cols[j].replace('_rel_l2', '')

                    # Get matched data points
                    mask = h2h_df[rel_l2_cols[i]].notna() & h2h_df[rel_l2_cols[j]].notna()
                    x_data = h2h_df.loc[mask, rel_l2_cols[i]]
                    y_data = h2h_df.loc[mask, rel_l2_cols[j]]

                    if len(x_data) > 0:
                        ax = axes[comparisons_made] if len(axes) > 1 else axes[0]
                        ax.scatter(x_data, y_data, alpha=0.7, s=60)

                        # Add diagonal line for equal performance
                        min_val = min(x_data.min(), y_data.min())
                        max_val = max(x_data.max(), y_data.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

                        ax.set_xlabel(f'{repr1} Rel L2 Error')
                        ax.set_ylabel(f'{repr2} Rel L2 Error')
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        ax.set_title(f'{repr1} vs {repr2} Head-to-Head')
                        ax.grid(True, alpha=0.3)

                        # Count wins
                        wins_repr1 = (x_data < y_data).sum()
                        wins_repr2 = (y_data < x_data).sum()
                        ties = (x_data == y_data).sum()

                        ax.text(0.05, 0.95, f'{repr1}: {wins_repr1} wins\n{repr2}: {wins_repr2} wins\nTies: {ties}',
                               transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                               verticalalignment='top', fontsize=10)

                        comparisons_made += 1

            plt.tight_layout()
            if output_dir:
                plt.savefig(output_path / 'head_to_head_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()

        # Print numerical comparison
        print("\nNumerical Head-to-Head Results:")
        print(h2h_df.to_string(index=False))

        # Determine overall winner
        print("\n=== OVERALL COMPARISON SUMMARY ===")
        for repr_type in representations:
            subset = df[df['rational_representation'] == repr_type]
            print(f"\n{repr_type.upper()}:")
            print(f"  Experiments: {len(subset)}")
            print(f"  Best rel L2: {subset['final_rel_l2'].min():.6e}")
            print(f"  Median rel L2: {subset['final_rel_l2'].median():.6e}")
            print(f"  Mean rel L2: {subset['final_rel_l2'].mean():.6e}")
            if 'param_count' in subset.columns:
                print(f"  Typical params: {subset['param_count'].median():.0f}")
    else:
        print("No common configurations found for direct head-to-head comparison")

    # 4. Training curves comparison (if available)
    plot_representation_training_curves(df, representations, output_dir)


def plot_representation_training_curves(df: pd.DataFrame, representations: List[str], output_dir: str = None):
    """Plot training curves comparison between representations."""
    # This would require loading the full training curves data
    # For now, we'll create a placeholder that shows the concept
    print("\n=== TRAINING CURVES COMPARISON ===")
    print("Training curves comparison would show convergence patterns for each representation")
    print("This requires the full iteration_history data from the JSON files")

    # TODO: Implement training curves comparison when data is available
    # This would involve:
    # 1. Loading training curves data from the original JSON files
    # 2. Plotting convergence curves for each representation
    # 3. Comparing convergence rates and final performance


def print_summary(df: pd.DataFrame):
    """Print summary statistics for barycentric attention results."""
    print("=== Barycentric Attention Experiment Summary ===")
    print(f"Total experiments: {len(df)}")
    print(f"Unique configurations: {len(df.groupby(['N', 'rational_representation', 'input_dependent_values', 'learn_nodes', 'learn_query']))}")
    print(f"N values tested: {sorted(df['N'].unique())}")
    print(f"k values tested: {sorted(df['k'].unique())}")
    print(f"Seeds used: {sorted(df['seed'].unique())}")
    print(f"Rational representations: {sorted(df['rational_representation'].unique())}")

    print("\n=== Best Results ===")
    best_overall = df.loc[df['final_rel_l2'].idxmin()]
    print(f"Best overall: N={best_overall['N']}, {best_overall['rational_representation']}, "
          f"k={best_overall['k']}, Rel L2={best_overall['final_rel_l2']:.6e}")

    print("\n=== Best by k value ===")
    for k in sorted(df['k'].unique()):
        subset = df[df['k'] == k]
        best_k = subset.loc[subset['final_rel_l2'].idxmin()]
        median_error = subset['final_rel_l2'].median()
        param_count = best_k.get('param_count', 'N/A')
        print(f"k={k}: Best N={best_k['N']}, {best_k['rational_representation']}, "
              f"Params={param_count}, Rel L2={best_k['final_rel_l2']:.6e}, Median={median_error:.6e}")

    print("\n=== Performance by Configuration ===")
    if 'param_count' in df.columns and df['param_count'].notna().any():
        config_stats = df.groupby(['N', 'rational_representation']).agg({
            'final_rel_l2': ['median', 'std', 'min', 'count'],
            'param_count': 'first'
        }).round(6)
        config_stats.columns = ['rel_l2_median', 'rel_l2_std', 'rel_l2_min', 'num_runs', 'param_count']
        print(config_stats.head(10))
    else:
        config_stats = df.groupby(['N', 'rational_representation'])['final_rel_l2'].agg(['median', 'std', 'min', 'count']).round(6)
        print(config_stats.head(10))

    print("\n=== Parameter Distribution Summary ===")
    print("Kernel init scales:", sorted(df['kernel_init_scale'].dropna().unique()))
    print("Learning rates:", sorted(df['lr'].dropna().unique()))
    print("c1 values:", sorted(df['c1'].dropna().unique()))
    print("c2 values:", sorted(df['c2'].dropna().unique()))
    print("Query hidden dims:", sorted(df['query_hdim'].dropna().unique()))


def main():
    parser = argparse.ArgumentParser(description="Analyze barycentric attention benchmark results")
    parser.add_argument("results_dir", help="Directory containing barycentric attention result JSON files")
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
    print(f"Loading barycentric attention results from {args.results_dir}...")
    df = load_results(args.results_dir)

    # Print summary
    print_summary(df)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        plot_results(df, args.output_dir)

        # Also plot training curves if available
        print("\nGenerating training curves...")
        plot_training_curves_from_dir(args.results_dir, args.output_dir)

    print("Analysis complete!")


if __name__ == "__main__":
    main()