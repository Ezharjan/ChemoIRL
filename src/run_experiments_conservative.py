"""Alternative baseline experiment configuration with lower learning rate and fewer iterations."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import warnings
from chemo_irl import ChemoIRL, GymDeFi, generate_synthetic_expert_trajectories
from baselines import ActionFrequencyBaseline, LinearGAIL
from config import TRAINING_CONFIG
import time

warnings.filterwarnings('ignore')
np.random.seed(42)

# Intent mapping for evaluation
INTENT_MAPPING = {
    "swap": 0,
    "lend": 1,
    "liquidity": 2,
    "yield": 3,
    "complex_leverage": 4,
    "complex_LP": 5
}

def main():
    print("="*70)
    print("ALTERNATIVE BASELINE EXPERIMENTS")
    print("="*70)
    
    # Generate synthetic data
    print("\nGenerating Synthetic Expert Trajectories...")
    env = GymDeFi()
    intent_types = ["swap", "lend", "liquidity", "yield", "complex_leverage", "complex_LP"]
    
    all_trajs = generate_synthetic_expert_trajectories(
        env,
        n_trajectories=TRAINING_CONFIG['n_expert_trajectories'],
        intent_types=intent_types
    )
    n_train = int(TRAINING_CONFIG['train_test_split'] * len(all_trajs))
    train_trajs = all_trajs[:n_train]
    test_trajs = all_trajs[n_train:]
    
    print(f"Generated {len(train_trajs)} training trajectories and {len(test_trajs)} test trajectories")
    
    results = {}
    
    # 1. Action Frequency (BC) - Heuristic Baseline
    print("\n" + "-"*70)
    print("1. Training Action Frequency (BC Heuristic)...")
    print("-"*70)
    
    bc_model = ActionFrequencyBaseline()
    bc_model.train(train_trajs, env)
    bc_results = bc_model.evaluate(test_trajs, INTENT_MAPPING)
    
    print(f"\nAction Frequency Results:")
    print(f"  F1 Simple:  {bc_results['simple']:.3f}")
    print(f"  F1 Complex: {bc_results['complex']:.3f}")
    print(f"  F1 Overall: {bc_results['overall']:.3f}")
    
    results["Action Frequency (BC)"] = bc_results
    
    # 2. Linear GAIL - Conservative settings
    print("\n" + "-"*70)
    print("2. Training Linear GAIL (Conservative Settings)...")
    print("-"*70)
    
    # Use conservative hyperparameters: lower lr, fewer iterations
    gail_model = LinearGAIL(n_features=21, learning_rate=0.001, gamma=0.99)
    gail_model.train(train_trajs, env, n_iterations=15)
    gail_results = gail_model.evaluate(test_trajs, INTENT_MAPPING, env=env)
    
    print(f"\nLinear GAIL Results:")
    print(f"  F1 Simple:  {gail_results['simple']:.3f}")
    print(f"  F1 Complex: {gail_results['complex']:.3f}")
    print(f"  F1 Overall: {gail_results['overall']:.3f}")
    print(f"  RRE:        {gail_results['rre']:.3f}")
    
    results["Linear GAIL"] = gail_results
    
    # 3. Chemo-IRL - Conservative settings
    print("\n" + "-"*70)
    print("3. Training Chemo-IRL (Conservative Settings)...")
    print("-"*70)
    
    # Use conservative hyperparameters: lower lr, fewer iterations
    chemo_model = ChemoIRL(n_features=21, learning_rate=0.001, gamma=0.99)
    chemo_model.train(train_trajs, env, n_iterations=15)
    chemo_results = chemo_model.evaluate(test_trajs, INTENT_MAPPING, env=env)
    
    print(f"\nChemo-IRL Results:")
    print(f"  F1 Simple:  {chemo_results['simple']:.3f}")
    print(f"  F1 Complex: {chemo_results['complex']:.3f}")
    print(f"  F1 Overall: {chemo_results['overall']:.3f}")
    print(f"  RRE:        {chemo_results['rre']:.3f}")
    
    results["Chemo-IRL"] = chemo_results
    
    # Save results
    print("\n" + "="*70)
    print("FINAL RESULTS (ALTERNATIVE SETTINGS)")
    print("="*70)
    print(json.dumps(results, indent=2))
    
    with open('conservative_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create figures directory if it doesn't exist
    os.makedirs('../figures', exist_ok=True)
    
    # Generate Figure 1: Performance Comparison Bar Chart
    print("\nGenerating performance comparison plot...")
    
    methods = list(results.keys())
    f1_scores = [results[m]['overall'] for m in methods]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    bars = ax.bar(methods, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Overall F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('Intent Classification Performance (Conservative Baselines)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('../figures/performance_bar_chart_conservative.png', dpi=300, bbox_inches='tight')
    print("  Saved to: figures/performance_bar_chart_conservative.png")
    plt.close()
    
    # Generate Figure 2: RRE Comparison
    print("Generating RRE comparison plot...")
    
    irl_methods = [m for m in methods if results[m].get('rre') is not None]
    rre_values = [results[m]['rre'] for m in irl_methods]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_rre = ['#e74c3c', '#3498db']
    bars = ax.bar(irl_methods, rre_values, color=colors_rre, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Reward Recovery Error (RRE) ↓', fontsize=14, fontweight='bold')
    ax.set_title('Reward Recovery Error Comparison', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, rre in zip(bars, rre_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rre:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('../figures/rre_comparison_conservative.png', dpi=300, bbox_inches='tight')
    print("  Saved to: figures/rre_comparison_conservative.png")
    plt.close()
    
    # Generate Figure 3: Convergence Plot for Chemo-IRL
    print("Generating convergence plot...")
    
    if chemo_model.training_history:
        history = chemo_model.training_history
        iterations = [h['iteration'] for h in history]
        param_norms = [h['param_norm'] for h in history]
        grad_norms = [h['grad_norm'] for h in history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Parameter norm
        ax1.plot(iterations, param_norms, marker='o', linewidth=2, markersize=4, color='#3498db')
        ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Parameter Norm ||θ||₂', fontsize=12, fontweight='bold')
        ax1.set_title('Chemo-IRL Parameter Convergence', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.axhline(y=param_norms[-1], color='r', linestyle='--', alpha=0.5, label=f'Final: {param_norms[-1]:.3f}')
        ax1.legend()
        
        # Gradient norm
        ax2.plot(iterations, grad_norms, marker='s', linewidth=2, markersize=4, color='#e74c3c')
        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Gradient Norm ||∇L||₂', fontsize=12, fontweight='bold')
        ax2.set_title('Gradient Norm Evolution', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.axhline(y=0.001, color='g', linestyle='--', alpha=0.5, label='Threshold ε=0.001')
        ax2.legend()
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('../figures/convergence_plot_conservative.png', dpi=300, bbox_inches='tight')
        print("  Saved to: figures/convergence_plot_conservative.png")
        plt.close()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\nGenerated Files:")
    print("  - conservative_results.json")
    print("  - figures/performance_bar_chart_conservative.png")
    print("  - figures/rre_comparison_conservative.png")
    print("  - figures/convergence_plot_conservative.png")

if __name__ == "__main__":
    main()
