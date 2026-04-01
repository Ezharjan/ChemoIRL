"""Experiment runner for Chemo-IRL and baseline comparisons."""

import numpy as np
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend
import matplotlib.pyplot as plt
import json
import os
import warnings
from chemo_irl import ChemoIRL, GymDeFi, Trajectory, generate_synthetic_expert_trajectories
from config import CHEMO_IRL_CONFIG, TRAINING_CONFIG
from baselines import (ActionFrequencyBaseline, LinearGAIL, TIMBaseline, 
                       RLPolicyBaseline, ValueDice, SQIL)
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

SIMPLE_INTENTS = ["swap", "lend", "liquidity"]
COMPLEX_INTENTS = ["yield", "complex_leverage", "complex_LP"]


def train_chemo_irl(train_trajs, test_trajs, env):
    """Train Chemo-IRL Discriminative Model"""
    print("\n" + "="*70)
    print("TRAINING CHEMO-IRL")
    print("="*70)
    
    # Initialize real IRL model
    model = ChemoIRL(
        n_features=21, 
        learning_rate=CHEMO_IRL_CONFIG['learning_rate'], 
        gamma=CHEMO_IRL_CONFIG['gamma']
    )
    
    # Train using actual MaxEnt IRL algorithm
    # This populates model.theta and model.training_history w/ real data
    model.train(train_trajs, env, n_iterations=CHEMO_IRL_CONFIG['max_iterations'])
    
    # Evaluate
    results = model.evaluate(test_trajs, INTENT_MAPPING, env=env)
    
    print(f"\nChemo-IRL Results:")
    print(f"  F1 Simple:  {results['simple']:.3f}")
    print(f"  F1 Complex: {results['complex']:.3f}")
    print(f"  F1 Overall: {results['overall']:.3f}")
    print(f"  RRE:        {results['rre']:.3f}")
    
    return model, results


def train_baselines(train_trajs, test_trajs, env):
    """Train all baseline methods."""
    print("\n" + "="*70)
    print("TRAINING BASELINE METHODS")
    print("="*70)
    
    results = {}
    
    # 1. TIM (Transaction Intent Mining)
    print("\n1. Training TIM (Transaction Intent Mining)...")
    tim_model = TIMBaseline()
    tim_model.train(train_trajs, env)
    tim_results = tim_model.evaluate(test_trajs, INTENT_MAPPING)
    print(f"  TIM Results: F1={tim_results['overall']:.3f}")
    results["TIM"] = tim_results
    
    # 2. BC (Behavioral Cloning - Action Frequency Heuristic)
    print("\n2. Training BC (Action Frequency Baseline)...")
    bc_model = ActionFrequencyBaseline()
    bc_model.train(train_trajs, env)
    bc_results = bc_model.evaluate(test_trajs, INTENT_MAPPING)
    print(f"  BC Results: F1={bc_results['overall']:.3f}")
    results["BC"] = bc_results
    
    # 3. GAIL (Generative Adversarial Imitation Learning)
    print("\n3. Training GAIL...")
    gail_model = LinearGAIL(n_features=21, learning_rate=0.05, gamma=0.99)
    gail_model.train(train_trajs, env, n_iterations=30)
    gail_results = gail_model.evaluate(test_trajs, INTENT_MAPPING, env=env)
    print(f"  GAIL Results: F1={gail_results['overall']:.3f}, RRE={gail_results['rre']:.3f}")
    results["GAIL"] = gail_results
    
    # 4. PPO (Proximal Policy Optimization)
    print("\n4. Training PPO...")
    ppo_model = RLPolicyBaseline(method_name='PPO')
    ppo_model.train(train_trajs, env)
    ppo_results = ppo_model.evaluate(test_trajs, INTENT_MAPPING)
    print(f"  PPO Results: F1={ppo_results['overall']:.3f}")
    results["PPO"] = ppo_results
    
    # 5. DQN (Deep Q-Network)
    print("\n5. Training DQN...")
    dqn_model = RLPolicyBaseline(method_name='DQN')
    dqn_model.train(train_trajs, env)
    dqn_results = dqn_model.evaluate(test_trajs, INTENT_MAPPING)
    print(f"  DQN Results: F1={dqn_results['overall']:.3f}")
    results["DQN"] = dqn_results
    
    # 6. A2C (Advantage Actor-Critic)
    print("\n6. Training A2C...")
    a2c_model = RLPolicyBaseline(method_name='A2C')
    a2c_model.train(train_trajs, env)
    a2c_results = a2c_model.evaluate(test_trajs, INTENT_MAPPING)
    print(f"  A2C Results: F1={a2c_results['overall']:.3f}")
    results["A2C"] = a2c_results
    
    # 7. TRPO (Trust Region Policy Optimization)
    print("\n7. Training TRPO...")
    trpo_model = RLPolicyBaseline(method_name='TRPO')
    trpo_model.train(train_trajs, env)
    trpo_results = trpo_model.evaluate(test_trajs, INTENT_MAPPING)
    print(f"  TRPO Results: F1={trpo_results['overall']:.3f}")
    results["TRPO"] = trpo_results
    
    # 8. SQIL (Soft Q Imitation Learning)
    print("\n8. Training SQIL...")
    sqil_model = SQIL(n_features=21, learning_rate=0.02, gamma=0.99)
    sqil_model.train(train_trajs, env, n_iterations=25)
    sqil_results = sqil_model.evaluate(test_trajs, INTENT_MAPPING, env=env)
    rre_val = f"{sqil_results.get('rre', 'N/A'):.3f}" if sqil_results.get('rre') is not None else "N/A"
    print(f"  SQIL Results: F1={sqil_results['overall']:.3f}, RRE={rre_val}")
    results["SQIL"] = sqil_results
    
    # 9. ValueDice
    print("\n9. Training ValueDice...")
    valuedice_model = ValueDice(n_features=21, learning_rate=0.02, gamma=0.99)
    valuedice_model.train(train_trajs, env, n_iterations=25)
    valuedice_results = valuedice_model.evaluate(test_trajs, INTENT_MAPPING, env=env)
    print(f"  ValueDice Results: F1={valuedice_results['overall']:.3f}, RRE={valuedice_results['rre']:.3f}")
    results["ValueDice"] = valuedice_results
    
    # 10. SAC (Soft Actor-Critic)
    print("\n10. Training SAC...")
    sac_model = RLPolicyBaseline(method_name='SAC')
    sac_model.train(train_trajs, env)
    sac_results = sac_model.evaluate(test_trajs, INTENT_MAPPING)
    print(f"  SAC Results: F1={sac_results['overall']:.3f}")
    results["SAC"] = sac_results
    
    # 11. TD3 (Twin Delayed DDPG)
    print("\n11. Training TD3...")
    td3_model = RLPolicyBaseline(method_name='TD3')
    td3_model.train(train_trajs, env)
    td3_results = td3_model.evaluate(test_trajs, INTENT_MAPPING)
    print(f"  TD3 Results: F1={td3_results['overall']:.3f}")
    results["TD3"] = td3_results
    
    return results


def plot_performance_comparison(all_results, output_dir='../figures'):
    """Generate performance bar chart for all methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating performance comparison plot...")
    
    methods = list(all_results.keys())
    simple_scores = [all_results[m]['simple'] for m in methods]
    complex_scores = [all_results[m]['complex'] for m in methods]
    overall_scores = [all_results[m]['overall'] for m in methods]
    
    # Sort by overall score
    sorted_idx = sorted(range(len(overall_scores)), key=lambda i: overall_scores[i], reverse=True)
    methods = [methods[i] for i in sorted_idx]
    simple_scores = [simple_scores[i] for i in sorted_idx]
    complex_scores = [complex_scores[i] for i in sorted_idx]
    overall_scores = [overall_scores[i] for i in sorted_idx]
    
    # Create figure with larger size to accommodate all methods
    fig, ax = plt.subplots(figsize=(18, 10))
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax.bar(x - width, simple_scores, width, label='Simple Intents',
                   color='#6A994E', edgecolor='black', linewidth=1.2, alpha=0.85)
    bars2 = ax.bar(x, complex_scores, width, label='Complex Intents',
                   color='#C73E1D', edgecolor='black', linewidth=1.2, alpha=0.85)
    bars3 = ax.bar(x + width, overall_scores, width, label='Overall F1',
                   color='#2E86AB', edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_title('Intent Classification Performance (All Baselines)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Highlight Chemo-IRL
    if 'Chemo-IRL' in methods:
        chemo_idx = methods.index('Chemo-IRL')
        ax.axvline(x=chemo_idx, color='gold', linestyle='--', linewidth=2.5, alpha=0.6)
        rect = plt.Rectangle((chemo_idx - 0.5, 0), 1, 1.05,
                            facecolor='gold', alpha=0.1, edgecolor='gold', linewidth=2)
        ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_bar_chart.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/performance_bar_chart.png")
    plt.close()


def plot_rre_comparison(all_results, output_dir='../figures'):
    """Plot RRE comparison for IRL methods"""
    print("Generating RRE comparison plot...")
    
    irl_methods = [m for m in all_results if all_results[m]['rre'] is not None]
    rre_values = [all_results[m]['rre'] for m in irl_methods]
    
    sorted_idx = sorted(range(len(rre_values)), key=lambda i: rre_values[i])
    irl_methods = [irl_methods[i] for i in sorted_idx]
    rre_values = [rre_values[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2E86AB' if m == 'Chemo-IRL' else '#F18F01' if 'GAIL' in m else '#9B59B6' 
              for m in irl_methods]
    
    bars = ax.bar(irl_methods, rre_values, color=colors,
                  edgecolor='black', linewidth=1.8, alpha=0.85, width=0.6)
    
    ax.set_ylabel('Reward Recovery Error (RRE)', fontsize=14, fontweight='bold')
    ax.set_xlabel('IRL Method', fontsize=14, fontweight='bold')
    # Title removed to reduce whitespace and follow academic standards
    # ax.set_title('Reward Recovery Error: Lower is Better', fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(rre_values) * 1.15)
    
    for bar, val in zip(bars, rre_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{val:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rre_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/rre_comparison.png")
    plt.close()


def plot_robustness_analysis(all_results, output_dir='../figures'):
    """Generate robustness to noise plot"""
    print("Generating robustness analysis plot...")
    print("Robustness plotting is disabled in this script.")
    return


def plot_ablation_study(output_dir='../figures'):
    """Generate ablation study plot"""
    print("Generating ablation study plot...")

    print("Ablation plotting is disabled in this script.")
    return


def plot_convergence(model, output_dir='../figures'):
    """Plot IRL convergence curves (Deep Reward Version) including parameters"""
    print("Generating convergence plot...")
    
    history = model.training_history
    epochs = [h['iteration'] for h in history]
    losses = [h['loss'] for h in history]
    expert_rewards = [h['expert_reward'] for h in history]
    agent_rewards = [h['agent_reward'] for h in history]
    
    # Handle missing metrics for backward compatibility
    grad_norms = [h.get('grad_norm', 0.0) for h in history]
    param_norms = [h.get('param_norm', 0.0) for h in history]
    
    # Create 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss
    axs[0, 0].plot(epochs, losses, '-o', color='#E63946', linewidth=2.5, markersize=6)
    axs[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axs[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axs[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # 2. Rewards
    axs[0, 1].plot(epochs, expert_rewards, '-o', color='#2A9D8F', linewidth=2.0, label='Expert R')
    axs[0, 1].plot(epochs, agent_rewards, '-s', color='#F4A261', linewidth=2.0, label='Agent R')
    axs[0, 1].set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    axs[0, 1].set_title('Reward Convergence', fontsize=14, fontweight='bold')
    axs[0, 1].grid(True, alpha=0.3, linestyle='--')
    axs[0, 1].legend(loc='lower right')
    
    # 3. Gradient Norm
    axs[1, 0].plot(epochs, grad_norms, '-^', color='#457B9D', linewidth=2.0)
    axs[1, 0].set_xlabel('IRL Iteration', fontsize=12, fontweight='bold')
    axs[1, 0].set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
    axs[1, 0].set_title('Gradient Convergence', fontsize=14, fontweight='bold')
    axs[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # 4. Parameter Norm
    axs[1, 1].plot(epochs, param_norms, '-d', color='#1D3557', linewidth=2.0)
    axs[1, 1].set_xlabel('IRL Iteration', fontsize=12, fontweight='bold')
    axs[1, 1].set_ylabel('Parameter Norm', fontsize=12, fontweight='bold')
    axs[1, 1].set_title('Parameter Stability', fontsize=14, fontweight='bold')
    axs[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/convergence_plot.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/convergence_plot.png")
    plt.close()


def main():
    """Main experiment pipeline"""
    print("\n" + "="*70)
    print("CHEMO-IRL EXPERIMENT PIPELINE")
    print("="*70)
    
    # Create environment
    env = GymDeFi(state_dim=64, action_dim=1000, obfuscated=True)
    
    # Generate synthetic expert trajectories
    print("\nGenerating synthetic expert trajectories (Obfuscated)...")
    all_intents = SIMPLE_INTENTS + COMPLEX_INTENTS
    # Use obfuscation_level=0.7 for the benchmark configuration.
    expert_trajs = generate_synthetic_expert_trajectories(env, n_trajectories=TRAINING_CONFIG['n_expert_trajectories'],
                                                           intent_types=all_intents,
                                                           obfuscation_level=0.7)
    
    # Split train/test (80/20)
    n_train = int(0.8 * len(expert_trajs))
    train_trajs = expert_trajs[:n_train]
    test_trajs = expert_trajs[n_train:]
    
    print(f"  Training trajectories: {len(train_trajs)}")
    print(f"  Testing trajectories:  {len(test_trajs)}")
    
    # Train Chemo-IRL
    chemo_model, chemo_results = train_chemo_irl(train_trajs, test_trajs, env)
    
    # Train baselines
    baseline_results = train_baselines(train_trajs, test_trajs, env)
    
    # Combine results
    all_results = {"Chemo-IRL": chemo_results}
    all_results.update(baseline_results)
    
    # Save results
    print("\nSaving results...")
    with open('evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("  Saved: evaluation_results.json")
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_performance_comparison(all_results)
    plot_rre_comparison(all_results)
    plot_robustness_analysis(all_results)
    plot_ablation_study() # Skipped
    plot_convergence(chemo_model)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nResults Summary:")
    print(f"  Chemo-IRL Overall F1: {chemo_results['overall']:.3f}")
    print(f"  Chemo-IRL RRE:        {chemo_results['rre']:.3f}")
    
    if baseline_results:
        print(f"  Top Baseline (F1):    {max(baseline_results, key=lambda k: baseline_results[k]['overall'])}")
        
    print(f"\nAll figures saved to: ../figures/")
    print(f"Results saved to: evaluation_results.json")
    print()


if __name__ == "__main__":
    main()
