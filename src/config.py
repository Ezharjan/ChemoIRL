"""
Configuration file for Chemo-IRL experiments
Centralizes all hyperparameters and settings
"""

# =============================================================================
# Environment Configuration
# =============================================================================

ENV_CONFIG = {
    'state_dim': 64,
    'action_dim': 1000,
    'user_state_dim': 20,
    'market_state_dim': 22,
    'gradient_dim': 22,
    'n_protocols': 5,
    'n_functions': 20,
    'n_value_bins': 10,
    'block_time': 12.0,  # Ethereum block time in seconds
}

# =============================================================================
# Chemo-IRL Configuration
# =============================================================================

CHEMO_IRL_CONFIG = {
    'n_features': 21,
    'learning_rate': 0.005,
    'gamma': 0.99,
    'epsilon': 1e-4,  # Convergence threshold
    'max_iterations': 60,
    'n_agent_trajectories': 40,
    'max_trajectory_steps': 15,
}

# =============================================================================
# Training Configuration
# =============================================================================

TRAINING_CONFIG = {
    'n_expert_trajectories': 200,  # Increased from 50 to 200 for better stability
    'train_test_split': 0.8,
    'random_seed': 42,
    'validation_split': 0.2,  # For hyperparameter tuning
}

# =============================================================================
# Baseline Algorithms Configuration
# =============================================================================
# Baselines are configured in src/baselines.py and orchestrated in
# src/run_experiments.py.

LINEAR_GAIL_CONFIG = {
    'learning_rate': 0.05,
    'n_iterations': 30,
    'gamma': 0.99,
}

# =============================================================================
# Intent Taxonomy Configuration
# =============================================================================

INTENT_MAPPING = {
    "swap": 0,
    "lend": 1,
    "liquidity": 2,
    "yield": 3,
    "complex_leverage": 4,
    "complex_LP": 5,
}

INTENT_NAMES = list(INTENT_MAPPING.keys())

SIMPLE_INTENTS = ["swap", "lend", "liquidity"]
COMPLEX_INTENTS = ["yield", "complex_leverage", "complex_LP"]

# Feature taxonomy aligned with TIM framework
FEATURE_NAMES = [
    # Swap indicators (0-2)
    'dex_swap_indicator',
    'price_impact',
    'swap_size',
    
    # Lending indicators (3-6)
    'deposit_indicator',
    'borrow_indicator',
    'collateral_ratio',
    'liquidation_risk',
    
    # Liquidity indicators (7-9)
    'add_liquidity',
    'remove_liquidity',
    'lp_token_value',
    
    # Yield indicators (10-12)
    'apy_signal',
    'farming_indicator',
    'staking_indicator',
    
    # Governance indicators (13-14)
    'voting_indicator',
    'proposal_indicator',
    
    # Risk indicators (15-17)
    'volatility_exposure',
    'leverage_indicator',
    'concentration_risk',
    
    # Temporal/MEV indicators (18-20)
    'time_sensitivity',
    'mev_exposure',
    'gas_efficiency',
]

# =============================================================================
# Visualization Configuration
# =============================================================================

PLOT_CONFIG = {
    'dpi': 300,
    'figsize_single': (10, 7),
    'figsize_double': (14, 6),
    'figsize_wide': (16, 8),
    'output_dir': '../figures',
    'color_scheme': {
        'chemo_irl': '#2E86AB',
        'gail': '#F18F01',
        'valuedice': '#9B59B6',
        'bc': '#C73E1D',
        'ppo': '#6A994E',
        'simple_intents': '#6A994E',
        'complex_intents': '#C73E1D',
        'overall': '#2E86AB',
    },
    'font_sizes': {
        'title': 16,
        'axis_label': 14,
        'tick_label': 12,
        'legend': 12,
        'annotation': 10,
    }
}

# =============================================================================
# Evaluation Configuration
# =============================================================================

EVAL_CONFIG = {
    'metrics': ['f1_macro', 'f1_micro', 'precision', 'recall', 'accuracy'],
    'compute_confusion_matrix': True,
    'compute_per_class_metrics': True,
    'save_predictions': True,
}

# =============================================================================
# Robustness Analysis Configuration
# =============================================================================
# Robustness analysis parameters can be added here when needed.

# =============================================================================
# Ablation Study Configuration
# =============================================================================

ABLATION_CONFIG = {
    'variants': [
        'full',
        'without_temporal_gradients',
        'random_features',
    ],
    'learning_rate': 0.01,
    'n_iterations': 15,
}


# =============================================================================
# Output Paths
# =============================================================================

OUTPUT_PATHS = {
    'results': 'evaluation_results.json',
    'model_checkpoint': 'checkpoints/chemo_irl_model.json',
    'training_history': 'logs/training_history.json',
    'figures': '../figures/',
    'logs': 'logs/',
}

# =============================================================================
# Protocol and Function Mappings
# =============================================================================

PROTOCOL_NAMES = [
    'Uniswap',
    'Aave',
    'Compound',
    'Curve',
    'Balancer',
]

FUNCTION_SIGNATURES = [
    'swap',
    'swapExactTokensForTokens',
    'swapTokensForExactTokens',
    'deposit',
    'borrow',
    'repay',
    'withdraw',
    'addLiquidity',
    'addLiquidityETH',
    'removeLiquidity',
    'stake',
    'unstake',
    'claimRewards',
    'vote',
    'propose',
    'flashLoan',
    'liquidate',
    'setApproval',
    'transferFrom',
    'multicall',
]

# =============================================================================
# Data Generation Configuration
# =============================================================================

SYNTHETIC_DATA_CONFIG = {
    'trajectory_length_range': (5, 15),
    'state_noise_std': 0.1,
    'market_volatility': 0.05,
    'mean_reversion_strength': 0.05,
    'price_distribution': 'lognormal',
    'price_mean': 0.0,
    'price_std': 0.5,
}

# =============================================================================
# Computational Resources
# =============================================================================

COMPUTE_CONFIG = {
    'device': 'cpu',  # 'cpu' or 'cuda'
    'n_workers': 4,
    'verbose': 1,
}
