"""
Utility functions for Chemo-IRL framework
Provides helper functions for data processing, metrics computation, and visualization
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import json


def extract_semantic_features(state: np.ndarray, action: int, n_features: int = 21, decoding_map: Dict = None) -> np.ndarray:
    """
    Extract semantic features from state-action pair.
    
    Shared feature extraction logic for Chemo-IRL and baselines.
    Args:
        state: The environment state
        action: The action identifier
        n_features: Number of features to extract
        decoding_map: Optional dictionary mapping action_id -> (protocol_id, function_id, value_bin)
                      If None, uses default fixed mapping (0-200..).
    """
    features = np.zeros(n_features)
    
    # Convert action to scalar if needed
    if isinstance(action, np.ndarray):
        if action.size == 1:
            action = int(action.item())
        else:
            action = int(action[0])  # Take first element
    else:
        action = int(action)
    
    # Extract user state (first 20 dims), market state (next 22), gradient (last 22)
    state_dim = len(state)
    user_state = state[:min(20, state_dim)]
    market_state = state[20:min(42, state_dim)] if state_dim > 20 else np.zeros(22)
    gradient = state[42:min(64, state_dim)] if state_dim > 42 else np.zeros(22)
    
    # Decode action (hierarchical: protocol, function, value)
    if decoding_map is not None and action in decoding_map:
        protocol_id, function_id, value_bin = decoding_map[action]
    else:
        # Default legacy mapping
        protocol_id = action // 200  # 0-4 for 5 protocols
        function_id = (action % 200) // 10  # 0-19 for 20 functions
        value_bin = action % 10  # 0-9 for value sizes
    
    # Feature 0-2: Swap indicators
    if function_id in [0, 1, 2]:  # swap-related functions
        features[0] = 0.8 + 0.2 * np.random.rand()  # DEX swap indicator
        features[1] = min(1.0, abs(gradient[:5]).mean())  # Price impact
        features[2] = value_bin / 10.0  # Swap size
        
    # Feature 3-6: Lending indicators
    if function_id in [3, 4, 5, 6]:  # lending-related
        features[3] = 0.7 + 0.3 * np.random.rand() if function_id == 3 else 0.0  # Deposit
        features[4] = 0.7 + 0.3 * np.random.rand() if function_id == 4 else 0.0  # Borrow
        features[5] = user_state[:5].mean() if len(user_state) >= 5 else 0.5  # Collateral
        features[6] = abs(gradient[5:10]).mean() if len(gradient) >= 10 else 0.0  # Liquidation risk
        
    # Feature 7-9: Liquidity indicators  
    if function_id in [7, 8, 9]:  # liquidity-related
        features[7] = 0.8 if function_id == 7 else 0.0  # Add liquidity
        features[8] = 0.8 if function_id == 8 else 0.0  # Remove liquidity
        features[9] = market_state[:5].std() if len(market_state) >= 5 else 0.5  # LP value
        
    # Feature 10-12: Yield indicators
    if function_id in [10, 11, 12]:  # yield farming
        features[10] = market_state[10:15].mean() if len(market_state) >= 15 else 0.6  # APY
        features[11] = 0.7 if function_id == 11 else 0.0  # Farming
        features[12] = 0.7 if function_id == 12 else 0.0  # Staking
        
    # Feature 13-14: Governance
    if function_id in [13, 14]:
        features[13] = 0.9 if function_id == 13 else 0.0  # Vote
        features[14] = 0.9 if function_id == 14 else 0.0  # Propose
        
    # Feature 15-17: Risk indicators
    features[15] = abs(gradient).mean()  # Volatility exposure
    features[16] = value_bin / 5.0  # Leverage indicator
    features[17] = 1.0 - user_state[:10].std() if len(user_state) >= 10 else 0.5  # Concentration
    
    # Feature 18-20: Temporal/MEV indicators
    features[18] = abs(gradient[:3]).max() if len(gradient) >= 3 else 0.0  # Time sensitivity
    features[19] = features[0] * features[18]  # MEV exposure (swap * time_sensitive)
    features[20] = 1.0 - (value_bin / 10.0)  # Gas efficiency (smaller = more efficient)
    
    # Normalize features to [0, 1]
    features = np.clip(features, 0, 1)
    
    return features


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   labels: List[int] = None) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label indices to include
        
    Returns:
        Dictionary containing F1, precision, recall, and accuracy
    """
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    metrics = {
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        'f1_micro': float(f1_score(y_true, y_pred, average='micro', zero_division=0, labels=labels)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        'accuracy': float(np.mean(y_true == y_pred))
    }
    
    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             intent_names: List[str] = None) -> Dict:
    """
    Compute and format confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        intent_names: List of intent names for labeling
        
    Returns:
        Dictionary with confusion matrix and related statistics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Compute per-class metrics
    per_class_metrics = {}
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_name = intent_names[i] if intent_names and i < len(intent_names) else f"Class_{i}"
        per_class_metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(cm[i, :].sum())
        }
    
    return {
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class_metrics
    }


def compute_reward_recovery_error(learned_rewards: np.ndarray, 
                                  true_rewards: np.ndarray,
                                  normalize: bool = True) -> float:
    """
    Compute Reward Recovery Error (RRE)
    
    Args:
        learned_rewards: Rewards from learned policy
        true_rewards: Ground truth rewards
        normalize: Whether to normalize by true reward magnitude
        
    Returns:
        RRE value (lower is better)
    """
    diff = np.abs(learned_rewards - true_rewards)
    
    if normalize:
        norm_factor = np.linalg.norm(true_rewards) + 1e-8
        rre = np.linalg.norm(diff) / norm_factor
    else:
        rre = np.mean(diff)
    
    return float(rre)


def compute_feature_statistics(features: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute statistics of feature distributions
    
    Args:
        features: Feature matrix (n_samples, n_features)
        
    Returns:
        Dictionary with mean, std, min, max, median for each feature
    """
    return {
        'mean': np.mean(features, axis=0),
        'std': np.std(features, axis=0),
        'min': np.min(features, axis=0),
        'max': np.max(features, axis=0),
        'median': np.median(features, axis=0),
        'q25': np.percentile(features, 25, axis=0),
        'q75': np.percentile(features, 75, axis=0)
    }


def validate_trajectory(trajectory, state_dim: int, action_dim: int) -> bool:
    """
    Validate trajectory data structure
    
    Args:
        trajectory: Trajectory object to validate
        state_dim: Expected state dimensionality
        action_dim: Maximum action value
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required attributes
        if not hasattr(trajectory, 'states') or not hasattr(trajectory, 'actions'):
            return False
        
        # Check dimensions
        if len(trajectory.states.shape) != 2:
            return False
        
        if trajectory.states.shape[1] != state_dim:
            return False
        
        # Check actions are within valid range
        if np.any(trajectory.actions < 0) or np.any(trajectory.actions >= action_dim):
            return False
        
        # Check trajectory length consistency
        if len(trajectory.actions) != len(trajectory.states) - 1:
            # Last state has no corresponding action
            if len(trajectory.actions) != len(trajectory.states):
                return False
        
        return True
    
    except Exception:
        return False


def split_train_test(trajectories: List, train_ratio: float = 0.8, 
                     shuffle: bool = True, random_seed: int = 42) -> Tuple[List, List]:
    """
    Split trajectories into train and test sets
    
    Args:
        trajectories: List of trajectory objects
        train_ratio: Ratio of training data
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_trajectories, test_trajectories)
    """
    n_total = len(trajectories)
    n_train = int(n_total * train_ratio)
    
    indices = np.arange(n_total)
    
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_trajs = [trajectories[i] for i in train_indices]
    test_trajs = [trajectories[i] for i in test_indices]
    
    return train_trajs, test_trajs


def save_results(results: Dict, filepath: str, indent: int = 2):
    """
    Save results dictionary to JSON file
    
    Args:
        results: Dictionary to save
        filepath: Output file path
        indent: JSON indentation level
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=indent)


def load_results(filepath: str) -> Dict:
    """
    Load results from JSON file
    
    Args:
        filepath: Input file path
        
    Returns:
        Dictionary of results
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_statistical_significance(results_a: np.ndarray, results_b: np.ndarray,
                                     test: str = 't-test') -> Dict:
    """
    Compute statistical significance between two sets of results
    
    Args:
        results_a: First set of results
        results_b: Second set of results
        test: Type of test ('t-test' or 'wilcoxon')
        
    Returns:
        Dictionary with test statistics and p-value
    """
    from scipy import stats
    
    if test == 't-test':
        statistic, pvalue = stats.ttest_ind(results_a, results_b)
        test_name = "Independent t-test"
    elif test == 'wilcoxon':
        statistic, pvalue = stats.wilcoxon(results_a, results_b)
        test_name = "Wilcoxon signed-rank test"
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return {
        'test': test_name,
        'statistic': float(statistic),
        'p_value': float(pvalue),
        'significant_at_0.05': pvalue < 0.05,
        'significant_at_0.01': pvalue < 0.01
    }


def normalize_features(features: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize features
    
    Args:
        features: Feature matrix (n_samples, n_features)
        method: Normalization method ('minmax', 'standard', or 'l2')
        
    Returns:
        Normalized features
    """
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        min_vals = np.min(features, axis=0)
        max_vals = np.max(features, axis=0)
        normalized = (features - min_vals) / (max_vals - min_vals + 1e-8)
    
    elif method == 'standard':
        # Z-score normalization
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        normalized = (features - mean) / (std + 1e-8)
    
    elif method == 'l2':
        # L2 normalization
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        normalized = features / (norms + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def create_intent_distribution_summary(trajectories: List, 
                                       intent_mapping: Dict[str, int]) -> Dict:
    """
    Create summary of intent distribution in dataset
    
    Args:
        trajectories: List of trajectories
        intent_mapping: Mapping from intent names to indices
        
    Returns:
        Dictionary with distribution statistics
    """
    intent_counts = {name: 0 for name in intent_mapping.keys()}
    
    for traj in trajectories:
        if hasattr(traj, 'intent_label') and traj.intent_label in intent_counts:
            intent_counts[traj.intent_label] += 1
    
    total = sum(intent_counts.values())
    
    distribution = {
        'counts': intent_counts,
        'percentages': {k: v/total*100 if total > 0 else 0 for k, v in intent_counts.items()},
        'total_trajectories': total
    }
    
    return distribution
