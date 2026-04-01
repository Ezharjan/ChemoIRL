import numpy as np
import warnings
import json
import sys
from chemo_irl import ChemoIRL, GymDeFi, generate_synthetic_expert_trajectories
from utils import extract_semantic_features as original_extractor

warnings.filterwarnings('ignore')
np.random.seed(42)

def no_gradient_extractor(state, action, n_features=21):
    """Wrapper that zeros out temporal gradient part of state before feature extraction"""
    state_copy = state.copy()
    # Temporal gradient occupies indices 42:64 in the 64D state.
    if len(state_copy) >= 64:
        state_copy[42:64] = 0.0
    return original_extractor(state_copy, action, n_features)

def random_extractor(state, action, n_features=21):
    """Returns random features for ablation runs."""
    return np.random.rand(n_features)

class ChemoIRLAblation(ChemoIRL):
    """ChemoIRL variant that allows custom feature extraction"""
    def __init__(self, n_features=21, learning_rate=0.01, gamma=0.99, feature_extractor=None):
        super().__init__(n_features, learning_rate, gamma)
        self.feature_extractor = feature_extractor if feature_extractor else original_extractor
    
    def extract_features(self, state, action):
        """Use custom feature extractor"""
        return self.feature_extractor(state, action, self.n_features)

def run_ablation_experiment():
    print("Generating Synthetic Data...")
    env = GymDeFi()
    intent_types = ["swap", "lend", "liquidity", "yield", "complex_leverage", "complex_LP"]
    
    # Generate data
    train_trajs = generate_synthetic_expert_trajectories(env, n_trajectories=100, intent_types=intent_types)
    test_trajs = generate_synthetic_expert_trajectories(env, n_trajectories=50, intent_types=intent_types)
    
    mapping = {
        "swap": 0, "lend": 1, "liquidity": 2, "yield": 3, 
        "complex_leverage": 4, "complex_LP": 5
    }
    
    results = {}
    
    # 1. Full Model (with all features including temporal gradients)
    print("\nRunning Full Model (Chemo-IRL)...")
    model = ChemoIRLAblation(n_features=21, learning_rate=0.01, gamma=0.99, feature_extractor=original_extractor)
    model.train(train_trajs, env, n_iterations=15)
    results['Full Model (Chemo-IRL)'] = model.evaluate(test_trajs, mapping, env=env)
    
    # 2. w/o Temporal Gradients
    print("\nRunning Ablation: w/o Temporal Gradients...")
    model = ChemoIRLAblation(n_features=21, learning_rate=0.01, gamma=0.99, feature_extractor=no_gradient_extractor)
    model.train(train_trajs, env, n_iterations=15)
    results['w/o Temporal Gradients'] = model.evaluate(test_trajs, mapping, env=env)
    
    # 3. Random Features
    print("\nRunning Ablation: Random Features...")
    model = ChemoIRLAblation(n_features=21, learning_rate=0.01, gamma=0.99, feature_extractor=random_extractor)
    model.train(train_trajs, env, n_iterations=15)
    results['Random Features'] = model.evaluate(test_trajs, mapping, env=env)
    
    print("\n" + "="*50)
    print("ABLATION RESULTS")
    print("="*50)
    print(json.dumps(results, indent=2))
    
    # Save to file
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_ablation_experiment()
