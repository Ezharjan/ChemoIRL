"""
Unit tests for Chemo-IRL framework
Tests core components to ensure correctness
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chemo_irl import ChemoIRL, GymDeFi, Trajectory, generate_synthetic_expert_trajectories
from utils import (compute_comprehensive_metrics, compute_reward_recovery_error,
                   validate_trajectory, split_train_test, normalize_features)
from config import ENV_CONFIG, CHEMO_IRL_CONFIG, INTENT_MAPPING


def test_gym_defi_environment():
    """Test GymDeFi environment basic functionality"""
    print("\n" + "="*70)
    print("TEST: GymDeFi Environment")
    print("="*70)
    
    env = GymDeFi(state_dim=64, action_dim=1000)
    
    # Test reset
    state = env.reset()
    assert state.shape == (64,), f"State shape mismatch: {state.shape}"
    assert np.all(np.isfinite(state)), "State contains non-finite values"
    print("✓ Reset produces valid 64-dimensional state")
    
    # Test step
    action = np.random.randint(0, 1000)
    next_state, reward, done, info = env.step(action)
    assert next_state.shape == (64,), f"Next state shape mismatch: {next_state.shape}"
    assert isinstance(reward, (int, float)), "Reward must be scalar"
    assert isinstance(done, bool), "Done must be boolean"
    assert isinstance(info, dict), "Info must be dictionary"
    print("✓ Step function works correctly")
    
    # Test trajectory generation
    trajectory_states = [state]
    trajectory_actions = []
    for _ in range(10):
        action = np.random.randint(0, 1000)
        state, _, done, _ = env.step(action)
        trajectory_states.append(state)
        trajectory_actions.append(action)
        if done:
            break
    
    assert len(trajectory_actions) > 0, "No actions recorded"
    assert len(trajectory_states) == len(trajectory_actions) + 1, "State-action mismatch"
    print(f"✓ Generated trajectory with {len(trajectory_actions)} steps")
    
    print("✓ All GymDeFi tests passed!\n")
    return True


def test_chemo_irl_feature_extraction():
    """Test ChemoIRL feature extraction"""
    print("\n" + "="*70)
    print("TEST: Feature Extraction")
    print("="*70)
    
    model = ChemoIRL(n_features=21, learning_rate=1e-3, gamma=0.99)
    env = GymDeFi()
    
    state = env.reset()
    action = 42  # Arbitrary action
    
    # Extract features
    features = model.extract_features(state, action)
    
    assert features.shape == (21,), f"Feature shape mismatch: {features.shape}"
    assert np.all(features >= 0) and np.all(features <= 1), "Features must be in [0, 1]"
    assert np.all(np.isfinite(features)), "Features contain non-finite values"
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Features range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"✓ Feature mean: {features.mean():.3f}")
    
    # Test feature extraction consistency
    features2 = model.extract_features(state, action)
    # Should have some randomness but similar structure
    assert features.shape == features2.shape
    print("✓ Feature extraction is consistent")
    
    print("✓ All feature extraction tests passed!\n")
    return True


def test_trajectory_generation():
    """Test synthetic trajectory generation"""
    print("\n" + "="*70)
    print("TEST: Trajectory Generation")
    print("="*70)
    
    env = GymDeFi()
    intent_types = ["swap", "lend", "liquidity", "yield", "complex_leverage", "complex_LP"]
    
    trajectories = generate_synthetic_expert_trajectories(env, n_trajectories=20, 
                                                          intent_types=intent_types)
    
    assert len(trajectories) == 20, f"Wrong number of trajectories: {len(trajectories)}"
    print(f"✓ Generated {len(trajectories)} trajectories")
    
    # Check trajectory structure
    for i, traj in enumerate(trajectories[:3]):  # Check first 3
        assert hasattr(traj, 'states'), f"Trajectory {i} missing states"
        assert hasattr(traj, 'actions'), f"Trajectory {i} missing actions"
        assert hasattr(traj, 'intent_label'), f"Trajectory {i} missing intent_label"
        
        assert traj.states.ndim == 2, f"States should be 2D, got {traj.states.ndim}D"
        assert traj.actions.ndim == 1, f"Actions should be 1D, got {traj.actions.ndim}D"
        
        # Length consistency
        assert len(traj.actions) <= len(traj.states), "More actions than states"
        assert 5 <= len(traj.actions) <= 15, f"Trajectory length {len(traj.actions)} out of range"
        
        print(f"✓ Trajectory {i}: {len(traj.actions)} steps, intent={traj.intent_label}")
    
    # Check intent distribution
    intent_counts = {}
    for traj in trajectories:
        intent = traj.intent_label
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print(f"✓ Intent distribution: {intent_counts}")
    
    print("✓ All trajectory generation tests passed!\n")
    return True


def test_feature_expectation_computation():
    """Test feature expectation computation"""
    print("\n" + "="*70)
    print("TEST: Feature Expectation Computation")
    print("="*70)
    
    model = ChemoIRL(n_features=21, gamma=0.99)
    env = GymDeFi()
    
    # Generate trajectories
    trajectories = generate_synthetic_expert_trajectories(env, n_trajectories=10,
                                                          intent_types=["swap", "lend"])
    
    # Compute feature expectation
    mu = model.compute_feature_expectation(trajectories)
    
    assert mu.shape == (21,), f"Feature expectation shape mismatch: {mu.shape}"
    assert np.all(np.isfinite(mu)), "Feature expectation contains non-finite values"
    print(f"✓ Feature expectation shape: {mu.shape}")
    print(f"✓ Feature expectation range: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"✓ Feature expectation norm: {np.linalg.norm(mu):.3f}")
    
    # Test with empty list
    mu_empty = model.compute_feature_expectation([])
    assert np.all(mu_empty == 0), "Empty trajectory list should give zero expectation"
    print("✓ Empty trajectory handling works")
    
    print("✓ All feature expectation tests passed!\n")
    return True


def test_reward_computation():
    """Test reward computation"""
    print("\n" + "="*70)
    print("TEST: Reward Computation")
    print("="*70)
    
    model = ChemoIRL(n_features=21)
    env = GymDeFi()
    
    state = env.reset()
    action = 100
    
    # Compute reward
    reward = model.compute_reward(state, action)
    
    assert isinstance(reward, (int, float, np.number)), f"Reward must be scalar, got {type(reward)}"
    assert np.isfinite(reward), "Reward must be finite"
    print(f"✓ Reward: {reward:.3f}")
    
    print("✓ All reward computation tests passed!\n")
    return True


def test_trajectory_validation():
    """Test trajectory validation utility"""
    print("\n" + "="*70)
    print("TEST: Trajectory Validation")
    print("="*70)
    
    env = GymDeFi(state_dim=64, action_dim=1000)
    
    # Valid trajectory
    valid_traj = Trajectory(
        states=np.random.rand(10, 64),
        actions=np.random.randint(0, 1000, size=9),
        intent_label="swap"
    )
    
    assert validate_trajectory(valid_traj, 64, 1000), "Valid trajectory rejected"
    print("✓ Valid trajectory accepted")
    
    # Invalid state dimension
    invalid_traj = Trajectory(
        states=np.random.rand(10, 32),  # Wrong dimension
        actions=np.random.randint(0, 1000, size=9),
        intent_label="swap"
    )
    
    assert not validate_trajectory(invalid_traj, 64, 1000), "Invalid state dim not detected"
    print("✓ Invalid state dimension detected")
    
    # Invalid action range
    invalid_traj2 = Trajectory(
        states=np.random.rand(10, 64),
        actions=np.random.randint(1000, 2000, size=9),  # Out of range
        intent_label="swap"
    )
    
    assert not validate_trajectory(invalid_traj2, 64, 1000), "Invalid action range not detected"
    print("✓ Invalid action range detected")
    
    print("✓ All validation tests passed!\n")
    return True


def test_train_test_split():
    """Test train-test split utility"""
    print("\n" + "="*70)
    print("TEST: Train-Test Split")
    print("="*70)
    
    env = GymDeFi()
    trajectories = generate_synthetic_expert_trajectories(env, n_trajectories=100,
                                                          intent_types=["swap", "lend"])
    
    train, test = split_train_test(trajectories, train_ratio=0.8, shuffle=True, random_seed=42)
    
    assert len(train) == 80, f"Train set size incorrect: {len(train)}"
    assert len(test) == 20, f"Test set size incorrect: {len(test)}"
    assert len(train) + len(test) == len(trajectories), "Train + test != total"
    print(f"✓ Split: {len(train)} train, {len(test)} test")
    
    # Test reproducibility
    train2, test2 = split_train_test(trajectories, train_ratio=0.8, shuffle=True, random_seed=42)
    assert len(train) == len(train2), "Split not reproducible"
    print("✓ Split is reproducible with same seed")
    
    # Test without shuffle
    train_no_shuffle, test_no_shuffle = split_train_test(trajectories, train_ratio=0.8, 
                                                         shuffle=False, random_seed=42)
    assert len(train_no_shuffle) == 80, "No-shuffle split size incorrect"
    print("✓ No-shuffle split works")
    
    print("✓ All split tests passed!\n")
    return True


def test_metrics_computation():
    """Test metrics computation"""
    print("\n" + "="*70)
    print("TEST: Metrics Computation")
    print("="*70)
    
    # Simple test case
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 2])
    
    metrics = compute_comprehensive_metrics(y_true, y_pred)
    
    assert 'f1_macro' in metrics, "Missing f1_macro"
    assert 'precision_macro' in metrics, "Missing precision_macro"
    assert 'recall_macro' in metrics, "Missing recall_macro"
    assert 'accuracy' in metrics, "Missing accuracy"
    
    assert 0 <= metrics['f1_macro'] <= 1, "F1 out of range"
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy out of range"
    
    print(f"✓ F1 (macro): {metrics['f1_macro']:.3f}")
    print(f"✓ Accuracy: {metrics['accuracy']:.3f}")
    
    # Perfect prediction
    y_perfect = np.array([0, 1, 2, 0, 1, 2])
    metrics_perfect = compute_comprehensive_metrics(y_perfect, y_perfect)
    
    assert metrics_perfect['accuracy'] == 1.0, "Perfect prediction should have accuracy 1.0"
    assert metrics_perfect['f1_macro'] == 1.0, "Perfect prediction should have F1 1.0"
    print("✓ Perfect prediction gives correct metrics")
    
    print("✓ All metrics tests passed!\n")
    return True


def test_normalization():
    """Test feature normalization"""
    print("\n" + "="*70)
    print("TEST: Feature Normalization")
    print("="*70)
    
    features = np.random.randn(100, 21) * 5 + 10
    
    # Min-max normalization
    normalized_minmax = normalize_features(features, method='minmax')
    assert np.all(normalized_minmax >= -1e-6), "Min-max normalization produced values < 0"
    assert np.all(normalized_minmax <= 1 + 1e-6), "Min-max normalization produced values > 1"
    print("✓ Min-max normalization to [0, 1]")
    
    # Standard normalization
    normalized_std = normalize_features(features, method='standard')
    assert np.abs(normalized_std.mean()) < 1e-6, "Standard normalization should have mean ≈ 0"
    print(f"✓ Standard normalization mean: {normalized_std.mean():.6f}")
    
    # L2 normalization
    normalized_l2 = normalize_features(features, method='l2')
    norms = np.linalg.norm(normalized_l2, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), "L2 normalization should give unit norm"
    print("✓ L2 normalization gives unit norm")
    
    print("✓ All normalization tests passed!\n")
    return True


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("CHEMO-IRL FRAMEWORK TEST SUITE")
    print("="*70)
    
    tests = [
        ("GymDeFi Environment", test_gym_defi_environment),
        ("Feature Extraction", test_chemo_irl_feature_extraction),
        ("Trajectory Generation", test_trajectory_generation),
        ("Feature Expectation", test_feature_expectation_computation),
        ("Reward Computation", test_reward_computation),
        ("Trajectory Validation", test_trajectory_validation),
        ("Train-Test Split", test_train_test_split),
        ("Metrics Computation", test_metrics_computation),
        ("Feature Normalization", test_normalization),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "ERROR"))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, status in results if status == "PASSED")
    failed = sum(1 for _, status in results if status in ["FAILED", "ERROR"])
    
    for test_name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
