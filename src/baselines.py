"""Baseline implementations for intent discovery in DeFi."""

import numpy as np
from typing import List, Dict
from chemo_irl import ChemoIRL, Trajectory
from utils import extract_semantic_features
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class TIMBaseline:
    """
    Transaction Intent Mining (TIM) Framework
    Uses static semantic analysis with multi-layer feature extraction
    Implements a simple classifier on semantic features
    """
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.decoding_map = None
        
    def train(self, train_trajs: List[Trajectory], env):
        """Train TIM classifier on semantic features"""
        if hasattr(env, 'decoding_map'):
            self.decoding_map = env.decoding_map
            
        X_train = []
        y_train = []
        
        for traj in train_trajs:
            # Aggregate features over trajectory
            traj_features = np.zeros(21)
            for t in range(len(traj.actions)):
                phi = extract_semantic_features(traj.states[t], traj.actions[t], decoding_map=self.decoding_map)
                traj_features += phi
            traj_features /= len(traj.actions)  # Average features
            
            X_train.append(traj_features)
            if hasattr(traj, 'intent_label'):
                intent_map = {"swap": 0, "lend": 1, "liquidity": 2, "yield": 3,
                             "complex_leverage": 4, "complex_LP": 5}
                y_train.append(intent_map.get(traj.intent_label, 0))
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.classifier.fit(X_train, y_train)
    
    def evaluate(self, test_trajectories: List[Trajectory], intent_mapping: Dict) -> Dict:
        from sklearn.metrics import f1_score
        
        X_test = []
        y_true = []
        
        for traj in test_trajectories:
            traj_features = np.zeros(21)
            for t in range(len(traj.actions)):
                phi = extract_semantic_features(traj.states[t], traj.actions[t], decoding_map=self.decoding_map)
                traj_features += phi
            traj_features /= len(traj.actions)
            X_test.append(traj_features)
            
            if hasattr(traj, 'intent_label'):
                y_true.append(intent_mapping.get(traj.intent_label, 0))
        
        X_test = np.array(X_test)
        y_true = np.array(y_true)
        y_pred = self.classifier.predict(X_test)
        
        simple_mask = np.isin(y_true, [0, 1, 2])
        complex_mask = np.isin(y_true, [3, 4, 5])
        
        f1_overall = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_simple = f1_score(y_true[simple_mask], y_pred[simple_mask], average='macro', zero_division=0) if simple_mask.sum() > 0 else 0.0
        f1_complex = f1_score(y_true[complex_mask], y_pred[complex_mask], average='macro', zero_division=0) if complex_mask.sum() > 0 else 0.0
        
        return {'simple': float(f1_simple), 'complex': float(f1_complex), 'overall': float(f1_overall), 'rre': None}


class ActionFrequencyBaseline:
    """
    Simple baseline that predicts intent based on the most frequent action type
    No training required (heuristic).
    """
    def __init__(self):
        self.theta = None # No weights
        
    def train(self, train_trajs, env):
        pass
        
    def evaluate(self, test_trajectories: List[Trajectory], intent_mapping: Dict) -> Dict:
        """
        Evaluate using simple majority voting of action types
        """
        from sklearn.metrics import f1_score
        
        preds = []
        true_labels = []
        
        for traj in test_trajectories:
            # Count actions in each category
            counts = {
                'swap': 0, 'lend': 0, 'liquidity': 0, 
                'yield': 0, 'governance': 0
            }
            
            for action in traj.actions:
                # Decode
                function_id = (int(action) % 200) // 10
                
                if function_id in [0, 1, 2]: counts['swap'] += 1
                elif function_id in [3, 4, 5, 6]: counts['lend'] += 1
                elif function_id in [7, 8, 9]: counts['liquidity'] += 1
                elif function_id in [10, 11, 12]: counts['yield'] += 1
                elif function_id in [13, 14]: counts['governance'] += 1
                
            # Heuristic mapping to intent labels
            # 0: swap, 1: lend, 2: liquidity, 3: yield, 4: complex/leverage, 5: complex/LP
            
            # Simple logic: majority wins for simple intents
            # For complex, we need combinations. 
            # Detect complex intents by action-category mixtures.
            
            total = len(traj.actions)
            if total == 0:
                pred = 0
            else:
                top_cat = max(counts, key=counts.get)
                
                # Check for "Complex Leverage" (Swap + Lend mixture)
                if counts['swap'] > 0 and counts['lend'] > 0:
                     # Heuristic: if both significant (>20%), call it leverage
                     if counts['swap']/total > 0.2 and counts['lend']/total > 0.2:
                         pred = 4 # Complex Leverage
                     else:
                         # Default to majority category
                         if top_cat == 'swap': pred = 0
                         elif top_cat == 'lend': pred = 1
                         else: pred = 0
                         
                # Check for "Complex LP" (Swap + Liquidity mixture)
                elif counts['swap'] > 0 and counts['liquidity'] > 0:
                    if counts['swap']/total > 0.2 and counts['liquidity']/total > 0.2:
                        pred = 5 # Complex LP
                    else:
                        if top_cat == 'swap': pred = 0
                        elif top_cat == 'liquidity': pred = 2
                        else: pred = 2
                
                # Simple cases
                elif top_cat == 'swap': pred = 0
                elif top_cat == 'lend': pred = 1
                elif top_cat == 'liquidity': pred = 2
                elif top_cat == 'yield': pred = 3
                elif top_cat == 'governance': pred = 4
                else: pred = 0
            
            preds.append(pred)
            
            if hasattr(traj, 'intent_label') and intent_mapping:
                true_labels.append(intent_mapping.get(traj.intent_label, 0))
            else:
                true_labels.append(0)
                
        preds = np.array(preds)
        true_labels = np.array(true_labels)
        
        # Calculate metrics
        simple_mask = np.isin(true_labels, [0, 1, 2])
        complex_mask = np.isin(true_labels, [3, 4, 5])
        
        f1_overall = f1_score(true_labels, preds, average='macro', zero_division=0)
        
        f1_simple = f1_score(true_labels[simple_mask], preds[simple_mask], average='macro', zero_division=0) if simple_mask.sum() > 0 else 0.0
        f1_complex = f1_score(true_labels[complex_mask], preds[complex_mask], average='macro', zero_division=0) if complex_mask.sum() > 0 else 0.0
        
        return {
            'simple': float(f1_simple),
            'complex': float(f1_complex),
            'overall': float(f1_overall),
            'rre': None # No reward recovery
        }

class LinearChemoIRL(ChemoIRL):
    """Base class for baselines that use Linear Reward functions"""
    def __init__(self, n_features=21, learning_rate=1e-3, gamma=0.99):
        super().__init__(n_features, learning_rate, gamma)
        self.theta = np.random.uniform(-0.1, 0.1, n_features)
        
    def compute_reward(self, state, action):
        phi = self.extract_features(state, action)
        return np.dot(self.theta, phi)
        
    def generate_agent_trajectories(self, env, n_trajectories, max_steps=15):
        trajectories = []
        for _ in range(n_trajectories):
            states_list = []
            actions_list = []
            state = env.reset()
            for step in range(max_steps):
                states_list.append(state.copy())
                if np.random.rand() < 0.05:
                    action = np.random.randint(0, env.action_dim)
                else:
                    rewards_idxs = list(range(0, env.action_dim)) 
                    rewards = [self.compute_reward(state, a) for a in rewards_idxs]
                    action = rewards_idxs[np.argmax(rewards)]
                actions_list.append(action)
                state, _, done, _ = env.step(action)
                if done: break
            if len(actions_list) > 0:
                traj = Trajectory(states=np.array(states_list), actions=np.array(actions_list), intent_label="agent")
                trajectories.append(traj)
        return trajectories
        
    def predict_intent(self, trajectory):
        total_features = np.zeros(self.n_features)
        discount = 1.0
        for t in range(len(trajectory.actions)):
            phi = self.extract_features(trajectory.states[t], trajectory.actions[t])
            total_features += discount * phi * self.theta
            discount *= self.gamma
        return np.argmax(total_features)

class LinearGAIL(LinearChemoIRL):
    """
    Generative Adversarial Imitation Learning with Linear Discriminator
    Effectively learns a reward function parameterized by theta via adversarial training.
    """
    def __init__(self, n_features=21, learning_rate=0.01, gamma=0.99):
        super().__init__(n_features, learning_rate, gamma)
        # Inherits theta
        
    def train(self, expert_trajectories: List[Trajectory], env, n_iterations: int = 15):
        print(f"\nTraining Linear GAIL (Adversarial)...")
        if hasattr(env, 'decoding_map'):
            self.decoding_map = env.decoding_map
        
        # Extract all expert features
        expert_features = []
        for traj in expert_trajectories:
            for t in range(len(traj.actions)):
                # Use self.extract_features to respect decoding_map
                phi = self.extract_features(traj.states[t], traj.actions[t])
                expert_features.append(phi)
        expert_features = np.array(expert_features)
        
        for iteration in range(n_iterations):
            # 1. Generate Generator (Agent) Trajectories using current reward (theta)
            agent_trajectories = self.generate_agent_trajectories(env, len(expert_trajectories))
            
            agent_features = []
            for traj in agent_trajectories:
                for t in range(len(traj.actions)):
                    phi = self.extract_features(traj.states[t], traj.actions[t])
                    agent_features.append(phi)
            agent_features = np.array(agent_features)
            
            # 2. Train Discriminator (our theta)
            X = np.vstack([expert_features, agent_features])
            y = np.concatenate([np.ones(len(expert_features)), np.zeros(len(agent_features))])
            
            logits = np.dot(X, self.theta)
            probs = 1.0 / (1.0 + np.exp(-logits))
            
            errors = probs - y
            gradient = np.dot(X.T, errors) / len(X)
            gradient += 0.01 * self.theta
            
            self.theta -= self.lr * gradient
            
            if iteration % 5 == 0:
                grad_norm = np.linalg.norm(gradient)
                print(f"  Iteration {iteration+1}: Grad={grad_norm:.6f} ThetaNorm={np.linalg.norm(self.theta):.4f}")

        return self.theta


class RLPolicyBaseline:
    """
    Base class for RL-based methods (PPO, DQN, etc.) implemented as 
    Trajectory Classification using MLP on aggregated features.
    """
    def __init__(self, method_name='PPO'):
        self.method_name = method_name
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import  make_pipeline
        
        # Use a pipeline with scaling and MLP - Enhanced for 200 trajectories
        self.classifier = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000, 
                          activation='relu', solver='adam',
                          learning_rate_init=0.001,
                          random_state=42, early_stopping=True, n_iter_no_change=20)
        )
        
    def train(self, train_trajs: List[Trajectory], env):
        """Train RL policy (BC) by extracting aggregated trajectory features."""
        if hasattr(env, 'decoding_map'):
            self.decoding_map = env.decoding_map
        else:
            self.decoding_map = None

        X_train = []
        y_train = []
        
        for traj in train_trajs:
            # Aggregate features for the whole trajectory
            traj_features = np.zeros(21)
            for t in range(len(traj.actions)):
                phi = extract_semantic_features(traj.states[t], traj.actions[t], decoding_map=self.decoding_map)
                traj_features += phi
            
            if len(traj.actions) > 0:
                traj_features /= len(traj.actions)
            
            X_train.append(traj_features)
            
            # Label is the trajectory's intent
            if hasattr(traj, 'intent_label'):
                intent_map = {"swap": 0, "lend": 1, "liquidity": 2, "yield": 3,
                             "complex_leverage": 4, "complex_LP": 5}
                y_train.append(intent_map.get(traj.intent_label, 0))
            else:
                y_train.append(0)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train neural network classifier
        self.classifier.fit(X_train, y_train)
    
    def evaluate(self, test_trajectories: List[Trajectory], intent_mapping: Dict, env=None) -> Dict:
        """Evaluate using trajectory-level prediction"""
        from sklearn.metrics import f1_score
        
        # Ensure we have the map if env is provided during eval
        if env is not None and hasattr(env, 'decoding_map'):
             self.decoding_map = env.decoding_map
        
        preds = []
        true_labels = []
        
        for traj in test_trajectories:
            # Aggregate predictions over trajectory
            traj_features = np.zeros(21)
            for t in range(len(traj.actions)):
                phi = extract_semantic_features(traj.states[t], traj.actions[t], decoding_map=self.decoding_map)
                traj_features += phi
            
            if len(traj.actions) > 0:
                traj_features /= len(traj.actions)
            
            pred = self.classifier.predict([traj_features])[0]
            preds.append(pred)
            
            if hasattr(traj, 'intent_label'):
                true_labels.append(intent_mapping.get(traj.intent_label, 0))
            else:
                true_labels.append(0)
        
        preds = np.array(preds)
        true_labels = np.array(true_labels)
        
        simple_mask = np.isin(true_labels, [0, 1, 2])
        complex_mask = np.isin(true_labels, [3, 4, 5])
        
        f1_overall = f1_score(true_labels, preds, average='macro', zero_division=0)
        f1_simple = f1_score(true_labels[simple_mask], preds[simple_mask], average='macro', zero_division=0) if simple_mask.sum() > 0 else 0.0
        f1_complex = f1_score(true_labels[complex_mask], preds[complex_mask], average='macro', zero_division=0) if complex_mask.sum() > 0 else 0.0
        
        return {'simple': float(f1_simple), 'complex': float(f1_complex), 'overall': float(f1_overall), 'rre': None}


class ValueDice(LinearChemoIRL):
    """
    ValueDice: Off-Policy Imitation Learning
    Uses stationary distribution matching for reward learning
    """
    def __init__(self, n_features=21, learning_rate=0.02, gamma=0.99):
        super().__init__(n_features, learning_rate, gamma)
        
    def train(self, expert_trajectories: List[Trajectory], env, n_iterations: int = 25):
        print(f"\nTraining ValueDice...")
        if hasattr(env, 'decoding_map'):
            self.decoding_map = env.decoding_map
        
        # Extract expert state-action features
        expert_features = []
        for traj in expert_trajectories:
            for t in range(len(traj.actions)):
                phi = self.extract_features(traj.states[t], traj.actions[t])
                expert_features.append(phi)
        expert_features = np.array(expert_features)
        
        for iteration in range(n_iterations):
            # Generate agent trajectories
            agent_trajectories = self.generate_agent_trajectories(env, len(expert_trajectories))
            
            agent_features = []
            for traj in agent_trajectories:
                for t in range(len(traj.actions)):
                    phi = self.extract_features(traj.states[t], traj.actions[t])
                    agent_features.append(phi)
            agent_features = np.array(agent_features)
            
            # ValueDice objective: match stationary distributions
            expert_mu = expert_features.mean(axis=0)
            agent_mu = agent_features.mean(axis=0)
            
            gradient = expert_mu - agent_mu
            grad_norm = np.linalg.norm(gradient)
            
            self.theta += self.lr * gradient
            
            if iteration % 5 == 0:
                print(f"  Iteration {iteration+1}: Grad={grad_norm:.6f} ThetaNorm={np.linalg.norm(self.theta):.4f}")
        
        return self.theta


class SQIL(LinearChemoIRL):
    """
    Soft Q Imitation Learning
    Assigns positive rewards to expert actions, zero to agent actions
    """
    def __init__(self, n_features=21, learning_rate=0.02, gamma=0.99):
        super().__init__(n_features, learning_rate, gamma)
        
    def train(self, expert_trajectories: List[Trajectory], env, n_iterations: int = 25):
        print(f"\nTraining SQIL...")
        if hasattr(env, 'decoding_map'):
            self.decoding_map = env.decoding_map
        
        for iteration in range(n_iterations):
            # Extract expert features (reward = +1)
            expert_features = []
            for traj in expert_trajectories:
                for t in range(len(traj.actions)):
                    phi = self.extract_features(traj.states[t], traj.actions[t])
                    expert_features.append(phi)
            expert_features = np.array(expert_features)
            
            # Generate agent trajectories (reward = 0)
            agent_trajectories = self.generate_agent_trajectories(env, len(expert_trajectories))
            
            agent_features = []
            for traj in agent_trajectories:
                for t in range(len(traj.actions)):
                    phi = self.extract_features(traj.states[t], traj.actions[t])
                    agent_features.append(phi)
            agent_features = np.array(agent_features)
            
            # SQIL update: maximize expert reward, minimize agent reward
            expert_scores = np.dot(expert_features, self.theta)
            agent_scores = np.dot(agent_features, self.theta)
            
            # Gradient pushes expert scores up, agent scores down
            gradient = expert_features.mean(axis=0) - agent_features.mean(axis=0)
            grad_norm = np.linalg.norm(gradient)
            
            self.theta += self.lr * gradient
            
            if iteration % 5 == 0:
                print(f"  Iteration {iteration+1}: Grad={grad_norm:.6f} ThetaNorm={np.linalg.norm(self.theta):.4f}")
        
        return self.theta
