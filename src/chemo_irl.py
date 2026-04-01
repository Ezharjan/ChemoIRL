"""
Chemo-IRL: Chemotactic Inverse Reinforcement Learning for DeFi Intent Discovery
Core implementation for training and evaluation
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass
from utils import extract_semantic_features
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class Trajectory:
    """Represents a single trajectory (state-action sequence)"""
    states: np.ndarray  # Shape: (T, state_dim)
    actions: np.ndarray  # Shape: (T,)
    intent_label: str  # Ground truth intent for evaluation

class RewardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RewardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class ChemoIRL:
    """
    Maximum Entropy Inverse Reinforcement Learning with bio-mimetic state representation
    Neural Network Reward Function
    """
    
    def __init__(self, n_features=21, learning_rate=1e-3, gamma=0.99, epsilon=1e-3):
        """
        Args:
            n_features: Number of semantic features (aligned with TIM taxonomy)
            learning_rate: IRL gradient ascent learning rate
            gamma: Discount factor
            epsilon: Convergence threshold
        """
        self.lr = learning_rate
        self.gamma = gamma
        self.n_features = n_features
        self.epsilon = epsilon
        self.training_history = []
        self.decoding_map = None  # Set from environment when available
        
        # Neural Reward Function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_net = RewardNet(n_features).to(self.device)
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=learning_rate, weight_decay=1e-4)

    def extract_features(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Extract semantic features from state-action pair.
        Delegates to utils.extract_semantic_features for consistency.
        """
        return extract_semantic_features(state, action, self.n_features, self.decoding_map)
    
    def compute_feature_expectation(self, trajectories: List[Trajectory]) -> np.ndarray:
        """
        Compute empirical feature expectation: mu = E[sum_t gamma^t * phi(s_t, a_t)]
        Used for RRE metric.
        """
        total_features = np.zeros(self.n_features)
        n_traj = len(trajectories)
        
        if n_traj == 0:
            return total_features
        
        for traj in trajectories:
            discount = 1.0
            for t in range(len(traj.actions)):
                state = traj.states[t]
                action = traj.actions[t]
                phi = self.extract_features(state, action)
                total_features += discount * phi
                discount *= self.gamma
                
        return total_features / n_traj
    
    def compute_reward(self, state: np.ndarray, action: int, return_tensor=False):
        """Compute reward using Neural Network"""
        phi = self.extract_features(state, action)
        phi_tensor = torch.FloatTensor(phi).to(self.device)
        reward = self.reward_net(phi_tensor)
        if return_tensor:
            return reward
        return reward.item()
    
    def generate_agent_trajectories(self, env, n_trajectories: int, max_steps: int = 15) -> List[Trajectory]:
        """
        Generate agent trajectories using epsilon-greedy policy based on current reward
        """
        trajectories = []
        
        for _ in range(n_trajectories):
            states_list = []
            actions_list = []
            
            state = env.reset()
            
            for step in range(max_steps):
                # Record current state before taking action
                states_list.append(state.copy())
                
                # Epsilon-greedy action selection (epsilon=0.05)
                if np.random.rand() < 0.05:
                    action = np.random.randint(0, env.action_dim)
                else:
                    # Select action with highest predicted reward
                    rewards_idxs = list(range(0, env.action_dim)) 
                    
                    # Batch processing for efficiency
                    phis = [self.extract_features(state, a) for a in rewards_idxs]
                    phis_tensor = torch.FloatTensor(np.array(phis)).to(self.device)
                    
                    with torch.no_grad():
                        rewards = self.reward_net(phis_tensor).squeeze().cpu().numpy()
                    
                    best_idx = np.argmax(rewards)
                    action = rewards_idxs[best_idx]
                
                actions_list.append(action)
                state, _, done, _ = env.step(action)
                
                if done:
                    break
            
            # Ensure we have valid trajectory
            if len(actions_list) > 0:
                traj = Trajectory(
                    states=np.array(states_list),
                    actions=np.array(actions_list),
                    intent_label="agent_generated"
                )
                trajectories.append(traj)
        
        return trajectories
    
    def train(self, expert_trajectories: List[Trajectory], env, n_iterations: int = 15) -> None:
        """
        Deep MaxEnt IRL training algorithm
        """
        if hasattr(env, 'decoding_map'):
            self.decoding_map = env.decoding_map

        print(f"\n{'='*70}")
        print(f"Chemo-IRL Training (Deep Reward)")
        print(f"{'='*70}")
        print(f"Expert trajectories: {len(expert_trajectories)}")
        print(f"Device: {self.device}")
        
        for iteration in range(n_iterations):
            # Generate agent trajectories with current policy
            agent_trajectories = self.generate_agent_trajectories(env, len(expert_trajectories))
            
            # Compute Loss: E_agent[R] - E_expert[R]
            self.optimizer.zero_grad()
            
            expert_loss = 0
            for traj in expert_trajectories:
                discount = 1.0
                for t in range(len(traj.actions)):
                    r = self.compute_reward(traj.states[t], traj.actions[t], return_tensor=True)
                    expert_loss -= (r * discount) # Maximize expert reward
                    discount *= self.gamma
            expert_loss /= len(expert_trajectories)
            
            agent_loss = 0
            for traj in agent_trajectories:
                discount = 1.0
                for t in range(len(traj.actions)):
                    r = self.compute_reward(traj.states[t], traj.actions[t], return_tensor=True)
                    agent_loss += (r * discount) # Minimize agent reward
                    discount *= self.gamma
            agent_loss /= len(agent_trajectories)
            
            loss = expert_loss + agent_loss
            loss.backward()
            
            # Metric tracking: Gradient and Parameter Norms
            grad_norm = 0.0
            for p in self.reward_net.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            param_norm = 0.0
            for p in self.reward_net.parameters():
                param_norm += p.data.norm(2).item() ** 2
            param_norm = param_norm ** 0.5
            
            self.optimizer.step()
            
            # Record history
            self.training_history.append({
                'iteration': iteration,
                'loss': loss.item(),
                'expert_reward': -expert_loss.item(),
                'agent_reward': agent_loss.item(),
                'grad_norm': grad_norm,
                'param_norm': param_norm
            })
            
            print(f"  Iteration {iteration+1}/{n_iterations}: "
                  f"Loss={loss.item():.4f} (Exp_R={-expert_loss.item():.4f}, Ag_R={agent_loss.item():.4f})")
        
        print(f"\n{'='*70}")
        print(f"Training Complete")
        print(f"{'='*70}\n")
    
    def predict_intent(self, trajectory: Trajectory) -> int:
        """
        Predict intent by finding the most activated feature via Gradient Attribution
        """
        total_attribution = np.zeros(self.n_features)
        discount = 1.0
        
        for t in range(len(trajectory.actions)):
            phi = self.extract_features(trajectory.states[t], trajectory.actions[t])
            # Create tensor directly on device and set requires_grad
            phi_tensor = torch.tensor(phi, dtype=torch.float32, device=self.device)
            phi_tensor.requires_grad_(True)
            
            reward = self.reward_net(phi_tensor)
            
            # Compute gradient w.r.t input features
            reward.backward()
            
            if phi_tensor.grad is not None:
                grad = phi_tensor.grad.cpu().numpy()
                attribution = phi * grad
                total_attribution += discount * attribution
            
            discount *= self.gamma
        
        return np.argmax(total_attribution)
    
    def evaluate(self, test_trajectories: List[Trajectory], intent_mapping: Dict, env=None) -> Dict:
        """
        Evaluate on test trajectories using actual F1 scores.
        Computes RRE by generating new agent trajectories if env is provided.
        """
        from sklearn.metrics import f1_score

        if env is not None and hasattr(env, 'decoding_map'):
             self.decoding_map = env.decoding_map
        
        if len(test_trajectories) == 0:
            return {'simple': 0.0, 'complex': 0.0, 'overall': 0.0, 'rre': 0.0}
        
        # Predict intents
        predictions = []
        true_labels = []
        
        for traj in test_trajectories:
            pred_feature = self.predict_intent(traj)
            
            # Map feature index to intent label
            if 0 <= pred_feature <= 2:
                pred_label = 0
            elif 3 <= pred_feature <= 6:
                pred_label = 1
            elif 7 <= pred_feature <= 9:
                pred_label = 2
            elif 10 <= pred_feature <= 12:
                pred_label = 3
            elif 13 <= pred_feature <= 17:
                pred_label = 4
            elif 18 <= pred_feature <= 20:
                pred_label = 5
            else:
                pred_label = 0 # Default
                
            predictions.append(pred_label)
            
            # Get true label from trajectory metadata if available
            if hasattr(traj, 'intent_label') and intent_mapping:
                # Convert string label to integer using mapping
                label_val = intent_mapping.get(traj.intent_label, 0)
                true_labels.append(label_val)
            else:
                true_labels.append(0)
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Compute accuracy by intent type
        # Labels 0,1,2 = Simple; 3,4,5 = Complex
        simple_mask = np.isin(true_labels, [0, 1, 2])
        complex_mask = np.isin(true_labels, [3, 4, 5])
        
        # Calculate F1 scores
        f1_overall = f1_score(true_labels, predictions, average='macro', zero_division=0)
        
        if simple_mask.sum() > 0:
            f1_simple = f1_score(true_labels[simple_mask], predictions[simple_mask], average='macro', zero_division=0)
        else:
            f1_simple = 0.0
            
        if complex_mask.sum() > 0:
            f1_complex = f1_score(true_labels[complex_mask], predictions[complex_mask], average='macro', zero_division=0)
        else:
            f1_complex = 0.0
            
        # Compute RRE (reward recovery error)
        expert_mu = self.compute_feature_expectation(test_trajectories)
        
        if env is not None:
             agent_trajectories = self.generate_agent_trajectories(env, n_trajectories=len(test_trajectories))
             learner_mu = self.compute_feature_expectation(agent_trajectories)
             rre = np.linalg.norm(expert_mu - learner_mu) / (np.linalg.norm(expert_mu) + 1e-6)
        else:
             rre = 0.0
        
        results = {
            'simple': float(f1_simple),
            'complex': float(f1_complex),
            'overall': float(f1_overall),
            'rre': float(rre)
        }
        
        return results
    
    def save(self, filepath: str):
        """Save model parameters"""
        torch.save(self.reward_net.state_dict(), filepath)
        # Also save history
        hist_path = filepath + ".hist.json"
        with open(hist_path, 'w') as f:
            json.dump({
                'training_history': self.training_history,
                'n_features': self.n_features,
                'gamma': self.gamma
            }, f)
    
    def load(self, filepath: str):
        """Load model parameters"""
        self.reward_net.load_state_dict(torch.load(filepath, map_location=self.device))
        hist_path = filepath + ".hist.json"
        if os.path.exists(hist_path):
            with open(hist_path, 'r') as f:
                data = json.load(f)
            self.training_history = data.get('training_history', [])
            self.n_features = data['n_features']
            self.gamma = data['gamma']


class GymDeFi:
    """
    Synthetic DeFi environment with bio-mimetic state representation
    """
    
    def __init__(self, state_dim=64, action_dim=1000, obfuscated=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.current_state = None
        self.prev_market_state = None
        self.timestep = 0
        self.obfuscated = obfuscated
        
        # State dimensions
        self.user_dim = 20
        self.market_dim = 22
        self.grad_dim = 22
        
        # Create action mapping (Logic <-> ActionID)
        # If obfuscated, shuffle the mapping.
        # Logic space: 5 protocols * 20 functions * 10 values = 1000 actions.
        self.decoding_map = {}  # ActionID -> (proto, func, val)
        self.reverse_map = {}   # (proto, func, val) -> ActionID
        
        # Generate canonical list of logics
        logics = []
        for p in range(5):
            for f in range(20):
                for v in range(10):
                    logics.append((p, f, v))
        
        # Assign ActionIDs
        action_ids = np.arange(self.action_dim)
        if self.obfuscated:
            np.random.shuffle(action_ids)
            
        for idx, (p, f, v) in enumerate(logics):
            # Map logic tuples to action ids.
            if idx < len(action_ids):
                aid = action_ids[idx]
                self.decoding_map[aid] = (p, f, v)
                self.reverse_map[(p, f, v)] = aid
            else:
                break
                
    def get_action_id(self, protocol, function, value):
        """Helper to get action ID for a logical action"""
        key = (protocol, function, value)
        if key in self.reverse_map:
            return self.reverse_map[key]
        return np.random.randint(0, self.action_dim)

    def reset(self) -> np.ndarray:
        """Reset to initial state"""
        self.timestep = 0
        
        # User state: token balances, nonce, etc.
        # Initialize 0-4 as Tokens, 5-9 as Collateral, 10-14 as LP tokens
        user_state = np.random.exponential(1.0, self.user_dim)
        user_state = user_state / (user_state.sum() + 1e-8)  # Normalize
        
        # Market state: prices, reserves (log-normal distribution for prices)
        market_state = np.random.lognormal(0, 0.5, self.market_dim)
        market_state = market_state / market_state.mean()  # Normalize around 1
        
        # Initial gradient is zero
        temporal_grad = np.zeros(self.grad_dim)
        
        self.current_state = np.concatenate([user_state, market_state, temporal_grad])
        self.prev_market_state = market_state.copy()
        
        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state"""
        self.timestep += 1
        
        # Decode action
        if action in self.decoding_map:
            protocol_id, function_id, value_bin = self.decoding_map[action]
        else:
            # Fallback for unmapped action ids
            protocol_id = 0
            function_id = 0 
            value_bin = 0
            
        # Update user state based on action (meaningful transitions)
        user_state = self.current_state[:self.user_dim].copy()
        market_state = self.prev_market_state.copy() # Will update later
        
        # Define meaningful indices
        # Tokens: 0-2 (Stable, ETH, WBTC)
        # Collateral: 3-5
        # LP: 6-8
        
        amount = (value_bin + 1) * 0.05 # Scaling factor
        
        # Meaningful state updates based on Function ID
        # 0,1,2: Swap
        if function_id in [0, 1, 2]:
            # Simple swap logic: Decrease random token, Increase random token
            # Function and protocol ids define swap direction.
            in_token = (protocol_id) % 3
            out_token = (protocol_id + 1) % 3
            if user_state[in_token] > amount * 0.1:
                user_state[in_token] -= amount * 0.1
                user_state[out_token] += amount * 0.095 # Fees
            
        # 3,4,5,6: Lending
        elif function_id in [3, 4, 5, 6]: 
            # 3: Deposit/Supply
            if function_id == 3:
                token = protocol_id % 3
                if user_state[token] > amount * 0.1:
                    user_state[token] -= amount * 0.1
                    user_state[token + 3] += amount * 0.1 # Get cToken/Collateral
            # 4: Borrow
            elif function_id == 4:
                # Need collateral
                if user_state[3:6].sum() > amount * 0.1:
                     user_state[protocol_id % 3] += amount * 0.08 # Receive token
            
        # 7,8,9: Liquidity
        elif function_id in [7, 8, 9]:
            # 7: Add Liq
            if function_id == 7:
                 if user_state[0:3].sum() > amount * 0.2:
                      user_state[0] -= amount * 0.05
                      user_state[1] -= amount * 0.05
                      user_state[6 + (protocol_id%3)] += amount * 0.1 # LP Token
                      
        # Apply some noise to simulate other factors
        user_state += np.random.normal(0, 0.01, self.user_dim)
        user_state = np.clip(user_state, 0, None)
        # Normalize user features for stable feature extraction.
        user_state = user_state / (user_state.sum() + 1e-8)
        
        # Update market state (random walk with mean reversion)
        market_state = self.prev_market_state.copy()
        market_state += np.random.normal(0, 0.05, self.market_dim)
        market_state = market_state * 0.95 + 1.0 * 0.05  # Mean reversion to 1
        market_state = np.clip(market_state, 0.1, 10.0)
        
        # Compute temporal gradient
        delta_t = 12.0  # Ethereum block time
        temporal_grad = (market_state - self.prev_market_state) / delta_t
        
        # Construct new state
        self.current_state = np.concatenate([user_state, market_state, temporal_grad])
        self.prev_market_state = market_state.copy()
        
        # Reward (not used in IRL, but needed for environment)
        reward = 0.0
        
        # Termination (stochastic or max steps)
        done = (np.random.rand() > 0.95) or (self.timestep >= 15)
        
        info = {'timestep': self.timestep}
        
        return self.current_state, reward, done, info


def generate_synthetic_expert_trajectories(env: GymDeFi, n_trajectories: int, 
                                           intent_types: List[str],
                                           obfuscation_level: float = 0.3) -> List[Trajectory]:
    """
    Generate synthetic expert trajectories with different intent patterns
    
    Args:
        env: environment
        n_trajectories: num trajs
        intent_types: list of intent strings
        obfuscation_level: Probability of using ambiguous or noisy actions to hide intent.
                          Default 0.3.
    
    Ensures proper alignment: states[t] corresponds to the state when action[t] was taken
    """
    trajectories = []
    
    for i in range(n_trajectories):
        intent = intent_types[i % len(intent_types)]
        
        states_list = []
        actions_list = []
        
        state = env.reset()
        
        # Generate actions based on intent type
        traj_length = np.random.randint(5, 15)
        
        for step in range(traj_length):
            # Record current state before taking action
            states_list.append(state.copy())
            
            # Helper to generate action from function_id
            def make_action(func_id):
                # Generates a valid action ID for the given function preference
                # Protocol and Value are chosen randomly to simulate variety
                protocol = np.random.randint(0, 5)
                value = np.random.randint(0, 10)
                # If env supports get_action_id (obfuscated), use it
                if hasattr(env, 'get_action_id'):
                    return env.get_action_id(protocol, func_id, value)
                else:
                    return int(protocol * 200 + func_id * 10 + value)
            
            # Obfuscation logic:
            is_obfuscated = (np.random.rand() < obfuscation_level)
            
            # Choose action based on intent
            if intent == "swap":
                # Primary: 0, 1, 2
                if is_obfuscated:
                    # Mix with Lending (3) and Noise/MEV (18, 19)
                    func = np.random.choice([1, 2, 3, 18, 19])
                else:
                    func = np.random.randint(0, 3)
                action = make_action(func)
                
            elif intent == "lend":
                # Primary: 3, 4, 5, 6
                if is_obfuscated:
                     # Mix with Swap (2) and Risk (15)
                    func = np.random.choice([2, 3, 4, 15, 18])
                else:
                    func = np.random.randint(3, 7)
                action = make_action(func)
                
            elif intent == "liquidity":
                # Primary: 7, 8, 9
                if is_obfuscated:
                    # Mix with Yield (10) and Gov (13)
                    func = np.random.choice([8, 9, 10, 13])
                else:
                    func = np.random.randint(7, 10)
                action = make_action(func)
                
            elif intent == "yield":
                # Primary: 10, 11, 12
                if is_obfuscated:
                    # Mix with Liq (9)
                    func = np.random.choice([9, 10, 11, 14])
                else:
                    func = np.random.randint(10, 13)
                action = make_action(func)
                
            elif intent == "complex_leverage":
                # Complex: lending + swap
                if step % 2 == 0:
                     # Borrow (3)
                    func = 3 if not is_obfuscated else np.random.choice([3, 4, 15])
                else:
                    # Swap (0)
                    func = 0 if not is_obfuscated else np.random.choice([0, 1, 18])
                action = make_action(func)
                
            elif intent == "complex_LP":
                # Complex: swap + add liquidity
                if step < traj_length // 2:
                    # Swap (0)
                    func = 0 if not is_obfuscated else np.random.choice([0, 1, 18])
                else:
                    # Add Liq (7)
                    func = 7 if not is_obfuscated else np.random.choice([7, 8, 19])
                action = make_action(func)
            else:
                action = int(np.random.randint(0, env.action_dim))
            
            actions_list.append(action)
            
            # Take action and get next state
            state, _, done, _ = env.step(action)
            
            # Check termination conditions
            if done and len(actions_list) >= 5:
                break
        
        # Ensure we have valid trajectory (at least one action)
        if len(actions_list) > 0:
            traj = Trajectory(
                states=np.array(states_list),
                actions=np.array(actions_list),
                intent_label=intent
            )
            # Validate state-action alignment
            assert len(traj.states) == len(traj.actions), \
                f"State-action mismatch: {len(traj.states)} states vs {len(traj.actions)} actions"
            trajectories.append(traj)
    
    return trajectories

