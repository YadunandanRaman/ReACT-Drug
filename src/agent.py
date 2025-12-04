"""
ReACT-Drug: PPO Agent for Dynamic Action Spaces
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CONFIG, DEVICE


class PPONetwork(nn.Module):
    """Actor-Critic network for dynamic action spaces"""
    
    def __init__(self, state_dim, action_embedding_dim=768):
        super().__init__()
        self.state_dim = state_dim
        self.action_embedding_dim = action_embedding_dim
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Policy head: outputs query vector
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_embedding_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        action_query = self.policy_head(features)
        value = self.value_head(features)
        return action_query, value


class PPOBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_contexts = []
        self.advantages = []
        self.returns = []
    
    def store(self, state, action, reward, value, log_prob, done, action_context):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.action_contexts.append(action_context)
    
    def compute_advantages(self, gamma, gae_lambda):
        """Compute GAE advantages"""
        advantages, returns = [], []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_val = 0.0 if self.dones[t] else self.values[t]
            else:
                next_val = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_val - self.values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batch(self):
        """Get batch tensors"""
        states = torch.stack([torch.FloatTensor(s) for s in self.states]).to(DEVICE)
        actions = torch.LongTensor(self.actions).to(DEVICE)
        old_log_probs = torch.FloatTensor(self.log_probs).to(DEVICE)
        advantages = torch.FloatTensor(self.advantages).to(DEVICE)
        returns = torch.FloatTensor(self.returns).to(DEVICE)
        
        return states, actions, old_log_probs, advantages, returns, self.action_contexts


class PPOAgent:
    """PPO agent for dynamic action spaces"""
    
    def __init__(self, state_dim, chemberta_encoder):
        self.state_dim = state_dim
        self.action_embedding_dim = CONFIG["chemberta"]["embedding_dim"]
        self.chemberta_encoder = chemberta_encoder
        
        self.network = PPONetwork(state_dim, self.action_embedding_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=CONFIG["training"]["learning_rate"]
        )
        self.buffer = PPOBuffer()
        
        # Hyperparameters
        self.clip_ratio = CONFIG["training"]["clip_ratio"]
        self.value_coef = CONFIG["training"]["value_coef"]
        self.entropy_coef = CONFIG["training"]["entropy_coef"]
        self.max_grad_norm = CONFIG["training"]["max_grad_norm"]
        
        print("🤖 PPO Agent initialized")
    
    def select_action(self, state, possible_actions):
        """Select action from dynamic action space"""
        if not possible_actions:
            return None, None, None, None
        
        possible_smiles = [a['smiles'] for a in possible_actions]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            action_query, value = self.network(state_tensor)
            
            # Encode candidate actions
            candidate_embs = self.chemberta_encoder.encode_smiles(possible_smiles)
            candidate_tensors = torch.stack(candidate_embs).to(DEVICE)
            
            # Compute logits via dot product
            logits = torch.matmul(candidate_tensors, action_query.T).squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            
            # Sample action
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
        
        return action_idx.item(), log_prob.item(), value.item(), possible_smiles
    
    def store_experience(self, state, action, reward, value, log_prob, done, action_context):
        """Store experience in buffer"""
        self.buffer.store(state, action, reward, value, log_prob, done, action_context)
    
    def update(self, episode):
        """PPO update"""
        if len(self.buffer.states) < CONFIG["training"]["batch_size"]:
            return {}
        
        self.buffer.compute_advantages(
            CONFIG["training"]["gamma"], 
            CONFIG["training"]["gae_lambda"]
        )
        
        states, actions, old_log_probs, advantages, returns, contexts = self.buffer.get_batch()
        
        batch_size = len(self.buffer.states)
        minibatch_size = CONFIG["training"]["batch_size"]
        
        all_policy_losses, all_value_losses, all_entropy_losses = [], [], []
        
        for _ in range(CONFIG["training"]["ppo_epochs"]):
            indices = np.random.permutation(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]
                
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_contexts = [contexts[i] for i in mb_idx]
                
                # Re-evaluate
                new_log_probs, new_values, entropies = [], [], []
                
                for i in range(len(mb_states)):
                    state = mb_states[i].unsqueeze(0)
                    context_smiles = mb_contexts[i]
                    action_taken = mb_actions[i]
                    
                    if not context_smiles:
                        new_log_probs.append(torch.tensor(-1e9).to(DEVICE))
                        entropies.append(torch.tensor(0.0).to(DEVICE))
                        _, value = self.network(state)
                        new_values.append(value.squeeze())
                        continue
                    
                    action_query, value = self.network(state)
                    new_values.append(value.squeeze())
                    
                    candidate_embs = self.chemberta_encoder.encode_smiles(context_smiles)
                    candidate_tensors = torch.stack(candidate_embs).to(DEVICE)
                    
                    logits = torch.matmul(candidate_tensors, action_query.T).squeeze()
                    if logits.dim() == 0:
                        logits = logits.unsqueeze(0)
                    
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs.append(dist.log_prob(action_taken))
                    entropies.append(dist.entropy())
                
                new_log_probs = torch.stack(new_log_probs)
                new_values = torch.stack(new_values)
                entropies = torch.stack(entropies)
                
                # PPO loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, mb_returns)
                entropy_loss = -entropies.mean()
                
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(entropy_loss.item())
        
        self.buffer.clear()
        
        return {
            'policy_loss': np.mean(all_policy_losses),
            'value_loss': np.mean(all_value_losses),
            'entropy_loss': np.mean(all_entropy_losses),
        }
    
    def save(self, path):
        """Save model"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])