import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

def compute_gae(rewards, value, dones, gamma, gae_lambda, last_value=None):
    """
    Compute Generalized Advantage Estimation (GAE).
    
    :param rewards: 1D rewards tensors of length T
    :param value: 1D value tensors of length T 
    :param dones: 1D done tensors of length T
    :param gamma: discount factor -> discount future rewards
    :param gae_lambda: lambda for GAE -> discount future error
    :param last_value: value of last state (if trajectory ended mid-episode)
    
    :return: tuple of (advantages, returns), both tensors of length T
    """
    T = rewards.shape[0]  # length of trajectory
    advantages = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    last_adv = torch.zeros((), dtype=torch.float32, device=rewards.device)
    if last_value is None:
        last_value = torch.zeros((), dtype=torch.float32, device=rewards.device)
    elif not torch.is_tensor(last_value):
        last_value = torch.as_tensor(last_value, dtype=torch.float32, device=rewards.device)

    # Iterate backwards so each step can use the value from the next state
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else value[t + 1]
        mask = 1.0 - dones[t].float()

        delta = rewards[t] + gamma * next_value * mask - value[t]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[t] = last_adv

    returns = advantages + value
    return advantages, returns

def evaluate_env_discrete(env, policy, device, num_episodes=3, deterministic=False):
    """Backwards-compatible evaluator (despite the name, works for any discrete-action env)."""
    total = 0.0
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policy(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            action_np = int(action.item())
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            ep_reward += float(reward)
        total += ep_reward
    return total / num_episodes

def save_stats(path, timesteps, rewards):
    np.savez(
        path,
        timesteps=np.array(timesteps),
        rewards=np.array(rewards),
    )

def evaluate_env_continuous(env, policy, device, num_episodes=5, max_steps=1000, obs_normalizer=None):
    total = 0.0
    action_low = env.action_space.low
    action_high = env.action_space.high

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < max_steps:
            if obs_normalizer is not None and getattr(obs_normalizer, "count", 0) > 0:
                obs_input = obs_normalizer.normalize(obs)
            else:
                obs_input = obs
            obs_tensor = torch.as_tensor(obs_input, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                mean, log_std, value = policy.forward(obs_tensor)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                action = dist.mean  # deterministic: use mean
            action_np = action.cpu().numpy()[0]
            action_np = np.clip(action_np, action_low, action_high)

            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            ep_reward += reward
            steps += 1
        total += ep_reward

    return total / num_episodes

class ObsNormalizer:
    def __init__(self, obs_dim, eps=1e-8):
        self.obs_dim = obs_dim
        self.eps = eps
        self.count = 0
        self.mean = np.zeros(obs_dim, dtype=np.float32)
        self.var = np.ones(obs_dim, dtype=np.float32)
    
    def update(self, obs_batch: np.ndarray):
        # obs_batch: [batch_size, obs_dim]
        if obs_batch.ndim == 1:
            obs_batch = obs_batch[None, :]  # make it 2D
        batch_mean = obs_batch.mean(axis=0)
        batch_var = obs_batch.var(axis=0)
        batch_count = obs_batch.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        if batch_count == 0:
            return
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / np.sqrt(self.var + self.eps)
    
    def normalize_tensor(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        # obs_tensor: [batch_size, obs_dim]
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=obs_tensor.device)
        std = torch.sqrt(torch.as_tensor(self.var, dtype=torch.float32, device=obs_tensor.device) + self.eps)
        return (obs_tensor - mean) / std


class RewardScaler:
    """SB3/VecNormalize-style reward normalization.

    Tracks a running variance of discounted returns and normalizes rewards by
    $1 / \sqrt{\operatorname{Var}(R_t) + \epsilon}$, with optional clipping.

    This is intentionally stateful (single-env) and should be used during rollout
    collection only (evaluation should use raw environment rewards).
    """

    def __init__(self, gamma: float = 0.99, clip_reward: float = 10.0, eps: float = 1e-8):
        self.gamma = float(gamma)
        self.clip_reward = float(clip_reward)
        self.eps = float(eps)
        self.reset()

        # Running mean/var of returns (scalar)
        self.count = 0
        self.mean = 0.0
        self.var = 1.0

    def reset(self):
        self.ret = 0.0

    def _update_return_rms(self, ret: float):
        # Welford update for scalar
        self.count += 1
        if self.count == 1:
            self.mean = ret
            self.var = 0.0
            return

        delta = ret - self.mean
        self.mean += delta / self.count
        delta2 = ret - self.mean
        # Store population variance
        self.var = ((self.count - 1) * self.var + delta * delta2) / self.count

    def get_scale(self) -> float:
        return float(np.sqrt(self.var + self.eps))

    def normalize(self, reward: float, done: bool) -> float:
        # Update discounted return
        self.ret = self.ret * self.gamma + float(reward)
        self._update_return_rms(self.ret)

        scaled = float(reward) / self.get_scale()
        if self.clip_reward is not None:
            scaled = float(np.clip(scaled, -self.clip_reward, self.clip_reward))

        if done:
            self.reset()

        return scaled

def save_policy_checkpoint(policy: torch.nn.Module,
                           save_path: Path,
                           env_id: str,
                           obs_normalizer: Optional[ObsNormalizer] = None) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "policy_state_dict": policy.state_dict(),
        "env_id": env_id,
    }
    if obs_normalizer is not None:
        checkpoint["obs_normalizer"] = {
            "mean": obs_normalizer.mean,
            "var": obs_normalizer.var,
            "count": obs_normalizer.count,
        }
    torch.save(checkpoint, save_path)
    print(f"Saved policy checkpoint to {save_path}")
