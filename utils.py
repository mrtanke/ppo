import torch
import numpy as np
import matplotlib.pyplot as plt

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

def evaluate_env_continuous(env, policy, device, num_episodes=5, max_steps=1000):
    total = 0.0
    action_low = env.action_space.low
    action_high = env.action_space.high

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < max_steps:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

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
        total += ep_reward

    return total / num_episodes