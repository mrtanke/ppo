import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import argparse

from models import PolicyValueNet, ContinuousPolicyValueNet
from utils import compute_gae, evaluate_env_discrete, save_stats, evaluate_env_continuous
from ppo_agent import ppo_update, ppo_update_continuous, collect_trajectories, collect_trajectories_continuous

DEFAULT_CONFIG = {
    "CartPole-v1": dict(total_timesteps=50_000, lr=3e-4, num_steps=2048),
    "Acrobot-v1": dict(total_timesteps=500_000, lr=1e-4, num_steps=2048),
    "Pendulum-v1": dict(total_timesteps=2_000_000, lr=3e-4, num_steps=2048),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--mode", type=str, default="auto", choices=["discrete", "continuous", "auto"])
    return parser.parse_args()


def train_ppo_discrete(env_id: str, total_timesteps: int = 50_000):
    cfg = DEFAULT_CONFIG.get(env_id, {})
    lr = cfg.get("lr", 3e-4)
    num_steps_per_rollout = cfg.get("num_steps", 2048)
    total_timesteps = cfg.get("total_timesteps", total_timesteps)

    train_env = gym.make(env_id)
    eval_env = gym.make(env_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.n

    # Define the neural network that contains both policy and value function
    policy = PolicyValueNet(
        obs_dim=obs_dim, # environment state -> input dimension
        act_dim=act_dim, # number of actions -> output dimension
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # placeholder config
    ppo_hyperparams = dict(
        gamma=0.99,
        lam=0.95,
        clip_range=0.2,
        train_epochs=4,
        batch_size=64,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    timesteps_collected = 0

    # history for logging
    timesteps_history = []
    rewards_history = []

    while timesteps_collected < total_timesteps:
        # Step 1: collect trajectories
        data = collect_trajectories(train_env, policy, num_steps_per_rollout, device=device)
        timesteps_collected += num_steps_per_rollout
        
        # Step 2: compute advantages and returns
        advantages, returns = compute_gae(
            rewards=data["rewards"],
            value=data["values"],
            dones=data["dones"],
            gamma=ppo_hyperparams["gamma"],
            gae_lambda=ppo_hyperparams["lam"],
            last_value=data["last_value"],
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalize advantages
        data["advantages"] = advantages
        data["returns"] = returns
        
        # Step 3: ppo update
        ppo_update(policy, optimizer, data, ppo_hyperparams)

        # Step 4: log progress
        eval_rewards = evaluate_env_discrete(eval_env, policy, device)
        timesteps_history.append(timesteps_collected)
        rewards_history.append(eval_rewards)
        print(f"Timesteps: {timesteps_collected}, Eval Reward: {eval_rewards:.2f}")

    # Save training statistics
    save_stats(f"{env_id}_training_stats.npz", timesteps_history, rewards_history)

    train_env.close()
    eval_env.close()


def train_ppo_continuous(env_id: str, total_timesteps: int = 200_000):
    cfg = DEFAULT_CONFIG.get(env_id, {})
    lr = cfg.get("lr", 3e-4)
    num_steps_per_rollout = cfg.get("num_steps", 2048)
    total_timesteps = cfg.get("total_timesteps", total_timesteps)

    train_env = gym.make(env_id)
    eval_env = gym.make(env_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]

    policy = ContinuousPolicyValueNet(
        obs_dim=obs_dim,
        act_dim=act_dim,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    ppo_hyperparams = dict(
        gamma=0.99,
        lam=0.95,
        clip_range=0.2,
        train_epochs=4,
        batch_size=64,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    timesteps_collected = 0

    # history for logging
    timesteps_history = []
    rewards_history = []

    while timesteps_collected < total_timesteps:
        # Step 1: collect trajectories
        data = collect_trajectories_continuous(train_env, policy, num_steps_per_rollout, device=device)
        timesteps_collected += num_steps_per_rollout
        
        # Step 2: compute advantages and returns
        advantages, returns = compute_gae(
            rewards=data["rewards"],
            value=data["values"],
            dones=data["dones"],
            gamma=ppo_hyperparams["gamma"],
            gae_lambda=ppo_hyperparams["lam"],
            last_value=data["last_value"],
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalize advantages
        data["advantages"] = advantages
        data["returns"] = returns
        
        # Step 3: ppo update
        ppo_update_continuous(policy, optimizer, data, ppo_hyperparams)

        # Step 4: log progress
        eval_rewards = evaluate_env_continuous(eval_env, policy, device)
        timesteps_history.append(timesteps_collected)
        rewards_history.append(eval_rewards)
        print(f"Timesteps: {timesteps_collected}, Eval Reward: {eval_rewards:.2f}")

    # Save training statistics
    save_stats(f"{env_id}_training_stats.npz", timesteps_history, rewards_history)

    train_env.close()
    eval_env.close()
    


if __name__ == "__main__":
    args = parse_args()

    # Determine mode: discrete or continuous
    if args.mode == "auto":
        # infer from env
        env = gym.make(args.env_id)
        if isinstance(env.action_space, gym.spaces.Discrete):
            train_ppo_discrete(env_id=args.env_id, total_timesteps=args.total_timesteps)
        else:
            train_ppo_continuous(env_id=args.env_id, total_timesteps=args.total_timesteps)
        env.close()
    
    elif args.mode == "discrete":
        train_ppo_discrete(env_id=args.env_id, total_timesteps=args.total_timesteps)
    elif args.mode == "continuous":
        train_ppo_continuous(env_id=args.env_id, total_timesteps=args.total_timesteps)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
