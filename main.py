import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
from typing import Optional

from models import PolicyValueNet, ContinuousPolicyValueNet
from utils import (
    compute_gae,
    evaluate_env_discrete,
    save_stats,
    evaluate_env_continuous,
    ObsNormalizer,
    RewardScaler,
    save_policy_checkpoint,
)
from wrappers import WalkerForwardRewardWrapper
from ppo_agent import ppo_update, ppo_update_continuous, collect_trajectories, collect_trajectories_continuous

DEFAULT_CONFIG = {
    "CartPole-v1": dict(total_timesteps=50_000, lr=3e-4, num_steps=2048),
    "Acrobot-v1": dict(total_timesteps=500_000, lr=1e-4, num_steps=2048),
    "Pendulum-v1": dict(total_timesteps=2_000_000, lr=3e-4, num_steps=2048,
                         ppo=dict(ent_coef=0.005, train_epochs=6)),
    "Walker2d-v5": dict(
        total_timesteps=2_000_000,
        lr=3e-4,
        num_steps=4096,
        use_obs_norm=True,
        use_reward_norm=True,
        ppo=dict(
            batch_size=256,
            clip_range=0.2,
            train_epochs=10,
            ent_coef=0.01,
            target_kl=0.02,
            vf_clip_range=1.0,
        ),
    ),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--mode", type=str, default="auto", choices=["discrete", "continuous", "auto"])
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides default)")
    parser.add_argument("--train_epochs", type=int, default=None, help="Number of training epochs per update")
    parser.add_argument("--use_obs_norm", action="store_true", help="Enable observation normalization (continuous envs only)")
    parser.add_argument("--use_reward_norm", action="store_true", help="Enable reward normalization (continuous envs only)")
    parser.add_argument("--save_model_path", type=str, default=None,
                        help="Where to save the trained policy (defaults to logs/<env>_policy.pt for Walker2d)")
    return parser.parse_args()




def make_env(env_id: str, render_mode: Optional[str] = None):
    env_kwargs = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode
    env = gym.make(env_id, **env_kwargs)
    if env_id.startswith("Walker2d"):
        env = WalkerForwardRewardWrapper(env)
    return env


def train_ppo_discrete(env_id: str, total_timesteps: int = 50_000,
                       lr_override: Optional[float] = None,
                       train_epochs_override: Optional[int] = None,
                       save_model_path: Optional[str] = None):
    cfg = DEFAULT_CONFIG.get(env_id, {})
    lr = lr_override if lr_override is not None else cfg.get("lr", 3e-4)
    num_steps_per_rollout = cfg.get("num_steps", 2048)
    total_timesteps = cfg.get("total_timesteps", total_timesteps)

    train_env = make_env(env_id)
    eval_env = make_env(env_id)
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
    ppo_hyperparams.update(cfg.get("ppo", {}))
    if train_epochs_override is not None:
        ppo_hyperparams["train_epochs"] = train_epochs_override

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

    if save_model_path:
        save_policy_checkpoint(policy, Path(save_model_path), env_id)

    train_env.close()
    eval_env.close()


def train_ppo_continuous(env_id: str, total_timesteps: int = 200_000,
                         lr_override: Optional[float] = None,
                         train_epochs_override: Optional[int] = None,
                         use_obs_norm: bool = False,
                         use_reward_norm: bool = False,
                         save_model_path: Optional[str] = None):
    cfg = DEFAULT_CONFIG.get(env_id, {})
    lr = lr_override if lr_override is not None else cfg.get("lr", 3e-4)
    num_steps_per_rollout = cfg.get("num_steps", 2048)
    total_timesteps = cfg.get("total_timesteps", total_timesteps)
    use_obs_norm = cfg.get("use_obs_norm", False) or use_obs_norm
    use_reward_norm = cfg.get("use_reward_norm", False) or use_reward_norm

    train_env = make_env(env_id)
    eval_env = make_env(env_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = train_env.observation_space.shape[0]
    obs_normalizer = ObsNormalizer(obs_dim) if use_obs_norm else None
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
    ppo_hyperparams.update(cfg.get("ppo", {}))
    if train_epochs_override is not None:
        ppo_hyperparams["train_epochs"] = train_epochs_override

    # SB3-style schedules (linear)
    initial_lr = lr
    initial_clip_range = float(ppo_hyperparams.get("clip_range", 0.2))

    # SB3-style reward normalization (based on discounted returns RMS)
    reward_scaler = RewardScaler(gamma=float(ppo_hyperparams["gamma"])) if use_reward_norm else None

    timesteps_collected = 0
    auto_save_path: Optional[Path] = None
    if save_model_path is not None:
        auto_save_path = Path(save_model_path)
    elif env_id.startswith("Walker2d"):
        auto_save_path = Path("logs") / f"{env_id}_policy.pt"

    # history for logging
    timesteps_history = []
    rewards_history = []

    while timesteps_collected < total_timesteps:
        # Update schedules (SB3 uses progress_remaining in [1..0])
        progress_remaining = 1.0 - (timesteps_collected / float(total_timesteps))
        progress_remaining = float(np.clip(progress_remaining, 0.0, 1.0))
        lr_now = max(initial_lr * progress_remaining, 1e-6)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now
        ppo_hyperparams["clip_range"] = max(initial_clip_range * progress_remaining, 0.0)

        # Step 1: collect trajectories
        data = collect_trajectories_continuous(
            train_env,
            policy,
            num_steps_per_rollout,
            device=device,
            obs_normalizer=obs_normalizer,
            reward_scaler=reward_scaler,
        )
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
        eval_rewards, eval_forward = evaluate_env_continuous(
            eval_env,
            policy,
            device,
            obs_normalizer=obs_normalizer,
        )
        timesteps_history.append(timesteps_collected)
        rewards_history.append(eval_rewards)
        print(
            f"Timesteps: {timesteps_collected}, Eval Reward: {eval_rewards:.2f}, "
            f"Avg Forward Vel: {eval_forward:.2f}"
        )

    # Save training statistics
    save_stats(f"{env_id}_training_stats.npz", timesteps_history, rewards_history)

    if auto_save_path is not None:
        save_policy_checkpoint(
            policy,
            auto_save_path,
            env_id,
            obs_normalizer=obs_normalizer,
        )

    train_env.close()
    eval_env.close()



if __name__ == "__main__":
    args = parse_args()

    # Determine mode: discrete or continuous
    if args.mode == "auto":
        # infer from env
        env = gym.make(args.env_id)
        if isinstance(env.action_space, gym.spaces.Discrete):
            train_ppo_discrete(
                env_id=args.env_id,
                total_timesteps=args.total_timesteps,
                lr_override=args.lr,
                train_epochs_override=args.train_epochs,
                save_model_path=args.save_model_path,
            )
        else:
            train_ppo_continuous(
                env_id=args.env_id,
                total_timesteps=args.total_timesteps,
                lr_override=args.lr,
                train_epochs_override=args.train_epochs,
                use_obs_norm=args.use_obs_norm,
                use_reward_norm=args.use_reward_norm,
                save_model_path=args.save_model_path,
            )
        env.close()
    
    elif args.mode == "discrete":
        train_ppo_discrete(
            env_id=args.env_id,
            total_timesteps=args.total_timesteps,
            lr_override=args.lr,
            train_epochs_override=args.train_epochs,
            save_model_path=args.save_model_path,
        )
    elif args.mode == "continuous":
        train_ppo_continuous(
            env_id=args.env_id,
            total_timesteps=args.total_timesteps,
            lr_override=args.lr,
            train_epochs_override=args.train_epochs,
            use_obs_norm=args.use_obs_norm,
            use_reward_norm=args.use_reward_norm,
            save_model_path=args.save_model_path,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
