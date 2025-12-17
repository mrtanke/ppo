import argparse
from pathlib import Path
from typing import Optional

import gymnasium as gym
import imageio
import numpy as np
import torch

from models import ContinuousPolicyValueNet
from utils import ObsNormalizer


def load_policy(model_path: Path, env_id: str, device: torch.device, obs_dim: int, act_dim: int):
    policy = ContinuousPolicyValueNet(obs_dim=obs_dim, act_dim=act_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("policy_state_dict", checkpoint)
    policy.load_state_dict(state_dict)
    policy.eval()

    obs_normalizer = None
    obs_state = checkpoint.get("obs_normalizer")
    if obs_state is not None:
        obs_normalizer = ObsNormalizer(obs_dim)
        obs_normalizer.mean = np.asarray(obs_state["mean"], dtype=np.float32)
        obs_normalizer.var = np.asarray(obs_state["var"], dtype=np.float32)
        obs_normalizer.count = float(obs_state["count"])

    return policy, obs_normalizer


def record_gif(env_id: str,
               model_path: Path,
               gif_path: Path,
               num_steps: int,
               fps: int,
               seed: Optional[int] = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_id, render_mode="rgb_array")
    if seed is not None:
        env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    policy, obs_normalizer = load_policy(model_path, env_id, device, obs_dim, act_dim)

    frames = []
    obs, _ = env.reset()
    steps = 0
    done = False

    while not done and steps < num_steps:
        obs_input = obs_normalizer.normalize(obs) if obs_normalizer is not None and obs_normalizer.count > 0 else obs
        obs_tensor = torch.tensor(obs_input, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            mean, _, _ = policy.forward(obs_tensor)
            action = mean

        action_np = action.cpu().numpy()[0]
        action_np = np.clip(action_np, action_low, action_high)

        obs, _, terminated, truncated, _ = env.step(action_np)
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        done = terminated or truncated
        steps += 1

    env.close()

    if not frames:
        raise RuntimeError("No frames were captured; ensure the environment supports rgb_array rendering.")

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"Saved GIF to {gif_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Record a Walker2d rollout GIF using a saved policy.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the saved policy checkpoint.")
    parser.add_argument("--gif_path", type=Path, default=Path("walker2d_rollout.gif"),
                        help="Where to write the output GIF.")
    parser.add_argument("--env_id", type=str, default="Walker2d-v5", help="Environment ID to load.")
    parser.add_argument("--num_steps", type=int, default=2000, help="Maximum number of steps to record.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the GIF.")
    parser.add_argument("--seed", type=int, default=1, help="Optional environment seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    record_gif(
        env_id=args.env_id,
        model_path=args.model_path,
        gif_path=args.gif_path,
        num_steps=args.num_steps,
        fps=args.fps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
