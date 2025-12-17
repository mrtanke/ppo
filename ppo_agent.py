import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import typing as Tuple
import numpy as np


def ppo_update(policy, optimizer, data, ppo_hyperparams):
    obs = data["obs"]
    actions = data["actions"]
    advantages = data["advantages"]
    returns = data["returns"]
    old_log_probs = data["log_probs"]
    old_values = data["values"]

    batch_size = ppo_hyperparams["batch_size"]
    clip_range = ppo_hyperparams["clip_range"]
    train_epochs = ppo_hyperparams["train_epochs"]
    target_kl = ppo_hyperparams.get("target_kl", None)
    vf_clip_range = ppo_hyperparams.get("vf_clip_range", None)

    N = obs.shape[0]

    early_stop = False
    last_approx_kl = torch.tensor(0.0, device=obs.device)

    for epoch in range(train_epochs):
        perm = torch.randperm(N) # shuffle indices, dimension: [N]
        for start in range(0, N, batch_size): # split N into chunks of batch_size
            end = start + batch_size
            mb_idx = perm[start:end] # dimenstion: [batch_size]

            mb_obs = obs[mb_idx] # [batch_size, obs_dim]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]
            mb_old_values = old_values[mb_idx]

            # forward pass
            logits, values = policy(mb_obs) # logits: [batch_size, act_dim], values: [batch_size]
            dist = torch.distributions.Categorical(logits=logits) # = softmax
            new_log_probs = dist.log_prob(mb_actions) 

            # ratio
            ratio = torch.exp(new_log_probs - mb_old_log_probs)

            # policy loss
            unclipped = ratio * mb_advantages
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * mb_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            # value loss (MSE)
            if vf_clip_range is not None:
                values_clipped = mb_old_values + torch.clamp(
                    values - mb_old_values,
                    -vf_clip_range,
                    vf_clip_range,
                )
                value_loss = torch.max(
                    (values - mb_returns).pow(2),
                    (values_clipped - mb_returns).pow(2),
                ).mean()
            else:
                value_loss = (mb_returns - values).pow(2).mean()

            # entropy bonus (optional)
            entropy = dist.entropy().mean()

            ent_coef = ppo_hyperparams.get("ent_coef", 0.0)
            vf_coef = ppo_hyperparams.get("vf_coef", 0.5)

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            max_grad_norm = ppo_hyperparams.get("max_grad_norm", None)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            last_approx_kl = (mb_old_log_probs - new_log_probs).mean()
            if target_kl is not None and last_approx_kl > 1.5 * target_kl:
                early_stop = True
                break
        if early_stop:
            break


def ppo_update_continuous(policy, optimizer, data, ppo_params):
    obs = data["obs"]
    actions = data["actions"]
    advantages = data["advantages"]
    returns = data["returns"]
    old_log_probs = data["log_probs"]
    old_values = data["values"]

    batch_size = ppo_params["batch_size"]
    clip_range = ppo_params["clip_range"]
    train_epochs = ppo_params["train_epochs"]
    target_kl = ppo_params.get("target_kl", None)
    vf_clip_range = ppo_params.get("vf_clip_range", None)

    N = obs.shape[0]
    idxs = torch.arange(N)

    early_stop = False
    last_approx_kl = torch.tensor(0.0, device=obs.device)

    for epoch in range(train_epochs):
        perm = idxs[torch.randperm(N)]
        for start in range(0, N, batch_size):
            end = start + batch_size
            mb_idx = perm[start:end]

            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]
            mb_old_values = old_values[mb_idx]

            mean, log_std, values = policy.forward(mb_obs)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std) # mean dimensions: [batch_size, act_dim], std dimensions: [act_dim]
            new_log_probs = dist.log_prob(mb_actions).sum(-1)
            value_preds = values

            ratio = torch.exp(new_log_probs - mb_old_log_probs)

            # policy loss
            unclipped = ratio * mb_advantages
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * mb_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            # value loss (MSE)
            if vf_clip_range is not None:
                values_clipped = mb_old_values + torch.clamp(
                    value_preds - mb_old_values,
                    -vf_clip_range,
                    vf_clip_range,
                )
                value_loss = torch.max(
                    (value_preds - mb_returns).pow(2),
                    (values_clipped - mb_returns).pow(2),
                ).mean()
            else:
                value_loss = (value_preds - mb_returns).pow(2).mean()

            # entropy bonus (optional)
            entropy = dist.entropy().sum(-1).mean()

            ent_coef = ppo_params.get("ent_coef", 0.0)
            vf_coef = ppo_params.get("vf_coef", 0.5)

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            max_grad_norm = ppo_params.get("max_grad_norm", None)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            last_approx_kl = (mb_old_log_probs - new_log_probs).mean()
            if target_kl is not None and last_approx_kl > 1.5 * target_kl:
                early_stop = True
                break
        if early_stop:
            break


def collect_trajectories(env, policy, num_steps, device):
    """
    Collect trajectories by interacting with the environment using the current policy.

    :param env: environment object
    :param policy: policy network
    :param num_steps: number of steps to collect
    :param device: torch device

    :return: dictionary containing collected data
    """
    obs_list = []
    actions_list = []
    rewards_list = []
    done_list = []
    value_list = []
    log_probs_list = []

    obs, info = env.reset()

    for _ in range(num_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) # [1, obs_dim]
    
        with torch.no_grad():
            action, log_prob, value, dist = policy.get_action_and_value(obs_tensor)

        action_np = int(action.item())
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        obs_list.append(obs)
        actions_list.append(action_np)
        rewards_list.append(reward)
        done_list.append(done)
        value_list.append(value.cpu().numpy()[0])
        log_probs_list.append(log_prob.cpu().numpy()[0])

        obs = next_obs
        if done:
            obs, info = env.reset()

    # Bootstrap value for the last observation (needed when the rollout ends mid-episode).
    with torch.no_grad():
        last_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        _, _, last_value, _ = policy.get_action_and_value(last_obs_tensor)
        last_value = last_value.squeeze(0)
    
    data = {
        "obs": torch.as_tensor(np.asarray(obs_list), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(np.asarray(actions_list), dtype=torch.int64, device=device),
        "rewards": torch.as_tensor(np.asarray(rewards_list), dtype=torch.float32, device=device),
        "dones": torch.as_tensor(np.asarray(done_list), dtype=torch.bool, device=device),
        "values": torch.as_tensor(np.asarray(value_list), dtype=torch.float32, device=device),
        "log_probs": torch.as_tensor(np.asarray(log_probs_list), dtype=torch.float32, device=device),
        "last_value": last_value,
    }
    return data


def collect_trajectories_continuous(env, policy, num_steps, device,
                                   obs_normalizer=None, reward_scaler=None):
    """
    Collect trajectories by interacting with the environment using the current policy.
    Similar to collect_trajectories but for continuous action spaces.
    - Actions are float arrays;
    - Need to clip actions to be within action space bounds.

    :param env: environment object
    :param policy: policy network
    :param num_steps: number of steps to collect
    :param device: torch device

    :return: dictionary containing collected data
    """
    obs_list = []  # raw observations for updating normalizer
    actions_list = []
    rewards_list = []
    done_list = []
    value_list = []
    log_probs_list = []

    obs, info = env.reset()
    if reward_scaler is not None:
        reward_scaler.reset()
    action_low = env.action_space.low
    action_high = env.action_space.high

    for _ in range(num_steps):
        obs_input = obs_normalizer.normalize(obs) if obs_normalizer is not None and obs_normalizer.count > 0 else obs
        obs_tensor = torch.tensor(obs_input, dtype=torch.float32, device=device).unsqueeze(0) # [1, obs_dim]
    
        with torch.no_grad():
            action, log_prob, value, dist = policy.get_action_and_value(obs_tensor)

        action_np = action.cpu().numpy()[0].astype(np.float32)
        # Clip action to be within action space bounds
        action_np = np.clip(action_np, action_low, action_high)

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        if reward_scaler is not None:
            reward = reward_scaler.normalize(float(reward), done)

        obs_list.append(obs)
        actions_list.append(action_np)
        rewards_list.append(reward)
        done_list.append(done)
        value_list.append(value.cpu().numpy()[0])
        log_probs_list.append(log_prob.cpu().numpy()[0])

        obs = next_obs
        if done:
            obs, info = env.reset()

            if reward_scaler is not None:
                reward_scaler.reset()

    obs_array = np.asarray(obs_list, dtype=np.float32)
    if obs_normalizer is not None:
        obs_normalizer.update(obs_array)
        norm_obs_array = obs_normalizer.normalize(obs_array)
    else:
        norm_obs_array = obs_array

    rewards_array = np.asarray(rewards_list, dtype=np.float32)
    # Bootstrap value for the last observation (needed when the rollout ends mid-episode).
    with torch.no_grad():
        last_obs_input = obs_normalizer.normalize(obs) if obs_normalizer is not None and obs_normalizer.count > 0 else obs
        last_obs_tensor = torch.tensor(last_obs_input, dtype=torch.float32, device=device).unsqueeze(0)
        _, _, last_value, _ = policy.get_action_and_value(last_obs_tensor)
        last_value = last_value.squeeze(0)
    
    data = {
        "obs": torch.as_tensor(norm_obs_array, dtype=torch.float32, device=device),
        "actions": torch.as_tensor(np.asarray(actions_list), dtype=torch.float32, device=device),
        "rewards": torch.as_tensor(rewards_array, dtype=torch.float32, device=device),
        "dones": torch.as_tensor(np.asarray(done_list), dtype=torch.bool, device=device),
        "values": torch.as_tensor(np.asarray(value_list), dtype=torch.float32, device=device),
        "log_probs": torch.as_tensor(np.asarray(log_probs_list), dtype=torch.float32, device=device),
        "last_value": last_value,
    }
    return data
