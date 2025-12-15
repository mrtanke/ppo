# PPO Training Logic (rl-basics)

This project contains a minimal PPO implementation for CartPole.

## Main Components

- `ppo_minimal.py`  
  - `collect_trajectories(...)`: runs the environment with the current policy and records observations, actions, rewards, dones, values, and log-probabilities.  
  - `ppo_update(...)`: performs the PPO update using:
    - clipped policy loss (ratio * advantage with clipping),
    - value loss (MSE between predicted values and returns),
    - optional entropy bonus for exploration.
  - `train_ppo_cartpole()`: main training loop:
    1. Collect a rollout.  
    2. Compute advantages and returns with `compute_gae(...)`.  
    3. Normalize advantages.  
    4. Run `ppo_update(...)` for a few epochs.  
    5. Evaluate and print average reward.

- `utils.py`  
  - `compute_gae(...)`: computes GAE advantages and returns by walking the trajectory backwards, using discount `gamma` and GAE parameter `gae_lambda`.

- `models.py`  
  - `PolicyValueNet`: a shared network that outputs both policy logits and state values used by PPO.

Use `train_ppo_cartpole()` in `ppo_minimal.py` as the entry point to run training.

## Data Collected (Rollout Buffer)
`collect_trajectories(...)` collects a fixed number of steps `N = num_steps_per_rollout` (not “one episode”).

Typical tensor shapes:
- `obs`: `[N, obs_dim]`
- `actions`: `[N]`
- `rewards`: `[N]`
- `dones`: `[N]` (1 if episode ended at that step)
- `values`: `[N]` (critic prediction $V(s_t)$ at collection time)
- `log_probs`: `[N]` (log-prob of the taken action under the policy used to collect data)

## Advantage and Return (GAE)
Goal: turn raw rewards into a learning signal for the policy (advantages) and a target for the value function (returns).

One-step TD error (“surprise”):
$$\delta_t = r_t + \gamma (1 - d_{t+1}) V(s_{t+1}) - V(s_t)$$
- d: done, if the current state is done, then no next state.

GAE advantage (backward, discounted sum of TD errors):
$$A_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l\,\Big(\prod_{k=1}^{l} (1-d_{t+k})\Big)\,\delta_{t+l}$$

$$A_t = \delta_{t} + (\gamma\lambda) A_{t+1}$$

Return target used for value regression:
$$R_t = A_t + V(s_t)$$

In code:
- `advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)`
- advantages are often normalized before PPO updates.

## PPO Update (Mini-batch SGD)
PPO trains on the rollout buffer for several epochs:
- Shuffle indices.
- Split into mini-batches of size `batch_size`.
- Each mini-batch performs **one** `optimizer.step()`.

Updates per rollout:
$$\text{train\_epochs} \times \left\lceil\frac{N}{\text{batch\_size}}\right\rceil$$

## Loss (PPO Clip + Value + Entropy)
Total loss (matches `ppo_minimal.py`):
$$L = L_{\text{policy}} + c_1 L_{\text{value}} - c_2 H$$
where `c1 = vf_coef`, `c2 = ent_coef`.

### Policy loss (clipped objective)
Compute the new action log-prob under the *current* policy:
- `logits, _ = policy(mb_obs)`
- `dist = Categorical(logits=logits)`
- `new_log_probs = dist.log_prob(mb_actions)`

Probability ratio:
$$r_t(\theta) = \exp(\log \pi_\theta(a_t|s_t) - \log \pi_{\text{old}}(a_t|s_t))$$

Clipped PPO objective:
$$L_{\text{policy}} = -\mathbb{E}\big[\min( r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t )\big]$$

Notes:
- `mb_old_log_probs` are from rollout time and are treated as constants.
- On the very first mini-batch after collection, `new_log_probs ≈ old_log_probs` so `r_t ≈ 1`, but gradients are still non-zero.

### Value loss (critic regression)
$$L_{\text{value}} = \mathbb{E}\big[(R_t - V_\theta(s_t))^2\big]$$

In code: `value_loss = (mb_returns - values).pow(2).mean()`.

### Entropy bonus (optional exploration)
$$H = \mathbb{E}[\mathcal{H}(\pi_\theta(\cdot|s_t))]$$

In code: `entropy = dist.entropy().mean()`.
