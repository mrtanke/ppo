# Hands-on PPO (minimal)

Minimal PPO implementations for:
- **Discrete action**: `CartPole-v1`, `Acrobot-v1`
- **Continuous action**: `Pendulum-v1`, `Walker2d-v5`

## Install

This repo assumes you already have Python + PyTorch working.
Install the basic dependencies:

```powershell
pip install -r requirements.txt
```

## Run training

`main.py` is the main entrypoint and can auto-detect discrete vs continuous environments.

### Discrete (CartPole / Acrobot)

```powershell
python main.py --env_id CartPole-v1 --total_timesteps 50000
python main.py --env_id Acrobot-v1 --total_timesteps 500000
```

### Continuous (Pendulum)

```powershell
python main.py --env_id Pendulum-v1 --mode continuous --total_timesteps 300000
python main.py --env_id Walker-v5 --mode continuous --total_timesteps 200000
```

## Outputs

- Training logs print periodic evaluation returns.
- Curves can be plotted via `plot_curve.py`.
- Notes and plots live under `notes/` and `images/`.

## Files

- `main.py`: training loops + CLI (discrete/continuous)
- `ppo_agent.py`: PPO update + rollout collection helpers
- `models.py`: policy/value networks (discrete + continuous)
- `utils.py`: GAE, evaluation helpers, stats saving
