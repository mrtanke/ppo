# Walker2d-v5 Notes

- **Observation space:** `Box(shape=(17,), dtype=float32)`
  - Contains joint angles, joint velocities, and torso positional signals (all normalized floats).
- **Action space:** `Box(shape=(6,), low=-1.0, high=1.0)`
  - Represents the torques applied to the six controllable joints; PPO outputs tanh-squashed values and the environment expects inputs clipped to this range.

## Reward characteristics (rough guide)

| Phase | Typical episode return |
|-------|------------------------|
| Initial random / untrained policy | roughly **-20 to +50** (falls quickly, little forward progress) |
| After reasonable PPO training (~1M steps) | roughly **+1,800 to +3,200** (sustained running, fewer crashes) |

These ranges are empirical ballparks to set expectations while monitoring training curves; individual seeds will vary.

## SB3 PPO hyperparameters used

```
policy        = "MlpPolicy"
learning_rate = 3e-4
n_steps       = 2048
batch_size    = 64
gamma         = 0.99
gae_lambda    = 0.95
clip_range    = 0.2
```

Model was trained for `1_000_000` timesteps (`model.learn(total_timesteps=1_000_000)`).
