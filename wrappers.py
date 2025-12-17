import gymnasium as gym


class WalkerForwardRewardWrapper(gym.Wrapper):
    """Encourage Walker2d agents to move forward instead of standing still.

    - Removes the constant healthy/survive reward so that standing in place
      no longer accumulates return.
    - Adds an extra bonus proportional to forward velocity, making locomotion
      strictly more rewarding than staying idle.
    """

    def __init__(self, env: gym.Env, forward_bonus_weight: float = 1.0):
        super().__init__(env)
        self.forward_bonus_weight = float(forward_bonus_weight)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        forward = info.get("x_velocity", info.get("reward_forward", 0.0))
        survive = info.get("reward_survive", 0.0)

        shaped = reward
        if survive is not None:
            shaped -= survive
        shaped += self.forward_bonus_weight * forward

        info = dict(info)
        info["raw_reward"] = reward
        info["forward_bonus"] = self.forward_bonus_weight * forward
        info["shaped_reward"] = shaped

        return obs, shaped, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
