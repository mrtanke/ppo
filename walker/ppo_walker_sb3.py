import gymnasium as gym
from stable_baselines3 import PPO

def main():
    env_id = "Walker2d-v5"
    env = gym.make(env_id)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    )

    model.learn(total_timesteps=1_000_000)

    model.save("ppo_walker_sb3")

    env.close()
    
if __name__ == "__main__":
    main()