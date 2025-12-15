import gymnasium as gym
from stable_baselines3 import PPO

def main():
    env_id = "Walker2d-v5"
    env = gym.make(env_id, render_mode="human")
    model = PPO.load("ppo_walker_sb3", env=env)

    obs, info = env.reset()
    for _ in range(2000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            obs, info = env.reset()
    
    env.close()

if __name__ == "__main__":
    main()