import gymnasium as gym

env = gym.make("Walker2d-v4")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

obs, info = env.reset()
print("Initial obs shape:", obs.shape)
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Step {i}: reward={reward}, done={done}")
env.close()
