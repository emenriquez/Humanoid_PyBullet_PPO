import gym
import Humanoid_Basic_Env
import time

from stable_baselines3 import PPO

env_name = 'HumanoidBasicEnv-v0'

env = gym.make(env_name)

model = PPO.load('tiny_agent_moving', env=env)

# model = PPO('MlpPolicy', env, verbose=0) # Use if no training is available


# Render performance of the agent in the given environment
    # Render 3 episodes
i = 0
total_reward = 0
step_count = 0
max_steps_per_episode = 1000

obs = env.reset()

while i < 3:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()
    step_count += 1
    if step_count >= max_steps_per_episode:
        done = True
    if done:
        step_count = 0
        print(f'cumulative reward: {total_reward:.2f}')
        total_reward = 0
        obs = env.reset()
        i += 1

env.close()