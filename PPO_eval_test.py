from stable_baselines3 import ppo
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import Humanoid_Basic_Env

env = gym.make('HumanoidTinyEnv-v0')


# model = PPO.load('test_ppo', env=env)
model = PPO('MlpPolicy', env=env)

results = evaluate_policy(model=model, env=env, render=False)
print(results)