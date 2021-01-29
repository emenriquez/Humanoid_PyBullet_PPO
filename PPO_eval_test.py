from stable_baselines3 import ppo
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import helloRLWorldEnv

env = gym.make('HelloRLWorldEnv-v0')


model = PPO.load('test_ppo', env=env)

results = evaluate_policy(model=model, env=env, render=False)
print(results)