import gym
import torch
import Humanoid_Basic_Env
from torch import nn

from stable_baselines3 import PPO

env = gym.make('HumanoidBasicEnv-v0')
eval_env = gym.make('HumanoidBasicEnv-v0')

policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[1024,512])

# model = PPO.load('walking_agent', env=env)
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0)
model.learn(total_timesteps=4000, eval_freq=400, eval_env=eval_env)
model.save('walking_agent')

env.close()