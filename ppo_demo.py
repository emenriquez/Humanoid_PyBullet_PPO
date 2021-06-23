import gym
import torch
import Humanoid_Basic_Env
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

env = gym.make('HumanoidBasicEnv-v0')
eval_env = gym.make('HumanoidBasicEnv-v0')

policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[1024,512])

# model = PPO.load('walking_agent', env=env)
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log='./walk/logs/')

# Save the best model periodically during training
bestModelCallback = EvalCallback(eval_env=eval_env, eval_freq=10000, log_path='./walk/logs/', best_model_save_path='./walk/logs/')

model.learn(total_timesteps=200, eval_freq=4000, eval_env=eval_env, tb_log_name='static_run', callback=bestModelCallback)
model.save('static_agent')

env.close()