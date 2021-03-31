import gym
import Humanoid_Basic_Env

from stable_baselines3 import PPO

env = gym.make('HumanoidBasicEnv-v0')
eval_env = gym.make('HumanoidBasicEnv-v0')

model = PPO.load('walking_agent', env=env)
# model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=200000, eval_freq=4000, eval_env=eval_env)
model.save('walking_agent')

env.close()