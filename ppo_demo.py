import gym
import Humanoid_Basic_Env

from stable_baselines3 import PPO

env = gym.make('HumanoidBasicEnv-v0')
eval_env = gym.make('HumanoidBasicEnv-v0')

#model = PPO.load('test_ppo', env=env)
model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=100, eval_freq=4000, eval_env=eval_env)
model.save('test_ppo')

env.close()