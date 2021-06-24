# Demo for tiny env
from sys import path
path.append(".")

from Humanoid_Basic_Env.envs.HumanoidTinyEnv import HumanoidTinyEnv
import numpy as np

test = HumanoidTinyEnv()

time_step = test.reset()

reward = 0
total_steps = 0
while total_steps<10:
    while not time_step[2] == True:
        time_step = test.step(np.random.uniform(low=-1, high=1, size=(28,)))
        total_steps += 1
        try:
            test.render()
        except:
            quit()
        if time_step[1] > 0:
            print(time_step[1])
            print(f'total steps: {total_steps}')
            reward += time_step[1]

    time_step = test.reset()