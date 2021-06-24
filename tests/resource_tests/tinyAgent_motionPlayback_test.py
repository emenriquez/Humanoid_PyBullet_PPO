from sys import path
path.append('.')

import pybullet as p
from Humanoid_Basic_Env.resources.tiny.tinyAgent import TinyAgent


# Debug tests

clientID = p.connect(p.GUI)
p.setRealTimeSimulation(False)
test = TinyAgent(clientID)


    # testing mocap file playback
for i in range(1):
    test.playReferenceMotion('../Motions/humanoid3d_dance_a.txt')

p.disconnect()