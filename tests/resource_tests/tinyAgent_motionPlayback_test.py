from sys import path
path.append('.')

import pybullet as p
from Humanoid_Basic_Env.resources.tiny.tinyAgent import TinyAgent


# Debug tests

clientID = p.connect(p.GUI)
p.setRealTimeSimulation(False)
test = TinyAgent(clientID)


    # testing mocap file playback
for i in range(3):
    test.playReferenceMotion('../Motions/humanoid3d_spinkick.txt')

p.disconnect()