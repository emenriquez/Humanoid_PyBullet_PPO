from sys import path
path.append('.')

import pybullet as p

from Humanoid_Basic_Env.resources.tiny.tinyTarget import TinyTarget

# Debug tests
clientID = p.connect(p.DIRECT)
test = TinyTarget(client=clientID, motionFile='Motions/humanoid3d_backflip.txt', staticTarget=True)

print(test.randomStartFrame())


result = test.initializeMotionTarget()

print(test.totalImitationReward(agentPose=test.processedMotionTarget[0]))