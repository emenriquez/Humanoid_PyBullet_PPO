from sys import path
path.append('.')

import pybullet as p
import time
import numpy as np

from Humanoid_Basic_Env.resources.tiny.tinyAgent import TinyAgent


# Debug tests

clientID = p.connect(p.GUI)
p.setRealTimeSimulation(False)
test = TinyAgent(clientID)

    # testing joint control

p.enableJointForceTorqueSensor(
    bodyUniqueId=test.get_ids(),
    jointIndex=1,
    enableSensor=True
)
for i in range(30):
    actions = 0.2*np.random.random(size=(3,)) - 0.1

    time.sleep(0.1)
    for x in range(3):
        test.applyActions(actions)
        # print(
        #     p.getJointStateMultiDof(
        #         bodyUniqueId=test.get_ids(),
        #         jointIndex=1,
        #     )[2:]
        # )

p.disconnect()