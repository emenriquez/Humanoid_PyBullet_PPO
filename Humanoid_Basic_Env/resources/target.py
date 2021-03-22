import pybullet as p
import os
import numpy as np
import json


class Target:
    '''
    Tracks and calculates the target pose that the agent should be attempting to replicate.
    Mocap files include positions at time intervals, so this will also add velocity information
    to the data upon initialization.
    '''
    def __init__(self, motionFile):
        self.agentPose = None
        motionFile = os.path.join(os.path.dirname(__file__), motionFile)
        with open(motionFile, "r") as f:
            self.targetMotion = json.load(f)
   
    def playReferenceMotion(self):

        JointFrameMapIndices = [
            0,                  #root
            [9, 10, 11, 8],     #chest
            [13, 14, 15, 12],   #neck
            [26, 27, 28, 25],   #rShoulder
            29,                 #rElbow
            0,                  #rWrist
            [40, 41, 42, 39],   #lShoulder
            43,                 #lElbow
            0,                  #lWrist
            [17, 18, 19, 16],   #rHip
            20,                 #rKnee
            [22, 23, 24, 21],   #rAnkle
            [31, 32, 33, 30],   #lHip
            34,                 #lKnee
            [36, 37, 38, 35],   #lAnkle
        ]

        for frame in self.targetMotion['Frames']:
            targetPos_orig = [frame[i] for i in [1, 2, 3]]
            targetOrn_orig = frame[5:8] + [frame[4]]
            # transform the position and orientation to have z-axis upward
            y2zPos = [0, 0, 0.0]
            y2zOrn = p.getQuaternionFromEuler([1.57, 0, 0])
            targetPos, targetOrn = p.multiplyTransforms(y2zPos, y2zOrn, targetPos_orig, targetOrn_orig)
    
    def computePoseReward(self):
        return 1
    def computeVelocityReward(self):
        return 1
    def computeEndEffectorReward(self):
        totalDistance = sum([dist for dist in [1,2,3]])
        return totalDistance
    def computeCenterOfMassReward(self):
        distance = np.linalg.norm(np.array([0,0,0]) - np.array([0,0,0.2]))
        return np.exp(-10*distance)

    def totalImitationReward(self, agentPose):
        self.agentPose = agentPose
        poseReward = self.computePoseReward()
        velocityReward = self.computeVelocityReward()
        endEffectorReward = self.computeEndEffectorReward()
        centerOfMassReward = self.computeCenterOfMassReward()
        totalReward = 0.65*poseReward + 0.1*velocityReward + 0.15*endEffectorReward + 0.1*centerOfMassReward
        return totalReward

    def computeGoalReward(self, frame):
        pass


test = Target(motionFile='Motions/humanoid3d_backflip.txt')

print(test.totalImitationReward(agentPose=None))