import pybullet as p
import os
import numpy as np
import json

# Used to perform relative imports from the resources folder
from sys import path
path.append(".")

from Humanoid_Basic_Env.resources.humanoid import Humanoid


class Target:
    '''
    Tracks and calculates the target pose that the agent should be attempting to replicate.
    Mocap files include positions at time intervals, so this will also add velocity information
    to the data upon initialization.
    '''
    def __init__(self, client, motionFile):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.agentPose = np.zeros(shape=(77,))
        self.targetPose = [
            -0.26454898715019226, 0.022854402661323547, 0.9122610688209534,
            0.6474829034058529, -0.29051448629257354, 0.2797007736899504, 0.6466333584408804,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            -0.69534, 0.0,
            0.867543, 0.0,
            -0.733329, 0.0,
            0.947101, 0.0,
            -0.015759001331352007, 0.009407000794722275, -0.0057410004850112235, 0.9998150844663816,
            0.0, 0.0, 0.0,
            0.003101813040613037, -0.031763325735139306, -0.24121322629597997, 0.9699472405002186,
            0.0, 0.0, 0.0,
            -0.11787694094614279, -0.11500194238645632, -0.10483494747990599, 0.9807595086602051,
            0.0, 0.0, 0.0,
            -0.02001800984225732, 0.023144011379218874, -0.22142810886958508, 0.9746964792291358,
            0.0, 0.0, 0.0,
            -0.5642888606830595, -0.0143471321709281, 0.5323338844841742, 0.6308667663916643,
            0.0, 0.0, 0.0,
            0.11165900540256528, 0.09369100453319251, -0.09715500470079644, 0.9845380476363823,
            0.0, 0.0, 0.0,
            0.02001800984225732, -0.023144011379218874, -0.22142810886958508, 0.9746964792291358,
            0.0, 0.0, 0.0,
            0.7851171556608757, -0.013963002768368036, 0.416618082600584, 0.45806709081845176,
            0.0, 0.0, 0.0,
            -0.04338079765310836, -0.15550348262029812, 0.11769126614205666,
            -0.5504272207179463, -0.3467609398159516, 1.659535929062007,
            -0.04108051249642725, 0.20917186490503967, 0.12540074110919452,
            -0.6488312005607257, 0.3900464487042496, 1.6567469444044791
            ]
        motionFile = os.path.join(os.path.dirname(__file__), motionFile)
        with open(motionFile, "r") as f:
            self.targetMotion = json.load(f)
        
        self.targetDummy = Humanoid(client, target=True)
        self.targetDummyID = self.targetDummy.get_ids()

        # Format the mocap data with the current agent on startup
        self.processedMotionTarget = self.initializeMotionTarget()
        self.framePosition = 0
 
    def initializeMotionTarget(self):
        '''
        Convert the motion file into a format that matches the observations collected from the agent in order to compare and compute rewards.
        '''
        JointFrameMapIndices = [
            0,                  #root
            [9, 10, 11, 8],     #chest
            [13, 14, 15, 12],   #neck
            [17, 18, 19, 16],   #rHip
            20,                 #rKnee
            [22, 23, 24, 21],   #rAnkle
            [26, 27, 28, 25],   #rShoulder
            29,                 #rElbow
            0,                  #rWrist
            [31, 32, 33, 30],   #lHip
            34,                 #lKnee
            [36, 37, 38, 35],   #lAnkle
            [40, 41, 42, 39],   #lShoulder
            43,                 #lElbow
            0,                  #lWrist
        ]

        processedTargetMotion = []

        targetFrames = self.targetMotion['Frames']
        for frameIndex in range(len(targetFrames)-1):
            # Calculate First Frame
            targetPos_orig = [targetFrames[frameIndex][i] for i in [1, 2, 3]]
            targetOrn_orig = targetFrames[frameIndex][5:8] + [targetFrames[frameIndex][4]]
            # transform the position and orientation to have z-axis upward
            y2zPos = [0, 0, 0.0]
            y2zOrn = p.getQuaternionFromEuler([1.57, 0, 0])
            basePos, baseOrn = p.multiplyTransforms(y2zPos, y2zOrn, targetPos_orig, targetOrn_orig)
            # set the agent's root position and orientation
            p.resetBasePositionAndOrientation(
                self.targetDummyID,
                posObj=basePos,
                ornObj=baseOrn
            )

            # Set joint positions
            for joint in self.targetDummy.revoluteJoints:
                p.resetJointState(
                    self.targetDummyID,
                    jointIndex=joint,
                    targetValue=targetFrames[frameIndex][JointFrameMapIndices[joint]]
                )
            for joint in self.targetDummy.sphericalJoints:
                p.resetJointStateMultiDof(
                    self.targetDummyID,
                    jointIndex=joint,
                    targetValue=[targetFrames[frameIndex][i] for i in JointFrameMapIndices[joint]]
                )
            currentFrame = self.targetDummy.collectObservations()
            deltaTime = targetFrames[frameIndex][0]

            if frameIndex < len(targetFrames)-1:
                targetPos_orig = [targetFrames[frameIndex+1][i] for i in [1, 2, 3]]
                targetOrn_orig = targetFrames[frameIndex+1][5:8] + [targetFrames[frameIndex+1][4]]
                # transform the position and orientation to have z-axis upward
                y2zPos = [0, 0, 0.0]
                y2zOrn = p.getQuaternionFromEuler([1.57, 0, 0])
                basePos, baseOrn = p.multiplyTransforms(y2zPos, y2zOrn, targetPos_orig, targetOrn_orig)
                # set the agent's root position and orientation
                p.resetBasePositionAndOrientation(
                    self.targetDummyID,
                    posObj=basePos,
                    ornObj=baseOrn
                )

                # Set joint positions
                for joint in self.targetDummy.revoluteJoints:
                    p.resetJointState(
                        self.targetDummyID,
                        jointIndex=joint,
                        targetValue=targetFrames[frameIndex+1][JointFrameMapIndices[joint]]
                    )
                for joint in self.targetDummy.sphericalJoints:
                    p.resetJointStateMultiDof(
                        self.targetDummyID,
                        jointIndex=joint,
                        targetValue=[targetFrames[frameIndex+1][i] for i in JointFrameMapIndices[joint]]
                    )
                nextFrame = self.targetDummy.collectObservations()

            processedTargetMotion.append(self.processVelocities(currentFrame, nextFrame, deltaTime))
                # Use the lines below for debug testing - safe to remove once confirmed to function
            # self.targetPose = currentFrame
            # print(f'{frameIndex}\t {self.totalImitationReward(agentPose=processedTargetMotion[-1]):.4f}')

        return processedTargetMotion
    
    def calculateLinearVelocity(self, posStart, posEnd, deltaTime):
        velocity = [(posEnd[i] - posStart[i])/deltaTime for i in range(len(posStart))]
        return velocity

    def calculateAngularVelocity(self, ornStart, ornEnd, deltaTime):
        dorn = p.getDifferenceQuaternion(ornStart, ornEnd)
        axis, angle = p.getAxisAngleFromQuaternion(dorn)
        angVel = [(x*angle) / deltaTime for x in axis]
        return angVel

    def processVelocities(self, frame, nextFrame, deltaTime):
        # pre-fill processed frame with currentFrame data
        processedFrame = frame[:]

        # fill processedFrame with velocities
        # base velocity
        processedFrame[7:10] = self.calculateLinearVelocity(frame[0:3], nextFrame[0:3], deltaTime)
        # base angular velocities
        processedFrame[10:13] = self.calculateAngularVelocity(frame[3:7], nextFrame[3:7], deltaTime)
        # 1D joint velocities
        for i in [13, 15, 17, 19]:
            processedFrame[i+1] = self.calculateLinearVelocity([frame[i]], [nextFrame[i]], deltaTime)[0]
        # Angular Velocities from Quaternions
        for i in [21, 28, 35, 42, 49, 56, 63, 70]:
            processedFrame[i+4:i+7] = self.calculateAngularVelocity(frame[i:i+4], nextFrame[i:i+4], deltaTime)

        return processedFrame

    def computePoseReward(self):
        totalQuatDistance = 0
        for i in [3, 21, 28, 35, 42, 49, 56, 63, 70]:
            rotation1 = self.targetPose[i:i+4]
            rotation2 = self.agentPose[i:i+4]
            diffQuat = p.getDifferenceQuaternion(rotation1,rotation2)
            _, quatMag = p.getAxisAngleFromQuaternion(diffQuat)
            # value is rounded because the calculations even for the same quaternions sometimes produce small errors
            totalQuatDistance += np.around(quatMag, decimals=2)
        return np.exp(-2*totalQuatDistance)
    def computeVelocityReward(self):
        velocityIndices = [7, 25, 32, 39, 46, 53, 60, 67, 74]
        totalVelocityDifference = sum([np.linalg.norm(np.array(self.targetPose[i:i+3]) - np.array(self.agentPose[i:i+3])) for i in velocityIndices])
        return np.exp(-0.1*totalVelocityDifference)
    def computeEndEffectorReward(self):
        totalDistance = sum([np.linalg.norm(np.array(self.targetPose[i:i+3]) - np.array(self.agentPose[i:i+3])) for i in [77, 80, 83, 86]])
        return np.exp(-40*totalDistance)
    def computeCenterOfMassReward(self):
        agentRoot = self.agentPose[0:3]
        targetRoot = self.targetPose[0:3]
        distance = np.linalg.norm(np.array(targetRoot) - np.array(agentRoot))
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

    def randomStartFrame(self):
        self.framePosition = np.random.randint(0,high=len(self.processedMotionTarget)-1)
        self.targetPose = self.processedMotionTarget[self.framePosition]
        self.displayTargetPose(processedFrame=self.processedMotionTarget[self.framePosition])
        return self.processedMotionTarget[self.framePosition]

    def restartMotion(self):
        self.framePosition = 0
        self.targetPose = self.processedMotionTarget[self.framePosition]
        self.displayTargetPose(processedFrame=self.processedMotionTarget[self.framePosition])
        return self.processedMotionTarget[self.framePosition]

    def nextFrame(self):
        if self.framePosition < len(self.processedMotionTarget)-1:
            self.framePosition += 1
            self.targetPose = self.processedMotionTarget[self.framePosition]
            self.displayTargetPose(processedFrame=self.processedMotionTarget[self.framePosition])
        return self.processedMotionTarget[self.framePosition]
    
    def checkIfLastFrame(self):
        return self.framePosition >= len(self.processedMotionTarget)-1
    
    def displayTargetPose(self, processedFrame):
        self.targetDummy.setStartingPositionAndVelocity(processedFrame)
        return None
    
# Debug tests
# clientID = p.connect(p.DIRECT)
# test = Target(client=clientID, motionFile='Motions/humanoid3d_backflip.txt')

# print(test.randomStartFrame())

# testPose2 = [
#     -0.35500800609588623, 0.02783391624689102, 1.0422571897506714,
#     0.6223108303817404, -0.37940458264375937, 0.32550806494422807, 0.6023503073086023,
#     0.0, 0.0, 0.0,
#     0.0, 0.0, 0.0,
#     -1.304763, 0.0,
#     1.049183, 0.0,
#     -1.455006, 0.0,
#     1.040217, 0.0,
#     -0.01234700117334175, 0.017517001664649506, -0.02341500222513947, 0.9994960949826182,
#     0.0, 0.0, 0.0,
#     0.008414392760023457, -0.02556603748749739, -0.2258530502836992, 0.9737894923438108,
#     0.0, 0.0, 0.0,
#     -0.157774963995284, -0.08523898054821115, 0.08140198142382576, 0.9804157762662042,
#     0.0, 0.0, 0.0,
#     -0.02029632723630169, 0.17125192259441938, -0.16635012950772768, 0.9708699565447446,
#     0.0, 0.0, 0.0,
#     -0.5675199873815431, -0.28422662422837225, 0.285825698816142, 0.7179414738670981,
#     0.0, 0.0, 0.0,
#     0.10870401641979728, 0.07920501196395756, 0.07324301106339429, 0.9882031492685912,
#     0.0, 0.0, 0.0,
#     0.02029632723630169, -0.17125192259441938, -0.16635012950772768, 0.9708699565447446,
#     0.0, 0.0, 0.0,
#     0.6554179738992906, 0.2886789885039369, 0.1664349933720594, 0.6777839730086095,
#     0.0, 0.0, 0.0,
#     0.0018463717634861695, -0.1512310135444939, 0.43374322483901195,
#     -0.7804912276107155, -0.5604335921107495, 1.4655383360806236,
#     -0.04402742031443441, 0.17440317058074054, 0.45523042153975224,
#     -0.7658689070738836, 0.6593761167752237, 1.4336295342435508
# ]

# result = test.initializeMotionTarget()

# print(f'total imitation reward: {test.totalImitationReward(agentPose=testPose2):.4f}')