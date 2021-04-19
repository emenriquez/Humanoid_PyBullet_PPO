from Humanoid_Basic_Env.resources.tiny.tinyAgent import TinyAgent
import pybullet as p
import os
import numpy as np
import json

# Used to perform relative imports from the resources folder
from sys import path
path.append(".")

from Humanoid_Basic_Env.resources.tiny.tinyAgent import TinyAgent


class TinyTarget:
    '''
    Tracks and calculates the target pose that the agent should be attempting to replicate.
    Mocap files include positions at time intervals, so this will also add velocity information
    to the data upon initialization.
    '''
    def __init__(self, client, motionFile, staticTarget=False):
        self.staticTarget = staticTarget
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.agentPose = np.zeros(shape=(77,))
        self.targetPose = [-0.22032800316810608, 0.004294916521757841, 0.6617388129234314, 0.7039509129823992, 0.16629285435255795, -0.1374280545944393, 0.6766929351728991, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.374817, 0.0, 0.179283, 0.0, -1.35178, 0.0, 0.244704, 0.0, -0.023734002022433733, -0.03917100333785926, 0.6930770590588314, 0.7194070613024768, 0.0, 0.0, 0.0, -0.023734002022433733, -0.03917100333785926, 0.6930770590588314, 0.7194070613024768, 0.0, 0.0, 0.0, -0.023734002022433733, -0.03917100333785926, 0.6930770590588314, 0.7194070613024768, 0.0, 0.0, 0.0, -0.023734002022433733, -0.03917100333785926, 0.6930770590588314, 0.7194070613024768, 0.0, 0.0, 0.0, -0.023734002022433733, -0.03917100333785926, 0.6930770590588314, 0.7194070613024768, 0.0, 0.0, 0.0, -0.023734002022433733, -0.03917100333785926, 0.6930770590588314, 0.7194070613024768, 0.0, 0.0, 0.0, -0.023734002022433733, -0.03917100333785926, 0.6930770590588314, 0.7194070613024768, 0.0, 0.0, 0.0, -0.023734002022433733, -0.03917100333785926, 0.6930770590588314, 0.7194070613024768, 0.0, 0.0, 0.0, 0.08092368212796099, -0.08589996560423413, 0.10751054955181422, -0.14534513385760567, -0.24799329812892978, 1.4415175836034657, 0.08382827858824551, 0.08317185385017556, 0.11371258083657892, -0.14028942157236113, 0.11820706621011437, 1.4564192379937035]
        motionFile = os.path.join(os.path.dirname(__file__), '../', motionFile)
        with open(motionFile, "r") as f:
            self.targetMotion = json.load(f)
        
        self.targetDummy = TinyAgent(client, target=True)
        self.targetDummyID = self.targetDummy.get_ids()

        # Format the mocap data with the current agent on startup
        if self.staticTarget:
            self.processedMotionTarget = self.initializeStaticTarget()
        else:
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
                    targetValue=[targetFrames[frameIndex][i] for i in JointFrameMapIndices[3]]
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
                        targetValue=[targetFrames[frameIndex+1][i] for i in JointFrameMapIndices[3]]
                    )
                nextFrame = self.targetDummy.collectObservations()

            processedTargetMotion.append(self.processVelocities(currentFrame, nextFrame, deltaTime))
                # Use the lines below for debug testing - safe to remove once confirmed to function
            # self.targetPose = currentFrame
            # print(f'{frameIndex}\t {self.totalImitationReward(agentPose=processedTargetMotion[-1]):.4f}')

        return processedTargetMotion

    def initializeStaticTarget(self):
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
                    targetValue=[targetFrames[frameIndex][i] for i in JointFrameMapIndices[3]]
                )
            currentFrame = self.targetDummy.collectObservations()

            processedTargetMotion.append(currentFrame)

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
        for i in []:
            processedFrame[i+1] = self.calculateLinearVelocity([frame[i]], [nextFrame[i]], deltaTime)[0]
        # Angular Velocities from Quaternions
        for i in [13]:
            processedFrame[i+4:i+7] = self.calculateAngularVelocity(frame[i:i+4], nextFrame[i:i+4], deltaTime)

        return processedFrame

    def computePoseReward(self):
        totalQuatDistance = 0
        for i in [3, 13]:
            rotation1 = self.targetPose[i:i+4]
            rotation2 = self.agentPose[i:i+4]
            diffQuat = p.getDifferenceQuaternion(rotation1,rotation2)
            _, quatMag = p.getAxisAngleFromQuaternion(diffQuat)
            # value is rounded because the calculations even for the same quaternions sometimes produce small errors
            totalQuatDistance += np.around(quatMag, decimals=2)
        return np.exp(-0.2*totalQuatDistance) # original value is -2*distance
    def computeVelocityReward(self):
        velocityIndices = [17]
        totalVelocityDifference = sum([np.linalg.norm(np.array(self.targetPose[i:i+3]) - np.array(self.agentPose[i:i+3])) for i in velocityIndices])
        return np.exp(-0.1*totalVelocityDifference) # original value is -1*distance
    def computeEndEffectorReward(self):
        # totalDistance = sum([np.linalg.norm(np.array(self.targetPose[i:i+3]) - np.array(self.agentPose[i:i+3])) for i in [77, 80, 83, 86]])
        return 1 # original value is -40*distance
    def computeCenterOfMassReward(self):
        agentRoot = self.agentPose[0:3]
        targetRoot = self.targetPose[0:3]
        distance = np.linalg.norm(np.array(targetRoot) - np.array(agentRoot))
        return np.exp(-1*distance) # original value is -10*distance

    def totalImitationReward(self, agentPose):
        self.agentPose = agentPose
        poseReward = self.computePoseReward()
        velocityReward = self.computeVelocityReward()
        endEffectorReward = self.computeEndEffectorReward()
        centerOfMassReward = self.computeCenterOfMassReward()
        totalReward = 0.76*poseReward + 0.12*velocityReward + 0*endEffectorReward + 0.12*centerOfMassReward # original weights are 0.65, 0.1, 0.15, 0.1
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
        if self.staticTarget:
            self.displayTargetPose(processedFrame=self.processedMotionTarget[self.framePosition])
            return self.processedMotionTarget[self.framePosition]
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
# test = Target(client=clientID, motionFile='../Motions/humanoid3d_backflip.txt', staticTarget=True)

# test.randomStartFrame()

# result = test.initializeMotionTarget()

# print(test.totalImitationReward(agentPose=test.processedMotionTarget[0]))