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
        self.agentPose = None
        self.targetPose = None
        self.deltaTime = 0.01
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

        # Angular Velocities from Quaternions
        for i in [0]:
            processedFrame[i+4:i+7] = self.calculateAngularVelocity(frame[i:i+4], nextFrame[i:i+4], deltaTime)

        return processedFrame

    def computePoseReward(self):
        totalQuatDistance = 0
        for i in [0]:
            rotation1 = self.targetPose[i:i+4]
            rotation2 = self.agentPose[i:i+4]
            diffQuat = p.getDifferenceQuaternion(rotation1,rotation2)
            _, quatMag = p.getAxisAngleFromQuaternion(diffQuat)
            # value is rounded because the calculations even for the same quaternions sometimes produce small errors
            totalQuatDistance += np.around(quatMag, decimals=2)
        return np.exp(-2*totalQuatDistance) # original value is -2*distance
    def computeVelocityReward(self):
        velocityIndices = [4]
        totalVelocityDifference = sum([np.linalg.norm(np.array(self.targetPose[i:i+3]) - np.array(self.agentPose[i:i+3])) for i in velocityIndices])
        return np.exp(-1*totalVelocityDifference) # original value is -1*distance
    def computeEndEffectorReward(self):
        # totalDistance = sum([np.linalg.norm(np.array(self.targetPose[i:i+3]) - np.array(self.agentPose[i:i+3])) for i in [77, 80, 83, 86]])
        return 1 # original value is -40*distance
    def computeCenterOfMassReward(self):
        # agentRoot = self.agentPose[0:3]
        # targetRoot = self.targetPose[0:3]
        # distance = np.linalg.norm(np.array(targetRoot) - np.array(agentRoot))
        return 1 # np.exp(-1*distance) # original value is -10*distance

    def totalImitationReward(self, agentPose):
        self.agentPose = agentPose
        poseReward = self.computePoseReward()
        velocityReward = self.computeVelocityReward()
        endEffectorReward = self.computeEndEffectorReward()
        centerOfMassReward = self.computeCenterOfMassReward()
        totalReward = 0.87*poseReward + 0.13*velocityReward + 0*endEffectorReward + 0*centerOfMassReward # original weights are 0.65, 0.1, 0.15, 0.1
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
