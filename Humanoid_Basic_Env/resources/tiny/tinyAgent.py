import time
import pybullet as p
import os
import numpy as np
import json

# from humanoid_pose_interpolator import *

class tinyAgent:
    def __init__(self, client, target=False) -> None:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        if target==True:
            f_name = os.path.join(os.path.dirname(__file__), 'tinyTarget.urdf')
            self.humanoidAgent = p.loadURDF(
                fileName=f_name,
                basePosition=[0,1,0],
                baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]),
                globalScaling=0.25,
                physicsClientId=client,
                useFixedBase=True,
                flags=(p.URDF_MAINTAIN_LINK_ORDER or p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
            )
        else:
            f_name = os.path.join(os.path.dirname(__file__), 'tiny.urdf')
            self.humanoidAgent = p.loadURDF(
                fileName=f_name,
                basePosition=[0,1,0],
                baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]),
                globalScaling=0.25,
                physicsClientId=client,
                useFixedBase=True,
                flags=(p.URDF_MAINTAIN_LINK_ORDER or p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
            )
        self.numJoints=p.getNumJoints(self.humanoidAgent)
        self.sphericalJoints=None
        self.revoluteJoints=None
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setTimeStep(1./240)
        self.initializeJoints()
        
    def get_ids(self):
        return self.humanoidAgent

    def getJointInfo(self):
        for joint in range(self.numJoints):

            jointID, name, jointTypeIndex = p.getJointInfo(self.humanoidAgent, joint)[:3]
            jointType = ['JOINT_REVOLUTE', 'JOINT_PRISMATIC', 'JOINT_SPHERICAL', 'JOINT_PLANAR', 'JOINT_FIXED'][jointTypeIndex]
            print(f'{jointID}\t Joint Name: {name}\t Type: {jointType}')

    def initializeJoints(self):
        '''
        Change control mode of joints to allow for torque control.
        '''
        self.sphericalJoints = []
        self.revoluteJoints = []
        for joint in range(self.numJoints):
            jointType = p.getJointInfo(self.humanoidAgent, joint)[2]
            if jointType == 2:
                p.setJointMotorControlMultiDof(
                    bodyUniqueId=self.humanoidAgent,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetVelocity=[0,0,0],
                    force=[1,1,1]
                )
                self.sphericalJoints.append(joint)
            elif jointType == 0:
                p.setJointMotorControl2(
                    bodyUniqueId=self.humanoidAgent,
                    jointIndex=joint,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=1
                )
                self.revoluteJoints.append(joint)
        # print(self.revoluteJoints, self.sphericalJoints)

    def collectObservations(self):
        '''
        Collect state information on humanoid joints. Output of getJointStates() will be in the following format:
             0 - position: float
             1 - velocity: float
             2 - Joint Reaction Forces: list of 6 floats
             3 - applied Joint Motor Torque: fload (does not add more information if we use torque control since we know the input torque)
        getJointStatesMultiDof() is used for Spherical Joints and returns 3 dimensional outputs instead of 1D output for revolute joints
        '''
        # Collect the base position, orientation and velocity of the "root" of the agent
        rootPosition, rootOrientation = p.getBasePositionAndOrientation(self.humanoidAgent)
        rootVelocity, rootAngularVelocity = p.getBaseVelocity(self.humanoidAgent)
        # Make calls to Pybullet to collect joint information
        revoluteJointStates = p.getJointStates(
            self.humanoidAgent,
            jointIndices=self.revoluteJoints,
        )
        sphericalJointStates = p.getJointStatesMultiDof(
            self.humanoidAgent,
            jointIndex=self.sphericalJoints,
        )

        # Collect the position of end effectors (Hands and Feet)
        endEffectorPositions = [position[0] for position in p.getLinkStates(bodyUniqueId=self.get_ids(), linkIndices=[])]
        endEffectorPositionsFlattened = [position for link in endEffectorPositions for position in link]
        
        # use the first two values from each joint state (position, velocity) and flatten all tuples into a list
        # output is a flattened list containing:
            # 1D position and velocity for revolute Joints, followed by
            #  Quaternion rotation and 3d angular velocity vectors for spherical Joints
        rootState = [position for position in rootPosition] + \
             [orientation for orientation in rootOrientation] + \
                 [velocity for velocity in rootVelocity] + \
                     [angularVelocity for angularVelocity in rootAngularVelocity]
        '''
        output index structure of allJointStates:
            [0, 1, 2] - root position (xyz)
            [3, 4, 5, 6] - root orientation quaternion (xyzw)
            [7, 8, 9] - root velocity (xyz)
            [10, 11, 12] - root angular velocity (xyz)
            [13-19] - (4D orientation Quaternion, 3D angular velocity) for each spherical joint in order:
                                            rHip,
        '''
        allJointStates = rootState + \
            [jointInfo for joint in [] for jointInfo in joint[:2]] + \
                [jointInfo for joint in sphericalJointStates for jointInfo in list(sum(joint[:2], ()))] + \
                    endEffectorPositionsFlattened # removed revoluteJointStates since we have no revolute joints currently

        return allJointStates
    
    def applyActions(self, actions):
        '''
        Action indices should correspond to the following:
              [0-2] - chest            Type: SPHERICAL
              [3-5] - neck             Type: SPHERICAL
              [6-8] - right_hip        Type: SPHERICAL
             [9-11] - right_ankle      Type: SPHERICAL
            [12-14] - right_shoulder   Type: SPHERICAL
            [15-17] - left_hip         Type: SPHERICAL
            [18-20] - left_ankle       Type: SPHERICAL
            [21-23] - left_shoulder    Type: SPHERICAL
               [24] - right_knee       Type: REVOLUTE
               [25] - right_elbow      Type: REVOLUTE
               [26] - left_knee        Type: REVOLUTE
               [27] - left_elbow       Type: REVOLUTE
        '''
        # Format actions from -1,1 to actual values.
        # New scaledAction values will fall in range of -maxForce, maxForce
        maxForce = 10
        scaledActions = [action*maxForce for action in actions]
        formattedActions = []
        # condense flat array into list of list format for spherical joint control
        for i in [0,3,6,9,12,15,18,21]:
            formattedActions.append(scaledActions[i:i+3])
        formattedActions.extend(scaledActions[24:])
        # Set spherical joint torques
        p.setJointMotorControlMultiDofArray(
            bodyUniqueId=self.humanoidAgent,
            jointIndices=self.sphericalJoints,
            controlMode=p.TORQUE_CONTROL,
            forces=formattedActions[:8]
        )
        # Set revolute joint torques (Commands are different)
        p.setJointMotorControlArray(
            bodyUniqueId=self.humanoidAgent,
            jointIndices=self.revoluteJoints,
            controlMode=p.TORQUE_CONTROL,
            forces=formattedActions[8:]
        )

        # Once all actions are sent, complete by stepping the simulation forward
        # p.stepSimulation()

    def playReferenceMotion(self, motionFile):
        motionFile = os.path.join(os.path.dirname(__file__), motionFile)
        with open(motionFile, "r") as motion_file:
            motion = json.load(motion_file)

        '''
        Joint indices should correspond to the following:
             0 - root             Type: FIXED
             1 - chest            Type: SPHERICAL
             2 - neck             Type: SPHERICAL
             3 - right_hip        Type: SPHERICAL
             4 - right_knee       Type: REVOLUTE
             5 - right_ankle      Type: SPHERICAL
             6 - right_shoulder   Type: SPHERICAL
             7 - right_elbow      Type: REVOLUTE
             8 - right_wrist      Type: FIXED
             9 - left_hip         Type: SPHERICAL
            10 - left_knee        Type: REVOLUTE
            11 - left_ankle       Type: SPHERICAL
            12 - left_shoulder    Type: SPHERICAL
            13 - left_elbow       Type: REVOLUTE
            14 - left_wrist       Type: FIXED
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

        for frame in motion['Frames']:
            basePos1 = [frame[i] for i in [1, 2, 3]]
            baseOrn1 = frame[5:8] + [frame[4]]
            # transform the position and orientation to have z-axis upward
            y2zPos = [0, 0, 0.0]
            y2zOrn = p.getQuaternionFromEuler([1.57, 0, 0])
            basePos, baseOrn = p.multiplyTransforms(y2zPos, y2zOrn, basePos1, baseOrn1)
            # set the agent's root position and orientation
            p.resetBasePositionAndOrientation(
                self.humanoidAgent,
                posObj=basePos,
                ornObj=baseOrn
            )

            # Set joint positions
            for joint in self.revoluteJoints:
                p.resetJointState(
                    self.humanoidAgent,
                    jointIndex=joint,
                    targetValue=frame[JointFrameMapIndices[joint]]
                )
            for joint in self.sphericalJoints:
                p.resetJointStateMultiDof(
                    self.humanoidAgent,
                    jointIndex=joint,
                    targetValue=[frame[i] for i in JointFrameMapIndices[joint]]
                )
            for i in range(16):
                p.stepSimulation()
                time.sleep(0.1/240)

    def setStartingPositionAndVelocity(self, inputFrame):
        # set the agent's root position and orientation
        p.resetBasePositionAndOrientation(
            self.humanoidAgent,
            posObj=inputFrame[0:3],
            ornObj=inputFrame[3:7]
        )
        # set the agent's root velocity and angular velocity
        p.resetBaseVelocity(
            self.humanoidAgent,
            linearVelocity=inputFrame[7:10],
            angularVelocity=inputFrame[10:13]
        )
        # Set joint positions
        revoluteJointIndices = [13, 15, 17, 19]
        for i, joint in enumerate(self.revoluteJoints):
            p.resetJointState(
                self.humanoidAgent,
                jointIndex=joint,
                targetValue=inputFrame[revoluteJointIndices[i]],
                targetVelocity=inputFrame[revoluteJointIndices[i]+1]
            )
        sphericalJointIndices = [21, 28, 35, 42, 49, 56, 63, 70]
        p.resetJointStatesMultiDof(
            self.humanoidAgent,
            jointIndices=self.sphericalJoints,
            targetValues=[inputFrame[index:index+4] for index in sphericalJointIndices],
            targetVelocities=[inputFrame[index+4:index+7] for index in sphericalJointIndices]
        )
        # slight delay is needed before agent will reset. Need to investigate before removing this delay
        time.sleep(0.005)
        # p.stepSimulation()

# Debug tests

clientID = p.connect(p.GUI)
p.setRealTimeSimulation(False)
test = tinyAgent(clientID)

print(test.collectObservations())

#     # testing mocap file playback
# for i in range(1):
#     test.playReferenceMotion('../Motions/humanoid3d_backflip.txt')

    # testing joint control
# actions = np.zeros(shape=(28,))
# for index in range(8):
#     for i in range(50):
#         actions[3*index:3*index+3] = [2*np.random.random() - np.random.random()] *3
#         test.applyActions(actions)
#         time.sleep(0.02)
# for i in range(1):
#     actions = np.random.random(size=(28,))
#     test.applyActions(actions)
#     time.sleep(0.02)

# p.disconnect()

