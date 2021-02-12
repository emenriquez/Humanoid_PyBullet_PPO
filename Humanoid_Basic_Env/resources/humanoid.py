import time
import pybullet as p
import os
import numpy as np
import json


class Humanoid:
    def __init__(self, client) -> None:
        f_name = os.path.join(os.path.dirname(__file__), 'humanoid.urdf')
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.humanoidAgent = p.loadURDF(
            fileName=f_name,
            basePosition=[0,1,0],
            baseOrientation=p.getQuaternionFromEuler([0,0,0]),
            globalScaling=0.25,
            physicsClientId=client,
            useFixedBase=1,
            flags=(p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        )
        self.numJoints=p.getNumJoints(self.humanoidAgent)
        self.sphericalJoints=None
        self.revoluteJoints=None
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
    def get_ids(self):
        return self.humanoidAgent

    def getJointInfo(self):
        for joint in range(self.numJoints):

            name, jointID, jointTypeIndex = p.getJointInfo(self.humanoidAgent, joint)[:3]
            # jointID = .getJointInfo(self.humanoidAgent, joint)[0]
            # jointTypeIndex = p.getJointInfo(self.humanoidAgent, joint)[2]
            jointType = ['JOINT_REVOLUTE', 'JOINT_PRISMATIC', 'JOINT_SPHERICAL', 'JOINT_PLANAR', 'JOINT_FIXED'][jointTypeIndex]
            print(f'{jointID} - Joint Name: {name}\t Type: {jointType}')

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
                    jointIndex=1,
                    controlMode=p.POSITION_CONTROL,
                    targetVelocity=[0,0,0],
                    force=[0,0,0]
                )
                self.sphericalJoints.append(joint)
            elif jointType == 0:
                p.setJointMotorControl2(
                    bodyUniqueId=self.humanoidAgent,
                    jointIndex=joint,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=0
                )
                self.revoluteJoints.append(joint)

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
        
        # use the first two values from each joint state (position, velocity) and flatten all tuples into a list
        # output is a flattened list containing:
            # 1D position and velocity for revolute Joints, followed by
            #  Quaternion position and 3D velocity vectors for spherical Joints
        rootState = [position for position in rootPosition] + \
             [orientation for orientation in rootOrientation] + \
                 [velocity for velocity in rootVelocity] + \
                     [angularVelocity for angularVelocity in rootAngularVelocity]
        allJointStates = rootState + \
            [jointInfo for joint in revoluteJointStates for jointInfo in joint[:2]] + \
                [jointInfo for joint in sphericalJointStates for jointInfo in list(sum(joint[:2], ()))]

        return allJointStates
    
    def applyActions(self, actions):
        '''
        Joint indices should correspond to the following:
             0 - root             Type: FIXED
             1 - chest            Type: SPHERICAL
             2 - neck             Type: SPHERICAL
             3 - right_shoulder   Type: SPHERICAL
             4 - right_elbow      Type: REVOLUTE
             5 - right_wrist      Type: FIXED
             6 - left_shoulder    Type: SPHERICAL
             7 - left_elbow       Type: REVOLUTE
             8 - left_wrist       Type: FIXED
             9 - right_hip        Type: SPHERICAL
            10 - right_knee       Type: REVOLUTE
            11 - right_ankle      Type: SPHERICAL
            12 - left_hip         Type: SPHERICAL
            13 - left_knee        Type: REVOLUTE
            14 - left_ankle       Type: SPHERICAL
        '''
        # Format actions from -1,1 to actual values.
        # New scaledAction values will fall in range of -maxForce, maxForce
        maxForce = 200000
        scaledActions = [action*maxForce for action in actions]

        # Set spherical joint torques
        p.setJointMotorControlMultiDofArray(
            bodyUniqueId=self.humanoidAgent,
            jointIndices=self.sphericalJoints,
            controlMode=p.TORQUE_CONTROL,
            forces=scaledActions[0:24]
        )
        # Set revolute joint torques (Commands are different)
        p.setJointMotorControlArray(
            bodyUniqueId=self.humanoidAgent,
            jointIndices=self.revoluteJoints,
            controlMode=p.TORQUE_CONTROL,
            forces=scaledActions[24:]
        )

        # Once all actions are sent, complete by stepping the simulation forward
        p.stepSimulation()

    def playReferenceMotion(self, motionFile):
        motionFile = os.path.join(os.path.dirname(__file__), motionFile)
        with open(motionFile, "r") as motion_file:
            motion = json.load(motion_file)

        JointFrameMapIndices = [
            0, #root
            [9, 10, 11, 8],     #chest
            [13, 14, 15, 12],   #neck
            [26, 27, 28, 25],    #rShoulder
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
            p.stepSimulation() 
            time.sleep(0.03)

# Debug tests

clientID = p.connect(p.GUI)
p.setRealTimeSimulation(False)
test = Humanoid(clientID)

# test.getJointInfo()
test.initializeJoints()
for i in range(5):
    test.playReferenceMotion('Motions/humanoid3d_backflip.txt')

# for i in range(1000):
#     actions = [2*np.random.random() - np.random.random()] * 28
#     test.applyActions(actions)
#     time.sleep(0.03)

# p.disconnect()