import time
import pybullet as p
import os
import numpy as np


class Humanoid:
    def __init__(self, client) -> None:
        f_name = os.path.join(os.path.dirname(__file__), 'humanoid.urdf')
        self.humanoidAgent = p.loadURDF(
            fileName=f_name,
            basePosition=[0,0,3],
            baseOrientation=p.getQuaternionFromEuler([0,0,0]),
            physicsClientId=client,
            useFixedBase=1,
            flags=(p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        )
        self.numJoints=p.getNumJoints(self.humanoidAgent)
        self.sphericalJoints=None
        self.revoluteJoints=None

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
        Collect information on humanoid joints. Output of getJointInfo() will be in the following format:
             0 - index: int
             1 - name: str
             2 - type: int <-- type of the joint, this also implies the number of position and velocity variables.
                            JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
             3 - gIndex: int
             4 - uIndex: int
             5 - flags: int
             6 - damping: float
             7 - friction: float
             8 - lowerLimit: float
             9 - upperLimit: float
            10 - maxForce: float
            11 - maxVelocity: float
            12 - linkName: str
            13 - axis: tuple <-- Vector3
            14 - parentFramePosition: tuple <-- Vector3
            15 - parentFrameOrientation: tuple <-- Quaternion
            16 - parentIndex: int
        '''
        observations_list = []
        jointList = self.revoluteJoints[0]
        jointPos, jointVel, _, _ = p.getJointState(
            self.humanoidAgent,
            jointIndex=4,
        )
        print(jointPos)
        print(jointVel)

        # return observations_list
    
    def applyActions(self, action):
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
    

# Debug tests

clientID = p.connect(p.DIRECT)
p.setRealTimeSimulation(False)
test = Humanoid(clientID)

# test.getJointInfo()
test.initializeJoints()
test.collectObservations()

# for i in range(1000):
#     actions = [2*np.random.random() - np.random.random()] * 28
#     test.applyActions(actions)
#     time.sleep(0.03)

p.disconnect()