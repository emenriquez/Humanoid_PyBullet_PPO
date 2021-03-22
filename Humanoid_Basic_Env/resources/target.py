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
        self.agentPose = np.zeros(shape=(77,))
        self.targetPose = [
            -0.26454898715019226, 0.022854402661323547, 0.9122610688209534,
            0.6474829034058529, -0.29051448629257354, 0.2797007736899504, 0.6466333584408804,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.9392771431776723, 2.3911381059224115,
            0.9818866497297722, 1.1595216576590746,
            -0.7149742790237188, -0.6544759674572938,
            -0.7509255528107431, -0.5865517603581043,
            -0.011300824231338314, 0.011183972519850557, 0.005038624557936806, 0.9998609015222727,
            0.2897713383005688, 0.10879788515515416, 0.7232058583108675,
            0.0032829310045616234, -0.030613089313622575, -0.23172806506017016, 0.9722932505114725,
            0.012815274322457514, 0.08422102372018905, 0.6504554225982309,
            -0.557081520118961, -0.0054891433954768944, 0.5123622256285788, 0.6535403575875486,
            1.45159897583345, 0.8899270768780816, -1.3185170906576993,
            0.7731724180516226, -0.02733566533656029, 0.39978692582419756, 0.4915562910846042,
            -2.505437131719275, -0.9266668423381779, -0.7332852741501034,
            -0.11282908812298957, -0.11101821098567483, -0.09999650401970536, 0.9823162692772182,
            0.3515394002242417, 0.26966724744615134, 0.3198455507945587,
            -0.019489282785801722, 0.02214128043023203, -0.21271679494060253, 0.9766685705537997,
            0.03835010565113387, -0.07202154405427952, 0.5946565443183871,
            0.10694754473069207, 0.09045802597721561, -0.0927923250252098, 0.9857835221937918,
            -0.32482451771327897, -0.2180212455818368, 0.2890557581879866,
            0.019502985446742713, -0.02214366984014484, -0.21271686684439264, 0.9766682271910865,
            -0.03742407785899163, 0.07206793233426356, 0.5946288472716175
            ]
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
        velocityIndices = [7]
        totalVelocityDifference = sum([np.linalg.norm(np.array(self.targetPose[i:i+3]) - np.array(self.agentPose[i:i+3])) for i in velocityIndices])
        return np.exp(-0.1*totalVelocityDifference)
    def computeEndEffectorReward(self):
        totalDistance = 0 #sum([np.linalg.norm(np.array(self.targetPose[i:i+3]) - np.array(self.agentPose[i:i+3])) for i in [1,2,3]])
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


# Debug tests
test = Target(motionFile='Motions/humanoid3d_backflip.txt')

testPose2 = [
    -0.44130900502204895, 0.02583223767578602, 1.0564767122268677,
    -0.24188393221053409, 0.6695399431026858, -0.6485714477105822, -0.269376140634284,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.5555630367106815, 2.0705012236893854,
    0.6915158740900548, 1.9381958030018247,
    -0.9538431984557761, 3.647093384807462,
    -0.8957132008622479, 3.713493304591733,
    0.001663352267994256, 0.0054796645024944995, -0.11647103652945383, 0.9931775793814154,
    0.08628664927992263, -0.24908711914674023, 0.31666258792230567,
    0.025506464687364606, -0.04944438309881907, 0.01688265144890089, 0.9983083938939834,
    -0.07365610969286648, 0.13601601350460607, -0.05089663342769386,
    -0.3098741478881212, 0.17608638499844134, 0.26293753719176294, 0.8965687084793458,
    0.9904951140466739, -0.1867593622373129, -0.8777807152781558,
    0.36278965706169924, -0.3737106116196629, 0.28543756814344606, 0.804518140369916,
    -1.4541873643069911, 0.5048816826323125, -1.297628507948555,
    -0.07958504348309814, -0.07652186454870559, 0.7254465355050566, 0.6793658434330457,
    -0.015819810408372194, 0.2931707535894853, -2.275912001217917,
    -0.020635606847958326, 0.16472373144773317, -0.1594845528972215, 0.973141788957357,
    -0.024871345875511125, -0.4615316779143598, 0.45686730882013404,
    0.07139572828468081, 0.050254427008476626, 0.7189899573892184, 0.689514745108616,
    -0.026078733124507793, -0.2307963076930653, -2.2382721784907442,
    0.020629449773169183, -0.1647236836409558, -0.159484374219689, 0.9731419568745172,
    0.024474107527542692, 0.461468648000345, 0.45695096654118383
    ]

print(f'total imitation reward: {test.totalImitationReward(agentPose=testPose2):.4f}')