import math
from pybullet_utils import bullet_client
import pybullet as p


class PoseInterpolator(object):

  def __init__(self):
    self.Reset()

  def Reset(self,
            basePos=[0, 0, 0],
            baseOrn=[0, 0, 0, 1],
            chestRot=[0, 0, 0, 1],
            neckRot=[0, 0, 0, 1],
            rightHipRot=[0, 0, 0, 1],
            rightKneeRot=[0],
            rightAnkleRot=[0, 0, 0, 1],
            rightShoulderRot=[0, 0, 0, 1],
            rightElbowRot=[0],
            leftHipRot=[0, 0, 0, 1],
            leftKneeRot=[0],
            leftAnkleRot=[0, 0, 0, 1],
            leftShoulderRot=[0, 0, 0, 1],
            leftElbowRot=[0],
            baseLinVel=[0, 0, 0],
            baseAngVel=[0, 0, 0],
            chestVel=[0, 0, 0],
            neckVel=[0, 0, 0],
            rightHipVel=[0, 0, 0],
            rightKneeVel=[0],
            rightAnkleVel=[0, 0, 0],
            rightShoulderVel=[0, 0, 0],
            rightElbowVel=[0],
            leftHipVel=[0, 0, 0],
            leftKneeVel=[0],
            leftAnkleVel=[0, 0, 0],
            leftShoulderVel=[0, 0, 0],
            leftElbowVel=[0]):

    self._basePos = basePos
    self._baseLinVel = baseLinVel
    self._baseOrn = baseOrn
    self._baseAngVel = baseAngVel

    self._chestRot = chestRot
    self._chestVel = chestVel
    self._neckRot = neckRot
    self._neckVel = neckVel

    self._rightHipRot = rightHipRot
    self._rightHipVel = rightHipVel
    self._rightKneeRot = rightKneeRot
    self._rightKneeVel = rightKneeVel
    self._rightAnkleRot = rightAnkleRot
    self._rightAnkleVel = rightAnkleVel

    self._rightShoulderRot = rightShoulderRot
    self._rightShoulderVel = rightShoulderVel
    self._rightElbowRot = rightElbowRot
    self._rightElbowVel = rightElbowVel

    self._leftHipRot = leftHipRot
    self._leftHipVel = leftHipVel
    self._leftKneeRot = leftKneeRot
    self._leftKneeVel = leftKneeVel
    self._leftAnkleRot = leftAnkleRot
    self._leftAnkleVel = leftAnkleVel

    self._leftShoulderRot = leftShoulderRot
    self._leftShoulderVel = leftShoulderVel
    self._leftElbowRot = leftElbowRot
    self._leftElbowVel = leftElbowVel

  def ComputeLinVel(self, posStart, posEnd, deltaTime):
    vel = [(posEnd[i] - posStart[i])/deltaTime for i in range(len(posStart))]
    return vel

  def ComputeAngVel(self, ornStart, ornEnd, deltaTime):
    dorn = p.getDifferenceQuaternion(ornStart, ornEnd)
    axis, angle = p.getAxisAngleFromQuaternion(dorn)
    angVel = [(x*angle) / deltaTime for x in axis]
    return angVel

  def ComputeAngVelRel(self, ornStart, ornEnd, deltaTime):
    ornStartConjugate = [-ornStart[0], -ornStart[1], -ornStart[2], ornStart[3]]
    _, q_diff = p.multiplyTransforms([0, 0, 0], ornStartConjugate, [0, 0, 0],
                                                        ornEnd)
    axis, angle = p.getAxisAngleFromQuaternion(q_diff)
    angVel = [axis_value * angle / deltaTime for axis_value in axis]
    return angVel

  def GetPose(self):
    pose = [0,
        self._basePos[0], self._basePos[1], self._basePos[2],
        self._baseOrn[0], self._baseOrn[1], self._baseOrn[2], self._baseOrn[3],
        self._chestRot[0], self._chestRot[1], self._chestRot[2], self._chestRot[3],
        self._neckRot[0], self._neckRot[1], self._neckRot[2], self._neckRot[3],
        self._rightHipRot[0], self._rightHipRot[1], self._rightHipRot[2], self._rightHipRot[3],
        self._rightKneeRot[0],
        self._rightAnkleRot[0], self._rightAnkleRot[1], self._rightAnkleRot[2], self._rightAnkleRot[3],
        self._rightShoulderRot[0], self._rightShoulderRot[1], self._rightShoulderRot[2], self._rightShoulderRot[3],
        self._rightElbowRot[0],
        self._leftHipRot[0], self._leftHipRot[1], self._leftHipRot[2], self._leftHipRot[3],
        self._leftKneeRot[0],
        self._leftAnkleRot[0], self._leftAnkleRot[1], self._leftAnkleRot[2], self._leftAnkleRot[3],
        self._leftShoulderRot[0], self._leftShoulderRot[1], self._leftShoulderRot[2], self._leftShoulderRot[3],
        self._leftElbowRot[0]
    ]
    return pose

  def InterpolateVector(self, startVector, endVector, fraction):
    assert len(startVector) == len(endVector), "startVector and EndVector must be the same length!"
    return [startVector[i] + fraction * (endVector[i] - startVector[i]) for i in range(len(startVector))]

  def getJointCurrentAndNextFrame(self, poseJoint, frameData, frameDataNext):
    jointDict = {
      'basePos': [1, 2, 3],
      'baseOrn': [5, 6, 7, 4],
      'chestRot': [9, 10, 11, 8],
      'neckRot': [13, 14, 15, 12],
      'rightHipRot': [17, 18, 19, 16],
      'rightKneeRot': [20],
      'rightAnkleRot': [22, 23, 24, 21],
      'rightShoulderRot': [26, 27, 28, 25],
      'rightElbowRot': [29],
      'leftHipRot': [31, 32, 33, 30],
      'leftKneeRot': [34],
      'leftAnkleRot': [36, 37, 38, 35],
      'leftShoulderRot': [40, 41, 42, 39],
      'leftElbowRot': [43],
    }
    currentFrameData = [frameData[i] for i in jointDict[poseJoint]]
    nextFrameData = [frameDataNext[i] for i in jointDict[poseJoint]]
    return currentFrameData, nextFrameData

  def SlerpJoint(self, poseJoint, frameData, frameDataNext, frameFraction, keyFrameDuration):
    jointStart, jointEnd = self.getJointCurrentAndNextFrame(poseJoint, frameData, frameDataNext)
    # Perform calculations in quaternion case
    if len(jointStart) == 4:
      orn = p.getQuaternionSlerp(jointStart, jointEnd, frameFraction)
      vel = self.ComputeAngVel(jointStart, jointEnd, keyFrameDuration)
    # Process everything else as a vector
    else:
      orn = self.InterpolateVector(jointStart, jointEnd, frameFraction)
      vel = self.ComputeLinVel(jointStart, jointEnd, keyFrameDuration)
    return orn, vel

  def Slerp(self, frameFraction, frameData, frameDataNext):
    keyFrameDuration = frameData[0]
    self._basePos, self._baseLinVel = self.SlerpJoint('basePos', frameData, frameDataNext, frameFraction, keyFrameDuration)
    self._baseOrn, self._baseAngVel = self.SlerpJoint('baseOrn', frameData, frameDataNext, frameFraction, keyFrameDuration)
    self._chestRot, self._chestVel = self.SlerpJoint('chestRot', frameData, frameDataNext, frameFraction, keyFrameDuration)
    self._neckRot, self._neckVel = self.SlerpJoint('neckRot', frameData, frameDataNext, frameFraction, keyFrameDuration)
    self._rightHipRot, self._rightHipVel = self.SlerpJoint('rightHipRot', frameData, frameDataNext, frameFraction, keyFrameDuration)
    self._rightKneeRot, self._rightKneeVel = self.SlerpJoint('rightKneeRot', frameData, frameDataNext, frameFraction, keyFrameDuration)
    self._rightAnkleRot, self._rightAnkleVel = self.SlerpJoint('rightAnkleRot', frameData, frameDataNext, frameFraction, keyFrameDuration)
    self._rightShoulderRot, self._rightShoulderVel = self.SlerpJoint('rightShoulderRot', frameData, frameDataNext, frameFraction, keyFrameDuration)
    self._rightElbowRot, self._rightElbowVel = self.SlerpJoint('rightElbowRot', frameData, frameDataNext, frameFraction, keyFrameDuration)
    self._leftHipRot, self._leftHipVel = self.SlerpJoint('leftHipRot', frameData, frameDataNext, frameFraction, keyFrameDuration)
    self._leftKneeRot, self._leftKneeVel = self.SlerpJoint('leftKneeRot', frameData, frameDataNext, frameFraction, keyFrameDuration)
    self._leftAnkleRot, self._leftAnkleVel = self.SlerpJoint('leftAnkleRot', frameData, frameDataNext, frameFraction, keyFrameDuration)
    self._leftShoulderRot, self._leftShoulderVel = self.SlerpJoint('leftShoulderRot', frameData, frameDataNext, frameFraction, keyFrameDuration)
    self._leftElbowRot, self._leftElbowVel = self.SlerpJoint('leftElbowRot', frameData, frameDataNext, frameFraction, keyFrameDuration)

    pose = self.GetPose()
    return pose
