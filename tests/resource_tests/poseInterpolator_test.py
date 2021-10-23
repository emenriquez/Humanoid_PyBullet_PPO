from sys import path
path.append('.')
import pybullet as p
from pybullet_utils import bullet_client

clientID = p.connect(p.DIRECT)

from Humanoid_Basic_Env.resources.poseInterpolator import PoseInterpolator

# initialize interpolator
test = PoseInterpolator()

# check that interpolator initialized correctly
print(test.GetPose())

# Sample frames to test interpolation
sampleFrame=[0.062500, 0.000000, 0.886733, 0.000000, 0.999412, 0.029215, -0.000525, -0.017963, 0.999985, 0.000432, 0.000572, 0.005500, 0.9897717, 0.001660, -0.0111650, -0.140564, 0.998945, -0.024450, -0.0008386, -0.0388686, -0.014186, 0.988643, 0.010702, -0.104681, 0.035223, 0.989539, -0.003003, -0.124234, 0.073280, 0.240463, 0.999678, -0.020920, -0.012925, -0.00630, -0.027859, 0.988643, -0.010702, 0.104681, -0.009223, 0.993344, 0.055661, -0.019608, 0.098917, 0.148934]
sampleFrameNext=[0.062500, -0.020268, 0.909379, 0.000735, 0.999197, 0.030395, 0.008425, -0.024696, 0.999998, -0.000024, -0.001583, -0.001104, 0.989013, 0.004416, -0.008083, -0.145576, 0.999424, -0.033001, -0.002606, -0.007386, -0.014254, 0.987239, 0.011060, -0.101618, -0.102208, 0.965615, -0.104331, -0.207997, 0.115930, 0.233349, 0.999385, -0.014931, -0.012806, 0.029035, -0.064696, 0.987239, -0.011060, 0.101618, -0.102208, 0.972982, 0.174223, 0.068809, 0.134972, 0.171986]


# Test Slerp
print(test.Slerp(0.5, frameData=sampleFrame, frameDataNext=sampleFrameNext))