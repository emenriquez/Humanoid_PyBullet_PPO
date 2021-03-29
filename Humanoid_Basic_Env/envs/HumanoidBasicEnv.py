import gym
import numpy as np
import pybullet as p

import time
import pybullet_data
import matplotlib.pyplot as plt

# Used to perform relative imports from the resources folder
from sys import path
path.append(".")

from Humanoid_Basic_Env.resources.humanoid import Humanoid
from Humanoid_Basic_Env.resources.target import Target
from Humanoid_Basic_Env.resources.plane import Plane

class HumanoidBasicEnv(gym.Env):
    '''
    Description here. Coming soon.
    '''

    def __init__(self):
        # Mocap file for imitation
        self.motionFile='Motions/humanoid3d_backflip.txt'

        # Actions
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1]*28, dtype=np.float32),
            high=np.array([1]*28, dtype=np.float32)
        )  

        # Observations
        observation_mins = [
            -15, -15, -15,
            -1, -1, -1, -1,
            -15, -15, -15,
            -15, -15, -15,
            -1, -15,
            -1, -15,
            -1, -15,
            -1, -15,
            -1, -1, -1, -1,
            -15, -15, -15,
            -1, -1, -1, -1,
            -15, -15, -15,
            -1, -1, -1, -1,
            -15, -15, -15,
            -1, -1, -1, -1,
            -15, -15, -15,
            -1, -1, -1, -1,
            -15, -15, -15,
            -1, -1, -1, -1,
            -15, -15, -15,
            -1, -1, -1, -1,
            -15, -15, -15,
            -1, -1, -1, -1,
            -15, -15, -15,
            -15, -15, -15,
            -15, -15, -15,
            -15, -15, -15,
            -15, -15, -15,
        ]
        observation_maxs = [-1*minimum for minimum in observation_mins]
        self.observation_space = gym.spaces.box.Box(
            low=np.array(observation_mins, dtype=np.float32),
            high=np.array(observation_maxs, dtype=np.float32)
        )

        # Basic environment settings
        self.np_random, _ = gym.utils.seeding.np_random()

        # Pybullet environment settings
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(fixedTimeStep=1./240)
        p.setPhysicsEngineParameter(numSolverIterations=12)
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

        # Pybullet Camera controls for rendering
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[4,0,5],
                                                            distance=15,
                                                            yaw=90,
                                                            pitch=-30,
                                                            roll=0,
                                                            upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)

        # Import URDF files
        planeId = Plane(self.client)
        self.agentId = Humanoid(self.client)

        self.target = Target(self.client, self.motionFile)

        # Initial state settings
        self.state = None
        self.done = False
        self.rendered_img = None
        self.touchingGround = False
        self.episode_reward = 0

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def collectObservations(self):
        observations_list = []
        observations_list.extend(self.agentId.collectObservations())

        return observations_list

    def reset(self):
        p.resetBasePositionAndOrientation(self.boxUid,
                                          [np.random.uniform(-4.5, 4.5),
                                           np.random.uniform(-4.5, 4.5),
                                           0.8],
                                          p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBasePositionAndOrientation(self.agentId,
                                          [0, 0, 6],
                                          p.getQuaternionFromEuler([90, 0, 0]))
        self.state = self.collectObservations()
        self.done = False
        self.episode_reward = 0
        self.ballPositionZ = 1
        self.step_counter = 0
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        reward = 0
        if self.done:
            # The last action ended the episode.
            # Ignore the current action and start a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if self.ballPositionZ < 0:
            reward = -1
            self.episode_reward -= 1
            self.done = True
        elif p.getContactPoints(self.agentId, self.boxUid):
            reward = 1 
            self.episode_reward += 1
            self.step_counter += 1
            p.resetBasePositionAndOrientation(self.boxUid, [np.random.uniform(
                -5, 5), np.random.uniform(-4.5, 4.5), 0.8], p.getQuaternionFromEuler([0, 0, 0]))
        else:
            reward = -0.01
            self.episode_reward -= 0.01
            self.step_counter += 1
            self.ballPositionZ = p.getBasePositionAndOrientation(self.agentId)[
                0][2]
            p.applyExternalForce(objectUniqueId=self.agentId, linkIndex=-1, forceObj=(
                20*action[0], 20*action[1], 0), posObj=p.getBasePositionAndOrientation(self.agentId)[
                0], flags=p.WORLD_FRAME)
        
        # Step the simulation
        p.stepSimulation()

        # Update state
        self.state = self.collectObservations()

        return np.array(self.state, dtype=np.float32), np.array(reward), self.done, dict()
    
    def render(self, mode='human'):
        # Create a new plot if one has not been made yet.
        if self.rendered_img is None:
            plt.axis('off')
            self.rendered_img = plt.imshow(np.zeros((720, 960, 4)))

        # Display image
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=self.view_matrix,
                                              projectionMatrix=self.proj_matrix,
                                              renderer=p.ER_TINY_RENDERER)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))
        self.rendered_img.set_data(rgb_array)

        # Add some data info to the plot to help with performance visualization
        annotation = plt.annotate(f'Step: {self.step_counter}\nEpisode Reward: {self.episode_reward:.3f}', xy=(0,0))
        plt.draw()
        plt.pause(.00001)
        
        # Remove the annotation or else it will persist and be typed over in the next step
        annotation.remove()

    def close(self):
        p.disconnect(self.client)


# Demo

# test = HumanoidBasicEnv()

# time_step = test.reset()

# reward = 0
# total_steps = 0
# while reward < 5:
#     while not time_step[2] == True:
#         time_step = test.step(
#             [np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
#         total_steps += 1
#         try:
#             test.render()
#         except:
#             quit()
#         if time_step[1] > 0:
#             print(time_step[1])
#             print(f'total steps: {total_steps}')
#             total_steps = 0
#             reward += time_step[1]

#     time_step = test.reset()


# minitest min-max values
[-0.6799079775810242, 0.02380305714905262, 1.1805833578109741,
-0.1667420640999807, -0.6905400446171615, 0.6981512593935416, -0.08908619335799633,
3.3253440856933594, 0.002926260232925415, 1.786172866821289,
-0.09215739836958381, -8.75097205024548, 0.35173200957394357,
-0.998702, 8.902032000000002,
0.868153, -0.14176000000000144,
-0.890919, 10.068096,
0.94688, 1.9909440000000007,
-0.08598599557477384, 0.047140997573912186, -0.055993997118297, 0.9936039488646707,
2.209115733354423, 1.2046558490051353, -0.8679064953512636,
0.019252264750213566, -0.09782824057445717, 0.04044566528419171, 0.9941947162441636,
-0.3813360381403612, -0.5773694894768511, -2.0656408153494112,
-0.026425993876067318, -0.12445097115985974, 0.9311397842186225, 0.3417489208034561,
-0.23969575388158404, 1.5421230470263936, 1.5275898575634983,
-0.020649305891224882, 0.11181417093320895, -0.1068271295003358, 0.9877546060370705,
0.006693770927006737, -1.2980716336641578, 1.3535037101334666,
-0.3124783593752776, 0.1640603545851045, 0.2232132152649167, 0.9086348746920528,
-0.7696849949897562, -1.6337595182954914, 3.52510452817576,
-0.0360090062730035, 0.09113101587561671, 0.8935971556704465, 0.4380440763101321,
-0.045205681906630435, 0.29419529459465643, 2.534765026422221,
0.020649305891224882, -0.11181417093320895, -0.1068271295003358, 0.9877546060370705,
-0.006693770927006737, 1.2980716336641578, 1.3535037101334666,
0.406211188383157, -0.5089544643860224, 0.13083838611967702, 0.747555442981642,
-2.713719844309651, -2.030431009817903, 4.682886022784714,
-1.259761615107517, -0.14609189718561952, 0.6601938464042352,
-0.9942192395145011, -0.21957797631863457, 0.8163674095889648,
-1.3147061946596212, 0.2253475068197199, 0.7289818014076757,
-0.9002054965988906, 0.2650005054017699, 0.9911003616929229
]