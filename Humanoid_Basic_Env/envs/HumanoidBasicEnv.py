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

class HumanoidBasicEnv(gym.Env):
    '''
    Description here. Coming soon.
    '''

    def __init__(self):
        # Actions
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1,-1], dtype=np.float32),
            high=np.array([1,1], dtype=np.float32)
        )  

        # Observations
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-10, -10, -10, -10, -10, -10, -10, -10,], dtype=np.float32),
            high=np.array([10,10,10,10,10,10,10,10,], dtype=np.float32)
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

        # Import URDF
        planeId = Plane(self.client)

        agentStartPos = [0,0,5]
        agentStartOrientation = p.getQuaternionFromEuler([0,0,0])
        self.agentId = p.loadURDF('Humanoid_Basic_Env/resources/humanoid.urdf', agentStartPos, agentStartOrientation)

        goal = Goal(self.client,base=[0,0,0.4])
        self.boxUid = goal.get_ids()

        # Initial state settings
        self.state = None
        self.done = False
        self.rendered_img = None
        self.ballPositionZ = 1
        self.episode_reward = 0

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def collectObservations(self):
        observations_list = []
        observations_list.append(
            p.getBasePositionAndOrientation(self.boxUid)[0]),
        observations_list.append(
            p.getBasePositionAndOrientation(self.agentId)[0])
        observations_list.append(p.getBaseVelocity(self.agentId)[0][:2])

        return [item for observation in observations_list for item in observation]

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
