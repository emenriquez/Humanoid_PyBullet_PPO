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
        self.motionFile='Motions/humanoid3d_walk.txt'

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
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.setRealTimeSimulation(0)

        # Pybullet Camera controls for rendering
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[4,1,5],
                                                            distance=2,
                                                            yaw=60,
                                                            pitch=-30,
                                                            roll=0,
                                                            upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)

        # Import URDF files
        self.plane = Plane(self.client)
        self.planeID = self.plane.get_ids()
        self.agent = Humanoid(self.client)
        self.agentID = self.agent.get_ids()
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
        observations_list.extend(self.agent.collectObservations())

        return observations_list

    def agentFallCheck(self):
        contactPoints = p.getContactPoints(
            bodyA=self.agentID,
            bodyB=self.planeID,
        )
        # point[3] indicates the link index of the agent body touching the ground
        # in this statement, link 5 is rFoot, and link 11 is lFoot
        # if any other link touches the ground, we assume the agent has fallen
        return any([True for point in contactPoints if point[3] != 5 and point[3] != 11])

    def motionCompleted(self):
        return self.target.checkIfLastFrame()

    def reset(self):
        randomStart = self.target.randomStartFrame()
        self.agent.setStartingPositionAndVelocity(inputFrame=randomStart)
        self.state = self.collectObservations()
        self.done = False
        self.episode_reward = 0
        self.step_counter = 0
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        reward = 0
        if self.done:
            # The last action ended the episode.
            # Ignore the current action and start a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if self.agentFallCheck():
            reward = -1
            self.episode_reward -= 1
            self.done = True
        elif self.motionCompleted():
            self.step_counter += 1
            self.done = True
        else:
            pose = self.agent.collectObservations()
            reward = self.target.totalImitationReward(agentPose=pose)
            self.episode_reward += reward
            self.step_counter += 1
            self.agent.applyActions(actions=action)
        
        # Step the simulation 16 times to increment frame by deltaTime
        for i in range(1):
            p.stepSimulation()

        # Update state
        self.state = self.collectObservations()
        # move target to the next frame pose
        self.target.nextFrame()

        return np.array(self.state, dtype=np.float32), np.array(reward), self.done, dict()
    
    def render(self, mode='human'):
        # Create a new plot if one has not been made yet.
        if self.rendered_img is None:
            plt.axis('off')
            self.rendered_img = plt.imshow(np.zeros((720, 960, 4)))

        # Pybullet Camera controls for rendering
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.target.targetPose[0:3], # Use self.state[0:3] to track agent instead of target
                                                            distance=4,
                                                            yaw=60,
                                                            pitch=-30,
                                                            roll=0,
                                                            upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)

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
        plt.pause(0.0625)
        
        # Remove the annotation or else it will persist and be typed over in the next step
        annotation.remove()

    def close(self):
        p.disconnect(self.client)


# Demo

# test = HumanoidBasicEnv()

# time_step = test.reset()

# reward = 0
# total_steps = 0
# while total_steps<50:
#     while not time_step[2] == True:
#         time_step = test.step(np.random.uniform(low=-1, high=1, size=(28,)))
#         total_steps += 1
#         try:
#             test.render()
#         except:
#             quit()
#         if time_step[1] > 0:
#             print(time_step[1])
#             print(f'total steps: {total_steps}')
#             reward += time_step[1]

#     time_step = test.reset()

