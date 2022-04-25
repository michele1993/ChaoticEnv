import gym
import numpy as np

# Ensure continuous MountainCar always starts in the same position

class ModifiedInvMCar(gym.Wrapper):

    def __init__(self,env, easier):

        super(ModifiedInvMCar,self).__init__(env)

        if easier:
            self.unwrapped.goal_position = 0.25


    def reset(self): # Ensure each episode starts with the pendulum in the same upward position


        super().reset()
        self.unwrapped.state = np.array([- 0.5, 0])

        return np.array(self.unwrapped.state)