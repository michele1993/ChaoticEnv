import gym
import numpy as np

class ModifiedCartPole(gym.Wrapper):

    def __init__(self,env):

        super(ModifiedCartPole,self).__init__(env)



    def reset(self): # Ensure each episode starts with the pendulum in the same upward position


        super().reset()
        self.unwrapped.state = np.array([0,0,0,0])

        return self.unwrapped.state