import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from math import pi

class hills(gym.Env):

    actions = {0:(1,0),1:(-1,0),2:(0,1),3:(0,-1),\
               4:(7,0),5:(-7,0),6:(0,7),7:(0,-7)}

    def __init__(self, init_position = (0,0)):

        self.state = init_position
        self.h     = self._height(*self.state)
        self.action_space = spaces.Discrete(8)
        self.seed()

    # Function takes position (x,y) returns hight update later : https://en.wikipedia.org/wiki/Test_functions_for_optimization
    # Lévi function N.13
    # Global max f(1,1)=0

    def _height(self, x,y):
        val = np.sin(3*pi*x)**2 + (x-1)**2*(1+np.sin(3*pi*y)**2)+(y-1)**2*(1+np.sin(2*pi*y)**2)
        return -val

    def step(self, action):

        # The second function is the step function, that will take an action variable
        # and will return the a list of four things — the next state,
        # the reward for the current state, a boolean representing whether
        # the current episode of our model is done
        # and some additional info on our problem.

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        f0 = self.h
        x, y = self.state
        delta_x , delta_y = self.actions[action]
        x+=delta_x
        y+=delta_y
        new_state = (x,y)

        f1 = self._height(*new_state)
        reward = f1 - f0   #  delta f
        done = bool(x<10 or x>-10 or y<10 or y >10) # agent moved outside defined boundary

        self.state = new_state
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = (0,0)
        # resets the state and other variables of the environment to the start state
        return np.array(self.state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        raise NotImplementedError
