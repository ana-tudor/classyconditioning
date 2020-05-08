import gym
import procgen
import time
import numpy as np
from abc import ABC, abstractmethod

class FruitbotModel(ABC):
    
    def __init__(self):
        super().__init__()
        
    """
    TO IMPLEMENT:
    
    Compute actions according to the model, return a vector of actions to take in the vectorized environment
    
    ==Inputs==
    state: State vector of size (N, 64, 64, 3)
    
    ==Outputs==
    action: An integer vector of actions of size (N), in which each action value is contained within range(0, 15).
        i.e., must be a valid input for venv.step()
    """
    @abstractmethod
    def step(self, state):
        pass

    
    
    """
    Simple implementation for epsilon-greedy exploration in vectorized form. Dependent on implementation for
    FruitbotModel.step()
    """
    def step_with_explore(self, state, epsilon):
        agent_step = self.step(state)
        
        # Keep probability for agent actions
        mask = np.random.sample(agent_step.size) > epsilon
        
        return np.where(mask, agent_step, np.random.choice(a = 15, size = agent_step.size))
        
        
    
    
    """
    Train the model at a certain timestep. 
    """
    @abstractmethod
    def train(self):
        pass
    
    """
    Train the model according to state, action, reward information obtained from the environment. 
    """
    @abstractmethod
    def train(self, state0, action, state1, reward):
        pass
    

    
    
"""
Base model implementation that always takes random actions.
"""
class BaseModel(FruitbotModel):
    
    def __init__(self, num_envs):
        self.N = num_envs
    
    def step(self, state):
        return np.random.choice(15, self.N)
    
    def train(self, state0, action, state1, reward):
        pass
    

    
    
    
    
    
    
"""
Sample training loop

TODO:
* Update to track timesteps

"""

num_envs = 16

venv = procgen.ProcgenEnv(num_envs=num_envs, env_name="fruitbot", distribution_mode = 'easy')
model = BaseModel(num_envs)
state0 = venv.reset()

while True:
    
    # Advance environment by one timestep
    
    action = model.step_with_explore(state0, .25)
    state1, rew, done, info = venv.step(action)
    
    # Train environment on observations
    model.train(state0, action, state1, rew)
    state0 = state1
    venv.render()
    
    if np.all(done):
        break
    # time.sleep(.05)
venv.close()


