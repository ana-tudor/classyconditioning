import gym
import time
from PIL import Image
from matplotlib import pyplot as plt


env = gym.make("procgen:procgen-fruitbot-v0")
obs = env.reset()
plt.ion()
while True:
    obs, reward, done, info = env.step(env.action_space.sample())
    if done:
        break

    print(obs.shape)
    env.render()
    plt.imshow(obs, interpolation='nearest')

    time.sleep(.1)

env.close()
