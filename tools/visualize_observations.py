import gym
import time
from matplotlib import pyplot as plt
import argparse

# Use this script to visualize Environment Observations.
# The observations processed by our algorithms are smaller versions of the rendered
# environments. This script helps visualize the difference with a random agent
# python visualize_observations.py
# --hide_obs hides the observations and only shows the rendered game
# --fps is the frames per second; slow it down for easier comparison
# --mode by default easy mode
# --env by default fruitbot, but it works for other environments as well

def main():
    parser = argparse.ArgumentParser(description='Visualize Environment Observations')
    parser.add_argument('--hide_obs', default=False, action='store_true')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--mode', type=str, default='easy')
    parser.add_argument('--env', type=str, default="procgen:procgen-fruitbot-v0")
    args = parser.parse_args()
    fps = args.fps
    mode = args.mode
    env = args.env
    hide_obs = args.hide_obs
    run_game(env, mode, hide_obs, fps)

def run_game(env="procgen:procgen-fruitbot-v0", mode='easy', hide_obs=False, fps=30):
    env = gym.make("procgen:procgen-fruitbot-v0", distribution_mode=mode)
    obs = env.reset()
    if not hide_obs:
        plt.ion()
    while True:
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            break
        env.render()
        if not hide_obs:
            plt.imshow(obs, interpolation='nearest')
        time.sleep(1/fps)
    env.close()

if __name__ == '__main__':
    main()
