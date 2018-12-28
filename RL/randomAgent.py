import argparse
import sys
import matplotlib
matplotlib.use("TkAgg")
import gym
import envs
#import gridworld_env
from gym import wrappers, logger
import numpy as np
import copy

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        print(self.action_space)

    def act(self, observation, reward, done):
        self.action_space.sample()
        return f





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='gridworld-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    envx = gym.make(args.env_id)



    outdir = 'gridworld-v0/random-agent-results'

    env = wrappers.Monitor(envx, directory=outdir, force=True, video_callable=False)

    env.seed(0)


    episode_count = 1000000
    reward = 0
    done = False
    envx.verbose = True

    envx.setPlan("gridworldPlans/plan0.txt",{0:-0.001,3:1,4:1,5:-1,6:-1})

    agent = RandomAgent(envx.action_space)
    #np.random.seed(5)
    rsum=0

    for i in range(episode_count):
        ob = env.reset()

        if i % 100 == 0 and i > 0:
            envx.verbose = True
        else:
            envx.verbose = False

        if envx.verbose:
            envx.render(1)
        j = 0
        #print(str(ob))
        while True:

            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            rsum+=reward
            j += 1
            if envx.verbose:
                envx.render()
            if done:
                print(ob)
                print(str(i)+" rsum="+str(rsum)+", "+str(j)+" actions")
                rsum=0
                break

    print("done")
    env.close()