import os
import tempfile

import gym

import torch
from torch.distributions.one_hot_categorical import OneHotCategorical

import matplotlib.pyplot as plt

from .navigation2D import Navigation2D
from .diayn import DIAYN

def build_diayn(n_skills=4, env_name="Navigation2D", alpha=.1):
    '''
    :param n_skills:
    :param env_name: "MountainCar-v0" or "Navigation2D"
    :return:
    '''
    if env_name == "Navigation2D" :
        env = Navigation2D(10)
    else :
        env = gym.make(env_name)
    prior = OneHotCategorical(torch.ones((1, n_skills)))
    hidden_sizes = {s: [300] for s in ("actor", "discriminator", "critic")}
    model = DIAYN(env, prior, hidden_sizes, alpha=alpha)
    return model

def main(n_skills):
    '''
    :param n_skills:
    :return:
    '''
    model = build_diayn(n_skills)
    model.train(10)
    model.plot_rewards()
    # plt.pause(1)
    input("Press Enter to see skills")
    for i in range(n_skills):
        model.show_skill(skill=i, wait_before_closing=True)
    # plt.show() not needed since plt.ion() is called in diayn.py
    # plt.pause(1)
    model.save(os.path.join(tempfile.gettempdir(), "diayn"))

if __name__ == "__main__":
    main(2)
    input("Press Enter to quit")
    
