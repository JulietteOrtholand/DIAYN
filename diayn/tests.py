import os
import tempfile

import gym

import torch
from torch.distributions.one_hot_categorical import OneHotCategorical

import matplotlib.pyplot as plt
import diayn


'''--------------------------------------------------------------------------------------------------------------------
                                                    FUNCTIONS
---------------------------------------------------------------------------------------------------------------------'''

def mountaincar(diayn):
    n = 10
    n_skills = diayn.prior.event_shape[0]
    for skill in range(n_skills):
        color = plt.rcParams["axes.color_cycle"][skill]
        for i in range(n):
            z = torch.zeros((1, diayn.prior.event_shape[0]))
            z[0, skill] = 1
            states = diayn.episode(train=False, z=z, return_states=True)
            positions = list(zip(*states))[0]
            kwargs = {"color": color, "alpha": .3}
            if i == 0:
                kwargs["label"] = "Skill n° {}".format(skill)
            plt.plot(positions, **kwargs)
    plt.xlabel("Step")
    plt.ylabel("$x$ position")
    plt.legend()
    plt.show()
    plt.pause(.1)

def build_diayn(n_skills=4, env_name="MountainCar-v0", alpha=.1):
    '''
    :param n_skills:
    :param env_name: "MountainCar-v0" or "Navigation2D"
    :return:
    '''
    env = gym.make(env_name)
    prior = OneHotCategorical(torch.ones((1, n_skills)))
    hidden_sizes = {s: [30,30] for s in ("actor", "discriminator", "critic")}
    model = diayn.DIAYN(env, prior, hidden_sizes, alpha=alpha)
    return model

'''--------------------------------------------------------------------------------------------------------------------
                                                    MAINS
---------------------------------------------------------------------------------------------------------------------'''

def main_train(n_skills):
    '''
    :param n_skills:
    :return:
    '''
    model = build_diayn(n_skills)
    model.train(10000)
    model.plot_rewards()
    plt.pause(1)
    #input("Press Enter to see skills")
    mountaincar(model, n_skills=n_skills)
    # plt.show() not needed since plt.ion() is called in diayn.py
    # plt.pause(1)
    path = "C:\\Users\\Juliette\\Dropbox\\ecole_ing\\DAC\\AS\\Projet\\DIAYN\\diayn\\models\\"
    model.save(path + "diayn")

def main_test():
    path = "C:\\Users\\Juliette\\Dropbox\\ecole_ing\\DAC\\AS\\Projet\\DIAYN\\diayn\\models\\diayn\\"

    diayn_mod = build_diayn(2)
    diayn_mod.load(path)
    mountaincar(diayn_mod)
    input('pause')

    pretrained = diayn.A2C.from_diayn(diayn_mod, 0)
    baseline = diayn.A2C(pretrained.env, {"actor": [30, 30], "critic": [30, 30]}, gamma=pretrained.gamma)

    titles = ["Baseline", "Pre-trained"]

    for i, model in enumerate((baseline, pretrained)):
        model.train(1000)
        plt.subplot(1, 2, i + 1)
        model.plot_rewards()
        plt.title(titles[i])
        plt.pause(.001)

    plt.tight_layout()
    plt.ioff()
    plt.show()


'''--------------------------------------------------------------------------------------------------------------------
                                                    START
---------------------------------------------------------------------------------------------------------------------'''


if __name__ == "__main__":
    n=1
    if n<2:
        main_train(5)
        input("Press Enter to quit")

    if n<3:
        main_test()
    
