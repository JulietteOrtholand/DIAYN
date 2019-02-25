import os
import tempfile

import gym

import torch
from torch.distributions.one_hot_categorical import OneHotCategorical

import matplotlib.pyplot as plt
import diayn
from a2c import A2C
from show_skills import lunarlander, mountaincar
import os
import datetime
import numpy as np

'''--------------------------------------------------------------------------------------------------------------------
                                                    FUNCTIONS
---------------------------------------------------------------------------------------------------------------------'''

def build_diayn(n_skills=4, env_name="MountainCar-v0", alpha=.1):
    '''
    :param n_skills:
    :param env_name: "MountainCar-v0" or "Navigation2D"
    :return:
    '''
    env = gym.make(env_name)
    alpha = .1
    gamma = .9
    prior = OneHotCategorical(torch.ones((1, n_skills)))
    hidden_sizes = {s: [30,30] for s in ("actor", "discriminator", "critic")}
    model = diayn.DIAYN(env, prior, hidden_sizes, alpha=alpha, gamma = gamma)
    return model

def lunarlander_baseline(diayn, filename):
    n = 20
    plt.figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color = colors[0]
    for i in range(n):
        states = diayn.episode(train=False,return_states=True)
        x, y = list(zip(*states))[:2]
        kwargs = {"color": color, "alpha": .3}
        if i == 0:
            kwargs["label"] = "Skill n° {}".format(1)
        plt.plot(x, y, **kwargs)
    plt.xlabel("$x$ position")
    plt.ylabel("$y$ position")
    plt.legend()
    plt.show()
    plt.pause(1)
    plt.savefig(filename)


def mountaincar_baseline(diayn,filename):
    n = 10
    plt.figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    color = colors[0]
    for i in range(n):
        states = diayn.episode(train=False, return_states=True)
        positions = list(zip(*states))[0]
        kwargs = {"color": color, "alpha": .3}
        if i == 0:
            kwargs["label"] = "Skill n° {}".format(1)
        #print(positions)
        plt.plot(positions, **kwargs)
    plt.xlabel("Step")
    plt.ylabel("$x$ position")
    plt.legend()
    plt.show()
    plt.pause(1)
    plt.savefig(filename)
    #plt.close()


'''--------------------------------------------------------------------------------------------------------------------
                                                    MAINS
---------------------------------------------------------------------------------------------------------------------'''

def dyian_train(n_skills, path):
    '''
    :param n_skills:
    :return:
    '''
    model = build_diayn(n_skills)
    path_plot = path + "plot_diayn\\"
    if not os.path.exists(path_plot):
        os.makedirs(path_plot)

    path_save = path + "save_diayn\\"
    if not os.path.exists(path_save):
        os.makedirs(path_save)


    for k in range(0, 1):
        iter_ = 200
        model.train(iter_)
        model.plot_rewards(path_plot + "diyan_train_rewards_" + str((k+1)*iter_))
        #input("Press Enter to see skills")
        mountaincar(model, path_plot + "diyan_train_trajectoires_" + str((k+1)*iter_))
        # plt.show() not needed since plt.ion() is called in diayn.py
        # plt.pause(1)
        model.save(path_save)

def dyian_test(path):

    path_plot = path + "plot_pt\\"
    if not os.path.exists(path_plot):
        os.makedirs(path_plot)

    path_save = path + "save_pt\\"
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    diayn_mod = build_diayn(2)
    diayn_mod.load(path + "save_diayn\\")
    mountaincar(diayn_mod,  path_plot + "pretrained_trajectoires_0")

    pretrained = A2C.from_diayn(diayn_mod, 0)

    model=pretrained
    for k in range(0, 1):
        iter_ = 200
        model.train(iter_)
        model.plot_rewards(path_plot + "pretrained_rewards_" + str((k + 1) * iter_))
        mountaincar_baseline(model, path_plot + "pretrained_trajectoires_" + str((k + 1) * iter_))
        model.save(path_save)

def baseline_test(path):
    path_plot = path + "plot_bl\\"
    if not os.path.exists(path_plot):
        os.makedirs(path_plot)

    path_save = path + "save_bl\\"
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    env = gym.make("MountainCar-v0")
    baseline = A2C(env, {"actor": [30, 30], "critic": [30, 30]}, gamma=0.99)
    model=baseline
    for k in range(0,1):
        iter_ = 200
        model.train(iter_)
        model.plot_rewards(path_plot + "baseline_rewards_" + str((k+1)*iter_))
        mountaincar_baseline(model, path_plot + "baseline_trajectoires_" + str((k+1)*iter_))
        model.save(path_save)

def plot_results(path):
    env = gym.make("MountainCar-v0")
    baseline = A2C(env, {"actor": [30, 30], "critic": [30, 30]}, gamma=0.99)
    baseline.load(path + "save_bl\\")

    diayn_mod = build_diayn(2)
    diayn_mod.load(path + "save_diayn\\")
    pretrained = A2C.from_diayn(diayn_mod, 0)
    pretrained.load(path + "save_pt\\")

    plt.figure()
    plt.plot(range(99, len(pretrained.rewards)),
             np.convolve(pretrained.rewards, np.ones(100) / 100, "valid"), label = "pretrained")
    plt.plot(range(99, len(baseline.rewards)),
             np.convolve(baseline.rewards, np.ones(100) / 100, "valid"), label = "baseline")
    plt.legend()
    plt.show()
    plt.savefig(path + "results")
    plt.pause(1)



'''--------------------------------------------------------------------------------------------------------------------
                                                    START
---------------------------------------------------------------------------------------------------------------------'''


if __name__ == "__main__":
    k =0
    n_skills = 2
    seed = 100
    torch.manual_seed(seed)

    if k<1 :
        path = "tests_moutain_car_" + str(datetime.datetime.today()).replace(":", "-").replace(" ", "-") + "\\"
        if not os.path.exists(path):
            os.makedirs(path)

        print("Stage 1 : Train DIAYN")
        print("Start : " + str(datetime.datetime.today()))
        dyian_train(n_skills, path)
        print("End : " + str(datetime.datetime.today()))
    else :
        path = "tests_2019-02-10-18-26-53.813433\\"

    if k<2 :
        print("Stage 2 : Test from DIAYN")
        print("Start : " + str(datetime.datetime.today()))
        dyian_test(path)
        print("End : " + str(datetime.datetime.today()))
    
    if k<3 : 
        print("Stage 3 : Test baseline")
        print("Start : " + str(datetime.datetime.today()))
        baseline_test(path)
        print("End : " + str(datetime.datetime.today()))

    if k<4 :
        print("Stage 4 : Plot Results")
        print("Start : " + str(datetime.datetime.today()))
        plot_results(path)
        print("End : " + str(datetime.datetime.today()))


    input("pause")
