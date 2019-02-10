import os
import tempfile
import gym
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical
import matplotlib.pyplot as plt
from navigation2D import Navigation2D
from diayn import DIAYN
import pandas as pd
import numpy as np
from show_skills import lunarlander, mountaincar


def build_diayn(n_skills=4, env_name="MountainCar-v0", alpha=0.1, gamma=0.1, seed = 101):
    '''
    :param n_skills:
    :param env_name: "MountainCar-v0" or "Navigation2D"
    :alpha=0.1,
    :gamma=0.1,
    :seed = 101
    :return:
    '''
    if env_name == "Navigation2D" :
        env = Navigation2D(20)
    else :
        env = gym.make(env_name)
    prior = OneHotCategorical(torch.ones((1, n_skills)))
    hidden_sizes = {s: [30, 30] for s in ("actor", "discriminator", "critic")}
    model = DIAYN(env, prior, hidden_sizes,  alpha, gamma, seed = seed)
    return model


def main(n_skills):
    '''

    :param n_skills:
    :return:
    '''
    folder = '0127_test3\\'
    alphas = [0.8]
    gammas= [0.2]
    seeds = [100,101,102]
    train_iter = 500
    mean_val = 100

    for seed in seeds:
        rewards_storage = []
        for alpha in alphas:
            alpha_store = []
            for gamma in gammas:
                filename = 'seed=' + str(seed) + ',alpha=' + str(alpha) + ',gamma=' + str(gamma) +'.png'
                rF = folder + 'rewards=' + filename
                model = build_diayn(n_skills, "MountainCar-v0", alpha, gamma, seed)
                model.train(train_iter)
                model.plot_rewards(rF)
                end = np.mean(model.rewards[len(model.rewards)-1-mean_val:])
                start = np.mean(model.rewards[: mean_val])
                alpha_store.append(end - start)
                # plt.pause(1)
                #input("Press Enter to see skills")
                mountaincar(model, folder + 'skills=' + filename)
                # plt.show() not needed since plt.ion() is called in diayn.py
                # plt.pause(1)
                model.save(os.path.join(tempfile.gettempdir(), "diayn"))
            rewards_storage.append(alpha_store)
        pd_info = pd.DataFrame(rewards_storage, columns = gammas, index = alphas)
        writer = pd.ExcelWriter(folder + 'seed=' + str(seed) + '.xlsx')
        pd_info.to_excel(writer, 'Sheet1')
        writer.save()

if __name__ == "__main__":
    main(2)
    input("Press Enter to quit")