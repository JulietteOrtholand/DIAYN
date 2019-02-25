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
import datetime


def show_skill_2D(model, nb_skills, filename):
    """Run one episode for each skill, plot them all on the same graph and save it"""
    plt.figure()
    plt.ylim(-0.1, 1.1)
    plt.xlim(-0.1, 1.1)
    for skill in range(nb_skills):
        z = torch.zeros((1, model.prior.event_shape[0]))
        z[0, skill] = 1
        print("Showing skill {}".format(z.argmax(1).item()))
        model.episode(train=False, render=False, z=z)
        plt.plot([i for i, j in model.env.memory], [j for i, j in model.env.memory],
                 label=str(skill))
    plt.legend()
    plt.savefig(filename)
    # plt.show()
    plt.pause(1)
    plt.close()


def build_diayn(n_skills=4, env_name="Navigation2D", alpha=0.1, gamma=0.1, seed=101):
    '''
    :param n_skills:
    :param env_name: "MountainCar-v0" or "Navigation2D"
    :alpha=0.1,
    :gamma=0.1,
    :seed = 101
    :return:
    '''
    if env_name == "Navigation2D":
        env = Navigation2D(20)
    else:
        env = gym.make(env_name)
    prior = OneHotCategorical(torch.ones((1, n_skills)))
    hidden_sizes = {s: [30, 30] for s in ("actor", "discriminator", "critic")}
    model = DIAYN(env, prior, hidden_sizes, alpha, gamma, seed=seed)
    return model


def main(n_skills):
    '''

    :param n_skills:
    :return:
    '''
    folder = "tests_stabilisations" + str(datetime.datetime.today()).replace(":", "-").replace(" ", "-") + "\\"
    if not os.path.exists(folder):
        os.makedirs(folder)

    alpha = 0.1
    gamma = 0.9
    seeds = [100, 101, 102]
    train_iter = 150

    seed_storage = []
    for seed in seeds:
        model = build_diayn(n_skills, "Navigation2D", alpha, gamma, seed)
        model.train(train_iter)
        seed_storage.append(model.rewards)
    pd_info = pd.DataFrame(seed_storage, index=seeds)
    writer = pd.ExcelWriter(folder + 'TEST2' + ',alpha=' + str(alpha) + ',gamma=' + str(gamma) + '.xlsx')
    pd_info.to_excel(writer, 'Sheet1')
    writer.save()

    plt.figure()
    for i in range(len(seeds)):
        #plt.plot(seed_storage[i], label = "seed " +str(seeds[i]))
        plt.plot(range(99, len(seed_storage[i])),
                 np.convolve(seed_storage[i], np.ones(100) / 100, "valid"), label = "seed " +str(seeds[i]))
    plt.title("Evolution des récompenses selon l'initialisation des graines aléatoires")
    plt.xlabel("Itérations")
    plt.ylabel("Récompenses")
    plt.legend()
    plt.show()

    plt.savefig(folder)


if __name__ == "__main__":
    main(4)
    input("Press Enter to quit")
