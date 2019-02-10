import diayn
import matplotlib.pyplot as plt
import gym

import torch
from torch.distributions.one_hot_categorical import OneHotCategorical



def lunarlander(diayn, n_skills=None):
    n = 10
    if n_skills is None:
        n_skills = diayn.prior.event_shape[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for skill in range(n_skills):
        print(plt.rcParams["axes.prop_cycle"])
        color = colors[skill]
        for i in range(n):
            z = torch.zeros((1, diayn.prior.event_shape[0]))
            z[0, skill] = 1
            states = diayn.episode(train=False, z=z, return_states=True)
            x, y = list(zip(*states))[:2]
            kwargs = {"color": color, "alpha": .3}
            if i == 0:
                kwargs["label"] = "Skill n° {}".format(skill + 1)
            plt.plot(x, y, **kwargs)
    plt.xlabel("$x$ position")
    plt.ylabel("$y$ position")
    plt.legend()
    plt.show()
    plt.pause(1)


def build_diayn(n_skills=4, env_name="LunarLander-v2", alpha=.1):
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


path = "C:\\Users\\Juliette\\Dropbox\\ecole_ing\\DAC\\AS\\Projet\\DIAYN\\diayn\\models\\diayn\\"

diayn_mod = build_diayn(2)
diayn_mod.load(path)
lunarlander(diayn_mod, n_skills=2)
input('pause')

pretrained = diayn.A2C.from_diayn(diayn_mod, 0)
baseline = diayn.A2C(pretrained.env, {"actor": [30,30], "critic": [30,30]}, gamma=pretrained.gamma)

titles = ["Baseline", "Pre-trained"]

for i, model in enumerate((baseline,pretrained)):
    model.train(10000)
    plt.subplot(1, 2, i + 1)
    model.plot_rewards()
    plt.title(titles[i])
    plt.pause(.001)
    
plt.tight_layout()
plt.ioff()
plt.show()
