import os
import tempfile
import gym
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical
import matplotlib.pyplot as plt

from diayn import DIAYN

def build_diayn(n_skills=2, env_name="MountainCar-v0"):
    env = gym.make(env_name)
    prior = OneHotCategorical(torch.ones((1, n_skills)))
    hidden_sizes = {s: [30, 30] for s in ("actor", "discriminator", "critic")}
    model = DIAYN(env, prior, hidden_sizes)
    return model

def main():
    model = build_diayn()
    model.train(20)
    model.plot_rewards()
    plt.pause(.0001)
    model.show_skill(skill=0)
    model.show_skill(skill=1)
    model.save(os.path.join(tempfile.gettempdir(), "diayn"))

if __name__ == "__main__":
    main()
    input("Press Enter to quit")
    
