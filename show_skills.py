import torch
import matplotlib.pyplot as plt

def cartpole(diayn, n_skills=None):
    n = 10
    if n_skills is None:
        n_skills = diayn.prior.event_shape[0]
    for skill in range(n_skills):
        color = plt.rcParams["axes.prop_cycle"][skill]
        for i in range(n):
            z = torch.zeros((1, diayn.prior.event_shape[0]))
            z[0, skill] = 1
            states = diayn.episode(train=False, z=z, return_states=True)
            positions = list(zip(*states))[0]
            kwargs = {"color": color, "alpha": .3}
            if i == 0:
                kwargs["label"] = "Skill n° {}".format(skill + 1)
            plt.plot(positions, **kwargs)
    plt.xlabel("Step")
    plt.ylabel("$x$ position")
    plt.legend()
    plt.show()
    plt.pause(.1)
    
def lunarlander(diayn,filename, n_skills=None):
    n = 20
    plt.figure()
    if n_skills is None:
        n_skills = diayn.prior.event_shape[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for skill in range(n_skills):
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
    plt.savefig(filename)


def navigation2d(diayn):
    n_skills = diayn.prior.event_shape[0]
    plt.figure()
    plt.ylim(-0.1, 1.1)
    plt.xlim(-0.1, 1.1)
    for skill in range(n_skills):
        z = torch.zeros((1, diayn.prior.event_shape[0]))
        z[0, skill] = 1
        print("Showing skill {}".format(z.argmax(1).item()))
        diayn.episode(train=False, render=False, z=z)
        plt.plot(
            [i for i, j in diayn.env.memory],
            [j for i, j in diayn.env.memory],
            label = "Skill n° {}".format(skill)
        )
    plt.legend()
    plt.show()
    plt.pause(5)

def mountaincar(diayn,filename):
    n = 1
    plt.figure()
    n_skills = diayn.prior.event_shape[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    for skill in range(n_skills):
        color = colors[skill]
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
    plt.pause(1)
    plt.savefig(filename)
    plt.close()

