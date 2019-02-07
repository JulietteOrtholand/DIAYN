import diayn

import matplotlib.pyplot as plt

pretrained = diayn.A2C.from_diayn(diayn.models.mountaincar, 0)
baseline = diayn.A2C(pretrained.env, {"actor": [300], "critic": [300]}, gamma=pretrained.gamma)

titles = ["Baseline", "Pre-trained"]

for i, model in enumerate((baseline, pretrained)):
    model.train(10000)
    plt.subplot(1, 2, i + 1)
    model.plot_rewards()
    plt.title(titles[i])
    plt.pause(.001)
    
plt.tight_layout()
plt.ioff()
plt.show()
