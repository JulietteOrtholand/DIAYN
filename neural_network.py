import torch
import torch.nn as nn

class NeuralNetwork(nn.Sequential):

    """Fully-connected neural network."""

    def __init__(self, in_size, out_size, hidden_sizes, 
                 activation=nn.Tanh):
        layers = []
        for size in hidden_sizes:
            layers.append(nn.Linear(in_size, size))
            layers.append(activation())
            in_size = size
        layers.append(nn.Linear(in_size, out_size))
        super(NeuralNetwork, self).__init__(*layers)
        
    @property
    def hidden_sizes(self):
        sizes = [
            c.in_features for c in self.children() if isinstance(c, nn.Linear)
        ]
        return sizes[1:]

    @property
    def in_size(self):
        sizes = [
            c.in_features for c in self.children() if isinstance(c, nn.Linear)
        ]
        return sizes[0]

    @property
    def out_size(self):
        sizes = [
            c.out_features for c in self.children() if isinstance(c, nn.Linear)
        ]
        return sizes[-1]
