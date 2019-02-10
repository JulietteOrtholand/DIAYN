import os
import json
import time
import shutil
import numpy as np

import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from neural_network import NeuralNetwork


class Actor(nn.Module):

    def __init__(self, env, hidden_sizes, **kwargs):
        super(Actor, self).__init__()
        in_size = env.observation_space.shape[0]
        out_size = env.action_space.n
        self.network = NeuralNetwork(in_size, out_size, hidden_sizes, **kwargs)

    def act(self, s):
        return Categorical(torch.exp(self(s, z))).sample().item()

    def forward(self, s):
        x = torch.Tensor([s])
        return F.log_softmax(self.network(x), dim=1)


class ActorFromDIAYN(nn.Module):

    def __init__(self, actor, skill, **kwargs):
        super(ActorFromDIAYN, self).__init__()
        # Build network
        in_size = actor.network.in_size
        out_size = actor.network.out_size
        hidden_sizes = actor.network.hidden_sizes
        self.network = NeuralNetwork(in_size, out_size, hidden_sizes, **kwargs)
        self.network.load_state_dict(actor.network.state_dict())
        # Store skill
        self.skill = skill

    def act(self, s):
        return Categorical(torch.exp(self(s))).sample().item()

    def forward(self, s):
        s = torch.Tensor([s])
        x = torch.cat((s, self.skill), dim=1)
        return F.log_softmax(self.network(x), dim=1)


class Critic(nn.Module):

    def __init__(self, env, hidden_sizes, **kwargs):
        super(Critic, self).__init__()
        in_size = env.observation_space.shape[0]
        out_size = 1
        self.network = NeuralNetwork(in_size, out_size, hidden_sizes, **kwargs)

    def forward(self, s):
        x = torch.Tensor([s])
        return self.network(x)


class CriticFromDIAYN(nn.Module):

    def __init__(self, critic, skill, **kwargs):
        super(CriticFromDIAYN, self).__init__()
        in_size = critic.network.in_size
        out_size = critic.network.out_size
        hidden_sizes = critic.network.hidden_sizes
        self.network = NeuralNetwork(in_size, out_size, hidden_sizes, **kwargs)
        self.network.load_state_dict(critic.network.state_dict())
        self.skill = skill

    def forward(self, s):
        s = torch.Tensor([s])
        x = torch.cat((s, self.skill), dim=1)
        return self.network(x)


class A2C:

    """Online A2C."""

    @staticmethod
    def from_diayn(model, skill):
        """
        Return A2C initialized with DIAYN weights.
        
        Parameters
        ----------
        model : diayn.DIAYN
            Base DIAYN model to initialize weights.
        skill : torch.Tensor
            Skill on which to condition `model`.
            
        Returns
        -------
        a2c : A2C
            A2C initialized with DIAYN weights.
        """
        env = model.env
        gamma = model.gamma
        hidden_sizes = {"actor": [1], "critic": [1]}
        tensor_skill = torch.zeros((1, model.prior.event_shape[0]))
        tensor_skill[0, skill] = 1
        a2c = A2C(env, hidden_sizes, gamma)
        a2c.actor = ActorFromDIAYN(model.actor, tensor_skill)
        a2c.critic = CriticFromDIAYN(model.critic, tensor_skill)
        a2c.optimizers = {
            name: torch.optim.Adam(a2c.__getattribute__(name).parameters())
            for name in ("actor", "critic")
        }
        return a2c

    def __init__(self, env, hidden_sizes, gamma=.99):
        """
        Build A2C model.

        Parameters
        ----------
        env : gym environment
            Environment on which to learn skills.
        hidden_sizes : dict
            Dictionnary with keys "actor" and "critic" containing the lists of 
            hidden layers sizes for corresponding networks.
        gamma : float
            Discount factor.
        """
        self.env, self.gamma = env, gamma
        self.actor = Actor(env, hidden_sizes["actor"])
        self.critic = Critic(env, hidden_sizes["critic"])
        self.optimizers = {
            "actor": torch.optim.Adam(self.__getattribute__("actor").parameters(), lr=0.001),
            "critic": torch.optim.Adam(self.__getattribute__("critic").parameters(), lr=0.01)
        }
        self.rewards = []
        self.n_episode = 0

    def episode(self, train=True, render=False, return_states=False):
        """
        Run one episode.

        Parameters
        ----------
        train : bool
            If True, perform update on underlying parameters and store reward
            into self.rewards.
        render : bool
            If True, display the episode with env.render and return total
            reward.
        return_states : bool
            If True, return the list of states of the episode.
        """
        s = self.env.reset()
        done, step, total_reward = False, 0, 0
        if return_states:
            states = [s]
        while not done:
            pi = self.actor(s) # log P(a | s)
            a = Categorical(torch.exp(pi)).sample() # Sample action
            new_s, reward, done, _ = self.env.step(a.item())
            reward = torch.Tensor([[reward]])
            if train: # Perform update
                self._update_models(pi, a, reward, s, new_s, done)
            #print(reward.item())
            total_reward += reward.item()
            if render: # Render the environment
                self.env.render()
            step += 1
            s = new_s
            if return_states:
                states.append(s)
        if train: # Store episode score
            self.n_episode += 1
            self.rewards.append(total_reward)
        if render: # Return episode score
            return total_reward
        if return_states:
            return states

    def load(self):
        """Load A2C model from path."""
        for name in ("actor", "critic"):
            state_dict = torch.load(os.path.join(path, name + ".pkl"))
            self.__getattribute__(name).load_state_dict(state_dict)
            state_dict = torch.load(os.path.join(path, name + "_optimizer.pkl"))
            self.optimizers[name].load_state_dict(state_dict)
        with open(os.path.join(path, "rewards.json"), "r") as f:
            self.rewards = json.load(f)
        self.n_episode = len(self.rewards)

    def plot_rewards(self):
        """Plot rewards accumulated throughout training."""
        plt.plot(self.rewards)
        if len(self.rewards) > 100:
            plt.plot(range(99, len(self.rewards)), 
                     np.convolve(self.rewards, np.ones(100)/100, "valid"))
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("True rewards throughout training")

    def save(self):
        """Save A2C model to path."""
        shutil.rmtree(path, ignore_errors=True)
        os.mkdir(path)
        for name in ("actor", "critic"):
            torch.save(
                self.__getattribute__(name).state_dict(),
                os.path.join(path, name + ".pkl")
            )
            torch.save(
                self.optimizers[name].state_dict(),
                os.path.join(path, name + "_optimizer.pkl")
            )
        with open(os.path.join(path, "rewards.json"), "w") as f:
            json.dump(self.rewards, f)

    def train(self, max_episodes=float("inf"), max_time=float("inf"), 
              verbose=True):
        """
        Train the model.

        Parameters
        ----------
        max_episodes : int
            Maximum number of episodes run during training.
        max_time : float
            Maximum duration of training in seconds.
        verbose : bool
            If True, print information about training.
        """
        if max_episodes == max_time == float("inf"):
            raise ValueError("One of max_episodes or max_time must be set")
        start = time.time()
        episode = 0
        while episode < max_episodes and time.time() - start < max_time:
            if verbose:
                time_str = self._time_to_str(time.time() - start)                    
                print("Episode {} (elapsed time: {})".format(episode + 1,
                                                             time_str),
                      end="\r")
            self.episode(train=True, render=False)
            episode += 1
            if episode % 100 == 0:
                print(episode)
        print(" "*70, end="\r")    

    def _time_to_str(self, t):
        """Turn a timestamp into a readable string."""
        t = int(t)
        time_str = ""
        if t > 3600:
            time_str += "{:02} h ".format(t // 3600)
            t %= 3600
        if t > 60:
            time_str += "{:02} m ".format(t // 60)
            t %= 60
        time_str += "{:02} s".format(t)
        return time_str

    def _update_models(self, pi, a, reward, s, new_s, done):
        """Perform update."""
        next_reward = 0 if done else self.gamma*self.critic(new_s)
        target = (reward + next_reward).detach()
        critic = self.critic(s)
        # Critic loss
        loss = F.smooth_l1_loss(critic, target)
        loss.backward()
        # Actor loss
        advantage = target - critic.detach()
        (-advantage*pi[:, a]).backward()
        # Update parameters and zero grads
        for optimizer in self.optimizers.values():
            optimizer.step()
        for model in self.actor, self.critic:
            model.zero_grad()
