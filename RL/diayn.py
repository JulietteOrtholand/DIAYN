import os
import json
import time
import shutil

import matplotlib.pyplot as plt
plt.ion()

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class NeuralNetwork(nn.Sequential):

    """Fully-connected neural network."""

    def __init__(self, in_size, out_size, hidden_sizes, 
                 activation=nn.LeakyReLU):
        layers = []
        for size in hidden_sizes:
            layers.append(nn.Linear(in_size, size))
            layers.append(activation())
            in_size = size
        layers.append(nn.Linear(in_size, out_size))
        super(NeuralNetwork, self).__init__(*layers)  


class Actor(nn.Module):

    """Estimate log p(a | s, z)."""

    def __init__(self, env, prior, hidden_sizes, **kwargs):
        super(Actor, self).__init__()
        prior_size = prior.event_shape[0] if prior.event_shape else 1
        in_size = env.observation_space.shape[0] + prior_size
        try : #if not with navigation2D
            out_size = env.action_space.n
        except: #with navigation2D
            out_size = env.action_space
        self.network = NeuralNetwork(in_size, out_size, hidden_sizes, **kwargs)

    def act(self, s, z):
        return Categorical(torch.exp(self(s, z))).sample().item()

    def forward(self, s, z):
        s = torch.Tensor([s])
        x = torch.cat((s, z), dim=1)
        return F.log_softmax(self.network(x), dim=1)


class Critic(nn.Module):

    """Critic part of the A2C optimization scheme."""

    def __init__(self, env, prior, hidden_sizes, **kwargs):
        super(Critic, self).__init__()
        prior_size = prior.event_shape[0] if prior.event_shape else 1
        in_size = env.observation_space.shape[0] + prior_size
        out_size = 1
        self.network = NeuralNetwork(in_size, out_size, hidden_sizes, **kwargs)

    def forward(self, s, z):
        s = torch.Tensor([s])
        x = torch.cat((s, z), dim=1)
        return self.network(x)


class Discriminator(nn.Module):

    """Estimate log p(z | s)."""

    def __init__(self, env, prior, hidden_sizes, **kwargs):
        super(Discriminator, self).__init__()
        in_size = env.observation_space.shape[0]
        out_size = prior.event_shape[0] if prior.event_shape else 1
        self.network = NeuralNetwork(in_size, out_size, hidden_sizes, **kwargs)
        self.out_size = out_size

    def forward(self, s):
        s = torch.Tensor([s])
        if self.out_size == 1:
            return torch.log(torch.sigmoid(self.network(s)))
        return F.log_softmax(self.network(s), dim=1)


class DIAYN:

    """Online A2C implementation of Diversity Is All You Need."""

    def __init__(self, env, prior, hidden_sizes, alpha=.1, gamma=.99):
        """Build DIAYN model.

        Parameters:
            env          -- gym environment on which to learn skills
            prior        -- OneHotCategorical distribution (from 
                            torch.distributions.one_hot_categorical) encoding 
                            the distribution of skills.
            hidden_sizes -- Dictionnary with keys "actor", "critic" and 
                            "discriminator" containing the lists of hidden 
                            layers sizes for corresponding networks.
            alpha        -- Scaling parameter for entropy regularization.
            gamma        -- Discount factor.
        """
        self.env, self.prior, self.alpha, self.gamma = env, prior, alpha, gamma
        self.actor = Actor(env, prior, hidden_sizes["actor"])
        self.critic = Critic(env, prior, hidden_sizes["critic"])
        self.discriminator = Discriminator(env, prior, 
                                           hidden_sizes["discriminator"])
        self.optimizers = {
            name: torch.optim.Adam(self.__getattribute__(name).parameters())
            for name in ("actor", "critic", "discriminator")
        }
        self.rewards = []
        self.n_episode = 0

    def episode(self, train=True, render=False, z=None):
        """Run one episode.

        Parameters:
            train  -- If True, perform update on underlying parameters and
                      store reward into self.rewards.
            render -- If True, display the episode with env.render and return 
                      total reward.
            z      -- Skill value. If None, a random skill is sampled from
                      self.prior.
        """
        s = self.env.reset()
        if z is None:
            z = self.prior.sample()
        p_z = self.prior.log_prob(z)
        done, step, total_reward = False, 0, 0
        while not done:
            pi = self.actor(s, z) # log P(a | s, z)
            a = Categorical(torch.exp(pi)).sample() # Sample action
            new_s, _, done, _ = self.env.step(a.item())
            q = self.discriminator(s) # log P(z | s)
            reward = q[:, z.argmax(dim=1)] - self.alpha*pi[:, a] - p_z
            if train: # Perform update
                self._update_models(pi, a, q, reward, s, z, new_s, done)
            total_reward += reward.item()
            if render: # Render the environment
                self.env.render()
            step += 1
            s = new_s
        if train: # Store episode score
            self.n_episode += 1
            self.rewards.append(total_reward/step)
        if render: # Return episode score
            return total_reward

    def load(self, path="/tmp/diayn"):
        """Load DIAYN model from path."""
        for name in ("actor", "critic", "discriminator"):
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
        plt.title("DIAYN rewards throughout training")

    def save(self, path=os.path.join("tmp", "diayn")):
        """Save DIAYN model to path."""
        shutil.rmtree(path, ignore_errors=True)
        os.mkdir(path)
        for name in ("actor", "critic", "discriminator"):
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

    def show_skill(self, skill=None, wait_before_closing=False):
        """Run one episode for skill (if None, random skill is chosen)."""
        if skill is None:
            z = self.prior.sample()
        else:
            z = torch.zeros((1, self.prior.event_shape[0]))
            z[0, skill] = 1
        print("Showing skill {}".format(z.argmax(1).item()))
        self.episode(train=False, render=True, z=z)
        if wait_before_closing:
            input("Press Enter to close")
        self.env.close()

    def train(self, max_episodes=float("inf"), max_time=float("inf"), 
              verbose=True):
        """Train the model.

        Parameters:
            max_episodes -- Maximum number of episodes run during training.
            max_time     -- Maximum duration of training in seconds.
            verbose      -- If True, print information about training.
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

    def _update_models(self, pi, a, q, reward, s, z, new_s, done):
        """Perform update."""
        next_reward = 0 if done else self.gamma*self.critic(new_s, z)
        target = (reward + next_reward).detach()
        critic = self.critic(s, z)
        # Critic loss
        loss = F.smooth_l1_loss(critic, target)
        loss.backward()
        # Actor loss
        advantage = target - critic.detach()
        (-advantage*pi[:, a]).backward()
        # Discriminator loss
        F.nll_loss(q, z.argmax(dim=1)).backward()
        # Update parameters and zero grads
        for optimizer in self.optimizers.values():
            optimizer.step()
        for model in self.actor, self.critic, self.discriminator:
            model.zero_grad()
