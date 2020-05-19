from model import Model 

import numpy as np
import argparse
import torch
from copy import deepcopy
import random

import cv2
from logger import Logger
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pusher_goal import PusherEnv
from ppo import PPO, Memory
from torch.utils import data
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":

    # from scratch -- PPO_continuous_ppo_scratch.pth
    # vanilla fine tune - PPO_continuous_ppo_vanilla.pth
    # joint - PPO_continuous_ppo_joint.pth

    parser = argparse.ArgumentParser(description="behavioral_cloning")

    parser.add_argument('--env', default='PusherEnvJoint', help='ROM to run')
    parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for numpy, torch, random.')
    parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
    parser.add_argument('--modeldir', type=str, default='models', help='Directory for saving model')
    parser.add_argument('--clip-value', type=float, default=0.5, help="clip value")
    parser.add_argument('--n-episodes', type=int, default=int(100), help='number of maximum epsisodes to take.') # bout 4 million
    parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
    parser.add_argument('--n-hidden-units', type=int, default=32, help="Hidden units in NN")
    parser.add_argument('--num-epochs', type=int, default=30, help="Epoch Number")

    args = parser.parse_args()
    logger = Logger(logdir=args.logdir, run_name=f"{args.env}-{time.ctime()}")

    env_name = "ppo_joint"
    env = PusherEnv()

    render = False
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 10000        # max timesteps in one episode
    
    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = None
    #############################################
    
    writer = SummaryWriter("runs" + env_name)

    # creating environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    model = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    model.load("PPO_continuous_ppo_joint.pth")

    done = False
    step = 0
    error = 0
    for ep in range(args.n_episodes):
        state = env.reset()
        total_reward = 0
        while not done:
            action = model.select_action(state, memory)
            next_state, reward, done, info = env.step(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            total_reward+=reward
            logger.log_step("reward", total_reward,step, ep)
            step+=1
        
        error += (np.linalg.norm(next_state[3:6] - next_state[6:9])) / args.n_episodes
        done = False

    print("Average L2 Distance", error)
    
    step = 0
    for ep in range(10):
        state = env.reset()
        total_reward = 0
        while not done:
            action = model.select_action(state, memory)
            next_state, reward, done, info = env.step(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            total_reward+=reward
            s = "episode" + str(ep) + " reward " + str(total_reward)
            plt.title(s)
            observation = env.render()
            plt.imsave("img_cloning_joint/" + str(step) + ".png", observation)
            step+=1

        done = False

    print("Done Saving 10 Episodes")