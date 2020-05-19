
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
from model import Model
from torch.utils import data
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Runner(object):
    def __init__(self, args):
        # Set the random seed during training and deterministic cudnn
        
        self.args = args
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

        # Create Logger 
        self.logger = Logger(logdir=args.logdir, run_name=f"{args.env}-{time.ctime()}")

        # Load Env
        self.env = PusherEnv()

        self.num_t_steps = 0
        self.num_e_steps = 0
        self.state = self.env.reset()
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = 2

        # Create Model 
        self.model = Model(self.args, self.num_states, self.num_actions)
        self.optim = torch.optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate)

        self.dataset = np.load('./expert.npz')
        self.tensor_dataset = data.TensorDataset(torch.Tensor(self.dataset['obs']), torch.Tensor(self.dataset['action']))
        self.dataloader = data.DataLoader(self.tensor_dataset, self.args.batch_size, shuffle=True)

    def get_loss(self,loader, model):
        total_loss = 0
        for data in loader:
            state, action = data
            
            predicted_action = model(state.float())
            loss = model.mse(predicted_action, action)         
            total_loss += float(loss.cpu().data)

        return total_loss/len(loader)

    def run(self):
        print("model")
        print(self.model)

        for epoch in range(self.args.num_epochs):  
            for data in self.dataloader:
                state, action = data
                loss = self.model.optimize(state.float(), action)
                self.logger.log_epoch("training_loss", loss,epoch)


        print("Done")
        print("Train Loss", self.get_loss(self.dataloader, self.model))
        self.model.save("models")
        
        done = False
        step = 0
        error = 0
        for ep in range(self.args.n_episodes):
            state = self.env.reset()
            total_reward = 0
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                total_reward+=reward
                self.logger.log_step("reward", total_reward,step, ep)
                step+=1
            
            error += (np.linalg.norm(next_state[3:6] - next_state[6:9])) / self.args.n_episodes
            done = False

        print("Average L2 Distance", error)
        
        step = 0
        for ep in range(10):
            state = self.env.reset()
            total_reward = 0
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                total_reward+=reward
                s = "episode" + str(ep) + " reward " + str(total_reward)
                plt.title(s)
                observation = self.env.render()
                plt.imsave("img_cloning/" + str(step) + ".png", observation)
                step+=1

            done = False
        print("Done Saving 10 Episodes")
        

