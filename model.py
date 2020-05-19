
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, Adadelta


class Model(nn.Module):

    def __init__(self,
                 args,
                 num_states,
                 num_actions):

        super(Model, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions

        self.args = args
        self.name = "cloning"

        self.fc1 = nn.Linear(num_states, args.n_hidden_units)
        self.fc2 = nn.Linear(args.n_hidden_units, args.n_hidden_units)
        self.fc3 = nn.Linear(args.n_hidden_units, num_actions)

        self.mse = nn.MSELoss()
        self.optim = Adadelta(self.parameters())
        # self.optim = Adam(self.parameters(), lr=args.learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def optimize(self, state, action):
        predicted_action = self(state)
        loss = self.mse(predicted_action.double(), action.double())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss.cpu().data)

    def get_action(self, state):
        with torch.no_grad():
            return self(torch.from_numpy(state).float())

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, "{}.pth".format(self.name)))

    def load(self, model_dir):
        self.load_state_dict(torch.load(os.path.join(model_dir, "{}.pth".format(self.name))))

