import argparse

from pusher_goal import PusherEnv
from ppo import train_ppo, train_ppo_vanilla, train_ppo_joint_loss
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from model import Model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="behavioral_cloning")

    parser.add_argument('--env', default='PusherEnv', help='ROM to run')
    parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for numpy, torch, random.')
    parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
    parser.add_argument('--modeldir', type=str, default='models', help='Directory for saving model')
    parser.add_argument('--clip-value', type=float, default=0.5, help="clip value")
    parser.add_argument('--n-episodes', type=int, default=int(100), help='number of maximum epsisodes to take.') # bout 4 million
    parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
    parser.add_argument('--n-hidden-units', type=int, default=64, help="Hidden units in NN")
    parser.add_argument('--num-epochs', type=int, default=30, help="Epoch Number")

    args = parser.parse_args()

    ## PPO policy learned form scratch
    #env = PusherEnv(render=False)
    #train_ppo(env)


    ## PPO vanilla fine-tuned policy
    #num_states = 8
    #num_actions = 2
    #env2 = PusherEnv(render=False)
    #train_ppo_vanilla(env2)



    ## PPO joint loss fine-tuned policy
    num_states = 8
    num_actions = 2
    env3 = PusherEnv(render=False)
    train_ppo_joint_loss(env3)


