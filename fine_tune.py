import argparse

from pusher_goal import PusherEnv
from ppo import train_ppo
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":

    ## PPO policy learned form scratch
    env = PusherEnv(render=False)
    train_ppo(env)


    ## PPO vanilla fine-tuned policy



    ## PPO joint loss fine-tuned policy


    ## learning curves
    ## Evaluate the three policies on 100 episodes and report the average L2 distance between the object location and the goal location (20 pts).
    ## Video for 10 episodes

